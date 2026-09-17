#pragma once

#include "wsvt/search_pruning.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace wsvt {

enum class FixedSpatialSupport { Point, Uniform3, Hamming3 };

inline const char* fixed_spatial_support_name(FixedSpatialSupport support) {
    switch (support) {
    case FixedSpatialSupport::Point: return "point";
    case FixedSpatialSupport::Uniform3: return "uniform3";
    case FixedSpatialSupport::Hamming3: return "hamming3";
    }
    throw std::invalid_argument("unknown fixed spatial support");
}

// Bit inspection remains valid with the existing -ffinite-math-only build.
inline bool finite_descriptor_value(float value) noexcept {
    return (std::bit_cast<std::uint32_t>(value) & 0x7f800000U) != 0x7f800000U;
}

// Validated immutable HWD view. No allocation or wraparound at the boundary.
struct SpatialFeatureView {
    std::span<const float> data;
    std::size_t h, w, depth;
    SpatialFeatureView(std::span<const float> values, std::size_t height,
                       std::size_t width, std::size_t d)
        : data(values), h(height), w(width), depth(d) {
        const auto max = std::numeric_limits<std::size_t>::max();
        if (!h || !w || !depth || h > max / w || h*w > max / depth ||
            h*w*depth != data.size() ||
            h > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
            w > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::invalid_argument("invalid spatial descriptor shape");
        }
        for (float v : data) {
            if (!finite_descriptor_value(v))
                throw std::invalid_argument("nonfinite spatial descriptor");
        }
    }
    const float* pixel(std::int64_t y, std::int64_t x) const noexcept {
        if (y < 0 || x < 0 || static_cast<std::uint64_t>(y) >= h ||
            static_cast<std::uint64_t>(x) >= w) return nullptr;
        return data.data() + (static_cast<std::size_t>(y)*w +
                              static_cast<std::size_t>(x))*depth;
    }
};

class FixedSpatialCost {
public:
    explicit FixedSpatialCost(FixedSpatialSupport support) : support_(support) {
        (void)fixed_spatial_support_name(support);
        if (support == FixedSpatialSupport::Uniform3) {
            weights_.fill(1.0/9.0);
        } else if (support == FixedSpatialSupport::Hamming3) {
            constexpr std::array<double,3> h{2.0/29.0,25.0/29.0,2.0/29.0};
            for (int y=0;y<3;++y) for (int x=0;x<3;++x)
                weights_[y*3+x] = h[y]*h[x];
        } else {
            weights_[4] = 1.0;
        }
    }
    const std::array<double,9>& weights() const noexcept { return weights_; }
    // Coordinates refer to these views, not to residual search indices.
    float operator()(const SpatialFeatureView& sample, const SpatialFeatureView& ref,
                     std::int64_t sy, std::int64_t sx,
                     std::int64_t ry, std::int64_t rx) const {
        return evaluate(sample,ref,sy,sx,ry,rx,point_distance);
    }
    static float point_distance(const SpatialFeatureView& sample, const SpatialFeatureView& ref,
                                std::int64_t sy,std::int64_t sx,std::int64_t ry,std::int64_t rx) {
        const float* a=sample.pixel(sy,sx);
        const float* b=ref.pixel(ry,rx);
        if(a&&b) return squared_distance_full_simd(a,b,sample.depth);
        float distance=0.0f;
        for(std::size_t c=0;c<sample.depth;++c) {
            const float delta=(a ? a[c] : 0.0f)-(b ? b[c] : 0.0f);
            distance+=delta*delta;
        }
        return distance;
    }
    template<class Distance>
    float evaluate(const SpatialFeatureView& sample,const SpatialFeatureView& ref,
                   std::int64_t sy,std::int64_t sx,std::int64_t ry,std::int64_t rx,
                   Distance&& point) const {
        if (sample.depth != ref.depth)
            throw std::invalid_argument("spatial descriptor depths differ");
        // Leave headroom for the +/-1 support before adding signed offsets.
        constexpr auto lim = std::numeric_limits<std::int64_t>::max()-1;
        if (sy < -lim || sx < -lim || ry < -lim || rx < -lim ||
            sy > lim || sx > lim || ry > lim || rx > lim)
            throw std::invalid_argument("spatial coordinate overflow");
        const int radius = support_ == FixedSpatialSupport::Point ? 0 : 1;
        double total = 0.0;
        for (int y=-radius;y<=radius;++y) for (int x=-radius;x<=radius;++x) {
            const float distance=point(sample,ref,sy+y,sx+x,ry+y,rx+x);
            total += weights_[(y+1)*3+x+1]*static_cast<double>(distance);
        }
        return static_cast<float>(total);
    }
private:
    FixedSpatialSupport support_;
    std::array<double,9> weights_{};
};

struct SpatialReuseStats {
    std::uint64_t hits=0,misses=0,resets=0,overflows=0;
    std::size_t peak_entries=0,allocated_bytes=0;
};

// One instance per worker and immutable descriptor level. Keys include actual
// sample AND reference coordinates, so different pyramid centers cannot alias.
class SpatialPointCache {
    struct Entry {
        std::array<std::int64_t,4> key{};
        float value=0;
        std::uint32_t epoch=0;
    };
    static constexpr std::size_t slots=32768;
    std::vector<Entry> table_{slots};
    std::uint32_t epoch_=1;
    std::size_t tile_=std::numeric_limits<std::size_t>::max(),entries_=0;
    SpatialReuseStats stats_;
public:
    SpatialPointCache() { stats_.allocated_bytes=table_.size()*sizeof(Entry)+sizeof(*this); }
    void select_tile(std::size_t tile) {
        if(tile==tile_) return;
        tile_=tile;entries_=0;++stats_.resets;
        if(++epoch_==0) {
            for(auto& e:table_) e.epoch=0;
            epoch_=1;
        }
    }
    const SpatialReuseStats& stats() const { return stats_; }
    float operator()(const SpatialFeatureView& sample,const SpatialFeatureView& ref,
                     std::int64_t sy,std::int64_t sx,std::int64_t ry,std::int64_t rx) {
        const std::array<std::int64_t,4> key{sy,sx,ry,rx};
        std::uint64_t h=0x9e3779b97f4a7c15ULL;
        for(auto v:key) {
            auto z=static_cast<std::uint64_t>(v)+0x9e3779b97f4a7c15ULL;
            z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
            z=(z^(z>>27))*0x94d049bb133111ebULL;
            h^=(z^(z>>31))+(h<<6)+(h>>2);
        }
        for(std::size_t probe=0;probe<slots;++probe) {
            auto& e=table_[(h+probe)&(slots-1)];
            if(e.epoch!=epoch_) {
                e.key=key;e.value=FixedSpatialCost::point_distance(sample,ref,sy,sx,ry,rx);
                e.epoch=epoch_;++stats_.misses;++entries_;
                if(entries_>stats_.peak_entries) stats_.peak_entries=entries_;
                return e.value;
            }
            if(e.key==key) { ++stats_.hits;return e.value; }
        }
        // Bounded, numerically exact fallback, never evict a value silently.
        ++stats_.overflows;++stats_.misses;
        return FixedSpatialCost::point_distance(sample,ref,sy,sx,ry,rx);
    }
};
} // namespace wsvt
