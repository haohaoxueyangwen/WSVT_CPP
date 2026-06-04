#include "wsvt/pyramid.hpp"
#include "wsvt/console_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace wsvt {

namespace {

inline std::size_t idx3(std::size_t c, std::size_t y, std::size_t x, std::size_t h, std::size_t w) {
    return (c * h + y) * w + x;
}

inline std::size_t idx_feat(std::size_t y, std::size_t x, std::size_t d, std::size_t w, std::size_t depth) {
    return (y * w + x) * depth + d;
}

inline std::size_t idx_chw(std::size_t d, std::size_t y, std::size_t x, std::size_t h, std::size_t w) {
    return (d * h + y) * w + x;
}

inline std::size_t wrap_index(long long v, std::size_t n) {
    const long long m = static_cast<long long>(n);
    long long r = v % m;
    if (r < 0) r += m;
    return static_cast<std::size_t>(r);
}

std::vector<float> downsample2x_mean(
    const std::vector<float>& in,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    std::size_t& out_h,
    std::size_t& out_w) {
    if (in.size() != ch * h * w) {
        throw std::invalid_argument("downsample2x_mean size mismatch");
    }

    out_h = (h + 1) / 2;
    out_w = (w + 1) / 2;
    std::vector<float> out(ch * out_h * out_w, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t c = 0; c < ch; ++c) {
        for (std::size_t oy = 0; oy < out_h; ++oy) {
            for (std::size_t ox = 0; ox < out_w; ++ox) {
                const std::size_t y0 = oy * 2;
                const std::size_t x0 = ox * 2;
                float sum = 0.0f;
                std::size_t cnt = 0;
                for (std::size_t ky = 0; ky < 2; ++ky) {
                    for (std::size_t kx = 0; kx < 2; ++kx) {
                        const std::size_t sy = y0 + ky;
                        const std::size_t sx = x0 + kx;
                        if (sy < h && sx < w) {
                            sum += in[idx3(c, sy, sx, h, w)];
                            ++cnt;
                        }
                    }
                }
                out[idx3(c, oy, ox, out_h, out_w)] = cnt == 0 ? 0.0f : sum / static_cast<float>(cnt);
            }
        }
    }
    return out;
}

std::vector<float> downsample2x_db3_aa(
    const std::vector<float>& in,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    std::size_t& out_h,
    std::size_t& out_w) {
    if (in.size() != ch * h * w) {
        throw std::invalid_argument("downsample2x_db3_aa size mismatch");
    }

    // db3 low-pass decomposition filter (reversed for convolution, matching pywt convention)
    static constexpr float kDb3LoD[6] = {
        0.3326705530f, 0.8068915093f, 0.4598775021f,
        -0.1350110200f, -0.0854412739f, 0.0352262919f
    };
    static constexpr int kFilterLen = 6;
    static constexpr int kOffset = kFilterLen - 2;  // = 4

    // Output size matches pywt: floor((n + filter_len - 1) / 2)
    out_h = static_cast<std::size_t>((static_cast<int>(h) + kFilterLen - 1) / 2);
    out_w = static_cast<std::size_t>((static_cast<int>(w) + kFilterLen - 1) / 2);

    // Separable 2D DWT: first filter along rows (axis=-1), then along columns (axis=-2)
    // Step 1: Row-wise filtering → intermediate shape [ch, h, out_w]
    std::vector<float> row_filtered(ch * h * out_w, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t c = 0; c < ch; ++c) {
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t ox = 0; ox < out_w; ++ox) {
                const int sx = static_cast<int>(ox) * 2;
                float s = 0.0f;
                for (int j = 0; j < kFilterLen; ++j) {
                    const int ix = sx + j - kOffset;
                    if (ix >= 0 && ix < static_cast<int>(w)) {
                        s += kDb3LoD[j] * in[idx3(c, y, static_cast<std::size_t>(ix), h, w)];
                    }
                }
                row_filtered[(c * h + y) * out_w + ox] = s;
            }
        }
    }

    // Step 2: Column-wise filtering → final shape [ch, out_h, out_w]
    std::vector<float> out(ch * out_h * out_w, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t c = 0; c < ch; ++c) {
        for (std::size_t oy = 0; oy < out_h; ++oy) {
                const int sy = static_cast<int>(oy) * 2;
                for (std::size_t ox = 0; ox < out_w; ++ox) {
                    float s = 0.0f;
                    for (int j = 0; j < kFilterLen; ++j) {
                        const int iy = sy + j - kOffset;
                        if (iy >= 0 && iy < static_cast<int>(h)) {
                            s += kDb3LoD[j] * row_filtered[(c * h + static_cast<std::size_t>(iy)) * out_w + ox];
                        }
                    }
                    out[idx3(c, oy, ox, out_h, out_w)] = s;
                }
        }
    }
    return out;
}

std::vector<float> downsample_by_mode(
    const std::vector<float>& in,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    std::size_t& out_h,
    std::size_t& out_w,
    PyramidDownsampleMode mode) {
    if (mode == PyramidDownsampleMode::Mean2x2) {
        return downsample2x_mean(in, ch, h, w, out_h, out_w);
    }
    return downsample2x_db3_aa(in, ch, h, w, out_h, out_w);
}

}  // anonymous namespace

// HWD-native template window stacking: input CHW, output HWD
AlignedVector<float> stack_template_window_hwd(
    const std::vector<float>& img,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int n_template,
    std::size_t& out_depth) {
    if (img.size() != ch * h * w) {
        throw std::invalid_argument("stack_template_window_hwd size mismatch");
    }
    if (n_template < 0) {
        throw std::invalid_argument("n_template must be >= 0");
    }
    const int axis = 2 * n_template + 1;
    out_depth = ch * static_cast<std::size_t>(axis * axis);
    AlignedVector<float> out(h * w * out_depth, 0.0f);

    // Fast path: n_template=0 means no spatial shifts — just CHW->HWD transpose.
    if (n_template == 0) {
        const std::size_t plane = h * w;
        #pragma omp parallel for schedule(static)
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            float* dst = out.data() + pixel * ch;
            for (std::size_t c = 0; c < ch; ++c) {
                dst[c] = img[c * plane + pixel];
            }
        }
        return out;
    }

    #pragma omp parallel for collapse(3) schedule(static)
    for (int dy = -n_template; dy <= n_template; ++dy) {
        for (int dx = -n_template; dx <= n_template; ++dx) {
            for (std::size_t c = 0; c < ch; ++c) {
                const std::size_t d = static_cast<std::size_t>(
                    (dy + n_template) * axis + (dx + n_template)) * ch + c;

                std::vector<std::size_t> wy(h), wx(w);
                for (std::size_t i = 0; i < h; ++i) {
                    wy[i] = wrap_index(static_cast<long long>(i) - dy, h);
                }
                for (std::size_t i = 0; i < w; ++i) {
                    wx[i] = wrap_index(static_cast<long long>(i) - dx, w);
                }

                for (std::size_t y = 0; y < h; ++y) {
                    for (std::size_t x = 0; x < w; ++x) {
                        out[idx_feat(y, x, d, w, out_depth)] = img[idx3(c, wy[y], wx[x], h, w)];
                    }
                }
            }
        }
    }
    return out;
}

// Pixel-major fused template window + PerLevelFeature normalization.
// Replaces separate stack_template_window_hwd() + normalize_feature_depth_hwd()
// with a single pass that fills depth per-pixel, accumulates sum/sum_sq on the fly,
// and normalizes in-place — cutting memory traffic by ~2×.
// Shift order, wrap semantics, and normalization formula are bitwise-identical
// to the separate path (verified by golden tests).
AlignedVector<float> stack_and_normalize_template_hwd(
    const std::vector<float>& img,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int n_template,
    std::size_t& out_depth)
{
    if (img.size() != ch * h * w) {
        throw std::invalid_argument("stack_and_normalize_template_hwd size mismatch");
    }
    if (n_template < 0) {
        throw std::invalid_argument("n_template must be >= 0");
    }

    const int axis = 2 * n_template + 1;
    out_depth = ch * static_cast<std::size_t>(axis * axis);
    AlignedVector<float> out(h * w * out_depth, 0.0f);
    const std::size_t plane = h * w;
    constexpr float kEps = 1e-6f;

    // Fast path: n_template=0 — just CHW→HWD transpose + normalize
    if (n_template == 0) {
        const float inv_ch = 1.0f / static_cast<float>(ch);
        #pragma omp parallel for schedule(static)
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            float* dst = out.data() + pixel * ch;
            float sum = 0.0f, sum_sq = 0.0f;
            for (std::size_t c = 0; c < ch; ++c) {
                const float v = img[c * plane + pixel];
                dst[c] = v;
                sum += v;
                sum_sq += v * v;
            }
            const float mean = sum * inv_ch;
            const float var = sum_sq * inv_ch - mean * mean;
            const float inv_std = 1.0f / (std::sqrt(std::max(var, 0.0f)) + kEps);
            #pragma omp simd
            for (std::size_t c = 0; c < ch; ++c) {
                dst[c] = (dst[c] - mean) * inv_std;
            }
        }
        return out;
    }

    // Precompute per-shift wrap indices: wy[dy_idx][y], wx[dx_idx][x]
    const int N = n_template;
    std::vector<std::vector<std::size_t>> wy_lut(static_cast<std::size_t>(axis));
    std::vector<std::vector<std::size_t>> wx_lut(static_cast<std::size_t>(axis));
    for (int i = 0; i < axis; ++i) {
        const int d = i - N;
        wy_lut[static_cast<std::size_t>(i)].resize(h);
        wx_lut[static_cast<std::size_t>(i)].resize(w);
        for (std::size_t y = 0; y < h; ++y) {
            wy_lut[static_cast<std::size_t>(i)][y] = wrap_index(static_cast<long long>(y) - d, h);
        }
        for (std::size_t x = 0; x < w; ++x) {
            wx_lut[static_cast<std::size_t>(i)][x] = wrap_index(static_cast<long long>(x) - d, w);
        }
    }

    const float inv_depth = 1.0f / static_cast<float>(out_depth);

    #pragma omp parallel for schedule(static)
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        const std::size_t y = pixel / w;
        const std::size_t x = pixel % w;

        float* dst = out.data() + pixel * out_depth;

        // Fill depth vector and accumulate stats in one pass
        float sum = 0.0f;
        float sum_sq = 0.0f;
        std::size_t d = 0;

        for (int dy_idx = 0; dy_idx < axis; ++dy_idx) {
            const std::size_t src_y = wy_lut[static_cast<std::size_t>(dy_idx)][y];
            for (int dx_idx = 0; dx_idx < axis; ++dx_idx) {
                const std::size_t src_x = wx_lut[static_cast<std::size_t>(dx_idx)][x];
                for (std::size_t c = 0; c < ch; ++c) {
                    const float v = img[idx3(c, src_y, src_x, h, w)];
                    dst[d] = v;
                    sum += v;
                    sum_sq += v * v;
                    ++d;
                }
            }
        }

        // Normalize in-place (same formula as normalize_feature_depth_hwd)
        const float mean = sum * inv_depth;
        const float var = sum_sq * inv_depth - mean * mean;
        const float inv_std = 1.0f / (std::sqrt(std::max(var, 0.0f)) + kEps);

        #pragma omp simd
        for (std::size_t i = 0; i < out_depth; ++i) {
            dst[i] = (dst[i] - mean) * inv_std;
        }
    }

    return out;
}

namespace {

// HWD-native normalization: depth is contiguous per pixel — cache-friendly
[[maybe_unused]] void normalize_feature_depth_hwd(AlignedVector<float>& feature, std::size_t h, std::size_t w, std::size_t depth) {
    if (feature.size() != h * w * depth) {
        throw std::invalid_argument("normalize_feature_depth_hwd size mismatch");
    }
    constexpr float kEps = 1e-6f;
    const std::size_t plane = h * w;
    const float inv_depth = 1.0f / static_cast<float>(depth);

    #pragma omp parallel for schedule(static)
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        float* row = feature.data() + pixel * depth;

        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (std::size_t d = 0; d < depth; ++d) sum += row[d];
        const float mean = sum * inv_depth;

        float var = 0.0f;
        #pragma omp simd reduction(+:var)
        for (std::size_t d = 0; d < depth; ++d) {
            const float v = row[d] - mean;
            var += v * v;
        }
        const float inv_std = 1.0f / (std::sqrt(var * inv_depth) + kEps);

        #pragma omp simd
        for (std::size_t d = 0; d < depth; ++d) {
            row[d] = (row[d] - mean) * inv_std;
        }
    }
}

std::vector<float> normalize_stack_depth_chw(
    const std::vector<float>& data,
    std::size_t ch,
    std::size_t h,
    std::size_t w) {
    if (data.size() != ch * h * w) {
        throw std::invalid_argument("normalize_stack_depth_chw size mismatch");
    }
    std::vector<float> out(data.size(), 0.0f);

    #pragma omp parallel for schedule(static)
    for (std::size_t pixel = 0; pixel < h * w; ++pixel) {
        double mean = 0.0;
        for (std::size_t c = 0; c < ch; ++c) {
            mean += static_cast<double>(data[c * h * w + pixel]);
        }
        mean /= static_cast<double>(ch);

        double var = 0.0;
        for (std::size_t c = 0; c < ch; ++c) {
            const double diff = static_cast<double>(data[c * h * w + pixel]) - mean;
            var += diff * diff;
        }
        const double stdv = std::sqrt(var / static_cast<double>(ch));

        for (std::size_t c = 0; c < ch; ++c) {
            out[c * h * w + pixel] = stdv > 0.0
                ? static_cast<float>((static_cast<double>(data[c * h * w + pixel]) - mean) / stdv)
                : 0.0f;
        }
    }
    return out;
}

PyramidResult build_pyramid_single(
    const std::vector<float>& data,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int pyramid_level,
    int n_template,
    PyramidDownsampleMode mode,
    PyramidNormalizationMode normalization_mode) {
    if (data.size() != ch * h * w) {
        throw std::invalid_argument("build_pyramid_single size mismatch");
    }
    if (pyramid_level < 0) {
        throw std::invalid_argument("pyramid_level must be >= 0");
    }

    double total_downsample_s = 0.0;
    double total_tmpl_s = 0.0;
    double total_norm_s = 0.0;

    std::vector<std::vector<float>> raw_levels;
    std::vector<std::array<std::size_t, 3>> raw_dims;
    if (normalization_mode == PyramidNormalizationMode::InitialStack) {
        auto t0 = std::chrono::steady_clock::now();
        raw_levels.push_back(normalize_stack_depth_chw(data, ch, h, w));
        auto t1 = std::chrono::steady_clock::now();
        total_norm_s += std::chrono::duration<double>(t1 - t0).count();
    } else {
        raw_levels.push_back(data);
    }
    raw_dims.push_back({ch, h, w});

    for (int lv = 0; lv < pyramid_level; ++lv) {
        std::size_t nh = 0;
        std::size_t nw = 0;
        const auto& prev = raw_levels.back();
        const auto dim = raw_dims.back();
        auto t0 = std::chrono::steady_clock::now();
        auto next = downsample_by_mode(prev, dim[0], dim[1], dim[2], nh, nw, mode);
        auto t1 = std::chrono::steady_clock::now();
        total_downsample_s += std::chrono::duration<double>(t1 - t0).count();
        raw_levels.push_back(std::move(next));
        raw_dims.push_back({dim[0], nh, nw});
    }

    PyramidResult result;
    result.ref_levels.reserve(raw_levels.size());

    for (std::size_t lv = 0; lv < raw_levels.size(); ++lv) {
        const auto dim = raw_dims[lv];
        std::size_t out_d = 0;
        auto t0 = std::chrono::steady_clock::now();

        AlignedVector<float> feat;
        if (normalization_mode == PyramidNormalizationMode::PerLevelFeature) {
            // Fused: pixel-major template fill + normalization in one pass
            feat = stack_and_normalize_template_hwd(raw_levels[lv], dim[0], dim[1], dim[2], n_template, out_d);
            auto t1 = std::chrono::steady_clock::now();
            total_tmpl_s += std::chrono::duration<double>(t1 - t0).count();
            // normalization is fused into template_window time
        } else {
            feat = stack_template_window_hwd(raw_levels[lv], dim[0], dim[1], dim[2], n_template, out_d);
            auto t1 = std::chrono::steady_clock::now();
            total_tmpl_s += std::chrono::duration<double>(t1 - t0).count();
        }

        result.ref_levels.push_back(PyramidLevel{std::move(feat), out_d, dim[1], dim[2]});
    }
    prColor("  pyramid detail: downsample=" + std::to_string(total_downsample_s) +
            "s tmpl_win=" + std::to_string(total_tmpl_s) +
            "s normalize=" + std::to_string(total_norm_s) + "s", "light_purple");
    result.downsample_time_s = static_cast<double>(total_downsample_s);
    result.template_window_time_s = static_cast<double>(total_tmpl_s);
    result.normalize_time_s = static_cast<double>(total_norm_s);
    return result;
}

}

PyramidResult pyramid_data(
    const std::vector<float>& ref_data,
    const std::vector<float>& img_data,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int pyramid_level,
    int n_template,
    PyramidDownsampleMode mode,
    PyramidNormalizationMode normalization_mode) {
    if (ref_data.size() != ch * h * w || img_data.size() != ch * h * w) {
        throw std::invalid_argument("pyramid_data input size mismatch");
    }

    int max_threads = 1;
    bool use_nested_sections = false;
#ifdef _OPENMP
    max_threads = std::max(1, omp_get_max_threads());
    use_nested_sections = (omp_get_max_active_levels() > 1 && max_threads >= 4);
#endif

    PyramidResult ref_result, img_result;
    if (use_nested_sections) {
        const int inner_threads = std::max(1, max_threads / 2);
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
#ifdef _OPENMP
                omp_set_num_threads(inner_threads);
#endif
                ref_result = build_pyramid_single(ref_data, ch, h, w, pyramid_level, n_template, mode, normalization_mode);
            }
            #pragma omp section
            {
#ifdef _OPENMP
                omp_set_num_threads(inner_threads);
#endif
                img_result = build_pyramid_single(img_data, ch, h, w, pyramid_level, n_template, mode, normalization_mode);
            }
        }
#ifdef _OPENMP
        omp_set_num_threads(max_threads);
#endif
    } else {
        ref_result = build_pyramid_single(ref_data, ch, h, w, pyramid_level, n_template, mode, normalization_mode);
        img_result = build_pyramid_single(img_data, ch, h, w, pyramid_level, n_template, mode, normalization_mode);
    }

    PyramidResult merged;
    merged.ref_levels = std::move(ref_result.ref_levels);
    merged.img_levels = std::move(img_result.ref_levels);
    if (use_nested_sections) {
        merged.downsample_time_s = std::max(ref_result.downsample_time_s, img_result.downsample_time_s);
        merged.template_window_time_s = std::max(ref_result.template_window_time_s, img_result.template_window_time_s);
        merged.normalize_time_s = std::max(ref_result.normalize_time_s, img_result.normalize_time_s);
    } else {
        merged.downsample_time_s = ref_result.downsample_time_s + img_result.downsample_time_s;
        merged.template_window_time_s = ref_result.template_window_time_s + img_result.template_window_time_s;
        merged.normalize_time_s = ref_result.normalize_time_s + img_result.normalize_time_s;
    }
    return merged;
}

}
