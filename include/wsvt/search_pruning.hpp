#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace wsvt {

inline constexpr std::size_t kMaxExactTopK = 4;

struct RankedSsdCandidate {
    float distance = std::numeric_limits<float>::infinity();
    std::size_t candidate_index = std::numeric_limits<std::size_t>::max();
};

// Fixed-capacity, allocation-free Top-K state for the displacement hot loop.
// Entries are ordered by increasing SSD, then by row-major candidate index.
class ExactTopK {
public:
    explicit ExactTopK(std::size_t k) : k_(k) {
        if (k_ == 0 || k_ > kMaxExactTopK) {
            throw std::invalid_argument("ExactTopK k must be in [1, 4]");
        }
    }

    void reset() noexcept {
        size_ = 0;
    }

    // Returns true only when the pruning threshold changes. The ranked state
    // may still change on an equal-distance row-major tie.
    bool consider(float distance, std::size_t candidate_index) noexcept {
        const float old_threshold = abandon_threshold();
        const RankedSsdCandidate candidate{
            std::isnan(distance) ? std::numeric_limits<float>::infinity() : distance,
            candidate_index};
        std::size_t pos = 0;
        while (pos < size_ && !less(candidate, entries_[pos])) {
            ++pos;
        }
        if (pos >= k_) {
            return false;
        }

        const std::size_t new_size = std::min(k_, size_ + 1);
        for (std::size_t i = new_size; i > pos + 1; --i) {
            entries_[i - 1] = entries_[i - 2];
        }
        entries_[pos] = candidate;
        size_ = new_size;
        return abandon_threshold() != old_threshold;
    }

    [[nodiscard]] bool full() const noexcept {
        return size_ == k_;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return size_;
    }

    [[nodiscard]] float abandon_threshold() const noexcept {
        return full() ? entries_[k_ - 1].distance
                      : std::numeric_limits<float>::infinity();
    }

    [[nodiscard]] const RankedSsdCandidate& operator[](std::size_t index) const noexcept {
        return entries_[index];
    }

private:
    static bool less(
        const RankedSsdCandidate& lhs,
        const RankedSsdCandidate& rhs) noexcept {
        return lhs.distance < rhs.distance ||
            (lhs.distance == rhs.distance && lhs.candidate_index < rhs.candidate_index);
    }

    std::array<RankedSsdCandidate, kMaxExactTopK> entries_ {};
    std::size_t k_ = 0;
    std::size_t size_ = 0;
};

struct BlockwiseSsdResult {
    float distance = 0.0f;
    std::size_t terms_evaluated = 0;
    bool abandoned = false;
};

// Keep the final-candidate and Hessian-neighbourhood reduction identical to
// the exhaustive WSVT loop. This removes block-boundary round-off from the
// publication-facing best/second scores and subpixel fit.
inline float squared_distance_full_simd(
    const float* lhs,
    const float* rhs,
    std::size_t depth) noexcept {
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (std::size_t k = 0; k < depth; ++k) {
        const float diff = lhs[k] - rhs[k];
        sum += diff * diff;
    }
    return sum;
}

// The caller supplies a complete Top-K distance as abandon_above. Every SSD
// term is non-negative, so a partial sum strictly above that threshold cannot
// enter the final Top-K. Checks occur only between SIMD-friendly blocks.
inline BlockwiseSsdResult squared_distance_blockwise(
    const float* lhs,
    const float* rhs,
    std::size_t depth,
    float abandon_above,
    std::size_t block_size) noexcept {
    float sum = 0.0f;
    for (std::size_t begin = 0; begin < depth; begin += block_size) {
        const std::size_t end = std::min(depth, begin + block_size);
        #pragma omp simd reduction(+:sum)
        for (std::size_t k = begin; k < end; ++k) {
            const float diff = lhs[k] - rhs[k];
            sum += diff * diff;
        }
        if (sum > abandon_above && end < depth) {
            return BlockwiseSsdResult{sum, end, true};
        }
    }
    return BlockwiseSsdResult{sum, depth, false};
}

// Convert a computed prefix SSD into a conservative lower bound for the value
// that the baseline full float reduction can return.  For non-negative terms,
// gamma_n bounds the round-off of any n-term float summation.  Using epsilon
// rather than unit round-off deliberately makes the guard more conservative.
// A non-finite or structurally invalid input disables pruning.
inline double conservative_full_ssd_lower_bound(
    float prefix_distance,
    std::size_t prefix_terms,
    std::size_t full_depth) noexcept {
    if (!std::isfinite(prefix_distance) || prefix_distance < 0.0f ||
        prefix_terms == 0 || prefix_terms > full_depth) {
        return -std::numeric_limits<double>::infinity();
    }

    const double epsilon = static_cast<double>(
        std::numeric_limits<float>::epsilon());
    const double prefix_product = static_cast<double>(prefix_terms) * epsilon;
    const double full_product = static_cast<double>(full_depth) * epsilon;
    if (prefix_product >= 1.0 || full_product >= 1.0) {
        return -std::numeric_limits<double>::infinity();
    }
    const double gamma_prefix = prefix_product / (1.0 - prefix_product);
    const double gamma_full = full_product / (1.0 - full_product);
    const double lower = static_cast<double>(prefix_distance) *
        (1.0 - gamma_full) / (1.0 + gamma_prefix);
    return std::nextafter(lower, -std::numeric_limits<double>::infinity());
}

// U2C preserves the U2B inequality exactly while moving invariant gamma work
// out of the candidate loop. For representable doubles x and t,
// nextafter(x,-inf) > t iff x > nextafter(t,+inf). The caller may therefore
// cache guarded_threshold() until the complete Top-K threshold changes.
class ConservativePrefixSsdGuard {
public:
    ConservativePrefixSsdGuard(
        std::size_t prefix_terms,
        std::size_t full_depth) noexcept {
        if (prefix_terms == 0 || prefix_terms > full_depth) {
            return;
        }
        const double epsilon = static_cast<double>(
            std::numeric_limits<float>::epsilon());
        const double prefix_product =
            static_cast<double>(prefix_terms) * epsilon;
        const double full_product = static_cast<double>(full_depth) * epsilon;
        if (prefix_product >= 1.0 || full_product >= 1.0) {
            return;
        }
        const double gamma_prefix =
            prefix_product / (1.0 - prefix_product);
        const double gamma_full = full_product / (1.0 - full_product);
        scale_ = (1.0 - gamma_full) / (1.0 + gamma_prefix);
        valid_ = true;
    }

    [[nodiscard]] double guarded_threshold(float threshold) const noexcept {
        if (!valid_ || !std::isfinite(threshold)) {
            return std::numeric_limits<double>::infinity();
        }
        return std::nextafter(
            static_cast<double>(threshold),
            std::numeric_limits<double>::infinity());
    }

    [[nodiscard]] bool proves_above(
        float prefix_distance,
        double guarded_threshold_value) const noexcept {
        return valid_ && std::isfinite(prefix_distance) &&
            prefix_distance >= 0.0f &&
            static_cast<double>(prefix_distance) * scale_ >
                guarded_threshold_value;
    }

    [[nodiscard]] bool valid() const noexcept {
        return valid_;
    }

private:
    double scale_ = 0.0;
    bool valid_ = false;
};

// Strict comparison preserves the row-major tie policy.  If the guard cannot
// prove that the baseline full-SIMD SSD is above the complete Top-K threshold,
// the caller must evaluate the full candidate.
inline bool prefix_proves_full_ssd_above(
    float prefix_distance,
    float complete_topk_threshold,
    std::size_t prefix_terms,
    std::size_t full_depth) noexcept {
    if (!std::isfinite(complete_topk_threshold)) {
        return false;
    }
    return conservative_full_ssd_lower_bound(
        prefix_distance, prefix_terms, full_depth) >
        static_cast<double>(complete_topk_threshold);
}

} // namespace wsvt
