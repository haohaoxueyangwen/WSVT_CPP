#pragma once

#include "wsvt/aligned_alloc.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace wsvt {

enum class WaveletFamily : uint8_t {
    Db2 = 0,
    Db3 = 1,
    Db6 = 2,
};

[[nodiscard]] inline std::string_view wavelet_family_name(WaveletFamily f) noexcept {
    switch (f) {
        case WaveletFamily::Db2: return "db2";
        case WaveletFamily::Db3: return "db3";
        case WaveletFamily::Db6: return "db6";
    }
    return "unknown";
}

[[nodiscard]] inline WaveletFamily parse_wavelet_family(std::string_view s) {
    if (s == "db2") return WaveletFamily::Db2;
    if (s == "db3") return WaveletFamily::Db3;
    if (s == "db6") return WaveletFamily::Db6;
    throw std::invalid_argument(std::string("unsupported wavelet method: ") + std::string(s));
}

struct WaveletResult {
    AlignedVector<float> coeffs_filter;
    std::vector<std::string> level_name;
    std::size_t out_h{};
    std::size_t out_w{};
    std::size_t out_depth{};
};

struct WaveletTaskResult {
    AlignedVector<float> coeffs_filter;
    std::vector<std::string> level_name;
    std::vector<std::size_t> y_list;
    std::size_t out_h{};
    std::size_t out_w{};
    std::size_t out_depth{};
};

[[nodiscard]] WaveletResult wavelet_transform(
    std::span<const float> img,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    WaveletFamily wavelet = WaveletFamily::Db6,
    int w_level = 1,
    int return_level = 1);

// Legacy CHW APIs kept for compatibility. Prefer wavelet_transform_hwd for new code.
[[nodiscard]] WaveletTaskResult wavedec_func(
    std::span<const float> img,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    const std::vector<std::size_t>& y_list,
    WaveletFamily wavelet = WaveletFamily::Db6,
    int w_level = 5,
    int return_level = 4);

[[nodiscard]] WaveletResult wavelet_transform_multiprocess(
    std::span<const float> img,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int n_cores,
    WaveletFamily wavelet = WaveletFamily::Db6,
    int w_level = 1,
    int return_level = 1);

// Overloads accepting string for CLI/Python boundary compatibility
[[nodiscard]] inline WaveletResult wavelet_transform(
    std::span<const float> img,
    std::size_t ch, std::size_t h, std::size_t w,
    const std::string& wavelet_method,
    int w_level = 1, int return_level = 1) {
    return wavelet_transform(img, ch, h, w, parse_wavelet_family(wavelet_method), w_level,
                             return_level);
}

[[nodiscard]] inline WaveletResult wavelet_transform_multiprocess(
    std::span<const float> img,
    std::size_t ch, std::size_t h, std::size_t w,
    int n_cores, const std::string& wavelet_method,
    int w_level = 1, int return_level = 1) {
    return wavelet_transform_multiprocess(img, ch, h, w, n_cores,
                                          parse_wavelet_family(wavelet_method), w_level,
                                          return_level);
}

/// Precomputed DWT tap for a single (input_channel, filter_coeff) pair.
struct DwtTap {
    std::size_t ic;   ///< source channel index (input depth dimension)
    float lo;         ///< precomputed low-pass coefficient (already time-reversed)
    float hi;         ///< precomputed high-pass coefficient (already time-reversed)
};

/// Precomputed DWT decomposition plan for a single level.
/// Eliminates per-pixel boundary checks and filter-index reversals.
struct DwtLevelPlan {
    std::vector<std::vector<DwtTap>> oc_taps;  ///< oc_taps[oc] = list of non-zero taps
};

/// Build DWT decomposition plans for all levels.  The plan encodes the exact
/// convolution pattern used by wavelet_transform_hwd, so applying a plan
/// produces bitwise-identical results.
[[nodiscard]] std::vector<DwtLevelPlan> compute_wavelet_plan(
    std::size_t depth_in,
    int w_level,
    WaveletFamily wavelet);

/// Pixelchain wavelet transform: each pixel's full DWT chain runs in
/// thread-local scratch (~1 KB), eliminating all global intermediate
/// approx buffers.  Single OpenMP parallel-for, no internal barriers.
/// Mathematically identical to wavelet_transform_hwd_planned.
[[nodiscard]] WaveletResult wavelet_transform_hwd_pixelchain(
    std::span<const float> img_hwd,
    std::size_t h,
    std::size_t w,
    std::size_t depth_in,
    WaveletFamily wavelet,
    int w_level,
    int return_level,
    const std::vector<DwtLevelPlan>& plans);

/// HWD-native wavelet transform using precomputed plans and a single OpenMP
/// parallel region.  Mathematically identical to wavelet_transform_hwd.
[[nodiscard]] WaveletResult wavelet_transform_hwd_planned(
    std::span<const float> img_hwd,
    std::size_t h,
    std::size_t w,
    std::size_t depth_in,
    WaveletFamily wavelet,
    int w_level,
    int return_level,
    const std::vector<DwtLevelPlan>& plans);

/// HWD-native wavelet transform: input is [h, w, depth_in] with depth contiguous per pixel.
/// Decomposes along the depth axis. Output is [h, w, out_depth] in HWD layout.
[[nodiscard]] WaveletResult wavelet_transform_hwd(
    std::span<const float> img_hwd,
    std::size_t h,
    std::size_t w,
    std::size_t depth_in,
    WaveletFamily wavelet = WaveletFamily::Db2,
    int w_level = 1,
    int return_level = 1);

/// Strict-equivalent streaming HWD transform. It computes every approximation
/// needed by the decomposition chain, but writes only returned coefficients.
[[nodiscard]] WaveletResult wavelet_transform_hwd_streamed(
    std::span<const float> img_hwd,
    std::size_t h,
    std::size_t w,
    std::size_t depth_in,
    WaveletFamily wavelet = WaveletFamily::Db2,
    int w_level = 1,
    int return_level = 1);

/// Holds two independent WaveletResult objects from a pair transform.
struct WaveletPairResult {
    WaveletResult img;
    WaveletResult ref;
};

/// Final-depth DWT basis used by the default-off B2 incremental-temporal
/// prototype. The prefix descriptor uses retained coefficients of the final
/// transform, with not-yet-observed frames set to zero. This makes every
/// retained coefficient extendable without treating an independently
/// normalized short-prefix DWT as a truncation of the final DWT.
struct PrefixCompatibleWaveletBasis {
    std::size_t final_depth{};
    std::size_t out_depth{};
    WaveletFamily wavelet{WaveletFamily::Db2};
    int wavelet_level{};
    int return_level{};
    // Frame-major [final_depth, out_depth] linear weights.
    std::vector<double> frame_coefficient_weights;
    // [final_depth + 1, out_depth], including the zero-frame row.
    std::vector<double> prefix_weight_sums;
    std::vector<std::string> level_name;
};

/// Per-pixel raw sufficient statistics and final-basis weighted sums. Pixels
/// may be extended to different frame counts so a later B2 stage can update
/// only unresolved sample/reference support.
struct PrefixCompatibleWaveletState {
    std::size_t h{};
    std::size_t w{};
    std::size_t final_depth{};
    std::size_t out_depth{};
    std::vector<std::uint16_t> frames_accumulated;
    std::vector<double> raw_sum;
    std::vector<double> raw_sum_squares;
    std::vector<double> weighted_raw_sum;
    std::uint64_t raw_frame_pixel_terms_accumulated{};
    std::uint64_t weighted_coefficient_terms_accumulated{};
};

[[nodiscard]] PrefixCompatibleWaveletBasis make_prefix_compatible_wavelet_basis(
    std::size_t final_depth,
    WaveletFamily wavelet,
    int wavelet_level,
    int return_level);

[[nodiscard]] PrefixCompatibleWaveletState make_prefix_compatible_wavelet_state(
    std::size_t h,
    std::size_t w,
    const PrefixCompatibleWaveletBasis& basis);

/// Extend selected pixels from their current prefix to target_frames. Empty
/// selected_mask means all pixels. Input remains the authoritative final-depth
/// CHW stack; no raw frame is copied into the state.
void extend_prefix_compatible_wavelet_state(
    PrefixCompatibleWaveletState& state,
    const PrefixCompatibleWaveletBasis& basis,
    std::span<const float> data_chw,
    std::size_t target_frames,
    std::span<const std::uint8_t> selected_mask = {});

/// Materialize normalized final-basis coefficients in HWD order. Unselected
/// pixels are zero-filled, matching the sparse-descriptor convention.
[[nodiscard]] WaveletResult materialize_prefix_compatible_wavelet(
    const PrefixCompatibleWaveletState& state,
    const PrefixCompatibleWaveletBasis& basis,
    std::span<const std::uint8_t> selected_mask = {});

/// Pair wavelet transform: processes img and ref HWD stacks in a single
/// OpenMP traversal, sharing filter coefficients and geometry iteration.
/// Produces identical results to two separate wavelet_transform_hwd calls.
[[nodiscard]] WaveletPairResult wavelet_transform_hwd_pair(
    std::span<const float> img_hwd,
    std::span<const float> ref_hwd,
    std::size_t h,
    std::size_t w,
    std::size_t depth_in,
    WaveletFamily wavelet = WaveletFamily::Db2,
    int w_level = 1,
    int return_level = 1);

[[nodiscard]] WaveletPairResult wavelet_transform_hwd_pair_streamed(
    std::span<const float> img_hwd,
    std::span<const float> ref_hwd,
    std::size_t h,
    std::size_t w,
    std::size_t depth_in,
    WaveletFamily wavelet = WaveletFamily::Db2,
    int w_level = 1,
    int return_level = 1);

}  // namespace wsvt
