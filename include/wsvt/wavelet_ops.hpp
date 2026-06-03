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

}  // namespace wsvt
