#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#if defined(WSVT_HAS_OPENCV)
#include <opencv2/core.hpp>
#endif

namespace wsvt {

[[nodiscard]] std::string now_iso8601_utc();

[[nodiscard]] double stddev_2d(const std::vector<float>& arr);

[[nodiscard]] double pv_2d(const std::vector<float>& arr);

#if defined(WSVT_HAS_OPENCV)
[[nodiscard]] float median_mat_f32(const cv::Mat& m);

[[nodiscard]] std::array<std::vector<float>, 2> slope_tracking(
    const float* ref_ch0,
    const float* img_ch0,
    std::size_t h,
    std::size_t w,
    int n_window);
#endif

}  // namespace wsvt
