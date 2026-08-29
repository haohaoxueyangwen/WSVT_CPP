#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#if defined(WSVT_HAS_OPENCV)
#include <opencv2/core.hpp>
#endif

namespace wsvt {

inline constexpr std::string_view kPythonReferenceSemantics = "python_reference_v1";

struct ManualWindowContract {
    int cal_half_window;
    int n_s_extend;
    int template_radius;
};

[[nodiscard]] std::string now_iso8601_utc();

[[nodiscard]] int configure_openmp_threads(
    int requested_cores,
    const char* label,
    int max_active_levels);

[[nodiscard]] std::vector<int> wavelet_add_list_for_depth(std::size_t depth);

void validate_manual_window_contract(
    ManualWindowContract window,
    std::size_t input_h,
    std::size_t input_w,
    int crop,
    std::string_view solver_name);

[[nodiscard]] std::vector<int> derived_search_half_windows(
    int pyramid_level,
    int cal_half_window,
    int n_s_extend);

void apply_python_reference_phase_scale(std::span<float> phase, double scale);

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
