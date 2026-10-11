#include "wsvt/solver_utils.hpp"

#include "wsvt/console_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(WSVT_HAS_OPENCV)
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "wsvt/common.hpp"
#endif

namespace wsvt {

std::string now_iso8601_utc() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto tt = system_clock::to_time_t(now);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &tt);
#else
    gmtime_r(&tt, &tm_utc);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

int configure_openmp_threads(int requested_cores, const char* label, int max_active_levels) {
    const unsigned int hw_cores_u = std::thread::hardware_concurrency();
    int cores = static_cast<int>(hw_cores_u == 0 ? 1 : hw_cores_u);
    if (label != nullptr) {
        prColor(std::string("Computer available cores: ") + std::to_string(cores), "green");
    }
    cores = std::max(1, std::min(cores, std::max(1, requested_cores)));
    if (label != nullptr) {
        prColor(std::string("Use ") + std::to_string(cores) + " cores for " + label, "light_purple");
    }
#ifdef _OPENMP
    omp_set_num_threads(cores);
    omp_set_max_active_levels(std::max(1, max_active_levels));
#endif
    return cores;
}

std::vector<int> wavelet_add_list_for_depth(std::size_t depth) {
    if (depth > 150) {
        return {0, 0, 0, 0, 0, 0};
    }
    if (depth > 50) {
        return {0, 0, 1, 2, 2, 2};
    }
    return {2, 2, 2, 2, 2, 2};
}

void validate_manual_window_contract(
    ManualWindowContract window,
    std::size_t input_h,
    std::size_t input_w,
    int crop,
    std::string_view solver_name) {
    const std::string prefix = std::string(solver_name) + ": ";
    if (input_h == 0 || input_w == 0) {
        throw std::invalid_argument(prefix + "input dimensions must be positive");
    }
    if (crop < 0) {
        throw std::invalid_argument(prefix + "crop must be >= 0; automatic crop sentinels are not supported");
    }
    if (window.cal_half_window <= 0) {
        throw std::invalid_argument(prefix + "cal_half_window must be > 0 and set manually");
    }
    if (window.n_s_extend <= 0) {
        throw std::invalid_argument(prefix + "n_s_extend must be > 0 and set manually");
    }
    if (window.template_radius < 0) {
        throw std::invalid_argument(prefix + "template radius must be >= 0 and set manually");
    }

    std::size_t effective_h = input_h;
    std::size_t effective_w = input_w;
    const std::size_t min_extent = std::min(input_h, input_w);
    if (crop > 0 && static_cast<std::size_t>(crop) < min_extent) {
        effective_h = static_cast<std::size_t>(crop);
        effective_w = static_cast<std::size_t>(crop);
    }
    const std::size_t required_margin = static_cast<std::size_t>(
        window.template_radius + window.cal_half_window);
    if (effective_h <= 2 * required_margin || effective_w <= 2 * required_margin) {
        throw std::invalid_argument(
            prefix + "effective ROI must be larger than twice (template_radius + cal_half_window)");
    }
}

std::vector<int> derived_search_half_windows(
    int pyramid_level,
    int cal_half_window,
    int n_s_extend) {
    if (pyramid_level < 0) {
        throw std::invalid_argument("pyramid_level must be >= 0");
    }
    std::vector<int> windows(static_cast<std::size_t>(pyramid_level), n_s_extend);
    windows.push_back(static_cast<int>(std::ceil(
        static_cast<double>(cal_half_window) /
        std::pow(2.0, static_cast<double>(pyramid_level)))));
    return windows;
}

void apply_python_reference_phase_scale(std::span<float> phase, double scale) {
    for (float& value : phase) {
        value = static_cast<float>(-static_cast<double>(value) * scale);
    }
}

double stddev_2d(const std::vector<float>& arr) {
    if (arr.empty()) return 0.0;
    const double mean = std::accumulate(arr.begin(), arr.end(), 0.0) / static_cast<double>(arr.size());
    double acc = 0.0;
    for (const float v : arr) {
        const double d = static_cast<double>(v) - mean;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(arr.size()));
}

double pv_2d(const std::vector<float>& arr) {
    if (arr.empty()) return 0.0;
    auto [mn_it, mx_it] = std::minmax_element(arr.begin(), arr.end());
    return static_cast<double>(*mx_it) - static_cast<double>(*mn_it);
}

#if defined(WSVT_HAS_OPENCV)
float median_mat_f32(const cv::Mat& m) {
    std::vector<float> vals;
    vals.reserve(static_cast<std::size_t>(m.rows * m.cols));
    for (int y = 0; y < m.rows; ++y) {
        const float* row = m.ptr<float>(y);
        for (int x = 0; x < m.cols; ++x) {
            vals.push_back(row[x]);
        }
    }
    const std::size_t mid = vals.size() / 2;
    std::nth_element(vals.begin(), vals.begin() + static_cast<long long>(mid), vals.end());
    return vals[mid];
}

std::array<std::vector<float>, 2> slope_tracking(
    const float* ref_ch0,
    const float* img_ch0,
    std::size_t h,
    std::size_t w,
    int n_window) {
    cv::Mat ref_mat(static_cast<int>(h), static_cast<int>(w), CV_32F);
    cv::Mat img_mat(static_cast<int>(h), static_cast<int>(w), CV_32F);
    for (std::size_t y = 0; y < h; ++y) {
        float* rr = ref_mat.ptr<float>(static_cast<int>(y));
        float* ir = img_mat.ptr<float>(static_cast<int>(y));
        for (std::size_t x = 0; x < w; ++x) {
            rr[x] = ref_ch0[idx2(y, x, w)];
            ir[x] = img_ch0[idx2(y, x, w)];
        }
    }
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(ref_mat, img_mat, flow, 0.5, 2, n_window, 10, 3, 1.2, 1);
    std::vector<cv::Mat> flow_ch(2);
    cv::split(flow, flow_ch);
    cv::Mat disp_y = -flow_ch[1];
    cv::Mat disp_x = -flow_ch[0];
    cv::GaussianBlur(disp_y, disp_y, cv::Size(0, 0), 2.0, 2.0, cv::BORDER_REFLECT_101);
    cv::GaussianBlur(disp_x, disp_x, cv::Size(0, 0), 2.0, 2.0, cv::BORDER_REFLECT_101);
    const float y_med = static_cast<float>(round_half_to_even(static_cast<double>(median_mat_f32(disp_y))));
    const float x_med = static_cast<float>(round_half_to_even(static_cast<double>(median_mat_f32(disp_x))));
    return {std::vector<float>(h * w, y_med), std::vector<float>(h * w, x_med)};
}
#endif

}  // namespace wsvt
