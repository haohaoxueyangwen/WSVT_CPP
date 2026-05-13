#include "wsvt/solver_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

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
