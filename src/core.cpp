#include "wsvt/core.hpp"

#include "wsvt/common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace wsvt {

DispResult find_disp(
    ImageView2D<const float> corr,
    std::span<const float> xx_axis,
    std::span<const float> yy_axis,
    bool sub_resolution) {
    const Shape2D s = corr.shape();
    if (s.size() == 0) {
        throw std::invalid_argument("correlation map is empty");
    }
    if (xx_axis.size() != s.size() || yy_axis.size() != s.size()) {
        throw std::invalid_argument("axis shape mismatch");
    }
    if (s.w < 2 || s.h < 2) {
        throw std::invalid_argument("find_disp requires corr shape >= 2x2");
    }

    const std::size_t h = s.h;
    const std::size_t w = s.w;

    float corr_max = -std::numeric_limits<float>::infinity();
    std::size_t max_y = 0;
    std::size_t max_x = 0;
    float sum_abs = 0.0f;
    for (std::size_t yy = 0; yy < h; ++yy) {
        for (std::size_t xx = 0; xx < w; ++xx) {
            const float v = corr(yy, xx);
            sum_abs += std::fabs(v);
            if (v > corr_max) {
                corr_max = v;
                max_y = yy;
                max_x = xx;
            }
        }
    }

    const std::size_t n = h * w;
    const float avg = n > 1 ? (sum_abs - std::fabs(corr_max)) / static_cast<float>(n - 1) : 0.0f;
    const float sn_ratio = avg != 0.0f ? corr_max / avg : 0.0f;

    const auto sample = [&](long long y, long long x) -> float {
        const auto yc = static_cast<std::size_t>(std::clamp<long long>(y, 0, static_cast<long long>(h - 1)));
        const auto xc = static_cast<std::size_t>(std::clamp<long long>(x, 0, static_cast<long long>(w - 1)));
        return corr(yc, xc);
    };

    const auto my = static_cast<long long>(max_y);
    const auto mx = static_cast<long long>(max_x);

    const float c_m10 = sample(my - 1, mx);
    const float c_p10 = sample(my + 1, mx);
    const float c_0m1 = sample(my, mx - 1);
    const float c_0p1 = sample(my, mx + 1);
    const float c_00 = sample(my, mx);
    const float c_pp = sample(my + 1, mx + 1);
    const float c_pm = sample(my + 1, mx - 1);
    const float c_mp = sample(my - 1, mx + 1);
    const float c_mm = sample(my - 1, mx - 1);

    const float dy = (c_p10 - c_m10) / 2.0f;
    const float dyy = c_p10 + c_m10 - 2.0f * c_00;
    const float dx = (c_0p1 - c_0m1) / 2.0f;
    const float dxx = c_0p1 + c_0m1 - 2.0f * c_00;
    const float dxy = (c_pp - c_pm - c_mp + c_mm) / 4.0f;

    const float denom = dxx * dyy - dxy * dxy;
    const float det = denom != 0.0f ? 1.0f / denom : 0.0f;

    const float pixel_res_x = xx_axis[idx2(0, 1, w)] - xx_axis[idx2(0, 0, w)];
    const float pixel_res_y = yy_axis[idx2(1, 0, w)] - yy_axis[idx2(0, 0, w)];

    const float minor_disp_x = (-(dyy * dx - dxy * dy) * det) * pixel_res_x;
    const float minor_disp_y = (-(dxx * dy - dxy * dx) * det) * pixel_res_y;

    float disp_x = xx_axis[idx2(max_y, max_x, w)];
    float disp_y = yy_axis[idx2(max_y, max_x, w)];

    if (sub_resolution) {
        disp_x += minor_disp_x;
        disp_y += minor_disp_y;
    }

    const float max_axis_x = xx_axis[idx2(0, w - 1, w)];
    const float min_axis_x = xx_axis[idx2(0, 0, w)];
    const float max_axis_y = yy_axis[idx2(h - 1, 0, w)];
    const float min_axis_y = yy_axis[idx2(0, 0, w)];

    if (disp_x > max_axis_x) {
        disp_x = max_axis_x;
    } else if (disp_x < min_axis_x) {
        disp_x = min_axis_x;
    }

    if (disp_y > max_axis_y) {
        disp_y = max_axis_y;
    } else if (disp_y < min_axis_y) {
        disp_y = min_axis_y;
    }

    return DispResult{disp_y, disp_x, sn_ratio, corr_max};
}

std::array<std::size_t, 3> infer_dims(
    std::size_t flat_size,
    std::size_t depth,
    std::size_t width) {
    if (depth == 0 || width == 0) {
        throw std::invalid_argument("depth and width must be positive");
    }
    const std::size_t plane = depth * width;
    if (flat_size % plane != 0) {
        throw std::invalid_argument("flat size cannot be inferred");
    }
    return {flat_size / plane, width, depth};
}

}
