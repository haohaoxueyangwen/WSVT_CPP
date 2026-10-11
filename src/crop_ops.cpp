#include "wsvt/crop_ops.hpp"

#include "wsvt/common.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace wsvt {

namespace {

std::pair<int, int> py_slice_to_range(int start, int stop, int n) {
    int s = start;
    int e = stop;
    if (s < 0) s += n;
    if (e < 0) e += n;
    s = std::max(0, std::min(s, n));
    e = std::max(0, std::min(e, n));
    if (e < s) e = s;
    return {s, e};
}

std::pair<double, double> center_of_mass_binary(
    ImageView2D<const float> img_seg) {
    const Shape2D s = img_seg.shape();
    double mass = 0.0;
    double sum_y = 0.0;
    double sum_x = 0.0;
    for (std::size_t y = 0; y < s.h; ++y) {
        for (std::size_t x = 0; x < s.w; ++x) {
            const double v = static_cast<double>(img_seg(y, x));
            mass += v;
            sum_y += static_cast<double>(y) * v;
            sum_x += static_cast<double>(x) * v;
        }
    }
    if (mass == 0.0) {
        throw std::runtime_error("center_of_mass failed: zero mass");
    }
    return {sum_y / mass, sum_x / mass};
}

std::vector<std::pair<int, int>> where_zero_local(
    ImageView2D<const float> img_seg,
    int y_start, int y_stop,
    int x_start, int x_stop) {
    const Shape2D s = img_seg.shape();
    const auto yr = py_slice_to_range(y_start, y_stop, static_cast<int>(s.h));
    const auto xr = py_slice_to_range(x_start, x_stop, static_cast<int>(s.w));
    std::vector<std::pair<int, int>> pos;
    for (int yy = yr.first; yy < yr.second; ++yy) {
        for (int xx = xr.first; xx < xr.second; ++xx) {
            if (img_seg(static_cast<std::size_t>(yy),
                        static_cast<std::size_t>(xx)) == 0.0f) {
                pos.push_back({yy - yr.first, xx - xr.first});
            }
        }
    }
    return pos;
}

int amax_second(const std::vector<std::pair<int, int>>& pos) {
    if (pos.empty()) throw std::runtime_error("np.amax on empty array");
    int v = std::numeric_limits<int>::min();
    for (const auto& p : pos) v = std::max(v, p.second);
    return v;
}

int amin_second(const std::vector<std::pair<int, int>>& pos) {
    if (pos.empty()) throw std::runtime_error("np.amin on empty array");
    int v = std::numeric_limits<int>::max();
    for (const auto& p : pos) v = std::min(v, p.second);
    return v;
}

int amax_first(const std::vector<std::pair<int, int>>& pos) {
    if (pos.empty()) throw std::runtime_error("np.amax on empty array");
    int v = std::numeric_limits<int>::min();
    for (const auto& p : pos) v = std::max(v, p.first);
    return v;
}

int amin_first(const std::vector<std::pair<int, int>>& pos) {
    if (pos.empty()) throw std::runtime_error("np.amin on empty array");
    int v = std::numeric_limits<int>::max();
    for (const auto& p : pos) v = std::min(v, p.first);
    return v;
}

}

AutoCropResult auto_crop(
    ImageView2D<const float> img,
    double shrink,
    std::optional<float> count) {
    const Shape2D s = img.shape();

    double thr = 0.0;
    if (count.has_value()) {
        thr = static_cast<double>(*count);
    } else {
        thr = mean_2d(img.flat());
    }

    Image2D<float> img_seg(s, 0.0f);
    for (std::size_t i = 0; i < s.size(); ++i) {
        img_seg.data()[i] = static_cast<double>(img.data()[i]) > thr ? 1.0f : 0.0f;
    }

    ImageView2D<const float> seg_view(img_seg.data(), s);
    const auto cen = center_of_mass_binary(seg_view);
    const int cen_x = static_cast<int>(cen.first);
    const int cen_y = static_cast<int>(cen.second);

    const int n_width = 50;
    auto pos = where_zero_local(seg_view, cen_y - n_width, cen_y + n_width, 0, cen_x);
    const int left_x = amax_second(pos);

    pos = where_zero_local(seg_view, cen_y - n_width, cen_y + n_width, cen_x, static_cast<int>(s.w));
    const int right_x = amin_second(pos) + cen_x;

    pos = where_zero_local(seg_view, 0, cen_y, cen_x - n_width, cen_x + n_width);
    const int up_y = amax_first(pos);

    pos = where_zero_local(seg_view, cen_y, static_cast<int>(s.h), cen_x - n_width, cen_x + n_width);
    const int down_y = amin_first(pos) + cen_y;

    const int x_width = static_cast<int>(shrink * static_cast<double>(right_x - left_x) / 2.0);
    const int y_width = static_cast<int>(shrink * static_cast<double>(down_y - up_y) / 2.0);
    const int x_cen = static_cast<int>(static_cast<double>(right_x + left_x) / 2.0);
    const int y_cen = static_cast<int>(static_cast<double>(down_y + up_y) / 2.0);

    std::cout << "auto-crop. center: " << y_cen << " y " << x_cen
              << " x, width: " << y_width << " y " << x_width << " x"
              << std::endl;

    return AutoCropResult{
        y_cen - y_width,
        y_cen + y_width,
        x_cen - x_width,
        x_cen + x_width
    };
}

}
