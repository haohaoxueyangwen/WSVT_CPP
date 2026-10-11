#include "wsvt/filter_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace wsvt {

namespace {

inline std::size_t mirror_index(int i, std::size_t n) {
    if (n == 0) {
        throw std::invalid_argument("mirror_index size must be > 0");
    }
    if (n == 1) {
        return 0;
    }
    int v = i;
    const int nn = static_cast<int>(n);
    while (v < 0 || v >= nn) {
        if (v < 0) {
            v = -v;
        } else {
            v = 2 * nn - v - 2;
        }
    }
    return static_cast<std::size_t>(v);
}

inline std::size_t py_index(int i, std::size_t n) {
    if (n == 0) {
        throw std::invalid_argument("py_index size must be > 0");
    }
    int v = i;
    const int nn = static_cast<int>(n);
    if (v < 0) v += nn;
    if (v < 0 || v >= nn) throw std::out_of_range("py_index out of range");
    return static_cast<std::size_t>(v);
}

}

Image2D<float> filter_erosion(
    ImageView2D<const float> image,
    float val_thresh,
    int filt_sz) {
    if (filt_sz <= 0) {
        throw std::invalid_argument("filt_sz must be > 0");
    }
    const Shape2D s = image.shape();
    const std::size_t h = s.h;
    const std::size_t w = s.w;

    Image2D<float> image_filt(s, 0.0f);
    const int k0 = -(filt_sz / 2);
    const int k1 = k0 + filt_sz - 1;
    std::vector<float> win;
    win.reserve(static_cast<std::size_t>(filt_sz * filt_sz));

    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            win.clear();
            for (int ky = k0; ky <= k1; ++ky) {
                for (int kx = k0; kx <= k1; ++kx) {
                    const std::size_t sy = mirror_index(static_cast<int>(y) + ky, h);
                    const std::size_t sx = mirror_index(static_cast<int>(x) + kx, w);
                    win.push_back(image(sy, sx));
                }
            }
            const auto mid = win.begin() + static_cast<std::ptrdiff_t>(win.size() / 2);
            std::nth_element(win.begin(), mid, win.end());
            image_filt(y, x) = *mid;
        }
    }

    Image2D<float> diff_img(s, 0.0f);
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            diff_img(y, x) = std::fabs(image(y, x) - image_filt(y, x));
        }
    }

    const int ys[12] = {0, 0, 0, 0, 1, 1,
                        static_cast<int>(h) - 2, static_cast<int>(h) - 2,
                        static_cast<int>(h) - 1, static_cast<int>(h) - 1,
                        static_cast<int>(h) - 1, static_cast<int>(h) - 1};
    const int xs[12] = {0, 1, static_cast<int>(w) - 2, static_cast<int>(w) - 1,
                        0, static_cast<int>(w) - 1,
                        0, static_cast<int>(w) - 1,
                        0, 1, static_cast<int>(w) - 2, static_cast<int>(w) - 1};
    for (int i = 0; i < 12; ++i) {
        diff_img(py_index(ys[i], h), py_index(xs[i], w)) = 0.0f;
    }

    Image2D<float> out(s);
    std::memcpy(out.data(), image.data(), s.size() * sizeof(float));
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            if (diff_img(y, x) > val_thresh) {
                out(y, x) = image_filt(y, x);
            }
        }
    }
    return out;
}

}
