#include "wsvt/image_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace wsvt {

namespace {

inline long long round_even_ll(double v) {
    return static_cast<long long>(std::nearbyint(v));
}

}

Image2D<float> image_roi(ImageView2D<const float> img, std::size_t m) {
    const Shape2D s = img.shape();
    if (m == 0 || m > std::min(s.h, s.w)) {
        // Python 语义：m 越界时原样返回
        Image2D<float> out(s);
        std::memcpy(out.data(), img.data(), s.size() * sizeof(float));
        return out;
    }

    const long long y0 = round_even_ll(static_cast<double>(s.h) / 2.0) -
                         round_even_ll(static_cast<double>(m) / 2.0);
    const long long x0 = round_even_ll(static_cast<double>(s.w) / 2.0) -
                         round_even_ll(static_cast<double>(m) / 2.0);

    Image2D<float> out(Shape2D{m, m});
    for (std::size_t y = 0; y < m; ++y) {
        std::memcpy(out.row(y),
                    &img(static_cast<std::size_t>(y0) + y,
                         static_cast<std::size_t>(x0)),
                    m * sizeof(float));
    }
    return out;
}

Tensor3D<float, Layout::CHW> image_roi(
    TensorView3D<const float, Layout::CHW> img, std::size_t m) {
    const Shape3D s = img.shape();  // d0=ch, d1=h, d2=w
    if (m == 0 || m > std::min(s.d1, s.d2)) {
        Tensor3D<float, Layout::CHW> out(s);
        std::memcpy(out.data(), img.data(), s.size() * sizeof(float));
        return out;
    }

    const long long y0 = round_even_ll(static_cast<double>(s.d1) / 2.0) -
                         round_even_ll(static_cast<double>(m) / 2.0);
    const long long x0 = round_even_ll(static_cast<double>(s.d2) / 2.0) -
                         round_even_ll(static_cast<double>(m) / 2.0);

    Tensor3D<float, Layout::CHW> out(Shape3D{s.d0, m, m});
    for (std::size_t c = 0; c < s.d0; ++c) {
        for (std::size_t y = 0; y < m; ++y) {
            std::memcpy(&out(c, y, 0),
                        &img(c, static_cast<std::size_t>(y0) + y,
                             static_cast<std::size_t>(x0)),
                        m * sizeof(float));
        }
    }
    return out;
}

}
