#include "wsvt/template_window.hpp"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace wsvt {

namespace {

inline std::size_t idx3(std::size_t c, std::size_t y, std::size_t x, std::size_t h, std::size_t w) {
    return (c * h + y) * w + x;
}

inline std::size_t idx3_out(std::size_t y, std::size_t x, std::size_t d, std::size_t w, std::size_t depth) {
    return (y * w + x) * depth + d;
}

inline std::size_t wrap_index(long long v, std::size_t n) {
    const long long m = static_cast<long long>(n);
    long long r = v % m;
    if (r < 0) {
        r += m;
    }
    return static_cast<std::size_t>(r);
}

}

AlignedVector<float> stack_template_window(
    const std::vector<float>& img,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int n_template,
    std::size_t& out_h,
    std::size_t& out_w,
    std::size_t& out_depth) {
    if (img.size() != ch * h * w) {
        throw std::invalid_argument("stack_template_window size mismatch");
    }
    if (n_template < 0) {
        throw std::invalid_argument("n_template must be >= 0");
    }

    const int axis = 2 * n_template + 1;
    out_h = h;
    out_w = w;
    out_depth = ch * static_cast<std::size_t>(axis * axis);
    AlignedVector<float> out(h * w * out_depth, 0.0f);

    // Match Python loop order:
    //   for dy in axis_Nw:                   # outer: h-axis / row shift
    //       for dx in axis_Nw:               # inner: w-axis / column shift
    //           np.roll(np.roll(img, dy, axis=-2), dx, axis=-1)
    //
    // np.roll(img, x, axis=-1) reads from (col - x) for output col,
    // so we use wrap_index(col - dx, w) and wrap_index(row - dy, h).

    // OpenMP parallel over (dx, dy, c) — each combination is independent
    #pragma omp parallel for collapse(3) schedule(static)
    for (int dy = -n_template; dy <= n_template; ++dy) {
        for (int dx = -n_template; dx <= n_template; ++dx) {
            for (std::size_t c = 0; c < ch; ++c) {
                const std::size_t d = static_cast<std::size_t>(
                    (dy + n_template) * axis + (dx + n_template)) * ch + c;

                // Precompute wrap indices to avoid modulo in inner loop
                std::vector<std::size_t> wy(h), wx(w);
                for (std::size_t i = 0; i < h; ++i) {
                    wy[i] = wrap_index(static_cast<long long>(i) - dy, h);
                }
                for (std::size_t i = 0; i < w; ++i) {
                    wx[i] = wrap_index(static_cast<long long>(i) - dx, w);
                }

                for (std::size_t y = 0; y < h; ++y) {
                    for (std::size_t x = 0; x < w; ++x) {
                        out[idx3_out(y, x, d, w, out_depth)] = img[idx3(c, wy[y], wx[x], h, w)];
                    }
                }
            }
        }
    }
    return out;
}

}
