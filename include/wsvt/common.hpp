#pragma once

#include "wsvt/image.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace wsvt {

// ---- Flat-array indexing helpers (kept for hot inner loops) ----

/// Row-major 2D index: arr[y * w + x]
inline std::size_t idx2(std::size_t y, std::size_t x, std::size_t w) {
    return y * w + x;
}

/// CHW (channel-first) 3D index: arr[(c * h + y) * w + x]
inline std::size_t idx3(std::size_t c, std::size_t y, std::size_t x,
                        std::size_t h, std::size_t w) {
    return (c * h + y) * w + x;
}

/// Alias for idx3 with clearer naming
inline std::size_t idx_chw(std::size_t c, std::size_t y, std::size_t x,
                           std::size_t h, std::size_t w) {
    return (c * h + y) * w + x;
}

/// HWD (depth-last) 3D index: arr[(y * w + x) * depth + d]
inline std::size_t idx_hwd(std::size_t y, std::size_t x, std::size_t d,
                           std::size_t w, std::size_t depth) {
    return (y * w + x) * depth + d;
}

// ---- Scalar utilities ----

inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

inline double round_half_to_even(double v) {
    if (!std::isfinite(v)) return v;
    const double f = std::floor(v);
    const double frac = v - f;
    if (frac < 0.5) return f;
    if (frac > 0.5) return f + 1.0;
    const long long fi = static_cast<long long>(f);
    return ((fi & 1LL) == 0LL) ? f : (f + 1.0);
}

inline int dwt_max_level_db2(std::size_t data_len) {
    if (data_len <= 3) return 0;
    return static_cast<int>(
        std::floor(std::log2(static_cast<double>(data_len) / 3.0)));
}

inline double cubic_weight(double x) {
    const double a = -0.5;
    const double ax = std::fabs(x);
    if (ax <= 1.0) {
        return (a + 2.0) * ax * ax * ax - (a + 3.0) * ax * ax + 1.0;
    }
    if (ax < 2.0) {
        return a * ax * ax * ax - 5.0 * a * ax * ax + 8.0 * a * ax - 4.0 * a;
    }
    return 0.0;
}

// ---- 2D array operations (View / span based) ----

/// In-place clamp; shape is irrelevant, just iterate the buffer.
inline void clamp_2d(std::span<float> data, float lo, float hi) {
    for (float& v : data) v = clampf(v, lo, hi);
}

/// Arithmetic mean over a flat buffer.
inline double mean_2d(std::span<const float> data) {
    if (data.empty()) return 0.0;
    double s = 0.0;
    for (float v : data) s += static_cast<double>(v);
    return s / static_cast<double>(data.size());
}

/// Crop symmetric border of `pad` pixels. Returns empty image if too small.
inline Image2D<float> crop_2d(ImageView2D<const float> img, std::size_t pad) {
    const Shape2D s = img.shape();
    if (pad == 0) {
        Image2D<float> out(s);
        std::memcpy(out.data(), img.data(), s.size() * sizeof(float));
        return out;
    }
    if (s.h <= 2 * pad || s.w <= 2 * pad) {
        return Image2D<float>(Shape2D{0, 0});
    }
    const Shape2D out_shape{s.h - 2 * pad, s.w - 2 * pad};
    Image2D<float> out(out_shape);
    for (std::size_t y = 0; y < out_shape.h; ++y) {
        std::memcpy(out.row(y),
                    &img(y + pad, pad),
                    out_shape.w * sizeof(float));
    }
    return out;
}

/// Bicubic sample at continuous (y, x) coordinates with edge clamping.
inline double sample_bicubic(ImageView2D<const float> img, double y, double x) {
    const Shape2D s = img.shape();
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    double out = 0.0;
    for (int m = -1; m <= 2; ++m) {
        for (int n = -1; n <= 2; ++n) {
            const int yy = std::clamp(y0 + m, 0, static_cast<int>(s.h) - 1);
            const int xx = std::clamp(x0 + n, 0, static_cast<int>(s.w) - 1);
            const double wy = cubic_weight(y - static_cast<double>(y0 + m));
            const double wx = cubic_weight(x - static_cast<double>(x0 + n));
            out += static_cast<double>(
                       img(static_cast<std::size_t>(yy),
                           static_cast<std::size_t>(xx))) *
                   wy * wx;
        }
    }
    return out;
}

// ---- HWD tensor operations ----

/// Depth-wise population standard deviation for a CHW tensor; produces a 2D image.
inline Image2D<float> std_depth_chw_per_pixel(
    std::span<const float> in,
    std::size_t ch,
    std::size_t h,
    std::size_t w) {
    if (in.size() != ch * h * w) {
        throw std::invalid_argument("std_depth_chw_per_pixel size mismatch");
    }
    Image2D<float> out(Shape2D{h, w}, 0.0f);
    const float* __restrict__ src = in.data();
    float* __restrict__ dst = out.data();
    const std::size_t plane = h * w;
    const float inv_depth = 1.0f / static_cast<float>(ch);
    std::vector<float> mean(plane, 0.0f);

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        std::size_t begin = 0;
        std::size_t end = plane;
#ifdef _OPENMP
        const auto tid = static_cast<std::size_t>(omp_get_thread_num());
        const auto nthreads = static_cast<std::size_t>(omp_get_num_threads());
        begin = (plane * tid) / nthreads;
        end = (plane * (tid + 1)) / nthreads;
#endif

        for (std::size_t c = 0; c < ch; ++c) {
            const float* __restrict__ src_plane = src + c * plane;
            #pragma omp simd
            for (std::size_t idx = begin; idx < end; ++idx) {
                mean[idx] += src_plane[idx];
            }
        }

        #pragma omp simd
        for (std::size_t idx = begin; idx < end; ++idx) {
            mean[idx] *= inv_depth;
            dst[idx] = 0.0f;
        }

        for (std::size_t c = 0; c < ch; ++c) {
            const float* __restrict__ src_plane = src + c * plane;
            #pragma omp simd
            for (std::size_t idx = begin; idx < end; ++idx) {
                const float d = src_plane[idx] - mean[idx];
                dst[idx] += d * d;
            }
        }

        #pragma omp simd
        for (std::size_t idx = begin; idx < end; ++idx) {
            dst[idx] = std::sqrt(dst[idx] * inv_depth);
        }
    }
    return out;
}

/// Zero-pad an HWD tensor by (pad_y, pad_x); depth is preserved.
inline Tensor3D<float, Layout::HWD> pad_hwd_zero(
    TensorView3D<const float, Layout::HWD> in,
    std::size_t pad_y, std::size_t pad_x) {
    const Shape3D s = in.shape();  // d0=h, d1=w, d2=depth
    const Shape3D out_shape{s.d0 + 2 * pad_y, s.d1 + 2 * pad_x, s.d2};
    Tensor3D<float, Layout::HWD> out(out_shape, 0.0f);
    const std::size_t in_row_bytes = s.d1 * s.d2 * sizeof(float);
    const std::size_t out_stride = out_shape.d1 * out_shape.d2;
    const std::size_t in_stride = s.d1 * s.d2;

    #pragma omp parallel for schedule(static) if(s.d0 > 64)
    for (std::size_t y = 0; y < s.d0; ++y) {
        std::memcpy(out.data() + (y + pad_y) * out_stride + pad_x * s.d2,
                    in.data() + y * in_stride,
                    in_row_bytes);
    }
    return out;
}

/// Slice rows [y0, y1) from an HWD tensor. Width/depth preserved.
inline Tensor3D<float, Layout::HWD> slice_hwd_y(
    TensorView3D<const float, Layout::HWD> in,
    std::size_t y0, std::size_t y1) {
    const Shape3D s = in.shape();
    const std::size_t out_h = y1 - y0;
    Tensor3D<float, Layout::HWD> out(Shape3D{out_h, s.d1, s.d2}, 0.0f);
    const std::size_t row_bytes = s.d1 * s.d2 * sizeof(float);
    const std::size_t stride = s.d1 * s.d2;

    #pragma omp parallel for schedule(static) if(out_h > 64)
    for (std::size_t y = 0; y < out_h; ++y) {
        std::memcpy(out.data() + y * stride,
                    in.data() + (y + y0) * stride,
                    row_bytes);
    }
    return out;
}

/// Depth-wise population standard deviation; produces a 2D image.
inline Image2D<float> std_depth_hwd(TensorView3D<const float, Layout::HWD> in) {
    const Shape3D s = in.shape();
    Image2D<float> out(Shape2D{s.d0, s.d1}, 0.0f);
    const float* __restrict__ src = in.data();
    float* __restrict__ dst = out.data();
    const std::size_t plane = s.d0 * s.d1;
    const std::size_t depth = s.d2;
    const float inv_depth = 1.0f / static_cast<float>(depth);

    #pragma omp parallel for schedule(static)
    for (std::size_t idx = 0; idx < plane; ++idx) {
        const float* row = src + idx * depth;
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (std::size_t k = 0; k < depth; ++k) sum += row[k];
        const float mean = sum * inv_depth;
        float var = 0.0f;
        #pragma omp simd reduction(+:var)
        for (std::size_t k = 0; k < depth; ++k) {
            const float d = row[k] - mean;
            var += d * d;
        }
        dst[idx] = std::sqrt(var * inv_depth);
    }
    return out;
}

}  // namespace wsvt
