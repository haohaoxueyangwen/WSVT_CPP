#include "wsvt/pyramid.hpp"
#include "wsvt/console_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace wsvt {

namespace {

inline std::size_t idx3(std::size_t c, std::size_t y, std::size_t x, std::size_t h, std::size_t w) {
    return (c * h + y) * w + x;
}

inline std::size_t idx_feat(std::size_t y, std::size_t x, std::size_t d, std::size_t w, std::size_t depth) {
    return (y * w + x) * depth + d;
}

inline std::size_t idx_chw(std::size_t d, std::size_t y, std::size_t x, std::size_t h, std::size_t w) {
    return (d * h + y) * w + x;
}

inline std::size_t wrap_index(long long v, std::size_t n) {
    const long long m = static_cast<long long>(n);
    long long r = v % m;
    if (r < 0) r += m;
    return static_cast<std::size_t>(r);
}

std::vector<float> downsample2x_mean(
    const std::vector<float>& in,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    std::size_t& out_h,
    std::size_t& out_w) {
    if (in.size() != ch * h * w) {
        throw std::invalid_argument("downsample2x_mean size mismatch");
    }

    out_h = (h + 1) / 2;
    out_w = (w + 1) / 2;
    std::vector<float> out(ch * out_h * out_w, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t c = 0; c < ch; ++c) {
        for (std::size_t oy = 0; oy < out_h; ++oy) {
            for (std::size_t ox = 0; ox < out_w; ++ox) {
                const std::size_t y0 = oy * 2;
                const std::size_t x0 = ox * 2;
                float sum = 0.0f;
                std::size_t cnt = 0;
                for (std::size_t ky = 0; ky < 2; ++ky) {
                    for (std::size_t kx = 0; kx < 2; ++kx) {
                        const std::size_t sy = y0 + ky;
                        const std::size_t sx = x0 + kx;
                        if (sy < h && sx < w) {
                            sum += in[idx3(c, sy, sx, h, w)];
                            ++cnt;
                        }
                    }
                }
                out[idx3(c, oy, ox, out_h, out_w)] = cnt == 0 ? 0.0f : sum / static_cast<float>(cnt);
            }
        }
    }
    return out;
}

std::vector<float> downsample2x_db3_aa(
    const std::vector<float>& in,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    std::size_t& out_h,
    std::size_t& out_w) {
    if (in.size() != ch * h * w) {
        throw std::invalid_argument("downsample2x_db3_aa size mismatch");
    }

    // db3 low-pass decomposition filter (reversed for convolution, matching pywt convention)
    static constexpr float kDb3LoD[6] = {
        0.3326705530f, 0.8068915093f, 0.4598775021f,
        -0.1350110200f, -0.0854412739f, 0.0352262919f
    };
    static constexpr int kFilterLen = 6;
    static constexpr int kOffset = kFilterLen - 2;  // = 4

    // Output size matches pywt: floor((n + filter_len - 1) / 2)
    out_h = static_cast<std::size_t>((static_cast<int>(h) + kFilterLen - 1) / 2);
    out_w = static_cast<std::size_t>((static_cast<int>(w) + kFilterLen - 1) / 2);

    // Separable 2D DWT: first filter along rows (axis=-1), then along columns (axis=-2)
    // Step 1: Row-wise filtering → intermediate shape [ch, h, out_w]
    std::vector<float> row_filtered(ch * h * out_w, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t c = 0; c < ch; ++c) {
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t ox = 0; ox < out_w; ++ox) {
                const int sx = static_cast<int>(ox) * 2;
                float s = 0.0f;
                for (int j = 0; j < kFilterLen; ++j) {
                    const int ix = sx + j - kOffset;
                    const std::size_t wrapped_x = wrap_index(static_cast<long long>(ix), w);
                    s += kDb3LoD[j] * in[idx3(c, y, wrapped_x, h, w)];
                }
                row_filtered[(c * h + y) * out_w + ox] = s;
            }
        }
    }

    // Step 2: Column-wise filtering → final shape [ch, out_h, out_w]
    std::vector<float> out(ch * out_h * out_w, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t c = 0; c < ch; ++c) {
        for (std::size_t oy = 0; oy < out_h; ++oy) {
            const int sy = static_cast<int>(oy) * 2;
            for (std::size_t ox = 0; ox < out_w; ++ox) {
                float s = 0.0f;
                for (int j = 0; j < kFilterLen; ++j) {
                    const int iy = sy + j - kOffset;
                    const std::size_t wrapped_y = wrap_index(static_cast<long long>(iy), h);
                    s += kDb3LoD[j] * row_filtered[(c * h + wrapped_y) * out_w + ox];
                }
                out[idx3(c, oy, ox, out_h, out_w)] = s;
            }
        }
    }
    return out;
}

std::vector<float> downsample_by_mode(
    const std::vector<float>& in,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    std::size_t& out_h,
    std::size_t& out_w,
    PyramidDownsampleMode mode) {
    if (mode == PyramidDownsampleMode::Mean2x2) {
        return downsample2x_mean(in, ch, h, w, out_h, out_w);
    }
    return downsample2x_db3_aa(in, ch, h, w, out_h, out_w);
}

AlignedVector<float> stack_template_window_chw(
    const std::vector<float>& img,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int n_template,
    std::size_t& out_ch) {
    if (img.size() != ch * h * w) {
        throw std::invalid_argument("stack_template_window_chw size mismatch");
    }
    if (n_template < 0) {
        throw std::invalid_argument("n_template must be >= 0");
    }
    const int axis = 2 * n_template + 1;
    out_ch = ch * static_cast<std::size_t>(axis * axis);
    AlignedVector<float> out(out_ch * h * w, 0.0f);

    // OpenMP parallel over (dx, dy, c) — each combination is independent
    #pragma omp parallel for collapse(3) schedule(static)
    for (int dx = -n_template; dx <= n_template; ++dx) {
        for (int dy = -n_template; dy <= n_template; ++dy) {
            for (std::size_t c = 0; c < ch; ++c) {
                const std::size_t d = static_cast<std::size_t>(
                    (dx + n_template) * axis + (dy + n_template)) * ch + c;

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
                        out[idx_chw(d, y, x, h, w)] = img[idx3(c, wy[y], wx[x], h, w)];
                    }
                }
            }
        }
    }
    return out;
}

void normalize_feature_depth_chw(AlignedVector<float>& feature, std::size_t depth, std::size_t h, std::size_t w) {
    if (feature.size() != depth * h * w) {
        throw std::invalid_argument("normalize_feature_depth_chw size mismatch");
    }
    constexpr float kEps = 1e-6f;

    // OpenMP parallel over pixels
    #pragma omp parallel for schedule(static)
    for (std::size_t pixel = 0; pixel < h * w; ++pixel) {
        const std::size_t y = pixel / w;
        const std::size_t x = pixel % w;

        float mean = 0.0f;
        for (std::size_t d = 0; d < depth; ++d) {
            mean += feature[idx_chw(d, y, x, h, w)];
        }
        mean /= static_cast<float>(depth);

        float var = 0.0f;
        for (std::size_t d = 0; d < depth; ++d) {
            const float v = feature[idx_chw(d, y, x, h, w)] - mean;
            var += v * v;
        }
        var /= static_cast<float>(depth);
        const float stdv = std::sqrt(var);

        for (std::size_t d = 0; d < depth; ++d) {
            auto& cell = feature[idx_chw(d, y, x, h, w)];
            cell = (cell - mean) / (stdv + kEps);
        }
    }
}

// HWD-native template window stacking: input CHW, output HWD
AlignedVector<float> stack_template_window_hwd(
    const std::vector<float>& img,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int n_template,
    std::size_t& out_depth) {
    if (img.size() != ch * h * w) {
        throw std::invalid_argument("stack_template_window_hwd size mismatch");
    }
    if (n_template < 0) {
        throw std::invalid_argument("n_template must be >= 0");
    }
    const int axis = 2 * n_template + 1;
    out_depth = ch * static_cast<std::size_t>(axis * axis);
    AlignedVector<float> out(h * w * out_depth, 0.0f);

    // Fast path: n_template=0 means no spatial shifts — just CHW->HWD transpose.
    if (n_template == 0) {
        const std::size_t plane = h * w;
        #pragma omp parallel for schedule(static)
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            float* dst = out.data() + pixel * ch;
            for (std::size_t c = 0; c < ch; ++c) {
                dst[c] = img[c * plane + pixel];
            }
        }
        return out;
    }

    #pragma omp parallel for collapse(3) schedule(static)
    for (int dx = -n_template; dx <= n_template; ++dx) {
        for (int dy = -n_template; dy <= n_template; ++dy) {
            for (std::size_t c = 0; c < ch; ++c) {
                const std::size_t d = static_cast<std::size_t>(
                    (dx + n_template) * axis + (dy + n_template)) * ch + c;

                std::vector<std::size_t> wy(h), wx(w);
                for (std::size_t i = 0; i < h; ++i) {
                    wy[i] = wrap_index(static_cast<long long>(i) - dy, h);
                }
                for (std::size_t i = 0; i < w; ++i) {
                    wx[i] = wrap_index(static_cast<long long>(i) - dx, w);
                }

                for (std::size_t y = 0; y < h; ++y) {
                    for (std::size_t x = 0; x < w; ++x) {
                        out[idx_feat(y, x, d, w, out_depth)] = img[idx3(c, wy[y], wx[x], h, w)];
                    }
                }
            }
        }
    }
    return out;
}

// HWD-native normalization: depth is contiguous per pixel — cache-friendly
void normalize_feature_depth_hwd(AlignedVector<float>& feature, std::size_t h, std::size_t w, std::size_t depth) {
    if (feature.size() != h * w * depth) {
        throw std::invalid_argument("normalize_feature_depth_hwd size mismatch");
    }
    constexpr float kEps = 1e-6f;
    const std::size_t plane = h * w;
    const float inv_depth = 1.0f / static_cast<float>(depth);

    #pragma omp parallel for schedule(static)
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        float* row = feature.data() + pixel * depth;

        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (std::size_t d = 0; d < depth; ++d) sum += row[d];
        const float mean = sum * inv_depth;

        float var = 0.0f;
        #pragma omp simd reduction(+:var)
        for (std::size_t d = 0; d < depth; ++d) {
            const float v = row[d] - mean;
            var += v * v;
        }
        const float inv_std = 1.0f / (std::sqrt(var * inv_depth) + kEps);

        #pragma omp simd
        for (std::size_t d = 0; d < depth; ++d) {
            row[d] = (row[d] - mean) * inv_std;
        }
    }
}

PyramidResult build_pyramid_single(
    const std::vector<float>& data,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int pyramid_level,
    int n_template,
    PyramidDownsampleMode mode) {
    if (data.size() != ch * h * w) {
        throw std::invalid_argument("build_pyramid_single size mismatch");
    }
    if (pyramid_level < 0) {
        throw std::invalid_argument("pyramid_level must be >= 0");
    }

    double total_downsample_s = 0.0;
    double total_tmpl_s = 0.0;
    double total_norm_s = 0.0;

    std::vector<std::vector<float>> raw_levels;
    std::vector<std::array<std::size_t, 3>> raw_dims;
    raw_levels.push_back(data);
    raw_dims.push_back({ch, h, w});

    for (int lv = 0; lv < pyramid_level; ++lv) {
        std::size_t nh = 0;
        std::size_t nw = 0;
        const auto& prev = raw_levels.back();
        const auto dim = raw_dims.back();
        auto t0 = std::chrono::steady_clock::now();
        auto next = downsample_by_mode(prev, dim[0], dim[1], dim[2], nh, nw, mode);
        auto t1 = std::chrono::steady_clock::now();
        total_downsample_s += std::chrono::duration<double>(t1 - t0).count();
        raw_levels.push_back(std::move(next));
        raw_dims.push_back({dim[0], nh, nw});
    }

    PyramidResult result;
    result.ref_levels.reserve(raw_levels.size());

    for (std::size_t lv = 0; lv < raw_levels.size(); ++lv) {
        const auto dim = raw_dims[lv];
        std::size_t out_d = 0;
        auto t0 = std::chrono::steady_clock::now();
        auto feat = stack_template_window_hwd(raw_levels[lv], dim[0], dim[1], dim[2], n_template, out_d);
        auto t1 = std::chrono::steady_clock::now();
        total_tmpl_s += std::chrono::duration<double>(t1 - t0).count();

        t0 = std::chrono::steady_clock::now();
        normalize_feature_depth_hwd(feat, dim[1], dim[2], out_d);
        t1 = std::chrono::steady_clock::now();
        total_norm_s += std::chrono::duration<double>(t1 - t0).count();

        result.ref_levels.push_back(PyramidLevel{std::move(feat), out_d, dim[1], dim[2]});
    }
    prColor("  pyramid detail: downsample=" + std::to_string(total_downsample_s) +
            "s tmpl_win=" + std::to_string(total_tmpl_s) +
            "s normalize=" + std::to_string(total_norm_s) + "s", "light_purple");
    return result;
}

}

PyramidResult pyramid_data(
    const std::vector<float>& ref_data,
    const std::vector<float>& img_data,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int pyramid_level,
    int n_template,
    PyramidDownsampleMode mode) {
    if (ref_data.size() != ch * h * w || img_data.size() != ch * h * w) {
        throw std::invalid_argument("pyramid_data input size mismatch");
    }

    int inner_threads = 1;
#ifdef _OPENMP
    inner_threads = std::max(1, omp_get_max_threads() / 2);
#endif

    PyramidResult ref_result, img_result;
    // ref and img builds are independent — run in parallel.
    // Each section reduces its inner thread pool to half the total to avoid oversubscription.
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
#ifdef _OPENMP
            omp_set_num_threads(inner_threads);
#endif
            ref_result = build_pyramid_single(ref_data, ch, h, w, pyramid_level, n_template, mode);
        }
        #pragma omp section
        {
#ifdef _OPENMP
            omp_set_num_threads(inner_threads);
#endif
            img_result = build_pyramid_single(img_data, ch, h, w, pyramid_level, n_template, mode);
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(inner_threads * 2);
#endif

    PyramidResult merged;
    merged.ref_levels = std::move(ref_result.ref_levels);
    merged.img_levels = std::move(img_result.ref_levels);
    return merged;
}

}
