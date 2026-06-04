#include "wsvt/wavelet_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace wsvt {

namespace {

inline std::size_t idx_chw(std::size_t c, std::size_t y, std::size_t x, std::size_t h,
                           std::size_t w) {
    return (c * h + y) * w + x;
}

inline std::size_t idx_hwd(std::size_t y, std::size_t x, std::size_t d, std::size_t w,
                           std::size_t depth) {
    return (y * w + x) * depth + d;
}

void get_wavelet_filters(WaveletFamily wavelet, std::vector<float>& dec_lo,
                         std::vector<float>& dec_hi) {
    switch (wavelet) {
        case WaveletFamily::Db2:
            dec_lo = {-0.1294095226f, 0.2241438680f, 0.8365163037f, 0.4829629131f};
            dec_hi = {-0.4829629131f, 0.8365163037f, -0.2241438680f, -0.1294095226f};
            return;
        case WaveletFamily::Db3:
            dec_lo = {0.0352262919f,  -0.0854412739f, -0.1350110200f,
                      0.4598775021f, 0.8068915093f,  0.3326705530f};
            dec_hi = {-0.3326705530f, 0.8068915093f,  -0.4598775021f,
                      -0.1350110200f, 0.0854412739f, 0.0352262919f};
            return;
        case WaveletFamily::Db6:
            dec_lo = {-0.0010773011f, 0.0047772575f,  0.0005538422f,  -0.0315820393f,
                      0.0275228655f,  0.0975016056f,  -0.1297668676f, -0.2262646940f,
                      0.3152503517f,  0.7511339080f,  0.4946238904f,  0.1115407434f};
            dec_hi = {-0.1115407434f, 0.4946238904f,  -0.7511339080f, 0.3152503517f,
                      0.2262646940f,  -0.1297668676f, -0.0975016056f, 0.0275228655f,
                      0.0315820393f,  0.0005538422f,  -0.0047772575f, -0.0010773011f};
            return;
    }
    throw std::invalid_argument("unsupported wavelet family");
}

std::vector<std::string> build_level_name(int w_level, int return_level) {
    std::vector<std::string> level_name;
    level_name.reserve(static_cast<std::size_t>(w_level + 1));
    for (int kk = 0; kk < w_level; ++kk) {
        std::string name = "D";
        name += std::to_string(kk + 1);
        level_name.push_back(std::move(name));
    }
    std::string approx_name = "A";
    approx_name += std::to_string(w_level);
    level_name.push_back(std::move(approx_name));
    if (return_level < 0 || return_level > static_cast<int>(level_name.size())) {
        throw std::invalid_argument("return_level out of range");
    }
    return std::vector<std::string>(level_name.end() - return_level, level_name.end());
}

// Optimized wavedec_axis0: inline DWT + OpenMP parallel over (y, x) pixels
std::vector<std::vector<float>> wavedec_axis0(std::span<const float> img, std::size_t ch,
                                              std::size_t h, std::size_t w,
                                              const std::vector<float>& dec_lo,
                                              const std::vector<float>& dec_hi, int level) {
    const std::size_t filt_len = dec_lo.size();
    std::vector<std::vector<float>> coeffs;
    coeffs.reserve(static_cast<std::size_t>(level + 1));

    std::vector<float> current(img.begin(), img.end());
    std::size_t cur_ch = ch;

    for (int lv = 0; lv < level; ++lv) {
        const std::size_t out_ch = (cur_ch + filt_len - 1) / 2;
        std::vector<float> approx(out_ch * h * w, 0.0f);
        std::vector<float> detail(out_ch * h * w, 0.0f);

#pragma omp parallel for schedule(static)
        for (std::size_t pixel = 0; pixel < h * w; ++pixel) {
            const std::size_t yy = pixel / w;
            const std::size_t xx = pixel % w;
            for (std::size_t oc = 0; oc < out_ch; ++oc) {
                float a = 0.0f, d = 0.0f;
                const auto offset = static_cast<long long>(filt_len - 2);
                for (std::size_t k = 0; k < filt_len; ++k) {
                    const auto ic_signed = static_cast<long long>(oc * 2 + k) - offset;
                    if (ic_signed < 0 || ic_signed >= static_cast<long long>(cur_ch)) {
                        continue;
                    }
                    const auto ic = static_cast<std::size_t>(ic_signed);
                    const float v = current[idx_chw(ic, yy, xx, h, w)];
                    const std::size_t fk = filt_len - 1 - k;
                    a += dec_lo[fk] * v;
                    d += dec_hi[fk] * v;
                }
                approx[idx_chw(oc, yy, xx, h, w)] = a;
                detail[idx_chw(oc, yy, xx, h, w)] = d;
            }
        }

        coeffs.push_back(std::move(detail));
        current = std::move(approx);
        cur_ch = out_ch;
    }
    coeffs.push_back(std::move(current));
    return coeffs;
}

}  // namespace

WaveletResult wavelet_transform(std::span<const float> img, std::size_t ch, std::size_t h,
                                std::size_t w, WaveletFamily wavelet, int w_level,
                                int return_level) {
    if (img.size() != ch * h * w) {
        throw std::invalid_argument("wavelet_transform: input size mismatch");
    }

    std::vector<float> dec_lo, dec_hi;
    get_wavelet_filters(wavelet, dec_lo, dec_hi);

    auto coeffs = wavedec_axis0(img, ch, h, w, dec_lo, dec_hi, w_level);

    const int total_levels = static_cast<int>(coeffs.size());
    const int effective_return_level = std::clamp(return_level, 1, total_levels);
    const int start_idx = total_levels - effective_return_level;

    std::size_t out_depth = 0;
    for (int i = start_idx; i < total_levels; ++i) {
        const auto& c = coeffs[static_cast<std::size_t>(i)];
        out_depth += c.size() / (h * w);
    }

    AlignedVector<float> out(h * w * out_depth, 0.0f);
    std::size_t d_offset = 0;

    for (int i = start_idx; i < total_levels; ++i) {
        const auto& c = coeffs[static_cast<std::size_t>(i)];
        const std::size_t c_depth = c.size() / (h * w);

#pragma omp parallel for schedule(static)
        for (std::size_t pixel = 0; pixel < h * w; ++pixel) {
            const std::size_t yy = pixel / w;
            const std::size_t xx = pixel % w;
            for (std::size_t dd = 0; dd < c_depth; ++dd) {
                out[idx_hwd(yy, xx, d_offset + dd, w, out_depth)] =
                    c[idx_chw(dd, yy, xx, h, w)];
            }
        }
        d_offset += c_depth;
    }

    WaveletResult result;
    result.coeffs_filter = std::move(out);
    result.out_h = h;
    result.out_w = w;
    result.out_depth = out_depth;
    result.level_name = build_level_name(w_level, effective_return_level);
    return result;
}

WaveletResult wavelet_transform_multiprocess(std::span<const float> img, std::size_t ch,
                                             std::size_t h, std::size_t w, int /*n_cores*/,
                                             WaveletFamily wavelet, int w_level,
                                             int return_level) {
    return wavelet_transform(img, ch, h, w, wavelet, w_level, return_level);
}

// ── HWD-native wavelet transform ────────────────────────────────────────────
// Input: [h, w, depth_in] in HWD layout (depth contiguous per pixel)
// Decomposes along the depth axis using 1D DWT per pixel.
// Output: [h, w, out_depth] in HWD layout — no transpose needed.

WaveletResult wavelet_transform_hwd(
    std::span<const float> img_hwd,
    std::size_t h, std::size_t w, std::size_t depth_in,
    WaveletFamily wavelet, int w_level, int return_level) {
    return wavelet_transform_hwd_streamed(img_hwd, h, w, depth_in, wavelet, w_level,
                                          return_level);
}

WaveletResult wavelet_transform_hwd_streamed(
    std::span<const float> img_hwd,
    std::size_t h, std::size_t w, std::size_t depth_in,
    WaveletFamily wavelet, int w_level, int return_level) {
    if (img_hwd.size() != h * w * depth_in) {
        throw std::invalid_argument("wavelet_transform_hwd_streamed: input size mismatch");
    }
    if (w_level < 0) {
        throw std::invalid_argument("wavelet_transform_hwd_streamed: w_level must be non-negative");
    }

    std::vector<float> dec_lo, dec_hi;
    get_wavelet_filters(wavelet, dec_lo, dec_hi);

    const std::size_t filt_len = dec_lo.size();
    const std::size_t plane = h * w;
    std::vector<std::size_t> detail_depths(static_cast<std::size_t>(w_level));
    std::size_t cur_depth_for_shape = depth_in;
    for (int lv = 0; lv < w_level; ++lv) {
        const std::size_t next_depth = (cur_depth_for_shape + filt_len - 1) / 2;
        detail_depths[static_cast<std::size_t>(lv)] = next_depth;
        cur_depth_for_shape = next_depth;
    }

    const int total_levels = w_level + 1;
    const int effective_return_level = std::clamp(return_level, 1, total_levels);
    const int start_idx = total_levels - effective_return_level;

    std::size_t out_depth = 0;
    std::vector<std::size_t> detail_offsets(static_cast<std::size_t>(w_level), 0);
    std::vector<bool> keep_detail(static_cast<std::size_t>(w_level), false);
    for (int lv = 0; lv < w_level; ++lv) {
        if (lv >= start_idx) {
            keep_detail[static_cast<std::size_t>(lv)] = true;
            detail_offsets[static_cast<std::size_t>(lv)] = out_depth;
            out_depth += detail_depths[static_cast<std::size_t>(lv)];
        }
    }
    const std::size_t approx_offset = out_depth;
    const std::size_t approx_depth = cur_depth_for_shape;
    out_depth += approx_depth;

    AlignedVector<float> out(plane * out_depth, 0.0f);

    if (w_level == 0) {
#pragma omp parallel for schedule(static)
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            const float* src = img_hwd.data() + pixel * depth_in;
            float* dst = out.data() + pixel * out_depth + approx_offset;
            std::memcpy(dst, src, depth_in * sizeof(float));
        }
    } else {
        const float* current_ptr = img_hwd.data();
        std::size_t cur_depth = depth_in;
        std::vector<float> approx_buf;

        for (int lv = 0; lv < w_level; ++lv) {
            const std::size_t next_depth = detail_depths[static_cast<std::size_t>(lv)];
            std::vector<float> approx(plane * next_depth);
            const bool write_detail = keep_detail[static_cast<std::size_t>(lv)];
            const std::size_t detail_offset = detail_offsets[static_cast<std::size_t>(lv)];

#pragma omp parallel for schedule(static)
            for (std::size_t pixel = 0; pixel < plane; ++pixel) {
                const float* pixel_in = current_ptr + pixel * cur_depth;
                float* pixel_a = approx.data() + pixel * next_depth;
                const auto offset = static_cast<long long>(filt_len - 2);
                if (write_detail) {
                    float* pixel_d = out.data() + pixel * out_depth + detail_offset;
                    for (std::size_t oc = 0; oc < next_depth; ++oc) {
                        float a = 0.0f, d = 0.0f;
                        for (std::size_t k = 0; k < filt_len; ++k) {
                            const auto ic_signed = static_cast<long long>(oc * 2 + k) - offset;
                            if (ic_signed < 0 || ic_signed >= static_cast<long long>(cur_depth)) {
                                continue;
                            }
                            const auto ic = static_cast<std::size_t>(ic_signed);
                            const float v = pixel_in[ic];
                            const std::size_t fk = filt_len - 1 - k;
                            a += dec_lo[fk] * v;
                            d += dec_hi[fk] * v;
                        }
                        pixel_a[oc] = a;
                        pixel_d[oc] = d;
                    }
                } else {
                    for (std::size_t oc = 0; oc < next_depth; ++oc) {
                        float a = 0.0f;
                        for (std::size_t k = 0; k < filt_len; ++k) {
                            const auto ic_signed = static_cast<long long>(oc * 2 + k) - offset;
                            if (ic_signed < 0 || ic_signed >= static_cast<long long>(cur_depth)) {
                                continue;
                            }
                            const auto ic = static_cast<std::size_t>(ic_signed);
                            const float v = pixel_in[ic];
                            const std::size_t fk = filt_len - 1 - k;
                            a += dec_lo[fk] * v;
                        }
                        pixel_a[oc] = a;
                    }
                }
            }

            approx_buf = std::move(approx);
            current_ptr = approx_buf.data();
            cur_depth = next_depth;
        }

#pragma omp parallel for schedule(static)
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            const float* src = approx_buf.data() + pixel * approx_depth;
            float* dst = out.data() + pixel * out_depth + approx_offset;
            std::memcpy(dst, src, approx_depth * sizeof(float));
        }
    }

    WaveletResult result;
    result.coeffs_filter = std::move(out);
    result.out_h = h;
    result.out_w = w;
    result.out_depth = out_depth;
    result.level_name = build_level_name(w_level, effective_return_level);
    return result;
}

WaveletPairResult wavelet_transform_hwd_pair(
    std::span<const float> img_hwd,
    std::span<const float> ref_hwd,
    std::size_t h, std::size_t w, std::size_t depth_in,
    WaveletFamily wavelet, int w_level, int return_level) {
    return wavelet_transform_hwd_pair_streamed(img_hwd, ref_hwd, h, w, depth_in,
                                               wavelet, w_level, return_level);
}

WaveletPairResult wavelet_transform_hwd_pair_streamed(
    std::span<const float> img_hwd,
    std::span<const float> ref_hwd,
    std::size_t h, std::size_t w, std::size_t depth_in,
    WaveletFamily wavelet, int w_level, int return_level) {
    if (img_hwd.size() != h * w * depth_in) {
        throw std::invalid_argument("wavelet_transform_hwd_pair: img size mismatch");
    }
    if (ref_hwd.size() != h * w * depth_in) {
        throw std::invalid_argument("wavelet_transform_hwd_pair: ref size mismatch");
    }
    if (w_level < 0) {
        throw std::invalid_argument("wavelet_transform_hwd_pair: w_level must be non-negative");
    }

    std::vector<float> dec_lo, dec_hi;
    get_wavelet_filters(wavelet, dec_lo, dec_hi);

    const std::size_t filt_len = dec_lo.size();
    const std::size_t plane = h * w;
    std::vector<std::size_t> detail_depths(static_cast<std::size_t>(w_level));
    std::size_t cur_depth_for_shape = depth_in;
    for (int lv = 0; lv < w_level; ++lv) {
        const std::size_t next_depth = (cur_depth_for_shape + filt_len - 1) / 2;
        detail_depths[static_cast<std::size_t>(lv)] = next_depth;
        cur_depth_for_shape = next_depth;
    }

    const int total_levels = w_level + 1;
    const int effective_return_level = std::clamp(return_level, 1, total_levels);
    const int start_idx = total_levels - effective_return_level;

    std::size_t out_depth = 0;
    std::vector<std::size_t> detail_offsets(static_cast<std::size_t>(w_level), 0);
    std::vector<bool> keep_detail(static_cast<std::size_t>(w_level), false);
    for (int lv = 0; lv < w_level; ++lv) {
        if (lv >= start_idx) {
            keep_detail[static_cast<std::size_t>(lv)] = true;
            detail_offsets[static_cast<std::size_t>(lv)] = out_depth;
            out_depth += detail_depths[static_cast<std::size_t>(lv)];
        }
    }
    const std::size_t approx_offset = out_depth;
    const std::size_t approx_depth = cur_depth_for_shape;
    out_depth += approx_depth;

    AlignedVector<float> img_out(plane * out_depth, 0.0f);
    AlignedVector<float> ref_out(plane * out_depth, 0.0f);

    if (w_level == 0) {
#pragma omp parallel for schedule(static)
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            const float* img_src = img_hwd.data() + pixel * depth_in;
            const float* ref_src = ref_hwd.data() + pixel * depth_in;
            float* img_dst = img_out.data() + pixel * out_depth + approx_offset;
            float* ref_dst = ref_out.data() + pixel * out_depth + approx_offset;
            std::memcpy(img_dst, img_src, depth_in * sizeof(float));
            std::memcpy(ref_dst, ref_src, depth_in * sizeof(float));
        }
    } else {
        const float* img_cur = img_hwd.data();
        const float* ref_cur = ref_hwd.data();
        std::size_t cur_depth = depth_in;
        std::vector<float> img_approx_buf;
        std::vector<float> ref_approx_buf;

        for (int lv = 0; lv < w_level; ++lv) {
            const std::size_t next_depth = detail_depths[static_cast<std::size_t>(lv)];
            std::vector<float> img_approx(plane * next_depth);
            std::vector<float> ref_approx(plane * next_depth);
            const bool write_detail = keep_detail[static_cast<std::size_t>(lv)];
            const std::size_t detail_offset = detail_offsets[static_cast<std::size_t>(lv)];

#pragma omp parallel for schedule(static)
            for (std::size_t pixel = 0; pixel < plane; ++pixel) {
                const float* img_in = img_cur + pixel * cur_depth;
                const float* ref_in = ref_cur + pixel * cur_depth;
                float* img_a = img_approx.data() + pixel * next_depth;
                float* ref_a = ref_approx.data() + pixel * next_depth;
                const auto offset = static_cast<long long>(filt_len - 2);
                if (write_detail) {
                    float* img_d = img_out.data() + pixel * out_depth + detail_offset;
                    float* ref_d = ref_out.data() + pixel * out_depth + detail_offset;
                    for (std::size_t oc = 0; oc < next_depth; ++oc) {
                        float img_acc = 0.0f, img_diff = 0.0f;
                        float ref_acc = 0.0f, ref_diff = 0.0f;
                        for (std::size_t k = 0; k < filt_len; ++k) {
                            const auto ic_signed = static_cast<long long>(oc * 2 + k) - offset;
                            if (ic_signed < 0 || ic_signed >= static_cast<long long>(cur_depth)) {
                                continue;
                            }
                            const auto ic = static_cast<std::size_t>(ic_signed);
                            const std::size_t fk = filt_len - 1 - k;
                            const float lo = dec_lo[fk];
                            const float hi = dec_hi[fk];
                            img_acc += lo * img_in[ic];
                            img_diff += hi * img_in[ic];
                            ref_acc += lo * ref_in[ic];
                            ref_diff += hi * ref_in[ic];
                        }
                        img_a[oc] = img_acc;
                        img_d[oc] = img_diff;
                        ref_a[oc] = ref_acc;
                        ref_d[oc] = ref_diff;
                    }
                } else {
                    for (std::size_t oc = 0; oc < next_depth; ++oc) {
                        float img_acc = 0.0f;
                        float ref_acc = 0.0f;
                        for (std::size_t k = 0; k < filt_len; ++k) {
                            const auto ic_signed = static_cast<long long>(oc * 2 + k) - offset;
                            if (ic_signed < 0 || ic_signed >= static_cast<long long>(cur_depth)) {
                                continue;
                            }
                            const auto ic = static_cast<std::size_t>(ic_signed);
                            const std::size_t fk = filt_len - 1 - k;
                            const float lo = dec_lo[fk];
                            img_acc += lo * img_in[ic];
                            ref_acc += lo * ref_in[ic];
                        }
                        img_a[oc] = img_acc;
                        ref_a[oc] = ref_acc;
                    }
                }
            }

            img_approx_buf = std::move(img_approx);
            ref_approx_buf = std::move(ref_approx);
            img_cur = img_approx_buf.data();
            ref_cur = ref_approx_buf.data();
            cur_depth = next_depth;
        }

#pragma omp parallel for schedule(static)
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            const float* img_src = img_approx_buf.data() + pixel * approx_depth;
            const float* ref_src = ref_approx_buf.data() + pixel * approx_depth;
            float* img_dst = img_out.data() + pixel * out_depth + approx_offset;
            float* ref_dst = ref_out.data() + pixel * out_depth + approx_offset;
            std::memcpy(img_dst, img_src, approx_depth * sizeof(float));
            std::memcpy(ref_dst, ref_src, approx_depth * sizeof(float));
        }
    }

    auto level_names = build_level_name(w_level, effective_return_level);

    WaveletPairResult pair_result;
    pair_result.img.coeffs_filter = std::move(img_out);
    pair_result.img.out_h = h;
    pair_result.img.out_w = w;
    pair_result.img.out_depth = out_depth;
    pair_result.img.level_name = level_names;
    pair_result.ref.coeffs_filter = std::move(ref_out);
    pair_result.ref.out_h = h;
    pair_result.ref.out_w = w;
    pair_result.ref.out_depth = out_depth;
    pair_result.ref.level_name = level_names;
    return pair_result;
}

WaveletTaskResult wavedec_func(std::span<const float> img, std::size_t ch, std::size_t h,
                               std::size_t w, const std::vector<std::size_t>& y_list,
                               WaveletFamily wavelet, int w_level, int return_level) {
    auto wr = wavelet_transform(img, ch, h, w, wavelet, w_level, return_level);
    WaveletTaskResult result;
    result.coeffs_filter = std::move(wr.coeffs_filter);
    result.level_name = std::move(wr.level_name);
    result.y_list = y_list;
    result.out_h = wr.out_h;
    result.out_w = wr.out_w;
    result.out_depth = wr.out_depth;
    return result;
}

}  // namespace wsvt
