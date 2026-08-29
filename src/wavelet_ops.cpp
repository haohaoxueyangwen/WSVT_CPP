#include "wsvt/wavelet_ops.hpp"
#include "wsvt/aligned_alloc.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
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
    std::string approx_name = "A";
    approx_name += std::to_string(w_level);
    level_name.push_back(std::move(approx_name));
    for (int level = w_level; level >= 1; --level) {
        std::string name = "D";
        name += std::to_string(level);
        level_name.push_back(std::move(name));
    }
    if (return_level < 0 || return_level > static_cast<int>(level_name.size())) {
        throw std::invalid_argument("return_level out of range");
    }
    return std::vector<std::string>(level_name.begin(), level_name.begin() + return_level);
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
    // PyWavelets wavedec order: [cA_n, cD_n, ..., cD1].
    std::reverse(coeffs.begin(), coeffs.end());
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
    std::size_t out_depth = 0;
    for (int i = 0; i < effective_return_level; ++i) {
        const auto& c = coeffs[static_cast<std::size_t>(i)];
        out_depth += c.size() / (h * w);
    }

    AlignedVector<float> out(h * w * out_depth, 0.0f);
    std::size_t d_offset = 0;

    for (int i = 0; i < effective_return_level; ++i) {
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

    const std::size_t approx_depth = cur_depth_for_shape;
    const std::size_t approx_offset = 0;
    std::size_t out_depth = approx_depth;
    std::vector<std::size_t> detail_offsets(static_cast<std::size_t>(w_level), 0);
    std::vector<bool> keep_detail(static_cast<std::size_t>(w_level), false);
    // PyWavelets wavedec order: approximation first, then details from the
    // coarsest retained level to the finest retained level.
    for (int lv = w_level - 1; lv >= start_idx; --lv) {
        keep_detail[static_cast<std::size_t>(lv)] = true;
        detail_offsets[static_cast<std::size_t>(lv)] = out_depth;
        out_depth += detail_depths[static_cast<std::size_t>(lv)];
    }

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
        AlignedBuffer<float> approx_buf;

        for (int lv = 0; lv < w_level; ++lv) {
            const std::size_t next_depth = detail_depths[static_cast<std::size_t>(lv)];
            AlignedBuffer<float> approx(plane * next_depth);
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

// ── Precomputed DWT plan + optimized transform ─────────────────────────────

std::vector<DwtLevelPlan> compute_wavelet_plan(
    std::size_t depth_in, int w_level, WaveletFamily wavelet)
{
    std::vector<float> dec_lo, dec_hi;
    get_wavelet_filters(wavelet, dec_lo, dec_hi);
    const std::size_t filt_len = dec_lo.size();
    const auto offset = static_cast<long long>(filt_len - 2);

    std::vector<DwtLevelPlan> plans(static_cast<std::size_t>(w_level));
    std::size_t cur_depth = depth_in;

    for (int lv = 0; lv < w_level; ++lv) {
        const std::size_t next_depth = (cur_depth + filt_len - 1) / 2;
        auto& plan = plans[static_cast<std::size_t>(lv)];
        plan.oc_taps.resize(next_depth);

        for (std::size_t oc = 0; oc < next_depth; ++oc) {
            for (std::size_t k = 0; k < filt_len; ++k) {
                const auto ic_signed =
                    static_cast<long long>(oc * 2 + k) - offset;
                if (ic_signed < 0 || ic_signed >= static_cast<long long>(cur_depth)) {
                    continue;
                }
                const std::size_t ic = static_cast<std::size_t>(ic_signed);
                const std::size_t fk = filt_len - 1 - k;
                plan.oc_taps[oc].push_back(DwtTap{ic, dec_lo[fk], dec_hi[fk]});
            }
        }
        cur_depth = next_depth;
    }
    return plans;
}

WaveletResult wavelet_transform_hwd_planned(
    std::span<const float> img_hwd,
    std::size_t h, std::size_t w, std::size_t depth_in,
    WaveletFamily wavelet, int w_level, int return_level,
    const std::vector<DwtLevelPlan>& plans)
{
    if (img_hwd.size() != h * w * depth_in) {
        throw std::invalid_argument("wavelet_transform_hwd_planned: input size mismatch");
    }
    if (w_level < 0) {
        throw std::invalid_argument("wavelet_transform_hwd_planned: w_level >= 0 required");
    }
    if (static_cast<std::size_t>(w_level) != plans.size()) {
        throw std::invalid_argument("wavelet_transform_hwd_planned: plans.size() != w_level");
    }

    std::vector<float> dec_lo, dec_hi;
    get_wavelet_filters(wavelet, dec_lo, dec_hi);
    const std::size_t filt_len = dec_lo.size();

    // ── Compute output layout (same as wavelet_transform_hwd_streamed) ──
    const std::size_t plane = h * w;
    std::vector<std::size_t> detail_depths(static_cast<std::size_t>(w_level));
    std::size_t cur_depth_for_shape = depth_in;
    for (int lv = 0; lv < w_level; ++lv) {
        const std::size_t next_depth = (cur_depth_for_shape + filt_len - 1) / 2;
        detail_depths[static_cast<std::size_t>(lv)] = next_depth;
        cur_depth_for_shape = next_depth;
    }

    const int total_levels = w_level + 1;
    const int eff_ret = std::clamp(return_level, 1, total_levels);
    const int start_idx = total_levels - eff_ret;

    const std::size_t approx_depth = cur_depth_for_shape;
    const std::size_t approx_offset = 0;
    std::size_t out_depth = approx_depth;
    std::vector<std::size_t> detail_offsets(static_cast<std::size_t>(w_level));
    std::vector<bool> keep_detail(static_cast<std::size_t>(w_level), false);
    for (int lv = w_level - 1; lv >= start_idx; --lv) {
        keep_detail[static_cast<std::size_t>(lv)] = true;
        detail_offsets[static_cast<std::size_t>(lv)] = out_depth;
        out_depth += detail_depths[static_cast<std::size_t>(lv)];
    }

    AlignedVector<float> out(plane * out_depth, 0.0f);

    if (w_level == 0) {
        #pragma omp parallel for schedule(static)
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            const float* src = img_hwd.data() + pixel * depth_in;
            float* dst = out.data() + pixel * out_depth + approx_offset;
            std::memcpy(dst, src, depth_in * sizeof(float));
        }
    } else {
        // ── Single OpenMP region for all levels ──
        // Double-buffer: buf_a / buf_b alternate as source / destination.
        // AlignedBuffer: uninitialized — avoids GB-scale zeroing; OpenMP
        // workers first-touch their own pages on first write.
        const std::size_t max_buf_depth = *std::max_element(detail_depths.begin(),
                                                             detail_depths.end());
        const std::size_t buf_elems = plane * std::max(max_buf_depth, approx_depth);
        AlignedBuffer<float> buf_a(buf_elems);
        AlignedBuffer<float> buf_b(buf_elems);

        #pragma omp parallel
        {
            const float* cur_ptr = img_hwd.data();
            std::size_t cur_depth = depth_in;
            float* prev_buf = nullptr;  // buffer holding previous approx (not used for lv=0)

            for (int lv = 0; lv < w_level; ++lv) {
                const auto& plan = plans[static_cast<std::size_t>(lv)];
                const std::size_t next_depth = detail_depths[static_cast<std::size_t>(lv)];
                const bool write_detail = keep_detail[static_cast<std::size_t>(lv)];
                const std::size_t detail_off = detail_offsets[static_cast<std::size_t>(lv)];

                // Pick output buffer: lv even → buf_a, lv odd → buf_b
                float* approx_out = (lv % 2 == 0) ? buf_a.data() : buf_b.data();

                #pragma omp for schedule(static)
                for (std::size_t pixel = 0; pixel < plane; ++pixel) {
                    const float* pixel_in;
                    if (lv == 0) {
                        pixel_in = cur_ptr + pixel * cur_depth;
                    } else {
                        pixel_in = prev_buf + pixel * cur_depth;
                    }

                    float* pixel_a = approx_out + pixel * next_depth;
                    float* pixel_d = write_detail
                        ? out.data() + pixel * out_depth + detail_off
                        : nullptr;

                    for (std::size_t oc = 0; oc < next_depth; ++oc) {
                        float a = 0.0f, d = 0.0f;
                        for (const auto& tap : plan.oc_taps[oc]) {
                            const float v = pixel_in[tap.ic];
                            a += tap.lo * v;
                            d += tap.hi * v;
                        }
                        pixel_a[oc] = a;
                        if (pixel_d) pixel_d[oc] = d;
                    }
                }
                // implicit omp barrier — all pixels must finish before next level

                prev_buf = approx_out;
                cur_depth = next_depth;
            }

            // Copy final approximation to output
            #pragma omp for schedule(static)
            for (std::size_t pixel = 0; pixel < plane; ++pixel) {
                const float* src = prev_buf + pixel * approx_depth;
                float* dst = out.data() + pixel * out_depth + approx_offset;
                std::memcpy(dst, src, approx_depth * sizeof(float));
            }
        }
    }

    WaveletResult result;
    result.coeffs_filter = std::move(out);
    result.out_h = h;
    result.out_w = w;
    result.out_depth = out_depth;
    result.level_name = build_level_name(w_level, eff_ret);
    return result;
}

// ── Pixelchain wavelet transform ────────────────────────────────────────────
// Each pixel's full DWT chain runs in thread-local scratch (~1 KB), removing
// all global intermediate approx buffers.  Single OpenMP parallel-for with
// no internal barriers.  Mathematically identical to wavelet_transform_hwd.

WaveletResult wavelet_transform_hwd_pixelchain(
    std::span<const float> img_hwd,
    std::size_t h, std::size_t w, std::size_t depth_in,
    WaveletFamily wavelet, int w_level, int return_level,
    const std::vector<DwtLevelPlan>& plans)
{
    if (img_hwd.size() != h * w * depth_in) {
        throw std::invalid_argument("wavelet_transform_hwd_pixelchain: size mismatch");
    }
    if (w_level < 0) {
        throw std::invalid_argument("wavelet_transform_hwd_pixelchain: w_level >= 0 required");
    }
    if (static_cast<std::size_t>(w_level) != plans.size()) {
        throw std::invalid_argument("wavelet_transform_hwd_pixelchain: plans.size() != w_level");
    }

    std::vector<float> dec_lo, dec_hi;
    get_wavelet_filters(wavelet, dec_lo, dec_hi);
    const std::size_t filt_len = dec_lo.size();

    // ── Output layout (same as planned) ──
    const std::size_t plane = h * w;
    std::vector<std::size_t> detail_depths(static_cast<std::size_t>(w_level));
    std::size_t cur_depth_for_shape = depth_in;
    for (int lv = 0; lv < w_level; ++lv) {
        const std::size_t next_depth = (cur_depth_for_shape + filt_len - 1) / 2;
        detail_depths[static_cast<std::size_t>(lv)] = next_depth;
        cur_depth_for_shape = next_depth;
    }

    const int total_levels = w_level + 1;
    const int eff_ret = std::clamp(return_level, 1, total_levels);
    const int start_idx = total_levels - eff_ret;

    const std::size_t approx_depth = cur_depth_for_shape;
    const std::size_t approx_offset = 0;
    std::size_t out_depth = approx_depth;
    std::vector<std::size_t> detail_offsets(static_cast<std::size_t>(w_level));
    std::vector<bool> keep_detail(static_cast<std::size_t>(w_level), false);
    for (int lv = w_level - 1; lv >= start_idx; --lv) {
        keep_detail[static_cast<std::size_t>(lv)] = true;
        detail_offsets[static_cast<std::size_t>(lv)] = out_depth;
        out_depth += detail_depths[static_cast<std::size_t>(lv)];
    }

    AlignedVector<float> out(plane * out_depth);

    if (w_level == 0) {
        #pragma omp parallel for schedule(static)
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            const float* src = img_hwd.data() + pixel * depth_in;
            float* dst = out.data() + pixel * out_depth + approx_offset;
            std::memcpy(dst, src, depth_in * sizeof(float));
        }
    } else {
        // Thread-local double buffer: ~1 KB per thread (depth_in ≤ 121 for WXST)
        #pragma omp parallel
        {
            std::vector<float> buf_a(depth_in);
            std::vector<float> buf_b(depth_in);

            #pragma omp for schedule(static)
            for (std::size_t pixel = 0; pixel < plane; ++pixel) {
                const float* src = img_hwd.data() + pixel * depth_in;
                float* cur = buf_a.data();
                float* nxt = buf_b.data();

                // Load input
                std::memcpy(cur, src, depth_in * sizeof(float));
                std::size_t cur_depth = depth_in;

                for (int lv = 0; lv < w_level; ++lv) {
                    const auto& plan = plans[static_cast<std::size_t>(lv)];
                    const std::size_t next_depth = detail_depths[static_cast<std::size_t>(lv)];

                    float* detail_out = keep_detail[static_cast<std::size_t>(lv)]
                        ? out.data() + pixel * out_depth + detail_offsets[static_cast<std::size_t>(lv)]
                        : nullptr;

                    for (std::size_t oc = 0; oc < next_depth; ++oc) {
                        float a = 0.0f, d = 0.0f;
                        for (const auto& tap : plan.oc_taps[oc]) {
                            const float v = cur[tap.ic];
                            a += tap.lo * v;
                            d += tap.hi * v;
                        }
                        nxt[oc] = a;
                        if (detail_out) detail_out[oc] = d;
                    }

                    // Swap: nxt becomes cur for next level
                    std::swap(cur, nxt);
                    cur_depth = next_depth;
                }

                // Copy final approximation
                float* dst = out.data() + pixel * out_depth + approx_offset;
                std::memcpy(dst, cur, approx_depth * sizeof(float));
            }
        }
    }

    WaveletResult result;
    result.coeffs_filter = std::move(out);
    result.out_h = h;
    result.out_w = w;
    result.out_depth = out_depth;
    result.level_name = build_level_name(w_level, eff_ret);
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

    const std::size_t approx_depth = cur_depth_for_shape;
    const std::size_t approx_offset = 0;
    std::size_t out_depth = approx_depth;
    std::vector<std::size_t> detail_offsets(static_cast<std::size_t>(w_level), 0);
    std::vector<bool> keep_detail(static_cast<std::size_t>(w_level), false);
    for (int lv = w_level - 1; lv >= start_idx; --lv) {
        keep_detail[static_cast<std::size_t>(lv)] = true;
        detail_offsets[static_cast<std::size_t>(lv)] = out_depth;
        out_depth += detail_depths[static_cast<std::size_t>(lv)];
    }

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
        AlignedBuffer<float> img_approx_buf;
        AlignedBuffer<float> ref_approx_buf;

        for (int lv = 0; lv < w_level; ++lv) {
            const std::size_t next_depth = detail_depths[static_cast<std::size_t>(lv)];
            AlignedBuffer<float> img_approx(plane * next_depth);
            AlignedBuffer<float> ref_approx(plane * next_depth);
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

PrefixCompatibleWaveletBasis make_prefix_compatible_wavelet_basis(
    std::size_t final_depth,
    WaveletFamily wavelet,
    int wavelet_level,
    int return_level) {
    if (final_depth == 0) {
        throw std::invalid_argument(
            "prefix-compatible wavelet final_depth must be positive");
    }
    if (final_depth > static_cast<std::size_t>(
            std::numeric_limits<std::uint16_t>::max())) {
        throw std::invalid_argument(
            "prefix-compatible wavelet final_depth exceeds state counter range");
    }

    // Each HWD pixel is one input-frame basis vector. Transforming all basis
    // pixels at once recovers the retained final-depth linear operator without
    // duplicating the DWT boundary convention in this prototype.
    AlignedVector<float> identity(final_depth * final_depth, 0.0f);
    for (std::size_t frame = 0; frame < final_depth; ++frame) {
        identity[frame * final_depth + frame] = 1.0f;
    }
    auto transformed = wavelet_transform_hwd(
        as_span(identity), final_depth, 1, final_depth,
        wavelet, wavelet_level, return_level);

    PrefixCompatibleWaveletBasis basis;
    basis.final_depth = final_depth;
    basis.out_depth = transformed.out_depth;
    basis.wavelet = wavelet;
    basis.wavelet_level = wavelet_level;
    basis.return_level = return_level;
    basis.level_name = std::move(transformed.level_name);
    basis.frame_coefficient_weights.resize(
        final_depth * basis.out_depth, 0.0);
    for (std::size_t frame = 0; frame < final_depth; ++frame) {
        for (std::size_t coefficient = 0;
             coefficient < basis.out_depth; ++coefficient) {
            basis.frame_coefficient_weights[
                frame * basis.out_depth + coefficient] =
                static_cast<double>(transformed.coeffs_filter[
                    frame * basis.out_depth + coefficient]);
        }
    }
    basis.prefix_weight_sums.resize(
        (final_depth + 1) * basis.out_depth, 0.0);
    for (std::size_t frame = 0; frame < final_depth; ++frame) {
        const double* previous = basis.prefix_weight_sums.data() +
            frame * basis.out_depth;
        double* next = basis.prefix_weight_sums.data() +
            (frame + 1) * basis.out_depth;
        const double* weights = basis.frame_coefficient_weights.data() +
            frame * basis.out_depth;
        for (std::size_t coefficient = 0;
             coefficient < basis.out_depth; ++coefficient) {
            next[coefficient] = previous[coefficient] + weights[coefficient];
        }
    }
    return basis;
}

PrefixCompatibleWaveletState make_prefix_compatible_wavelet_state(
    std::size_t h,
    std::size_t w,
    const PrefixCompatibleWaveletBasis& basis) {
    if (h == 0 || w == 0 || basis.final_depth == 0 || basis.out_depth == 0) {
        throw std::invalid_argument(
            "prefix-compatible wavelet state requires non-zero dimensions");
    }
    const std::size_t plane = h * w;
    PrefixCompatibleWaveletState state;
    state.h = h;
    state.w = w;
    state.final_depth = basis.final_depth;
    state.out_depth = basis.out_depth;
    state.frames_accumulated.assign(plane, 0);
    state.raw_sum.assign(plane, 0.0);
    state.raw_sum_squares.assign(plane, 0.0);
    state.weighted_raw_sum.assign(plane * basis.out_depth, 0.0);
    return state;
}

void extend_prefix_compatible_wavelet_state(
    PrefixCompatibleWaveletState& state,
    const PrefixCompatibleWaveletBasis& basis,
    std::span<const float> data_chw,
    std::size_t target_frames,
    std::span<const std::uint8_t> selected_mask) {
    const std::size_t plane = state.h * state.w;
    if (state.final_depth != basis.final_depth ||
        state.out_depth != basis.out_depth ||
        state.frames_accumulated.size() != plane ||
        state.raw_sum.size() != plane ||
        state.raw_sum_squares.size() != plane ||
        state.weighted_raw_sum.size() != plane * state.out_depth) {
        throw std::invalid_argument(
            "prefix-compatible wavelet state/basis shape mismatch");
    }
    if (data_chw.size() != basis.final_depth * plane) {
        throw std::invalid_argument(
            "prefix-compatible wavelet input size mismatch");
    }
    if (target_frames == 0 || target_frames > basis.final_depth) {
        throw std::invalid_argument(
            "prefix-compatible wavelet target_frames out of range");
    }
    if (!selected_mask.empty() && selected_mask.size() != plane) {
        throw std::invalid_argument(
            "prefix-compatible wavelet selected mask size mismatch");
    }
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        if ((!selected_mask.empty() && selected_mask[pixel] == 0) ||
            target_frames >= state.frames_accumulated[pixel]) {
            continue;
        }
        throw std::invalid_argument(
            "prefix-compatible wavelet state cannot move backward");
    }

    std::uint64_t raw_frame_terms = 0;
    std::uint64_t coefficient_terms = 0;
    #pragma omp parallel for schedule(static) reduction(+:raw_frame_terms, coefficient_terms)
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        if (!selected_mask.empty() && selected_mask[pixel] == 0) {
            continue;
        }
        const std::size_t begin = state.frames_accumulated[pixel];
        const auto added_frames = static_cast<std::uint64_t>(
            target_frames - begin);
        raw_frame_terms += added_frames;
        coefficient_terms += added_frames *
            static_cast<std::uint64_t>(state.out_depth);
        double sum = state.raw_sum[pixel];
        double sum_squares = state.raw_sum_squares[pixel];
        double* weighted = state.weighted_raw_sum.data() +
            pixel * state.out_depth;
        for (std::size_t frame = begin; frame < target_frames; ++frame) {
            const double value = static_cast<double>(
                data_chw[frame * plane + pixel]);
            sum += value;
            sum_squares += value * value;
            const double* weights = basis.frame_coefficient_weights.data() +
                frame * state.out_depth;
            for (std::size_t coefficient = 0;
                 coefficient < state.out_depth; ++coefficient) {
                weighted[coefficient] += value * weights[coefficient];
            }
        }
        state.raw_sum[pixel] = sum;
        state.raw_sum_squares[pixel] = sum_squares;
        state.frames_accumulated[pixel] =
            static_cast<std::uint16_t>(target_frames);
    }
    state.raw_frame_pixel_terms_accumulated += raw_frame_terms;
    state.weighted_coefficient_terms_accumulated += coefficient_terms;
}

WaveletResult materialize_prefix_compatible_wavelet(
    const PrefixCompatibleWaveletState& state,
    const PrefixCompatibleWaveletBasis& basis,
    std::span<const std::uint8_t> selected_mask) {
    const std::size_t plane = state.h * state.w;
    if (state.final_depth != basis.final_depth ||
        state.out_depth != basis.out_depth ||
        state.frames_accumulated.size() != plane ||
        state.raw_sum.size() != plane ||
        state.raw_sum_squares.size() != plane ||
        state.weighted_raw_sum.size() != plane * state.out_depth) {
        throw std::invalid_argument(
            "prefix-compatible wavelet state/basis shape mismatch");
    }
    if (!selected_mask.empty() && selected_mask.size() != plane) {
        throw std::invalid_argument(
            "prefix-compatible wavelet selected mask size mismatch");
    }

    AlignedVector<float> output(plane * state.out_depth, 0.0f);
    #pragma omp parallel for schedule(static)
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        if (!selected_mask.empty() && selected_mask[pixel] == 0) {
            continue;
        }
        const std::size_t frames = state.frames_accumulated[pixel];
        if (frames == 0) {
            continue;
        }
        const double inv_frames = 1.0 / static_cast<double>(frames);
        const double mean = state.raw_sum[pixel] * inv_frames;
        const double variance = std::max(
            0.0,
            state.raw_sum_squares[pixel] * inv_frames - mean * mean);
        const double stddev = std::sqrt(variance);
        if (stddev == 0.0) {
            continue;
        }
        const double* weighted = state.weighted_raw_sum.data() +
            pixel * state.out_depth;
        const double* prefix_weights = basis.prefix_weight_sums.data() +
            frames * state.out_depth;
        float* destination = output.data() + pixel * state.out_depth;
        for (std::size_t coefficient = 0;
             coefficient < state.out_depth; ++coefficient) {
            destination[coefficient] = static_cast<float>(
                (weighted[coefficient] - mean * prefix_weights[coefficient]) /
                stddev);
        }
    }

    WaveletResult result;
    result.coeffs_filter = std::move(output);
    result.level_name = basis.level_name;
    result.out_h = state.h;
    result.out_w = state.w;
    result.out_depth = state.out_depth;
    return result;
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
