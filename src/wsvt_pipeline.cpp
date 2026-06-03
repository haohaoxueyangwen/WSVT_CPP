#include "wsvt/wsvt_pipeline.hpp"

#include "wsvt/common.hpp"
#include "wsvt/console_ops.hpp"
#include "wsvt/core.hpp"
#include "wsvt/euclidean_dist.hpp"
#include "wsvt/image_ops.hpp"
#include "wsvt/io_h5.hpp"
#include "wsvt/io_json.hpp"
#include "wsvt/phase_recovery.hpp"
#include "wsvt/solver_utils.hpp"
#include "wsvt/template_window.hpp"
#include "wsvt/wavelet_ops.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <span>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace wsvt {

WSVT::WSVT(
    const std::vector<float>& img_stack,
    const std::vector<float>& ref_stack,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int crop,
    int cal_half_window,
    int n_template,
    int n_s_extend,
    int n_cores,
    int n_group,
    double energy,
    double p_x,
    double mag_factor,
    double z,
    int wavelet_level_cut,
    int pyramid_level,
    int n_iter,
    bool use_estimate,
    bool use_wavelet,
    int use_gpu)
    : img_data_(img_stack),
      ref_data_(ref_stack),
      ch_(ch),
      h_(h),
      w_(w),
      crop_(crop),
      cal_half_window_(cal_half_window),
      n_s_extend_(n_s_extend),
      n_template_(n_template),
      n_cores_(n_cores),
      n_group_(n_group),
      mag_factor_(mag_factor),
      energy_(energy),
      wavelength_(1.2398419843320026e-6 / energy),
      p_x_(p_x),
      z_(z),
      wavelet_level_cut_(wavelet_level_cut),
      pyramid_level_(pyramid_level),
      n_iter_(n_iter),
      use_estimate_(use_estimate),
      use_wavelet_(use_wavelet),
      use_gpu_(use_gpu == 1),
      wavelet_level_(0),
      displace_estimate_h_(h_),
    displace_estimate_w_(w_),
    last_pyramid_time_s_(0.0),
    last_wavelet_time_s_(0.0) {
    if (img_data_.size() != ch_ * h_ * w_ || ref_data_.size() != ch_ * h_ * w_) {
        throw std::invalid_argument("WSVT init size mismatch");
    }
    if (use_gpu_) {
        prColor("Use GPU found. Enable multi-resolution", "cyan");
    } else {
        prColor("No gpu found. Use CPU instead.", "cyan");
    }
    if (use_estimate_) {
#if defined(WSVT_HAS_OPENCV)
        // CHW layout: channel 0 is the first h*w elements — zero-copy pointer
        const auto disp_init = slope_tracking(ref_data_.data(), img_data_.data(), h_, w_, cal_half_window_);
        displace_estimate_y_ = disp_init[0];
        displace_estimate_x_ = disp_init[1];
#else
        throw std::runtime_error("use_estimate requires OpenCV (WSVT_HAS_OPENCV)");
#endif
    } else {
        displace_estimate_y_.assign(h_ * w_, 0.0f);
        displace_estimate_x_.assign(h_ * w_, 0.0f);
    }
}

WSVT::WSVT(
    std::vector<float>&& img_stack,
    std::vector<float>&& ref_stack,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int crop,
    int cal_half_window,
    int n_template,
    int n_s_extend,
    int n_cores,
    int n_group,
    double energy,
    double p_x,
    double mag_factor,
    double z,
    int wavelet_level_cut,
    int pyramid_level,
    int n_iter,
    bool use_estimate,
    bool use_wavelet,
    int use_gpu)
    : img_data_(std::move(img_stack)),
      ref_data_(std::move(ref_stack)),
      ch_(ch),
      h_(h),
      w_(w),
      crop_(crop),
      cal_half_window_(cal_half_window),
      n_s_extend_(n_s_extend),
      n_template_(n_template),
      n_cores_(n_cores),
      n_group_(n_group),
      mag_factor_(mag_factor),
      energy_(energy),
      wavelength_(1.2398419843320026e-6 / energy),
      p_x_(p_x),
      z_(z),
      wavelet_level_cut_(wavelet_level_cut),
      pyramid_level_(pyramid_level),
      n_iter_(n_iter),
      use_estimate_(use_estimate),
      use_wavelet_(use_wavelet),
      use_gpu_(use_gpu == 1),
      wavelet_level_(0),
      displace_estimate_h_(h_),
    displace_estimate_w_(w_),
    last_pyramid_time_s_(0.0),
    last_wavelet_time_s_(0.0) {
    if (img_data_.size() != ch_ * h_ * w_ || ref_data_.size() != ch_ * h_ * w_) {
        throw std::invalid_argument("WSVT init size mismatch");
    }
    if (use_gpu_) {
        prColor("Use GPU found. Enable multi-resolution", "cyan");
    } else {
        prColor("No gpu found. Use CPU instead.", "cyan");
    }
    if (use_estimate_) {
#if defined(WSVT_HAS_OPENCV)
        // CHW layout: channel 0 is the first h*w elements — zero-copy pointer
        const auto disp_init = slope_tracking(ref_data_.data(), img_data_.data(), h_, w_, cal_half_window_);
        displace_estimate_y_ = disp_init[0];
        displace_estimate_x_ = disp_init[1];
#else
        throw std::runtime_error("use_estimate requires OpenCV (WSVT_HAS_OPENCV)");
#endif
    } else {
        displace_estimate_y_.assign(h_ * w_, 0.0f);
        displace_estimate_x_.assign(h_ * w_, 0.0f);
    }
}

AlignedVector<float> WSVT::stack_TemplateWindow(
    const std::vector<float>& img,
    std::size_t in_ch,
    std::size_t in_h,
    std::size_t in_w,
    std::size_t& out_h,
    std::size_t& out_w,
    std::size_t& out_d) const {
    return stack_template_window(img, in_ch, in_h, in_w, n_template_, out_h, out_w, out_d);
}

PyramidResult WSVT::pyramid_data() {
    prColor("obtain pyramid image with pyramid level: " + std::to_string(pyramid_level_), "green");
    if (crop_ > 0 && static_cast<std::size_t>(crop_) < std::min(h_, w_)) {
        auto cropped_ref = image_roi(
            TensorView3D<const float, Layout::CHW>(ref_data_.data(), Shape3D{ch_, h_, w_}),
            static_cast<std::size_t>(crop_));
        auto cropped_img = image_roi(
            TensorView3D<const float, Layout::CHW>(img_data_.data(), Shape3D{ch_, h_, w_}),
            static_cast<std::size_t>(crop_));
        ref_data_ = std::move(cropped_ref).take();
        img_data_ = std::move(cropped_img).take();
        h_ = static_cast<std::size_t>(crop_);
        w_ = static_cast<std::size_t>(crop_);
    }
    auto p = wsvt::pyramid_data(
        ref_data_, img_data_, ch_, h_, w_,
        pyramid_level_, n_template_, PyramidDownsampleMode::Db3Aa,
        PyramidNormalizationMode::InitialStack);
    return p;
}

PyramidResult WSVT::wavelet_data() {
    const auto pyr_t0 = std::chrono::steady_clock::now();
    auto p = pyramid_data();
    const auto pyr_t1 = std::chrono::steady_clock::now();
    last_pyramid_time_s_ = std::chrono::duration<double>(pyr_t1 - pyr_t0).count();
    prColor("pyramid time: " + std::to_string(last_pyramid_time_s_) + " s", "light_purple");
    last_wavelet_time_s_ = 0.0;
    if (use_wavelet_) {
        prColor("obtain wavelet data...", "green");
        constexpr WaveletFamily wavelet_method = WaveletFamily::Db2;
        wavelet_level_ = dwt_max_level_db2(p.ref_levels[0].d0);
        prColor("max wavelet level: " + std::to_string(wavelet_level_), "green");
        int coefs_level = wavelet_level_ + 1 - wavelet_level_cut_;

        wavelet_add_list_ = wavelet_add_list_for_depth(p.ref_levels[0].d0);

        const auto wavelet_t0 = std::chrono::steady_clock::now();
        std::string wavelet_detail;
        for (std::size_t lv = 0; lv < p.ref_levels.size(); ++lv) {
            int wavelevel_add = (lv >= wavelet_add_list_.size() ? 2 : wavelet_add_list_[lv]);
            WaveletResult img_wa;
            WaveletResult ref_wa;
            double img_wavelet_s = 0.0;
            double ref_wavelet_s = 0.0;
            bool use_nested_wavelet = false;
            int outer_threads = 1;
#ifdef _OPENMP
            outer_threads = omp_get_max_threads();
            use_nested_wavelet = (omp_get_max_active_levels() > 1 && outer_threads >= 4);
#endif
            if (use_nested_wavelet) {
                const int inner_threads = std::max(1, outer_threads / 2);
                #pragma omp parallel sections num_threads(2)
                {
                    #pragma omp section
                    {
#ifdef _OPENMP
                        omp_set_num_threads(inner_threads);
#endif
                        const auto wt0 = std::chrono::steady_clock::now();
                        img_wa = wavelet_transform_hwd(
                            as_span(p.img_levels[lv].data), p.img_levels[lv].d1, p.img_levels[lv].d2, p.img_levels[lv].d0,
                            wavelet_method, wavelet_level_, coefs_level + wavelevel_add);
                        const auto wt1 = std::chrono::steady_clock::now();
                        img_wavelet_s = std::chrono::duration<double>(wt1 - wt0).count();
                    }
                    #pragma omp section
                    {
#ifdef _OPENMP
                        omp_set_num_threads(inner_threads);
#endif
                        const auto wt0 = std::chrono::steady_clock::now();
                        ref_wa = wavelet_transform_hwd(
                            as_span(p.ref_levels[lv].data), p.ref_levels[lv].d1, p.ref_levels[lv].d2, p.ref_levels[lv].d0,
                            wavelet_method, wavelet_level_, coefs_level + wavelevel_add);
                        const auto wt1 = std::chrono::steady_clock::now();
                        ref_wavelet_s = std::chrono::duration<double>(wt1 - wt0).count();
                    }
                }
#ifdef _OPENMP
                omp_set_num_threads(outer_threads);
#endif
            } else {
                const auto wt0 = std::chrono::steady_clock::now();
                img_wa = wavelet_transform_hwd(
                    as_span(p.img_levels[lv].data), p.img_levels[lv].d1, p.img_levels[lv].d2, p.img_levels[lv].d0,
                    wavelet_method, wavelet_level_, coefs_level + wavelevel_add);
                const auto wt1 = std::chrono::steady_clock::now();
                ref_wa = wavelet_transform_hwd(
                    as_span(p.ref_levels[lv].data), p.ref_levels[lv].d1, p.ref_levels[lv].d2, p.ref_levels[lv].d0,
                    wavelet_method, wavelet_level_, coefs_level + wavelevel_add);
                const auto wt2 = std::chrono::steady_clock::now();
                img_wavelet_s = std::chrono::duration<double>(wt1 - wt0).count();
                ref_wavelet_s = std::chrono::duration<double>(wt2 - wt1).count();
            }
            p.img_levels[lv] = PyramidLevel{std::move(img_wa.coeffs_filter), img_wa.out_depth, img_wa.out_h, img_wa.out_w};
            p.ref_levels[lv] = PyramidLevel{std::move(ref_wa.coeffs_filter), ref_wa.out_depth, ref_wa.out_h, ref_wa.out_w};
            wavelet_detail += " lv" + std::to_string(lv) + "_img=" + std::to_string(img_wavelet_s) +
                              "s_ref=" + std::to_string(ref_wavelet_s) + "s";
            prColor("pyramid level: " + std::to_string(lv) + "\nvector length: " + std::to_string(ref_wa.out_depth), "green");
        }
        const auto wavelet_t1 = std::chrono::steady_clock::now();
        last_wavelet_time_s_ = std::chrono::duration<double>(wavelet_t1 - wavelet_t0).count();
        prColor("wavelet time: " + std::to_string(last_wavelet_time_s_) + " s", "light_purple");
        prColor("  wavelet detail:" + wavelet_detail, "light_purple");
    } else {
        wavelet_level_ = 0;
        wavelet_add_list_.clear();
    }
    return p;
}

std::vector<float> WSVT::resampling_spline(const std::vector<float>& img, std::size_t in_h, std::size_t in_w, std::size_t out_h, std::size_t out_w) const {
    if (img.size() != in_h * in_w) {
        throw std::invalid_argument("resampling_spline input size mismatch");
    }
    std::vector<float> out(out_h * out_w, 0.0f);
    if (out_h == 0 || out_w == 0) {
        return out;
    }
    if (in_h == 0 || in_w == 0) {
        throw std::invalid_argument("resampling_spline input shape must be positive");
    }
    if (out_h == 1 && out_w == 1) {
        out[0] = img[0];
        return out;
    }
    const double x_scale = out_w == 1 ? 0.0 : static_cast<double>(in_w - 1) / static_cast<double>(out_w - 1);
    const double y_scale = out_h == 1 ? 0.0 : static_cast<double>(in_h - 1) / static_cast<double>(out_h - 1);
    #pragma omp parallel for schedule(static)
    for (std::size_t y = 0; y < out_h; ++y) {
        for (std::size_t x = 0; x < out_w; ++x) {
            const double src_y = y_scale * static_cast<double>(y);
            const double src_x = x_scale * static_cast<double>(x);
            out[idx2(y, x, out_w)] = static_cast<float>(sample_bicubic(
                ImageView2D<const float>{img.data(), {in_h, in_w}},
                src_y, src_x));
        }
    }
    return out;
}

std::array<std::vector<float>, 3> WSVT::displace_wavelet(
    std::span<const float> img_wa_stack,
    std::size_t img_h,
    std::size_t img_w,
    std::span<const float> ref_wa_stack,
    std::size_t ref_h,
    std::size_t ref_w,
    std::size_t depth,
    const std::vector<float>& displace_y,
    const std::vector<float>& displace_x,
    int cal_half_window,
    int n_pad) const {
    if (img_wa_stack.size() != img_h * img_w * depth || ref_wa_stack.size() != ref_h * ref_w * depth) {
        throw std::invalid_argument("displace_wavelet stack shape mismatch");
    }
    if (displace_y.size() != img_h * img_w || displace_x.size() != img_h * img_w) {
        throw std::invalid_argument("displace_wavelet displacement shape mismatch");
    }
    const std::size_t window_size = static_cast<std::size_t>(2 * cal_half_window + 1);
    const std::size_t ws2 = window_size * window_size;

    // Pre-compute axis grids (shared, read-only)
    std::vector<float> yy_axis(ws2, 0.0f);
    std::vector<float> xx_axis(ws2, 0.0f);
    for (std::size_t y = 0; y < window_size; ++y) {
        for (std::size_t x = 0; x < window_size; ++x) {
            yy_axis[y * window_size + x] = static_cast<float>(static_cast<int>(y) - cal_half_window);
            xx_axis[y * window_size + x] = static_cast<float>(static_cast<int>(x) - cal_half_window);
        }
    }

    // Pre-compute pixel_res for find_disp (axis grids are uniform)
    const float pixel_res_x = (window_size >= 2) ? (xx_axis[1] - xx_axis[0]) : 1.0f;
    const float pixel_res_y = (window_size >= 2) ? (yy_axis[window_size] - yy_axis[0]) : 1.0f;

    std::vector<float> disp_y(img_h * img_w, 0.0f);
    std::vector<float> disp_x(img_h * img_w, 0.0f);
    std::vector<float> darkfield_nd(img_h * img_w, 0.0f);
    const std::size_t npad = static_cast<std::size_t>(n_pad);

    const float* __restrict__ img_ptr = std::assume_aligned<64>(img_wa_stack.data());
    const float* __restrict__ ref_ptr = std::assume_aligned<64>(ref_wa_stack.data());
    const float* __restrict__ dy_in = displace_y.data();
    const float* __restrict__ dx_in = displace_x.data();

    #pragma omp parallel
    {
        // Thread-local buffers — allocated once per thread
        std::vector<float> corr_data(ws2, 0.0f);

        #pragma omp for schedule(guided, 64)
        for (std::size_t pixel = 0; pixel < img_h * img_w; ++pixel) {
            const std::size_t yy = pixel / img_w;
            const std::size_t xx = pixel % img_w;

            const float* img_line = img_ptr + pixel * depth;

            const int dy_int = static_cast<int>(dy_in[pixel]);
            const int dx_int = static_cast<int>(dx_in[pixel]);
            const std::size_t y0n = static_cast<std::size_t>(
                static_cast<long long>(npad + yy) + static_cast<long long>(dy_int));
            const std::size_t x0n = static_cast<std::size_t>(
                static_cast<long long>(npad + xx) + static_cast<long long>(dx_int));

            float corr_max = -std::numeric_limits<float>::infinity();
            std::size_t max_idx = 0;
            float sum_abs = 0.0f;

            for (std::size_t wy = 0; wy < window_size; ++wy) {
                for (std::size_t wx = 0; wx < window_size; ++wx) {
                    const float* ref_row = ref_ptr + ((y0n + wy) * ref_w + (x0n + wx)) * depth;
                    float s = 0.0f;
                    #pragma omp simd reduction(+:s)
                    for (std::size_t k = 0; k < depth; ++k) {
                        const float diff = img_line[k] - ref_row[k];
                        s += diff * diff;
                    }
                    const float val = -s;
                    const std::size_t ci = wy * window_size + wx;
                    corr_data[ci] = val;
                    sum_abs += std::fabs(val);
                    if (val > corr_max) {
                        corr_max = val;
                        max_idx = ci;
                    }
                }
            }

            // 4. Inline find_disp: sub-pixel peak finding
            const std::size_t max_y_idx = max_idx / window_size;
            const std::size_t max_x_idx = max_idx % window_size;

            // Sample neighbors with clamping
            const auto sample_corr = [&](long long r, long long c) -> float {
                const auto rc = static_cast<std::size_t>(std::clamp<long long>(r, 0, static_cast<long long>(window_size - 1)));
                const auto cc = static_cast<std::size_t>(std::clamp<long long>(c, 0, static_cast<long long>(window_size - 1)));
                return corr_data[rc * window_size + cc];
            };

            const auto my = static_cast<long long>(max_y_idx);
            const auto mx = static_cast<long long>(max_x_idx);

            const float c_m10 = sample_corr(my - 1, mx);
            const float c_p10 = sample_corr(my + 1, mx);
            const float c_0m1 = sample_corr(my, mx - 1);
            const float c_0p1 = sample_corr(my, mx + 1);
            const float c_00  = sample_corr(my, mx);
            const float c_pp  = sample_corr(my + 1, mx + 1);
            const float c_pm  = sample_corr(my + 1, mx - 1);
            const float c_mp  = sample_corr(my - 1, mx + 1);
            const float c_mm  = sample_corr(my - 1, mx - 1);

            const float grad_y = (c_p10 - c_m10) / 2.0f;
            const float dyy = c_p10 + c_m10 - 2.0f * c_00;
            const float grad_x = (c_0p1 - c_0m1) / 2.0f;
            const float dxx = c_0p1 + c_0m1 - 2.0f * c_00;
            const float dxy = (c_pp - c_pm - c_mp + c_mm) / 4.0f;

            const float denom = dxx * dyy - dxy * dxy;
            const float det = denom != 0.0f ? 1.0f / denom : 0.0f;

            const float minor_disp_x = (-(dyy * grad_x - dxy * grad_y) * det) * pixel_res_x;
            const float minor_disp_y = (-(dxx * grad_y - dxy * grad_x) * det) * pixel_res_y;

            float result_disp_x = xx_axis[max_idx] + minor_disp_x;
            float result_disp_y = yy_axis[max_idx] + minor_disp_y;

            const float max_axis_x = xx_axis[window_size - 1];
            const float min_axis_x = xx_axis[0];
            const float max_axis_y = yy_axis[(window_size - 1) * window_size];
            const float min_axis_y = yy_axis[0];

            if (result_disp_x > max_axis_x) {
                result_disp_x = max_axis_x;
            } else if (result_disp_x < min_axis_x) {
                result_disp_x = min_axis_x;
            }
            if (result_disp_y > max_axis_y) {
                result_disp_y = max_axis_y;
            } else if (result_disp_y < min_axis_y) {
                result_disp_y = min_axis_y;
            }

            disp_y[pixel] = result_disp_y + dy_in[pixel];
            disp_x[pixel] = result_disp_x + dx_in[pixel];
            darkfield_nd[pixel] = corr_max;
        }
    }
    return {std::move(disp_y), std::move(disp_x), std::move(darkfield_nd)};
}


SolverOutput WSVT::solver() {
    const auto processing_t0 = std::chrono::steady_clock::now();
    const int solver_threads = configure_openmp_threads(n_cores_, "pyramid/wavelet/displace", 2);
    auto p = wavelet_data();
    const double pyramid_time = last_pyramid_time_s_;
    const double wavelet_time = last_wavelet_time_s_;
    std::size_t out_h = h_;
    std::size_t out_w = w_;
    std::size_t out_d = ch_;
    std::vector<float> darkfield(out_h * out_w, 0.0f);

    if (n_template_ == 0) {
        auto std_img = std_depth_chw_per_pixel(img_data_, ch_, h_, w_).take();
        auto std_ref = std_depth_chw_per_pixel(ref_data_, ch_, h_, w_).take();
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < darkfield.size(); ++i) {
            darkfield[i] = std_img[i] / (std_ref[i] + 1e-6f);
        }
    } else {
        auto img_stack = stack_TemplateWindow(img_data_, ch_, h_, w_, out_h, out_w, out_d);
        auto ref_stack = stack_TemplateWindow(ref_data_, ch_, h_, w_, out_h, out_w, out_d);
        auto std_img = std_depth_hwd(
            TensorView3D<const float, Layout::HWD>{img_stack.data(), {out_h, out_w, out_d}}).take();
        auto std_ref = std_depth_hwd(
            TensorView3D<const float, Layout::HWD>{ref_stack.data(), {out_h, out_w, out_d}}).take();
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < darkfield.size(); ++i) {
            darkfield[i] = std_img[i] / (std_ref[i] + 1e-6f);
        }
    }

    // Transmission follows the Python WSVT reference: first frame sample/ref with clipping.
    std::vector<float> transmission(out_h * out_w, 0.0f);
    #pragma omp parallel for schedule(static)
    for (std::size_t idx = 0; idx < out_h * out_w; ++idx) {
        const float den = std::max(ref_data_[idx], 1.0e-10f);
        const float ratio = img_data_[idx] / den;
        transmission[idx] = std::clamp(ratio, 0.01f, 10.0f);
    }

    const int max_pyramid_searching_window = static_cast<int>(std::ceil(static_cast<double>(cal_half_window_) / std::pow(2.0, static_cast<double>(pyramid_level_))));
    std::vector<int> searching_window_pyramid_list(static_cast<std::size_t>(pyramid_level_), n_s_extend_);
    searching_window_pyramid_list.push_back(max_pyramid_searching_window);
    std::size_t m0 = p.img_levels[0].d1;
    std::size_t n0 = p.img_levels[0].d2;
    std::vector<float> displace_y;
    std::vector<float> displace_x;
    if (displace_estimate_h_ == m0 && displace_estimate_w_ == n0) {
        displace_y = std::move(displace_estimate_y_);
        displace_x = std::move(displace_estimate_x_);
    } else {
        displace_y = resampling_spline(displace_estimate_y_, displace_estimate_h_, displace_estimate_w_, m0, n0);
        displace_x = resampling_spline(displace_estimate_x_, displace_estimate_h_, displace_estimate_w_, m0, n0);
    }
    std::vector<float> darkfield_nd(m0 * n0, 0.0f);
    double total_displace_upsample_s = 0.0;
    double total_displace_pad_s = 0.0;
    double total_displace_search_s = 0.0;
    std::string displace_detail;
    const auto displace_t0 = std::chrono::steady_clock::now();
    for (int k_iter = 0; k_iter < n_iter_; ++k_iter) {
        const float inv_scale = 1.0f / static_cast<float>(std::pow(2.0, static_cast<double>(pyramid_level_)));
        for (float& v : displace_y) v *= inv_scale;
        for (float& v : displace_x) v *= inv_scale;
        const std::size_t coarse_h = p.img_levels[static_cast<std::size_t>(pyramid_level_)].d1;
        const std::size_t coarse_w = p.img_levels[static_cast<std::size_t>(pyramid_level_)].d2;
        auto t_up0 = std::chrono::steady_clock::now();
        displace_y = resampling_spline(displace_y, m0, n0, coarse_h, coarse_w);
        displace_x = resampling_spline(displace_x, m0, n0, coarse_h, coarse_w);
        auto t_up1 = std::chrono::steady_clock::now();
        total_displace_upsample_s += std::chrono::duration<double>(t_up1 - t_up0).count();
        prColor("down sampling the dispalce to size: (" + std::to_string(coarse_h) + ", " + std::to_string(coarse_w) + ")", "green");
        const float lim_coarse = static_cast<float>(cal_half_window_) / static_cast<float>(std::pow(2.0, static_cast<double>(pyramid_level_)));
        clamp_2d(displace_y, -lim_coarse, lim_coarse);
        clamp_2d(displace_x, -lim_coarse, lim_coarse);
        for (int p_level = pyramid_level_; p_level >= 0; --p_level) {
            const int pyramid_seaching_window = searching_window_pyramid_list[static_cast<std::size_t>(p_level)];
            const std::size_t lv = static_cast<std::size_t>(p_level);
            const std::size_t ph = p.img_levels[lv].d1;
            const std::size_t pw = p.img_levels[lv].d2;
            const std::size_t depth = p.img_levels[lv].d0;
            std::vector<float> displace_pyramid_y(ph * pw, 0.0f);
            std::vector<float> displace_pyramid_x(ph * pw, 0.0f);
            if (p_level == pyramid_level_) {
                for (std::size_t i = 0; i < ph * pw; ++i) {
                    displace_pyramid_y[i] = static_cast<float>(round_half_to_even(static_cast<double>(displace_y[i])));
                    displace_pyramid_x[i] = static_cast<float>(round_half_to_even(static_cast<double>(displace_x[i])));
                }
            } else {
                const std::size_t old_h = p.img_levels[lv + 1].d1;
                const std::size_t old_w = p.img_levels[lv + 1].d2;
                t_up0 = std::chrono::steady_clock::now();
                auto up_y = resampling_spline(displace_y, old_h, old_w, ph, pw);
                auto up_x = resampling_spline(displace_x, old_h, old_w, ph, pw);
                t_up1 = std::chrono::steady_clock::now();
                total_displace_upsample_s += std::chrono::duration<double>(t_up1 - t_up0).count();
                for (std::size_t i = 0; i < ph * pw; ++i) {
                    displace_pyramid_y[i] = static_cast<float>(round_half_to_even(static_cast<double>(up_y[i] * 2.0f)));
                    displace_pyramid_x[i] = static_cast<float>(round_half_to_even(static_cast<double>(up_x[i] * 2.0f)));
                }
            }
            const float lim = static_cast<float>(cal_half_window_) / static_cast<float>(std::pow(2.0, static_cast<double>(p_level)));
            clamp_2d(displace_pyramid_y, -lim, lim);
            clamp_2d(displace_pyramid_x, -lim, lim);
            const int n_pad = static_cast<int>(std::ceil(static_cast<double>(cal_half_window_) / std::pow(2.0, static_cast<double>(p_level))));
            prColor("pyramid level: " + std::to_string(p_level) + "\nImage size: (" + std::to_string(ph) + ", " + std::to_string(pw) + ", " + std::to_string(depth) + ")\nsearching window:" + std::to_string(pyramid_seaching_window), "cyan");
            const std::size_t pad_all = static_cast<std::size_t>(n_pad + pyramid_seaching_window);
            auto t_pad0 = std::chrono::steady_clock::now();
            auto ref_wa_pad_t = pad_hwd_zero(
                TensorView3D<const float, Layout::HWD>{p.ref_levels[lv].data.data(), {ph, pw, depth}},
                pad_all, pad_all);
            const std::size_t ref_pad_h = ref_wa_pad_t.shape().d0;
            const std::size_t ref_pad_w = ref_wa_pad_t.shape().d1;
            const auto ref_wa_pad = std::move(ref_wa_pad_t).take();
            auto t_pad1 = std::chrono::steady_clock::now();
            total_displace_pad_s += std::chrono::duration<double>(t_pad1 - t_pad0).count();

            auto t_search0 = std::chrono::steady_clock::now();
            auto result = displace_wavelet(
                as_span(p.img_levels[lv].data), ph, pw,
                ref_wa_pad, ref_pad_h, ref_pad_w, depth,
                displace_pyramid_y, displace_pyramid_x,
                pyramid_seaching_window, n_pad);
            auto t_search1 = std::chrono::steady_clock::now();
            total_displace_search_s += std::chrono::duration<double>(t_search1 - t_search0).count();

            clamp_2d(result[0], -lim, lim);
            clamp_2d(result[1], -lim, lim);
            displace_y = std::move(result[0]);
            displace_x = std::move(result[1]);
            darkfield_nd = std::move(result[2]);
            prColor("displace map wrapping: (" + std::to_string(ph) + ", " + std::to_string(pw) + ")", "green");
        }
    }
    const auto displace_t1 = std::chrono::steady_clock::now();
    const double displace_time_s = std::chrono::duration<double>(displace_t1 - displace_t0).count();
    prColor("displace time: " + std::to_string(displace_time_s) + " s", "light_purple");
    prColor("  displace detail: upsample=" + std::to_string(total_displace_upsample_s) +
            "s pad=" + std::to_string(total_displace_pad_s) +
            "s search=" + std::to_string(total_displace_search_s) + "s", "light_purple");
    (void)configure_openmp_threads(1, "post-process/FFTW phase recovery", 1);
    const auto post_t0 = std::chrono::steady_clock::now();
    const std::size_t disp_h = p.img_levels[0].d1;
    const std::size_t disp_w = p.img_levels[0].d2;
    const std::size_t pad_crop = static_cast<std::size_t>(cal_half_window_);
    auto displace_y_crop_img = crop_2d(
        ImageView2D<const float>{displace_y.data(), {disp_h, disp_w}}, pad_crop);
    auto displace_x_crop_img = crop_2d(
        ImageView2D<const float>{displace_x.data(), {disp_h, disp_w}}, pad_crop);
    auto darkfield_nd_crop_img = crop_2d(
        ImageView2D<const float>{darkfield_nd.data(), {disp_h, disp_w}}, pad_crop);
    const std::size_t cropped_h = displace_y_crop_img.shape().h;
    const std::size_t cropped_w = displace_y_crop_img.shape().w;
    auto displace_y_crop = std::move(displace_y_crop_img).take();
    auto displace_x_crop = std::move(displace_x_crop_img).take();
    auto darkfield_nd_crop = std::move(darkfield_nd_crop_img).take();
    for (float& v : displace_y_crop) v = -v;
    for (float& v : displace_x_crop) v = -v;
    const double mean_dy = mean_2d(displace_y_crop);
    const double mean_dx = mean_2d(displace_x_crop);
    std::vector<float> dpc_y(displace_y_crop.size(), 0.0f);
    std::vector<float> dpc_x(displace_x_crop.size(), 0.0f);
    const double scale = p_x_ / (z_ / mag_factor_);
    auto t_fc0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < displace_y_crop.size(); ++i) {
        dpc_y[i] = static_cast<float>((static_cast<double>(displace_y_crop[i]) - mean_dy) * scale);
        dpc_x[i] = static_cast<float>((static_cast<double>(displace_x_crop[i]) - mean_dx) * scale);
    }
    auto phase = frankot_chellappa(
        ImageView2D<const float>{dpc_x.data(), {cropped_h, cropped_w}},
        ImageView2D<const float>{dpc_y.data(), {cropped_h, cropped_w}}).take();
    const double phase_scale = p_x_ * 2.0 * 3.14159265358979323846 / wavelength_;
    for (float& v : phase) {
        v = static_cast<float>(static_cast<double>(v) * phase_scale);
    }
    auto t_fc1 = std::chrono::steady_clock::now();
    const auto post_t1 = std::chrono::steady_clock::now();
    const double postprocess_time_s = std::chrono::duration<double>(post_t1 - post_t0).count();
    const double fc_time_s = std::chrono::duration<double>(t_fc1 - t_fc0).count();
    prColor("post-process time: " + std::to_string(postprocess_time_s) + " s", "light_purple");
    prColor("  post detail: dpc_conv=" + std::to_string(postprocess_time_s - fc_time_s) +
            "s fc_phase=" + std::to_string(fc_time_s) + "s", "light_purple");
#ifdef _OPENMP
    omp_set_num_threads(solver_threads);
#endif
    const auto processing_t1 = std::chrono::steady_clock::now();
    const double time_cost_s = std::chrono::duration<double>(processing_t1 - processing_t0).count();
    prColor("total time: " + std::to_string(time_cost_s) + " s", "light_purple");
    prColor("  pyramid:    " + std::to_string(pyramid_time) + " s", "light_purple");
    prColor("  wavelet:    " + std::to_string(wavelet_time) + " s", "light_purple");
    prColor("  displace:   " + std::to_string(displace_time_s) + " s", "light_purple");
    prColor("  post-proc:  " + std::to_string(postprocess_time_s) + " s", "light_purple");

    return SolverOutput{
        std::move(displace_y_crop),
        std::move(displace_x_crop),
        cropped_h,
        cropped_w,
        std::move(dpc_y),
        std::move(dpc_x),
        std::move(phase),
        std::move(transmission),
        std::move(darkfield),
        std::move(darkfield_nd_crop),
        out_h,
        out_w,
        time_cost_s,
        pyramid_time,
        wavelet_time,
        displace_time_s,
        postprocess_time_s
    };
}

SolverOutput WSVT::run(const std::string& result_path, bool cleansave) {
    const std::string started_at_utc = now_iso8601_utc();
    SolverOutput out = solver();
    const std::string ended_at_utc = now_iso8601_utc();
    if (!result_path.empty()) {
        std::vector<H5ItemF32> items;
        if (cleansave) {
            items.push_back(H5ItemF32{"displace_x", NdArrayF32{{out.h, out.w}, out.displace_x}});
            items.push_back(H5ItemF32{"displace_y", NdArrayF32{{out.h, out.w}, out.displace_y}});
            items.push_back(H5ItemF32{"transmission_image", NdArrayF32{{out.transmission_h, out.transmission_w}, out.transmission}});
            items.push_back(H5ItemF32{"darkfield", NdArrayF32{{out.transmission_h, out.transmission_w}, out.darkfield}});
            items.push_back(H5ItemF32{"darkfield_nd", NdArrayF32{{out.h, out.w}, out.darkfield_nd}});
        } else {
            items.push_back(H5ItemF32{"displace_x", NdArrayF32{{out.h, out.w}, out.displace_x}});
            items.push_back(H5ItemF32{"displace_y", NdArrayF32{{out.h, out.w}, out.displace_y}});
            items.push_back(H5ItemF32{"DPC_x", NdArrayF32{{out.h, out.w}, out.dpc_x}});
            items.push_back(H5ItemF32{"DPC_y", NdArrayF32{{out.h, out.w}, out.dpc_y}});
            items.push_back(H5ItemF32{"phase", NdArrayF32{{out.h, out.w}, out.phase}});
            items.push_back(H5ItemF32{"transmission_image", NdArrayF32{{out.transmission_h, out.transmission_w}, out.transmission}});
            items.push_back(H5ItemF32{"darkfield", NdArrayF32{{out.transmission_h, out.transmission_w}, out.darkfield}});
            items.push_back(H5ItemF32{"darkfield_nd", NdArrayF32{{out.h, out.w}, out.darkfield_nd}});
        }
        write_h5(result_path, "WSVT_result", items);

        const double phase_rms = stddev_2d(out.phase);
        const double phase_pv = pv_2d(out.phase);

        JsonObject parameter_dict;
        parameter_dict["crop"] = static_cast<double>(crop_);
        parameter_dict["N_s extend"] = static_cast<double>(n_s_extend_);
        parameter_dict["half_window"] = static_cast<double>(cal_half_window_);
        parameter_dict["energy"] = energy_;
        parameter_dict["wavelength"] = wavelength_;
        parameter_dict["p_x"] = p_x_;
        parameter_dict["d"] = z_;
        parameter_dict["cpu_cores"] = static_cast<double>(n_cores_);
        parameter_dict["n_group"] = static_cast<double>(n_group_);
        parameter_dict["wavelet_level"] = static_cast<double>(wavelet_level_);
        parameter_dict["pyramid_level"] = static_cast<double>(pyramid_level_);
        parameter_dict["n_iter"] = static_cast<double>(n_iter_);
        parameter_dict["time_cost"] = out.time_cost_s;
        parameter_dict["pyramid_time"] = out.pyramid_time_s;
        parameter_dict["wavelet_time"] = out.wavelet_time_s;
        parameter_dict["displace_time"] = out.displace_time_s;
        parameter_dict["postprocess_time"] = out.postprocess_time_s;
        parameter_dict["use_wavelet"] = use_wavelet_;
        parameter_dict["use_GPU"] = use_gpu_;
        parameter_dict["wavelet_level_cut"] = static_cast<double>(wavelet_level_cut_);
        JsonArray wavelet_add;
        for (const int v : wavelet_add_list_) {
            wavelet_add.emplace_back(v);
        }
        parameter_dict["wavelet_add"] = wavelet_add;
        parameter_dict["phase_rms"] = phase_rms;
        parameter_dict["phase_pv"] = phase_pv;
        write_json(result_path, "WSVT_result", parameter_dict);

        JsonObject events;
        events["run_id"] = "WSVT_result";
        events["solver"] = "wsvt";
        events["started_at_utc"] = started_at_utc;
        events["processing_time_s"] = out.time_cost_s;
        events["pyramid_time_s"] = out.pyramid_time_s;
        events["wavelet_time_s"] = out.wavelet_time_s;
        events["displace_time_s"] = out.displace_time_s;
        events["postprocess_time_s"] = out.postprocess_time_s;
        JsonArray stage_records;
        JsonObject pyr_stage;
        pyr_stage["stage"] = "pyramid";
        pyr_stage["elapsed_s"] = out.pyramid_time_s;
        stage_records.emplace_back(pyr_stage);
        JsonObject wavelet_stage;
        wavelet_stage["stage"] = "wavelet_transform";
        wavelet_stage["elapsed_s"] = out.wavelet_time_s;
        stage_records.emplace_back(wavelet_stage);
        JsonObject displace_stage;
        displace_stage["stage"] = "displace";
        displace_stage["elapsed_s"] = out.displace_time_s;
        stage_records.emplace_back(displace_stage);
        JsonObject post_stage;
        post_stage["stage"] = "postprocess";
        post_stage["elapsed_s"] = out.postprocess_time_s;
        stage_records.emplace_back(post_stage);
        JsonObject solver_stage;
        solver_stage["stage"] = "solver_total";
        solver_stage["elapsed_s"] = out.time_cost_s;
        stage_records.emplace_back(solver_stage);
        events["stages"] = stage_records;
        events["ended_at_utc"] = ended_at_utc;
        write_json(result_path, "WSVT_events", events);
    }
    return out;
}

}
