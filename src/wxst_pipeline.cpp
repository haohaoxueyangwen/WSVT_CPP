#include "wsvt/wxst_pipeline.hpp"

#include "wsvt/common.hpp"
#include "wsvt/console_ops.hpp"
#include "wsvt/core.hpp"
#include "wsvt/euclidean_dist.hpp"
#include "wsvt/image_ops.hpp"
#include "wsvt/io_h5.hpp"
#include "wsvt/io_json.hpp"
#include "wsvt/phase_recovery.hpp"
#include "wsvt/solver_utils.hpp"
#include "wsvt/wavelet_ops.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
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

WXST::WXST(
    const std::vector<float>& img,
    const std::vector<float>& ref,
    std::size_t h,
    std::size_t w,
    int m_image,
    int n_s,
    int cal_half_window,
    int n_s_extend,
    int n_cores,
    int n_group,
    double energy,
    double p_x,
    double z,
    int wavelet_level_cut,
    int pyramid_level,
    int n_iter,
    bool use_estimate,
    bool use_wavelet,
    int use_gpu,
    int wavelet_impl,
    int phase_cores)
    : img_data_(img),
      ref_data_(ref),
      h_(h),
      w_(w),
      m_image_(m_image),
      n_s_(n_s),
      cal_half_window_(cal_half_window),
      n_s_extend_(n_s_extend),
      n_cores_(n_cores),
      phase_cores_(phase_cores > 0 ? phase_cores : n_cores),
      n_group_(n_group),
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
    wavelet_impl_(wavelet_impl),
    wavelet_level_(0),
      last_pyramid_time_s_(0.0),
      last_wavelet_time_s_(0.0) {
    validate_manual_window_contract(
        {cal_half_window_, n_s_extend_, n_s_}, h_, w_, 0, "WXST");
    if (img_data_.size() != h_ * w_ || ref_data_.size() != h_ * w_) {
        throw std::invalid_argument("WXST init size mismatch");
    }
    if (n_s_extend_ > cal_half_window_) {
        prColor("WXST warning: n_s_extend exceeds cal_half_window; keeping the manual values unchanged", "yellow");
    }
    if (n_group_ != 1) {
        prColor("WXST warning: n_group is compatibility metadata only and does not control C++ parallelism", "yellow");
    }
    if (wavelet_impl_ < 0 || wavelet_impl_ > 2) {
        throw std::invalid_argument("WXST wavelet_impl must be 0=streamed, 1=planned, or 2=pixelchain");
    }
    if (use_gpu_) {
        prColor("Use GPU found. Enable multi-resolution", "cyan");
    } else {
        prColor("No gpu found. Use CPU instead.", "cyan");
    }
    if (use_estimate_) {
#if defined(WSVT_HAS_OPENCV)
        const auto init = slope_tracking(ref_data_.data(), img_data_.data(), h_, w_, cal_half_window_);
        displace_estimate_y_ = init[0];
        displace_estimate_x_ = init[1];
#else
        throw std::runtime_error("use_estimate requires OpenCV (WSVT_HAS_OPENCV)");
#endif
    } else {
        displace_estimate_y_.assign(h_ * w_, 0.0f);
        displace_estimate_x_.assign(h_ * w_, 0.0f);
    }
}

PyramidResult WXST::pyramid_data() {
    prColor("obtain pyramid image and stack the window with pyramid level: " + std::to_string(pyramid_level_), "green");
    if (m_image_ > 0 && static_cast<std::size_t>(m_image_) < std::min(h_, w_)) {
        auto cropped_estimate_y = image_roi(
            TensorView3D<const float, Layout::CHW>(
                displace_estimate_y_.data(), Shape3D{1, h_, w_}),
            static_cast<std::size_t>(m_image_));
        auto cropped_estimate_x = image_roi(
            TensorView3D<const float, Layout::CHW>(
                displace_estimate_x_.data(), Shape3D{1, h_, w_}),
            static_cast<std::size_t>(m_image_));
        auto cropped_ref = image_roi(
            TensorView3D<const float, Layout::CHW>(ref_data_.data(), Shape3D{1, h_, w_}),
            static_cast<std::size_t>(m_image_));
        auto cropped_img = image_roi(
            TensorView3D<const float, Layout::CHW>(img_data_.data(), Shape3D{1, h_, w_}),
            static_cast<std::size_t>(m_image_));
        ref_data_ = std::move(cropped_ref).take();
        img_data_ = std::move(cropped_img).take();
        displace_estimate_y_ = std::move(cropped_estimate_y).take();
        displace_estimate_x_ = std::move(cropped_estimate_x).take();
        h_ = static_cast<std::size_t>(m_image_);
        w_ = static_cast<std::size_t>(m_image_);
    }
    return wsvt::pyramid_data(ref_data_, img_data_, 1, h_, w_, pyramid_level_, n_s_, PyramidDownsampleMode::Db3Aa);
}

PyramidResult WXST::wavelet_data() {
    const auto pyr_t0 = std::chrono::steady_clock::now();
    auto p = pyramid_data();
    const auto pyr_t1 = std::chrono::steady_clock::now();
    last_pyramid_time_s_ = std::chrono::duration<double>(pyr_t1 - pyr_t0).count();
    last_template_window_time_s_ = p.template_window_time_s;
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

        if (wavelet_impl_ == 1 || wavelet_impl_ == 2) {
            // P0: Precompute DWT plans, then dispatch to planned or pixelchain
            std::vector<std::vector<DwtLevelPlan>> all_plans;
            all_plans.reserve(p.ref_levels.size());
            for (std::size_t lv = 0; lv < p.ref_levels.size(); ++lv) {
                all_plans.push_back(compute_wavelet_plan(
                    p.ref_levels[lv].d0, wavelet_level_, wavelet_method));
            }
            for (std::size_t lv = 0; lv < p.ref_levels.size(); ++lv) {
                int wavelevel_add = (lv >= wavelet_add_list_.size() ? 2 : wavelet_add_list_[lv]);
                const int ret_level = coefs_level + wavelevel_add;
                WaveletResult img_wa, ref_wa;
                if (wavelet_impl_ == 2) {
                    img_wa = wavelet_transform_hwd_pixelchain(
                        as_span(p.img_levels[lv].data), p.img_levels[lv].d1, p.img_levels[lv].d2,
                        p.img_levels[lv].d0, wavelet_method, wavelet_level_,
                        ret_level, all_plans[lv]);
                    ref_wa = wavelet_transform_hwd_pixelchain(
                        as_span(p.ref_levels[lv].data), p.ref_levels[lv].d1, p.ref_levels[lv].d2,
                        p.ref_levels[lv].d0, wavelet_method, wavelet_level_,
                        ret_level, all_plans[lv]);
                } else {
                    img_wa = wavelet_transform_hwd_planned(
                        as_span(p.img_levels[lv].data), p.img_levels[lv].d1, p.img_levels[lv].d2,
                        p.img_levels[lv].d0, wavelet_method, wavelet_level_,
                        ret_level, all_plans[lv]);
                    ref_wa = wavelet_transform_hwd_planned(
                        as_span(p.ref_levels[lv].data), p.ref_levels[lv].d1, p.ref_levels[lv].d2,
                        p.ref_levels[lv].d0, wavelet_method, wavelet_level_,
                        ret_level, all_plans[lv]);
                }
                p.img_levels[lv] = PyramidLevel{std::move(img_wa.coeffs_filter), img_wa.out_depth, img_wa.out_h, img_wa.out_w};
                p.ref_levels[lv] = PyramidLevel{std::move(ref_wa.coeffs_filter), ref_wa.out_depth, ref_wa.out_h, ref_wa.out_w};
                prColor("pyramid level: " + std::to_string(lv) + "\nvector length: " + std::to_string(ref_wa.out_depth), "green");
            }
        } else {
            for (std::size_t lv = 0; lv < p.ref_levels.size(); ++lv) {
                int wavelevel_add = (lv >= wavelet_add_list_.size() ? 2 : wavelet_add_list_[lv]);
                auto img_wa = wavelet_transform_hwd(
                    as_span(p.img_levels[lv].data), p.img_levels[lv].d1, p.img_levels[lv].d2, p.img_levels[lv].d0,
                    wavelet_method, wavelet_level_, coefs_level + wavelevel_add);
                auto ref_wa = wavelet_transform_hwd(
                    as_span(p.ref_levels[lv].data), p.ref_levels[lv].d1, p.ref_levels[lv].d2, p.ref_levels[lv].d0,
                    wavelet_method, wavelet_level_, coefs_level + wavelevel_add);
                p.img_levels[lv] = PyramidLevel{std::move(img_wa.coeffs_filter), img_wa.out_depth, img_wa.out_h, img_wa.out_w};
                p.ref_levels[lv] = PyramidLevel{std::move(ref_wa.coeffs_filter), ref_wa.out_depth, ref_wa.out_h, ref_wa.out_w};
                prColor("pyramid level: " + std::to_string(lv) + "\nvector length: " + std::to_string(ref_wa.out_depth), "green");
            }
        }
        const auto wavelet_t1 = std::chrono::steady_clock::now();
        last_wavelet_time_s_ = std::chrono::duration<double>(wavelet_t1 - wavelet_t0).count();
        prColor("wavelet time: " + std::to_string(last_wavelet_time_s_) + " s", "light_purple");
    } else {
        wavelet_level_ = 0;
        wavelet_add_list_.clear();
    }
    return p;
}

std::vector<float> WXST::resampling_spline(const std::vector<float>& img, std::size_t in_h, std::size_t in_w, std::size_t out_h, std::size_t out_w) const {
    if (img.size() != in_h * in_w) {
        throw std::invalid_argument("resampling_spline input size mismatch");
    }
    return std::move(resample_rect_bivariate_spline(
        ImageView2D<const float>{img.data(), {in_h, in_w}}, {out_h, out_w})).take();
}

std::array<std::vector<float>, 3> WXST::displace_wavelet(
    std::span<const float> img_wa_stack,
    std::size_t img_h,
    std::size_t img_w,
    std::span<const float> ref_wa_stack,
    std::size_t ref_h,
    std::size_t ref_w,
    std::size_t depth,
    const std::vector<float>& displace_y,
    const std::vector<float>& displace_x,
    int cal_half_window) const {
    if (img_wa_stack.size() != img_h * img_w * depth || ref_wa_stack.size() != ref_h * ref_w * depth) {
        throw std::invalid_argument("displace_wavelet stack shape mismatch");
    }
    if (displace_y.size() != img_h * img_w || displace_x.size() != img_h * img_w) {
        throw std::invalid_argument("displace_wavelet displacement shape mismatch");
    }
    const std::size_t window_size = static_cast<std::size_t>(2 * cal_half_window + 1);
    const std::size_t ws2 = window_size * window_size;

    std::vector<float> yy_axis(ws2, 0.0f);
    std::vector<float> xx_axis(ws2, 0.0f);
    for (std::size_t y = 0; y < window_size; ++y) {
        for (std::size_t x = 0; x < window_size; ++x) {
            yy_axis[y * window_size + x] = static_cast<float>(static_cast<int>(y) - cal_half_window);
            xx_axis[y * window_size + x] = static_cast<float>(static_cast<int>(x) - cal_half_window);
        }
    }

    const float pixel_res_x = (window_size >= 2) ? (xx_axis[1] - xx_axis[0]) : 1.0f;
    const float pixel_res_y = (window_size >= 2) ? (yy_axis[window_size] - yy_axis[0]) : 1.0f;

    std::vector<float> disp_y(img_h * img_w, 0.0f);
    std::vector<float> disp_x(img_h * img_w, 0.0f);
    std::vector<float> darkfield_nd(img_h * img_w, 0.0f);

    const float* __restrict__ img_ptr = std::assume_aligned<64>(img_wa_stack.data());
    const float* __restrict__ ref_ptr = std::assume_aligned<64>(ref_wa_stack.data());
    const std::size_t ref_stride = ref_w * depth;
    const long long ref_h_s = static_cast<long long>(ref_h);
    const long long ref_w_s = static_cast<long long>(ref_w);
    const int hw = cal_half_window;
    const long long win_s = static_cast<long long>(window_size);

    #pragma omp parallel
    {
        std::vector<float> corr_data(ws2, 0.0f);

        #pragma omp for schedule(guided, 64)
        for (std::size_t pixel = 0; pixel < img_h * img_w; ++pixel) {
            const std::size_t yy = pixel / img_w;
            const std::size_t xx = pixel % img_w;

            const float* img_line = img_ptr + pixel * depth;

            const int dy_int = static_cast<int>(displace_y[pixel]);
            const int dx_int = static_cast<int>(displace_x[pixel]);

            // Corner of search window in original reference coordinates
            const long long ref_y0 = static_cast<long long>(yy) + dy_int - static_cast<long long>(hw);
            const long long ref_x0 = static_cast<long long>(xx) + dx_int - static_cast<long long>(hw);

            // Precompute img squared sum (used for OOB zero-padding positions)
            float img_sq_sum = 0.0f;
            #pragma omp simd reduction(+:img_sq_sum)
            for (std::size_t k = 0; k < depth; ++k) {
                img_sq_sum += img_line[k] * img_line[k];
            }
            const float corr_oob = -img_sq_sum;  // SSD with zero ref

            // Interior: entire search window fits inside original reference bounds
            const bool interior =
                ref_y0 >= 0 && ref_y0 + win_s <= ref_h_s &&
                ref_x0 >= 0 && ref_x0 + win_s <= ref_w_s;

            float corr_max = -std::numeric_limits<float>::infinity();
            std::size_t max_idx = 0;
            float sum_abs = 0.0f;

            if (interior) {
                // Fast path: direct access, no bounds checks
                const float* ref_base = ref_ptr + (static_cast<std::size_t>(ref_y0) * ref_w
                                                   + static_cast<std::size_t>(ref_x0)) * depth;
                for (std::size_t wy = 0; wy < window_size; ++wy) {
                    for (std::size_t wx = 0; wx < window_size; ++wx) {
                        const float* ref_row = ref_base + (wy * ref_w + wx) * depth;
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
                        if (val > corr_max) { corr_max = val; max_idx = ci; }
                    }
                }
            } else {
                // Boundary: bounds-checked access (zero-padding semantics).
                // OOB reference values are treated as zero.
                for (std::size_t wy = 0; wy < window_size; ++wy) {
                    const long long ry = ref_y0 + static_cast<long long>(wy);
                    for (std::size_t wx = 0; wx < window_size; ++wx) {
                        const long long rx = ref_x0 + static_cast<long long>(wx);
                        if (ry >= 0 && ry < ref_h_s && rx >= 0 && rx < ref_w_s) {
                            const float* ref_row = ref_ptr
                                + static_cast<std::size_t>(ry) * ref_stride
                                + static_cast<std::size_t>(rx) * depth;
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
                            if (val > corr_max) { corr_max = val; max_idx = ci; }
                        } else {
                            // OOB: reference is zero → SSD = sum(img²)
                            corr_data[wy * window_size + wx] = corr_oob;
                            sum_abs += img_sq_sum;  // |corr_oob| = img_sq_sum
                            if (corr_oob > corr_max) {
                                corr_max = corr_oob;
                                max_idx = wy * window_size + wx;
                            }
                        }
                    }
                }
            }

            const std::size_t max_y_idx = max_idx / window_size;
            const std::size_t max_x_idx = max_idx % window_size;

            const auto sample_corr = [&](long long r, long long c) -> float {
                const auto rc = static_cast<std::size_t>(std::clamp<long long>(r, 0, win_s - 1));
                const auto cc = static_cast<std::size_t>(std::clamp<long long>(c, 0, win_s - 1));
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

            if (result_disp_x > max_axis_x) result_disp_x = max_axis_x;
            else if (result_disp_x < min_axis_x) result_disp_x = min_axis_x;
            if (result_disp_y > max_axis_y) result_disp_y = max_axis_y;
            else if (result_disp_y < min_axis_y) result_disp_y = min_axis_y;

            disp_y[pixel] = result_disp_y + displace_y[pixel];
            disp_x[pixel] = result_disp_x + displace_x[pixel];
            darkfield_nd[pixel] = corr_max;
        }
    }
    return {std::move(disp_y), std::move(disp_x), std::move(darkfield_nd)};
}

WXSTOutput WXST::solver() {
    const auto processing_t0 = std::chrono::steady_clock::now();
    const int solver_threads = configure_openmp_threads(n_cores_, "pyramid/wavelet/displace", 2);
    auto p = wavelet_data();
    const double pyramid_time = last_pyramid_time_s_;
    const double wavelet_time = last_wavelet_time_s_;
    const auto searching_window_pyramid_list = derived_search_half_windows(
        pyramid_level_, cal_half_window_, n_s_extend_);
    std::vector<float> displace_y = std::move(displace_estimate_y_);
    std::vector<float> displace_x = std::move(displace_estimate_x_);
    std::vector<float> darkfield_nd;
    const auto displace_t0 = std::chrono::steady_clock::now();
    for (int k_iter = 0; k_iter < n_iter_; ++k_iter) {
        const float inv_scale = 1.0f / static_cast<float>(std::pow(2.0, static_cast<double>(pyramid_level_)));
        for (float& v : displace_y) v *= inv_scale;
        for (float& v : displace_x) v *= inv_scale;
        const std::size_t coarse_h = p.img_levels[static_cast<std::size_t>(pyramid_level_)].d1;
        const std::size_t coarse_w = p.img_levels[static_cast<std::size_t>(pyramid_level_)].d2;
        displace_y = resampling_spline(displace_y, h_, w_, coarse_h, coarse_w);
        displace_x = resampling_spline(displace_x, h_, w_, coarse_h, coarse_w);
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
                auto up_y = resampling_spline(displace_y, old_h, old_w, ph, pw);
                auto up_x = resampling_spline(displace_x, old_h, old_w, ph, pw);
                for (std::size_t i = 0; i < ph * pw; ++i) {
                    displace_pyramid_y[i] = static_cast<float>(round_half_to_even(static_cast<double>(up_y[i] * 2.0f)));
                    displace_pyramid_x[i] = static_cast<float>(round_half_to_even(static_cast<double>(up_x[i] * 2.0f)));
                }
            }
            const float lim = static_cast<float>(cal_half_window_) / static_cast<float>(std::pow(2.0, static_cast<double>(p_level)));
            clamp_2d(displace_pyramid_y, -lim, lim);
            clamp_2d(displace_pyramid_x, -lim, lim);
            prColor("pyramid level: " + std::to_string(p_level) + "\nImage size: (" + std::to_string(ph) + ", " + std::to_string(pw) + ", " + std::to_string(depth) + ")\nsearching window:" + std::to_string(pyramid_seaching_window), "cyan");

            auto result = displace_wavelet(
                as_span(p.img_levels[lv].data), ph, pw,
                as_span(p.ref_levels[lv].data), ph, pw, depth,
                displace_pyramid_y, displace_pyramid_x,
                pyramid_seaching_window);

            clamp_2d(result[0], -lim, lim);
            clamp_2d(result[1], -lim, lim);
            displace_y = std::move(result[0]);
            displace_x = std::move(result[1]);
            darkfield_nd = std::move(result[2]);
        }
    }
    const auto displace_t1 = std::chrono::steady_clock::now();
    const double displace_time_s = std::chrono::duration<double>(displace_t1 - displace_t0).count();
    prColor("displace time: " + std::to_string(displace_time_s) + " s", "light_purple");
    (void)configure_openmp_threads(phase_cores_, "post-process/FFTW phase recovery", 1);
    const auto post_t0 = std::chrono::steady_clock::now();
    std::vector<float> transmission(h_ * w_, 0.0f);
    for (std::size_t i = 0; i < transmission.size(); ++i) {
        const double ref_safe = std::max(static_cast<double>(ref_data_[i]), 1e-10);
        const double ratio = static_cast<double>(img_data_[i]) / ref_safe;
        transmission[i] = static_cast<float>(std::clamp(ratio, 0.01, 10.0));
    }
    const auto t_xmsn1 = std::chrono::steady_clock::now();
    const std::size_t pad_crop = static_cast<std::size_t>(cal_half_window_);
    auto displace_y_crop_img = crop_2d(
        ImageView2D<const float>{displace_y.data(), {h_, w_}}, pad_crop);
    auto displace_x_crop_img = crop_2d(
        ImageView2D<const float>{displace_x.data(), {h_, w_}}, pad_crop);
    auto darkfield_nd_crop_img = crop_2d(
        ImageView2D<const float>{darkfield_nd.data(), {h_, w_}}, pad_crop);
    const std::size_t out_h = displace_y_crop_img.shape().h;
    const std::size_t out_w = displace_y_crop_img.shape().w;
    auto displace_y_crop = std::move(displace_y_crop_img).take();
    auto displace_x_crop = std::move(displace_x_crop_img).take();
    auto darkfield_nd_crop = std::move(darkfield_nd_crop_img).take();
    for (float& v : displace_y_crop) v = -v;
    for (float& v : displace_x_crop) v = -v;
    const double mean_dy = mean_2d(displace_y_crop);
    const double mean_dx = mean_2d(displace_x_crop);
    const auto t_dpc0 = std::chrono::steady_clock::now();
    std::vector<float> dpc_y(displace_y_crop.size(), 0.0f);
    std::vector<float> dpc_x(displace_x_crop.size(), 0.0f);
    const double scale = p_x_ / z_;
    for (std::size_t i = 0; i < dpc_y.size(); ++i) {
        dpc_y[i] = static_cast<float>((static_cast<double>(displace_y_crop[i]) - mean_dy) * scale);
        dpc_x[i] = static_cast<float>((static_cast<double>(displace_x_crop[i]) - mean_dx) * scale);
    }
    const auto t_phase0 = std::chrono::steady_clock::now();
    auto phase = frankot_chellappa(
        ImageView2D<const float>{dpc_x.data(), {out_h, out_w}},
        ImageView2D<const float>{dpc_y.data(), {out_h, out_w}}).take();
    const double phase_scale = p_x_ * 2.0 * 3.14159265358979323846 / wavelength_;
    apply_python_reference_phase_scale(phase, phase_scale);
    const auto post_t1 = std::chrono::steady_clock::now();
    const double postprocess_time_s = std::chrono::duration<double>(post_t1 - post_t0).count();
    const double transmission_time_s = std::chrono::duration<double>(t_xmsn1 - post_t0).count();
    const double crop_sign_time_s = std::chrono::duration<double>(t_dpc0 - t_xmsn1).count();
    const double dpc_conv_time_s = std::chrono::duration<double>(t_phase0 - t_dpc0).count();
    const double phase_recovery_time_s = std::chrono::duration<double>(post_t1 - t_phase0).count();
    prColor("post-process time: " + std::to_string(postprocess_time_s) + " s", "light_purple");
    prColor("  post detail: transmission=" + std::to_string(transmission_time_s) +
            "s crop_sign=" + std::to_string(crop_sign_time_s) +
            "s dpc_conv=" + std::to_string(dpc_conv_time_s) +
            "s phase_recovery=" + std::to_string(phase_recovery_time_s) + "s", "light_purple");
#ifdef _OPENMP
    omp_set_num_threads(solver_threads);
#endif
    const auto processing_t1 = std::chrono::steady_clock::now();
    const double time_cost_s = std::chrono::duration<double>(processing_t1 - processing_t0).count();
    prColor("total time: " + std::to_string(time_cost_s) + " s", "light_purple");
    prColor("  pyramid:    " + std::to_string(pyramid_time) + " s", "light_purple");
    if (last_template_window_time_s_ > 0.0) {
        prColor("  tmpl_win:   " + std::to_string(last_template_window_time_s_) + " s", "light_purple");
    }
    prColor("  wavelet:    " + std::to_string(wavelet_time) + " s", "light_purple");
    prColor("  displace:   " + std::to_string(displace_time_s) + " s", "light_purple");
    prColor("  post-proc:  " + std::to_string(postprocess_time_s) + " s", "light_purple");

    return WXSTOutput{
        std::move(displace_y_crop),
        std::move(displace_x_crop),
        out_h,
        out_w,
        std::move(dpc_y),
        std::move(dpc_x),
        std::move(phase),
        std::move(transmission),
        h_,
        w_,
        std::move(darkfield_nd_crop),
        time_cost_s,
        pyramid_time,
        wavelet_time,
        displace_time_s,
        postprocess_time_s,
        last_template_window_time_s_,
        transmission_time_s,
        crop_sign_time_s,
        dpc_conv_time_s,
        phase_recovery_time_s
    };
}

WXSTOutput WXST::run(const std::string& result_path, int h5_deflate) {
    const std::string started_at_utc = now_iso8601_utc();
    auto out = solver();
    if (!result_path.empty()) {
        const auto t_write_t0 = std::chrono::steady_clock::now();
        std::vector<H5ItemF32> items{
            H5ItemF32{"displace_x", NdArrayF32{{out.h, out.w}, out.displace_x}},
            H5ItemF32{"displace_y", NdArrayF32{{out.h, out.w}, out.displace_y}},
            H5ItemF32{"DPC_x", NdArrayF32{{out.h, out.w}, out.dpc_x}},
            H5ItemF32{"DPC_y", NdArrayF32{{out.h, out.w}, out.dpc_y}},
            H5ItemF32{"phase", NdArrayF32{{out.h, out.w}, out.phase}},
            H5ItemF32{"transmission_image", NdArrayF32{{out.transmission_h, out.transmission_w}, out.transmission}},
            H5ItemF32{"darkfield_nd", NdArrayF32{{out.h, out.w}, out.darkfield_nd}}
        };
        write_h5(result_path, "WXST_result", items, h5_deflate);

        const double phase_rms = stddev_2d(out.phase);
        const double phase_pv = pv_2d(out.phase);
        const auto t_write_t1 = std::chrono::steady_clock::now();
        out.result_write_time_s = std::chrono::duration<double>(t_write_t1 - t_write_t0).count();

        JsonObject parameter_dict;
        parameter_dict["half_window"] = static_cast<double>(cal_half_window_);
        parameter_dict["N_s extend"] = static_cast<double>(n_s_extend_);
        parameter_dict["n_s"] = static_cast<double>(n_s_);
        parameter_dict["window_policy"] = "manual_fixed";
        parameter_dict["semantics_profile"] = std::string(kPythonReferenceSemantics);
        parameter_dict["input_axis_order"] = "HW";
        parameter_dict["descriptor_axis_order"] = "HWD";
        parameter_dict["output_dtype"] = "float32";
        parameter_dict["phase_convention"] = "-frankot_chellappa(DPC_x,DPC_y)*p_x*2pi/wavelength";
        parameter_dict["valid_roi_margin_input_px"] =
            static_cast<double>(n_s_ + cal_half_window_);
        parameter_dict["n_group_effective"] = false;
        JsonArray search_half_windows;
        for (const int radius : derived_search_half_windows(
                 pyramid_level_, cal_half_window_, n_s_extend_)) {
            search_half_windows.emplace_back(radius);
        }
        parameter_dict["search_half_window_by_level"] = search_half_windows;
        parameter_dict["energy"] = energy_;
        parameter_dict["wavelength"] = wavelength_;
        parameter_dict["p_x"] = p_x_;
        parameter_dict["d"] = z_;
        parameter_dict["cpu_cores"] = static_cast<double>(n_cores_);
        parameter_dict["phase_cores"] = static_cast<double>(phase_cores_);
        parameter_dict["n_group"] = static_cast<double>(n_group_);
        parameter_dict["wavelet_level"] = static_cast<double>(wavelet_level_);
        parameter_dict["wavelet_impl"] = static_cast<double>(wavelet_impl_);
        parameter_dict["pyramid_level"] = static_cast<double>(pyramid_level_);
        parameter_dict["n_iter"] = static_cast<double>(n_iter_);
        parameter_dict["time_cost"] = out.time_cost_s;
        parameter_dict["pyramid_time"] = out.pyramid_time_s;
        parameter_dict["template_window_time"] = out.template_window_time_s;
        parameter_dict["wavelet_time"] = out.wavelet_time_s;
        parameter_dict["displace_time"] = out.displace_time_s;
        parameter_dict["postprocess_time"] = out.postprocess_time_s;
        parameter_dict["post_transmission_time"] = out.post_transmission_time_s;
        parameter_dict["post_crop_sign_time"] = out.post_crop_sign_time_s;
        parameter_dict["post_dpc_conv_time"] = out.post_dpc_conv_time_s;
        parameter_dict["post_phase_recovery_time"] = out.post_phase_recovery_time_s;
        parameter_dict["result_write_time"] = out.result_write_time_s;
        parameter_dict["h5_deflate"] = static_cast<double>(h5_deflate);
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
        write_json(result_path, "WXST_result", parameter_dict);

        JsonObject events;
        events["run_id"] = "WXST_result";
        events["solver"] = "wxst";
        events["started_at_utc"] = started_at_utc;
        events["processing_time_s"] = out.time_cost_s;
        events["pyramid_time_s"] = out.pyramid_time_s;
        events["template_window_time_s"] = out.template_window_time_s;
        events["wavelet_time_s"] = out.wavelet_time_s;
        events["wavelet_impl"] = static_cast<double>(wavelet_impl_);
        events["displace_time_s"] = out.displace_time_s;
        events["postprocess_time_s"] = out.postprocess_time_s;
        events["post_transmission_time_s"] = out.post_transmission_time_s;
        events["post_crop_sign_time_s"] = out.post_crop_sign_time_s;
        events["post_dpc_conv_time_s"] = out.post_dpc_conv_time_s;
        events["post_phase_recovery_time_s"] = out.post_phase_recovery_time_s;
        events["result_write_time_s"] = out.result_write_time_s;
        events["h5_deflate"] = static_cast<double>(h5_deflate);
        events["cpu_cores"] = static_cast<double>(n_cores_);
        events["phase_cores"] = static_cast<double>(phase_cores_);
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
        events["ended_at_utc"] = now_iso8601_utc();
        write_json(result_path, "WXST_events", events);
    }
    return out;
}

}
