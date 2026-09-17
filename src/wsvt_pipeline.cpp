#include "wsvt/wsvt_pipeline.hpp"

#include "wsvt/common.hpp"
#include "wsvt/console_ops.hpp"
#include "wsvt/core.hpp"
#include "wsvt/euclidean_dist.hpp"
#include "wsvt/image_ops.hpp"
#include "wsvt/io_h5.hpp"
#include "wsvt/io_json.hpp"
#include "wsvt/phase_recovery.hpp"
#include "wsvt/search_pruning.hpp"
#include "wsvt/solver_utils.hpp"
#include "wsvt/template_window.hpp"
#include "wsvt/wavelet_ops.hpp"
#include "wsvt/wsvt_umpa_refine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include "wsvt/guarded_subpixel.hpp"
#include <sstream>
#include <span>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace wsvt {

namespace {

void validate_search_pruning_options(
    bool early_abandon,
    bool two_pass,
    bool guard_cache,
    int top_k,
    int block_size,
    int prefix_size) {
    if (early_abandon && two_pass) {
        throw std::invalid_argument(
            "WSVT search_early_abandon and search_two_pass are mutually exclusive");
    }
    if (guard_cache && !two_pass) {
        throw std::invalid_argument(
            "WSVT search_guard_cache requires search_two_pass");
    }
    if (top_k < 2 || top_k > static_cast<int>(kMaxExactTopK)) {
        throw std::invalid_argument("WSVT search_top_k must be in [2, 4]");
    }
    if (block_size <= 0) {
        throw std::invalid_argument("WSVT search_block_size must be positive");
    }
    if (prefix_size <= 0) {
        throw std::invalid_argument("WSVT search_prefix_size must be positive");
    }
}

const char* search_profile_name(
    bool early_abandon,
    bool two_pass,
    bool guard_cache) noexcept {
    if (two_pass) {
        if (guard_cache) {
            return "exact_topk_two_pass_cached_guard_v1";
        }
        return "exact_topk_two_pass_prefix_v1";
    }
    if (early_abandon) {
        return "exact_topk_blockwise_early_abandon_v1";
    }
    return "exhaustive_ssd_v1";
}

struct SparseDescriptorBuild {
    PyramidLevel level;
    std::uint64_t selected_pixel_count = 0;
    double gather_scatter_time_s = 0.0;
    double wavelet_time_s = 0.0;
};

SparseDescriptorBuild build_sparse_temporal_descriptor(
    const RawPyramidLevel& raw,
    std::span<const std::uint8_t> selected_mask,
    bool use_wavelet,
    int wavelet_level,
    int return_level) {
    const std::size_t plane = raw.h * raw.w;
    if (raw.data.size() != raw.ch * plane || selected_mask.size() != plane) {
        throw std::invalid_argument(
            "build_sparse_temporal_descriptor shape mismatch");
    }

    const auto gather_t0 = std::chrono::steady_clock::now();
    std::vector<std::size_t> selected_pixels;
    selected_pixels.reserve(static_cast<std::size_t>(std::count(
        selected_mask.begin(), selected_mask.end(), static_cast<std::uint8_t>(1))));
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        if (selected_mask[pixel] != 0) {
            selected_pixels.push_back(pixel);
        }
    }
    if (selected_pixels.empty()) {
        throw std::invalid_argument(
            "build_sparse_temporal_descriptor requires a non-empty mask");
    }

    AlignedVector<float> compact(selected_pixels.size() * raw.ch, 0.0f);
    #pragma omp parallel for schedule(static)
    for (std::size_t selected = 0; selected < selected_pixels.size(); ++selected) {
        const std::size_t pixel = selected_pixels[selected];
        float* dst = compact.data() + selected * raw.ch;
        for (std::size_t frame = 0; frame < raw.ch; ++frame) {
            dst[frame] = raw.data[frame * plane + pixel];
        }
    }
    const auto gather_t1 = std::chrono::steady_clock::now();

    WaveletResult transformed;
    const auto wavelet_t0 = std::chrono::steady_clock::now();
    if (use_wavelet) {
        transformed = wavelet_transform_hwd(
            as_span(compact), selected_pixels.size(), 1, raw.ch,
            WaveletFamily::Db2, wavelet_level, return_level);
    } else {
        transformed.coeffs_filter = std::move(compact);
        transformed.out_depth = raw.ch;
        transformed.out_h = selected_pixels.size();
        transformed.out_w = 1;
    }
    const auto wavelet_t1 = std::chrono::steady_clock::now();

    const auto scatter_t0 = std::chrono::steady_clock::now();
    AlignedVector<float> dense(plane * transformed.out_depth, 0.0f);
    #pragma omp parallel for schedule(static)
    for (std::size_t selected = 0; selected < selected_pixels.size(); ++selected) {
        const float* src = transformed.coeffs_filter.data() +
            selected * transformed.out_depth;
        float* dst = dense.data() +
            selected_pixels[selected] * transformed.out_depth;
        std::copy_n(src, transformed.out_depth, dst);
    }
    const auto scatter_t1 = std::chrono::steady_clock::now();

    SparseDescriptorBuild result;
    result.level = PyramidLevel{
        std::move(dense), transformed.out_depth, raw.h, raw.w};
    result.selected_pixel_count = static_cast<std::uint64_t>(
        selected_pixels.size());
    result.gather_scatter_time_s =
        std::chrono::duration<double>(gather_t1 - gather_t0).count() +
        std::chrono::duration<double>(scatter_t1 - scatter_t0).count();
    result.wavelet_time_s =
        std::chrono::duration<double>(wavelet_t1 - wavelet_t0).count();
    return result;
}

std::vector<std::uint8_t> candidate_reference_mask(
    std::span<const std::uint8_t> active_mask,
    std::size_t h,
    std::size_t w,
    const std::vector<float>& displace_y,
    const std::vector<float>& displace_x,
    int search_half_window) {
    const std::size_t plane = h * w;
    if (active_mask.size() != plane || displace_y.size() != plane ||
        displace_x.size() != plane) {
        throw std::invalid_argument("candidate_reference_mask shape mismatch");
    }
    std::vector<std::uint8_t> mask(plane, 0);
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        if (active_mask[pixel] == 0) {
            continue;
        }
        const long long y = static_cast<long long>(pixel / w);
        const long long x = static_cast<long long>(pixel % w);
        const long long dy = static_cast<long long>(displace_y[pixel]);
        const long long dx = static_cast<long long>(displace_x[pixel]);
        for (int local_y = -search_half_window;
             local_y <= search_half_window; ++local_y) {
            const long long ref_y = y + dy + local_y;
            if (ref_y < 0 || ref_y >= static_cast<long long>(h)) {
                continue;
            }
            for (int local_x = -search_half_window;
                 local_x <= search_half_window; ++local_x) {
                const long long ref_x = x + dx + local_x;
                if (ref_x < 0 || ref_x >= static_cast<long long>(w)) {
                    continue;
                }
                mask[static_cast<std::size_t>(ref_y) * w +
                     static_cast<std::size_t>(ref_x)] = 1;
            }
        }
    }
    return mask;
}

} // namespace

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
    int use_gpu,
    bool calc_darkfield,
    int phase_cores,
    bool search_early_abandon,
    int search_top_k,
    int search_block_size,
    bool search_two_pass,
    int search_prefix_size,
    bool search_guard_cache)
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
      phase_cores_(phase_cores > 0 ? phase_cores : n_cores),
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
      calc_darkfield_(calc_darkfield),
      search_early_abandon_(search_early_abandon),
      search_two_pass_(search_two_pass),
      search_top_k_(search_top_k),
      search_block_size_(search_block_size),
      search_prefix_size_(search_prefix_size),
      search_guard_cache_(search_guard_cache),
      wavelet_level_(0),
      displace_estimate_h_(h_),
      displace_estimate_w_(w_),
      last_pyramid_time_s_(0.0),
      last_wavelet_time_s_(0.0) {
    validate_manual_window_contract(
        {cal_half_window_, n_s_extend_, n_template_}, h_, w_, crop_, "WSVT");
    validate_search_pruning_options(
        search_early_abandon_, search_two_pass_, search_guard_cache_, search_top_k_,
        search_block_size_, search_prefix_size_);
    if (img_data_.size() != ch_ * h_ * w_ || ref_data_.size() != ch_ * h_ * w_) {
        throw std::invalid_argument("WSVT init size mismatch");
    }
    if (n_s_extend_ > cal_half_window_) {
        prColor("WSVT warning: n_s_extend exceeds cal_half_window; keeping the manual values unchanged", "yellow");
    }
    if (n_group_ != 1) {
        prColor("WSVT warning: n_group is compatibility metadata only and does not control C++ parallelism", "yellow");
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
    int use_gpu,
    bool calc_darkfield,
    int phase_cores,
    bool search_early_abandon,
    int search_top_k,
    int search_block_size,
    bool search_two_pass,
    int search_prefix_size,
    bool search_guard_cache)
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
      phase_cores_(phase_cores > 0 ? phase_cores : n_cores),
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
      calc_darkfield_(calc_darkfield),
      search_early_abandon_(search_early_abandon),
      search_two_pass_(search_two_pass),
      search_top_k_(search_top_k),
      search_block_size_(search_block_size),
      search_prefix_size_(search_prefix_size),
      search_guard_cache_(search_guard_cache),
      wavelet_level_(0),
      displace_estimate_h_(h_),
      displace_estimate_w_(w_),
      last_pyramid_time_s_(0.0),
      last_wavelet_time_s_(0.0) {
    validate_manual_window_contract(
        {cal_half_window_, n_s_extend_, n_template_}, h_, w_, crop_, "WSVT");
    validate_search_pruning_options(
        search_early_abandon_, search_two_pass_, search_guard_cache_, search_top_k_,
        search_block_size_, search_prefix_size_);
    if (img_data_.size() != ch_ * h_ * w_ || ref_data_.size() != ch_ * h_ * w_) {
        throw std::invalid_argument("WSVT init size mismatch");
    }
    if (n_s_extend_ > cal_half_window_) {
        prColor("WSVT warning: n_s_extend exceeds cal_half_window; keeping the manual values unchanged", "yellow");
    }
    if (n_group_ != 1) {
        prColor("WSVT warning: n_group is compatibility metadata only and does not control C++ parallelism", "yellow");
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

void WSVT::crop_inputs_if_requested() {
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
}

PyramidResult WSVT::pyramid_data() {
    prColor("obtain pyramid image with pyramid level: " + std::to_string(pyramid_level_), "green");
    crop_inputs_if_requested();
    auto p = wsvt::pyramid_data_consume(
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

PyramidResult WSVT::wavelet_data_for_frame_prefix(
    std::size_t frame_count,
    double& pyramid_time_s,
    double& template_window_time_s,
    double& wavelet_time_s) {
    if (frame_count == 0 || frame_count > ch_) {
        throw std::invalid_argument("WSVT frame prefix must be in [1, ch]");
    }
    if (frame_count == ch_) {
        auto result = wavelet_data();
        pyramid_time_s = last_pyramid_time_s_;
        template_window_time_s = last_template_window_time_s_;
        wavelet_time_s = last_wavelet_time_s_;
        return result;
    }

    const std::size_t plane = h_ * w_;
    std::vector<float> img_prefix(
        img_data_.begin(),
        img_data_.begin() + static_cast<std::ptrdiff_t>(frame_count * plane));
    std::vector<float> ref_prefix(
        ref_data_.begin(),
        ref_data_.begin() + static_cast<std::ptrdiff_t>(frame_count * plane));
    auto saved_img = std::move(img_data_);
    auto saved_ref = std::move(ref_data_);
    const std::size_t saved_ch = ch_;
    img_data_ = std::move(img_prefix);
    ref_data_ = std::move(ref_prefix);
    ch_ = frame_count;
    try {
        auto result = wavelet_data();
        pyramid_time_s = last_pyramid_time_s_;
        template_window_time_s = last_template_window_time_s_;
        wavelet_time_s = last_wavelet_time_s_;
        img_data_ = std::move(saved_img);
        ref_data_ = std::move(saved_ref);
        ch_ = saved_ch;
        return result;
    } catch (...) {
        img_data_ = std::move(saved_img);
        ref_data_ = std::move(saved_ref);
        ch_ = saved_ch;
        throw;
    }
}

std::vector<float> WSVT::temporal_speckle_contrast_map(
    std::size_t frame_count) const {
    if (frame_count == 0 || frame_count > ch_) {
        throw std::invalid_argument("WSVT contrast frame count must be in [1, ch]");
    }
    const std::size_t plane = h_ * w_;
    std::vector<float> contrast(plane, 0.0f);
    constexpr double kEps = 1.0e-12;
    #pragma omp parallel for schedule(static)
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        double img_sum = 0.0;
        double img_sum_sq = 0.0;
        double ref_sum = 0.0;
        double ref_sum_sq = 0.0;
        for (std::size_t frame = 0; frame < frame_count; ++frame) {
            const double img = static_cast<double>(img_data_[frame * plane + pixel]);
            const double ref = static_cast<double>(ref_data_[frame * plane + pixel]);
            img_sum += img;
            img_sum_sq += img * img;
            ref_sum += ref;
            ref_sum_sq += ref * ref;
        }
        const double inv_n = 1.0 / static_cast<double>(frame_count);
        const double img_mean = img_sum * inv_n;
        const double ref_mean = ref_sum * inv_n;
        const double img_var = std::max(0.0, img_sum_sq * inv_n - img_mean * img_mean);
        const double ref_var = std::max(0.0, ref_sum_sq * inv_n - ref_mean * ref_mean);
        const double img_cv = std::sqrt(img_var) / std::max(std::fabs(img_mean), kEps);
        const double ref_cv = std::sqrt(ref_var) / std::max(std::fabs(ref_mean), kEps);
        contrast[pixel] = static_cast<float>(std::min(img_cv, ref_cv));
    }
    return contrast;
}

std::vector<float> WSVT::resampling_spline(const std::vector<float>& img, std::size_t in_h, std::size_t in_w, std::size_t out_h, std::size_t out_w) const {
    if (img.size() != in_h * in_w) {
        throw std::invalid_argument("resampling_spline input size mismatch");
    }
    return std::move(resample_rect_bivariate_spline(
        ImageView2D<const float>{img.data(), {in_h, in_w}}, {out_h, out_w})).take();
}

DisplaceWaveletOutput WSVT::displace_wavelet(
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
    int n_pad,
    std::span<const std::uint8_t> active_mask,
    std::span<const std::uint8_t> easy_mask,
    std::span<const float> temporal_contrast,
    std::span<const int> set_hypothesis_y,
    std::span<const int> set_hypothesis_x,
    int current_pyramid_level) const {
    if (img_wa_stack.size() != img_h * img_w * depth || ref_wa_stack.size() != ref_h * ref_w * depth) {
        throw std::invalid_argument("displace_wavelet stack shape mismatch");
    }
    if (displace_y.size() != img_h * img_w || displace_x.size() != img_h * img_w) {
        throw std::invalid_argument("displace_wavelet displacement shape mismatch");
    }
    const std::size_t pixel_count = img_h * img_w;
    const bool use_spatial_cost =
        fixed_spatial_support_ != FixedSpatialSupport::Point &&
        current_pyramid_level == 0;
    std::optional<SpatialFeatureView> spatial_sample, spatial_ref;
    const FixedSpatialCost spatial_cost(fixed_spatial_support_);
    const bool use_spatial_reuse=use_spatial_cost&&fixed_spatial_reuse_;
    if(use_spatial_reuse) spatial_reuse_stats_={};
    const bool use_guarded_subpixel = guarded_subpixel_ && current_pyramid_level == 0;
    if(use_guarded_subpixel) {
        guarded_subpixel_reasons_.assign(pixel_count,1);
        guarded_subpixel_offsets_.assign(2*pixel_count,0.0);
    }
    if (use_spatial_cost) {
        // Validate once outside the OpenMP region. Existing B0 has no new scan.
        spatial_sample.emplace(img_wa_stack, img_h, img_w, depth);
        spatial_ref.emplace(ref_wa_stack, ref_h, ref_w, depth);
    }
    if (!active_mask.empty() && active_mask.size() != pixel_count) {
        throw std::invalid_argument("displace_wavelet active mask shape mismatch");
    }
    if (!easy_mask.empty() && easy_mask.size() != pixel_count) {
        throw std::invalid_argument("displace_wavelet easy mask shape mismatch");
    }
    if (!temporal_contrast.empty() && temporal_contrast.size() != pixel_count) {
        throw std::invalid_argument("displace_wavelet contrast shape mismatch");
    }
    constexpr std::size_t kSetWidth = 4;
    const bool use_set_transport = !set_hypothesis_y.empty();
    if (set_hypothesis_y.empty() != set_hypothesis_x.empty() ||
        (use_set_transport &&
         (set_hypothesis_y.size() != pixel_count * kSetWidth ||
          set_hypothesis_x.size() != pixel_count * kSetWidth))) {
        throw std::invalid_argument(
            "displace_wavelet fixed set hypothesis shape mismatch");
    }
    if (use_set_transport && (!fixed_set_transport_ || current_pyramid_level != 0)) {
        throw std::invalid_argument(
            "fixed set hypotheses are valid only at the final pyramid level");
    }
    const std::size_t window_size = static_cast<std::size_t>(2 * cal_half_window + 1);
    const std::size_t ws2 = window_size * window_size;
    if (static_cast<std::size_t>(search_top_k_) > ws2) {
        throw std::invalid_argument("WSVT search_top_k exceeds the candidate window");
    }

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
    std::vector<float> best_score_neg_ssd(img_h * img_w, 0.0f);
    std::vector<float> second_best_score_neg_ssd(img_h * img_w, 0.0f);
    std::vector<float> score_margin(img_h * img_w, 0.0f);
    std::vector<float> peak_hessian_det(img_h * img_w, 0.0f);
    std::vector<float> normalized_peak_curvature(img_h * img_w, 0.0f);
    std::vector<float> interlevel_displacement_delta(img_h * img_w, 0.0f);
    std::vector<float> search_boundary_hit(img_h * img_w, 0.0f);
    std::vector<float> search_geometry_valid(img_h * img_w, 0.0f);
    std::vector<float> confidence_accept(img_h * img_w, 0.0f);
    std::vector<float> effective_search_half_window(
        img_h * img_w, static_cast<float>(cal_half_window));
    std::vector<float> confidence_path(img_h * img_w, 0.0f);
    const bool capture_integer_proposal =
        wavelet_guided_umpa_.enabled || use_set_transport;
    std::vector<int> integer_displace_y(
        capture_integer_proposal ? img_h * img_w : 0U, 0);
    std::vector<int> integer_displace_x(
        capture_integer_proposal ? img_h * img_w : 0U, 0);
    const bool capture_coarse_set = fixed_set_transport_ &&
        current_pyramid_level == pyramid_level_;
    std::vector<float> coarse_set_hypothesis_y(
        capture_coarse_set ? pixel_count * kSetWidth : 0U, 0.0f);
    std::vector<float> coarse_set_hypothesis_x(
        capture_coarse_set ? pixel_count * kSetWidth : 0U, 0.0f);
    std::vector<float> set_transport_candidates_evaluated(
        use_set_transport ? pixel_count : 0U, 0.0f);
    std::vector<int> set_transport_representative_y(
        use_set_transport ? pixel_count * kSetWidth : 0U, 0);
    std::vector<int> set_transport_representative_x(
        use_set_transport ? pixel_count * kSetWidth : 0U, 0);
    const std::size_t npad = static_cast<std::size_t>(n_pad);

    if (!is_pointer_aligned<64>(img_wa_stack.data()) ||
        !is_pointer_aligned<64>(ref_wa_stack.data())) {
        throw std::invalid_argument(
            "displace_wavelet requires 64-byte aligned sample/reference storage");
    }
    const float* __restrict__ img_ptr = std::assume_aligned<64>(img_wa_stack.data());
    const float* __restrict__ ref_ptr = std::assume_aligned<64>(ref_wa_stack.data());
    const float* __restrict__ dy_in = displace_y.data();
    const float* __restrict__ dx_in = displace_x.data();

    std::uint64_t search_candidate_count = 0;
    std::uint64_t search_abandoned_candidate_count = 0;
    std::uint64_t search_distance_terms_evaluated = 0;
    std::uint64_t search_distance_terms_possible = 0;
    std::uint64_t search_refine_terms_evaluated = 0;
    std::uint64_t search_prefix_terms_evaluated = 0;
    std::uint64_t search_full_candidate_count = 0;
    std::uint64_t search_guard_check_count = 0;
    std::uint64_t search_guard_refresh_count = 0;
    std::uint64_t search_dense_baseline_candidate_count = 0;
    std::uint64_t easy_path_eligible_pixel_count = 0;
    std::uint64_t easy_path_accepted_pixel_count = 0;
    std::uint64_t easy_path_fallback_pixel_count = 0;
    std::uint64_t inactive_pixel_count = 0;
    std::uint64_t set_transport_unique_candidate_count = 0;
    std::uint64_t set_transport_nominal_candidate_count = 0;
    std::uint64_t set_transport_duplicate_candidate_count = 0;
    const std::size_t two_pass_prefix_terms = std::min(
        depth, static_cast<std::size_t>(search_prefix_size_));
    const ConservativePrefixSsdGuard cached_prefix_guard(
        two_pass_prefix_terms, depth);
    const bool capture_topk = diagnostic_top_k_ > 0 &&
        current_pyramid_level == diagnostic_pyramid_level_;
    if (capture_topk) {
        if (diagnostic_pixels_.back() >= pixel_count) {
            throw std::logic_error(
                "Top-K diagnostic pixel is outside the selected pyramid grid");
        }
        search_topk_diagnostics_.assign(
            diagnostic_pixels_.size() * diagnostic_top_k_,
            SearchTopKDiagnostic{});
    }

    #pragma omp parallel reduction(+:search_candidate_count,search_abandoned_candidate_count,search_distance_terms_evaluated,search_distance_terms_possible,search_refine_terms_evaluated,search_prefix_terms_evaluated,search_full_candidate_count,search_guard_check_count,search_guard_refresh_count,search_dense_baseline_candidate_count,easy_path_eligible_pixel_count,easy_path_accepted_pixel_count,easy_path_fallback_pixel_count,inactive_pixel_count,set_transport_unique_candidate_count,set_transport_nominal_candidate_count,set_transport_duplicate_candidate_count)
    {
        // Thread-local buffers — allocated once per thread
        std::vector<float> corr_data(ws2, 0.0f);
        std::vector<std::uint32_t> exact_epoch(ws2, 0);
        std::uint32_t current_exact_epoch = 0;
        ExactTopK top_k(static_cast<std::size_t>(search_top_k_));
        ExactTopK prefix_top_k(static_cast<std::size_t>(search_top_k_));
        ExactTopK exact_top_k(static_cast<std::size_t>(search_top_k_));
        const int set_coordinate_limit = cal_half_window_ + cal_half_window;
        const std::size_t set_coordinate_side = static_cast<std::size_t>(
            2 * set_coordinate_limit + 1);
        std::vector<float> set_corr(
            use_set_transport ? set_coordinate_side * set_coordinate_side : 0U,
            -std::numeric_limits<float>::infinity());
        std::vector<std::uint32_t> set_epoch(set_corr.size(), 0U);
        std::uint32_t current_set_epoch = 0;
        std::optional<SpatialPointCache> spatial_cache;
        if(use_spatial_reuse) spatial_cache.emplace();

        #if defined(WSVT_SEARCH_STATIC_SCHEDULE)
        #pragma omp for schedule(static)
        #else
        #pragma omp for schedule(guided, 64)
        #endif
        for (std::size_t pixel = 0; pixel < img_h * img_w; ++pixel) {
            const std::size_t yy = pixel / img_w;
            const std::size_t xx = pixel % img_w;
            if(spatial_cache) spatial_cache->select_tile(
                yy*((img_w+spatial_reuse_tile_width_-1)/spatial_reuse_tile_width_)+
                xx/spatial_reuse_tile_width_);

            search_dense_baseline_candidate_count +=
                static_cast<std::uint64_t>(ws2);
            if (!active_mask.empty() && active_mask[pixel] == 0) {
                disp_y[pixel] = dy_in[pixel];
                disp_x[pixel] = dx_in[pixel];
                if (capture_integer_proposal) {
                    integer_displace_y[pixel] = static_cast<int>(
                        round_half_to_even(static_cast<double>(dy_in[pixel])));
                    integer_displace_x[pixel] = static_cast<int>(
                        round_half_to_even(static_cast<double>(dx_in[pixel])));
                }
                effective_search_half_window[pixel] = 0.0f;
                confidence_path[pixel] = 0.0f;
                ++inactive_pixel_count;
                continue;
            }

            const float* img_line = img_ptr + pixel * depth;
            std::fill(
                corr_data.begin(), corr_data.end(),
                -std::numeric_limits<float>::infinity());
            top_k.reset();
            prefix_top_k.reset();
            exact_top_k.reset();
            ++current_exact_epoch;
            if (current_exact_epoch == 0) {
                std::fill(exact_epoch.begin(), exact_epoch.end(), 0);
                current_exact_epoch = 1;
            }

            const int dy_int = static_cast<int>(dy_in[pixel]);
            const int dx_int = static_cast<int>(dx_in[pixel]);
            int fit_center_y = dy_int;
            int fit_center_x = dx_int;
            const std::size_t y0n = static_cast<std::size_t>(
                static_cast<long long>(npad + yy) + static_cast<long long>(dy_int));
            const std::size_t x0n = static_cast<std::size_t>(
                static_cast<long long>(npad + xx) + static_cast<long long>(dx_int));

            float corr_max = -std::numeric_limits<float>::infinity();
            float corr_second = -std::numeric_limits<float>::infinity();
            std::size_t max_idx = 0;
            int effective_half_window = cal_half_window;
            float path_code = 1.0f;
            std::size_t diagnostic_request = diagnostic_pixels_.size();
            if (capture_topk) {
                const auto found = std::lower_bound(
                    diagnostic_pixels_.begin(), diagnostic_pixels_.end(), pixel);
                if (found != diagnostic_pixels_.end() && *found == pixel) {
                    diagnostic_request = static_cast<std::size_t>(
                        found - diagnostic_pixels_.begin());
                }
            }

            if (use_spatial_cost) {
                // The original absolute candidate locations and row-major tie
                // order are unchanged. Only the finest-level objective differs.
                for (std::size_t wy=0; wy<window_size; ++wy) {
                    for (std::size_t wx=0; wx<window_size; ++wx) {
                        const std::size_t ci=wy*window_size+wx;
                        const auto misses_before=spatial_cache ? spatial_cache->stats().misses : 0;
                        const float value=spatial_cache ? -spatial_cost.evaluate(
                            *spatial_sample,*spatial_ref,yy,xx,y0n+wy,x0n+wx,*spatial_cache) : -spatial_cost(
                            *spatial_sample, *spatial_ref,
                            static_cast<std::int64_t>(yy),
                            static_cast<std::int64_t>(xx),
                            static_cast<std::int64_t>(y0n+wy),
                            static_cast<std::int64_t>(x0n+wx));
                        corr_data[ci]=value;
                        ++search_candidate_count;
                        ++search_full_candidate_count;
                        search_distance_terms_possible += 9U*depth;
                        search_distance_terms_evaluated += spatial_cache ?
                            (spatial_cache->stats().misses-misses_before)*depth : 9U*depth;
                        if (value > corr_max) {
                            corr_second=corr_max; corr_max=value; max_idx=ci;
                        } else if (value > corr_second) {
                            corr_second=value;
                        }
                    }
                }
            } else if (use_set_transport) {
                ++current_set_epoch;
                if (current_set_epoch == 0) {
                    std::fill(set_epoch.begin(), set_epoch.end(), 0U);
                    current_set_epoch = 1;
                }
                std::size_t best_key = set_corr.size();
                std::size_t second_key = set_corr.size();
                std::uint64_t unique_candidates = 0;
                const std::uint64_t nominal_candidates =
                    static_cast<std::uint64_t>(kSetWidth * ws2);
                const std::size_t hypothesis_base = pixel * kSetWidth;
                const std::size_t pad_origin = static_cast<std::size_t>(
                    n_pad + cal_half_window);
                for (std::size_t hypothesis = 0; hypothesis < kSetWidth;
                     ++hypothesis) {
                    const int center_y = set_hypothesis_y[
                        hypothesis_base + hypothesis];
                    const int center_x = set_hypothesis_x[
                        hypothesis_base + hypothesis];
                    for (int local_y = -cal_half_window;
                         local_y <= cal_half_window; ++local_y) {
                        for (int local_x = -cal_half_window;
                             local_x <= cal_half_window; ++local_x) {
                            const int candidate_y = center_y + local_y;
                            const int candidate_x = center_x + local_x;
                            if (std::abs(candidate_y) > set_coordinate_limit ||
                                std::abs(candidate_x) > set_coordinate_limit) {
                                throw std::logic_error(
                                    "fixed set candidate exceeds padded coordinate range");
                            }
                            const std::size_t key = static_cast<std::size_t>(
                                candidate_y + set_coordinate_limit) *
                                set_coordinate_side + static_cast<std::size_t>(
                                candidate_x + set_coordinate_limit);
                            if (set_epoch[key] == current_set_epoch) {
                                continue;
                            }
                            const std::size_t ref_y = static_cast<std::size_t>(
                                static_cast<long long>(pad_origin + yy) +
                                static_cast<long long>(candidate_y));
                            const std::size_t ref_x = static_cast<std::size_t>(
                                static_cast<long long>(pad_origin + xx) +
                                static_cast<long long>(candidate_x));
                            const float* ref_row = ref_ptr +
                                (ref_y * ref_w + ref_x) * depth;
                            const float value = -squared_distance_full_simd(
                                img_line, ref_row, depth);
                            set_corr[key] = value;
                            set_epoch[key] = current_set_epoch;
                            ++unique_candidates;
                            ++search_candidate_count;
                            ++search_full_candidate_count;
                            search_distance_terms_possible +=
                                static_cast<std::uint64_t>(depth);
                            search_distance_terms_evaluated +=
                                static_cast<std::uint64_t>(depth);
                            const auto better_key = [&](std::size_t lhs,
                                                        std::size_t rhs) {
                                return rhs == set_corr.size() ||
                                    set_corr[lhs] > set_corr[rhs] ||
                                    (set_corr[lhs] == set_corr[rhs] && lhs < rhs);
                            };
                            if (better_key(key, best_key)) {
                                second_key = best_key;
                                best_key = key;
                            } else if (key != best_key &&
                                       better_key(key, second_key)) {
                                second_key = key;
                            }
                        }
                    }
                }
                const auto better_set_key = [&](std::size_t lhs,
                                                std::size_t rhs) {
                    return rhs == set_corr.size() ||
                        set_corr[lhs] > set_corr[rhs] ||
                        (set_corr[lhs] == set_corr[rhs] && lhs < rhs);
                };
                for (std::size_t hypothesis = 0; hypothesis < kSetWidth;
                     ++hypothesis) {
                    const int center_y = set_hypothesis_y[
                        hypothesis_base + hypothesis];
                    const int center_x = set_hypothesis_x[
                        hypothesis_base + hypothesis];
                    std::size_t representative_key = set_corr.size();
                    for (int local_y = -cal_half_window;
                         local_y <= cal_half_window; ++local_y) {
                        for (int local_x = -cal_half_window;
                             local_x <= cal_half_window; ++local_x) {
                            const int candidate_y = center_y + local_y;
                            const int candidate_x = center_x + local_x;
                            const std::size_t key = static_cast<std::size_t>(
                                candidate_y + set_coordinate_limit) *
                                set_coordinate_side + static_cast<std::size_t>(
                                candidate_x + set_coordinate_limit);
                            if (set_epoch[key] != current_set_epoch) {
                                throw std::logic_error(
                                    "SET4 representative domain was not evaluated");
                            }
                            if (better_set_key(key, representative_key)) {
                                representative_key = key;
                            }
                        }
                    }
                    const std::size_t output_index =
                        hypothesis_base + hypothesis;
                    set_transport_representative_y[output_index] =
                        static_cast<int>(representative_key / set_coordinate_side) -
                        set_coordinate_limit;
                    set_transport_representative_x[output_index] =
                        static_cast<int>(representative_key % set_coordinate_side) -
                        set_coordinate_limit;
                }
                if (best_key == set_corr.size() || second_key == set_corr.size()) {
                    throw std::logic_error(
                        "fixed set transport produced fewer than two candidates");
                }
                const int winner_y = static_cast<int>(
                    best_key / set_coordinate_side) - set_coordinate_limit;
                const int winner_x = static_cast<int>(
                    best_key % set_coordinate_side) - set_coordinate_limit;
                std::size_t owner = 0;
                for (; owner < kSetWidth; ++owner) {
                    const int center_y = set_hypothesis_y[hypothesis_base + owner];
                    const int center_x = set_hypothesis_x[hypothesis_base + owner];
                    if (std::abs(winner_y - center_y) <= cal_half_window &&
                        std::abs(winner_x - center_x) <= cal_half_window) {
                        fit_center_y = center_y;
                        fit_center_x = center_x;
                        break;
                    }
                }
                if (owner == kSetWidth) {
                    throw std::logic_error(
                        "fixed set winner has no contributing hypothesis domain");
                }
                for (int local_y = -cal_half_window;
                     local_y <= cal_half_window; ++local_y) {
                    for (int local_x = -cal_half_window;
                         local_x <= cal_half_window; ++local_x) {
                        const int candidate_y = fit_center_y + local_y;
                        const int candidate_x = fit_center_x + local_x;
                        const std::size_t key = static_cast<std::size_t>(
                            candidate_y + set_coordinate_limit) *
                            set_coordinate_side + static_cast<std::size_t>(
                            candidate_x + set_coordinate_limit);
                        if (set_epoch[key] != current_set_epoch) {
                            throw std::logic_error(
                                "fixed set winning domain was not fully evaluated");
                        }
                        const std::size_t local_index = static_cast<std::size_t>(
                            local_y + cal_half_window) * window_size +
                            static_cast<std::size_t>(local_x + cal_half_window);
                        corr_data[local_index] = set_corr[key];
                    }
                }
                max_idx = static_cast<std::size_t>(
                    winner_y - fit_center_y + cal_half_window) * window_size +
                    static_cast<std::size_t>(
                    winner_x - fit_center_x + cal_half_window);
                corr_max = set_corr[best_key];
                corr_second = set_corr[second_key];
                set_transport_candidates_evaluated[pixel] =
                    static_cast<float>(unique_candidates);
                set_transport_unique_candidate_count += unique_candidates;
                set_transport_nominal_candidate_count += nominal_candidates;
                set_transport_duplicate_candidate_count +=
                    nominal_candidates - unique_candidates;
            } else if (easy_to_hard_.confidence_aware_refinement) {
                const auto evaluate_candidate = [&](std::size_t wy, std::size_t wx) {
                    const std::size_t ci = wy * window_size + wx;
                    if (exact_epoch[ci] == current_exact_epoch) {
                        return;
                    }
                    const float* ref_row = ref_ptr +
                        ((y0n + wy) * ref_w + (x0n + wx)) * depth;
                    const float s = squared_distance_full_simd(
                        img_line, ref_row, depth);
                    ++search_candidate_count;
                    search_distance_terms_possible +=
                        static_cast<std::uint64_t>(depth);
                    search_distance_terms_evaluated +=
                        static_cast<std::uint64_t>(depth);
                    ++search_full_candidate_count;
                    const float val = -s;
                    corr_data[ci] = val;
                    exact_epoch[ci] = current_exact_epoch;
                    if (val > corr_max) {
                        corr_second = corr_max;
                        corr_max = val;
                        max_idx = ci;
                    } else if (val > corr_second) {
                        corr_second = val;
                    }
                };
                const auto evaluate_square = [&](int radius, bool outer_only) {
                    const int center = cal_half_window;
                    for (int local_y = -radius; local_y <= radius; ++local_y) {
                        for (int local_x = -radius; local_x <= radius; ++local_x) {
                            if (outer_only &&
                                std::abs(local_y) <= easy_to_hard_.easy_half_window &&
                                std::abs(local_x) <= easy_to_hard_.easy_half_window) {
                                continue;
                            }
                            evaluate_candidate(
                                static_cast<std::size_t>(center + local_y),
                                static_cast<std::size_t>(center + local_x));
                        }
                    }
                };
                const auto current_peak_is_confident = [&](int radius) {
                    const std::size_t max_y = max_idx / window_size;
                    const std::size_t max_x = max_idx % window_size;
                    const int local_y = static_cast<int>(max_y) - cal_half_window;
                    const int local_x = static_cast<int>(max_x) - cal_half_window;
                    const bool boundary = std::abs(local_y) >= radius ||
                        std::abs(local_x) >= radius;
                    if (boundary || max_y == 0 || max_x == 0 ||
                        max_y + 1 >= window_size || max_x + 1 >= window_size ||
                        !std::isfinite(corr_max) ||
                        !std::isfinite(corr_second)) {
                        return false;
                    }
                    const auto value = [&](long long y, long long x) {
                        return corr_data[static_cast<std::size_t>(y) * window_size +
                                         static_cast<std::size_t>(x)];
                    };
                    const auto my = static_cast<long long>(max_y);
                    const auto mx = static_cast<long long>(max_x);
                    const float c_m10 = value(my - 1, mx);
                    const float c_p10 = value(my + 1, mx);
                    const float c_0m1 = value(my, mx - 1);
                    const float c_0p1 = value(my, mx + 1);
                    const float c_00 = value(my, mx);
                    const float c_pp = value(my + 1, mx + 1);
                    const float c_pm = value(my + 1, mx - 1);
                    const float c_mp = value(my - 1, mx + 1);
                    const float c_mm = value(my - 1, mx - 1);
                    const float grad_y = (c_p10 - c_m10) / 2.0f;
                    const float dyy = c_p10 + c_m10 - 2.0f * c_00;
                    const float grad_x = (c_0p1 - c_0m1) / 2.0f;
                    const float dxx = c_0p1 + c_0m1 - 2.0f * c_00;
                    const float dxy = (c_pp - c_pm - c_mp + c_mm) / 4.0f;
                    const float denom = dxx * dyy - dxy * dxy;
                    if (!std::isfinite(denom) || denom <= 0.0f) {
                        return false;
                    }
                    const float inv_det = 1.0f / denom;
                    const float minor_x =
                        (-(dyy * grad_x - dxy * grad_y) * inv_det) * pixel_res_x;
                    const float minor_y =
                        (-(dxx * grad_y - dxy * grad_x) * inv_det) * pixel_res_y;
                    const float delta = std::hypot(
                        static_cast<float>(local_x) + minor_x,
                        static_cast<float>(local_y) + minor_y);
                    const float margin = (corr_max - corr_second) /
                        std::max(std::fabs(corr_max),
                                 std::numeric_limits<float>::epsilon());
                    const float curvature = std::sqrt(denom) /
                        std::max(std::fabs(corr_max),
                                 std::numeric_limits<float>::epsilon());
                    const float contrast = temporal_contrast.empty()
                        ? std::numeric_limits<float>::infinity()
                        : temporal_contrast[pixel];
                    return margin >= easy_to_hard_.score_margin_min &&
                        curvature >= easy_to_hard_.normalized_curvature_min &&
                        contrast >= easy_to_hard_.temporal_speckle_contrast_min &&
                        delta <= easy_to_hard_.interlevel_delta_max_px;
                };

                const bool request_easy = !easy_mask.empty() &&
                    easy_mask[pixel] != 0 &&
                    easy_to_hard_.easy_half_window < cal_half_window;
                if (request_easy) {
                    ++easy_path_eligible_pixel_count;
                    effective_half_window = easy_to_hard_.easy_half_window;
                    evaluate_square(effective_half_window, false);
                    bool easy_confident =
                        current_peak_is_confident(effective_half_window);
                    if (easy_confident) {
                        // A 3x3 local peak can still be a confident-looking
                        // alias while a much better basin exists farther away.
                        // Probe a deterministic stride-2 sentinel grid across
                        // the full fine-level window before accepting it.
                        for (int local_y = -cal_half_window;
                             local_y <= cal_half_window; local_y += 2) {
                            for (int local_x = -cal_half_window;
                                 local_x <= cal_half_window; local_x += 2) {
                                evaluate_candidate(
                                    static_cast<std::size_t>(
                                        cal_half_window + local_y),
                                    static_cast<std::size_t>(
                                        cal_half_window + local_x));
                            }
                        }
                        easy_confident =
                            current_peak_is_confident(effective_half_window);
                    }
                    if (easy_confident) {
                        path_code = 2.0f;
                        ++easy_path_accepted_pixel_count;
                    } else {
                        evaluate_square(cal_half_window, true);
                        effective_half_window = cal_half_window;
                        path_code = 3.0f;
                        ++easy_path_fallback_pixel_count;
                    }
                } else {
                    evaluate_square(cal_half_window, false);
                }
            } else if (search_two_pass_) {
                const std::size_t prefix_terms = two_pass_prefix_terms;

                // Stage A: one contiguous SIMD prefix for every candidate.  No
                // block loop or threshold branch occurs inside this pass.
                for (std::size_t ci = 0; ci < ws2; ++ci) {
                    const std::size_t wy = ci / window_size;
                    const std::size_t wx = ci % window_size;
                    const float* ref_row =
                        ref_ptr + ((y0n + wy) * ref_w + (x0n + wx)) * depth;
                    const float prefix_distance = squared_distance_full_simd(
                        img_line, ref_row, prefix_terms);
                    ++search_candidate_count;
                    search_distance_terms_possible += static_cast<std::uint64_t>(depth);
                    search_distance_terms_evaluated +=
                        static_cast<std::uint64_t>(prefix_terms);
                    search_prefix_terms_evaluated +=
                        static_cast<std::uint64_t>(prefix_terms);
                    corr_data[ci] = prefix_distance;
                    prefix_top_k.consider(prefix_distance, ci);
                }

                if (prefix_terms == depth) {
                    // The prefix already is the baseline full-SIMD reduction.
                    for (std::size_t ci = 0; ci < ws2; ++ci) {
                        const float exact_distance = corr_data[ci];
                        corr_data[ci] = -exact_distance;
                        exact_epoch[ci] = current_exact_epoch;
                        exact_top_k.consider(exact_distance, ci);
                        ++search_full_candidate_count;
                    }
                } else {
                    // Evaluate the K best prefix seeds completely to establish
                    // a valid upper bound before screening other candidates.
                    for (std::size_t rank = 0; rank < prefix_top_k.size(); ++rank) {
                        const std::size_t ci = prefix_top_k[rank].candidate_index;
                        const std::size_t wy = ci / window_size;
                        const std::size_t wx = ci % window_size;
                        const float* ref_row =
                            ref_ptr + ((y0n + wy) * ref_w + (x0n + wx)) * depth;
                        const float exact_distance = squared_distance_full_simd(
                            img_line, ref_row, depth);
                        corr_data[ci] = -exact_distance;
                        exact_epoch[ci] = current_exact_epoch;
                        exact_top_k.consider(exact_distance, ci);
                        search_distance_terms_evaluated +=
                            static_cast<std::uint64_t>(depth);
                        ++search_full_candidate_count;
                    }

                    double guarded_threshold = std::numeric_limits<double>::infinity();
                    if (search_guard_cache_) {
                        guarded_threshold = cached_prefix_guard.guarded_threshold(
                            exact_top_k.abandon_threshold());
                        ++search_guard_refresh_count;
                    }

                    // Stage B: one conservative bound check per non-seed
                    // candidate, followed by the unchanged full-SIMD SSD only
                    // for survivors.  The threshold may tighten as exact
                    // survivors enter Top-K; strict ties always survive.
                    for (std::size_t ci = 0; ci < ws2; ++ci) {
                        if (exact_epoch[ci] == current_exact_epoch) {
                            continue;
                        }
                        const float prefix_distance = corr_data[ci];
                        ++search_guard_check_count;
                        const bool prune = search_guard_cache_
                            ? cached_prefix_guard.proves_above(
                                prefix_distance, guarded_threshold)
                            : prefix_proves_full_ssd_above(
                                prefix_distance,
                                exact_top_k.abandon_threshold(),
                                prefix_terms,
                                depth);
                        if (prune) {
                            corr_data[ci] = -prefix_distance;
                            ++search_abandoned_candidate_count;
                            continue;
                        }
                        const std::size_t wy = ci / window_size;
                        const std::size_t wx = ci % window_size;
                        const float* ref_row =
                            ref_ptr + ((y0n + wy) * ref_w + (x0n + wx)) * depth;
                        const float exact_distance = squared_distance_full_simd(
                            img_line, ref_row, depth);
                        corr_data[ci] = -exact_distance;
                        exact_epoch[ci] = current_exact_epoch;
                        const bool top_k_changed =
                            exact_top_k.consider(exact_distance, ci);
                        if (search_guard_cache_ && top_k_changed) {
                            guarded_threshold = cached_prefix_guard.guarded_threshold(
                                exact_top_k.abandon_threshold());
                            ++search_guard_refresh_count;
                        }
                        search_distance_terms_evaluated +=
                            static_cast<std::uint64_t>(depth);
                        ++search_full_candidate_count;
                    }
                }
            } else {
                for (std::size_t wy = 0; wy < window_size; ++wy) {
                    for (std::size_t wx = 0; wx < window_size; ++wx) {
                        const float* ref_row = ref_ptr +
                            ((y0n + wy) * ref_w + (x0n + wx)) * depth;
                        const std::size_t ci = wy * window_size + wx;
                        ++search_candidate_count;
                        search_distance_terms_possible +=
                            static_cast<std::uint64_t>(depth);

                        if (search_early_abandon_) {
                            const auto result = squared_distance_blockwise(
                                img_line,
                                ref_row,
                                depth,
                                top_k.abandon_threshold(),
                                static_cast<std::size_t>(search_block_size_));
                            search_distance_terms_evaluated +=
                                static_cast<std::uint64_t>(result.terms_evaluated);
                            corr_data[ci] = -result.distance;
                            if (result.abandoned) {
                                ++search_abandoned_candidate_count;
                            } else {
                                top_k.consider(result.distance, ci);
                                ++search_full_candidate_count;
                            }
                        } else {
                            const float s = squared_distance_full_simd(
                                img_line, ref_row, depth);
                            search_distance_terms_evaluated +=
                                static_cast<std::uint64_t>(depth);
                            ++search_full_candidate_count;
                            const float val = -s;
                            corr_data[ci] = val;
                            if (val > corr_max) {
                                corr_second = corr_max;
                                corr_max = val;
                                max_idx = ci;
                            } else if (val > corr_second) {
                                corr_second = val;
                            }
                        }
                    }
                }

                if (search_early_abandon_) {
                    for (std::size_t rank = 0; rank < top_k.size(); ++rank) {
                        const std::size_t ci = top_k[rank].candidate_index;
                        const std::size_t wy = ci / window_size;
                        const std::size_t wx = ci % window_size;
                        const float* ref_row =
                            ref_ptr + ((y0n + wy) * ref_w + (x0n + wx)) * depth;
                        const float exact_distance = squared_distance_full_simd(
                            img_line, ref_row, depth);
                        corr_data[ci] = -exact_distance;
                        exact_epoch[ci] = current_exact_epoch;
                        exact_top_k.consider(exact_distance, ci);
                        search_refine_terms_evaluated +=
                            static_cast<std::uint64_t>(depth);
                    }
                }
            }

            if (search_early_abandon_ || search_two_pass_) {
                max_idx = exact_top_k[0].candidate_index;
                corr_max = -exact_top_k[0].distance;
                corr_second = -exact_top_k[1].distance;

                // Pruned candidates hold only a lower-bound distance. Recompute
                // the 3x3 neighbourhood around the exact winner so the existing
                // 2D Hessian subpixel fit always sees complete SSD values.
                const std::size_t max_y = max_idx / window_size;
                const std::size_t max_x = max_idx % window_size;
                const std::size_t y_begin = max_y == 0 ? 0 : max_y - 1;
                const std::size_t x_begin = max_x == 0 ? 0 : max_x - 1;
                const std::size_t y_end = std::min(window_size - 1, max_y + 1);
                const std::size_t x_end = std::min(window_size - 1, max_x + 1);
                for (std::size_t wy = y_begin; wy <= y_end; ++wy) {
                    for (std::size_t wx = x_begin; wx <= x_end; ++wx) {
                        const std::size_t ci = wy * window_size + wx;
                        if (exact_epoch[ci] == current_exact_epoch) {
                            continue;
                        }
                        const float* ref_row =
                            ref_ptr + ((y0n + wy) * ref_w + (x0n + wx)) * depth;
                        corr_data[ci] =
                            -squared_distance_full_simd(img_line, ref_row, depth);
                        exact_epoch[ci] = current_exact_epoch;
                        search_refine_terms_evaluated += static_cast<std::uint64_t>(depth);
                    }
                }
            }

            if (diagnostic_request < diagnostic_pixels_.size()) {
                std::vector<std::size_t> order(ws2);
                std::iota(order.begin(), order.end(), std::size_t{0});
                const auto better = [&](std::size_t lhs, std::size_t rhs) {
                    if (corr_data[lhs] != corr_data[rhs]) {
                        return corr_data[lhs] > corr_data[rhs];
                    }
                    return lhs < rhs;
                };
                std::partial_sort(
                    order.begin(),
                    order.begin() + static_cast<std::ptrdiff_t>(diagnostic_top_k_),
                    order.end(),
                    better);
                for (std::size_t rank = 0; rank < diagnostic_top_k_; ++rank) {
                    const std::size_t candidate = order[rank];
                    const int local_y = static_cast<int>(candidate / window_size) -
                        cal_half_window;
                    const int local_x = static_cast<int>(candidate % window_size) -
                        cal_half_window;
                    SearchTopKDiagnostic diagnostic;
                    diagnostic.request_index = diagnostic_request;
                    diagnostic.pyramid_level = current_pyramid_level;
                    diagnostic.raw_y = yy;
                    diagnostic.raw_x = xx;
                    diagnostic.rank = rank;
                    diagnostic.initial_guess_y = dy_int;
                    diagnostic.initial_guess_x = dx_int;
                    diagnostic.local_offset_y = local_y;
                    diagnostic.local_offset_x = local_x;
                    diagnostic.internal_candidate_y = dy_int + local_y;
                    diagnostic.internal_candidate_x = dx_int + local_x;
                    diagnostic.saved_candidate_y = -diagnostic.internal_candidate_y;
                    diagnostic.saved_candidate_x = -diagnostic.internal_candidate_x;
                    diagnostic.descriptor_score_neg_ssd = corr_data[candidate];
                    diagnostic.descriptor_cost_ssd = -corr_data[candidate];
                    search_topk_diagnostics_[
                        diagnostic_request * diagnostic_top_k_ + rank] = diagnostic;
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
            if(use_guarded_subpixel) {
                const long long support_radius=use_spatial_cost ? 1 : 0;
                const long long reference_y=static_cast<long long>(yy)+fit_center_y+
                    static_cast<long long>(max_y_idx)-cal_half_window;
                const long long reference_x=static_cast<long long>(xx)+fit_center_x+
                    static_cast<long long>(max_x_idx)-cal_half_window;
                const bool complete = max_y_idx>0 && max_x_idx>0 &&
                    max_y_idx+1<window_size && max_x_idx+1<window_size &&
                    static_cast<long long>(yy)>=support_radius &&
                    static_cast<long long>(xx)>=support_radius &&
                    yy+support_radius<img_h && xx+support_radius<img_w &&
                    reference_y-1-support_radius>=0 && reference_x-1-support_radius>=0 &&
                    reference_y+1+support_radius<static_cast<long long>(img_h) &&
                    reference_x+1+support_radius<static_cast<long long>(img_w);
                const std::array<float,9> costs{
                    -c_mm,-c_m10,-c_mp,-c_0m1,-c_00,-c_0p1,-c_pm,-c_p10,-c_pp};
                const auto fit=guarded_subpixel(costs,complete,depth,constrained_subpixel_);
                guarded_subpixel_reasons_[pixel]=static_cast<unsigned char>(fit.reason);
                guarded_subpixel_offsets_[2*pixel]=fit.dx;
                guarded_subpixel_offsets_[2*pixel+1]=fit.dy;
                result_disp_x=xx_axis[max_idx]+static_cast<float>(fit.dx);
                result_disp_y=yy_axis[max_idx]+static_cast<float>(fit.dy);
            }

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

            disp_y[pixel] = result_disp_y + static_cast<float>(fit_center_y);
            disp_x[pixel] = result_disp_x + static_cast<float>(fit_center_x);
            const int winner_local_y =
                static_cast<int>(max_y_idx) - cal_half_window;
            const int winner_local_x =
                static_cast<int>(max_x_idx) - cal_half_window;
            if (capture_integer_proposal) {
                integer_displace_y[pixel] = fit_center_y + winner_local_y;
                integer_displace_x[pixel] = fit_center_x + winner_local_x;
            }
            if (capture_coarse_set) {
                std::vector<int> roots(ws2, -1);
                std::vector<std::size_t> path;
                path.reserve(ws2);
                const auto better_surface = [&](std::size_t lhs,
                                                std::size_t rhs) {
                    return corr_data[lhs] > corr_data[rhs] ||
                        (corr_data[lhs] == corr_data[rhs] && lhs < rhs);
                };
                for (std::size_t start = 0; start < ws2; ++start) {
                    if (roots[start] >= 0) continue;
                    path.clear();
                    std::size_t current = start;
                    while (roots[current] < 0) {
                        path.push_back(current);
                        const std::size_t cy = current / window_size;
                        const std::size_t cx = current % window_size;
                        std::size_t best = current;
                        for (int oy = -1; oy <= 1; ++oy) {
                            for (int ox = -1; ox <= 1; ++ox) {
                                const long long ny = static_cast<long long>(cy) + oy;
                                const long long nx = static_cast<long long>(cx) + ox;
                                if (ny < 0 || nx < 0 ||
                                    ny >= static_cast<long long>(window_size) ||
                                    nx >= static_cast<long long>(window_size)) {
                                    continue;
                                }
                                const std::size_t neighbour =
                                    static_cast<std::size_t>(ny) * window_size +
                                    static_cast<std::size_t>(nx);
                                if (better_surface(neighbour, best)) best = neighbour;
                            }
                        }
                        if (best == current) {
                            roots[current] = static_cast<int>(current);
                            break;
                        }
                        current = best;
                    }
                    const int root = roots[current];
                    for (auto it = path.rbegin(); it != path.rend(); ++it) {
                        roots[*it] = root;
                    }
                }
                std::vector<std::size_t> ordered_roots;
                ordered_roots.reserve(ws2);
                for (const int root : roots) {
                    const std::size_t value = static_cast<std::size_t>(root);
                    if (std::find(ordered_roots.begin(), ordered_roots.end(),
                                  value) == ordered_roots.end()) {
                        ordered_roots.push_back(value);
                    }
                }
                std::sort(ordered_roots.begin(), ordered_roots.end(),
                          better_surface);
                const std::size_t output_base = pixel * kSetWidth;
                coarse_set_hypothesis_y[output_base] = disp_y[pixel];
                coarse_set_hypothesis_x[output_base] = disp_x[pixel];
                for (std::size_t hypothesis = 1; hypothesis < kSetWidth;
                     ++hypothesis) {
                    if (hypothesis < ordered_roots.size()) {
                        const std::size_t root = ordered_roots[hypothesis];
                        coarse_set_hypothesis_y[output_base + hypothesis] =
                            static_cast<float>(dy_int +
                            static_cast<int>(root / window_size) -
                            cal_half_window);
                        coarse_set_hypothesis_x[output_base + hypothesis] =
                            static_cast<float>(dx_int +
                            static_cast<int>(root % window_size) -
                            cal_half_window);
                    } else {
                        coarse_set_hypothesis_y[output_base + hypothesis] =
                            disp_y[pixel];
                        coarse_set_hypothesis_x[output_base + hypothesis] =
                            disp_x[pixel];
                    }
                }
            }
            const bool boundary_hit =
                std::abs(winner_local_y) == effective_half_window ||
                std::abs(winner_local_x) == effective_half_window;
            const bool complete_spatial_domain = !use_spatial_cost ||
                (yy >= 1 && xx >= 1 && yy+1 < img_h && xx+1 < img_w &&
                 static_cast<long long>(yy)+dy_int-cal_half_window >= 1 &&
                 static_cast<long long>(xx)+dx_int-cal_half_window >= 1 &&
                 static_cast<long long>(yy)+dy_int+cal_half_window+1 <
                     static_cast<long long>(img_h) &&
                 static_cast<long long>(xx)+dx_int+cal_half_window+1 <
                     static_cast<long long>(img_w));
            const bool geometry_valid = complete_spatial_domain &&
                (!use_guarded_subpixel || guarded_subpixel_reasons_[pixel]==0 ||
                 guarded_subpixel_reasons_[pixel]==6) &&
                !boundary_hit && std::isfinite(denom) &&
                denom > 0.0f && std::isfinite(corr_max) && std::isfinite(corr_second);
            const float normalized_curvature = geometry_valid
                ? std::sqrt(denom) /
                    std::max(std::fabs(corr_max),
                             std::numeric_limits<float>::epsilon())
                : 0.0f;
            const float margin = (corr_max - corr_second) /
                std::max(std::fabs(corr_max),
                         std::numeric_limits<float>::epsilon());
            const float contrast = temporal_contrast.empty()
                ? std::numeric_limits<float>::infinity()
                : temporal_contrast[pixel];
            const float interlevel_delta = std::hypot(result_disp_x, result_disp_y);
            const bool confident = geometry_valid && std::isfinite(margin) &&
                margin >= easy_to_hard_.score_margin_min &&
                normalized_curvature >= easy_to_hard_.normalized_curvature_min &&
                contrast >= easy_to_hard_.temporal_speckle_contrast_min &&
                interlevel_delta <= easy_to_hard_.interlevel_delta_max_px;
            best_score_neg_ssd[pixel] = corr_max;
            second_best_score_neg_ssd[pixel] = corr_second;
            score_margin[pixel] = margin;
            peak_hessian_det[pixel] = denom;
            normalized_peak_curvature[pixel] = normalized_curvature;
            interlevel_displacement_delta[pixel] = interlevel_delta;
            search_boundary_hit[pixel] = boundary_hit ? 1.0f : 0.0f;
            search_geometry_valid[pixel] = geometry_valid ? 1.0f : 0.0f;
            confidence_accept[pixel] = confident ? 1.0f : 0.0f;
            effective_search_half_window[pixel] =
                static_cast<float>(effective_half_window);
            if (!confident && path_code != 0.0f) {
                path_code = 4.0f;
            }
            confidence_path[pixel] = path_code;
        }
        if(spatial_cache) {
            const auto s=spatial_cache->stats();
            #pragma omp critical(wsvt_spatial_reuse_stats)
            {
                spatial_reuse_stats_.hits+=s.hits;
                spatial_reuse_stats_.misses+=s.misses;
                spatial_reuse_stats_.resets+=s.resets;
                spatial_reuse_stats_.overflows+=s.overflows;
                spatial_reuse_stats_.allocated_bytes+=s.allocated_bytes;
                spatial_reuse_stats_.peak_entries=std::max(spatial_reuse_stats_.peak_entries,s.peak_entries);
            }
        }
    }
    return DisplaceWaveletOutput{
        std::move(disp_y),
        std::move(disp_x),
        std::move(best_score_neg_ssd),
        std::move(second_best_score_neg_ssd),
        std::move(score_margin),
        std::move(peak_hessian_det),
        std::move(normalized_peak_curvature),
        std::move(interlevel_displacement_delta),
        std::move(search_boundary_hit),
        std::move(search_geometry_valid),
        std::move(confidence_accept),
        std::move(effective_search_half_window),
        std::move(confidence_path),
        std::move(integer_displace_y),
        std::move(integer_displace_x),
        search_candidate_count,
        search_abandoned_candidate_count,
        search_distance_terms_evaluated,
        search_distance_terms_possible,
        search_refine_terms_evaluated,
        search_prefix_terms_evaluated,
        search_full_candidate_count,
        search_guard_check_count,
        search_guard_refresh_count,
        search_dense_baseline_candidate_count,
        easy_path_eligible_pixel_count,
        easy_path_accepted_pixel_count,
        easy_path_fallback_pixel_count,
        inactive_pixel_count,
        std::move(coarse_set_hypothesis_y),
        std::move(coarse_set_hypothesis_x),
        std::move(set_transport_representative_y),
        std::move(set_transport_representative_x),
        std::move(set_transport_candidates_evaluated),
        set_transport_unique_candidate_count,
        set_transport_nominal_candidate_count,
        set_transport_duplicate_candidate_count};
}


SolverOutput WSVT::solver() {
    const auto processing_t0 = std::chrono::steady_clock::now();
    // Revalidate at use time: setter order must not bypass profile exclusions.
    configure_fixed_spatial_support(fixed_spatial_support_);
    configure_fixed_spatial_reuse(fixed_spatial_reuse_,spatial_reuse_tile_width_);
    if(constrained_subpixel_) configure_constrained_subpixel(true);
    else configure_guarded_subpixel(guarded_subpixel_);
    if (fixed_spatial_support_ != FixedSpatialSupport::Point || guarded_subpixel_) {
        for (float v : img_data_) if (!finite_descriptor_value(v))
            throw std::invalid_argument("nonfinite fixed-support sample input");
        for (float v : ref_data_) if (!finite_descriptor_value(v))
            throw std::invalid_argument("nonfinite fixed-support reference input");
    }
    const int solver_threads = configure_openmp_threads(n_cores_, "pyramid/wavelet/displace", 2);
    crop_inputs_if_requested();
    std::vector<float> umpa_sample_raw;
    std::vector<float> umpa_reference_raw;
    std::uint64_t umpa_retained_raw_bytes = 0;
    const bool raw_rerank_enabled = wavelet_guided_umpa_.enabled ||
        set_transport_raw_rerank_.enabled;
    if (raw_rerank_enabled) {
        umpa_sample_raw = img_data_;
        umpa_reference_raw = ref_data_;
        umpa_retained_raw_bytes = static_cast<std::uint64_t>(
            (umpa_sample_raw.size() + umpa_reference_raw.size()) * sizeof(float));
    }
    std::size_t out_h = h_;
    std::size_t out_w = w_;
    std::size_t out_d = ch_;
    std::vector<float> darkfield;
    double darkfield_time_s = 0.0;

    if (calc_darkfield_) {
        const auto darkfield_t0 = std::chrono::steady_clock::now();
        if (n_template_ == 0) {
            darkfield.assign(out_h * out_w, 0.0f);
            auto std_img = std_depth_chw_per_pixel(img_data_, ch_, h_, w_).take();
            auto std_ref = std_depth_chw_per_pixel(ref_data_, ch_, h_, w_).take();
            #pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < darkfield.size(); ++i) {
                darkfield[i] = std_img[i] / (std_ref[i] + 1e-6f);
            }
        } else {
            auto img_stack = stack_TemplateWindow(img_data_, ch_, h_, w_, out_h, out_w, out_d);
            auto ref_stack = stack_TemplateWindow(ref_data_, ch_, h_, w_, out_h, out_w, out_d);
            darkfield.assign(out_h * out_w, 0.0f);
            auto std_img = std_depth_hwd(
                TensorView3D<const float, Layout::HWD>{img_stack.data(), {out_h, out_w, out_d}}).take();
            auto std_ref = std_depth_hwd(
                TensorView3D<const float, Layout::HWD>{ref_stack.data(), {out_h, out_w, out_d}}).take();
            #pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < darkfield.size(); ++i) {
                darkfield[i] = std_img[i] / (std_ref[i] + 1e-6f);
            }
        }
        const auto darkfield_t1 = std::chrono::steady_clock::now();
        darkfield_time_s = std::chrono::duration<double>(darkfield_t1 - darkfield_t0).count();
        prColor("raw darkfield time: " + std::to_string(darkfield_time_s) + " s", "light_purple");
    } else {
        prColor("raw darkfield time: 0.000000 s (disabled)", "light_purple");
    }

    // Transmission follows the Python WSVT reference: first frame sample/ref with clipping.
    std::vector<float> transmission(out_h * out_w, 0.0f);
    #pragma omp parallel for schedule(static)
    for (std::size_t idx = 0; idx < out_h * out_w; ++idx) {
        const float den = std::max(ref_data_[idx], 1.0e-10f);
        const float ratio = img_data_[idx] / den;
        transmission[idx] = std::clamp(ratio, 0.01f, 10.0f);
    }

    // B0 builds one descriptor pyramid. B2 independently constructs a valid
    // normalized/DWT descriptor for each nested frame prefix; coefficient
    // truncation would not be mathematically equivalent. B2 v2 may materialize
    // later-stage descriptors only at active sample/candidate-reference pixels.
    std::vector<std::size_t> temporal_stages = easy_to_hard_.adaptive_frames
        ? easy_to_hard_.frame_stages
        : std::vector<std::size_t>{ch_};
    PrefixCompatibleWaveletBasis prefix_wavelet_basis;
    PrefixCompatibleWaveletState prefix_img_state;
    PrefixCompatibleWaveletState prefix_ref_state;
    if (easy_to_hard_.prefix_compatible_temporal_wavelet) {
        wavelet_level_ = dwt_max_level_db2(ch_);
        wavelet_add_list_ = wavelet_add_list_for_depth(ch_);
        const int add = wavelet_add_list_.empty() ? 2 : wavelet_add_list_.front();
        const int return_level = wavelet_level_ + 1 - wavelet_level_cut_ + add;
        prefix_wavelet_basis = make_prefix_compatible_wavelet_basis(
            ch_, WaveletFamily::Db2, wavelet_level_, return_level);
        prefix_img_state = make_prefix_compatible_wavelet_state(
            h_, w_, prefix_wavelet_basis);
        prefix_ref_state = make_prefix_compatible_wavelet_state(
            h_, w_, prefix_wavelet_basis);
    }
    const auto searching_window_pyramid_list = derived_search_half_windows(
        pyramid_level_, cal_half_window_, n_s_extend_);
    PyramidResult p;
    std::size_t m0 = 0;
    std::size_t n0 = 0;
    std::vector<float> merged_displace_y;
    std::vector<float> merged_displace_x;
    std::vector<float> base_displace_y;
    std::vector<float> base_displace_x;
    std::vector<int> merged_integer_proposal_y;
    std::vector<int> merged_integer_proposal_x;
    std::vector<float> merged_set_transport_center_y;
    std::vector<float> merged_set_transport_center_x;
    std::vector<int> merged_set_transport_representative_y;
    std::vector<int> merged_set_transport_representative_x;
    std::vector<float> merged_set_transport_candidates_evaluated;
    std::vector<float> darkfield_nd;
    std::vector<float> second_best_score_neg_ssd;
    std::vector<float> score_margin;
    std::vector<float> peak_hessian_det;
    std::vector<float> normalized_peak_curvature;
    std::vector<float> interlevel_displacement_delta;
    std::vector<float> search_boundary_hit;
    std::vector<float> search_geometry_valid;
    std::vector<float> temporal_speckle_contrast;
    std::vector<float> effective_search_half_window;
    std::vector<float> temporal_frames_used;
    std::vector<float> confidence_path;
    std::vector<std::uint8_t> active_full;
    std::vector<float> previous_temporal_y;
    std::vector<float> previous_temporal_x;
    double pyramid_time = 0.0;
    double wavelet_time = 0.0;
    double total_template_window_time = 0.0;
    double total_displace_upsample_s = 0.0;
    double total_displace_pad_s = 0.0;
    double total_displace_search_s = 0.0;
    std::uint64_t total_search_candidate_count = 0;
    std::uint64_t total_search_abandoned_candidate_count = 0;
    std::uint64_t total_search_distance_terms_evaluated = 0;
    std::uint64_t total_search_distance_terms_possible = 0;
    std::uint64_t total_search_refine_terms_evaluated = 0;
    std::uint64_t total_search_prefix_terms_evaluated = 0;
    std::uint64_t total_search_full_candidate_count = 0;
    std::uint64_t total_search_guard_check_count = 0;
    std::uint64_t total_search_guard_refresh_count = 0;
    std::uint64_t dense_baseline_candidate_count = 0;
    std::uint64_t total_easy_path_eligible_pixel_count = 0;
    std::uint64_t total_easy_path_accepted_pixel_count = 0;
    std::uint64_t total_easy_path_fallback_pixel_count = 0;
    std::uint64_t total_inactive_pixel_count = 0;
    std::uint64_t total_set_transport_unique_candidate_count = 0;
    std::uint64_t total_set_transport_nominal_candidate_count = 0;
    std::uint64_t total_set_transport_duplicate_candidate_count = 0;
    std::uint64_t temporal_frame_candidate_terms_actual = 0;
    std::uint64_t temporal_descriptor_pixel_count_actual = 0;
    std::uint64_t temporal_descriptor_pixel_count_dense_stage_baseline = 0;
    std::uint64_t temporal_descriptor_frame_terms_actual = 0;
    std::uint64_t temporal_descriptor_frame_terms_dense_stage_baseline = 0;
    std::uint64_t temporal_descriptor_frame_terms_single_final_baseline = 0;
    std::vector<std::size_t> executed_temporal_stages;
    std::vector<std::uint64_t> temporal_stage_active_pixel_counts;
    std::vector<std::uint64_t> temporal_stage_accepted_pixel_counts;
    std::vector<std::uint64_t> temporal_stage_sample_descriptor_pixel_counts;
    std::vector<std::uint64_t> temporal_stage_reference_descriptor_pixel_counts;
    std::string displace_detail;
    const auto displace_t0 = std::chrono::steady_clock::now();
    for (std::size_t stage_index = 0; stage_index < temporal_stages.size(); ++stage_index) {
        if (stage_index > 0 && std::none_of(
                active_full.begin(), active_full.end(),
                [](std::uint8_t value) { return value != 0; })) {
            break;
        }
        const std::size_t stage_frames = temporal_stages[stage_index];
        executed_temporal_stages.push_back(stage_frames);
        const bool lazy_descriptor_stage =
            easy_to_hard_.lazy_temporal_descriptors &&
            !easy_to_hard_.prefix_compatible_temporal_wavelet &&
            stage_index > 0;
        std::vector<float> stage_contrast_raw;
        if (easy_to_hard_.confidence_aware_refinement) {
            stage_contrast_raw = temporal_speckle_contrast_map(stage_frames);
        }
        RawPyramidResult lazy_img_raw;
        RawPyramidResult lazy_ref_raw;
        if (easy_to_hard_.prefix_compatible_temporal_wavelet) {
            const auto incremental_t0 = std::chrono::steady_clock::now();
            extend_prefix_compatible_wavelet_state(
                prefix_img_state, prefix_wavelet_basis,
                img_data_, stage_frames);
            extend_prefix_compatible_wavelet_state(
                prefix_ref_state, prefix_wavelet_basis,
                ref_data_, stage_frames);
            auto img_base = materialize_prefix_compatible_wavelet(
                prefix_img_state, prefix_wavelet_basis);
            auto ref_base = materialize_prefix_compatible_wavelet(
                prefix_ref_state, prefix_wavelet_basis);
            const auto incremental_t1 = std::chrono::steady_clock::now();
            wavelet_time += std::chrono::duration<double>(
                incremental_t1 - incremental_t0).count();

            const auto spatial_t0 = std::chrono::steady_clock::now();
            p.img_levels = descriptor_spatial_pyramid_hwd(
                img_base.coeffs_filter, h_, w_, img_base.out_depth,
                pyramid_level_, PyramidDownsampleMode::Db3Aa);
            p.ref_levels = descriptor_spatial_pyramid_hwd(
                ref_base.coeffs_filter, h_, w_, ref_base.out_depth,
                pyramid_level_, PyramidDownsampleMode::Db3Aa);
            const auto spatial_t1 = std::chrono::steady_clock::now();
            pyramid_time += std::chrono::duration<double>(
                spatial_t1 - spatial_t0).count();
        } else if (lazy_descriptor_stage) {
            const std::size_t prefix_values = stage_frames * h_ * w_;
            std::vector<float> img_prefix(
                img_data_.begin(),
                img_data_.begin() + static_cast<std::ptrdiff_t>(prefix_values));
            std::vector<float> ref_prefix(
                ref_data_.begin(),
                ref_data_.begin() + static_cast<std::ptrdiff_t>(prefix_values));
            lazy_img_raw = normalized_raw_pyramid_data(
                img_prefix, stage_frames, h_, w_, pyramid_level_,
                PyramidDownsampleMode::Db3Aa);
            lazy_ref_raw = normalized_raw_pyramid_data(
                ref_prefix, stage_frames, h_, w_, pyramid_level_,
                PyramidDownsampleMode::Db3Aa);
            pyramid_time += lazy_img_raw.normalize_time_s +
                lazy_img_raw.downsample_time_s + lazy_ref_raw.normalize_time_s +
                lazy_ref_raw.downsample_time_s;

            p.img_levels.clear();
            p.ref_levels.clear();
            p.img_levels.reserve(lazy_img_raw.levels.size());
            p.ref_levels.reserve(lazy_ref_raw.levels.size());
            for (std::size_t level = 0; level < lazy_img_raw.levels.size(); ++level) {
                const auto& img_level = lazy_img_raw.levels[level];
                const auto& ref_level = lazy_ref_raw.levels[level];
                p.img_levels.push_back(PyramidLevel{
                    {}, 0, img_level.h, img_level.w});
                p.ref_levels.push_back(PyramidLevel{
                    {}, 0, ref_level.h, ref_level.w});
            }
            wavelet_level_ = use_wavelet_
                ? dwt_max_level_db2(stage_frames)
                : 0;
            wavelet_add_list_ = use_wavelet_
                ? wavelet_add_list_for_depth(stage_frames)
                : std::vector<int>{};
        } else {
            double stage_pyramid_time = 0.0;
            double stage_template_time = 0.0;
            double stage_wavelet_time = 0.0;
            auto stage_p = wavelet_data_for_frame_prefix(
                stage_frames, stage_pyramid_time, stage_template_time,
                stage_wavelet_time);
            pyramid_time += stage_pyramid_time;
            total_template_window_time += stage_template_time;
            wavelet_time += stage_wavelet_time;
            p = std::move(stage_p);
        }

        if (stage_index == 0) {
            m0 = p.img_levels[0].d1;
            n0 = p.img_levels[0].d2;
            if (displace_estimate_h_ == m0 && displace_estimate_w_ == n0) {
                merged_displace_y = std::move(displace_estimate_y_);
                merged_displace_x = std::move(displace_estimate_x_);
            } else {
                merged_displace_y = resampling_spline(
                    displace_estimate_y_, displace_estimate_h_,
                    displace_estimate_w_, m0, n0);
                merged_displace_x = resampling_spline(
                    displace_estimate_x_, displace_estimate_h_,
                    displace_estimate_w_, m0, n0);
            }
            base_displace_y = merged_displace_y;
            base_displace_x = merged_displace_x;
            const std::size_t pixels = m0 * n0;
            darkfield_nd.assign(pixels, 0.0f);
            second_best_score_neg_ssd.assign(pixels, 0.0f);
            score_margin.assign(pixels, 0.0f);
            peak_hessian_det.assign(pixels, 0.0f);
            normalized_peak_curvature.assign(pixels, 0.0f);
            interlevel_displacement_delta.assign(pixels, 0.0f);
            search_boundary_hit.assign(pixels, 0.0f);
            search_geometry_valid.assign(pixels, 0.0f);
            temporal_speckle_contrast.assign(pixels, 0.0f);
            effective_search_half_window.assign(pixels, 0.0f);
            temporal_frames_used.assign(pixels, 0.0f);
            confidence_path.assign(pixels, 0.0f);
            if (wavelet_guided_umpa_.enabled || fixed_set_transport_) {
                merged_integer_proposal_y.assign(pixels, 0);
                merged_integer_proposal_x.assign(pixels, 0);
            }
            active_full.assign(pixels, 1);
        }

        std::vector<float> stage_contrast_full;
        if (easy_to_hard_.confidence_aware_refinement) {
            if (h_ == m0 && w_ == n0) {
                stage_contrast_full = std::move(stage_contrast_raw);
            } else {
                stage_contrast_full = resampling_spline(
                    stage_contrast_raw, h_, w_, m0, n0);
            }
            for (float& value : stage_contrast_full) {
                value = std::max(value, 0.0f);
            }
        }
        const std::vector<std::uint8_t> stage_active_full = active_full;
        temporal_stage_active_pixel_counts.push_back(
            static_cast<std::uint64_t>(std::count(
                stage_active_full.begin(), stage_active_full.end(),
                static_cast<std::uint8_t>(1))));

        // A later temporal stage is an evidence fallback, not another motion
        // scale. Restart active pixels from the original external estimate so
        // an unstable short-prefix displacement cannot steer the full-prefix
        // Pyramid WSVT into a different local basin. Inactive pixels are
        // skipped by the active mask and retain their merged early result.
        std::vector<float> displace_y = stage_index == 0
            ? merged_displace_y
            : base_displace_y;
        std::vector<float> displace_x = stage_index == 0
            ? merged_displace_x
            : base_displace_x;
        DisplaceWaveletOutput final_level_result;
        std::uint64_t stage_sample_descriptor_pixels = 0;
        std::uint64_t stage_reference_descriptor_pixels = 0;
        std::uint64_t stage_dense_descriptor_pixels = 0;
        for (int k_iter = 0; k_iter < n_iter_; ++k_iter) {
            const float inv_scale = 1.0f / static_cast<float>(
                std::pow(2.0, static_cast<double>(pyramid_level_)));
            for (float& value : displace_y) value *= inv_scale;
            for (float& value : displace_x) value *= inv_scale;
            const std::size_t coarse_h =
                p.img_levels[static_cast<std::size_t>(pyramid_level_)].d1;
            const std::size_t coarse_w =
                p.img_levels[static_cast<std::size_t>(pyramid_level_)].d2;
            auto t_up0 = std::chrono::steady_clock::now();
            displace_y = resampling_spline(
                displace_y, m0, n0, coarse_h, coarse_w);
            displace_x = resampling_spline(
                displace_x, m0, n0, coarse_h, coarse_w);
            auto t_up1 = std::chrono::steady_clock::now();
            total_displace_upsample_s +=
                std::chrono::duration<double>(t_up1 - t_up0).count();
            const float lim_coarse = static_cast<float>(cal_half_window_) /
                static_cast<float>(std::pow(
                    2.0, static_cast<double>(pyramid_level_)));
            clamp_2d(displace_y, -lim_coarse, lim_coarse);
            clamp_2d(displace_x, -lim_coarse, lim_coarse);
            std::vector<float> prior_level_confidence;
            std::size_t prior_h = 0;
            std::size_t prior_w = 0;
            std::vector<float> coarse_set_hypothesis_y;
            std::vector<float> coarse_set_hypothesis_x;
            std::size_t coarse_set_h = 0;
            std::size_t coarse_set_w = 0;

            for (int p_level = pyramid_level_; p_level >= 0; --p_level) {
                const int configured_search_half_window =
                    searching_window_pyramid_list[
                        static_cast<std::size_t>(p_level)];
                const bool prefix_search_stage =
                    easy_to_hard_.prefix_compatible_temporal_wavelet &&
                    stage_index + 1 < temporal_stages.size();
                const int search_half_window = prefix_search_stage
                    ? std::min(
                        configured_search_half_window,
                        easy_to_hard_.temporal_prefix_half_window)
                    : configured_search_half_window;
                const std::size_t lv = static_cast<std::size_t>(p_level);
                const std::size_t ph = p.img_levels[lv].d1;
                const std::size_t pw = p.img_levels[lv].d2;
                std::size_t depth = p.img_levels[lv].d0;
                std::vector<float> displace_pyramid_y(ph * pw, 0.0f);
                std::vector<float> displace_pyramid_x(ph * pw, 0.0f);
                std::vector<int> set_level_hypothesis_y;
                std::vector<int> set_level_hypothesis_x;
                if (p_level == pyramid_level_) {
                    for (std::size_t i = 0; i < ph * pw; ++i) {
                        displace_pyramid_y[i] = static_cast<float>(
                            round_half_to_even(static_cast<double>(displace_y[i])));
                        displace_pyramid_x[i] = static_cast<float>(
                            round_half_to_even(static_cast<double>(displace_x[i])));
                    }
                } else {
                    const std::size_t old_h = p.img_levels[lv + 1].d1;
                    const std::size_t old_w = p.img_levels[lv + 1].d2;
                    t_up0 = std::chrono::steady_clock::now();
                    auto up_y = resampling_spline(
                        displace_y, old_h, old_w, ph, pw);
                    auto up_x = resampling_spline(
                        displace_x, old_h, old_w, ph, pw);
                    t_up1 = std::chrono::steady_clock::now();
                    total_displace_upsample_s +=
                        std::chrono::duration<double>(t_up1 - t_up0).count();
                    for (std::size_t i = 0; i < ph * pw; ++i) {
                        displace_pyramid_y[i] = static_cast<float>(
                            round_half_to_even(static_cast<double>(up_y[i] * 2.0f)));
                        displace_pyramid_x[i] = static_cast<float>(
                            round_half_to_even(static_cast<double>(up_x[i] * 2.0f)));
                    }
                }
                const float lim = static_cast<float>(cal_half_window_) /
                    static_cast<float>(std::pow(
                        2.0, static_cast<double>(p_level)));
                clamp_2d(displace_pyramid_y, -lim, lim);
                clamp_2d(displace_pyramid_x, -lim, lim);
                if (fixed_set_transport_ && p_level == 0) {
                    constexpr std::size_t kSetWidth = 4;
                    if (coarse_set_hypothesis_y.size() !=
                            coarse_set_h * coarse_set_w * kSetWidth ||
                        coarse_set_hypothesis_x.size() !=
                            coarse_set_h * coarse_set_w * kSetWidth ||
                        coarse_set_h < 2 || coarse_set_w < 2) {
                        throw std::logic_error(
                            "coarse fixed set hypotheses are unavailable");
                    }
                    set_level_hypothesis_y.resize(ph * pw * kSetWidth);
                    set_level_hypothesis_x.resize(ph * pw * kSetWidth);
                    merged_set_transport_center_y.resize(ph * pw * kSetWidth);
                    merged_set_transport_center_x.resize(ph * pw * kSetWidth);
                    for (std::size_t y = 0; y < ph; ++y) {
                        const std::size_t parent_y = static_cast<std::size_t>(
                            round_half_to_even(
                                static_cast<double>(y) *
                                static_cast<double>(coarse_set_h - 1) /
                                static_cast<double>(ph - 1)));
                        for (std::size_t x = 0; x < pw; ++x) {
                            const std::size_t parent_x = static_cast<std::size_t>(
                                round_half_to_even(
                                    static_cast<double>(x) *
                                    static_cast<double>(coarse_set_w - 1) /
                                    static_cast<double>(pw - 1)));
                            const std::size_t parent_base =
                                (parent_y * coarse_set_w + parent_x) * kSetWidth;
                            const std::size_t fine_base =
                                (y * pw + x) * kSetWidth;
                            for (std::size_t hypothesis = 0;
                                 hypothesis < kSetWidth; ++hypothesis) {
                                const int center_y = std::clamp(
                                    static_cast<int>(round_half_to_even(
                                        static_cast<double>(
                                            coarse_set_hypothesis_y[
                                                parent_base + hypothesis]) *
                                        2.0)),
                                    -cal_half_window_, cal_half_window_);
                                const int center_x = std::clamp(
                                    static_cast<int>(round_half_to_even(
                                        static_cast<double>(
                                            coarse_set_hypothesis_x[
                                                parent_base + hypothesis]) *
                                        2.0)),
                                    -cal_half_window_, cal_half_window_);
                                set_level_hypothesis_y[fine_base + hypothesis] =
                                    center_y;
                                set_level_hypothesis_x[fine_base + hypothesis] =
                                    center_x;
                                merged_set_transport_center_y[
                                    fine_base + hypothesis] =
                                    static_cast<float>(center_y);
                                merged_set_transport_center_x[
                                    fine_base + hypothesis] =
                                    static_cast<float>(center_x);
                            }
                            displace_pyramid_y[y * pw + x] = static_cast<float>(
                                set_level_hypothesis_y[fine_base]);
                            displace_pyramid_x[y * pw + x] = static_cast<float>(
                                set_level_hypothesis_x[fine_base]);
                        }
                    }
                }
                const int n_pad = static_cast<int>(std::ceil(
                    static_cast<double>(cal_half_window_) /
                    std::pow(2.0, static_cast<double>(p_level))));

                std::vector<std::uint8_t> active_level;
                std::vector<std::uint8_t> easy_level;
                std::vector<float> contrast_level;
                if (easy_to_hard_.confidence_aware_refinement) {
                    std::vector<float> active_float(
                        stage_active_full.begin(), stage_active_full.end());
                    auto active_resampled = (ph == m0 && pw == n0)
                        ? std::move(active_float)
                        : resampling_spline(active_float, m0, n0, ph, pw);
                    active_level.resize(ph * pw, 0);
                    for (std::size_t i = 0; i < ph * pw; ++i) {
                        active_level[i] = active_resampled[i] > 0.01f ? 1 : 0;
                    }
                    contrast_level = (ph == m0 && pw == n0)
                        ? stage_contrast_full
                        : resampling_spline(
                            stage_contrast_full, m0, n0, ph, pw);
                    easy_level.assign(ph * pw, 0);
                    if (!prior_level_confidence.empty()) {
                        auto projected = resampling_spline(
                            prior_level_confidence, prior_h, prior_w, ph, pw);
                        for (std::size_t i = 0; i < ph * pw; ++i) {
                            easy_level[i] = projected[i] >= 0.5f ? 1 : 0;
                        }
                    }
                }

                const std::uint64_t dense_level_pixels =
                    static_cast<std::uint64_t>(2 * ph * pw);
                stage_dense_descriptor_pixels += dense_level_pixels;
                if (lazy_descriptor_stage) {
                    auto ref_mask = candidate_reference_mask(
                        active_level, ph, pw, displace_pyramid_y,
                        displace_pyramid_x, search_half_window);
                    const int coefs_level =
                        wavelet_level_ + 1 - wavelet_level_cut_;
                    const int wavelevel_add = lv >= wavelet_add_list_.size()
                        ? 2
                        : wavelet_add_list_[lv];
                    const int return_level = coefs_level + wavelevel_add;
                    auto img_descriptor = build_sparse_temporal_descriptor(
                        lazy_img_raw.levels[lv], active_level, use_wavelet_,
                        wavelet_level_, return_level);
                    auto ref_descriptor = build_sparse_temporal_descriptor(
                        lazy_ref_raw.levels[lv], ref_mask, use_wavelet_,
                        wavelet_level_, return_level);
                    if (img_descriptor.level.d0 != ref_descriptor.level.d0) {
                        throw std::runtime_error(
                            "lazy temporal descriptor depth mismatch");
                    }
                    depth = img_descriptor.level.d0;
                    stage_sample_descriptor_pixels +=
                        img_descriptor.selected_pixel_count;
                    stage_reference_descriptor_pixels +=
                        ref_descriptor.selected_pixel_count;
                    const double sparse_gather_scatter_s =
                        img_descriptor.gather_scatter_time_s +
                        ref_descriptor.gather_scatter_time_s;
                    total_template_window_time += sparse_gather_scatter_s;
                    pyramid_time += sparse_gather_scatter_s;
                    wavelet_time += img_descriptor.wavelet_time_s +
                        ref_descriptor.wavelet_time_s;
                    p.img_levels[lv] = std::move(img_descriptor.level);
                    p.ref_levels[lv] = std::move(ref_descriptor.level);
                } else {
                    stage_sample_descriptor_pixels +=
                        static_cast<std::uint64_t>(ph * pw);
                    stage_reference_descriptor_pixels +=
                        static_cast<std::uint64_t>(ph * pw);
                }

                const std::size_t pad_all = static_cast<std::size_t>(
                    n_pad + search_half_window);
                auto t_pad0 = std::chrono::steady_clock::now();
                auto ref_wa_pad = pad_hwd_zero_aligned(
                    TensorView3D<const float, Layout::HWD>{
                        p.ref_levels[lv].data.data(), {ph, pw, depth}},
                    pad_all, pad_all);
                const std::size_t ref_pad_h = ref_wa_pad.shape().d0;
                const std::size_t ref_pad_w = ref_wa_pad.shape().d1;
                auto t_pad1 = std::chrono::steady_clock::now();
                total_displace_pad_s +=
                    std::chrono::duration<double>(t_pad1 - t_pad0).count();

                auto t_search0 = std::chrono::steady_clock::now();
                auto result = displace_wavelet(
                    as_span(p.img_levels[lv].data), ph, pw,
                    ref_wa_pad.flat(), ref_pad_h, ref_pad_w, depth,
                    displace_pyramid_y, displace_pyramid_x,
                    search_half_window, n_pad,
                    active_level, easy_level, contrast_level,
                    set_level_hypothesis_y, set_level_hypothesis_x, p_level);
                auto t_search1 = std::chrono::steady_clock::now();
                total_displace_search_s +=
                    std::chrono::duration<double>(t_search1 - t_search0).count();
                total_search_candidate_count += result.search_candidate_count;
                total_search_abandoned_candidate_count +=
                    result.search_abandoned_candidate_count;
                total_search_distance_terms_evaluated +=
                    result.search_distance_terms_evaluated;
                total_search_distance_terms_possible +=
                    result.search_distance_terms_possible;
                total_search_refine_terms_evaluated +=
                    result.search_refine_terms_evaluated;
                total_search_prefix_terms_evaluated +=
                    result.search_prefix_terms_evaluated;
                total_search_full_candidate_count +=
                    result.search_full_candidate_count;
                total_search_guard_check_count += result.search_guard_check_count;
                total_search_guard_refresh_count +=
                    result.search_guard_refresh_count;
                total_easy_path_eligible_pixel_count +=
                    result.easy_path_eligible_pixel_count;
                total_easy_path_accepted_pixel_count +=
                    result.easy_path_accepted_pixel_count;
                total_easy_path_fallback_pixel_count +=
                    result.easy_path_fallback_pixel_count;
                total_inactive_pixel_count += result.inactive_pixel_count;
                total_set_transport_unique_candidate_count +=
                    result.set_transport_unique_candidate_count;
                total_set_transport_nominal_candidate_count +=
                    result.set_transport_nominal_candidate_count;
                total_set_transport_duplicate_candidate_count +=
                    result.set_transport_duplicate_candidate_count;
                temporal_frame_candidate_terms_actual +=
                    result.search_candidate_count *
                    static_cast<std::uint64_t>(stage_frames);
                if (stage_index == 0) {
                    dense_baseline_candidate_count +=
                        result.search_dense_baseline_candidate_count;
                }

                clamp_2d(result.displace_y, -lim, lim);
                clamp_2d(result.displace_x, -lim, lim);
                displace_y = result.displace_y;
                displace_x = result.displace_x;
                prior_level_confidence = result.confidence_accept;
                prior_h = ph;
                prior_w = pw;
                if (fixed_set_transport_ && p_level == pyramid_level_) {
                    coarse_set_hypothesis_y =
                        std::move(result.coarse_set_hypothesis_y);
                    coarse_set_hypothesis_x =
                        std::move(result.coarse_set_hypothesis_x);
                    coarse_set_h = ph;
                    coarse_set_w = pw;
                }
                if (p_level == 0) {
                    if (fixed_set_transport_) {
                        merged_set_transport_candidates_evaluated =
                            result.set_transport_candidates_evaluated;
                        merged_set_transport_representative_y =
                            result.set_transport_representative_y;
                        merged_set_transport_representative_x =
                            result.set_transport_representative_x;
                    }
                    final_level_result = std::move(result);
                }
            }
        }
        const std::uint64_t stage_descriptor_pixels =
            stage_sample_descriptor_pixels + stage_reference_descriptor_pixels;
        temporal_descriptor_pixel_count_actual += stage_descriptor_pixels;
        temporal_descriptor_pixel_count_dense_stage_baseline +=
            stage_dense_descriptor_pixels;
        const std::size_t descriptor_frames_added =
            easy_to_hard_.prefix_compatible_temporal_wavelet
            ? stage_frames - (stage_index == 0
                ? 0
                : temporal_stages[stage_index - 1])
            : stage_frames;
        temporal_descriptor_frame_terms_actual += stage_descriptor_pixels *
            static_cast<std::uint64_t>(descriptor_frames_added);
        temporal_descriptor_frame_terms_dense_stage_baseline +=
            stage_dense_descriptor_pixels *
            static_cast<std::uint64_t>(stage_frames);
        if (stage_index == 0) {
            temporal_descriptor_frame_terms_single_final_baseline =
                stage_dense_descriptor_pixels *
                static_cast<std::uint64_t>(temporal_stages.back());
        }
        temporal_stage_sample_descriptor_pixel_counts.push_back(
            stage_sample_descriptor_pixels);
        temporal_stage_reference_descriptor_pixel_counts.push_back(
            stage_reference_descriptor_pixels);

        std::uint64_t accepted_this_stage = 0;
        for (std::size_t pixel = 0; pixel < m0 * n0; ++pixel) {
            if (stage_active_full[pixel] == 0) {
                continue;
            }
            merged_displace_y[pixel] = final_level_result.displace_y[pixel];
            merged_displace_x[pixel] = final_level_result.displace_x[pixel];
            darkfield_nd[pixel] = final_level_result.best_score_neg_ssd[pixel];
            second_best_score_neg_ssd[pixel] =
                final_level_result.second_best_score_neg_ssd[pixel];
            score_margin[pixel] = final_level_result.score_margin[pixel];
            peak_hessian_det[pixel] = final_level_result.peak_hessian_det[pixel];
            normalized_peak_curvature[pixel] =
                final_level_result.normalized_peak_curvature[pixel];
            interlevel_displacement_delta[pixel] =
                final_level_result.interlevel_displacement_delta[pixel];
            search_boundary_hit[pixel] =
                final_level_result.search_boundary_hit[pixel];
            search_geometry_valid[pixel] =
                final_level_result.search_geometry_valid[pixel];
            effective_search_half_window[pixel] =
                final_level_result.effective_search_half_window[pixel];
            confidence_path[pixel] = final_level_result.confidence_path[pixel];
            if (wavelet_guided_umpa_.enabled || fixed_set_transport_) {
                merged_integer_proposal_y[pixel] =
                    final_level_result.integer_displace_y[pixel];
                merged_integer_proposal_x[pixel] =
                    final_level_result.integer_displace_x[pixel];
            }

            bool accept = final_level_result.confidence_accept[pixel] != 0.0f;
            if (easy_to_hard_.temporal_probe_first_stage_only &&
                stage_index == 0) {
                accept = true;
            }
            if (easy_to_hard_.adaptive_frames && stage_index > 0) {
                const float temporal_delta = std::hypot(
                    final_level_result.displace_y[pixel] - previous_temporal_y[pixel],
                    final_level_result.displace_x[pixel] - previous_temporal_x[pixel]);
                accept = accept &&
                    temporal_delta <= easy_to_hard_.temporal_delta_max_px;
                if (temporal_delta > easy_to_hard_.temporal_delta_max_px) {
                    confidence_path[pixel] = 4.0f;
                }
            }
            if (!easy_to_hard_.adaptive_frames ||
                stage_index + 1 == temporal_stages.size()) {
                accept = true;
            }
            if (accept) {
                active_full[pixel] = 0;
                temporal_frames_used[pixel] = static_cast<float>(stage_frames);
                temporal_speckle_contrast[pixel] =
                    easy_to_hard_.confidence_aware_refinement
                    ? stage_contrast_full[pixel]
                    : 0.0f;
                ++accepted_this_stage;
            }
        }
        temporal_stage_accepted_pixel_counts.push_back(accepted_this_stage);
        previous_temporal_y = final_level_result.displace_y;
        previous_temporal_x = final_level_result.displace_x;
    }

    // The last temporal stage consumes the owned CHW inputs. Release any
    // implementation-specific residual capacity before post-processing.
    std::vector<float>().swap(img_data_);
    std::vector<float>().swap(ref_data_);
    const auto displace_t1 = std::chrono::steady_clock::now();
    const double displace_time_s = std::chrono::duration<double>(displace_t1 - displace_t0).count();
    prColor("displace time: " + std::to_string(displace_time_s) + " s", "light_purple");
    prColor("  displace detail: upsample=" + std::to_string(total_displace_upsample_s) +
            "s pad=" + std::to_string(total_displace_pad_s) +
            "s search=" + std::to_string(total_displace_search_s) + "s", "light_purple");
    if (search_early_abandon_ || search_two_pass_) {
        const double abandon_fraction = total_search_candidate_count == 0 ? 0.0 :
            static_cast<double>(total_search_abandoned_candidate_count) /
            static_cast<double>(total_search_candidate_count);
        const double term_fraction = total_search_distance_terms_possible == 0 ? 1.0 :
            static_cast<double>(total_search_distance_terms_evaluated) /
            static_cast<double>(total_search_distance_terms_possible);
        prColor("  exact screening: profile=" + std::string(search_profile_name(
                    search_early_abandon_, search_two_pass_, search_guard_cache_)) +
                " candidates=" + std::to_string(total_search_candidate_count) +
                " abandoned=" + std::to_string(total_search_abandoned_candidate_count) +
                " fraction=" + std::to_string(abandon_fraction) +
                " term_fraction=" + std::to_string(term_fraction) +
                " refine_terms=" + std::to_string(total_search_refine_terms_evaluated) +
                " prefix_terms=" + std::to_string(total_search_prefix_terms_evaluated) +
                " complete_candidates=" + std::to_string(total_search_full_candidate_count) +
                " guard_checks=" + std::to_string(total_search_guard_check_count) +
                " guard_refreshes=" + std::to_string(total_search_guard_refresh_count),
                "light_purple");
    }
    (void)configure_openmp_threads(phase_cores_, "post-process/FFTW phase recovery", 1);
    const auto post_t0 = std::chrono::steady_clock::now();
    const std::size_t disp_h = p.img_levels[0].d1;
    const std::size_t disp_w = p.img_levels[0].d2;
    const std::size_t pad_crop = static_cast<std::size_t>(cal_half_window_);
    if (!easy_to_hard_.confidence_aware_refinement) {
        std::fill(temporal_speckle_contrast.begin(),
                  temporal_speckle_contrast.end(), 0.0f);
        std::fill(effective_search_half_window.begin(),
                  effective_search_half_window.end(), 0.0f);
        std::fill(temporal_frames_used.begin(),
                  temporal_frames_used.end(), 0.0f);
        std::fill(confidence_path.begin(), confidence_path.end(), 0.0f);
    }
    auto displace_y_crop_img = crop_2d(
        ImageView2D<const float>{merged_displace_y.data(), {disp_h, disp_w}}, pad_crop);
    auto displace_x_crop_img = crop_2d(
        ImageView2D<const float>{merged_displace_x.data(), {disp_h, disp_w}}, pad_crop);
    auto darkfield_nd_crop_img = crop_2d(
        ImageView2D<const float>{darkfield_nd.data(), {disp_h, disp_w}}, pad_crop);
    auto second_best_score_crop_img = crop_2d(
        ImageView2D<const float>{second_best_score_neg_ssd.data(), {disp_h, disp_w}}, pad_crop);
    auto score_margin_crop_img = crop_2d(
        ImageView2D<const float>{score_margin.data(), {disp_h, disp_w}}, pad_crop);
    auto peak_hessian_det_crop_img = crop_2d(
        ImageView2D<const float>{peak_hessian_det.data(), {disp_h, disp_w}}, pad_crop);
    auto normalized_peak_curvature_crop_img = crop_2d(
        ImageView2D<const float>{normalized_peak_curvature.data(), {disp_h, disp_w}}, pad_crop);
    auto interlevel_displacement_delta_crop_img = crop_2d(
        ImageView2D<const float>{interlevel_displacement_delta.data(), {disp_h, disp_w}}, pad_crop);
    auto search_boundary_hit_crop_img = crop_2d(
        ImageView2D<const float>{search_boundary_hit.data(), {disp_h, disp_w}}, pad_crop);
    auto search_geometry_valid_crop_img = crop_2d(
        ImageView2D<const float>{search_geometry_valid.data(), {disp_h, disp_w}}, pad_crop);
    auto temporal_speckle_contrast_crop_img = crop_2d(
        ImageView2D<const float>{temporal_speckle_contrast.data(), {disp_h, disp_w}}, pad_crop);
    auto effective_search_half_window_crop_img = crop_2d(
        ImageView2D<const float>{effective_search_half_window.data(), {disp_h, disp_w}}, pad_crop);
    auto temporal_frames_used_crop_img = crop_2d(
        ImageView2D<const float>{temporal_frames_used.data(), {disp_h, disp_w}}, pad_crop);
    auto confidence_path_crop_img = crop_2d(
        ImageView2D<const float>{confidence_path.data(), {disp_h, disp_w}}, pad_crop);
    const std::size_t cropped_h = displace_y_crop_img.shape().h;
    const std::size_t cropped_w = displace_y_crop_img.shape().w;
    auto displace_y_crop = std::move(displace_y_crop_img).take();
    auto displace_x_crop = std::move(displace_x_crop_img).take();
    auto darkfield_nd_crop = std::move(darkfield_nd_crop_img).take();
    auto second_best_score_crop = std::move(second_best_score_crop_img).take();
    auto score_margin_crop = std::move(score_margin_crop_img).take();
    auto peak_hessian_det_crop = std::move(peak_hessian_det_crop_img).take();
    auto normalized_peak_curvature_crop =
        std::move(normalized_peak_curvature_crop_img).take();
    auto interlevel_displacement_delta_crop =
        std::move(interlevel_displacement_delta_crop_img).take();
    auto search_boundary_hit_crop = std::move(search_boundary_hit_crop_img).take();
    auto search_geometry_valid_crop = std::move(search_geometry_valid_crop_img).take();
    auto temporal_speckle_contrast_crop =
        std::move(temporal_speckle_contrast_crop_img).take();
    auto effective_search_half_window_crop =
        std::move(effective_search_half_window_crop_img).take();
    auto temporal_frames_used_crop =
        std::move(temporal_frames_used_crop_img).take();
    auto confidence_path_crop = std::move(confidence_path_crop_img).take();
    std::vector<float> set_transport_center_y_crop;
    std::vector<float> set_transport_center_x_crop;
    std::vector<float> set_transport_representative_y_crop;
    std::vector<float> set_transport_representative_x_crop;
    std::vector<float> set_transport_candidates_evaluated_crop;
    if (fixed_set_transport_) {
        constexpr std::size_t kSetWidth = 4;
        if (merged_set_transport_center_y.size() != disp_h * disp_w * kSetWidth ||
            merged_set_transport_center_x.size() != disp_h * disp_w * kSetWidth ||
            merged_set_transport_representative_y.size() !=
                disp_h * disp_w * kSetWidth ||
            merged_set_transport_representative_x.size() !=
                disp_h * disp_w * kSetWidth ||
            merged_set_transport_candidates_evaluated.size() != disp_h * disp_w) {
            throw std::logic_error(
                "fixed set transport final diagnostics are incomplete");
        }
        set_transport_center_y_crop.resize(
            kSetWidth * cropped_h * cropped_w);
        set_transport_center_x_crop.resize(
            kSetWidth * cropped_h * cropped_w);
        set_transport_representative_y_crop.resize(
            kSetWidth * cropped_h * cropped_w);
        set_transport_representative_x_crop.resize(
            kSetWidth * cropped_h * cropped_w);
        for (std::size_t hypothesis = 0; hypothesis < kSetWidth; ++hypothesis) {
            std::vector<float> full_y(disp_h * disp_w);
            std::vector<float> full_x(disp_h * disp_w);
            for (std::size_t pixel = 0; pixel < disp_h * disp_w; ++pixel) {
                full_y[pixel] = merged_set_transport_center_y[
                    pixel * kSetWidth + hypothesis];
                full_x[pixel] = merged_set_transport_center_x[
                    pixel * kSetWidth + hypothesis];
            }
            auto cropped_y = crop_2d(
                ImageView2D<const float>{full_y.data(), {disp_h, disp_w}},
                pad_crop).take();
            auto cropped_x = crop_2d(
                ImageView2D<const float>{full_x.data(), {disp_h, disp_w}},
                pad_crop).take();
            const std::size_t output_base = hypothesis * cropped_h * cropped_w;
            for (std::size_t pixel = 0; pixel < cropped_y.size(); ++pixel) {
                set_transport_center_y_crop[output_base + pixel] =
                    -cropped_y[pixel];
                set_transport_center_x_crop[output_base + pixel] =
                    -cropped_x[pixel];
            }
            std::vector<float> full_representative_y(disp_h * disp_w);
            std::vector<float> full_representative_x(disp_h * disp_w);
            for (std::size_t pixel = 0; pixel < disp_h * disp_w; ++pixel) {
                full_representative_y[pixel] = static_cast<float>(
                    merged_set_transport_representative_y[
                        pixel * kSetWidth + hypothesis]);
                full_representative_x[pixel] = static_cast<float>(
                    merged_set_transport_representative_x[
                        pixel * kSetWidth + hypothesis]);
            }
            auto cropped_representative_y = crop_2d(
                ImageView2D<const float>{
                    full_representative_y.data(), {disp_h, disp_w}},
                pad_crop).take();
            auto cropped_representative_x = crop_2d(
                ImageView2D<const float>{
                    full_representative_x.data(), {disp_h, disp_w}},
                pad_crop).take();
            for (std::size_t pixel = 0;
                 pixel < cropped_representative_y.size(); ++pixel) {
                set_transport_representative_y_crop[output_base + pixel] =
                    -cropped_representative_y[pixel];
                set_transport_representative_x_crop[output_base + pixel] =
                    -cropped_representative_x[pixel];
            }
        }
        set_transport_candidates_evaluated_crop = crop_2d(
            ImageView2D<const float>{
                merged_set_transport_candidates_evaluated.data(),
                {disp_h, disp_w}},
            pad_crop).take();
    }
    for (float& v : displace_y_crop) v = -v;
    for (float& v : displace_x_crop) v = -v;
    std::vector<float> integer_winner_y_crop;
    std::vector<float> integer_winner_x_crop;
    if (wavelet_guided_umpa_.enabled || fixed_set_transport_) {
        if (merged_integer_proposal_y.size() != disp_h * disp_w ||
            merged_integer_proposal_x.size() != disp_h * disp_w) {
            throw std::logic_error(
                "integer descriptor winner diagnostics are incomplete");
        }
        std::vector<float> integer_winner_y_full(
            merged_integer_proposal_y.begin(), merged_integer_proposal_y.end());
        std::vector<float> integer_winner_x_full(
            merged_integer_proposal_x.begin(), merged_integer_proposal_x.end());
        integer_winner_y_crop = crop_2d(
            ImageView2D<const float>{
                integer_winner_y_full.data(), {disp_h, disp_w}},
            pad_crop).take();
        integer_winner_x_crop = crop_2d(
            ImageView2D<const float>{
                integer_winner_x_full.data(), {disp_h, disp_w}},
            pad_crop).take();
        for (float& value : integer_winner_y_crop) value = -value;
        for (float& value : integer_winner_x_crop) value = -value;
    }
    WaveletGuidedUmpaOutput umpa_output;
    double umpa_refine_time_s = 0.0;
    if (wavelet_guided_umpa_.enabled) {
        std::vector<int> integer_proposal_y_crop(
            integer_winner_y_crop.size());
        std::vector<int> integer_proposal_x_crop(
            integer_winner_x_crop.size());
        for (std::size_t pixel = 0;
              pixel < integer_proposal_y_crop.size(); ++pixel) {
            integer_proposal_y_crop[pixel] = static_cast<int>(
                integer_winner_y_crop[pixel]);
            integer_proposal_x_crop[pixel] = static_cast<int>(
                integer_winner_x_crop[pixel]);
        }
        const auto umpa_t0 = std::chrono::steady_clock::now();
        umpa_output = refine_wavelet_guided_umpa(
            umpa_sample_raw, umpa_reference_raw,
            ch_, h_, w_,
            integer_proposal_y_crop, integer_proposal_x_crop,
            cropped_h, cropped_w, pad_crop, pad_crop,
            wavelet_guided_umpa_);
        const auto umpa_t1 = std::chrono::steady_clock::now();
        umpa_refine_time_s = std::chrono::duration<double>(
            umpa_t1 - umpa_t0).count();
        displace_y_crop = umpa_output.displace_y;
        displace_x_crop = umpa_output.displace_x;
        std::vector<float>().swap(umpa_sample_raw);
        std::vector<float>().swap(umpa_reference_raw);
    } else if (set_transport_raw_rerank_.enabled) {
        constexpr std::size_t kSetWidth = 4U;
        const std::size_t output_pixels = cropped_h * cropped_w;
        if (set_transport_center_y_crop.size() != output_pixels * kSetWidth ||
            set_transport_center_x_crop.size() != output_pixels * kSetWidth) {
            throw std::logic_error(
                "SET4 raw rerank requires complete cropped hypothesis centres");
        }
        std::vector<int> set_center_y(output_pixels * kSetWidth);
        std::vector<int> set_center_x(output_pixels * kSetWidth);
        std::vector<int> set_representative_y(output_pixels * kSetWidth);
        std::vector<int> set_representative_x(output_pixels * kSetWidth);
        for (std::size_t pixel = 0; pixel < output_pixels; ++pixel) {
            for (std::size_t hypothesis = 0; hypothesis < kSetWidth; ++hypothesis) {
                const std::size_t source = hypothesis * output_pixels + pixel;
                const std::size_t target = pixel * kSetWidth + hypothesis;
                const float center_y = set_transport_center_y_crop[source];
                const float center_x = set_transport_center_x_crop[source];
                const int integer_y = static_cast<int>(center_y);
                const int integer_x = static_cast<int>(center_x);
                if (center_y != static_cast<float>(integer_y) ||
                    center_x != static_cast<float>(integer_x)) {
                    throw std::logic_error(
                        "SET4 raw rerank centres must be exact integers");
                }
                set_center_y[target] = integer_y;
                set_center_x[target] = integer_x;
                set_representative_y[target] = static_cast<int>(
                    set_transport_representative_y_crop[source]);
                set_representative_x[target] = static_cast<int>(
                    set_transport_representative_x_crop[source]);
            }
        }
        const auto umpa_t0 = std::chrono::steady_clock::now();
        umpa_output = rerank_set_transport_raw(
            umpa_sample_raw, umpa_reference_raw, ch_, h_, w_,
            set_center_y, set_center_x,
            set_representative_y, set_representative_x,
            displace_y_crop, displace_x_crop,
            cropped_h, cropped_w, pad_crop, pad_crop,
            set_transport_raw_rerank_);
        const auto umpa_t1 = std::chrono::steady_clock::now();
        umpa_refine_time_s = std::chrono::duration<double>(
            umpa_t1 - umpa_t0).count();
        displace_y_crop = umpa_output.displace_y;
        displace_x_crop = umpa_output.displace_x;
        std::vector<float>().swap(umpa_sample_raw);
        std::vector<float>().swap(umpa_reference_raw);
    }
    const double mean_dy = mean_2d(displace_y_crop);
    const double mean_dx = mean_2d(displace_x_crop);
    const auto t_dpc0 = std::chrono::steady_clock::now();
    std::vector<float> dpc_y(displace_y_crop.size(), 0.0f);
    std::vector<float> dpc_x(displace_x_crop.size(), 0.0f);
    const double scale = p_x_ / (z_ / mag_factor_);
    for (std::size_t i = 0; i < displace_y_crop.size(); ++i) {
        dpc_y[i] = static_cast<float>((static_cast<double>(displace_y_crop[i]) - mean_dy) * scale);
        dpc_x[i] = static_cast<float>((static_cast<double>(displace_x_crop[i]) - mean_dx) * scale);
    }
    const auto t_phase0 = std::chrono::steady_clock::now();
    auto phase = frankot_chellappa(
        ImageView2D<const float>{dpc_x.data(), {cropped_h, cropped_w}},
        ImageView2D<const float>{dpc_y.data(), {cropped_h, cropped_w}}).take();
    const double phase_scale = p_x_ * 2.0 * 3.14159265358979323846 / wavelength_;
    apply_python_reference_phase_scale(phase, phase_scale);
    const auto post_t1 = std::chrono::steady_clock::now();
    const double postprocess_time_s = std::chrono::duration<double>(post_t1 - post_t0).count();
    const double crop_sign_time_s = std::chrono::duration<double>(t_dpc0 - post_t0).count();
    const double dpc_conv_time_s = std::chrono::duration<double>(t_phase0 - t_dpc0).count();
    const double phase_recovery_time_s = std::chrono::duration<double>(post_t1 - t_phase0).count();
    prColor("post-process time: " + std::to_string(postprocess_time_s) + " s", "light_purple");
    prColor("  post detail: crop_sign=" + std::to_string(crop_sign_time_s) +
            "s dpc_conv=" + std::to_string(dpc_conv_time_s) +
            "s phase_recovery=" + std::to_string(phase_recovery_time_s) + "s", "light_purple");
#ifdef _OPENMP
    omp_set_num_threads(solver_threads);
#endif
    const auto processing_t1 = std::chrono::steady_clock::now();
    const double time_cost_s = std::chrono::duration<double>(processing_t1 - processing_t0).count();
    prColor("total time: " + std::to_string(time_cost_s) + " s", "light_purple");
    prColor("  pyramid:    " + std::to_string(pyramid_time) + " s", "light_purple");
    if (total_template_window_time > 0.0) {
        prColor("  tmpl_win:   " + std::to_string(total_template_window_time) + " s", "light_purple");
    }
    prColor("  wavelet:    " + std::to_string(wavelet_time) + " s", "light_purple");
    prColor("  darkfield:  " + std::to_string(darkfield_time_s) + " s", "light_purple");
    prColor("  displace:   " + std::to_string(displace_time_s) + " s", "light_purple");
    prColor("  post-proc:  " + std::to_string(postprocess_time_s) + " s", "light_purple");

    SolverOutput output{
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
        std::move(second_best_score_crop),
        std::move(score_margin_crop),
        std::move(peak_hessian_det_crop),
        std::move(normalized_peak_curvature_crop),
        std::move(interlevel_displacement_delta_crop),
        std::move(search_boundary_hit_crop),
        std::move(search_geometry_valid_crop),
        std::move(temporal_speckle_contrast_crop),
        std::move(effective_search_half_window_crop),
        std::move(temporal_frames_used_crop),
        std::move(confidence_path_crop),
        out_h,
        out_w,
        time_cost_s,
        pyramid_time,
        wavelet_time,
        displace_time_s,
        postprocess_time_s,
        total_template_window_time,
        crop_sign_time_s,
        dpc_conv_time_s,
        phase_recovery_time_s,
        darkfield_time_s
    };
    output.search_candidate_count = total_search_candidate_count;
    output.search_abandoned_candidate_count = total_search_abandoned_candidate_count;
    output.search_distance_terms_evaluated = total_search_distance_terms_evaluated;
    output.search_distance_terms_possible = total_search_distance_terms_possible;
    output.search_refine_terms_evaluated = total_search_refine_terms_evaluated;
    output.search_prefix_terms_evaluated = total_search_prefix_terms_evaluated;
    output.search_full_candidate_count = total_search_full_candidate_count;
    output.search_guard_check_count = total_search_guard_check_count;
    output.search_guard_refresh_count = total_search_guard_refresh_count;
    output.search_dense_baseline_candidate_count =
        dense_baseline_candidate_count;
    output.set_transport_center_y = std::move(set_transport_center_y_crop);
    output.set_transport_center_x = std::move(set_transport_center_x_crop);
    if (fixed_set_transport_) {
        output.set_transport_integer_winner_y =
            std::move(integer_winner_y_crop);
        output.set_transport_integer_winner_x =
            std::move(integer_winner_x_crop);
        output.set_transport_representative_y =
            std::move(set_transport_representative_y_crop);
        output.set_transport_representative_x =
            std::move(set_transport_representative_x_crop);
    }
    output.set_transport_candidates_evaluated =
        std::move(set_transport_candidates_evaluated_crop);
    output.set_transport_unique_candidate_count =
        total_set_transport_unique_candidate_count;
    output.set_transport_nominal_candidate_count =
        total_set_transport_nominal_candidate_count;
    output.set_transport_duplicate_candidate_count =
        total_set_transport_duplicate_candidate_count;
    output.easy_path_eligible_pixel_count =
        total_easy_path_eligible_pixel_count;
    output.easy_path_accepted_pixel_count =
        total_easy_path_accepted_pixel_count;
    output.easy_path_fallback_pixel_count =
        total_easy_path_fallback_pixel_count;
    output.inactive_pixel_count = total_inactive_pixel_count;
    output.temporal_frame_candidate_terms_actual =
        temporal_frame_candidate_terms_actual;
    output.temporal_frame_candidate_terms_dense_baseline =
        dense_baseline_candidate_count *
        static_cast<std::uint64_t>(temporal_stages.back());
    output.temporal_descriptor_pixel_count_actual =
        temporal_descriptor_pixel_count_actual;
    output.temporal_descriptor_pixel_count_dense_stage_baseline =
        temporal_descriptor_pixel_count_dense_stage_baseline;
    output.temporal_descriptor_frame_terms_actual =
        temporal_descriptor_frame_terms_actual;
    output.temporal_descriptor_frame_terms_dense_stage_baseline =
        temporal_descriptor_frame_terms_dense_stage_baseline;
    output.temporal_descriptor_frame_terms_single_final_baseline =
        temporal_descriptor_frame_terms_single_final_baseline;
    if (easy_to_hard_.prefix_compatible_temporal_wavelet) {
        output.temporal_prefix_raw_frame_pixel_terms_actual =
            prefix_img_state.raw_frame_pixel_terms_accumulated +
            prefix_ref_state.raw_frame_pixel_terms_accumulated;
        output.temporal_prefix_raw_frame_pixel_terms_final_baseline =
            static_cast<std::uint64_t>(2 * h_ * w_) *
            static_cast<std::uint64_t>(ch_);
        output.temporal_prefix_weighted_coefficient_terms_actual =
            prefix_img_state.weighted_coefficient_terms_accumulated +
            prefix_ref_state.weighted_coefficient_terms_accumulated;
    }
    output.temporal_stage_frame_counts = std::move(executed_temporal_stages);
    output.temporal_stage_active_pixel_counts =
        std::move(temporal_stage_active_pixel_counts);
    output.temporal_stage_accepted_pixel_counts =
        std::move(temporal_stage_accepted_pixel_counts);
    output.temporal_stage_sample_descriptor_pixel_counts =
        std::move(temporal_stage_sample_descriptor_pixel_counts);
    output.temporal_stage_reference_descriptor_pixel_counts =
        std::move(temporal_stage_reference_descriptor_pixel_counts);
    if (raw_rerank_enabled) {
        output.umpa_proposal_y = std::move(umpa_output.proposal_y);
        output.umpa_proposal_x = std::move(umpa_output.proposal_x);
        output.umpa_relative_offset_y = std::move(umpa_output.relative_offset_y);
        output.umpa_relative_offset_x = std::move(umpa_output.relative_offset_x);
        output.umpa_best_cost = std::move(umpa_output.best_cost);
        output.umpa_second_best_cost = std::move(umpa_output.second_best_cost);
        output.umpa_cost_margin = std::move(umpa_output.cost_margin);
        output.umpa_transmission = std::move(umpa_output.transmission);
        output.umpa_visibility = std::move(umpa_output.visibility);
        output.umpa_delta = std::move(umpa_output.delta);
        output.umpa_condition = std::move(umpa_output.condition);
        output.umpa_numerical_valid = std::move(umpa_output.numerical_valid);
        output.umpa_physical_valid = std::move(umpa_output.physical_valid);
        output.umpa_local_boundary_hit =
            std::move(umpa_output.local_boundary_hit);
        output.umpa_candidates_evaluated =
            std::move(umpa_output.candidates_evaluated);
        output.umpa_refine_time_s = umpa_refine_time_s;
        output.umpa_raw_candidate_count = umpa_output.raw_candidate_count;
        output.umpa_raw_observation_count = umpa_output.raw_observation_count;
        output.umpa_numerical_valid_pixel_count =
            umpa_output.numerical_valid_pixel_count;
        output.umpa_physical_valid_pixel_count =
            umpa_output.physical_valid_pixel_count;
        output.umpa_retained_raw_bytes = umpa_retained_raw_bytes;
    }
    return output;
}

SolverOutput WSVT::run(const std::string& result_path, bool cleansave, int h5_deflate) {
    const std::string started_at_utc = now_iso8601_utc();
    SolverOutput out = solver();
    if (!result_path.empty()) {
        const auto t_write_t0 = std::chrono::steady_clock::now();
        std::vector<H5ItemF32> items;
        if (cleansave) {
            items.push_back(H5ItemF32{"displace_x", NdArrayF32{{out.h, out.w}, out.displace_x}});
            items.push_back(H5ItemF32{"displace_y", NdArrayF32{{out.h, out.w}, out.displace_y}});
            items.push_back(H5ItemF32{"transmission_image", NdArrayF32{{out.transmission_h, out.transmission_w}, out.transmission}});
            if (!out.darkfield.empty()) {
                items.push_back(H5ItemF32{"darkfield", NdArrayF32{{out.transmission_h, out.transmission_w}, out.darkfield}});
            }
            items.push_back(H5ItemF32{"darkfield_nd", NdArrayF32{{out.h, out.w}, out.darkfield_nd}});
            items.push_back(H5ItemF32{"match_score_neg_ssd", NdArrayF32{{out.h, out.w}, out.darkfield_nd}});
            items.push_back(H5ItemF32{"second_best_score_neg_ssd", NdArrayF32{{out.h, out.w}, out.second_best_score_neg_ssd}});
            items.push_back(H5ItemF32{"score_margin", NdArrayF32{{out.h, out.w}, out.score_margin}});
            items.push_back(H5ItemF32{"peak_hessian_det", NdArrayF32{{out.h, out.w}, out.peak_hessian_det}});
            items.push_back(H5ItemF32{"normalized_peak_curvature", NdArrayF32{{out.h, out.w}, out.normalized_peak_curvature}});
            items.push_back(H5ItemF32{"interlevel_displacement_delta", NdArrayF32{{out.h, out.w}, out.interlevel_displacement_delta}});
            items.push_back(H5ItemF32{"search_boundary_hit", NdArrayF32{{out.h, out.w}, out.search_boundary_hit}});
            items.push_back(H5ItemF32{"search_geometry_valid", NdArrayF32{{out.h, out.w}, out.search_geometry_valid}});
            if (easy_to_hard_.confidence_aware_refinement) {
                items.push_back(H5ItemF32{"temporal_speckle_contrast", NdArrayF32{{out.h, out.w}, out.temporal_speckle_contrast}});
                items.push_back(H5ItemF32{"effective_search_half_window", NdArrayF32{{out.h, out.w}, out.effective_search_half_window}});
                items.push_back(H5ItemF32{"temporal_frames_used", NdArrayF32{{out.h, out.w}, out.temporal_frames_used}});
                items.push_back(H5ItemF32{"confidence_path", NdArrayF32{{out.h, out.w}, out.confidence_path}});
            }
        } else {
            items.push_back(H5ItemF32{"displace_x", NdArrayF32{{out.h, out.w}, out.displace_x}});
            items.push_back(H5ItemF32{"displace_y", NdArrayF32{{out.h, out.w}, out.displace_y}});
            items.push_back(H5ItemF32{"DPC_x", NdArrayF32{{out.h, out.w}, out.dpc_x}});
            items.push_back(H5ItemF32{"DPC_y", NdArrayF32{{out.h, out.w}, out.dpc_y}});
            items.push_back(H5ItemF32{"phase", NdArrayF32{{out.h, out.w}, out.phase}});
            items.push_back(H5ItemF32{"transmission_image", NdArrayF32{{out.transmission_h, out.transmission_w}, out.transmission}});
            if (!out.darkfield.empty()) {
                items.push_back(H5ItemF32{"darkfield", NdArrayF32{{out.transmission_h, out.transmission_w}, out.darkfield}});
            }
            items.push_back(H5ItemF32{"darkfield_nd", NdArrayF32{{out.h, out.w}, out.darkfield_nd}});
            items.push_back(H5ItemF32{"match_score_neg_ssd", NdArrayF32{{out.h, out.w}, out.darkfield_nd}});
            items.push_back(H5ItemF32{"second_best_score_neg_ssd", NdArrayF32{{out.h, out.w}, out.second_best_score_neg_ssd}});
            items.push_back(H5ItemF32{"score_margin", NdArrayF32{{out.h, out.w}, out.score_margin}});
            items.push_back(H5ItemF32{"peak_hessian_det", NdArrayF32{{out.h, out.w}, out.peak_hessian_det}});
            items.push_back(H5ItemF32{"normalized_peak_curvature", NdArrayF32{{out.h, out.w}, out.normalized_peak_curvature}});
            items.push_back(H5ItemF32{"interlevel_displacement_delta", NdArrayF32{{out.h, out.w}, out.interlevel_displacement_delta}});
            items.push_back(H5ItemF32{"search_boundary_hit", NdArrayF32{{out.h, out.w}, out.search_boundary_hit}});
            items.push_back(H5ItemF32{"search_geometry_valid", NdArrayF32{{out.h, out.w}, out.search_geometry_valid}});
            if (easy_to_hard_.confidence_aware_refinement) {
                items.push_back(H5ItemF32{"temporal_speckle_contrast", NdArrayF32{{out.h, out.w}, out.temporal_speckle_contrast}});
                items.push_back(H5ItemF32{"effective_search_half_window", NdArrayF32{{out.h, out.w}, out.effective_search_half_window}});
                items.push_back(H5ItemF32{"temporal_frames_used", NdArrayF32{{out.h, out.w}, out.temporal_frames_used}});
                items.push_back(H5ItemF32{"confidence_path", NdArrayF32{{out.h, out.w}, out.confidence_path}});
            }
        }
        if (fixed_set_transport_) {
            items.push_back(H5ItemF32{
                "set_transport_center_y",
                NdArrayF32{{4U, out.h, out.w}, out.set_transport_center_y}});
            items.push_back(H5ItemF32{
                "set_transport_center_x",
                NdArrayF32{{4U, out.h, out.w}, out.set_transport_center_x}});
            items.push_back(H5ItemF32{
                "set_transport_integer_winner_y",
                NdArrayF32{{out.h, out.w},
                           out.set_transport_integer_winner_y}});
            items.push_back(H5ItemF32{
                "set_transport_integer_winner_x",
                NdArrayF32{{out.h, out.w},
                           out.set_transport_integer_winner_x}});
            items.push_back(H5ItemF32{
                "set_transport_representative_y",
                NdArrayF32{{4U, out.h, out.w},
                           out.set_transport_representative_y}});
            items.push_back(H5ItemF32{
                "set_transport_representative_x",
                NdArrayF32{{4U, out.h, out.w},
                           out.set_transport_representative_x}});
            items.push_back(H5ItemF32{
                "set_transport_candidates_evaluated",
                NdArrayF32{{out.h, out.w},
                           out.set_transport_candidates_evaluated}});
        }
        if (wavelet_guided_umpa_.enabled || set_transport_raw_rerank_.enabled) {
            items.push_back(H5ItemF32{"umpa_proposal_x", NdArrayF32{{out.h, out.w}, out.umpa_proposal_x}});
            items.push_back(H5ItemF32{"umpa_proposal_y", NdArrayF32{{out.h, out.w}, out.umpa_proposal_y}});
            items.push_back(H5ItemF32{"umpa_relative_offset_x", NdArrayF32{{out.h, out.w}, out.umpa_relative_offset_x}});
            items.push_back(H5ItemF32{"umpa_relative_offset_y", NdArrayF32{{out.h, out.w}, out.umpa_relative_offset_y}});
            items.push_back(H5ItemF32{"umpa_best_cost", NdArrayF32{{out.h, out.w}, out.umpa_best_cost}});
            items.push_back(H5ItemF32{"umpa_second_best_cost", NdArrayF32{{out.h, out.w}, out.umpa_second_best_cost}});
            items.push_back(H5ItemF32{"umpa_cost_margin", NdArrayF32{{out.h, out.w}, out.umpa_cost_margin}});
            items.push_back(H5ItemF32{"umpa_transmission", NdArrayF32{{out.h, out.w}, out.umpa_transmission}});
            items.push_back(H5ItemF32{"umpa_visibility", NdArrayF32{{out.h, out.w}, out.umpa_visibility}});
            items.push_back(H5ItemF32{"umpa_delta", NdArrayF32{{out.h, out.w}, out.umpa_delta}});
            items.push_back(H5ItemF32{"umpa_condition", NdArrayF32{{out.h, out.w}, out.umpa_condition}});
            items.push_back(H5ItemF32{"umpa_numerical_valid", NdArrayF32{{out.h, out.w}, out.umpa_numerical_valid}});
            items.push_back(H5ItemF32{"umpa_physical_valid", NdArrayF32{{out.h, out.w}, out.umpa_physical_valid}});
            items.push_back(H5ItemF32{"umpa_local_boundary_hit", NdArrayF32{{out.h, out.w}, out.umpa_local_boundary_hit}});
            items.push_back(H5ItemF32{"umpa_candidates_evaluated", NdArrayF32{{out.h, out.w}, out.umpa_candidates_evaluated}});
        }
        write_h5(result_path, "WSVT_result", items, h5_deflate);

        const double phase_rms = stddev_2d(out.phase);
        const double phase_pv = pv_2d(out.phase);
        const auto t_write_t1 = std::chrono::steady_clock::now();
        out.result_write_time_s = std::chrono::duration<double>(t_write_t1 - t_write_t0).count();

        JsonObject parameter_dict;
        parameter_dict["crop"] = static_cast<double>(crop_);
        parameter_dict["N_s extend"] = static_cast<double>(n_s_extend_);
        parameter_dict["half_window"] = static_cast<double>(cal_half_window_);
        parameter_dict["n_template"] = static_cast<double>(n_template_);
        parameter_dict["fixed_spatial_support"] =
            std::string(fixed_spatial_support_name(fixed_spatial_support_));
        parameter_dict["fixed_spatial_reuse"]=fixed_spatial_reuse_;
        parameter_dict["spatial_reuse_tile_width"]=static_cast<double>(spatial_reuse_tile_width_);
        parameter_dict["spatial_cost_radius"] =
            fixed_spatial_support_ == FixedSpatialSupport::Point ? 0.0 : 1.0;
        parameter_dict["spatial_cost_boundary"] = "zero_extension";
        parameter_dict["subpixel_policy"] = constrained_subpixel_ ? "box_finest_v2" :
            (guarded_subpixel_ ? "guarded_finest_v1" : "legacy");
        parameter_dict["window_policy"] = "manual_fixed";
        parameter_dict["semantics_profile"] = std::string(kPythonReferenceSemantics);
        parameter_dict["search_profile"] = std::string(search_profile_name(
            search_early_abandon_, search_two_pass_, search_guard_cache_));
        parameter_dict["algorithm_profile"] =
            (fixed_set_transport_ && set_transport_raw_rerank_.enabled)
            ? (set_transport_raw_rerank_.representatives_only
                ? "pyramid_wsvt_set4_raw_rep4_v1"
                : "pyramid_wsvt_set4_raw_rerank_v1")
            : fixed_set_transport_
            ? "pyramid_wsvt_fixed_set_transport_v1"
            : (wavelet_guided_umpa_.enabled
            ? "wavelet_guided_umpa_h1_n1"
            : (easy_to_hard_.adaptive_frames
            ? (easy_to_hard_.prefix_compatible_temporal_wavelet
                ? (easy_to_hard_.temporal_probe_first_stage_only
                    ? "pyramid_easy_to_hard_b2_v3_prefix_probe"
                    : "pyramid_easy_to_hard_b2_v3_prefix_basis")
                : (easy_to_hard_.lazy_temporal_descriptors
                    ? "pyramid_easy_to_hard_b2_v2_lazy"
                    : "pyramid_easy_to_hard_b2_v1"))
            : (easy_to_hard_.confidence_aware_refinement
                ? "pyramid_easy_to_hard_b1_v1"
                : "pyramid_wsvt_b0")));
        parameter_dict["fixed_set_transport"] = fixed_set_transport_;
        parameter_dict["set_transport_hypothesis_width"] = 4.0;
        parameter_dict["set_transport_parent_mapping"] =
            "endpoint-aligned nearest parent with round-half-to-even";
        parameter_dict["set_transport_basin_definition"] =
            "deterministic 8-neighbour steepest descent; roots ordered by exact descriptor SSD then linear index";
        parameter_dict["set_transport_subpixel_definition"] =
            "existing float32 Hessian fit inside the lowest-index transported domain containing the global union winner";
        parameter_dict["wavelet_guided_umpa"] =
            wavelet_guided_umpa_.enabled;
        parameter_dict["set_transport_raw_rerank"] =
            set_transport_raw_rerank_.enabled;
        parameter_dict["set_transport_raw_representatives"] =
            set_transport_raw_rerank_.enabled &&
            set_transport_raw_rerank_.representatives_only;
        parameter_dict["raw_rerank_objective"] = set_transport_raw_rerank_.enabled
            ? set_transport_raw_objective_name(set_transport_raw_rerank_.objective)
            : (wavelet_guided_umpa_.enabled ? "ModelDF" : "disabled");
        parameter_dict["umpa_model"] = set_transport_raw_rerank_.enabled
            ? set_transport_raw_objective_name(set_transport_raw_rerank_.objective)
            : "ModelDF";
        parameter_dict["umpa_coordinate_assignment"] = "sample";
        parameter_dict["umpa_local_half_window"] =
            static_cast<double>(set_transport_raw_rerank_.enabled
                ? set_transport_raw_rerank_.domain_half_window
                : wavelet_guided_umpa_.local_half_window);
        parameter_dict["umpa_analysis_radius"] =
            static_cast<double>(set_transport_raw_rerank_.enabled
                ? set_transport_raw_rerank_.analysis_radius
                : wavelet_guided_umpa_.analysis_radius);
        parameter_dict["umpa_weighting"] = "fixed_normalized_hamming";
        parameter_dict["umpa_subpixel"] = false;
        parameter_dict["umpa_reference_self"] = false;
        parameter_dict["confidence_aware_refinement"] =
            easy_to_hard_.confidence_aware_refinement;
        parameter_dict["adaptive_frames"] = easy_to_hard_.adaptive_frames;
        parameter_dict["lazy_temporal_descriptors"] =
            easy_to_hard_.lazy_temporal_descriptors;
        parameter_dict["prefix_compatible_temporal_wavelet"] =
            easy_to_hard_.prefix_compatible_temporal_wavelet;
        parameter_dict["temporal_probe_first_stage_only"] =
            easy_to_hard_.temporal_probe_first_stage_only;
        parameter_dict["temporal_prefix_half_window"] =
            static_cast<double>(easy_to_hard_.temporal_prefix_half_window);
        parameter_dict["easy_half_window"] =
            static_cast<double>(easy_to_hard_.easy_half_window);
        parameter_dict["confidence_margin_min"] =
            static_cast<double>(easy_to_hard_.score_margin_min);
        parameter_dict["confidence_curvature_min"] =
            static_cast<double>(easy_to_hard_.normalized_curvature_min);
        parameter_dict["confidence_contrast_min"] =
            static_cast<double>(easy_to_hard_.temporal_speckle_contrast_min);
        parameter_dict["confidence_interlevel_delta_max_px"] =
            static_cast<double>(easy_to_hard_.interlevel_delta_max_px);
        parameter_dict["confidence_temporal_delta_max_px"] =
            static_cast<double>(easy_to_hard_.temporal_delta_max_px);
        JsonArray frame_stages;
        for (const std::size_t value : out.temporal_stage_frame_counts) {
            frame_stages.emplace_back(static_cast<double>(value));
        }
        parameter_dict["temporal_frame_stages_executed"] = frame_stages;
        parameter_dict["search_early_abandon"] = search_early_abandon_;
        parameter_dict["search_two_pass"] = search_two_pass_;
        parameter_dict["search_guard_cache"] = search_guard_cache_;
        parameter_dict["search_top_k"] = static_cast<double>(search_top_k_);
        parameter_dict["search_block_size"] = static_cast<double>(search_block_size_);
        parameter_dict["search_prefix_size"] = static_cast<double>(search_prefix_size_);
        parameter_dict["input_axis_order"] = "CHW";
        parameter_dict["descriptor_axis_order"] = "HWD";
        parameter_dict["output_dtype"] = "float32";
        parameter_dict["darkfield_nd_semantics"] = "legacy alias of match_score_neg_ssd; not a physical dark-field observable";
        parameter_dict["score_margin_definition"] = "(best_score-second_best_score)/max(abs(best_score),float32_epsilon)";
        parameter_dict["normalized_peak_curvature_definition"] = "sqrt(max(Hxx*Hyy-Hxy^2,0))/max(abs(best_score),float32_epsilon)";
        parameter_dict["interlevel_displacement_delta_definition"] = "Euclidean norm of the final-level local displacement correction before adding the pyramid prediction";
        parameter_dict["temporal_speckle_contrast_definition"] = "min(sample_CV,reference_CV) over the accepted raw frame prefix at each pixel";
        parameter_dict["confidence_path_codes"] = "0=inactive/B0,1=full-search confident,2=easy-search accepted,3=easy-search fallback then confident,4=ambiguous/final fallback";
        parameter_dict["search_geometry_valid_definition"] = "finite best/second scores, non-boundary integer peak, finite positive Hessian determinant; no score-margin threshold";
        parameter_dict["phase_convention"] = "-frankot_chellappa(DPC_x,DPC_y)*p_x*2pi/wavelength";
        parameter_dict["valid_roi_margin_input_px"] =
            static_cast<double>(n_template_ + cal_half_window_);
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
        parameter_dict["pyramid_level"] = static_cast<double>(pyramid_level_);
        parameter_dict["n_iter"] = static_cast<double>(n_iter_);
        parameter_dict["time_cost"] = out.time_cost_s;
        parameter_dict["pyramid_time"] = out.pyramid_time_s;
        parameter_dict["template_window_time"] = out.template_window_time_s;
        parameter_dict["wavelet_time"] = out.wavelet_time_s;
        parameter_dict["displace_time"] = out.displace_time_s;
        parameter_dict["postprocess_time"] = out.postprocess_time_s;
        parameter_dict["post_crop_sign_time"] = out.post_crop_sign_time_s;
        parameter_dict["post_dpc_conv_time"] = out.post_dpc_conv_time_s;
        parameter_dict["post_phase_recovery_time"] = out.post_phase_recovery_time_s;
        parameter_dict["darkfield_time"] = out.darkfield_time_s;
        parameter_dict["umpa_refine_time"] = out.umpa_refine_time_s;
        parameter_dict["result_write_time"] = out.result_write_time_s;
        parameter_dict["search_candidate_count"] =
            static_cast<double>(out.search_candidate_count);
        parameter_dict["search_abandoned_candidate_count"] =
            static_cast<double>(out.search_abandoned_candidate_count);
        parameter_dict["search_distance_terms_evaluated"] =
            static_cast<double>(out.search_distance_terms_evaluated);
        parameter_dict["search_distance_terms_possible"] =
            static_cast<double>(out.search_distance_terms_possible);
        parameter_dict["search_refine_terms_evaluated"] =
            static_cast<double>(out.search_refine_terms_evaluated);
        parameter_dict["search_prefix_terms_evaluated"] =
            static_cast<double>(out.search_prefix_terms_evaluated);
        parameter_dict["search_full_candidate_count"] =
            static_cast<double>(out.search_full_candidate_count);
        parameter_dict["search_guard_check_count"] =
            static_cast<double>(out.search_guard_check_count);
        parameter_dict["search_guard_refresh_count"] =
            static_cast<double>(out.search_guard_refresh_count);
        parameter_dict["search_dense_baseline_candidate_count"] =
            static_cast<double>(out.search_dense_baseline_candidate_count);
        parameter_dict["set_transport_unique_candidate_count"] =
            static_cast<double>(out.set_transport_unique_candidate_count);
        parameter_dict["set_transport_nominal_candidate_count"] =
            static_cast<double>(out.set_transport_nominal_candidate_count);
        parameter_dict["set_transport_duplicate_candidate_count"] =
            static_cast<double>(out.set_transport_duplicate_candidate_count);
        parameter_dict["umpa_raw_candidate_count"] =
            static_cast<double>(out.umpa_raw_candidate_count);
        parameter_dict["umpa_raw_observation_count"] =
            static_cast<double>(out.umpa_raw_observation_count);
        parameter_dict["umpa_numerical_valid_pixel_count"] =
            static_cast<double>(out.umpa_numerical_valid_pixel_count);
        parameter_dict["umpa_physical_valid_pixel_count"] =
            static_cast<double>(out.umpa_physical_valid_pixel_count);
        parameter_dict["umpa_retained_raw_bytes"] =
            static_cast<double>(out.umpa_retained_raw_bytes);
        parameter_dict["easy_path_eligible_pixel_count"] =
            static_cast<double>(out.easy_path_eligible_pixel_count);
        parameter_dict["easy_path_accepted_pixel_count"] =
            static_cast<double>(out.easy_path_accepted_pixel_count);
        parameter_dict["easy_path_fallback_pixel_count"] =
            static_cast<double>(out.easy_path_fallback_pixel_count);
        parameter_dict["inactive_pixel_count"] =
            static_cast<double>(out.inactive_pixel_count);
        parameter_dict["temporal_frame_candidate_terms_actual"] =
            static_cast<double>(out.temporal_frame_candidate_terms_actual);
        parameter_dict["temporal_frame_candidate_terms_dense_baseline"] =
            static_cast<double>(out.temporal_frame_candidate_terms_dense_baseline);
        parameter_dict["temporal_descriptor_pixel_count_actual"] =
            static_cast<double>(out.temporal_descriptor_pixel_count_actual);
        parameter_dict["temporal_descriptor_pixel_count_dense_stage_baseline"] =
            static_cast<double>(
                out.temporal_descriptor_pixel_count_dense_stage_baseline);
        parameter_dict["temporal_descriptor_frame_terms_actual"] =
            static_cast<double>(out.temporal_descriptor_frame_terms_actual);
        parameter_dict["temporal_descriptor_frame_terms_dense_stage_baseline"] =
            static_cast<double>(
                out.temporal_descriptor_frame_terms_dense_stage_baseline);
        parameter_dict["temporal_descriptor_frame_terms_single_final_baseline"] =
            static_cast<double>(
                out.temporal_descriptor_frame_terms_single_final_baseline);
        parameter_dict["temporal_prefix_raw_frame_pixel_terms_actual"] =
            static_cast<double>(
                out.temporal_prefix_raw_frame_pixel_terms_actual);
        parameter_dict["temporal_prefix_raw_frame_pixel_terms_final_baseline"] =
            static_cast<double>(
                out.temporal_prefix_raw_frame_pixel_terms_final_baseline);
        parameter_dict["temporal_prefix_weighted_coefficient_terms_actual"] =
            static_cast<double>(
                out.temporal_prefix_weighted_coefficient_terms_actual);
        JsonArray stage_active_counts;
        for (const std::uint64_t value : out.temporal_stage_active_pixel_counts) {
            stage_active_counts.emplace_back(static_cast<double>(value));
        }
        parameter_dict["temporal_stage_active_pixel_counts"] = stage_active_counts;
        JsonArray stage_accepted_counts;
        for (const std::uint64_t value : out.temporal_stage_accepted_pixel_counts) {
            stage_accepted_counts.emplace_back(static_cast<double>(value));
        }
        parameter_dict["temporal_stage_accepted_pixel_counts"] = stage_accepted_counts;
        JsonArray stage_sample_descriptor_counts;
        for (const std::uint64_t value :
             out.temporal_stage_sample_descriptor_pixel_counts) {
            stage_sample_descriptor_counts.emplace_back(
                static_cast<double>(value));
        }
        parameter_dict["temporal_stage_sample_descriptor_pixel_counts"] =
            stage_sample_descriptor_counts;
        JsonArray stage_reference_descriptor_counts;
        for (const std::uint64_t value :
             out.temporal_stage_reference_descriptor_pixel_counts) {
            stage_reference_descriptor_counts.emplace_back(
                static_cast<double>(value));
        }
        parameter_dict["temporal_stage_reference_descriptor_pixel_counts"] =
            stage_reference_descriptor_counts;
        parameter_dict["h5_deflate"] = static_cast<double>(h5_deflate);
        parameter_dict["use_wavelet"] = use_wavelet_;
        parameter_dict["use_GPU"] = use_gpu_;
        parameter_dict["calc_darkfield"] = calc_darkfield_;
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
        events["template_window_time_s"] = out.template_window_time_s;
        events["wavelet_time_s"] = out.wavelet_time_s;
        events["displace_time_s"] = out.displace_time_s;
        events["postprocess_time_s"] = out.postprocess_time_s;
        events["post_crop_sign_time_s"] = out.post_crop_sign_time_s;
        events["post_dpc_conv_time_s"] = out.post_dpc_conv_time_s;
        events["post_phase_recovery_time_s"] = out.post_phase_recovery_time_s;
        events["darkfield_time_s"] = out.darkfield_time_s;
        events["umpa_refine_time_s"] = out.umpa_refine_time_s;
        events["result_write_time_s"] = out.result_write_time_s;
        events["h5_deflate"] = static_cast<double>(h5_deflate);
        events["calc_darkfield"] = calc_darkfield_;
        events["search_profile"] = std::string(search_profile_name(
            search_early_abandon_, search_two_pass_, search_guard_cache_));
        events["algorithm_profile"] =
            (fixed_set_transport_ && set_transport_raw_rerank_.enabled)
            ? (set_transport_raw_rerank_.representatives_only
                ? "pyramid_wsvt_set4_raw_rep4_v1"
                : "pyramid_wsvt_set4_raw_rerank_v1")
            : fixed_set_transport_
            ? "pyramid_wsvt_fixed_set_transport_v1"
            : (wavelet_guided_umpa_.enabled
            ? "wavelet_guided_umpa_h1_n1"
            : (easy_to_hard_.adaptive_frames
            ? (easy_to_hard_.prefix_compatible_temporal_wavelet
                ? (easy_to_hard_.temporal_probe_first_stage_only
                    ? "pyramid_easy_to_hard_b2_v3_prefix_probe"
                    : "pyramid_easy_to_hard_b2_v3_prefix_basis")
                : (easy_to_hard_.lazy_temporal_descriptors
                    ? "pyramid_easy_to_hard_b2_v2_lazy"
                    : "pyramid_easy_to_hard_b2_v1"))
            : (easy_to_hard_.confidence_aware_refinement
                ? "pyramid_easy_to_hard_b1_v1"
                : "pyramid_wsvt_b0")));
        events["fixed_set_transport"] = fixed_set_transport_;
        events["set_transport_hypothesis_width"] = 4.0;
        events["wavelet_guided_umpa"] = wavelet_guided_umpa_.enabled;
        events["set_transport_raw_rerank"] =
            set_transport_raw_rerank_.enabled;
        events["set_transport_raw_representatives"] =
            set_transport_raw_rerank_.enabled &&
            set_transport_raw_rerank_.representatives_only;
        events["raw_rerank_objective"] = set_transport_raw_rerank_.enabled
            ? set_transport_raw_objective_name(set_transport_raw_rerank_.objective)
            : (wavelet_guided_umpa_.enabled ? "ModelDF" : "disabled");
        events["umpa_model"] = set_transport_raw_rerank_.enabled
            ? set_transport_raw_objective_name(set_transport_raw_rerank_.objective)
            : "ModelDF";
        events["umpa_coordinate_assignment"] = "sample";
        events["umpa_local_half_window"] =
            static_cast<double>(set_transport_raw_rerank_.enabled
                ? set_transport_raw_rerank_.domain_half_window
                : wavelet_guided_umpa_.local_half_window);
        events["umpa_analysis_radius"] =
            static_cast<double>(set_transport_raw_rerank_.enabled
                ? set_transport_raw_rerank_.analysis_radius
                : wavelet_guided_umpa_.analysis_radius);
        events["confidence_aware_refinement"] =
            easy_to_hard_.confidence_aware_refinement;
        events["adaptive_frames"] = easy_to_hard_.adaptive_frames;
        events["lazy_temporal_descriptors"] =
            easy_to_hard_.lazy_temporal_descriptors;
        events["prefix_compatible_temporal_wavelet"] =
            easy_to_hard_.prefix_compatible_temporal_wavelet;
        events["temporal_probe_first_stage_only"] =
            easy_to_hard_.temporal_probe_first_stage_only;
        events["temporal_prefix_half_window"] =
            static_cast<double>(easy_to_hard_.temporal_prefix_half_window);
        events["search_early_abandon"] = search_early_abandon_;
        events["search_two_pass"] = search_two_pass_;
        events["search_guard_cache"] = search_guard_cache_;
        events["search_top_k"] = static_cast<double>(search_top_k_);
        events["search_block_size"] = static_cast<double>(search_block_size_);
        events["search_prefix_size"] = static_cast<double>(search_prefix_size_);
        events["search_candidate_count"] = static_cast<double>(out.search_candidate_count);
        events["search_abandoned_candidate_count"] =
            static_cast<double>(out.search_abandoned_candidate_count);
        events["search_distance_terms_evaluated"] =
            static_cast<double>(out.search_distance_terms_evaluated);
        events["search_distance_terms_possible"] =
            static_cast<double>(out.search_distance_terms_possible);
        events["search_refine_terms_evaluated"] =
            static_cast<double>(out.search_refine_terms_evaluated);
        events["search_prefix_terms_evaluated"] =
            static_cast<double>(out.search_prefix_terms_evaluated);
        events["search_full_candidate_count"] =
            static_cast<double>(out.search_full_candidate_count);
        events["search_guard_check_count"] =
            static_cast<double>(out.search_guard_check_count);
        events["search_guard_refresh_count"] =
            static_cast<double>(out.search_guard_refresh_count);
        events["search_dense_baseline_candidate_count"] =
            static_cast<double>(out.search_dense_baseline_candidate_count);
        events["set_transport_unique_candidate_count"] =
            static_cast<double>(out.set_transport_unique_candidate_count);
        events["set_transport_nominal_candidate_count"] =
            static_cast<double>(out.set_transport_nominal_candidate_count);
        events["set_transport_duplicate_candidate_count"] =
            static_cast<double>(out.set_transport_duplicate_candidate_count);
        events["umpa_raw_candidate_count"] =
            static_cast<double>(out.umpa_raw_candidate_count);
        events["umpa_raw_observation_count"] =
            static_cast<double>(out.umpa_raw_observation_count);
        events["umpa_numerical_valid_pixel_count"] =
            static_cast<double>(out.umpa_numerical_valid_pixel_count);
        events["umpa_physical_valid_pixel_count"] =
            static_cast<double>(out.umpa_physical_valid_pixel_count);
        events["umpa_retained_raw_bytes"] =
            static_cast<double>(out.umpa_retained_raw_bytes);
        events["easy_path_eligible_pixel_count"] =
            static_cast<double>(out.easy_path_eligible_pixel_count);
        events["easy_path_accepted_pixel_count"] =
            static_cast<double>(out.easy_path_accepted_pixel_count);
        events["easy_path_fallback_pixel_count"] =
            static_cast<double>(out.easy_path_fallback_pixel_count);
        events["inactive_pixel_count"] =
            static_cast<double>(out.inactive_pixel_count);
        events["temporal_frame_candidate_terms_actual"] =
            static_cast<double>(out.temporal_frame_candidate_terms_actual);
        events["temporal_frame_candidate_terms_dense_baseline"] =
            static_cast<double>(out.temporal_frame_candidate_terms_dense_baseline);
        events["temporal_descriptor_pixel_count_actual"] =
            static_cast<double>(out.temporal_descriptor_pixel_count_actual);
        events["temporal_descriptor_pixel_count_dense_stage_baseline"] =
            static_cast<double>(
                out.temporal_descriptor_pixel_count_dense_stage_baseline);
        events["temporal_descriptor_frame_terms_actual"] =
            static_cast<double>(out.temporal_descriptor_frame_terms_actual);
        events["temporal_descriptor_frame_terms_dense_stage_baseline"] =
            static_cast<double>(
                out.temporal_descriptor_frame_terms_dense_stage_baseline);
        events["temporal_descriptor_frame_terms_single_final_baseline"] =
            static_cast<double>(
                out.temporal_descriptor_frame_terms_single_final_baseline);
        events["temporal_prefix_raw_frame_pixel_terms_actual"] =
            static_cast<double>(
                out.temporal_prefix_raw_frame_pixel_terms_actual);
        events["temporal_prefix_raw_frame_pixel_terms_final_baseline"] =
            static_cast<double>(
                out.temporal_prefix_raw_frame_pixel_terms_final_baseline);
        events["temporal_prefix_weighted_coefficient_terms_actual"] =
            static_cast<double>(
                out.temporal_prefix_weighted_coefficient_terms_actual);
        JsonArray event_frame_stages;
        for (const std::size_t value : out.temporal_stage_frame_counts) {
            event_frame_stages.emplace_back(static_cast<double>(value));
        }
        events["temporal_frame_stages_executed"] = event_frame_stages;
        JsonArray event_active_counts;
        for (const std::uint64_t value : out.temporal_stage_active_pixel_counts) {
            event_active_counts.emplace_back(static_cast<double>(value));
        }
        events["temporal_stage_active_pixel_counts"] = event_active_counts;
        JsonArray event_accepted_counts;
        for (const std::uint64_t value : out.temporal_stage_accepted_pixel_counts) {
            event_accepted_counts.emplace_back(static_cast<double>(value));
        }
        events["temporal_stage_accepted_pixel_counts"] = event_accepted_counts;
        JsonArray event_sample_descriptor_counts;
        for (const std::uint64_t value :
             out.temporal_stage_sample_descriptor_pixel_counts) {
            event_sample_descriptor_counts.emplace_back(
                static_cast<double>(value));
        }
        events["temporal_stage_sample_descriptor_pixel_counts"] =
            event_sample_descriptor_counts;
        JsonArray event_reference_descriptor_counts;
        for (const std::uint64_t value :
             out.temporal_stage_reference_descriptor_pixel_counts) {
            event_reference_descriptor_counts.emplace_back(
                static_cast<double>(value));
        }
        events["temporal_stage_reference_descriptor_pixel_counts"] =
            event_reference_descriptor_counts;
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
        JsonObject darkfield_stage;
        darkfield_stage["stage"] = "raw_darkfield";
        darkfield_stage["elapsed_s"] = out.darkfield_time_s;
        stage_records.emplace_back(darkfield_stage);
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
        write_json(result_path, "WSVT_events", events);
    }
    return out;
}

void WSVT::configure_search_topk_diagnostics(
    std::vector<std::size_t> raw_linear_pixels,
    std::size_t top_k,
    int pyramid_level) {
    if (fixed_set_transport_) {
        throw std::invalid_argument(
            "Top-K diagnostics cannot be combined with fixed set transport");
    }
    if (wavelet_guided_umpa_.enabled) {
        throw std::invalid_argument(
            "Top-K diagnostics cannot be combined with WG-UMPA H1");
    }
    if (easy_to_hard_.confidence_aware_refinement) {
        throw std::invalid_argument(
            "Top-K diagnostics cannot be combined with easy-to-hard B1/B2");
    }
    if (search_early_abandon_ || search_two_pass_) {
        throw std::invalid_argument(
            "Top-K diagnostics currently require exhaustive WSVT search");
    }
    if (pyramid_level < 0 || pyramid_level > pyramid_level_) {
        throw std::invalid_argument(
            "diagnostic pyramid level must be within the configured pyramid");
    }
    const auto diagnostic_half_windows = derived_search_half_windows(
        pyramid_level_, cal_half_window_, n_s_extend_);
    const int diagnostic_half_window = diagnostic_half_windows[
        static_cast<std::size_t>(pyramid_level)];
    const std::size_t diagnostic_window = static_cast<std::size_t>(
        2 * diagnostic_half_window + 1);
    const std::size_t diagnostic_candidate_count =
        diagnostic_window * diagnostic_window;
    if (top_k < 2 || top_k > diagnostic_candidate_count) {
        throw std::invalid_argument(
            "diagnostic rank count must be in [2, search-window candidate count]");
    }
    if (raw_linear_pixels.empty()) {
        throw std::invalid_argument("Top-K diagnostics require at least one pixel");
    }
    std::sort(raw_linear_pixels.begin(), raw_linear_pixels.end());
    if (std::adjacent_find(raw_linear_pixels.begin(), raw_linear_pixels.end()) !=
        raw_linear_pixels.end()) {
        throw std::invalid_argument("Top-K diagnostic pixels must be unique");
    }
    std::size_t diagnostic_h = h_;
    std::size_t diagnostic_w = w_;
    for (int level = 0; level < pyramid_level; ++level) {
        diagnostic_h = (diagnostic_h + 5U) / 2U;
        diagnostic_w = (diagnostic_w + 5U) / 2U;
    }
    const std::size_t pixel_count = diagnostic_h * diagnostic_w;
    if (raw_linear_pixels.back() >= pixel_count) {
        throw std::out_of_range(
            "Top-K diagnostic pixel is outside the selected pyramid grid");
    }
    diagnostic_pixels_ = std::move(raw_linear_pixels);
    diagnostic_top_k_ = top_k;
    diagnostic_pyramid_level_ = pyramid_level;
    search_topk_diagnostics_.clear();
}

const std::vector<SearchTopKDiagnostic>&
WSVT::search_topk_diagnostics() const noexcept {
    return search_topk_diagnostics_;
}

void WSVT::configure_easy_to_hard(EasyToHardConfig config) {
    if (fixed_set_transport_ &&
        (config.confidence_aware_refinement || config.adaptive_frames)) {
        throw std::invalid_argument(
            "fixed set transport cannot be combined with easy-to-hard B1/B2");
    }
    if (wavelet_guided_umpa_.enabled &&
        (config.confidence_aware_refinement || config.adaptive_frames)) {
        throw std::invalid_argument(
            "WG-UMPA H1 cannot be combined with the closed easy-to-hard profiles");
    }
    if (!config.confidence_aware_refinement && !config.adaptive_frames) {
        easy_to_hard_ = std::move(config);
        return;
    }
    if (config.adaptive_frames && !config.confidence_aware_refinement) {
        throw std::invalid_argument(
            "adaptive frames require confidence-aware refinement");
    }
    if (config.lazy_temporal_descriptors && !config.adaptive_frames) {
        throw std::invalid_argument(
            "lazy temporal descriptors require adaptive frames");
    }
    if (config.prefix_compatible_temporal_wavelet && !config.adaptive_frames) {
        throw std::invalid_argument(
            "prefix-compatible temporal wavelet requires adaptive frames");
    }
    if (config.prefix_compatible_temporal_wavelet && !use_wavelet_) {
        throw std::invalid_argument(
            "prefix-compatible temporal wavelet requires use_wavelet=true");
    }
    if (config.prefix_compatible_temporal_wavelet &&
        config.lazy_temporal_descriptors) {
        throw std::invalid_argument(
            "prefix-compatible temporal wavelet and legacy lazy descriptors are mutually exclusive");
    }
    if (config.temporal_probe_first_stage_only &&
        !config.prefix_compatible_temporal_wavelet) {
        throw std::invalid_argument(
            "temporal first-stage probe requires prefix-compatible temporal wavelet");
    }
    if (search_early_abandon_ || search_two_pass_) {
        throw std::invalid_argument(
            "easy-to-hard B1/B2 currently require exhaustive SSD search");
    }
    if (diagnostic_top_k_ != 0) {
        throw std::invalid_argument(
            "easy-to-hard B1/B2 cannot be combined with Top-K diagnostics");
    }
    if (n_template_ != 0) {
        throw std::invalid_argument(
            "B1/B2 freeze spatial support at n_template=0");
    }
    if (n_iter_ != 1) {
        throw std::invalid_argument(
            "B1/B2 research profile currently requires n_iter=1");
    }
    if (config.easy_half_window < 1 ||
        config.easy_half_window >= n_s_extend_) {
        throw std::invalid_argument(
            "easy_half_window must be in [1, n_s_extend-1]");
    }
    if (config.temporal_prefix_half_window < 1 ||
        config.temporal_prefix_half_window > n_s_extend_) {
        throw std::invalid_argument(
            "temporal_prefix_half_window must be in [1, n_s_extend]");
    }
    const auto finite_nonnegative = [](float value) {
        return std::isfinite(value) && value >= 0.0f;
    };
    if (!finite_nonnegative(config.score_margin_min) ||
        !finite_nonnegative(config.normalized_curvature_min) ||
        !finite_nonnegative(config.temporal_speckle_contrast_min) ||
        !finite_nonnegative(config.interlevel_delta_max_px) ||
        !finite_nonnegative(config.temporal_delta_max_px)) {
        throw std::invalid_argument(
            "easy-to-hard thresholds must be finite and non-negative");
    }
    if (config.adaptive_frames) {
        if (config.frame_stages.size() < 2) {
            throw std::invalid_argument(
                "adaptive frames require at least two frame stages");
        }
        if (config.frame_stages.front() < 2 ||
            config.frame_stages.back() != ch_) {
            throw std::invalid_argument(
                "frame stages must start at >=2 and end at the input frame count");
        }
        if (!std::is_sorted(
                config.frame_stages.begin(), config.frame_stages.end()) ||
            std::adjacent_find(
                config.frame_stages.begin(), config.frame_stages.end()) !=
                config.frame_stages.end()) {
            throw std::invalid_argument(
                "frame stages must be strictly increasing");
        }
        if (config.prefix_compatible_temporal_wavelet) {
            const int final_wavelet_level = dwt_max_level_db2(ch_);
            const auto add_list = wavelet_add_list_for_depth(ch_);
            int effective_return_level = -1;
            for (int pyramid = 0; pyramid <= pyramid_level_; ++pyramid) {
                const std::size_t index = static_cast<std::size_t>(pyramid);
                const int add = index < add_list.size() ? add_list[index] : 2;
                const int requested = final_wavelet_level + 1 -
                    wavelet_level_cut_ + add;
                const int effective = std::clamp(
                    requested, 1, final_wavelet_level + 1);
                if (effective_return_level < 0) {
                    effective_return_level = effective;
                } else if (effective != effective_return_level) {
                    throw std::invalid_argument(
                        "prefix-compatible temporal wavelet currently requires one retained depth across pyramid levels");
                }
            }
        }
    } else {
        config.frame_stages.clear();
    }
    easy_to_hard_ = std::move(config);
}

void WSVT::configure_fixed_spatial_support(FixedSpatialSupport support) {
    (void)fixed_spatial_support_name(support);
    if (support != FixedSpatialSupport::Point &&
        (cal_half_window_ != 16 || n_s_extend_ != 4 || pyramid_level_ != 1 ||
         n_template_ != 0 || wavelet_level_cut_ != 1 || n_iter_ != 1 ||
         use_estimate_ || use_gpu_ || !use_wavelet_ ||
         search_early_abandon_ || search_two_pass_ ||
         easy_to_hard_.confidence_aware_refinement || easy_to_hard_.adaptive_frames ||
         fixed_set_transport_ || wavelet_guided_umpa_.enabled ||
         set_transport_raw_rerank_.enabled)) {
        throw std::invalid_argument(
            "WSVT-FS-v1 requires fixed window16/residual4/pyramid1/cut1, "
            "n_template=0, one iteration, CPU wavelets, no other research/pruning");
    }
    fixed_spatial_support_ = support;
}

void WSVT::configure_guarded_subpixel(bool enabled) {
    if(enabled) {
        // Share the frozen CPU/profile exclusions without changing the support.
        const auto previous=fixed_spatial_support_;
        configure_fixed_spatial_support(FixedSpatialSupport::Hamming3);
        fixed_spatial_support_=previous;
    }
    guarded_subpixel_=enabled;
    constrained_subpixel_=false;
    if(!enabled) {
        guarded_subpixel_reasons_.clear();
        guarded_subpixel_offsets_.clear();
    }
}

void WSVT::configure_fixed_spatial_reuse(bool enabled,int tile_width) {
    if(tile_width!=8&&tile_width!=16&&tile_width!=32)
        throw std::invalid_argument("fixed reuse tile width must be 8/16/32");
    if(enabled) {
        if(fixed_spatial_support_==FixedSpatialSupport::Point)
            throw std::invalid_argument("spatial reuse requires nonpoint support");
        configure_fixed_spatial_support(fixed_spatial_support_);
    }
    fixed_spatial_reuse_=enabled;
    spatial_reuse_tile_width_=tile_width;
    spatial_reuse_stats_={};
}

void WSVT::configure_constrained_subpixel(bool enabled) {
    configure_guarded_subpixel(enabled);
    constrained_subpixel_=enabled;
}

void WSVT::configure_wavelet_guided_umpa(WaveletGuidedUmpaConfig config) {
    if (!config.enabled) {
        wavelet_guided_umpa_ = std::move(config);
        return;
    }
    if (fixed_set_transport_) {
        throw std::invalid_argument(
            "fixed set transport cannot be combined with WG-UMPA H1");
    }
    if (easy_to_hard_.confidence_aware_refinement ||
        easy_to_hard_.adaptive_frames) {
        throw std::invalid_argument(
            "WG-UMPA H1 cannot be combined with the closed easy-to-hard profiles");
    }
    if (search_early_abandon_ || search_two_pass_ || diagnostic_top_k_ != 0U) {
        throw std::invalid_argument(
            "WG-UMPA H1 currently requires exhaustive Top-1 WSVT without diagnostics");
    }
    if (cal_half_window_ != 16) {
        throw std::invalid_argument(
            "WG-UMPA v1 freezes the manual WSVT cal_half_window at 16");
    }
    if (n_template_ != 0 || n_iter_ != 1) {
        throw std::invalid_argument(
            "WG-UMPA H1 requires n_template=0 and n_iter=1");
    }
    if (config.local_half_window != 1 || config.analysis_radius != 1U) {
        throw std::invalid_argument(
            "WG-UMPA H1 freezes both displacement and analysis half-widths at 1");
    }
    if (!std::isfinite(config.relative_delta_tolerance) ||
        !std::isfinite(config.transmission_epsilon) ||
        config.relative_delta_tolerance < 0.0 ||
        config.transmission_epsilon < 0.0) {
        throw std::invalid_argument(
            "WG-UMPA H1 tolerances must be finite and non-negative");
    }
    wavelet_guided_umpa_ = std::move(config);
}

void WSVT::configure_fixed_set_transport(bool enabled) {
    if (!enabled) {
        if (set_transport_raw_rerank_.enabled) {
            throw std::invalid_argument(
                "cannot disable fixed set transport while raw rerank is enabled");
        }
        fixed_set_transport_ = false;
        return;
    }
    if (easy_to_hard_.confidence_aware_refinement ||
        easy_to_hard_.adaptive_frames || wavelet_guided_umpa_.enabled) {
        throw std::invalid_argument(
            "fixed set transport cannot be combined with adaptive or raw-UMPA profiles");
    }
    if (diagnostic_top_k_ != 0U || search_early_abandon_ || search_two_pass_) {
        throw std::invalid_argument(
            "fixed set transport v1 requires exhaustive WSVT without diagnostics");
    }
    if (cal_half_window_ != 16 || n_s_extend_ != 4 || pyramid_level_ != 1 ||
        n_template_ != 0 || n_iter_ != 1 || use_estimate_ || use_gpu_ ||
        !use_wavelet_) {
        throw std::invalid_argument(
            "fixed set transport v1 freezes half-window=16, residual=4, pyramid=1, n_template=0, n_iter=1, wavelet CPU, no estimate");
    }
    fixed_set_transport_ = true;
}

void WSVT::configure_set_transport_raw_rerank(
    SetTransportRawRerankConfig config) {
    if (!config.enabled) {
        set_transport_raw_rerank_ = std::move(config);
        return;
    }
    if (!fixed_set_transport_) {
        throw std::invalid_argument(
            "SET4 raw rerank requires fixed set transport");
    }
    if (wavelet_guided_umpa_.enabled ||
        easy_to_hard_.confidence_aware_refinement ||
        easy_to_hard_.adaptive_frames || diagnostic_top_k_ != 0U ||
        search_early_abandon_ || search_two_pass_) {
        throw std::invalid_argument(
            "SET4 raw rerank cannot be combined with H1, adaptive, pruning, or diagnostics");
    }
    if (config.domain_half_window != n_s_extend_ ||
        config.domain_half_window != 4 || config.analysis_radius != 1U) {
        throw std::invalid_argument(
            "SET4 raw rerank v1 freezes the raw domain at +/-4 and Hamming radius N=1");
    }
    if (config.representatives_only &&
        config.objective != SetTransportRawObjective::WindowedZncc) {
        throw std::invalid_argument(
            "SET4 REP4 v1 freezes the raw objective to windowed ZNCC");
    }
    if (!std::isfinite(config.relative_delta_tolerance) ||
        !std::isfinite(config.transmission_epsilon) ||
        !std::isfinite(config.variance_epsilon) ||
        config.relative_delta_tolerance < 0.0 ||
        config.transmission_epsilon < 0.0 ||
        config.variance_epsilon < 0.0) {
        throw std::invalid_argument(
            "SET4 raw rerank tolerances must be finite and non-negative");
    }
    set_transport_raw_rerank_ = std::move(config);
}

} // namespace wsvt
