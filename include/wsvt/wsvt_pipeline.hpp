#pragma once

#include "wsvt/pyramid.hpp"
#include "wsvt/wsvt_umpa_refine.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace wsvt {

/// Default-off research controls for the Pyramid WSVT easy-to-hard ablation.
/// The pyramid continues to determine displacement scale.  These controls only
/// decide how much evidence an individual pixel receives at each scale.
struct EasyToHardConfig {
    bool confidence_aware_refinement = false;
    bool adaptive_frames = false;
    bool lazy_temporal_descriptors = false;
    bool prefix_compatible_temporal_wavelet = false;
    bool temporal_probe_first_stage_only = false;
    int easy_half_window = 1;
    int temporal_prefix_half_window = 1;
    float score_margin_min = 0.02f;
    float normalized_curvature_min = 0.0f;
    float temporal_speckle_contrast_min = 0.0f;
    float interlevel_delta_max_px = 1.5f;
    float temporal_delta_max_px = 0.5f;
    std::vector<std::size_t> frame_stages;
};

struct SolverOutput {
    std::vector<float> displace_y;
    std::vector<float> displace_x;
    std::size_t h;
    std::size_t w;
    std::vector<float> dpc_y;
    std::vector<float> dpc_x;
    std::vector<float> phase;
    std::vector<float> transmission;
    std::vector<float> darkfield;
    // Legacy compatibility name. This is the best negative-SSD match score,
    // not a physical dark-field observable.
    std::vector<float> darkfield_nd;
    std::vector<float> second_best_score_neg_ssd;
    std::vector<float> score_margin;
    std::vector<float> peak_hessian_det;
    std::vector<float> normalized_peak_curvature;
    std::vector<float> interlevel_displacement_delta;
    std::vector<float> search_boundary_hit;
    std::vector<float> search_geometry_valid;
    // Easy-to-hard diagnostics. confidence_path codes are documented in the
    // result JSON; maps remain zero for the default B0 solver.
    std::vector<float> temporal_speckle_contrast;
    std::vector<float> effective_search_half_window;
    std::vector<float> temporal_frames_used;
    std::vector<float> confidence_path;
    std::size_t transmission_h;
    std::size_t transmission_w;
    double time_cost_s = 0.0;
    double pyramid_time_s = 0.0;
    double wavelet_time_s = 0.0;
    double displace_time_s = 0.0;
    double postprocess_time_s = 0.0;
    double template_window_time_s = 0.0;
    double post_crop_sign_time_s = 0.0;
    double post_dpc_conv_time_s = 0.0;
    double post_phase_recovery_time_s = 0.0;
    double darkfield_time_s = 0.0;
    double result_write_time_s = 0.0;
    double process_wall_s = 0.0;
    double load_time_s = 0.0;
    double save_time_s = 0.0;
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
    std::uint64_t temporal_frame_candidate_terms_actual = 0;
    std::uint64_t temporal_frame_candidate_terms_dense_baseline = 0;
    std::uint64_t temporal_descriptor_pixel_count_actual = 0;
    std::uint64_t temporal_descriptor_pixel_count_dense_stage_baseline = 0;
    std::uint64_t temporal_descriptor_frame_terms_actual = 0;
    std::uint64_t temporal_descriptor_frame_terms_dense_stage_baseline = 0;
    std::uint64_t temporal_descriptor_frame_terms_single_final_baseline = 0;
    std::uint64_t temporal_prefix_raw_frame_pixel_terms_actual = 0;
    std::uint64_t temporal_prefix_raw_frame_pixel_terms_final_baseline = 0;
    std::uint64_t temporal_prefix_weighted_coefficient_terms_actual = 0;
    std::vector<std::size_t> temporal_stage_frame_counts;
    std::vector<std::uint64_t> temporal_stage_active_pixel_counts;
    std::vector<std::uint64_t> temporal_stage_accepted_pixel_counts;
    std::vector<std::uint64_t> temporal_stage_sample_descriptor_pixel_counts;
    std::vector<std::uint64_t> temporal_stage_reference_descriptor_pixel_counts;
    // Default-off fixed Wavelet-guided UMPA H1 diagnostics.
    std::vector<float> umpa_proposal_y;
    std::vector<float> umpa_proposal_x;
    std::vector<float> umpa_relative_offset_y;
    std::vector<float> umpa_relative_offset_x;
    std::vector<float> umpa_best_cost;
    std::vector<float> umpa_second_best_cost;
    std::vector<float> umpa_cost_margin;
    std::vector<float> umpa_transmission;
    std::vector<float> umpa_visibility;
    std::vector<float> umpa_delta;
    std::vector<float> umpa_condition;
    std::vector<float> umpa_numerical_valid;
    std::vector<float> umpa_physical_valid;
    std::vector<float> umpa_local_boundary_hit;
    std::vector<float> umpa_candidates_evaluated;
    double umpa_refine_time_s = 0.0;
    std::uint64_t umpa_raw_candidate_count = 0;
    std::uint64_t umpa_raw_observation_count = 0;
    std::uint64_t umpa_numerical_valid_pixel_count = 0;
    std::uint64_t umpa_physical_valid_pixel_count = 0;
    std::uint64_t umpa_retained_raw_bytes = 0;
};

struct DisplaceWaveletOutput {
    std::vector<float> displace_y;
    std::vector<float> displace_x;
    std::vector<float> best_score_neg_ssd;
    std::vector<float> second_best_score_neg_ssd;
    std::vector<float> score_margin;
    std::vector<float> peak_hessian_det;
    std::vector<float> normalized_peak_curvature;
    std::vector<float> interlevel_displacement_delta;
    std::vector<float> search_boundary_hit;
    std::vector<float> search_geometry_valid;
    std::vector<float> confidence_accept;
    std::vector<float> effective_search_half_window;
    std::vector<float> confidence_path;
    std::vector<int> integer_displace_y;
    std::vector<int> integer_displace_x;
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
};

/// One exact descriptor candidate retained only for an explicitly requested
/// diagnostic pixel.  Internal search displacement addresses reference
/// (y+dy,x+dx); the saved project convention is its sign inverse.
struct SearchTopKDiagnostic {
    std::size_t request_index = 0;
    std::size_t raw_y = 0;
    std::size_t raw_x = 0;
    std::size_t rank = 0;
    int initial_guess_y = 0;
    int initial_guess_x = 0;
    int local_offset_y = 0;
    int local_offset_x = 0;
    int internal_candidate_y = 0;
    int internal_candidate_x = 0;
    int saved_candidate_y = 0;
    int saved_candidate_x = 0;
    float descriptor_score_neg_ssd = 0.0f;
    float descriptor_cost_ssd = 0.0f;
};

class WSVT {
public:
    WSVT(
        const std::vector<float>& img_stack,
        const std::vector<float>& ref_stack,
        std::size_t ch,
        std::size_t h,
        std::size_t w,
        int crop = 512,
        int cal_half_window = 20,
        int n_template = 0,
        int n_s_extend = 4,
        int n_cores = 4,
        int n_group = 4,
        double energy = 14e3,
        double p_x = 0.65e-6,
        double mag_factor = 1.0,
        double z = 500e-3,
        int wavelet_level_cut = 2,
        int pyramid_level = 2,
        int n_iter = 1,
        bool use_estimate = false,
        bool use_wavelet = true,
        int use_gpu = 0,
        bool calc_darkfield = true,
        int phase_cores = 0,
        bool search_early_abandon = false,
        int search_top_k = 2,
        int search_block_size = 16,
        bool search_two_pass = false,
        int search_prefix_size = 16,
        bool search_guard_cache = false);

    /// Move-semantics overload: takes ownership of img/ref data without copying
    WSVT(
        std::vector<float>&& img_stack,
        std::vector<float>&& ref_stack,
        std::size_t ch,
        std::size_t h,
        std::size_t w,
        int crop = 512,
        int cal_half_window = 20,
        int n_template = 0,
        int n_s_extend = 4,
        int n_cores = 4,
        int n_group = 4,
        double energy = 14e3,
        double p_x = 0.65e-6,
        double mag_factor = 1.0,
        double z = 500e-3,
        int wavelet_level_cut = 2,
        int pyramid_level = 2,
        int n_iter = 1,
        bool use_estimate = false,
        bool use_wavelet = true,
        int use_gpu = 0,
        bool calc_darkfield = true,
        int phase_cores = 0,
        bool search_early_abandon = false,
        int search_top_k = 2,
        int search_block_size = 16,
        bool search_two_pass = false,
        int search_prefix_size = 16,
        bool search_guard_cache = false);

    AlignedVector<float> stack_TemplateWindow(const std::vector<float>& img, std::size_t in_ch, std::size_t in_h, std::size_t in_w, std::size_t& out_h, std::size_t& out_w, std::size_t& out_d) const;
    PyramidResult pyramid_data();
    PyramidResult wavelet_data();
    SolverOutput solver();
    SolverOutput run(const std::string& result_path = "", bool cleansave = false, int h5_deflate = 9);

    /// Enable exact Top-K capture at a small set of raw-grid pixels.  This is
    /// a diagnostic side channel; it is empty and has no hot-loop cost in the
    /// default solver.  The current diagnostic is deliberately restricted to
    /// exhaustive search so every ranked score is an exact full SSD.
    void configure_search_topk_diagnostics(
        std::vector<std::size_t> raw_linear_pixels,
        std::size_t top_k);
    [[nodiscard]] const std::vector<SearchTopKDiagnostic>&
    search_topk_diagnostics() const noexcept;

    /// Configure the B1/B2 easy-to-hard research profile.  The default
    /// constructed configuration leaves the established B0 path unchanged.
    void configure_easy_to_hard(EasyToHardConfig config);

    /// Enable the fixed, default-off Wavelet-guided UMPA H1 raw refinement.
    /// H1 is deliberately incompatible with the closed easy-to-hard profiles
    /// and exact-pruning diagnostics in its first integration version.
    void configure_wavelet_guided_umpa(WaveletGuidedUmpaConfig config);

private:
    void crop_inputs_if_requested();
    std::vector<float> resampling_spline(const std::vector<float>& img, std::size_t in_h, std::size_t in_w, std::size_t out_h, std::size_t out_w) const;
    DisplaceWaveletOutput displace_wavelet(
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
        std::span<const std::uint8_t> active_mask = {},
        std::span<const std::uint8_t> easy_mask = {},
        std::span<const float> temporal_contrast = {}) const;
    PyramidResult wavelet_data_for_frame_prefix(
        std::size_t frame_count,
        double& pyramid_time_s,
        double& template_window_time_s,
        double& wavelet_time_s);
    [[nodiscard]] std::vector<float> temporal_speckle_contrast_map(
        std::size_t frame_count) const;

    std::vector<float> img_data_;
    std::vector<float> ref_data_;
    std::size_t ch_;
    std::size_t h_;
    std::size_t w_;
    int crop_;
    int cal_half_window_;
    int n_s_extend_;
    int n_template_;
    int n_cores_;
    int phase_cores_;
    int n_group_;
    double mag_factor_;
    double energy_;
    double wavelength_;
    double p_x_;
    double z_;
    int wavelet_level_cut_;
    int pyramid_level_;
    int n_iter_;
    bool use_estimate_;
    bool use_wavelet_;
    bool use_gpu_;
    bool calc_darkfield_;
    bool search_early_abandon_;
    bool search_two_pass_;
    int search_top_k_;
    int search_block_size_;
    int search_prefix_size_;
    bool search_guard_cache_;
    int wavelet_level_;
    std::vector<int> wavelet_add_list_;
    std::vector<float> displace_estimate_y_;
    std::vector<float> displace_estimate_x_;
    std::size_t displace_estimate_h_;
    std::size_t displace_estimate_w_;
    double last_pyramid_time_s_;
    double last_template_window_time_s_ = 0.0;
    double last_wavelet_time_s_;
    std::vector<std::size_t> diagnostic_pixels_;
    std::size_t diagnostic_top_k_ = 0;
    mutable std::vector<SearchTopKDiagnostic> search_topk_diagnostics_;
    EasyToHardConfig easy_to_hard_;
    WaveletGuidedUmpaConfig wavelet_guided_umpa_;
};

}
