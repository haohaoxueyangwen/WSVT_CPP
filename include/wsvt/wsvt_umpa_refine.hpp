#pragma once

#include "wsvt/export.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace wsvt {

/// Fixed-parameter Wavelet-guided UMPA H1 configuration.
///
/// The first identifiable profile uses a 3x3 displacement neighbourhood and a
/// fixed 3x3 Hamming analysis support.  Radius zero is rejected by the ModelDF
/// rank gate because reference and reference_mean become identical columns.
struct WaveletGuidedUmpaConfig {
    bool enabled = false;
    int local_half_window = 1;
    std::size_t analysis_radius = 1;
    double relative_delta_tolerance = 1.0e-12;
    double transmission_epsilon = 1.0e-12;
};

/// Dense H1 maps on the saved sample-coordinate output grid.
struct WaveletGuidedUmpaOutput {
    std::vector<float> proposal_y;
    std::vector<float> proposal_x;
    std::vector<float> displace_y;
    std::vector<float> displace_x;
    std::vector<float> relative_offset_y;
    std::vector<float> relative_offset_x;
    std::vector<float> best_cost;
    std::vector<float> second_best_cost;
    std::vector<float> cost_margin;
    std::vector<float> transmission;
    std::vector<float> visibility;
    std::vector<float> delta;
    std::vector<float> condition;
    std::vector<float> numerical_valid;
    std::vector<float> physical_valid;
    std::vector<float> local_boundary_hit;
    std::vector<float> candidates_evaluated;
    std::uint64_t raw_candidate_count = 0;
    std::uint64_t raw_observation_count = 0;
    std::uint64_t numerical_valid_pixel_count = 0;
    std::uint64_t physical_valid_pixel_count = 0;
};

/// Evaluate fixed-radius ModelDF raw costs around integer WSVT proposals.
///
/// `sample_stack` and `reference_stack` are frame-major FHW float arrays. The
/// proposal maps use the saved project convention: sample(y,x) matches
/// reference(y-dy,x-dx). Output pixel (0,0) is assigned to raw sample
/// coordinate (`sample_origin_y`, `sample_origin_x`). Geometrically incomplete
/// candidates are skipped; if no numerical candidate remains, displacement is
/// left at the proposal and validity stays false. Exact cost ties keep the
/// first row-major candidate (dy then dx, both ascending).
WSVT_API WaveletGuidedUmpaOutput refine_wavelet_guided_umpa(
    std::span<const float> sample_stack,
    std::span<const float> reference_stack,
    std::size_t frames,
    std::size_t raw_height,
    std::size_t raw_width,
    std::span<const int> proposal_y,
    std::span<const int> proposal_x,
    std::size_t output_height,
    std::size_t output_width,
    std::size_t sample_origin_y,
    std::size_t sample_origin_x,
    const WaveletGuidedUmpaConfig& config = {});

}  // namespace wsvt
