#pragma once

#include "wsvt/export.hpp"
#include "wsvt/umpa_physical_fit.hpp"

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

/// Objective used to score the same fixed SET4 raw-intensity candidate union.
///
/// This enum separates the effect of spatial support from the physical model:
/// WindowedZncc is the XST-XSVT-style window control, ModelT fits only local
/// transmission, and ModelDF fits transmission plus modulation visibility.
enum class SetTransportRawObjective {
    WindowedZncc,
    ModelT,
    ModelDF,
};

WSVT_API const char* set_transport_raw_objective_name(
    SetTransportRawObjective objective) noexcept;

/// Evaluate one fixed raw-intensity candidate with the exact production
/// SET4-rerank objective implementation.  This is a diagnostic entry point:
/// it does not search, route, interpolate, or alter the WSVT displacement.
/// Both sample/reference analysis patches must be complete.
WSVT_API UmpaPhysicalFit evaluate_set_transport_raw_candidate(
    SetTransportRawObjective objective,
    std::span<const float> sample_stack,
    std::span<const float> reference_stack,
    std::size_t frames,
    std::size_t height,
    std::size_t width,
    std::size_t sample_y,
    std::size_t sample_x,
    std::size_t reference_y,
    std::size_t reference_x,
    std::size_t analysis_radius,
    std::span<const double> normalized_window,
    double relative_delta_tolerance = 1.0e-12,
    double transmission_epsilon = 1.0e-12,
    double variance_epsilon = 1.0e-12);

/// Fixed, default-off raw rerank over the exact SET4 fine candidate union.
struct SetTransportRawRerankConfig {
    bool enabled = false;
    bool representatives_only = false;
    SetTransportRawObjective objective = SetTransportRawObjective::ModelDF;
    int domain_half_window = 4;
    std::size_t analysis_radius = 1;
    double relative_delta_tolerance = 1.0e-12;
    double transmission_epsilon = 1.0e-12;
    double variance_epsilon = 1.0e-12;
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

/// Rerank the deduplicated union of four fixed SET4 displacement domains.
///
/// `set_center_y/x` are pixel-major arrays with four saved-convention integer
/// centres per output pixel. `fallback_y/x` are the established SET4 result and
/// are retained when no raw candidate is numerically valid. Every objective
/// uses the same candidate union, frames, 3x3 normalized-Hamming support and
/// row-major tie ordering. No adaptive routing or subpixel interpolation is
/// applied by this bounded mechanism-ablation path.
WSVT_API WaveletGuidedUmpaOutput rerank_set_transport_raw(
    std::span<const float> sample_stack,
    std::span<const float> reference_stack,
    std::size_t frames,
    std::size_t raw_height,
    std::size_t raw_width,
    std::span<const int> set_center_y,
    std::span<const int> set_center_x,
    std::span<const int> set_representative_y,
    std::span<const int> set_representative_x,
    std::span<const float> fallback_y,
    std::span<const float> fallback_x,
    std::size_t output_height,
    std::size_t output_width,
    std::size_t sample_origin_y,
    std::size_t sample_origin_x,
    const SetTransportRawRerankConfig& config = {});

}  // namespace wsvt
