#pragma once

#include "wsvt/export.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace wsvt {

/// Six sufficient statistics used by the UMPA++ ModelDF normal equation.
struct UmpaSufficientStatistics {
    double l1 = 0.0;  ///< sum(w * sample^2)
    double l2 = 0.0;  ///< sum(w * reference_mean^2)
    double l3 = 0.0;  ///< sum(w * reference^2)
    double l4 = 0.0;  ///< sum(w * reference_mean * sample)
    double l5 = 0.0;  ///< sum(w * reference * sample)
    double l6 = 0.0;  ///< sum(w * reference * reference_mean)
    double weight_sum = 0.0;
    std::size_t observation_count = 0;
};

/// Unconstrained fixed-displacement raw-intensity fit.
///
/// No value is clipped into a physical range. `numerical_valid` reports
/// identifiability/finite arithmetic; `physical_valid` additionally requires
/// T>0 and 0<=D<=1.
struct UmpaPhysicalFit {
    UmpaSufficientStatistics statistics{};
    double alpha = 0.0;         ///< T*D
    double beta = 0.0;          ///< T*(1-D)
    double transmission = 0.0;  ///< T
    double visibility = 0.0;    ///< D
    double cost = 0.0;          ///< weighted mean squared raw residual
    double delta = 0.0;         ///< determinant of the 2x2 normal matrix
    double condition = 0.0;     ///< spectral condition number of that matrix
    bool numerical_valid = false;
    bool physical_valid = false;
};

/// NumPy-compatible separable Hamming window, flattened row-major and
/// normalized to unit sum. Radius two yields the frozen 5x5 analysis window.
WSVT_API std::vector<double> normalized_hamming_window_2d(std::size_t radius);

/// Accumulate sufficient statistics from already aligned raw observations.
/// A zero weight is an explicit mask and may cover non-finite data.
WSVT_API UmpaSufficientStatistics accumulate_umpa_statistics(
    std::span<const double> sample,
    std::span<const double> reference,
    std::span<const double> reference_mean,
    std::span<const double> weights = {});

/// Solve the two-by-two normal equation without applying physical constraints.
WSVT_API UmpaPhysicalFit solve_umpa_physical_fit(
    const UmpaSufficientStatistics& statistics,
    double relative_delta_tolerance = 1.0e-12,
    double transmission_epsilon = 1.0e-12);

/// Accumulate, solve, and recompute the non-negative weighted residual cost.
WSVT_API UmpaPhysicalFit fit_umpa_physical(
    std::span<const double> sample,
    std::span<const double> reference,
    std::span<const double> reference_mean,
    std::span<const double> weights = {},
    double relative_delta_tolerance = 1.0e-12,
    double transmission_epsilon = 1.0e-12);

/// Extract matching raw sample/reference patches from frame-major image stacks
/// and perform the fixed integer-displacement fit. Positive dx/dy follows the
/// project convention: sample(y,x) matches reference(y-dy,x-dx).
/// Both complete analysis patches must be in bounds; otherwise this function
/// throws std::out_of_range instead of silently cropping or padding.
WSVT_API UmpaPhysicalFit fit_umpa_physical_integer_at(
    std::span<const double> sample_stack,
    std::span<const double> reference_stack,
    std::size_t frames,
    std::size_t height,
    std::size_t width,
    std::size_t sample_y,
    std::size_t sample_x,
    std::ptrdiff_t displacement_y,
    std::ptrdiff_t displacement_x,
    std::size_t analysis_radius,
    double relative_delta_tolerance = 1.0e-12,
    double transmission_epsilon = 1.0e-12);

}  // namespace wsvt
