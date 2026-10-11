#pragma once

#include "wsvt/export.hpp"
#include "wsvt/umpa_physical_fit.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace wsvt {

/// Parameters of the discrete anisotropic Gaussian used by official UMPA-DDF.
/// The kernel is exp(-a*i*i-b*i*j-c*j*j), normalized over the finite support
/// [-kernel_radius, kernel_radius]^2.  i is image row and j is image column.
struct UmpaDdfKernelParameters {
    std::size_t kernel_radius = 8U;
    double a = 0.5;
    double b = 0.0;
    double c = 0.5;
};

/// Sufficient statistics and result of one fixed-u official ModelDF-kernel
/// evaluation.  No parameter is clipped and no physical-validity claim is
/// inferred from a finite cost.  `cost` is the official algebraic cost after
/// eliminating T, divided by the unmasked frame count.
struct UmpaDdfKernelFit {
    double t1 = 0.0;             ///< sum(Gamma * sample^2)
    double t3 = 0.0;             ///< sum(Gamma * blurred_reference^2)
    double t5 = 0.0;             ///< sum(Gamma * blurred_reference * sample)
    double weight_sum = 0.0;    ///< official unmasked wt (number of frames)
    std::size_t observation_count = 0U;
    double transmission = 0.0;  ///< T = t5 / t3
    double cost = 0.0;           ///< (t1 - t5*T) / weight_sum
    double residual_cost = 0.0;  ///< direct sum of weighted squared residuals / weight_sum
    bool numerical_valid = false;
};

/// Evaluate one official UMPA++ ModelDF-kernel candidate on full frame-major
/// double stacks.  `official_shift` is the public UMPA shift: the reference
/// center is sample center + official_shift.  Components must be strictly
/// inside (-max_shift,max_shift), matching the official open boundary.
///
/// Unlike the scalar ModelDF helper, this function uses the raw reference
/// intensity and a finite normalized Gaussian convolution.  The full input
/// array is required to contain analysis_radius + max_shift + kernel_radius
/// pixels around the requested coordinate; no wrap, padding, or implicit crop
/// is performed.  This is an evaluation primitive only and is not connected to
/// the WSVT production pipeline.
WSVT_API UmpaDdfKernelFit fit_umpa_ddf_kernel_official_integer_at(
    std::span<const double> sample_stack,
    std::span<const double> reference_stack,
    std::size_t frames,
    std::size_t height,
    std::size_t width,
    std::size_t coordinate_y,
    std::size_t coordinate_x,
    std::ptrdiff_t official_shift_y,
    std::ptrdiff_t official_shift_x,
    std::size_t analysis_radius,
    std::ptrdiff_t max_shift,
    UmpaDdfKernelParameters kernel = {},
    UmpaAssignCoordinates assign_coordinates = UmpaAssignCoordinates::Sample);

}  // namespace wsvt
