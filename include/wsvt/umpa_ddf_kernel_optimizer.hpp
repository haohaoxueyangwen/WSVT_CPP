#pragma once

#include "wsvt/export.hpp"
#include "wsvt/umpa_ddf_kernel.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace wsvt {

/// Deterministic optimizer settings for a fixed-u raw-intensity DDF fit.
/// The search is performed in log-precision / Fisher-z coordinates, so every
/// evaluated kernel is finite and positive definite.  These settings are
/// deliberately fixed for a run; they are not a data-dependent sweep.
struct UmpaDdfKernelOptimizerParameters {
    std::size_t kernel_radius = 8U;
    double initial_a = 0.8;
    double initial_b = 0.0;
    double initial_c = 0.8;
    std::size_t max_iterations = 400U;
    double simplex_step = 0.35;
    double parameter_tolerance = 1.0e-5;
    double objective_tolerance = 1.0e-10;
};

struct UmpaDdfKernelOptimization {
    UmpaDdfKernelParameters kernel{};
    UmpaDdfKernelFit fit{};
    double initial_residual_cost = 0.0;
    std::size_t iterations = 0U;
    std::size_t evaluations = 0U;
    bool converged = false;
    bool numerical_valid = false;
};

/// Fit the positive-definite finite UMPA-DDF kernel at one fixed registration.
/// The objective is the direct weighted squared residual from
/// `fit_umpa_ddf_kernel_official_integer_at`, not the cancellation-prone
/// `t1-t5*T` expression.  The WSVT displacement is an input and is never
/// optimized here.
WSVT_API UmpaDdfKernelOptimization optimize_umpa_ddf_kernel_fixed_u(
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
    UmpaDdfKernelOptimizerParameters optimizer = {},
    UmpaAssignCoordinates assign_coordinates = UmpaAssignCoordinates::Sample);

}  // namespace wsvt
