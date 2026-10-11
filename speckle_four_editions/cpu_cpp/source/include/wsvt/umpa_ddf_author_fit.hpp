#pragma once
#include "wsvt/umpa_ddf_kernel.hpp"
#include <array>
namespace wsvt {
struct UmpaDdfAuthorSettings {
    std::array<double,3> initial{0.,0.,0.4};
    double blur_extra=0.05, max_sigma=10.;
    std::size_t max_iterations=500;
    /// Finite kernel support radius. Author baseline is 8 (17×17).
    /// Independent of the analysis window (`window_half`).
    std::size_t kernel_radius=8U;
    /// Nelder–Mead absolute objective spread. Author baseline is 1e-12.
    /// The application path may pass a larger value: algebraic cost and the
    /// direct residual already differ by about 1e-7 from cancellation, so
    /// 1e-12 is finer than that noise.
    double objective_atol=1e-12;
    /// Application path. Author fits leave this false and minimize algebraic
    /// cost + penalty. When true, the simplex minimizes direct residual + penalty.
    /// Both terms stay in residual units; a later division by data_scale is only
    /// a report of that same pair.
    bool minimize_direct_residual=false;
};
struct UmpaDdfAuthorResult {
    std::array<double,3> transform{};
    UmpaDdfKernelParameters kernel{};
    UmpaDdfKernelFit fit{};
    double penalty=0.;
    std::size_t iterations=0, evaluations=0;
    bool converged=false;
    double final_dx=0.;
    double final_df=0.;
    bool author_converged=false;
    /// Weighted mean square of the sample window, t1/weight_sum. Fixed for a
    /// pixel. Divides residual and penalty together when a normalized value is reported.
    double data_scale=1.;
};
WSVT_API UmpaDdfAuthorResult fit_umpa_ddf_author_fixed_u(
    std::span<const double> sample, std::span<const double> reference,
    std::size_t frames, std::size_t height, std::size_t width,
    std::size_t y, std::size_t x, std::ptrdiff_t uy, std::ptrdiff_t ux,
    std::size_t window_half, std::ptrdiff_t max_shift,
    UmpaDdfAuthorSettings settings = {});
}
