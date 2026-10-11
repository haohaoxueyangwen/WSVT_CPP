#pragma once
#include "wsvt/umpa_ddf_kernel.hpp"
#include <array>
namespace wsvt {
struct UmpaDdfAuthorSettings {
    std::array<double,3> initial{0.,0.,0.4};
    double blur_extra=0.05, max_sigma=10.;
    std::size_t max_iterations=500;
};
struct UmpaDdfAuthorResult {
    std::array<double,3> transform{};
    UmpaDdfKernelParameters kernel{};
    UmpaDdfKernelFit fit{};
    double penalty=0.;
    std::size_t iterations=0, evaluations=0;
    bool converged=false;
};
WSVT_API UmpaDdfAuthorResult fit_umpa_ddf_author_fixed_u(
    std::span<const double> sample, std::span<const double> reference,
    std::size_t frames, std::size_t height, std::size_t width,
    std::size_t y, std::size_t x, std::ptrdiff_t uy, std::ptrdiff_t ux,
    std::size_t window_half, std::ptrdiff_t max_shift,
    UmpaDdfAuthorSettings settings = {});
}
