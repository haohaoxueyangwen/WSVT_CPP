#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "wsvt/umpa_ddf_kernel.hpp"
#include "wsvt/umpa_physical_fit.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>

using namespace wsvt;

namespace {

std::size_t at(std::size_t frame, std::size_t y, std::size_t x,
               std::size_t height, std::size_t width) {
    return (frame * height + y) * width + x;
}

std::vector<double> kernel_values(UmpaDdfKernelParameters p) {
    const std::size_t n = 2U * p.kernel_radius + 1U;
    std::vector<double> k(n * n);
    double sum = 0.0;
    for (std::ptrdiff_t i = -static_cast<std::ptrdiff_t>(p.kernel_radius);
         i <= static_cast<std::ptrdiff_t>(p.kernel_radius); ++i) {
        for (std::ptrdiff_t j = -static_cast<std::ptrdiff_t>(p.kernel_radius);
             j <= static_cast<std::ptrdiff_t>(p.kernel_radius); ++j) {
            const double v = std::exp(
                -p.a * static_cast<double>(i * i) -
                p.b * static_cast<double>(i * j) -
                p.c * static_cast<double>(j * j));
            k[static_cast<std::size_t>(i + static_cast<std::ptrdiff_t>(p.kernel_radius)) * n +
              static_cast<std::size_t>(j + static_cast<std::ptrdiff_t>(p.kernel_radius))] = v;
            sum += v;
        }
    }
    for (double& v : k) v /= sum;
    return k;
}

}  // namespace

TEST_CASE("UMPA-DDF fixed-u kernel recovers raw T and official cost", "[umpa][ddf-kernel]") {
    constexpr std::size_t frames = 5U;
    constexpr std::size_t height = 48U;
    constexpr std::size_t width = 51U;
    constexpr std::size_t y0 = 24U;
    constexpr std::size_t x0 = 25U;
    constexpr std::ptrdiff_t shift_y = 2;
    constexpr std::ptrdiff_t shift_x = -3;
    constexpr std::size_t analysis_radius = 2U;
    constexpr std::ptrdiff_t max_shift = 6;
    constexpr double transmission = 0.73;
    UmpaDdfKernelParameters params{1U, 0.42, 0.06, 0.71};
    const auto kernel = kernel_values(params);
    const std::size_t kw = 2U * params.kernel_radius + 1U;
    std::vector<double> reference(frames * height * width, 0.0);
    std::vector<double> sample(reference.size(), 0.0);
    for (std::size_t f = 0; f < frames; ++f) {
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                reference[at(f, y, x, height, width)] =
                    30.0 + 2.0 * static_cast<double>(f) +
                    0.017 * static_cast<double>(y * y) +
                    0.031 * static_cast<double>(x * x) +
                    0.11 * static_cast<double>(x * y) +
                    std::sin(0.17 * static_cast<double>(x + 2U * y + f));
            }
        }
        for (std::size_t y = params.kernel_radius + 8U;
             y + params.kernel_radius + 8U < height; ++y) {
            for (std::size_t x = params.kernel_radius + 8U;
                 x + params.kernel_radius + 8U < width; ++x) {
                double blurred = 0.0;
                for (std::size_t ky = 0; ky < kw; ++ky) {
                    for (std::size_t kx = 0; kx < kw; ++kx) {
                        const auto dy = static_cast<std::ptrdiff_t>(ky) -
                            static_cast<std::ptrdiff_t>(params.kernel_radius);
                        const auto dx = static_cast<std::ptrdiff_t>(kx) -
                            static_cast<std::ptrdiff_t>(params.kernel_radius);
                        blurred += kernel[ky * kw + kx] * reference[
                            at(f, static_cast<std::size_t>(static_cast<std::ptrdiff_t>(y) + dy + shift_y),
                               static_cast<std::size_t>(static_cast<std::ptrdiff_t>(x) + dx + shift_x),
                               height, width)];
                    }
                }
                sample[at(f, y, x, height, width)] = transmission * blurred;
            }
        }
    }
    const auto fit = fit_umpa_ddf_kernel_official_integer_at(
        sample, reference, frames, height, width, y0, x0,
        shift_y, shift_x, analysis_radius, max_shift, params);
    REQUIRE(fit.numerical_valid);
    REQUIRE(fit.transmission == Catch::Approx(transmission).margin(1.0e-12));
    REQUIRE(fit.cost <= 1.0e-9);
    REQUIRE(fit.residual_cost <= 1.0e-9);
    REQUIRE(fit.weight_sum == Catch::Approx(static_cast<double>(frames)));
    REQUIRE(fit.observation_count == frames * 25U);
}

TEST_CASE("UMPA-DDF coordinate assignment and sign are equivalent", "[umpa][ddf-kernel][sign]") {
    constexpr std::size_t frames = 3U;
    constexpr std::size_t height = 32U;
    constexpr std::size_t width = 35U;
    constexpr std::size_t sample_y = 16U;
    constexpr std::size_t sample_x = 18U;
    constexpr std::ptrdiff_t sy = 2;
    constexpr std::ptrdiff_t sx = -3;
    std::vector<double> reference(frames * height * width);
    std::vector<double> sample(reference.size());
    for (std::size_t f = 0; f < frames; ++f) {
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                reference[at(f, y, x, height, width)] =
                    5.0 + 0.7 * static_cast<double>(f) +
                    0.2 * static_cast<double>(y) +
                    0.13 * static_cast<double>(x) +
                    std::sin(static_cast<double>(x * 3U + y * 5U + f));
            }
        }
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const auto source_y = static_cast<std::ptrdiff_t>(y) + sy;
                const auto source_x = static_cast<std::ptrdiff_t>(x) + sx;
                sample[at(f, y, x, height, width)] =
                    (source_y >= 0 && source_x >= 0 &&
                     source_y < static_cast<std::ptrdiff_t>(height) &&
                     source_x < static_cast<std::ptrdiff_t>(width))
                    ? 0.81 * reference[at(
                        f, static_cast<std::size_t>(source_y),
                        static_cast<std::size_t>(source_x), height, width)]
                    : 0.0;
            }
        }
    }
    UmpaDdfKernelParameters delta{0U, 1.0, 0.0, 1.0};
    const auto from_sample = fit_umpa_ddf_kernel_official_integer_at(
        sample, reference, frames, height, width, sample_y, sample_x,
        sy, sx, 1U, 5, delta, UmpaAssignCoordinates::Sample);
    const auto from_reference = fit_umpa_ddf_kernel_official_integer_at(
        sample, reference, frames, height, width, sample_y + sy, sample_x + sx,
        sy, sx, 1U, 5, delta, UmpaAssignCoordinates::Reference);
    REQUIRE(from_sample.numerical_valid);
    REQUIRE(from_reference.numerical_valid);
    REQUIRE(from_sample.transmission == Catch::Approx(from_reference.transmission).margin(1.0e-12));
    REQUIRE(from_sample.cost == Catch::Approx(from_reference.cost).margin(1.0e-12));
    REQUIRE(from_sample.transmission == Catch::Approx(0.81).margin(1.0e-12));
}

TEST_CASE("UMPA-DDF rejects open shift and full halo boundary", "[umpa][ddf-kernel][boundary]") {
    const std::vector<double> stack(2U * 64U * 64U, 1.0);
    UmpaDdfKernelParameters params{8U, 0.5, 0.0, 0.5};
    REQUIRE_THROWS_AS(
        fit_umpa_ddf_kernel_official_integer_at(
            stack, stack, 2U, 64U, 64U, 32U, 32U, 6, 0, 1U, 6, params),
        std::out_of_range);
    REQUIRE_THROWS_AS(
        fit_umpa_ddf_kernel_official_integer_at(
            stack, stack, 2U, 64U, 64U, 10U, 10U, 1, 1, 1U, 6, params),
        std::out_of_range);
    REQUIRE_THROWS_AS(
        fit_umpa_ddf_kernel_official_integer_at(
            stack, stack, 2U, 64U, 64U, 32U, 32U, 1, 1, 1U, 6,
            params, static_cast<UmpaAssignCoordinates>(99)),
        std::invalid_argument);
}
