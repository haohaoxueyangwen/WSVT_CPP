#include "wsvt/umpa_ddf_kernel.hpp"

#include "wsvt/umpa_physical_fit.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace wsvt {
namespace {

bool finite_value(double value) noexcept {
    constexpr std::uint64_t exponent_mask = 0x7ff0000000000000ULL;
    return (std::bit_cast<std::uint64_t>(value) & exponent_mask) != exponent_mask;
}

std::size_t checked_product(std::size_t lhs, std::size_t rhs) {
    if (lhs != 0U && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error("UMPA-DDF dimensions overflow");
    }
    return lhs * rhs;
}

std::size_t shifted_center(
    std::size_t center, std::ptrdiff_t displacement, std::size_t extent) {
    if (displacement >= 0) {
        const auto amount = static_cast<std::size_t>(displacement);
        if (amount > center) {
            throw std::out_of_range("UMPA-DDF shifted center is out of bounds");
        }
        return center - amount;
    }
    const auto amount = static_cast<std::size_t>(-(displacement + 1)) + 1U;
    if (amount >= extent || center >= extent - amount) {
        throw std::out_of_range("UMPA-DDF shifted center is out of bounds");
    }
    return center + amount;
}

void validate_open_shift(std::ptrdiff_t shift, std::ptrdiff_t max_shift) {
    if (max_shift <= 0 || shift <= -max_shift || shift >= max_shift) {
        throw std::out_of_range("UMPA-DDF shift lies outside the open interval");
    }
}

void validate_center(
    std::size_t center, std::size_t extent, std::size_t padding,
    const char* message) {
    if (extent == 0U || padding > (extent - 1U) / 2U ||
        center < padding || center >= extent - padding) {
        throw std::out_of_range(message);
    }
}

std::vector<double> normalized_kernel(const UmpaDdfKernelParameters& params) {
    if (!finite_value(params.a) || !finite_value(params.b) ||
        !finite_value(params.c) || params.a <= 0.0 || params.c <= 0.0 ||
        4.0 * params.a * params.c <= params.b * params.b) {
        throw std::invalid_argument("UMPA-DDF kernel precision matrix is invalid");
    }
    if (params.kernel_radius > 1024U) {
        throw std::invalid_argument("UMPA-DDF kernel radius is unreasonable");
    }
    const std::size_t width = params.kernel_radius * 2U + 1U;
    std::vector<double> kernel(checked_product(width, width), 0.0);
    double total = 0.0;
    const auto radius = static_cast<std::ptrdiff_t>(params.kernel_radius);
    for (std::ptrdiff_t i = -radius; i <= radius; ++i) {
        for (std::ptrdiff_t j = -radius; j <= radius; ++j) {
            const double ii = static_cast<double>(i);
            const double jj = static_cast<double>(j);
            const double value = std::exp(
                -params.a * ii * ii - params.b * ii * jj - params.c * jj * jj);
            if (!finite_value(value)) {
                throw std::invalid_argument("UMPA-DDF kernel is non-finite");
            }
            const auto y = static_cast<std::size_t>(i + radius);
            const auto x = static_cast<std::size_t>(j + radius);
            kernel[y * width + x] = value;
            total += value;
        }
    }
    if (!(total > 0.0) || !finite_value(total)) {
        throw std::invalid_argument("UMPA-DDF kernel normalization failed");
    }
    for (double& value : kernel) {
        value /= total;
    }
    return kernel;
}

}  // namespace

UmpaDdfKernelFit fit_umpa_ddf_kernel_official_integer_at(
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
    UmpaDdfKernelParameters kernel,
    UmpaAssignCoordinates assign_coordinates) {
    validate_open_shift(official_shift_y, max_shift);
    validate_open_shift(official_shift_x, max_shift);
    if (frames == 0U || height == 0U || width == 0U) {
        throw std::invalid_argument("UMPA-DDF requires non-empty image stacks");
    }
    const std::size_t plane = checked_product(height, width);
    const std::size_t stack_size = checked_product(frames, plane);
    if (sample_stack.size() != stack_size || reference_stack.size() != stack_size) {
        throw std::invalid_argument("UMPA-DDF stack size does not match dimensions");
    }
    if (kernel.kernel_radius >
        std::numeric_limits<std::size_t>::max() - analysis_radius) {
        throw std::overflow_error("UMPA-DDF support padding overflows");
    }
    const std::size_t support = analysis_radius + kernel.kernel_radius;
    if (support > std::numeric_limits<std::size_t>::max() -
        static_cast<std::size_t>(max_shift)) {
        throw std::overflow_error("UMPA-DDF support padding overflows");
    }
    const std::size_t padding = support + static_cast<std::size_t>(max_shift);
    validate_center(coordinate_y, height, padding, "UMPA-DDF coordinate y is out of bounds");
    validate_center(coordinate_x, width, padding, "UMPA-DDF coordinate x is out of bounds");

    std::size_t sample_y = coordinate_y;
    std::size_t sample_x = coordinate_x;
    std::size_t reference_y = coordinate_y;
    std::size_t reference_x = coordinate_x;
    switch (assign_coordinates) {
    case UmpaAssignCoordinates::Sample:
        reference_y = shifted_center(coordinate_y, -official_shift_y, height);
        reference_x = shifted_center(coordinate_x, -official_shift_x, width);
        break;
    case UmpaAssignCoordinates::Reference:
        sample_y = shifted_center(coordinate_y, official_shift_y, height);
        sample_x = shifted_center(coordinate_x, official_shift_x, width);
        break;
    default:
        throw std::invalid_argument("unknown UMPA-DDF coordinate assignment");
    }
    validate_center(sample_y, height, support, "UMPA-DDF sample support is out of bounds");
    validate_center(sample_x, width, support, "UMPA-DDF sample support is out of bounds");
    validate_center(reference_y, height, support, "UMPA-DDF reference support is out of bounds");
    validate_center(reference_x, width, support, "UMPA-DDF reference support is out of bounds");

    const std::vector<double> window = normalized_hamming_window_2d(analysis_radius);
    const std::size_t analysis_width = analysis_radius * 2U + 1U;
    const std::size_t kernel_width = kernel.kernel_radius * 2U + 1U;
    const std::vector<double> discrete_kernel = normalized_kernel(kernel);
    UmpaDdfKernelFit fit;
    fit.weight_sum = static_cast<double>(frames);
    fit.observation_count = checked_product(frames, checked_product(analysis_width, analysis_width));

    for (std::size_t frame = 0; frame < frames; ++frame) {
        const std::size_t offset = frame * plane;
        for (std::size_t wy = 0; wy < analysis_width; ++wy) {
            const auto dy = static_cast<std::ptrdiff_t>(wy) -
                static_cast<std::ptrdiff_t>(analysis_radius);
            for (std::size_t wx = 0; wx < analysis_width; ++wx) {
                const auto dx = static_cast<std::ptrdiff_t>(wx) -
                    static_cast<std::ptrdiff_t>(analysis_radius);
                const std::size_t sample_index =
                    offset + static_cast<std::size_t>(
                        static_cast<std::ptrdiff_t>(sample_y) + dy) * width +
                    static_cast<std::size_t>(static_cast<std::ptrdiff_t>(sample_x) + dx);
                const std::size_t ref_y = static_cast<std::size_t>(
                    static_cast<std::ptrdiff_t>(reference_y) + dy);
                const std::size_t ref_x = static_cast<std::size_t>(
                    static_cast<std::ptrdiff_t>(reference_x) + dx);
                double blurred = 0.0;
                for (std::size_t ky = 0; ky < kernel_width; ++ky) {
                    const auto sy = static_cast<std::ptrdiff_t>(ky) -
                        static_cast<std::ptrdiff_t>(kernel.kernel_radius);
                    for (std::size_t kx = 0; kx < kernel_width; ++kx) {
                        const auto sx = static_cast<std::ptrdiff_t>(kx) -
                            static_cast<std::ptrdiff_t>(kernel.kernel_radius);
                        const std::size_t index = offset +
                            static_cast<std::size_t>(static_cast<std::ptrdiff_t>(ref_y) + sy) * width +
                            static_cast<std::size_t>(static_cast<std::ptrdiff_t>(ref_x) + sx);
                        blurred += discrete_kernel[ky * kernel_width + kx] * reference_stack[index];
                    }
                }
                const double sample = sample_stack[sample_index];
                if (!finite_value(sample) || !finite_value(blurred)) {
                    throw std::invalid_argument("UMPA-DDF active observations must be finite");
                }
                const double weight = window[wy * analysis_width + wx];
                fit.t1 += weight * sample * sample;
                fit.t3 += weight * blurred * blurred;
                fit.t5 += weight * blurred * sample;
            }
        }
    }
    if (!(fit.weight_sum > 0.0) || !(fit.t3 > 0.0) ||
        !finite_value(fit.t1) || !finite_value(fit.t3) || !finite_value(fit.t5)) {
        fit.transmission = std::numeric_limits<double>::quiet_NaN();
        fit.cost = std::numeric_limits<double>::quiet_NaN();
        fit.residual_cost = std::numeric_limits<double>::quiet_NaN();
        return fit;
    }
    fit.transmission = fit.t5 / fit.t3;
    fit.cost = (fit.t1 - fit.t5 * fit.transmission) / fit.weight_sum;
    double residual_sum = 0.0;
    for (std::size_t frame = 0; frame < frames; ++frame) {
        const std::size_t offset = frame * plane;
        for (std::size_t wy = 0; wy < analysis_width; ++wy) {
            const auto dy = static_cast<std::ptrdiff_t>(wy) -
                static_cast<std::ptrdiff_t>(analysis_radius);
            for (std::size_t wx = 0; wx < analysis_width; ++wx) {
                const auto dx = static_cast<std::ptrdiff_t>(wx) -
                    static_cast<std::ptrdiff_t>(analysis_radius);
                const std::size_t sample_index =
                    offset + static_cast<std::size_t>(
                        static_cast<std::ptrdiff_t>(sample_y) + dy) * width +
                    static_cast<std::size_t>(static_cast<std::ptrdiff_t>(sample_x) + dx);
                const std::size_t ref_y = static_cast<std::size_t>(
                    static_cast<std::ptrdiff_t>(reference_y) + dy);
                const std::size_t ref_x = static_cast<std::size_t>(
                    static_cast<std::ptrdiff_t>(reference_x) + dx);
                double blurred = 0.0;
                for (std::size_t ky = 0; ky < kernel_width; ++ky) {
                    const auto sy = static_cast<std::ptrdiff_t>(ky) -
                        static_cast<std::ptrdiff_t>(kernel.kernel_radius);
                    for (std::size_t kx = 0; kx < kernel_width; ++kx) {
                        const auto sx = static_cast<std::ptrdiff_t>(kx) -
                            static_cast<std::ptrdiff_t>(kernel.kernel_radius);
                        const std::size_t index = offset +
                            static_cast<std::size_t>(static_cast<std::ptrdiff_t>(ref_y) + sy) * width +
                            static_cast<std::size_t>(static_cast<std::ptrdiff_t>(ref_x) + sx);
                        blurred += discrete_kernel[ky * kernel_width + kx] * reference_stack[index];
                    }
                }
                const double sample = sample_stack[sample_index];
                const double weight = window[wy * analysis_width + wx];
                const double residual = sample - fit.transmission * blurred;
                residual_sum += weight * residual * residual;
            }
        }
    }
    fit.residual_cost = residual_sum / fit.weight_sum;
    fit.numerical_valid = finite_value(fit.transmission) && finite_value(fit.cost) &&
        finite_value(fit.residual_cost);
    return fit;
}

}  // namespace wsvt
