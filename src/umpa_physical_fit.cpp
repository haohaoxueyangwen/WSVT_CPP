#include "wsvt/umpa_physical_fit.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace wsvt {

namespace {

double quiet_nan() noexcept {
    return std::numeric_limits<double>::quiet_NaN();
}

double positive_infinity() noexcept {
    return std::numeric_limits<double>::infinity();
}

bool is_finite_value(double value) noexcept {
    constexpr std::uint64_t exponent_mask = 0x7ff0000000000000ULL;
    return (std::bit_cast<std::uint64_t>(value) & exponent_mask) != exponent_mask;
}

bool is_visibility_physical(double visibility) noexcept {
    const double slack = 64.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::fabs(visibility));
    return visibility >= -slack && visibility <= 1.0 + slack;
}

void validate_tolerances(
    double relative_delta_tolerance, double transmission_epsilon) {
    if (!is_finite_value(relative_delta_tolerance) ||
        !is_finite_value(transmission_epsilon) ||
        relative_delta_tolerance < 0.0 || transmission_epsilon < 0.0) {
        throw std::invalid_argument("UMPA fit tolerances must be finite and non-negative");
    }
}

std::size_t checked_product(std::size_t first, std::size_t second) {
    if (first != 0U && second > std::numeric_limits<std::size_t>::max() / first) {
        throw std::overflow_error("UMPA image dimensions overflow");
    }
    return first * second;
}

std::size_t shifted_center(
    std::size_t sample_center,
    std::ptrdiff_t displacement,
    std::size_t extent) {
    if (displacement >= 0) {
        const std::size_t amount = static_cast<std::size_t>(displacement);
        if (amount > sample_center) {
            throw std::out_of_range("UMPA shifted reference center is out of bounds");
        }
        return sample_center - amount;
    }
    const std::size_t amount =
        static_cast<std::size_t>(-(displacement + 1)) + 1U;
    if (amount >= extent || sample_center >= extent - amount) {
        throw std::out_of_range("UMPA shifted reference center is out of bounds");
    }
    return sample_center + amount;
}

void validate_patch_center(
    std::size_t center,
    std::size_t extent,
    std::size_t radius,
    const char* message) {
    if (extent == 0U || radius > (extent - 1U) / 2U ||
        center < radius || center >= extent - radius) {
        throw std::out_of_range(message);
    }
}

}  // namespace

std::vector<double> normalized_hamming_window_2d(std::size_t radius) {
    if (radius > (std::numeric_limits<std::size_t>::max() - 1U) / 2U) {
        throw std::overflow_error("UMPA analysis radius is too large");
    }
    const std::size_t width = radius * 2U + 1U;
    std::vector<double> one_d(width, 1.0);
    if (width > 1U) {
        const double denominator = static_cast<double>(width - 1U);
        for (std::size_t index = 0; index < width; ++index) {
            one_d[index] = 0.54 - 0.46 * std::cos(
                2.0 * std::numbers::pi_v<double> *
                static_cast<double>(index) / denominator);
        }
    }
    std::vector<double> window(width * width, 0.0);
    double total = 0.0;
    for (std::size_t y = 0; y < width; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const double value = one_d[y] * one_d[x];
            window[y * width + x] = value;
            total += value;
        }
    }
    if (!(total > 0.0) || !is_finite_value(total)) {
        throw std::runtime_error("UMPA Hamming window normalization failed");
    }
    for (double& value : window) {
        value /= total;
    }
    return window;
}

UmpaSufficientStatistics accumulate_umpa_statistics(
    std::span<const double> sample,
    std::span<const double> reference,
    std::span<const double> reference_mean,
    std::span<const double> weights) {
    if (sample.empty()) {
        throw std::invalid_argument("UMPA fit requires at least one observation");
    }
    if (sample.size() != reference.size() || sample.size() != reference_mean.size()) {
        throw std::invalid_argument("UMPA sample/reference/reference_mean size mismatch");
    }
    if (!weights.empty() && weights.size() != sample.size()) {
        throw std::invalid_argument("UMPA weights size mismatch");
    }

    UmpaSufficientStatistics statistics;
    for (std::size_t index = 0; index < sample.size(); ++index) {
        const double weight = weights.empty() ? 1.0 : weights[index];
        if (!is_finite_value(weight) || weight < 0.0) {
            throw std::invalid_argument("UMPA weights must be finite and non-negative");
        }
        if (weight == 0.0) {
            continue;
        }
        const double sample_value = sample[index];
        const double reference_value = reference[index];
        const double mean_value = reference_mean[index];
        if (!is_finite_value(sample_value) || !is_finite_value(reference_value) ||
            !is_finite_value(mean_value)) {
            throw std::invalid_argument("active UMPA observations must be finite");
        }
        statistics.l1 += weight * sample_value * sample_value;
        statistics.l2 += weight * mean_value * mean_value;
        statistics.l3 += weight * reference_value * reference_value;
        statistics.l4 += weight * mean_value * sample_value;
        statistics.l5 += weight * reference_value * sample_value;
        statistics.l6 += weight * reference_value * mean_value;
        statistics.weight_sum += weight;
        ++statistics.observation_count;
    }
    if (!(statistics.weight_sum > 0.0)) {
        throw std::invalid_argument("UMPA fit requires at least one positive weight");
    }
    return statistics;
}

UmpaPhysicalFit solve_umpa_physical_fit(
    const UmpaSufficientStatistics& statistics,
    double relative_delta_tolerance,
    double transmission_epsilon) {
    validate_tolerances(relative_delta_tolerance, transmission_epsilon);
    UmpaPhysicalFit fit;
    fit.statistics = statistics;
    fit.delta = statistics.l3 * statistics.l2 - statistics.l6 * statistics.l6;

    const double trace = statistics.l3 + statistics.l2;
    const double discriminant = std::hypot(
        statistics.l3 - statistics.l2, 2.0 * statistics.l6);
    const double eigen_max = 0.5 * (trace + discriminant);
    const double eigen_min = 0.5 * (trace - discriminant);
    fit.condition = eigen_min > 0.0 && is_finite_value(eigen_max)
        ? eigen_max / eigen_min
        : positive_infinity();

    const bool statistics_finite =
        is_finite_value(statistics.l1) && is_finite_value(statistics.l2) &&
        is_finite_value(statistics.l3) && is_finite_value(statistics.l4) &&
        is_finite_value(statistics.l5) && is_finite_value(statistics.l6) &&
        is_finite_value(statistics.weight_sum);
    const double scale = std::max({
        std::fabs(statistics.l3 * statistics.l2),
        std::fabs(statistics.l6 * statistics.l6),
        std::numeric_limits<double>::min(),
    });
    if (!statistics_finite || !(statistics.weight_sum > 0.0) ||
        !is_finite_value(fit.delta) ||
        !(fit.delta > relative_delta_tolerance * scale)) {
        fit.alpha = quiet_nan();
        fit.beta = quiet_nan();
        fit.transmission = quiet_nan();
        fit.visibility = quiet_nan();
        fit.cost = quiet_nan();
        return fit;
    }

    fit.alpha = (
        statistics.l2 * statistics.l5 - statistics.l4 * statistics.l6
    ) / fit.delta;
    fit.beta = (
        statistics.l3 * statistics.l4 - statistics.l5 * statistics.l6
    ) / fit.delta;
    fit.transmission = fit.alpha + fit.beta;
    fit.numerical_valid =
        is_finite_value(fit.alpha) && is_finite_value(fit.beta) &&
        is_finite_value(fit.transmission) &&
        std::fabs(fit.transmission) > transmission_epsilon;
    fit.visibility = fit.numerical_valid
        ? fit.alpha / fit.transmission
        : quiet_nan();
    fit.cost = (
        statistics.l1 + fit.beta * fit.beta * statistics.l2 +
        fit.alpha * fit.alpha * statistics.l3 -
        2.0 * fit.beta * statistics.l4 -
        2.0 * fit.alpha * statistics.l5 +
        2.0 * fit.beta * fit.alpha * statistics.l6
    ) / statistics.weight_sum;
    fit.numerical_valid = fit.numerical_valid &&
        is_finite_value(fit.visibility) && is_finite_value(fit.cost);
    fit.physical_valid = fit.numerical_valid && fit.transmission > 0.0 &&
        is_visibility_physical(fit.visibility);
    return fit;
}

UmpaPhysicalFit fit_umpa_physical(
    std::span<const double> sample,
    std::span<const double> reference,
    std::span<const double> reference_mean,
    std::span<const double> weights,
    double relative_delta_tolerance,
    double transmission_epsilon) {
    const UmpaSufficientStatistics statistics = accumulate_umpa_statistics(
        sample, reference, reference_mean, weights);
    UmpaPhysicalFit fit = solve_umpa_physical_fit(
        statistics, relative_delta_tolerance, transmission_epsilon);
    if (!fit.numerical_valid) {
        return fit;
    }
    double residual_sum = 0.0;
    for (std::size_t index = 0; index < sample.size(); ++index) {
        const double weight = weights.empty() ? 1.0 : weights[index];
        if (weight == 0.0) {
            continue;
        }
        const double residual = sample[index] -
            fit.alpha * reference[index] - fit.beta * reference_mean[index];
        residual_sum += weight * residual * residual;
    }
    fit.cost = residual_sum / statistics.weight_sum;
    fit.numerical_valid = fit.numerical_valid && is_finite_value(fit.cost);
    fit.physical_valid = fit.numerical_valid && fit.transmission > 0.0 &&
        is_visibility_physical(fit.visibility);
    return fit;
}

UmpaPhysicalFit fit_umpa_physical_integer_at(
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
    double relative_delta_tolerance,
    double transmission_epsilon) {
    validate_tolerances(relative_delta_tolerance, transmission_epsilon);
    if (frames == 0U) {
        throw std::invalid_argument("UMPA image fit requires at least one frame");
    }
    const std::size_t plane_size = checked_product(height, width);
    const std::size_t stack_size = checked_product(frames, plane_size);
    if (sample_stack.size() != stack_size || reference_stack.size() != stack_size) {
        throw std::invalid_argument("UMPA image stack size does not match dimensions");
    }
    validate_patch_center(
        sample_y, height, analysis_radius,
        "UMPA sample analysis patch is out of bounds");
    validate_patch_center(
        sample_x, width, analysis_radius,
        "UMPA sample analysis patch is out of bounds");
    const std::size_t reference_y = shifted_center(
        sample_y, displacement_y, height);
    const std::size_t reference_x = shifted_center(
        sample_x, displacement_x, width);
    validate_patch_center(
        reference_y, height, analysis_radius,
        "UMPA reference analysis patch is out of bounds");
    validate_patch_center(
        reference_x, width, analysis_radius,
        "UMPA reference analysis patch is out of bounds");

    const std::vector<double> window =
        normalized_hamming_window_2d(analysis_radius);
    const std::size_t analysis_width = analysis_radius * 2U + 1U;
    const std::size_t patch_size = checked_product(analysis_width, analysis_width);
    const std::size_t observation_count = checked_product(frames, patch_size);
    std::vector<double> sample;
    std::vector<double> reference;
    std::vector<double> reference_mean;
    std::vector<double> weights;
    sample.reserve(observation_count);
    reference.reserve(observation_count);
    reference_mean.reserve(observation_count);
    weights.reserve(observation_count);

    for (std::size_t frame = 0; frame < frames; ++frame) {
        const std::size_t frame_offset = frame * plane_size;
        double mean = 0.0;
        for (std::size_t wy = 0; wy < analysis_width; ++wy) {
            const std::size_t ry = reference_y + wy - analysis_radius;
            for (std::size_t wx = 0; wx < analysis_width; ++wx) {
                const std::size_t rx = reference_x + wx - analysis_radius;
                const std::size_t window_index = wy * analysis_width + wx;
                mean += window[window_index] *
                    reference_stack[frame_offset + ry * width + rx];
            }
        }
        for (std::size_t wy = 0; wy < analysis_width; ++wy) {
            const std::size_t sy = sample_y + wy - analysis_radius;
            const std::size_t ry = reference_y + wy - analysis_radius;
            for (std::size_t wx = 0; wx < analysis_width; ++wx) {
                const std::size_t sx = sample_x + wx - analysis_radius;
                const std::size_t rx = reference_x + wx - analysis_radius;
                const std::size_t window_index = wy * analysis_width + wx;
                sample.push_back(sample_stack[frame_offset + sy * width + sx]);
                reference.push_back(reference_stack[frame_offset + ry * width + rx]);
                reference_mean.push_back(mean);
                weights.push_back(window[window_index]);
            }
        }
    }
    return fit_umpa_physical(
        sample, reference, reference_mean, weights,
        relative_delta_tolerance, transmission_epsilon);
}

}  // namespace wsvt
