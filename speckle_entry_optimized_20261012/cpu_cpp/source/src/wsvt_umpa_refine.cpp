#include "wsvt/wsvt_umpa_refine.hpp"

#include "wsvt/umpa_physical_fit.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace wsvt {

namespace {

bool is_finite_value(double value) noexcept {
    constexpr std::uint64_t exponent_mask = 0x7ff0000000000000ULL;
    return (std::bit_cast<std::uint64_t>(value) & exponent_mask) != exponent_mask;
}

std::size_t checked_product(std::size_t first, std::size_t second) {
    if (first != 0U && second > std::numeric_limits<std::size_t>::max() / first) {
        throw std::overflow_error("WG-UMPA dimensions overflow");
    }
    return first * second;
}

bool complete_patch(
    long long center_y,
    long long center_x,
    std::size_t radius,
    std::size_t height,
    std::size_t width) noexcept {
    const long long r = static_cast<long long>(radius);
    return center_y >= r && center_x >= r &&
        center_y + r < static_cast<long long>(height) &&
        center_x + r < static_cast<long long>(width);
}

UmpaPhysicalFit fit_candidate(
    std::span<const float> sample_stack,
    std::span<const float> reference_stack,
    std::size_t frames,
    std::size_t height,
    std::size_t width,
    std::size_t sample_y,
    std::size_t sample_x,
    std::size_t reference_y,
    std::size_t reference_x,
    std::size_t radius,
    std::span<const double> window,
    double relative_delta_tolerance,
    double transmission_epsilon) {
    const std::size_t plane = height * width;
    const std::size_t analysis_width = radius * 2U + 1U;
    UmpaSufficientStatistics statistics;

    for (std::size_t frame = 0; frame < frames; ++frame) {
        const std::size_t frame_offset = frame * plane;
        double reference_mean = 0.0;
        for (std::size_t wy = 0; wy < analysis_width; ++wy) {
            const std::size_t ry = reference_y + wy - radius;
            for (std::size_t wx = 0; wx < analysis_width; ++wx) {
                const std::size_t rx = reference_x + wx - radius;
                const std::size_t wi = wy * analysis_width + wx;
                reference_mean += window[wi] * static_cast<double>(
                    reference_stack[frame_offset + ry * width + rx]);
            }
        }
        for (std::size_t wy = 0; wy < analysis_width; ++wy) {
            const std::size_t sy = sample_y + wy - radius;
            const std::size_t ry = reference_y + wy - radius;
            for (std::size_t wx = 0; wx < analysis_width; ++wx) {
                const std::size_t sx = sample_x + wx - radius;
                const std::size_t rx = reference_x + wx - radius;
                const std::size_t wi = wy * analysis_width + wx;
                const double weight = window[wi];
                const double sample = static_cast<double>(
                    sample_stack[frame_offset + sy * width + sx]);
                const double reference = static_cast<double>(
                    reference_stack[frame_offset + ry * width + rx]);
                if (!is_finite_value(sample) || !is_finite_value(reference) ||
                    !is_finite_value(reference_mean)) {
                    throw std::invalid_argument(
                        "WG-UMPA raw stacks must contain finite values");
                }
                statistics.l1 += weight * sample * sample;
                statistics.l2 += weight * reference_mean * reference_mean;
                statistics.l3 += weight * reference * reference;
                statistics.l4 += weight * reference_mean * sample;
                statistics.l5 += weight * reference * sample;
                statistics.l6 += weight * reference * reference_mean;
                statistics.weight_sum += weight;
                ++statistics.observation_count;
            }
        }
    }

    UmpaPhysicalFit fit = solve_umpa_physical_fit(
        statistics, relative_delta_tolerance, transmission_epsilon);
    if (!fit.numerical_valid) {
        return fit;
    }

    double residual_sum = 0.0;
    for (std::size_t frame = 0; frame < frames; ++frame) {
        const std::size_t frame_offset = frame * plane;
        double reference_mean = 0.0;
        for (std::size_t wy = 0; wy < analysis_width; ++wy) {
            const std::size_t ry = reference_y + wy - radius;
            for (std::size_t wx = 0; wx < analysis_width; ++wx) {
                const std::size_t rx = reference_x + wx - radius;
                const std::size_t wi = wy * analysis_width + wx;
                reference_mean += window[wi] * static_cast<double>(
                    reference_stack[frame_offset + ry * width + rx]);
            }
        }
        for (std::size_t wy = 0; wy < analysis_width; ++wy) {
            const std::size_t sy = sample_y + wy - radius;
            const std::size_t ry = reference_y + wy - radius;
            for (std::size_t wx = 0; wx < analysis_width; ++wx) {
                const std::size_t sx = sample_x + wx - radius;
                const std::size_t rx = reference_x + wx - radius;
                const std::size_t wi = wy * analysis_width + wx;
                const double sample = static_cast<double>(
                    sample_stack[frame_offset + sy * width + sx]);
                const double reference = static_cast<double>(
                    reference_stack[frame_offset + ry * width + rx]);
                const double residual = sample -
                    fit.alpha * reference - fit.beta * reference_mean;
                residual_sum += window[wi] * residual * residual;
            }
        }
    }
    fit.cost = residual_sum / statistics.weight_sum;
    fit.numerical_valid = fit.numerical_valid && is_finite_value(fit.cost);
    fit.physical_valid = fit.physical_valid && fit.numerical_valid;
    return fit;
}

}  // namespace

WaveletGuidedUmpaOutput refine_wavelet_guided_umpa(
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
    const WaveletGuidedUmpaConfig& config) {
    if (!config.enabled) {
        throw std::invalid_argument("WG-UMPA refinement requires enabled=true");
    }
    if (config.local_half_window != 1) {
        throw std::invalid_argument("WG-UMPA H1 local_half_window is frozen at 1");
    }
    if (config.analysis_radius != 1U) {
        throw std::invalid_argument(
            "WG-UMPA H1 analysis_radius is frozen at 1 after the N=0 rank rejection");
    }
    if (frames == 0U || raw_height == 0U || raw_width == 0U) {
        throw std::invalid_argument("WG-UMPA raw dimensions must be positive");
    }
    const std::size_t raw_plane = checked_product(raw_height, raw_width);
    const std::size_t raw_size = checked_product(frames, raw_plane);
    if (sample_stack.size() != raw_size || reference_stack.size() != raw_size) {
        throw std::invalid_argument("WG-UMPA raw stack size mismatch");
    }
    const std::size_t output_pixels = checked_product(output_height, output_width);
    if (proposal_y.size() != output_pixels || proposal_x.size() != output_pixels) {
        throw std::invalid_argument("WG-UMPA proposal map size mismatch");
    }
    if (sample_origin_y > raw_height || sample_origin_x > raw_width ||
        output_height > raw_height - sample_origin_y ||
        output_width > raw_width - sample_origin_x) {
        throw std::out_of_range("WG-UMPA output grid is outside the raw sample stack");
    }
    if (!is_finite_value(config.relative_delta_tolerance) ||
        !is_finite_value(config.transmission_epsilon) ||
        config.relative_delta_tolerance < 0.0 ||
        config.transmission_epsilon < 0.0) {
        throw std::invalid_argument("WG-UMPA tolerances must be finite and non-negative");
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    WaveletGuidedUmpaOutput output;
    output.proposal_y.resize(output_pixels);
    output.proposal_x.resize(output_pixels);
    output.displace_y.resize(output_pixels);
    output.displace_x.resize(output_pixels);
    output.relative_offset_y.assign(output_pixels, 0.0f);
    output.relative_offset_x.assign(output_pixels, 0.0f);
    output.best_cost.assign(output_pixels, nan);
    output.second_best_cost.assign(output_pixels, nan);
    output.cost_margin.assign(output_pixels, nan);
    output.transmission.assign(output_pixels, nan);
    output.visibility.assign(output_pixels, nan);
    output.delta.assign(output_pixels, nan);
    output.condition.assign(output_pixels, nan);
    output.numerical_valid.assign(output_pixels, 0.0f);
    output.physical_valid.assign(output_pixels, 0.0f);
    output.local_boundary_hit.assign(output_pixels, 0.0f);
    output.candidates_evaluated.assign(output_pixels, 0.0f);
    for (std::size_t pixel = 0; pixel < output_pixels; ++pixel) {
        output.proposal_y[pixel] = static_cast<float>(proposal_y[pixel]);
        output.proposal_x[pixel] = static_cast<float>(proposal_x[pixel]);
        output.displace_y[pixel] = static_cast<float>(proposal_y[pixel]);
        output.displace_x[pixel] = static_cast<float>(proposal_x[pixel]);
    }

    const std::vector<double> window =
        normalized_hamming_window_2d(config.analysis_radius);
    const std::uint64_t observations_per_candidate =
        static_cast<std::uint64_t>(frames) *
        static_cast<std::uint64_t>(window.size());
    std::uint64_t raw_candidate_count = 0;
    std::uint64_t numerical_valid_pixel_count = 0;
    std::uint64_t physical_valid_pixel_count = 0;

    #pragma omp parallel for schedule(guided, 32) reduction(+:raw_candidate_count,numerical_valid_pixel_count,physical_valid_pixel_count)
    for (std::size_t pixel = 0; pixel < output_pixels; ++pixel) {
        const std::size_t output_y = pixel / output_width;
        const std::size_t output_x = pixel % output_width;
        const std::size_t sample_y = sample_origin_y + output_y;
        const std::size_t sample_x = sample_origin_x + output_x;
        double best_cost = std::numeric_limits<double>::infinity();
        double second_cost = std::numeric_limits<double>::infinity();
        int best_dy = proposal_y[pixel];
        int best_dx = proposal_x[pixel];
        UmpaPhysicalFit best_fit;
        std::uint64_t candidates = 0;

        if (complete_patch(
                static_cast<long long>(sample_y),
                static_cast<long long>(sample_x),
                config.analysis_radius, raw_height, raw_width)) {
            for (int offset_y = -config.local_half_window;
                 offset_y <= config.local_half_window; ++offset_y) {
                for (int offset_x = -config.local_half_window;
                     offset_x <= config.local_half_window; ++offset_x) {
                    const int candidate_y = proposal_y[pixel] + offset_y;
                    const int candidate_x = proposal_x[pixel] + offset_x;
                    const long long reference_y =
                        static_cast<long long>(sample_y) - candidate_y;
                    const long long reference_x =
                        static_cast<long long>(sample_x) - candidate_x;
                    if (!complete_patch(
                            reference_y, reference_x, config.analysis_radius,
                            raw_height, raw_width)) {
                        continue;
                    }
                    ++candidates;
                    const UmpaPhysicalFit fit = fit_candidate(
                        sample_stack, reference_stack, frames,
                        raw_height, raw_width, sample_y, sample_x,
                        static_cast<std::size_t>(reference_y),
                        static_cast<std::size_t>(reference_x),
                        config.analysis_radius, window,
                        config.relative_delta_tolerance,
                        config.transmission_epsilon);
                    if (!fit.numerical_valid || !is_finite_value(fit.cost)) {
                        continue;
                    }
                    if (fit.cost < best_cost) {
                        second_cost = best_cost;
                        best_cost = fit.cost;
                        best_dy = candidate_y;
                        best_dx = candidate_x;
                        best_fit = fit;
                    } else if (fit.cost < second_cost) {
                        second_cost = fit.cost;
                    }
                }
            }
        }

        raw_candidate_count += candidates;
        output.candidates_evaluated[pixel] = static_cast<float>(candidates);
        if (!is_finite_value(best_cost)) {
            continue;
        }
        output.displace_y[pixel] = static_cast<float>(best_dy);
        output.displace_x[pixel] = static_cast<float>(best_dx);
        output.relative_offset_y[pixel] = static_cast<float>(
            best_dy - proposal_y[pixel]);
        output.relative_offset_x[pixel] = static_cast<float>(
            best_dx - proposal_x[pixel]);
        output.best_cost[pixel] = static_cast<float>(best_cost);
        if (is_finite_value(second_cost)) {
            output.second_best_cost[pixel] = static_cast<float>(second_cost);
            output.cost_margin[pixel] = static_cast<float>(second_cost - best_cost);
        }
        output.transmission[pixel] = static_cast<float>(best_fit.transmission);
        output.visibility[pixel] = static_cast<float>(best_fit.visibility);
        output.delta[pixel] = static_cast<float>(best_fit.delta);
        output.condition[pixel] = static_cast<float>(best_fit.condition);
        output.numerical_valid[pixel] = 1.0f;
        output.physical_valid[pixel] = best_fit.physical_valid ? 1.0f : 0.0f;
        const int offset_y = best_dy - proposal_y[pixel];
        const int offset_x = best_dx - proposal_x[pixel];
        output.local_boundary_hit[pixel] =
            (std::abs(offset_y) == config.local_half_window ||
             std::abs(offset_x) == config.local_half_window)
            ? 1.0f
            : 0.0f;
        ++numerical_valid_pixel_count;
        if (best_fit.physical_valid) {
            ++physical_valid_pixel_count;
        }
    }

    output.raw_candidate_count = raw_candidate_count;
    output.raw_observation_count =
        raw_candidate_count * observations_per_candidate;
    output.numerical_valid_pixel_count = numerical_valid_pixel_count;
    output.physical_valid_pixel_count = physical_valid_pixel_count;
    return output;
}

}  // namespace wsvt
