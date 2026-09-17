#include "wsvt/wsvt_umpa_refine.hpp"

#include "wsvt/umpa_physical_fit.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

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
    return fit_umpa_physical_float_candidate_at(
        sample_stack, reference_stack, frames, height, width,
        sample_y, sample_x, reference_y, reference_x, radius, window,
        relative_delta_tolerance, transmission_epsilon);
}
UmpaPhysicalFit fit_model_t_candidate(
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
    double transmission_epsilon) {
    const std::size_t plane = height * width;
    const std::size_t analysis_width = radius * 2U + 1U;
    double sample_squared = 0.0;
    double reference_squared = 0.0;
    double sample_reference = 0.0;
    double weight_sum = 0.0;
    std::size_t observation_count = 0;
    for (std::size_t frame = 0; frame < frames; ++frame) {
        const std::size_t frame_offset = frame * plane;
        for (std::size_t wy = 0; wy < analysis_width; ++wy) {
            const std::size_t sy = sample_y + wy - radius;
            const std::size_t ry = reference_y + wy - radius;
            for (std::size_t wx = 0; wx < analysis_width; ++wx) {
                const std::size_t sx = sample_x + wx - radius;
                const std::size_t rx = reference_x + wx - radius;
                const double weight = window[wy * analysis_width + wx];
                const double sample = static_cast<double>(
                    sample_stack[frame_offset + sy * width + sx]);
                const double reference = static_cast<double>(
                    reference_stack[frame_offset + ry * width + rx]);
                if (!is_finite_value(sample) || !is_finite_value(reference)) {
                    throw std::invalid_argument(
                        "SET4 raw stacks must contain finite values");
                }
                sample_squared += weight * sample * sample;
                reference_squared += weight * reference * reference;
                sample_reference += weight * sample * reference;
                weight_sum += weight;
                ++observation_count;
            }
        }
    }

    UmpaPhysicalFit fit;
    fit.statistics.l1 = sample_squared;
    fit.statistics.l3 = reference_squared;
    fit.statistics.l5 = sample_reference;
    fit.statistics.weight_sum = weight_sum;
    fit.statistics.observation_count = observation_count;
    fit.delta = reference_squared;
    fit.condition = 1.0;
    const double scale = std::max(1.0, sample_squared);
    if (!(weight_sum > 0.0) ||
        !(reference_squared > transmission_epsilon * scale) ||
        !is_finite_value(reference_squared) ||
        !is_finite_value(sample_reference)) {
        fit.alpha = std::numeric_limits<double>::quiet_NaN();
        fit.beta = std::numeric_limits<double>::quiet_NaN();
        fit.transmission = std::numeric_limits<double>::quiet_NaN();
        fit.visibility = std::numeric_limits<double>::quiet_NaN();
        fit.cost = std::numeric_limits<double>::quiet_NaN();
        return fit;
    }
    fit.transmission = sample_reference / reference_squared;
    fit.alpha = fit.transmission;
    fit.beta = 0.0;
    fit.visibility = 1.0;
    const double residual = sample_squared -
        2.0 * fit.transmission * sample_reference +
        fit.transmission * fit.transmission * reference_squared;
    fit.cost = std::max(0.0, residual) / weight_sum;
    fit.numerical_valid = is_finite_value(fit.transmission) &&
        is_finite_value(fit.cost);
    fit.physical_valid = fit.numerical_valid && fit.transmission > 0.0;
    return fit;
}

UmpaPhysicalFit fit_windowed_zncc_candidate(
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
    double variance_epsilon) {
    const std::size_t plane = height * width;
    const std::size_t analysis_width = radius * 2U + 1U;
    double sample_sum = 0.0;
    double reference_sum = 0.0;
    double sample_squared = 0.0;
    double reference_squared = 0.0;
    double sample_reference = 0.0;
    double weight_sum = 0.0;
    std::size_t observation_count = 0;
    for (std::size_t frame = 0; frame < frames; ++frame) {
        const std::size_t frame_offset = frame * plane;
        for (std::size_t wy = 0; wy < analysis_width; ++wy) {
            const std::size_t sy = sample_y + wy - radius;
            const std::size_t ry = reference_y + wy - radius;
            for (std::size_t wx = 0; wx < analysis_width; ++wx) {
                const std::size_t sx = sample_x + wx - radius;
                const std::size_t rx = reference_x + wx - radius;
                const double weight = window[wy * analysis_width + wx];
                const double sample = static_cast<double>(
                    sample_stack[frame_offset + sy * width + sx]);
                const double reference = static_cast<double>(
                    reference_stack[frame_offset + ry * width + rx]);
                if (!is_finite_value(sample) || !is_finite_value(reference)) {
                    throw std::invalid_argument(
                        "SET4 raw stacks must contain finite values");
                }
                sample_sum += weight * sample;
                reference_sum += weight * reference;
                sample_squared += weight * sample * sample;
                reference_squared += weight * reference * reference;
                sample_reference += weight * sample * reference;
                weight_sum += weight;
                ++observation_count;
            }
        }
    }

    UmpaPhysicalFit fit;
    fit.statistics.l1 = sample_squared;
    fit.statistics.l3 = reference_squared;
    fit.statistics.l5 = sample_reference;
    fit.statistics.weight_sum = weight_sum;
    fit.statistics.observation_count = observation_count;
    if (!(weight_sum > 0.0)) {
        fit.cost = std::numeric_limits<double>::quiet_NaN();
        return fit;
    }
    const double covariance = sample_reference -
        sample_sum * reference_sum / weight_sum;
    const double sample_variance = std::max(
        0.0, sample_squared - sample_sum * sample_sum / weight_sum);
    const double reference_variance = std::max(
        0.0, reference_squared - reference_sum * reference_sum / weight_sum);
    const double variance_scale = std::max({
        sample_squared, reference_squared, 1.0});
    const double denominator = std::sqrt(sample_variance * reference_variance);
    if (!(sample_variance > variance_epsilon * variance_scale) ||
        !(reference_variance > variance_epsilon * variance_scale) ||
        !(denominator > 0.0) || !is_finite_value(denominator)) {
        fit.cost = std::numeric_limits<double>::quiet_NaN();
        return fit;
    }
    const double correlation = std::clamp(covariance / denominator, -1.0, 1.0);
    fit.cost = 1.0 - correlation;
    fit.transmission = std::sqrt(sample_variance / reference_variance);
    fit.visibility = std::numeric_limits<double>::quiet_NaN();
    fit.delta = denominator;
    fit.condition = 1.0;
    fit.numerical_valid = is_finite_value(fit.cost) &&
        is_finite_value(fit.transmission);
    fit.physical_valid = false;
    return fit;
}

UmpaPhysicalFit score_raw_candidate(
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
    std::size_t radius,
    std::span<const double> window,
    const SetTransportRawRerankConfig& config) {
    switch (objective) {
    case SetTransportRawObjective::WindowedZncc:
        return fit_windowed_zncc_candidate(
            sample_stack, reference_stack, frames, height, width,
            sample_y, sample_x, reference_y, reference_x,
            radius, window, config.variance_epsilon);
    case SetTransportRawObjective::ModelT:
        return fit_model_t_candidate(
            sample_stack, reference_stack, frames, height, width,
            sample_y, sample_x, reference_y, reference_x,
            radius, window, config.transmission_epsilon);
    case SetTransportRawObjective::ModelDF:
        return fit_candidate(
            sample_stack, reference_stack, frames, height, width,
            sample_y, sample_x, reference_y, reference_x,
            radius, window, config.relative_delta_tolerance,
            config.transmission_epsilon);
    }
    throw std::invalid_argument("unknown SET4 raw objective");
}

}  // namespace

const char* set_transport_raw_objective_name(
    SetTransportRawObjective objective) noexcept {
    switch (objective) {
    case SetTransportRawObjective::WindowedZncc:
        return "windowed_zncc";
    case SetTransportRawObjective::ModelT:
        return "ModelT";
    case SetTransportRawObjective::ModelDF:
        return "ModelDF";
    }
    return "unknown";
}

UmpaPhysicalFit evaluate_set_transport_raw_candidate(
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
    double relative_delta_tolerance,
    double transmission_epsilon,
    double variance_epsilon) {
    SetTransportRawRerankConfig config;
    config.objective = objective;
    config.analysis_radius = analysis_radius;
    config.relative_delta_tolerance = relative_delta_tolerance;
    config.transmission_epsilon = transmission_epsilon;
    config.variance_epsilon = variance_epsilon;
    return score_raw_candidate(
        objective, sample_stack, reference_stack, frames, height, width,
        sample_y, sample_x, reference_y, reference_x, analysis_radius,
        normalized_window, config);
}

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

WaveletGuidedUmpaOutput rerank_set_transport_raw(
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
    const SetTransportRawRerankConfig& config) {
    constexpr std::size_t kSetWidth = 4U;
    if (!config.enabled) {
        throw std::invalid_argument(
            "SET4 raw rerank requires enabled=true");
    }
    if (config.domain_half_window != 4 || config.analysis_radius != 1U) {
        throw std::invalid_argument(
            "SET4 raw rerank v1 freezes domain_half_window=4 and analysis_radius=1");
    }
    if (frames == 0U || raw_height == 0U || raw_width == 0U) {
        throw std::invalid_argument("SET4 raw dimensions must be positive");
    }
    if (!is_finite_value(config.relative_delta_tolerance) ||
        !is_finite_value(config.transmission_epsilon) ||
        !is_finite_value(config.variance_epsilon) ||
        config.relative_delta_tolerance < 0.0 ||
        config.transmission_epsilon < 0.0 ||
        config.variance_epsilon < 0.0) {
        throw std::invalid_argument(
            "SET4 raw tolerances must be finite and non-negative");
    }
    const std::size_t raw_plane = checked_product(raw_height, raw_width);
    const std::size_t raw_size = checked_product(frames, raw_plane);
    if (sample_stack.size() != raw_size || reference_stack.size() != raw_size) {
        throw std::invalid_argument("SET4 raw stack size mismatch");
    }
    const std::size_t output_pixels = checked_product(output_height, output_width);
    if (set_center_y.size() != output_pixels * kSetWidth ||
        set_center_x.size() != output_pixels * kSetWidth ||
        fallback_y.size() != output_pixels || fallback_x.size() != output_pixels) {
        throw std::invalid_argument("SET4 raw candidate/fallback shape mismatch");
    }
    if (config.representatives_only &&
        (set_representative_y.size() != output_pixels * kSetWidth ||
         set_representative_x.size() != output_pixels * kSetWidth)) {
        throw std::invalid_argument(
            "SET4 REP4 raw rerank requires four representatives per pixel");
    }
    if (sample_origin_y > raw_height || sample_origin_x > raw_width ||
        output_height > raw_height - sample_origin_y ||
        output_width > raw_width - sample_origin_x) {
        throw std::out_of_range("SET4 raw output grid is outside the raw stack");
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    WaveletGuidedUmpaOutput output;
    output.proposal_y.assign(fallback_y.begin(), fallback_y.end());
    output.proposal_x.assign(fallback_x.begin(), fallback_x.end());
    output.displace_y.assign(fallback_y.begin(), fallback_y.end());
    output.displace_x.assign(fallback_x.begin(), fallback_x.end());
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

    const std::vector<double> window =
        normalized_hamming_window_2d(config.analysis_radius);
    const std::uint64_t observations_per_candidate =
        static_cast<std::uint64_t>(frames) *
        static_cast<std::uint64_t>(window.size());
    std::uint64_t raw_candidate_count = 0;
    std::uint64_t numerical_valid_pixel_count = 0;
    std::uint64_t physical_valid_pixel_count = 0;

    #pragma omp parallel for schedule(guided, 16) reduction(+:raw_candidate_count,numerical_valid_pixel_count,physical_valid_pixel_count)
    for (std::size_t pixel = 0; pixel < output_pixels; ++pixel) {
        const std::size_t output_y = pixel / output_width;
        const std::size_t output_x = pixel % output_width;
        const std::size_t sample_y = sample_origin_y + output_y;
        const std::size_t sample_x = sample_origin_x + output_x;
        if (!complete_patch(
                static_cast<long long>(sample_y),
                static_cast<long long>(sample_x), config.analysis_radius,
                raw_height, raw_width)) {
            continue;
        }

        std::vector<std::pair<int, int>> candidates;
        candidates.reserve(config.representatives_only
            ? kSetWidth
            : kSetWidth * static_cast<std::size_t>(
                (2 * config.domain_half_window + 1) *
                (2 * config.domain_half_window + 1)));
        const std::size_t center_base = pixel * kSetWidth;
        if (config.representatives_only) {
            for (std::size_t hypothesis = 0; hypothesis < kSetWidth; ++hypothesis) {
                candidates.emplace_back(
                    set_representative_y[center_base + hypothesis],
                    set_representative_x[center_base + hypothesis]);
            }
        } else {
            for (std::size_t hypothesis = 0; hypothesis < kSetWidth; ++hypothesis) {
                const int center_y = set_center_y[center_base + hypothesis];
                const int center_x = set_center_x[center_base + hypothesis];
                for (int local_y = -config.domain_half_window;
                     local_y <= config.domain_half_window; ++local_y) {
                    for (int local_x = -config.domain_half_window;
                         local_x <= config.domain_half_window; ++local_x) {
                        candidates.emplace_back(
                            center_y + local_y, center_x + local_x);
                    }
                }
            }
        }
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(
            std::unique(candidates.begin(), candidates.end()), candidates.end());

        double best_cost = std::numeric_limits<double>::infinity();
        double second_cost = std::numeric_limits<double>::infinity();
        int best_y = 0;
        int best_x = 0;
        UmpaPhysicalFit best_fit;
        std::uint64_t evaluated = 0;
        for (const auto& [candidate_y, candidate_x] : candidates) {
            const long long reference_y =
                static_cast<long long>(sample_y) - candidate_y;
            const long long reference_x =
                static_cast<long long>(sample_x) - candidate_x;
            if (!complete_patch(
                    reference_y, reference_x, config.analysis_radius,
                    raw_height, raw_width)) {
                continue;
            }
            ++evaluated;
            const UmpaPhysicalFit fit = score_raw_candidate(
                config.objective, sample_stack, reference_stack, frames,
                raw_height, raw_width, sample_y, sample_x,
                static_cast<std::size_t>(reference_y),
                static_cast<std::size_t>(reference_x),
                config.analysis_radius, window, config);
            if (!fit.numerical_valid || !is_finite_value(fit.cost)) {
                continue;
            }
            if (fit.cost < best_cost) {
                second_cost = best_cost;
                best_cost = fit.cost;
                best_y = candidate_y;
                best_x = candidate_x;
                best_fit = fit;
            } else if (fit.cost < second_cost) {
                second_cost = fit.cost;
            }
        }

        raw_candidate_count += evaluated;
        output.candidates_evaluated[pixel] = static_cast<float>(evaluated);
        if (!is_finite_value(best_cost)) {
            continue;
        }
        output.displace_y[pixel] = static_cast<float>(best_y);
        output.displace_x[pixel] = static_cast<float>(best_x);
        output.relative_offset_y[pixel] =
            static_cast<float>(best_y) - fallback_y[pixel];
        output.relative_offset_x[pixel] =
            static_cast<float>(best_x) - fallback_x[pixel];
        output.best_cost[pixel] = static_cast<float>(best_cost);
        if (is_finite_value(second_cost)) {
            output.second_best_cost[pixel] = static_cast<float>(second_cost);
            output.cost_margin[pixel] =
                static_cast<float>(second_cost - best_cost);
        }
        output.transmission[pixel] =
            static_cast<float>(best_fit.transmission);
        output.visibility[pixel] = static_cast<float>(best_fit.visibility);
        output.delta[pixel] = static_cast<float>(best_fit.delta);
        output.condition[pixel] = static_cast<float>(best_fit.condition);
        output.numerical_valid[pixel] = 1.0f;
        output.physical_valid[pixel] = best_fit.physical_valid ? 1.0f : 0.0f;

        bool belongs_to_domain = false;
        bool has_interior_owner = false;
        for (std::size_t hypothesis = 0; hypothesis < kSetWidth; ++hypothesis) {
            const int offset_y = best_y - set_center_y[center_base + hypothesis];
            const int offset_x = best_x - set_center_x[center_base + hypothesis];
            if (std::abs(offset_y) <= config.domain_half_window &&
                std::abs(offset_x) <= config.domain_half_window) {
                belongs_to_domain = true;
                if (std::abs(offset_y) < config.domain_half_window &&
                    std::abs(offset_x) < config.domain_half_window) {
                    has_interior_owner = true;
                }
            }
        }
        output.local_boundary_hit[pixel] =
            belongs_to_domain && !has_interior_owner ? 1.0f : 0.0f;
        ++numerical_valid_pixel_count;
        if (best_fit.physical_valid) {
            ++physical_valid_pixel_count;
        }
    }

    output.raw_candidate_count = raw_candidate_count;
    output.raw_observation_count = raw_candidate_count * observations_per_candidate;
    output.numerical_valid_pixel_count = numerical_valid_pixel_count;
    output.physical_valid_pixel_count = physical_valid_pixel_count;
    return output;
}

}  // namespace wsvt
