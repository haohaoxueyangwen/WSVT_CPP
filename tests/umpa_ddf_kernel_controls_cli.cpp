#include "wsvt/umpa_ddf_kernel.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::size_t at(std::size_t frame, std::size_t y, std::size_t x,
               std::size_t height, std::size_t width) {
    return (frame * height + y) * width + x;
}

std::vector<double> normalized_kernel(const wsvt::UmpaDdfKernelParameters& p) {
    const std::size_t side = 2U * p.kernel_radius + 1U;
    std::vector<double> result(side * side, 0.0);
    const auto radius = static_cast<std::ptrdiff_t>(p.kernel_radius);
    double total = 0.0;
    for (std::ptrdiff_t y = -radius; y <= radius; ++y) {
        for (std::ptrdiff_t x = -radius; x <= radius; ++x) {
            const double value = std::exp(
                -p.a * static_cast<double>(y * y) -
                p.b * static_cast<double>(y * x) -
                p.c * static_cast<double>(x * x));
            result[static_cast<std::size_t>(y + radius) * side +
                   static_cast<std::size_t>(x + radius)] = value;
            total += value;
        }
    }
    for (double& value : result) value /= total;
    return result;
}

struct Stack {
    std::size_t frames = 0U;
    std::size_t height = 0U;
    std::size_t width = 0U;
    std::vector<double> reference;
    std::vector<double> sample;
};

Stack make_stack(std::size_t frames, std::size_t height, std::size_t width,
                 double transmission,
                 std::ptrdiff_t shift_y, std::ptrdiff_t shift_x,
                 wsvt::UmpaDdfKernelParameters kernel, bool scatter) {
    Stack stack{frames, height, width,
                std::vector<double>(frames * height * width, 0.0),
                std::vector<double>(frames * height * width, 0.0)};
    const auto kernel_values = normalized_kernel(kernel);
    const std::size_t side = 2U * kernel.kernel_radius + 1U;
    for (std::size_t frame = 0U; frame < frames; ++frame) {
        for (std::size_t y = 0U; y < height; ++y) {
            for (std::size_t x = 0U; x < width; ++x) {
                stack.reference[at(frame, y, x, height, width)] =
                    40.0 + 1.7 * static_cast<double>(frame) +
                    0.021 * static_cast<double>(y * y) +
                    0.037 * static_cast<double>(x * x) +
                    0.083 * static_cast<double>(x * y) +
                    std::sin(0.13 * static_cast<double>(x) +
                             0.19 * static_cast<double>(y) +
                             0.31 * static_cast<double>(frame));
            }
        }
        for (std::size_t y = 0U; y < height; ++y) {
            for (std::size_t x = 0U; x < width; ++x) {
                if (!scatter) {
                    stack.sample[at(frame, y, x, height, width)] =
                        transmission * stack.reference[at(frame, y, x, height, width)];
                    continue;
                }
                const auto source_y = static_cast<std::ptrdiff_t>(y) + shift_y;
                const auto source_x = static_cast<std::ptrdiff_t>(x) + shift_x;
                double blurred = 0.0;
                bool complete = true;
                for (std::size_t ky = 0U; ky < side && complete; ++ky) {
                    for (std::size_t kx = 0U; kx < side; ++kx) {
                        const auto sy = source_y +
                            static_cast<std::ptrdiff_t>(ky) -
                            static_cast<std::ptrdiff_t>(kernel.kernel_radius);
                        const auto sx = source_x +
                            static_cast<std::ptrdiff_t>(kx) -
                            static_cast<std::ptrdiff_t>(kernel.kernel_radius);
                        if (sy < 0 || sx < 0 || sy >= static_cast<std::ptrdiff_t>(height) ||
                            sx >= static_cast<std::ptrdiff_t>(width)) {
                            complete = false;
                            break;
                        }
                        blurred += kernel_values[ky * side + kx] *
                            stack.reference[at(frame, static_cast<std::size_t>(sy),
                                               static_cast<std::size_t>(sx), height, width)];
                    }
                }
                if (complete) {
                    stack.sample[at(frame, y, x, height, width)] = transmission * blurred;
                }
            }
        }
    }
    return stack;
}

struct Result {
    double transmission = std::numeric_limits<double>::quiet_NaN();
    double cost = std::numeric_limits<double>::quiet_NaN();
    bool valid = false;
};

Result evaluate(const Stack& stack, std::size_t y, std::size_t x,
                std::ptrdiff_t shift_y, std::ptrdiff_t shift_x,
                std::size_t analysis_radius, std::ptrdiff_t max_shift,
                wsvt::UmpaDdfKernelParameters kernel) {
    const auto fit = wsvt::fit_umpa_ddf_kernel_official_integer_at(
        stack.sample, stack.reference, stack.frames, stack.height, stack.width,
        y, x, shift_y, shift_x, analysis_radius, max_shift, kernel);
    return {fit.transmission, fit.cost, fit.numerical_valid};
}

void emit_number(std::ostringstream& out, double value) {
    if (std::isfinite(value)) out << std::setprecision(17) << value;
    else out << "null";
}

void emit_result(std::ostringstream& out, const Result& result) {
    out << "{\"valid\":" << (result.valid ? "true" : "false")
        << ",\"T\":";
    emit_number(out, result.transmission);
    out << ",\"cost\":";
    emit_number(out, result.cost);
    out << '}';
}

}  // namespace

int main() try {
    constexpr std::size_t frames = 5U;
    constexpr std::size_t height = 96U;
    constexpr std::size_t width = 99U;
    constexpr std::size_t y0 = 48U;
    constexpr std::size_t x0 = 49U;
    constexpr std::size_t analysis_radius = 2U;
    constexpr std::ptrdiff_t max_shift = 6;
    constexpr double transmission = 0.73;
    const wsvt::UmpaDdfKernelParameters truth{8U, 0.42, 0.06, 0.71};
    const wsvt::UmpaDdfKernelParameters wrong{8U, 0.85, -0.12, 0.34};
    const wsvt::UmpaDdfKernelParameters delta{0U, 1.0, 0.0, 1.0};
    const std::ptrdiff_t shift_y = 2;
    const std::ptrdiff_t shift_x = -3;
    const Stack known = make_stack(frames, height, width, transmission,
                                   shift_y, shift_x, truth, true);
    const Result exact = evaluate(known, y0, x0, shift_y, shift_x,
                                  analysis_radius, max_shift, truth);
    const Result wrong_kernel = evaluate(known, y0, x0, shift_y, shift_x,
                                         analysis_radius, max_shift, wrong);
    const Result wrong_sign = evaluate(known, y0, x0, -shift_y, -shift_x,
                                       analysis_radius, max_shift, truth);

    const Stack no_scatter = make_stack(frames, height, width, transmission,
                                        0, 0, truth, false);
    const Result zero_delta = evaluate(no_scatter, y0, x0, 0, 0,
                                       analysis_radius, max_shift, delta);
    const Result zero_broad = evaluate(no_scatter, y0, x0, 0, 0,
                                       analysis_radius, max_shift, truth);

    bool open_boundary = false;
    try {
        (void)evaluate(known, y0, x0, max_shift, 0,
                       analysis_radius, max_shift, truth);
    } catch (const std::out_of_range&) {
        open_boundary = true;
    }
    bool halo_boundary = false;
    try {
        (void)evaluate(known, analysis_radius + truth.kernel_radius + max_shift - 1U,
                       x0, 0, 0, analysis_radius, max_shift, truth);
    } catch (const std::out_of_range&) {
        halo_boundary = true;
    }

    const bool gates = exact.valid && zero_delta.valid &&
        std::abs(exact.transmission - transmission) < 1.0e-11 &&
        std::abs(zero_delta.transmission - transmission) < 1.0e-11 &&
        exact.cost < 1.0e-9 && zero_delta.cost < 1.0e-9 &&
        wrong_kernel.cost > exact.cost + 1.0e-8 &&
        wrong_sign.cost > exact.cost + 1.0e-8 &&
        zero_broad.cost > zero_delta.cost + 1.0e-8 &&
        open_boundary && halo_boundary;

    std::ostringstream out;
    out << "{\"schema_version\":1,\"experiment_id\":\"umpapp_ddf_kernel_controls_v1\","
        << "\"status\":\"" << (gates ? "passed" : "failed") << "\","
        << "\"claim_boundary\":\"known synthetic controls for the fixed official operator; no real-data accuracy or production acceptance\","
        << "\"cases\":{\"known_kernel_exact\":";
    emit_result(out, exact);
    out << ",\"known_kernel_wrong_parameters\":";
    emit_result(out, wrong_kernel);
    out << ",\"known_kernel_wrong_sign\":";
    emit_result(out, wrong_sign);
    out << ",\"zero_scatter_delta\":";
    emit_result(out, zero_delta);
    out << ",\"zero_scatter_broad_kernel\":";
    emit_result(out, zero_broad);
    out << "},\"gates\":{\"exact_T_error\":";
    emit_number(out, std::abs(exact.transmission - transmission));
    out << ",\"zero_delta_T_error\":";
    emit_number(out, std::abs(zero_delta.transmission - transmission));
    out << ",\"open_shift_rejected\":" << (open_boundary ? "true" : "false")
        << ",\"incomplete_halo_rejected\":" << (halo_boundary ? "true" : "false")
        << ",\"wrong_kernel_cost_separated\":"
        << ((wrong_kernel.cost > exact.cost + 1.0e-8) ? "true" : "false")
        << ",\"wrong_sign_cost_separated\":"
        << ((wrong_sign.cost > exact.cost + 1.0e-8) ? "true" : "false")
        << ",\"zero_scatter_broad_cost_separated\":"
        << ((zero_broad.cost > zero_delta.cost + 1.0e-8) ? "true" : "false")
        << "},\"production_pipeline_changed\":false}\n";
    std::cout << out.str();
    return gates ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
