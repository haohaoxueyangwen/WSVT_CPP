#include "wsvt/umpa_ddf_kernel_optimizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace wsvt {
namespace {

using State = std::array<double, 3>;

bool finite_value(double value) noexcept {
    return std::isfinite(value);
}

double clamp_value(double value, double lower, double upper) {
    return std::max(lower, std::min(upper, value));
}

State kernel_to_state(const UmpaDdfKernelParameters& kernel) {
    if (!finite_value(kernel.a) || !finite_value(kernel.b) ||
        !finite_value(kernel.c) || kernel.a <= 0.0 || kernel.c <= 0.0 ||
        4.0 * kernel.a * kernel.c <= kernel.b * kernel.b) {
        throw std::invalid_argument("initial UMPA-DDF kernel is not positive definite");
    }
    const double rho = kernel.b / (2.0 * std::sqrt(kernel.a * kernel.c));
    if (!finite_value(rho) || std::abs(rho) >= 1.0) {
        throw std::invalid_argument("initial UMPA-DDF kernel correlation is invalid");
    }
    return {std::log(kernel.a), std::log(kernel.c),
            std::atanh(clamp_value(rho, -0.999999, 0.999999))};
}

UmpaDdfKernelParameters state_to_kernel(const State& state, std::size_t radius) {
    const double log_a = clamp_value(state[0], -6.0, 6.0);
    const double log_c = clamp_value(state[1], -6.0, 6.0);
    const double a = std::exp(log_a);
    const double c = std::exp(log_c);
    const double rho = 0.999 * std::tanh(clamp_value(state[2], -6.0, 6.0));
    return UmpaDdfKernelParameters{radius, a, 2.0 * std::sqrt(a * c) * rho, c};
}

struct Vertex {
    State state{};
    UmpaDdfKernelFit fit{};
    double objective = std::numeric_limits<double>::infinity();
};

bool better(const Vertex& lhs, const Vertex& rhs) {
    if (lhs.objective != rhs.objective) return lhs.objective < rhs.objective;
    return lhs.state < rhs.state;
}

double state_span(const State& a, const State& b) {
    double result = 0.0;
    for (std::size_t i = 0U; i < a.size(); ++i) {
        result = std::max(result, std::abs(a[i] - b[i]));
    }
    return result;
}

double simplex_span(const std::vector<Vertex>& simplex) {
    double result = 0.0;
    for (std::size_t i = 1U; i < simplex.size(); ++i) {
        result = std::max(result, state_span(simplex[0].state, simplex[i].state));
    }
    return result;
}

}  // namespace

UmpaDdfKernelOptimization optimize_umpa_ddf_kernel_fixed_u(
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
    UmpaDdfKernelOptimizerParameters settings,
    UmpaAssignCoordinates assign_coordinates) {
    if (settings.max_iterations == 0U || settings.max_iterations > 100000U ||
        !finite_value(settings.simplex_step) || settings.simplex_step <= 0.0 ||
        !finite_value(settings.parameter_tolerance) || settings.parameter_tolerance <= 0.0 ||
        !finite_value(settings.objective_tolerance) || settings.objective_tolerance <= 0.0) {
        throw std::invalid_argument("invalid UMPA-DDF optimizer settings");
    }
    const UmpaDdfKernelParameters initial_kernel{
        settings.kernel_radius, settings.initial_a, settings.initial_b, settings.initial_c};
    const State initial_state = kernel_to_state(initial_kernel);
    UmpaDdfKernelOptimization result;
    result.kernel = initial_kernel;

    std::size_t evaluations = 0U;
    const auto evaluate = [&](const State& state) {
        Vertex vertex;
        vertex.state = state;
        vertex.fit = fit_umpa_ddf_kernel_official_integer_at(
            sample_stack, reference_stack, frames, height, width,
            coordinate_y, coordinate_x, official_shift_y, official_shift_x,
            analysis_radius, max_shift, state_to_kernel(state, settings.kernel_radius),
            assign_coordinates);
        ++evaluations;
        if (vertex.fit.numerical_valid && finite_value(vertex.fit.residual_cost) &&
            vertex.fit.residual_cost >= 0.0) {
            vertex.objective = vertex.fit.residual_cost;
        }
        return vertex;
    };

    std::vector<Vertex> simplex;
    simplex.reserve(4U);
    simplex.push_back(evaluate(initial_state));
    result.initial_residual_cost = simplex.front().objective;
    for (std::size_t axis = 0U; axis < 3U; ++axis) {
        State state = initial_state;
        state[axis] += settings.simplex_step;
        simplex.push_back(evaluate(state));
    }

    constexpr double alpha = 1.0;
    constexpr double gamma = 2.0;
    constexpr double rho = 0.5;
    constexpr double sigma = 0.5;
    for (std::size_t iteration = 0U; iteration < settings.max_iterations; ++iteration) {
        std::stable_sort(simplex.begin(), simplex.end(), better);
        const double objective_scale = std::max(1.0, std::abs(simplex.front().objective));
        double objective_span = 0.0;
        for (const Vertex& vertex : simplex) {
            objective_span = std::max(objective_span,
                                      std::abs(vertex.objective - simplex.front().objective));
        }
        result.iterations = iteration;
        if (simplex_span(simplex) <= settings.parameter_tolerance &&
            objective_span <= settings.objective_tolerance * objective_scale) {
            result.converged = true;
            break;
        }

        State centroid{};
        for (std::size_t i = 0U; i < 3U; ++i) {
            for (std::size_t axis = 0U; axis < centroid.size(); ++axis) {
                centroid[axis] += simplex[i].state[axis] / 3.0;
            }
        }
        State reflected_state{};
        for (std::size_t axis = 0U; axis < centroid.size(); ++axis) {
            reflected_state[axis] = centroid[axis] +
                alpha * (centroid[axis] - simplex[3].state[axis]);
        }
        Vertex reflected = evaluate(reflected_state);
        if (better(reflected, simplex.front())) {
            State expanded_state{};
            for (std::size_t axis = 0U; axis < centroid.size(); ++axis) {
                expanded_state[axis] = centroid[axis] +
                    gamma * (reflected.state[axis] - centroid[axis]);
            }
            Vertex expanded = evaluate(expanded_state);
            simplex[3] = better(expanded, reflected) ? expanded : reflected;
            continue;
        }
        if (better(reflected, simplex[2])) {
            simplex[3] = reflected;
            continue;
        }

        State contracted_state{};
        if (better(reflected, simplex[3])) {
            for (std::size_t axis = 0U; axis < centroid.size(); ++axis) {
                contracted_state[axis] = centroid[axis] +
                    rho * (reflected.state[axis] - centroid[axis]);
            }
        } else {
            for (std::size_t axis = 0U; axis < centroid.size(); ++axis) {
                contracted_state[axis] = centroid[axis] +
                    rho * (simplex[3].state[axis] - centroid[axis]);
            }
        }
        Vertex contracted = evaluate(contracted_state);
        if (better(contracted, simplex[3])) {
            simplex[3] = contracted;
            continue;
        }
        for (std::size_t i = 1U; i < simplex.size(); ++i) {
            State shrunk_state{};
            for (std::size_t axis = 0U; axis < shrunk_state.size(); ++axis) {
                shrunk_state[axis] = simplex.front().state[axis] +
                    sigma * (simplex[i].state[axis] - simplex.front().state[axis]);
            }
            simplex[i] = evaluate(shrunk_state);
        }
    }

    std::stable_sort(simplex.begin(), simplex.end(), better);
    result.kernel = state_to_kernel(simplex.front().state, settings.kernel_radius);
    result.fit = simplex.front().fit;
    result.evaluations = evaluations;
    result.numerical_valid = result.fit.numerical_valid;
    if (result.iterations + 1U >= settings.max_iterations && !result.converged) {
        result.iterations = settings.max_iterations;
    }
    return result;
}

}  // namespace wsvt
