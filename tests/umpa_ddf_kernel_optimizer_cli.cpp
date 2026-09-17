#include "wsvt/umpa_ddf_kernel_optimizer.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T> T read_value(std::ifstream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) throw std::runtime_error("truncated UMPA-DDF optimizer fixture");
    return value;
}

template <typename T>
std::vector<T> read_vector(std::ifstream& input, std::size_t count) {
    std::vector<T> values(count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(count * sizeof(T)));
    if (!input) throw std::runtime_error("truncated UMPA-DDF optimizer stack");
    return values;
}

void emit_number(std::ostream& output, double value) {
    if (std::isfinite(value)) output << std::setprecision(17) << value;
    else output << "null";
}

std::array<double, 3> covariance(const wsvt::UmpaDdfKernelParameters& params) {
    const std::size_t side = 2U * params.kernel_radius + 1U;
    const auto radius = static_cast<std::ptrdiff_t>(params.kernel_radius);
    std::vector<double> values(side * side, 0.0);
    double total = 0.0;
    for (std::ptrdiff_t y = -radius; y <= radius; ++y) {
        for (std::ptrdiff_t x = -radius; x <= radius; ++x) {
            const double value = std::exp(
                -params.a * static_cast<double>(y * y) -
                params.b * static_cast<double>(y * x) -
                params.c * static_cast<double>(x * x));
            values[static_cast<std::size_t>(y + radius) * side +
                   static_cast<std::size_t>(x + radius)] = value;
            total += value;
        }
    }
    double yy = 0.0;
    double yx = 0.0;
    double xx = 0.0;
    for (std::ptrdiff_t y = -radius; y <= radius; ++y) {
        for (std::ptrdiff_t x = -radius; x <= radius; ++x) {
            const double weight = values[static_cast<std::size_t>(y + radius) * side +
                                         static_cast<std::size_t>(x + radius)] / total;
            yy += weight * static_cast<double>(y * y);
            yx += weight * static_cast<double>(y * x);
            xx += weight * static_cast<double>(x * x);
        }
    }
    return {yy, yx, xx};
}

}  // namespace

int main(int argc, char** argv) try {
    if ((argc != 3 && argc != 7) || std::string(argv[1]) != "--input" ||
        (argc == 7 && std::string(argv[3]) != "--initial")) {
        std::cerr << "usage: wsvt_umpa_ddf_kernel_optimizer_cli --input FIXTURE.bin [--initial A B C]\n";
        return 2;
    }
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) throw std::runtime_error("cannot open UMPA-DDF optimizer fixture");
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    const std::array<char, 8> expected{'U','M','P','A','D','D','F','1'};
    if (!input || magic != expected) throw std::runtime_error("invalid optimizer fixture magic");
    const auto version = read_value<std::uint32_t>(input);
    const auto frames = read_value<std::uint32_t>(input);
    const auto height = read_value<std::uint32_t>(input);
    const auto width = read_value<std::uint32_t>(input);
    const auto analysis_radius = read_value<std::uint32_t>(input);
    const auto max_shift = read_value<std::int32_t>(input);
    const auto kernel_radius = read_value<std::uint32_t>(input);
    const auto candidate_count = read_value<std::uint32_t>(input);
    double initial_a = read_value<double>(input);
    double initial_b = read_value<double>(input);
    double initial_c = read_value<double>(input);
    if (version != 1U || frames == 0U || height == 0U || width == 0U ||
        max_shift <= 0 || candidate_count != 1U) {
        throw std::runtime_error("invalid optimizer fixture header");
    }
    const std::size_t stack_size = static_cast<std::size_t>(frames) * height * width;
    const std::vector<double> reference = read_vector<double>(input, stack_size);
    const std::vector<double> sample = read_vector<double>(input, stack_size);
    const auto y = read_value<std::int32_t>(input);
    const auto x = read_value<std::int32_t>(input);
    const auto shift_y = read_value<std::int32_t>(input);
    const auto shift_x = read_value<std::int32_t>(input);
    const auto assign = read_value<std::int32_t>(input);
    if (assign != 0 && assign != 1) throw std::runtime_error("invalid assignment");

    if (argc == 7) {
        initial_a = std::stod(argv[4]);
        initial_b = std::stod(argv[5]);
        initial_c = std::stod(argv[6]);
    }
    const wsvt::UmpaDdfKernelOptimizerParameters settings{
        kernel_radius, initial_a, initial_b, initial_c, 400U, 0.35, 1.0e-5, 1.0e-10};
    const auto result = wsvt::optimize_umpa_ddf_kernel_fixed_u(
        sample, reference, frames, height, width,
        static_cast<std::size_t>(y), static_cast<std::size_t>(x),
        shift_y, shift_x, analysis_radius, max_shift, settings,
        assign == 0 ? wsvt::UmpaAssignCoordinates::Sample
                    : wsvt::UmpaAssignCoordinates::Reference);
    const auto moments = covariance(result.kernel);
    std::cout << std::setprecision(17)
              << "{\"status\":\"" << (result.numerical_valid ? "valid" : "invalid")
              << "\",\"converged\":" << (result.converged ? "true" : "false")
              << ",\"iterations\":" << result.iterations
              << ",\"evaluations\":" << result.evaluations
              << ",\"initial_residual_cost\":";
    emit_number(std::cout, result.initial_residual_cost);
    std::cout << ",\"residual_cost\":";
    emit_number(std::cout, result.fit.residual_cost);
    std::cout << ",\"official_cost\":";
    emit_number(std::cout, result.fit.cost);
    std::cout << ",\"T\":";
    emit_number(std::cout, result.fit.transmission);
    std::cout << ",\"a\":";
    emit_number(std::cout, result.kernel.a);
    std::cout << ",\"b\":";
    emit_number(std::cout, result.kernel.b);
    std::cout << ",\"c\":";
    emit_number(std::cout, result.kernel.c);
    std::cout << ",\"covariance_yy_yx_xx\":[";
    emit_number(std::cout, moments[0]);
    std::cout << ',';
    emit_number(std::cout, moments[1]);
    std::cout << ',';
    emit_number(std::cout, moments[2]);
    std::cout << "]}\n";
    return result.numerical_valid ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
