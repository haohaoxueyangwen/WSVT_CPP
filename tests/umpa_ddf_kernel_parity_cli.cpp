#include "wsvt/umpa_ddf_kernel.hpp"

#include <array>
#include <bit>
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
    if (!input) throw std::runtime_error("truncated UMPA-DDF parity fixture");
    return value;
}

template <typename T>
std::vector<T> read_vector(std::ifstream& input, std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        throw std::overflow_error("UMPA-DDF parity fixture size overflow");
    std::vector<T> values(count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(count * sizeof(T)));
    if (!input) throw std::runtime_error("truncated UMPA-DDF parity array");
    return values;
}

bool finite_value(double value) noexcept {
    constexpr std::uint64_t exponent_mask = 0x7ff0000000000000ULL;
    return (std::bit_cast<std::uint64_t>(value) & exponent_mask) != exponent_mask;
}

void emit_number(double value) {
    if (finite_value(value)) std::cout << value;
    else std::cout << "null";
}
}

int main(int argc, char** argv) try {
    if (argc != 3 || std::string(argv[1]) != "--input") {
        std::cerr << "usage: wsvt_umpa_ddf_kernel_parity_cli --input FIXTURE.bin\n";
        return 2;
    }
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) throw std::runtime_error("cannot open UMPA-DDF parity fixture");
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    const std::array<char, 8> expected{'U','M','P','A','D','D','F','1'};
    if (!input || magic != expected) throw std::runtime_error("invalid UMPA-DDF parity magic");
    const auto version = read_value<std::uint32_t>(input);
    const auto frames = read_value<std::uint32_t>(input);
    const auto height = read_value<std::uint32_t>(input);
    const auto width = read_value<std::uint32_t>(input);
    const auto analysis_radius = read_value<std::uint32_t>(input);
    const auto max_shift = read_value<std::int32_t>(input);
    const auto kernel_radius = read_value<std::uint32_t>(input);
    const auto candidate_count = read_value<std::uint32_t>(input);
    const double a = read_value<double>(input);
    const double b = read_value<double>(input);
    const double c = read_value<double>(input);
    if (version != 1U || frames == 0U || height == 0U || width == 0U ||
        max_shift <= 0 || candidate_count == 0U)
        throw std::runtime_error("invalid UMPA-DDF parity header");
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    const std::size_t stack_size = static_cast<std::size_t>(frames) * plane;
    const std::vector<double> reference = read_vector<double>(input, stack_size);
    const std::vector<double> sample = read_vector<double>(input, stack_size);
    struct Candidate { std::int32_t y, x, sy, sx, assign; };
    std::vector<Candidate> candidates;
    candidates.reserve(candidate_count);
    for (std::uint32_t i = 0; i < candidate_count; ++i) {
        candidates.push_back({read_value<std::int32_t>(input), read_value<std::int32_t>(input),
                              read_value<std::int32_t>(input), read_value<std::int32_t>(input),
                              read_value<std::int32_t>(input)});
    }
    const wsvt::UmpaDdfKernelParameters params{
        static_cast<std::size_t>(kernel_radius), a, b, c};
    std::cout << std::setprecision(17) << "{\"schema_version\":1,\"cases\":[";
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (i != 0U) std::cout << ',';
        const auto& q = candidates[i];
        std::cout << "{\"index\":" << i << ",\"y\":" << q.y << ",\"x\":" << q.x
                  << ",\"shift_y\":" << q.sy << ",\"shift_x\":" << q.sx
                  << ",\"assign_coordinates\":\"" << (q.assign == 0 ? "sam" : "ref") << "\"";
        if (q.assign != 0 && q.assign != 1) {
            std::cout << ",\"status\":\"invalid_assignment\"}";
            continue;
        }
        try {
            const auto fit = wsvt::fit_umpa_ddf_kernel_official_integer_at(
                sample, reference, frames, height, width,
                static_cast<std::size_t>(q.y), static_cast<std::size_t>(q.x),
                q.sy, q.sx, analysis_radius, max_shift, params,
                q.assign == 0 ? wsvt::UmpaAssignCoordinates::Sample
                               : wsvt::UmpaAssignCoordinates::Reference);
            std::cout << ",\"status\":\"valid\",\"t1\":"; emit_number(fit.t1);
            std::cout << ",\"t3\":"; emit_number(fit.t3);
            std::cout << ",\"t5\":"; emit_number(fit.t5);
            std::cout << ",\"T\":"; emit_number(fit.transmission);
            std::cout << ",\"cost\":"; emit_number(fit.cost);
            std::cout << ",\"residual_cost\":"; emit_number(fit.residual_cost);
            std::cout << ",\"numerical_valid\":" << (fit.numerical_valid ? "true" : "false") << '}';
        } catch (const std::out_of_range&) {
            std::cout << ",\"status\":\"boundary\"}";
        }
    }
    std::cout << "]}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
