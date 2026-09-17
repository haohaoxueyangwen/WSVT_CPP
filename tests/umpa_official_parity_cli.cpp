#include "wsvt/umpa_physical_fit.hpp"

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

struct Candidate {
    std::int32_t y = 0;
    std::int32_t x = 0;
    std::int32_t shift_y = 0;
    std::int32_t shift_x = 0;
    std::int32_t assign_coordinates = 0;
};

template <typename T>
T read_value(std::ifstream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error("truncated official UMPA parity fixture");
    }
    return value;
}

template <typename T>
std::vector<T> read_vector(std::ifstream& input, std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::overflow_error("official UMPA parity fixture size overflow");
    }
    std::vector<T> values(count);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(T)));
    if (!input) {
        throw std::runtime_error("truncated official UMPA parity array");
    }
    return values;
}

bool is_finite_value(double value) noexcept {
    constexpr std::uint64_t exponent_mask = 0x7ff0000000000000ULL;
    return (std::bit_cast<std::uint64_t>(value) & exponent_mask) != exponent_mask;
}

void emit_number(double value) {
    if (is_finite_value(value)) {
        std::cout << value;
    } else {
        std::cout << "null";
    }
}

void emit_fit(const wsvt::UmpaPhysicalFit& fit) {
    std::cout << ",\"cost\":";
    emit_number(fit.cost);
    std::cout << ",\"T\":";
    emit_number(fit.transmission);
    std::cout << ",\"D\":";
    emit_number(fit.visibility);
    std::cout << ",\"alpha\":";
    emit_number(fit.alpha);
    std::cout << ",\"beta\":";
    emit_number(fit.beta);
    std::cout << ",\"l1\":";
    emit_number(fit.statistics.l1);
    std::cout << ",\"l2\":";
    emit_number(fit.statistics.l2);
    std::cout << ",\"l3\":";
    emit_number(fit.statistics.l3);
    std::cout << ",\"l4\":";
    emit_number(fit.statistics.l4);
    std::cout << ",\"l5\":";
    emit_number(fit.statistics.l5);
    std::cout << ",\"l6\":";
    emit_number(fit.statistics.l6);
    std::cout << ",\"weight_sum\":";
    emit_number(fit.statistics.weight_sum);
    std::cout << ",\"numerical_valid\":"
              << (fit.numerical_valid ? "true" : "false")
              << ",\"physical_valid\":"
              << (fit.physical_valid ? "true" : "false");
}

std::size_t checked_stack_size(
    std::uint32_t frames, std::uint32_t height, std::uint32_t width) {
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    if (frames != 0U && plane > std::numeric_limits<std::size_t>::max() / frames) {
        throw std::overflow_error("official UMPA parity dimensions overflow");
    }
    return static_cast<std::size_t>(frames) * plane;
}

bool outside_shift_bound(
    const Candidate& candidate, std::ptrdiff_t max_shift) noexcept {
    return candidate.shift_y <= -max_shift || candidate.shift_y >= max_shift ||
        candidate.shift_x <= -max_shift || candidate.shift_x >= max_shift;
}

bool outside_public_coordinate_bound(
    const Candidate& candidate,
    std::size_t height,
    std::size_t width,
    std::size_t radius,
    std::ptrdiff_t max_shift) noexcept {
    if (candidate.y < 0 || candidate.x < 0) {
        return true;
    }
    const std::size_t padding = radius + static_cast<std::size_t>(max_shift);
    const std::size_t y = static_cast<std::size_t>(candidate.y);
    const std::size_t x = static_cast<std::size_t>(candidate.x);
    return padding > height || padding > width ||
        y < padding || y >= height - padding ||
        x < padding || x >= width - padding;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc != 3 || std::string(argv[1]) != "--input") {
        std::cerr << "usage: wsvt_umpa_official_parity --input FIXTURE.bin\n";
        return 2;
    }
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error("official UMPA parity requires little endian");
    }
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open official UMPA parity fixture");
    }
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    const std::array<char, 8> expected_magic{'U', 'M', 'P', 'A', 'P', 'R', '1', '\0'};
    if (!input || magic != expected_magic) {
        throw std::runtime_error("invalid official UMPA parity fixture magic");
    }
    const std::uint32_t schema_version = read_value<std::uint32_t>(input);
    const std::uint32_t frames = read_value<std::uint32_t>(input);
    const std::uint32_t height = read_value<std::uint32_t>(input);
    const std::uint32_t width = read_value<std::uint32_t>(input);
    const std::uint32_t radius = read_value<std::uint32_t>(input);
    const std::uint32_t max_shift_raw = read_value<std::uint32_t>(input);
    const std::uint32_t candidate_count = read_value<std::uint32_t>(input);
    if (schema_version != 1U || frames == 0U || height == 0U || width == 0U ||
        max_shift_raw == 0U || candidate_count == 0U ||
        max_shift_raw > static_cast<std::uint32_t>(
            std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("invalid official UMPA parity header");
    }
    const std::ptrdiff_t max_shift = static_cast<std::ptrdiff_t>(max_shift_raw);
    const std::size_t stack_size = checked_stack_size(frames, height, width);
    const std::vector<double> reference_exchange =
        read_vector<double>(input, stack_size);
    const std::vector<double> sample_exchange =
        read_vector<double>(input, stack_size);
    std::vector<float> reference(reference_exchange.begin(), reference_exchange.end());
    std::vector<float> sample(sample_exchange.begin(), sample_exchange.end());
    const std::vector<double> window =
        wsvt::normalized_hamming_window_2d(radius);
    std::vector<Candidate> candidates;
    candidates.reserve(candidate_count);
    for (std::uint32_t index = 0; index < candidate_count; ++index) {
        Candidate candidate;
        candidate.y = read_value<std::int32_t>(input);
        candidate.x = read_value<std::int32_t>(input);
        candidate.shift_y = read_value<std::int32_t>(input);
        candidate.shift_x = read_value<std::int32_t>(input);
        candidate.assign_coordinates = read_value<std::int32_t>(input);
        candidates.push_back(candidate);
    }

    std::cout << std::setprecision(17)
              << "{\"schema_version\":1,\"implementation\":"
                 "\"cpp_production_modeldf_float32\",\"cases\":[";
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        if (index != 0U) {
            std::cout << ',';
        }
        const Candidate& candidate = candidates[index];
        const char* mode = candidate.assign_coordinates == 0 ? "sam" : "ref";
        std::cout << "{\"index\":" << index << ",\"y\":" << candidate.y
                  << ",\"x\":" << candidate.x
                  << ",\"shift_y\":" << candidate.shift_y
                  << ",\"shift_x\":" << candidate.shift_x
                  << ",\"assign_coordinates\":\"" << mode << "\"";
        if (candidate.assign_coordinates != 0 && candidate.assign_coordinates != 1) {
            std::cout << ",\"status\":\"invalid_assignment\"}";
            continue;
        }
        if (outside_shift_bound(candidate, max_shift)) {
            std::cout << ",\"status\":\"shift_bound\"}";
            continue;
        }
        if (outside_public_coordinate_bound(
                candidate, height, width, radius, max_shift)) {
            std::cout << ",\"status\":\"coordinate_bound\"}";
            continue;
        }
        try {
            const long long coordinate_y = candidate.y;
            const long long coordinate_x = candidate.x;
            const long long sample_y = candidate.assign_coordinates == 0
                ? coordinate_y
                : coordinate_y - candidate.shift_y;
            const long long sample_x = candidate.assign_coordinates == 0
                ? coordinate_x
                : coordinate_x - candidate.shift_x;
            const long long reference_y = candidate.assign_coordinates == 0
                ? coordinate_y + candidate.shift_y
                : coordinate_y;
            const long long reference_x = candidate.assign_coordinates == 0
                ? coordinate_x + candidate.shift_x
                : coordinate_x;
            if (sample_y < 0 || sample_x < 0 ||
                reference_y < 0 || reference_x < 0) {
                throw std::out_of_range("negative official UMPA mapped center");
            }
            const wsvt::UmpaPhysicalFit production_fit =
                wsvt::fit_umpa_physical_float_candidate_at(
                    sample, reference, frames, height, width,
                    static_cast<std::size_t>(sample_y),
                    static_cast<std::size_t>(sample_x),
                    static_cast<std::size_t>(reference_y),
                    static_cast<std::size_t>(reference_x),
                    radius, window, 0.0, 0.0);
            const wsvt::UmpaPhysicalFit fit = wsvt::solve_umpa_physical_fit(
                production_fit.statistics, 0.0, 0.0);
            std::cout << ",\"status\":\"valid\"";
            emit_fit(fit);
            std::cout << '}';
        } catch (const std::out_of_range&) {
            std::cout << ",\"status\":\"coordinate_bound\"}";
        }
    }
    std::cout << "]}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
