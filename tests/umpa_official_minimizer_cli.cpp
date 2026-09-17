#include "wsvt/umpa_physical_fit.hpp"

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

struct Query {
    std::int32_t y = 0;
    std::int32_t x = 0;
    std::int32_t assign_coordinates = 0;
};

template <typename T>
T read_value(std::ifstream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error("truncated official UMPA minimizer fixture");
    }
    return value;
}

template <typename T>
std::vector<T> read_vector(std::ifstream& input, std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::overflow_error("official UMPA minimizer fixture size overflow");
    }
    std::vector<T> values(count);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(T)));
    if (!input) {
        throw std::runtime_error("truncated official UMPA minimizer array");
    }
    return values;
}

std::size_t checked_stack_size(
    std::uint32_t frames, std::uint32_t height, std::uint32_t width) {
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    if (frames != 0U && plane > std::numeric_limits<std::size_t>::max() / frames) {
        throw std::overflow_error("official UMPA minimizer dimensions overflow");
    }
    return static_cast<std::size_t>(frames) * plane;
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

}  // namespace

int main(int argc, char** argv) try {
    if (argc != 3 || std::string(argv[1]) != "--input") {
        std::cerr << "usage: wsvt_umpa_official_minimizer --input FIXTURE.bin\n";
        return 2;
    }
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error("official UMPA minimizer fixture requires little endian");
    }
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open official UMPA minimizer fixture");
    }
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    const std::array<char, 8> expected_magic{'U', 'M', 'P', 'A', 'M', 'N', '1', '\0'};
    if (!input || magic != expected_magic) {
        throw std::runtime_error("invalid official UMPA minimizer fixture magic");
    }
    const std::uint32_t schema_version = read_value<std::uint32_t>(input);
    const std::uint32_t frames = read_value<std::uint32_t>(input);
    const std::uint32_t height = read_value<std::uint32_t>(input);
    const std::uint32_t width = read_value<std::uint32_t>(input);
    const std::uint32_t radius = read_value<std::uint32_t>(input);
    const std::uint32_t max_shift_raw = read_value<std::uint32_t>(input);
    const std::uint32_t query_count = read_value<std::uint32_t>(input);
    if (schema_version != 1U || frames == 0U || height == 0U || width == 0U ||
        max_shift_raw == 0U || query_count == 0U ||
        max_shift_raw > static_cast<std::uint32_t>(
            std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("invalid official UMPA minimizer fixture header");
    }
    const std::ptrdiff_t max_shift = static_cast<std::ptrdiff_t>(max_shift_raw);
    const std::size_t stack_size = checked_stack_size(frames, height, width);
    const std::vector<double> reference_exchange =
        read_vector<double>(input, stack_size);
    const std::vector<double> sample_exchange =
        read_vector<double>(input, stack_size);
    std::vector<float> reference(reference_exchange.begin(), reference_exchange.end());
    std::vector<float> sample(sample_exchange.begin(), sample_exchange.end());
    std::vector<Query> queries;
    queries.reserve(query_count);
    for (std::uint32_t index = 0; index < query_count; ++index) {
        Query query;
        query.y = read_value<std::int32_t>(input);
        query.x = read_value<std::int32_t>(input);
        query.assign_coordinates = read_value<std::int32_t>(input);
        if (query.assign_coordinates != 0 && query.assign_coordinates != 1) {
            throw std::runtime_error("invalid coordinate assignment in minimizer fixture");
        }
        queries.push_back(query);
    }
    char trailing = 0;
    if (input.read(&trailing, 1)) {
        throw std::runtime_error("unexpected trailing bytes in minimizer fixture");
    }

    const std::vector<double> window = wsvt::normalized_hamming_window_2d(radius);
    std::cout << std::setprecision(17)
              << "{\"schema_version\":1,\"implementation\":"
                 "\"cpp_production_modeldf_float32_exhaustive_integer\",\"queries\":[";
    for (std::size_t query_index = 0; query_index < queries.size(); ++query_index) {
        if (query_index != 0U) {
            std::cout << ',';
        }
        const Query& query = queries[query_index];
        bool found = false;
        double best_cost = std::numeric_limits<double>::infinity();
        double second_cost = std::numeric_limits<double>::infinity();
        std::ptrdiff_t best_shift_y = 0;
        std::ptrdiff_t best_shift_x = 0;
        wsvt::UmpaPhysicalFit best_fit;
        std::size_t evaluated = 0;

        for (std::ptrdiff_t shift_y = -max_shift + 1;
             shift_y < max_shift; ++shift_y) {
            for (std::ptrdiff_t shift_x = -max_shift + 1;
                 shift_x < max_shift; ++shift_x) {
                const long long coordinate_y = query.y;
                const long long coordinate_x = query.x;
                const bool sample_assignment = query.assign_coordinates == 0;
                const long long sample_y = sample_assignment
                    ? coordinate_y
                    : coordinate_y - shift_y;
                const long long sample_x = sample_assignment
                    ? coordinate_x
                    : coordinate_x - shift_x;
                const long long reference_y = sample_assignment
                    ? coordinate_y + shift_y
                    : coordinate_y;
                const long long reference_x = sample_assignment
                    ? coordinate_x + shift_x
                    : coordinate_x;
                if (sample_y < 0 || sample_x < 0 || reference_y < 0 ||
                    reference_x < 0) {
                    throw std::out_of_range("negative mapped minimizer center");
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
                ++evaluated;
                if (!fit.numerical_valid || !is_finite_value(fit.cost)) {
                    continue;
                }
                if (!found || fit.cost < best_cost) {
                    second_cost = best_cost;
                    best_cost = fit.cost;
                    best_shift_y = shift_y;
                    best_shift_x = shift_x;
                    best_fit = fit;
                    found = true;
                } else if (fit.cost < second_cost) {
                    second_cost = fit.cost;
                }
            }
        }

        std::cout << "{\"index\":" << query_index
                  << ",\"y\":" << query.y
                  << ",\"x\":" << query.x
                  << ",\"assign_coordinates\":\""
                  << (query.assign_coordinates == 0 ? "sam" : "ref") << "\""
                  << ",\"status\":\"" << (found ? "valid" : "no_valid_candidate")
                  << "\",\"evaluated_candidates\":" << evaluated;
        if (found) {
            std::cout << ",\"shift_y\":" << best_shift_y
                      << ",\"shift_x\":" << best_shift_x
                      << ",\"project_displacement_y\":" << -best_shift_y
                      << ",\"project_displacement_x\":" << -best_shift_x
                      << ",\"cost\":";
            emit_number(best_cost);
            std::cout << ",\"second_cost\":";
            emit_number(second_cost);
            std::cout << ",\"margin\":";
            emit_number(second_cost - best_cost);
            std::cout << ",\"T\":";
            emit_number(best_fit.transmission);
            std::cout << ",\"D\":";
            emit_number(best_fit.visibility);
        }
        std::cout << '}';
    }
    std::cout << "]}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
