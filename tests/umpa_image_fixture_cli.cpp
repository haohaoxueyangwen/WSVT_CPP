#include "wsvt/umpa_physical_fit.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ImageCase {
    std::int32_t y = 0;
    std::int32_t x = 0;
    std::int32_t dy = 0;
    std::int32_t dx = 0;
    double expected_t = 0.0;
    double expected_d = 0.0;
};

template <typename T>
T read_value(std::ifstream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error("truncated UMPA image fixture");
    }
    return value;
}

template <typename T>
std::vector<T> read_vector(std::ifstream& input, std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::overflow_error("UMPA image fixture array size overflow");
    }
    std::vector<T> values(count);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(T)));
    if (!input) {
        throw std::runtime_error("truncated UMPA image fixture array");
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

void emit_fit(
    std::size_t index,
    const ImageCase& fixture,
    const wsvt::UmpaPhysicalFit& fit,
    double wrong_sign_cost) {
    std::cout << "{\"id\":\"case_" << index << "\",\"y\":" << fixture.y
              << ",\"x\":" << fixture.x << ",\"dy\":" << fixture.dy
              << ",\"dx\":" << fixture.dx << ",\"expected_T\":";
    emit_number(fixture.expected_t);
    std::cout << ",\"expected_D\":";
    emit_number(fixture.expected_d);
    std::cout << ",\"wrong_sign_cost\":";
    emit_number(wrong_sign_cost);
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
    std::cout << ",\"observation_count\":" << fit.statistics.observation_count;
    std::cout << ",\"alpha\":";
    emit_number(fit.alpha);
    std::cout << ",\"beta\":";
    emit_number(fit.beta);
    std::cout << ",\"transmission\":";
    emit_number(fit.transmission);
    std::cout << ",\"visibility\":";
    emit_number(fit.visibility);
    std::cout << ",\"cost\":";
    emit_number(fit.cost);
    std::cout << ",\"delta\":";
    emit_number(fit.delta);
    std::cout << ",\"condition\":";
    emit_number(fit.condition);
    std::cout << ",\"numerical_valid\":"
              << (fit.numerical_valid ? "true" : "false")
              << ",\"physical_valid\":"
              << (fit.physical_valid ? "true" : "false") << '}';
}

std::size_t checked_stack_size(
    std::uint32_t frames, std::uint32_t height, std::uint32_t width) {
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    if (frames != 0U && plane > std::numeric_limits<std::size_t>::max() / frames) {
        throw std::overflow_error("UMPA image fixture dimensions overflow");
    }
    return static_cast<std::size_t>(frames) * plane;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc != 3 || std::string(argv[1]) != "--input") {
        std::cerr << "usage: wsvt_umpa_image_fixture --input FIXTURE.bin\n";
        return 2;
    }
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error("UMPA image fixture requires a little-endian host");
    }
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open UMPA image fixture");
    }
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    const std::array<char, 8> expected_magic{'U', 'M', 'P', 'I', 'M', 'G', '1', '\0'};
    if (!input || magic != expected_magic) {
        throw std::runtime_error("invalid UMPA image fixture magic");
    }
    const std::uint32_t schema_version = read_value<std::uint32_t>(input);
    const std::uint32_t frames = read_value<std::uint32_t>(input);
    const std::uint32_t height = read_value<std::uint32_t>(input);
    const std::uint32_t width = read_value<std::uint32_t>(input);
    const std::uint32_t radius = read_value<std::uint32_t>(input);
    const std::uint32_t case_count = read_value<std::uint32_t>(input);
    if (schema_version != 1U || frames == 0U || height == 0U || width == 0U ||
        case_count == 0U) {
        throw std::runtime_error("invalid UMPA image fixture header");
    }
    const std::size_t stack_size = checked_stack_size(frames, height, width);
    const std::vector<double> reference = read_vector<double>(input, stack_size);
    const std::vector<double> sample = read_vector<double>(input, stack_size);
    std::vector<ImageCase> cases;
    cases.reserve(case_count);
    for (std::uint32_t index = 0; index < case_count; ++index) {
        ImageCase fixture;
        fixture.y = read_value<std::int32_t>(input);
        fixture.x = read_value<std::int32_t>(input);
        fixture.dy = read_value<std::int32_t>(input);
        fixture.dx = read_value<std::int32_t>(input);
        fixture.expected_t = read_value<double>(input);
        fixture.expected_d = read_value<double>(input);
        if (fixture.y < 0 || fixture.x < 0) {
            throw std::runtime_error("negative UMPA sample center");
        }
        cases.push_back(fixture);
    }

    bool boundary_rejected = false;
    try {
        (void)wsvt::fit_umpa_physical_integer_at(
            sample, reference, frames, height, width,
            1U, static_cast<std::size_t>(cases[0].x), 0, 0, radius);
    } catch (const std::out_of_range&) {
        boundary_rejected = true;
    }

    std::cout << std::setprecision(17)
              << "{\"schema_version\":1,\"implementation\":"
                 "\"cpp_image_integer_float64\",\"frames\":" << frames
              << ",\"height\":" << height << ",\"width\":" << width
              << ",\"analysis_radius\":" << radius
              << ",\"boundary_rejected\":"
              << (boundary_rejected ? "true" : "false") << ",\"cases\":[";
    for (std::size_t index = 0; index < cases.size(); ++index) {
        if (index != 0U) {
            std::cout << ',';
        }
        const ImageCase& fixture = cases[index];
        const auto correct = wsvt::fit_umpa_physical_integer_at(
            sample, reference, frames, height, width,
            static_cast<std::size_t>(fixture.y),
            static_cast<std::size_t>(fixture.x),
            fixture.dy, fixture.dx, radius);
        const auto wrong_sign = wsvt::fit_umpa_physical_integer_at(
            sample, reference, frames, height, width,
            static_cast<std::size_t>(fixture.y),
            static_cast<std::size_t>(fixture.x),
            -static_cast<std::ptrdiff_t>(fixture.dy),
            -static_cast<std::ptrdiff_t>(fixture.dx), radius);
        emit_fit(index, fixture, correct, wrong_sign.cost);
    }
    std::cout << "]}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
