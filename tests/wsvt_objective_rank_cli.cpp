#include "wsvt/search_pruning.hpp"
#include "wsvt/wsvt_pipeline.hpp"
#include "wsvt/wsvt_umpa_refine.hpp"

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
    std::uint32_t candidate_id = 0;
    std::uint32_t label_code = 0;
    std::int32_t sample_y = 0;
    std::int32_t sample_x = 0;
    std::int32_t displacement_y = 0;
    std::int32_t displacement_x = 0;
};

template <typename T>
T read_value(std::ifstream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error("truncated WSVT objective-rank fixture");
    }
    return value;
}

template <typename T>
std::vector<T> read_vector(std::ifstream& input, std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::overflow_error("WSVT objective-rank fixture size overflow");
    }
    std::vector<T> values(count);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(T)));
    if (!input) {
        throw std::runtime_error("truncated WSVT objective-rank array");
    }
    return values;
}

std::size_t checked_product(std::size_t first, std::size_t second) {
    if (first != 0U && second > std::numeric_limits<std::size_t>::max() / first) {
        throw std::overflow_error("WSVT objective-rank dimensions overflow");
    }
    return first * second;
}

bool complete_patch(
    long long y, long long x, std::size_t radius,
    std::size_t height, std::size_t width) noexcept {
    const long long r = static_cast<long long>(radius);
    return y >= r && x >= r &&
        y + r < static_cast<long long>(height) &&
        x + r < static_cast<long long>(width);
}

void emit_fit(std::ostream& output, const wsvt::UmpaPhysicalFit& fit) {
    output << ',' << fit.cost
           << ',' << (fit.numerical_valid ? 1 : 0)
           << ',' << (fit.physical_valid ? 1 : 0)
           << ',' << fit.transmission
           << ',' << fit.visibility
           << ',' << fit.condition;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc != 5 || std::string(argv[1]) != "--input" ||
        std::string(argv[3]) != "--output") {
        std::cerr << "usage: wsvt_objective_rank --input FIXTURE.bin "
                     "--output CANDIDATES.csv\n";
        return 2;
    }
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error("WSVT objective-rank fixture requires little endian");
    }
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open WSVT objective-rank fixture");
    }
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    const std::array<char, 8> expected{'W', 'S', 'V', 'T', 'M', '0', '1', '\0'};
    if (!input || magic != expected) {
        throw std::runtime_error("invalid WSVT objective-rank fixture magic");
    }
    const std::uint32_t schema = read_value<std::uint32_t>(input);
    const std::uint32_t frames = read_value<std::uint32_t>(input);
    const std::uint32_t height = read_value<std::uint32_t>(input);
    const std::uint32_t width = read_value<std::uint32_t>(input);
    const std::uint32_t cal_half_window = read_value<std::uint32_t>(input);
    const std::uint32_t n_s_extend = read_value<std::uint32_t>(input);
    const std::uint32_t pyramid_level = read_value<std::uint32_t>(input);
    const std::uint32_t wavelet_level_cut = read_value<std::uint32_t>(input);
    const std::uint32_t n_template = read_value<std::uint32_t>(input);
    const std::uint32_t analysis_radius = read_value<std::uint32_t>(input);
    const std::uint32_t candidate_count = read_value<std::uint32_t>(input);
    if (schema != 1U || frames == 0U || height == 0U || width == 0U ||
        candidate_count == 0U || pyramid_level > 8U || analysis_radius > 16U) {
        throw std::runtime_error("invalid WSVT objective-rank fixture header");
    }
    const std::size_t plane = checked_product(height, width);
    const std::size_t stack_size = checked_product(frames, plane);
    const std::vector<float> sample = read_vector<float>(input, stack_size);
    const std::vector<float> reference = read_vector<float>(input, stack_size);
    std::vector<Candidate> candidates;
    candidates.reserve(candidate_count);
    for (std::uint32_t index = 0; index < candidate_count; ++index) {
        Candidate candidate;
        candidate.candidate_id = read_value<std::uint32_t>(input);
        candidate.label_code = read_value<std::uint32_t>(input);
        candidate.sample_y = read_value<std::int32_t>(input);
        candidate.sample_x = read_value<std::int32_t>(input);
        candidate.displacement_y = read_value<std::int32_t>(input);
        candidate.displacement_x = read_value<std::int32_t>(input);
        candidates.push_back(candidate);
    }
    char trailing = 0;
    input.read(&trailing, 1);
    if (input.gcount() != 0) {
        throw std::runtime_error("WSVT objective-rank fixture has trailing bytes");
    }

    wsvt::WSVT solver(
        sample, reference, frames, height, width,
        /*crop=*/0, static_cast<int>(cal_half_window),
        static_cast<int>(n_template), static_cast<int>(n_s_extend),
        /*n_cores=*/1, /*n_group=*/4,
        /*energy=*/14000.0, /*p_x=*/6.5e-7, /*mag_factor=*/1.0, /*z=*/0.425,
        static_cast<int>(wavelet_level_cut), static_cast<int>(pyramid_level),
        /*n_iter=*/1, /*use_estimate=*/false, /*use_wavelet=*/true,
        /*use_gpu=*/0, /*calc_darkfield=*/false, /*phase_cores=*/1);
    const wsvt::PyramidResult descriptors = solver.wavelet_data();
    if (descriptors.img_levels.empty() || descriptors.ref_levels.empty()) {
        throw std::runtime_error("WSVT objective-rank descriptor pyramid is empty");
    }
    const auto& sample_descriptor = descriptors.img_levels.front();
    const auto& reference_descriptor = descriptors.ref_levels.front();
    if (sample_descriptor.d1 != height || sample_descriptor.d2 != width ||
        reference_descriptor.d1 != height || reference_descriptor.d2 != width ||
        sample_descriptor.d0 != reference_descriptor.d0) {
        throw std::runtime_error("WSVT objective-rank final descriptor shape mismatch");
    }
    const std::size_t depth = sample_descriptor.d0;
    const std::vector<double> window =
        wsvt::normalized_hamming_window_2d(analysis_radius);

    std::ofstream output(argv[4]);
    if (!output) {
        throw std::runtime_error("cannot open WSVT objective-rank output CSV");
    }
    output << std::setprecision(17)
           << "candidate_id,label_code,sample_y,sample_x,displacement_y,"
              "displacement_x,reference_y,reference_x,descriptor_cost_ssd,"
              "raw_patch_complete,zncc_cost,zncc_numerical_valid,"
              "zncc_physical_valid,zncc_T,zncc_D,zncc_condition,"
              "modelt_cost,modelt_numerical_valid,modelt_physical_valid,"
              "modelt_T,modelt_D,modelt_condition,modeldf_cost,"
              "modeldf_numerical_valid,modeldf_physical_valid,modeldf_T,"
              "modeldf_D,modeldf_condition\n";
    for (const Candidate& candidate : candidates) {
        const long long sy = candidate.sample_y;
        const long long sx = candidate.sample_x;
        const long long ry = sy - candidate.displacement_y;
        const long long rx = sx - candidate.displacement_x;
        if (sy < 0 || sx < 0 ||
            sy >= static_cast<long long>(height) ||
            sx >= static_cast<long long>(width)) {
            throw std::out_of_range("objective-rank sample coordinate is out of bounds");
        }
        const float* sample_line = sample_descriptor.data.data() +
            (static_cast<std::size_t>(sy) * width + static_cast<std::size_t>(sx)) * depth;
        const bool reference_center_in_bounds =
            ry >= 0 && rx >= 0 &&
            ry < static_cast<long long>(height) &&
            rx < static_cast<long long>(width);
        float descriptor_cost = 0.0f;
        if (reference_center_in_bounds) {
            const float* reference_line = reference_descriptor.data.data() +
                (static_cast<std::size_t>(ry) * width +
                 static_cast<std::size_t>(rx)) * depth;
            descriptor_cost = wsvt::squared_distance_full_simd(
                sample_line, reference_line, depth);
        } else {
            // The production search explicitly zero-pads reference descriptors.
            // Candidates whose center lies in that padding therefore compare
            // against an all-zero descriptor, even though no raw N=1 patch is
            // available for the auxiliary objectives.
            for (std::size_t index = 0; index < depth; ++index) {
                descriptor_cost += sample_line[index] * sample_line[index];
            }
        }
        const bool raw_complete =
            complete_patch(sy, sx, analysis_radius, height, width) &&
            complete_patch(ry, rx, analysis_radius, height, width);
        output << candidate.candidate_id << ',' << candidate.label_code
               << ',' << sy << ',' << sx
               << ',' << candidate.displacement_y
               << ',' << candidate.displacement_x
               << ',' << ry << ',' << rx
               << ',' << descriptor_cost
               << ',' << (raw_complete ? 1 : 0);
        if (!raw_complete) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            for (int objective = 0; objective < 3; ++objective) {
                output << ',' << nan << ",0,0," << nan << ',' << nan << ',' << nan;
            }
            output << '\n';
            continue;
        }
        for (const auto objective : {
                 wsvt::SetTransportRawObjective::WindowedZncc,
                 wsvt::SetTransportRawObjective::ModelT,
                 wsvt::SetTransportRawObjective::ModelDF}) {
            const auto fit = wsvt::evaluate_set_transport_raw_candidate(
                objective, sample, reference, frames, height, width,
                static_cast<std::size_t>(sy), static_cast<std::size_t>(sx),
                static_cast<std::size_t>(ry), static_cast<std::size_t>(rx),
                analysis_radius, window);
            emit_fit(output, fit);
        }
        output << '\n';
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
