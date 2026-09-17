#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "umpa_fixture_data.hpp"
#include "wsvt/umpa_physical_fit.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace wsvt;

TEST_CASE("UMPA Hamming analysis window is normalized and symmetric", "[umpa][window]") {
    const std::vector<double> window = normalized_hamming_window_2d(2U);
    REQUIRE(window.size() == 25U);
    double total = 0.0;
    for (std::size_t y = 0; y < 5U; ++y) {
        for (std::size_t x = 0; x < 5U; ++x) {
            total += window[y * 5U + x];
            REQUIRE(window[y * 5U + x] == Catch::Approx(window[(4U - y) * 5U + x]));
            REQUIRE(window[y * 5U + x] == Catch::Approx(window[y * 5U + (4U - x)]));
        }
    }
    REQUIRE(total == Catch::Approx(1.0).margin(1.0e-15));
}

TEST_CASE("UMPA raw fit recovers analytic T and D", "[umpa][physical-fit]") {
    const auto fixtures = umpa_fixture::all_fixtures();
    for (std::size_t index = 0; index < 3U; ++index) {
        const auto& fixture = fixtures[index];
        const UmpaPhysicalFit fit = fit_umpa_physical(
            fixture.sample, fixture.reference, fixture.reference_mean, fixture.weights);
        REQUIRE(fit.numerical_valid);
        REQUIRE(fit.physical_valid);
        REQUIRE(fit.transmission == Catch::Approx(fixture.expected_transmission).margin(1.0e-11));
        REQUIRE(fit.visibility == Catch::Approx(fixture.expected_visibility).margin(1.0e-11));
        REQUIRE(fit.cost <= 1.0e-20);
    }
}

TEST_CASE("UMPA zero weights mask non-finite observations", "[umpa][mask]") {
    const auto fixture = umpa_fixture::all_fixtures()[2];
    const UmpaPhysicalFit fit = fit_umpa_physical(
        fixture.sample, fixture.reference, fixture.reference_mean, fixture.weights);
    REQUIRE(fit.numerical_valid);
    REQUIRE(fit.statistics.observation_count < fixture.sample.size());
}

TEST_CASE("UMPA unphysical solution remains visible and unclipped", "[umpa][physical-fit]") {
    const auto fixture = umpa_fixture::all_fixtures()[3];
    const UmpaPhysicalFit fit = fit_umpa_physical(
        fixture.sample, fixture.reference, fixture.reference_mean, fixture.weights);
    REQUIRE(fit.numerical_valid);
    REQUIRE_FALSE(fit.physical_valid);
    REQUIRE(fit.transmission == Catch::Approx(0.90).margin(1.0e-11));
    REQUIRE(fit.visibility == Catch::Approx(1.20).margin(1.0e-11));
}

TEST_CASE("UMPA singular normal equation is invalid", "[umpa][physical-fit][singular]") {
    const auto fixture = umpa_fixture::all_fixtures()[4];
    const UmpaPhysicalFit fit = fit_umpa_physical(
        fixture.sample, fixture.reference, fixture.reference_mean, fixture.weights);
    REQUIRE_FALSE(fit.numerical_valid);
    REQUIRE_FALSE(fit.physical_valid);
    REQUIRE(std::isnan(fit.transmission));
    REQUIRE((std::isinf(fit.condition) || fit.condition > 1.0e12));
}

TEST_CASE("UMPA active non-finite observations are rejected", "[umpa][validation]") {
    const std::vector<double> sample{1.0, std::numeric_limits<double>::quiet_NaN()};
    const std::vector<double> reference{1.0, 2.0};
    const std::vector<double> mean{1.0, 1.0};
    const std::vector<double> weights{1.0, 1.0};
    REQUIRE_THROWS_AS(
        fit_umpa_physical(sample, reference, mean, weights),
        std::invalid_argument);
}

TEST_CASE("UMPA integer image sampling obeys the declared displacement sign", "[umpa][image]") {
    constexpr std::size_t frames = 4U;
    constexpr std::size_t height = 32U;
    constexpr std::size_t width = 32U;
    constexpr std::size_t center_y = 16U;
    constexpr std::size_t center_x = 16U;
    constexpr std::ptrdiff_t dy = -2;
    constexpr std::ptrdiff_t dx = 3;
    constexpr double expected_t = 0.82;
    constexpr double expected_d = 0.63;
    std::vector<double> reference(frames * height * width, 0.0);
    std::vector<double> sample(frames * height * width, 0.0);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const std::size_t pattern =
                    (17U * y + 13U * x + 7U * frame + 5U * x * y) % 29U;
                reference[(frame * height + y) * width + x] =
                    80.0 + 3.0 * static_cast<double>(frame) +
                    static_cast<double>(pattern) +
                    0.125 * static_cast<double>(y) +
                    0.0625 * static_cast<double>(x);
            }
        }
    }
    const std::vector<double> window = normalized_hamming_window_2d(2U);
    const std::size_t reference_y = center_y + 2U;
    const std::size_t reference_x = center_x - 3U;
    for (std::size_t frame = 0; frame < frames; ++frame) {
        double mean = 0.0;
        for (std::size_t wy = 0; wy < 5U; ++wy) {
            for (std::size_t wx = 0; wx < 5U; ++wx) {
                const std::size_t ry = reference_y + wy - 2U;
                const std::size_t rx = reference_x + wx - 2U;
                mean += window[wy * 5U + wx] *
                    reference[(frame * height + ry) * width + rx];
            }
        }
        for (std::size_t wy = 0; wy < 5U; ++wy) {
            for (std::size_t wx = 0; wx < 5U; ++wx) {
                const std::size_t sy = center_y + wy - 2U;
                const std::size_t sx = center_x + wx - 2U;
                const std::size_t ry = reference_y + wy - 2U;
                const std::size_t rx = reference_x + wx - 2U;
                const double value = reference[(frame * height + ry) * width + rx];
                sample[(frame * height + sy) * width + sx] =
                    expected_t * (expected_d * (value - mean) + mean);
            }
        }
    }

    const UmpaPhysicalFit correct = fit_umpa_physical_integer_at(
        sample, reference, frames, height, width,
        center_y, center_x, dy, dx, 2U);
    const UmpaPhysicalFit wrong_sign = fit_umpa_physical_integer_at(
        sample, reference, frames, height, width,
        center_y, center_x, -dy, -dx, 2U);
    REQUIRE(correct.physical_valid);
    REQUIRE(correct.transmission == Catch::Approx(expected_t).margin(1.0e-11));
    REQUIRE(correct.visibility == Catch::Approx(expected_d).margin(1.0e-11));
    REQUIRE(correct.cost <= 1.0e-20);
    REQUIRE(wrong_sign.cost > correct.cost + 1.0e-3);
}

TEST_CASE("UMPA integer image sampling rejects incomplete patches", "[umpa][image][boundary]") {
    const std::vector<double> stack(2U * 16U * 16U, 1.0);
    REQUIRE_THROWS_AS(
        fit_umpa_physical_integer_at(
            stack, stack, 2U, 16U, 16U, 1U, 8U, 0, 0, 2U),
        std::out_of_range);
    REQUIRE_THROWS_AS(
        fit_umpa_physical_integer_at(
            stack, stack, 2U, 16U, 16U, 8U, 8U, 0, 7, 2U),
        std::out_of_range);
}

TEST_CASE("UMPA official integer wrapper matches sign assignment and open bounds", "[umpa][official-parity]") {
    constexpr std::size_t frames = 4U;
    constexpr std::size_t height = 32U;
    constexpr std::size_t width = 34U;
    constexpr std::size_t sample_y = 15U;
    constexpr std::size_t sample_x = 16U;
    constexpr std::ptrdiff_t official_shift_y = 2;
    constexpr std::ptrdiff_t official_shift_x = -3;
    constexpr std::size_t reference_y = 17U;
    constexpr std::size_t reference_x = 13U;
    constexpr std::size_t radius = 1U;
    constexpr std::ptrdiff_t max_shift = 4;
    constexpr double expected_t = 0.83;
    constexpr double expected_d = 0.61;
    std::vector<double> reference(frames * height * width, 0.0);
    std::vector<double> sample(frames * height * width, 0.0);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const std::size_t pattern =
                    (11U * y + 19U * x + 7U * frame + 3U * x * y) % 31U;
                reference[(frame * height + y) * width + x] =
                    70.0 + 2.0 * static_cast<double>(frame) +
                    static_cast<double>(pattern) +
                    0.125 * static_cast<double>(y) +
                    0.0625 * static_cast<double>(x);
            }
        }
    }
    const std::vector<double> window = normalized_hamming_window_2d(radius);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        double mean = 0.0;
        for (std::size_t wy = 0; wy < 3U; ++wy) {
            for (std::size_t wx = 0; wx < 3U; ++wx) {
                const std::size_t ry = reference_y + wy - radius;
                const std::size_t rx = reference_x + wx - radius;
                mean += window[wy * 3U + wx] *
                    reference[(frame * height + ry) * width + rx];
            }
        }
        for (std::size_t wy = 0; wy < 3U; ++wy) {
            for (std::size_t wx = 0; wx < 3U; ++wx) {
                const std::size_t sy = sample_y + wy - radius;
                const std::size_t sx = sample_x + wx - radius;
                const std::size_t ry = reference_y + wy - radius;
                const std::size_t rx = reference_x + wx - radius;
                const double value =
                    reference[(frame * height + ry) * width + rx];
                sample[(frame * height + sy) * width + sx] =
                    expected_t * (expected_d * value + (1.0 - expected_d) * mean);
            }
        }
    }

    const UmpaPhysicalFit sample_assigned =
        fit_umpa_physical_official_integer_at(
            sample, reference, frames, height, width,
            sample_y, sample_x, official_shift_y, official_shift_x,
            radius, max_shift, UmpaAssignCoordinates::Sample);
    const UmpaPhysicalFit reference_assigned =
        fit_umpa_physical_official_integer_at(
            sample, reference, frames, height, width,
            reference_y, reference_x, official_shift_y, official_shift_x,
            radius, max_shift, UmpaAssignCoordinates::Reference);
    for (const UmpaPhysicalFit* fit : {&sample_assigned, &reference_assigned}) {
        REQUIRE(fit->numerical_valid);
        REQUIRE(fit->physical_valid);
        REQUIRE(fit->transmission == Catch::Approx(expected_t).margin(1.0e-11));
        REQUIRE(fit->visibility == Catch::Approx(expected_d).margin(1.0e-11));
        REQUIRE(std::abs(fit->cost) <= 1.0e-10);
    }
    REQUIRE(sample_assigned.cost == Catch::Approx(reference_assigned.cost).margin(1.0e-12));
    REQUIRE(sample_assigned.transmission == Catch::Approx(reference_assigned.transmission).margin(1.0e-12));
    REQUIRE(sample_assigned.visibility == Catch::Approx(reference_assigned.visibility).margin(1.0e-12));

    const std::vector<float> sample_float(sample.begin(), sample.end());
    const std::vector<float> reference_float(reference.begin(), reference.end());
    const std::vector<double> sample_quantized(
        sample_float.begin(), sample_float.end());
    const std::vector<double> reference_quantized(
        reference_float.begin(), reference_float.end());
    const UmpaPhysicalFit official_quantized =
        fit_umpa_physical_official_integer_at(
            sample_quantized, reference_quantized, frames, height, width,
            sample_y, sample_x, official_shift_y, official_shift_x,
            radius, max_shift, UmpaAssignCoordinates::Sample);
    const UmpaPhysicalFit production_residual =
        fit_umpa_physical_float_candidate_at(
            sample_float, reference_float, frames, height, width,
            sample_y, sample_x, reference_y, reference_x,
            radius, window, 0.0, 0.0);
    const UmpaPhysicalFit production_expanded = solve_umpa_physical_fit(
        production_residual.statistics, 0.0, 0.0);
    REQUIRE(production_expanded.cost ==
        Catch::Approx(official_quantized.cost).margin(1.0e-9));
    REQUIRE(production_expanded.transmission ==
        Catch::Approx(official_quantized.transmission).margin(1.0e-10));
    REQUIRE(production_expanded.visibility ==
        Catch::Approx(official_quantized.visibility).margin(1.0e-10));

    REQUIRE_THROWS_AS(
        fit_umpa_physical_official_integer_at(
            sample, reference, frames, height, width,
            sample_y, sample_x, -max_shift, 0, radius, max_shift),
        std::out_of_range);
    REQUIRE_THROWS_AS(
        fit_umpa_physical_official_integer_at(
            sample, reference, frames, height, width,
            sample_y, sample_x, 0, max_shift, radius, max_shift),
        std::out_of_range);
    REQUIRE_THROWS_AS(
        fit_umpa_physical_official_integer_at(
            sample, reference, frames, height, width,
            radius + static_cast<std::size_t>(max_shift) - 1U,
            sample_x, 0, 0, radius, max_shift),
        std::out_of_range);
}

TEST_CASE("UMPA N zero ModelDF is structurally rank deficient", "[umpa][N0][rank]") {
    constexpr std::size_t frames = 8U;
    constexpr std::size_t height = 8U;
    constexpr std::size_t width = 8U;
    std::vector<double> reference(frames * height * width, 0.0);
    std::vector<double> sample(frames * height * width, 0.0);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const std::size_t index = (frame * height + y) * width + x;
                reference[index] = 20.0 + static_cast<double>(
                    7U * frame + 3U * y + 5U * x + frame * x);
                sample[index] = 0.82 * reference[index];
            }
        }
    }
    const UmpaPhysicalFit fit = fit_umpa_physical_integer_at(
        sample, reference, frames, height, width, 4U, 4U, 0, 0, 0U);
    REQUIRE_FALSE(fit.numerical_valid);
    REQUIRE_FALSE(fit.physical_valid);
    REQUIRE(std::isnan(fit.transmission));
    REQUIRE((std::isinf(fit.condition) || fit.condition > 1.0e12));
}
