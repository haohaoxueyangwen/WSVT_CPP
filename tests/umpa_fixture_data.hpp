#pragma once

#include "wsvt/umpa_physical_fit.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace umpa_fixture {

struct Fixture {
    std::string id;
    std::vector<double> sample;
    std::vector<double> reference;
    std::vector<double> reference_mean;
    std::vector<double> weights;
    double expected_transmission = 0.0;
    double expected_visibility = 0.0;
    bool expect_numerical_valid = false;
    bool expect_physical_valid = false;
};

inline Fixture make_fixture(
    std::string id,
    double transmission,
    double visibility,
    bool masked,
    bool singular,
    bool numerical_valid,
    bool physical_valid) {
    Fixture fixture;
    fixture.id = std::move(id);
    fixture.expected_transmission = transmission;
    fixture.expected_visibility = visibility;
    fixture.expect_numerical_valid = numerical_valid;
    fixture.expect_physical_valid = physical_valid;
    const std::vector<double> window = wsvt::normalized_hamming_window_2d(2U);
    fixture.sample.reserve(100U);
    fixture.reference.reserve(100U);
    fixture.reference_mean.reserve(100U);
    fixture.weights.reserve(100U);
    for (std::size_t frame = 0; frame < 4U; ++frame) {
        std::vector<double> reference_patch(25U, 0.0);
        for (std::size_t y = 0; y < 5U; ++y) {
            for (std::size_t x = 0; x < 5U; ++x) {
                const std::size_t index = y * 5U + x;
                if (singular) {
                    reference_patch[index] = 80.0 + 3.0 * static_cast<double>(frame);
                } else {
                    const std::size_t texture_index =
                        ((frame + 1U) * (y + 2U * x)) % 7U;
                    reference_patch[index] =
                        80.0 + 3.0 * static_cast<double>(frame) +
                        0.5 * static_cast<double>(y) +
                        0.25 * static_cast<double>(x) +
                        static_cast<double>(texture_index);
                }
            }
        }
        double mean = reference_patch[0];
        if (!singular) {
            mean = 0.0;
            for (std::size_t index = 0; index < 25U; ++index) {
                mean += window[index] * reference_patch[index];
            }
        }
        const double alpha = transmission * visibility;
        const double beta = transmission * (1.0 - visibility);
        for (std::size_t index = 0; index < 25U; ++index) {
            const bool excluded = masked && ((index + frame) % 7U == 0U);
            fixture.sample.push_back(excluded
                ? std::numeric_limits<double>::quiet_NaN()
                : alpha * reference_patch[index] + beta * mean);
            fixture.reference.push_back(excluded
                ? std::numeric_limits<double>::quiet_NaN()
                : reference_patch[index]);
            fixture.reference_mean.push_back(mean);
            fixture.weights.push_back(excluded ? 0.0 : window[index]);
        }
    }
    return fixture;
}

inline std::vector<Fixture> all_fixtures() {
    return {
        make_fixture("identity_T1_D1", 1.0, 1.0, false, false, true, true),
        make_fixture("known_T082_D063", 0.82, 0.63, false, false, true, true),
        make_fixture("masked_known_T091_D074", 0.91, 0.74, true, false, true, true),
        make_fixture("unphysical_T090_D120", 0.90, 1.20, false, false, true, false),
        make_fixture("near_singular_reference", 0.85, 0.70, false, true, false, false),
    };
}

}  // namespace umpa_fixture
