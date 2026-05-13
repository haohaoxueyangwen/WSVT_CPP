#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "wsvt/image.hpp"
#include "wsvt/phase_recovery.hpp"

#include <cmath>
#include <cstddef>

using namespace wsvt;

TEST_CASE("frankot_chellappa recovers sinusoidal phase", "[phase]") {
    constexpr std::size_t h = 64, w = 64;
    constexpr float kx = 2.0f * 3.14159265f / static_cast<float>(w);
    constexpr float ky = 2.0f * 3.14159265f / static_cast<float>(h);

    Image2D<float> dpc_x({h, w});
    Image2D<float> dpc_y({h, w});
    Image2D<float> expected_phase({h, w});

    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            float fx = static_cast<float>(x);
            float fy = static_cast<float>(y);
            expected_phase(y, x) = std::sin(kx * fx) + std::sin(ky * fy);
            dpc_x(y, x) = kx * std::cos(kx * fx);
            dpc_y(y, x) = ky * std::cos(ky * fy);
        }
    }

    auto phase = frankot_chellappa(dpc_x, dpc_y);

    REQUIRE(phase.shape().h == h);
    REQUIRE(phase.shape().w == w);

    float mean_expected = 0.0f, mean_phase = 0.0f;
    for (std::size_t i = 0; i < h * w; ++i) {
        mean_expected += expected_phase.data()[i];
        mean_phase += phase.data()[i];
    }
    mean_expected /= static_cast<float>(h * w);
    mean_phase /= static_cast<float>(h * w);

    double max_err = 0.0;
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            double err = std::abs(
                (phase(y, x) - mean_phase) - (expected_phase(y, x) - mean_expected));
            if (err > max_err) max_err = err;
        }
    }
    REQUIRE(max_err < 0.01);
}

TEST_CASE("frankot_chellappa zero gradient gives zero phase", "[phase]") {
    constexpr std::size_t h = 32, w = 32;
    Image2D<float> dpc_x({h, w}, 0.0f);
    Image2D<float> dpc_y({h, w}, 0.0f);

    auto phase = frankot_chellappa(dpc_x, dpc_y);

    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            REQUIRE_THAT(static_cast<double>(phase(y, x)),
                         Catch::Matchers::WithinAbs(0.0, 1e-5));
        }
    }
}

TEST_CASE("frankot_chellappa rejects mismatched shapes", "[phase]") {
    Image2D<float> a({32, 32}, 0.0f);
    Image2D<float> b({16, 16}, 0.0f);
    REQUIRE_THROWS_AS(frankot_chellappa(a, b), std::invalid_argument);
}
