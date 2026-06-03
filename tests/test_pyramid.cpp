#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "wsvt/pyramid.hpp"

#include <cstddef>
#include <numeric>
#include <vector>

using namespace wsvt;

static std::vector<float> make_ramp(std::size_t ch, std::size_t h, std::size_t w) {
    std::vector<float> data(ch * h * w);
    std::iota(data.begin(), data.end(), 1.0f);
    return data;
}

TEST_CASE("pyramid_data basic shapes", "[pyramid]") {
    constexpr std::size_t ch = 1, h = 64, w = 64;
    auto ref = make_ramp(ch, h, w);
    auto img = make_ramp(ch, h, w);

    SECTION("Mean2x2 mode, pyramid_level=1") {
        auto result = pyramid_data(ref, img, ch, h, w, 1, 0, PyramidDownsampleMode::Mean2x2);
        REQUIRE(result.ref_levels.size() == 2);
        REQUIRE(result.img_levels.size() == 2);
    }

    SECTION("Db3Aa mode, pyramid_level=1") {
        auto result = pyramid_data(ref, img, ch, h, w, 1, 0, PyramidDownsampleMode::Db3Aa);
        REQUIRE(result.ref_levels.size() == 2);
        REQUIRE(result.img_levels.size() == 2);
    }

    SECTION("pyramid_level=2") {
        auto result = pyramid_data(ref, img, ch, h, w, 2, 0, PyramidDownsampleMode::Mean2x2);
        REQUIRE(result.ref_levels.size() == 3);
    }
}

TEST_CASE("pyramid_data dimension consistency", "[pyramid]") {
    constexpr std::size_t ch = 1, h = 128, w = 128;
    auto ref = make_ramp(ch, h, w);
    auto img = make_ramp(ch, h, w);

    auto result = pyramid_data(ref, img, ch, h, w, 2, 0, PyramidDownsampleMode::Mean2x2);

    for (const auto& lv : result.ref_levels) {
        REQUIRE(lv.data.size() == lv.d0 * lv.d1 * lv.d2);
    }
    for (const auto& lv : result.img_levels) {
        REQUIRE(lv.data.size() == lv.d0 * lv.d1 * lv.d2);
    }
}

TEST_CASE("pyramid_data rejects mismatched input", "[pyramid]") {
    std::vector<float> ref(64 * 64, 1.0f);
    std::vector<float> img(32 * 32, 1.0f);
    REQUIRE_THROWS_AS(
        pyramid_data(ref, img, 1, 64, 64, 1, 0, PyramidDownsampleMode::Mean2x2),
        std::invalid_argument);
}

TEST_CASE("pyramid_data db3 aa matches PyWavelets zero-mode golden values", "[pyramid]") {
    constexpr std::size_t ch = 2, h = 5, w = 5;
    auto ref = make_ramp(ch, h, w);
    auto img = make_ramp(ch, h, w);

    const auto result = pyramid_data(
        ref, img, ch, h, w, 1, 0,
        PyramidDownsampleMode::Db3Aa,
        PyramidNormalizationMode::InitialStack);

    REQUIRE(result.ref_levels.size() == 2);
    const auto& level = result.ref_levels[1];
    REQUIRE(level.d0 == ch);
    REQUIRE(level.d1 == 5);
    REQUIRE(level.d2 == 5);

    const std::vector<float> expected = {
        -2.5215445e-03f, 2.5215445e-03f, 1.3791621e-02f, -1.3791621e-02f,
        6.9245823e-02f, -6.9245823e-02f, 8.0315828e-02f, -8.0315828e-02f,
        1.6705045e-02f, -1.6705045e-02f, 1.3791620e-02f, -1.3791620e-02f,
        -7.5433448e-02f, 7.5433448e-02f, -3.7874091e-01f, 3.7874091e-01f,
        -4.3928847e-01f, 4.3928847e-01f, -9.1368459e-02f, 9.1368459e-02f,
        6.9245830e-02f, -6.9245830e-02f, -3.7874091e-01f, 3.7874091e-01f,
        -1.9016060e+00f, 1.9016060e+00f, -2.2056069e+00f, 2.2056069e+00f,
        -4.5874846e-01f, 4.5874846e-01f, 8.0315821e-02f, -8.0315821e-02f,
        -4.3928847e-01f, 4.3928847e-01f, -2.2056067e+00f, 2.2056067e+00f,
        -2.5582068e+00f, 2.5582068e+00f, -5.3208637e-01f, 5.3208637e-01f,
        1.6705045e-02f, -1.6705045e-02f, -9.1368452e-02f, 9.1368452e-02f,
        -4.5874840e-01f, 4.5874840e-01f, -5.3208643e-01f, 5.3208643e-01f,
        -1.1066969e-01f, 1.1066969e-01f,
    };

    REQUIRE(level.data.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        REQUIRE_THAT(static_cast<double>(level.data[i]),
                     Catch::Matchers::WithinAbs(static_cast<double>(expected[i]), 2e-5));
    }
}
