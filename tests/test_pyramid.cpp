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
