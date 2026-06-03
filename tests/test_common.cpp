#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "wsvt/common.hpp"

#include <cstddef>
#include <vector>

using namespace wsvt;

TEST_CASE("std_depth_chw_per_pixel matches HWD standard deviation", "[common][stats]") {
    constexpr std::size_t ch = 5;
    constexpr std::size_t h = 3;
    constexpr std::size_t w = 4;

    std::vector<float> chw(ch * h * w, 0.0f);
    for (std::size_t c = 0; c < ch; ++c) {
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t x = 0; x < w; ++x) {
                chw[(c * h + y) * w + x] =
                    static_cast<float>(10 * c + 3 * y + x) * 0.25f;
            }
        }
    }

    std::vector<float> hwd(h * w * ch, 0.0f);
    for (std::size_t c = 0; c < ch; ++c) {
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t x = 0; x < w; ++x) {
                hwd[(y * w + x) * ch + c] = chw[(c * h + y) * w + x];
            }
        }
    }

    const auto direct = std_depth_chw_per_pixel(chw, ch, h, w);
    const auto hwd_std = std_depth_hwd(
        TensorView3D<const float, Layout::HWD>{hwd.data(), Shape3D{h, w, ch}});

    REQUIRE(direct.shape().h == h);
    REQUIRE(direct.shape().w == w);
    REQUIRE(hwd_std.shape().h == h);
    REQUIRE(hwd_std.shape().w == w);

    for (std::size_t i = 0; i < h * w; ++i) {
        REQUIRE_THAT(static_cast<double>(direct.flat()[i]),
                     Catch::Matchers::WithinAbs(static_cast<double>(hwd_std.flat()[i]), 1.0e-6));
    }
}
