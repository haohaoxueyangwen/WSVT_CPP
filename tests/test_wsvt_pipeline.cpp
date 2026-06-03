#include <catch2/catch_test_macros.hpp>

#include "wsvt/wsvt_pipeline.hpp"

#include <cstddef>
#include <vector>

using namespace wsvt;

namespace {

std::vector<float> make_stack(std::size_t ch, std::size_t h, std::size_t w, float bias) {
    std::vector<float> data(ch * h * w, 0.0f);
    for (std::size_t c = 0; c < ch; ++c) {
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t x = 0; x < w; ++x) {
                const float ramp = static_cast<float>(0.07 * static_cast<double>(y) +
                                                      0.11 * static_cast<double>(x) +
                                                      0.19 * static_cast<double>(c));
                data[c * h * w + y * w + x] = bias + ramp;
            }
        }
    }
    return data;
}

} // namespace

TEST_CASE("WSVT computes raw darkfield by default", "[wsvt][darkfield]") {
    constexpr std::size_t ch = 4;
    constexpr std::size_t h = 24;
    constexpr std::size_t w = 24;
    auto img = make_stack(ch, h, w, 12.0f);
    auto ref = make_stack(ch, h, w, 10.0f);

    WSVT solver(
        img, ref, ch, h, w,
        /*crop=*/0, /*cal_half_window=*/2, /*n_template=*/0,
        /*n_s_extend=*/1, /*n_cores=*/2, /*n_group=*/1,
        /*energy=*/14000.0, /*p_x=*/0.65e-6, /*mag_factor=*/1.0, /*z=*/0.5,
        /*wavelet_level_cut=*/1, /*pyramid_level=*/0, /*n_iter=*/1,
        /*use_estimate=*/false, /*use_wavelet=*/false, /*use_gpu=*/0);

    const auto result = solver.solver();

    REQUIRE(result.h > 0);
    REQUIRE(result.w > 0);
    REQUIRE(result.darkfield.size() == result.transmission_h * result.transmission_w);
    REQUIRE(result.darkfield_time_s >= 0.0);
    REQUIRE(result.darkfield_nd.size() == result.h * result.w);
}

TEST_CASE("WSVT can skip raw darkfield while keeping core outputs", "[wsvt][darkfield]") {
    constexpr std::size_t ch = 4;
    constexpr std::size_t h = 24;
    constexpr std::size_t w = 24;
    auto img = make_stack(ch, h, w, 12.0f);
    auto ref = make_stack(ch, h, w, 10.0f);

    WSVT solver(
        img, ref, ch, h, w,
        /*crop=*/0, /*cal_half_window=*/2, /*n_template=*/0,
        /*n_s_extend=*/1, /*n_cores=*/2, /*n_group=*/1,
        /*energy=*/14000.0, /*p_x=*/0.65e-6, /*mag_factor=*/1.0, /*z=*/0.5,
        /*wavelet_level_cut=*/1, /*pyramid_level=*/0, /*n_iter=*/1,
        /*use_estimate=*/false, /*use_wavelet=*/false, /*use_gpu=*/0,
        /*calc_darkfield=*/false);

    const auto result = solver.solver();

    REQUIRE(result.h > 0);
    REQUIRE(result.w > 0);
    REQUIRE(result.displace_x.size() == result.h * result.w);
    REQUIRE(result.displace_y.size() == result.h * result.w);
    REQUIRE(result.darkfield.empty());
    REQUIRE(result.darkfield_time_s == 0.0);
    REQUIRE(result.darkfield_nd.size() == result.h * result.w);
}
