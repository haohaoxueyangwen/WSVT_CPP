#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "wsvt/wxst_pipeline.hpp"

#include <cstddef>
#include <vector>

using namespace wsvt;

TEST_CASE("WXST solver smoke test", "[wxst][smoke]") {
    constexpr std::size_t h = 32, w = 32;
    std::vector<float> img(h * w);
    std::vector<float> ref(h * w);
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            ref[y * w + x] = static_cast<float>((y + 1) * (x + 1));
            img[y * w + x] = ref[y * w + ((x + 1) % w)];
        }
    }

    WXST wxst(img, ref, h, w,
              /*m_image=*/32, /*n_s=*/1, /*cal_half_window=*/2,
              /*n_s_extend=*/2, /*n_cores=*/2, /*n_group=*/2,
              /*energy=*/14000.0, /*p_x=*/0.65e-6, /*z=*/0.5,
              /*wavelet_level_cut=*/2, /*pyramid_level=*/1,
              /*n_iter=*/1, /*use_estimate=*/false, /*use_wavelet=*/true,
              /*use_gpu=*/0);

    auto result = wxst.solver();

    REQUIRE(result.h > 0);
    REQUIRE(result.w > 0);
    REQUIRE(result.displace_y.size() == result.h * result.w);
    REQUIRE(result.displace_x.size() == result.h * result.w);
    REQUIRE(result.dpc_y.size() == result.h * result.w);
    REQUIRE(result.dpc_x.size() == result.h * result.w);
}

TEST_CASE("WXST preserves Python full-ROI transmission semantics", "[wxst][transmission]") {
    constexpr std::size_t h = 16, w = 16;
    std::vector<float> img(h * w);
    std::vector<float> ref(h * w);
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            const std::size_t idx = y * w + x;
            ref[idx] = 10.0f + static_cast<float>(y + x);
            img[idx] = 1.25f * ref[idx];
        }
    }

    WXST wxst(img, ref, h, w,
              /*m_image=*/16, /*n_s=*/0, /*cal_half_window=*/2,
              /*n_s_extend=*/1, /*n_cores=*/1, /*n_group=*/1,
              /*energy=*/14000.0, /*p_x=*/0.65e-6, /*z=*/0.5,
              /*wavelet_level_cut=*/1, /*pyramid_level=*/0,
              /*n_iter=*/1, /*use_estimate=*/false, /*use_wavelet=*/false,
              /*use_gpu=*/0);

    auto result = wxst.solver();

    REQUIRE(result.h == h - 4);
    REQUIRE(result.w == w - 4);
    REQUIRE(result.transmission_h == h);
    REQUIRE(result.transmission_w == w);
    REQUIRE(result.transmission.size() == h * w);
    for (const float value : result.transmission) {
        REQUIRE_THAT(static_cast<double>(value), Catch::Matchers::WithinAbs(1.25, 1e-6));
    }
}
