#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "wsvt/wavelet_ops.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

using namespace wsvt;

static std::vector<float> make_signal_chw(std::size_t ch, std::size_t h, std::size_t w) {
    std::vector<float> data(ch * h * w);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.07f) + 0.5f * std::cos(static_cast<float>(i) * 0.13f);
    }
    return data;
}

static std::vector<float> chw_to_hwd(const std::vector<float>& chw, std::size_t ch, std::size_t h, std::size_t w) {
    std::vector<float> hwd(h * w * ch);
    for (std::size_t c = 0; c < ch; ++c) {
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t x = 0; x < w; ++x) {
                hwd[(y * w + x) * ch + c] = chw[(c * h + y) * w + x];
            }
        }
    }
    return hwd;
}

TEST_CASE("wavelet_transform_hwd matches CHW path numerically", "[wavelet][hwd]") {
    constexpr std::size_t ch = 9, h = 32, w = 32;
    auto signal_chw = make_signal_chw(ch, h, w);
    auto signal_hwd = chw_to_hwd(signal_chw, ch, h, w);

    for (auto wavelet : {WaveletFamily::Db2, WaveletFamily::Db3, WaveletFamily::Db6}) {
        DYNAMIC_SECTION("wavelet=" << static_cast<int>(wavelet)) {
            SECTION("level=1, return_level=1") {
                auto chw_result = wavelet_transform(signal_chw, ch, h, w, wavelet, 1, 1);
                auto hwd_result = wavelet_transform_hwd(signal_hwd, h, w, ch, wavelet, 1, 1);

                REQUIRE(chw_result.out_h == hwd_result.out_h);
                REQUIRE(chw_result.out_w == hwd_result.out_w);
                REQUIRE(chw_result.out_depth == hwd_result.out_depth);
                REQUIRE(chw_result.coeffs_filter.size() == hwd_result.coeffs_filter.size());

                for (std::size_t i = 0; i < chw_result.coeffs_filter.size(); ++i) {
                    REQUIRE_THAT(static_cast<double>(hwd_result.coeffs_filter[i]),
                                 Catch::Matchers::WithinAbs(static_cast<double>(chw_result.coeffs_filter[i]), 1e-5));
                }
            }

            SECTION("level=2, return_level=2") {
                auto chw_result = wavelet_transform(signal_chw, ch, h, w, wavelet, 2, 2);
                auto hwd_result = wavelet_transform_hwd(signal_hwd, h, w, ch, wavelet, 2, 2);

                REQUIRE(chw_result.out_depth == hwd_result.out_depth);
                REQUIRE(chw_result.coeffs_filter.size() == hwd_result.coeffs_filter.size());

                for (std::size_t i = 0; i < chw_result.coeffs_filter.size(); ++i) {
                    REQUIRE_THAT(static_cast<double>(hwd_result.coeffs_filter[i]),
                                 Catch::Matchers::WithinAbs(static_cast<double>(chw_result.coeffs_filter[i]), 1e-5));
                }
            }
        }
    }
}

TEST_CASE("wavelet_transform_hwd output shape", "[wavelet][hwd]") {
    constexpr std::size_t depth = 25, h = 64, w = 64;
    std::vector<float> signal(h * w * depth);
    for (std::size_t i = 0; i < signal.size(); ++i) {
        signal[i] = static_cast<float>(i % 100) * 0.01f;
    }

    auto result = wavelet_transform_hwd(signal, h, w, depth, WaveletFamily::Db2, 2, 2);
    REQUIRE(result.out_h == h);
    REQUIRE(result.out_w == w);
    REQUIRE(result.out_depth > 0);
    REQUIRE(result.coeffs_filter.size() == h * w * result.out_depth);
    REQUIRE(result.level_name.size() == 2);
}
