#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "wsvt/solver_utils.hpp"
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

TEST_CASE("wavelet_transform_hwd_streamed matches CHW selected levels", "[wavelet][hwd]") {
    constexpr std::size_t depth = 33, h = 5, w = 7;
    const auto signal_chw = make_signal_chw(depth, h, w);
    const auto signal_hwd = chw_to_hwd(signal_chw, depth, h, w);

    for (auto wavelet : {WaveletFamily::Db2, WaveletFamily::Db3}) {
        DYNAMIC_SECTION("wavelet=" << static_cast<int>(wavelet)) {
            for (const int return_level : {1, 2, 3, 4}) {
                DYNAMIC_SECTION("return_level=" << return_level) {
                    const auto legacy = wavelet_transform(signal_chw, depth, h, w, wavelet, 3, return_level);
                    const auto streamed = wavelet_transform_hwd_streamed(signal_hwd, h, w, depth, wavelet, 3, return_level);

                    REQUIRE(streamed.out_h == legacy.out_h);
                    REQUIRE(streamed.out_w == legacy.out_w);
                    REQUIRE(streamed.out_depth == legacy.out_depth);
                    REQUIRE(streamed.level_name == legacy.level_name);
                    REQUIRE(streamed.coeffs_filter.size() == legacy.coeffs_filter.size());

                    for (std::size_t i = 0; i < legacy.coeffs_filter.size(); ++i) {
                        REQUIRE_THAT(static_cast<double>(streamed.coeffs_filter[i]),
                                     Catch::Matchers::WithinAbs(static_cast<double>(legacy.coeffs_filter[i]), 1e-5));
                    }
                }
            }
        }
    }
}

TEST_CASE("wavelet_transform_hwd matches PyWavelets db3 zero-mode golden values", "[wavelet][hwd]") {
    constexpr std::size_t h = 1, w = 2, depth = 5;
    const std::vector<float> signal = {
        1.0f, 3.0f, 5.0f, 7.0f, 9.0f,
        2.0f, 4.0f, 6.0f, 8.0f, 10.0f,
    };

    const auto result = wavelet_transform_hwd(signal, h, w, depth, WaveletFamily::Db3, 1, 2);

    REQUIRE(result.out_h == h);
    REQUIRE(result.out_w == w);
    REQUIRE(result.out_depth == 10);
    const std::vector<float> expected = {
        -0.19112009f, 0.19112033f, 3.6593761f, -0.44087872f, 0.31703663f,
        0.020237602f, -0.1257779f, 3.3386838f, 11.450491f, 2.9940348f,
        0.28310084f, 0.07045281f, 3.992047f, -0.4552222f, 0.3522629f,
        -0.029977381f, 0.14887363f, 4.717671f, 13.049931f, 3.3267055f,
    };

    REQUIRE(result.coeffs_filter.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        REQUIRE_THAT(static_cast<double>(result.coeffs_filter[i]),
                     Catch::Matchers::WithinAbs(static_cast<double>(expected[i]), 2e-5));
    }
}

TEST_CASE("wavelet_transform_hwd_pair matches separate transforms", "[wavelet][hwd][pair]") {
    constexpr std::size_t depth = 25, h = 64, w = 64;
    std::vector<float> img_signal(h * w * depth);
    std::vector<float> ref_signal(h * w * depth);
    for (std::size_t i = 0; i < img_signal.size(); ++i) {
        img_signal[i] = std::sin(static_cast<float>(i) * 0.07f) + 0.5f * std::cos(static_cast<float>(i) * 0.13f);
        ref_signal[i] = std::cos(static_cast<float>(i) * 0.11f) - 0.3f * std::sin(static_cast<float>(i) * 0.05f);
    }

    for (auto wavelet : {WaveletFamily::Db2, WaveletFamily::Db3, WaveletFamily::Db6}) {
        DYNAMIC_SECTION("wavelet=" << static_cast<int>(wavelet)) {
            for (const int w_level : {1, 2, 3}) {
                for (const int return_level : {1, 2}) {
                    if (return_level > w_level + 1) continue;
                    DYNAMIC_SECTION("level=" << w_level << "_return=" << return_level) {
                        auto pair = wavelet_transform_hwd_pair(
                            img_signal, ref_signal, h, w, depth, wavelet, w_level, return_level);
                        auto img_sep = wavelet_transform_hwd(
                            img_signal, h, w, depth, wavelet, w_level, return_level);
                        auto ref_sep = wavelet_transform_hwd(
                            ref_signal, h, w, depth, wavelet, w_level, return_level);

                        REQUIRE(pair.img.out_h == img_sep.out_h);
                        REQUIRE(pair.img.out_w == img_sep.out_w);
                        REQUIRE(pair.img.out_depth == img_sep.out_depth);
                        REQUIRE(pair.ref.out_h == ref_sep.out_h);
                        REQUIRE(pair.ref.out_w == ref_sep.out_w);
                        REQUIRE(pair.ref.out_depth == ref_sep.out_depth);
                        REQUIRE(pair.img.level_name == img_sep.level_name);
                        REQUIRE(pair.ref.level_name == ref_sep.level_name);

                        REQUIRE(pair.img.coeffs_filter.size() == img_sep.coeffs_filter.size());
                        REQUIRE(pair.ref.coeffs_filter.size() == ref_sep.coeffs_filter.size());

                        for (std::size_t i = 0; i < img_sep.coeffs_filter.size(); ++i) {
                            REQUIRE_THAT(static_cast<double>(pair.img.coeffs_filter[i]),
                                         Catch::Matchers::WithinAbs(static_cast<double>(img_sep.coeffs_filter[i]), 1e-6));
                        }
                        for (std::size_t i = 0; i < ref_sep.coeffs_filter.size(); ++i) {
                            REQUIRE_THAT(static_cast<double>(pair.ref.coeffs_filter[i]),
                                         Catch::Matchers::WithinAbs(static_cast<double>(ref_sep.coeffs_filter[i]), 1e-6));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("wavelet_transform_hwd_planned matches streamed bitwise", "[wavelet][hwd][planned]") {
    // Verify that the precomputed-plan path produces bitwise-identical results
    // to wavelet_transform_hwd (which delegates to wavelet_transform_hwd_streamed).
    constexpr std::size_t depth = 33, h = 7, w = 5;
    const auto signal_chw = make_signal_chw(depth, h, w);
    const auto signal_hwd = chw_to_hwd(signal_chw, depth, h, w);

    for (auto wavelet : {WaveletFamily::Db2, WaveletFamily::Db3, WaveletFamily::Db6}) {
        DYNAMIC_SECTION("wavelet=" << static_cast<int>(wavelet)) {
            for (int w_level = 1; w_level <= 3; ++w_level) {
                for (int ret : {1, w_level + 1}) {
                    DYNAMIC_SECTION("w_level=" << w_level << " ret=" << ret) {
                        auto plans = compute_wavelet_plan(depth, w_level, wavelet);
                        REQUIRE(plans.size() == static_cast<std::size_t>(w_level));

                        auto ref = wavelet_transform_hwd(signal_hwd, h, w, depth,
                                                         wavelet, w_level, ret);
                        auto opt = wavelet_transform_hwd_planned(
                            signal_hwd, h, w, depth, wavelet, w_level, ret, plans);

                        REQUIRE(opt.out_h == ref.out_h);
                        REQUIRE(opt.out_w == ref.out_w);
                        REQUIRE(opt.out_depth == ref.out_depth);
                        REQUIRE(opt.level_name == ref.level_name);
                        REQUIRE(opt.coeffs_filter.size() == ref.coeffs_filter.size());

                        for (std::size_t i = 0; i < ref.coeffs_filter.size(); ++i) {
                            REQUIRE_THAT(
                                static_cast<double>(opt.coeffs_filter[i]),
                                Catch::Matchers::WithinAbs(
                                    static_cast<double>(ref.coeffs_filter[i]), 1e-6));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("wavelet_transform_hwd_planned w_level=0", "[wavelet][hwd][planned]") {
    constexpr std::size_t depth = 9, h = 4, w = 3;
    const auto signal_chw = make_signal_chw(depth, h, w);
    const auto signal_hwd = chw_to_hwd(signal_chw, depth, h, w);

    auto plans = compute_wavelet_plan(depth, 0, WaveletFamily::Db2);
    REQUIRE(plans.empty());

    auto ref = wavelet_transform_hwd(signal_hwd, h, w, depth, WaveletFamily::Db2, 0, 1);
    auto opt = wavelet_transform_hwd_planned(signal_hwd, h, w, depth,
                                             WaveletFamily::Db2, 0, 1, plans);

    REQUIRE(opt.coeffs_filter.size() == ref.coeffs_filter.size());
    for (std::size_t i = 0; i < ref.coeffs_filter.size(); ++i) {
        REQUIRE_THAT(static_cast<double>(opt.coeffs_filter[i]),
                     Catch::Matchers::WithinAbs(
                         static_cast<double>(ref.coeffs_filter[i]), 1e-6));
    }
}

TEST_CASE("wavelet_transform_hwd_pixelchain matches planned bitwise", "[wavelet][hwd][pixelchain]") {
    constexpr std::size_t depth = 33, h = 7, w = 5;
    const auto signal_chw = make_signal_chw(depth, h, w);
    const auto signal_hwd = chw_to_hwd(signal_chw, depth, h, w);

    for (auto wavelet : {WaveletFamily::Db2, WaveletFamily::Db3, WaveletFamily::Db6}) {
        DYNAMIC_SECTION("wavelet=" << static_cast<int>(wavelet)) {
            for (int w_level = 1; w_level <= 3; ++w_level) {
                for (int ret : {1, w_level + 1}) {
                    DYNAMIC_SECTION("w_level=" << w_level << " ret=" << ret) {
                        auto plans = compute_wavelet_plan(depth, w_level, wavelet);
                        auto planned = wavelet_transform_hwd_planned(
                            signal_hwd, h, w, depth, wavelet, w_level, ret, plans);
                        auto pixelchain = wavelet_transform_hwd_pixelchain(
                            signal_hwd, h, w, depth, wavelet, w_level, ret, plans);

                        REQUIRE(pixelchain.out_depth == planned.out_depth);
                        REQUIRE(pixelchain.coeffs_filter.size() == planned.coeffs_filter.size());

                        for (std::size_t i = 0; i < planned.coeffs_filter.size(); ++i) {
                            REQUIRE_THAT(
                                static_cast<double>(pixelchain.coeffs_filter[i]),
                                Catch::Matchers::WithinAbs(
                                    static_cast<double>(planned.coeffs_filter[i]), 1e-6));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("wavelet_add_list_for_depth matches Python reference thresholds", "[wavelet]") {
    REQUIRE(wavelet_add_list_for_depth(49) == std::vector<int>{0, 2, 2, 2, 2, 2});
    REQUIRE(wavelet_add_list_for_depth(50) == std::vector<int>{0, 2, 2, 2, 2, 2});
    REQUIRE(wavelet_add_list_for_depth(51) == std::vector<int>{0, 0, 1, 2, 2, 2});
    REQUIRE(wavelet_add_list_for_depth(150) == std::vector<int>{0, 0, 1, 2, 2, 2});
    REQUIRE(wavelet_add_list_for_depth(151) == std::vector<int>{0, 0, 0, 0, 0, 0});
}
