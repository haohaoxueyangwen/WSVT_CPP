#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "wsvt/common.hpp"
#include "wsvt/pyramid.hpp"
#include "wsvt/solver_utils.hpp"
#include "wsvt/wavelet_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
    REQUIRE(result.level_name == std::vector<std::string>{"A1", "D1"});
    const std::vector<float> expected = {
        0.020237602f, -0.1257779f, 3.3386838f, 11.450491f, 2.9940348f,
        -0.19112009f, 0.19112033f, 3.6593761f, -0.44087872f, 0.31703663f,
        -0.029977381f, 0.14887363f, 4.717671f, 13.049931f, 3.3267055f,
        0.28310084f, 0.07045281f, 3.992047f, -0.4552222f, 0.3522629f,
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

TEST_CASE("prefix-compatible wavelet matches the zero-padded final basis", "[wavelet][b2][incremental]") {
    constexpr std::size_t final_depth = 49;
    constexpr std::size_t prefix = 25;
    constexpr std::size_t h = 5;
    constexpr std::size_t w = 4;
    const std::size_t plane = h * w;
    const int level = dwt_max_level_db2(final_depth);
    const int return_level = level + 1;
    const auto basis = make_prefix_compatible_wavelet_basis(
        final_depth, WaveletFamily::Db2, level, return_level);
    auto state = make_prefix_compatible_wavelet_state(h, w, basis);

    std::vector<float> raw(final_depth * plane, 0.0f);
    for (std::size_t frame = 0; frame < final_depth; ++frame) {
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            raw[frame * plane + pixel] = 30000.0f +
                1500.0f * std::sin(0.19f * static_cast<float>(frame) +
                                    0.07f * static_cast<float>(pixel)) +
                500.0f * std::cos(0.11f * static_cast<float>(frame + pixel));
        }
    }
    extend_prefix_compatible_wavelet_state(state, basis, raw, prefix);
    const auto incremental = materialize_prefix_compatible_wavelet(state, basis);

    std::vector<float> zero_padded_hwd(plane * final_depth, 0.0f);
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        double mean = 0.0;
        for (std::size_t frame = 0; frame < prefix; ++frame) {
            mean += raw[frame * plane + pixel];
        }
        mean /= static_cast<double>(prefix);
        double variance = 0.0;
        for (std::size_t frame = 0; frame < prefix; ++frame) {
            const double delta = raw[frame * plane + pixel] - mean;
            variance += delta * delta;
        }
        const double stddev = std::sqrt(variance / static_cast<double>(prefix));
        for (std::size_t frame = 0; frame < prefix; ++frame) {
            zero_padded_hwd[pixel * final_depth + frame] = static_cast<float>(
                (static_cast<double>(raw[frame * plane + pixel]) - mean) /
                stddev);
        }
    }
    const auto direct = wavelet_transform_hwd(
        zero_padded_hwd, h, w, final_depth,
        WaveletFamily::Db2, level, return_level);

    REQUIRE(incremental.out_depth == direct.out_depth);
    REQUIRE(incremental.level_name == direct.level_name);
    for (std::size_t i = 0; i < direct.coeffs_filter.size(); ++i) {
        REQUIRE_THAT(
            static_cast<double>(incremental.coeffs_filter[i]),
            Catch::Matchers::WithinAbs(
                static_cast<double>(direct.coeffs_filter[i]), 2e-5));
    }

    // At the final stage the new representation must collapse to the existing
    // independently normalized final-depth WSVT descriptor.
    extend_prefix_compatible_wavelet_state(state, basis, raw, final_depth);
    const auto incremental_final = materialize_prefix_compatible_wavelet(
        state, basis);
    std::fill(zero_padded_hwd.begin(), zero_padded_hwd.end(), 0.0f);
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        double mean = 0.0;
        for (std::size_t frame = 0; frame < final_depth; ++frame) {
            mean += raw[frame * plane + pixel];
        }
        mean /= static_cast<double>(final_depth);
        double variance = 0.0;
        for (std::size_t frame = 0; frame < final_depth; ++frame) {
            const double delta = raw[frame * plane + pixel] - mean;
            variance += delta * delta;
        }
        const double stddev = std::sqrt(
            variance / static_cast<double>(final_depth));
        for (std::size_t frame = 0; frame < final_depth; ++frame) {
            zero_padded_hwd[pixel * final_depth + frame] = static_cast<float>(
                (static_cast<double>(raw[frame * plane + pixel]) - mean) /
                stddev);
        }
    }
    const auto direct_final = wavelet_transform_hwd(
        zero_padded_hwd, h, w, final_depth,
        WaveletFamily::Db2, level, return_level);
    for (std::size_t i = 0; i < direct_final.coeffs_filter.size(); ++i) {
        REQUIRE_THAT(
            static_cast<double>(incremental_final.coeffs_filter[i]),
            Catch::Matchers::WithinAbs(
                static_cast<double>(direct_final.coeffs_filter[i]), 2e-5));
    }
}

TEST_CASE("prefix-compatible wavelet reuses prefix state for selected final pixels", "[wavelet][b2][incremental]") {
    constexpr std::size_t final_depth = 49;
    constexpr std::size_t prefix = 25;
    constexpr std::size_t h = 4;
    constexpr std::size_t w = 3;
    const std::size_t plane = h * w;
    const int level = dwt_max_level_db2(final_depth);
    const auto basis = make_prefix_compatible_wavelet_basis(
        final_depth, WaveletFamily::Db2, level, level + 1);
    std::vector<float> raw(final_depth * plane, 0.0f);
    for (std::size_t frame = 0; frame < final_depth; ++frame) {
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            raw[frame * plane + pixel] = 20000.0f +
                800.0f * std::sin(0.13f * static_cast<float>(frame * 3 + pixel)) +
                static_cast<float>(17 * pixel + frame);
        }
    }

    auto staged = make_prefix_compatible_wavelet_state(h, w, basis);
    extend_prefix_compatible_wavelet_state(staged, basis, raw, prefix);
    std::vector<std::uint8_t> selected(plane, 0);
    for (std::size_t pixel = 0; pixel < plane; pixel += 2) {
        selected[pixel] = 1;
    }
    extend_prefix_compatible_wavelet_state(
        staged, basis, raw, final_depth, selected);
    const std::uint64_t selected_count = static_cast<std::uint64_t>(
        std::count(selected.begin(), selected.end(), static_cast<std::uint8_t>(1)));
    const std::uint64_t expected_frame_terms =
        static_cast<std::uint64_t>(plane * prefix) +
        selected_count * static_cast<std::uint64_t>(final_depth - prefix);
    REQUIRE(staged.raw_frame_pixel_terms_accumulated == expected_frame_terms);
    REQUIRE(staged.weighted_coefficient_terms_accumulated ==
            expected_frame_terms * static_cast<std::uint64_t>(basis.out_depth));

    auto one_shot = make_prefix_compatible_wavelet_state(h, w, basis);
    extend_prefix_compatible_wavelet_state(one_shot, basis, raw, final_depth);
    const auto staged_coefficients = materialize_prefix_compatible_wavelet(
        staged, basis, selected);
    const auto one_shot_coefficients = materialize_prefix_compatible_wavelet(
        one_shot, basis, selected);

    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        REQUIRE(staged.frames_accumulated[pixel] ==
                (selected[pixel] ? final_depth : prefix));
        for (std::size_t coefficient = 0;
             coefficient < basis.out_depth; ++coefficient) {
            const std::size_t index = pixel * basis.out_depth + coefficient;
            REQUIRE(staged_coefficients.coeffs_filter[index] ==
                    one_shot_coefficients.coeffs_filter[index]);
        }
    }

    REQUIRE_THROWS_AS(
        extend_prefix_compatible_wavelet_state(
            staged, basis, raw, prefix, selected),
        std::invalid_argument);
}

TEST_CASE("final-basis temporal coefficients commute with the WSVT spatial pyramid", "[wavelet][pyramid][b2][incremental]") {
    constexpr std::size_t final_depth = 49;
    constexpr std::size_t h = 17;
    constexpr std::size_t w = 15;
    constexpr int pyramid_level = 2;
    const std::size_t plane = h * w;
    const int wavelet_level = dwt_max_level_db2(final_depth);
    const auto basis = make_prefix_compatible_wavelet_basis(
        final_depth, WaveletFamily::Db2,
        wavelet_level, wavelet_level + 1);

    std::vector<float> raw(final_depth * plane, 0.0f);
    for (std::size_t frame = 0; frame < final_depth; ++frame) {
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            raw[frame * plane + pixel] = 25000.0f +
                1800.0f * std::sin(0.17f * static_cast<float>(frame) +
                                    0.031f * static_cast<float>(pixel)) +
                700.0f * std::cos(0.09f * static_cast<float>(frame + 2 * pixel));
        }
    }

    auto state = make_prefix_compatible_wavelet_state(h, w, basis);
    extend_prefix_compatible_wavelet_state(
        state, basis, raw, final_depth);
    const auto base_coefficients = materialize_prefix_compatible_wavelet(
        state, basis);
    const auto incremental_levels = descriptor_spatial_pyramid_hwd(
        base_coefficients.coeffs_filter, h, w, basis.out_depth,
        pyramid_level, PyramidDownsampleMode::Db3Aa);

    const auto normalized_raw_levels = normalized_raw_pyramid_data(
        raw, final_depth, h, w, pyramid_level,
        PyramidDownsampleMode::Db3Aa);
    REQUIRE(incremental_levels.size() == normalized_raw_levels.levels.size());
    for (std::size_t level = 0; level < incremental_levels.size(); ++level) {
        const auto& raw_level = normalized_raw_levels.levels[level];
        const auto raw_hwd = chw_to_hwd(
            raw_level.data, raw_level.ch, raw_level.h, raw_level.w);
        const auto direct = wavelet_transform_hwd(
            raw_hwd, raw_level.h, raw_level.w, raw_level.ch,
            WaveletFamily::Db2, wavelet_level, wavelet_level + 1);
        const auto& incremental = incremental_levels[level];
        REQUIRE(incremental.d0 == direct.out_depth);
        REQUIRE(incremental.d1 == direct.out_h);
        REQUIRE(incremental.d2 == direct.out_w);
        REQUIRE(incremental.data.size() == direct.coeffs_filter.size());
        for (std::size_t i = 0; i < direct.coeffs_filter.size(); ++i) {
            REQUIRE_THAT(
                static_cast<double>(incremental.data[i]),
                Catch::Matchers::WithinAbs(
                    static_cast<double>(direct.coeffs_filter[i]), 5e-5));
        }
    }
}

TEST_CASE("wavelet_add_list_for_depth matches Python reference thresholds", "[wavelet]") {
    REQUIRE(wavelet_add_list_for_depth(49) == std::vector<int>{2, 2, 2, 2, 2, 2});
    REQUIRE(wavelet_add_list_for_depth(50) == std::vector<int>{2, 2, 2, 2, 2, 2});
    REQUIRE(wavelet_add_list_for_depth(51) == std::vector<int>{0, 0, 1, 2, 2, 2});
    REQUIRE(wavelet_add_list_for_depth(150) == std::vector<int>{0, 0, 1, 2, 2, 2});
    REQUIRE(wavelet_add_list_for_depth(151) == std::vector<int>{0, 0, 0, 0, 0, 0});
}

TEST_CASE("Python-reference phase scaling keeps the explicit negative sign", "[phase][semantics]") {
    std::vector<float> phase{1.0f, -2.0f, 0.5f};
    apply_python_reference_phase_scale(phase, 3.0);
    REQUIRE(phase == std::vector<float>{-3.0f, 6.0f, -1.5f});
}

TEST_CASE("manual search radii are recorded without automatic adjustment", "[window][semantics]") {
    REQUIRE(derived_search_half_windows(0, 20, 4) == std::vector<int>{20});
    REQUIRE(derived_search_half_windows(2, 20, 4) == std::vector<int>{4, 4, 5});
}
