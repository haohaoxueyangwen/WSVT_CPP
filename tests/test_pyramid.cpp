#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "wsvt/pyramid.hpp"

#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

using namespace wsvt;

static std::vector<float> make_ramp(std::size_t ch, std::size_t h, std::size_t w) {
    std::vector<float> data(ch * h * w);
    std::iota(data.begin(), data.end(), 1.0f);
    return data;
}

static std::size_t wrap_index(long long v, std::size_t n) {
    const long long m = static_cast<long long>(n);
    long long r = v % m;
    if (r < 0) {
        r += m;
    }
    return static_cast<std::size_t>(r);
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

TEST_CASE("stack_and_normalize_template_hwd matches separate stack + normalize", "[pyramid][template]") {
    // Golden test: verify fused pixel-major function produces bitwise-identical
    // results as the old separate stack_template_window_hwd + normalization.
    constexpr std::size_t ch = 2;
    constexpr std::size_t h = 5;
    constexpr std::size_t w = 7;
    constexpr int N = 2;
    constexpr int axis = 2 * N + 1;

    auto img = make_ramp(ch, h, w);

    // Fused path
    std::size_t fused_depth = 0;
    auto fused = stack_and_normalize_template_hwd(img, ch, h, w, N, fused_depth);
    REQUIRE(fused_depth == ch * axis * axis);

    // Separate path: stack first
    std::size_t sep_depth = 0;
    auto sep = stack_template_window_hwd(img, ch, h, w, N, sep_depth);
    REQUIRE(sep_depth == fused_depth);

    // Then normalize manually (same formula as normalize_feature_depth_hwd)
    const std::size_t plane = h * w;
    const float inv_depth = 1.0f / static_cast<float>(sep_depth);
    constexpr float kEps = 1e-6f;
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        float* row = sep.data() + pixel * sep_depth;
        float sum = 0.0f, sum_sq = 0.0f;
        for (std::size_t d = 0; d < sep_depth; ++d) {
            sum += row[d];
            sum_sq += row[d] * row[d];
        }
        const float mean = sum * inv_depth;
        const float var = sum_sq * inv_depth - mean * mean;
        const float inv_std = 1.0f / (std::sqrt(std::max(var, 0.0f)) + kEps);
        for (std::size_t d = 0; d < sep_depth; ++d) {
            row[d] = (row[d] - mean) * inv_std;
        }
    }

    // Compare element-by-element
    REQUIRE(fused.size() == sep.size());
    for (std::size_t i = 0; i < fused.size(); ++i) {
        REQUIRE_THAT(static_cast<double>(fused[i]),
                     Catch::Matchers::WithinAbs(static_cast<double>(sep[i]), 1e-6));
    }
}

TEST_CASE("stack_and_normalize_template_hwd n_template=0 fast path", "[pyramid][template]") {
    constexpr std::size_t ch = 3;
    constexpr std::size_t h = 4;
    constexpr std::size_t w = 5;

    auto img = make_ramp(ch, h, w);

    std::size_t fused_depth = 0;
    auto fused = stack_and_normalize_template_hwd(img, ch, h, w, 0, fused_depth);
    REQUIRE(fused_depth == ch);

    // Separate path
    std::size_t sep_depth = 0;
    auto sep = stack_template_window_hwd(img, ch, h, w, 0, sep_depth);
    REQUIRE(sep_depth == ch);

    // Normalize manually
    const std::size_t plane = h * w;
    const float inv_ch = 1.0f / static_cast<float>(ch);
    constexpr float kEps = 1e-6f;
    for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        float* row = sep.data() + pixel * ch;
        float sum = 0.0f, sum_sq = 0.0f;
        for (std::size_t d = 0; d < ch; ++d) {
            sum += row[d];
            sum_sq += row[d] * row[d];
        }
        const float mean = sum * inv_ch;
        const float var = sum_sq * inv_ch - mean * mean;
        const float inv_std = 1.0f / (std::sqrt(std::max(var, 0.0f)) + kEps);
        for (std::size_t d = 0; d < ch; ++d) {
            row[d] = (row[d] - mean) * inv_std;
        }
    }

    REQUIRE(fused.size() == sep.size());
    for (std::size_t i = 0; i < fused.size(); ++i) {
        REQUIRE_THAT(static_cast<double>(fused[i]),
                     Catch::Matchers::WithinAbs(static_cast<double>(sep[i]), 1e-6));
    }
}

TEST_CASE("stack_and_normalize_template_hwd edge cases", "[pyramid][template]") {
    // Test with odd/even sizes, ch=1, N_s=1/2
    constexpr int Ns[] = {1, 2};
    constexpr std::size_t h_vals[] = {3, 4, 7, 8};
    constexpr std::size_t w_vals[] = {3, 5, 6, 8};

    for (int n : Ns) {
        for (int si = 0; si < 4; ++si) {
            const std::size_t h = h_vals[si];
            const std::size_t w = w_vals[si];
            auto img = make_ramp(1, h, w);

            std::size_t fd = 0;
            auto fused = stack_and_normalize_template_hwd(img, 1, h, w, n, fd);

            std::size_t sd = 0;
            auto sep = stack_template_window_hwd(img, 1, h, w, n, sd);
            REQUIRE(sd == fd);

            const std::size_t plane = h * w;
            const float inv_d = 1.0f / static_cast<float>(sd);
            constexpr float kEps = 1e-6f;
            for (std::size_t pixel = 0; pixel < plane; ++pixel) {
                float* row = sep.data() + pixel * sd;
                float sum = 0.0f, sum_sq = 0.0f;
                for (std::size_t d = 0; d < sd; ++d) {
                    sum += row[d];
                    sum_sq += row[d] * row[d];
                }
                const float mean = sum * inv_d;
                const float var = sum_sq * inv_d - mean * mean;
                const float inv_std = 1.0f / (std::sqrt(std::max(var, 0.0f)) + kEps);
                for (std::size_t d = 0; d < sd; ++d) {
                    row[d] = (row[d] - mean) * inv_std;
                }
            }

            REQUIRE(fused.size() == sep.size());
            for (std::size_t i = 0; i < fused.size(); ++i) {
                REQUIRE_THAT(static_cast<double>(fused[i]),
                             Catch::Matchers::WithinAbs(static_cast<double>(sep[i]), 2e-6));
            }
        }
    }
}

TEST_CASE("pyramid_data template stack uses Python row-major shift order", "[pyramid]") {
    constexpr std::size_t ch = 1;
    constexpr std::size_t h = 3;
    constexpr std::size_t w = 4;
    constexpr int n_template = 1;
    constexpr int axis = 2 * n_template + 1;
    constexpr std::size_t depth = ch * axis * axis;

    auto ref = make_ramp(ch, h, w);
    auto img = make_ramp(ch, h, w);

    const auto result = pyramid_data(
        ref, img, ch, h, w, 0, n_template,
        PyramidDownsampleMode::Mean2x2,
        PyramidNormalizationMode::PerLevelFeature);

    REQUIRE(result.ref_levels.size() == 1);
    const auto& level = result.ref_levels[0];
    REQUIRE(level.d0 == depth);
    REQUIRE(level.d1 == h);
    REQUIRE(level.d2 == w);

    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            std::vector<double> expected_raw(depth, 0.0);
            for (int dy = -n_template; dy <= n_template; ++dy) {
                for (int dx = -n_template; dx <= n_template; ++dx) {
                    const std::size_t d = static_cast<std::size_t>(
                        (dy + n_template) * axis + (dx + n_template));
                    const auto src_y = wrap_index(static_cast<long long>(y) - dy, h);
                    const auto src_x = wrap_index(static_cast<long long>(x) - dx, w);
                    expected_raw[d] = static_cast<double>(ref[src_y * w + src_x]);
                }
            }

            double mean = 0.0;
            for (const double v : expected_raw) {
                mean += v;
            }
            mean /= static_cast<double>(depth);

            double var = 0.0;
            for (const double v : expected_raw) {
                const double diff = v - mean;
                var += diff * diff;
            }
            const double inv_std = 1.0 / (std::sqrt(var / static_cast<double>(depth)) + 1.0e-6);

            for (std::size_t d = 0; d < depth; ++d) {
                const double expected = (expected_raw[d] - mean) * inv_std;
                const double actual = level.data[(y * w + x) * depth + d];
                REQUIRE_THAT(actual, Catch::Matchers::WithinAbs(expected, 1.0e-5));
            }
        }
    }
}
