#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "wsvt/common.hpp"
#include "wsvt/image_ops.hpp"
#include "wsvt/io_json.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

using namespace wsvt;

TEST_CASE("JSON serialization preserves exact large integer-valued counters",
          "[common][json]") {
    REQUIRE(json_dumps(JsonValue(1327104.0)) == "1327104");
    REQUIRE(json_dumps(JsonValue(7962624.0)) == "7962624");
}

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

TEST_CASE("pad_hwd_zero_aligned preserves data and SIMD alignment", "[common][alignment]") {
    constexpr Shape3D shape{3, 5, 7};
    std::vector<float> input(shape.size(), 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1);
    }

    const auto padded = pad_hwd_zero_aligned(
        TensorView3D<const float, Layout::HWD>{input.data(), shape}, 2, 3);

    REQUIRE(padded.shape().d0 == 7);
    REQUIRE(padded.shape().d1 == 11);
    REQUIRE(padded.shape().d2 == 7);
    REQUIRE(is_pointer_aligned<64>(padded.data()));

    const auto out_shape = padded.shape();
    for (std::size_t y = 0; y < out_shape.d0; ++y) {
        for (std::size_t x = 0; x < out_shape.d1; ++x) {
            for (std::size_t d = 0; d < out_shape.d2; ++d) {
                const float actual = padded.values[idx_hwd(y, x, d, out_shape.d1, out_shape.d2)];
                if (y >= 2 && y < 5 && x >= 3 && x < 8) {
                    const float expected = input[idx_hwd(y - 2, x - 3, d, shape.d1, shape.d2)];
                    REQUIRE(actual == expected);
                } else {
                    REQUIRE(actual == 0.0f);
                }
            }
        }
    }
}

TEST_CASE("not-a-knot tensor spline reproduces bicubic fields", "[image][spline]") {
    constexpr Shape2D input_shape{5, 6};
    constexpr Shape2D output_shape{9, 11};
    Image2D<float> input(input_shape, 0.0f);
    const auto polynomial = [](double y, double x) {
        return 0.25 * x * x * x - 0.5 * y * y * y +
               0.75 * x * y + 2.0 * x - 3.0 * y + 4.0;
    };
    for (std::size_t y = 0; y < input_shape.h; ++y) {
        for (std::size_t x = 0; x < input_shape.w; ++x) {
            input(y, x) = static_cast<float>(polynomial(y, x));
        }
    }

    const auto output = resample_rect_bivariate_spline(input, output_shape);
    const double y_scale = static_cast<double>(input_shape.h - 1) /
                           static_cast<double>(output_shape.h - 1);
    const double x_scale = static_cast<double>(input_shape.w - 1) /
                           static_cast<double>(output_shape.w - 1);
    for (std::size_t y = 0; y < output_shape.h; ++y) {
        for (std::size_t x = 0; x < output_shape.w; ++x) {
            const double expected = polynomial(y_scale * y, x_scale * x);
            REQUIRE_THAT(static_cast<double>(output(y, x)),
                         Catch::Matchers::WithinAbs(expected, 2.0e-5));
        }
    }
}
