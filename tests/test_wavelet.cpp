#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "wsvt/wavelet_ops.hpp"

#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

using namespace wsvt;

static std::vector<float> make_signal(std::size_t ch, std::size_t h, std::size_t w) {
    std::vector<float> data(ch * h * w);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.1f);
    }
    return data;
}

TEST_CASE("wavelet_transform output shape", "[wavelet]") {
    constexpr std::size_t ch = 1, h = 64, w = 64;
    auto signal = make_signal(ch, h, w);

    SECTION("db6, level=1, return_level=1") {
        auto result = wavelet_transform(signal, ch, h, w, "db6", 1, 1);
        REQUIRE(result.out_h == h);
        REQUIRE(result.out_w == w);
        REQUIRE(result.out_depth > 0);
        REQUIRE(result.coeffs_filter.size() == result.out_h * result.out_w * result.out_depth);
        REQUIRE(result.level_name.size() == 1);
    }

    SECTION("db6, level=2, return_level=2") {
        auto result = wavelet_transform(signal, ch, h, w, "db6", 2, 2);
        REQUIRE(result.out_h == h);
        REQUIRE(result.out_w == w);
        REQUIRE(result.level_name.size() == 2);
        REQUIRE(result.coeffs_filter.size() == result.out_h * result.out_w * result.out_depth);
    }
}

TEST_CASE("wavelet_transform supports db2/db3/db6", "[wavelet]") {
    constexpr std::size_t ch = 1, h = 64, w = 64;
    auto signal = make_signal(ch, h, w);

    for (const auto& method : {"db2", "db3", "db6"}) {
        DYNAMIC_SECTION("method=" << method) {
            auto result = wavelet_transform(signal, ch, h, w, method, 1, 1);
            REQUIRE(result.out_depth > 0);
            REQUIRE(result.coeffs_filter.size() == h * w * result.out_depth);
        }
    }
}

TEST_CASE("wavelet_transform rejects unknown method", "[wavelet]") {
    std::vector<float> signal(64 * 64, 1.0f);
    REQUIRE_THROWS_AS(
        wavelet_transform(signal, 1, 64, 64, "unknown", 1, 1),
        std::invalid_argument);
}

TEST_CASE("wavelet_transform_multiprocess matches single-thread", "[wavelet]") {
    constexpr std::size_t ch = 1, h = 64, w = 64;
    auto signal = make_signal(ch, h, w);

    auto single = wavelet_transform(signal, ch, h, w, "db6", 1, 1);
    auto multi = wavelet_transform_multiprocess(signal, ch, h, w, 2, "db6", 1, 1);

    REQUIRE(single.out_depth == multi.out_depth);
    REQUIRE(single.coeffs_filter.size() == multi.coeffs_filter.size());
    for (std::size_t i = 0; i < single.coeffs_filter.size(); ++i) {
        REQUIRE_THAT(static_cast<double>(multi.coeffs_filter[i]),
                     Catch::Matchers::WithinAbs(static_cast<double>(single.coeffs_filter[i]), 1e-5));
    }
}
