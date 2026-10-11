#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "wsvt/template_window.hpp"

#include <cstddef>
#include <numeric>
#include <vector>

using namespace wsvt;

namespace {

std::size_t wrap_index(long long v, std::size_t n) {
    const long long m = static_cast<long long>(n);
    long long r = v % m;
    if (r < 0) {
        r += m;
    }
    return static_cast<std::size_t>(r);
}

}  // namespace

TEST_CASE("stack_template_window uses Python row-major shift ordering", "[template_window]") {
    constexpr std::size_t ch = 2;
    constexpr std::size_t h = 3;
    constexpr std::size_t w = 4;
    constexpr int n_template = 1;
    constexpr int axis = 2 * n_template + 1;

    std::vector<float> img(ch * h * w);
    std::iota(img.begin(), img.end(), 1.0f);

    std::size_t out_h = 0;
    std::size_t out_w = 0;
    std::size_t out_depth = 0;
    const auto stacked = stack_template_window(img, ch, h, w, n_template, out_h, out_w, out_depth);

    REQUIRE(out_h == h);
    REQUIRE(out_w == w);
    REQUIRE(out_depth == ch * axis * axis);
    REQUIRE(stacked.size() == h * w * out_depth);

    for (int dy = -n_template; dy <= n_template; ++dy) {
        for (int dx = -n_template; dx <= n_template; ++dx) {
            for (std::size_t c = 0; c < ch; ++c) {
                const std::size_t d = static_cast<std::size_t>(
                    (dy + n_template) * axis + (dx + n_template)) * ch + c;
                for (std::size_t y = 0; y < h; ++y) {
                    for (std::size_t x = 0; x < w; ++x) {
                        const auto src_y = wrap_index(static_cast<long long>(y) - dy, h);
                        const auto src_x = wrap_index(static_cast<long long>(x) - dx, w);
                        const float expected = img[(c * h + src_y) * w + src_x];
                        const float actual = stacked[(y * w + x) * out_depth + d];
                        REQUIRE_THAT(static_cast<double>(actual),
                                     Catch::Matchers::WithinAbs(static_cast<double>(expected), 0.0));
                    }
                }
            }
        }
    }
}
