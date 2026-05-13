#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include "wsvt/image.hpp"
#include "wsvt/phase_recovery.hpp"
#include "wsvt/pyramid.hpp"
#include "wsvt/wavelet_ops.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

static std::vector<float> make_signal(std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = std::sin(static_cast<float>(i) * 0.01f);
    }
    return v;
}

int main() {
    using namespace wsvt;
    namespace nb = ankerl::nanobench;

    nb::Bench bench;
    bench.title("WSVT core benchmarks").warmup(3).minEpochIterations(3);

    constexpr std::size_t h1 = 512, w1 = 512;
    constexpr std::size_t h2 = 1024, w2 = 1024;

    auto sig_512 = make_signal(h1 * w1);
    auto sig_1024 = make_signal(h2 * w2);

    bench.run("pyramid_data 512x512 lv=2 Mean2x2", [&] {
        auto r = pyramid_data(sig_512, sig_512, 1, h1, w1, 2, 0, PyramidDownsampleMode::Mean2x2);
        nb::doNotOptimizeAway(r);
    });

    bench.run("pyramid_data 1024x1024 lv=2 Mean2x2", [&] {
        auto r = pyramid_data(sig_1024, sig_1024, 1, h2, w2, 2, 0, PyramidDownsampleMode::Mean2x2);
        nb::doNotOptimizeAway(r);
    });

    bench.run("wavelet_transform 512x512 db6 lv=1", [&] {
        auto r = wavelet_transform(sig_512, 1, h1, w1, "db6", 1, 1);
        nb::doNotOptimizeAway(r);
    });

    bench.run("wavelet_transform 1024x1024 db6 lv=1", [&] {
        auto r = wavelet_transform(sig_1024, 1, h2, w2, "db6", 1, 1);
        nb::doNotOptimizeAway(r);
    });

    {
        Image2D<float> dpc_x({h1, w1}, 1.0f);
        Image2D<float> dpc_y({h1, w1}, 1.0f);
        bench.run("frankot_chellappa 512x512", [&] {
            auto r = frankot_chellappa(dpc_x, dpc_y);
            nb::doNotOptimizeAway(r);
        });
    }

    {
        Image2D<float> dpc_x({h2, w2}, 1.0f);
        Image2D<float> dpc_y({h2, w2}, 1.0f);
        bench.run("frankot_chellappa 1024x1024", [&] {
            auto r = frankot_chellappa(dpc_x, dpc_y);
            nb::doNotOptimizeAway(r);
        });
    }
}
