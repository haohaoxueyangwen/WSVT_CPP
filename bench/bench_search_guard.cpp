#include "wsvt/search_pruning.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Result {
    double median_s = 0.0;
    double min_s = 0.0;
    std::uint64_t checksum = 0;
};

template <typename Function>
Result measure(Function&& function, int repeats) {
    std::vector<double> seconds;
    seconds.reserve(static_cast<std::size_t>(repeats));
    std::uint64_t checksum = 0;
    for (int repeat = 0; repeat < repeats + 2; ++repeat) {
        const auto start = Clock::now();
        const std::uint64_t value = function();
        const auto end = Clock::now();
        if (repeat >= 2) {
            seconds.push_back(std::chrono::duration<double>(end - start).count());
            checksum ^= value;
        }
    }
    std::sort(seconds.begin(), seconds.end());
    return Result{
        seconds[seconds.size() / 2],
        seconds.front(),
        checksum};
}

} // namespace

int main() {
    constexpr std::size_t prefix_terms = 24;
    constexpr std::size_t full_depth = 63;
    constexpr std::size_t groups = 100'000;
    constexpr std::size_t candidates_per_group = 81;
    constexpr std::size_t total = groups * candidates_per_group;
    constexpr int repeats = 7;

    std::mt19937 rng(20260827U);
    std::uniform_real_distribution<float> prefix_distribution(0.0f, 8.0f);
    std::uniform_real_distribution<float> threshold_distribution(0.25f, 2.0f);
    std::vector<float> prefixes(total);
    std::vector<float> thresholds(groups);
    for (float& value : prefixes) {
        value = prefix_distribution(rng);
    }
    for (float& value : thresholds) {
        value = threshold_distribution(rng);
    }

    const auto legacy = measure([&] {
        std::uint64_t count = 0;
        for (std::size_t group = 0; group < groups; ++group) {
            const float threshold = thresholds[group];
            const std::size_t begin = group * candidates_per_group;
            const std::size_t end = begin + candidates_per_group;
            for (std::size_t i = begin; i < end; ++i) {
                count += static_cast<std::uint64_t>(
                    wsvt::prefix_proves_full_ssd_above(
                        prefixes[i], threshold, prefix_terms, full_depth));
            }
        }
        return count;
    }, repeats);

    const wsvt::ConservativePrefixSsdGuard cached_guard(
        prefix_terms, full_depth);
    const auto cached = measure([&] {
        std::uint64_t count = 0;
        for (std::size_t group = 0; group < groups; ++group) {
            const double guarded_threshold =
                cached_guard.guarded_threshold(thresholds[group]);
            const std::size_t begin = group * candidates_per_group;
            const std::size_t end = begin + candidates_per_group;
            for (std::size_t i = begin; i < end; ++i) {
                count += static_cast<std::uint64_t>(
                    cached_guard.proves_above(prefixes[i], guarded_threshold));
            }
        }
        return count;
    }, repeats);

    if (legacy.checksum != cached.checksum) {
        throw std::runtime_error("cached guard changed the pruning decision");
    }

    std::cout << std::setprecision(9)
              << "checks=" << total << '\n'
              << "legacy_median_s=" << legacy.median_s << '\n'
              << "legacy_min_s=" << legacy.min_s << '\n'
              << "cached_median_s=" << cached.median_s << '\n'
              << "cached_min_s=" << cached.min_s << '\n'
              << "median_speedup=" << legacy.median_s / cached.median_s << '\n'
              << "checksum=" << legacy.checksum << '\n';
}
