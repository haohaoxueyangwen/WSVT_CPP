#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "wsvt/search_pruning.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using namespace wsvt;

namespace {

std::vector<RankedSsdCandidate> rank_exhaustive(
    const std::vector<float>& query,
    const std::vector<float>& candidates,
    std::size_t candidate_count,
    std::size_t top_k,
    std::size_t block_size) {
    ExactTopK ranked(top_k);
    for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
        const auto result = squared_distance_blockwise(
            query.data(),
            candidates.data() + candidate * query.size(),
            query.size(),
            std::numeric_limits<float>::infinity(),
            block_size);
        REQUIRE_FALSE(result.abandoned);
        ranked.consider(result.distance, candidate);
    }

    std::vector<RankedSsdCandidate> out;
    for (std::size_t i = 0; i < ranked.size(); ++i) {
        out.push_back(ranked[i]);
    }
    return out;
}

std::vector<RankedSsdCandidate> rank_pruned(
    const std::vector<float>& query,
    const std::vector<float>& candidates,
    std::size_t candidate_count,
    std::size_t top_k,
    std::size_t block_size,
    std::size_t& terms_evaluated,
    std::size_t& abandoned_count) {
    ExactTopK ranked(top_k);
    terms_evaluated = 0;
    abandoned_count = 0;
    for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
        const auto result = squared_distance_blockwise(
            query.data(),
            candidates.data() + candidate * query.size(),
            query.size(),
            ranked.abandon_threshold(),
            block_size);
        terms_evaluated += result.terms_evaluated;
        if (result.abandoned) {
            ++abandoned_count;
        } else {
            ranked.consider(result.distance, candidate);
        }
    }

    std::vector<RankedSsdCandidate> out;
    for (std::size_t i = 0; i < ranked.size(); ++i) {
        out.push_back(ranked[i]);
    }
    return out;
}

struct TwoPassFixtureStats {
    std::size_t terms_evaluated = 0;
    std::size_t prefix_terms_evaluated = 0;
    std::size_t full_candidate_count = 0;
    std::size_t abandoned_count = 0;
};

std::vector<RankedSsdCandidate> rank_two_pass(
    const std::vector<float>& query,
    const std::vector<float>& candidates,
    std::size_t candidate_count,
    std::size_t top_k,
    std::size_t prefix_size,
    TwoPassFixtureStats& stats) {
    const std::size_t depth = query.size();
    const std::size_t prefix_terms = std::min(depth, prefix_size);
    std::vector<float> prefix(candidate_count, 0.0f);
    std::vector<bool> exact(candidate_count, false);
    ExactTopK prefix_top_k(top_k);
    ExactTopK exact_top_k(top_k);
    stats = {};

    for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
        prefix[candidate] = squared_distance_full_simd(
            query.data(), candidates.data() + candidate * depth, prefix_terms);
        prefix_top_k.consider(prefix[candidate], candidate);
        stats.terms_evaluated += prefix_terms;
        stats.prefix_terms_evaluated += prefix_terms;
    }
    if (prefix_terms == depth) {
        for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
            exact_top_k.consider(prefix[candidate], candidate);
            exact[candidate] = true;
            ++stats.full_candidate_count;
        }
    } else {
        for (std::size_t rank = 0; rank < prefix_top_k.size(); ++rank) {
            const std::size_t candidate = prefix_top_k[rank].candidate_index;
            const float distance = squared_distance_full_simd(
                query.data(), candidates.data() + candidate * depth, depth);
            exact_top_k.consider(distance, candidate);
            exact[candidate] = true;
            stats.terms_evaluated += depth;
            ++stats.full_candidate_count;
        }
        for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
            if (exact[candidate]) {
                continue;
            }
            if (prefix_proves_full_ssd_above(
                    prefix[candidate], exact_top_k.abandon_threshold(),
                    prefix_terms, depth)) {
                ++stats.abandoned_count;
                continue;
            }
            const float distance = squared_distance_full_simd(
                query.data(), candidates.data() + candidate * depth, depth);
            exact_top_k.consider(distance, candidate);
            exact[candidate] = true;
            stats.terms_evaluated += depth;
            ++stats.full_candidate_count;
        }
    }

    std::vector<RankedSsdCandidate> out;
    for (std::size_t rank = 0; rank < exact_top_k.size(); ++rank) {
        out.push_back(exact_top_k[rank]);
    }
    return out;
}

} // namespace

TEST_CASE("ExactTopK uses deterministic row-major tie ordering", "[search][topk]") {
    ExactTopK top_k(4);
    top_k.consider(3.0f, 7);
    top_k.consider(1.0f, 9);
    top_k.consider(1.0f, 2);
    top_k.consider(2.0f, 5);
    top_k.consider(1.0f, 4);

    REQUIRE(top_k.full());
    REQUIRE(top_k[0].candidate_index == 2);
    REQUIRE(top_k[1].candidate_index == 4);
    REQUIRE(top_k[2].candidate_index == 9);
    REQUIRE(top_k[3].candidate_index == 5);
    REQUIRE(top_k.abandon_threshold() == 2.0f);
}

TEST_CASE("ExactTopK reports only abandon-threshold changes", "[search][topk][guard-cache]") {
    ExactTopK top_k(2);
    REQUIRE_FALSE(top_k.consider(1.0f, 4));
    REQUIRE(top_k.consider(2.0f, 5));
    REQUIRE_FALSE(top_k.consider(2.0f, 3));
    REQUIRE(top_k[1].candidate_index == 3);
    REQUIRE(top_k.consider(1.5f, 7));
    REQUIRE(top_k.abandon_threshold() == 1.5f);
    REQUIRE_FALSE(top_k.consider(3.0f, 0));
}

TEST_CASE("Blockwise pruning preserves exhaustive Top-K on random fixtures", "[search][topk][pruning]") {
    constexpr std::size_t depth = 73;
    constexpr std::size_t candidate_count = 97;

    std::mt19937 rng(20260824U);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    std::vector<float> query(depth, 0.0f);
    std::vector<float> candidates(candidate_count * depth, 0.0f);
    for (float& value : query) {
        value = distribution(rng);
    }
    for (float& value : candidates) {
        value = distribution(rng);
    }
    // Put the strongest candidates first so later poor candidates exercise the
    // early-abandon path deterministically.
    for (std::size_t candidate = 0; candidate < kMaxExactTopK; ++candidate) {
        for (std::size_t k = 0; k < depth; ++k) {
            candidates[candidate * depth + k] =
                query[k] + static_cast<float>(candidate + 1) * 1.0e-3f;
        }
    }

    for (const std::size_t top_k : std::array<std::size_t, 2>{2, 4}) {
        for (const std::size_t block_size : std::array<std::size_t, 4>{1, 8, 16, 32}) {
            const auto exhaustive = rank_exhaustive(
                query, candidates, candidate_count, top_k, block_size);
            std::size_t terms_evaluated = 0;
            std::size_t abandoned_count = 0;
            const auto pruned = rank_pruned(
                query, candidates, candidate_count, top_k, block_size,
                terms_evaluated, abandoned_count);

            REQUIRE(pruned.size() == exhaustive.size());
            for (std::size_t i = 0; i < top_k; ++i) {
                REQUIRE(pruned[i].candidate_index == exhaustive[i].candidate_index);
                REQUIRE(pruned[i].distance ==
                        Catch::Approx(exhaustive[i].distance).epsilon(1.0e-7));
            }
            REQUIRE(abandoned_count > 0);
            REQUIRE(terms_evaluated < candidate_count * depth);
        }
    }
}

TEST_CASE("A candidate tied at the threshold is fully evaluated", "[search][topk][ties]") {
    const std::array<float, 5> lhs{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    const std::array<float, 5> rhs{0.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    const auto result = squared_distance_blockwise(
        lhs.data(), rhs.data(), lhs.size(), /*abandon_above=*/1.0f,
        /*block_size=*/2);

    REQUIRE_FALSE(result.abandoned);
    REQUIRE(result.terms_evaluated == lhs.size());
    REQUIRE(result.distance == Catch::Approx(1.0f));
}

TEST_CASE("A cutoff crossed only after the final block is not early-abandoned", "[search][pruning][counter]") {
    const std::array<float, 4> lhs{0.0f, 0.0f, 0.0f, 0.0f};
    const std::array<float, 4> rhs{2.0f, 2.0f, 2.0f, 2.0f};
    const auto result = squared_distance_blockwise(
        lhs.data(), rhs.data(), lhs.size(), /*abandon_above=*/1.0f,
        /*block_size=*/8);

    REQUIRE_FALSE(result.abandoned);
    REQUIRE(result.terms_evaluated == lhs.size());
    REQUIRE(result.distance ==
            squared_distance_full_simd(lhs.data(), rhs.data(), lhs.size()));
}

TEST_CASE("Two-pass prefix screening preserves exhaustive Top-K", "[search][topk][two-pass]") {
    constexpr std::size_t depth = 73;
    constexpr std::size_t candidate_count = 97;
    std::mt19937 rng(20260827U);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    std::vector<float> query(depth, 0.0f);
    std::vector<float> candidates(candidate_count * depth, 0.0f);
    for (float& value : query) {
        value = distribution(rng);
    }
    for (float& value : candidates) {
        value = distribution(rng);
    }
    for (std::size_t candidate = 0; candidate < kMaxExactTopK; ++candidate) {
        for (std::size_t k = 0; k < depth; ++k) {
            candidates[candidate * depth + k] =
                query[k] + static_cast<float>(candidate + 1) * 1.0e-3f;
        }
    }

    for (const std::size_t top_k : std::array<std::size_t, 2>{2, 4}) {
        const auto exhaustive = rank_exhaustive(
            query, candidates, candidate_count, top_k, depth);
        for (const std::size_t prefix_size :
             std::array<std::size_t, 4>{1, 8, 16, 32}) {
            TwoPassFixtureStats stats;
            const auto screened = rank_two_pass(
                query, candidates, candidate_count, top_k, prefix_size, stats);
            REQUIRE(screened.size() == exhaustive.size());
            for (std::size_t rank = 0; rank < top_k; ++rank) {
                REQUIRE(screened[rank].candidate_index ==
                        exhaustive[rank].candidate_index);
                REQUIRE(screened[rank].distance == exhaustive[rank].distance);
            }
            REQUIRE(stats.abandoned_count > 0);
            REQUIRE(stats.full_candidate_count < candidate_count);
            REQUIRE(stats.prefix_terms_evaluated == candidate_count * prefix_size);
            REQUIRE(stats.terms_evaluated < candidate_count * depth);
        }
    }
}

TEST_CASE("Two-pass bound is conservative and preserves threshold ties", "[search][two-pass][ties]") {
    REQUIRE(prefix_proves_full_ssd_above(
        /*prefix_distance=*/2.0f, /*threshold=*/1.0f,
        /*prefix_terms=*/1, /*full_depth=*/8));
    REQUIRE_FALSE(prefix_proves_full_ssd_above(
        /*prefix_distance=*/1.0f, /*threshold=*/1.0f,
        /*prefix_terms=*/1, /*full_depth=*/8));
    REQUIRE_FALSE(prefix_proves_full_ssd_above(
        std::numeric_limits<float>::quiet_NaN(), 1.0f, 1, 8));
    REQUIRE_FALSE(prefix_proves_full_ssd_above(
        2.0f, std::numeric_limits<float>::infinity(), 1, 8));
}

TEST_CASE("Cached two-pass guard exactly matches the legacy decision", "[search][two-pass][guard-cache]") {
    constexpr std::size_t prefix_terms = 24;
    constexpr std::size_t full_depth = 100;
    const ConservativePrefixSsdGuard guard(prefix_terms, full_depth);
    REQUIRE(guard.valid());

    std::mt19937 rng(20260827U);
    std::uniform_real_distribution<float> value_distribution(0.0f, 1.0e5f);
    for (std::size_t i = 0; i < 100000; ++i) {
        const float prefix_distance = value_distribution(rng);
        const float threshold = value_distribution(rng);
        const bool legacy = prefix_proves_full_ssd_above(
            prefix_distance, threshold, prefix_terms, full_depth);
        const bool cached = guard.proves_above(
            prefix_distance, guard.guarded_threshold(threshold));
        REQUIRE(cached == legacy);
    }

    for (const float threshold : std::array<float, 4>{
             0.0f, 1.0f, std::numeric_limits<float>::max(),
             std::numeric_limits<float>::infinity()}) {
        for (const float prefix_distance : std::array<float, 5>{
                 0.0f, threshold, std::nextafter(threshold, 0.0f),
                 std::nextafter(threshold, std::numeric_limits<float>::infinity()),
                 std::numeric_limits<float>::quiet_NaN()}) {
            REQUIRE(guard.proves_above(
                        prefix_distance, guard.guarded_threshold(threshold)) ==
                    prefix_proves_full_ssd_above(
                        prefix_distance, threshold, prefix_terms, full_depth));
        }
    }

    const ConservativePrefixSsdGuard invalid_zero(0, full_depth);
    const ConservativePrefixSsdGuard invalid_order(full_depth + 1, full_depth);
    REQUIRE_FALSE(invalid_zero.valid());
    REQUIRE_FALSE(invalid_order.valid());
    REQUIRE_FALSE(invalid_zero.proves_above(2.0f, 1.0));
    REQUIRE_FALSE(invalid_order.proves_above(2.0f, 1.0));
}

TEST_CASE("Two-pass full-depth prefix falls back to one exhaustive reduction", "[search][two-pass][fallback]") {
    constexpr std::size_t depth = 5;
    constexpr std::size_t candidate_count = 6;
    const std::vector<float> query(depth, 0.0f);
    std::vector<float> candidates(candidate_count * depth, 0.0f);
    for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
        for (std::size_t k = 0; k < depth; ++k) {
            candidates[candidate * depth + k] = static_cast<float>(candidate);
        }
    }
    TwoPassFixtureStats stats;
    const auto screened = rank_two_pass(
        query, candidates, candidate_count, 2, /*prefix_size=*/16, stats);
    const auto exhaustive = rank_exhaustive(
        query, candidates, candidate_count, 2, depth);

    REQUIRE(screened[0].candidate_index == exhaustive[0].candidate_index);
    REQUIRE(screened[1].candidate_index == exhaustive[1].candidate_index);
    REQUIRE(stats.abandoned_count == 0);
    REQUIRE(stats.full_candidate_count == candidate_count);
    REQUIRE(stats.prefix_terms_evaluated == candidate_count * depth);
    REQUIRE(stats.terms_evaluated == candidate_count * depth);
}
