#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "wsvt/umpa_physical_fit.hpp"
#include "wsvt/wsvt_umpa_refine.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

struct Fixture {
    std::vector<float> sample;
    std::vector<float> reference;
    std::size_t frames = 8U;
    std::size_t height = 24U;
    std::size_t width = 24U;
    std::size_t sample_y = 12U;
    std::size_t sample_x = 12U;
    int truth_y = -2;
    int truth_x = 3;
    double transmission = 0.82;
    double visibility = 0.63;
};

Fixture make_fixture(double visibility = 0.63) {
    Fixture fixture;
    fixture.visibility = visibility;
    const std::size_t plane = fixture.height * fixture.width;
    fixture.reference.resize(fixture.frames * plane, 0.0f);
    fixture.sample.resize(fixture.frames * plane, 0.0f);
    for (std::size_t frame = 0; frame < fixture.frames; ++frame) {
        for (std::size_t y = 0; y < fixture.height; ++y) {
            for (std::size_t x = 0; x < fixture.width; ++x) {
                const std::size_t pattern =
                    (11U * frame + 7U * y + 13U * x + 3U * x * y) % 37U;
                fixture.reference[frame * plane + y * fixture.width + x] =
                    static_cast<float>(
                        100.0 + 4.0 * static_cast<double>(frame) +
                        static_cast<double>(pattern));
            }
        }
    }
    const auto window = wsvt::normalized_hamming_window_2d(1U);
    const std::size_t reference_y = static_cast<std::size_t>(
        static_cast<int>(fixture.sample_y) - fixture.truth_y);
    const std::size_t reference_x = static_cast<std::size_t>(
        static_cast<int>(fixture.sample_x) - fixture.truth_x);
    for (std::size_t frame = 0; frame < fixture.frames; ++frame) {
        double mean = 0.0;
        for (std::size_t wy = 0; wy < 3U; ++wy) {
            for (std::size_t wx = 0; wx < 3U; ++wx) {
                const std::size_t ry = reference_y + wy - 1U;
                const std::size_t rx = reference_x + wx - 1U;
                mean += window[wy * 3U + wx] * fixture.reference[
                    frame * plane + ry * fixture.width + rx];
            }
        }
        for (std::size_t wy = 0; wy < 3U; ++wy) {
            for (std::size_t wx = 0; wx < 3U; ++wx) {
                const std::size_t sy = fixture.sample_y + wy - 1U;
                const std::size_t sx = fixture.sample_x + wx - 1U;
                const std::size_t ry = reference_y + wy - 1U;
                const std::size_t rx = reference_x + wx - 1U;
                const double reference = fixture.reference[
                    frame * plane + ry * fixture.width + rx];
                fixture.sample[frame * plane + sy * fixture.width + sx] =
                    static_cast<float>(fixture.transmission * (
                        fixture.visibility * (reference - mean) + mean));
            }
        }
    }
    return fixture;
}

std::vector<int> repeated_centres(int value) {
    return std::vector<int>(4U, value);
}

}  // namespace

TEST_CASE("fixed WG-UMPA H1 recovers an integer neighbour and T D", "[wsvt][umpa][H1]") {
    const Fixture fixture = make_fixture();
    const std::vector<int> proposal_y{fixture.truth_y};
    const std::vector<int> proposal_x{fixture.truth_x - 1};
    wsvt::WaveletGuidedUmpaConfig config;
    config.enabled = true;
    const auto output = wsvt::refine_wavelet_guided_umpa(
        fixture.sample, fixture.reference,
        fixture.frames, fixture.height, fixture.width,
        proposal_y, proposal_x, 1U, 1U,
        fixture.sample_y, fixture.sample_x, config);

    REQUIRE(output.raw_candidate_count == 9U);
    REQUIRE(output.raw_observation_count == 9U * fixture.frames * 9U);
    REQUIRE(output.numerical_valid_pixel_count == 1U);
    REQUIRE(output.physical_valid_pixel_count == 1U);
    REQUIRE(output.displace_y[0] == Catch::Approx(static_cast<float>(fixture.truth_y)));
    REQUIRE(output.displace_x[0] == Catch::Approx(static_cast<float>(fixture.truth_x)));
    REQUIRE(output.relative_offset_y[0] == Catch::Approx(0.0f));
    REQUIRE(output.relative_offset_x[0] == Catch::Approx(1.0f));
    REQUIRE(output.transmission[0] == Catch::Approx(fixture.transmission).margin(2.0e-5));
    REQUIRE(output.visibility[0] == Catch::Approx(fixture.visibility).margin(2.0e-5));
    REQUIRE(output.local_boundary_hit[0] == Catch::Approx(1.0f));
    REQUIRE(output.best_cost[0] < output.second_best_cost[0]);
}

TEST_CASE("fixed WG-UMPA H1 rejects the structurally singular N zero profile", "[wsvt][umpa][H1][N0]") {
    const Fixture fixture = make_fixture();
    const std::vector<int> proposal_y{fixture.truth_y};
    const std::vector<int> proposal_x{fixture.truth_x};
    wsvt::WaveletGuidedUmpaConfig config;
    config.enabled = true;
    config.analysis_radius = 0U;
    REQUIRE_THROWS_AS(
        wsvt::refine_wavelet_guided_umpa(
            fixture.sample, fixture.reference,
            fixture.frames, fixture.height, fixture.width,
            proposal_y, proposal_x, 1U, 1U,
            fixture.sample_y, fixture.sample_x, config),
        std::invalid_argument);
}

TEST_CASE("fixed WG-UMPA H1 keeps the proposal when no complete patch exists", "[wsvt][umpa][H1][boundary]") {
    const Fixture fixture = make_fixture();
    const std::vector<int> proposal_y{0};
    const std::vector<int> proposal_x{0};
    wsvt::WaveletGuidedUmpaConfig config;
    config.enabled = true;
    const auto output = wsvt::refine_wavelet_guided_umpa(
        fixture.sample, fixture.reference,
        fixture.frames, fixture.height, fixture.width,
        proposal_y, proposal_x, 1U, 1U, 0U, 0U, config);
    REQUIRE(output.raw_candidate_count == 0U);
    REQUIRE(output.numerical_valid_pixel_count == 0U);
    REQUIRE(output.displace_y[0] == Catch::Approx(0.0f));
    REQUIRE(output.displace_x[0] == Catch::Approx(0.0f));
    REQUIRE(output.numerical_valid[0] == Catch::Approx(0.0f));
}

TEST_CASE("SET4 raw objectives share one deduplicated candidate union", "[wsvt][umpa][set4][ablation]") {
    const Fixture model_df_fixture = make_fixture();
    const Fixture model_t_fixture = make_fixture(1.0);
    const std::vector<int> centres_y = repeated_centres(model_df_fixture.truth_y);
    const std::vector<int> centres_x = repeated_centres(model_df_fixture.truth_x);
    const std::vector<float> fallback_y{0.25f};
    const std::vector<float> fallback_x{-0.5f};

    for (const auto objective : {
             wsvt::SetTransportRawObjective::WindowedZncc,
             wsvt::SetTransportRawObjective::ModelT,
             wsvt::SetTransportRawObjective::ModelDF}) {
        const Fixture& fixture = objective == wsvt::SetTransportRawObjective::ModelT
            ? model_t_fixture : model_df_fixture;
        wsvt::SetTransportRawRerankConfig config;
        config.enabled = true;
        config.objective = objective;
        const auto output = wsvt::rerank_set_transport_raw(
             fixture.sample, fixture.reference,
             fixture.frames, fixture.height, fixture.width,
             centres_y, centres_x,
             std::span<const int>{}, std::span<const int>{},
             fallback_y, fallback_x,
            1U, 1U, fixture.sample_y, fixture.sample_x, config);

        REQUIRE(output.raw_candidate_count == 81U);
        REQUIRE(output.raw_observation_count == 81U * fixture.frames * 9U);
        REQUIRE(output.numerical_valid_pixel_count == 1U);
        REQUIRE(output.displace_y[0] == Catch::Approx(
            static_cast<float>(fixture.truth_y)));
        REQUIRE(output.displace_x[0] == Catch::Approx(
            static_cast<float>(fixture.truth_x)));
        REQUIRE(output.best_cost[0] < output.second_best_cost[0]);
        REQUIRE(output.candidates_evaluated[0] == Catch::Approx(81.0f));
        if (objective == wsvt::SetTransportRawObjective::ModelT) {
            REQUIRE(output.transmission[0] == Catch::Approx(
                fixture.transmission).margin(2.0e-5));
            REQUIRE(output.visibility[0] == Catch::Approx(1.0f));
            REQUIRE(output.physical_valid[0] == Catch::Approx(1.0f));
        } else if (objective == wsvt::SetTransportRawObjective::ModelDF) {
            REQUIRE(output.transmission[0] == Catch::Approx(
                fixture.transmission).margin(2.0e-5));
            REQUIRE(output.visibility[0] == Catch::Approx(
                fixture.visibility).margin(2.0e-5));
            REQUIRE(output.physical_valid[0] == Catch::Approx(1.0f));
        } else {
            REQUIRE(output.physical_valid[0] == Catch::Approx(0.0f));
        }
    }
}

TEST_CASE("SET4 raw rerank validates its frozen support", "[wsvt][umpa][set4][validation]") {
    const Fixture fixture = make_fixture();
    wsvt::SetTransportRawRerankConfig config;
    config.enabled = true;
    config.analysis_radius = 2U;
    REQUIRE_THROWS_AS(
        wsvt::rerank_set_transport_raw(
            fixture.sample, fixture.reference,
             fixture.frames, fixture.height, fixture.width,
             repeated_centres(fixture.truth_y),
             repeated_centres(fixture.truth_x),
             std::span<const int>{}, std::span<const int>{},
             std::vector<float>{0.0f}, std::vector<float>{0.0f},
            1U, 1U, fixture.sample_y, fixture.sample_x, config),
        std::invalid_argument);
}

TEST_CASE("SET4 REP4 raw rerank evaluates only deduplicated domain representatives",
          "[wsvt][umpa][set4][rep4]") {
    const Fixture fixture = make_fixture();
    const std::vector<int> centres_y = repeated_centres(fixture.truth_y);
    const std::vector<int> centres_x = repeated_centres(fixture.truth_x);
    const std::vector<int> representatives_y{
        fixture.truth_y, fixture.truth_y, fixture.truth_y + 1,
        fixture.truth_y - 1};
    const std::vector<int> representatives_x{
        fixture.truth_x, fixture.truth_x + 1, fixture.truth_x,
        fixture.truth_x - 1};
    wsvt::SetTransportRawRerankConfig config;
    config.enabled = true;
    config.representatives_only = true;
    config.objective = wsvt::SetTransportRawObjective::WindowedZncc;
    const auto output = wsvt::rerank_set_transport_raw(
        fixture.sample, fixture.reference,
        fixture.frames, fixture.height, fixture.width,
        centres_y, centres_x, representatives_y, representatives_x,
        std::vector<float>{0.0f}, std::vector<float>{0.0f},
        1U, 1U, fixture.sample_y, fixture.sample_x, config);
    REQUIRE(output.raw_candidate_count == 4U);
    REQUIRE(output.raw_observation_count == 4U * fixture.frames * 9U);
    REQUIRE(output.displace_y[0] == Catch::Approx(
        static_cast<float>(fixture.truth_y)));
    REQUIRE(output.displace_x[0] == Catch::Approx(
        static_cast<float>(fixture.truth_x)));
}
