#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "wsvt/wsvt_pipeline.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>

using namespace wsvt;

namespace {

std::vector<float> make_stack(std::size_t ch, std::size_t h, std::size_t w, float bias) {
    std::vector<float> data(ch * h * w, 0.0f);
    for (std::size_t c = 0; c < ch; ++c) {
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t x = 0; x < w; ++x) {
                const float ramp = static_cast<float>(0.07 * static_cast<double>(y) +
                                                      0.11 * static_cast<double>(x) +
                                                      0.19 * static_cast<double>(c));
                data[c * h * w + y * w + x] = bias + ramp;
            }
        }
    }
    return data;
}

std::vector<float> make_textured_stack(
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    float bias) {
    std::vector<float> data(ch * h * w, 0.0f);
    for (std::size_t c = 0; c < ch; ++c) {
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t x = 0; x < w; ++x) {
                const double cd = static_cast<double>(c);
                const double yd = static_cast<double>(y);
                const double xd = static_cast<double>(x);
                const double texture =
                    std::sin(0.31 * (xd + 2.0 * cd)) +
                    std::cos(0.27 * (yd + cd)) +
                    0.002 * xd * yd * (cd + 1.0);
                data[c * h * w + y * w + x] =
                    bias + static_cast<float>(texture);
            }
        }
    }
    return data;
}

} // namespace

TEST_CASE("WSVT computes raw darkfield by default", "[wsvt][darkfield]") {
    constexpr std::size_t ch = 4;
    constexpr std::size_t h = 24;
    constexpr std::size_t w = 24;
    auto img = make_stack(ch, h, w, 12.0f);
    auto ref = make_stack(ch, h, w, 10.0f);

    WSVT solver(
        img, ref, ch, h, w,
        /*crop=*/0, /*cal_half_window=*/2, /*n_template=*/0,
        /*n_s_extend=*/1, /*n_cores=*/2, /*n_group=*/1,
        /*energy=*/14000.0, /*p_x=*/0.65e-6, /*mag_factor=*/1.0, /*z=*/0.5,
        /*wavelet_level_cut=*/1, /*pyramid_level=*/0, /*n_iter=*/1,
        /*use_estimate=*/false, /*use_wavelet=*/false, /*use_gpu=*/0);

    const auto result = solver.solver();

    REQUIRE(result.h > 0);
    REQUIRE(result.w > 0);
    REQUIRE(result.darkfield.size() == result.transmission_h * result.transmission_w);
    REQUIRE(result.darkfield_time_s >= 0.0);
    REQUIRE(result.darkfield_nd.size() == result.h * result.w);
    REQUIRE(result.second_best_score_neg_ssd.size() == result.h * result.w);
    REQUIRE(result.score_margin.size() == result.h * result.w);
    REQUIRE(result.peak_hessian_det.size() == result.h * result.w);
    REQUIRE(result.search_boundary_hit.size() == result.h * result.w);
    REQUIRE(result.search_geometry_valid.size() == result.h * result.w);
    for (std::size_t i = 0; i < result.h * result.w; ++i) {
        REQUIRE(result.darkfield_nd[i] >= result.second_best_score_neg_ssd[i]);
        REQUIRE(result.score_margin[i] >= 0.0f);
        REQUIRE((result.search_boundary_hit[i] == 0.0f || result.search_boundary_hit[i] == 1.0f));
        REQUIRE((result.search_geometry_valid[i] == 0.0f || result.search_geometry_valid[i] == 1.0f));
        if (result.search_geometry_valid[i] == 1.0f) {
            REQUIRE(result.search_boundary_hit[i] == 0.0f);
            REQUIRE(result.peak_hessian_det[i] > 0.0f);
        }
    }
}

TEST_CASE("WSVT can skip raw darkfield while keeping core outputs", "[wsvt][darkfield]") {
    constexpr std::size_t ch = 4;
    constexpr std::size_t h = 24;
    constexpr std::size_t w = 24;
    auto img = make_stack(ch, h, w, 12.0f);
    auto ref = make_stack(ch, h, w, 10.0f);

    WSVT solver(
        img, ref, ch, h, w,
        /*crop=*/0, /*cal_half_window=*/2, /*n_template=*/0,
        /*n_s_extend=*/1, /*n_cores=*/2, /*n_group=*/1,
        /*energy=*/14000.0, /*p_x=*/0.65e-6, /*mag_factor=*/1.0, /*z=*/0.5,
        /*wavelet_level_cut=*/1, /*pyramid_level=*/0, /*n_iter=*/1,
        /*use_estimate=*/false, /*use_wavelet=*/false, /*use_gpu=*/0,
        /*calc_darkfield=*/false);

    const auto result = solver.solver();

    REQUIRE(result.h > 0);
    REQUIRE(result.w > 0);
    REQUIRE(result.displace_x.size() == result.h * result.w);
    REQUIRE(result.displace_y.size() == result.h * result.w);
    REQUIRE(result.darkfield.empty());
    REQUIRE(result.darkfield_time_s == 0.0);
    REQUIRE(result.darkfield_nd.size() == result.h * result.w);
    REQUIRE(result.second_best_score_neg_ssd.size() == result.h * result.w);
    REQUIRE(result.score_margin.size() == result.h * result.w);
}

TEST_CASE("WSVT captures exact Top-K only at requested exhaustive-search pixels", "[wsvt][search][diagnostic]") {
    constexpr std::size_t ch = 8;
    constexpr std::size_t h = 24;
    constexpr std::size_t w = 24;
    const auto ref = make_textured_stack(ch, h, w, 10.0f);
    const auto img = ref;
    WSVT solver(
        img, ref, ch, h, w,
        /*crop=*/0, /*cal_half_window=*/2, /*n_template=*/0,
        /*n_s_extend=*/1, /*n_cores=*/1, /*n_group=*/1,
        14000.0, 0.65e-6, 1.0, 0.5,
        /*wavelet_level_cut=*/1, /*pyramid_level=*/0, /*n_iter=*/1,
        /*use_estimate=*/false, /*use_wavelet=*/false, /*use_gpu=*/0,
        /*calc_darkfield=*/false, /*phase_cores=*/1,
        /*search_early_abandon=*/false, /*search_top_k=*/2,
        /*search_block_size=*/4);
    REQUIRE(solver.search_topk_diagnostics().empty());
    solver.configure_search_topk_diagnostics({12 * w + 12}, 4);

    const auto result = solver.solver();
    const auto& diagnostics = solver.search_topk_diagnostics();
    REQUIRE(result.h == h - 4);
    REQUIRE(result.w == w - 4);
    REQUIRE(diagnostics.size() == 4);
    for (std::size_t rank = 0; rank < diagnostics.size(); ++rank) {
        const auto& value = diagnostics[rank];
        REQUIRE(value.request_index == 0);
        REQUIRE(value.raw_y == 12);
        REQUIRE(value.raw_x == 12);
        REQUIRE(value.rank == rank);
        REQUIRE(value.saved_candidate_y == -value.internal_candidate_y);
        REQUIRE(value.saved_candidate_x == -value.internal_candidate_x);
        REQUIRE(value.descriptor_cost_ssd == -value.descriptor_score_neg_ssd);
        if (rank > 0) {
            REQUIRE(diagnostics[rank - 1].descriptor_cost_ssd <=
                    value.descriptor_cost_ssd);
        }
    }
    REQUIRE(diagnostics[0].saved_candidate_y == 0);
    REQUIRE(diagnostics[0].saved_candidate_x == 0);
    REQUIRE(diagnostics[0].descriptor_cost_ssd == Catch::Approx(0.0f));
}

TEST_CASE("WSVT rejects invalid manual windows before allocating solver stages", "[wsvt][window]") {
    constexpr std::size_t ch = 4;
    constexpr std::size_t h = 24;
    constexpr std::size_t w = 24;
    const auto img = make_stack(ch, h, w, 12.0f);
    const auto ref = make_stack(ch, h, w, 10.0f);

    REQUIRE_THROWS_AS(
        WSVT(img, ref, ch, h, w,
             /*crop=*/0, /*cal_half_window=*/0, /*n_template=*/0,
             /*n_s_extend=*/1, /*n_cores=*/2, /*n_group=*/1,
             14000.0, 0.65e-6, 1.0, 0.5, 1, 0, 1, false, false, 0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        WSVT(img, ref, ch, h, w,
             /*crop=*/0, /*cal_half_window=*/2, /*n_template=*/-1,
             /*n_s_extend=*/1, /*n_cores=*/2, /*n_group=*/1,
             14000.0, 0.65e-6, 1.0, 0.5, 1, 0, 1, false, false, 0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        WSVT(img, ref, ch, h, w,
             /*crop=*/8, /*cal_half_window=*/3, /*n_template=*/1,
             /*n_s_extend=*/1, /*n_cores=*/2, /*n_group=*/1,
             14000.0, 0.65e-6, 1.0, 0.5, 1, 0, 1, false, false, 0),
        std::invalid_argument);
}

TEST_CASE("WSVT blockwise early abandon preserves exhaustive displacement outputs", "[wsvt][search][pruning]") {
    constexpr std::size_t ch = 8;
    constexpr std::size_t h = 24;
    constexpr std::size_t w = 24;
    const auto img = make_textured_stack(ch, h, w, 10.0f);
    const auto ref = img;

    WSVT exhaustive(
        img, ref, ch, h, w,
        /*crop=*/0, /*cal_half_window=*/2, /*n_template=*/0,
        /*n_s_extend=*/1, /*n_cores=*/1, /*n_group=*/1,
        /*energy=*/14000.0, /*p_x=*/0.65e-6, /*mag_factor=*/1.0, /*z=*/0.5,
        /*wavelet_level_cut=*/1, /*pyramid_level=*/0, /*n_iter=*/1,
        /*use_estimate=*/false, /*use_wavelet=*/false, /*use_gpu=*/0,
        /*calc_darkfield=*/false, /*phase_cores=*/1,
        /*search_early_abandon=*/false, /*search_top_k=*/4,
        /*search_block_size=*/4);
    WSVT pruned(
        img, ref, ch, h, w,
        /*crop=*/0, /*cal_half_window=*/2, /*n_template=*/0,
        /*n_s_extend=*/1, /*n_cores=*/1, /*n_group=*/1,
        /*energy=*/14000.0, /*p_x=*/0.65e-6, /*mag_factor=*/1.0, /*z=*/0.5,
        /*wavelet_level_cut=*/1, /*pyramid_level=*/0, /*n_iter=*/1,
        /*use_estimate=*/false, /*use_wavelet=*/false, /*use_gpu=*/0,
        /*calc_darkfield=*/false, /*phase_cores=*/1,
        /*search_early_abandon=*/true, /*search_top_k=*/4,
        /*search_block_size=*/4);

    const auto expected = exhaustive.solver();
    const auto actual = pruned.solver();

    REQUIRE(actual.h == expected.h);
    REQUIRE(actual.w == expected.w);
    REQUIRE(actual.search_candidate_count == expected.search_candidate_count);
    REQUIRE(expected.search_abandoned_candidate_count == 0);
    REQUIRE(expected.search_distance_terms_evaluated ==
            expected.search_distance_terms_possible);
    REQUIRE(actual.search_abandoned_candidate_count > 0);
    REQUIRE(actual.search_distance_terms_evaluated <
            actual.search_distance_terms_possible);
    REQUIRE(actual.search_refine_terms_evaluated > 0);

    for (std::size_t i = 0; i < actual.displace_x.size(); ++i) {
        REQUIRE(actual.displace_x[i] == Catch::Approx(expected.displace_x[i]).margin(1.0e-5));
        REQUIRE(actual.displace_y[i] == Catch::Approx(expected.displace_y[i]).margin(1.0e-5));
        REQUIRE(actual.darkfield_nd[i] == Catch::Approx(expected.darkfield_nd[i]).margin(1.0e-5));
        REQUIRE(actual.second_best_score_neg_ssd[i] ==
                Catch::Approx(expected.second_best_score_neg_ssd[i]).margin(1.0e-5));
    }
}

TEST_CASE("WSVT validates exact Top-K pruning options", "[wsvt][search][validation]") {
    constexpr std::size_t ch = 4;
    constexpr std::size_t h = 24;
    constexpr std::size_t w = 24;
    const auto img = make_stack(ch, h, w, 12.0f);
    const auto ref = make_stack(ch, h, w, 10.0f);

    REQUIRE_THROWS_AS(
        WSVT(img, ref, ch, h, w,
             0, 2, 0, 1, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
             1, 0, 1, false, false, 0, false, 1, true, 1, 16),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        WSVT(img, ref, ch, h, w,
             0, 2, 0, 1, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
             1, 0, 1, false, false, 0, false, 1, true, 2, 0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        WSVT(img, ref, ch, h, w,
             0, 2, 0, 1, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
             1, 0, 1, false, false, 0, false, 1,
             true, 2, 4, true, 4),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        WSVT(img, ref, ch, h, w,
             0, 2, 0, 1, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
             1, 0, 1, false, false, 0, false, 1,
             false, 2, 4, true, 0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        WSVT(img, ref, ch, h, w,
             0, 2, 0, 1, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
             1, 0, 1, false, false, 0, false, 1,
             false, 2, 4, false, 4, true),
        std::invalid_argument);
}

TEST_CASE("WSVT two-pass prefix screen preserves exhaustive outputs", "[wsvt][search][two-pass]") {
    constexpr std::size_t ch = 8;
    constexpr std::size_t h = 24;
    constexpr std::size_t w = 24;
    const auto img = make_textured_stack(ch, h, w, 10.0f);
    const auto ref = img;

    WSVT exhaustive(
        img, ref, ch, h, w,
        0, 2, 0, 1, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
        1, 0, 1, false, false, 0, false, 1,
        false, 2, 4, false, 4);
    WSVT screened(
        img, ref, ch, h, w,
        0, 2, 0, 1, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
        1, 0, 1, false, false, 0, false, 1,
        false, 2, 4, true, 4);
    WSVT cached_guard(
        img, ref, ch, h, w,
        0, 2, 0, 1, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
        1, 0, 1, false, false, 0, false, 1,
        false, 2, 4, true, 4, true);

    const auto expected = exhaustive.solver();
    const auto actual = screened.solver();
    const auto cached = cached_guard.solver();
    REQUIRE(actual.search_candidate_count == expected.search_candidate_count);
    REQUIRE(actual.search_abandoned_candidate_count > 0);
    REQUIRE(actual.search_prefix_terms_evaluated > 0);
    REQUIRE(actual.search_full_candidate_count < actual.search_candidate_count);
    REQUIRE(actual.search_distance_terms_evaluated <
            actual.search_distance_terms_possible);
    REQUIRE(actual.search_refine_terms_evaluated > 0);
    REQUIRE(expected.search_prefix_terms_evaluated == 0);
    REQUIRE(expected.search_full_candidate_count == expected.search_candidate_count);
    REQUIRE(actual.search_guard_check_count > 0);
    REQUIRE(actual.search_guard_refresh_count == 0);
    REQUIRE(cached.search_guard_check_count == actual.search_guard_check_count);
    REQUIRE(cached.search_guard_refresh_count > 0);
    REQUIRE(cached.search_guard_refresh_count < cached.search_guard_check_count);
    REQUIRE(cached.search_candidate_count == actual.search_candidate_count);
    REQUIRE(cached.search_abandoned_candidate_count ==
            actual.search_abandoned_candidate_count);
    REQUIRE(cached.search_distance_terms_evaluated ==
            actual.search_distance_terms_evaluated);
    REQUIRE(cached.search_distance_terms_possible ==
            actual.search_distance_terms_possible);
    REQUIRE(cached.search_refine_terms_evaluated ==
            actual.search_refine_terms_evaluated);
    REQUIRE(cached.search_prefix_terms_evaluated ==
            actual.search_prefix_terms_evaluated);
    REQUIRE(cached.search_full_candidate_count ==
            actual.search_full_candidate_count);

    const auto require_same = [](const std::vector<float>& lhs,
                                 const std::vector<float>& rhs) {
        REQUIRE(lhs.size() == rhs.size());
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            REQUIRE(lhs[i] == Catch::Approx(rhs[i]).margin(1.0e-5));
        }
    };
    require_same(actual.displace_x, expected.displace_x);
    require_same(actual.displace_y, expected.displace_y);
    require_same(actual.dpc_x, expected.dpc_x);
    require_same(actual.dpc_y, expected.dpc_y);
    require_same(actual.phase, expected.phase);
    require_same(actual.transmission, expected.transmission);
    require_same(actual.darkfield_nd, expected.darkfield_nd);
    require_same(actual.second_best_score_neg_ssd,
                 expected.second_best_score_neg_ssd);
    require_same(actual.score_margin, expected.score_margin);
    require_same(actual.peak_hessian_det, expected.peak_hessian_det);
    require_same(actual.interlevel_displacement_delta,
                 expected.interlevel_displacement_delta);
    require_same(actual.search_boundary_hit, expected.search_boundary_hit);
    require_same(actual.search_geometry_valid, expected.search_geometry_valid);

    // U2C changes only threshold bookkeeping. Its twelve publication-facing
    // fields must be bitwise identical to the U2B two-pass implementation.
    REQUIRE(cached.displace_x == actual.displace_x);
    REQUIRE(cached.displace_y == actual.displace_y);
    REQUIRE(cached.dpc_x == actual.dpc_x);
    REQUIRE(cached.dpc_y == actual.dpc_y);
    REQUIRE(cached.phase == actual.phase);
    REQUIRE(cached.transmission == actual.transmission);
    REQUIRE(cached.darkfield_nd == actual.darkfield_nd);
    REQUIRE(cached.second_best_score_neg_ssd ==
            actual.second_best_score_neg_ssd);
    REQUIRE(cached.score_margin == actual.score_margin);
    REQUIRE(cached.peak_hessian_det == actual.peak_hessian_det);
    REQUIRE(cached.interlevel_displacement_delta ==
            actual.interlevel_displacement_delta);
    REQUIRE(cached.search_boundary_hit == actual.search_boundary_hit);
    REQUIRE(cached.search_geometry_valid == actual.search_geometry_valid);
}

TEST_CASE("WSVT B1 accepts confident fine-level pixels through the easy window",
          "[wsvt][easy-to-hard][b1]") {
    constexpr std::size_t ch = 8;
    constexpr std::size_t h = 32;
    constexpr std::size_t w = 32;
    const auto ref = make_textured_stack(ch, h, w, 10.0f);
    const auto img = ref;

    WSVT baseline(
        img, ref, ch, h, w,
        0, 4, 0, 2, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
        1, 1, 1, false, false, 0, false, 1);
    WSVT candidate(
        img, ref, ch, h, w,
        0, 4, 0, 2, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
        1, 1, 1, false, false, 0, false, 1);
    EasyToHardConfig config;
    config.confidence_aware_refinement = true;
    config.easy_half_window = 1;
    config.score_margin_min = 0.0f;
    config.interlevel_delta_max_px = 1.5f;
    candidate.configure_easy_to_hard(config);

    const auto expected = baseline.solver();
    const auto actual = candidate.solver();
    REQUIRE(actual.easy_path_eligible_pixel_count > 0);
    REQUIRE(actual.easy_path_accepted_pixel_count > 0);
    REQUIRE(actual.search_candidate_count <
            actual.search_dense_baseline_candidate_count);
    REQUIRE(actual.displace_x == expected.displace_x);
    REQUIRE(actual.displace_y == expected.displace_y);
    REQUIRE(actual.normalized_peak_curvature.size() == actual.h * actual.w);
    REQUIRE(actual.interlevel_displacement_delta.size() == actual.h * actual.w);
    REQUIRE(actual.effective_search_half_window.size() == actual.h * actual.w);
    REQUIRE(std::any_of(
        actual.confidence_path.begin(), actual.confidence_path.end(),
        [](float value) { return value == 2.0f; }));
}

TEST_CASE("WSVT B2 stops easy pixels before the final temporal stage",
          "[wsvt][easy-to-hard][b2]") {
    constexpr std::size_t ch = 16;
    constexpr std::size_t h = 32;
    constexpr std::size_t w = 32;
    const auto ref = make_textured_stack(ch, h, w, 10.0f);
    const auto img = ref;
    WSVT solver(
        img, ref, ch, h, w,
        0, 4, 0, 2, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
        1, 1, 1, false, false, 0, false, 1);
    EasyToHardConfig config;
    config.confidence_aware_refinement = true;
    config.adaptive_frames = true;
    config.easy_half_window = 1;
    config.score_margin_min = 0.0f;
    config.frame_stages = {8, 16};
    solver.configure_easy_to_hard(config);

    const auto result = solver.solver();
    CAPTURE(result.temporal_stage_frame_counts,
            result.temporal_stage_active_pixel_counts,
            result.temporal_stage_accepted_pixel_counts,
            result.temporal_frame_candidate_terms_actual,
            result.temporal_frame_candidate_terms_dense_baseline);
    REQUIRE(result.temporal_stage_frame_counts ==
            std::vector<std::size_t>{8, 16});
    REQUIRE(result.temporal_stage_active_pixel_counts.size() == 2);
    REQUIRE(result.temporal_stage_accepted_pixel_counts.size() == 2);
    REQUIRE(result.temporal_stage_accepted_pixel_counts[0] > 0);
    REQUIRE(result.temporal_stage_active_pixel_counts[1] <
            result.temporal_stage_active_pixel_counts[0]);
    REQUIRE(result.temporal_frame_candidate_terms_actual > 0);
    REQUIRE(result.temporal_frame_candidate_terms_dense_baseline > 0);
    REQUIRE(std::any_of(
        result.temporal_frames_used.begin(), result.temporal_frames_used.end(),
        [](float frames) { return frames == 8.0f; }));
    REQUIRE(std::any_of(
        result.temporal_frames_used.begin(), result.temporal_frames_used.end(),
        [](float frames) { return frames == 16.0f; }));
}

TEST_CASE("WSVT B2 sends ambiguous pixels to the final temporal stage",
          "[wsvt][easy-to-hard][b2]") {
    constexpr std::size_t ch = 8;
    constexpr std::size_t h = 32;
    constexpr std::size_t w = 32;
    const std::vector<float> img(ch * h * w, 5.0f);
    const auto ref = img;
    WSVT solver(
        img, ref, ch, h, w,
        0, 4, 0, 2, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
        1, 1, 1, false, false, 0, false, 1);
    EasyToHardConfig config;
    config.confidence_aware_refinement = true;
    config.adaptive_frames = true;
    config.easy_half_window = 1;
    config.score_margin_min = 0.01f;
    config.frame_stages = {4, 8};
    solver.configure_easy_to_hard(config);

    const auto result = solver.solver();
    REQUIRE(result.temporal_stage_frame_counts ==
            std::vector<std::size_t>{4, 8});
    REQUIRE(result.temporal_stage_active_pixel_counts.size() == 2);
    REQUIRE(result.temporal_stage_active_pixel_counts[1] > 0);
    for (const float frames : result.temporal_frames_used) {
        REQUIRE(frames == 8.0f);
    }
    REQUIRE(std::all_of(
        result.confidence_path.begin(), result.confidence_path.end(),
        [](float value) { return value == 4.0f; }));
}

TEST_CASE("WSVT B2 v2 lazy descriptors preserve the dense B2 result",
          "[wsvt][easy-to-hard][b2][lazy]") {
    constexpr std::size_t ch = 16;
    constexpr std::size_t h = 32;
    constexpr std::size_t w = 32;
    const auto ref = make_textured_stack(ch, h, w, 10.0f);
    const auto img = ref;
    WSVT dense(
        img, ref, ch, h, w,
        0, 4, 0, 2, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
        1, 1, 1, false, false, 0, false, 1);
    WSVT lazy(
        img, ref, ch, h, w,
        0, 4, 0, 2, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
        1, 1, 1, false, false, 0, false, 1);
    EasyToHardConfig dense_config;
    dense_config.confidence_aware_refinement = true;
    dense_config.adaptive_frames = true;
    dense_config.easy_half_window = 1;
    dense_config.score_margin_min = 0.0f;
    dense_config.frame_stages = {8, 16};
    auto lazy_config = dense_config;
    lazy_config.lazy_temporal_descriptors = true;
    dense.configure_easy_to_hard(dense_config);
    lazy.configure_easy_to_hard(lazy_config);

    const auto expected = dense.solver();
    const auto actual = lazy.solver();
    REQUIRE(actual.displace_x == expected.displace_x);
    REQUIRE(actual.displace_y == expected.displace_y);
    REQUIRE(actual.score_margin == expected.score_margin);
    REQUIRE(actual.interlevel_displacement_delta ==
            expected.interlevel_displacement_delta);
    REQUIRE(actual.temporal_frames_used == expected.temporal_frames_used);
    REQUIRE(actual.temporal_stage_active_pixel_counts ==
            expected.temporal_stage_active_pixel_counts);
    REQUIRE(actual.temporal_stage_accepted_pixel_counts ==
            expected.temporal_stage_accepted_pixel_counts);
    REQUIRE(actual.search_candidate_count == expected.search_candidate_count);
    REQUIRE(actual.temporal_stage_sample_descriptor_pixel_counts.size() == 2);
    REQUIRE(actual.temporal_stage_reference_descriptor_pixel_counts.size() == 2);
    REQUIRE(actual.temporal_descriptor_pixel_count_actual <
            actual.temporal_descriptor_pixel_count_dense_stage_baseline);
    REQUIRE(actual.temporal_descriptor_frame_terms_actual <
            actual.temporal_descriptor_frame_terms_dense_stage_baseline);
    REQUIRE(actual.temporal_descriptor_frame_terms_single_final_baseline > 0);
}

TEST_CASE("WSVT B2 full-prefix fallback restarts from the original estimate",
          "[wsvt][easy-to-hard][b2][fallback]") {
    constexpr std::size_t ch = 16;
    constexpr std::size_t h = 32;
    constexpr std::size_t w = 32;
    const auto ref = make_textured_stack(ch, h, w, 10.0f);
    auto img = ref;
    for (std::size_t frame = 0; frame < ch / 2; ++frame) {
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t x = 0; x < w; ++x) {
                img[(frame * h + y) * w + x] =
                    ref[(frame * h + y) * w + ((x + 2) % w)];
            }
        }
    }
    WSVT baseline(
        img, ref, ch, h, w,
        0, 4, 0, 2, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
        1, 1, 1, false, false, 0, false, 1);
    WSVT adaptive(
        img, ref, ch, h, w,
        0, 4, 0, 2, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
        1, 1, 1, false, false, 0, false, 1);
    EasyToHardConfig config;
    config.confidence_aware_refinement = true;
    config.adaptive_frames = true;
    config.lazy_temporal_descriptors = true;
    config.easy_half_window = 1;
    config.score_margin_min = 1.0e9f;
    config.interlevel_delta_max_px = 1.0e9f;
    config.frame_stages = {8, 16};
    adaptive.configure_easy_to_hard(config);

    const auto expected = baseline.solver();
    const auto actual = adaptive.solver();
    REQUIRE(actual.temporal_stage_accepted_pixel_counts[0] == 0);
    REQUIRE(actual.temporal_stage_active_pixel_counts[1] == h * w);
    REQUIRE(actual.displace_x == expected.displace_x);
    REQUIRE(actual.displace_y == expected.displace_y);
}

TEST_CASE("WSVT B2 prefix-compatible full fallback matches fixed final WSVT",
          "[wsvt][easy-to-hard][b2][incremental]") {
    constexpr std::size_t ch = 16;
    constexpr std::size_t h = 32;
    constexpr std::size_t w = 32;
    const auto ref = make_textured_stack(ch, h, w, 10.0f);
    auto img = ref;
    for (std::size_t frame = 0; frame < ch / 2; ++frame) {
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t x = 0; x < w; ++x) {
                img[(frame * h + y) * w + x] =
                    ref[(frame * h + y) * w + ((x + 2) % w)];
            }
        }
    }
    WSVT baseline(
        img, ref, ch, h, w,
        0, 4, 0, 2, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
        1, 1, 1, false, true, 0, false, 1);
    WSVT probe(
        img, ref, ch, h, w,
        0, 4, 0, 2, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
        1, 1, 1, false, true, 0, false, 1);
    WSVT adaptive(
        img, ref, ch, h, w,
        0, 4, 0, 2, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
        1, 1, 1, false, true, 0, false, 1);
    EasyToHardConfig config;
    config.confidence_aware_refinement = true;
    config.adaptive_frames = true;
    config.prefix_compatible_temporal_wavelet = true;
    config.easy_half_window = 1;
    config.temporal_prefix_half_window = 1;
    config.score_margin_min = 1.0e9f;
    config.interlevel_delta_max_px = 1.0e9f;
    config.frame_stages = {8, 16};
    adaptive.configure_easy_to_hard(config);

    const auto expected = baseline.solver();
    const auto actual = adaptive.solver();
    REQUIRE(actual.temporal_stage_accepted_pixel_counts[0] == 0);
    REQUIRE(actual.temporal_stage_active_pixel_counts[1] == h * w);
    REQUIRE(actual.displace_x.size() == expected.displace_x.size());
    REQUIRE(actual.displace_y.size() == expected.displace_y.size());
    double squared_error = 0.0;
    double max_error = 0.0;
    for (std::size_t pixel = 0; pixel < actual.displace_x.size(); ++pixel) {
        const double dx = static_cast<double>(actual.displace_x[pixel]) -
            static_cast<double>(expected.displace_x[pixel]);
        const double dy = static_cast<double>(actual.displace_y[pixel]) -
            static_cast<double>(expected.displace_y[pixel]);
        const double error = std::hypot(dx, dy);
        squared_error += error * error;
        max_error = std::max(max_error, error);
    }
    const double vector_rmse = std::sqrt(
        squared_error / static_cast<double>(actual.displace_x.size()));
    CAPTURE(vector_rmse, max_error);
    REQUIRE(vector_rmse < 1.0e-4);
    REQUIRE(max_error < 1.0e-3);
    REQUIRE(actual.temporal_prefix_raw_frame_pixel_terms_actual ==
            static_cast<std::uint64_t>(2 * h * w * ch));
    REQUIRE(actual.temporal_prefix_raw_frame_pixel_terms_actual ==
            actual.temporal_prefix_raw_frame_pixel_terms_final_baseline);
    REQUIRE(actual.temporal_descriptor_frame_terms_actual ==
            actual.temporal_descriptor_frame_terms_single_final_baseline);
    REQUIRE(actual.search_candidate_count > expected.search_candidate_count);
    REQUIRE(actual.search_candidate_count <
            2 * expected.search_candidate_count);

    config.temporal_probe_first_stage_only = true;
    probe.configure_easy_to_hard(config);
    const auto probe_result = probe.solver();
    REQUIRE(probe_result.temporal_stage_frame_counts ==
            std::vector<std::size_t>{8});
    REQUIRE(probe_result.temporal_stage_accepted_pixel_counts ==
            std::vector<std::uint64_t>{h * w});
    REQUIRE(std::all_of(
        probe_result.temporal_frames_used.begin(),
        probe_result.temporal_frames_used.end(),
        [](float frames) { return frames == 8.0f; }));
    REQUIRE(probe_result.search_candidate_count <
            expected.search_candidate_count);
}

TEST_CASE("WSVT validates the B1 and B2 research contract",
          "[wsvt][easy-to-hard][validation]") {
    constexpr std::size_t ch = 8;
    constexpr std::size_t h = 24;
    constexpr std::size_t w = 24;
    const auto img = make_textured_stack(ch, h, w, 10.0f);
    const auto ref = img;
    WSVT solver(
        img, ref, ch, h, w,
        0, 4, 0, 2, 1, 1, 14000.0, 0.65e-6, 1.0, 0.5,
        1, 1, 1, false, false, 0, false, 1);
    EasyToHardConfig invalid;
    invalid.adaptive_frames = true;
    invalid.frame_stages = {4, 8};
    REQUIRE_THROWS_AS(solver.configure_easy_to_hard(invalid),
                      std::invalid_argument);
    invalid.confidence_aware_refinement = true;
    invalid.easy_half_window = 2;
    REQUIRE_THROWS_AS(solver.configure_easy_to_hard(invalid),
                      std::invalid_argument);
    invalid.easy_half_window = 1;
    invalid.frame_stages = {4, 7};
    REQUIRE_THROWS_AS(solver.configure_easy_to_hard(invalid),
                      std::invalid_argument);
    EasyToHardConfig lazy_without_frames;
    lazy_without_frames.confidence_aware_refinement = true;
    lazy_without_frames.lazy_temporal_descriptors = true;
    REQUIRE_THROWS_AS(solver.configure_easy_to_hard(lazy_without_frames),
                      std::invalid_argument);
    EasyToHardConfig prefix_without_frames;
    prefix_without_frames.confidence_aware_refinement = true;
    prefix_without_frames.prefix_compatible_temporal_wavelet = true;
    REQUIRE_THROWS_AS(solver.configure_easy_to_hard(prefix_without_frames),
                      std::invalid_argument);
    EasyToHardConfig probe_without_prefix;
    probe_without_prefix.confidence_aware_refinement = true;
    probe_without_prefix.temporal_probe_first_stage_only = true;
    REQUIRE_THROWS_AS(solver.configure_easy_to_hard(probe_without_prefix),
                      std::invalid_argument);
    EasyToHardConfig conflicting;
    conflicting.confidence_aware_refinement = true;
    conflicting.adaptive_frames = true;
    conflicting.lazy_temporal_descriptors = true;
    conflicting.prefix_compatible_temporal_wavelet = true;
    conflicting.frame_stages = {4, 8};
    REQUIRE_THROWS_AS(solver.configure_easy_to_hard(conflicting),
                      std::invalid_argument);
}
