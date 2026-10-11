#pragma once

#include "wsvt/export.hpp"
#include "wsvt/umpa_physical_fit.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace wsvt {

inline constexpr std::size_t kScalarDarkfieldDefaultRadius = 7;

/// Optional post-tracking scalar ModelDF fit. Default off: WSVT/WXST search
/// math is unchanged when this stage is disabled.
struct ScalarDarkfieldConfig {
    bool enabled = false;
    std::size_t analysis_radius = kScalarDarkfieldDefaultRadius;
    double relative_delta_tolerance = 1.0e-12;
    double transmission_epsilon = 1.0e-12;
};

/// Dense maps on the displacement grid. NaN marks refused estimates.
/// D is never clipped into [0, 1]. constraint_active is always false in v1.
struct ScalarDarkfieldMaps {
    std::size_t height = 0;
    std::size_t width = 0;
    std::vector<double> T;
    std::vector<double> D;
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> cost;
    std::vector<double> delta;
    std::vector<double> weight_sum;
    std::vector<double> lnD;
    std::vector<std::uint8_t> numerical_valid;
    std::vector<std::uint8_t> physical_valid;
    std::vector<std::uint8_t> constraint_active;
    std::uint64_t numerical_valid_pixel_count = 0;
    std::uint64_t physical_valid_pixel_count = 0;
};

/// Public WSVT/WXST convention: sample (y,x) matches reference (y-u_y, x-u_x).
/// sample/reference stacks are frame-major FHW float32 (WXST: F=1).
/// Displacement pixel (i,j) is intensity coordinate (origin_y+i, origin_x+j).
/// The analysis window is Hamming of `config.analysis_radius`, not the search
/// half-window. Per-frame local reference means are used; a one-pixel mean is
/// rejected by the normal-equation gate.
WSVT_API ScalarDarkfieldMaps fit_scalar_darkfield_field(
    std::span<const float> sample_stack,
    std::span<const float> reference_stack,
    std::size_t frames,
    std::size_t raw_height,
    std::size_t raw_width,
    std::span<const float> displace_y,
    std::span<const float> displace_x,
    std::size_t output_height,
    std::size_t output_width,
    double origin_y,
    double origin_x,
    const ScalarDarkfieldConfig& config = {});

WSVT_API std::vector<float> scalar_darkfield_as_f32(
    const std::vector<double>& values);

WSVT_API std::vector<float> scalar_darkfield_mask_as_f32(
    const std::vector<std::uint8_t>& values);

struct ScalarDarkfieldF32 {
    std::vector<float> T;
    std::vector<float> D;
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> cost;
    std::vector<float> delta;
    std::vector<float> weight_sum;
    std::vector<float> lnD;
    std::vector<float> numerical_valid;
    std::vector<float> physical_valid;
    double time_s = 0.0;
    std::uint64_t numerical_valid_pixel_count = 0;
    std::uint64_t physical_valid_pixel_count = 0;
};

WSVT_API ScalarDarkfieldF32 pack_scalar_darkfield_f32(
    const ScalarDarkfieldMaps& maps);

}  // namespace wsvt
