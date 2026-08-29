#pragma once

#include "wsvt/aligned_alloc.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace wsvt {

enum class PyramidDownsampleMode {
    Mean2x2 = 0,
    Db3Aa = 1
};

enum class PyramidNormalizationMode {
    PerLevelFeature = 0,
    InitialStack = 1
};

struct PyramidLevel {
    AlignedVector<float> data;
    std::size_t d0{};
    std::size_t d1{};
    std::size_t d2{};
};

struct PyramidResult {
    std::vector<PyramidLevel> ref_levels;
    std::vector<PyramidLevel> img_levels;
    double template_window_time_s = 0.0;
    double downsample_time_s = 0.0;
    double normalize_time_s = 0.0;
};

/// Raw CHW pyramid after the established InitialStack normalization.  This is
/// used by the default-off adaptive-frame prototype so temporal descriptors
/// can be materialized only at active sample/candidate-reference locations.
struct RawPyramidLevel {
    std::vector<float> data;
    std::size_t ch{};
    std::size_t h{};
    std::size_t w{};
};

struct RawPyramidResult {
    std::vector<RawPyramidLevel> levels;
    double downsample_time_s = 0.0;
    double normalize_time_s = 0.0;
};

/// HWD-native template window stacking: input CHW, output HWD.
/// Kept as reference implementation; hot path uses stack_and_normalize_template_hwd.
[[nodiscard]] AlignedVector<float> stack_template_window_hwd(
    const std::vector<float>& img,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int n_template,
    std::size_t& out_depth);

/// Pixel-major fused template window + PerLevelFeature normalization.
/// Replaces separate stack_template_window_hwd() + normalize_feature_depth_hwd()
/// with a single pass.  Shift order, wrap semantics, and normalization formula
/// are identical to the separate path.
[[nodiscard]] AlignedVector<float> stack_and_normalize_template_hwd(
    const std::vector<float>& img,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int n_template,
    std::size_t& out_depth);

[[nodiscard]] PyramidResult pyramid_data(
    const std::vector<float>& ref_data,
    const std::vector<float>& img_data,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int pyramid_level,
    int n_template,
    PyramidDownsampleMode mode = PyramidDownsampleMode::Db3Aa,
    PyramidNormalizationMode normalization_mode = PyramidNormalizationMode::PerLevelFeature);

/// Consuming variant for pipeline-owned CHW stacks. Both input vectors are
/// empty on successful return, avoiding a second full normalized CHW copy.
[[nodiscard]] PyramidResult pyramid_data_consume(
    std::vector<float>& ref_data,
    std::vector<float>& img_data,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int pyramid_level,
    int n_template,
    PyramidDownsampleMode mode = PyramidDownsampleMode::Db3Aa,
    PyramidNormalizationMode normalization_mode = PyramidNormalizationMode::PerLevelFeature);

/// Build normalized raw CHW levels without materializing HWD descriptors.
/// Arithmetic and zero-boundary db3 downsampling match the InitialStack path
/// used by Pyramid WSVT.
[[nodiscard]] RawPyramidResult normalized_raw_pyramid_data(
    const std::vector<float>& data,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int pyramid_level,
    PyramidDownsampleMode mode = PyramidDownsampleMode::Db3Aa);

/// Spatially downsample an already materialized HWD temporal descriptor. This
/// is the second half of the default-off prefix-compatible B2 representation:
/// temporal DWT and spatial db3 filtering are linear, so final-depth temporal
/// coefficients may be accumulated first and then propagated through the same
/// spatial pyramid. No feature-depth renormalization is applied here.
[[nodiscard]] std::vector<PyramidLevel> descriptor_spatial_pyramid_hwd(
    std::span<const float> descriptor_hwd,
    std::size_t h,
    std::size_t w,
    std::size_t depth,
    int pyramid_level,
    PyramidDownsampleMode mode = PyramidDownsampleMode::Db3Aa);

}  // namespace wsvt
