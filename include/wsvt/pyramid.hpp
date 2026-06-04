#pragma once

#include "wsvt/aligned_alloc.hpp"

#include <array>
#include <cstddef>
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

}  // namespace wsvt
