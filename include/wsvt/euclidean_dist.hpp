#pragma once

#include "wsvt/image.hpp"

#include <span>

namespace wsvt {

// Per-pixel negative squared Euclidean distance between a feature vector
// `v1` (length = depth) and every feature vector in `array2` (HWD layout,
// inner stride = depth). Output has shape (H, W).
Image2D<float> dist_correlation(
    std::span<const float> v1,
    TensorView3D<const float, Layout::HWD> array2);

}
