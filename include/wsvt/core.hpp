#pragma once

#include "wsvt/image.hpp"

#include <array>
#include <cstddef>
#include <span>

namespace wsvt {

struct DispResult {
    float disp_y {};
    float disp_x {};
    float sn_ratio {};
    float corr_max {};
};

// Locate sub-pixel peak of a 2D correlation map using a quadratic fit.
// `xx_axis` / `yy_axis` are flattened (H*W) grids matching `corr`.
DispResult find_disp(
    ImageView2D<const float> corr,
    std::span<const float> xx_axis,
    std::span<const float> yy_axis,
    bool sub_resolution);

std::array<std::size_t, 3> infer_dims(
    std::size_t flat_size,
    std::size_t depth,
    std::size_t width);

}
