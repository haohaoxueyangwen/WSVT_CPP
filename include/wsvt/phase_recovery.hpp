#pragma once

#include "wsvt/image.hpp"

namespace wsvt {

[[nodiscard]] Image2D<float> frankot_chellappa(
    ImageView2D<const float> dpc_x,
    ImageView2D<const float> dpc_y);

}  // namespace wsvt
