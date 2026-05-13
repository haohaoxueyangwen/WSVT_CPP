#pragma once

#include "wsvt/image.hpp"

#include <optional>

namespace wsvt {

struct AutoCropResult {
    int y_begin;
    int y_end;
    int x_begin;
    int x_end;
};

AutoCropResult auto_crop(
    ImageView2D<const float> img,
    double shrink = 0.9,
    std::optional<float> count = std::nullopt);

}
