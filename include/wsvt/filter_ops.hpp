#pragma once

#include "wsvt/image.hpp"

namespace wsvt {

// Outlier-rejection median filter: replace pixels whose deviation from a
// local median exceeds `val_thresh` with the median value. Corner pixels
// are exempted to match the original Python behaviour.
Image2D<float> filter_erosion(
    ImageView2D<const float> image,
    float val_thresh,
    int filt_sz = 2);

}
