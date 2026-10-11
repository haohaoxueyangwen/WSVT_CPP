#pragma once

#include "wsvt/image.hpp"

namespace wsvt {

// Center-crop a 2D image to an m x m ROI. If m == 0 or m > min(h, w),
// returns a copy of the original image unchanged (matches Python semantics).
Image2D<float> image_roi(ImageView2D<const float> img, std::size_t m);

// Center-crop each channel of a CHW tensor to m x m in the (H, W) plane.
// Returns a new CHW tensor with shape {ch, m, m} (or unchanged on m == 0
// or m > min(h, w)).
Tensor3D<float, Layout::CHW> image_roi(
    TensorView3D<const float, Layout::CHW> img, std::size_t m);

// Tensor-product cubic interpolation with uniform-grid not-a-knot boundary
// conditions, matching scipy.interpolate.RectBivariateSpline(kx=3, ky=3,
// s=0) for the WSVT/WXST coarse-to-fine displacement path.
Image2D<float> resample_rect_bivariate_spline(
    ImageView2D<const float> img, Shape2D output_shape);

}
