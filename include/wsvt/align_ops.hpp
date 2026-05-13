#pragma once

#include "wsvt/image.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace wsvt {

struct ImageAlignResult {
    std::array<double, 2> shift;
    double error;
    double diffphase;
    Image2D<float> image_back;
};

// Phase-correlation sub-pixel alignment (brute-force DFT reference).
// Returns the estimated shift, correlation error, phase difference, and
// the offset image back-shifted onto `image`.
ImageAlignResult image_align(
    ImageView2D<const float> image,
    ImageView2D<const float> offset_image);

struct StackAlignResult {
    std::array<double, 2> shift;
    double error;
    double diffphase;
};

// Align an entire image stack (CHW layout) to a reference first frame.
// Uses phase_cross_correlation on the first frame to find the shift,
// then applies fourier_shift to every frame in ref_stack (in-place).
StackAlignResult stack_image_align(
    std::vector<float>& ref_stack,
    const float* img_first_frame, std::size_t img_stride,
    std::size_t ch, std::size_t h, std::size_t w);

}
