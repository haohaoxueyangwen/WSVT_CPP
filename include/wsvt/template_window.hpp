#pragma once

#include "wsvt/aligned_alloc.hpp"

#include <cstddef>
#include <vector>

namespace wsvt {

AlignedVector<float> stack_template_window(
    const std::vector<float>& img,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int n_template,
    std::size_t& out_h,
    std::size_t& out_w,
    std::size_t& out_depth);

}
