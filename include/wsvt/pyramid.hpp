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

struct PyramidLevel {
    AlignedVector<float> data;
    std::size_t d0{};
    std::size_t d1{};
    std::size_t d2{};
};

struct PyramidResult {
    std::vector<PyramidLevel> ref_levels;
    std::vector<PyramidLevel> img_levels;
};

[[nodiscard]] PyramidResult pyramid_data(
    const std::vector<float>& ref_data,
    const std::vector<float>& img_data,
    std::size_t ch,
    std::size_t h,
    std::size_t w,
    int pyramid_level,
    int n_template,
    PyramidDownsampleMode mode = PyramidDownsampleMode::Db3Aa);

}  // namespace wsvt
