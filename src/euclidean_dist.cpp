#include "wsvt/euclidean_dist.hpp"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace wsvt {

Image2D<float> dist_correlation(
    std::span<const float> v1,
    TensorView3D<const float, Layout::HWD> array2) {
    const Shape3D s = array2.shape();
    const std::size_t h = s.d0;
    const std::size_t w = s.d1;
    const std::size_t depth = s.d2;
    if (v1.size() != depth) {
        throw std::invalid_argument("v1 size must equal depth");
    }

    Image2D<float> out(Shape2D{h, w});

    const float* v1_ptr = v1.data();
    const float* a2_ptr = array2.data();
    float* out_ptr = out.data();
    const std::size_t plane = h * w;

    // OpenMP parallel + compiler auto-vectorization (AVX-512 with -march=native -ffast-math)
    // Inner depth loop accesses contiguous memory (HWD layout), ideal for SIMD.
    #pragma omp parallel for schedule(static)
    for (std::size_t idx = 0; idx < plane; ++idx) {
        const float* row = a2_ptr + idx * depth;
        float ss = 0.0f;
        for (std::size_t kk = 0; kk < depth; ++kk) {
            const float diff = v1_ptr[kk] - row[kk];
            ss += diff * diff;
        }
        out_ptr[idx] = -ss;
    }
    return out;
}

}
