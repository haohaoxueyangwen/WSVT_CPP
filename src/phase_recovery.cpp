#include "wsvt/phase_recovery.hpp"
#include "wsvt/types.hpp"

#include <cmath>
#include <complex>
#include <mutex>
#include <stdexcept>
#include <vector>

#if defined(WSVT_HAS_FFTW)
#include <fftw3.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace wsvt {

namespace {

inline std::size_t idx2(std::size_t y, std::size_t x, std::size_t w) {
    return y * w + x;
}

#if defined(WSVT_HAS_FFTW)

// Initialize FFTW threading once (thread-safe)
static std::once_flag fftw_init_flag;

static void init_fftw_threads() {
    std::call_once(fftw_init_flag, [] {
        if (fftwf_init_threads()) {
            int nthreads = 1;
            #ifdef _OPENMP
            nthreads = omp_get_max_threads();
            #endif
            fftwf_plan_with_nthreads(nthreads);
        }
    });
}

Image2D<float> frankot_chellappa_fftw(
    ImageView2D<const float> dpc_x,
    ImageView2D<const float> dpc_y) {

    init_fftw_threads();

    const Shape2D shape = dpc_x.shape();
    const std::size_t h = shape.h;
    const std::size_t w = shape.w;
    const int rows = static_cast<int>(h);
    const int cols = static_cast<int>(w);
    const std::size_t n_real = h * w;
    // r2c output: h * (w/2 + 1) complex values
    const std::size_t n_complex = h * (w / 2 + 1);

    // RAII-managed FFTW buffers: automatic cleanup on any exit path
    FFTWBuffer<float> in_x(n_real);
    FFTWBuffer<float> in_y(n_real);
    FFTWBuffer<fftwf_complex> fx(n_complex);
    FFTWBuffer<fftwf_complex> fy(n_complex);
    FFTWBuffer<fftwf_complex> div(n_complex);
    FFTWBuffer<float> out(n_real);

    if (!in_x.data() || !in_y.data() || !fx.data() || !fy.data() ||
        !div.data() || !out.data()) {
        throw std::runtime_error("FFTW allocation failed");
    }

    // Copy input data
    {
        float* ix = in_x.data();
        float* iy = in_y.data();
        const float* dx = dpc_x.data();
        const float* dy = dpc_y.data();
        for (std::size_t i = 0; i < n_real; ++i) {
            ix[i] = dx[i];
            iy[i] = dy[i];
        }
    }

    // RAII-managed plans
    FFTWPlan plan_fx{fftwf_plan_dft_r2c_2d(rows, cols, in_x.data(), fx.data(), FFTW_ESTIMATE)};
    FFTWPlan plan_fy{fftwf_plan_dft_r2c_2d(rows, cols, in_y.data(), fy.data(), FFTW_ESTIMATE)};
    FFTWPlan plan_inv{fftwf_plan_dft_c2r_2d(rows, cols, div.data(), out.data(), FFTW_ESTIMATE)};

    // Forward transforms
    plan_fx.execute();
    plan_fy.execute();

    // Frequency-domain integration: (-j*wx*Fx - j*wy*Fy) / (wx^2 + wy^2)
    const float pi = static_cast<float>(std::acos(-1.0));
    const float eps = 1e-12f;
    const std::size_t half_w = w / 2 + 1;

    fftwf_complex* fx_ptr = fx.data();
    fftwf_complex* fy_ptr = fy.data();
    fftwf_complex* div_ptr = div.data();

    for (std::size_t v = 0; v < h; ++v) {
        float wy = 2.0f * pi *
            static_cast<float>(v <= h / 2
                ? static_cast<long long>(v)
                : static_cast<long long>(v) - static_cast<long long>(h)) /
            static_cast<float>(h);
        for (std::size_t u = 0; u < half_w; ++u) {
            float wx = 2.0f * pi *
                static_cast<float>(u <= w / 2
                    ? static_cast<long long>(u)
                    : static_cast<long long>(u) - static_cast<long long>(w)) /
                static_cast<float>(w);

            const float den = std::max(wx * wx + wy * wy, eps);
            const std::size_t k = v * half_w + u;

            // num = -j*wx*fx[k] - j*wy*fy[k]
            // -j*(a+bi) = b - ja
            const float num_r = wx * fx_ptr[k][1] + wy * fy_ptr[k][1];
            const float num_i = -(wx * fx_ptr[k][0] + wy * fy_ptr[k][0]);

            div_ptr[k][0] = num_r / den;
            div_ptr[k][1] = num_i / den;
        }
    }

    // Inverse transform (FFTW c2r does NOT normalize)
    plan_inv.execute();

    // Copy result and normalize (FFTW convention: unnormalized, divide by N)
    const float scale = 1.0f / static_cast<float>(n_real);
    std::vector<float> phi(n_real, 0.0f);
    const float* out_ptr = out.data();
    float mean = 0.0f;
    for (std::size_t i = 0; i < n_real; ++i) {
        phi[i] = out_ptr[i] * scale;
        mean += phi[i];
    }
    mean /= static_cast<float>(n_real);
    for (float& v : phi) {
        v -= mean;
    }

    return Image2D<float>(std::move(phi), shape);
}

#endif  // WSVT_HAS_FFTW

// Brute-force O(n^4) fallback when FFTW is not available
std::vector<std::complex<float>> dft2(
    const float* in,
    std::size_t h,
    std::size_t w) {
    std::vector<std::complex<float>> out(h * w, {0.0f, 0.0f});
    const float pi = static_cast<float>(std::acos(-1.0));
    for (std::size_t v = 0; v < h; ++v) {
        for (std::size_t u = 0; u < w; ++u) {
            std::complex<float> sum(0.0f, 0.0f);
            for (std::size_t y = 0; y < h; ++y) {
                for (std::size_t x = 0; x < w; ++x) {
                    const float ang = -2.0f * pi * (
                        static_cast<float>(u * x) / static_cast<float>(w) +
                        static_cast<float>(v * y) / static_cast<float>(h));
                    const std::complex<float> ph(std::cos(ang), std::sin(ang));
                    sum += in[idx2(y, x, w)] * ph;
                }
            }
            out[idx2(v, u, w)] = sum;
        }
    }
    return out;
}

std::vector<float> idft2_real(
    const std::vector<std::complex<float>>& in,
    std::size_t h,
    std::size_t w) {
    std::vector<float> out(h * w, 0.0f);
    const float pi = static_cast<float>(std::acos(-1.0));
    const float scale = 1.0f / static_cast<float>(h * w);
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            std::complex<float> sum(0.0f, 0.0f);
            for (std::size_t v = 0; v < h; ++v) {
                for (std::size_t u = 0; u < w; ++u) {
                    const float ang = 2.0f * pi * (
                        static_cast<float>(u * x) / static_cast<float>(w) +
                        static_cast<float>(v * y) / static_cast<float>(h));
                    const std::complex<float> ph(std::cos(ang), std::sin(ang));
                    sum += in[idx2(v, u, w)] * ph;
                }
            }
            out[idx2(y, x, w)] = sum.real() * scale;
        }
    }
    return out;
}

Image2D<float> frankot_chellappa_bruteforce(
    ImageView2D<const float> dpc_x,
    ImageView2D<const float> dpc_y) {
    const Shape2D shape = dpc_x.shape();
    const std::size_t h = shape.h;
    const std::size_t w = shape.w;

    const auto fx = dft2(dpc_x.data(), h, w);
    const auto fy = dft2(dpc_y.data(), h, w);
    std::vector<std::complex<float>> div_arr(h * w, {0.0f, 0.0f});
    const float pi = static_cast<float>(std::acos(-1.0));
    const float eps = 1e-12f;
    const std::complex<float> j(0.0f, 1.0f);

    for (std::size_t v = 0; v < h; ++v) {
        float wy = 2.0f * pi * static_cast<float>(v <= h / 2 ? v : static_cast<long long>(v) - static_cast<long long>(h)) / static_cast<float>(h);
        for (std::size_t u = 0; u < w; ++u) {
            float wx = 2.0f * pi * static_cast<float>(u <= w / 2 ? u : static_cast<long long>(u) - static_cast<long long>(w)) / static_cast<float>(w);
            const float den = std::max(wx * wx + wy * wy, eps);
            const std::complex<float> num = -j * wx * fx[idx2(v, u, w)] - j * wy * fy[idx2(v, u, w)];
            div_arr[idx2(v, u, w)] = num / den;
        }
    }

    auto phi = idft2_real(div_arr, h, w);
    float mean = 0.0f;
    for (float v : phi) mean += v;
    mean /= static_cast<float>(phi.size());
    for (float& v : phi) v -= mean;
    return Image2D<float>(std::move(phi), shape);
}

}  // namespace

Image2D<float> frankot_chellappa(
    ImageView2D<const float> dpc_x,
    ImageView2D<const float> dpc_y) {
    if (dpc_x.shape() != dpc_y.shape()) {
        throw std::invalid_argument("frankot_chellappa: dpc_x and dpc_y shape mismatch");
    }
    if (dpc_x.data() == nullptr || dpc_y.data() == nullptr) {
        throw std::invalid_argument("frankot_chellappa: null input");
    }

#if defined(WSVT_HAS_FFTW)
    return frankot_chellappa_fftw(dpc_x, dpc_y);
#else
    return frankot_chellappa_bruteforce(dpc_x, dpc_y);
#endif
}

}
