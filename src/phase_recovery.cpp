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
    const std::size_t n = h * w;

    // Use complex-to-complex transforms for clarity and correctness.
    // RAII-managed FFTW buffers.
    FFTWBuffer<fftwf_complex> buf_x(n);
    FFTWBuffer<fftwf_complex> buf_y(n);
    FFTWBuffer<fftwf_complex> buf_div(n);
    FFTWBuffer<fftwf_complex> buf_out(n);

    if (!buf_x.data() || !buf_y.data() || !buf_div.data() || !buf_out.data()) {
        throw std::runtime_error("FFTW allocation failed");
    }

    // Copy real input into complex buffers (imaginary = 0)
    const float* dx = dpc_x.data();
    const float* dy = dpc_y.data();
    for (std::size_t i = 0; i < n; ++i) {
        buf_x.data()[i][0] = dx[i];
        buf_x.data()[i][1] = 0.0f;
        buf_y.data()[i][0] = dy[i];
        buf_y.data()[i][1] = 0.0f;
    }

    // Forward FFT (complex-to-complex)
    FFTWPlanT<> plan_fx(fftwf_plan_dft_2d(rows, cols, buf_x.data(), buf_x.data(), FFTW_FORWARD, FFTW_ESTIMATE));
    FFTWPlanT<> plan_fy(fftwf_plan_dft_2d(rows, cols, buf_y.data(), buf_y.data(), FFTW_FORWARD, FFTW_ESTIMATE));
    plan_fx.execute();
    plan_fy.execute();

    // Frequency-domain integration: (-j*wx*Fx - j*wy*Fy) / (wx^2 + wy^2)
    const double pi = std::acos(-1.0);
    const double eps = 1e-12;

    for (std::size_t v = 0; v < h; ++v) {
        // fftfreq: v/h for v <= h/2, (v-h)/h for v > h/2
        const double fv = (v <= h / 2) ? static_cast<double>(v) : static_cast<double>(v) - static_cast<double>(h);
        const double wy = 2.0 * pi * fv / static_cast<double>(h);
        for (std::size_t u = 0; u < w; ++u) {
            const double fu = (u <= w / 2) ? static_cast<double>(u) : static_cast<double>(u) - static_cast<double>(w);
            const double wx = 2.0 * pi * fu / static_cast<double>(w);

            const double den = std::max(wx * wx + wy * wy, eps);
            const std::size_t k = v * w + u;

            // -j * wx * Fx - j * wy * Fy
            // If F = a + bi, then -j*F = b - ja
            // So: -j*wx*F = wx*b - j*wx*a; -j*wy*F = wy*b - j*wy*a
            const double fx_r = static_cast<double>(buf_x.data()[k][0]);
            const double fx_i = static_cast<double>(buf_x.data()[k][1]);
            const double fy_r = static_cast<double>(buf_y.data()[k][0]);
            const double fy_i = static_cast<double>(buf_y.data()[k][1]);

            const double num_r = wx * fx_i + wy * fy_i;
            const double num_i = -(wx * fx_r + wy * fy_r);

            buf_div.data()[k][0] = static_cast<float>(num_r / den);
            buf_div.data()[k][1] = static_cast<float>(num_i / den);
        }
    }

    // Inverse FFT (complex-to-complex)
    FFTWPlanT<> plan_inv(fftwf_plan_dft_2d(rows, cols, buf_div.data(), buf_out.data(), FFTW_BACKWARD, FFTW_ESTIMATE));
    plan_inv.execute();

    // Extract real part and normalize (FFTW convention: unnormalized, divide by N)
    const double scale = 1.0 / static_cast<double>(n);
    std::vector<float> phi(n, 0.0f);
    double mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        phi[i] = static_cast<float>(static_cast<double>(buf_out.data()[i][0]) * scale);
        mean += static_cast<double>(phi[i]);
    }
    mean /= static_cast<double>(n);
    for (float& v : phi) {
        v -= static_cast<float>(mean);
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
