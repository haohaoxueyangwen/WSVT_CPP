#include "wsvt/align_ops.hpp"
#include "wsvt/common.hpp"

#if defined(WSVT_HAS_FFTW)
#include "wsvt/types.hpp"
#include <fftw3.h>
#include <mutex>
#endif

#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace wsvt {

namespace {

inline std::size_t aidx2(std::size_t y, std::size_t x, std::size_t w) {
    return y * w + x;
}

#if defined(WSVT_HAS_FFTW)

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

// FFTW-based FFT: real input -> full complex output (padded from r2c)
std::vector<std::complex<double>> fft2_real_fftw(ImageView2D<const float> in) {
    const Shape2D s = in.shape();
    const std::size_t h = s.h;
    const std::size_t w = s.w;
    const std::size_t n_real = h * w;
    const std::size_t n_complex = h * (w / 2 + 1);

    FFTWBuffer<float> buf_in(n_real);
    FFTWBuffer<fftwf_complex> buf_out(n_complex);

    std::memcpy(buf_in.data(), in.data(), n_real * sizeof(float));

    FFTWPlanT<> plan(fftwf_plan_dft_r2c_2d(
        static_cast<int>(h), static_cast<int>(w),
        buf_in.data(), buf_out.data(), FFTW_ESTIMATE));
    plan.execute();

    // Expand r2c output to full complex array
    std::vector<std::complex<double>> out(n_real, {0.0, 0.0});
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x <= w / 2; ++x) {
            const auto& c = buf_out.data()[y * (w / 2 + 1) + x];
            out[aidx2(y, x, w)] = std::complex<double>(c[0], c[1]);
        }
    }
    // Mirror conjugate for x > w/2
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = w / 2 + 1; x < w; ++x) {
            const std::size_t mx = w - x;
            const std::size_t my = (y == 0) ? 0 : h - y;
            out[aidx2(y, x, w)] = std::conj(out[aidx2(my, mx, w)]);
        }
    }
    return out;
}

// FFTW-based IFFT: full complex input -> real output
Image2D<float> ifft2_real_fftw(const std::vector<std::complex<double>>& in,
                               std::size_t h, std::size_t w) {
    const std::size_t n_real = h * w;
    const std::size_t n_complex = h * (w / 2 + 1);

    FFTWBuffer<fftwf_complex> buf_in(n_complex);
    FFTWBuffer<float> buf_out(n_real);

    // Pack only the r2c half
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x <= w / 2; ++x) {
            auto& c = buf_in.data()[y * (w / 2 + 1) + x];
            const auto& v = in[aidx2(y, x, w)];
            c[0] = static_cast<float>(v.real());
            c[1] = static_cast<float>(v.imag());
        }
    }

    FFTWPlanT<> plan(fftwf_plan_dft_c2r_2d(
        static_cast<int>(h), static_cast<int>(w),
        buf_in.data(), buf_out.data(), FFTW_ESTIMATE));
    plan.execute();

    Image2D<float> out(Shape2D{h, w});
    const float scale = 1.0f / static_cast<float>(n_real);
    for (std::size_t i = 0; i < n_real; ++i) {
        out.data()[i] = buf_out.data()[i] * scale;
    }
    return out;
}

#else  // !WSVT_HAS_FFTW — brute-force fallback

std::vector<std::complex<double>> fft2_real_naive(ImageView2D<const float> in) {
    const Shape2D s = in.shape();
    std::vector<std::complex<double>> out(s.size(), {0.0, 0.0});
    const double pi = std::acos(-1.0);
    for (std::size_t v = 0; v < s.h; ++v) {
        for (std::size_t u = 0; u < s.w; ++u) {
            std::complex<double> sum(0.0, 0.0);
            for (std::size_t y = 0; y < s.h; ++y) {
                for (std::size_t x = 0; x < s.w; ++x) {
                    const double ang = -2.0 * pi * (
                        static_cast<double>(u * x) / static_cast<double>(s.w) +
                        static_cast<double>(v * y) / static_cast<double>(s.h));
                    const std::complex<double> ph(std::cos(ang), std::sin(ang));
                    sum += static_cast<double>(in(y, x)) * ph;
                }
            }
            out[aidx2(v, u, s.w)] = sum;
        }
    }
    return out;
}

Image2D<float> ifft2_real_naive(const std::vector<std::complex<double>>& in,
                                 std::size_t h, std::size_t w) {
    std::vector<std::complex<double>> out(h * w, {0.0, 0.0});
    const double pi = std::acos(-1.0);
    const double scale = 1.0 / static_cast<double>(h * w);
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            std::complex<double> sum(0.0, 0.0);
            for (std::size_t v = 0; v < h; ++v) {
                for (std::size_t u = 0; u < w; ++u) {
                    const double ang = 2.0 * pi * (
                        static_cast<double>(u * x) / static_cast<double>(w) +
                        static_cast<double>(v * y) / static_cast<double>(h));
                    const std::complex<double> ph(std::cos(ang), std::sin(ang));
                    sum += in[aidx2(v, u, w)] * ph;
                }
            }
            out[aidx2(y, x, w)] = sum * scale;
        }
    }
    Image2D<float> result(Shape2D{h, w});
    for (std::size_t i = 0; i < out.size(); ++i) {
        result.data()[i] = static_cast<float>(out[i].real());
    }
    return result;
}

#endif  // WSVT_HAS_FFTW

std::vector<std::complex<double>> fourier_shift(
    const std::vector<std::complex<double>>& spectrum,
    std::size_t h,
    std::size_t w,
    double shift_y,
    double shift_x) {
    std::vector<std::complex<double>> out(h * w, {0.0, 0.0});
    const double pi = std::acos(-1.0);
    for (std::size_t v = 0; v < h; ++v) {
        const double fv = (v <= h / 2) ? static_cast<double>(v) : static_cast<double>(v) - static_cast<double>(h);
        for (std::size_t u = 0; u < w; ++u) {
            const double fu = (u <= w / 2) ? static_cast<double>(u) : static_cast<double>(u) - static_cast<double>(w);
            const double ang = -2.0 * pi * (fu * shift_x / static_cast<double>(w) + fv * shift_y / static_cast<double>(h));
            const std::complex<double> ph(std::cos(ang), std::sin(ang));
            out[aidx2(v, u, w)] = spectrum[aidx2(v, u, w)] * ph;
        }
    }
    return out;
}

double mse_for_shift(
    ImageView2D<const float> image,
    const std::vector<std::complex<double>>& off_fft,
    double shift_y,
    double shift_x) {
    const Shape2D s = image.shape();
    const auto shifted_fft = fourier_shift(off_fft, s.h, s.w, shift_y, shift_x);
#if defined(WSVT_HAS_FFTW)
    const auto shifted = ifft2_real_fftw(shifted_fft, s.h, s.w);
#else
    const auto shifted = ifft2_real_naive(shifted_fft, s.h, s.w);
#endif
    double mse = 0.0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const double d = static_cast<double>(image.data()[i]) -
                         static_cast<double>(shifted.data()[i]);
        mse += d * d;
    }
    return mse / static_cast<double>(s.size());
}

}  // anonymous namespace

ImageAlignResult image_align(
    ImageView2D<const float> image,
    ImageView2D<const float> offset_image) {
    if (image.shape() != offset_image.shape()) {
        throw std::invalid_argument("image_align: shape mismatch");
    }
#if defined(WSVT_HAS_FFTW)
    init_fftw_threads();
#endif
    const Shape2D s = image.shape();
    const std::size_t h = s.h;
    const std::size_t w = s.w;

#if defined(WSVT_HAS_FFTW)
    const auto image_fft = fft2_real_fftw(image);
    const auto offset_fft = fft2_real_fftw(offset_image);
#else
    const auto image_fft = fft2_real_naive(image);
    const auto offset_fft = fft2_real_naive(offset_image);
#endif

    std::vector<std::complex<double>> cps(h * w, {0.0, 0.0});
    for (std::size_t i = 0; i < cps.size(); ++i) {
        const std::complex<double> v = image_fft[i] * std::conj(offset_fft[i]);
        const double n = std::abs(v);
        cps[i] = (n > 0.0) ? (v / n) : std::complex<double>(0.0, 0.0);
    }

#if defined(WSVT_HAS_FFTW)
    const auto cc = [&] {
        const std::size_t n_complex = h * (w / 2 + 1);
        FFTWBuffer<fftwf_complex> buf_in(n_complex);
        FFTWBuffer<float> buf_out(h * w);
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t x = 0; x <= w / 2; ++x) {
                auto& c = buf_in.data()[y * (w / 2 + 1) + x];
                const auto& v = cps[aidx2(y, x, w)];
                c[0] = static_cast<float>(v.real());
                c[1] = static_cast<float>(v.imag());
            }
        }
        FFTWPlanT<> plan(fftwf_plan_dft_c2r_2d(
            static_cast<int>(h), static_cast<int>(w),
            buf_in.data(), buf_out.data(), FFTW_ESTIMATE));
        plan.execute();
        std::vector<std::complex<double>> result(h * w);
        const float scale = 1.0f / static_cast<float>(h * w);
        for (std::size_t i = 0; i < h * w; ++i) {
            result[i] = std::complex<double>(static_cast<double>(buf_out.data()[i] * scale), 0.0);
        }
        return result;
    }();
#else
    // brute-force IFFT for cross-correlation
    std::vector<std::complex<double>> cc(h * w, {0.0, 0.0});
    const double pi = std::acos(-1.0);
    const double scale = 1.0 / static_cast<double>(h * w);
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            std::complex<double> sum(0.0, 0.0);
            for (std::size_t v = 0; v < h; ++v) {
                for (std::size_t u = 0; u < w; ++u) {
                    const double ang = 2.0 * pi * (
                        static_cast<double>(u * x) / static_cast<double>(w) +
                        static_cast<double>(v * y) / static_cast<double>(h));
                    const std::complex<double> ph(std::cos(ang), std::sin(ang));
                    sum += cps[aidx2(v, u, w)] * ph;
                }
            }
            cc[aidx2(y, x, w)] = sum * scale;
        }
    }
#endif

    double best_abs = -1.0;
    std::size_t peak_y = 0;
    std::size_t peak_x = 0;
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            const double a = std::abs(cc[aidx2(y, x, w)]);
            if (a > best_abs) {
                best_abs = a;
                peak_y = y;
                peak_x = x;
            }
        }
    }

    double shift_y = static_cast<double>(peak_y);
    double shift_x = static_cast<double>(peak_x);
    if (shift_y > static_cast<double>(h) / 2.0) shift_y -= static_cast<double>(h);
    if (shift_x > static_cast<double>(w) / 2.0) shift_x -= static_cast<double>(w);

    const int upsample_factor = 100;
    const double step = 1.0 / static_cast<double>(upsample_factor);
    double best_mse = std::numeric_limits<double>::infinity();
    double best_sy = shift_y;
    double best_sx = shift_x;
    for (int iy = -upsample_factor; iy <= upsample_factor; ++iy) {
        for (int ix = -upsample_factor; ix <= upsample_factor; ++ix) {
            const double sy = shift_y + static_cast<double>(iy) * step;
            const double sx = shift_x + static_cast<double>(ix) * step;
            const double mse = mse_for_shift(image, offset_fft, sy, sx);
            if (mse < best_mse) {
                best_mse = mse;
                best_sy = sy;
                best_sx = sx;
            }
        }
    }

    const auto back_fft = fourier_shift(offset_fft, h, w, best_sy, best_sx);
#if defined(WSVT_HAS_FFTW)
    auto image_back = ifft2_real_fftw(back_fft, h, w);
#else
    auto image_back = ifft2_real_naive(back_fft, h, w);
#endif

    double src_power = 0.0;
    double dst_power = 0.0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const double a = static_cast<double>(image.data()[i]);
        const double b = static_cast<double>(offset_image.data()[i]);
        src_power += a * a;
        dst_power += b * b;
    }
    const double norm = std::sqrt(std::max(1e-30, src_power * dst_power));
    const double ccmax = best_abs;
    double error = 1.0;
    if (norm > 0.0) {
        const double ratio = std::min(1.0, std::max(0.0, (ccmax * ccmax) / (norm * norm)));
        error = std::sqrt(std::max(0.0, 1.0 - ratio));
    }
    const double diffphase = std::arg(cc[aidx2(peak_y, peak_x, w)]);

    std::cout << "shift dist: [" << best_sy << ", " << best_sx
              << "], alignment error: " << error
              << " and phase difference: " << diffphase
              << std::endl;

    return ImageAlignResult{{best_sy, best_sx}, error, diffphase, std::move(image_back)};
}

StackAlignResult stack_image_align(
    std::vector<float>& ref_stack,
    const float* img_first_frame, std::size_t /*img_stride*/,
    std::size_t ch, std::size_t h, std::size_t w) {
    if (ref_stack.size() != ch * h * w) {
        throw std::invalid_argument("stack_image_align: ref_stack size mismatch");
    }
#if defined(WSVT_HAS_FFTW)
    init_fftw_threads();
#endif

    ImageView2D<const float> img_view{img_first_frame, Shape2D{h, w}};
    ImageView2D<const float> ref_first{ref_stack.data(), Shape2D{h, w}};

    const auto result = image_align(img_view, ref_first);
    const double shift_y = result.shift[0];
    const double shift_x = result.shift[1];

    // Apply fourier_shift to every frame in ref_stack
    for (std::size_t i = 0; i < ch; ++i) {
        ImageView2D<const float> frame{
            ref_stack.data() + i * h * w, Shape2D{h, w}};
#if defined(WSVT_HAS_FFTW)
        auto spectrum = fft2_real_fftw(frame);
#else
        auto spectrum = fft2_real_naive(frame);
#endif
        auto shifted = fourier_shift(spectrum, h, w, shift_y, shift_x);
#if defined(WSVT_HAS_FFTW)
        auto back = ifft2_real_fftw(shifted, h, w);
#else
        auto back = ifft2_real_naive(shifted, h, w);
#endif
        std::memcpy(ref_stack.data() + i * h * w, back.data(), h * w * sizeof(float));
    }

    return StackAlignResult{{shift_y, shift_x}, result.error, result.diffphase};
}

}
