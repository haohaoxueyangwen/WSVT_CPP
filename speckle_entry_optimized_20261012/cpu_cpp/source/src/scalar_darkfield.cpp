#include "wsvt/scalar_darkfield.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace wsvt {
namespace {

double quiet_nan() noexcept {
    return std::numeric_limits<double>::quiet_NaN();
}

bool is_finite_value(double value) noexcept {
    return std::isfinite(value);
}

std::size_t checked_product(std::size_t first, std::size_t second) {
    if (first != 0U && second > std::numeric_limits<std::size_t>::max() / first) {
        throw std::overflow_error("scalar dark-field dimensions overflow");
    }
    return first * second;
}

bool sample_nearest(
    const float* image,
    std::size_t height,
    std::size_t width,
    double y,
    double x,
    double& value) {
    if (!is_finite_value(y) || !is_finite_value(x)) {
        return false;
    }
    const auto yi = static_cast<std::ptrdiff_t>(std::nearbyint(y));
    const auto xi = static_cast<std::ptrdiff_t>(std::nearbyint(x));
    if (yi < 0 || xi < 0) {
        return false;
    }
    const auto uy = static_cast<std::size_t>(yi);
    const auto ux = static_cast<std::size_t>(xi);
    if (uy >= height || ux >= width) {
        return false;
    }
    value = static_cast<double>(image[uy * width + ux]);
    return is_finite_value(value);
}

bool sample_bilinear(
    const float* image,
    std::size_t height,
    std::size_t width,
    double y,
    double x,
    double& value) {
    if (!is_finite_value(y) || !is_finite_value(x)) {
        return false;
    }
    const double y0f = std::floor(y);
    const double x0f = std::floor(x);
    const double wy = y - y0f;
    const double wx = x - x0f;
    const auto y0 = static_cast<std::ptrdiff_t>(y0f);
    const auto x0 = static_cast<std::ptrdiff_t>(x0f);
    const bool need_y1 = wy > 0.0;
    const bool need_x1 = wx > 0.0;
    if (y0 < 0 || x0 < 0) {
        return false;
    }
    const auto uy0 = static_cast<std::size_t>(y0);
    const auto ux0 = static_cast<std::size_t>(x0);
    if (uy0 >= height || ux0 >= width) {
        return false;
    }
    if (need_y1 && uy0 + 1U >= height) {
        return false;
    }
    if (need_x1 && ux0 + 1U >= width) {
        return false;
    }
    const std::size_t uy1 = need_y1 ? uy0 + 1U : uy0;
    const std::size_t ux1 = need_x1 ? ux0 + 1U : ux0;
    const double v00 = static_cast<double>(image[uy0 * width + ux0]);
    const double v01 = static_cast<double>(image[uy0 * width + ux1]);
    const double v10 = static_cast<double>(image[uy1 * width + ux0]);
    const double v11 = static_cast<double>(image[uy1 * width + ux1]);
    if (!is_finite_value(v00) || !is_finite_value(v01) ||
        !is_finite_value(v10) || !is_finite_value(v11)) {
        return false;
    }
    value =
        (1.0 - wy) * (1.0 - wx) * v00 +
        (1.0 - wy) * wx * v01 +
        wy * (1.0 - wx) * v10 +
        wy * wx * v11;
    return is_finite_value(value);
}

void fill_invalid_pixel(ScalarDarkfieldMaps& maps, std::size_t pixel) {
    maps.T[pixel] = quiet_nan();
    maps.D[pixel] = quiet_nan();
    maps.a[pixel] = quiet_nan();
    maps.b[pixel] = quiet_nan();
    maps.cost[pixel] = quiet_nan();
    maps.delta[pixel] = quiet_nan();
    maps.weight_sum[pixel] = 0.0;
    maps.lnD[pixel] = quiet_nan();
    maps.numerical_valid[pixel] = 0;
    maps.physical_valid[pixel] = 0;
    maps.constraint_active[pixel] = 0;
}

}  // namespace

std::vector<float> scalar_darkfield_as_f32(const std::vector<double>& values) {
    std::vector<float> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = static_cast<float>(values[i]);
    }
    return out;
}

std::vector<float> scalar_darkfield_mask_as_f32(
    const std::vector<std::uint8_t>& values) {
    std::vector<float> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = values[i] != 0U ? 1.0f : 0.0f;
    }
    return out;
}

ScalarDarkfieldF32 pack_scalar_darkfield_f32(const ScalarDarkfieldMaps& maps) {
    ScalarDarkfieldF32 packed;
    packed.T = scalar_darkfield_as_f32(maps.T);
    packed.D = scalar_darkfield_as_f32(maps.D);
    packed.a = scalar_darkfield_as_f32(maps.a);
    packed.b = scalar_darkfield_as_f32(maps.b);
    packed.cost = scalar_darkfield_as_f32(maps.cost);
    packed.delta = scalar_darkfield_as_f32(maps.delta);
    packed.weight_sum = scalar_darkfield_as_f32(maps.weight_sum);
    packed.lnD = scalar_darkfield_as_f32(maps.lnD);
    packed.numerical_valid = scalar_darkfield_mask_as_f32(maps.numerical_valid);
    packed.physical_valid = scalar_darkfield_mask_as_f32(maps.physical_valid);
    packed.numerical_valid_pixel_count = maps.numerical_valid_pixel_count;
    packed.physical_valid_pixel_count = maps.physical_valid_pixel_count;
    return packed;
}

ScalarDarkfieldMaps fit_scalar_darkfield_field(
    std::span<const float> sample_stack,
    std::span<const float> reference_stack,
    std::size_t frames,
    std::size_t raw_height,
    std::size_t raw_width,
    std::span<const float> displace_y,
    std::span<const float> displace_x,
    std::size_t output_height,
    std::size_t output_width,
    double origin_y,
    double origin_x,
    const ScalarDarkfieldConfig& config) {
    if (!config.enabled) {
        return {};
    }
    if (config.relative_delta_tolerance < 0.0 ||
        config.transmission_epsilon < 0.0 ||
        !is_finite_value(config.relative_delta_tolerance) ||
        !is_finite_value(config.transmission_epsilon)) {
        throw std::invalid_argument(
            "scalar dark-field tolerances must be finite and non-negative");
    }
    if (frames == 0U || raw_height == 0U || raw_width == 0U) {
        throw std::invalid_argument("scalar dark-field requires a non-empty stack");
    }
    const std::size_t plane = checked_product(raw_height, raw_width);
    const std::size_t stack = checked_product(frames, plane);
    if (sample_stack.size() != stack || reference_stack.size() != stack) {
        throw std::invalid_argument("scalar dark-field stack size mismatch");
    }
    const std::size_t pixels = checked_product(output_height, output_width);
    if (displace_y.size() != pixels || displace_x.size() != pixels) {
        throw std::invalid_argument("scalar dark-field displacement size mismatch");
    }
    if (!is_finite_value(origin_y) || !is_finite_value(origin_x)) {
        throw std::invalid_argument("scalar dark-field origin must be finite");
    }

    const std::vector<double> window =
        normalized_hamming_window_2d(config.analysis_radius);
    const std::size_t width = config.analysis_radius * 2U + 1U;
    const auto radius = static_cast<std::ptrdiff_t>(config.analysis_radius);

    ScalarDarkfieldMaps maps;
    maps.height = output_height;
    maps.width = output_width;
    maps.T.assign(pixels, quiet_nan());
    maps.D.assign(pixels, quiet_nan());
    maps.a.assign(pixels, quiet_nan());
    maps.b.assign(pixels, quiet_nan());
    maps.cost.assign(pixels, quiet_nan());
    maps.delta.assign(pixels, quiet_nan());
    maps.weight_sum.assign(pixels, 0.0);
    maps.lnD.assign(pixels, quiet_nan());
    maps.numerical_valid.assign(pixels, 0);
    maps.physical_valid.assign(pixels, 0);
    maps.constraint_active.assign(pixels, 0);

    long long numerical_count = 0;
    long long physical_count = 0;
#pragma omp parallel
    {
        std::vector<double> s_store(width * width);
        std::vector<double> r_store(width * width);
        std::vector<double> w_store(width * width);
#pragma omp for schedule(static) reduction(+ : numerical_count, physical_count)
        for (std::ptrdiff_t linear = 0; linear < static_cast<std::ptrdiff_t>(pixels);
             ++linear) {
            const auto pixel = static_cast<std::size_t>(linear);
            const std::size_t iy = pixel / output_width;
            const std::size_t ix = pixel % output_width;
            const double uy = static_cast<double>(displace_y[pixel]);
            const double ux = static_cast<double>(displace_x[pixel]);
            if (!is_finite_value(uy) || !is_finite_value(ux)) {
                fill_invalid_pixel(maps, pixel);
                continue;
            }
            const double cy = origin_y + static_cast<double>(iy);
            const double cx = origin_x + static_cast<double>(ix);

            UmpaSufficientStatistics statistics;
            bool any_weight = false;
            for (std::size_t frame = 0; frame < frames; ++frame) {
                const float* sample_img = sample_stack.data() + frame * plane;
                const float* reference_img = reference_stack.data() + frame * plane;
                double r_acc = 0.0;
                double w_acc = 0.0;
                std::size_t n_obs = 0;
                for (std::ptrdiff_t oy = -radius; oy <= radius; ++oy) {
                    for (std::ptrdiff_t ox = -radius; ox <= radius; ++ox) {
                        const std::size_t wy =
                            static_cast<std::size_t>(oy + radius);
                        const std::size_t wx =
                            static_cast<std::size_t>(ox + radius);
                        const double wt = window[wy * width + wx];
                        double s = 0.0;
                        double r = 0.0;
                        const bool ok =
                            wt > 0.0 &&
                            sample_nearest(
                                sample_img, raw_height, raw_width,
                                cy + static_cast<double>(oy),
                                cx + static_cast<double>(ox), s) &&
                            sample_bilinear(
                                reference_img, raw_height, raw_width,
                                cy + static_cast<double>(oy) - uy,
                                cx + static_cast<double>(ox) - ux, r);
                        const double wpix = ok ? wt : 0.0;
                        s_store[n_obs] = ok ? s : 0.0;
                        r_store[n_obs] = ok ? r : 0.0;
                        w_store[n_obs] = wpix;
                        if (ok) {
                            r_acc += wpix * r;
                            w_acc += wpix;
                        }
                        ++n_obs;
                    }
                }
                if (!(w_acc > 0.0)) {
                    continue;
                }
                const double mu = r_acc / w_acc;
                for (std::size_t obs = 0; obs < n_obs; ++obs) {
                    const double wpix = w_store[obs];
                    if (!(wpix > 0.0)) {
                        continue;
                    }
                    const double s = s_store[obs];
                    const double r = r_store[obs];
                    statistics.l1 += wpix * s * s;
                    statistics.l2 += wpix * mu * mu;
                    statistics.l3 += wpix * r * r;
                    statistics.l4 += wpix * mu * s;
                    statistics.l5 += wpix * r * s;
                    statistics.l6 += wpix * r * mu;
                    statistics.weight_sum += wpix;
                    ++statistics.observation_count;
                    any_weight = true;
                }
            }
            if (!any_weight || !(statistics.weight_sum > 0.0)) {
                fill_invalid_pixel(maps, pixel);
                continue;
            }
            const UmpaPhysicalFit fit = solve_umpa_physical_fit(
                statistics,
                config.relative_delta_tolerance,
                config.transmission_epsilon);
            maps.delta[pixel] = fit.delta;
            maps.weight_sum[pixel] = statistics.weight_sum;
            maps.constraint_active[pixel] = 0;
            if (!fit.numerical_valid) {
                maps.a[pixel] = quiet_nan();
                maps.b[pixel] = quiet_nan();
                maps.T[pixel] = quiet_nan();
                maps.D[pixel] = quiet_nan();
                maps.cost[pixel] = quiet_nan();
                maps.lnD[pixel] = quiet_nan();
                maps.numerical_valid[pixel] = 0;
                maps.physical_valid[pixel] = 0;
                continue;
            }
            maps.a[pixel] = fit.alpha;
            maps.b[pixel] = fit.beta;
            maps.T[pixel] = fit.transmission;
            maps.D[pixel] = fit.visibility;
            maps.cost[pixel] = fit.cost;
            maps.numerical_valid[pixel] = 1;
            maps.physical_valid[pixel] = fit.physical_valid ? 1 : 0;
            ++numerical_count;
            if (fit.physical_valid) {
                ++physical_count;
                if (fit.visibility > 1.0e-12 && fit.visibility <= 1.0) {
                    maps.lnD[pixel] = -std::log(fit.visibility);
                }
            }
        }
    }
    maps.numerical_valid_pixel_count = static_cast<std::uint64_t>(numerical_count);
    maps.physical_valid_pixel_count = static_cast<std::uint64_t>(physical_count);
    return maps;
}

}  // namespace wsvt
