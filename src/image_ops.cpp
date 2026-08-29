#include "wsvt/image_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace wsvt {

namespace {

inline long long round_even_ll(double v) {
    return static_cast<long long>(std::nearbyint(v));
}

std::vector<double> not_a_knot_second_derivatives(std::span<const double> values) {
    const std::size_t n = values.size();
    std::vector<double> second(n, 0.0);
    if (n < 4) {
        return second;
    }

    // Unknowns are M[1]..M[n-2]. The first and last unknowns follow directly
    // from the uniform-grid not-a-knot equations; interior rows are the usual
    // cubic-spline tridiagonal system.
    const std::size_t count = n - 2;
    std::vector<double> lower(count, 0.0);
    std::vector<double> diagonal(count, 4.0);
    std::vector<double> upper(count, 0.0);
    std::vector<double> rhs(count, 0.0);
    diagonal.front() = 1.0;
    rhs.front() = values[2] - 2.0 * values[1] + values[0];
    diagonal.back() = 1.0;
    rhs.back() = values[n - 1] - 2.0 * values[n - 2] + values[n - 3];
    for (std::size_t row = 1; row + 1 < count; ++row) {
        lower[row] = 1.0;
        upper[row] = 1.0;
        const std::size_t i = row + 1;
        rhs[row] = 6.0 * (values[i + 1] - 2.0 * values[i] + values[i - 1]);
    }

    for (std::size_t row = 1; row < count; ++row) {
        const double factor = lower[row] / diagonal[row - 1];
        diagonal[row] -= factor * upper[row - 1];
        rhs[row] -= factor * rhs[row - 1];
    }
    std::vector<double> solution(count, 0.0);
    solution.back() = rhs.back() / diagonal.back();
    for (std::size_t row = count - 1; row-- > 0;) {
        solution[row] =
            (rhs[row] - upper[row] * solution[row + 1]) / diagonal[row];
    }
    for (std::size_t row = 0; row < count; ++row) {
        second[row + 1] = solution[row];
    }
    second[0] = 2.0 * second[1] - second[2];
    second[n - 1] = 2.0 * second[n - 2] - second[n - 3];
    return second;
}

double eval_uniform_cubic_spline(
    std::span<const double> values,
    std::span<const double> second,
    double coordinate) {
    const std::size_t n = values.size();
    if (n == 1 || coordinate <= 0.0) return values.front();
    if (coordinate >= static_cast<double>(n - 1)) return values.back();
    const std::size_t left = static_cast<std::size_t>(std::floor(coordinate));
    const double t = coordinate - static_cast<double>(left);
    const double one_minus_t = 1.0 - t;
    return second[left] * one_minus_t * one_minus_t * one_minus_t / 6.0 +
           second[left + 1] * t * t * t / 6.0 +
           (values[left] - second[left] / 6.0) * one_minus_t +
           (values[left + 1] - second[left + 1] / 6.0) * t;
}

std::vector<double> resample_spline_1d(
    std::span<const double> values, std::size_t output_size) {
    std::vector<double> output(output_size, 0.0);
    if (output_size == 0) return output;
    if (values.empty()) {
        throw std::invalid_argument("spline input must not be empty");
    }
    if (output_size == 1) {
        output[0] = values.front();
        return output;
    }
    const auto second = not_a_knot_second_derivatives(values);
    const double scale = static_cast<double>(values.size() - 1) /
                         static_cast<double>(output_size - 1);
    for (std::size_t i = 0; i < output_size; ++i) {
        output[i] = eval_uniform_cubic_spline(
            values, second, scale * static_cast<double>(i));
    }
    return output;
}

}

Image2D<float> image_roi(ImageView2D<const float> img, std::size_t m) {
    const Shape2D s = img.shape();
    if (m == 0 || m > std::min(s.h, s.w)) {
        // Python 语义：m 越界时原样返回
        Image2D<float> out(s);
        std::memcpy(out.data(), img.data(), s.size() * sizeof(float));
        return out;
    }

    const long long y0 = round_even_ll(static_cast<double>(s.h) / 2.0) -
                         round_even_ll(static_cast<double>(m) / 2.0);
    const long long x0 = round_even_ll(static_cast<double>(s.w) / 2.0) -
                         round_even_ll(static_cast<double>(m) / 2.0);

    Image2D<float> out(Shape2D{m, m});
    for (std::size_t y = 0; y < m; ++y) {
        std::memcpy(out.row(y),
                    &img(static_cast<std::size_t>(y0) + y,
                         static_cast<std::size_t>(x0)),
                    m * sizeof(float));
    }
    return out;
}

Tensor3D<float, Layout::CHW> image_roi(
    TensorView3D<const float, Layout::CHW> img, std::size_t m) {
    const Shape3D s = img.shape();  // d0=ch, d1=h, d2=w
    if (m == 0 || m > std::min(s.d1, s.d2)) {
        Tensor3D<float, Layout::CHW> out(s);
        std::memcpy(out.data(), img.data(), s.size() * sizeof(float));
        return out;
    }

    const long long y0 = round_even_ll(static_cast<double>(s.d1) / 2.0) -
                         round_even_ll(static_cast<double>(m) / 2.0);
    const long long x0 = round_even_ll(static_cast<double>(s.d2) / 2.0) -
                         round_even_ll(static_cast<double>(m) / 2.0);

    Tensor3D<float, Layout::CHW> out(Shape3D{s.d0, m, m});
    for (std::size_t c = 0; c < s.d0; ++c) {
        for (std::size_t y = 0; y < m; ++y) {
            std::memcpy(&out(c, y, 0),
                        &img(c, static_cast<std::size_t>(y0) + y,
                             static_cast<std::size_t>(x0)),
                        m * sizeof(float));
        }
    }
    return out;
}

Image2D<float> resample_rect_bivariate_spline(
    ImageView2D<const float> img, Shape2D output_shape) {
    const Shape2D input_shape = img.shape();
    if (input_shape.h == 0 || input_shape.w == 0) {
        throw std::invalid_argument("resample_rect_bivariate_spline input is empty");
    }
    if (input_shape.h < 4 || input_shape.w < 4) {
        throw std::invalid_argument(
            "RectBivariateSpline-compatible cubic interpolation requires both input dimensions >= 4");
    }
    Image2D<float> output(output_shape, 0.0f);
    if (output_shape.h == 0 || output_shape.w == 0) return output;

    std::vector<double> horizontal(input_shape.h * output_shape.w, 0.0);
    #pragma omp parallel for schedule(static)
    for (std::size_t y = 0; y < input_shape.h; ++y) {
        std::vector<double> row(input_shape.w);
        for (std::size_t x = 0; x < input_shape.w; ++x) row[x] = img(y, x);
        const auto interpolated = resample_spline_1d(row, output_shape.w);
        std::copy(interpolated.begin(), interpolated.end(),
                  horizontal.begin() + static_cast<std::ptrdiff_t>(y * output_shape.w));
    }

    #pragma omp parallel for schedule(static)
    for (std::size_t x = 0; x < output_shape.w; ++x) {
        std::vector<double> column(input_shape.h);
        for (std::size_t y = 0; y < input_shape.h; ++y) {
            column[y] = horizontal[y * output_shape.w + x];
        }
        const auto interpolated = resample_spline_1d(column, output_shape.h);
        for (std::size_t y = 0; y < output_shape.h; ++y) {
            output(y, x) = static_cast<float>(interpolated[y]);
        }
    }
    return output;
}

}
