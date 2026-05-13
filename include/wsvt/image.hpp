#pragma once

#include "wsvt/types.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace wsvt {

template<typename T = float>
class Image2D {
    std::vector<T> data_;
    Shape2D shape_{};
public:
    Image2D() = default;

    explicit Image2D(Shape2D s) : data_(s.size()), shape_(s) {}

    Image2D(Shape2D s, T init) : data_(s.size(), init), shape_(s) {}

    Image2D(std::vector<T>&& data, Shape2D s) : data_(std::move(data)), shape_(s) {
        if (data_.size() != shape_.size()) {
            throw std::invalid_argument("Image2D: data size does not match shape");
        }
    }

    T& operator()(std::size_t y, std::size_t x) noexcept {
        return data_[y * shape_.w + x];
    }
    const T& operator()(std::size_t y, std::size_t x) const noexcept {
        return data_[y * shape_.w + x];
    }

    T* row(std::size_t y) noexcept { return data_.data() + y * shape_.w; }
    const T* row(std::size_t y) const noexcept { return data_.data() + y * shape_.w; }

    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    std::span<T> flat() noexcept { return {data_.data(), data_.size()}; }
    std::span<const T> flat() const noexcept { return {data_.data(), data_.size()}; }

    Shape2D shape() const noexcept { return shape_; }
    std::size_t h() const noexcept { return shape_.h; }
    std::size_t w() const noexcept { return shape_.w; }
    std::size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }

    // 所有权转移（pybind11 返回 numpy 用）
    std::vector<T> take() && noexcept { return std::move(data_); }

    // 对 Phase 2-3 adapter 过渡期友好：给出底层 vector 的 const 引用
    const std::vector<T>& as_vector() const& noexcept { return data_; }
};

template<typename T = float, Layout L = Layout::CHW>
class Tensor3D {
    std::vector<T> data_;
    Shape3D shape_{};
public:
    Tensor3D() = default;

    explicit Tensor3D(Shape3D s) : data_(s.size()), shape_(s) {}

    Tensor3D(Shape3D s, T init) : data_(s.size(), init), shape_(s) {}

    Tensor3D(std::vector<T>&& data, Shape3D s) : data_(std::move(data)), shape_(s) {
        if (data_.size() != shape_.size()) {
            throw std::invalid_argument("Tensor3D: data size does not match shape");
        }
    }

    // Layout-aware access
    T& operator()(std::size_t c, std::size_t y, std::size_t x) noexcept
        requires (L == Layout::CHW)
    {
        return data_[(c * shape_.d1 + y) * shape_.d2 + x];
    }
    const T& operator()(std::size_t c, std::size_t y, std::size_t x) const noexcept
        requires (L == Layout::CHW)
    {
        return data_[(c * shape_.d1 + y) * shape_.d2 + x];
    }

    T& operator()(std::size_t y, std::size_t x, std::size_t d) noexcept
        requires (L == Layout::HWD)
    {
        return data_[(y * shape_.d1 + x) * shape_.d2 + d];
    }
    const T& operator()(std::size_t y, std::size_t x, std::size_t d) const noexcept
        requires (L == Layout::HWD)
    {
        return data_[(y * shape_.d1 + x) * shape_.d2 + d];
    }

    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    std::span<T> flat() noexcept { return {data_.data(), data_.size()}; }
    std::span<const T> flat() const noexcept { return {data_.data(), data_.size()}; }

    Shape3D shape() const noexcept { return shape_; }

    // Semantic accessors (depend on layout)
    std::size_t channels() const noexcept requires (L == Layout::CHW) { return shape_.d0; }
    std::size_t height()   const noexcept requires (L == Layout::CHW) { return shape_.d1; }
    std::size_t width()    const noexcept requires (L == Layout::CHW) { return shape_.d2; }

    std::size_t height()   const noexcept requires (L == Layout::HWD) { return shape_.d0; }
    std::size_t width()    const noexcept requires (L == Layout::HWD) { return shape_.d1; }
    std::size_t depth()    const noexcept requires (L == Layout::HWD) { return shape_.d2; }

    std::size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }

    std::vector<T> take() && noexcept { return std::move(data_); }
    const std::vector<T>& as_vector() const& noexcept { return data_; }
};

// Non-owning view, zero-copy parameter type (pybind11-friendly)
template<typename T>
class ImageView2D {
    T* data_{nullptr};
    Shape2D shape_{};
public:
    ImageView2D() = default;
    ImageView2D(T* data, Shape2D s) noexcept : data_(data), shape_(s) {}

    ImageView2D(Image2D<std::remove_const_t<T>>& img) noexcept
        requires (!std::is_const_v<T>)
        : data_(img.data()), shape_(img.shape()) {}

    ImageView2D(const Image2D<std::remove_const_t<T>>& img) noexcept
        requires std::is_const_v<T>
        : data_(img.data()), shape_(img.shape()) {}

    T& operator()(std::size_t y, std::size_t x) const noexcept {
        return data_[y * shape_.w + x];
    }

    T* data() const noexcept { return data_; }
    Shape2D shape() const noexcept { return shape_; }
    std::size_t h() const noexcept { return shape_.h; }
    std::size_t w() const noexcept { return shape_.w; }
    std::size_t size() const noexcept { return shape_.size(); }
    std::span<T> flat() const noexcept { return {data_, shape_.size()}; }
};

template<typename T, Layout L = Layout::CHW>
class TensorView3D {
    T* data_{nullptr};
    Shape3D shape_{};
public:
    TensorView3D() = default;
    TensorView3D(T* data, Shape3D s) noexcept : data_(data), shape_(s) {}

    TensorView3D(Tensor3D<std::remove_const_t<T>, L>& t) noexcept
        requires (!std::is_const_v<T>)
        : data_(t.data()), shape_(t.shape()) {}

    TensorView3D(const Tensor3D<std::remove_const_t<T>, L>& t) noexcept
        requires std::is_const_v<T>
        : data_(t.data()), shape_(t.shape()) {}

    T& operator()(std::size_t c, std::size_t y, std::size_t x) const noexcept
        requires (L == Layout::CHW)
    {
        return data_[(c * shape_.d1 + y) * shape_.d2 + x];
    }

    T& operator()(std::size_t y, std::size_t x, std::size_t d) const noexcept
        requires (L == Layout::HWD)
    {
        return data_[(y * shape_.d1 + x) * shape_.d2 + d];
    }

    T* data() const noexcept { return data_; }
    Shape3D shape() const noexcept { return shape_; }
    std::size_t size() const noexcept { return shape_.size(); }
    std::span<T> flat() const noexcept { return {data_, shape_.size()}; }
};

}  // namespace wsvt
