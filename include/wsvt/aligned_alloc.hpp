#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>
#include <span>
#include <vector>

namespace wsvt {

template<typename T, std::size_t Alignment = 64>
struct AlignedAllocator {
    using value_type = T;

    template<typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    AlignedAllocator() noexcept = default;

    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        const std::size_t bytes = ((n * sizeof(T) + Alignment - 1) / Alignment) * Alignment;
        void* p = std::aligned_alloc(Alignment, bytes);
        if (!p) throw std::bad_alloc{};
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t) noexcept {
        std::free(p);
    }

    template<typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept { return true; }
};

template<typename T, std::size_t Alignment = 64>
using AlignedVector = std::vector<T, AlignedAllocator<T, Alignment>>;

// libstdc++ doesn't implicitly convert vector<T,CustomAlloc> to span<const T>
template<typename T, typename Alloc>
std::span<const T> as_span(const std::vector<T, Alloc>& v) noexcept {
    return {v.data(), v.size()};
}

template<typename T, typename Alloc>
std::span<T> as_span(std::vector<T, Alloc>& v) noexcept {
    return {v.data(), v.size()};
}

/// Uninitialized aligned buffer. Allocated via std::aligned_alloc (no
/// zero-init). Movable; not copyable. Use for scratch buffers that will
/// be fully overwritten before first read.
template<typename T, std::size_t Alignment = 64>
class AlignedBuffer {
public:
    AlignedBuffer() noexcept = default;

    explicit AlignedBuffer(std::size_t n) { resize(n); }

    ~AlignedBuffer() noexcept { std::free(data_); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            std::free(data_);
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void resize(std::size_t n) {
        std::free(data_);
        size_ = n;
        if (n == 0) { data_ = nullptr; return; }
        const std::size_t bytes = ((n * sizeof(T) + Alignment - 1) / Alignment) * Alignment;
        data_ = static_cast<T*>(std::aligned_alloc(Alignment, bytes));
        if (!data_) throw std::bad_alloc{};
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    [[nodiscard]] T* begin() noexcept { return data_; }
    [[nodiscard]] T* end() noexcept { return data_ + size_; }

    void clear() noexcept {
        std::free(data_);
        data_ = nullptr;
        size_ = 0;
    }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace wsvt
