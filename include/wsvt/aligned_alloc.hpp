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

}  // namespace wsvt
