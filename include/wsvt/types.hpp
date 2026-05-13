#pragma once

#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

#if defined(WSVT_HAS_FFTW)
#include <fftw3.h>
#endif

namespace wsvt {

struct Shape2D {
    std::size_t h{0};
    std::size_t w{0};

    constexpr std::size_t size() const noexcept { return h * w; }
    constexpr bool operator==(const Shape2D&) const = default;
};

struct Shape3D {
    std::size_t d0{0};
    std::size_t d1{0};
    std::size_t d2{0};

    constexpr std::size_t size() const noexcept { return d0 * d1 * d2; }
    constexpr bool operator==(const Shape3D&) const = default;
};

enum class Layout { CHW, HWD };

#if defined(WSVT_HAS_FFTW)

namespace detail {

template<typename T>
inline constexpr bool fftw_is_single_v =
    std::is_same_v<T, float> || std::is_same_v<T, fftwf_complex>;

template<typename T>
inline constexpr bool fftw_is_double_v =
    std::is_same_v<T, double> || std::is_same_v<T, fftw_complex>;

template<typename T>
inline constexpr bool fftw_supported_v = fftw_is_single_v<T> || fftw_is_double_v<T>;

template<typename T>
T* fftw_alloc(std::size_t n) {
    static_assert(fftw_supported_v<T>,
        "FFTWBuffer<T>: T must be float, double, fftwf_complex, or fftw_complex");
    if constexpr (std::is_same_v<T, float>)              return fftwf_alloc_real(n);
    else if constexpr (std::is_same_v<T, double>)        return fftw_alloc_real(n);
    else if constexpr (std::is_same_v<T, fftwf_complex>) return fftwf_alloc_complex(n);
    else                                                 return fftw_alloc_complex(n);
}

template<typename T>
void fftw_dealloc(T* p) noexcept {
    if constexpr (fftw_is_single_v<T>) fftwf_free(p);
    else                               fftw_free(p);
}

}  // namespace detail

// RAII wrapper around fftw_alloc_* / fftw_free (and single-precision variants).
// Supported T: float, double, fftwf_complex, fftw_complex.
template<typename T>
class FFTWBuffer {
    static_assert(detail::fftw_supported_v<T>,
        "FFTWBuffer<T>: T must be float, double, fftwf_complex, or fftw_complex");

    T* ptr_{nullptr};
    std::size_t n_{0};

public:
    FFTWBuffer() = default;

    explicit FFTWBuffer(std::size_t n) : ptr_(detail::fftw_alloc<T>(n)), n_(n) {}

    ~FFTWBuffer() { if (ptr_) detail::fftw_dealloc(ptr_); }

    FFTWBuffer(const FFTWBuffer&) = delete;
    FFTWBuffer& operator=(const FFTWBuffer&) = delete;

    FFTWBuffer(FFTWBuffer&& other) noexcept
        : ptr_(other.ptr_), n_(other.n_) {
        other.ptr_ = nullptr;
        other.n_ = 0;
    }

    FFTWBuffer& operator=(FFTWBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) detail::fftw_dealloc(ptr_);
            ptr_ = other.ptr_;
            n_ = other.n_;
            other.ptr_ = nullptr;
            other.n_ = 0;
        }
        return *this;
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return n_; }

    // span() only makes sense for real scalar types — fftw_complex is a
    // 2-element array, not a std::span-friendly type
    auto span() noexcept requires std::is_arithmetic_v<T> { return std::span<T>{ptr_, n_}; }
    auto span() const noexcept requires std::is_arithmetic_v<T> { return std::span<const T>{ptr_, n_}; }
};

// RAII wrapper for fftwf_plan / fftw_plan. Template-parameterized on precision
// so both single- and double-precision plans share the same code.
template<bool SinglePrecision = true>
class FFTWPlanT {
    using plan_t = std::conditional_t<SinglePrecision, fftwf_plan, fftw_plan>;
    plan_t plan_{nullptr};

    static void destroy(plan_t p) noexcept {
        if constexpr (SinglePrecision) fftwf_destroy_plan(p);
        else                           fftw_destroy_plan(p);
    }
    static void run(plan_t p) noexcept {
        if constexpr (SinglePrecision) fftwf_execute(p);
        else                           fftw_execute(p);
    }

public:
    FFTWPlanT() = default;
    explicit FFTWPlanT(plan_t p) noexcept : plan_(p) {}

    ~FFTWPlanT() { if (plan_) destroy(plan_); }

    FFTWPlanT(const FFTWPlanT&) = delete;
    FFTWPlanT& operator=(const FFTWPlanT&) = delete;

    FFTWPlanT(FFTWPlanT&& other) noexcept : plan_(other.plan_) { other.plan_ = nullptr; }

    FFTWPlanT& operator=(FFTWPlanT&& other) noexcept {
        if (this != &other) {
            if (plan_) destroy(plan_);
            plan_ = other.plan_;
            other.plan_ = nullptr;
        }
        return *this;
    }

    void execute() const { run(plan_); }
    plan_t get() const noexcept { return plan_; }
    explicit operator bool() const noexcept { return plan_ != nullptr; }
};

using FFTWPlan  = FFTWPlanT<true>;   // single precision (fftwf_plan)
using FFTWPlanD = FFTWPlanT<false>;  // double precision (fftw_plan)

#endif  // WSVT_HAS_FFTW

}  // namespace wsvt
