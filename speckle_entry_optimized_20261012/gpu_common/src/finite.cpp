#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

// Integer exponent test prevents fast-math assumptions from deleting checks.
// memcpy is alias-safe; the reduction vectorizes without early-exit branches.
// This uses one host thread, exactly as the original native validation did.
extern "C" bool speckle_all_finite(const float* input, std::size_t count) {
    static_assert(sizeof(float)==sizeof(std::uint32_t));
    static_assert(std::numeric_limits<float>::is_iec559);
    std::uint32_t invalid=0;
    for(std::size_t i=0;i<count;++i) {
        std::uint32_t bits;
        std::memcpy(&bits,input+i,sizeof(bits));
        invalid |= std::uint32_t((bits & 0x7f800000u)==0x7f800000u);
    }
    return invalid==0;
}
