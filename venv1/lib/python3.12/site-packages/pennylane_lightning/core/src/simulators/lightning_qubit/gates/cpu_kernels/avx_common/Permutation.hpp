
// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file
 * Defines permutation of AVX intrinsics
 */
#pragma once
#include <array>
#include <cstdint>
#include <cstdlib>
#include <immintrin.h>
#include <limits>

#include "AVXUtil.hpp"

// Clang warns constexpr __m256i constructor, but it works.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Winvalid-constexpr"
#endif

namespace Pennylane::LightningQubit::Gates::AVXCommon::Permutation {
/**
 * @brief Maintain permutation related data in a compile time.
 */
template <typename PrecisionT, size_t packed_size> struct CompiledPermutation {
    // Cannot use unspecialized version
    static_assert(sizeof(PrecisionT) == std::numeric_limits<size_t>::max(),
                  "Unsupported data typed and packed size.");
};

template <> struct CompiledPermutation<float, 8> {
    using PrecisionT_ = float;
    constexpr static size_t packed_size_ = 8;

    const bool within_lane_;
    const int imm8_ = 0;
    __m256i permute256_ = {
        0,
    };

    constexpr CompiledPermutation(bool within_lane, uint8_t imm8)
        : within_lane_{within_lane}, imm8_{imm8} {}

    constexpr CompiledPermutation(bool within_lane, __m256i permute)
        : within_lane_{within_lane}, permute256_{permute} {}
};

template <> struct CompiledPermutation<double, 4> {
    using PrecisionT_ = double;
    constexpr static size_t packed_size_ = 4;

    const bool within_lane_;
    const int imm8_ = 0;
    __m256i permute256_ = {
        0,
    };

    constexpr CompiledPermutation(bool within_lane, uint8_t imm8)
        : within_lane_{within_lane}, imm8_{imm8} {}

    constexpr CompiledPermutation(bool within_lane, __m256i permute)
        : within_lane_{within_lane}, permute256_{permute} {}
};
template <> struct CompiledPermutation<float, 16> {
    using PrecisionT_ = float;
    constexpr static size_t packed_size_ = 16;

    const bool within_lane_;
    const int imm8_ = 0;
    __m512i permute512_ = {
        0,
    };

    constexpr CompiledPermutation(bool within_lane, uint8_t imm8)
        : within_lane_{within_lane}, imm8_{imm8} {}

    constexpr CompiledPermutation(bool within_lane, __m512i permute)
        : within_lane_{within_lane}, permute512_{permute} {}
};
template <> struct CompiledPermutation<double, 8> {
    using PrecisionT_ = float;
    constexpr static size_t packed_size_ = 8;

    const bool within_lane_;
    const int imm8_ = 0;
    __m512i permute512_ = {
        0,
    };

    constexpr CompiledPermutation(bool within_lane, uint8_t imm8)
        : within_lane_{within_lane}, imm8_{imm8} {}

    constexpr CompiledPermutation(bool within_lane, __m512i permute)
        : within_lane_{within_lane}, permute512_{permute} {}
};
/**
 * @brief Compute whether the given permutation acts only within 128bit lane.
 *
 * @tparam PrecisionT Floating point precision type
 * @tparam size Size of the permutation
 * @param permutation Permutation as an array
 */
template <typename PrecisionT, size_t size>
constexpr bool isWithinLane(const std::array<uint8_t, size> &permutation) {
    constexpr size_t size_within_lane = 16 / sizeof(PrecisionT);

    std::array<uint32_t, size_within_lane> lane = {
        0,
    };
    for (size_t i = 0; i < size_within_lane; i++) {
        lane[i] = permutation[i];
    }
    {
        auto lane2 = lane;
        std::sort(lane2.begin(), lane2.end());
        for (size_t i = 0; i < size_within_lane; i++) {
            if (lane2[i] != i) {
                return false;
            }
        }
    }

    for (size_t k = 0; k < permutation.size(); k += size_within_lane) {
        for (size_t idx = 0; idx < size_within_lane; idx++) {
            if (lane[idx] + k != permutation[idx + k]) {
                return false;
            }
        }
    }
    return true;
}

///@cond DEV
template <size_t size>
constexpr uint8_t
getPermutation2x(const std::array<uint8_t, size> &permutation) {
    uint8_t res = static_cast<uint8_t>(permutation[1] << 1U) | permutation[0];
    // NOLINTNEXTLINE(readability-magic-numbers, hicpp-signed-bitwise)
    return (res << 6U) | (res << 4U) | (res << 2U) | res;
}
template <size_t size>
constexpr uint8_t
getPermutation4x(const std::array<uint8_t, size> &permutation) {
    uint8_t res = 0;
    for (int idx = 3; idx >= 0; idx--) {
        res <<= 2U;
        res |= (permutation[idx]);
    }
    return res;
}
//@endcond

#ifdef PL_USE_AVX2
constexpr __m256i
getPermutation8x256i(const std::array<uint8_t, 8> &permutation) {
    // NOLINTBEGIN(readability-magic-numbers)
    return setr256i(permutation[0], permutation[1], permutation[2],
                    permutation[3], permutation[4], permutation[5],
                    permutation[6], permutation[7]);
    // NOLINTEND(readability-magic-numbers)
}
#endif
#ifdef PL_USE_AVX512F
// LCOV_EXCL_START
constexpr __m512i
getPermutation8x512i(const std::array<uint8_t, 8> &permutation) {
    // NOLINTBEGIN(readability-magic-numbers)
    return setr512i(permutation[0], permutation[1], permutation[2],
                    permutation[3], permutation[4], permutation[5],
                    permutation[6], permutation[7]);
    // NOLINTEND(readability-magic-numbers)
}
constexpr __m512i
getPermutation16x512i(const std::array<uint8_t, 16> &permutation) {
    // NOLINTBEGIN(readability-magic-numbers)
    return setr512i(
        permutation[0], permutation[1], permutation[2], permutation[3],
        permutation[4], permutation[5], permutation[6], permutation[7],
        permutation[8], permutation[9], permutation[10], permutation[11],
        permutation[12], permutation[13], permutation[14], permutation[15]);
    // NOLINTEND(readability-magic-numbers)
}
// LCOV_EXCL_STOP
#endif

template <typename PrecisionT, size_t packed_size>
constexpr auto compilePermutation(
    [[maybe_unused]] const std::array<uint8_t, packed_size> &permutation)
    -> CompiledPermutation<PrecisionT, packed_size> {
    // Raise a compile error when instantiated
    static_assert(sizeof(PrecisionT) == std::numeric_limits<size_t>::max(),
                  "Only specialized classes can be used");
};

#ifdef PL_USE_AVX2 // Specializations for AVX2 begin
template <>
constexpr auto
compilePermutation<float, 8>(const std::array<uint8_t, 8> &permutation)
    -> CompiledPermutation<float, 8> {
    bool within_lane = isWithinLane<float>(permutation);

    if (within_lane) {
        const auto imm8 = getPermutation4x(permutation);
        return {within_lane, imm8};
    } // else
    return {within_lane, getPermutation8x256i(permutation)};
}
template <>
constexpr auto
compilePermutation<double, 4>(const std::array<uint8_t, 4> &permutation)
    -> CompiledPermutation<double, 4> {
    bool within_lane = isWithinLane<double>(permutation);

    if (within_lane) {
        const auto permute_val = getPermutation2x(permutation);
        return {within_lane, permute_val};
    } // else
    return {within_lane, getPermutation4x(permutation)};
}
#endif // Specializations for AVX2 end

#ifdef PL_USE_AVX512F // Specializations for AVX512 begin
// LCOV_EXCL_START
template <>
constexpr auto
compilePermutation<float, 16>(const std::array<uint8_t, 16> &permutation)
    -> CompiledPermutation<float, 16> {
    bool within_lane = isWithinLane<float>(permutation);

    if (within_lane) {
        const auto imm8 = getPermutation4x(permutation);
        return {within_lane, imm8};
    } // else
    return {within_lane, getPermutation16x512i(permutation)};
}
template <>
constexpr auto
compilePermutation<double, 8>(const std::array<uint8_t, 8> &permutation)
    -> CompiledPermutation<double, 8> {
    bool within_lane = isWithinLane<double>(permutation);

    if (within_lane) {
        const auto permute_val = getPermutation2x(permutation);
        return {within_lane, permute_val};
    } // else
    return {within_lane, getPermutation8x512i(permutation)};
}
// LCOV_EXCL_STOP
#endif // Specializations for AVX512 end

template <size_t packed_size>
constexpr auto identity() -> std::array<uint8_t, packed_size> {
    std::array<uint8_t, packed_size> res{};
    for (uint8_t i = 0; i < packed_size; i++) {
        res[i] = i;
    }
    return res;
}

/**
 * @brief Create a permutation that flip a bit in rev_wire. This is same as
 * applying Pauli X gate to rev_wire.
 *
 * @tparam packed_size Number of elements in a packed type
 * @param perm Previous permutation
 * @param rev_wire Reverse wire
 */
template <size_t packed_size>
constexpr auto flip(const std::array<uint8_t, packed_size> &perm,
                    size_t rev_wire) -> std::array<uint8_t, packed_size> {
    std::array<uint8_t, packed_size> res{};

    for (size_t k = 0; k < packed_size / 2; k++) {
        res[2 * k + 0] = perm[2 * (k ^ (1U << rev_wire)) + 0];
        res[2 * k + 1] = perm[2 * (k ^ (1U << rev_wire)) + 1];
    }
    return res;
}

/**
 * @brief Create a permutation that swap real and imaginary parts of a packed
 * array.
 *
 * @tparam packed_size Number of elements in a packed type
 * @param perm Previous permutation
 */
template <size_t packed_size>
constexpr auto swapRealImag(const std::array<uint8_t, packed_size> &perm)
    -> std::array<uint8_t, packed_size> {
    std::array<uint8_t, packed_size> res{};

    for (uint8_t k = 0; k < packed_size / 2; k++) {
        res[2 * k + 0] = perm[2 * k + 1];
        res[2 * k + 1] = perm[2 * k + 0];
    }
    return res;
}

template <const auto &compiled_permutation>
PL_FORCE_INLINE auto permute(const __m256 &v) {
    static_assert(compiled_permutation.packed_size_ == 8);

    if constexpr (compiled_permutation.within_lane_) {
        constexpr static auto imm8 = compiled_permutation.imm8_;
        return _mm256_permute_ps(v, imm8);
    } else {
        constexpr auto permute256 = compiled_permutation.permute256_;
        return _mm256_permutevar8x32_ps(v, permute256);
    }
}

template <const auto &compiled_permutation>
PL_FORCE_INLINE auto permute(const __m256d &v) {
    static_assert(compiled_permutation.packed_size_ == 4);

    constexpr static auto imm8 = compiled_permutation.imm8_;
    if constexpr (compiled_permutation.within_lane_) {
        constexpr static int imm8_trunc = imm8 % 16; // to suppress error
        return _mm256_permute_pd(v, imm8_trunc);
    } else {
        return _mm256_permute4x64_pd(v, imm8);
    }
}

template <const auto &compiled_permutation>
PL_FORCE_INLINE auto permute(const __m512 &v) {
    static_assert(compiled_permutation.packed_size_ == 16);

    if constexpr (compiled_permutation.within_lane_) {
        constexpr static auto imm8 = compiled_permutation.imm8_;
        return _mm512_permute_ps(v, imm8);
    } else {
        constexpr auto permute512 = compiled_permutation.permute512_;
        return _mm512_permutexvar_ps(permute512, v);
    }
}

template <const auto &compiled_permutation>
PL_FORCE_INLINE auto permute(const __m512d &v) {
    static_assert(compiled_permutation.packed_size_ == 8);

    if constexpr (compiled_permutation.within_lane_) {
        constexpr static auto imm8 = compiled_permutation.imm8_;
        return _mm512_permute_pd(v, imm8);
    } else {
        constexpr auto permute512 = compiled_permutation.permute512_;
        return _mm512_permutexvar_pd(permute512, v);
    }
}

template <const auto &perm, int imm8>
PL_FORCE_INLINE __m256 maskPermute(const __m256 &src, const __m256 &a) {
    return _mm256_blend_ps(src, permute<perm>(a), imm8);
}
template <const auto &perm, int imm8>
PL_FORCE_INLINE __m256d maskPermute(const __m256d &src, const __m256d &a) {
    return _mm256_blend_pd(src, permute<perm>(a), imm8);
}
template <const auto &perm, __mmask16 k>
PL_FORCE_INLINE __m512 maskPermute(const __m512 &src, const __m512 &a) {
    if constexpr (perm.within_lane_) {
        constexpr static auto imm8 = perm.imm8_;
        return _mm512_mask_permute_ps(src, k, a, imm8);
    } else {
        return _mm512_mask_permutexvar_ps(src, k, perm.permute512_, a);
    }
}
template <const auto &perm, __mmask8 k>
PL_FORCE_INLINE __m512d maskPermute(const __m512d &src, const __m512d &a) {
    if constexpr (perm.within_lane_) {
        constexpr static auto imm8 = perm.imm8_;
        return _mm512_mask_permute_pd(src, k, a, imm8);
    } else {
        return _mm512_mask_permutexvar_pd(src, k, perm.permute512_, a);
    }
}
} // namespace Pennylane::LightningQubit::Gates::AVXCommon::Permutation
