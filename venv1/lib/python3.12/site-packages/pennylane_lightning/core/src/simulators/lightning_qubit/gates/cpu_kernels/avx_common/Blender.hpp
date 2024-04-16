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
 * Defines utility functions for all AVX blend functions
 */
#pragma once
#include <immintrin.h>
#include <limits>

#include "AVXUtil.hpp"
#include "Macros.hpp"

namespace Pennylane::LightningQubit::Gates::AVXCommon {
template <typename PrecisionT, size_t packed_size> struct CompileMask {
    static_assert(sizeof(PrecisionT) == std::numeric_limits<size_t>::max(),
                  "Unsupported type and/or packed size.");
};

template <> struct CompileMask<float, 8> {
    // AVX2 with float
    constexpr static auto create(const std::array<bool, 8> &mask) -> int {
        int imm8 = 0;
        for (uint8_t i = 0; i < 8; i++) {
            imm8 |= int(mask[i]) << i; // NOLINT(hicpp-signed-bitwise)
        }
        return imm8;
    }
};
template <> struct CompileMask<double, 4> {
    // AVX2 with double
    constexpr static auto create(const std::array<bool, 4> &mask) {
        int imm8 = 0;
        for (uint8_t i = 0; i < 4; i++) {
            imm8 |= int(mask[i]) << i; // NOLINT(hicpp-signed-bitwise)
        }
        return imm8;
    }
};
template <> struct CompileMask<float, 16> {
    // AVX512 with float
    constexpr static auto create(const std::array<bool, 16> &mask) {
        __mmask16 m = 0;
        for (uint8_t i = 0; i < 16; i++) {
            m |= int(mask[i]) << i; // NOLINT(hicpp-signed-bitwise)
        }
        return m;
    }
};
template <> struct CompileMask<double, 8> {
    // AVX512 with double
    constexpr static auto create(const std::array<bool, 8> &mask) {
        __mmask8 m = 0;
        for (uint8_t i = 0; i < 8; i++) {
            m |= int(mask[i]) << i; // NOLINT(hicpp-signed-bitwise)
        }
        return m;
    }
};

constexpr int negate(int imm8) {
    return 0B11111111 ^ imm8; // NOLINT
}
constexpr __mmask8 negate(__mmask8 m) {
    return 0B11111111 ^ m; // NOLINT
}
constexpr __mmask16 negate(__mmask16 m) {
    return 0B1111'1111'1111'1111 ^ m; // NOLINT
}

template <typename PrecisionT, size_t packed_size>
constexpr static auto compileMask(const std::array<bool, packed_size> &mask) {
    return CompileMask<PrecisionT, packed_size>::create(mask);
}

template <int imm8>
PL_FORCE_INLINE __m256 blend(const __m256 &a, const __m256 &b) {
    return _mm256_blend_ps(a, b, imm8);
}
template <int imm8>
PL_FORCE_INLINE __m256d blend(const __m256d &a, const __m256d &b) {
    return _mm256_blend_pd(a, b, imm8);
}
template <__mmask16 k>
PL_FORCE_INLINE __m512 blend(const __m512 &a, const __m512 &b) {
    return _mm512_mask_blend_ps(k, a, b);
}
template <__mmask8 k>
PL_FORCE_INLINE __m512d blend(const __m512d &a, const __m512d &b) {
    return _mm512_mask_blend_pd(k, a, b);
}

} // namespace Pennylane::LightningQubit::Gates::AVXCommon
