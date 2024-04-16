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
 * Defines common AVX256 concept
 */
#pragma once
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <type_traits>

namespace Pennylane::LightningQubit::Gates::AVXCommon {
///@cond DEV
namespace Internal {
template <typename T> struct AVX2Intrinsic {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
};
template <> struct AVX2Intrinsic<float> {
    using Type = __m256;
};
template <> struct AVX2Intrinsic<double> {
    using Type = __m256d;
};
} // namespace Internal
///@endcond

template <typename T> struct AVX2Concept {
    using PrecisionT = T;
    using IntrinsicType = typename Internal::AVX2Intrinsic<PrecisionT>::Type;

    PL_FORCE_INLINE
    static auto load(const std::complex<PrecisionT> *p) -> IntrinsicType {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm256_load_ps(reinterpret_cast<const PrecisionT *>(p));
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm256_load_pd(reinterpret_cast<const PrecisionT *>(p));
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static auto loadu(const std::complex<PrecisionT> *p) -> IntrinsicType {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm256_loadu_ps(reinterpret_cast<const PrecisionT *>(p));
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm256_loadu_pd(reinterpret_cast<const PrecisionT *>(p));
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static auto loadu(PrecisionT *p) -> IntrinsicType {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm256_loadu_ps(p);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm256_loadu_pd(p);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static void store(std::complex<PrecisionT> *p, IntrinsicType value) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            _mm256_store_ps(reinterpret_cast<PrecisionT *>(p), value);
        } else if (std::is_same_v<PrecisionT, double>) {
            _mm256_store_pd(reinterpret_cast<PrecisionT *>(p), value);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static auto mul(IntrinsicType v0, IntrinsicType v1) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm256_mul_ps(v0, v1);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm256_mul_pd(v0, v1);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static auto add(IntrinsicType v0, IntrinsicType v1) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm256_add_ps(v0, v1);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm256_add_pd(v0, v1);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }
};
} // namespace Pennylane::LightningQubit::Gates::AVXCommon
