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
 * Defines common AVX512 concept
 */
#pragma once
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Macros.hpp"
#include "Util.hpp"

#include <immintrin.h>

#include <type_traits>

namespace Pennylane::LightningQubit::Gates::AVXCommon {
///@cond DEV
namespace Internal {
template <typename T> struct AVX512Intrinsic {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
};
template <> struct AVX512Intrinsic<float> {
    using Type = __m512;
};
template <> struct AVX512Intrinsic<double> {
    using Type = __m512d;
};
} // namespace Internal
///@endcond

template <typename T> struct AVX512Concept {
    using PrecisionT = T;
    using IntrinsicType = typename Internal::AVX512Intrinsic<PrecisionT>::Type;

    PL_FORCE_INLINE
    static auto load(std::complex<PrecisionT> *p) -> IntrinsicType {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm512_load_ps(p);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm512_load_pd(p);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static auto loadu(std::complex<PrecisionT> *p) -> IntrinsicType {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm512_loadu_ps(p);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm512_loadu_pd(p);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static auto loadu(PrecisionT *p) -> IntrinsicType {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm512_loadu_ps(p);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm512_loadu_pd(p);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static void store(std::complex<PrecisionT> *p, IntrinsicType value) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            _mm512_store_ps(p, value);
        } else if (std::is_same_v<PrecisionT, double>) {
            _mm512_store_pd(p, value);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static auto mul(IntrinsicType v0, IntrinsicType v1) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm512_mul_ps(v0, v1);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm512_mul_pd(v0, v1);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }

    PL_FORCE_INLINE
    static auto add(IntrinsicType v0, IntrinsicType v1) {
        if constexpr (std::is_same_v<PrecisionT, float>) {
            return _mm512_add_ps(v0, v1);
        } else if (std::is_same_v<PrecisionT, double>) {
            return _mm512_add_pd(v0, v1);
        } else {
            static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>);
        }
    }
};

} // namespace Pennylane::LightningQubit::Gates::AVXCommon
