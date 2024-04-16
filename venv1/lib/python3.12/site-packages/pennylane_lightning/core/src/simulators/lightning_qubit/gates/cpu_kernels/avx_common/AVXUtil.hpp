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
 * Defines common utility functions for all AVX
 */
#pragma once
#include "BitUtil.hpp" // fillTrailingOnes, fillLeadingOnes, log2PerfectPower
#include "Error.hpp"
#include "Macros.hpp"
#include "Util.hpp" // INVSQRT2

#include <immintrin.h>

#include <cstdlib>

namespace Pennylane::LightningQubit::Gates::AVXCommon {
using Pennylane::Util::fillLeadingOnes;
using Pennylane::Util::fillTrailingOnes;
using Pennylane::Util::INVSQRT2;
using Pennylane::Util::log2PerfectPower;

template <typename PrecisionT, size_t packed_size> struct AVXIntrinsic {
    static_assert((sizeof(PrecisionT) * packed_size == 32) ||
                  (sizeof(PrecisionT) * packed_size == 64));
};
template <typename T, size_t size>
using AVXIntrinsicType = typename AVXIntrinsic<T, size>::Type;
#ifdef PL_USE_AVX2
template <> struct AVXIntrinsic<float, 8> {
    // AVX2
    using Type = __m256;
};
template <> struct AVXIntrinsic<double, 4> {
    // AVX2
    using Type = __m256d;
};
#endif
#ifdef PL_USE_AVX512F
template <> struct AVXIntrinsic<float, 16> {
    // AVX512
    using Type = __m512;
};
template <> struct AVXIntrinsic<double, 8> {
    // AVX512
    using Type = __m512d;
};
#endif

/**
 * @brief one or minus one parity for reverse wire in packed data.
 */
template <typename PrecisionT, size_t packed_size>
constexpr auto internalParity(size_t rev_wire)
    -> AVXIntrinsicType<PrecisionT, packed_size>;
#ifdef PL_USE_AVX2
template <> constexpr auto internalParity<float, 8>(size_t rev_wire) -> __m256 {
    switch (rev_wire) {
    case 0:
        // When Z is applied to the 0th qubit
        return __m256{1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F};
    case 1:
        // When Z is applied to the 1st qubit
        return __m256{1.0F, 1.0F, 1.0F, 1.0F, -1.0F, -1.0F, -1.0F, -1.0F};
    default:
        PL_UNREACHABLE;
    }
    return _mm256_setzero_ps();
}
template <>
constexpr auto internalParity<double, 4>([[maybe_unused]] size_t rev_wire)
    -> __m256d {
    PL_ASSERT(rev_wire == 0);
    // When Z is applied to the 0th qubit
    return __m256d{1.0, 1.0, -1.0, -1.0};
}
#endif
#ifdef PL_USE_AVX512F
// LCOV_EXCL_START
template <>
constexpr auto internalParity<float, 16>(size_t rev_wire) -> __m512 {
    // AVX512 with float
    // clang-format off
    switch(rev_wire) {
    case 0:
        // When Z is applied to the 0th qubit
        return __m512{1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F,
                      1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, -1.0F, -1.0F};
    case 1:
        // When Z is applied to the 1st qubit
        return __m512{1.0F, 1.0F, 1.0F, 1.0F, -1.0F, -1.0F, -1.0F, -1.0F,
                      1.0F, 1.0F, 1.0F, 1.0F, -1.0F,- 1.0F, -1.0F, -1.0F};
    case 2:
        // When Z is applied to the 2nd qubit
        return __m512{ 1.0F,  1.0F,  1.0F,  1.0F,
                       1.0F,  1.0F,  1.0F,  1.0F,
                      -1.0F, -1.0F, -1.0F, -1.0F,
                      -1.0F,- 1.0F, -1.0F, -1.0F};
    default:
        PL_UNREACHABLE;
    }
    // clang-format on
    return __m512{
        0,
    };
};
template <>
constexpr auto internalParity<double, 8>(size_t rev_wire) -> __m512d {
    // AVX512 with double
    switch (rev_wire) {
    case 0:
        // When Z is applied to the 0th qubit
        return __m512d{1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0};
    case 1:
        // When Z is applied to the 1st qubit
        return __m512d{1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0};
    default:
        PL_UNREACHABLE;
    }
    return __m512d{
        0,
    };
}
// LCOV_EXCL_STOP
#endif

/**
 * @brief Factor that is applied to the intrinsic type for product of
 * pure imaginary value.
 */
template <typename PrecisionT, size_t packed_size> struct ImagFactor;

template <typename PrecisionT, size_t packed_size>
constexpr auto imagFactor(PrecisionT val = 1.0) {
    return ImagFactor<PrecisionT, packed_size>::create(val);
}
#ifdef PL_USE_AVX2
template <> struct ImagFactor<float, 8> {
    constexpr static auto create(float val) -> AVXIntrinsicType<float, 8> {
        return __m256{-val, val, -val, val, -val, val, -val, val};
    };
};
template <> struct ImagFactor<double, 4> {
    constexpr static auto create(double val) -> AVXIntrinsicType<double, 4> {
        return __m256d{-val, val, -val, val};
    };
};
#endif
#ifdef PL_USE_AVX512F
// LCOV_EXCL_START
template <> struct ImagFactor<float, 16> {
    constexpr static auto create(float val) -> AVXIntrinsicType<float, 16> {
        return __m512{-val, val, -val, val, -val, val, -val, val,
                      -val, val, -val, val, -val, val, -val, val};
    };
};
template <> struct ImagFactor<double, 8> {
    constexpr static auto create(double val) -> AVXIntrinsicType<double, 8> {
        return __m512d{-val, val, -val, val, -val, val, -val, val};
    };
};
// LCOV_EXCL_STOP
#endif

template <typename PrecisionT, size_t packed_size> struct Set1;
#ifdef PL_USE_AVX2
template <> struct Set1<float, 8> {
    constexpr static auto create(float val) -> AVXIntrinsicType<float, 8> {
        return __m256{val, val, val, val, val, val, val, val};
    }
};
template <> struct Set1<double, 4> {
    constexpr static auto create(double val) -> AVXIntrinsicType<double, 4> {
        return __m256d{val, val, val, val};
    }
};
#endif
#ifdef PL_USE_AVX512F
// LCOV_EXCL_START
template <> struct Set1<float, 16> {
    constexpr static auto create(float val) -> AVXIntrinsicType<float, 16> {
        return __m512{val, val, val, val, val, val, val, val,
                      val, val, val, val, val, val, val, val};
    }
};
template <> struct Set1<double, 8> {
    constexpr static auto create(double val) -> AVXIntrinsicType<double, 8> {
        return __m512d{val, val, val, val, val, val, val, val};
    }
};
// LCOV_EXCL_STOP
#endif

template <typename PrecisionT, size_t packed_size>
constexpr auto set1(PrecisionT val) {
    return Set1<PrecisionT, packed_size>::create(val);
}

template <size_t packed_size> struct InternalWires {
    constexpr static auto value = log2PerfectPower(packed_size / 2);
};
template <size_t packed_size>
constexpr auto internal_wires_v = InternalWires<packed_size>::value;

#ifdef PL_USE_AVX2
constexpr static auto setValue(const std::array<float, 8> &arr)
    -> AVXIntrinsicType<float, 8> {
    // NOLINTBEGIN(readability-magic-numbers)
    return __m256{arr[0], arr[1], arr[2], arr[3],
                  arr[4], arr[5], arr[6], arr[7]};
    // NOLINTEND(readability-magic-numbers)
}
constexpr static auto setValue(const std::array<double, 4> &arr)
    -> AVXIntrinsicType<double, 4> {
    // NOLINTBEGIN(readability-magic-numbers)
    return __m256d{arr[0], arr[1], arr[2], arr[3]};
    // NOLINTEND(readability-magic-numbers)
}
#endif
#ifdef PL_USE_AVX512F
constexpr static auto setValue(const std::array<float, 16> &arr)
    -> AVXIntrinsicType<float, 16> {
    // NOLINTBEGIN(readability-magic-numbers)
    return __m512{arr[0],  arr[1],  arr[2],  arr[3], arr[4],  arr[5],
                  arr[6],  arr[7],  arr[8],  arr[9], arr[10], arr[11],
                  arr[12], arr[13], arr[14], arr[15]};
    // NOLINTEND(readability-magic-numbers)
}
constexpr static auto setValue(const std::array<double, 8> &arr)
    -> AVXIntrinsicType<double, 8> {
    // NOLINTBEGIN(readability-magic-numbers)
    return __m512d{arr[0], arr[1], arr[2], arr[3],
                   arr[4], arr[5], arr[6], arr[7]};
    // NOLINTEND(readability-magic-numbers)
}
#endif

// clang-format off
#ifdef PL_USE_AVX2
constexpr __m256i setr256i(int32_t  e0, int32_t  e1, int32_t  e2, int32_t  e3,
		                   int32_t  e4, int32_t  e5, int32_t  e6, int32_t  e7) {
    // NOLINTBEGIN(hicpp-signed-bitwise)
    return __m256i{(static_cast<int64_t>(e1) << 32) | e0,
                   (static_cast<int64_t>(e3) << 32) | e2,
                   (static_cast<int64_t>(e5) << 32) | e4,
                   (static_cast<int64_t>(e7) << 32) | e6};
    // NOLINTEND(hicpp-signed-bitwise)
}
#endif
#ifdef PL_USE_AVX512F
// LCOV_EXCL_START
constexpr __m512i setr512i(int32_t  e0, int32_t  e1, int32_t  e2, int32_t  e3,
		                   int32_t  e4, int32_t  e5, int32_t  e6, int32_t  e7,
		                   int32_t  e8, int32_t  e9, int32_t e10, int32_t e11,
		                   int32_t e12, int32_t e13, int32_t e14, int32_t e15) {
    // NOLINTBEGIN(hicpp-signed-bitwise)
    return __m512i{(static_cast<int64_t>(e1) << 32)  |  e0,
                   (static_cast<int64_t>(e3) << 32)  |  e2,
                   (static_cast<int64_t>(e5) << 32)  |  e4,
                   (static_cast<int64_t>(e7) << 32)  |  e6,
                   (static_cast<int64_t>(e9) << 32)  |  e8,
                   (static_cast<int64_t>(e11) << 32) | e10,
                   (static_cast<int64_t>(e13) << 32) | e12,
                   (static_cast<int64_t>(e15) << 32) | e14};
    // NOLINTEND(hicpp-signed-bitwise)
}
constexpr __m512i setr512i(int64_t  e0, int64_t  e1, int64_t  e2, int64_t  e3,
		                   int64_t  e4, int64_t  e5, int64_t  e6, int64_t  e7) {
    return __m512i{e0, e1, e2, e3, e4, e5, e6, e7};
}
// LCOV_EXCL_STOP
#endif
// clang-format on

/**
 * @brief @rst
 * For a function :math:`f(x)` with binary output, this function creates
 * an AVX intrinsic floating-point type with values :math:`(-1)^{f(x)}`
 * where :math:`x` is index of an array (viewed as a complex-valued array).
 * @endrst
 *
 * @rst
 * For example, when :math:`f(x) = x % 2`, this returns a packed array
 * with values [1, 1, -1, -1, 1, 1, -1, -1]. Note that each value is repeated
 * twice as it applies to the both real and imaginary parts. This function is
 * used e.g. in CZ gate.
 * @endrst
 *
 * @tparam PrecisionT Floating point precision type
 * @tparam packed_size Number of packed values for a AVX intrinsic type
 * @tparam Func Type of a function
 * @param func Binary output function
 */
template <typename PrecisionT, size_t packed_size, typename Func>
auto toParity(Func &&func) -> AVXIntrinsicType<PrecisionT, packed_size> {
    std::array<PrecisionT, packed_size> data{};
    PL_LOOP_SIMD
    for (size_t idx = 0; idx < packed_size / 2; idx++) {
        data[2 * idx + 0] = static_cast<PrecisionT>(1.0) -
                            2 * static_cast<PrecisionT>(func(idx));
        data[2 * idx + 1] = static_cast<PrecisionT>(1.0) -
                            2 * static_cast<PrecisionT>(func(idx));
    }
    return setValue(data);
}

/**
 * @brief Repeat the value of the function twice.
 *
 * As we treat a complex number as two real numbers, this helps when we
 * multiply function outcomes to a AVX intrinsic type.
 */
template <typename PrecisionT, size_t packed_size, typename Func>
auto setValueOneTwo(Func &&func) -> AVXIntrinsicType<PrecisionT, packed_size> {
    std::array<PrecisionT, packed_size> data{};
    PL_LOOP_SIMD
    for (size_t idx = 0; idx < packed_size / 2; idx++) {
        data[2 * idx + 0] = static_cast<PrecisionT>(func(idx));
        data[2 * idx + 1] = data[2 * idx + 0];
    }
    return setValue(data);
}
} // namespace Pennylane::LightningQubit::Gates::AVXCommon
