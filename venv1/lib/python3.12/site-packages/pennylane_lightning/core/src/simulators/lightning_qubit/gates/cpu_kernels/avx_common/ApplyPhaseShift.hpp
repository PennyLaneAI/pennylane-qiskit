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
 * Defines PhaseShift gate
 */
#pragma once
#include "AVXConceptType.hpp"
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Permutation.hpp"
#include "Util.hpp"

#include <complex>

namespace Pennylane::LightningQubit::Gates::AVXCommon {
template <typename PrecisionT, size_t packed_size> struct ApplyPhaseShift {
    using Precision = PrecisionT;
    using PrecisionAVXConcept =
        typename AVXConcept<PrecisionT, packed_size>::Type;

    constexpr static size_t packed_size_ = packed_size;

    /**
     * @brief Permutation for applying `i` if a bit is 1
     *
     * FIXME: clang++-12 currently does not accept consteval here.
     */
    static constexpr auto applyInternalPermutation(size_t rev_wire) {
        std::array<uint8_t, packed_size> perm{};
        for (size_t n = 0; n < packed_size / 2; n++) {
            if (((n >> rev_wire) & 1U) == 0) {
                perm[2 * n + 0] = 2 * n + 0;
                perm[2 * n + 1] = 2 * n + 1;
            } else {
                perm[2 * n + 0] = 2 * n + 1;
                perm[2 * n + 1] = 2 * n + 0;
            }
        }

        return Permutation::compilePermutation<PrecisionT>(perm);
    }

    /**
     * @brief Factor for applying [1, 1, cos(phi/2), cos(phi/2)]
     */
    static auto cosFactor(size_t rev_wire, PrecisionT angle)
        -> AVXIntrinsicType<PrecisionT, packed_size> {
        std::array<PrecisionT, packed_size> arr{};
        PL_LOOP_SIMD
        for (size_t n = 0; n < packed_size / 2; n++) {
            if (((n >> rev_wire) & 1U) == 0) {
                arr[2 * n + 0] = 1.0;
                arr[2 * n + 1] = 1.0;
            } else {
                arr[2 * n + 0] = std::cos(angle);
                arr[2 * n + 1] = arr[2 * n + 0];
            }
        }
        return setValue(arr);
    }

    /**
     * @brief Factor for applying [0, 0, -sin(phi/2), sin(phi/2)]
     */
    static auto isinFactor(size_t rev_wire, PrecisionT angle)
        -> AVXIntrinsicType<PrecisionT, packed_size> {
        std::array<PrecisionT, packed_size> arr{};
        PL_LOOP_SIMD
        for (size_t n = 0; n < packed_size / 2; n++) {
            if (((n >> rev_wire) & 1U) == 0) {
                arr[2 * n + 0] = 0.0;
                arr[2 * n + 1] = 0.0;
            } else {
                arr[2 * n + 0] = -std::sin(angle);
                arr[2 * n + 1] = -arr[2 * n + 0];
            }
        }
        return setValue(arr);
    }

    template <size_t rev_wire, typename ParamT>
    static void applyInternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits, bool inverse,
                              ParamT angle) {
        constexpr static auto perm = applyInternalPermutation(rev_wire);
        const auto cos_factor = cosFactor(rev_wire, angle);
        const auto isin_factor =
            isinFactor(rev_wire, (inverse ? -angle : angle));
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < (1U << num_qubits); k += packed_size / 2) {
            const auto v = PrecisionAVXConcept::load(arr + k);
            const auto w =
                cos_factor * v + isin_factor * Permutation::permute<perm>(v);
            PrecisionAVXConcept::store(arr + k, w);
        }
    }

    template <typename ParamT>
    static void applyExternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits, const size_t rev_wire,
                              bool inverse, ParamT angle) {
        using namespace Permutation;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        const auto cos_factor =
            set1<PrecisionT, packed_size>(static_cast<PrecisionT>(cos(angle)));
        const auto isin_factor =
            set1<PrecisionT, packed_size>(inverse ? -1.0 : 1.0) *
            imagFactor<PrecisionT, packed_size>(
                static_cast<PrecisionT>(sin(angle)));
        constexpr static auto perm = compilePermutation<PrecisionT>(
            swapRealImag(identity<packed_size>()));
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const auto v1 = PrecisionAVXConcept::load(arr + i1);
            const auto w1 = cos_factor * v1 + isin_factor * permute<perm>(v1);
            PrecisionAVXConcept::store(arr + i1, w1);
        }
    }
};
} // namespace Pennylane::LightningQubit::Gates::AVXCommon
