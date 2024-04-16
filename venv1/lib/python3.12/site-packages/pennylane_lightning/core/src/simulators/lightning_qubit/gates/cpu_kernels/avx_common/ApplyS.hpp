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
 * Defines S gate
 */
#pragma once
#include "AVXConceptType.hpp"
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Permutation.hpp"
#include "Util.hpp"

#include <complex>

namespace Pennylane::LightningQubit::Gates::AVXCommon {
template <typename PrecisionT, size_t packed_size> struct ApplyS {
    using Precision = PrecisionT;
    using PrecisionAVXConcept = AVXConceptType<PrecisionT, packed_size>;

    constexpr static size_t packed_size_ = packed_size;

    /**
     * @brief Permutation for applying `i` to
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

    static auto createFactor(size_t rev_wire, bool inverse)
        -> AVXIntrinsicType<PrecisionT, packed_size> {
        std::array<PrecisionT, packed_size> data{};
        PL_LOOP_SIMD
        for (size_t n = 0; n < packed_size / 2; n++) {
            if (((n >> rev_wire) & 1U) == 0) {
                data[2 * n + 0] = 1.0;
                data[2 * n + 1] = 1.0;
            } else {
                if (inverse) {
                    data[2 * n + 0] = 1.0;
                    data[2 * n + 1] = -1.0;
                } else {
                    data[2 * n + 0] = -1.0;
                    data[2 * n + 1] = 1.0;
                }
            }
        }
        return PrecisionAVXConcept::loadu(data.data());
    }

    template <size_t rev_wire>
    static void applyInternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits, bool inverse) {
        constexpr static auto perm = applyInternalPermutation(rev_wire);
        const auto factor = createFactor(rev_wire, inverse);
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < (1U << num_qubits); k += packed_size / 2) {
            const auto v = PrecisionAVXConcept::load(arr + k);
            PrecisionAVXConcept::store(arr + k,
                                       factor * Permutation::permute<perm>(v));
        }
    }

    static void applyExternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits, const size_t rev_wire,
                              bool inverse) {
        using namespace Permutation;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        const auto factor =
            set1<PrecisionT, packed_size>(inverse ? -1.0 : 1.0) *
            imagFactor<PrecisionT, packed_size>();
        constexpr static auto perm = compilePermutation<PrecisionT>(
            swapRealImag(identity<packed_size>()));
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const auto v1 = PrecisionAVXConcept::load(arr + i1);
            PrecisionAVXConcept::store(arr + i1, factor * permute<perm>(v1));
        }
    }
};
} // namespace Pennylane::LightningQubit::Gates::AVXCommon
