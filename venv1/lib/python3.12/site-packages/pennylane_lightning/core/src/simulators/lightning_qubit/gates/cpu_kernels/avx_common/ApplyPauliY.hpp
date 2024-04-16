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
 * Defines PauliY gate
 */
#pragma once
#include "AVXConceptType.hpp"
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Permutation.hpp"
#include "Util.hpp"

#include <complex>

namespace Pennylane::LightningQubit::Gates::AVXCommon {
template <typename PrecisionT, size_t packed_size> struct ApplyPauliY {
    using Precision = PrecisionT;
    using PrecisionAVXConcept = AVXConceptType<PrecisionT, packed_size>;

    constexpr static size_t packed_size_ = packed_size;

    template <size_t rev_wire>
    static void applyInternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              [[maybe_unused]] bool inverse) {
        using namespace Permutation;
        const auto factor = internalParity<PrecisionT, packed_size>(rev_wire) *
                            (-imagFactor<PrecisionT, packed_size>());
        constexpr static auto compiled_permutation =
            compilePermutation<PrecisionT>(
                swapRealImag(flip(identity<packed_size>(), rev_wire)));
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < (1U << num_qubits); k += packed_size / 2) {
            const auto v = PrecisionAVXConcept::load(arr + k);
            const auto w = permute<compiled_permutation>(v);
            PrecisionAVXConcept::store(arr + k, w * factor);
        }
    }

    static void applyExternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits, const size_t rev_wire,
                              [[maybe_unused]] bool inverse) {
        using namespace Permutation;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        constexpr static auto m_imag =
            imagFactor<PrecisionT, packed_size>(-1.0);
        constexpr static auto p_imag = imagFactor<PrecisionT, packed_size>(1.0);

        constexpr static auto compiled_permutation =
            compilePermutation<PrecisionT>(
                swapRealImag(identity<packed_size>()));
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const auto v0 = PrecisionAVXConcept::load(arr + i0);
            const auto v1 = PrecisionAVXConcept::load(arr + i1);
            PrecisionAVXConcept::store(
                arr + i0, m_imag * permute<compiled_permutation>(v1));
            PrecisionAVXConcept::store(
                arr + i1, p_imag * permute<compiled_permutation>(v0));
        }
    }
};
} // namespace Pennylane::LightningQubit::Gates::AVXCommon
