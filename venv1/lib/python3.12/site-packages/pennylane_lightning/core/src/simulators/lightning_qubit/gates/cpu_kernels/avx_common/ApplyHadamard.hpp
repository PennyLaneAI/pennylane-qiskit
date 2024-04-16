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
 * Defines Hadamard gate
 */
#pragma once
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Permutation.hpp"
#include "Util.hpp"

#include <complex>

namespace Pennylane::LightningQubit::Gates::AVXCommon {
template <typename PrecisionT, size_t packed_size> struct ApplyHadamard {
    using Precision = PrecisionT;
    using PrecisionAVXConcept = AVXConceptType<PrecisionT, packed_size>;

    constexpr static size_t packed_size_ = packed_size;

    template <size_t rev_wire>
    static void applyInternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              [[maybe_unused]] bool inverse) {
        using namespace Permutation;
        constexpr static auto isqrt2 = INVSQRT2<PrecisionT>();

        constexpr static auto mat_diag =
            internalParity<PrecisionT, packed_size>(rev_wire) * isqrt2;
        constexpr static auto mat_offdiag =
            set1<PrecisionT, packed_size>(isqrt2);

        constexpr static auto compiled_permutation =
            compilePermutation<PrecisionT>(
                flip(identity<packed_size>(), rev_wire));
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits); k += packed_size / 2) {
            const auto v = PrecisionAVXConcept::load(arr + k);

            const auto w_diag = mat_diag * v;
            const auto v_offdiag = permute<compiled_permutation>(v);
            const auto w_offdiag = mat_offdiag * v_offdiag;
            PrecisionAVXConcept::store(arr + k, w_diag + w_offdiag);
        }
    }

    static void applyExternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits, const size_t rev_wire,
                              [[maybe_unused]] bool inverse) {
        constexpr auto isqrt2 = INVSQRT2<PrecisionT>();

        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        const auto p_isqrt2 = set1<PrecisionT, packed_size>(isqrt2);
        const auto m_isqrt2 = set1<PrecisionT, packed_size>(-isqrt2);
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const auto v0 = PrecisionAVXConcept::load(arr + i0);
            const auto v1 = PrecisionAVXConcept::load(arr + i1);

            const auto w0 = (p_isqrt2 * v0) + (p_isqrt2 * v1);
            const auto w1 = (p_isqrt2 * v0) + (m_isqrt2 * v1);

            PrecisionAVXConcept::store(arr + i0, w0);
            PrecisionAVXConcept::store(arr + i1, w1);
        }
    }
};
} // namespace Pennylane::LightningQubit::Gates::AVXCommon
