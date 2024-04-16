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
 * Defines RX gate
 */
#pragma once
#include "AVXConceptType.hpp"
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Permutation.hpp"
#include "Util.hpp"

#include <complex>

namespace Pennylane::LightningQubit::Gates::AVXCommon {
template <typename PrecisionT, size_t packed_size> struct ApplyRX {
    using Precision = PrecisionT;
    using PrecisionAVXConcept =
        typename AVXConcept<PrecisionT, packed_size>::Type;

    constexpr static size_t packed_size_ = packed_size;

    template <size_t rev_wire, class ParamT>
    static void applyInternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              [[maybe_unused]] bool inverse, ParamT angle) {
        using namespace Permutation;
        const PrecisionT isin =
            inverse ? std::sin(angle / 2) : -std::sin(angle / 2);

        const auto diag_real =
            set1<PrecisionT, packed_size>(std::cos(angle / 2));
        const auto offdiag_imag = imagFactor<PrecisionT, packed_size>(isin);

        constexpr static auto perm = compilePermutation<PrecisionT>(
            swapRealImag(flip(identity<packed_size>(), rev_wire)));
        PL_LOOP_PARALLEL(1)
        for (size_t n = 0; n < (1U << num_qubits); n += packed_size / 2) {
            const auto v = PrecisionAVXConcept::load(arr + n);
            const auto w_diag = diag_real * v;
            const auto w_offdiag = offdiag_imag * permute<perm>(v);
            PrecisionAVXConcept::store(arr + n, w_diag + w_offdiag);
        }
    }

    template <class ParamT>
    static void applyExternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits, const size_t rev_wire,
                              [[maybe_unused]] bool inverse, ParamT angle) {
        using namespace Permutation;

        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        const auto cos_factor =
            set1<PrecisionT, packed_size>(std::cos(angle / 2));
        const PrecisionT isin =
            inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
        const auto isin_factor = imagFactor<PrecisionT, packed_size>(isin);

        constexpr static auto perm = compilePermutation<PrecisionT>(
            swapRealImag(identity<packed_size>()));
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const auto v0 = PrecisionAVXConcept::load(arr + i0);
            const auto v1 = PrecisionAVXConcept::load(arr + i1);

            const auto w0 = cos_factor * v0 + isin_factor * permute<perm>(v1);
            const auto w1 = cos_factor * v1 + isin_factor * permute<perm>(v0);

            PrecisionAVXConcept::store(arr + i0, w0);
            PrecisionAVXConcept::store(arr + i1, w1);
        }
    }
};
} // namespace Pennylane::LightningQubit::Gates::AVXCommon
