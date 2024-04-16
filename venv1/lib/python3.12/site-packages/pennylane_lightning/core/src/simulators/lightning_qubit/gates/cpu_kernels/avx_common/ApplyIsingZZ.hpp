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
 * Defines IsingZZ gate
 */
#pragma once
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Permutation.hpp"
#include "Util.hpp"

#include <complex>

namespace Pennylane::LightningQubit::Gates::AVXCommon {
template <typename PrecisionT, size_t packed_size> struct ApplyIsingZZ {
    using PrecisionAVXConcept =
        typename AVXConcept<PrecisionT, packed_size>::Type;
    using Precision = PrecisionT;

    constexpr static auto perm = Permutation::compilePermutation<PrecisionT>(
        Permutation::swapRealImag(Permutation::identity<packed_size>()));
    constexpr static size_t packed_size_ = packed_size;
    constexpr static bool symmetric = true;

    template <size_t rev_wire0, size_t rev_wire1, class ParamT>
    static void applyInternalInternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits, bool inverse,
                                      ParamT angle) {
        const auto isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
        const auto parity = toParity<PrecisionT, packed_size>([=](size_t idx) {
            return ((idx >> rev_wire0) & 1U) ^ ((idx >> rev_wire1) & 1U);
        });
        const auto real_cos =
            set1<PrecisionT, packed_size>(std::cos(angle / 2));
        const auto imag_sin =
            imagFactor<PrecisionT, packed_size>(isin) * parity;
        PL_LOOP_PARALLEL(1)
        for (size_t n = 0; n < exp2(num_qubits); n += packed_size / 2) {
            const auto v = PrecisionAVXConcept::load(arr + n);

            const auto prod_cos = real_cos * v;
            const auto prod_sin = imag_sin * Permutation::permute<perm>(v);

            PrecisionAVXConcept::store(arr + n, prod_cos + prod_sin);
        }
    }

    template <size_t min_rev_wire, class ParamT>
    static void applyInternalExternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits, size_t max_rev_wire,
                                      bool inverse, ParamT angle) {
        const size_t max_rev_wire_shift =
            (static_cast<size_t>(1U) << max_rev_wire);
        const size_t max_wire_parity = fillTrailingOnes(max_rev_wire);
        const size_t max_wire_parity_inv = fillLeadingOnes(max_rev_wire + 1);

        const auto isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);
        const auto real_cos =
            set1<PrecisionT, packed_size>(std::cos(angle / 2));
        const auto imag_sin = imagFactor<PrecisionT, packed_size>(isin);

        const auto imag_sin_parity0 =
            imag_sin * internalParity<PrecisionT, packed_size>(min_rev_wire);
        const auto imag_sin_parity1 = imag_sin_parity0 * -1.0;
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 =
                ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
            const size_t i1 = i0 | max_rev_wire_shift;

            const auto v0 = PrecisionAVXConcept::load(arr + i0);
            const auto v1 = PrecisionAVXConcept::load(arr + i1);

            const auto prod_cos0 = real_cos * v0;
            const auto prod_sin0 =
                imag_sin_parity0 * Permutation::permute<perm>(v0);

            const auto prod_cos1 = real_cos * v1;
            const auto prod_sin1 =
                imag_sin_parity1 * Permutation::permute<perm>(v1);

            PrecisionAVXConcept::store(arr + i0, prod_cos0 + prod_sin0);
            PrecisionAVXConcept::store(arr + i1, prod_cos1 + prod_sin1);
        }
    }

    template <class ParamT>
    static void
    applyExternalExternal(std::complex<PrecisionT> *arr,
                          const size_t num_qubits, const size_t rev_wire0,
                          const size_t rev_wire1, bool inverse, ParamT angle) {
        using namespace Permutation;
        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
        const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

        const size_t parity_low = fillTrailingOnes(rev_wire_min);
        const size_t parity_high = fillLeadingOnes(rev_wire_max + 1);
        const size_t parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        const auto isin = inverse ? std::sin(angle / 2) : -std::sin(angle / 2);

        const auto real_cos =
            set1<PrecisionT, packed_size>(std::cos(angle / 2));

        const auto p_isin = imagFactor<PrecisionT, packed_size>(isin);
        const auto m_isin = imagFactor<PrecisionT, packed_size>(-isin);
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 2); k += packed_size / 2) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            const auto v00 = PrecisionAVXConcept::load(arr + i00); // 00
            const auto v01 = PrecisionAVXConcept::load(arr + i01); // 01
            const auto v10 = PrecisionAVXConcept::load(arr + i10); // 10
            const auto v11 = PrecisionAVXConcept::load(arr + i11); // 11

            const auto prod_cos00 = real_cos * v00;
            const auto prod_isin00 = p_isin * permute<perm>(v00);

            const auto prod_cos01 = real_cos * v01;
            const auto prod_isin01 = m_isin * permute<perm>(v01);

            const auto prod_cos10 = real_cos * v10;
            const auto prod_isin10 = m_isin * permute<perm>(v10);

            const auto prod_cos11 = real_cos * v11;
            const auto prod_isin11 = p_isin * permute<perm>(v11);

            PrecisionAVXConcept::store(arr + i00, prod_cos00 + prod_isin00);
            PrecisionAVXConcept::store(arr + i01, prod_cos01 + prod_isin01);
            PrecisionAVXConcept::store(arr + i10, prod_cos10 + prod_isin10);
            PrecisionAVXConcept::store(arr + i11, prod_cos11 + prod_isin11);
        }
    }
};
} // namespace Pennylane::LightningQubit::Gates::AVXCommon
