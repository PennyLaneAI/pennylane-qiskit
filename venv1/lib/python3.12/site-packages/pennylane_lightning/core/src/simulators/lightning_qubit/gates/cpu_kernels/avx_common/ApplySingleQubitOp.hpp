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
 * Defines applySingleQubitOp for AVX
 */
#pragma once
#include "AVXConceptType.hpp"
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Permutation.hpp"
#include "Util.hpp"

#include <complex>

namespace Pennylane::LightningQubit::Gates::AVXCommon {
template <typename PrecisionT, size_t packed_size> struct ApplySingleQubitOp {
    using PrecisionAVXConcept =
        typename AVXConcept<PrecisionT, packed_size>::Type;

    template <size_t rev_wire>
    static void applyInternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              const std::complex<PrecisionT> *matrix,
                              bool inverse = false) {
        using namespace Permutation;

        const AVXIntrinsicType<PrecisionT, packed_size> diag_real =
            setValueOneTwo<PrecisionT, packed_size>([=](size_t idx) {
                return (((idx >> rev_wire) & 1U) == 0) ? real(matrix[0])
                                                       : real(matrix[3]);
            });
        const AVXIntrinsicType<PrecisionT, packed_size> diag_imag =
            setValueOneTwo<PrecisionT, packed_size>([=](size_t idx) {
                if (inverse) {
                    return (((idx >> rev_wire) & 1U) == 0) ? -imag(matrix[0])
                                                           : -imag(matrix[3]);
                } // else
                return (((idx >> rev_wire) & 1U) == 0) ? imag(matrix[0])
                                                       : imag(matrix[3]);
            }) *
            imagFactor<PrecisionT, packed_size>();
        const AVXIntrinsicType<PrecisionT, packed_size> offdiag_real =
            setValueOneTwo<PrecisionT, packed_size>([=](size_t idx) {
                if (inverse) {
                    return (((idx >> rev_wire) & 1U) == 0) ? real(matrix[2])
                                                           : real(matrix[1]);
                } // else
                return (((idx >> rev_wire) & 1U) == 0) ? real(matrix[1])
                                                       : real(matrix[2]);
            });
        const AVXIntrinsicType<PrecisionT, packed_size> offdiag_imag =
            setValueOneTwo<PrecisionT, packed_size>([=](size_t idx) {
                if (inverse) {
                    return (((idx >> rev_wire) & 1U) == 0) ? -imag(matrix[2])
                                                           : -imag(matrix[1]);
                } // else
                return (((idx >> rev_wire) & 1U) == 0) ? imag(matrix[1])
                                                       : imag(matrix[2]);
            }) *
            imagFactor<PrecisionT, packed_size>();
        ;

        constexpr static auto flip_rev_wire = compilePermutation<PrecisionT>(
            flip(identity<packed_size>(), rev_wire));
        constexpr static auto swap_real_imag = compilePermutation<PrecisionT>(
            swapRealImag(identity<packed_size>()));
        constexpr static auto flip_swap_real_imag =
            compilePermutation<PrecisionT>(
                swapRealImag(flip(identity<packed_size>(), rev_wire)));
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits); k += packed_size / 2) {
            const auto v = PrecisionAVXConcept::load(arr + k);
            const auto w_diag =
                diag_real * v + diag_imag * permute<swap_real_imag>(v);

            const auto v_off_real = offdiag_real * permute<flip_rev_wire>(v);

            const auto v_off_imag =
                offdiag_imag * permute<flip_swap_real_imag>(v);

            PrecisionAVXConcept::store(arr + k,
                                       w_diag + v_off_imag + v_off_real);
        }
    }

    static void applyExternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits, const size_t rev_wire,
                              const std::complex<PrecisionT> *matrix,
                              bool inverse = false) {
        using namespace Permutation;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        std::complex<PrecisionT> u00;
        std::complex<PrecisionT> u01;
        std::complex<PrecisionT> u10;
        std::complex<PrecisionT> u11;

        if (inverse) {
            u00 = std::conj(matrix[0]);
            u01 = std::conj(matrix[2]);
            u10 = std::conj(matrix[1]);
            u11 = std::conj(matrix[3]);
        } else {
            u00 = matrix[0];
            u01 = matrix[1];
            u10 = matrix[2];
            u11 = matrix[3];
        }

        const auto u00_real = set1<PrecisionT, packed_size>(real(u00));
        const auto u00_imag = set1<PrecisionT, packed_size>(imag(u00)) *
                              imagFactor<PrecisionT, packed_size>();

        const auto u01_real = set1<PrecisionT, packed_size>(real(u01));
        const auto u01_imag = set1<PrecisionT, packed_size>(imag(u01)) *
                              imagFactor<PrecisionT, packed_size>();

        const auto u10_real = set1<PrecisionT, packed_size>(real(u10));
        const auto u10_imag = set1<PrecisionT, packed_size>(imag(u10)) *
                              imagFactor<PrecisionT, packed_size>();

        const auto u11_real = set1<PrecisionT, packed_size>(real(u11));
        const auto u11_imag = set1<PrecisionT, packed_size>(imag(u11)) *
                              imagFactor<PrecisionT, packed_size>();

        constexpr static auto swap_real_imag = compilePermutation<PrecisionT>(
            swapRealImag(identity<packed_size>()));
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;

            const auto v0 = PrecisionAVXConcept::load(arr + i0);
            const auto v1 = PrecisionAVXConcept::load(arr + i1);

            // w0 = u00 * v0 + u01 * v1
            const auto w0_real = u00_real * v0 + u01_real * v1;
            const auto w0_imag = u00_imag * permute<swap_real_imag>(v0) +
                                 u01_imag * permute<swap_real_imag>(v1);

            // w1 = u11 * v1 + u10 * v0
            const auto w1_real = u11_real * v1 + u10_real * v0;
            const auto w1_imag = u11_imag * permute<swap_real_imag>(v1) +
                                 u10_imag * permute<swap_real_imag>(v0);

            PrecisionAVXConcept::store(arr + i0, w0_real + w0_imag);
            PrecisionAVXConcept::store(arr + i1, w1_real + w1_imag);
        }
    }
};
/// @endcond
} // namespace Pennylane::LightningQubit::Gates::AVXCommon
