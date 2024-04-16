// Copyright 2023 Xanadu Quantum Technologies Inc.

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
 * Defines CRX gate
 */
#pragma once
#include "AVXConceptType.hpp"
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Blender.hpp"
#include "Permutation.hpp"

#include "ConstantUtil.hpp"
#include "Util.hpp"

#include <complex>
#include <utility>

namespace Pennylane::LightningQubit::Gates::AVXCommon {
template <typename PrecisionT, size_t packed_size> struct ApplyCRX {
    using Precision = PrecisionT;
    using PrecisionAVXConcept = AVXConceptType<PrecisionT, packed_size>;

    constexpr static auto packed_size_ = packed_size;
    constexpr static bool symmetric = false;

    // clang-format off
    /**
     * We implement CRX gate by dividing the matrix into diagonal and
     * off-diagonal parts. The matrix is written as:
     * [1   0   0               0            ]
     * [0   1   0               0            ]
     * [0   0   cos(phi/2)      -i sin(phi/2)]
     * [0   0   -i sin(phi/2)   cos(phi/2)   ]
     *
     * Applying the matrix to a vector v, we thus
     * (1) compute [v[0], v[1], cos(phi/2) v[2], cos(phi/2) v[3]]
     * (2) compute [0, 0, -i sin(phi/2) v[3], -i sin(phi/2) v[2])]
     * and sum them.
     *
     * Functions related to (1) contains "Diag" in the name whereas those
     * related to (2) contains "OffDiang".
     * */
    // clang-format on

    /**
     * @brief Permutation for (2).
     * After applying this permutation, the array will be
     * [Re(v[0]), Im(v[0]), Re(v[1]), Im(v[1]), Im(v[3]), Re(v[3]), Im(v[2]),
     * Re(v[2])]
     */
    template <size_t control, size_t target>
    static consteval auto applyInternalInternalPermutation() {
        std::array<uint8_t, packed_size> perm{};
        for (size_t k = 0; k < packed_size / 2; k++) {
            if ((k >> control) & 1U) { // if control bit is 1
                perm[2 * k + 0] = 2 * (k ^ (1U << target)) + 1;
                perm[2 * k + 1] = 2 * (k ^ (1U << target)) + 0;
            } else {
                perm[2 * k + 0] = 2 * k + 0;
                perm[2 * k + 1] = 2 * k + 1;
            }
        }
        return Permutation::compilePermutation<PrecisionT>(perm);
    }

    /**
     * @brief Factor for (2).
     * [0, 0, 0, 0, sin(phi/2), -sin(phi/2), sin(phi/2), -sin(phi/2)]
     */
    template <size_t control, size_t target, class ParamT>
    static auto applyInternalInternalOffDiagFactor(ParamT angle) {
        std::array<PrecisionT, packed_size> arr{};

        PL_LOOP_SIMD
        for (size_t k = 0; k < packed_size / 2; k++) {
            if ((k >> control) & 1U) { // if control bit is 1
                arr[2 * k + 0] = std::sin(angle / 2);
                arr[2 * k + 1] = -std::sin(angle / 2);
            } else {
                arr[2 * k + 0] = Precision{0.0};
                arr[2 * k + 1] = Precision{0.0};
            }
        }
        return setValue(arr);
    }

    /**
     * @brief Factor for (1)
     * [1, 1, 1, 1, cos(phi/2), cos(phi/2), cos(phi/2), cos(phi/2)]
     */
    template <size_t control, size_t target, class ParamT>
    static auto applyInternalInternalDiagFactor(ParamT angle) {
        std::array<PrecisionT, packed_size> arr{};
        PL_LOOP_SIMD
        for (size_t k = 0; k < packed_size / 2; k++) {
            if ((k >> control) & 1U) { // if control bit is 1
                arr[2 * k + 0] = std::cos(angle / 2);
                arr[2 * k + 1] = std::cos(angle / 2);
            } else {
                arr[2 * k + 0] = Precision{1.0};
                arr[2 * k + 1] = Precision{1.0};
            }
        }
        return setValue(arr);
    }

    template <size_t control, size_t target, class ParamT>
    static void applyInternalInternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits, bool inverse,
                                      ParamT angle) {
        constexpr static auto perm =
            applyInternalInternalPermutation<control, target>();

        if (inverse) {
            angle *= -1.0;
        }

        const auto off_diag_factor =
            applyInternalInternalOffDiagFactor<control, target>(angle);
        const auto diag_factor =
            applyInternalInternalDiagFactor<control, target>(angle);
        PL_LOOP_PARALLEL(1)
        for (size_t n = 0; n < exp2(num_qubits); n += packed_size / 2) {
            const auto v = PrecisionAVXConcept::load(arr + n);
            const auto diag_w = diag_factor * v;
            const auto off_diag_w =
                off_diag_factor * Permutation::permute<perm>(v);
            PrecisionAVXConcept::store(arr + n, diag_w + off_diag_w);
        }
    }

    /**
     * @brief Factor for (1) when the target bit is 0/1.
     */
    template <size_t control, typename ParamT>
    static auto applyInternalExternalDiagFactor(ParamT angle) {
        std::array<Precision, packed_size> arr{};
        PL_LOOP_SIMD
        for (size_t k = 0; k < packed_size / 2; k++) {
            if ((k >> control) & 1U) {
                // if control is 1
                arr[2 * k + 0] = std::cos(angle / 2);
                arr[2 * k + 1] = std::cos(angle / 2);
            } else {
                arr[2 * k + 0] = 1.0;
                arr[2 * k + 1] = 1.0;
            }
        }
        return setValue(arr);
    }

    /**
     * @brief Factor for (2) when the target bit is 0/1.
     */
    template <size_t control, typename ParamT>
    static auto applyInternalExternalOffDiagFactor(ParamT angle) {
        std::array<Precision, packed_size> arr{};
        PL_LOOP_SIMD
        for (size_t k = 0; k < packed_size / 2; k++) {
            if ((k >> control) & 1U) {
                // if control is 1
                arr[2 * k + 0] = std::sin(angle / 2);
                arr[2 * k + 1] = -std::sin(angle / 2);
            } else {
                arr[2 * k + 0] = 0.0;
                arr[2 * k + 1] = 0.0;
            }
        }
        return setValue(arr);
    }

    /**
     * @brief Implementation for the case where the control qubit acts
     * on internal wires (inside of packed bytes) but the target acts on
     * external wires.
     */
    template <size_t control, typename ParamT>
    static void applyInternalExternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits, size_t target,
                                      bool inverse, ParamT angle) {
        // control qubit is internal but target qubit is external
        using namespace Permutation;

        const size_t target_rev_wire_shift =
            (static_cast<size_t>(1U) << target);
        const size_t target_wire_parity = fillTrailingOnes(target);
        const size_t target_wire_parity_inv = fillLeadingOnes(target + 1);

        if (inverse) {
            angle *= -1.0;
        }

        const auto diag_factor =
            applyInternalExternalDiagFactor<control>(angle);
        const auto off_diag_factor =
            applyInternalExternalOffDiagFactor<control>(angle);

        constexpr static auto perm = compilePermutation<PrecisionT>(
            swapRealImag(identity<packed_size>()));
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 =
                ((k << 1U) & target_wire_parity_inv) | (target_wire_parity & k);
            const size_t i1 = i0 | target_rev_wire_shift;

            const auto v0 = PrecisionAVXConcept::load(arr + i0); // target is 0
            const auto v1 = PrecisionAVXConcept::load(arr + i1); // target is 1

            PrecisionAVXConcept::store(arr + i0,
                                       diag_factor * v0 +
                                           off_diag_factor * permute<perm>(v1));
            PrecisionAVXConcept::store(arr + i1,
                                       diag_factor * v1 +
                                           off_diag_factor * permute<perm>(v0));
        }
    }

    /**
     * @brief Permutation that flips the target bit.
     */
    template <size_t target>
    static consteval auto applyExternalInternalOffDiagPerm() {
        std::array<uint8_t, packed_size> arr{};

        uint8_t s = (uint8_t{1U} << target);
        for (size_t k = 0; k < packed_size / 2; k++) {
            arr[2 * k + 0] = 2 * (k ^ s) + 1;
            arr[2 * k + 1] = 2 * (k ^ s) + 0;
        }
        return Permutation::compilePermutation<PrecisionT>(arr);
    }

    template <size_t target, typename ParamT>
    static void applyExternalInternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits, size_t control,
                                      bool inverse, ParamT angle) {
        // control qubit is external but target qubit is external
        using namespace Permutation;

        const size_t control_shift = (static_cast<size_t>(1U) << control);
        const size_t max_wire_parity = fillTrailingOnes(control);
        const size_t max_wire_parity_inv = fillLeadingOnes(control + 1);

        constexpr static auto perm = applyExternalInternalOffDiagPerm<target>();

        if (inverse) {
            angle *= -1.0;
        }
        const auto diag_factor =
            set1<PrecisionT, packed_size>(std::cos(angle / 2));
        const auto offdiag_factor =
            imagFactor<PrecisionT, packed_size>(-std::sin(angle / 2));
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 =
                ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
            const size_t i1 = i0 | control_shift;

            const auto v1 =
                PrecisionAVXConcept::load(arr + i1); // control bit is 1
            const auto w1 = Permutation::permute<perm>(v1);
            PrecisionAVXConcept::store(arr + i1,
                                       diag_factor * v1 + offdiag_factor * w1);
        }
    }

    template <typename ParamT>
    static void applyExternalExternal(std::complex<PrecisionT> *arr,
                                      const size_t num_qubits,
                                      const size_t control, const size_t target,
                                      bool inverse, ParamT angle) {
        using namespace Permutation;
        const size_t control_shift = static_cast<size_t>(1U) << control;
        const size_t target_shift = static_cast<size_t>(1U) << target;

        const size_t rev_wire_min = std::min(control, target);
        const size_t rev_wire_max = std::max(control, target);

        const size_t parity_low = fillTrailingOnes(rev_wire_min);
        const size_t parity_high = fillLeadingOnes(rev_wire_max + 1);
        const size_t parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        if (inverse) {
            angle *= -1.0;
        }

        const auto cos_factor =
            set1<PrecisionT, packed_size>(std::cos(angle / 2));
        const auto sin_factor =
            imagFactor<PrecisionT, packed_size>(-std::sin(angle / 2));

        constexpr static auto perm = compilePermutation<PrecisionT>(
            swapRealImag(identity<packed_size>()));
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 2); k += packed_size / 2) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | control_shift;
            const size_t i11 = i00 | control_shift | target_shift;

            const auto v10 = PrecisionAVXConcept::load(arr + i10); // 10
            const auto v11 = PrecisionAVXConcept::load(arr + i11); // 11

            PrecisionAVXConcept::store(
                arr + i10, cos_factor * v10 + sin_factor * permute<perm>(v11));
            PrecisionAVXConcept::store(
                arr + i11, cos_factor * v11 + sin_factor * permute<perm>(v10));
        }
    }
};
} // namespace Pennylane::LightningQubit::Gates::AVXCommon
