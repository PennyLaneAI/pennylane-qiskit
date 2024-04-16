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
 * Defines CRZ gate
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
template <typename PrecisionT, size_t packed_size> struct ApplyCRZ {
    using Precision = PrecisionT;
    using PrecisionAVXConcept = AVXConceptType<PrecisionT, packed_size>;

    constexpr static auto packed_size_ = packed_size;
    constexpr static bool symmetric = false;

    /**
     * @brief Permutation for applying `i` when the control bit is 1
     */
    template <size_t control>
    static consteval auto applyInternalImagPermutation() {
        std::array<uint8_t, packed_size> perm{};
        for (size_t k = 0; k < packed_size / 2; k++) {
            if ((k >> control) & 1U) { // if control bit is 1
                perm[2 * k + 0] = 2 * k + 1;
                perm[2 * k + 1] = 2 * k + 0;
            } else {
                perm[2 * k + 0] = 2 * k + 0;
                perm[2 * k + 1] = 2 * k + 1;
            }
        }
        return Permutation::compilePermutation<PrecisionT>(perm);
    }

    /**
     * @brief Factor for real parts
     * [1, 1, 1, 1, cos(phi/2), cos(phi/2), cos(phi/2), cos(phi/2)]
     */
    template <size_t control, size_t target, class ParamT>
    static auto applyInternalInternalRealFactor(ParamT angle) {
        std::array<PrecisionT, packed_size> arr{};

        PL_LOOP_SIMD
        for (size_t k = 0; k < packed_size / 2; k++) {
            if ((k >> control) & 1U) { // if control bit is 1
                arr[2 * k + 0] = std::cos(angle / 2);
                arr[2 * k + 1] = std::cos(angle / 2);
            } else {
                arr[2 * k + 0] = Precision{1};
                arr[2 * k + 1] = Precision{1};
            }
        }
        return setValue(arr);
    }

    /**
     * @brief Factor for imaginary parts
     * [0, 0, 0, 0, sin(phi/2), -sin(phi/2), -sin(phi/2), sin(phi/2)]
     */
    template <size_t control, size_t target, class ParamT>
    static auto applyInternalInternalImagFactor(ParamT angle) {
        std::array<PrecisionT, packed_size> arr{};

        PL_LOOP_SIMD
        for (size_t k = 0; k < packed_size / 2; k++) {
            if ((k >> control) & 1U) {    // if control bit is 1
                if ((k >> target) & 1U) { // if target bit is 1
                    arr[2 * k + 0] = -std::sin(angle / 2);
                    arr[2 * k + 1] = std::sin(angle / 2);
                } else { // if target bit is 0
                    arr[2 * k + 0] = std::sin(angle / 2);
                    arr[2 * k + 1] = -std::sin(angle / 2);
                }
            } else {
                arr[2 * k + 0] = Precision{0.0};
                arr[2 * k + 1] = Precision{0.0};
            }
        }
        return setValue(arr);
    }

    template <size_t control, size_t target, class ParamT>
    static void applyInternalInternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits, bool inverse,
                                      ParamT angle) {
        constexpr static auto perm = applyInternalImagPermutation<control>();

        if (inverse) {
            angle *= -1.0;
        }

        const auto real_factor =
            applyInternalInternalRealFactor<control, target>(angle);
        const auto imag_factor =
            applyInternalInternalImagFactor<control, target>(angle);
        PL_LOOP_PARALLEL(1)
        for (size_t n = 0; n < exp2(num_qubits); n += packed_size / 2) {
            const auto v = PrecisionAVXConcept::load(arr + n);
            PrecisionAVXConcept::store(
                arr + n,
                real_factor * v + imag_factor * Permutation::permute<perm>(v));
        }
    }

    /**
     * @brief Factor for real parts when the target bit is 1
     */
    template <size_t control, typename ParamT>
    static auto applyInternalExternalRealFactor(ParamT angle) {
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

    template <size_t control, typename ParamT>
    static auto applyInternalExternalImagFactor(ParamT angle) {
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
        constexpr static auto perm = applyInternalImagPermutation<control>();

        const size_t target_rev_wire_shift =
            (static_cast<size_t>(1U) << target);
        const size_t target_wire_parity = fillTrailingOnes(target);
        const size_t target_wire_parity_inv = fillLeadingOnes(target + 1);

        if (inverse) {
            angle *= -1.0;
        }

        const auto real_factor =
            applyInternalExternalRealFactor<control>(angle);
        const auto imag_factor =
            applyInternalExternalImagFactor<control>(angle);
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 =
                ((k << 1U) & target_wire_parity_inv) | (target_wire_parity & k);
            const size_t i1 = i0 | target_rev_wire_shift;

            const auto v0 = PrecisionAVXConcept::load(arr + i0); // target is 0
            const auto v1 = PrecisionAVXConcept::load(arr + i1); // target is 1

            PrecisionAVXConcept::store(
                arr + i0, real_factor * v0 +
                              imag_factor * Permutation::permute<perm>(v0));
            PrecisionAVXConcept::store(
                arr + i1, real_factor * v1 -
                              imag_factor * Permutation::permute<perm>(v1));
        }
    }

    /**
     * @brief Factor for real parts when the control bit is 1
     */
    template <size_t target, typename ParamT>
    static auto applyExternalInternalRealFactor(ParamT angle) {
        std::array<Precision, packed_size> arr{};
        arr.fill(std::cos(angle / 2));
        return setValue(arr);
    }

    /**
     * @brief Factor for real parts when the control bit is 1
     */
    template <size_t target, typename ParamT>
    static auto applyExternalInternalImagFactor(ParamT angle) {
        std::array<Precision, packed_size> arr{};
        PL_LOOP_SIMD
        for (size_t k = 0; k < packed_size / 2; k++) {
            if ((k >> target) & 1U) { // target bit is 1
                arr[2 * k + 0] = -std::sin(angle / 2);
                arr[2 * k + 1] = std::sin(angle / 2);
            } else {
                arr[2 * k + 0] = std::sin(angle / 2);
                arr[2 * k + 1] = -std::sin(angle / 2);
            }
        }
        return setValue(arr);
    }

    template <size_t target, typename ParamT>
    static void applyExternalInternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits, size_t control,
                                      bool inverse, ParamT angle) {
        using namespace Permutation;

        const size_t control_shift = (static_cast<size_t>(1U) << control);
        const size_t max_wire_parity = fillTrailingOnes(control);
        const size_t max_wire_parity_inv = fillLeadingOnes(control + 1);

        constexpr static auto perm = compilePermutation<Precision>(
            swapRealImag(identity<packed_size>()));

        if (inverse) {
            angle *= -1.0;
        }
        const auto real_factor = applyExternalInternalRealFactor<target>(angle);
        const auto imag_factor = applyExternalInternalImagFactor<target>(angle);
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 =
                ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
            const size_t i1 = i0 | control_shift;

            const auto v1 =
                PrecisionAVXConcept::load(arr + i1); // control bit is 1
            const auto w1 = Permutation::permute<perm>(v1);
            PrecisionAVXConcept::store(arr + i1,
                                       real_factor * v1 + imag_factor * w1);
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

        constexpr static auto perm = compilePermutation<Precision>(
            swapRealImag(identity<packed_size>()));

        if (inverse) {
            angle *= -1.0;
        }

        const auto real_factor =
            set1<PrecisionT, packed_size>(std::cos(angle / 2));
        const auto imag_factor_p =
            imagFactor<PrecisionT, packed_size>(-std::sin(angle / 2));
        const auto imag_factor_m = -imag_factor_p;
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 2); k += packed_size / 2) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | control_shift;
            const size_t i11 = i00 | control_shift | target_shift;

            const auto v10 = PrecisionAVXConcept::load(arr + i10); // 10
            const auto v11 = PrecisionAVXConcept::load(arr + i11); // 11

            PrecisionAVXConcept::store(arr + i10,
                                       real_factor * v10 +
                                           imag_factor_p * permute<perm>(v10));
            PrecisionAVXConcept::store(arr + i11,
                                       real_factor * v11 +
                                           imag_factor_m * permute<perm>(v11));
        }
    }
};
} // namespace Pennylane::LightningQubit::Gates::AVXCommon
