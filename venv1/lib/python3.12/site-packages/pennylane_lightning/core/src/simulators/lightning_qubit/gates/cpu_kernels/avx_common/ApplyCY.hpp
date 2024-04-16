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
 * Defines CY gate
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
template <typename PrecisionT, size_t packed_size> struct ApplyCY {
    using Precision = PrecisionT;
    using PrecisionAVXConcept = AVXConceptType<PrecisionT, packed_size>;

    constexpr static auto packed_size_ = packed_size;
    constexpr static bool symmetric = false;

    /**
     * @brief Permutation for multiplying `i` and flip the target bit if control
     * is 1
     */
    template <size_t control, size_t target>
    static consteval auto applyInternalInternalPermuation() {
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
     * @brief Factor to applying `-i` and `i`
     */
    template <size_t control, size_t target>
    static consteval auto applyInternalInternalFactor() {
        std::array<PrecisionT, packed_size> signs{};
        for (size_t k = 0; k < packed_size / 2; k++) {
            if ((k >> control) & 1U) {    // if control bit is 1
                if ((k >> target) & 1U) { // if target bit is 1 (was 0) == -> i
                    signs[2 * k + 0] = Precision{-1.0};
                    signs[2 * k + 1] = Precision{1.0};
                } else { // if target bit is 0 (was 1) -> -i
                    signs[2 * k + 0] = Precision{1.0};
                    signs[2 * k + 1] = Precision{-1.0};
                }
            } else {
                signs[2 * k + 0] = Precision{1.0};
                signs[2 * k + 1] = Precision{1.0};
            }
        }
        return setValue(signs);
    }

    template <size_t control, size_t target>
    static void applyInternalInternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits,
                                      [[maybe_unused]] bool inverse) {
        constexpr static auto perm =
            applyInternalInternalPermuation<control, target>();
        constexpr static auto factor =
            applyInternalInternalFactor<control, target>();
        PL_LOOP_PARALLEL(1)
        for (size_t n = 0; n < exp2(num_qubits); n += packed_size / 2) {
            const auto v = PrecisionAVXConcept::load(arr + n);
            PrecisionAVXConcept::store(arr + n,
                                       factor * Permutation::permute<perm>(v));
        }
    }

    /**
     * @brief Mask for blending. Using this mask, blending v0 and v1 will output
     * v0 if the control bit is 0 v1 otherwise.
     */
    template <size_t control>
    static consteval auto applyInternalExternalMask() {
        std::array<bool, packed_size> mask{};
        for (size_t k = 0; k < packed_size / 2; k++) {
            if ((k >> control) & 1U) {
                mask[2 * k + 0] = true;
                mask[2 * k + 1] = true;
            }
        }
        return compileMask<PrecisionT>(mask);
    }

    /**
     * @brief Permutation when the target bit is 1
     */
    template <size_t control>
    static consteval auto applyInternalExternalPermutation() {
        std::array<uint8_t, packed_size> permutation{};
        for (size_t k = 0; k < packed_size / 2; k++) {
            if ((k >> control) & 1U) { // if control bit is 1
                permutation[2 * k + 0] = 2 * k + 1;
                permutation[2 * k + 1] = 2 * k + 0;
            } else {
                permutation[2 * k + 0] = 2 * k + 0;
                permutation[2 * k + 1] = 2 * k + 1;
            }
        }
        return Permutation::compilePermutation<Precision>(permutation);
    }

    /**
     * @brief Sign factor when the target bit is 0
     */
    template <size_t control>
    static consteval auto applyInternalExternalSign_target0() {
        // Signs when the target is 0
        std::array<Precision, packed_size> signs = {
            1.0,
        };
        for (size_t k = 0; k < packed_size / 2; k++) {
            if ((k >> control) & 1U) {
                signs[2 * k + 0] = 1.0;
                signs[2 * k + 1] = -1.0;
            } else {
                signs[2 * k + 0] = 1.0;
                signs[2 * k + 1] = 1.0;
            }
        }
        return setValue(signs);
    }

    /**
     * @brief Sign factor when the target bit is 1
     */
    template <size_t control>
    static consteval auto applyInternalExternalSign_target1() {
        // Signs when the target is 1
        std::array<Precision, packed_size> signs = {
            1.0,
        };
        for (size_t k = 0; k < packed_size / 2; k++) {
            if ((k >> control) & 1U) {
                signs[2 * k + 0] = -1.0;
                signs[2 * k + 1] = 1.0;
            } else {
                signs[2 * k + 0] = 1.0;
                signs[2 * k + 1] = 1.0;
            }
        }
        return setValue(signs);
    }

    /**
     * @brief Implementation for the case where the control qubit acts
     * on internal wires (inside of packed bytes) but the target acts on
     * external wires.
     */
    template <size_t control>
    static void applyInternalExternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits, size_t target,
                                      [[maybe_unused]] bool inverse) {
        // control qubit is internal but target qubit is external
        using namespace Permutation;

        const size_t target_rev_wire_shift =
            (static_cast<size_t>(1U) << target);
        const size_t target_wire_parity = fillTrailingOnes(target);
        const size_t target_wire_parity_inv = fillLeadingOnes(target + 1);

        constexpr static auto mask = applyInternalExternalMask<control>();
        constexpr static auto perm =
            applyInternalExternalPermutation<control>();

        constexpr static auto sign0 =
            applyInternalExternalSign_target0<control>();
        constexpr static auto sign1 =
            applyInternalExternalSign_target1<control>();
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 =
                ((k << 1U) & target_wire_parity_inv) | (target_wire_parity & k);
            const size_t i1 = i0 | target_rev_wire_shift;

            const auto v0 = PrecisionAVXConcept::load(arr + i0); // target is 0
            const auto v1 = PrecisionAVXConcept::load(arr + i1); // target is 1

            const auto w0 = blend<mask>(v0, v1);
            const auto w1 = blend<mask>(v1, v0);

            PrecisionAVXConcept::store(arr + i0, sign0 * permute<perm>(w0));
            PrecisionAVXConcept::store(arr + i1, sign1 * permute<perm>(w1));
        }
    }

    /**
     * @brief Sign factor when the control bit is 1.
     */
    template <size_t target> static consteval auto applyExternalInternalSign() {
        std::array<Precision, packed_size> signs = {
            1.0,
        };
        for (size_t k = 0; k < packed_size / 2; k++) {
            if ((k >> target) & 1U) { // target is 1 (was 0)
                signs[2 * k + 0] = -1.0;
                signs[2 * k + 1] = 1.0;
            } else {
                signs[2 * k + 0] = 1.0;
                signs[2 * k + 1] = -1.0;
            }
        }
        return setValue(signs);
    }

    template <size_t target>
    static void applyExternalInternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits, size_t control,
                                      [[maybe_unused]] bool inverse) {
        // control qubit is external but target qubit is external
        using namespace Permutation;

        const size_t control_shift = (static_cast<size_t>(1U) << control);
        const size_t max_wire_parity = fillTrailingOnes(control);
        const size_t max_wire_parity_inv = fillLeadingOnes(control + 1);

        constexpr static auto perm = compilePermutation<Precision>(
            swapRealImag(flip(identity<packed_size>(), target)));
        constexpr static auto factor = applyExternalInternalSign<target>();
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 =
                ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
            const size_t i1 = i0 | control_shift;

            const auto v1 = PrecisionAVXConcept::load(arr + i1); // control is 1
            const auto w1 = Permutation::permute<perm>(v1);
            PrecisionAVXConcept::store(arr + i1, factor * w1);
        }
    }

    static void applyExternalExternal(std::complex<PrecisionT> *arr,
                                      const size_t num_qubits,
                                      const size_t control, const size_t target,
                                      [[maybe_unused]] bool inverse) {
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
        constexpr static auto factor = imagFactor<PrecisionT, packed_size>();
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 2); k += packed_size / 2) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | control_shift;
            const size_t i11 = i00 | control_shift | target_shift;

            const auto v10 = PrecisionAVXConcept::load(arr + i10); // 10
            const auto v11 = PrecisionAVXConcept::load(arr + i11); // 11

            PrecisionAVXConcept::store(arr + i10, -factor * permute<perm>(v11));
            PrecisionAVXConcept::store(arr + i11, factor * permute<perm>(v10));
        }
    }
};
} // namespace Pennylane::LightningQubit::Gates::AVXCommon
