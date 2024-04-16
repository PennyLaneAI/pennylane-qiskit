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
 * Defines SWAP gate
 */
#pragma once
#include "AVXConceptType.hpp"
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Blender.hpp"
#include "Permutation.hpp"
#include "Util.hpp"

#include <complex>

namespace Pennylane::LightningQubit::Gates::AVXCommon {
template <typename PrecisionT, size_t packed_size> struct ApplySWAP {
    using Precision = PrecisionT;
    using PrecisionAVXConcept =
        typename AVXConcept<PrecisionT, packed_size>::Type;

    constexpr static size_t packed_size_ = packed_size;
    constexpr static bool symmetric = true;

    /**
     * @brief Permutation that swaps bits in two wires
     */
    template <size_t rev_wire0, size_t rev_wire1>
    static consteval auto applyInternalInternalPermutation() {
        const auto identity_perm = Permutation::identity<packed_size>();
        std::array<uint8_t, packed_size> perm{};
        for (size_t i = 0; i < packed_size / 2; i++) {
            // swap rev_wire1 and rev_wire0 bits
            const size_t b = ((i >> rev_wire0) ^ (i >> rev_wire1)) & 1U;
            const size_t j = i ^ ((b << rev_wire0) | (b << rev_wire1));
            perm[2 * i + 0] = identity_perm[2 * j + 0];
            perm[2 * i + 1] = identity_perm[2 * j + 1];
        }
        return Permutation::compilePermutation<PrecisionT, packed_size>(perm);
    }

    template <size_t rev_wire0, size_t rev_wire1>
    static void applyInternalInternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits,
                                      [[maybe_unused]] bool inverse) {
        using namespace Permutation;
        constexpr static auto perm =
            applyInternalInternalPermutation<rev_wire0, rev_wire1>();
        PL_LOOP_PARALLEL(1)
        for (size_t n = 0; n < exp2(num_qubits); n += packed_size / 2) {
            const auto v = PrecisionAVXConcept::load(arr + n);
            PrecisionAVXConcept::store(arr + n, permute<perm>(v));
        }
    }

    /**
     * @brief Setting a mask. Mask is 1 if bits in min_rev_wire is set
     */
    template <size_t min_rev_wire> static consteval auto createMask0() {
        std::array<bool, packed_size> m{};
        for (size_t i = 0; i < packed_size / 2; i++) {
            if ((i & (1U << min_rev_wire)) != 0) {
                m[2 * i + 0] = true;
                m[2 * i + 1] = true;
            } else {
                m[2 * i + 0] = false;
                m[2 * i + 1] = false;
            }
        }
        return compileMask<PrecisionT, packed_size>(m);
    }

    /**
     * @brief Setting a mask. Mask is 1 if bits in min_rev_wire is unset
     */
    template <size_t min_rev_wire> static consteval auto createMask1() {
        std::array<bool, packed_size> m = {};
        for (size_t i = 0; i < packed_size / 2; i++) {
            if ((i & (1U << min_rev_wire)) != 0) {
                m[2 * i + 0] = false;
                m[2 * i + 1] = false;
            } else {
                m[2 * i + 0] = true;
                m[2 * i + 1] = true;
            }
        }
        return compileMask<PrecisionT, packed_size>(m);
    }

    template <size_t min_rev_wire>
    static void applyInternalExternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits, size_t max_rev_wire,
                                      [[maybe_unused]] bool inverse) {
        using namespace Permutation;

        const size_t max_rev_wire_shift =
            (static_cast<size_t>(1U) << max_rev_wire);
        const size_t max_wire_parity = fillTrailingOnes(max_rev_wire);
        const size_t max_wire_parity_inv = fillLeadingOnes(max_rev_wire + 1);

        constexpr static auto compiled_mask0 = createMask0<min_rev_wire>();
        constexpr static auto compiled_mask1 = createMask1<min_rev_wire>();
        constexpr static auto compiled_perm = compilePermutation<PrecisionT>(
            flip(identity<packed_size>(), min_rev_wire));
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 =
                ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
            const size_t i1 = i0 | max_rev_wire_shift;

            const auto v0 = PrecisionAVXConcept::load(arr + i0);
            const auto v1 = PrecisionAVXConcept::load(arr + i1);

            const auto w0 = maskPermute<compiled_perm, compiled_mask0>(v0, v1);
            const auto w1 = maskPermute<compiled_perm, compiled_mask1>(v1, v0);

            PrecisionAVXConcept::store(arr + i0, w0);
            PrecisionAVXConcept::store(arr + i1, w1);
        }
    }

    static void applyExternalExternal(std::complex<PrecisionT> *arr,
                                      const size_t num_qubits,
                                      const size_t rev_wire0,
                                      const size_t rev_wire1,
                                      [[maybe_unused]] bool inverse) {
        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
        const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

        const size_t parity_low = fillTrailingOnes(rev_wire_min);
        const size_t parity_high = fillLeadingOnes(rev_wire_max + 1);
        const size_t parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 2); k += packed_size / 2) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i10 = i00 | rev_wire1_shift;

            const auto v01 = PrecisionAVXConcept::load(arr + i01); // 01
            const auto v10 = PrecisionAVXConcept::load(arr + i10); // 10
            PrecisionAVXConcept::store(arr + i10, v01);
            PrecisionAVXConcept::store(arr + i01, v10);
        }
    }
};
} // namespace Pennylane::LightningQubit::Gates::AVXCommon
