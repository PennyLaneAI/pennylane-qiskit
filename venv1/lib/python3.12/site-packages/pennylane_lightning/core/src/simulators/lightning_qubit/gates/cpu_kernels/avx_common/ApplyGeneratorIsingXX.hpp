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
 * Defines IsingXX generator
 */
#pragma once
#include "AVXConceptType.hpp"
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Permutation.hpp"
#include "Util.hpp"

#include <complex>

namespace Pennylane::LightningQubit::Gates::AVXCommon {
template <typename PrecisionT, size_t packed_size>
struct ApplyGeneratorIsingXX {
    using Precision = PrecisionT;
    using PrecisionAVXConcept =
        typename AVXConcept<PrecisionT, packed_size>::Type;

    constexpr static size_t packed_size_ = packed_size;
    constexpr static bool symmetric = true;

    template <size_t rev_wire0, size_t rev_wire1>
    static auto applyInternalInternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits,
                                      [[maybe_unused]] bool adj) -> PrecisionT {
        using namespace Permutation;
        constexpr static auto perm = compilePermutation<Precision, packed_size>(
            flip(flip(identity<packed_size>(), rev_wire0), rev_wire1));
        PL_LOOP_PARALLEL(1)
        for (size_t n = 0; n < exp2(num_qubits); n += packed_size / 2) {
            const auto v = PrecisionAVXConcept::load(arr + n);
            PrecisionAVXConcept::store(arr + n, permute<perm>(v));
        }
        return -static_cast<PrecisionT>(
            0.5); // NOLINT(readability-magic-numbers)
    }

    template <size_t min_rev_wire>
    static auto applyInternalExternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits, size_t max_rev_wire,
                                      [[maybe_unused]] bool adj) -> PrecisionT {
        using namespace Permutation;

        const size_t max_rev_wire_shift =
            (static_cast<size_t>(1U) << max_rev_wire);
        const size_t max_wire_parity = fillTrailingOnes(max_rev_wire);
        const size_t max_wire_parity_inv = fillLeadingOnes(max_rev_wire + 1);

        constexpr static auto perm = compilePermutation<PrecisionT>(
            flip(identity<packed_size>(), min_rev_wire));
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 =
                ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
            const size_t i1 = i0 | max_rev_wire_shift;

            const auto v0 = PrecisionAVXConcept::load(arr + i0);
            const auto v1 = PrecisionAVXConcept::load(arr + i1);

            PrecisionAVXConcept::store(arr + i0, permute<perm>(v1));
            PrecisionAVXConcept::store(arr + i1, permute<perm>(v0));
        }
        return -static_cast<PrecisionT>(
            0.5); // NOLINT(readability-magic-numbers)
    }

    static auto applyExternalExternal(std::complex<PrecisionT> *arr,
                                      const size_t num_qubits,
                                      const size_t rev_wire0,
                                      const size_t rev_wire1,
                                      [[maybe_unused]] bool adj) -> PrecisionT {
        using namespace Permutation;

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

            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i01 = i00 | rev_wire0_shift;
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            const auto v00 = PrecisionAVXConcept::load(arr + i00); // 00
            const auto v01 = PrecisionAVXConcept::load(arr + i01); // 01
            const auto v10 = PrecisionAVXConcept::load(arr + i10); // 10
            const auto v11 = PrecisionAVXConcept::load(arr + i11); // 11

            PrecisionAVXConcept::store(arr + i00, v11);
            PrecisionAVXConcept::store(arr + i01, v10);
            PrecisionAVXConcept::store(arr + i10, v01);
            PrecisionAVXConcept::store(arr + i11, v00);
        }
        return -static_cast<PrecisionT>(
            0.5); // NOLINT(readability-magic-numbers)
    }
};
} // namespace Pennylane::LightningQubit::Gates::AVXCommon
