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
 * Defines PhaseShift generator
 */
#pragma once
#include "AVXUtil.hpp"
#include "BitUtil.hpp"
#include "Permutation.hpp"
#include "Util.hpp"

#include <complex>

namespace Pennylane::LightningQubit::Gates::AVXCommon {
template <typename PrecisionT, size_t packed_size>
struct ApplyGeneratorPhaseShift {
    using Precision = PrecisionT;
    using PrecisionAVXConcept =
        typename AVXConcept<PrecisionT, packed_size>::Type;

    constexpr static size_t packed_size_ = packed_size;

    template <size_t rev_wire>
    static consteval auto factorInternal() ->
        typename PrecisionAVXConcept::IntrinsicType {
        std::array<PrecisionT, packed_size> factors{};
        for (size_t k = 0; k < packed_size_ / 2; k++) {
            if (((k >> rev_wire) & size_t{1U}) == 0) {
                factors[2 * k + 0] = 0.0;
                factors[2 * k + 1] = 0.0;
            } else {
                factors[2 * k + 0] = 1.0;
                factors[2 * k + 1] = 1.0;
            }
        }
        return setValue(factors);
    }

    template <size_t rev_wire>
    static auto applyInternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              [[maybe_unused]] bool inverse) -> PrecisionT {
        constexpr auto factor = factorInternal<rev_wire>();
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < (1U << num_qubits); k += packed_size / 2) {
            const auto v = PrecisionAVXConcept::load(arr + k);
            PrecisionAVXConcept::store(arr + k, factor * v);
        }
        return static_cast<PrecisionT>(1.0);
    }

    static auto applyExternal(std::complex<PrecisionT> *arr,
                              const size_t num_qubits, const size_t rev_wire,
                              [[maybe_unused]] bool inverse) -> PrecisionT {
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        constexpr auto zero =
            typename PrecisionAVXConcept::IntrinsicType{PrecisionT{0.0}};
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            PrecisionAVXConcept::store(arr + i0, zero);
        }
        return static_cast<PrecisionT>(1.0);
    }
};
} // namespace Pennylane::LightningQubit::Gates::AVXCommon
