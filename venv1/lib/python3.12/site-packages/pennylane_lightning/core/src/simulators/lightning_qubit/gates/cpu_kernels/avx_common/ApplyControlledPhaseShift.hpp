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
 * Defines ControlledPhaseShift gate
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
struct ApplyControlledPhaseShift {
    using PrecisionAVXConcept =
        typename AVXConcept<PrecisionT, packed_size>::Type;
    using Precision = PrecisionT;

    constexpr static auto perm = Permutation::compilePermutation<PrecisionT>(
        Permutation::swapRealImag(Permutation::identity<packed_size>()));
    constexpr static size_t packed_size_ = packed_size;
    constexpr static bool symmetric = true;

    /**
     * @brief Permutation applying imaginary `i` to |11>
     */
    template <size_t rev_wire0, size_t rev_wire1>
    static consteval auto applyInternalInternalPermutation() {
        // Swap real and imaginary part of 11
        std::array<uint8_t, packed_size> perm{};
        for (size_t k = 0; k < (packed_size / 2); k++) {
            if ((((k >> rev_wire0) & 1U) & ((k >> rev_wire1) & 1U)) == 1) {
                // Only swap real and image for 11
                perm[2 * k + 0] = 2 * k + 1;
                perm[2 * k + 1] = 2 * k + 0;
            } else {
                perm[2 * k + 0] = 2 * k + 0;
                perm[2 * k + 1] = 2 * k + 1;
            }
        }

        return Permutation::compilePermutation<PrecisionT>(perm);
    }

    template <size_t rev_wire0, size_t rev_wire1, class ParamT>
    static void applyInternalInternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits, bool inverse,
                                      ParamT angle) {
        const auto isin = inverse ? -std::sin(angle) : std::sin(angle);

        const auto real_factor = [angle]() {
            std::array<PrecisionT, packed_size> arr{};
            PL_LOOP_SIMD
            for (size_t k = 0; k < (packed_size / 2); k++) {
                if ((((k >> rev_wire0) & 1U) & ((k >> rev_wire1) & 1U)) == 1) {
                    // for 11
                    arr[2 * k + 0] = std::cos(angle);
                    arr[2 * k + 1] = std::cos(angle);
                } else {
                    arr[2 * k + 0] = 1.0;
                    arr[2 * k + 1] = 1.0;
                }
            }
            return setValue(arr);
        }();
        const auto imag_factor = [isin]() {
            std::array<PrecisionT, packed_size> arr{};
            PL_LOOP_SIMD
            for (size_t k = 0; k < (packed_size / 2); k++) {
                if ((((k >> rev_wire0) & 1U) & ((k >> rev_wire1) & 1U)) == 1) {
                    // for 11
                    arr[2 * k + 0] = -isin;
                    arr[2 * k + 1] = isin;
                } else {
                    arr[2 * k + 0] = 0.0;
                    arr[2 * k + 1] = 0.0;
                }
            }
            return setValue(arr);
        }();

        constexpr static auto perm =
            applyInternalInternalPermutation<rev_wire0, rev_wire1>();
        PL_LOOP_PARALLEL(1)
        for (size_t n = 0; n < exp2(num_qubits); n += packed_size / 2) {
            const auto v = PrecisionAVXConcept::load(arr + n);

            const auto prod_cos = real_factor * v;
            const auto prod_sin = imag_factor * Permutation::permute<perm>(v);

            PrecisionAVXConcept::store(arr + n, prod_cos + prod_sin);
        }
    }

    /**
     * @brief Permutation applying product `i` when the target bit is 1
     */
    template <size_t min_rev_wire>
    static consteval auto applyInternalExternalPermutation() {
        std::array<uint8_t, packed_size> perm{};
        for (size_t k = 0; k < (packed_size / 2); k++) {
            if (((k >> min_rev_wire) & 1U) == 1) {
                // Only swap real and imag when 1
                perm[2 * k + 0] = 2 * k + 1;
                perm[2 * k + 1] = 2 * k + 0;
            } else {
                perm[2 * k + 0] = 2 * k + 0;
                perm[2 * k + 1] = 2 * k + 1;
            }
        }

        return Permutation::compilePermutation<PrecisionT>(perm);
    }

    template <size_t min_rev_wire, class ParamT>
    static void applyInternalExternal(std::complex<PrecisionT> *arr,
                                      size_t num_qubits, size_t max_rev_wire,
                                      bool inverse, ParamT angle) {
        const size_t max_rev_wire_shift =
            (static_cast<size_t>(1U) << max_rev_wire);
        const size_t max_wire_parity = fillTrailingOnes(max_rev_wire);
        const size_t max_wire_parity_inv = fillLeadingOnes(max_rev_wire + 1);

        const auto isin = inverse ? -std::sin(angle) : std::sin(angle);
        const auto real_factor = [angle]() {
            std::array<Precision, packed_size> arr{};
            PL_LOOP_SIMD
            for (size_t k = 0; k < (packed_size / 2); k++) {
                if (((k >> min_rev_wire) & 1U) == 1) {
                    // for 11
                    arr[2 * k + 0] = std::cos(angle);
                    arr[2 * k + 1] = std::cos(angle);
                } else {
                    arr[2 * k + 0] = 1.0;
                    arr[2 * k + 1] = 1.0;
                }
            }

            return setValue(arr);
        }();

        const auto imag_factor = [isin]() {
            std::array<Precision, packed_size> arr{};
            PL_LOOP_SIMD
            for (size_t k = 0; k < (packed_size / 2); k++) {
                if (((k >> min_rev_wire) & 1U) == 1) {
                    // for 11
                    arr[2 * k + 0] = -isin;
                    arr[2 * k + 1] = isin;
                } else {
                    arr[2 * k + 0] = 0.0;
                    arr[2 * k + 1] = 0.0;
                }
            }

            return setValue(arr);
        }();

        constexpr static auto perm =
            applyInternalExternalPermutation<min_rev_wire>();
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 1); k += packed_size / 2) {
            const size_t i0 =
                ((k << 1U) & max_wire_parity_inv) | (max_wire_parity & k);
            const size_t i1 = i0 | max_rev_wire_shift;

            const auto v1 = PrecisionAVXConcept::load(arr + i1);

            const auto prod_real = real_factor * v1;
            const auto prod_imag = imag_factor * Permutation::permute<perm>(v1);

            PrecisionAVXConcept::store(arr + i1, prod_real + prod_imag);
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

        const auto isin = inverse ? -std::sin(angle) : std::sin(angle);

        const auto real_cos = set1<PrecisionT, packed_size>(std::cos(angle));
        const auto imag_sin = imagFactor<PrecisionT, packed_size>(isin);

        constexpr static auto perm = compilePermutation<Precision>(
            swapRealImag(identity<packed_size>()));
        PL_LOOP_PARALLEL(1)
        for (size_t k = 0; k < exp2(num_qubits - 2); k += packed_size / 2) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

            const auto v11 = PrecisionAVXConcept::load(arr + i11); // 11

            const auto prod_cos11 = real_cos * v11;
            const auto prod_isin11 = imag_sin * permute<perm>(v11);

            PrecisionAVXConcept::store(arr + i11, prod_cos11 + prod_isin11);
        }
    }
};
} // namespace Pennylane::LightningQubit::Gates::AVXCommon
