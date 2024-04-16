// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include <Kokkos_Core.hpp>

#include "BitUtil.hpp"
#include "BitUtilKokkos.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using Pennylane::LightningKokkos::Util::one;
using Pennylane::LightningKokkos::Util::wires2Parity;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Functors {

template <class PrecisionT> struct getExpectationValueIdentityFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    getExpectationValueIdentityFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
        [[maybe_unused]] std::size_t num_qubits,
        [[maybe_unused]] const std::vector<size_t> &wires) {
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {
        expval += real(conj(arr[k]) * arr[k]);
    }
};

template <class PrecisionT> struct getExpectationValuePauliXFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    getExpectationValuePauliXFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;

        expval += real(conj(arr[i0]) * arr[i1]);
        expval += real(conj(arr[i1]) * arr[i0]);
    }
};

template <class PrecisionT> struct getExpectationValuePauliYFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    getExpectationValuePauliYFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        const auto v0 = arr[i0];
        const auto v1 = arr[i1];

        expval += real(conj(arr[i0]) *
                       Kokkos::complex<PrecisionT>{imag(v1), -real(v1)});
        expval += real(conj(arr[i1]) *
                       Kokkos::complex<PrecisionT>{-imag(v0), real(v0)});
    }
};

template <class PrecisionT> struct getExpectationValuePauliZFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    getExpectationValuePauliZFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        expval += real(conj(arr[i1]) * (-arr[i1]));
        expval += real(conj(arr[i0]) * (arr[i0]));
    }
};

template <class PrecisionT> struct getExpectationValueHadamardFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    getExpectationValueHadamardFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        const Kokkos::complex<PrecisionT> v0 = arr[i0];
        const Kokkos::complex<PrecisionT> v1 = arr[i1];

        expval += real(M_SQRT1_2 *
                       (conj(arr[i0]) * (v0 + v1) + conj(arr[i1]) * (v0 - v1)));
    }
};

template <class PrecisionT> struct getExpValMultiQubitOpFunctor {
    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;
    using ScratchViewComplex =
        Kokkos::View<ComplexT *,
                     Kokkos::DefaultExecutionSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using MemberType = Kokkos::TeamPolicy<>::member_type;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    KokkosIntVector wires;
    KokkosIntVector parity;
    KokkosIntVector rev_wire_shifts;
    std::size_t dim;
    std::size_t num_qubits;

    getExpValMultiQubitOpFunctor(const KokkosComplexVector &arr_,
                                 std::size_t num_qubits_,
                                 const KokkosComplexVector &matrix_,
                                 const std::vector<std::size_t> &wires_) {
        Kokkos::View<const std::size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            wires_host(wires_.data(), wires_.size());
        Kokkos::resize(wires, wires_.size());
        Kokkos::deep_copy(wires, wires_host);

        dim = one << wires_.size();
        num_qubits = num_qubits_;
        arr = arr_;
        matrix = matrix_;
        std::tie(parity, rev_wire_shifts) = wires2Parity(num_qubits_, wires_);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const MemberType &teamMember, PrecisionT &expval) const {
        const std::size_t k = teamMember.league_rank();
        PrecisionT tempExpVal = 0.0;
        ScratchViewComplex coeffs_in(teamMember.team_scratch(0), dim);
        if (teamMember.team_rank() == 0) {
            std::size_t idx = (k & parity(0));
            for (std::size_t i = 1; i < parity.size(); i++) {
                idx |= ((k << i) & parity(i));
            }
            coeffs_in(0) = arr(idx);

            Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember, 1, dim),
                                 [&](const std::size_t inner_idx) {
                                     std::size_t index = idx;
                                     for (std::size_t i = 0; i < wires.size();
                                          i++) {
                                         if ((inner_idx & (one << i)) != 0) {
                                             index |= rev_wire_shifts(i);
                                         }
                                     }
                                     coeffs_in(inner_idx) = arr(index);
                                 });
        }
        teamMember.team_barrier();
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(teamMember, dim),
            [&](const std::size_t i, PrecisionT &innerExpVal) {
                const std::size_t base_idx = i * dim;
                ComplexT tmp{0.0};
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(teamMember, dim),
                    [&](const std::size_t j, ComplexT &isum) {
                        isum = isum + matrix(base_idx + j) * coeffs_in(j);
                    },
                    tmp);
                innerExpVal += real(conj(coeffs_in(i)) * tmp);
            },
            tempExpVal);
        if (teamMember.team_rank() == 0) {
            expval += tempExpVal;
        }
    }
};

template <class PrecisionT> struct getExpectationValueSparseFunctor {
    using KokkosComplexVector = Kokkos::View<Kokkos::complex<PrecisionT> *>;
    using KokkosSizeTVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector data;
    KokkosSizeTVector indices;
    KokkosSizeTVector indptr;
    std::size_t length;

    getExpectationValueSparseFunctor(KokkosComplexVector arr_,
                                     const KokkosComplexVector data_,
                                     const KokkosSizeTVector indices_,
                                     const KokkosSizeTVector indptr_) {
        length = indices_.size();
        indices = indices_;
        indptr = indptr_;
        data = data_;
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t row, PrecisionT &expval) const {
        for (size_t j = indptr[row]; j < indptr[row + 1]; j++) {
            expval += real(conj(arr[row]) * data[j] * arr[indices[j]]);
        }
    }
};

template <class PrecisionT> struct getExpVal1QubitOpFunctor {
    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    const std::size_t n_wires = 1;

    const std::size_t dim = one << n_wires;
    std::size_t num_qubits;
    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    getExpVal1QubitOpFunctor(
        const KokkosComplexVector &arr_, const std::size_t num_qubits_,
        const KokkosComplexVector &matrix_,
        [[maybe_unused]] const std::vector<std::size_t> &wires_) {
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;
        rev_wire = num_qubits - wires_[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;

        expval += real(conj(arr(i0)) *
                       (matrix(0B00) * arr(i0) + matrix(0B01) * arr(i1)));
        expval += real(conj(arr(i1)) *
                       (matrix(0B10) * arr(i0) + matrix(0B11) * arr(i1)));
    }
};

#define EXPVALENTRY2(xx, yy) xx << 2 | yy
#define EXPVALTERM2(xx, yy, iyy) matrix(EXPVALENTRY2(xx, yy)) * arr(iyy)
#define EXPVAL2(ixx, xx)                                                       \
    conj(arr(ixx)) *                                                           \
        (EXPVALTERM2(xx, 0B00, i00) + EXPVALTERM2(xx, 0B01, i01) +             \
         EXPVALTERM2(xx, 0B10, i10) + EXPVALTERM2(xx, 0B11, i11))

template <class PrecisionT> struct getExpVal2QubitOpFunctor {
    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    const std::size_t n_wires = 2;

    const std::size_t dim = one << n_wires;
    std::size_t num_qubits;
    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    getExpVal2QubitOpFunctor(
        const KokkosComplexVector &arr_, const std::size_t num_qubits_,
        const KokkosComplexVector &matrix_,
        [[maybe_unused]] const std::vector<std::size_t> &wires_) {
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;

        rev_wire0 = num_qubits - wires_[1] - 1;
        rev_wire1 = num_qubits - wires_[0] - 1;
        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);
        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        expval += real(EXPVAL2(i00, 0B00));
        expval += real(EXPVAL2(i10, 0B10));
        expval += real(EXPVAL2(i01, 0B01));
        expval += real(EXPVAL2(i11, 0B11));
    }
};

#define EXPVALENTRY3(xx, yy) xx << 3 | yy
#define EXPVALTERM3(xx, yy, iyy) matrix(EXPVALENTRY3(xx, yy)) * arr(iyy)
#define EXPVAL3(ixx, xx)                                                       \
    conj(arr(ixx)) *                                                           \
        (EXPVALTERM3(xx, 0B000, i000) + EXPVALTERM3(xx, 0B001, i001) +         \
         EXPVALTERM3(xx, 0B010, i010) + EXPVALTERM3(xx, 0B011, i011) +         \
         EXPVALTERM3(xx, 0B100, i100) + EXPVALTERM3(xx, 0B101, i101) +         \
         EXPVALTERM3(xx, 0B110, i110) + EXPVALTERM3(xx, 0B111, i111))

template <class PrecisionT> struct getExpVal3QubitOpFunctor {
    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    KokkosIntVector wires;
    KokkosIntVector parity;
    KokkosIntVector rev_wire_shifts;

    const std::size_t n_wires = 3;

    const std::size_t dim = one << n_wires;
    std::size_t num_qubits;

    getExpVal3QubitOpFunctor(const KokkosComplexVector &arr_,
                             const std::size_t num_qubits_,
                             const KokkosComplexVector &matrix_,
                             const std::vector<std::size_t> &wires_) {
        Kokkos::View<const std::size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            wires_host(wires_.data(), wires_.size());
        Kokkos::resize(wires, wires_.size());
        Kokkos::deep_copy(wires, wires_host);

        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;
        std::tie(parity, rev_wire_shifts) = wires2Parity(num_qubits_, wires_);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {
        std::size_t i000 = (k & parity(0));
        for (std::size_t i = 1; i < parity.size(); i++) {
            i000 |= ((k << i) & parity(i));
        }

        std::size_t i001 = i000 | rev_wire_shifts(0);
        std::size_t i010 = i000 | rev_wire_shifts(1);
        std::size_t i011 = i000 | rev_wire_shifts(0) | rev_wire_shifts(1);
        std::size_t i100 = i000 | rev_wire_shifts(2);
        std::size_t i101 = i000 | rev_wire_shifts(0) | rev_wire_shifts(2);
        std::size_t i110 = i000 | rev_wire_shifts(1) | rev_wire_shifts(2);
        std::size_t i111 =
            i000 | rev_wire_shifts(0) | rev_wire_shifts(1) | rev_wire_shifts(2);
        expval += real(EXPVAL3(i000, 0B000));
        expval += real(EXPVAL3(i001, 0B001));
        expval += real(EXPVAL3(i010, 0B010));
        expval += real(EXPVAL3(i011, 0B011));
        expval += real(EXPVAL3(i100, 0B100));
        expval += real(EXPVAL3(i101, 0B101));
        expval += real(EXPVAL3(i110, 0B110));
        expval += real(EXPVAL3(i111, 0B111));
    }
};

#define EXPVALENTRY4(xx, yy) xx << 4 | yy
#define EXPVALTERM4(xx, yy, iyy) matrix(EXPVALENTRY4(xx, yy)) * arr(iyy)
#define EXPVAL4(ixx, xx)                                                       \
    conj(arr(ixx)) *                                                           \
        (EXPVALTERM4(xx, 0B0000, i0000) + EXPVALTERM4(xx, 0B0001, i0001) +     \
         EXPVALTERM4(xx, 0B0010, i0010) + EXPVALTERM4(xx, 0B0011, i0011) +     \
         EXPVALTERM4(xx, 0B0100, i0100) + EXPVALTERM4(xx, 0B0101, i0101) +     \
         EXPVALTERM4(xx, 0B0110, i0110) + EXPVALTERM4(xx, 0B0111, i0111) +     \
         EXPVALTERM4(xx, 0B1000, i1000) + EXPVALTERM4(xx, 0B1001, i1001) +     \
         EXPVALTERM4(xx, 0B1010, i1010) + EXPVALTERM4(xx, 0B1011, i1011) +     \
         EXPVALTERM4(xx, 0B1100, i1100) + EXPVALTERM4(xx, 0B1101, i1101) +     \
         EXPVALTERM4(xx, 0B1110, i1110) + EXPVALTERM4(xx, 0B1111, i1111))

template <class PrecisionT> struct getExpVal4QubitOpFunctor {
    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    KokkosIntVector wires;
    KokkosIntVector parity;
    KokkosIntVector rev_wire_shifts;

    const std::size_t n_wires = 4;

    const std::size_t dim = one << n_wires;
    std::size_t num_qubits;

    getExpVal4QubitOpFunctor(const KokkosComplexVector &arr_,
                             const std::size_t num_qubits_,
                             const KokkosComplexVector &matrix_,
                             const std::vector<std::size_t> &wires_) {
        Kokkos::View<const std::size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            wires_host(wires_.data(), wires_.size());
        Kokkos::resize(wires, wires_.size());
        Kokkos::deep_copy(wires, wires_host);
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;
        std::tie(parity, rev_wire_shifts) = wires2Parity(num_qubits_, wires_);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {
        std::size_t i0000 = (k & parity(0));
        for (std::size_t i = 1; i < parity.size(); i++) {
            i0000 |= ((k << i) & parity(i));
        }

        std::size_t i0001 = i0000 | rev_wire_shifts(0);
        std::size_t i0010 = i0000 | rev_wire_shifts(1);
        std::size_t i0011 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(1);
        std::size_t i0100 = i0000 | rev_wire_shifts(2);
        std::size_t i0101 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(2);
        std::size_t i0110 = i0000 | rev_wire_shifts(1) | rev_wire_shifts(2);
        std::size_t i0111 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                            rev_wire_shifts(2);
        std::size_t i1000 = i0000 | rev_wire_shifts(3);
        std::size_t i1001 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(3);
        std::size_t i1010 = i0000 | rev_wire_shifts(1) | rev_wire_shifts(3);
        std::size_t i1011 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                            rev_wire_shifts(3);
        std::size_t i1100 = i0000 | rev_wire_shifts(2) | rev_wire_shifts(3);
        std::size_t i1101 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(2) |
                            rev_wire_shifts(3);
        std::size_t i1110 = i0000 | rev_wire_shifts(1) | rev_wire_shifts(2) |
                            rev_wire_shifts(3);
        std::size_t i1111 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                            rev_wire_shifts(2) | rev_wire_shifts(3);

        expval += real(EXPVAL4(i0000, 0B0000));
        expval += real(EXPVAL4(i0001, 0B0001));
        expval += real(EXPVAL4(i0010, 0B0010));
        expval += real(EXPVAL4(i0011, 0B0011));
        expval += real(EXPVAL4(i0100, 0B0100));
        expval += real(EXPVAL4(i0101, 0B0101));
        expval += real(EXPVAL4(i0110, 0B0110));
        expval += real(EXPVAL4(i0111, 0B0111));
        expval += real(EXPVAL4(i1000, 0B1000));
        expval += real(EXPVAL4(i1001, 0B1001));
        expval += real(EXPVAL4(i1010, 0B1010));
        expval += real(EXPVAL4(i1011, 0B1011));
        expval += real(EXPVAL4(i1100, 0B1100));
        expval += real(EXPVAL4(i1101, 0B1101));
        expval += real(EXPVAL4(i1110, 0B1110));
        expval += real(EXPVAL4(i1111, 0B1111));
    }
};

#define EXPVALENTRY5(xx, yy) xx << 5 | yy
#define EXPVALTERM5(xx, yy, iyy) matrix(EXPVALENTRY5(xx, yy)) * arr(iyy)
#define EXPVAL5(ixx, xx)                                                       \
    conj(arr(ixx)) *                                                           \
        (EXPVALTERM5(xx, 0B00000, i00000) + EXPVALTERM5(xx, 0B00001, i00001) + \
         EXPVALTERM5(xx, 0B00010, i00010) + EXPVALTERM5(xx, 0B00011, i00011) + \
         EXPVALTERM5(xx, 0B00100, i00100) + EXPVALTERM5(xx, 0B00101, i00101) + \
         EXPVALTERM5(xx, 0B00110, i00110) + EXPVALTERM5(xx, 0B00111, i00111) + \
         EXPVALTERM5(xx, 0B01000, i01000) + EXPVALTERM5(xx, 0B01001, i01001) + \
         EXPVALTERM5(xx, 0B01010, i01010) + EXPVALTERM5(xx, 0B01011, i01011) + \
         EXPVALTERM5(xx, 0B01100, i01100) + EXPVALTERM5(xx, 0B01101, i01101) + \
         EXPVALTERM5(xx, 0B01110, i01110) + EXPVALTERM5(xx, 0B01111, i01111) + \
         EXPVALTERM5(xx, 0B10000, i10000) + EXPVALTERM5(xx, 0B10001, i10001) + \
         EXPVALTERM5(xx, 0B10010, i10010) + EXPVALTERM5(xx, 0B10011, i10011) + \
         EXPVALTERM5(xx, 0B10100, i10100) + EXPVALTERM5(xx, 0B10101, i10101) + \
         EXPVALTERM5(xx, 0B10110, i10110) + EXPVALTERM5(xx, 0B10111, i10111) + \
         EXPVALTERM5(xx, 0B11000, i11000) + EXPVALTERM5(xx, 0B11001, i11001) + \
         EXPVALTERM5(xx, 0B11010, i11010) + EXPVALTERM5(xx, 0B11011, i11011) + \
         EXPVALTERM5(xx, 0B11100, i11100) + EXPVALTERM5(xx, 0B11101, i11101) + \
         EXPVALTERM5(xx, 0B11110, i11110) + EXPVALTERM5(xx, 0B11111, i11111))

template <class PrecisionT> struct getExpVal5QubitOpFunctor {
    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    KokkosIntVector wires;
    KokkosIntVector parity;
    KokkosIntVector rev_wire_shifts;

    const std::size_t n_wires = 5;

    const std::size_t dim = one << n_wires;
    std::size_t num_qubits;

    getExpVal5QubitOpFunctor(const KokkosComplexVector &arr_,
                             const std::size_t num_qubits_,
                             const KokkosComplexVector &matrix_,
                             const std::vector<std::size_t> &wires_) {
        Kokkos::View<const std::size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            wires_host(wires_.data(), wires_.size());
        Kokkos::resize(wires, wires_.size());
        Kokkos::deep_copy(wires, wires_host);
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;
        std::tie(parity, rev_wire_shifts) = wires2Parity(num_qubits_, wires_);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &expval) const {
        std::size_t i00000 = (k & parity(0));
        for (std::size_t i = 1; i < parity.size(); i++) {
            i00000 |= ((k << i) & parity(i));
        }

        std::size_t i00001 = i00000 | rev_wire_shifts(0);
        std::size_t i00010 = i00000 | rev_wire_shifts(1);
        std::size_t i00011 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1);
        std::size_t i00100 = i00000 | rev_wire_shifts(2);
        std::size_t i00101 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(2);
        std::size_t i00110 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(2);
        std::size_t i00111 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(2);
        std::size_t i01000 = i00000 | rev_wire_shifts(3);
        std::size_t i01001 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(3);
        std::size_t i01010 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(3);
        std::size_t i01011 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(3);
        std::size_t i01100 = i00000 | rev_wire_shifts(2) | rev_wire_shifts(3);
        std::size_t i01101 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(2) |
                             rev_wire_shifts(3);
        std::size_t i01110 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(2) |
                             rev_wire_shifts(3);
        std::size_t i01111 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(2) | rev_wire_shifts(3);
        std::size_t i10000 = i00000 | rev_wire_shifts(4);
        std::size_t i10001 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(4);
        std::size_t i10010 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(4);
        std::size_t i10011 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(4);
        std::size_t i10100 = i00000 | rev_wire_shifts(2) | rev_wire_shifts(4);
        std::size_t i10101 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(2) |
                             rev_wire_shifts(4);
        std::size_t i10110 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(2) |
                             rev_wire_shifts(4);
        std::size_t i10111 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(2) | rev_wire_shifts(4);
        std::size_t i11000 = i00000 | rev_wire_shifts(3) | rev_wire_shifts(4);
        std::size_t i11001 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(3) |
                             rev_wire_shifts(4);
        std::size_t i11010 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(3) |
                             rev_wire_shifts(4);
        std::size_t i11011 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(3) | rev_wire_shifts(4);
        std::size_t i11100 = i00000 | rev_wire_shifts(2) | rev_wire_shifts(3) |
                             rev_wire_shifts(4);
        std::size_t i11101 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(2) |
                             rev_wire_shifts(3) | rev_wire_shifts(4);
        std::size_t i11110 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(2) |
                             rev_wire_shifts(3) | rev_wire_shifts(4);
        std::size_t i11111 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(2) | rev_wire_shifts(3) |
                             rev_wire_shifts(4);

        expval += real(EXPVAL5(i00000, 0B00000));
        expval += real(EXPVAL5(i00001, 0B00001));
        expval += real(EXPVAL5(i00010, 0B00010));
        expval += real(EXPVAL5(i00011, 0B00011));
        expval += real(EXPVAL5(i00100, 0B00100));
        expval += real(EXPVAL5(i00101, 0B00101));
        expval += real(EXPVAL5(i00110, 0B00110));
        expval += real(EXPVAL5(i00111, 0B00111));
        expval += real(EXPVAL5(i01000, 0B01000));
        expval += real(EXPVAL5(i01001, 0B01001));
        expval += real(EXPVAL5(i01010, 0B01010));
        expval += real(EXPVAL5(i01011, 0B01011));
        expval += real(EXPVAL5(i01100, 0B01100));
        expval += real(EXPVAL5(i01101, 0B01101));
        expval += real(EXPVAL5(i01110, 0B01110));
        expval += real(EXPVAL5(i01111, 0B01111));
        expval += real(EXPVAL5(i10000, 0B10000));
        expval += real(EXPVAL5(i10001, 0B10001));
        expval += real(EXPVAL5(i10010, 0B10010));
        expval += real(EXPVAL5(i10011, 0B10011));
        expval += real(EXPVAL5(i10100, 0B10100));
        expval += real(EXPVAL5(i10101, 0B10101));
        expval += real(EXPVAL5(i10110, 0B10110));
        expval += real(EXPVAL5(i10111, 0B10111));
        expval += real(EXPVAL5(i11000, 0B11000));
        expval += real(EXPVAL5(i11001, 0B11001));
        expval += real(EXPVAL5(i11010, 0B11010));
        expval += real(EXPVAL5(i11011, 0B11011));
        expval += real(EXPVAL5(i11100, 0B11100));
        expval += real(EXPVAL5(i11101, 0B11101));
        expval += real(EXPVAL5(i11110, 0B11110));
        expval += real(EXPVAL5(i11111, 0B11111));
    }
};

} // namespace Pennylane::LightningKokkos::Functors
