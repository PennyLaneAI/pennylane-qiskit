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
#include <Kokkos_StdAlgorithms.hpp>

#include "BitUtil.hpp"
#include "BitUtilKokkos.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using Kokkos::Experimental::swap;
using Pennylane::LightningKokkos::Util::one;
using Pennylane::LightningKokkos::Util::wires2Parity;
using std::size_t;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Functors {

template <class Precision> struct multiQubitOpFunctor {
    using KokkosComplexVector = Kokkos::View<Kokkos::complex<Precision> *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;
    using ScratchViewComplex =
        Kokkos::View<Kokkos::complex<Precision> *,
                     Kokkos::DefaultExecutionSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ScratchViewSizeT =
        Kokkos::View<std::size_t *,
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

    multiQubitOpFunctor(KokkosComplexVector &arr_, std::size_t num_qubits_,
                        const KokkosComplexVector &matrix_,
                        const std::vector<std::size_t> &wires_) {
        Kokkos::View<const size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            wires_host(wires_.data(), wires_.size());
        Kokkos::resize(wires, wires_host.size());
        Kokkos::deep_copy(wires, wires_host);
        dim = one << wires_.size();
        num_qubits = num_qubits_;
        arr = arr_;
        matrix = matrix_;
        std::tie(parity, rev_wire_shifts) = wires2Parity(num_qubits_, wires_);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const MemberType &teamMember) const {
        const std::size_t k = teamMember.league_rank();
        ScratchViewComplex coeffs_in(teamMember.team_scratch(0), dim);
        ScratchViewSizeT indices(teamMember.team_scratch(0), dim);
        if (teamMember.team_rank() == 0) {
            std::size_t idx = (k & parity(0));
            for (std::size_t i = 1; i < parity.size(); i++) {
                idx |= ((k << i) & parity(i));
            }
            indices(0) = idx;
            coeffs_in(0) = arr(idx);

            Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember, 1, dim),
                                 [&](const std::size_t inner_idx) {
                                     std::size_t index = indices(0);
                                     for (std::size_t i = 0; i < wires.size();
                                          i++) {
                                         if ((inner_idx & (one << i)) != 0) {
                                             index |= rev_wire_shifts(i);
                                         }
                                     }
                                     indices(inner_idx) = index;
                                     coeffs_in(inner_idx) = arr(index);
                                 });
        }
        teamMember.team_barrier();
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(teamMember, dim), [&](const std::size_t i) {
                const auto idx = indices(i);
                arr(idx) = 0.0;
                const std::size_t base_idx = i * dim;

                for (std::size_t j = 0; j < dim; j++) {
                    arr(idx) += matrix(base_idx + j) * coeffs_in(j);
                }
            });
    }
};

template <class PrecisionT, bool inverse = false> struct phaseShiftFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;
    Kokkos::complex<PrecisionT> s;

    phaseShiftFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                      size_t num_qubits, const std::vector<size_t> &wires,
                      const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        const PrecisionT &angle = params[0];

        s = inverse ? exp(-Kokkos::complex<PrecisionT>(0, angle))
                    : exp(Kokkos::complex<PrecisionT>(0, angle));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;
        arr[i1] *= s;
    }
};

template <class PrecisionT, bool inverse = false> struct rxFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;
    PrecisionT c;
    PrecisionT s;
    rxFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
              size_t num_qubits, const std::vector<size_t> &wires,
              const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        const PrecisionT &angle = params[0];
        c = cos(angle * static_cast<PrecisionT>(0.5));
        s = (inverse) ? sin(angle * static_cast<PrecisionT>(0.5))
                      : sin(-angle * static_cast<PrecisionT>(0.5));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;
        const auto v0 = arr[i0];
        const auto v1 = arr[i1];

        arr[i0] =
            c * v0 + Kokkos::complex<PrecisionT>{-imag(v1) * s, real(v1) * s};
        arr[i1] =
            Kokkos::complex<PrecisionT>{-imag(v0) * s, real(v0) * s} + c * v1;
    }
};

template <class PrecisionT, bool inverse = false> struct ryFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;
    PrecisionT c;
    PrecisionT s;
    ryFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
              size_t num_qubits, const std::vector<size_t> &wires,
              const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        const PrecisionT &angle = params[0];
        c = cos(angle * static_cast<PrecisionT>(0.5));
        s = (inverse) ? -sin(angle * static_cast<PrecisionT>(0.5))
                      : sin(angle * static_cast<PrecisionT>(0.5));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;
        const auto v0 = arr[i0];
        const auto v1 = arr[i1];

        arr[i0] = Kokkos::complex<PrecisionT>{c * real(v0) - s * real(v1),
                                              c * imag(v0) - s * imag(v1)};
        arr[i1] = Kokkos::complex<PrecisionT>{s * real(v0) + c * real(v1),
                                              s * imag(v0) + c * imag(v1)};
    }
};

template <class PrecisionT, bool inverse = false> struct rzFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;
    Kokkos::complex<PrecisionT> shift_0;
    Kokkos::complex<PrecisionT> shift_1;

    rzFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
              size_t num_qubits, const std::vector<size_t> &wires,
              const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        const PrecisionT &angle = params[0];
        PrecisionT cos_angle = cos(angle * static_cast<PrecisionT>(0.5));
        PrecisionT sin_angle = sin(angle * static_cast<PrecisionT>(0.5));
        Kokkos::complex<PrecisionT> first{cos_angle, -sin_angle};
        Kokkos::complex<PrecisionT> second{cos_angle, sin_angle};
        shift_0 = (inverse) ? Kokkos::conj(first) : first;
        shift_1 = (inverse) ? Kokkos::conj(second) : second;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;
        arr[i0] *= shift_0;
        arr[i1] *= shift_1;
    }
};

template <class PrecisionT, bool inverse = false> struct cRotFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    Kokkos::complex<PrecisionT> rot_mat_0b00;
    Kokkos::complex<PrecisionT> rot_mat_0b10;
    Kokkos::complex<PrecisionT> rot_mat_0b01;
    Kokkos::complex<PrecisionT> rot_mat_0b11;

    cRotFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                size_t num_qubits, const std::vector<size_t> &wires,
                const std::vector<PrecisionT> &params) {
        const PrecisionT phi = (inverse) ? -params[2] : params[0];
        const PrecisionT theta = (inverse) ? -params[1] : params[1];
        const PrecisionT omega = (inverse) ? -params[0] : params[2];
        const PrecisionT c = std::cos(theta / 2);
        const PrecisionT s = std::sin(theta / 2);
        const PrecisionT p{phi + omega};
        const PrecisionT m{phi - omega};

        auto imag = Kokkos::complex<PrecisionT>(0, 1);
        rot_mat_0b00 =
            Kokkos::exp(static_cast<PrecisionT>(p / 2) * (-imag)) * c;
        rot_mat_0b01 = -Kokkos::exp(static_cast<PrecisionT>(m / 2) * imag) * s;
        rot_mat_0b10 =
            Kokkos::exp(static_cast<PrecisionT>(m / 2) * (-imag)) * s;
        rot_mat_0b11 = Kokkos::exp(static_cast<PrecisionT>(p / 2) * imag) * c;

        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const auto v0 = arr[i10];
        const auto v1 = arr[i11];

        arr[i10] = rot_mat_0b00 * v0 + rot_mat_0b01 * v1;
        arr[i11] = rot_mat_0b10 * v0 + rot_mat_0b11 * v1;
    }
};

template <class PrecisionT, bool inverse = false> struct isingXXFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    PrecisionT cr;
    PrecisionT sj;

    isingXXFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                   size_t num_qubits, const std::vector<size_t> &wires,
                   const std::vector<PrecisionT> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        const PrecisionT &angle = params[0];

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<PrecisionT> v00 = arr[i00];
        const Kokkos::complex<PrecisionT> v01 = arr[i01];
        const Kokkos::complex<PrecisionT> v10 = arr[i10];
        const Kokkos::complex<PrecisionT> v11 = arr[i11];

        arr[i00] = Kokkos::complex<PrecisionT>{cr * real(v00) + sj * imag(v11),
                                               cr * imag(v00) - sj * real(v11)};
        arr[i01] = Kokkos::complex<PrecisionT>{cr * real(v01) + sj * imag(v10),
                                               cr * imag(v01) - sj * real(v10)};
        arr[i10] = Kokkos::complex<PrecisionT>{cr * real(v10) + sj * imag(v01),
                                               cr * imag(v10) - sj * real(v01)};
        arr[i11] = Kokkos::complex<PrecisionT>{cr * real(v11) + sj * imag(v00),
                                               cr * imag(v11) - sj * real(v00)};
    }
};

template <class PrecisionT, bool inverse = false> struct isingXYFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    PrecisionT cr;
    PrecisionT sj;

    isingXYFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                   size_t num_qubits, const std::vector<size_t> &wires,
                   const std::vector<PrecisionT> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        const PrecisionT &angle = params[0];

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<PrecisionT> v00 = arr[i00];
        const Kokkos::complex<PrecisionT> v01 = arr[i01];
        const Kokkos::complex<PrecisionT> v10 = arr[i10];
        const Kokkos::complex<PrecisionT> v11 = arr[i11];

        arr[i00] = Kokkos::complex<PrecisionT>{real(v00), imag(v00)};
        arr[i01] = Kokkos::complex<PrecisionT>{cr * real(v01) - sj * imag(v10),
                                               cr * imag(v01) + sj * real(v10)};
        arr[i10] = Kokkos::complex<PrecisionT>{cr * real(v10) - sj * imag(v01),
                                               cr * imag(v10) + sj * real(v01)};
        arr[i11] = Kokkos::complex<PrecisionT>{real(v11), imag(v11)};
    }
};

template <class PrecisionT, bool inverse = false> struct isingYYFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    PrecisionT cr;
    PrecisionT sj;

    isingYYFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                   size_t num_qubits, const std::vector<size_t> &wires,
                   const std::vector<PrecisionT> &params) {
        const PrecisionT &angle = params[0];
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<PrecisionT> v00 = arr[i00];
        const Kokkos::complex<PrecisionT> v01 = arr[i01];
        const Kokkos::complex<PrecisionT> v10 = arr[i10];
        const Kokkos::complex<PrecisionT> v11 = arr[i11];

        arr[i00] = Kokkos::complex<PrecisionT>{cr * real(v00) - sj * imag(v11),
                                               cr * imag(v00) + sj * real(v11)};
        arr[i01] = Kokkos::complex<PrecisionT>{cr * real(v01) + sj * imag(v10),
                                               cr * imag(v01) - sj * real(v10)};
        arr[i10] = Kokkos::complex<PrecisionT>{cr * real(v10) + sj * imag(v01),
                                               cr * imag(v10) - sj * real(v01)};
        arr[i11] = Kokkos::complex<PrecisionT>{cr * real(v11) - sj * imag(v00),
                                               cr * imag(v11) + sj * real(v00)};
    }
};

template <class PrecisionT, bool inverse = false> struct isingZZFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    Kokkos::complex<PrecisionT> first;
    Kokkos::complex<PrecisionT> second;
    Kokkos::complex<PrecisionT> shift_0;
    Kokkos::complex<PrecisionT> shift_1;

    isingZZFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                   size_t num_qubits, const std::vector<size_t> &wires,
                   const std::vector<PrecisionT> &params) {
        const PrecisionT &angle = params[0];

        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        first = Kokkos::complex<PrecisionT>{std::cos(angle / 2),
                                            -std::sin(angle / 2)};
        second = Kokkos::complex<PrecisionT>{std::cos(angle / 2),
                                             std::sin(angle / 2)};

        shift_0 = (inverse) ? conj(first) : first;
        shift_1 = (inverse) ? conj(second) : second;

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i00] *= shift_0;
        arr[i01] *= shift_1;
        arr[i10] *= shift_1;
        arr[i11] *= shift_0;
    }
};

template <class PrecisionT, bool inverse = false>
struct singleExcitationFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    PrecisionT cr;
    PrecisionT sj;

    singleExcitationFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                            size_t num_qubits, const std::vector<size_t> &wires,
                            const std::vector<PrecisionT> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        const PrecisionT &angle = params[0];

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;

        const Kokkos::complex<PrecisionT> v01 = arr[i01];
        const Kokkos::complex<PrecisionT> v10 = arr[i10];

        arr[i01] = cr * v01 - sj * v10;
        arr[i10] = sj * v01 + cr * v10;
    }
};

template <class PrecisionT, bool inverse = false>
struct singleExcitationMinusFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    PrecisionT cr;
    PrecisionT sj;
    Kokkos::complex<PrecisionT> e;

    singleExcitationMinusFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_, size_t num_qubits,
        const std::vector<size_t> &wires,
        const std::vector<PrecisionT> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        const PrecisionT &angle = params[0];

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        e = inverse ? exp(Kokkos::complex<PrecisionT>(0, angle / 2))
                    : exp(Kokkos::complex<PrecisionT>(0, -angle / 2));

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<PrecisionT> v01 = arr[i01];
        const Kokkos::complex<PrecisionT> v10 = arr[i10];

        arr[i00] *= e;
        arr[i01] = cr * v01 - sj * v10;
        arr[i10] = sj * v01 + cr * v10;
        arr[i11] *= e;
    }
};

template <class PrecisionT, bool inverse = false>
struct singleExcitationPlusFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    PrecisionT cr;
    PrecisionT sj;
    Kokkos::complex<PrecisionT> e;

    singleExcitationPlusFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_, size_t num_qubits,
        const std::vector<size_t> &wires,
        const std::vector<PrecisionT> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        const PrecisionT &angle = params[0];

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        e = inverse ? exp(Kokkos::complex<PrecisionT>(0, -angle / 2))
                    : exp(Kokkos::complex<PrecisionT>(0, angle / 2));

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<PrecisionT> v01 = arr[i01];
        const Kokkos::complex<PrecisionT> v10 = arr[i10];

        arr[i00] *= e;
        arr[i01] = cr * v01 - sj * v10;
        arr[i10] = sj * v01 + cr * v10;
        arr[i11] *= e;
    }
};

template <class PrecisionT, bool inverse = false>
struct doubleExcitationFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire2;
    size_t rev_wire3;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire2_shift;
    size_t rev_wire3_shift;
    size_t rev_wire_min;
    size_t rev_wire_min_mid;
    size_t rev_wire_max_mid;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;
    size_t parity_hmiddle;
    size_t parity_lmiddle;

    Kokkos::complex<PrecisionT> shifts_0;
    Kokkos::complex<PrecisionT> shifts_1;
    Kokkos::complex<PrecisionT> shifts_2;
    Kokkos::complex<PrecisionT> shifts_3;

    PrecisionT cr;
    PrecisionT sj;

    doubleExcitationFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                            size_t num_qubits, const std::vector<size_t> &wires,
                            const std::vector<PrecisionT> &params) {
        const PrecisionT &angle = params[0];
        rev_wire0 = num_qubits - wires[3] - 1;
        rev_wire1 = num_qubits - wires[2] - 1;
        rev_wire2 = num_qubits - wires[1] - 1;
        rev_wire3 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;
        rev_wire3_shift = static_cast<size_t>(1U) << rev_wire3;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_min_mid = std::max(rev_wire0, rev_wire1);
        rev_wire_max_mid = std::min(rev_wire2, rev_wire3);
        rev_wire_max = std::max(rev_wire2, rev_wire3);

        if (rev_wire_max_mid > rev_wire_min_mid) {
        } else if (rev_wire_max_mid < rev_wire_min) {
            if (rev_wire_max < rev_wire_min) {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = tmp;

                tmp = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else if (rev_wire_max > rev_wire_min_mid) {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            }
        } else {
            if (rev_wire_max > rev_wire_min_mid) {
                size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = tmp;
            } else {
                size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = tmp;
            }
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle = fillLeadingOnes(rev_wire_min + 1) &
                         fillTrailingOnes(rev_wire_min_mid);
        parity_hmiddle = fillLeadingOnes(rev_wire_max_mid + 1) &
                         fillTrailingOnes(rev_wire_max);
        parity_middle = fillLeadingOnes(rev_wire_min_mid + 1) &
                        fillTrailingOnes(rev_wire_max_mid);

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i0000 = ((k << 4U) & parity_high) |
                             ((k << 3U) & parity_hmiddle) |
                             ((k << 2U) & parity_middle) |
                             ((k << 1U) & parity_lmiddle) | (k & parity_low);
        const size_t i0011 = i0000 | rev_wire1_shift | rev_wire0_shift;
        const size_t i1100 = i0000 | rev_wire3_shift | rev_wire2_shift;

        const Kokkos::complex<PrecisionT> v3 = arr[i0011];
        const Kokkos::complex<PrecisionT> v12 = arr[i1100];

        arr[i0011] = cr * v3 - sj * v12;
        arr[i1100] = sj * v3 + cr * v12;
    }
};

template <class PrecisionT, bool inverse = false>
struct doubleExcitationMinusFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire2;
    size_t rev_wire3;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire2_shift;
    size_t rev_wire3_shift;
    size_t rev_wire_min;
    size_t rev_wire_min_mid;
    size_t rev_wire_max_mid;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;
    size_t parity_hmiddle;
    size_t parity_lmiddle;

    Kokkos::complex<PrecisionT> shifts_0;
    Kokkos::complex<PrecisionT> shifts_1;
    Kokkos::complex<PrecisionT> shifts_2;
    Kokkos::complex<PrecisionT> shifts_3;

    PrecisionT cr;
    PrecisionT sj;
    Kokkos::complex<PrecisionT> e;

    doubleExcitationMinusFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_, size_t num_qubits,
        const std::vector<size_t> &wires,
        const std::vector<PrecisionT> &params) {
        const PrecisionT &angle = params[0];
        rev_wire0 = num_qubits - wires[3] - 1;
        rev_wire1 = num_qubits - wires[2] - 1;
        rev_wire2 = num_qubits - wires[1] - 1;
        rev_wire3 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;
        rev_wire3_shift = static_cast<size_t>(1U) << rev_wire3;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_min_mid = std::max(rev_wire0, rev_wire1);
        rev_wire_max_mid = std::min(rev_wire2, rev_wire3);
        rev_wire_max = std::max(rev_wire2, rev_wire3);

        if (rev_wire_max_mid > rev_wire_min_mid) {
        } else if (rev_wire_max_mid < rev_wire_min) {
            if (rev_wire_max < rev_wire_min) {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = tmp;

                tmp = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else if (rev_wire_max > rev_wire_min_mid) {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            }
        } else {
            if (rev_wire_max > rev_wire_min_mid) {
                size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = tmp;
            } else {
                size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = tmp;
            }
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle = fillLeadingOnes(rev_wire_min + 1) &
                         fillTrailingOnes(rev_wire_min_mid);
        parity_hmiddle = fillLeadingOnes(rev_wire_max_mid + 1) &
                         fillTrailingOnes(rev_wire_max);
        parity_middle = fillLeadingOnes(rev_wire_min_mid + 1) &
                        fillTrailingOnes(rev_wire_max_mid);

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        e = inverse ? exp(Kokkos::complex<PrecisionT>(0, angle / 2))
                    : exp(Kokkos::complex<PrecisionT>(0, -angle / 2));

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i0000 = ((k << 4U) & parity_high) |
                             ((k << 3U) & parity_hmiddle) |
                             ((k << 2U) & parity_middle) |
                             ((k << 1U) & parity_lmiddle) | (k & parity_low);
        const size_t i0001 = i0000 | rev_wire0_shift;
        const size_t i0010 = i0000 | rev_wire1_shift;
        const size_t i0011 = i0000 | rev_wire1_shift | rev_wire0_shift;
        const size_t i0100 = i0000 | rev_wire2_shift;
        const size_t i0101 = i0000 | rev_wire2_shift | rev_wire0_shift;
        const size_t i0110 = i0000 | rev_wire2_shift | rev_wire1_shift;
        const size_t i0111 =
            i0000 | rev_wire2_shift | rev_wire1_shift | rev_wire0_shift;
        const size_t i1000 = i0000 | rev_wire3_shift;
        const size_t i1001 = i0000 | rev_wire3_shift | rev_wire0_shift;
        const size_t i1010 = i0000 | rev_wire3_shift | rev_wire1_shift;
        const size_t i1011 =
            i0000 | rev_wire3_shift | rev_wire1_shift | rev_wire0_shift;
        const size_t i1100 = i0000 | rev_wire3_shift | rev_wire2_shift;
        const size_t i1101 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire0_shift;
        const size_t i1110 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire1_shift;
        const size_t i1111 = i0000 | rev_wire3_shift | rev_wire2_shift |
                             rev_wire1_shift | rev_wire0_shift;

        const Kokkos::complex<PrecisionT> v3 = arr[i0011];
        const Kokkos::complex<PrecisionT> v12 = arr[i1100];

        arr[i0000] *= e;
        arr[i0001] *= e;
        arr[i0010] *= e;
        arr[i0011] = cr * v3 - sj * v12;
        arr[i0100] *= e;
        arr[i0101] *= e;
        arr[i0110] *= e;
        arr[i0111] *= e;
        arr[i1000] *= e;
        arr[i1001] *= e;
        arr[i1010] *= e;
        arr[i1011] *= e;
        arr[i1100] = sj * v3 + cr * v12;
        arr[i1101] *= e;
        arr[i1110] *= e;
        arr[i1111] *= e;
    }
};

template <class PrecisionT, bool inverse = false>
struct doubleExcitationPlusFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire2;
    size_t rev_wire3;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire2_shift;
    size_t rev_wire3_shift;
    size_t rev_wire_min;
    size_t rev_wire_min_mid;
    size_t rev_wire_max_mid;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;
    size_t parity_hmiddle;
    size_t parity_lmiddle;

    Kokkos::complex<PrecisionT> shifts_0;
    Kokkos::complex<PrecisionT> shifts_1;
    Kokkos::complex<PrecisionT> shifts_2;
    Kokkos::complex<PrecisionT> shifts_3;

    PrecisionT cr;
    PrecisionT sj;
    Kokkos::complex<PrecisionT> e;

    doubleExcitationPlusFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_, size_t num_qubits,
        const std::vector<size_t> &wires,
        const std::vector<PrecisionT> &params) {
        const PrecisionT &angle = params[0];
        rev_wire0 = num_qubits - wires[3] - 1;
        rev_wire1 = num_qubits - wires[2] - 1;
        rev_wire2 = num_qubits - wires[1] - 1;
        rev_wire3 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;
        rev_wire3_shift = static_cast<size_t>(1U) << rev_wire3;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_min_mid = std::max(rev_wire0, rev_wire1);
        rev_wire_max_mid = std::min(rev_wire2, rev_wire3);
        rev_wire_max = std::max(rev_wire2, rev_wire3);

        if (rev_wire_max_mid > rev_wire_min_mid) {
        } else if (rev_wire_max_mid < rev_wire_min) {
            if (rev_wire_max < rev_wire_min) {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = tmp;

                tmp = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else if (rev_wire_max > rev_wire_min_mid) {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else {
                size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            }
        } else {
            if (rev_wire_max > rev_wire_min_mid) {
                size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = tmp;
            } else {
                size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = tmp;
            }
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle = fillLeadingOnes(rev_wire_min + 1) &
                         fillTrailingOnes(rev_wire_min_mid);
        parity_hmiddle = fillLeadingOnes(rev_wire_max_mid + 1) &
                         fillTrailingOnes(rev_wire_max);
        parity_middle = fillLeadingOnes(rev_wire_min_mid + 1) &
                        fillTrailingOnes(rev_wire_max_mid);

        cr = std::cos(angle / 2);
        sj = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        e = inverse ? exp(Kokkos::complex<PrecisionT>(0, -angle / 2))
                    : exp(Kokkos::complex<PrecisionT>(0, angle / 2));

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i0000 = ((k << 4U) & parity_high) |
                             ((k << 3U) & parity_hmiddle) |
                             ((k << 2U) & parity_middle) |
                             ((k << 1U) & parity_lmiddle) | (k & parity_low);
        const size_t i0001 = i0000 | rev_wire0_shift;
        const size_t i0010 = i0000 | rev_wire1_shift;
        const size_t i0011 = i0000 | rev_wire1_shift | rev_wire0_shift;
        const size_t i0100 = i0000 | rev_wire2_shift;
        const size_t i0101 = i0000 | rev_wire2_shift | rev_wire0_shift;
        const size_t i0110 = i0000 | rev_wire2_shift | rev_wire1_shift;
        const size_t i0111 =
            i0000 | rev_wire2_shift | rev_wire1_shift | rev_wire0_shift;
        const size_t i1000 = i0000 | rev_wire3_shift;
        const size_t i1001 = i0000 | rev_wire3_shift | rev_wire0_shift;
        const size_t i1010 = i0000 | rev_wire3_shift | rev_wire1_shift;
        const size_t i1011 =
            i0000 | rev_wire3_shift | rev_wire1_shift | rev_wire0_shift;
        const size_t i1100 = i0000 | rev_wire3_shift | rev_wire2_shift;
        const size_t i1101 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire0_shift;
        const size_t i1110 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire1_shift;
        const size_t i1111 = i0000 | rev_wire3_shift | rev_wire2_shift |
                             rev_wire1_shift | rev_wire0_shift;

        const Kokkos::complex<PrecisionT> v3 = arr[i0011];
        const Kokkos::complex<PrecisionT> v12 = arr[i1100];

        arr[i0000] *= e;
        arr[i0001] *= e;
        arr[i0010] *= e;
        arr[i0011] = cr * v3 - sj * v12;
        arr[i0100] *= e;
        arr[i0101] *= e;
        arr[i0110] *= e;
        arr[i0111] *= e;
        arr[i1000] *= e;
        arr[i1001] *= e;
        arr[i1010] *= e;
        arr[i1011] *= e;
        arr[i1100] = sj * v3 + cr * v12;
        arr[i1101] *= e;
        arr[i1110] *= e;
        arr[i1111] *= e;
    }
};

template <class PrecisionT, bool inverse = false>
struct controlledPhaseShiftFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    Kokkos::complex<PrecisionT> s;

    controlledPhaseShiftFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_, size_t num_qubits,
        const std::vector<size_t> &wires,
        const std::vector<PrecisionT> &params) {
        const PrecisionT &angle = params[0];
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        s = inverse ? exp(-Kokkos::complex<PrecisionT>(0, angle))
                    : exp(Kokkos::complex<PrecisionT>(0, angle));

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i11 = i00 | rev_wire1_shift | rev_wire0_shift;

        arr[i11] *= s;
    }
};

template <class PrecisionT, bool inverse = false> struct crxFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    PrecisionT c;
    PrecisionT js;

    crxFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
               size_t num_qubits, const std::vector<size_t> &wires,
               const std::vector<PrecisionT> &params) {
        const PrecisionT &angle = params[0];
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        c = std::cos(angle / 2);
        js = (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<PrecisionT> v10 = arr[i10];
        const Kokkos::complex<PrecisionT> v11 = arr[i11];

        arr[i10] = Kokkos::complex<PrecisionT>{
            c * Kokkos::real(v10) + js * Kokkos::imag(v11),
            c * Kokkos::imag(v10) - js * Kokkos::real(v11)};
        arr[i11] = Kokkos::complex<PrecisionT>{
            c * Kokkos::real(v11) + js * Kokkos::imag(v10),
            c * Kokkos::imag(v11) - js * Kokkos::real(v10)};
    }
};

template <class PrecisionT, bool inverse = false> struct cryFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    PrecisionT c;
    PrecisionT s;

    cryFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
               size_t num_qubits, const std::vector<size_t> &wires,
               const std::vector<PrecisionT> &params) {
        const PrecisionT &angle = params[0];
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        c = std::cos(angle / 2);
        s = (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<PrecisionT> v10 = arr[i10];
        const Kokkos::complex<PrecisionT> v11 = arr[i11];

        arr[i10] = c * v10 - s * v11;
        arr[i11] = s * v10 + c * v11;
    }
};

template <class PrecisionT, bool inverse = false> struct crzFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t rev_wire0;
    size_t rev_wire1;
    size_t rev_wire0_shift;
    size_t rev_wire1_shift;
    size_t rev_wire_min;
    size_t rev_wire_max;
    size_t parity_low;
    size_t parity_high;
    size_t parity_middle;

    Kokkos::complex<PrecisionT> shifts_0;
    Kokkos::complex<PrecisionT> shifts_1;

    crzFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
               size_t num_qubits, const std::vector<size_t> &wires,
               const std::vector<PrecisionT> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        const PrecisionT &angle = params[0];

        const Kokkos::complex<PrecisionT> first = Kokkos::complex<PrecisionT>{
            std::cos(angle / 2), -std::sin(angle / 2)};
        const Kokkos::complex<PrecisionT> second = Kokkos::complex<PrecisionT>{
            std::cos(angle / 2), std::sin(angle / 2)};

        shifts_0 = (inverse) ? conj(first) : first;
        shifts_1 = (inverse) ? conj(second) : second;

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i10] *= shifts_0;
        arr[i11] *= shifts_1;
    }
};

template <class PrecisionT, bool inverse = false> struct multiRZFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    size_t wires_parity;

    Kokkos::complex<PrecisionT> shift_0;
    Kokkos::complex<PrecisionT> shift_1;

    multiRZFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                   size_t num_qubits, const std::vector<size_t> &wires,
                   const std::vector<PrecisionT> &params) {
        const PrecisionT &angle = params[0];
        const Kokkos::complex<PrecisionT> first = Kokkos::complex<PrecisionT>{
            std::cos(angle / 2), -std::sin(angle / 2)};
        const Kokkos::complex<PrecisionT> second = Kokkos::complex<PrecisionT>{
            std::cos(angle / 2), std::sin(angle / 2)};

        shift_0 = (inverse) ? conj(first) : first;
        shift_1 = (inverse) ? conj(second) : second;

        size_t wires_parity_ = 0U;
        for (size_t wire : wires) {
            wires_parity_ |=
                (static_cast<size_t>(1U) << (num_qubits - wire - 1));
        }

        wires_parity = wires_parity_;
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        arr[k] *= (Kokkos::Impl::bit_count(k & wires_parity) % 2 == 0)
                      ? shift_0
                      : shift_1;
    }
};

template <class PrecisionT, bool inverse = false> struct globalPhaseFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;
    Kokkos::complex<PrecisionT> phase;

    globalPhaseFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                       [[maybe_unused]] size_t num_qubits,
                       [[maybe_unused]] const std::vector<size_t> &wires,
                       const std::vector<PrecisionT> &params) {
        phase = Kokkos::exp(
            Kokkos::complex<PrecisionT>{0, (inverse) ? params[0] : -params[0]});
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const { arr[k] *= phase; }
};

template <class PrecisionT, bool inverse = false> struct rotFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    Kokkos::complex<PrecisionT> rot_mat_0b00;
    Kokkos::complex<PrecisionT> rot_mat_0b10;
    Kokkos::complex<PrecisionT> rot_mat_0b01;
    Kokkos::complex<PrecisionT> rot_mat_0b11;

    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;

    rotFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
               size_t num_qubits, const std::vector<size_t> &wires,
               const std::vector<PrecisionT> &params) {
        const PrecisionT phi = (inverse) ? -params[2] : params[0];
        const PrecisionT theta = (inverse) ? -params[1] : params[1];
        const PrecisionT omega = (inverse) ? -params[0] : params[2];
        const PrecisionT c = std::cos(theta / 2);
        const PrecisionT s = std::sin(theta / 2);
        const PrecisionT p{phi + omega};
        const PrecisionT m{phi - omega};

        auto imag = Kokkos::complex<PrecisionT>(0, 1);
        rot_mat_0b00 =
            Kokkos::exp(static_cast<PrecisionT>(p / 2) * (-imag)) * c;
        rot_mat_0b01 = -Kokkos::exp(static_cast<PrecisionT>(m / 2) * imag) * s;
        rot_mat_0b10 =
            Kokkos::exp(static_cast<PrecisionT>(m / 2) * (-imag)) * s;
        rot_mat_0b11 = Kokkos::exp(static_cast<PrecisionT>(p / 2) * imag) * c;

        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;
        const Kokkos::complex<PrecisionT> v0 = arr[i0];
        const Kokkos::complex<PrecisionT> v1 = arr[i1];
        arr[i0] = rot_mat_0b00 * v0 +
                  rot_mat_0b01 * v1; // NOLINT(readability-magic-numbers)
        arr[i1] = rot_mat_0b10 * v0 +
                  rot_mat_0b11 * v1; // NOLINT(readability-magic-numbers)
    }
};

template <class PrecisionT> struct apply1QubitOpFunctor {
    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    using KokkosIntVector = Kokkos::View<std::size_t *>;

    KokkosComplexVector arr;
    KokkosComplexVector matrix;
    const std::size_t n_wires = 1;
    const std::size_t dim = one << n_wires;
    std::size_t num_qubits;
    size_t rev_wire;
    size_t rev_wire_shift;
    size_t wire_parity;
    size_t wire_parity_inv;

    apply1QubitOpFunctor(
        KokkosComplexVector &arr_, std::size_t num_qubits_,
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
    void operator()(const std::size_t k) const {
        const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const size_t i1 = i0 | rev_wire_shift;
        const Kokkos::complex<PrecisionT> v0 = arr[i0];
        const Kokkos::complex<PrecisionT> v1 = arr[i1];

        arr(i0) = matrix(0B00) * v0 + matrix(0B01) * v1;
        arr(i1) = matrix(0B10) * v0 + matrix(0B11) * v1;
    }
};

template <class PrecisionT> struct apply2QubitOpFunctor {
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

    apply2QubitOpFunctor(
        KokkosComplexVector &arr_, std::size_t num_qubits_,
        const KokkosComplexVector &matrix_,
        [[maybe_unused]] const std::vector<std::size_t> &wires_) {
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;

        rev_wire0 = num_qubits - wires_[1] - 1;
        rev_wire1 = num_qubits - wires_[0] - 1; // Control qubit
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
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const Kokkos::complex<PrecisionT> v00 = arr[i00];
        const Kokkos::complex<PrecisionT> v01 = arr[i01];
        const Kokkos::complex<PrecisionT> v10 = arr[i10];
        const Kokkos::complex<PrecisionT> v11 = arr[i11];

        arr(i00) = matrix(0B0000) * v00 + matrix(0B0001) * v01 +
                   matrix(0B0010) * v10 + matrix(0B0011) * v11;
        arr(i01) = matrix(0B0100) * v00 + matrix(0B0101) * v01 +
                   matrix(0B0110) * v10 + matrix(0B0111) * v11;
        arr(i10) = matrix(0B1000) * v00 + matrix(0B1001) * v01 +
                   matrix(0B1010) * v10 + matrix(0B1011) * v11;
        arr(i11) = matrix(0B1100) * v00 + matrix(0B1101) * v01 +
                   matrix(0B1110) * v10 + matrix(0B1111) * v11;
    }
};

#define GATEENTRY3(xx, yy) xx << 3 | yy
#define GATETERM3(xx, yy, vyy) matrix(GATEENTRY3(xx, yy)) * vyy
#define GATESUM3(xx)                                                           \
    GATETERM3(xx, 0B000, v000) + GATETERM3(xx, 0B001, v001) +                  \
        GATETERM3(xx, 0B010, v010) + GATETERM3(xx, 0B011, v011) +              \
        GATETERM3(xx, 0B100, v100) + GATETERM3(xx, 0B101, v101) +              \
        GATETERM3(xx, 0B110, v110) + GATETERM3(xx, 0B111, v111)

template <class PrecisionT> struct apply3QubitOpFunctor {
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

    apply3QubitOpFunctor(KokkosComplexVector &arr_, std::size_t num_qubits_,
                         const KokkosComplexVector &matrix_,
                         const std::vector<std::size_t> &wires_) {
        Kokkos::View<const size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            wires_host(wires_.data(), wires_.size());
        Kokkos::resize(wires, wires_host.size());
        Kokkos::deep_copy(wires, wires_host);
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;
        std::tie(parity, rev_wire_shifts) = wires2Parity(num_qubits_, wires_);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        std::size_t i000 = (k & parity(0));
        for (std::size_t i = 1; i < parity.size(); i++) {
            i000 |= ((k << i) & parity(i));
        }
        ComplexT v000 = arr(i000);

        std::size_t i001 = i000 | rev_wire_shifts(0);
        ComplexT v001 = arr(i001);
        std::size_t i010 = i000 | rev_wire_shifts(1);
        ComplexT v010 = arr(i010);
        std::size_t i011 = i000 | rev_wire_shifts(0) | rev_wire_shifts(1);
        ComplexT v011 = arr(i011);
        std::size_t i100 = i000 | rev_wire_shifts(2);
        ComplexT v100 = arr(i100);
        std::size_t i101 = i000 | rev_wire_shifts(0) | rev_wire_shifts(2);
        ComplexT v101 = arr(i101);
        std::size_t i110 = i000 | rev_wire_shifts(1) | rev_wire_shifts(2);
        ComplexT v110 = arr(i110);
        std::size_t i111 =
            i000 | rev_wire_shifts(0) | rev_wire_shifts(1) | rev_wire_shifts(2);
        ComplexT v111 = arr(i111);
        arr(i000) = GATESUM3(0B000);
        arr(i001) = GATESUM3(0B001);
        arr(i010) = GATESUM3(0B010);
        arr(i011) = GATESUM3(0B011);
        arr(i100) = GATESUM3(0B100);
        arr(i101) = GATESUM3(0B101);
        arr(i110) = GATESUM3(0B110);
        arr(i111) = GATESUM3(0B111);
    }
};

#define GATEENTRY4(xx, yy) xx << 4 | yy
#define GATETERM4(xx, yy, vyy) matrix(GATEENTRY4(xx, yy)) * vyy
#define GATESUM4(xx)                                                           \
    GATETERM4(xx, 0B0000, v0000) + GATETERM4(xx, 0B0001, v0001) +              \
        GATETERM4(xx, 0B0010, v0010) + GATETERM4(xx, 0B0011, v0011) +          \
        GATETERM4(xx, 0B0100, v0100) + GATETERM4(xx, 0B0101, v0101) +          \
        GATETERM4(xx, 0B0110, v0110) + GATETERM4(xx, 0B0111, v0111) +          \
        GATETERM4(xx, 0B1000, v1000) + GATETERM4(xx, 0B1001, v1001) +          \
        GATETERM4(xx, 0B1010, v1010) + GATETERM4(xx, 0B1011, v1011) +          \
        GATETERM4(xx, 0B1100, v1100) + GATETERM4(xx, 0B1101, v1101) +          \
        GATETERM4(xx, 0B1110, v1110) + GATETERM4(xx, 0B1111, v1111)

template <class PrecisionT> struct apply4QubitOpFunctor {
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

    apply4QubitOpFunctor(KokkosComplexVector &arr_, std::size_t num_qubits_,
                         const KokkosComplexVector &matrix_,
                         const std::vector<std::size_t> &wires_) {
        Kokkos::View<const size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            wires_host(wires_.data(), wires_.size());
        Kokkos::resize(wires, wires_host.size());
        Kokkos::deep_copy(wires, wires_host);
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;
        std::tie(parity, rev_wire_shifts) = wires2Parity(num_qubits_, wires_);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        std::size_t i0000 = (k & parity(0));
        for (std::size_t i = 1; i < parity.size(); i++) {
            i0000 |= ((k << i) & parity(i));
        }
        ComplexT v0000 = arr(i0000);

        std::size_t i0001 = i0000 | rev_wire_shifts(0);
        ComplexT v0001 = arr(i0001);
        std::size_t i0010 = i0000 | rev_wire_shifts(1);
        ComplexT v0010 = arr(i0010);
        std::size_t i0011 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(1);
        ComplexT v0011 = arr(i0011);
        std::size_t i0100 = i0000 | rev_wire_shifts(2);
        ComplexT v0100 = arr(i0100);
        std::size_t i0101 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(2);
        ComplexT v0101 = arr(i0101);
        std::size_t i0110 = i0000 | rev_wire_shifts(1) | rev_wire_shifts(2);
        ComplexT v0110 = arr(i0110);
        std::size_t i0111 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                            rev_wire_shifts(2);
        ComplexT v0111 = arr(i0111);
        std::size_t i1000 = i0000 | rev_wire_shifts(3);
        ComplexT v1000 = arr(i1000);
        std::size_t i1001 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(3);
        ComplexT v1001 = arr(i1001);
        std::size_t i1010 = i0000 | rev_wire_shifts(1) | rev_wire_shifts(3);
        ComplexT v1010 = arr(i1010);
        std::size_t i1011 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                            rev_wire_shifts(3);
        ComplexT v1011 = arr(i1011);
        std::size_t i1100 = i0000 | rev_wire_shifts(2) | rev_wire_shifts(3);
        ComplexT v1100 = arr(i1100);
        std::size_t i1101 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(2) |
                            rev_wire_shifts(3);
        ComplexT v1101 = arr(i1101);
        std::size_t i1110 = i0000 | rev_wire_shifts(1) | rev_wire_shifts(2) |
                            rev_wire_shifts(3);
        ComplexT v1110 = arr(i1110);
        std::size_t i1111 = i0000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                            rev_wire_shifts(2) | rev_wire_shifts(3);
        ComplexT v1111 = arr(i1111);

        arr(i0000) = GATESUM4(0B0000);
        arr(i0001) = GATESUM4(0B0001);
        arr(i0010) = GATESUM4(0B0010);
        arr(i0011) = GATESUM4(0B0011);
        arr(i0100) = GATESUM4(0B0100);
        arr(i0101) = GATESUM4(0B0101);
        arr(i0110) = GATESUM4(0B0110);
        arr(i0111) = GATESUM4(0B0111);
        arr(i1000) = GATESUM4(0B1000);
        arr(i1001) = GATESUM4(0B1001);
        arr(i1010) = GATESUM4(0B1010);
        arr(i1011) = GATESUM4(0B1011);
        arr(i1100) = GATESUM4(0B1100);
        arr(i1101) = GATESUM4(0B1101);
        arr(i1110) = GATESUM4(0B1110);
        arr(i1111) = GATESUM4(0B1111);
    }
};

#define GATEENTRY5(xx, yy) xx << 5 | yy
#define GATETERM5(xx, yy, vyy) matrix(GATEENTRY5(xx, yy)) * vyy
#define GATESUM5(xx)                                                           \
    GATETERM5(xx, 0B00000, v00000) + GATETERM5(xx, 0B00001, v00001) +          \
        GATETERM5(xx, 0B00010, v00010) + GATETERM5(xx, 0B00011, v00011) +      \
        GATETERM5(xx, 0B00100, v00100) + GATETERM5(xx, 0B00101, v00101) +      \
        GATETERM5(xx, 0B00110, v00110) + GATETERM5(xx, 0B00111, v00111) +      \
        GATETERM5(xx, 0B01000, v01000) + GATETERM5(xx, 0B01001, v01001) +      \
        GATETERM5(xx, 0B01010, v01010) + GATETERM5(xx, 0B01011, v01011) +      \
        GATETERM5(xx, 0B01100, v01100) + GATETERM5(xx, 0B01101, v01101) +      \
        GATETERM5(xx, 0B01110, v01110) + GATETERM5(xx, 0B01111, v01111) +      \
        GATETERM5(xx, 0B10000, v10000) + GATETERM5(xx, 0B10001, v10001) +      \
        GATETERM5(xx, 0B10010, v10010) + GATETERM5(xx, 0B10011, v10011) +      \
        GATETERM5(xx, 0B10100, v10100) + GATETERM5(xx, 0B10101, v10101) +      \
        GATETERM5(xx, 0B10110, v10110) + GATETERM5(xx, 0B10111, v10111) +      \
        GATETERM5(xx, 0B11000, v11000) + GATETERM5(xx, 0B11001, v11001) +      \
        GATETERM5(xx, 0B11010, v11010) + GATETERM5(xx, 0B11011, v11011) +      \
        GATETERM5(xx, 0B11100, v11100) + GATETERM5(xx, 0B11101, v11101) +      \
        GATETERM5(xx, 0B11110, v11110) + GATETERM5(xx, 0B11111, v11111)
template <class PrecisionT> struct apply5QubitOpFunctor {
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

    apply5QubitOpFunctor(KokkosComplexVector &arr_, std::size_t num_qubits_,
                         const KokkosComplexVector &matrix_,
                         const std::vector<std::size_t> &wires_) {
        Kokkos::View<const size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            wires_host(wires_.data(), wires_.size());
        Kokkos::resize(wires, wires_host.size());
        Kokkos::deep_copy(wires, wires_host);
        arr = arr_;
        matrix = matrix_;
        num_qubits = num_qubits_;
        std::tie(parity, rev_wire_shifts) = wires2Parity(num_qubits_, wires_);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        std::size_t i00000 = (k & parity(0));
        for (std::size_t i = 1; i < parity.size(); i++) {
            i00000 |= ((k << i) & parity(i));
        }
        ComplexT v00000 = arr(i00000);

        std::size_t i00001 = i00000 | rev_wire_shifts(0);
        ComplexT v00001 = arr(i00001);
        std::size_t i00010 = i00000 | rev_wire_shifts(1);
        ComplexT v00010 = arr(i00010);
        std::size_t i00011 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1);
        ComplexT v00011 = arr(i00011);
        std::size_t i00100 = i00000 | rev_wire_shifts(2);
        ComplexT v00100 = arr(i00100);
        std::size_t i00101 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(2);
        ComplexT v00101 = arr(i00101);
        std::size_t i00110 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(2);
        ComplexT v00110 = arr(i00110);
        std::size_t i00111 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(2);
        ComplexT v00111 = arr(i00111);
        std::size_t i01000 = i00000 | rev_wire_shifts(3);
        ComplexT v01000 = arr(i01000);
        std::size_t i01001 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(3);
        ComplexT v01001 = arr(i01001);
        std::size_t i01010 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(3);
        ComplexT v01010 = arr(i01010);
        std::size_t i01011 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(3);
        ComplexT v01011 = arr(i01011);
        std::size_t i01100 = i00000 | rev_wire_shifts(2) | rev_wire_shifts(3);
        ComplexT v01100 = arr(i01100);
        std::size_t i01101 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(2) |
                             rev_wire_shifts(3);
        ComplexT v01101 = arr(i01101);
        std::size_t i01110 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(2) |
                             rev_wire_shifts(3);
        ComplexT v01110 = arr(i01110);
        std::size_t i01111 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(2) | rev_wire_shifts(3);
        ComplexT v01111 = arr(i01111);
        std::size_t i10000 = i00000 | rev_wire_shifts(4);
        ComplexT v10000 = arr(i10000);
        std::size_t i10001 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(4);
        ComplexT v10001 = arr(i10001);
        std::size_t i10010 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(4);
        ComplexT v10010 = arr(i10010);
        std::size_t i10011 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(4);
        ComplexT v10011 = arr(i10011);
        std::size_t i10100 = i00000 | rev_wire_shifts(2) | rev_wire_shifts(4);
        ComplexT v10100 = arr(i10100);
        std::size_t i10101 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(2) |
                             rev_wire_shifts(4);
        ComplexT v10101 = arr(i10101);
        std::size_t i10110 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(2) |
                             rev_wire_shifts(4);
        ComplexT v10110 = arr(i10110);
        std::size_t i10111 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(2) | rev_wire_shifts(4);
        ComplexT v10111 = arr(i10111);
        std::size_t i11000 = i00000 | rev_wire_shifts(3) | rev_wire_shifts(4);
        ComplexT v11000 = arr(i11000);
        std::size_t i11001 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(3) |
                             rev_wire_shifts(4);
        ComplexT v11001 = arr(i11001);
        std::size_t i11010 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(3) |
                             rev_wire_shifts(4);
        ComplexT v11010 = arr(i11010);
        std::size_t i11011 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(3) | rev_wire_shifts(4);
        ComplexT v11011 = arr(i11011);
        std::size_t i11100 = i00000 | rev_wire_shifts(2) | rev_wire_shifts(3) |
                             rev_wire_shifts(4);
        ComplexT v11100 = arr(i11100);
        std::size_t i11101 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(2) |
                             rev_wire_shifts(3) | rev_wire_shifts(4);
        ComplexT v11101 = arr(i11101);
        std::size_t i11110 = i00000 | rev_wire_shifts(1) | rev_wire_shifts(2) |
                             rev_wire_shifts(3) | rev_wire_shifts(4);
        ComplexT v11110 = arr(i11110);
        std::size_t i11111 = i00000 | rev_wire_shifts(0) | rev_wire_shifts(1) |
                             rev_wire_shifts(2) | rev_wire_shifts(3) |
                             rev_wire_shifts(4);
        ComplexT v11111 = arr(i11111);

        arr(i00000) = GATESUM5(0B00000);
        arr(i00001) = GATESUM5(0B00001);
        arr(i00010) = GATESUM5(0B00010);
        arr(i00011) = GATESUM5(0B00011);
        arr(i00100) = GATESUM5(0B00100);
        arr(i00101) = GATESUM5(0B00101);
        arr(i00110) = GATESUM5(0B00110);
        arr(i00111) = GATESUM5(0B00111);
        arr(i01000) = GATESUM5(0B01000);
        arr(i01001) = GATESUM5(0B01001);
        arr(i01010) = GATESUM5(0B01010);
        arr(i01011) = GATESUM5(0B01011);
        arr(i01100) = GATESUM5(0B01100);
        arr(i01101) = GATESUM5(0B01101);
        arr(i01110) = GATESUM5(0B01110);
        arr(i01111) = GATESUM5(0B01111);
        arr(i10000) = GATESUM5(0B10000);
        arr(i10001) = GATESUM5(0B10001);
        arr(i10010) = GATESUM5(0B10010);
        arr(i10011) = GATESUM5(0B10011);
        arr(i10100) = GATESUM5(0B10100);
        arr(i10101) = GATESUM5(0B10101);
        arr(i10110) = GATESUM5(0B10110);
        arr(i10111) = GATESUM5(0B10111);
        arr(i11000) = GATESUM5(0B11000);
        arr(i11001) = GATESUM5(0B11001);
        arr(i11010) = GATESUM5(0B11010);
        arr(i11011) = GATESUM5(0B11011);
        arr(i11100) = GATESUM5(0B11100);
        arr(i11101) = GATESUM5(0B11101);
        arr(i11110) = GATESUM5(0B11110);
        arr(i11111) = GATESUM5(0B11111);
    }
};

} // namespace Pennylane::LightningKokkos::Functors