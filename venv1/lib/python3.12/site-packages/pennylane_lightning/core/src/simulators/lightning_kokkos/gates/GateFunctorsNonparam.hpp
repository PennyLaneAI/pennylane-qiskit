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

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using Kokkos::Experimental::swap;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Functors {
template <class PrecisionT, bool inverse = false> struct hadamardFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    hadamardFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                    std::size_t num_qubits, const std::vector<size_t> &wires,
                    [[maybe_unused]] const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        if constexpr (inverse) {
            const std::size_t i0 =
                ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const std::size_t i1 = i0 | rev_wire_shift;
            const Kokkos::complex<PrecisionT> v0 = arr[i0];
            const Kokkos::complex<PrecisionT> v1 = arr[i1];
            arr[i0] = M_SQRT1_2 * v0 +
                      M_SQRT1_2 * v1; // NOLINT(readability-magic-numbers)
            arr[i1] = M_SQRT1_2 * v0 +
                      (-M_SQRT1_2) * v1; // NOLINT(readability-magic-numbers)
                                         // }
        } else {
            const std::size_t i0 =
                ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const std::size_t i1 = i0 | rev_wire_shift;
            const Kokkos::complex<PrecisionT> v0 = arr[i0];
            const Kokkos::complex<PrecisionT> v1 = arr[i1];
            arr[i0] = M_SQRT1_2 * v0 +
                      M_SQRT1_2 * v1; // NOLINT(readability-magic-numbers)
            arr[i1] = M_SQRT1_2 * v0 +
                      -M_SQRT1_2 * v1; // NOLINT(readability-magic-numbers)
                                       // }
        }
    }
};

template <class PrecisionT, bool inverse = false> struct pauliXFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    pauliXFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                  std::size_t num_qubits, const std::vector<size_t> &wires,
                  [[maybe_unused]] const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        swap(arr[i0], arr[i1]);
    }
};

template <class PrecisionT, bool inverse = false> struct pauliYFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    pauliYFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                  std::size_t num_qubits, const std::vector<size_t> &wires,
                  [[maybe_unused]] const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        const auto v0 = arr[i0];
        const auto v1 = arr[i1];
        arr[i0] = Kokkos::complex<PrecisionT>{imag(v1), -real(v1)};
        arr[i1] = Kokkos::complex<PrecisionT>{-imag(v0), real(v0)};
    }
};

template <class PrecisionT, bool inverse = false> struct pauliZFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    pauliZFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                  std::size_t num_qubits, const std::vector<size_t> &wires,
                  [[maybe_unused]] const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        arr[i1] *= -1;
    }
};

template <class PrecisionT, bool inverse = false> struct sFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;
    Kokkos::complex<PrecisionT> shift;

    sFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
             std::size_t num_qubits, const std::vector<size_t> &wires,
             [[maybe_unused]] const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        shift =
            (inverse) ? -Kokkos::complex{0.0, 1.0} : Kokkos::complex{0.0, 1.0};
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        arr[i1] *= shift;
    }
};

template <class PrecisionT, bool inverse = false> struct tFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;
    Kokkos::complex<PrecisionT> shift;

    tFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
             std::size_t num_qubits, const std::vector<size_t> &wires,
             [[maybe_unused]] const std::vector<PrecisionT> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
        shift = (inverse) ? conj(exp(Kokkos::complex<PrecisionT>(
                                0, static_cast<PrecisionT>(M_PI / 4))))
                          : exp(Kokkos::complex<PrecisionT>(
                                0, static_cast<PrecisionT>(M_PI / 4)));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        const std::size_t i1 = i0 | rev_wire_shift;
        arr[i1] *= shift;
    }
};

template <class PrecisionT, bool inverse = false> struct cnotFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    cnotFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                std::size_t num_qubits, const std::vector<size_t> &wires,
                [[maybe_unused]] const std::vector<PrecisionT> &params) {
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
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire1_shift | rev_wire0_shift;

        swap(arr[i10], arr[i11]);
    }
};

template <class PrecisionT, bool inverse = false> struct cyFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    cyFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
              std::size_t num_qubits, const std::vector<size_t> &wires,
              [[maybe_unused]] const std::vector<PrecisionT> &params) {
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
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire1_shift | rev_wire0_shift;
        Kokkos::complex<PrecisionT> v10 = arr[i10];
        arr[i10] = Kokkos::complex<PrecisionT>{imag(arr[i11]), -real(arr[i11])};
        arr[i11] = Kokkos::complex<PrecisionT>{-imag(v10), real(v10)};
    }
};

template <class PrecisionT, bool inverse = false> struct czFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    czFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
              std::size_t num_qubits, const std::vector<size_t> &wires,
              [[maybe_unused]] const std::vector<PrecisionT> &params) {
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
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;
        arr[i11] *= -1;
    }
};

template <class PrecisionT, bool inverse = false> struct swapFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    swapFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                std::size_t num_qubits, const std::vector<size_t> &wires,
                [[maybe_unused]] const std::vector<PrecisionT> &params) {
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
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i01 = i00 | rev_wire0_shift;
        swap(arr[i10], arr[i01]);
    }
};

template <class PrecisionT, bool inverse = false> struct cSWAPFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire2;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire2_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_mid;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_hmiddle;
    std::size_t parity_lmiddle;

    Kokkos::complex<PrecisionT> shifts_0;
    Kokkos::complex<PrecisionT> shifts_1;
    Kokkos::complex<PrecisionT> shifts_2;

    cSWAPFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                 std::size_t num_qubits, const std::vector<size_t> &wires,
                 [[maybe_unused]] const std::vector<PrecisionT> &params) {
        rev_wire0 = num_qubits - wires[2] - 1;
        rev_wire1 = num_qubits - wires[1] - 1;
        rev_wire2 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        if (rev_wire2 < rev_wire_min) {
            rev_wire_mid = rev_wire_min;
            rev_wire_min = rev_wire2;
        } else if (rev_wire2 > rev_wire_max) {
            rev_wire_mid = rev_wire_max;
            rev_wire_max = rev_wire2;
        } else {
            rev_wire_mid = rev_wire2;
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_mid);
        parity_hmiddle =
            fillLeadingOnes(rev_wire_mid + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i000 =
            ((k << 3U) & parity_high) | ((k << 2U) & parity_hmiddle) |
            ((k << 1U) & parity_lmiddle) | (k & parity_low);
        const std::size_t i101 = i000 | rev_wire2_shift | rev_wire0_shift;
        const std::size_t i110 = i000 | rev_wire2_shift | rev_wire1_shift;

        swap(arr[i101], arr[i110]);
    }
};

template <class PrecisionT, bool inverse = false> struct toffoliFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire2;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire2_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_mid;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_hmiddle;
    std::size_t parity_lmiddle;

    Kokkos::complex<PrecisionT> shifts_0;
    Kokkos::complex<PrecisionT> shifts_1;
    Kokkos::complex<PrecisionT> shifts_2;

    toffoliFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> &arr_,
                   std::size_t num_qubits, const std::vector<size_t> &wires,
                   [[maybe_unused]] const std::vector<PrecisionT> &params) {
        rev_wire0 = num_qubits - wires[2] - 1;
        rev_wire1 = num_qubits - wires[1] - 1;
        rev_wire2 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        if (rev_wire2 < rev_wire_min) {
            rev_wire_mid = rev_wire_min;
            rev_wire_min = rev_wire2;
        } else if (rev_wire2 > rev_wire_max) {
            rev_wire_mid = rev_wire_max;
            rev_wire_max = rev_wire2;
        } else {
            rev_wire_mid = rev_wire2;
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_mid);
        parity_hmiddle =
            fillLeadingOnes(rev_wire_mid + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i000 =
            ((k << 3U) & parity_high) | ((k << 2U) & parity_hmiddle) |
            ((k << 1U) & parity_lmiddle) | (k & parity_low);
        const std::size_t i111 =
            i000 | rev_wire2_shift | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i110 = i000 | rev_wire2_shift | rev_wire1_shift;

        swap(arr[i111], arr[i110]);
    }
};

} // namespace Pennylane::LightningKokkos::Functors