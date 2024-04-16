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
 * @file PauliGenerator.hpp
 * Defines generators for RX, RY, RZ
 */
#pragma once
#include <complex>
#include <cstdlib>
#include <vector>

namespace Pennylane::LightningQubit::Gates {
/**
 * @brief Define generators of RX, RY, RZ using the Pauli gates.
 * @rst
 * A Generator for a unitary operator :math:`U` is :math:`G` such that
 * :math:`e^{iGt} = U`.
 * @endrst
 */
template <class GateImplementation> class PauliGenerator {
  public:
    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorRX(std::complex<PrecisionT> *data, size_t num_qubits,
                     const std::vector<size_t> &wires, bool adj) -> PrecisionT {
        GateImplementation::applyPauliX(data, num_qubits, wires, adj);
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }
    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorRY(std::complex<PrecisionT> *data, size_t num_qubits,
                     const std::vector<size_t> &wires, bool adj) -> PrecisionT {
        GateImplementation::applyPauliY(data, num_qubits, wires, adj);
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }
    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorRZ(std::complex<PrecisionT> *data, size_t num_qubits,
                     const std::vector<size_t> &wires, bool adj) -> PrecisionT {
        GateImplementation::applyPauliZ(data, num_qubits, wires, adj);
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }
};
} // namespace Pennylane::LightningQubit::Gates
