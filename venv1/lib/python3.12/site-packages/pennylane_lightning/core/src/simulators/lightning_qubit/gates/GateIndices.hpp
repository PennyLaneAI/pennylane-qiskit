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
 * Defines a non-template utility functions
 */

#pragma once

#include <cstdlib>
#include <set>
#include <vector>

#include "Util.hpp"

/**
 * This namespace defines helper functions that calculate indices appears in
 * implementations of gate operations. However, as using bit operations within a
 * for loop is usually better performing, consider to use other options before
 * using functions below.
 */
namespace Pennylane::LightningQubit::Gates {
/**
 * @brief Get indices of statevector data not participating in application
 * operation.
 *
 * @param indicesToExclude Indices to exclude from this call.
 * @param num_qubits Total number of qubits for statevector.
 * @return std::vector<size_t>
 */
auto getIndicesAfterExclusion(const std::vector<size_t> &indicesToExclude,
                              size_t num_qubits) -> std::vector<size_t>;

/**
 * @brief Generate indices for applying operations.
 *
 * This method will return the statevector indices participating in the
 * application of a gate to a given set of qubits.
 *
 * @param qubitIndices Indices of the qubits to apply operations.
 * @param num_qubits Number of qubits in register.
 * @return std::vector<size_t>
 */
auto generateBitPatterns(const std::vector<size_t> &qubitIndices,
                         size_t num_qubits) -> std::vector<size_t>;

/**
 * @brief Internal utility struct to track data indices of application for
 * operations.
 *
 * Each external index represents a bitstring outside of wires. For example, if
 * we have four qubits and wires = {1,2}, external indices are {0000B, 0001B,
 * 1000B, 1001B}. For each external index, internal indices are bitstrings for
 * the wires. In the example, we have internal indices {0000B, 0010B, 0100B,
 * 0110B}.
 */
struct GateIndices {
    const std::vector<size_t> internal; /**< Internal indices.
                                          For the given wires with size n_wire,
                                          the output size is 2^n_wire. */

    const std::vector<size_t>
        external; /**< external External indices.
                    For the given wires with size n_wire, the
                    output size is 2^(num_qubits - n_wires). */

    /**
     * @brief Create indices for gates.
     */
    GateIndices(const std::vector<size_t> &wires, size_t num_qubits)
        : internal{generateBitPatterns(wires, num_qubits)},
          external{generateBitPatterns(
              getIndicesAfterExclusion(wires, num_qubits), num_qubits)} {}
};
} // namespace Pennylane::LightningQubit::Gates
