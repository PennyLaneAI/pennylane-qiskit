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
 * Defines test helper methods for wires.
 */

#pragma once

#include <cstdlib>
#include <numeric> // iota
#include <vector>

#include "BitUtil.hpp"
#include "Constant.hpp"
#include "ConstantUtil.hpp" // array_has_elem, lookup
#include "Error.hpp"
#include "GateOperation.hpp"
#include "Macros.hpp"

namespace Pennylane::Util {
inline auto createWires(Pennylane::Gates::GateOperation op, size_t num_qubits)
    -> std::vector<size_t> {
    if (array_has_elem(Pennylane::Gates::Constant::multi_qubit_gates, op)) {
        std::vector<size_t> wires(num_qubits);
        std::iota(wires.begin(), wires.end(), 0);
        return wires;
    }
    switch (lookup(Pennylane::Gates::Constant::gate_wires, op)) {
    case 1:
        return {0};
    case 2:
        return {0, 1};
    case 3:
        return {0, 1, 2};
    case 4:
        return {0, 1, 2, 3};
    default:
        PL_ABORT("The number of wires for a given gate is unknown.");
    }
    return {};
}

inline auto createWires(Pennylane::Gates::ControlledGateOperation op,
                        size_t num_qubits) -> std::vector<size_t> {
    if (array_has_elem(Pennylane::Gates::Constant::controlled_multi_qubit_gates,
                       op)) {
        std::vector<size_t> wires(num_qubits - 2);
        std::iota(wires.begin(), wires.end(), 0);
        return wires;
    }
    switch (lookup(Pennylane::Gates::Constant::controlled_gate_wires, op)) {
    case 1:
        return {0};
    case 2:
        return {0, 1};
    case 3:
        return {0, 1, 2};
    case 4:
        return {0, 1, 2, 3};
    default:
        PL_ABORT("The number of wires for a given gate is unknown.");
    }
    return {};
}

template <class PrecisionT>
auto createParams(Pennylane::Gates::GateOperation op)
    -> std::vector<PrecisionT> {
    switch (lookup(Pennylane::Gates::Constant::gate_num_params, op)) {
    case 0:
        return {};
    case 1:
        return {static_cast<PrecisionT>(0.312)};
    case 3:
        return {static_cast<PrecisionT>(0.128), static_cast<PrecisionT>(-0.563),
                static_cast<PrecisionT>(1.414)};
    default:
        PL_ABORT("The number of parameters for a given gate is unknown.");
    }
    return {};
}

class WiresGenerator {
  public:
    [[nodiscard]] virtual auto all_perms() const
        -> const std::vector<std::vector<size_t>> & = 0;
};

/**
 * @brief
 * @rst Generating all permutation of wires without ordering (often called
 * as combination). The size of all combination is given as :math:`n \choose r`.
 *
 * We use the recursion formula
 * :math:`{n \choose r} = {n \choose r-1} + {n-1 \choose r}`
 * @endrst
 */
class CombinationGenerator : public WiresGenerator {
  private:
    std::vector<size_t> v_;
    std::vector<std::vector<size_t>> all_perms_;

  public:
    void comb(size_t n, size_t r) {
        if (r == 0) {
            all_perms_.push_back(v_);
            return;
        }
        if (n < r) {
            return;
        }

        v_[r - 1] = n - 1;
        comb(n - 1, r - 1);

        comb(n - 1, r);
    }

    CombinationGenerator(size_t n, size_t r) {
        v_.resize(r);
        comb(n, r);
    }

    [[nodiscard]] auto all_perms() const
        -> const std::vector<std::vector<size_t>> & override {
        return all_perms_;
    }
};

/**
 * @brief
 * @rst Generating all permutation of wires with ordering. The size of all
 * permutation is given as :math:`{}_{n}P_r=n!/(n-r)!r!`.
 * @endrst
 *
 * We use the recursion formula
 * :math:`{}_n P_r = n {}_{n-1} P_{r-1}`
 */
class PermutationGenerator : public WiresGenerator {
  private:
    std::vector<std::vector<size_t>> all_perms_;
    std::vector<size_t> available_elems_;
    std::vector<size_t> v;

  public:
    void perm(size_t n, size_t r) {
        if (r == 0) {
            all_perms_.push_back(v);
            return;
        }
        for (size_t i = 0; i < n; i++) {
            v[r - 1] = available_elems_[i];
            std::swap(available_elems_[n - 1], available_elems_[i]);
            perm(n - 1, r - 1);
            std::swap(available_elems_[n - 1], available_elems_[i]);
        }
    }

    PermutationGenerator(size_t n, size_t r) {
        v.resize(r);

        available_elems_.resize(n);
        std::iota(available_elems_.begin(), available_elems_.end(), 0);
        perm(n, r);
    }

    [[nodiscard]] auto all_perms() const
        -> const std::vector<std::vector<size_t>> & override {
        return all_perms_;
    }
};

/**
 * @brief Create all possible combination of wires
 * for a given number of qubits and gate operation
 *
 * @param n_qubits Number of qubits
 * @param gate_op Gate operation
 * @param order Whether the ordering matters (if true, permutation is used)
 */
auto inline createAllWires(size_t n_qubits,
                           Pennylane::Gates::GateOperation gate_op, bool order)
    -> std::vector<std::vector<size_t>> {
    if (array_has_elem(Pennylane::Gates::Constant::multi_qubit_gates,
                       gate_op)) {
        // make all possible 2^N permutations
        std::vector<std::vector<size_t>> res;
        res.reserve((1U << n_qubits) - 1);
        for (size_t k = 1; k < (static_cast<size_t>(1U) << n_qubits); k++) {
            std::vector<size_t> wires;
            wires.reserve(std::popcount(k));

            for (size_t i = 0; i < n_qubits; i++) {
                if (((k >> i) & 1U) == 1U) {
                    wires.emplace_back(i);
                }
            }

            res.push_back(wires);
        }
        return res;
    } // else
    const size_t n_wires =
        lookup(Pennylane::Gates::Constant::gate_wires, gate_op);
    if (order) {
        return PermutationGenerator(n_qubits, n_wires).all_perms();
    } // else
    return CombinationGenerator(n_qubits, n_wires).all_perms();
}

} // namespace Pennylane::Util
