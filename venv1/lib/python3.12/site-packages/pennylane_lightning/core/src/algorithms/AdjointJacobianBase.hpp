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
 * @file AdjointJacobianBase.hpp
 * Defines the base class to support the adjoint Jacobian differentiation
 * method.
 */
#pragma once
#include <span>

#include "JacobianData.hpp"
#include "Observables.hpp"

namespace Pennylane::Algorithms {
/**
 * @brief Adjoint Jacobian evaluator following the method of arXiV:2009.02823.
 *
 * @tparam StateVectorT State vector type.
 */
template <class StateVectorT, class Derived> class AdjointJacobianBase {
  private:
    using ComplexT = typename StateVectorT::ComplexT;
    using PrecisionT = typename StateVectorT::PrecisionT;

  protected:
    AdjointJacobianBase() = default;
    AdjointJacobianBase(const AdjointJacobianBase &) = default;
    AdjointJacobianBase(AdjointJacobianBase &&) noexcept = default;
    AdjointJacobianBase &operator=(const AdjointJacobianBase &) = default;
    AdjointJacobianBase &operator=(AdjointJacobianBase &&) noexcept = default;

    /**
     * @brief Apply all operations from given
     * `%OpsData<StateVectorT>` object to `%UpdatedStateVectorT`.
     *
     * @tparam UpdatedStateVectorT
     * @param state Statevector to be updated.
     * @param operations Operations to apply.
     * @param adj Take the adjoint of the given operations.
     */
    template <class UpdatedStateVectorT>
    inline void applyOperations(UpdatedStateVectorT &state,
                                const OpsData<StateVectorT> &operations,
                                bool adj = false) {
        for (size_t op_idx = 0; op_idx < operations.getOpsName().size();
             op_idx++) {
            if (operations.getOpsControlledWires()[op_idx].empty()) {
                state.applyOperation(operations.getOpsName()[op_idx],
                                     operations.getOpsWires()[op_idx],
                                     operations.getOpsInverses()[op_idx] ^ adj,
                                     operations.getOpsParams()[op_idx],
                                     operations.getOpsMatrices()[op_idx]);
            } else {
                state.applyOperation(
                    operations.getOpsName()[op_idx],
                    operations.getOpsControlledWires()[op_idx],
                    operations.getOpsControlledValues()[op_idx],
                    operations.getOpsWires()[op_idx],
                    operations.getOpsInverses()[op_idx] ^ adj,
                    operations.getOpsParams()[op_idx],
                    operations.getOpsMatrices()[op_idx]);
            }
        }
    }

    /**
     * @brief Apply the adjoint indexed operation from
     * `%OpsData<StateVectorT>` object to `%UpdatedStateVectorT`.
     *
     * @tparam UpdatedStateVectorT updated state vector type.
     * @param state Statevector to be updated.
     * @param operations Operations to apply.
     * @param op_idx Adjointed operation index to apply.
     */
    template <class UpdatedStateVectorT>
    inline void applyOperationAdj(UpdatedStateVectorT &state,
                                  const OpsData<StateVectorT> &operations,
                                  size_t op_idx) {
        if (operations.getOpsControlledWires()[op_idx].empty()) {
            state.applyOperation(operations.getOpsName()[op_idx],
                                 operations.getOpsWires()[op_idx],
                                 !operations.getOpsInverses()[op_idx],
                                 operations.getOpsParams()[op_idx],
                                 operations.getOpsMatrices()[op_idx]);
        } else {
            state.applyOperation(operations.getOpsName()[op_idx],
                                 operations.getOpsControlledWires()[op_idx],
                                 operations.getOpsControlledValues()[op_idx],
                                 operations.getOpsWires()[op_idx],
                                 !operations.getOpsInverses()[op_idx],
                                 operations.getOpsParams()[op_idx],
                                 operations.getOpsMatrices()[op_idx]);
        }
    }

    /**
     * @brief Apply the adjoint indexed operation from several
     * `%OpsData<StateVectorT>` objects to `%UpdatedStateVectorT` objects.
     *
     * @param states Vector of all statevectors; 1 per observable
     * @param operations Operations list.
     * @param op_idx Index of given operation within operations list to take
     * adjoint of.
     */
    inline void applyOperationsAdj(std::vector<StateVectorT> &states,
                                   const OpsData<StateVectorT> &operations,
                                   size_t op_idx) {
        for (auto &state : states) {
            applyOperationAdj(state, operations, op_idx);
        }
    }

    /**
     * @brief Applies the gate generator for a given parametric gate. Returns
     * the associated scaling coefficient.
     *
     * @param sv Statevector data to operate upon.
     * @param op_name Name of parametric gate.
     * @param wires Wires to operate upon.
     * @param adj Indicate whether to take the adjoint of the operation.
     * @return PrecisionT Generator scaling coefficient.
     */
    inline auto applyGenerator(StateVectorT &sv, const std::string &op_name,
                               const std::vector<size_t> &wires, const bool adj)
        -> PrecisionT {
        return sv.applyGenerator(op_name, wires, adj);
    }

    /**
     * @brief Applies the gate generator for a given parametric gate. Returns
     * the associated scaling coefficient.
     *
     * @param sv Statevector data to operate upon.
     * @param op_name Name of parametric gate.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to operate upon.
     * @param adj Indicate whether to take the adjoint of the operation.
     * @return PrecisionT Generator scaling coefficient.
     */
    inline auto applyGenerator(StateVectorT &sv, const std::string &op_name,
                               const std::vector<size_t> &controlled_wires,
                               const std::vector<bool> &controlled_values,
                               const std::vector<size_t> &wires, const bool adj)
        -> PrecisionT {
        return sv.applyGenerator(op_name, controlled_wires, controlled_values,
                                 wires, adj);
    }

    /**
     * @brief Apply a given `%Observable<StateVectorT>` object to
     * `%StateVectorT`.
     *
     * @param state Statevector to be updated.
     * @param observable Observable to apply.
     */
    inline void applyObservable(StateVectorT &state,
                                const Observable<StateVectorT> &observable) {
        observable.applyInPlace(state);
    }

    /**
     * @brief Apply several `%Observable<StateVectorT>` object. to
     * `%StateVectorT` objects.
     *
     * @param states Vector of statevector copies, one per observable.
     * @param reference_state Reference statevector
     * @param observables Vector of observables to apply to each statevector.
     */
    inline void applyObservables(
        std::vector<StateVectorT> &states, const StateVectorT &reference_state,
        const std::vector<std::shared_ptr<Observable<StateVectorT>>>
            &observables) {
        size_t num_observables = observables.size();
        for (size_t i = 0; i < num_observables; i++) {
            states[i].updateData(reference_state);
            applyObservable(states[i], *observables[i]);
        }
    }

    /**
     * @brief Calculates the statevector's Jacobian for the selected set
     * of parametric gates.
     *
     * @param jac Preallocated vector for Jacobian data results.
     * @param jd JacobianData represents the QuantumTape to differentiate.
     * @param apply_operations Indicate whether to apply operations to tape.psi
     * prior to calculation.
     */
    inline void adjointJacobian(std::span<PrecisionT> jac,
                                const JacobianData<StateVectorT> &jd,
                                const StateVectorT &ref_data = {0},
                                bool apply_operations = false) {
        return static_cast<Derived *>(this)->adjointJacobian(jac, jd, ref_data,
                                                             apply_operations);
    }

  public:
    ~AdjointJacobianBase() = default;
};
} // namespace Pennylane::Algorithms