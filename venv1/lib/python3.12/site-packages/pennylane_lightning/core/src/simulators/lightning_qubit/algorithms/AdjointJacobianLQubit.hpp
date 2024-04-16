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
 * @file AdjointJacobian.hpp
 * Defines the methods and class to support the adjoint Jacobian differentiation
 * method.
 */
#pragma once
#include <span>
#include <type_traits>
#include <vector>

#include "AdjointJacobianBase.hpp"
#include "JacobianData.hpp"
#include "LinearAlgebra.hpp" // innerProdC, Transpose
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"

// using namespace Pennylane;
/// @cond DEV
namespace {
using namespace Pennylane::Algorithms;
using namespace Pennylane::Util::MemoryStorageLocation;

using Pennylane::LightningQubit::Util::innerProdC;
using Pennylane::LightningQubit::Util::Transpose;

} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Algorithms {
/**
 * @brief Adjoint Jacobian evaluator following the method of arXiV:2009.02823.
 *
 * @tparam StateVectorT State vector type.
 */
template <class StateVectorT>
class AdjointJacobian final
    : public AdjointJacobianBase<StateVectorT, AdjointJacobian<StateVectorT>> {
  private:
    using ComplexT = typename StateVectorT::ComplexT;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using BaseType =
        AdjointJacobianBase<StateVectorT, AdjointJacobian<StateVectorT>>;

    /**
     * @brief Utility method to update the Jacobian at a given index by
     * calculating the overlap between two given states.
     *
     * @param states Vector of all statevectors, 1 per observable. Data will be
     * conjugated.
     * @param sv Statevector |sv>
     * @param jac Jacobian receiving the values.
     * @param scaling_coeff Generator coefficient for given gate derivative.
     * @param obs_index Observable index position of Jacobian to update.
     * @param param_index Parameter index position of Jacobian to update.
     */
    template <class OtherStateVectorT>
    inline void
    updateJacobian(std::vector<StateVectorT> &states, OtherStateVectorT &sv,
                   std::span<PrecisionT> &jac, PrecisionT scaling_coeff,
                   size_t obs_idx, size_t mat_row_idx) {
        jac[mat_row_idx + obs_idx] =
            -2 * scaling_coeff *
            std::imag(innerProdC(states[obs_idx].getData(), sv.getData(),
                                 sv.getLength()));
    }

    /**
     * @brief OpenMP accelerated application of observables to given
     * statevectors.
     *
     * @tparam RefStateVectorT reference state vector type.
     * @param states Vector of statevector copies, one per observable.
     * @param reference_state Reference statevector
     * @param observables Vector of observables to apply to each statevector.
     */
    template <class RefStateVectorT>
    inline void applyObservables(
        std::vector<StateVectorT> &states,
        const RefStateVectorT &reference_state,
        const std::vector<std::shared_ptr<Observable<StateVectorT>>>
            &observables) {
        std::exception_ptr ex = nullptr;
        size_t num_observables = observables.size();

        if (num_observables > 1) {
            /* Globally scoped exception value to be captured within OpenMP
             * block. See the following for OpenMP design decisions:
             * https://www.openmp.org/wp-content/uploads/openmp-examples-4.5.0.pdf
             * */
            // clang-format off

            #if defined(_OPENMP)
                #pragma omp parallel default(none)                                 \
                shared(states, reference_state, observables, ex, num_observables)
            {
                #pragma omp for
            #endif
                for (size_t h_i = 0; h_i < num_observables; h_i++) {
                    try {
                        states[h_i].updateData(reference_state.getData(),
                                               reference_state.getLength());
                        BaseType::applyObservable(states[h_i], *observables[h_i]);
                    } catch (...) {
                        #if defined(_OPENMP)
                            #pragma omp critical
                        #endif
                        ex = std::current_exception();
                        #if defined(_OPENMP)
                            #pragma omp cancel for
                        #endif
                    }
                }
            #if defined(_OPENMP)
                if (ex) {
                    #pragma omp cancel parallel
                }
            }
            #endif
            if (ex) {
                std::rethrow_exception(ex);
            }
            // clang-format on
        } else {
            states[0].updateData(reference_state.getData(),
                                 reference_state.getLength());
            BaseType::applyObservable(states[0], *observables[0]);
        }
    }

    /**
     * @brief OpenMP accelerated application of adjoint operations to
     * statevectors.
     *
     * @param states Vector of all statevectors; 1 per observable
     * @param operations Operations list.
     * @param op_idx Index of given operation within operations list to take
     * adjoint of.
     */
    inline void applyOperationsAdj(std::vector<StateVectorT> &states,
                                   const OpsData<StateVectorT> &operations,
                                   size_t op_idx) {
        // clang-format off
        // Globally scoped exception value to be captured within OpenMP block.
        // See the following for OpenMP design decisions:
        // https://www.openmp.org/wp-content/uploads/openmp-examples-4.5.0.pdf
        std::exception_ptr ex = nullptr;
        size_t num_states = states.size();
        #if defined(_OPENMP)
            #pragma omp parallel default(none)                                 \
                shared(states, operations, op_idx, ex, num_states)
        {
            #pragma omp for
        #endif
            for (size_t st_idx = 0; st_idx < num_states; st_idx++) {
                try {
                    BaseType::applyOperationAdj(states[st_idx], operations, op_idx);
                } catch (...) {
                    #if defined(_OPENMP)
                        #pragma omp critical
                    #endif
                    ex = std::current_exception();
                    #if defined(_OPENMP)
                        #pragma omp cancel for
                    #endif
                }
            }
        #if defined(_OPENMP)
            if (ex) {
                #pragma omp cancel parallel
            }
        }
        #endif
        if (ex) {
            std::rethrow_exception(ex);
        }
        // clang-format on
    }

  public:
    /**
     * @brief Calculates the Jacobian for the statevector for the selected set
     * of parametric gates.
     *
     * For the statevector data associated with `psi` of length `num_elements`,
     * we make internal copies, one per required observable. The `operations`
     * will be applied to the internal statevector copies, with the operation
     * indices participating in the gradient calculations given in
     * `trainableParams`, and the overall number of parameters for the gradient
     * calculation provided within `num_params`. The resulting row-major ordered
     * `jac` matrix representation will be of size `jd.getSizeStateVec() *
     * jd.getObservables().size()`. OpenMP is used to enable independent
     * operations to be offloaded to threads.
     *
     * @param jac Preallocated vector for Jacobian data results.
     * @param jd JacobianData represents the QuantumTape to differentiate.
     * @param apply_operations Indicate whether to apply operations to tape.psi
     * prior to calculation.
     */
    void adjointJacobian(std::span<PrecisionT> jac,
                         const JacobianData<StateVectorT> &jd,
                         [[maybe_unused]] const StateVectorT &ref_data = {0},
                         bool apply_operations = false) {
        const OpsData<StateVectorT> &ops = jd.getOperations();
        const std::vector<std::string> &ops_name = ops.getOpsName();

        const auto &obs = jd.getObservables();
        const size_t num_observables = obs.size();

        // We can assume the trainable params are sorted (from Python)
        const std::vector<size_t> &tp = jd.getTrainableParams();
        const size_t tp_size = tp.size();
        const size_t num_param_ops = ops.getNumParOps();

        if (!jd.hasTrainableParams()) {
            return;
        }

        PL_ABORT_IF_NOT(
            jac.size() == tp_size * num_observables,
            "The size of preallocated jacobian must be same as "
            "the number of trainable parameters times the number of "
            "observables provided.");

        // Track positions within par and non-par operations
        size_t trainableParamNumber = tp_size - 1;
        size_t current_param_idx =
            num_param_ops - 1; // total number of parametric ops

        // Create $U_{1:p}\vert \lambda \rangle$
        StateVectorLQubitManaged<PrecisionT> lambda(jd.getPtrStateVec(),
                                                    jd.getSizeStateVec());
        // Apply given operations to statevector if requested
        if (apply_operations) {
            BaseType::applyOperations(lambda, ops);
        }

        const auto tp_rend = tp.rend();
        auto tp_it = tp.rbegin();

        // Create observable-applied and mu state vectors
        std::unique_ptr<std::vector<StateVectorT>> H_lambda;

        // Pointer to data storage for StateVectorLQubitRaw<PrecisionT>:
        std::unique_ptr<std::vector<std::vector<ComplexT>>> H_lambda_storage;
        size_t lambda_qubits = lambda.getNumQubits();
        if constexpr (std::is_same_v<typename StateVectorT::MemoryStorageT,
                                     MemoryStorageLocation::Internal>) {
            H_lambda = std::make_unique<std::vector<StateVectorT>>(
                num_observables, StateVectorT{lambda_qubits});
        } else if constexpr (std::is_same_v<
                                 typename StateVectorT::MemoryStorageT,
                                 MemoryStorageLocation::External>) {
            H_lambda_storage =
                std::make_unique<std::vector<std::vector<ComplexT>>>(
                    num_observables, std::vector<ComplexT>(lambda.getLength()));
            H_lambda = std::make_unique<std::vector<StateVectorT>>();
            for (size_t ind = 0; ind < num_observables; ind++) {
                (*H_lambda_storage)[ind][0] = {1.0, 0};

                StateVectorT sv((*H_lambda_storage)[ind].data(),
                                (*H_lambda_storage)[ind].size());
                H_lambda->push_back(sv);
            }
        } else {
            /// LCOV_EXCL_START
            PL_ABORT("Undefined memory storage location for StateVectorT.");
            /// LCOV_EXCL_STOP
        }

        StateVectorLQubitManaged<PrecisionT> mu(lambda_qubits);

        applyObservables(*H_lambda, lambda, obs);

        for (int op_idx = static_cast<int>(ops_name.size() - 1); op_idx >= 0;
             op_idx--) {
            PL_ABORT_IF(ops.getOpsParams()[op_idx].size() > 1,
                        "The operation is not supported using the adjoint "
                        "differentiation method");
            if ((ops_name[op_idx] == "QubitStateVector") ||
                (ops_name[op_idx] == "StatePrep") ||
                (ops_name[op_idx] == "BasisState")) {
                continue; // Ignore them
            }

            if (tp_it == tp_rend) {
                break; // All done
            }
            mu.updateData(lambda.getData(), lambda.getLength());
            BaseType::applyOperationAdj(lambda, ops, op_idx);

            if (ops.hasParams(op_idx)) {
                if (current_param_idx == *tp_it) {
                    // if current parameter is a trainable parameter
                    const PrecisionT scalingFactor =
                        (ops.getOpsControlledWires()[op_idx].empty())
                            ? mu.applyGenerator(ops_name[op_idx],
                                                ops.getOpsWires()[op_idx],
                                                !ops.getOpsInverses()[op_idx]) *
                                  (ops.getOpsInverses()[op_idx] ? -1 : 1)
                            : mu.applyGenerator(
                                  ops_name[op_idx],
                                  ops.getOpsControlledWires()[op_idx],
                                  ops.getOpsControlledValues()[op_idx],
                                  ops.getOpsWires()[op_idx],
                                  !ops.getOpsInverses()[op_idx]) *
                                  (ops.getOpsInverses()[op_idx] ? -1 : 1);

                    const size_t mat_row_idx =
                        trainableParamNumber * num_observables;

                    // clang-format off

                #if defined(_OPENMP)
                #pragma omp parallel for default(none)                         \
                    shared(H_lambda, jac, mu, scalingFactor, mat_row_idx,      \
                            num_observables)
                #endif
                    // clang-format on

                    for (size_t obs_idx = 0; obs_idx < num_observables;
                         obs_idx++) {
                        updateJacobian(*H_lambda, mu, jac, scalingFactor,
                                       obs_idx, mat_row_idx);
                    }
                    trainableParamNumber--;
                    ++tp_it;
                }
                current_param_idx--;
            }
            applyOperationsAdj(*H_lambda, ops, static_cast<size_t>(op_idx));
        }
        const auto jac_transpose = Transpose(std::span<const PrecisionT>{jac},
                                             tp_size, num_observables);
        std::copy(std::begin(jac_transpose), std::end(jac_transpose),
                  std::begin(jac));
    }
};
} // namespace Pennylane::LightningQubit::Algorithms