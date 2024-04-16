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
#include <span>

#include "AdjointJacobianBase.hpp"
#include "ObservablesKokkos.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos::Observables;
using namespace Pennylane::Algorithms;
using Pennylane::LightningKokkos::Util::getImagOfComplexInnerProduct;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Algorithms {
/**
 * @brief Kokkos-enabled adjoint Jacobian evaluator following the method of
 * arXiV:2009.02823
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
     * @param sv1 Statevector <sv1|. Data will be conjugated.
     * @param sv2 Statevector |sv2>
     * @param jac Jacobian receiving the values.
     * @param scaling_coeff Generator coefficient for given gate derivative.
     * @param idx Linear Jacobian index.
     */
    inline void updateJacobian(StateVectorT &sv1, StateVectorT &sv2,
                               std::span<PrecisionT> &jac,
                               PrecisionT scaling_coeff, size_t idx) {
        jac[idx] = -2 * scaling_coeff *
                   getImagOfComplexInnerProduct<PrecisionT>(sv1.getView(),
                                                            sv2.getView());
    }

  public:
    AdjointJacobian() = default;

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
                         const StateVectorT &ref_data,
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
        auto tp_it = tp.rbegin();
        const auto tp_rend = tp.rend();

        // Create $U_{1:p}\vert \lambda \rangle$
        StateVectorT lambda{ref_data};

        // Apply given operations to statevector if requested
        if (apply_operations) {
            BaseType::applyOperations(lambda, ops);
        }

        // Create observable-applied state-vectors
        std::vector<StateVectorT> H_lambda(num_observables,
                                           StateVectorT(lambda.getNumQubits()));
        BaseType::applyObservables(H_lambda, lambda, obs);

        StateVectorT mu{lambda.getNumQubits()};

        for (int op_idx = static_cast<int>(ops_name.size() - 1); op_idx >= 0;
             op_idx--) {
            PL_ABORT_IF(ops.getOpsParams()[op_idx].size() > 1,
                        "The operation is not supported using the adjoint "
                        "differentiation method");
            if ((ops_name[op_idx] == "QubitStateVector") ||
                (ops_name[op_idx] == "StatePrep") ||
                (ops_name[op_idx] == "BasisState")) {
                continue;
            }
            if (tp_it == tp_rend) {
                break; // All done
            }
            mu.updateData(lambda);
            BaseType::applyOperationAdj(lambda, ops, op_idx);

            if (ops.hasParams(op_idx)) {
                if (current_param_idx == *tp_it) {
                    const PrecisionT scalingFactor =
                        BaseType::applyGenerator(
                            mu, ops.getOpsName()[op_idx],
                            ops.getOpsWires()[op_idx],
                            !ops.getOpsInverses()[op_idx]) *
                        (ops.getOpsInverses()[op_idx] ? -1 : 1);
                    for (size_t obs_idx = 0; obs_idx < num_observables;
                         obs_idx++) {
                        const size_t idx =
                            trainableParamNumber + obs_idx * tp_size;
                        updateJacobian(H_lambda[obs_idx], mu, jac,
                                       scalingFactor, idx);
                    }
                    trainableParamNumber--;
                    ++tp_it;
                }
                current_param_idx--;
            }
            BaseType::applyOperationsAdj(H_lambda, ops,
                                         static_cast<size_t>(op_idx));
        }
    }
};

} // namespace Pennylane::LightningKokkos::Algorithms
