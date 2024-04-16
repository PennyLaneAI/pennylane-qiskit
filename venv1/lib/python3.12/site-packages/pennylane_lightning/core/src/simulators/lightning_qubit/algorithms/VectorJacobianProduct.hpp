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
 * Define vjp algorithm for a statevector
 */
#pragma once

#include <span>

#include "AdjointJacobianBase.hpp"
#include "BitUtil.hpp" // log2PerfectPower
#include "JacobianData.hpp"
#include "LinearAlgebra.hpp" // innerProdC
#include "StateVectorLQubitManaged.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Algorithms;
using Pennylane::LightningQubit::Util::innerProdC;
using Pennylane::Util::log2PerfectPower;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Algorithms {
/**
 * @brief Vector Jacobian Product (VJP) functor.
 *
 * @tparam StateVectorT State vector type.
 */
template <class StateVectorT>
class VectorJacobianProduct final
    : public AdjointJacobianBase<StateVectorT,
                                 VectorJacobianProduct<StateVectorT>> {
  private:
    using ComplexT = typename StateVectorT::ComplexT;
    using PrecisionT = typename StateVectorT::PrecisionT;

  public:
    /**
     * @brief Compute vector Jacobian product for a statevector Jacobian.
     *
     * @rst
     * Product of statevector Jacobian :math:`J_{ij} = \partial_{\theta_j}
     * \psi_{\pmb{\theta}}(i)` and a vector, i.e. this function returns
     * :math:`w = v^\dagger J`. This is
     * equivalent to
     *
     * .. math::
     *
     *     w_j = \langle v | \partial_{\theta_j} \psi_{\pmb{\theta}} \rangle
     *
     * where :math:`\pmb{\theta}=(\theta_1, \theta_2, \cdots)` is a list of all
     * parameters and $v = dy$.
     *
     * Note that :math:`J` is :math:`2^n \times m` matrix where
     * :math:`n` is the number of qubits and :math:`m` is the number of
     * trainable parameters in the tape.
     * Thus the result vector is length :math:`m`.
     * @endrst
     *
     * @param jac Preallocated vector for Jacobian data results.
     * @param jd Jacobian data
     * @param vec A cotangent vector of size 2^n
     * @param apply_operations Assume the given state is an input state and
     * apply operations if true
     */
    void operator()(std::span<ComplexT> jac,
                    const JacobianData<StateVectorT> &jd,
                    std::span<const ComplexT> dy,
                    bool apply_operations = false) {
        PL_ASSERT(dy.size() == jd.getSizeStateVec());

        if (!jd.hasTrainableParams()) {
            return;
        }

        const OpsData<StateVectorT> &ops = jd.getOperations();
        const std::vector<std::string> &ops_name = ops.getOpsName();

        // We can assume the trainable params are sorted (from Python)
        const size_t num_param_ops = ops.getNumParOps();
        const auto &trainable_params = jd.getTrainableParams();

        PL_ABORT_IF_NOT(jac.size() == trainable_params.size(),
                        "The size of preallocated jacobian must be same as "
                        "the number of trainable parameters.");

        // Create $U_{1:p}\vert \lambda \rangle$
        StateVectorLQubitManaged<PrecisionT> lambda(jd.getPtrStateVec(),
                                                    jd.getSizeStateVec());

        // Apply given operations to statevector if requested
        if (apply_operations) {
            this->applyOperations(lambda, ops);
        }
        StateVectorLQubitManaged<PrecisionT> mu(dy.data(), dy.size());
        StateVectorLQubitManaged<PrecisionT> mu_d(
            log2PerfectPower(jd.getSizeStateVec()));

        const auto tp_rend = trainable_params.rend();
        auto tp_it = trainable_params.rbegin();
        size_t current_param_idx =
            num_param_ops - 1; // total number of parametric ops
        size_t trainable_param_idx = trainable_params.size() - 1;

        for (int op_idx = static_cast<int>(ops_name.size() - 1); op_idx >= 0;
             op_idx--) {
            PL_ABORT_IF(ops.getOpsParams()[op_idx].size() > 1,
                        "The operation is not supported using the adjoint "
                        "differentiation method");
            if ((ops_name[op_idx] == "QubitStateVector") ||
                (ops_name[op_idx] == "StatePrep") ||
                (ops_name[op_idx] == "BasisState")) {
                continue; // ignore them
            }

            if (tp_it == tp_rend) {
                break; // All done
            }

            if (ops.hasParams(op_idx)) {
                if (current_param_idx == *tp_it) {
                    // if current parameter is a trainable parameter
                    mu_d.updateData(mu.getDataVector());
                    const auto scalingFactor =
                        (ops.getOpsControlledWires()[op_idx].empty())
                            ? mu_d.applyGenerator(
                                  ops_name[op_idx], ops.getOpsWires()[op_idx],
                                  !ops.getOpsInverses()[op_idx]) *
                                  (ops.getOpsInverses()[op_idx] ? -1 : 1)
                            : mu_d.applyGenerator(
                                  ops_name[op_idx],
                                  ops.getOpsControlledWires()[op_idx],
                                  ops.getOpsControlledValues()[op_idx],
                                  ops.getOpsWires()[op_idx],
                                  !ops.getOpsInverses()[op_idx]) *
                                  (ops.getOpsInverses()[op_idx] ? -1 : 1);

                    jac[trainable_param_idx] =
                        ComplexT{0.0, scalingFactor} *
                        innerProdC(mu_d.getDataVector(),
                                   lambda.getDataVector());
                    --trainable_param_idx;
                    ++tp_it;
                }
                --current_param_idx;
            }
            this->applyOperationAdj(lambda, ops, static_cast<size_t>(op_idx));
            this->applyOperationAdj(mu, ops, static_cast<size_t>(op_idx));
        }
    }
};
} // namespace Pennylane::LightningQubit::Algorithms