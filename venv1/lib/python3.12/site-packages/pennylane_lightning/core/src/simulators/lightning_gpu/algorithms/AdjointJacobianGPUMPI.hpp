// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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
 * @file AdjointJacobianGPUMPI.hpp
 */

#pragma once

#include <chrono>
#include <future>
#include <omp.h>
#include <span>
#include <variant>

#include "AdjointJacobianBase.hpp"
#include "DevTag.hpp"
#include "DevicePool.hpp"
#include "JacobianDataMPI.hpp"
#include "LinearAlg.hpp"
#include "MPILinearAlg.hpp"
#include "MPIManager.hpp"
#include "ObservablesGPUMPI.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Algorithms;
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningGPU::Observables;
namespace cuUtil = Pennylane::LightningGPU::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningGPU::Algorithms {

/**
 * @brief GPU-enabled adjoint Jacobian evaluator following the method of
 * arXiV:2009.02823
 *
 * @tparam StateVectorT State vector type.
 */
template <class StateVectorT>
class AdjointJacobianMPI final
    : public AdjointJacobianBase<StateVectorT,
                                 AdjointJacobianMPI<StateVectorT>> {
  private:
    using ComplexT = typename StateVectorT::ComplexT;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));
    using BaseType =
        AdjointJacobianBase<StateVectorT, AdjointJacobianMPI<StateVectorT>>;

    /**
     * @brief Utility method to update the Jacobian at a given index by
     * calculating the overlap between two given states.
     *
     * @param sv1s vector of statevector <sv1| (one for each observable). Data
     * will be conjugated.
     * @param sv2 Statevector |sv2>
     * @param jac Jacobian receiving the values.
     * @param scaling_coeff Generator coefficient for given gate derivative.
     * @param num_observables the number of observables of Jacobian to update.
     * @param param_index Parameter index position of Jacobian to update.
     * @param tp_size Size of trainable parameters.
     * @param device_buffer_jac_single_param a workspace buffer on the device of
     * size num_observables elements. Data will be ignored and overwritten.
     * @param host_buffer_jac_single_param a workspace buffer on the host of
     * size num_observables elements. Data will be ignored and overwritten.
     */
    inline void updateJacobian(const StateVectorT &sv1, const StateVectorT &sv2,
                               std::span<PrecisionT> &jac,
                               PrecisionT scaling_coeff, size_t obs_idx,
                               size_t param_index, size_t tp_size) {
        PL_ABORT_IF_NOT(sv1.getDataBuffer().getDevTag().getDeviceID() ==
                            sv2.getDataBuffer().getDevTag().getDeviceID(),
                        "Data exists on different GPUs. Aborting.");
        CFP_t result;
        innerProdC_CUDA_device(sv1.getData(), sv2.getData(), sv1.getLength(),
                               sv1.getDataBuffer().getDevTag().getDeviceID(),
                               sv1.getDataBuffer().getDevTag().getStreamID(),
                               sv1.getCublasCaller(), &result);
        auto jac_single_param =
            sv2.getMPIManager().template allreduce<CFP_t>(result, "sum");

        size_t idx = param_index + obs_idx * tp_size;
        jac[idx] = -2 * scaling_coeff * jac_single_param.y;
    }

  public:
    AdjointJacobianMPI() = default;

    /**
     * @brief Batches the adjoint_jacobian method over the available GPUs.
     *
     * @param jac Preallocated vector for Jacobian data results.
     * @param jd JacobianData represents the QuantumTape to differentiate.
     * @param apply_operations Indicate whether to apply operations to psi prior
     * to calculation.
     */
    void adjointJacobian_serial(std::span<PrecisionT> jac,
                                const JacobianDataMPI<StateVectorT> &jd,
                                bool apply_operations = false) {
        if (!jd.hasTrainableParams()) {
            return;
        }
        MPIManager mpi_manager_(jd.getMPIManager());
        DevTag<int> dt_local(jd.getDevTag());

        const OpsData<StateVectorT> &ops = jd.getOperations();
        const std::vector<std::string> &ops_name = ops.getOpsName();

        const auto &obs = jd.getObservables();
        const size_t num_observables = obs.size();

        const std::vector<size_t> &trainableParams = jd.getTrainableParams();
        const size_t tp_size = trainableParams.size();
        const size_t num_param_ops = ops.getNumParOps();

        PL_ABORT_IF_NOT(
            jac.size() == tp_size * num_observables,
            "The size of preallocated jacobian must be same as "
            "the number of trainable parameters times the number of "
            "observables provided.");

        StateVectorT lambda_ref(dt_local, jd.getNumGlobalQubits(),
                                jd.getNumLocalQubits(), jd.getPtrStateVec());

        // Apply given operations to statevector if requested
        if (apply_operations) {
            BaseType::applyOperations(lambda_ref, ops);
        }
        StateVectorT mu(dt_local, lambda_ref.getNumGlobalQubits(),
                        lambda_ref.getNumLocalQubits());

        StateVectorT H_lambda(dt_local, lambda_ref.getNumGlobalQubits(),
                              lambda_ref.getNumLocalQubits(),
                              lambda_ref.getData());

        StateVectorT lambda(dt_local, lambda_ref.getNumGlobalQubits(),
                            lambda_ref.getNumLocalQubits(),
                            lambda_ref.getData());

        for (size_t obs_idx = 0; obs_idx < num_observables; obs_idx++) {
            lambda.updateData(lambda_ref);

            // Create observable-applied state-vectors
            H_lambda.updateData(lambda_ref);

            BaseType::applyObservable(H_lambda, *obs[obs_idx]);

            size_t trainableParamNumber = tp_size - 1;
            // Track positions within par and non-par operations
            size_t current_param_idx =
                num_param_ops - 1; // total number of parametric ops
            auto tp_it = trainableParams.rbegin();
            const auto tp_rend = trainableParams.rend();

            for (int op_idx = static_cast<int>(ops_name.size() - 1);
                 op_idx >= 0; op_idx--) {
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
                            this->applyGenerator(
                                mu, ops.getOpsName()[op_idx],
                                ops.getOpsWires()[op_idx],
                                !ops.getOpsInverses()[op_idx]) *
                            (ops.getOpsInverses()[op_idx] ? -1 : 1);
                        updateJacobian(H_lambda, mu, jac, scalingFactor,
                                       obs_idx, trainableParamNumber, tp_size);
                        trainableParamNumber--;
                        ++tp_it;
                    }
                    current_param_idx--;
                }
                BaseType::applyOperationAdj(H_lambda, ops,
                                            static_cast<size_t>(op_idx));
            }
        }
    }

    /**
     * @brief Calculates the Jacobian for the statevector for the selected set
     * of parametric gates.
     *
     * For the statevector data associated with `psi` of length `num_elements`,
     * we make internal copies to a `%StateVectorCudaManaged<T>` object, with
     * one per required observable. The `operations` will be applied to the
     * internal statevector copies, with the operation indices participating in
     * the gradient calculations given in `trainableParams`, and the overall
     * number of parameters for the gradient calculation provided within
     * `num_params`. The resulting row-major ordered `jac` matrix representation
     * will be of size `jd.getSizeStateVec() * jd.getObservables().size()`.
     * OpenMP is used to enable independent operations to be offloaded to
     * threads.
     *
     * @param jac Preallocated vector for Jacobian data results.
     * @param jd JacobianData represents the QuantumTape to differentiate.
     * @param ref_data Reference to a `%StateVectorT` object.
     * @param apply_operations Indicate whether to apply operations to psi prior
     * to calculation.
     */
    void adjointJacobian(std::span<PrecisionT> jac,
                         const JacobianDataMPI<StateVectorT> &jd,
                         const StateVectorT &ref_data,
                         bool apply_operations = false) {
        if (!jd.hasTrainableParams()) {
            return;
        }

        const OpsData<StateVectorT> &ops = jd.getOperations();
        const std::vector<std::string> &ops_name = ops.getOpsName();

        const auto &obs = jd.getObservables();
        const size_t num_observables = obs.size();

        const std::vector<size_t> &trainableParams = jd.getTrainableParams();
        const size_t tp_size = trainableParams.size();
        const size_t num_param_ops = ops.getNumParOps();

        PL_ABORT_IF_NOT(
            jac.size() == tp_size * num_observables,
            "The size of preallocated jacobian must be same as "
            "the number of trainable parameters times the number of "
            "observables provided.");

        // Track positions within par and non-par operations
        size_t trainableParamNumber = tp_size - 1;
        size_t current_param_idx =
            num_param_ops - 1; // total number of parametric ops
        auto tp_it = trainableParams.rbegin();
        const auto tp_rend = trainableParams.rend();

        DevTag<int> dt_local(ref_data.getDataBuffer().getDevTag());
        dt_local.refresh();
        // Create $U_{1:p}\vert \lambda \rangle$
        SharedCusvHandle cusvhandle = make_shared_cusv_handle();
        SharedCublasCaller cublascaller = make_shared_cublas_caller();
        SharedCusparseHandle cusparsehandle = make_shared_cusparse_handle();

        StateVectorT lambda(ref_data);

        // Apply given operations to statevector if requested
        if (apply_operations) {
            BaseType::applyOperations(lambda, ops);
        }

        lambda.getMPIManager().Barrier();

        // Create observable-applied state-vectors
        using SVTypePtr = std::unique_ptr<StateVectorT>;
        std::unique_ptr<SVTypePtr[]> H_lambda(new SVTypePtr[num_observables]);

        for (size_t h_i = 0; h_i < num_observables; h_i++) {
            H_lambda[h_i] = std::make_unique<StateVectorT>(
                dt_local, lambda.getNumGlobalQubits(),
                lambda.getNumLocalQubits(), lambda.getData());
            BaseType::applyObservable(*H_lambda[h_i], *obs[h_i]);
        }

        StateVectorT mu(dt_local, lambda.getNumGlobalQubits(),
                        lambda.getNumLocalQubits());

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
                        updateJacobian(*H_lambda[obs_idx], mu, jac,
                                       scalingFactor, obs_idx,
                                       trainableParamNumber, tp_size);
                    }

                    trainableParamNumber--;
                    ++tp_it;
                }
                current_param_idx--;
            }

            for (size_t obs_idx = 0; obs_idx < num_observables; obs_idx++) {
                BaseType::applyOperationAdj(*H_lambda[obs_idx], ops, op_idx);
            }
        }
    }
};

} // namespace Pennylane::LightningGPU::Algorithms
