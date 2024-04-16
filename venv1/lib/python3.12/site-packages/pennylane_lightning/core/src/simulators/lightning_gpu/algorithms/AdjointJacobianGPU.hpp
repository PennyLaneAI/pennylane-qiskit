// Copyright 2022-2023 Xanadu Quantum Technologies Inc. and contributors.

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
 * @file AdjointJacobianGPU.hpp
 */

#pragma once

#include <chrono>
#include <future>
#include <omp.h>
#include <span>
#include <thread>
#include <variant>

#include "AdjointJacobianBase.hpp"
#include "DevTag.hpp"
#include "DevicePool.hpp"
#include "LinearAlg.hpp"
#include "ObservablesGPU.hpp"

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
class AdjointJacobian final
    : public AdjointJacobianBase<StateVectorT, AdjointJacobian<StateVectorT>> {
  private:
    using ComplexT = typename StateVectorT::ComplexT;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));
    using BaseType =
        AdjointJacobianBase<StateVectorT, AdjointJacobian<StateVectorT>>;

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
    inline void
    updateJacobian(const std::vector<StateVectorT> &sv1s,
                   const StateVectorT &sv2, std::span<PrecisionT> &jac,
                   PrecisionT scaling_coeff, size_t num_observables,
                   size_t param_index, size_t tp_size,
                   DataBuffer<CFP_t, int> &device_buffer_jac_single_param,
                   std::vector<CFP_t> &host_buffer_jac_single_param) {
        host_buffer_jac_single_param.clear();
        for (size_t obs_idx = 0; obs_idx < num_observables; obs_idx++) {
            const StateVectorT &sv1 = sv1s[obs_idx];
            PL_ABORT_IF_NOT(sv1.getDataBuffer().getDevTag().getDeviceID() ==
                                sv2.getDataBuffer().getDevTag().getDeviceID(),
                            "Data exists on different GPUs. Aborting.");
            innerProdC_CUDA_device(
                sv1.getData(), sv2.getData(), sv1.getLength(),
                sv1.getDataBuffer().getDevTag().getDeviceID(),
                sv1.getDataBuffer().getDevTag().getStreamID(),
                sv1.getCublasCaller(),
                device_buffer_jac_single_param.getData() + obs_idx);
        }
        host_buffer_jac_single_param.resize(num_observables);
        device_buffer_jac_single_param.CopyGpuDataToHost(
            host_buffer_jac_single_param.data(),
            host_buffer_jac_single_param.size(), false);
        for (size_t obs_idx = 0; obs_idx < num_observables; obs_idx++) {
            size_t idx = param_index + obs_idx * tp_size;
            jac[idx] =
                -2 * scaling_coeff * host_buffer_jac_single_param[obs_idx].y;
        }

        host_buffer_jac_single_param.clear();
    }

  public:
    AdjointJacobian() = default;

    /**
     * @brief Batches the adjoint_jacobian method over the available GPUs.
     * Explicitly forbids OMP_NUM_THREADS>1 to avoid issues with std::thread
     * contention and state access issues.
     *
     * @param jac Preallocated vector for Jacobian data results.
     * @param jd JacobianData represents the QuantumTape to differentiate.
     * @param apply_operations Indicate whether to apply operations to psi prior
     * to calculation.
     */
    void batchAdjointJacobian(std::span<PrecisionT> jac,
                              const JacobianData<StateVectorT> &jd,
                              bool apply_operations = false) {
        // Create a pool of available GPU devices
        DevicePool<int> dp;
        const auto num_gpus = dp.getTotalDevices();
        const auto num_chunks = num_gpus;

        const auto &obs = jd.getObservables();

        const std::vector<size_t> &trainableParams = jd.getTrainableParams();
        const size_t tp_size = trainableParams.size();

        // Create a vector of threads for separate GPU executions
        using namespace std::chrono_literals;
        std::vector<std::thread> threads;
        threads.reserve(num_gpus);

        // Hold results of threaded GPU executions
        std::vector<std::future<std::vector<PrecisionT>>> jac_futures;

        // Iterate over the chunked observables, and submit the Jacobian task
        // for execution
        for (std::size_t i = 0; i < num_chunks; i++) {
            const auto first = static_cast<std::size_t>(
                std::ceil(obs.size() * i / num_chunks));
            const auto last = static_cast<std::size_t>(
                std::ceil((obs.size() * (i + 1) / num_chunks) - 1));

            std::promise<std::vector<PrecisionT>> jac_subset_promise;
            jac_futures.emplace_back(jac_subset_promise.get_future());

            auto adj_lambda =
                [&](std::promise<std::vector<PrecisionT>> j_promise,
                    std::size_t offset_first, std::size_t offset_last) {
                    // Ensure No OpenMP threads spawned;
                    // to be resolved with streams in future releases
                    omp_set_num_threads(1);
                    // Grab a GPU index, and set a device tag
                    const auto id = dp.acquireDevice();
                    DevTag<int> dt_local(id, 0);
                    dt_local.refresh();

                    // Create a sv copy on this thread and device; may not be
                    // necessary, could do in adjoint calculation directly
                    StateVectorT local_sv(jd.getPtrStateVec(),
                                          jd.getSizeStateVec(), dt_local);

                    // Create local store for Jacobian subset
                    std::vector<PrecisionT> jac_local_vector(
                        (offset_last - offset_first + 1) *
                            trainableParams.size(),
                        0);

                    const JacobianData<StateVectorT> jd_local(
                        tp_size, local_sv.getLength(), local_sv.getData(),
                        {obs.begin() + offset_first,
                         obs.begin() + offset_last + 1},
                        jd.getOperations(), jd.getTrainableParams());

                    std::span<PrecisionT> jac_local(jac_local_vector.data(),
                                                    jac_local_vector.size());

                    adjointJacobian(jac_local, jd_local, local_sv,
                                    apply_operations, dt_local);

                    j_promise.set_value(std::move(jac_local_vector));
                    dp.releaseDevice(id);
                };
            threads.emplace_back(adj_lambda, std::move(jac_subset_promise),
                                 first, last);
        }

        /// Ensure the new local jacs are inserted and
        /// overwrite the 0 jacs values before returning
        for (std::size_t i = 0; i < jac_futures.size(); i++) {
            const auto first = static_cast<std::size_t>(
                std::ceil(obs.size() * i / num_chunks));

            auto jac_chunk = jac_futures[i].get();
            for (std::size_t j = 0; j < jac_chunk.size(); j++) {
                std::copy(jac_chunk.begin(), jac_chunk.end(),
                          jac.begin() + first * tp_size);
            }
        }
        for (std::size_t t = 0; t < threads.size(); t++) {
            threads[t].join();
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
     * @param apply_operations Indicate whether to apply operations to psi prior
     * to calculation.
     * @param dev_tag Device tag represents the device id and stream id.
     */
    void adjointJacobian(std::span<PrecisionT> jac,
                         const JacobianData<StateVectorT> &jd,
                         const StateVectorT &ref_data,
                         bool apply_operations = false,
                         DevTag<int> dev_tag = {0, 0}) {
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

        DevTag<int> dt_local(std::move(dev_tag));
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

        // Create observable-applied state-vectors
        std::vector<StateVectorT> H_lambda;
        for (size_t n = 0; n < num_observables; n++) {
            H_lambda.emplace_back(lambda.getNumQubits(), dt_local, true,
                                  cusvhandle, cublascaller, cusparsehandle);
        }
        BaseType::applyObservables(H_lambda, lambda, obs);

        StateVectorT mu(lambda.getNumQubits(), dt_local, true, cusvhandle,
                        cublascaller, cusparsehandle);

        auto device_id = mu.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = mu.getDataBuffer().getDevTag().getStreamID();

        // The following buffers are only used as temporary workspace
        // inside updateJacobian and do not carry any state beyond a
        // single updateJacobian function call.
        // We create the buffers here instead of inside updateJacobian
        // to avoid expensive reallocations.
        DataBuffer<CFP_t, int> device_buffer_jac_single_param{
            static_cast<std::size_t>(num_observables), device_id, stream_id,
            true};
        std::vector<CFP_t> host_buffer_jac_single_param;

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

                    updateJacobian(H_lambda, mu, jac, scalingFactor,
                                   num_observables, trainableParamNumber,
                                   tp_size, device_buffer_jac_single_param,
                                   host_buffer_jac_single_param);

                    trainableParamNumber--;
                    ++tp_it;
                }
                current_param_idx--;
            }
            this->applyOperationsAdj(H_lambda, ops,
                                     static_cast<size_t>(op_idx));
        }
    }
};

} // namespace Pennylane::LightningGPU::Algorithms
