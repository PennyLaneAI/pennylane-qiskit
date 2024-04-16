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
 * @file StateVectorCudaMPI.hpp
 */
#pragma once

#include <functional>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuComplex.h> // cuDoubleComplex
#include <cuda.h>
#include <custatevec.h> // custatevecApplyMatrix

#include "CSRMatrix.hpp"
#include "Constant.hpp"
#include "Error.hpp"
#include "MPIManager.hpp"
#include "MPIWorker.hpp"
#include "MPI_helpers.hpp"
#include "StateVectorCudaBase.hpp"
#include "cuGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

#include "CPUMemoryModel.hpp"

#include "LinearAlg.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningGPU::MPI;
} // namespace
/// @endcond

namespace Pennylane::LightningGPU {

// declarations of external functions (defined in initSV.cu).
extern void setStateVector_CUDA(cuComplex *sv, int &num_indices,
                                cuComplex *value, int *indices,
                                size_t thread_per_block,
                                cudaStream_t stream_id);
extern void setStateVector_CUDA(cuDoubleComplex *sv, long &num_indices,
                                cuDoubleComplex *value, long *indices,
                                size_t thread_per_block,
                                cudaStream_t stream_id);

extern void setBasisState_CUDA(cuComplex *sv, cuComplex &value,
                               const size_t index, bool async,
                               cudaStream_t stream_id);
extern void setBasisState_CUDA(cuDoubleComplex *sv, cuDoubleComplex &value,
                               const size_t index, bool async,
                               cudaStream_t stream_id);

/**
 * @brief Managed memory CUDA state-vector class using custateVec backed
 * gate-calls.
 *
 * @tparam Precision Floating-point precision type.
 */
template <class Precision = double>
class StateVectorCudaMPI final
    : public StateVectorCudaBase<Precision, StateVectorCudaMPI<Precision>> {
  private:
    using BaseType = StateVectorCudaBase<Precision, StateVectorCudaMPI>;

    size_t numGlobalQubits_;
    size_t numLocalQubits_;
    MPIManager mpi_manager_;

    SharedCusvHandle handle_;
    SharedCublasCaller cublascaller_;
    mutable SharedCusparseHandle
        cusparsehandle_; // This member is mutable to allow lazy initialization.
    SharedLocalStream localStream_;
    SharedMPIWorker svSegSwapWorker_;
    GateCache<Precision> gate_cache_;

  public:
    using CFP_t =
        typename StateVectorCudaBase<Precision,
                                     StateVectorCudaMPI<Precision>>::CFP_t;
    using PrecisionT = Precision;
    using ComplexT = std::complex<PrecisionT>;
    using MemoryStorageT = Pennylane::Util::MemoryStorageLocation::Undefined;

    StateVectorCudaMPI() = delete;

    StateVectorCudaMPI(MPIManager mpi_manager, const DevTag<int> &dev_tag,
                       size_t mpi_buf_size, size_t num_global_qubits,
                       size_t num_local_qubits)
        : StateVectorCudaBase<Precision, StateVectorCudaMPI<Precision>>(
              num_local_qubits, dev_tag, true),
          numGlobalQubits_(num_global_qubits),
          numLocalQubits_(num_local_qubits), mpi_manager_(mpi_manager),
          handle_(make_shared_cusv_handle()),
          cublascaller_(make_shared_cublas_caller()),
          localStream_(make_shared_local_stream()),
          svSegSwapWorker_(make_shared_mpi_worker<CFP_t>(
              handle_.get(), mpi_manager_, mpi_buf_size, BaseType::getData(),
              num_local_qubits, localStream_.get())),
          gate_cache_(true, dev_tag) {
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();
    };

    StateVectorCudaMPI(MPI_Comm mpi_communicator, const DevTag<int> &dev_tag,
                       size_t mpi_buf_size, size_t num_global_qubits,
                       size_t num_local_qubits)
        : StateVectorCudaBase<Precision, StateVectorCudaMPI<Precision>>(
              num_local_qubits, dev_tag, true),
          numGlobalQubits_(num_global_qubits),
          numLocalQubits_(num_local_qubits), mpi_manager_(mpi_communicator),
          handle_(make_shared_cusv_handle()),
          cublascaller_(make_shared_cublas_caller()),
          localStream_(make_shared_local_stream()),
          svSegSwapWorker_(make_shared_mpi_worker<CFP_t>(
              handle_.get(), mpi_manager_, mpi_buf_size, BaseType::getData(),
              num_local_qubits, localStream_.get())),
          gate_cache_(true, dev_tag) {
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();
    };

    StateVectorCudaMPI(const DevTag<int> &dev_tag, size_t mpi_buf_size,
                       size_t num_global_qubits, size_t num_local_qubits)
        : StateVectorCudaBase<Precision, StateVectorCudaMPI<Precision>>(
              num_local_qubits, dev_tag, true),
          numGlobalQubits_(num_global_qubits),
          numLocalQubits_(num_local_qubits), mpi_manager_(MPI_COMM_WORLD),
          handle_(make_shared_cusv_handle()),
          cublascaller_(make_shared_cublas_caller()),
          localStream_(make_shared_local_stream()),
          svSegSwapWorker_(make_shared_mpi_worker<CFP_t>(
              handle_.get(), mpi_manager_, mpi_buf_size, BaseType::getData(),
              num_local_qubits, localStream_.get())),
          gate_cache_(true, dev_tag) {
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();
    };

    StateVectorCudaMPI(const DevTag<int> &dev_tag, size_t num_global_qubits,
                       size_t num_local_qubits, const CFP_t *gpu_data)
        : StateVectorCudaBase<Precision, StateVectorCudaMPI<Precision>>(
              num_local_qubits, dev_tag, true),
          numGlobalQubits_(num_global_qubits),
          numLocalQubits_(num_local_qubits), mpi_manager_(MPI_COMM_WORLD),
          handle_(make_shared_cusv_handle()),
          cublascaller_(make_shared_cublas_caller()),
          localStream_(make_shared_local_stream()),
          svSegSwapWorker_(make_shared_mpi_worker<CFP_t>(
              handle_.get(), mpi_manager_, 0, BaseType::getData(),
              num_local_qubits, localStream_.get())),
          gate_cache_(true, dev_tag) {
        size_t length = 1 << numLocalQubits_;
        BaseType::CopyGpuDataToGpuIn(gpu_data, length, false);
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize())
        mpi_manager_.Barrier();
    }

    StateVectorCudaMPI(const DevTag<int> &dev_tag, size_t num_global_qubits,
                       size_t num_local_qubits)
        : StateVectorCudaBase<Precision, StateVectorCudaMPI<Precision>>(
              num_local_qubits, dev_tag, true),
          numGlobalQubits_(num_global_qubits),
          numLocalQubits_(num_local_qubits), mpi_manager_(MPI_COMM_WORLD),
          handle_(make_shared_cusv_handle()),
          cublascaller_(make_shared_cublas_caller()),
          localStream_(make_shared_local_stream()),
          svSegSwapWorker_(make_shared_mpi_worker<CFP_t>(
              handle_.get(), mpi_manager_, 0, BaseType::getData(),
              num_local_qubits, localStream_.get())),
          gate_cache_(true, dev_tag) {
        BaseType::initSV();
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();
    }

    StateVectorCudaMPI(const StateVectorCudaMPI &other)
        : StateVectorCudaBase<Precision, StateVectorCudaMPI<Precision>>(
              other.getNumLocalQubits(), other.getDataBuffer().getDevTag(),
              true),
          numGlobalQubits_(other.getNumGlobalQubits()),
          numLocalQubits_(other.getNumLocalQubits()),
          mpi_manager_(other.getMPIManager()),
          handle_(make_shared_cusv_handle()),
          cublascaller_(make_shared_cublas_caller()),
          localStream_(make_shared_local_stream()),
          svSegSwapWorker_(make_shared_mpi_worker<CFP_t>(
              handle_.get(), mpi_manager_, 0, BaseType::getData(),
              numLocalQubits_, localStream_.get())),
          gate_cache_(true, other.getDataBuffer().getDevTag()) {
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();
        BaseType::CopyGpuDataToGpuIn(other.getData(), other.getLength(), false);
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();
    }

    ~StateVectorCudaMPI() final = default;

    /**
     * @brief Get MPI manager
     */
    auto getMPIManager() const { return mpi_manager_; }

    /**
     * @brief Get the total number of wires.
     */
    auto getTotalNumQubits() const -> size_t {
        return numGlobalQubits_ + numLocalQubits_;
    }

    /**
     * @brief Get the number of wires distributed across devices.
     */
    auto getNumGlobalQubits() const -> size_t { return numGlobalQubits_; }

    /**
     * @brief Get the number of wires within the local devices.
     */
    auto getNumLocalQubits() const -> size_t { return numLocalQubits_; }

    /**
     * @brief Get pointer to custatevecSVSwapWorkerDescriptor.
     */
    auto getSwapWorker() -> custatevecSVSwapWorkerDescriptor_t {
        return svSegSwapWorker_.get();
    }

    /**
     * @brief Set value for a single element of the state-vector on device. This
     * method is implemented by cudaMemcpy.
     *
     * @param value Value to be set for the target element.
     * @param index Index of the target element.
     * @param async Use an asynchronous memory copy.
     */
    void setBasisState(const std::complex<Precision> &value, const size_t index,
                       const bool async = false) {
        size_t rankId = index >> BaseType::getNumQubits();

        size_t local_index =
            static_cast<size_t>(rankId *
                                std::pow(2.0, static_cast<long double>(
                                                  BaseType::getNumQubits()))) ^
            index;
        BaseType::getDataBuffer().zeroInit();

        CFP_t value_cu = cuUtil::complexToCu<std::complex<Precision>>(value);
        auto stream_id = localStream_.get();

        if (mpi_manager_.getRank() == rankId) {
            setBasisState_CUDA(BaseType::getData(), value_cu, local_index,
                               async, stream_id);
        }
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();
    }

    /**
     * @brief Set values for a batch of elements of the state-vector. This
     * method is implemented by the customized CUDA kernel defined in the
     * DataBuffer class.
     *
     * @param num_indices Number of elements to be passed to the state vector.
     * @param values Pointer to values to be set for the target elements.
     * @param indices Pointer to indices of the target elements.
     * @param async Use an asynchronous memory copy.
     */
    template <class index_type, size_t thread_per_block = 256>
    void setStateVector(const index_type num_indices,
                        const std::complex<Precision> *values,
                        const index_type *indices, const bool async = false) {
        BaseType::getDataBuffer().zeroInit();

        std::vector<index_type> indices_local;
        std::vector<std::complex<Precision>> values_local;

        for (size_t i = 0; i < static_cast<size_t>(num_indices); i++) {
            int index = indices[i];
            PL_ASSERT(index >= 0);
            size_t rankId =
                static_cast<size_t>(index) >> BaseType::getNumQubits();

            if (rankId == mpi_manager_.getRank()) {
                int local_index =
                    static_cast<size_t>(
                        rankId * std::pow(2.0, static_cast<long double>(
                                                   BaseType::getNumQubits()))) ^
                    index;
                indices_local.push_back(local_index);
                values_local.push_back(values[i]);
            }
        }

        auto device_id = BaseType::getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = BaseType::getDataBuffer().getDevTag().getStreamID();

        index_type num_elements = indices_local.size();

        DataBuffer<index_type, int> d_indices{
            static_cast<std::size_t>(num_elements), device_id, stream_id, true};

        DataBuffer<CFP_t, int> d_values{static_cast<std::size_t>(num_elements),
                                        device_id, stream_id, true};

        d_indices.CopyHostDataToGpu(indices_local.data(), d_indices.getLength(),
                                    async);
        d_values.CopyHostDataToGpu(values_local.data(), d_values.getLength(),
                                   async);

        setStateVector_CUDA(BaseType::getData(), num_elements,
                            d_values.getData(), d_indices.getData(),
                            thread_per_block, stream_id);
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();
    }

    /**
     * @brief Apply a single gate to the state-vector. Offloads to custatevec
     * specific API calls if available. If unable, attempts to use prior cached
     * gate values on the device. Lastly, accepts a host-provided matrix if
     * otherwise, and caches on the device for later reuse.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param adjoint Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param matrix Matrix representation of gate.
     */
    void applyOperation(const std::string &opName,
                        const std::vector<size_t> &wires, bool adjoint,
                        const std::vector<Precision> &params,
                        [[maybe_unused]] const std::vector<ComplexT> &matrix) {
        std::vector<CFP_t> matrix_cu(matrix.size());
        std::transform(matrix.begin(), matrix.end(), matrix_cu.begin(),
                       [](const std::complex<Precision> &x) {
                           return cuUtil::complexToCu<std::complex<Precision>>(
                               x);
                       });
        applyOperation(opName, wires, adjoint, params, matrix_cu);
    }

    /**
     * @brief Apply a single gate to the state-vector. Offloads to custatevec
     * specific API calls if available. If unable, attempts to use prior cached
     * gate values on the device. Lastly, accepts a host-provided matrix if
     * otherwise, and caches on the device for later reuse.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param adjoint Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Matrix representation of gate.
     */
    void applyOperation(
        const std::string &opName, const std::vector<size_t> &wires,
        bool adjoint = false, const std::vector<Precision> &params = {0.0},
        [[maybe_unused]] const std::vector<CFP_t> &gate_matrix = {}) {
        const auto ctrl_offset = (BaseType::getCtrlMap().find(opName) !=
                                  BaseType::getCtrlMap().end())
                                     ? BaseType::getCtrlMap().at(opName)
                                     : 0;
        const std::vector<std::size_t> ctrls{wires.begin(),
                                             wires.begin() + ctrl_offset};
        const std::vector<std::size_t> tgts{wires.begin() + ctrl_offset,
                                            wires.end()};

        if (opName == "Identity") {
            return;
        } else if (native_gates_.find(opName) != native_gates_.end()) {
            applyParametricPauliGate({opName}, ctrls, tgts, params.front(),
                                     adjoint);
        } else if (opName == "Rot" || opName == "CRot") {
            if (adjoint) {
                auto rot_matrix =
                    cuGates::getRot<CFP_t>(params[2], params[1], params[0]);
                applyDeviceMatrixGate(rot_matrix.data(), ctrls, tgts, true);
            } else {
                auto rot_matrix =
                    cuGates::getRot<CFP_t>(params[0], params[1], params[2]);
                applyDeviceMatrixGate(rot_matrix.data(), ctrls, tgts, false);
            }
        } else if (par_gates_.find(opName) != par_gates_.end()) {
            par_gates_.at(opName)(wires, adjoint, params);
        } else { // No offloadable function call; defer to matrix passing
            auto &&par =
                (params.empty()) ? std::vector<Precision>{0.0} : params;
            // ensure wire indexing correctly preserved for tensor-observables
            const std::vector<std::size_t> ctrls_local{ctrls.rbegin(),
                                                       ctrls.rend()};
            const std::vector<std::size_t> tgts_local{tgts.rbegin(),
                                                      tgts.rend()};

            if (!gate_cache_.gateExists(opName, par[0]) &&
                gate_matrix.empty()) {
                std::string message = "Currently unsupported gate: " + opName;
                throw LightningException(message);
            } else if (!gate_cache_.gateExists(opName, par[0])) {
                gate_cache_.add_gate(opName, par[0], gate_matrix);
            }
            applyDeviceMatrixGate(
                gate_cache_.get_gate_device_ptr(opName, par[0]), ctrls_local,
                tgts_local, adjoint);
        }
    }

    /**
     * @brief Apply a single gate to the state vector.
     *
     * @param opName Name of gate to apply.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional std gate matrix if opName doesn't exist.
     */
    template <template <typename...> class complex_t>
    void
    applyOperation(const std::string &opName,
                   const std::vector<std::size_t> &controlled_wires,
                   const std::vector<bool> &controlled_values,
                   const std::vector<std::size_t> &wires, bool inverse = false,
                   const std::vector<Precision> &params = {0.0},
                   const std::vector<complex_t<Precision>> &gate_matrix = {}) {
        PL_ABORT_IF_NOT(controlled_wires.empty(),
                        "Controlled kernels not implemented.");
        PL_ABORT_IF_NOT(controlled_wires.size() == controlled_values.size(),
                        "`controlled_wires` must have the same size as "
                        "`controlled_values`.");
        applyOperation(opName, wires, inverse, params, gate_matrix);
    }

    /**
     * @brief Apply a single generator to the state vector using the given
     * kernel.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param adjoint Indicates whether to use adjoint of gate.
     */
    auto applyGenerator(const std::string &opName,
                        const std::vector<size_t> &wires, bool adjoint = false)
        -> PrecisionT {
        auto it = generator_map_.find(opName);
        PL_ABORT_IF(it == generator_map_.end(), "Unsupported generator!");
        return (it->second)(wires, adjoint);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a
     * raw matrix pointer vector.
     *
     * @param matrix Pointer to the array data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param adjoint Indicate whether inverse should be taken.
     */
    void applyMatrix(const std::complex<PrecisionT> *gate_matrix,
                     const std::vector<size_t> &wires, bool adjoint = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        const std::string opName = {};
        size_t n = size_t{1} << wires.size();
        const std::vector<std::complex<PrecisionT>> matrix(gate_matrix,
                                                           gate_matrix + n * n);
        std::vector<CFP_t> matrix_cu(matrix.size());
        std::transform(matrix.begin(), matrix.end(), matrix_cu.begin(),
                       [](const std::complex<Precision> &x) {
                           return cuUtil::complexToCu<std::complex<Precision>>(
                               x);
                       });
        applyOperation(opName, wires, adjoint, {}, matrix_cu);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a
     * std vector.
     *
     * @param matrix Pointer to the array data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param adjoint Indicate whether inverse should be taken.
     */
    void applyMatrix(const std::vector<std::complex<PrecisionT>> &gate_matrix,
                     const std::vector<size_t> &wires, bool adjoint = false) {
        PL_ABORT_IF(gate_matrix.size() !=
                        Pennylane::Util::exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");
        applyMatrix(gate_matrix.data(), wires, adjoint);
    }

    //****************************************************************************//
    // Explicit gate calls for bindings
    //****************************************************************************//
    /* one-qubit gates */
    inline void applyIdentity(const std::vector<std::size_t> &wires,
                              bool adjoint) {
        static_cast<void>(wires);
        static_cast<void>(adjoint);
    }
    inline void applyPauliX(const std::vector<std::size_t> &wires,
                            bool adjoint) {
        static const std::string name{"PauliX"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyPauliY(const std::vector<std::size_t> &wires,
                            bool adjoint) {
        static const std::string name{"PauliY"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyPauliZ(const std::vector<std::size_t> &wires,
                            bool adjoint) {
        static const std::string name{"PauliZ"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyHadamard(const std::vector<std::size_t> &wires,
                              bool adjoint) {
        static const std::string name{"Hadamard"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyS(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"S"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyT(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"T"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyRX(const std::vector<std::size_t> &wires, bool adjoint,
                        Precision param) {
        static const std::vector<std::string> name{{"RX"}};
        applyParametricPauliGate(name, {wires.begin(), wires.end() - 1},
                                 {wires.back()}, param, adjoint);
    }
    inline void applyRY(const std::vector<std::size_t> &wires, bool adjoint,
                        Precision param) {
        static const std::vector<std::string> name{{"RY"}};
        applyParametricPauliGate(name, {wires.begin(), wires.end() - 1},
                                 {wires.back()}, param, adjoint);
    }
    inline void applyRZ(const std::vector<std::size_t> &wires, bool adjoint,
                        Precision param) {
        static const std::vector<std::string> name{{"RZ"}};
        applyParametricPauliGate(name, {wires.begin(), wires.end() - 1},
                                 {wires.back()}, param, adjoint);
    }
    inline void applyRot(const std::vector<std::size_t> &wires, bool adjoint,
                         Precision param0, Precision param1, Precision param2) {
        const std::string opName = "Rot";
        const std::vector<Precision> params = {param0, param1, param2};
        applyOperation(opName, wires, adjoint, params);
    }
    inline void applyRot(const std::vector<std::size_t> &wires, bool adjoint,
                         const std::vector<Precision> &params) {
        applyRot(wires, adjoint, params[0], params[1], params[2]);
    }
    inline void applyPhaseShift(const std::vector<std::size_t> &wires,
                                bool adjoint, Precision param) {
        static const std::string name{"PhaseShift"};
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key,
                                 cuGates::getPhaseShift<CFP_t>(param));
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }

    /* two-qubit gates */
    inline void applyCNOT(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"CNOT"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyCY(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"CY"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyCZ(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"CZ"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applySWAP(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"SWAP"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param), {},
                              wires, adjoint);
    }
    inline void applyIsingXX(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        static const std::vector<std::string> names(wires.size(), {"RX"});
        applyParametricPauliGate(names, {}, wires, param, adjoint);
    }
    inline void applyIsingYY(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        static const std::vector<std::string> names(wires.size(), {"RY"});
        applyParametricPauliGate(names, {}, wires, param, adjoint);
    }
    inline void applyIsingZZ(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        static const std::vector<std::string> names(wires.size(), {"RZ"});
        applyParametricPauliGate(names, {}, wires, param, adjoint);
    }
    inline void applyIsingXY(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        static const std::string name{"IsingXY"};
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key, cuGates::getIsingXY<CFP_t>(param));
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }
    inline void applyCRot(const std::vector<std::size_t> &wires, bool adjoint,
                          const std::vector<Precision> &params) {
        applyCRot(wires, adjoint, params[0], params[1], params[2]);
    }
    inline void applyCRot(const std::vector<std::size_t> &wires, bool adjoint,
                          Precision param0, Precision param1,
                          Precision param2) {
        const std::string opName = "CRot";
        const std::vector<Precision> params = {param0, param1, param2};
        applyOperation(opName, wires, adjoint, params);
    }

    inline void applyCRX(const std::vector<std::size_t> &wires, bool adjoint,
                         Precision param) {
        applyRX(wires, adjoint, param);
    }
    inline void applyCRY(const std::vector<std::size_t> &wires, bool adjoint,
                         Precision param) {
        applyRY(wires, adjoint, param);
    }
    inline void applyCRZ(const std::vector<std::size_t> &wires, bool adjoint,
                         Precision param) {
        applyRZ(wires, adjoint, param);
    }
    inline void applyControlledPhaseShift(const std::vector<std::size_t> &wires,
                                          bool adjoint, Precision param) {
        applyPhaseShift(wires, adjoint, param);
    }
    inline void applySingleExcitation(const std::vector<std::size_t> &wires,
                                      bool adjoint, Precision param) {
        static const std::string name{"SingleExcitation"};
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key,
                                 cuGates::getSingleExcitation<CFP_t>(param));
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }
    inline void
    applySingleExcitationMinus(const std::vector<std::size_t> &wires,
                               bool adjoint, Precision param) {
        static const std::string name{"SingleExcitationMinus"};
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getSingleExcitationMinus<CFP_t>(param));
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }
    inline void applySingleExcitationPlus(const std::vector<std::size_t> &wires,
                                          bool adjoint, Precision param) {
        static const std::string name{"SingleExcitationPlus"};
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getSingleExcitationPlus<CFP_t>(param));
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }

    /* three-qubit gates */
    inline void applyToffoli(const std::vector<std::size_t> &wires,
                             bool adjoint) {
        static const std::string name{"Toffoli"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyCSWAP(const std::vector<std::size_t> &wires,
                           bool adjoint) {
        static const std::string name{"SWAP"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.front()}, {wires.begin() + 1, wires.end()},
                              adjoint);
    }

    /* four-qubit gates */
    inline void applyDoubleExcitation(const std::vector<std::size_t> &wires,
                                      bool adjoint, Precision param) {
        auto &&mat = cuGates::getDoubleExcitation<CFP_t>(param);
        applyDeviceMatrixGate(mat.data(), {}, wires, adjoint);
    }
    inline void
    applyDoubleExcitationMinus(const std::vector<std::size_t> &wires,
                               bool adjoint, Precision param) {
        auto &&mat = cuGates::getDoubleExcitationMinus<CFP_t>(param);
        applyDeviceMatrixGate(mat.data(), {}, wires, adjoint);
    }
    inline void applyDoubleExcitationPlus(const std::vector<std::size_t> &wires,
                                          bool adjoint, Precision param) {
        auto &&mat = cuGates::getDoubleExcitationPlus<CFP_t>(param);
        applyDeviceMatrixGate(mat.data(), {}, wires, adjoint);
    }

    /* Multi-qubit gates */
    inline void applyMultiRZ(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        const std::vector<std::string> names(wires.size(), {"RZ"});
        applyParametricPauliGate(names, {}, wires, param, adjoint);
    }

    /* Gate generators */
    /**
     * @brief Gradient generator function associated with the RX gate.
     *
     * @param sv Statevector
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorRX(const std::vector<size_t> &wires,
                                       bool adj = false) {
        applyPauliX(wires, adj);
        return -static_cast<PrecisionT>(0.5);
    }

    /**
     * @brief Gradient generator function associated with the RY gate.
     *
     * @param sv Statevector
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorRY(const std::vector<size_t> &wires,
                                       bool adj = false) {
        applyPauliY(wires, adj);
        return -static_cast<PrecisionT>(0.5);
    }

    /**
     * @brief Gradient generator function associated with the RZ gate.
     *
     * @param sv Statevector
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorRZ(const std::vector<size_t> &wires,
                                       bool adj = false) {
        applyPauliZ(wires, adj);
        return -static_cast<PrecisionT>(0.5);
    }

    inline PrecisionT
    applyGeneratorIsingXX(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"GeneratorIsingXX"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key,
                                 cuGates::getGeneratorIsingXX<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }
    inline PrecisionT
    applyGeneratorIsingYY(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"GeneratorIsingYY"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key,
                                 cuGates::getGeneratorIsingYY<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }
    inline PrecisionT
    applyGeneratorIsingZZ(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"GeneratorIsingZZ"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key,
                                 cuGates::getGeneratorIsingZZ<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }

    inline PrecisionT
    applyGeneratorIsingXY(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"GeneratorIsingXY"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key,
                                 cuGates::getGeneratorIsingXY<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return static_cast<PrecisionT>(0.5);
    }

    /**
     * @brief Gradient generator function associated with the PhaseShift gate.
     *
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorPhaseShift(const std::vector<size_t> &wires,
                                               bool adj = false) {
        applyOperation("P_11", wires, adj, {0.0}, cuGates::getP11_CU<CFP_t>());
        return static_cast<PrecisionT>(1.0);
    }

    /**
     * @brief Gradient generator function associated with the controlled RX
     * gate.
     *
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorCRX(const std::vector<size_t> &wires,
                                        bool adj = false) {
        applyOperation("P_11", {wires.front()}, adj, {0.0},
                       cuGates::getP11_CU<CFP_t>());
        applyPauliX(std::vector<size_t>{wires.back()}, adj);
        return -static_cast<PrecisionT>(0.5);
    }

    /**
     * @brief Gradient generator function associated with the controlled RY
     * gate.
     *
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorCRY(const std::vector<size_t> &wires,
                                        bool adj = false) {
        applyOperation("P_11", {wires.front()}, adj, {0.0},
                       cuGates::getP11_CU<CFP_t>());
        applyPauliY(std::vector<size_t>{wires.back()}, adj);
        return -static_cast<PrecisionT>(0.5);
    }

    /**
     * @brief Gradient generator function associated with the controlled RZ
     * gate.
     *
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorCRZ(const std::vector<size_t> &wires,
                                        bool adj = false) {
        applyOperation("P_11", {wires.front()}, adj, {0.0},
                       cuGates::getP11_CU<CFP_t>());
        applyPauliZ(std::vector<size_t>{wires.back()}, adj);
        return -static_cast<PrecisionT>(0.5);
    }

    /**
     * @brief Gradient generator function associated with the controlled
     * PhaseShift gate.
     *
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT
    applyGeneratorControlledPhaseShift(const std::vector<size_t> &wires,
                                       bool adj = false) {
        applyOperation("P_1111", {wires}, adj, {0.0},
                       cuGates::getP1111_CU<CFP_t>());
        return static_cast<PrecisionT>(1.0);
    }

    inline PrecisionT
    applyGeneratorSingleExcitation(const std::vector<std::size_t> &wires,
                                   bool adjoint) {
        static const std::string name{"GeneratorSingleExcitation"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorSingleExcitation<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }
    inline PrecisionT
    applyGeneratorSingleExcitationMinus(const std::vector<std::size_t> &wires,
                                        bool adjoint) {
        static const std::string name{"GeneratorSingleExcitationMinus"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorSingleExcitationMinus<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }
    inline PrecisionT
    applyGeneratorSingleExcitationPlus(const std::vector<std::size_t> &wires,
                                       bool adjoint) {
        static const std::string name{"GeneratorSingleExcitationPlus"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorSingleExcitationPlus<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }

    inline PrecisionT
    applyGeneratorDoubleExcitation(const std::vector<std::size_t> &wires,
                                   bool adjoint) {
        static const std::string name{"GeneratorDoubleExcitation"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorDoubleExcitation<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }
    inline PrecisionT
    applyGeneratorDoubleExcitationMinus(const std::vector<std::size_t> &wires,
                                        bool adjoint) {
        static const std::string name{"GeneratorDoubleExcitationMinus"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorDoubleExcitationMinus<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }
    inline PrecisionT
    applyGeneratorDoubleExcitationPlus(const std::vector<std::size_t> &wires,
                                       bool adjoint) {
        static const std::string name{"GeneratorDoubleExcitationPlus"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorDoubleExcitationPlus<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }

    inline PrecisionT
    applyGeneratorMultiRZ(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"PauliZ"};
        static const Precision param = 0.0;
        for (const auto &w : wires) {
            applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                                  {}, {w}, adjoint);
        }
        return -static_cast<PrecisionT>(0.5);
    }

    /**
     * @brief Access the CublasCaller the object is using.
     *
     * @return a reference to the object's CublasCaller object.
     */
    auto getCublasCaller() const -> const CublasCaller & {
        return *cublascaller_;
    }

    /**
     * @brief Get the cuSPARSE handle that the object is using.
     *
     * @return cusparseHandle_t returns the cuSPARSE handle.
     */
    auto getCusparseHandle() const -> cusparseHandle_t {
        if (!cusparsehandle_)
            cusparsehandle_ = make_shared_cusparse_handle();
        return cusparsehandle_.get();
    }

    /**
     * @brief Get the cuStateVec handle that the object is using.
     *
     * @return custatevecHandle_t returns the cuStateVec handle.
     */
    auto getCusvHandle() const -> custatevecHandle_t { return handle_.get(); }

    /**
     * @brief Get a host data copy.
     *
     * @return std::vector<std::complex<PrecisionT>>
     */
    auto getDataVector() -> std::vector<std::complex<PrecisionT>> {
        std::vector<std::complex<PrecisionT>> data_host(BaseType::getLength());
        this->CopyGpuDataToHost(data_host.data(), data_host.size());
        return data_host;
    }

    /**
     * @brief Utility method for expectation value calculations.
     *
     * @param obsName String label for observable. If already exists, will used
     * cached device value. If not, `gate_matrix` is expected, and will
     * automatically cache for future reuse.
     * @param wires Target wires for expectation value.
     * @param params Parameters for a parametric gate.
     * @param gate_matrix Optional matrix for observable. Caches for future use
     * if does not exist.
     * @return auto Expectation value.
     */
    auto expval(const std::string &obsName, const std::vector<size_t> &wires,
                const std::vector<Precision> &params = {0.0},
                const std::vector<CFP_t> &gate_matrix = {}) {
        auto &&par = (params.empty()) ? std::vector<Precision>{0.0} : params;
        auto &&local_wires = wires;

        if (!gate_cache_.gateExists(obsName, par[0]) && gate_matrix.empty()) {
            std::string message =
                "Currently unsupported observable: " + obsName;
            throw LightningException(message.c_str());
        }
        auto expect_val = getExpectationValueDeviceMatrix(
            gate_cache_.get_gate_device_ptr(obsName, par[0]), local_wires);
        return expect_val;
    }

    /**
     * @brief See `expval(std::vector<CFP_t> &gate_matrix = {})`
     */
    auto expval(const std::vector<size_t> &wires,
                const std::vector<std::complex<Precision>> &gate_matrix) {
        std::vector<CFP_t> matrix_cu(gate_matrix.size());

        for (std::size_t i = 0; i < gate_matrix.size(); i++) {
            matrix_cu[i] =
                cuUtil::complexToCu<std::complex<Precision>>(gate_matrix[i]);
        }

        if (gate_matrix.empty()) {
            std::string message = "Currently unsupported observable";
            throw LightningException(message.c_str());
        }

        // Wire order reversed to match expected custatevec wire ordering for
        // tensor observables.
        auto &&local_wires = std::vector<size_t>{wires.rbegin(), wires.rend()};
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        auto expect_val =
            getExpectationValueDeviceMatrix(matrix_cu.data(), local_wires).x;
        return expect_val;
    }

    /**
     * @brief Get expectation value for a sum of Pauli words. This function
     * accepts a vector of words, where each word contains a set of Pauli
     * operations along with their corresponding wires and coefficients of the
     * Hamiltonian. The expval calculation is performed based on the word and
     * its corresponding target wires. The distribution of target wires can be
     * categorized into three different types:1. All target wires are local, and
     * no MPI operation is required. 2. Some target wires are located at the
     * global wires, and there are sufficient local wires available for bit swap
     * operations. 3. Some target wires are located at the global wires, and
     * there are insufficient local wires available for bit swap operations. For
     * the first scenario, the `expvalOnPauliBasis` method can be called
     * directly to obtain the `expval`. In the second scenario, bit swap
     * operations are necessary before and after the `expvalOnPauliBasis`
     * operation to get the `expval`. In the third scenario, a temporary state
     * vector is created to calculate the `bra`, and then the inner product is
     * used to calculate the `expval`. This function here will check the
     * corresponding target wires of all word first. If all target wires for
     * each word are local, the `expvalOnPauliBasis` will be called just once.
     * Otherwise, each word will be looped and corresponding expval calculation
     * methods will be adopted.
     *
     * @param pauli_words Vector of Pauli-words to evaluate expectation value.
     * @param tgts Coupled qubit index to apply each Pauli term.
     * @param coeffs Numpy array buffer of size |pauli_words|
     * @return auto Expectation value.
     */
    auto getExpectationValuePauliWords(
        const std::vector<std::string> &pauli_words,
        const std::vector<std::vector<std::size_t>> &tgts,
        const std::complex<Precision> *coeffs) {
        std::vector<double> expect_local(pauli_words.size());

        std::vector<std::size_t> tgtsSwapStatus;
        std::vector<std::vector<int2>> tgtswirePairs;
        // Local target wires (required by expvalOnPauliBasis)
        std::vector<std::vector<size_t>> localTgts;
        localTgts.reserve(tgts.size());

        tgtsVecProcess(this->getNumLocalQubits(), this->getTotalNumQubits(),
                       tgts, localTgts, tgtsSwapStatus, tgtswirePairs);

        // Check if all target wires are local
        auto threshold = WiresSwapStatus::Swappable;
        bool isAllTargetsLocal = std::all_of(
            tgtsSwapStatus.begin(), tgtsSwapStatus.end(),
            [&threshold](size_t status) { return status < threshold; });

        mpi_manager_.Barrier();

        if (isAllTargetsLocal) {
            expvalOnPauliBasis(pauli_words, localTgts, expect_local);
        } else {
            size_t wirePairsIdx = 0;
            for (size_t i = 0; i < pauli_words.size(); i++) {
                if (tgtsSwapStatus[i] == WiresSwapStatus::UnSwappable) {
                    auto opsNames = pauliStringToOpNames(pauli_words[i]);
                    StateVectorCudaMPI<Precision> tmp(
                        this->getDataBuffer().getDevTag(),
                        this->getNumGlobalQubits(), this->getNumLocalQubits(),
                        this->getData());

                    for (size_t opsIdx = 0; opsIdx < tgts[i].size(); opsIdx++) {
                        std::vector<size_t> wires = {tgts[i][opsIdx]};
                        tmp.applyOperation({opsNames[opsIdx]},
                                           {tgts[i][opsIdx]}, {false});
                    }

                    expect_local[i] =
                        innerProdC_CUDA(
                            tmp.getData(), BaseType::getData(),
                            BaseType::getLength(),
                            BaseType::getDataBuffer().getDevTag().getDeviceID(),
                            BaseType::getDataBuffer().getDevTag().getStreamID(),
                            this->getCublasCaller())
                            .x;
                } else {
                    std::vector<std::string> pauli_words_idx(
                        1, std::string(pauli_words[i]));
                    std::vector<std::vector<size_t>> tgts_idx;
                    tgts_idx.push_back(localTgts[i]);
                    std::vector<double> expval_local(1);

                    if (tgtsSwapStatus[i] == WiresSwapStatus::Local) {
                        expvalOnPauliBasis(pauli_words_idx, tgts_idx,
                                           expval_local);
                    } else {
                        applyMPI_Dispatcher(
                            tgtswirePairs[wirePairsIdx],
                            &StateVectorCudaMPI::expvalOnPauliBasis,
                            pauli_words_idx, tgts_idx, expval_local);
                        wirePairsIdx++;
                    }
                    expect_local[i] = expval_local[0];
                }
            }
        }

        auto expect = mpi_manager_.allreduce<double>(expect_local, "sum");
        std::complex<PrecisionT> result{0, 0};

        for (std::size_t idx = 0; idx < expect.size(); idx++) {
            result += static_cast<PrecisionT>(expect[idx]) * coeffs[idx];
        }
        return std::real(result);
    }

  private:
    using ParFunc = std::function<void(const std::vector<size_t> &, bool,
                                       const std::vector<Precision> &)>;
    using GeneratorFunc =
        std::function<Precision(const std::vector<size_t> &, bool)>;

    using FMap = std::unordered_map<std::string, ParFunc>;
    using GMap = std::unordered_map<std::string, GeneratorFunc>;

    const FMap par_gates_{
        // LCOV_EXCL_START
        {"RX",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyRX(std::forward<decltype(wires)>(wires),
                     std::forward<decltype(adjoint)>(adjoint),
                     std::forward<decltype(params[0])>(params[0]));
         }},
        {"RY",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyRY(std::forward<decltype(wires)>(wires),
                     std::forward<decltype(adjoint)>(adjoint),
                     std::forward<decltype(params[0])>(params[0]));
         }},
        {"RZ",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyRZ(std::forward<decltype(wires)>(wires),
                     std::forward<decltype(adjoint)>(adjoint),
                     std::forward<decltype(params[0])>(params[0]));
         }},
        // LCOV_EXCL_STOP
        {"PhaseShift",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyPhaseShift(std::forward<decltype(wires)>(wires),
                             std::forward<decltype(adjoint)>(adjoint),
                             std::forward<decltype(params[0])>(params[0]));
         }},
        {"MultiRZ",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyMultiRZ(std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingXX",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyIsingXX(std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingYY",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyIsingYY(std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingZZ",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyIsingZZ(std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingXY",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyIsingXY(std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params[0])>(params[0]));
         }},
        // LCOV_EXCL_START
        // Calculation passed to applyParametricPauliGate
        {"CRX",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyCRX(std::forward<decltype(wires)>(wires),
                      std::forward<decltype(adjoint)>(adjoint),
                      std::forward<decltype(params[0])>(params[0]));
         }},
        {"CRY",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyCRY(std::forward<decltype(wires)>(wires),
                      std::forward<decltype(adjoint)>(adjoint),
                      std::forward<decltype(params[0])>(params[0]));
         }},
        {"CRZ",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyCRZ(std::forward<decltype(wires)>(wires),
                      std::forward<decltype(adjoint)>(adjoint),
                      std::forward<decltype(params[0])>(params[0]));
         }},
        // LCOV_EXCL_STOP
        {"SingleExcitation",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applySingleExcitation(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"SingleExcitationPlus",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applySingleExcitationPlus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"SingleExcitationMinus",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applySingleExcitationMinus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitation",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyDoubleExcitation(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitationPlus",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyDoubleExcitationPlus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitationMinus",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyDoubleExcitationMinus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"ControlledPhaseShift",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyControlledPhaseShift(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        // LCOV_EXCL_START
        {"Rot",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyRot(std::forward<decltype(wires)>(wires),
                      std::forward<decltype(adjoint)>(adjoint),
                      std::forward<decltype(params)>(params));
         }},
        {"CRot",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyCRot(std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
         }}
        // LCOV_EXCL_STOP
    };

    const std::unordered_map<std::string, custatevecPauli_t> native_gates_{
        {"RX", CUSTATEVEC_PAULI_X},       {"RY", CUSTATEVEC_PAULI_Y},
        {"RZ", CUSTATEVEC_PAULI_Z},       {"CRX", CUSTATEVEC_PAULI_X},
        {"CRY", CUSTATEVEC_PAULI_Y},      {"CRZ", CUSTATEVEC_PAULI_Z},
        {"Identity", CUSTATEVEC_PAULI_I}, {"I", CUSTATEVEC_PAULI_I}};

    // Holds the mapping from gate labels to associated generator functions.
    const GMap generator_map_{
        {"RX",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorRX(std::forward<decltype(wires)>(wires),
                                     std::forward<decltype(adjoint)>(adjoint));
         }},
        {"RY",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorRY(std::forward<decltype(wires)>(wires),
                                     std::forward<decltype(adjoint)>(adjoint));
         }},
        {"RZ",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorRZ(std::forward<decltype(wires)>(wires),
                                     std::forward<decltype(adjoint)>(adjoint));
         }},
        {"IsingXX",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorIsingXX(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"IsingXY",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorIsingXY(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"IsingYY",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorIsingYY(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"IsingZZ",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorIsingZZ(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"IsingXY",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorIsingXY(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"CRX",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorCRX(std::forward<decltype(wires)>(wires),
                                      std::forward<decltype(adjoint)>(adjoint));
         }},
        {"CRY",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorCRY(std::forward<decltype(wires)>(wires),
                                      std::forward<decltype(adjoint)>(adjoint));
         }},
        {"CRZ",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorCRZ(std::forward<decltype(wires)>(wires),
                                      std::forward<decltype(adjoint)>(adjoint));
         }},
        {"PhaseShift",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorPhaseShift(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"ControlledPhaseShift",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorControlledPhaseShift(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"SingleExcitation",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorSingleExcitation(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"SingleExcitationMinus",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorSingleExcitationMinus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"SingleExcitationPlus",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorSingleExcitationPlus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"DoubleExcitation",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorDoubleExcitation(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"DoubleExcitationMinus",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorDoubleExcitationMinus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"DoubleExcitationPlus",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorDoubleExcitationPlus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},

        {"MultiRZ", [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorMultiRZ(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }}};

    /**
     * @brief Normalize the index ordering to match PennyLane.
     *
     * @tparam IndexType Integer value type.
     * @param indices Given indices to transform.
     */
    template <typename IndexType>
    inline auto NormalizeIndices(std::vector<IndexType> indices)
        -> std::vector<IndexType> {
        std::vector<IndexType> t_indices(std::move(indices));
        std::transform(t_indices.begin(), t_indices.end(), t_indices.begin(),
                       [&](IndexType i) -> IndexType {
                           return BaseType::getNumQubits() - 1 - i;
                       });
        return t_indices;
    }

    /**
     * @brief Get expectation value for a sum of Pauli words.
     *
     * @param pauli_words Vector of Pauli-words to evaluate expectation value.
     * @param tgts Coupled qubit index to apply each Pauli term.
     * @param coeffs Numpy array buffer of size |pauli_words|
     * @return auto Expectation value.
     */
    auto expvalOnPauliBasis(const std::vector<std::string> &pauli_words,
                            const std::vector<std::vector<std::size_t>> &tgts,
                            std::vector<double> &local_expect) {
        uint32_t nIndexBits = static_cast<uint32_t>(this->getNumLocalQubits());
        cudaDataType_t data_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
        } else {
            data_type = CUDA_C_32F;
        }

        // Note: due to API design, cuStateVec assumes this is always a double.
        // Push NVIDIA to move this to behind API for future releases, and
        // support 32/64 bits.

        std::vector<std::vector<custatevecPauli_t>> pauliOps;

        std::vector<custatevecPauli_t *> pauliOps_ptr;

        for (auto &p_word : pauli_words) {
            pauliOps.push_back(cuUtil::pauliStringToEnum(p_word));
            pauliOps_ptr.push_back((*pauliOps.rbegin()).data());
        }

        std::vector<std::vector<int32_t>> basisBits;
        std::vector<int32_t *> basisBits_ptr;
        std::vector<uint32_t> n_basisBits;

        for (auto &wires : tgts) {
            std::vector<int32_t> wiresInt(wires.size());
            std::transform(wires.begin(), wires.end(), wiresInt.begin(),
                           [&](std::size_t x) { return static_cast<int>(x); });
            basisBits.push_back(wiresInt);
            basisBits_ptr.push_back((*basisBits.rbegin()).data());
            n_basisBits.push_back(wiresInt.size());
        }

        // compute expectation
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectationsOnPauliBasis(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* double* */ local_expect.data(),
            /* const custatevecPauli_t ** */
            const_cast<const custatevecPauli_t **>(pauliOps_ptr.data()),
            /* const uint32_t */ static_cast<uint32_t>(pauliOps.size()),
            /* const int32_t ** */
            const_cast<const int32_t **>(basisBits_ptr.data()),
            /* const uint32_t */ n_basisBits.data()));
    }

    /**
     * @brief Apply parametric Pauli gates to local statevector using custateVec
     * calls.
     *
     * @param pauli_words List of Pauli words representing operation.
     * @param ctrls Control wires
     * @param tgts target wires.
     * @param param Gate parameter.
     * @param use_adjoint Take adjoint of operation.
     */
    void applyCuSVPauliGate(const std::vector<std::string> &pauli_words,
                            std::vector<int> &ctrls, std::vector<int> &tgts,
                            Precision param, bool use_adjoint = false) {
        int nIndexBits = BaseType::getNumQubits();

        cudaDataType_t data_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
        } else {
            data_type = CUDA_C_32F;
        }

        std::vector<custatevecPauli_t> pauli_enums;
        pauli_enums.reserve(pauli_words.size());
        for (const auto &pauli_str : pauli_words) {
            pauli_enums.push_back(native_gates_.at(pauli_str));
        }
        const auto local_angle = (use_adjoint) ? param / 2 : -param / 2;

        PL_CUSTATEVEC_IS_SUCCESS(custatevecApplyPauliRotation(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* double */ local_angle,
            /* const custatevecPauli_t* */ pauli_enums.data(),
            /* const int32_t* */ tgts.data(),
            /* const uint32_t */ tgts.size(),
            /* const int32_t* */ ctrls.data(),
            /* const int32_t* */ nullptr,
            /* const uint32_t */ ctrls.size()));
    }

    /**
     * @brief Apply parametric Pauli gates using custateVec calls.
     *
     * @param pauli_words List of Pauli words representing operation.
     * @param ctrls Control wires
     * @param tgts target wires.
     * @param param Parametric gate parameter.
     * @param use_adjoint Take adjoint of operation.
     */
    void applyParametricPauliGate(const std::vector<std::string> &pauli_words,
                                  std::vector<std::size_t> ctrls,
                                  std::vector<std::size_t> tgts,
                                  Precision param, bool use_adjoint = false) {
        std::vector<int> ctrlsInt(ctrls.size());
        std::vector<int> tgtsInt(tgts.size());

        // Transform indices between PL & cuQuantum ordering
        std::transform(
            ctrls.begin(), ctrls.end(), ctrlsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(this->getTotalNumQubits() - 1 - x);
            });
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(this->getTotalNumQubits() - 1 - x);
            });

        // Initialize a vector to store the status of wires and default its
        // elements as zeros, which assumes there is no target and control wire.
        std::vector<int> statusWires(this->getTotalNumQubits(),
                                     WireStatus::Default);

        // Update wire status based on the gate information
        for (size_t i = 0; i < ctrlsInt.size(); i++) {
            statusWires[ctrlsInt[i]] = WireStatus::Control;
        }
        // Update wire status based on the gate information
        for (size_t i = 0; i < tgtsInt.size(); i++) {
            statusWires[tgtsInt[i]] = WireStatus::Target;
        }

        int StatusGlobalWires = std::reduce(
            statusWires.begin() + this->getNumLocalQubits(), statusWires.end());

        mpi_manager_.Barrier();

        if (!StatusGlobalWires) {
            applyCuSVPauliGate(pauli_words, ctrlsInt, tgtsInt, param,
                               use_adjoint);
        } else {
            size_t counts_global_wires =
                std::count_if(statusWires.begin(),
                              statusWires.begin() + this->getNumLocalQubits(),
                              [](int i) { return i != WireStatus::Default; });
            size_t counts_local_wires =
                ctrlsInt.size() + tgtsInt.size() - counts_global_wires;
            PL_ABORT_IF(
                counts_global_wires >
                    (this->getNumLocalQubits() - counts_local_wires),
                "There is not enough local wires for bit swap operation.");

            std::vector<int> localCtrls(ctrlsInt);
            std::vector<int> localTgts(tgtsInt);

            auto wirePairs = createWirePairs(
                this->getNumLocalQubits(), this->getTotalNumQubits(),
                localCtrls, localTgts, statusWires);

            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());

            applyMPI_Dispatcher(
                wirePairs, &StateVectorCudaMPI::applyCuSVPauliGate, pauli_words,
                localCtrls, localTgts, param, use_adjoint);
            PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(localStream_.get()));
            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        }
        // Ensure sync for all local target wires scenarios
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();
    }

    /**
     * @brief Apply a given host or device-stored array representing the gate
     * `matrix` to the local statevector at qubit indices given by `tgts` and
     * control-lines given by `ctrls`. The adjoint can be taken by setting
     * `use_adjoint` to true.
     *
     * @param matrix Host- or device data array in row-major order representing
     * a given gate.
     * @param ctrls Control line qubits.
     * @param tgts Target qubits.
     * @param use_adjoint Use adjoint of given gate.
     */
    void applyCuSVDeviceMatrixGate(const CFP_t *matrix,
                                   const std::vector<int> &ctrls,
                                   const std::vector<int> &tgts,
                                   bool use_adjoint = false) {
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;
        int nIndexBits = BaseType::getNumQubits();

        cudaDataType_t data_type;
        custatevecComputeType_t compute_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
            compute_type = CUSTATEVEC_COMPUTE_64F;
        } else {
            data_type = CUDA_C_32F;
            compute_type = CUSTATEVEC_COMPUTE_32F;
        }

        // check the size of external workspace
        PL_CUSTATEVEC_IS_SUCCESS(custatevecApplyMatrixGetWorkspaceSize(
            /* custatevecHandle_t */ handle_.get(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix,
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t */ use_adjoint,
            /* const uint32_t */ tgts.size(),
            /* const uint32_t */ ctrls.size(),
            /* custatevecComputeType_t */ compute_type,
            /* size_t* */ &extraWorkspaceSizeInBytes));

        // allocate external workspace if necessary
        // LCOV_EXCL_START
        if (extraWorkspaceSizeInBytes > 0) {
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
        }
        // LCOV_EXCL_STOP

        // apply gate
        PL_CUSTATEVEC_IS_SUCCESS(custatevecApplyMatrix(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix,
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t */ use_adjoint,
            /* const int32_t* */ tgts.data(),
            /* const uint32_t */ tgts.size(),
            /* const int32_t* */ ctrls.data(),
            /* const int32_t* */ nullptr,
            /* const uint32_t */ ctrls.size(),
            /* custatevecComputeType_t */ compute_type,
            /* void* */ extraWorkspace,
            /* size_t */ extraWorkspaceSizeInBytes));
        // LCOV_EXCL_START
        if (extraWorkspaceSizeInBytes)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
        // LCOV_EXCL_STOP
    }

    /**
     * @brief Apply a given host or device-stored array representing the gate
     * `matrix` to the statevector at qubit indices given by `tgts` and
     * control-lines given by `ctrls`. The adjoint can be taken by setting
     * `use_adjoint` to true.
     *
     * @param matrix Host- or device data array in row-major order representing
     * a given gate.
     * @param ctrls Control line qubits.
     * @param tgts Target qubits.
     * @param use_adjoint Use adjoint of given gate.
     */
    void applyDeviceMatrixGate(const CFP_t *matrix,
                               const std::vector<std::size_t> &ctrls,
                               const std::vector<std::size_t> &tgts,
                               bool use_adjoint = false) {
        std::vector<int> ctrlsInt(ctrls.size());
        std::vector<int> tgtsInt(tgts.size());

        std::transform(
            ctrls.begin(), ctrls.end(), ctrlsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(this->getTotalNumQubits() - 1 - x);
            });
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(this->getTotalNumQubits() - 1 - x);
            });

        // Initialize a vector to store the status of wires and default its
        // elements as zeros, which assumes there is no target and control wire.
        std::vector<int> statusWires(this->getTotalNumQubits(),
                                     WireStatus::Default);

        // Update wire status based on the gate information
        for (size_t i = 0; i < ctrlsInt.size(); i++) {
            statusWires[ctrlsInt[i]] = WireStatus::Control;
        }
        // Update wire status based on the gate information
        for (size_t i = 0; i < tgtsInt.size(); i++) {
            statusWires[tgtsInt[i]] = WireStatus::Target;
        }

        int StatusGlobalWires = std::reduce(
            statusWires.begin() + this->getNumLocalQubits(), statusWires.end());

        mpi_manager_.Barrier();

        if (!StatusGlobalWires) {
            applyCuSVDeviceMatrixGate(matrix, ctrlsInt, tgtsInt, use_adjoint);
        } else {
            size_t counts_global_wires =
                std::count_if(statusWires.begin(),
                              statusWires.begin() + this->getNumLocalQubits(),
                              [](int i) { return i != WireStatus::Default; });
            size_t counts_local_wires =
                ctrlsInt.size() + tgtsInt.size() - counts_global_wires;
            PL_ABORT_IF(
                counts_global_wires >
                    (this->getNumLocalQubits() - counts_local_wires),
                "There is not enough local wires for bit swap operation.");

            std::vector<int> localCtrls = ctrlsInt;
            std::vector<int> localTgts = tgtsInt;

            auto wirePairs = createWirePairs(
                this->getNumLocalQubits(), this->getTotalNumQubits(),
                localCtrls, localTgts, statusWires);

            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());

            applyMPI_Dispatcher(wirePairs,
                                &StateVectorCudaMPI::applyCuSVDeviceMatrixGate,
                                matrix, localCtrls, localTgts, use_adjoint);
            PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(localStream_.get()));
            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        }
        // Ensure sync for all local target wires scenarios
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();
    }

    /**
     * @brief Get expectation of a given host or device defined array.
     *
     * @param matrix Host or device defined row-major order gate matrix array.
     * @param tgts Target qubits.
     * @param expect Local expectation value.
     * @return auto Expectation value.
     */
    void getCuSVExpectationValueDeviceMatrix(const CFP_t *matrix,
                                             const std::vector<int> &tgts,
                                             CFP_t &expect) {
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        size_t nIndexBits = BaseType::getNumQubits();
        cudaDataType_t data_type;
        custatevecComputeType_t compute_type;
        cudaDataType_t expectationDataType = CUDA_C_64F;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
            compute_type = CUSTATEVEC_COMPUTE_64F;
        } else {
            data_type = CUDA_C_32F;
            compute_type = CUSTATEVEC_COMPUTE_32F;
        }

        // check the size of external workspace
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectationGetWorkspaceSize(
            /* custatevecHandle_t */ handle_.get(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix,
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const uint32_t */ tgts.size(),
            /* custatevecComputeType_t */ compute_type,
            /* size_t* */ &extraWorkspaceSizeInBytes));
        // LCOV_EXCL_START
        if (extraWorkspaceSizeInBytes > 0) {
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
        }
        // LCOV_EXCL_STOP

        cuDoubleComplex expect_;
        // compute expectation
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectation(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* void* */ &expect_,
            /* cudaDataType_t */ expectationDataType,
            /* double* */ nullptr,
            /* const void* */ matrix,
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t* */ tgts.data(),
            /* const uint32_t */ tgts.size(),
            /* custatevecComputeType_t */ compute_type,
            /* void* */ extraWorkspace,
            /* size_t */ extraWorkspaceSizeInBytes));
        // LCOV_EXCL_START
        if (extraWorkspaceSizeInBytes) {
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
        }
        // LCOV_EXCL_STOP
        expect.x = static_cast<PrecisionT>(expect_.x);
        expect.y = static_cast<PrecisionT>(expect_.y);
    }

    /**
     * @brief Get expectation value of a given host or device defined array.
     *
     * @param matrix Host or device defined row-major order gate matrix array.
     * @param tgts Target qubits.
     * @return auto Expectation value.
     */
    auto getExpectationValueDeviceMatrix(const CFP_t *matrix,
                                         const std::vector<std::size_t> &tgts) {
        std::vector<int> tgtsInt(tgts.size());
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(this->getTotalNumQubits() - 1 - x);
            });

        // Initialize a vector to store the status of wires and default its
        // elements as zeros, which assumes there is no target and control wire.
        std::vector<int> statusWires(this->getTotalNumQubits(),
                                     WireStatus::Default);

        // Update wire status based on the gate information
        for (size_t i = 0; i < tgtsInt.size(); i++) {
            statusWires[tgtsInt[i]] = WireStatus::Target;
        }

        int StatusGlobalWires = std::reduce(
            statusWires.begin() + this->getNumLocalQubits(), statusWires.end());

        mpi_manager_.Barrier();

        CFP_t local_expect;
        if (!StatusGlobalWires) {
            getCuSVExpectationValueDeviceMatrix(matrix, tgtsInt, local_expect);
        } else {
            std::vector<int> localTgts = tgtsInt;

            auto wirePairs = createWirePairs(this->getNumLocalQubits(),
                                             this->getTotalNumQubits(),
                                             localTgts, statusWires);

            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());

            applyMPI_Dispatcher(
                wirePairs,
                &StateVectorCudaMPI::getCuSVExpectationValueDeviceMatrix,
                matrix, localTgts, local_expect);
            PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(localStream_.get()));
            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        }
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        auto expect = mpi_manager_.allreduce<CFP_t>(local_expect, "sum");
        return expect;
    }

    /**
     * @brief MPI dispatcher for the target and control gates at global qubits.
     *
     * @tparam F Return type of the callable.
     * @tparam Args Types of arguments of t the callable.
     *
     * @param wirePairs Vector of wire pairs for bit index swap operations.
     * @param functor The callable.
     * @param args Arguments of the callable.
     */
    template <typename F, typename... Args>
    void applyMPI_Dispatcher(std::vector<int2> &wirePairs, F &&functor,
                             Args &&...args) {
        int maskBitString[] = {}; // specify the values of mask qubits
        int maskOrdering[] = {};  // specify the mask qubits

        cudaDataType_t svDataType;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            svDataType = CUDA_C_64F;
        } else {
            svDataType = CUDA_C_32F;
        }
        //
        // create distributed index bit swap scheduler
        //
        custatevecDistIndexBitSwapSchedulerDescriptor_t scheduler;
        PL_CUSTATEVEC_IS_SUCCESS(custatevecDistIndexBitSwapSchedulerCreate(
            /* custatevecHandle_t */ handle_.get(),
            /* custatevecDistIndexBitSwapSchedulerDescriptor_t */
            &scheduler,
            /* uint32_t */ this->getNumGlobalQubits(),
            /* uint32_t */ this->getNumLocalQubits()));

        // set the index bit swaps to the scheduler
        // nSwapBatches is obtained by the call.  This value specifies the
        // number of loops
        unsigned nSwapBatches = 0;
        PL_CUSTATEVEC_IS_SUCCESS(
            custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps(
                /* custatevecHandle_t */ handle_.get(),
                /* custatevecDistIndexBitSwapSchedulerDescriptor_t */
                scheduler,
                /* const int2* */ wirePairs.data(),
                /* const uint32_t */
                static_cast<unsigned>(wirePairs.size()),
                /* const int32_t* */ maskBitString,
                /* const int32_t* */ maskOrdering,
                /* const uint32_t */ 0,
                /* uint32_t* */ &nSwapBatches));

        //
        // the main loop of index bit swaps
        //
        constexpr size_t nLoops = 2;
        for (size_t loop = 0; loop < nLoops; loop++) {
            for (int swapBatchIndex = 0;
                 swapBatchIndex < static_cast<int>(nSwapBatches);
                 swapBatchIndex++) {
                // get parameters
                custatevecSVSwapParameters_t parameters;
                PL_CUSTATEVEC_IS_SUCCESS(
                    custatevecDistIndexBitSwapSchedulerGetParameters(
                        /* custatevecHandle_t */ handle_.get(),
                        /* custatevecDistIndexBitSwapSchedulerDescriptor_t*/
                        scheduler,
                        /* const int32_t */ swapBatchIndex,
                        /* const int32_t */ mpi_manager_.getRank(),
                        /* custatevecSVSwapParameters_t* */
                        &parameters));

                // the rank of the communication endpoint is
                // parameters.dstSubSVIndex as "rank == subSVIndex" is assumed
                // in the present sample.
                int rank = parameters.dstSubSVIndex;
                // set parameters to the worker
                PL_CUSTATEVEC_IS_SUCCESS(custatevecSVSwapWorkerSetParameters(
                    /* custatevecHandle_t */ handle_.get(),
                    /* custatevecSVSwapWorkerDescriptor_t */
                    this->getSwapWorker(),
                    /* const custatevecSVSwapParameters_t* */
                    &parameters,
                    /* int */ rank));
                PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(localStream_.get()));
                PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize())
                // execute swap
                PL_CUSTATEVEC_IS_SUCCESS(custatevecSVSwapWorkerExecute(
                    /* custatevecHandle_t */ handle_.get(),
                    /* custatevecSVSwapWorkerDescriptor_t */
                    this->getSwapWorker(),
                    /* custatevecIndex_t */ 0,
                    /* custatevecIndex_t */ parameters.transferSize));
                // all internal CUDA calls are serialized on localStream
                PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize())
                mpi_manager_.Barrier();
            }
            if (loop == 0) {
                std::invoke(std::forward<F>(functor), this,
                            std::forward<Args>(args)...);
            }
            // synchronize all operations on device
            PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(localStream_.get()));
            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
            mpi_manager_.Barrier();
        }

        PL_CUSTATEVEC_IS_SUCCESS(custatevecDistIndexBitSwapSchedulerDestroy(
            handle_.get(), scheduler));
    }
};

}; // namespace Pennylane::LightningGPU
