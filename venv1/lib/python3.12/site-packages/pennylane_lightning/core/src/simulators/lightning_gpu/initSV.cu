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
 * @file initSV.cu
 */
#include "cuError.hpp"
#include <cuComplex.h>

#include "cuda_helpers.hpp"
namespace {
using Pennylane::LightningGPU::Util::Cmul;
using Pennylane::LightningGPU::Util::Conj;
} // namespace

namespace Pennylane::LightningGPU {

/**
 * @brief Explicitly set state vector data on GPU device from the input values
 * (on device) and their corresponding indices (on device) information.
 *
 * @param sv Complex data pointer of state vector on device.
 * @param num_indices Number of elements of the value array.
 * @param value Complex data pointer of input values (on device).
 * @param indices Integer data pointer of the indices (on device) of sv elements
 * to be set with corresponding elements in values.
 * @param thread_per_block Number of threads set per block.
 * @param stream_id Stream id of CUDA calls
 */
void setStateVector_CUDA(cuComplex *sv, int &num_indices, cuComplex *value,
                         int *indices, size_t thread_per_block,
                         cudaStream_t stream_id);
void setStateVector_CUDA(cuDoubleComplex *sv, long &num_indices,
                         cuDoubleComplex *value, long *indices,
                         size_t thread_per_block, cudaStream_t stream_id);

/**
 * @brief Explicitly set basis state data on GPU device from the input values
 * (on device) and their corresponding indices (on device) information.
 *
 * @param sv Complex data pointer of state vector on device.
 * @param value Complex data of the input value.
 * @param index Integer data of the sv index to be set with the value.
 * @param async Use an asynchronous memory copy.
 * @param stream_id Stream id of CUDA calls
 */
void setBasisState_CUDA(cuComplex *sv, cuComplex &value, const size_t index,
                        bool async, cudaStream_t stream_id);
void setBasisState_CUDA(cuDoubleComplex *sv, cuDoubleComplex &value,
                        const size_t index, bool async, cudaStream_t stream_id);

/**
 * @brief The CUDA kernel that setS state vector data on GPU device from the
 * input values (on device) and their corresponding indices (on device)
 * information.
 *
 * @param sv Complex data pointer of state vector on device.
 * @param num_indices Number of elements of the value array.
 * @param value Complex data pointer of input values (on device).
 * @param indices Integer data pointer of the sv indices (on device) to be set
 * with corresponding elements in values.
 */
template <class GPUDataT, class index_type>
__global__ void setStateVectorkernel(GPUDataT *sv, index_type num_indices,
                                     GPUDataT *value, index_type *indices) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_indices) {
        sv[indices[i]] = value[i];
    }
}

/**
 * @brief The CUDA kernel call wrapper.
 *
 * @param sv Complex data pointer of state vector on device.
 * @param num_indices Number of elements of the value array.
 * @param value Complex data pointer of input values (on device).
 * @param indices Integer data pointer of the sv indices (on device) to be set
 * by corresponding elements in values.
 * @param thread_per_block Number of threads set per block.
 * @param stream_id Stream id of CUDA calls
 */
template <class GPUDataT, class index_type>
void setStateVector_CUDA_call(GPUDataT *sv, index_type &num_indices,
                              GPUDataT *value, index_type *indices,
                              size_t thread_per_block, cudaStream_t stream_id) {
    auto dv = std::div(num_indices, thread_per_block);
    size_t num_blocks = dv.quot + (dv.rem == 0 ? 0 : 1);
    const size_t block_per_grid = (num_blocks == 0 ? 1 : num_blocks);
    dim3 blockSize(thread_per_block, 1, 1);
    dim3 gridSize(block_per_grid, 1);

    setStateVectorkernel<GPUDataT, index_type>
        <<<gridSize, blockSize, 0, stream_id>>>(sv, num_indices, value,
                                                indices);
    PL_CUDA_IS_SUCCESS(cudaGetLastError());
}

/**
 * @brief The CUDA kernel that multiplies the state vector data on GPU device
 * by a global phase.
 *
 * @param sv Complex data pointer of state vector on device.
 * @param num_sv Number of state vector elements.
 * @param phase Complex data pointer of input values (on device).
 */
template <class GPUDataT, class index_type>
__global__ void globalPhaseStateVectorkernel(GPUDataT *sv, index_type num_sv,
                                             GPUDataT phase) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_sv) {
        sv[i] = Cmul(sv[i], phase);
    }
}

/**
 * @brief The CUDA kernel call wrapper.
 *
 * @param sv Complex data pointer of state vector on device.
 * @param num_sv Number of state vector elements.
 * @param phase Constant complex phase.
 * @param thread_per_block Number of threads set per block.
 * @param stream_id Stream id of CUDA calls
 */
template <class GPUDataT, class index_type>
void globalPhaseStateVector_CUDA_call(GPUDataT *sv, index_type num_sv,
                                      GPUDataT phase, size_t thread_per_block,
                                      cudaStream_t stream_id) {
    auto dv = std::div(static_cast<long>(num_sv), thread_per_block);
    size_t num_blocks = dv.quot + (dv.rem == 0 ? 0 : 1);
    const size_t block_per_grid = (num_blocks == 0 ? 1 : num_blocks);
    dim3 blockSize(thread_per_block, 1, 1);
    dim3 gridSize(block_per_grid, 1);

    globalPhaseStateVectorkernel<GPUDataT, index_type>
        <<<gridSize, blockSize, 0, stream_id>>>(sv, num_sv, phase);
    PL_CUDA_IS_SUCCESS(cudaGetLastError());
}

/**
 * @brief The CUDA kernel that multiplies the state vector data on GPU device
 * by a controlled global phase.
 *
 * @param sv Complex data pointer of state vector on device.
 * @param num_sv Number of state vector elements.
 * @param phase Complex data pointer of controlled global phase values (on
 * device).
 */
template <class GPUDataT, class index_type, bool adjoint = false>
__global__ void cGlobalPhaseStateVectorkernel(GPUDataT *sv, index_type num_sv,
                                              GPUDataT *phase) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_sv) {
        if constexpr (adjoint) {
            sv[i] = Cmul(sv[i], Conj(phase[i]));
        } else {
            sv[i] = Cmul(sv[i], phase[i]);
        }
    }
}

/**
 * @brief The CUDA kernel call wrapper.
 *
 * @param sv Complex data pointer of state vector on device.
 * @param num_sv Number of state vector elements.
 * @param phase Complex data pointer of controlled global phase values (on
 * device).
 * @param thread_per_block Number of threads set per block.
 * @param stream_id Stream id of CUDA calls
 */
template <class GPUDataT, class index_type, bool adjoint = false>
void cGlobalPhaseStateVector_CUDA_call(GPUDataT *sv, index_type num_sv,
                                       GPUDataT *phase, size_t thread_per_block,
                                       cudaStream_t stream_id) {
    auto dv = std::div(static_cast<long>(num_sv), thread_per_block);
    size_t num_blocks = dv.quot + (dv.rem == 0 ? 0 : 1);
    const size_t block_per_grid = (num_blocks == 0 ? 1 : num_blocks);
    dim3 blockSize(thread_per_block, 1, 1);
    dim3 gridSize(block_per_grid, 1);

    cGlobalPhaseStateVectorkernel<GPUDataT, index_type, adjoint>
        <<<gridSize, blockSize, 0, stream_id>>>(sv, num_sv, phase);
    PL_CUDA_IS_SUCCESS(cudaGetLastError());
}

/**
 * @brief CUDA runtime API call wrapper.
 *
 * @param sv Complex data pointer of state vector on device.
 * @param value Complex data of the input value.
 * @param index Integer data of the sv index to be set by value.
 * @param async Use an asynchronous memory copy.
 * @param stream_id Stream id of CUDA calls
 */
template <class GPUDataT>
void setBasisState_CUDA_call(GPUDataT *sv, GPUDataT &value, const size_t index,
                             bool async, cudaStream_t stream_id) {
    if (!async) {
        PL_CUDA_IS_SUCCESS(cudaMemcpy(&sv[index], &value, sizeof(GPUDataT),
                                      cudaMemcpyHostToDevice));
    } else {
        PL_CUDA_IS_SUCCESS(cudaMemcpyAsync(&sv[index], &value, sizeof(GPUDataT),
                                           cudaMemcpyHostToDevice, stream_id));
    }
}

// Definitions
void setStateVector_CUDA(cuComplex *sv, int &num_indices, cuComplex *value,
                         int *indices, size_t thread_per_block,
                         cudaStream_t stream_id) {
    setStateVector_CUDA_call(sv, num_indices, value, indices, thread_per_block,
                             stream_id);
}
void setStateVector_CUDA(cuDoubleComplex *sv, long &num_indices,
                         cuDoubleComplex *value, long *indices,
                         size_t thread_per_block, cudaStream_t stream_id) {
    setStateVector_CUDA_call(sv, num_indices, value, indices, thread_per_block,
                             stream_id);
}

void setBasisState_CUDA(cuComplex *sv, cuComplex &value, const size_t index,
                        bool async, cudaStream_t stream_id) {
    setBasisState_CUDA_call(sv, value, index, async, stream_id);
}
void setBasisState_CUDA(cuDoubleComplex *sv, cuDoubleComplex &value,
                        const size_t index, bool async,
                        cudaStream_t stream_id) {
    setBasisState_CUDA_call(sv, value, index, async, stream_id);
}

void globalPhaseStateVector_CUDA(cuComplex *sv, size_t num_sv, cuComplex phase,
                                 size_t thread_per_block,
                                 cudaStream_t stream_id) {
    globalPhaseStateVector_CUDA_call(sv, num_sv, phase, thread_per_block,
                                     stream_id);
}
void globalPhaseStateVector_CUDA(cuDoubleComplex *sv, size_t num_sv,
                                 cuDoubleComplex phase, size_t thread_per_block,
                                 cudaStream_t stream_id) {
    globalPhaseStateVector_CUDA_call(sv, num_sv, phase, thread_per_block,
                                     stream_id);
}

void cGlobalPhaseStateVector_CUDA(cuComplex *sv, size_t num_sv, bool adjoint,
                                  cuComplex *phase, size_t thread_per_block,
                                  cudaStream_t stream_id) {
    if (adjoint) {
        cGlobalPhaseStateVector_CUDA_call<cuComplex, size_t, true>(
            sv, num_sv, phase, thread_per_block, stream_id);
    } else {
        cGlobalPhaseStateVector_CUDA_call<cuComplex, size_t, false>(
            sv, num_sv, phase, thread_per_block, stream_id);
    }
}
void cGlobalPhaseStateVector_CUDA(cuDoubleComplex *sv, size_t num_sv,
                                  bool adjoint, cuDoubleComplex *phase,
                                  size_t thread_per_block,
                                  cudaStream_t stream_id) {
    if (adjoint) {
        cGlobalPhaseStateVector_CUDA_call<cuDoubleComplex, size_t, true>(
            sv, num_sv, phase, thread_per_block, stream_id);
    } else {
        cGlobalPhaseStateVector_CUDA_call<cuDoubleComplex, size_t, false>(
            sv, num_sv, phase, thread_per_block, stream_id);
    }
}

} // namespace Pennylane::LightningGPU