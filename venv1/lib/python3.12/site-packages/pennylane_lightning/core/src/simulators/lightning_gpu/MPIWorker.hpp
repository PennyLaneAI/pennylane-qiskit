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
#pragma once
#include <algorithm>
#include <bit>
#include <memory>
#include <string>
#include <vector>

#include "MPIManager.hpp"
#include "MPI_helpers.hpp"
#include "cuError.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <custatevec.h>
/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;

using SharedLocalStream =
    std::shared_ptr<std::remove_pointer<cudaStream_t>::type>;
using SharedMPIWorker = std::shared_ptr<
    std::remove_pointer<custatevecSVSwapWorkerDescriptor_t>::type>;

inline size_t mebibyteToBytes(const size_t mebibytes) {
    return mebibytes * size_t{1024 * 1024};
}

inline double bytesToMebibytes(const size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

} // namespace
/// @endcond

namespace Pennylane::LightningGPU::MPI {
/**
 * @brief Utility function object to tell std::shared_ptr how to
 * release/destroy various custatevecSVSwapWorker related objects.
 */
struct MPIWorkerDeleter {
    void operator()(cudaStream_t localStream) const {
        PL_CUDA_IS_SUCCESS(cudaStreamDestroy(localStream));
    }

    void operator()(custatevecSVSwapWorkerDescriptor_t svSegSwapWorker,
                    custatevecHandle_t handle,
                    custatevecCommunicatorDescriptor_t communicator,
                    void *d_extraWorkspace, void *d_transferWorkspace,
                    std::vector<void *> d_subSVsP2P,
                    std::vector<cudaEvent_t> remoteEvents,
                    cudaEvent_t localEvent) const {
        PL_CUSTATEVEC_IS_SUCCESS(
            custatevecSVSwapWorkerDestroy(handle, svSegSwapWorker));
        PL_CUSTATEVEC_IS_SUCCESS(
            custatevecCommunicatorDestroy(handle, communicator));
        PL_CUDA_IS_SUCCESS(cudaFree(d_extraWorkspace));
        PL_CUDA_IS_SUCCESS(cudaFree(d_transferWorkspace));
        // LCOV_EXCL_START
        for (auto *d_subSV : d_subSVsP2P)
            PL_CUDA_IS_SUCCESS(cudaIpcCloseMemHandle(d_subSV));
        for (auto event : remoteEvents)
            PL_CUDA_IS_SUCCESS(cudaEventDestroy(event));
        // LCOV_EXCL_STOP
        PL_CUDA_IS_SUCCESS(cudaEventDestroy(localEvent));
    }
};

/**
 * @brief Creates a SharedLocalStream (a shared pointer to a cuda stream)
 */
inline SharedLocalStream make_shared_local_stream() {
    cudaStream_t localStream;
    PL_CUDA_IS_SUCCESS(cudaStreamCreate(&localStream));
    return {localStream, MPIWorkerDeleter()};
}

/**
 * @brief Creates a SharedMPIWorker (a shared pointer to a
 * custatevecSVSwapWorker)
 *
 * @param handle custatevecHandle.
 * @param mpi_manager MPI manager object.
 * @param mpi_buf_size Size to set MPI buffer in MiB (mebibytes).
 * @param sv Pointer to the data requires MPI operation.
 * @param numLocalQubits Number of local qubits.
 * @param localStream Local cuda stream.
 */
template <typename CFP_t>
SharedMPIWorker
make_shared_mpi_worker(custatevecHandle_t handle, MPIManager &mpi_manager,
                       const size_t mpi_buf_size, CFP_t *sv,
                       const size_t numLocalQubits, cudaStream_t localStream) {
    custatevecSVSwapWorkerDescriptor_t svSegSwapWorker = nullptr;

    int nDevices_int = 0;
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&nDevices_int));

    size_t nDevices = static_cast<size_t>(nDevices_int);

    // Ensure the number of P2P devices is calculated based on the number of MPI
    // processes within the node
    nDevices = mpi_manager.getSizeNode() < nDevices ? mpi_manager.getSizeNode()
                                                    : nDevices;

    size_t nP2PDeviceBits = std::bit_width(nDevices) - 1;

    size_t p2pEnabled_local = 1;
    // P2P access check
    if (nP2PDeviceBits != 0) {
        size_t local_device_id = mpi_manager.getRank() % nDevices;

        for (size_t devId = 0; devId < nDevices; ++devId) {
            if (devId != local_device_id) {
                int accessEnabled;
                PL_CUDA_IS_SUCCESS(cudaDeviceCanAccessPeer(
                    &accessEnabled, static_cast<int>(devId),
                    static_cast<int>(local_device_id)));

                if (!accessEnabled) {
                    p2pEnabled_local = 0;
                }
            }
        }

        auto isP2PEnabled =
            mpi_manager.allreduce<size_t>(p2pEnabled_local, "min");
        // P2PDeviceBits is set as 0 for all MPI processes if P2P access is not
        // supported by any pair of devices of any node
        if (!isP2PEnabled) {
            nP2PDeviceBits = 0;
        }
    }

    cudaDataType_t svDataType;
    if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                  std::is_same_v<CFP_t, double2>) {
        svDataType = CUDA_C_64F;
    } else {
        svDataType = CUDA_C_32F;
    }

    cudaEvent_t localEvent = nullptr;
    custatevecCommunicatorDescriptor_t communicator = nullptr;

    PL_CUDA_IS_SUCCESS(cudaEventCreateWithFlags(
        &localEvent, cudaEventInterprocess | cudaEventDisableTiming));

    custatevecCommunicatorType_t communicatorType;
    if (mpi_manager.getVendor() == "MPICH") {
        communicatorType = CUSTATEVEC_COMMUNICATOR_TYPE_MPICH;
    }
    if (mpi_manager.getVendor() == "Open MPI") {
        communicatorType = CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI;
    }

    // LCOV_EXCL_START
    // The following lines are for handling the errors caused by python
    // layer calls. Won't be covered by cpp unit tests.
    auto err = custatevecCommunicatorCreate(handle, &communicator,
                                            communicatorType, nullptr);
    if (err != CUSTATEVEC_STATUS_SUCCESS) {
        communicator = nullptr;
        PL_CUSTATEVEC_IS_SUCCESS(custatevecCommunicatorCreate(
            handle, &communicator, communicatorType, "libmpi.so"));
    }
    // LCOV_EXCL_STOP
    mpi_manager.Barrier();

    void *d_extraWorkspace = nullptr;
    void *d_transferWorkspace = nullptr;

    std::vector<void *> d_subSVsP2P;
    std::vector<int> subSVIndicesP2P;
    std::vector<cudaEvent_t> remoteEvents;

    size_t extraWorkspaceSize = 0;
    size_t minTransferWorkspaceSize = 0;

    PL_CUSTATEVEC_IS_SUCCESS(custatevecSVSwapWorkerCreate(
        /* custatevecHandle_t */ handle,
        /* custatevecSVSwapWorkerDescriptor_t* */ &svSegSwapWorker,
        /* custatevecCommunicatorDescriptor_t */ communicator,
        /* void* */ sv,
        /* int32_t */ mpi_manager.getRank(),
        /* cudaEvent_t */ localEvent,
        /* cudaDataType_t */ svDataType,
        /* cudaStream_t */ localStream,
        /* size_t* */ &extraWorkspaceSize,
        /* size_t* */ &minTransferWorkspaceSize));

    PL_CUDA_IS_SUCCESS(cudaMalloc(&d_extraWorkspace, extraWorkspaceSize));

    PL_CUSTATEVEC_IS_SUCCESS(custatevecSVSwapWorkerSetExtraWorkspace(
        /* custatevecHandle_t */ handle,
        /* custatevecSVSwapWorkerDescriptor_t */ svSegSwapWorker,
        /* void* */ d_extraWorkspace,
        /* size_t */ extraWorkspaceSize));

    size_t transferWorkspaceSize; // In bytes and its value should be power
                                  // of 2.

    if (mpi_buf_size == 0) {
        transferWorkspaceSize = size_t{1} << numLocalQubits;
        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            transferWorkspaceSize = transferWorkspaceSize * sizeof(double) * 2;
        } else {
            transferWorkspaceSize = transferWorkspaceSize * sizeof(float) * 2;
        }
        // With the default setting, transfer work space is limited to 64 MiB
        // based on the benchmark tests on the Perlmutter.
        size_t buffer_limit = 64;
        double transferWorkspaceSizeInMiB =
            bytesToMebibytes(transferWorkspaceSize);
        if (transferWorkspaceSizeInMiB > static_cast<double>(buffer_limit)) {
            transferWorkspaceSize = mebibyteToBytes(buffer_limit);
        }
    } else {
        transferWorkspaceSize = mebibyteToBytes(mpi_buf_size);
    }

    transferWorkspaceSize =
        std::max(minTransferWorkspaceSize, transferWorkspaceSize);
    PL_CUDA_IS_SUCCESS(cudaMalloc(&d_transferWorkspace, transferWorkspaceSize));

    PL_CUSTATEVEC_IS_SUCCESS(custatevecSVSwapWorkerSetTransferWorkspace(
        /* custatevecHandle_t */ handle,
        /* custatevecSVSwapWorkerDescriptor_t */ svSegSwapWorker,
        /* void* */ d_transferWorkspace,
        /* size_t */ transferWorkspaceSize));

    // LCOV_EXCL_START
    // Won't be covered by CI checks with 2 nvidia GPUs without nvlink
    // connection
    if (nP2PDeviceBits != 0) {
        cudaIpcMemHandle_t ipcMemHandle;
        PL_CUDA_IS_SUCCESS(cudaIpcGetMemHandle(&ipcMemHandle, sv));
        std::vector<cudaIpcMemHandle_t> ipcMemHandles(mpi_manager.getSize());

        mpi_manager.Allgather<cudaIpcMemHandle_t>(ipcMemHandle, ipcMemHandles,
                                                  sizeof(ipcMemHandle));
        cudaIpcEventHandle_t eventHandle;
        PL_CUDA_IS_SUCCESS(cudaIpcGetEventHandle(&eventHandle, localEvent));
        // distribute event handles
        std::vector<cudaIpcEventHandle_t> ipcEventHandles(
            mpi_manager.getSize());

        mpi_manager.Allgather<cudaIpcEventHandle_t>(
            eventHandle, ipcEventHandles, sizeof(eventHandle));
        //  get remove device pointers and events
        size_t nSubSVsP2P = size_t{1} << nP2PDeviceBits;
        size_t p2pSubSVIndexBegin =
            (mpi_manager.getRank() / nSubSVsP2P) * nSubSVsP2P;
        size_t p2pSubSVIndexEnd = p2pSubSVIndexBegin + nSubSVsP2P;
        for (size_t p2pSubSVIndex = p2pSubSVIndexBegin;
             p2pSubSVIndex < p2pSubSVIndexEnd; p2pSubSVIndex++) {
            if (static_cast<size_t>(mpi_manager.getRank()) == p2pSubSVIndex)
                continue; // don't need local sub state vector pointer
            void *d_subSVP2P = nullptr;
            const auto &dstMemHandle = ipcMemHandles[p2pSubSVIndex];
            PL_CUDA_IS_SUCCESS(cudaIpcOpenMemHandle(
                &d_subSVP2P, dstMemHandle, cudaIpcMemLazyEnablePeerAccess));
            d_subSVsP2P.push_back(d_subSVP2P);
            cudaEvent_t eventP2P = nullptr;
            PL_CUDA_IS_SUCCESS(cudaIpcOpenEventHandle(
                &eventP2P, ipcEventHandles[p2pSubSVIndex]));
            remoteEvents.push_back(eventP2P);
            subSVIndicesP2P.push_back(p2pSubSVIndex);
        }

        // set p2p sub state vectors
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSVSwapWorkerSetSubSVsP2P(
            /* custatevecHandle_t */ handle,
            /* custatevecSVSwapWorkerDescriptor_t */ svSegSwapWorker,
            /* void** */ d_subSVsP2P.data(),
            /* const int32_t* */ subSVIndicesP2P.data(),
            /* cudaEvent_t */ remoteEvents.data(),
            /* const uint32_t */ static_cast<uint32_t>(d_subSVsP2P.size())));
    }
    // LCOV_EXCL_START

    return {svSegSwapWorker,
            std::bind(MPIWorkerDeleter(), std::placeholders::_1, handle,
                      communicator, d_extraWorkspace, d_transferWorkspace,
                      d_subSVsP2P, remoteEvents, localEvent)};
}
} // namespace Pennylane::LightningGPU::MPI