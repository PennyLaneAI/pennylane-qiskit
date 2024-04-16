#pragma once

#include <functional>
#include <memory>
#include <mutex>

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cusparse_v2.h>
#include <custatevec.h>

#include "DataBuffer.hpp"
#include "cuError.hpp"

#include "CSRMatrix.hpp"
#include "LinearAlg.hpp"
#include "MPIManager.hpp"

using namespace Pennylane::LightningGPU::MPI;

namespace Pennylane::LightningGPU::Util {
/**
 * @brief Sparse matrix vector multiply offloaded to cuSparse (Y =
 * alpha*SparseMat*X + beta)
 *
 * @tparam index_type Integer type for offsets, indices and number of elements
 * (size_t for the moment).
 * @tparam Precision Floating data-type.
 * @tparam DevTypeID Integer type of device id.
 *
 * @param mpi_manager MPI operation wrapper.
 * @param length_local Length of X.
 * @param csrOffsets_ptr Pointer to offsets in CSR format.
 * @param csrOffsets_size Number of elements of offsets.
 * @param columns_ptr Pointer to column indices in CSR format.
 * @param values_ptr Pointer to value of each non-zero elements in CSR format.
 * @param numNNZ Number of non-zero elements.
 * @param X Pointer to vector.
 * @param Y Pointer to vector.
 * @param device_id Device id.
 * @param cudaStream_t Stream id.
 * @param handle cuSparse handle.
 */
template <class index_type, class Precision, class CFP_t, class DevTypeID = int>
inline void SparseMV_cuSparseMPI(
    MPIManager &mpi_manager, const size_t &length_local,
    const index_type *csrOffsets_ptr, const int64_t csrOffsets_size,
    const index_type *columns_ptr, const std::complex<Precision> *values_ptr,
    CFP_t *X, CFP_t *Y, DevTypeID device_id, cudaStream_t stream_id,
    cusparseHandle_t handle) {
    std::vector<std::vector<CSRMatrix<Precision, index_type>>> csrmatrix_blocks;
    if (mpi_manager.getRank() == 0) {
        csrmatrix_blocks = splitCSRMatrix<Precision, index_type>(
            mpi_manager, static_cast<size_t>(csrOffsets_size - 1),
            csrOffsets_ptr, columns_ptr, values_ptr);
    }
    mpi_manager.Barrier();

    std::vector<CSRMatrix<Precision, index_type>> localCSRMatVector;
    for (size_t i = 0; i < mpi_manager.getSize(); i++) {
        auto localCSRMat = scatterCSRMatrix<Precision, index_type>(
            mpi_manager, csrmatrix_blocks[i], length_local, 0);
        localCSRMatVector.push_back(localCSRMat);
    }
    mpi_manager.Barrier();

    DataBuffer<CFP_t, int> d_res_per_block{length_local, device_id, stream_id,
                                           true};

    for (size_t i = 0; i < mpi_manager.getSize(); i++) {
        // Need to investigate if non-blocking MPI operation can improve
        // performace here.
        auto &localCSRMatrix = localCSRMatVector[i];
        size_t color = 0;

        if (localCSRMatrix.getValues().size() != 0) {
            d_res_per_block.zeroInit();
            color = 1;
            SparseMV_cuSparse<index_type, Precision, CFP_t>(
                localCSRMatrix.getCsrOffsets().data(),
                static_cast<int64_t>(localCSRMatrix.getCsrOffsets().size()),
                localCSRMatrix.getColumns().data(),
                localCSRMatrix.getValues().data(),
                static_cast<int64_t>(localCSRMatrix.getValues().size()), X,
                d_res_per_block.getData(), device_id, stream_id, handle);
        }

        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager.Barrier();

        if (mpi_manager.getRank() == i) {
            color = 1;
            if (localCSRMatrix.getValues().size() == 0) {
                d_res_per_block.zeroInit();
            }
        }

        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager.Barrier();

        auto new_mpi_manager = mpi_manager.split(color, mpi_manager.getRank());
        int reduce_root_rank = -1;

        if (mpi_manager.getRank() == i) {
            reduce_root_rank = new_mpi_manager.getRank();
        }

        mpi_manager.Bcast<int>(reduce_root_rank, i);

        if (new_mpi_manager.getComm() != MPI_COMM_NULL) {
            new_mpi_manager.Reduce<CFP_t>(d_res_per_block.getData(), Y,
                                          length_local, reduce_root_rank,
                                          "sum");
        }
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager.Barrier();
    }
}

} // namespace Pennylane::LightningGPU::Util