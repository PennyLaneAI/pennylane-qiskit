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

namespace Pennylane::LightningGPU::Util {
/**
 * @brief A wrapper for cuBLAS calls. This should be used for all calls to
 * cuBLAS.
 *
 * In most cases you do not want to create objects of this class directly but
 * use
 * %
 * This classes purpose is to manage the cuBLAS handle and avoid data races
 * between setting the active CUDA device, the current cuBLAS stream and the
 * call of the cuBLAS function via the call method.
 * The class creates and destroys a cuBLAS handle on construction/destruction.
 */
class CublasCaller {
  public:
    /** Creates a new CublasCaller with a new cuBLAS handle */
    CublasCaller() { PL_CUBLAS_IS_SUCCESS(cublasCreate(&handle)); }
    /** Destructs the CublasCaller and destroys its cuBLAS handle */
    ~CublasCaller() { PL_CUBLAS_IS_SUCCESS(cublasDestroy(handle)); }

    CublasCaller(CublasCaller const &) = delete;
    CublasCaller(CublasCaller &&) = delete;
    CublasCaller &operator=(CublasCaller const &) = delete;
    CublasCaller &operator=(CublasCaller &&) = delete;

    /**
     * @brief Call a cuBLAS function.
     *
     * The call function executes a cuBLAS function on a specified CUDA device
     * and stream pair. It ensures thread safety for this operation, i.e.,
     * it prevents other thread from changing the active device and stream
     * of the cuBLAS handle before the passed cuBLAS call could be queued.
     *
     * @param func the cuBLAS function to be called.
     * @param dev_id the CUDA device id on which the function should be
     * executed.
     * @param stream the CUDA stream on which the cuBLAS function should be
     * queued.
     * @param args the arguments for the cuBLAS function.
     */
    template <typename F, typename... Args>
    void call(F &&func, int dev_id, cudaStream_t stream, Args &&...args) const {
        std::lock_guard lk(mtx);
        PL_CUDA_IS_SUCCESS(cudaSetDevice(dev_id));
        PL_CUBLAS_IS_SUCCESS(cublasSetStream(handle, stream));
        PL_CUBLAS_IS_SUCCESS(std::invoke(std::forward<F>(func), handle,
                                         std::forward<Args>(args)...));
    }

  private:
    mutable std::mutex mtx;
    cublasHandle_t handle;
};

/**
 * @brief cuBLAS backed inner product for GPU data.
 *
 * @tparam T Complex data-type. Accepts cuFloatComplex and cuDoubleComplex
 * @param v1 Device data pointer 1
 * @param v2 Device data pointer 2
 * @param data_size Length of device data.
 * @return T Device data pointer to store inner-product result
 */
template <class T = cuDoubleComplex, class DevTypeID = int>
inline auto innerProdC_CUDA_device(const T *v1, const T *v2,
                                   const int data_size, int dev_id,
                                   cudaStream_t stream_id,
                                   const CublasCaller &cublas, T *d_result) {
    if constexpr (std::is_same_v<T, cuFloatComplex>) {
        cublas.call(cublasCdotc, dev_id, stream_id, data_size, v1, 1, v2, 1,
                    d_result);
    } else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
        cublas.call(cublasZdotc, dev_id, stream_id, data_size, v1, 1, v2, 1,
                    d_result);
    }
}

/**
 * @brief cuBLAS backed inner product for GPU data.
 *
 * @tparam T Complex data-type. Accepts cuFloatComplex and cuDoubleComplex
 * @param v1 Device data pointer 1
 * @param v2 Device data pointer 2
 * @param data_size Length of device data.
 * @param dev_id the device on which the function should be executed.
 * @param stream_id the CUDA stream on which the operation should be executed.
 * @param cublas the CublasCaller object that manages the cuBLAS handle.
 * @return T Inner-product result
 */
template <class T = cuDoubleComplex, class DevTypeID = int>
inline auto innerProdC_CUDA(const T *v1, const T *v2, const int data_size,
                            int dev_id, cudaStream_t stream_id,
                            const CublasCaller &cublas) -> T {
    T result{0.0, 0.0}; // Host result

    if constexpr (std::is_same_v<T, cuFloatComplex>) {
        cublas.call(cublasCdotc, dev_id, stream_id, data_size, v1, 1, v2, 1,
                    &result);
    } else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
        cublas.call(cublasZdotc, dev_id, stream_id, data_size, v1, 1, v2, 1,
                    &result);
    }
    return result;
}

/**
 * @brief cuBLAS backed GPU C/ZAXPY.
 *
 * @tparam CFP_t Complex data-type. Accepts std::complex<float> and
 * std::complex<double>
 * @param a scaling factor
 * @param v1 Device data pointer 1 (data to be modified)
 * @param v2 Device data pointer 2 (the result data)
 * @param data_size Length of device data.
 * @param dev_id the device on which the function should be executed.
 * @param stream_id the CUDA stream on which the operation should be executed.
 * @param cublas the CublasCaller object that manages the cuBLAS handle.
 */
template <class CFP_t = std::complex<double>, class T = cuDoubleComplex,
          class DevTypeID = int>
inline auto scaleAndAddC_CUDA(const CFP_t a, const T *v1, T *v2,
                              const int data_size, DevTypeID dev_id,
                              cudaStream_t stream_id,
                              const CublasCaller &cublas) {
    if constexpr (std::is_same_v<T, cuComplex>) {
        const cuComplex alpha{a.real(), a.imag()};
        cublas.call(cublasCaxpy, dev_id, stream_id, data_size, &alpha, v1, 1,
                    v2, 1);
    } else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
        const cuDoubleComplex alpha{a.real(), a.imag()};
        cublas.call(cublasZaxpy, dev_id, stream_id, data_size, &alpha, v1, 1,
                    v2, 1);
    }
}

/**
 * @brief cuBLAS backed GPU data scaling.
 *
 * @tparam CFP_t Complex data-type. Accepts std::complex<float> and
 * std::complex<double>
 * @param a scaling factor
 * @param v1 Device data pointer
 * @param data_size Length of device data.
 * @param dev_id the device on which the function should be executed.
 * @param stream_id the CUDA stream on which the operation should be executed.
 * @param cublas the CublasCaller object that manages the cuBLAS handle.
 */
template <class CFP_t = std::complex<double>, class T = cuDoubleComplex,
          class DevTypeID = int>
inline auto scaleC_CUDA(const CFP_t a, T *v1, const int data_size,
                        DevTypeID dev_id, cudaStream_t stream_id,
                        const CublasCaller &cublas) {
    cudaDataType_t data_type;

    if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                  std::is_same_v<CFP_t, double2>) {
        data_type = CUDA_C_64F;
    } else {
        data_type = CUDA_C_32F;
    }

    cublas.call(cublasScalEx, dev_id, stream_id, data_size,
                reinterpret_cast<const void *>(&a), data_type, v1, data_type, 1,
                data_type);
}

/** @brief `%CudaScopedDevice` uses RAII to select a CUDA device context.
 *
 * @see https://taskflow.github.io/taskflow/classtf_1_1cudaScopedDevice.html
 *
 * @note A `%CudaScopedDevice` instance cannot be moved or copied.
 *
 * @warning This class is not thread-safe.
 */
class CudaScopedDevice {
  public:
    /**
     * @brief Constructs a `%CudaScopedDevice` using a CUDA device.
     *
     *  @param device CUDA device to scope in the guard.
     */
    CudaScopedDevice(int device) {
        PL_CUDA_IS_SUCCESS(cudaGetDevice(&prev_device_));
        if (prev_device_ == device) {
            prev_device_ = -1;
        } else {
            PL_CUDA_IS_SUCCESS(cudaSetDevice(device));
        }
    }

    /**
     * @brief Destructs a `%CudaScopedDevice`, switching back to the
     * previous CUDA device context.
     */
    ~CudaScopedDevice() {
        if (prev_device_ != -1) {
            // Throwing exceptions from a destructor can be dangerous.
            // See https://isocpp.org/wiki/faq/exceptions#ctor-exceptions.
            cudaSetDevice(prev_device_);
        }
    }

    CudaScopedDevice() = delete;
    CudaScopedDevice(const CudaScopedDevice &) = delete;
    CudaScopedDevice(CudaScopedDevice &&) = delete;

  private:
    /// The previous CUDA device (or -1 if the device passed to the
    /// constructor is the current CUDA device).
    int prev_device_;
};

/**
 * Utility function object to tell std::shared_ptr how to
 * release/destroy various CUDA objects.
 */
struct HandleDeleter {
    void operator()(cublasHandle_t handle) const {
        PL_CUBLAS_IS_SUCCESS(cublasDestroy(handle));
    }
    void operator()(custatevecHandle_t handle) const {
        PL_CUSTATEVEC_IS_SUCCESS(custatevecDestroy(handle));
    }
    void operator()(cusparseHandle_t handle) const {
        PL_CUSPARSE_IS_SUCCESS(cusparseDestroy(handle));
    }
};

using SharedCublasCaller = std::shared_ptr<CublasCaller>;
using SharedCusvHandle =
    std::shared_ptr<std::remove_pointer<custatevecHandle_t>::type>;
using SharedCusparseHandle =
    std::shared_ptr<std::remove_pointer<cusparseHandle_t>::type>;

/**
 * @brief Creates a SharedCublasCaller (a shared pointer to a CublasCaller)
 */
inline SharedCublasCaller make_shared_cublas_caller() {
    return std::make_shared<CublasCaller>();
}

/**
 * @brief Creates a SharedCusvHandle (a shared pointer to a custatevecHandle)
 */
inline SharedCusvHandle make_shared_cusv_handle() {
    custatevecHandle_t h;
    PL_CUSTATEVEC_IS_SUCCESS(custatevecCreate(&h));
    return {h, HandleDeleter()};
}

/**
 * @brief Creates a SharedCusparseHandle (a shared pointer to a cusparseHandle)
 */
inline SharedCusparseHandle make_shared_cusparse_handle() {
    cusparseHandle_t h;
    PL_CUSPARSE_IS_SUCCESS(cusparseCreate(&h));
    return {h, HandleDeleter()};
}

/**
 * @brief Sparse matrix vector multiply offloaded to cuSparse (Y =
 * alpha*SparseMat*X + beta)
 *
 * @tparam index_type Integer type for offsets, indices and number of elements
 * (size_t for the moment).
 * @tparam Precision Floating data-type.
 * @tparam DevTypeID Integer type of device id.
 *
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
inline void
SparseMV_cuSparse(const index_type *csrOffsets_ptr,
                  const int64_t csrOffsets_size, const index_type *columns_ptr,
                  const std::complex<Precision> *values_ptr,
                  const int64_t numNNZ, CFP_t *X, CFP_t *Y, DevTypeID device_id,
                  cudaStream_t stream_id, cusparseHandle_t handle) {
    const int64_t num_rows =
        csrOffsets_size -
        1; // int64_t is required for num_rows by cusparseCreateCsr
    const int64_t num_cols =
        num_rows; // int64_t is required for num_cols by cusparseCreateCsr
    const int64_t nnz =
        numNNZ; // int64_t is required for nnz by cusparseCreateCsr

    const CFP_t alpha = {1.0, 0.0};
    const CFP_t beta = {0.0, 0.0};

    DataBuffer<index_type, int> d_csrOffsets{
        static_cast<std::size_t>(csrOffsets_size), device_id, stream_id, true};
    DataBuffer<index_type, int> d_columns{static_cast<std::size_t>(numNNZ),
                                          device_id, stream_id, true};
    DataBuffer<CFP_t, int> d_values{static_cast<std::size_t>(numNNZ), device_id,
                                    stream_id, true};

    d_csrOffsets.CopyHostDataToGpu(csrOffsets_ptr, d_csrOffsets.getLength(),
                                   false);
    d_columns.CopyHostDataToGpu(columns_ptr, d_columns.getLength(), false);
    d_values.CopyHostDataToGpu(values_ptr, d_values.getLength(), false);

    cudaDataType_t data_type;
    cusparseIndexType_t compute_type;

    if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                  std::is_same_v<CFP_t, double2>) {
        data_type = CUDA_C_64F;
        compute_type = CUSPARSE_INDEX_64I;
    } else {
        data_type = CUDA_C_32F;
        compute_type = CUSPARSE_INDEX_32I;
    }

    // CUSPARSE APIs
    cusparseSpMatDescr_t mat;
    cusparseDnVecDescr_t vecX, vecY;

    size_t bufferSize = 0;

    // Create sparse matrix A in CSR format
    PL_CUSPARSE_IS_SUCCESS(cusparseCreateCsr(
        /* cusparseSpMatDescr_t* */ &mat,
        /* int64_t */ num_rows,
        /* int64_t */ num_cols,
        /* int64_t */ nnz,
        /* void* */ d_csrOffsets.getData(),
        /* void* */ d_columns.getData(),
        /* void* */ d_values.getData(),
        /* cusparseIndexType_t */ compute_type,
        /* cusparseIndexType_t */ compute_type,
        /* cusparseIndexBase_t */ CUSPARSE_INDEX_BASE_ZERO,
        /* cudaDataType */ data_type));

    // Create dense vector X
    PL_CUSPARSE_IS_SUCCESS(cusparseCreateDnVec(
        /* cusparseDnVecDescr_t* */ &vecX,
        /* int64_t */ num_cols,
        /* void* */ X,
        /* cudaDataType */ data_type));

    // Create dense vector y
    PL_CUSPARSE_IS_SUCCESS(cusparseCreateDnVec(
        /* cusparseDnVecDescr_t* */ &vecY,
        /* int64_t */ num_rows,
        /* void* */ Y,
        /* cudaDataType */ data_type));

    // allocate an external buffer if needed
    PL_CUSPARSE_IS_SUCCESS(cusparseSpMV_bufferSize(
        /* cusparseHandle_t */ handle,
        /* cusparseOperation_t */ CUSPARSE_OPERATION_NON_TRANSPOSE,
        /* const void* */ &alpha,
        /* cusparseSpMatDescr_t */ mat,
        /* cusparseDnVecDescr_t */ vecX,
        /* const void* */ &beta,
        /* cusparseDnVecDescr_t */ vecY,
        /* cudaDataType */ data_type,
        /* cusparseSpMVAlg_t */ CUSPARSE_SPMV_ALG_DEFAULT,
        /* size_t* */ &bufferSize));

    DataBuffer<CFP_t, int> dBuffer{bufferSize, device_id, stream_id, true};

    // execute SpMV
    PL_CUSPARSE_IS_SUCCESS(cusparseSpMV(
        /* cusparseHandle_t */ handle,
        /* cusparseOperation_t */ CUSPARSE_OPERATION_NON_TRANSPOSE,
        /* const void* */ &alpha,
        /* cusparseSpMatDescr_t */ mat,
        /* cusparseDnVecDescr_t */ vecX,
        /* const void* */ &beta,
        /* cusparseDnVecDescr_t */ vecY,
        /* cudaDataType */ data_type,
        /* cusparseSpMVAlg_t */ CUSPARSE_SPMV_ALG_DEFAULT,
        /* void* */
        reinterpret_cast<void *>(dBuffer.getData())));

    // destroy matrix/vector descriptors
    PL_CUSPARSE_IS_SUCCESS(cusparseDestroySpMat(mat));
    PL_CUSPARSE_IS_SUCCESS(cusparseDestroyDnVec(vecX));
    PL_CUSPARSE_IS_SUCCESS(cusparseDestroyDnVec(vecY));
}

/**
 * @brief Sparse matrix vector multiply offloaded to cuSparse (Y =
 * alpha*SparseMat*X + beta)
 *
 * @tparam index_type Integer type for offsets, indices and number of elements
 * (size_t for the moment).
 * @tparam Precision Floating data-type.
 * @tparam DevTypeID Integer type of device id.
 *
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
inline void SparseMV_cuSparse(const index_type *csrOffsets_ptr,
                              const int64_t csrOffsets_size,
                              const index_type *columns_ptr,
                              const std::complex<Precision> *values_ptr,
                              const int64_t numNNZ, const CFP_t *X, CFP_t *Y,
                              DevTypeID device_id, cudaStream_t stream_id,
                              cusparseHandle_t handle) {
    const int64_t num_rows =
        csrOffsets_size -
        1; // int64_t is required for num_rows by cusparseCreateCsr
    const int64_t num_cols =
        num_rows; // int64_t is required for num_cols by cusparseCreateCsr
    const int64_t nnz =
        numNNZ; // int64_t is required for nnz by cusparseCreateCsr

    const CFP_t alpha = {1.0, 0.0};
    const CFP_t beta = {0.0, 0.0};

    DataBuffer<index_type, int> d_csrOffsets{
        static_cast<std::size_t>(csrOffsets_size), device_id, stream_id, true};
    DataBuffer<index_type, int> d_columns{static_cast<std::size_t>(numNNZ),
                                          device_id, stream_id, true};
    DataBuffer<CFP_t, int> d_values{static_cast<std::size_t>(numNNZ), device_id,
                                    stream_id, true};

    d_csrOffsets.CopyHostDataToGpu(csrOffsets_ptr, d_csrOffsets.getLength(),
                                   false);
    d_columns.CopyHostDataToGpu(columns_ptr, d_columns.getLength(), false);
    d_values.CopyHostDataToGpu(values_ptr, d_values.getLength(), false);

    cudaDataType_t data_type;
    cusparseIndexType_t compute_type = CUSPARSE_INDEX_64I;

    if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                  std::is_same_v<CFP_t, double2>) {
        data_type = CUDA_C_64F;
    } else {
        data_type = CUDA_C_32F;
    }

    // CUSPARSE APIs
    cusparseSpMatDescr_t mat;
    cusparseDnVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;

    size_t bufferSize = 0;

    // Create sparse matrix A in CSR format
    PL_CUSPARSE_IS_SUCCESS(cusparseCreateCsr(
        /* cusparseSpMatDescr_t* */ &mat,
        /* int64_t */ num_rows,
        /* int64_t */ num_cols,
        /* int64_t */ nnz,
        /* void* */ d_csrOffsets.getData(),
        /* void* */ d_columns.getData(),
        /* void* */ d_values.getData(),
        /* cusparseIndexType_t */ compute_type,
        /* cusparseIndexType_t */ compute_type,
        /* cusparseIndexBase_t */ CUSPARSE_INDEX_BASE_ZERO,
        /* cudaDataType */ data_type));

    // Create dense vector X
    PL_CUSPARSE_IS_SUCCESS(cusparseCreateDnVec(
        /* cusparseDnVecDescr_t* */ &vecX,
        /* int64_t */ num_cols,
        /* void* */ const_cast<void *>(static_cast<const void *>(X)),
        /* cudaDataType */ data_type));

    // Create dense vector y
    PL_CUSPARSE_IS_SUCCESS(cusparseCreateDnVec(
        /* cusparseDnVecDescr_t* */ &vecY,
        /* int64_t */ num_rows,
        /* void* */ Y,
        /* cudaDataType */ data_type));

    // allocate an external buffer if needed
    PL_CUSPARSE_IS_SUCCESS(cusparseSpMV_bufferSize(
        /* cusparseHandle_t */ handle,
        /* cusparseOperation_t */ CUSPARSE_OPERATION_NON_TRANSPOSE,
        /* const void* */ &alpha,
        /* cusparseSpMatDescr_t */ mat,
        /* cusparseDnVecDescr_t */ vecX,
        /* const void* */ &beta,
        /* cusparseDnVecDescr_t */ vecY,
        /* cudaDataType */ data_type,
        /* cusparseSpMVAlg_t */ CUSPARSE_SPMV_ALG_DEFAULT,
        /* size_t* */ &bufferSize));

    DataBuffer<cudaDataType_t, int> dBuffer{bufferSize, device_id, stream_id,
                                            true};

    // execute SpMV
    PL_CUSPARSE_IS_SUCCESS(cusparseSpMV(
        /* cusparseHandle_t */ handle,
        /* cusparseOperation_t */ CUSPARSE_OPERATION_NON_TRANSPOSE,
        /* const void* */ &alpha,
        /* cusparseSpMatDescr_t */ mat,
        /* cusparseDnVecDescr_t */ vecX,
        /* const void* */ &beta,
        /* cusparseDnVecDescr_t */ vecY,
        /* cudaDataType */ data_type,
        /* cusparseSpMVAlg_t */ CUSPARSE_SPMV_ALG_DEFAULT,
        /* void* */
        reinterpret_cast<void *>(dBuffer.getData())));

    // destroy matrix/vector descriptors
    PL_CUSPARSE_IS_SUCCESS(cusparseDestroySpMat(mat));
    PL_CUSPARSE_IS_SUCCESS(cusparseDestroyDnVec(vecX));
    PL_CUSPARSE_IS_SUCCESS(cusparseDestroyDnVec(vecY));
}
} // namespace Pennylane::LightningGPU::Util
