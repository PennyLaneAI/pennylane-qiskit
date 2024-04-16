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
 * Defines a class for the measurement of observables in quantum states
 * represented by a Lightning Qubit StateVector class.
 */

#pragma once

#include <algorithm>
#include <complex>
#include <cuda.h>
#include <cusparse.h>
#include <custatevec.h> // custatevecApplyMatrix
#include <random>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "MeasurementsBase.hpp"
#include "Observables.hpp"
#include "ObservablesGPU.hpp"
#include "StateVectorCudaManaged.hpp"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::Measures;
using namespace Pennylane::Observables;
using namespace Pennylane::LightningGPU::Observables;
namespace cuUtil = Pennylane::LightningGPU::Util;
using Pennylane::LightningGPU::StateVectorCudaManaged;
using namespace Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningGPU::Measures {
/**
 * @brief Observable's Measurement Class.
 *
 * This class couples with a statevector to performs measurements.
 * Observables are defined by its operator(matrix), the observable class,
 * or through a string-based function dispatch.
 *
 * @tparam StateVectorT type of the statevector to be measured.
 */
template <class StateVectorT>
class Measurements final
    : public MeasurementsBase<StateVectorT, Measurements<StateVectorT>> {
  private:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using BaseType = MeasurementsBase<StateVectorT, Measurements<StateVectorT>>;
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));
    cudaDataType_t data_type_;

    GateCache<PrecisionT> gate_cache_;

  public:
    explicit Measurements(StateVectorT &statevector)
        : BaseType{statevector},
          gate_cache_(true, statevector.getDataBuffer().getDevTag()) {
        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type_ = CUDA_C_64F;
        } else {
            data_type_ = CUDA_C_32F;
        }
    };

    /**
     * @brief Utility method for probability calculation using given wires.
     *
     * @param wires List of wires to return probabilities for in lexicographical
     * order.
     * @return std::vector<PrecisionT>
     */
    auto probs(const std::vector<size_t> &wires) -> std::vector<PrecisionT> {
        // Data return type fixed as double in custatevec function call
        std::vector<double> probabilities(Pennylane::Util::exp2(wires.size()));
        // this should be built upon by the wires not participating
        int maskLen =
            0; // static_cast<int>(BaseType::getNumQubits() - wires.size());
        int *maskBitString = nullptr; //
        int *maskOrdering = nullptr;

        cudaDataType_t data_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
        } else {
            data_type = CUDA_C_32F;
        }

        std::vector<int> wires_int(wires.size());

        // Transform indices between PL & cuQuantum ordering
        std::transform(wires.begin(), wires.end(), wires_int.begin(),
                       [&](std::size_t x) {
                           return static_cast<int>(
                               this->_statevector.getNumQubits() - 1 - x);
                       });

        PL_CUSTATEVEC_IS_SUCCESS(custatevecAbs2SumArray(
            /* custatevecHandle_t */ this->_statevector.getCusvHandle(),
            /* const void* */ this->_statevector.getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ this->_statevector.getNumQubits(),
            /* double* */ probabilities.data(),
            /* const int32_t* */ wires_int.data(),
            /* const uint32_t */ wires_int.size(),
            /* const int32_t* */ maskBitString,
            /* const int32_t* */ maskOrdering,
            /* const uint32_t */ maskLen));

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            return probabilities;
        } else {
            std::vector<PrecisionT> probs(Pennylane::Util::exp2(wires.size()));
            std::transform(
                probabilities.begin(), probabilities.end(), probs.begin(),
                [&](double x) { return static_cast<PrecisionT>(x); });
            return probs;
        }
    }

    /**
     * @brief Utility method for probability calculation for a full wires.
     *
     * @return std::vector<PrecisionT>
     */
    auto probs() -> std::vector<PrecisionT> {
        std::vector<size_t> wires;
        for (size_t i = 0; i < this->_statevector.getNumQubits(); i++) {
            wires.push_back(i);
        }
        return this->probs(wires);
    }

    /**
     * @brief Probabilities to measure rotated basis states.
     *
     * @param obs An observable object.
     * @param num_shots Number of shots(Optional).If specified with a non-zero
     * number, shot-noise will be added to return probabilities
     *
     * @return Floating point std::vector with probabilities
     * in lexicographic order.
     */
    std::vector<PrecisionT> probs(const Observable<StateVectorT> &obs,
                                  size_t num_shots = 0) {
        return BaseType::probs(obs, num_shots);
    }

    /**
     * @brief Probabilities with shot-noise.
     *
     * @param num_shots Number of shots.
     *
     * @return Floating point std::vector with probabilities.
     */
    std::vector<PrecisionT> probs(size_t num_shots) {
        return BaseType::probs(num_shots);
    }

    /**
     * @brief Probabilities with shot-noise for a subset of the full system.
     *
     * @param num_shots Number of shots.
     * @param wires Wires will restrict probabilities to a subset
     * of the full system.
     *
     * @return Floating point std::vector with probabilities.
     */

    std::vector<PrecisionT> probs(const std::vector<size_t> &wires,
                                  size_t num_shots) {
        return BaseType::probs(wires, num_shots);
    }

    /**
     * @brief Utility method for samples.
     *
     * @param num_samples Number of Samples
     *
     * @return std::vector<size_t> A 1-d array storing the samples.
     * Each sample has a length equal to the number of qubits. Each sample can
     * be accessed using the stride sample_id*num_qubits, where sample_id is a
     * number between 0 and num_samples-1.
     */
    auto generate_samples(size_t num_samples) -> std::vector<size_t> {
        std::vector<double> rand_nums(num_samples);
        custatevecSamplerDescriptor_t sampler;

        const size_t num_qubits = this->_statevector.getNumQubits();
        const int bitStringLen = this->_statevector.getNumQubits();

        std::vector<int> bitOrdering(num_qubits);
        std::iota(std::begin(bitOrdering), std::end(bitOrdering),
                  0); // Fill with 0, 1, ...,

        cudaDataType_t data_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
        } else {
            data_type = CUDA_C_32F;
        }

        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<PrecisionT> dis(0.0, 1.0);
        for (size_t n = 0; n < num_samples; n++) {
            rand_nums[n] = dis(gen);
        }
        std::vector<size_t> samples(num_samples * num_qubits, 0);
        std::unordered_map<size_t, size_t> cache;
        std::vector<custatevecIndex_t> bitStrings(num_samples);

        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;
        // create sampler and check the size of external workspace
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerCreate(
            this->_statevector.getCusvHandle(), this->_statevector.getData(),
            data_type, num_qubits, &sampler, num_samples,
            &extraWorkspaceSizeInBytes));

        // allocate external workspace if necessary
        if (extraWorkspaceSizeInBytes > 0)
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));

        // sample preprocess
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerPreprocess(
            this->_statevector.getCusvHandle(), sampler, extraWorkspace,
            extraWorkspaceSizeInBytes));

        // sample bit strings
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerSample(
            this->_statevector.getCusvHandle(), sampler, bitStrings.data(),
            bitOrdering.data(), bitStringLen, rand_nums.data(), num_samples,
            CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER));

        // destroy descriptor and handle
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerDestroy(sampler));

        // Pick samples
        for (size_t i = 0; i < num_samples; i++) {
            auto idx = bitStrings[i];
            // If cached, retrieve sample from cache
            if (cache.count(idx) != 0) {
                size_t cache_id = cache[idx];
                auto it_temp = samples.begin() + cache_id * num_qubits;
                std::copy(it_temp, it_temp + num_qubits,
                          samples.begin() + i * num_qubits);
            }
            // If not cached, compute
            else {
                for (size_t j = 0; j < num_qubits; j++) {
                    samples[i * num_qubits + (num_qubits - 1 - j)] =
                        (idx >> j) & 1U;
                }
                cache[idx] = i;
            }
        }

        if (extraWorkspaceSizeInBytes > 0)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));

        return samples;
    }

    /**
     * @brief expval(H) calculation with cuSparseSpMV.
     *
     * @tparam index_type Integer type used as indices of the sparse matrix.
     * @param csr_Offsets_ptr Pointer to the array of row offsets of the sparse
     * matrix. Array of size csrOffsets_size.
     * @param csrOffsets_size Number of Row offsets of the sparse matrix.
     * @param columns_ptr Pointer to the array of column indices of the sparse
     * matrix. Array of size numNNZ
     * @param values_ptr Pointer to the array of the non-zero elements
     * @param numNNZ Number of non-zero elements.
     * @return auto Expectation value.
     */
    template <class index_type>
    auto expval(const index_type *csrOffsets_ptr, const int64_t csrOffsets_size,
                const index_type *columns_ptr,
                const std::complex<PrecisionT> *values_ptr,
                const int64_t numNNZ) -> PrecisionT {
        const std::size_t nIndexBits = this->_statevector.getNumQubits();
        const std::size_t length = std::size_t{1} << nIndexBits;

        auto device_id =
            this->_statevector.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id =
            this->_statevector.getDataBuffer().getDevTag().getStreamID();

        std::unique_ptr<DataBuffer<CFP_t>> d_sv_prime =
            std::make_unique<DataBuffer<CFP_t>>(length, device_id, stream_id,
                                                true);

        cuUtil::SparseMV_cuSparse<index_type, PrecisionT, CFP_t>(
            csrOffsets_ptr, csrOffsets_size, columns_ptr, values_ptr, numNNZ,
            this->_statevector.getData(), d_sv_prime->getData(), device_id,
            stream_id, this->_statevector.getCusparseHandle());

        auto expect =
            innerProdC_CUDA(this->_statevector.getData(), d_sv_prime->getData(),
                            this->_statevector.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x;

        return expect;
    }

    /**
     * @brief Expected value of an observable.
     *
     * @param operation String with the operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point expected value of the observable.
     */
    auto expval(const std::string &operation, const std::vector<size_t> &wires)
        -> PrecisionT {
        std::vector<PrecisionT> params = {0.0};
        std::vector<ComplexT> gate_matrix = {};
        return this->expval_(operation, wires, params, gate_matrix);
    }

    /**
     * @brief Expected value for a list of observables.
     *
     * @tparam op_type Operation type.
     * @param operations_list List of operations to measure.
     * @param wires_list List of wires where to apply the operators.
     * @return Floating point std::vector with expected values for the
     * observables.
     */
    template <typename op_type>
    auto expval(const std::vector<op_type> &operations_list,
                const std::vector<std::vector<size_t>> &wires_list)
        -> std::vector<PrecisionT> {
        PL_ABORT_IF(
            (operations_list.size() != wires_list.size()),
            "The lengths of the list of operations and wires do not match.");
        std::vector<PrecisionT> expected_value_list;

        for (size_t index = 0; index < operations_list.size(); index++) {
            expected_value_list.emplace_back(
                expval(operations_list[index], wires_list[index]));
        }

        return expected_value_list;
    }

    /**
     * @brief Calculate expectation value for a general Observable.
     *
     * @param ob Observable.
     * @return Expectation value with respect to the given observable.
     */
    auto expval(const Observable<StateVectorT> &ob) -> PrecisionT {
        StateVectorT ob_sv(this->_statevector);
        ob.applyInPlace(ob_sv);

        auto device_id = ob_sv.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = ob_sv.getDataBuffer().getDevTag().getStreamID();

        auto expect =
            innerProdC_CUDA(this->_statevector.getData(), ob_sv.getData(),
                            this->_statevector.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x;
        return static_cast<PrecisionT>(expect);
    }

    /**
     * @brief Expectation value for a Observable with shots
     *
     * @param obs Observable.
     * @param num_shots Number of shots used to generate samples
     * @param shot_range The range of samples to use. All samples are used
     * by default.
     * @return Floating point expected value of the observable.
     */

    auto expval(const Observable<StateVectorT> &obs, const size_t &num_shots,
                const std::vector<size_t> &shot_range) -> PrecisionT {
        return BaseType::expval(obs, num_shots, shot_range);
    }

    /**
     * @brief Expected value of an observable.
     *
     * @param matrix Square matrix in row-major order.
     * @param wires Wires where to apply the operator.
     * @return Floating point expected value of the observable.
     */
    auto expval(const std::vector<ComplexT> &matrix,
                const std::vector<size_t> &wires) -> PrecisionT {
        return this->expval_(wires, matrix);
    }

    /**
     * @brief Expected value of an observable.
     *
     * @param pauli_words Vector of operators' name strings.
     * @param target_wires Vector of wires where to apply the operator.
     * @param coeffs Complex buffer of size |pauli_words|
     * @return Floating point expected value of the observable.
     */
    auto expval(const std::vector<std::string> &pauli_words,
                const std::vector<std::vector<std::size_t>> &target_wires,
                const std::complex<PrecisionT> *coeffs) -> PrecisionT {
        uint32_t nIndexBits =
            static_cast<uint32_t>(this->_statevector.getNumQubits());
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
        std::vector<double> expect(pauli_words.size());

        std::vector<std::vector<custatevecPauli_t>> pauliOps;

        std::vector<custatevecPauli_t *> pauliOps_ptr;

        for (auto &p_word : pauli_words) {
            pauliOps.push_back(cuUtil::pauliStringToEnum(p_word));
            pauliOps_ptr.push_back((*pauliOps.rbegin()).data());
        }

        std::vector<std::vector<int32_t>> basisBits;
        std::vector<int32_t *> basisBits_ptr;
        std::vector<uint32_t> n_basisBits;

        for (auto &wires : target_wires) {
            std::vector<int32_t> wiresInt(wires.size());
            std::transform(wires.begin(), wires.end(), wiresInt.begin(),
                           [&](std::size_t x) {
                               return static_cast<int>(
                                   this->_statevector.getNumQubits() - 1 - x);
                           });
            basisBits.push_back(wiresInt);
            basisBits_ptr.push_back((*basisBits.rbegin()).data());
            n_basisBits.push_back(wiresInt.size());
        }

        // compute expectation
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectationsOnPauliBasis(
            /* custatevecHandle_t */ this->_statevector.getCusvHandle(),
            /* const void* */ this->_statevector.getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* double* */ expect.data(),
            /* const custatevecPauli_t ** */
            const_cast<const custatevecPauli_t **>(pauliOps_ptr.data()),
            /* const uint32_t */ static_cast<uint32_t>(pauliOps.size()),
            /* const int32_t ** */
            const_cast<const int32_t **>(basisBits_ptr.data()),
            /* const uint32_t */ n_basisBits.data()));

        std::complex<PrecisionT> result{0, 0};

        if constexpr (std::is_same_v<PrecisionT, double>) {
            for (std::size_t idx = 0; idx < expect.size(); idx++) {
                result += expect[idx] * coeffs[idx];
            }
            return std::real(result);
        } else {
            std::vector<PrecisionT> expect_cast(expect.size());
            std::transform(expect.begin(), expect.end(), expect_cast.begin(),
                           [](double x) { return static_cast<float>(x); });

            for (std::size_t idx = 0; idx < expect_cast.size(); idx++) {
                result += expect_cast[idx] * coeffs[idx];
            }

            return std::real(result);
        }
    }

    /**
     * @brief Calculate variance of a general observable.
     *
     * @param ob An observable object.
     * @return Variance with respect to the given observable.
     */
    auto var(const Observable<StateVectorT> &ob) -> PrecisionT {
        StateVectorT ob_sv(this->_statevector.getData(),
                           this->_statevector.getLength());
        ob.applyInPlace(ob_sv);

        auto device_id = ob_sv.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = ob_sv.getDataBuffer().getDevTag().getStreamID();

        const PrecisionT mean_square =
            innerProdC_CUDA(ob_sv.getData(), ob_sv.getData(), ob_sv.getLength(),
                            device_id, stream_id, ob_sv.getCublasCaller())
                .x;

        const PrecisionT squared_mean = static_cast<PrecisionT>(std::pow(
            innerProdC_CUDA(this->_statevector.getData(), ob_sv.getData(),
                            this->_statevector.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x,
            2));
        return (mean_square - squared_mean);
    }

    /**
     * @brief Variance of an observable.
     *
     * @param operation String with the operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point with the variance of the observable.
     */
    auto var(const std::string &operation, const std::vector<size_t> &wires)
        -> PrecisionT {
        StateVectorT ob_sv(this->_statevector.getData(),
                           this->_statevector.getLength());
        ob_sv.applyOperation(operation, wires);

        auto device_id = ob_sv.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = ob_sv.getDataBuffer().getDevTag().getStreamID();

        const PrecisionT mean_square =
            innerProdC_CUDA(ob_sv.getData(), ob_sv.getData(), ob_sv.getLength(),
                            device_id, stream_id, ob_sv.getCublasCaller())
                .x;

        const PrecisionT squared_mean = static_cast<PrecisionT>(std::pow(
            innerProdC_CUDA(this->_statevector.getData(), ob_sv.getData(),
                            this->_statevector.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x,
            2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Variance of an observable.
     *
     * @param matrix Square matrix in row-major order.
     * @param wires Wires where to apply the operator.
     * @return Floating point with the variance of the observable.
     */
    auto var(const std::vector<ComplexT> &matrix,
             const std::vector<size_t> &wires) -> PrecisionT {
        StateVectorT ob_sv(this->_statevector.getData(),
                           this->_statevector.getLength());
        ob_sv.applyMatrix(matrix, wires);

        auto device_id = ob_sv.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = ob_sv.getDataBuffer().getDevTag().getStreamID();

        const PrecisionT mean_square =
            innerProdC_CUDA(ob_sv.getData(), ob_sv.getData(), ob_sv.getLength(),
                            device_id, stream_id, ob_sv.getCublasCaller())
                .x;

        const PrecisionT squared_mean = static_cast<PrecisionT>(std::pow(
            innerProdC_CUDA(this->_statevector.getData(), ob_sv.getData(),
                            this->_statevector.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x,
            2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Variance for a list of observables.
     *
     * @tparam op_type Operation type.
     * @param operations_list List of operations to measure.
     * Square matrix in row-major order or string with the operator name.
     * @param wires_list List of wires where to apply the operators.
     * @return Floating point std::vector with the variance of the
     observables.
     */
    template <typename op_type>
    auto var(const std::vector<op_type> &operations_list,
             const std::vector<std::vector<size_t>> &wires_list)
        -> std::vector<PrecisionT> {
        PL_ABORT_IF(
            (operations_list.size() != wires_list.size()),
            "The lengths of the list of operations and wires do not match.");

        std::vector<PrecisionT> expected_value_list;

        for (size_t index = 0; index < operations_list.size(); index++) {
            expected_value_list.emplace_back(
                var(operations_list[index], wires_list[index]));
        }

        return expected_value_list;
    };

    /**
     * @brief Variance of a sparse Hamiltonian.
     *
     * @tparam index_type integer type used as indices of the sparse matrix.
     * @param row_map_ptr   row_map array pointer.
     *                      The j element encodes the number of non-zeros
     above
     * row j.
     * @param row_map_size  row_map array size.
     * @param entries_ptr   pointer to an array with column indices of the
     * non-zero elements.
     * @param values_ptr    pointer to an array with the non-zero elements.
     * @param numNNZ        number of non-zero elements.
     * @return Floating point with the variance of the sparse Hamiltonian.
     */
    template <class index_type>
    PrecisionT var(const index_type *csrOffsets_ptr,
                   const int64_t csrOffsets_size, const index_type *columns_ptr,
                   const std::complex<PrecisionT> *values_ptr,
                   const int64_t numNNZ) {
        PL_ABORT_IF(
            (this->_statevector.getLength() != (size_t(csrOffsets_size) - 1)),
            "Statevector and Hamiltonian have incompatible sizes.");

        StateVectorT ob_sv(this->_statevector.getData(),
                           this->_statevector.getLength());

        auto device_id =
            this->_statevector.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id =
            this->_statevector.getDataBuffer().getDevTag().getStreamID();
        cusparseHandle_t handle = this->_statevector.getCusparseHandle();

        cuUtil::SparseMV_cuSparse<index_type, PrecisionT, CFP_t>(
            csrOffsets_ptr, csrOffsets_size, columns_ptr, values_ptr, numNNZ,
            this->_statevector.getData(), ob_sv.getData(), device_id, stream_id,
            handle);

        const PrecisionT mean_square =
            innerProdC_CUDA(ob_sv.getData(), ob_sv.getData(), ob_sv.getLength(),
                            device_id, stream_id, ob_sv.getCublasCaller())
                .x;

        const PrecisionT squared_mean = static_cast<PrecisionT>(std::pow(
            innerProdC_CUDA(this->_statevector.getData(), ob_sv.getData(),
                            this->_statevector.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x,
            2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Calculate the variance for an observable with the number of shots.
     *
     * @param obs An observable object.
     * @param num_shots Number of shots.
     *
     * @return Variance of the given observable.
     */

    auto var(const Observable<StateVectorT> &obs, const size_t &num_shots)
        -> PrecisionT {
        return BaseType::var(obs, num_shots);
    }

  private:
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
    auto expval_(const std::string &obsName, const std::vector<size_t> &wires,
                 const std::vector<PrecisionT> &params = {0.0},
                 const std::vector<std::complex<PrecisionT>> &gate_matrix = {})
        -> PrecisionT {
        auto &&par = (params.empty()) ? std::vector<PrecisionT>{0.0} : params;

        auto &&local_wires = wires;

        std::vector<CFP_t> matrix_cu(gate_matrix.size());
        if (!gate_cache_.gateExists(obsName, par[0]) && gate_matrix.empty()) {
            std::string message =
                "Currently unsupported observable: " + obsName;
            throw LightningException(message.c_str());
        }
        return this->getExpectationValueDeviceMatrix_(
            gate_cache_.get_gate_device_ptr(obsName, par[0]), local_wires);
    }
    /**
     * @brief See `expval(std::vector<CFP_t> &gate_matrix = {})`
     */
    auto expval_(const std::vector<size_t> &wires,
                 const std::vector<std::complex<PrecisionT>> &gate_matrix)
        -> PrecisionT {
        std::vector<CFP_t> matrix_cu(gate_matrix.size());

        for (std::size_t i = 0; i < gate_matrix.size(); i++) {
            matrix_cu[i] =
                cuUtil::complexToCu<std::complex<PrecisionT>>(gate_matrix[i]);
        }

        if (gate_matrix.empty()) {
            std::string message = "Currently unsupported observable";
            throw LightningException(message.c_str());
        }

        // Wire order reversed to match expected custatevec wire ordering for
        // tensor observables.
        auto &&local_wires = std::vector<size_t>{wires.rbegin(), wires.rend()};

        auto expect_val = this->getExpectationValueDeviceMatrix_(
            matrix_cu.data(), local_wires);
        return expect_val;
    }

    /**
     * @brief Get expectation of a given host or device defined array.
     *
     * @param matrix Host or device defined row-major order gate matrix array.
     * @param tgts Target qubits.
     * @return auto Expectation value.
     */
    auto
    getExpectationValueDeviceMatrix_(const CFP_t *matrix,
                                     const std::vector<std::size_t> &tgts) {
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        std::vector<int> tgtsInt(tgts.size());
        std::transform(tgts.begin(), tgts.end(), tgtsInt.begin(),
                       [&](std::size_t x) {
                           return static_cast<int>(
                               this->_statevector.getNumQubits() - 1 - x);
                       });

        size_t nIndexBits = this->_statevector.getNumQubits();
        cudaDataType_t data_type;
        cudaDataType_t expectationDataType =
            CUDA_C_64F; // Requested by the custatevecComputeExpectation API
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
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectationGetWorkspaceSize(
            /* custatevecHandle_t */ this->_statevector.getCusvHandle(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix,
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const uint32_t */ tgtsInt.size(),
            /* custatevecComputeType_t */ compute_type,
            /* size_t* */ &extraWorkspaceSizeInBytes));

        // LCOV_EXCL_START
        if (extraWorkspaceSizeInBytes > 0) {
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
        }
        // LCOV_EXCL_STOP

        cuDoubleComplex expect;

        // compute expectation
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectation(
            /* custatevecHandle_t */ this->_statevector.getCusvHandle(),
            /* const void* */ this->_statevector.getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* void* */ &expect,
            /* cudaDataType_t */ expectationDataType,
            /* double* */ nullptr,
            /* const void* */ matrix,
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t* */ tgtsInt.data(),
            /* const uint32_t */ tgtsInt.size(),
            /* custatevecComputeType_t */ compute_type,
            /* void* */ extraWorkspace,
            /* size_t */ extraWorkspaceSizeInBytes));

        // LCOV_EXCL_START
        if (extraWorkspaceSizeInBytes)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
        // LCOV_EXCL_STOP

        return static_cast<PrecisionT>(expect.x);
    }
}; // class Measurements
} // namespace Pennylane::LightningGPU::Measures