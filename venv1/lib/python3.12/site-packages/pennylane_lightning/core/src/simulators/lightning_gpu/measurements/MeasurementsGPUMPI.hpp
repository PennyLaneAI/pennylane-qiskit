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

#include "MPILinearAlg.hpp"
#include "MPIManager.hpp"
#include "MPIWorker.hpp"
#include "MeasurementsBase.hpp"
#include "Observables.hpp"
#include "ObservablesGPU.hpp"
#include "ObservablesGPUMPI.hpp"
#include "StateVectorCudaMPI.hpp"
#include "StateVectorCudaManaged.hpp"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::Measures;
using namespace Pennylane::Observables;
using namespace Pennylane::LightningGPU::Observables;
using namespace Pennylane::LightningGPU::MPI;
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
class MeasurementsMPI final
    : public MeasurementsBase<StateVectorT, MeasurementsMPI<StateVectorT>> {
  private:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using BaseType =
        MeasurementsBase<StateVectorT, MeasurementsMPI<StateVectorT>>;
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));
    cudaDataType_t data_type_;
    MPIManager mpi_manager_;
    GateCache<PrecisionT> gate_cache_;

  public:
    explicit MeasurementsMPI(StateVectorT &statevector)
        : BaseType{statevector}, mpi_manager_(statevector.getMPIManager()),
          gate_cache_(true, statevector.getDataBuffer().getDevTag()) {
        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type_ = CUDA_C_64F;
        } else {
            data_type_ = CUDA_C_32F;
        }
        mpi_manager_.Barrier();
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
        std::vector<double> subgroup_probabilities;

        // this should be built upon by the wires not participating
        int maskLen = 0;
        int *maskBitString = nullptr;
        int *maskOrdering = nullptr;

        std::vector<int> wires_int(wires.size());

        // Transform indices between PL & cuQuantum ordering
        std::transform(wires.begin(), wires.end(), wires_int.begin(),
                       [&](std::size_t x) {
                           return static_cast<int>(
                               this->_statevector.getTotalNumQubits() - 1 - x);
                       });

        // split wires_int to global and local ones
        std::vector<int> wires_local;
        std::vector<int> wires_global;

        for (const auto &wire : wires_int) {
            if (wire <
                static_cast<int>(this->_statevector.getNumLocalQubits())) {
                wires_local.push_back(wire);
            } else {
                wires_global.push_back(wire);
            }
        }

        std::vector<double> local_probabilities(
            Pennylane::Util::exp2(wires_local.size()));

        PL_CUSTATEVEC_IS_SUCCESS(custatevecAbs2SumArray(
            /* custatevecHandle_t */ this->_statevector.getCusvHandle(),
            /* const void* */ this->_statevector.getData(),
            /* cudaDataType_t */ data_type_,
            /* const uint32_t */ this->_statevector.getNumLocalQubits(),
            /* double* */ local_probabilities.data(),
            /* const int32_t* */ wires_local.data(),
            /* const uint32_t */ wires_local.size(),
            /* const int32_t* */ maskBitString,
            /* const int32_t* */ maskOrdering,
            /* const uint32_t */ maskLen));

        // create new MPI communicator groups
        size_t subCommGroupId = 0;
        for (size_t i = 0; i < wires_global.size(); i++) {
            size_t mask =
                1 << (wires_global[i] - this->_statevector.getNumLocalQubits());
            size_t bitValue = mpi_manager_.getRank() & mask;
            subCommGroupId += bitValue
                              << (wires_global[i] -
                                  this->_statevector.getNumLocalQubits());
        }
        auto sub_mpi_manager0 =
            mpi_manager_.split(subCommGroupId, mpi_manager_.getRank());

        if (sub_mpi_manager0.getSize() == 1) {
            if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                          std::is_same_v<CFP_t, double2>) {
                return local_probabilities;
            } else {
                std::vector<PrecisionT> local_probs(
                    Pennylane::Util::exp2(wires_local.size()));
                std::transform(
                    local_probabilities.begin(), local_probabilities.end(),
                    local_probs.begin(),
                    [&](double x) { return static_cast<PrecisionT>(x); });
                return local_probs;
            }
        } else {
            // LCOV_EXCL_START
            // Won't be covered with 2 MPI processes in CI checks
            if (sub_mpi_manager0.getRank() == 0) {
                subgroup_probabilities.resize(
                    Pennylane::Util::exp2(wires_local.size()));
            }

            sub_mpi_manager0.Reduce<double>(local_probabilities,
                                            subgroup_probabilities, 0, "sum");

            if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                          std::is_same_v<CFP_t, double2>) {
                return subgroup_probabilities;
            } else {
                std::vector<PrecisionT> local_probs(
                    Pennylane::Util::exp2(wires_local.size()));
                std::transform(
                    subgroup_probabilities.begin(),
                    subgroup_probabilities.end(), local_probs.begin(),
                    [&](double x) { return static_cast<PrecisionT>(x); });
                return local_probs;
            }
            // LCOV_EXCL_STOP
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
     * @param num_shots Number of shots(Optional).  If specified with a non-zero
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
        double epsilon = 1e-15;
        size_t nSubSvs = 1UL << (this->_statevector.getNumGlobalQubits());
        std::vector<double> rand_nums(num_samples);
        std::vector<size_t> samples(
            num_samples * this->_statevector.getTotalNumQubits(), 0);

        size_t bitStringLen = this->_statevector.getNumGlobalQubits() +
                              this->_statevector.getNumLocalQubits();

        std::vector<int> bitOrdering(bitStringLen);

        for (size_t i = 0; i < bitOrdering.size(); i++) {
            bitOrdering[i] = i;
        }

        std::vector<custatevecIndex_t> localBitStrings(num_samples);
        std::vector<custatevecIndex_t> globalBitStrings(num_samples);

        if (mpi_manager_.getRank() == 0) {
            for (size_t n = 0; n < num_samples; n++) {
                rand_nums[n] = (n + 1.0) / (num_samples + 2.0);
            }
        }

        mpi_manager_.Bcast<double>(rand_nums, 0);

        custatevecSamplerDescriptor_t sampler;

        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerCreate(
            /* custatevecHandle_t */ this->_statevector.getCusvHandle(),
            /* const void* */ this->_statevector.getData(),
            /* cudaDataType_t */ data_type_,
            /* const uint32_t */ this->_statevector.getNumLocalQubits(),
            /* custatevecSamplerDescriptor_t * */ &sampler,
            /* uint32_t */ num_samples,
            /* size_t* */ &extraWorkspaceSizeInBytes));

        if (extraWorkspaceSizeInBytes > 0)
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));

        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerPreprocess(
            /* custatevecHandle_t */ this->_statevector.getCusvHandle(),
            /* custatevecSamplerDescriptor_t */ sampler,
            /* void* */ extraWorkspace,
            /* const size_t */ extraWorkspaceSizeInBytes));

        double subNorm = 0;
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerGetSquaredNorm(
            /* custatevecHandle_t */ this->_statevector.getCusvHandle(),
            /* custatevecSamplerDescriptor_t */ sampler,
            /* double * */ &subNorm));

        int source = (mpi_manager_.getRank() - 1 + mpi_manager_.getSize()) %
                     mpi_manager_.getSize();
        int dest = (mpi_manager_.getRank() + 1) % mpi_manager_.getSize();

        double cumulative = 0;
        mpi_manager_.Scan<double>(subNorm, cumulative, "sum");

        double norm = cumulative;
        mpi_manager_.Bcast<double>(norm, mpi_manager_.getSize() - 1);

        double precumulative;
        mpi_manager_.Sendrecv<double>(cumulative, dest, precumulative, source);
        if (mpi_manager_.getRank() == 0) {
            precumulative = 0;
        }
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();

        // Ensure the 'custatevecSamplerApplySubSVOffset' function can be called
        // successfully without reducing accuracy.
        // LCOV_EXCL_START
        if (precumulative == norm) {
            precumulative = norm - epsilon;
        }
        // LCOV_EXCL_STOP
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerApplySubSVOffset(
            /* custatevecHandle_t */ this->_statevector.getCusvHandle(),
            /* custatevecSamplerDescriptor_t */ sampler,
            /* int32_t */ static_cast<int>(mpi_manager_.getRank()),
            /* uint32_t */ nSubSvs,
            /* double */ precumulative,
            /* double */ norm));

        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        auto low = std::lower_bound(rand_nums.begin(), rand_nums.end(),
                                    cumulative / norm);
        int shotOffset = std::distance(rand_nums.begin(), low);
        if (mpi_manager_.getRank() == (mpi_manager_.getSize() - 1)) {
            shotOffset = num_samples;
        }

        int preshotOffset;
        mpi_manager_.Sendrecv<int>(shotOffset, dest, preshotOffset, source);
        if (mpi_manager_.getRank() == 0) {
            preshotOffset = 0;
        }
        mpi_manager_.Barrier();

        int nSubShots = shotOffset - preshotOffset;
        if (nSubShots > 0) {
            PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerSample(
                /* custatevecHandle_t */ this->_statevector.getCusvHandle(),
                /* custatevecSamplerDescriptor_t */ sampler,
                /* custatevecIndex_t* */ &localBitStrings[preshotOffset],
                /* const int32_t * */ bitOrdering.data(),
                /* const uint32_t */ bitStringLen,
                /* const double * */ &rand_nums[preshotOffset],
                /* const uint32_t */ nSubShots,
                /* enum custatevecSamplerOutput_t */
                CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER));
        }

        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerDestroy(sampler));

        if (extraWorkspaceSizeInBytes > 0) {
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
        }

        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();

        mpi_manager_.Allreduce<custatevecIndex_t>(localBitStrings,
                                                  globalBitStrings, "sum");

        for (size_t i = 0; i < num_samples; i++) {
            for (size_t j = 0; j < bitStringLen; j++) {
                samples[i * bitStringLen + (bitStringLen - 1 - j)] =
                    (globalBitStrings[i] >> j) & 1U;
            }
        }
        mpi_manager_.Barrier();
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
        if (mpi_manager_.getRank() == 0) {
            PL_ABORT_IF_NOT(
                static_cast<size_t>(csrOffsets_size - 1) ==
                    (size_t{1} << this->_statevector.getTotalNumQubits()),
                "Incorrect size of CSR Offsets.");
            PL_ABORT_IF_NOT(numNNZ > 0, "Empty CSR matrix.");
        }

        PrecisionT local_expect = 0;

        auto device_id =
            this->_statevector.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id =
            this->_statevector.getDataBuffer().getDevTag().getStreamID();

        const size_t length_local = size_t{1}
                                    << this->_statevector.getNumLocalQubits();
        DataBuffer<CFP_t, int> d_res_per_rowblock{length_local, device_id,
                                                  stream_id, true};
        d_res_per_rowblock.zeroInit();

        // with new wrapper
        cuUtil::SparseMV_cuSparseMPI<index_type, PrecisionT, CFP_t>(
            mpi_manager_, length_local, csrOffsets_ptr, csrOffsets_size,
            columns_ptr, values_ptr,
            const_cast<CFP_t *>(this->_statevector.getData()),
            d_res_per_rowblock.getData(), device_id, stream_id,
            this->_statevector.getCusparseHandle());

        local_expect =
            innerProdC_CUDA(d_res_per_rowblock.getData(),
                            this->_statevector.getData(),
                            this->_statevector.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x;

        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();

        auto expect = mpi_manager_.allreduce<PrecisionT>(local_expect, "sum");
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
        std::vector<CFP_t> gate_matrix = {};
        auto expect =
            this->_statevector.expval(operation, wires, params, gate_matrix);

        return static_cast<PrecisionT>(expect.x);
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
            PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
            mpi_manager_.Barrier();
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

        auto local_expect =
            innerProdC_CUDA(this->_statevector.getData(), ob_sv.getData(),
                            this->_statevector.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x;

        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
        mpi_manager_.Barrier();

        auto expect = mpi_manager_.allreduce<PrecisionT>(local_expect, "sum");

        return static_cast<PrecisionT>(expect);
    }

    /**
     * @brief Expectation value for a Observable with shots
     *
     * @param obs Observable.
     * @param num_shots Number of shots used to generate samples.
     * @param shot_range The range of samples to use. All samples are used
     * by default.
     * @return Floating point expected value of the observable.
     */

    auto expval(const Observable<StateVectorT> &obs, const size_t &num_shots,
                const std::vector<size_t> &shot_range) -> PrecisionT {
        mpi_manager_.Barrier();
        PrecisionT result = BaseType::expval(obs, num_shots, shot_range);
        mpi_manager_.Barrier();
        return result;
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
        auto expect = this->_statevector.expval(wires, matrix);
        return static_cast<PrecisionT>(expect);
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
                const std::vector<std::vector<std::size_t>> &tgts,
                const std::complex<PrecisionT> *coeffs) -> PrecisionT {
        return this->_statevector.getExpectationValuePauliWords(pauli_words,
                                                                tgts, coeffs);
    }

    /**
     * @brief Calculate variance of a general Observable.
     *
     * @param ob Observable.
     * @return Variance with respect to the given observable.
     */
    auto var(const Observable<StateVectorT> &ob) -> PrecisionT {
        StateVectorT ob_sv(this->_statevector);
        ob.applyInPlace(ob_sv);

        auto device_id = ob_sv.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = ob_sv.getDataBuffer().getDevTag().getStreamID();

        PrecisionT mean_square_local =
            innerProdC_CUDA(ob_sv.getData(), ob_sv.getData(), ob_sv.getLength(),
                            device_id, stream_id, ob_sv.getCublasCaller())
                .x;

        PrecisionT squared_mean_local =
            innerProdC_CUDA(this->_statevector.getData(), ob_sv.getData(),
                            this->_statevector.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x;

        PrecisionT mean_square =
            mpi_manager_.allreduce<PrecisionT>(mean_square_local, "sum");
        PrecisionT squared_mean =
            mpi_manager_.allreduce<PrecisionT>(squared_mean_local, "sum");

        return mean_square - squared_mean * squared_mean;
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
        StateVectorT ob_sv(this->_statevector);
        ob_sv.applyOperation(operation, wires);

        auto device_id = ob_sv.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = ob_sv.getDataBuffer().getDevTag().getStreamID();

        PrecisionT mean_square_local =
            innerProdC_CUDA(ob_sv.getData(), ob_sv.getData(), ob_sv.getLength(),
                            device_id, stream_id, ob_sv.getCublasCaller())
                .x;

        PrecisionT squared_mean_local =
            innerProdC_CUDA(this->_statevector.getData(), ob_sv.getData(),
                            this->_statevector.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x;

        PrecisionT mean_square =
            mpi_manager_.allreduce<PrecisionT>(mean_square_local, "sum");
        PrecisionT squared_mean =
            mpi_manager_.allreduce<PrecisionT>(squared_mean_local, "sum");

        return mean_square - squared_mean * squared_mean;
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
        StateVectorT ob_sv(this->_statevector);
        ob_sv.applyMatrix(matrix, wires);

        auto device_id = ob_sv.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = ob_sv.getDataBuffer().getDevTag().getStreamID();

        PrecisionT mean_square_local =
            innerProdC_CUDA(ob_sv.getData(), ob_sv.getData(), ob_sv.getLength(),
                            device_id, stream_id, ob_sv.getCublasCaller())
                .x;

        PrecisionT squared_mean_local =
            innerProdC_CUDA(this->_statevector.getData(), ob_sv.getData(),
                            this->_statevector.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x;

        PrecisionT mean_square =
            mpi_manager_.allreduce<PrecisionT>(mean_square_local, "sum");
        PrecisionT squared_mean =
            mpi_manager_.allreduce<PrecisionT>(squared_mean_local, "sum");

        return mean_square - squared_mean * squared_mean;
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

        std::vector<PrecisionT> var_list;

        for (size_t index = 0; index < operations_list.size(); index++) {
            var_list.emplace_back(
                var(operations_list[index], wires_list[index]));
        }

        return var_list;
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
        if (mpi_manager_.getRank() == 0) {
            PL_ABORT_IF_NOT(
                static_cast<size_t>(csrOffsets_size - 1) ==
                    (size_t{1} << this->_statevector.getTotalNumQubits()),
                "Incorrect size of CSR Offsets.");
            PL_ABORT_IF_NOT(numNNZ > 0, "Empty CSR matrix.");
        }

        auto device_id =
            this->_statevector.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id =
            this->_statevector.getDataBuffer().getDevTag().getStreamID();
        const size_t length_local = size_t{1}
                                    << this->_statevector.getNumLocalQubits();

        DataBuffer<CFP_t, int> d_res_per_rowblock{length_local, device_id,
                                                  stream_id, true};
        d_res_per_rowblock.zeroInit();

        cuUtil::SparseMV_cuSparseMPI<index_type, PrecisionT, CFP_t>(
            mpi_manager_, length_local, csrOffsets_ptr, csrOffsets_size,
            columns_ptr, values_ptr,
            const_cast<CFP_t *>(this->_statevector.getData()),
            d_res_per_rowblock.getData(), device_id, stream_id,
            this->_statevector.getCusparseHandle());

        PrecisionT mean_square_local =
            innerProdC_CUDA(d_res_per_rowblock.getData(),
                            d_res_per_rowblock.getData(),
                            d_res_per_rowblock.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x;

        PrecisionT squared_mean_local =
            innerProdC_CUDA(this->_statevector.getData(),
                            d_res_per_rowblock.getData(),
                            this->_statevector.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x;

        PrecisionT mean_square =
            mpi_manager_.allreduce<PrecisionT>(mean_square_local, "sum");
        PrecisionT squared_mean =
            mpi_manager_.allreduce<PrecisionT>(squared_mean_local, "sum");

        return mean_square - squared_mean * squared_mean;
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
}; // class Measurements
} // namespace Pennylane::LightningGPU::Measures