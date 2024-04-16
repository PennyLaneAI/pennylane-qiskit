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
#include <cstdio>
#include <random>
#include <stack>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "LinearAlgebra.hpp"
#include "MeasurementsBase.hpp"
#include "Observables.hpp"
#include "SparseLinAlg.hpp"
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"
#include "TransitionKernels.hpp"
#include "Util.hpp" //transpose_state_tensor, sorting_indices

/// @cond DEV
namespace {
using namespace Pennylane::Measures;
using namespace Pennylane::Observables;
using Pennylane::LightningQubit::StateVectorLQubitManaged;
using Pennylane::LightningQubit::Util::innerProdC;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Measures {
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

  public:
    explicit Measurements(const StateVectorT &statevector)
        : BaseType{statevector} {};

    /**
     * @brief Probabilities of each computational basis state.
     *
     * @return Floating point std::vector with probabilities
     * in lexicographic order.
     */
    std::vector<PrecisionT> probs() {
        const ComplexT *arr_data = this->_statevector.getData();
        std::vector<PrecisionT> basis_probs(this->_statevector.getLength(), 0);

        std::transform(
            arr_data, arr_data + this->_statevector.getLength(),
            basis_probs.begin(),
            [](const ComplexT &z) -> PrecisionT { return std::norm(z); });
        return basis_probs;
    };

    /**
     * @brief Probabilities for a subset of the full system.
     *
     * @param wires Wires will restrict probabilities to a subset
     * of the full system.
     * @return Floating point std::vector with probabilities.
     * The basis columns are rearranged according to wires.
     */
    std::vector<PrecisionT>
    probs(const std::vector<size_t> &wires,
          [[maybe_unused]] const std::vector<size_t> &device_wires = {}) {
        // Determining index that would sort the vector.
        // This information is needed later.
        const auto sorted_ind_wires = Pennylane::Util::sorting_indices(wires);
        // Sorting wires.
        std::vector<size_t> sorted_wires(wires.size());
        for (size_t pos = 0; pos < wires.size(); pos++) {
            sorted_wires[pos] = wires[sorted_ind_wires[pos]];
        }
        // Determining probabilities for the sorted wires.
        const ComplexT *arr_data = this->_statevector.getData();

        size_t num_qubits = this->_statevector.getNumQubits();

        const std::vector<size_t> all_indices =
            Gates::generateBitPatterns(sorted_wires, num_qubits);
        const std::vector<size_t> all_offsets = Gates::generateBitPatterns(
            Gates::getIndicesAfterExclusion(sorted_wires, num_qubits),
            num_qubits);

        std::vector<PrecisionT> probabilities(all_indices.size(), 0);

        size_t ind_probs = 0;
        for (auto index : all_indices) {
            for (auto offset : all_offsets) {
                probabilities[ind_probs] += std::norm(arr_data[index + offset]);
            }
            ind_probs++;
        }
        // Transposing the probabilities tensor with the indices determined
        // at the beginning.
        if (wires != sorted_wires) {
            probabilities = Pennylane::Util::transpose_state_tensor(
                probabilities, sorted_ind_wires);
        }
        return probabilities;
    }

    /**
     * @brief Probabilities to measure rotated basis states.
     *
     * @param obs An observable object.
     * @param num_shots Number of shots (Optional). If specified with a non-zero
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
     * @brief Expected value of an observable.
     *
     * @param matrix Square matrix in row-major order.
     * @param wires Wires where to apply the operator.
     * @return Floating point expected value of the observable.
     */
    PrecisionT expval(const std::vector<ComplexT> &matrix,
                      const std::vector<size_t> &wires) {
        // Copying the original state vector, for the application of the
        // observable operator.
        StateVectorLQubitManaged<PrecisionT> operator_statevector(
            this->_statevector);

        operator_statevector.applyMatrix(matrix, wires);

        ComplexT expected_value = innerProdC(this->_statevector.getData(),
                                             operator_statevector.getData(),
                                             this->_statevector.getLength());
        return std::real(expected_value);
    };

    /**
     * @brief Expected value of an observable.
     *
     * @param operation String with the operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point expected value of the observable.
     */
    PrecisionT expval(const std::string &operation,
                      const std::vector<size_t> &wires) {
        // Copying the original state vector, for the application of the
        // observable operator.
        StateVectorLQubitManaged<PrecisionT> operator_statevector(
            this->_statevector);

        operator_statevector.applyOperation(operation, wires);

        ComplexT expected_value = innerProdC(this->_statevector.getData(),
                                             operator_statevector.getData(),
                                             this->_statevector.getLength());
        return std::real(expected_value);
    };

    /**
     * @brief Expected value of a Sparse Hamiltonian.
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
     * @return Floating point expected value of the observable.
     */
    template <class index_type>
    PrecisionT expval(const index_type *row_map_ptr,
                      const index_type row_map_size,
                      const index_type *entries_ptr, const ComplexT *values_ptr,
                      const index_type numNNZ) {
        PL_ABORT_IF(
            (this->_statevector.getLength() != (size_t(row_map_size) - 1)),
            "Statevector and Hamiltonian have incompatible sizes.");
        auto operator_vector = Util::apply_Sparse_Matrix(
            this->_statevector.getData(),
            static_cast<index_type>(this->_statevector.getLength()),
            row_map_ptr, row_map_size, entries_ptr, values_ptr, numNNZ);

        ComplexT expected_value =
            innerProdC(this->_statevector.getData(), operator_vector.data(),
                       this->_statevector.getLength());
        return std::real(expected_value);
    };

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
    std::vector<PrecisionT>
    expval(const std::vector<op_type> &operations_list,
           const std::vector<std::vector<size_t>> &wires_list) {
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
     * @brief Expectation value for a general Observable
     *
     * @param ob Observable
     * @return Floating point expected value of the observable.
     */
    auto expval(const Observable<StateVectorT> &ob) -> PrecisionT {
        PrecisionT result{};

        if constexpr (std::is_same_v<typename StateVectorT::MemoryStorageT,
                                     MemoryStorageLocation::Internal>) {
            StateVectorT sv(this->_statevector);
            result = calculateObsExpval(sv, ob, this->_statevector);
        } else if constexpr (std::is_same_v<
                                 typename StateVectorT::MemoryStorageT,
                                 MemoryStorageLocation::External>) {
            std::vector<ComplexT> data_storage(
                this->_statevector.getData(),
                this->_statevector.getData() + this->_statevector.getLength());
            StateVectorT sv(data_storage.data(), data_storage.size());
            result = calculateObsExpval(sv, ob, this->_statevector);
        } else {
            /// LCOV_EXCL_START
            PL_ABORT("Undefined memory storage location for StateVectorT.");
            /// LCOV_EXCL_STOP
        }

        return result;
    }

    /**
     * @brief Expectation value for a Observable with shots
     *
     * @param obs Observable
     * @param num_shots Number of shots.
     * @param shots_range Vector of shot number to measurement
     * @return Floating point expected value of the observable.
     */

    auto expval(const Observable<StateVectorT> &obs, const size_t &num_shots,
                const std::vector<size_t> &shot_range) -> PrecisionT {
        return BaseType::expval(obs, num_shots, shot_range);
    }

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

    /**
     * @brief Variance value for a general Observable
     *
     * @param ob Observable
     * @return Floating point with the variance of the observable.
     */
    auto var(const Observable<StateVectorT> &ob) -> PrecisionT {
        PrecisionT result{};
        if constexpr (std::is_same_v<typename StateVectorT::MemoryStorageT,
                                     MemoryStorageLocation::Internal>) {
            StateVectorT sv(this->_statevector);
            result = calculateObsVar(sv, ob, this->_statevector);

        } else if constexpr (std::is_same_v<
                                 typename StateVectorT::MemoryStorageT,
                                 MemoryStorageLocation::External>) {
            std::vector<ComplexT> data_storage(
                this->_statevector.getData(),
                this->_statevector.getData() + this->_statevector.getLength());
            StateVectorT sv(data_storage.data(), data_storage.size());
            result = calculateObsVar(sv, ob, this->_statevector);
        } else {
            /// LCOV_EXCL_START
            PL_ABORT("Undefined memory storage location for StateVectorT.");
            /// LCOV_EXCL_STOP
        }
        return result;
    }

    /**
     * @brief Variance of an observable.
     *
     * @param operation String with the operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point with the variance of the observable.
     */
    PrecisionT var(const std::string &operation,
                   const std::vector<size_t> &wires) {
        // Copying the original state vector, for the application of the
        // observable operator.
        StateVectorLQubitManaged<PrecisionT> operator_statevector(
            this->_statevector);

        operator_statevector.applyOperation(operation, wires);

        const std::complex<PrecisionT> *opsv_data =
            operator_statevector.getData();
        size_t orgsv_len = this->_statevector.getLength();

        PrecisionT mean_square =
            std::real(innerProdC(opsv_data, opsv_data, orgsv_len));
        PrecisionT squared_mean = std::real(
            innerProdC(this->_statevector.getData(), opsv_data, orgsv_len));
        squared_mean = static_cast<PrecisionT>(std::pow(squared_mean, 2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Variance of an observable.
     *
     * @param matrix Square matrix in row-major order.
     * @param wires Wires where to apply the operator.
     * @return Floating point with the variance of the observable.
     */
    PrecisionT var(const std::vector<ComplexT> &matrix,
                   const std::vector<size_t> &wires) {
        // Copying the original state vector, for the application of the
        // observable operator.
        StateVectorLQubitManaged<PrecisionT> operator_statevector(
            this->_statevector);

        operator_statevector.applyMatrix(matrix, wires);

        const std::complex<PrecisionT> *opsv_data =
            operator_statevector.getData();
        size_t orgsv_len = this->_statevector.getLength();

        PrecisionT mean_square =
            std::real(innerProdC(opsv_data, opsv_data, orgsv_len));
        PrecisionT squared_mean = std::real(
            innerProdC(this->_statevector.getData(), opsv_data, orgsv_len));
        squared_mean = static_cast<PrecisionT>(std::pow(squared_mean, 2));
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
    std::vector<PrecisionT>
    var(const std::vector<op_type> &operations_list,
        const std::vector<std::vector<size_t>> &wires_list) {
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
     * @brief Generate samples using the Metropolis-Hastings method.
     * Reference: Numerical Recipes, NetKet paper
     *
     * @param transition_kernel User-defined functor for producing
     transitions
     * between metropolis states.
     * @param num_burnin Number of Metropolis burn-in steps.
     * @param num_samples The number of samples to generate.
     * @return 1-D vector of samples in binary, each sample is
     * separated by a stride equal to the number of qubits.
     */
    std::vector<size_t>
    generate_samples_metropolis(const std::string &kernelname,
                                size_t num_burnin, size_t num_samples) {
        size_t num_qubits = this->_statevector.getNumQubits();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<PrecisionT> distrib(0.0, 1.0);
        std::vector<size_t> samples(num_samples * num_qubits, 0);
        std::unordered_map<size_t, size_t> cache;

        TransitionKernelType transition_kernel = TransitionKernelType::Local;
        if (kernelname == "NonZeroRandom") {
            transition_kernel = TransitionKernelType::NonZeroRandom;
        }

        auto tk =
            kernel_factory(transition_kernel, this->_statevector.getData(),
                           this->_statevector.getNumQubits());
        size_t idx = 0;

        // Burn In
        for (size_t i = 0; i < num_burnin; i++) {
            idx = metropolis_step(this->_statevector, tk, gen, distrib,
                                  idx); // Burn-in.
        }

        // Sample
        for (size_t i = 0; i < num_samples; i++) {
            idx = metropolis_step(this->_statevector, tk, gen, distrib, idx);

            if (cache.contains(idx)) {
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
        return samples;
    }

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
    PrecisionT var(const index_type *row_map_ptr, const index_type row_map_size,
                   const index_type *entries_ptr, const ComplexT *values_ptr,
                   const index_type numNNZ) {
        PL_ABORT_IF(
            (this->_statevector.getLength() != (size_t(row_map_size) - 1)),
            "Statevector and Hamiltonian have incompatible sizes.");
        auto operator_vector = Util::apply_Sparse_Matrix(
            this->_statevector.getData(),
            static_cast<index_type>(this->_statevector.getLength()),
            row_map_ptr, row_map_size, entries_ptr, values_ptr, numNNZ);

        const PrecisionT mean_square =
            std::real(innerProdC(operator_vector.data(), operator_vector.data(),
                                 operator_vector.size()));
        const auto squared_mean = static_cast<PrecisionT>(
            std::pow(std::real(innerProdC(operator_vector.data(),
                                          this->_statevector.getData(),
                                          operator_vector.size())),
                     2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Generate samples using the alias method.
     * Reference: https://en.wikipedia.org/wiki/Alias_method
     *
     * @param num_samples The number of samples to generate.
     * @return 1-D vector of samples in binary, each sample is
     * separated by a stride equal to the number of qubits.
     */
    std::vector<size_t> generate_samples(size_t num_samples) {
        const size_t num_qubits = this->_statevector.getNumQubits();
        auto &&probabilities = probs();

        std::vector<size_t> samples(num_samples * num_qubits, 0);
        std::mt19937 generator(std::random_device{}());
        std::uniform_real_distribution<PrecisionT> distribution(0.0, 1.0);
        std::unordered_map<size_t, size_t> cache;

        const size_t N = probabilities.size();
        std::vector<double> bucket(N);
        std::vector<size_t> bucket_partner(N);
        std::stack<size_t> overfull_bucket_ids;
        std::stack<size_t> underfull_bucket_ids;

        for (size_t i = 0; i < N; i++) {
            bucket[i] = N * probabilities[i];
            bucket_partner[i] = i;
            if (bucket[i] > 1.0) {
                overfull_bucket_ids.push(i);
            }
            if (bucket[i] < 1.0) {
                underfull_bucket_ids.push(i);
            }
        }

        // Run alias algorithm
        while (!underfull_bucket_ids.empty() && !overfull_bucket_ids.empty()) {
            // get an overfull bucket
            size_t i = overfull_bucket_ids.top();

            // get an underfull bucket
            size_t j = underfull_bucket_ids.top();
            underfull_bucket_ids.pop();

            // underfull bucket is partned with an overfull bucket
            bucket_partner[j] = i;
            bucket[i] = bucket[i] + bucket[j] - 1;

            // if overfull bucket is now underfull
            // put in underfull stack
            if (bucket[i] < 1) {
                overfull_bucket_ids.pop();
                underfull_bucket_ids.push(i);
            }

            // if overfull bucket is full -> remove
            else if (bucket[i] == 1.0) {
                overfull_bucket_ids.pop();
            }
        }

        // Pick samples
        for (size_t i = 0; i < num_samples; i++) {
            PrecisionT pct = distribution(generator) * N;
            auto idx = static_cast<size_t>(pct);
            if (pct - idx > bucket[idx]) {
                idx = bucket_partner[idx];
            }
            // If cached, retrieve sample from cache
            if (cache.contains(idx)) {
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
        return samples;
    }

  private:
    /**
     * @brief Support function that calculates <bra|obs|ket> to obtain the
     * observable's expectation value.
     *
     * @param bra Reference to the statevector where the observable will be
     * applied, must be mutable.
     * @param obs Constant reference to an observable.
     * @param ket Constant reference to the base statevector.
     * @return PrecisionT
     */
    auto inline calculateObsExpval(StateVectorT &bra,
                                   const Observable<StateVectorT> &obs,
                                   const StateVectorT &ket) -> PrecisionT {
        obs.applyInPlace(bra);
        return std::real(
            innerProdC(bra.getData(), ket.getData(), ket.getLength()));
    }

    /**
     * @brief Support function that calculates <bra|obs^2|ket> and
     * (<bra|obs|ket>)^2 to obtain the observable's variance.
     *
     * @param bra Reference to the statevector where the observable will be
     * applied, must be mutable.
     * @param obs Constant reference to an observable.
     * @param ket Constant reference to the base statevector.
     * @return PrecisionT
     */
    auto inline calculateObsVar(StateVectorT &bra,
                                const Observable<StateVectorT> &obs,
                                const StateVectorT &ket) -> PrecisionT {
        obs.applyInPlace(bra);
        PrecisionT mean_square = std::real(
            innerProdC(bra.getData(), bra.getData(), bra.getLength()));
        auto squared_mean = static_cast<PrecisionT>(
            std::pow(std::real(innerProdC(bra.getData(), ket.getData(),
                                          ket.getLength())),
                     2));
        return (mean_square - squared_mean);
    }

    /**
     * @brief Complete a single Metropolis-Hastings step.
     *
     * @param sv state vector
     * @param tk User-defined functor for producing transitions
     * between metropolis states.
     * @param gen Random number generator.
     * @param distrib Random number distribution.
     * @param init_idx Init index of basis state.
     */
    size_t
    metropolis_step(const StateVectorT &sv,
                    const std::unique_ptr<TransitionKernel<PrecisionT>> &tk,
                    std::mt19937 &gen,
                    std::uniform_real_distribution<PrecisionT> &distrib,
                    size_t init_idx) {
        auto init_plog = std::log(
            (sv.getData()[init_idx] * std::conj(sv.getData()[init_idx]))
                .real());

        auto init_qratio = tk->operator()(init_idx);

        // transition kernel outputs these two
        auto &trans_idx = init_qratio.first;
        auto &trans_qratio = init_qratio.second;

        auto trans_plog = std::log(
            (sv.getData()[trans_idx] * std::conj(sv.getData()[trans_idx]))
                .real());

        auto alph = std::min<PrecisionT>(
            1., trans_qratio * std::exp(trans_plog - init_plog));
        auto ran = distrib(gen);

        if (ran < alph) {
            return trans_idx;
        }
        return init_idx;
    }
}; // class Measurements
} // namespace Pennylane::LightningQubit::Measures