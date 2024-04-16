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
 * @file MeasurementsBase.hpp
 * Defines the Measurements CRTP base class.
 */
#pragma once

#include <string>
#include <vector>

#include "Observables.hpp"

#include "CPUMemoryModel.hpp"

#include "Util.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Observables;
using namespace Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane::Measures {
/**
 * @brief Observable's Measurement Class.
 *
 * This class performs measurements in the state vector provided to its
 * constructor. Observables are defined by its operator(matrix), the observable
 * class, or through a string-based function dispatch.
 *
 * @tparam StateVectorT
 * @tparam Derived
 */
template <class StateVectorT, class Derived> class MeasurementsBase {
  private:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

  protected:
#ifdef _ENABLE_PLGPU
    StateVectorT &_statevector;
#else
    const StateVectorT &_statevector;
#endif

  public:
#ifdef _ENABLE_PLGPU
    explicit MeasurementsBase(StateVectorT &statevector)
        : _statevector{statevector} {};
#else
    explicit MeasurementsBase(const StateVectorT &statevector)
        : _statevector{statevector} {};
#endif

    /**
     * @brief Calculate the expectation value for a general Observable.
     *
     * @param obs Observable.
     * @return Expectation value with respect to the given observable.
     */
    auto expval(const Observable<StateVectorT> &obs) -> PrecisionT {
        return static_cast<Derived *>(this)->expval(obs);
    }

    /**
     * @brief Calculate the variance for a general Observable.
     *
     * @param obs Observable.
     * @return Variance with respect to the given observable.
     */
    auto var(const Observable<StateVectorT> &obs) -> PrecisionT {
        return static_cast<Derived *>(this)->var(obs);
    }

    /**
     * @brief Probabilities of each computational basis state.
     *
     * @return Floating point std::vector with probabilities
     * in lexicographic order.
     */
    auto probs() -> std::vector<PrecisionT> {
        return static_cast<Derived *>(this)->probs();
    };

    /**
     * @brief Probabilities for a subset of the full system.
     *
     * @param wires Wires will restrict probabilities to a subset
     * of the full system.
     * @return Floating point std::vector with probabilities.
     * The basis columns are rearranged according to wires.
     */
    auto probs(const std::vector<size_t> &wires) -> std::vector<PrecisionT> {
        return static_cast<Derived *>(this)->probs(wires);
    };

    /**
     * @brief  Generate samples
     *
     * @param num_samples Number of samples
     * @return 1-D vector of samples in binary with each sample
     * separated by a stride equal to the number of qubits.
     */
    auto generate_samples(size_t num_samples) -> std::vector<size_t> {
        return static_cast<Derived *>(this)->generate_samples(num_samples);
    };

    /**
     * @brief Calculate the expectation value for a general Observable.
     *
     * @param obs Observable.
     * @param num_shots Number of shots used to generate samples
     * @param shot_range The range of samples to use. All samples are used
     * by default.
     *
     * @return Expectation value with respect to the given observable.
     */
    auto expval(const Observable<StateVectorT> &obs, const size_t &num_shots,
                const std::vector<size_t> &shot_range = {}) -> PrecisionT {
        PrecisionT result{0.0};

        if (obs.getObsName().find("SparseHamiltonian") != std::string::npos) {
            // SparseHamiltonian does not support shot measurement in pennylane.
            PL_ABORT("SparseHamiltonian observables do not support shot "
                     "measurement.");
        } else if (obs.getObsName().find("Hamiltonian") != std::string::npos) {
            auto coeffs = obs.getCoeffs();
            auto obsTerms = obs.getObs();
            for (size_t obs_term_idx = 0; obs_term_idx < coeffs.size();
                 obs_term_idx++) {
                result += coeffs[obs_term_idx] * expval(*obsTerms[obs_term_idx],
                                                        num_shots, shot_range);
            }
        } else {
            auto obs_samples = measure_with_samples(obs, num_shots, shot_range);
            result =
                std::accumulate(obs_samples.begin(), obs_samples.end(), 0.0);
            result /= obs_samples.size();
        }
        return result;
    }

    /**
     * @brief Calculate the expectation value for a general Observable.
     *
     * @param obs Observable.
     * @param num_shots Number of shots used to generate samples
     * @param shot_range The range of samples to use. All samples are used
     * by default.
     *
     * @return Expectation value with respect to the given observable.
     */
    auto measure_with_samples(const Observable<StateVectorT> &obs,
                              const size_t &num_shots,
                              const std::vector<size_t> &shot_range)
        -> std::vector<PrecisionT> {
        const size_t num_qubits = _statevector.getTotalNumQubits();
        std::vector<size_t> obs_wires;
        std::vector<std::vector<PrecisionT>> eigenValues;

        auto sub_samples =
            _sample_state(obs, num_shots, shot_range, obs_wires, eigenValues);

        size_t num_samples = shot_range.empty() ? num_shots : shot_range.size();

        std::vector<PrecisionT> obs_samples(num_samples, 0);

        std::vector<PrecisionT> eigenVals = eigenValues[0];

        for (size_t i = 1; i < eigenValues.size(); i++) {
            eigenVals = kronProd(eigenVals, eigenValues[i]);
        }

        for (size_t i = 0; i < num_samples; i++) {
            size_t idx = 0;
            size_t wire_idx = 0;
            for (auto &obs_wire : obs_wires) {
                idx += sub_samples[i * num_qubits + obs_wire]
                       << (obs_wires.size() - 1 - wire_idx);
                wire_idx++;
            }

            obs_samples[i] = eigenVals[idx];
        }
        return obs_samples;
    }

    /**
     * @brief Calculate the variance for an observable with the number of shots.
     *
     * @param obs An observable object.
     * @param num_shots Number of shots used to generate samples
     *
     * @return Variance of the given observable.
     */
    auto var(const Observable<StateVectorT> &obs, const size_t &num_shots)
        -> PrecisionT {
        PrecisionT result{0.0};
        if (obs.getObsName().find("SparseHamiltonian") != std::string::npos) {
            // SparseHamiltonian does not support shot measurement in pennylane.
            PL_ABORT("SparseHamiltonian observables do not support shot "
                     "measurement.");
        } else if (obs.getObsName().find("Hamiltonian") != std::string::npos) {
            // Branch for Hamiltonian observables
            auto coeffs = obs.getCoeffs();
            auto obs_terms = obs.getObs();

            size_t obs_term_idx = 0;
            for (const auto &coeff : coeffs) {
                result +=
                    coeff * coeff * var(*obs_terms[obs_term_idx], num_shots);
                obs_term_idx++;
            }
        } else {
            auto obs_samples = measure_with_samples(obs, num_shots, {});
            auto square_mean =
                std::accumulate(obs_samples.begin(), obs_samples.end(), 0.0) /
                obs_samples.size();
            auto mean_square =
                std::accumulate(obs_samples.begin(), obs_samples.end(), 0.0,
                                [](PrecisionT acc, PrecisionT element) {
                                    return acc + element * element;
                                }) /
                obs_samples.size();
            result = mean_square - square_mean * square_mean;
        }
        return result;
    }

    /**
     * @brief Probabilities to measure rotated basis states.
     *
     * @param obs An observable object.
     * @param num_shots Number of shots (Optional). If specified with a non-zero
     * number, shot-noise will be added to return probabilities
     *
     * @return Floating point std::vector with probabilities.
     * The basis columns are rearranged according to wires.
     */
    auto probs(const Observable<StateVectorT> &obs, size_t num_shots = 0)
        -> std::vector<PrecisionT> {
        PL_ABORT_IF(
            obs.getObsName().find("Hamiltonian") != std::string::npos,
            "Hamiltonian and Sparse Hamiltonian do not support samples().");
        std::vector<size_t> obs_wires;
        std::vector<std::vector<PrecisionT>> eigenvalues;
        if constexpr (std::is_same_v<
                          typename StateVectorT::MemoryStorageT,
                          Pennylane::Util::MemoryStorageLocation::External>) {
            std::vector<ComplexT> data_storage(
                this->_statevector.getData(),
                this->_statevector.getData() + this->_statevector.getLength());
            StateVectorT sv(data_storage.data(), data_storage.size());
            sv.updateData(data_storage.data(), data_storage.size());
            obs.applyInPlaceShots(sv, eigenvalues, obs_wires);
            Derived measure(sv);
            if (num_shots > size_t{0}) {
                return measure.probs(obs_wires, num_shots);
            }
            return measure.probs(obs_wires);
        } else {
            StateVectorT sv(_statevector);
            obs.applyInPlaceShots(sv, eigenvalues, obs_wires);
            Derived measure(sv);
            if (num_shots > size_t{0}) {
                return measure.probs(obs_wires, num_shots);
            }
            return measure.probs(obs_wires);
        }
    }

    /**
     * @brief Probabilities with shot-noise for a subset of the full system.
     *
     * @param wires Wires will restrict probabilities to a subset
     * of the full system.
     * @param num_shots Number of shots.
     *
     * @return Floating point std::vector with probabilities.
     */
    auto probs(const std::vector<size_t> &wires, size_t num_shots)
        -> std::vector<PrecisionT> {
        auto counts_map = counts(num_shots);

        size_t num_wires = _statevector.getTotalNumQubits();

        std::vector<PrecisionT> prob_shots(size_t{1} << wires.size(), 0.0);

        for (auto &it : counts_map) {
            size_t bitVal = 0;
            for (size_t bit = 0; bit < wires.size(); bit++) {
                // Mapping the value of wires[bit]th bit to local [bit]th bit of
                // the output
                bitVal += ((it.first >> (num_wires - size_t{1} - wires[bit])) &
                           size_t{1})
                          << (wires.size() - size_t{1} - bit);
            }

            prob_shots[bitVal] +=
                it.second / static_cast<PrecisionT>(num_shots);
        }

        return prob_shots;
    }

    /**
     * @brief Probabilities with shot-noise.
     *
     * @param num_shots Number of shots.
     *
     * @return Floating point std::vector with probabilities.
     */
    auto probs(size_t num_shots) -> std::vector<PrecisionT> {
        auto counts_map = counts(num_shots);

        size_t num_wires = _statevector.getTotalNumQubits();

        std::vector<PrecisionT> prob_shots(size_t{1} << num_wires, 0.0);

        for (auto &it : counts_map) {
            prob_shots[it.first] =
                it.second / static_cast<PrecisionT>(num_shots);
        }

        return prob_shots;
    }

    /**
     * @brief Return samples drawn from eigenvalues of the observable
     *
     * @param obs The observable object to sample
     * @param num_shots Number of shots used to generate samples
     *
     * @return Samples of eigenvalues of the observable
     */
    auto sample(const Observable<StateVectorT> &obs, const size_t &num_shots)
        -> std::vector<PrecisionT> {
        PL_ABORT_IF(
            obs.getObsName().find("Hamiltonian") != std::string::npos,
            "Hamiltonian and Sparse Hamiltonian do not support samples().");
        std::vector<size_t> obs_wires;
        std::vector<size_t> shot_range = {};

        return measure_with_samples(obs, num_shots, shot_range);
    }

    /**
     * @brief Return the raw basis state samples
     *
     * @param num_shots Number of shots used to generate samples
     *
     * @return Raw basis state samples
     */
    auto sample(const size_t &num_shots) -> std::vector<size_t> {
        Derived measure(_statevector);
        return measure.generate_samples(num_shots);
    }

    /**
     * @brief Groups the eigenvalues of samples into a dictionary showing
     * number of occurences for each possible outcome with the number of shots.
     *
     * @param obs The observable to sample
     * @param num_shots Number of wires the sampled observable was performed on
     *
     * @return std::unordered_map<PrecisionT, size_t> with format
     * ``{'EigenValue': num_occurences}``
     */
    auto counts(const Observable<StateVectorT> &obs, const size_t &num_shots)
        -> std::unordered_map<PrecisionT, size_t> {
        std::unordered_map<PrecisionT, size_t> outcome_map;
        auto sample_data = sample(obs, num_shots);
        for (size_t i = 0; i < num_shots; i++) {
            auto key = sample_data[i];
            auto it = outcome_map.find(key);
            if (it != outcome_map.end()) {
                it->second += 1;
            } else {
                outcome_map[key] = 1;
            }
        }
        return outcome_map;
    }

    /**
     * @brief Groups the samples into a dictionary showing number of occurences
     * for each possible outcome with the number of shots.
     *
     * @param num_shots Number of wires the sampled observable was performed on
     *
     * @return std::unordered_map<size_t, size_t> with format ``{'outcome':
     * num_occurences}``
     */
    auto counts(const size_t &num_shots) -> std::unordered_map<size_t, size_t> {
        std::unordered_map<size_t, size_t> outcome_map;
        auto sample_data = sample(num_shots);

        size_t num_wires = _statevector.getTotalNumQubits();
        for (size_t i = 0; i < num_shots; i++) {
            size_t key = 0;
            for (size_t j = 0; j < num_wires; j++) {
                key += sample_data[i * num_wires + j] << (num_wires - 1 - j);
            }

            auto it = outcome_map.find(key);
            if (it != outcome_map.end()) {
                it->second += 1;
            } else {
                outcome_map[key] = 1;
            }
        }
        return outcome_map;
    }

  private:
    /**
     * @brief Return samples of a observable
     *
     * @param obs The observable to sample
     * @param num_shots Number of shots used to generate samples
     * @param shot_range The range of samples to use. All samples are used by
     * default.
     * @param obs_wires Observable wires.
     * @param eigenValues eigenvalues of the observable.
     *
     * @return std::vector<size_t> samples in std::vector
     */
    auto _sample_state(const Observable<StateVectorT> &obs,
                       const size_t &num_shots,
                       const std::vector<size_t> &shot_range,
                       std::vector<size_t> &obs_wires,
                       std::vector<std::vector<PrecisionT>> &eigenValues)
        -> std::vector<size_t> {
        const size_t num_qubits = _statevector.getTotalNumQubits();
        std::vector<size_t> samples;
        if constexpr (std::is_same_v<
                          typename StateVectorT::MemoryStorageT,
                          Pennylane::Util::MemoryStorageLocation::External>) {
            std::vector<ComplexT> data_storage(
                this->_statevector.getData(),
                this->_statevector.getData() + this->_statevector.getLength());
            StateVectorT sv(data_storage.data(), data_storage.size());
            obs.applyInPlaceShots(sv, eigenValues, obs_wires);
            Derived measure(sv);
            samples = measure.generate_samples(num_shots);
        } else {
            StateVectorT sv(_statevector);
            obs.applyInPlaceShots(sv, eigenValues, obs_wires);
            Derived measure(sv);
            samples = measure.generate_samples(num_shots);
        }

        if (!shot_range.empty()) {
            std::vector<size_t> sub_samples(shot_range.size() * num_qubits);
            // Get a slice of samples based on the shot_range vector
            size_t shot_idx = 0;
            for (const auto &i : shot_range) {
                for (size_t j = i * num_qubits; j < (i + 1) * num_qubits; j++) {
                    // TODO some extra work to make it cache-friendly
                    sub_samples[shot_idx * num_qubits + j - i * num_qubits] =
                        samples[j];
                }
                shot_idx++;
            }
            return sub_samples;
        }
        return samples;
    }
};
} // namespace Pennylane::Measures
