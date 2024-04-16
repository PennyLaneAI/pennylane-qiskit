// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

namespace Pennylane::LightningKokkos::Functors {
/**
 * @brief Compute probability distribution from StateVector.
 *
 * @param arr_ StateVector data.
 * @param probabilities_ Discrete probability distribution.
 */
template <class PrecisionT> struct getProbFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> arr;
    Kokkos::View<PrecisionT *> probability;

    getProbFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> arr_,
                   Kokkos::View<PrecisionT *> probability_)
        : arr(arr_), probability(probability_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        const PrecisionT REAL = arr(k).real();
        const PrecisionT IMAG = arr(k).imag();
        probability(k) = REAL * REAL + IMAG * IMAG;
    }
};

/**
 *@brief Sampling using Random_XorShift64_Pool
 *
 * @param samples_ Kokkos::View of the generated samples.
 * @param cdf_  Kokkos::View of cumulative probability distribution.
 * @param rand_pool_ The generatorPool.
 * @param num_qubits_ Number of qubits.
 * @param length_ Length of cumulative probability distribution.
 */

template <class PrecisionT, template <class ExecutionSpace> class GeneratorPool,
          class ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct Sampler {
    Kokkos::View<size_t *> samples;
    Kokkos::View<PrecisionT *> cdf;
    GeneratorPool<ExecutionSpace> rand_pool;

    const size_t num_qubits;
    const size_t length;

    Sampler(Kokkos::View<size_t *> samples_, Kokkos::View<PrecisionT *> cdf_,
            GeneratorPool<ExecutionSpace> rand_pool_, const size_t num_qubits_,
            const size_t length_)
        : samples(samples_), cdf(cdf_), rand_pool(rand_pool_),
          num_qubits(num_qubits_), length(length_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k) const {
        // Get a random number state from the pool for the active thread
        auto rand_gen = rand_pool.get_state();
        PrecisionT U_rand = rand_gen.drand(0.0, 1.0);
        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
        size_t index;

        // Binary search for the bin index of cumulative probability
        // distribution that generated random number U falls into.
        if (U_rand <= cdf(1)) {
            index = 0;
        } else {
            size_t low_idx = 1, high_idx = length;
            size_t mid_idx;
            PrecisionT cdf_t;
            while (high_idx - low_idx > 1) {
                mid_idx = high_idx - ((high_idx - low_idx) >> 1U);
                if (mid_idx == length)
                    cdf_t = 1;
                else
                    cdf_t = cdf(mid_idx);
                if (cdf_t < U_rand)
                    low_idx = mid_idx;
                else
                    high_idx = mid_idx;
            }
            index = high_idx - 1;
        }
        for (size_t j = 0; j < num_qubits; j++) {
            samples(k * num_qubits + (num_qubits - 1 - j)) = (index >> j) & 1U;
        }
    }
};

/**
 * @brief Determines the transposed index of a tensor stored linearly.
 *  This function assumes each axis will have a length of 2 (|0>, |1>).
 *
 * @param sorted_ind_wires Data of indices for transposition.
 * @param trans_index Data of indices after transposition.
 * @param max_index_sorted_ind_wires_ Length of sorted_ind_wires.
 */
struct getTransposedIndexFunctor {
    Kokkos::View<size_t *> sorted_ind_wires;
    Kokkos::View<size_t *> trans_index;
    const size_t max_index_sorted_ind_wires;
    getTransposedIndexFunctor(Kokkos::View<size_t *> sorted_ind_wires_,
                              Kokkos::View<size_t *> trans_index_,
                              const int length_sorted_ind_wires_)
        : sorted_ind_wires(sorted_ind_wires_), trans_index(trans_index_),
          max_index_sorted_ind_wires(length_sorted_ind_wires_ - 1) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i, const size_t j) const {
        const size_t axis = sorted_ind_wires(j);
        const size_t index = i / (1L << (max_index_sorted_ind_wires - j));
        const size_t sub_index = (index % 2)
                                 << (max_index_sorted_ind_wires - axis);
        Kokkos::atomic_add(&trans_index(i), sub_index);
    }
};

/**
 * @brief Template for the transposition of state tensors,
 * axes are assumed to have a length of 2 (|0>, |1>).
 *
 * @tparam T Tensor data type.
 * @param tensor Tensor to be transposed.
 * @param new_axes new axes distribution.
 * @return Transposed Tensor.
 */
template <class PrecisionT> struct getTransposedFunctor {
    Kokkos::View<PrecisionT *> transProb;
    Kokkos::View<PrecisionT *> probability;
    Kokkos::View<size_t *> trans_index;
    getTransposedFunctor(Kokkos::View<PrecisionT *> transProb_,
                         Kokkos::View<PrecisionT *> probability_,
                         Kokkos::View<size_t *> trans_index_)
        : transProb(transProb_), probability(probability_),
          trans_index(trans_index_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i) const {
        const size_t new_index = trans_index(i);
        transProb(i) = probability(new_index);
    }
};

} // namespace Pennylane::LightningKokkos::Functors
