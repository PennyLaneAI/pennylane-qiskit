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
 * @file TransitionKernels.hpp
 * @brief Defines transition kernels classes for Markov chain Monte Carlo (MCMC)
 * sampling.
 *
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

#include "Util.hpp" // exp2

namespace Pennylane::LightningQubit::Measures {
enum class TransitionKernelType { Local, NonZeroRandom };

/**
 * @brief Parent class to define interface for Transition Kernel
 *
 * @tparam fp_t Floating point precision of underlying measurements.
 */
template <typename fp_t> class TransitionKernel {
  protected:
    TransitionKernel() = default;
    TransitionKernel(const TransitionKernel &) = default;
    TransitionKernel(TransitionKernel &&) noexcept = default;
    TransitionKernel &operator=(const TransitionKernel &) = default;
    TransitionKernel &operator=(TransitionKernel &&) noexcept = default;

  public:
    //  outputs the next state and the qratio
    virtual std::pair<size_t, fp_t> operator()(size_t) = 0;
    virtual ~TransitionKernel() = default;
};

/**
 * @brief Transition Kernel for a 'SpinFlip' local transition between states
 *
 * This class implements a local transition kernel for a spin flip operation.
 * It goes about this by generating a random qubit site and then generating
 * a random number to determine the new bit at that qubit site.
 * @tparam fp_t Floating point precision of underlying measurements.
 */
template <typename fp_t>
class LocalTransitionKernel : public TransitionKernel<fp_t> {
  private:
    size_t num_qubits_;
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_int_distribution<size_t> distrib_num_qubits_;
    std::uniform_int_distribution<size_t> distrib_binary_;

  public:
    explicit LocalTransitionKernel(size_t num_qubits)
        : num_qubits_(num_qubits), gen_(std::mt19937(rd_())),
          distrib_num_qubits_(
              std::uniform_int_distribution<size_t>(0, num_qubits - 1)),
          distrib_binary_(std::uniform_int_distribution<size_t>(0, 1)) {}

    std::pair<size_t, fp_t> operator()(size_t init_idx) final {
        size_t qubit_site = distrib_num_qubits_(gen_);
        size_t qubit_value = distrib_binary_(gen_);
        size_t current_bit = (static_cast<unsigned>(init_idx) >>
                              static_cast<unsigned>(qubit_site)) &
                             1U;

        if (qubit_value == current_bit) {
            return std::pair<size_t, fp_t>(init_idx, 1);
        }
        if (current_bit == 0) {
            return std::pair<size_t, fp_t>(init_idx + std::pow(2, qubit_site),
                                           1);
        }
        return std::pair<size_t, fp_t>(init_idx - std::pow(2, qubit_site), 1);
    }
};

/**
 * @brief Transition Kernel for a random transition between non-zero states
 *
 * This class randomly transitions between states that have nonzero probability.
 * To determine the states with non-zero probability we have O(2^num_qubits)
 * overhead. Despite this, this method is still fast. This transition kernel
 * can sample even GHZ states.
 */
template <typename fp_t>
class NonZeroRandomTransitionKernel : public TransitionKernel<fp_t> {
  private:
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_int_distribution<size_t> distrib_;
    size_t sv_length_;
    std::vector<size_t> non_zeros_;

  public:
    NonZeroRandomTransitionKernel(const std::complex<fp_t> *sv,
                                  size_t sv_length, fp_t min_error) {
        auto data = sv;
        sv_length_ = sv_length;
        // find nonzero candidates
        for (size_t i = 0; i < sv_length_; i++) {
            if (std::abs(data[i]) > min_error) {
                non_zeros_.push_back(i);
            }
        }
        gen_ = std::mt19937(rd_());
        distrib_ =
            std::uniform_int_distribution<size_t>(0, non_zeros_.size() - 1);
    }
    std::pair<size_t, fp_t> operator()([[maybe_unused]] size_t init_idx) final {
        auto trans_idx = distrib_(gen_);
        return std::pair<size_t, fp_t>(non_zeros_[trans_idx], 1);
    }
};

/**
 * @brief Factory function to create a transition kernel
 *
 * @param kernel_type Type of transition kernel to create
 * @param sv pointer to the statevector data
 * @param num_qubits number of qubits
 * @tparam fp_t Floating point precision of underlying measurements.
 * @return std::unique_ptr of the transition kernel
 */
template <typename fp_t>
std::unique_ptr<TransitionKernel<fp_t>>
kernel_factory(const TransitionKernelType kernel_type,
               const std::complex<fp_t> *sv, size_t num_qubits) {
    auto sv_length = Pennylane::Util::exp2(num_qubits);
    if (kernel_type == TransitionKernelType::Local) {
        return std::unique_ptr<TransitionKernel<fp_t>>(
            new NonZeroRandomTransitionKernel<fp_t>(
                sv, sv_length, std::numeric_limits<fp_t>::epsilon()));
    }
    return std::unique_ptr<TransitionKernel<fp_t>>(
        new LocalTransitionKernel<fp_t>(num_qubits));
}
} // namespace Pennylane::LightningQubit::Measures
