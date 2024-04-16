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
#pragma once

#include <cstring>
#include <memory>
#include <stdexcept>
#include <typeinfo>
#include <utility>
#include <vector>

#include "Macros.hpp"
#include "Observables.hpp"
#include "Util.hpp"

// using namespace Pennylane;
/// @cond DEV
namespace {
using Pennylane::Observables::Observable;
} // namespace
/// @endcond

namespace Pennylane::Algorithms {
/**
 * @brief Utility class for encapsulating operations used by AdjointJacobian
 * class.
 */

template <class StateVectorT> class OpsData {
  private:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    size_t num_par_ops_;
    size_t num_nonpar_ops_;
    const std::vector<std::string> ops_name_;
    const std::vector<std::vector<PrecisionT>> ops_params_;
    const std::vector<std::vector<size_t>> ops_wires_;
    const std::vector<bool> ops_inverses_;
    const std::vector<std::vector<ComplexT>> ops_matrices_;
    const std::vector<std::vector<size_t>> ops_controlled_wires_;
    const std::vector<std::vector<bool>> ops_controlled_values_;

  public:
    /**
     * @brief Construct an OpsData object, representing the serialized
     * operations to apply upon the `%StateVector`.
     *
     * @param ops_name Name of each operation to apply.
     * @param ops_params Parameters for a given operation ({} if optional).
     * @param ops_wires Wires upon which to apply operation.
     * @param ops_inverses Value to represent whether given operation is
     * adjoint.
     * @param ops_matrices Numerical representation of given matrix if not
     * supported.
     * @param ops_controlled_wires Control wires
     * @param ops_controlled_wires Control values
     */
    OpsData(std::vector<std::string> ops_name,
            const std::vector<std::vector<PrecisionT>> &ops_params,
            std::vector<std::vector<size_t>> ops_wires,
            std::vector<bool> ops_inverses,
            std::vector<std::vector<ComplexT>> ops_matrices,
            std::vector<std::vector<size_t>> ops_controlled_wires,
            std::vector<std::vector<bool>> ops_controlled_values)
        : num_par_ops_{0}, ops_name_{std::move(ops_name)},
          ops_params_{ops_params}, ops_wires_{std::move(ops_wires)},
          ops_inverses_{std::move(ops_inverses)},
          ops_matrices_{std::move(ops_matrices)},
          ops_controlled_wires_{std::move(ops_controlled_wires)},
          ops_controlled_values_{std::move(ops_controlled_values)} {
        for (const auto &p : ops_params) {
            num_par_ops_ += static_cast<size_t>(!p.empty());
        }
        num_nonpar_ops_ = ops_params.size() - num_par_ops_;
    };

    /**
     * @brief Construct an OpsData object, representing the serialized
     * operations to apply upon the `%StateVector`.
     *
     * @param ops_name Name of each operation to apply.
     * @param ops_params Parameters for a given operation ({} if optional).
     * @param ops_wires Wires upon which to apply operation
     * @param ops_inverses Value to represent whether given operation is
     * adjoint.
     * @param ops_matrices Numerical representation of given matrix if not
     * supported.
     */
    OpsData(std::vector<std::string> ops_name,
            const std::vector<std::vector<PrecisionT>> &ops_params,
            std::vector<std::vector<size_t>> ops_wires,
            std::vector<bool> ops_inverses,
            std::vector<std::vector<ComplexT>> ops_matrices)
        : num_par_ops_{0}, ops_name_{std::move(ops_name)},
          ops_params_{ops_params}, ops_wires_{std::move(ops_wires)},
          ops_inverses_{std::move(ops_inverses)},
          ops_matrices_{std::move(ops_matrices)},
          ops_controlled_wires_(ops_name.size()),
          ops_controlled_values_(ops_name.size()) {
        for (const auto &p : ops_params) {
            num_par_ops_ += static_cast<size_t>(!p.empty());
        }
        num_nonpar_ops_ = ops_params.size() - num_par_ops_;
    };

    /**
     * @brief Construct an OpsData object, representing the serialized
     operations to apply upon the `%StateVector`.
     *
     * @see  OpsData(const std::vector<std::string> &ops_name,
            const std::vector<std::vector<PrecisionT>> &ops_params,
            const std::vector<std::vector<size_t>> &ops_wires,
            const std::vector<bool> &ops_inverses,
            const std::vector<std::vector<ComplexT>> &ops_matrices)
     */
    OpsData(const std::vector<std::string> &ops_name,
            const std::vector<std::vector<PrecisionT>> &ops_params,
            std::vector<std::vector<size_t>> ops_wires,
            std::vector<bool> ops_inverses)
        : num_par_ops_{0}, ops_name_{ops_name}, ops_params_{ops_params},
          ops_wires_{std::move(ops_wires)},
          ops_inverses_{std::move(ops_inverses)},
          ops_matrices_(ops_name.size()),
          ops_controlled_wires_(ops_name.size()),
          ops_controlled_values_(ops_name.size()) {
        for (const auto &p : ops_params) {
            num_par_ops_ += static_cast<size_t>(!p.empty());
        }
        num_nonpar_ops_ = ops_params.size() - num_par_ops_;
    };

    /**
     * @brief Get the number of operations to be applied.
     *
     * @return size_t Number of operations.
     */
    [[nodiscard]] auto getSize() const -> size_t { return ops_name_.size(); }

    /**
     * @brief Get the names of the operations to be applied.
     *
     * @return const std::vector<std::string>&
     */
    [[nodiscard]] auto getOpsName() const -> const std::vector<std::string> & {
        return ops_name_;
    }
    /**
     * @brief Get the (optional) parameters for each operation. Given entries
     * are empty ({}) if not required.
     *
     * @return const std::vector<std::vector<PrecisionT>>&
     */
    [[nodiscard]] auto getOpsParams() const
        -> const std::vector<std::vector<PrecisionT>> & {
        return ops_params_;
    }
    /**
     * @brief Get the wires for each operation.
     *
     * @return const std::vector<std::vector<size_t>>&
     */
    [[nodiscard]] auto getOpsWires() const
        -> const std::vector<std::vector<size_t>> & {
        return ops_wires_;
    }
    /**
     * @brief Get the controlled wires for each operation.
     *
     * @return const std::vector<std::vector<size_t>>&
     */
    [[nodiscard]] auto getOpsControlledWires() const
        -> const std::vector<std::vector<size_t>> & {
        return ops_controlled_wires_;
    }
    /**
     * @brief Get the controlled values for each operation.
     *
     * @return const std::vector<std::vector<size_t>>&
     */
    [[nodiscard]] auto getOpsControlledValues() const
        -> const std::vector<std::vector<bool>> & {
        return ops_controlled_values_;
    }
    /**
     * @brief Get the adjoint flag for each operation.
     *
     * @return const std::vector<bool>&
     */
    [[nodiscard]] auto getOpsInverses() const -> const std::vector<bool> & {
        return ops_inverses_;
    }
    /**
     * @brief Get the numerical matrix for a given unsupported operation. Given
     * entries are empty ({}) if not required.
     *
     * @return const std::vector<std::vector<ComplexT>>&
     */
    [[nodiscard]] auto getOpsMatrices() const
        -> const std::vector<std::vector<ComplexT>> & {
        return ops_matrices_;
    }

    /**
     * @brief Notify if the operation at a given index is parametric.
     *
     * @param index Operation index.
     * @return true Gate is parametric (has parameters).
     * @return false Gate in non-parametric.
     */
    [[nodiscard]] inline auto hasParams(size_t index) const -> bool {
        return !ops_params_[index].empty();
    }

    /**
     * @brief Get the number of parametric operations.
     *
     * @return size_t
     */
    [[nodiscard]] auto getNumParOps() const -> size_t { return num_par_ops_; }

    /**
     * @brief Get the number of non-parametric ops.
     *
     * @return size_t
     */
    [[nodiscard]] auto getNumNonParOps() const -> size_t {
        return num_nonpar_ops_;
    }

    /**
     * @brief Get total number of parameters.
     */
    [[nodiscard]] auto getTotalNumParams() const -> size_t {
        return std::accumulate(
            ops_params_.begin(), ops_params_.end(), size_t{0U},
            [](size_t acc, auto &params) { return acc + params.size(); });
    }
};

/**
 * @brief Represent the serialized data of a QuantumTape to differentiate
 *
 * @tparam StateVectorT
 */
template <class StateVectorT> class JacobianData {
  private:
    using CFP_t = typename StateVectorT::CFP_t;
    size_t num_parameters; /**< Number of parameters in the tape */
    size_t num_elements;   /**< Length of the statevector data */
    const CFP_t *psi;      /**< Pointer to the statevector data */

    /**
     * @var observables
     * Observables for which to calculate Jacobian.
     */
    const std::vector<std::shared_ptr<Observable<StateVectorT>>> observables;

    /**
     * @var operations
     * operations Operations used to create given state.
     */
    const OpsData<StateVectorT> operations;

    /* @var trainableParams      */
    const std::vector<size_t> trainableParams;

  public:
    JacobianData(const JacobianData &) = default;
    JacobianData(JacobianData &&) noexcept = default;
    JacobianData &operator=(const JacobianData &) = default;
    JacobianData &operator=(JacobianData &&) noexcept = default;
    virtual ~JacobianData() = default;

    /**
     * @brief Construct a JacobianData object
     *
     * @param num_params Number of parameters in the Tape.
     * @param num_elem Length of the statevector data.
     * @param ps Pointer to the statevector data.
     * @param obs Observables for which to calculate Jacobian.
     * @param ops Operations used to create given state.
     * @param trainP Sorted list of parameters participating in Jacobian
     * computation.
     *
     * @rst
     * Each value :math:`i` in trainable params means that
     * we want to take a derivative respect to the :math:`i`-th operation.
     *
     * Further note that ``ops`` does not contain state preparation operations
     * (e.g. StatePrep) or Hamiltonian coefficients.
     * @endrst
     */
    JacobianData(size_t num_params, size_t num_elem, const CFP_t *sv_ptr,
                 std::vector<std::shared_ptr<Observable<StateVectorT>>> obs,
                 OpsData<StateVectorT> ops, std::vector<size_t> trainP)
        : num_parameters(num_params), num_elements(num_elem), psi(sv_ptr),
          observables(std::move(obs)), operations(std::move(ops)),
          trainableParams(std::move(trainP)) {
        /* When the Hamiltonian has parameters, trainable parameters include
         * these. We explicitly ignore them. */
    }

    /**
     * @brief Get Number of parameters in the Tape.
     *
     * @return size_t
     */
    [[nodiscard]] auto getNumParams() const -> size_t { return num_parameters; }

    /**
     * @brief Get the length of the statevector data.
     *
     * @return size_t
     */
    [[nodiscard]] auto getSizeStateVec() const -> size_t {
        return num_elements;
    }

    /**
     * @brief Get the pointer to the statevector data.
     *
     * @return CFP_t *
     */
    [[nodiscard]] auto getPtrStateVec() const -> const CFP_t * { return psi; }

    /**
     * @brief Get observables for which to calculate Jacobian.
     *
     * @return List of observables
     */
    [[nodiscard]] auto getObservables() const
        -> const std::vector<std::shared_ptr<Observable<StateVectorT>>> & {
        return observables;
    }

    /**
     * @brief Get the number of observables for which to calculate
     * Jacobian.
     *
     * @return size_t
     */
    [[nodiscard]] auto getNumObservables() const -> size_t {
        return observables.size();
    }

    /**
     * @brief Get operations used to create given state.
     *
     * @return OpsData<StateVectorT>&
     */
    [[nodiscard]] auto getOperations() const -> const OpsData<StateVectorT> & {
        return operations;
    }

    /**
     * @brief Get list of parameters participating in Jacobian
     * calculation.
     *
     * @return std::vector<size_t>&
     */
    [[nodiscard]] auto getTrainableParams() const
        -> const std::vector<size_t> & {
        return trainableParams;
    }

    /**
     * @brief Get if the number of parameters participating in Jacobian
     * calculation is zero.
     *
     * @return true If it has trainable parameters; false otherwise.
     */
    [[nodiscard]] auto hasTrainableParams() const -> bool {
        return !trainableParams.empty();
    }
};
} // namespace Pennylane::Algorithms
