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
 * Minimal class for the Lightning qubit state vector interfacing with the
 * dynamic dispatcher and threading functionalities. This class is a bridge
 * between the base (agnostic) class and specializations for distinct data
 * storage types.
 */

#pragma once
#include <complex>
#include <unordered_map>

#include "CPUMemoryModel.hpp"
#include "GateOperation.hpp"
#include "KernelMap.hpp"
#include "KernelType.hpp"
#include "StateVectorBase.hpp"
#include "Threading.hpp"

/// @cond DEV
namespace {
using Pennylane::LightningQubit::Util::Threading;
using Pennylane::Util::CPUMemoryModel;
using Pennylane::Util::exp2;
using namespace Pennylane::LightningQubit::Gates;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit {
/**
 * @brief Lightning qubit state vector class.
 *
 * Minimal class, without data storage, for the Lightning qubit state vector.
 * This class interfaces with the dynamic dispatcher and threading
 * functionalities and is a bridge between the base (agnostic) class and
 * specializations for distinct data storage types.
 *
 * @tparam PrecisionT Floating point precision of underlying state vector data.
 * @tparam Derived Derived class for CRTP.
 */
template <class PrecisionT, class Derived>
class StateVectorLQubit : public StateVectorBase<PrecisionT, Derived> {
  public:
    using ComplexT = std::complex<PrecisionT>;
    using MemoryStorageT = Pennylane::Util::MemoryStorageLocation::Undefined;

  protected:
    const Threading threading_;
    const CPUMemoryModel memory_model_;

  private:
    using BaseType = StateVectorBase<PrecisionT, Derived>;
    using GateKernelMap = std::unordered_map<GateOperation, KernelType>;
    using GeneratorKernelMap =
        std::unordered_map<GeneratorOperation, KernelType>;
    using MatrixKernelMap = std::unordered_map<MatrixOperation, KernelType>;
    using ControlledGateKernelMap =
        std::unordered_map<ControlledGateOperation, KernelType>;
    using ControlledGeneratorKernelMap =
        std::unordered_map<ControlledGeneratorOperation, KernelType>;
    using ControlledMatrixKernelMap =
        std::unordered_map<ControlledMatrixOperation, KernelType>;

    GateKernelMap kernel_for_gates_;
    GeneratorKernelMap kernel_for_generators_;
    MatrixKernelMap kernel_for_matrices_;
    ControlledGateKernelMap kernel_for_controlled_gates_;
    ControlledGeneratorKernelMap kernel_for_controlled_generators_;
    ControlledMatrixKernelMap kernel_for_controlled_matrices_;

    /**
     * @brief Internal function to set kernels for all operations depending on
     * provided dispatch options.
     *
     * @param num_qubits Number of qubits of the statevector
     * @param threading Threading option
     * @param memory_model Memory model
     */
    void setKernels(size_t num_qubits, Threading threading,
                    CPUMemoryModel memory_model) {
        using KernelMap::OperationKernelMap;
        kernel_for_gates_ =
            OperationKernelMap<GateOperation>::getInstance().getKernelMap(
                num_qubits, threading, memory_model);
        kernel_for_generators_ =
            OperationKernelMap<GeneratorOperation>::getInstance().getKernelMap(
                num_qubits, threading, memory_model);
        kernel_for_matrices_ =
            OperationKernelMap<MatrixOperation>::getInstance().getKernelMap(
                num_qubits, threading, memory_model);
        kernel_for_controlled_gates_ =
            OperationKernelMap<ControlledGateOperation>::getInstance()
                .getKernelMap(num_qubits, threading, memory_model);
        kernel_for_controlled_generators_ =
            OperationKernelMap<ControlledGeneratorOperation>::getInstance()
                .getKernelMap(num_qubits, threading, memory_model);
        kernel_for_controlled_matrices_ =
            OperationKernelMap<ControlledMatrixOperation>::getInstance()
                .getKernelMap(num_qubits, threading, memory_model);
    }

    /**
     * @brief Get a kernel for a gate operation.
     *
     * @param gate_op Gate operation
     * @return KernelType
     */
    [[nodiscard]] inline auto getKernelForGate(GateOperation gate_op) const
        -> KernelType {
        return kernel_for_gates_.at(gate_op);
    }

    /**
     * @brief Get a kernel for a controlled gate operation.
     *
     * @param gate_op Gate operation
     * @return KernelType
     */
    [[nodiscard]] inline auto
    getKernelForControlledGate(ControlledGateOperation gate_op) const
        -> KernelType {
        return kernel_for_controlled_gates_.at(gate_op);
    }

    /**
     * @brief Get a kernel for a generator operation.
     *
     * @param gen_op Generator operation
     * @return KernelType
     */
    [[nodiscard]] inline auto
    getKernelForGenerator(GeneratorOperation gen_op) const -> KernelType {
        return kernel_for_generators_.at(gen_op);
    }

    /**
     * @brief Get a kernel for a controlled generator operation.
     *
     * @param gen_op Generator operation
     * @return KernelType
     */
    [[nodiscard]] inline auto
    getKernelForControlledGenerator(ControlledGeneratorOperation gen_op) const
        -> KernelType {
        return kernel_for_controlled_generators_.at(gen_op);
    }

    /**
     * @brief Get a kernel for a matrix operation.
     *
     * @param mat_op Matrix operation
     * @return KernelType
     */
    [[nodiscard]] inline auto getKernelForMatrix(MatrixOperation mat_op) const
        -> KernelType {
        return kernel_for_matrices_.at(mat_op);
    }

    /**
     * @brief Get a kernel for a controlled matrix operation.
     *
     * @param mat_op Controlled matrix operation
     * @return KernelType
     */
    [[nodiscard]] inline auto
    getKernelForControlledMatrix(ControlledMatrixOperation mat_op) const
        -> KernelType {
        return kernel_for_controlled_matrices_.at(mat_op);
    }

    /**
     * @brief Get kernels for all gate operations.
     */
    [[nodiscard]] inline auto
    getGateKernelMap() const & -> const GateKernelMap & {
        return kernel_for_gates_;
    }

    [[nodiscard]] inline auto getGateKernelMap() && -> GateKernelMap {
        return kernel_for_gates_;
    }

    /**
     * @brief Get kernels for all controlled gate operations.
     */
    [[nodiscard]] inline auto
    getControlledGateKernelMap() const & -> const ControlledGateKernelMap & {
        return kernel_for_controlled_gates_;
    }

    [[nodiscard]] inline auto
    getControlledGateKernelMap() && -> ControlledGateKernelMap {
        return kernel_for_controlled_gates_;
    }

    /**
     * @brief Get kernels for all generator operations.
     */
    [[nodiscard]] inline auto
    getGeneratorKernelMap() const & -> const GeneratorKernelMap & {
        return kernel_for_generators_;
    }

    [[nodiscard]] inline auto getGeneratorKernelMap() && -> GeneratorKernelMap {
        return kernel_for_generators_;
    }

    /**
     * @brief Get kernels for all controlled generator operations.
     */
    [[nodiscard]] inline auto getControlledGeneratorKernelMap() const & -> const
        ControlledGeneratorKernelMap & {
        return kernel_for_controlled_generators_;
    }

    [[nodiscard]] inline auto
    getControlledGeneratorKernelMap() && -> ControlledGeneratorKernelMap {
        return kernel_for_controlled_generators_;
    }

    /**
     * @brief Get kernels for all matrix operations.
     */
    [[nodiscard]] inline auto
    getMatrixKernelMap() const & -> const MatrixKernelMap & {
        return kernel_for_matrices_;
    }

    [[nodiscard]] inline auto getMatrixKernelMap() && -> MatrixKernelMap {
        return kernel_for_matrices_;
    }

    /**
     * @brief Get kernels for all controlled matrix operations.
     */
    [[nodiscard]] inline auto getControlledMatrixKernelMap() const & -> const
        ControlledMatrixKernelMap & {
        return kernel_for_controlled_matrices_;
    }

    [[nodiscard]] inline auto
    getControlledMatrixKernelMap() && -> ControlledMatrixKernelMap {
        return kernel_for_controlled_matrices_;
    }

  protected:
    explicit StateVectorLQubit(size_t num_qubits, Threading threading,
                               CPUMemoryModel memory_model)
        : BaseType(num_qubits), threading_{threading},
          memory_model_{memory_model} {
        setKernels(num_qubits, threading, memory_model);
    }

  public:
    /**
     * @brief Get the statevector's memory model.
     */
    [[nodiscard]] inline CPUMemoryModel memoryModel() const {
        return memory_model_;
    }

    /**
     * @brief Get the statevector's threading mode.
     */
    [[nodiscard]] inline Threading threading() const { return threading_; }

    /**
     *  @brief Returns a tuple containing the gate, generator, and controlled
     * matrix kernel maps respectively.
     */
    [[nodiscard]] auto getSupportedKernels() const & -> std::tuple<
        const GateKernelMap &, const GeneratorKernelMap &,
        const MatrixKernelMap &, const ControlledGateKernelMap &,
        const ControlledGeneratorKernelMap &,
        const ControlledMatrixKernelMap &> {
        return {
            getGateKernelMap(),
            getGeneratorKernelMap(),
            getMatrixKernelMap(),
            getControlledGateKernelMap(),
            getControlledGeneratorKernelMap(),
            getControlledMatrixKernelMap(),
        };
    }

    [[nodiscard]] auto getSupportedKernels() && -> std::tuple<
        GateKernelMap &&, GeneratorKernelMap &&, MatrixKernelMap &&,
        ControlledGateKernelMap &&, ControlledGeneratorKernelMap &&,
        ControlledMatrixKernelMap &&> {
        return {
            getGateKernelMap(),
            getGeneratorKernelMap(),
            getMatrixKernelMap(),
            getControlledGateKernelMap(),
            getControlledGeneratorKernelMap(),
            getControlledMatrixKernelMap(),
        };
    }

    /**
     * @brief Apply a single gate to the state-vector using a given kernel.
     *
     * @param kernel Kernel to run the operation.
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(Pennylane::Gates::KernelType kernel,
                        const std::string &opName,
                        const std::vector<size_t> &wires, bool inverse = false,
                        const std::vector<PrecisionT> &params = {}) {
        auto *arr = this->getData();
        DynamicDispatcher<PrecisionT>::getInstance().applyOperation(
            kernel, arr, this->getNumQubits(), opName, wires, inverse, params);
    }

    /**
     * @brief Apply a single gate to the state-vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(const std::string &opName,
                        const std::vector<size_t> &wires, bool inverse = false,
                        const std::vector<PrecisionT> &params = {}) {
        auto *arr = this->getData();
        auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        const auto gate_op = dispatcher.strToGateOp(opName);
        dispatcher.applyOperation(getKernelForGate(gate_op), arr,
                                  this->getNumQubits(), gate_op, wires, inverse,
                                  params);
    }

    /**
     * @brief Apply a single gate to the state-vector.
     *
     * @param opName Name of gate to apply.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(const std::string &opName,
                        const std::vector<size_t> &controlled_wires,
                        const std::vector<bool> &controlled_values,
                        const std::vector<size_t> &wires, bool inverse = false,
                        const std::vector<PrecisionT> &params = {}) {
        PL_ABORT_IF_NOT(controlled_wires.size() == controlled_values.size(),
                        "`controlled_wires` must have the same size as "
                        "`controlled_values`.");
        auto *arr = this->getData();
        const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        const auto gate_op = dispatcher.strToControlledGateOp(opName);
        const auto kernel = getKernelForControlledGate(gate_op);
        dispatcher.applyControlledGate(
            kernel, arr, this->getNumQubits(), opName, controlled_wires,
            controlled_values, wires, inverse, params);
    }
    /**
     * @brief Apply a single gate to the state-vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     * @param matrix Matrix data (in row-major format).
     */
    template <typename Alloc>
    void applyOperation(const std::string &opName,
                        const std::vector<size_t> &wires, bool inverse,
                        const std::vector<PrecisionT> &params,
                        const std::vector<ComplexT, Alloc> &matrix) {
        auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        if (dispatcher.hasGateOp(opName)) {
            applyOperation(opName, wires, inverse, params);
        } else {
            applyMatrix(matrix, wires, inverse);
        }
    }

    /**
     * @brief Apply a single gate to the state-vector.
     *
     * @param opName Name of gate to apply.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     * @param matrix Matrix data (in row-major format).
     */
    template <typename Alloc>
    void applyOperation(const std::string &opName,
                        const std::vector<size_t> &controlled_wires,
                        const std::vector<bool> &controlled_values,
                        const std::vector<size_t> &wires, bool inverse,
                        const std::vector<PrecisionT> &params,
                        const std::vector<ComplexT, Alloc> &matrix) {
        PL_ABORT_IF_NOT(controlled_wires.size() == controlled_values.size(),
                        "`controlled_wires` must have the same size as "
                        "`controlled_values`.");
        if (!controlled_wires.empty()) {
            applyOperation(opName, controlled_wires, controlled_values, wires,
                           inverse, params);
            return;
        }
        auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        if (dispatcher.hasGateOp(opName)) {
            applyOperation(opName, wires, inverse, params);
        } else {
            applyMatrix(matrix, wires, inverse);
        }
    }

    /**
     * @brief Apply a single generator to the state-vector using a given
     * kernel.
     *
     * @param kernel Kernel to run the operation.
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param adj Indicates whether to use adjoint of operator.
     */
    [[nodiscard]] inline auto
    applyGenerator(Pennylane::Gates::KernelType kernel,
                   const std::string &opName, const std::vector<size_t> &wires,
                   bool adj = false) -> PrecisionT {
        auto *arr = this->getData();
        return DynamicDispatcher<PrecisionT>::getInstance().applyGenerator(
            kernel, arr, this->getNumQubits(), opName, wires, adj);
    }

    /**
     * @brief Apply a single generator to the state-vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires the gate applies to.
     * @param adj Indicates whether to use adjoint of operator.
     */
    [[nodiscard]] auto applyGenerator(const std::string &opName,
                                      const std::vector<size_t> &wires,
                                      bool adj = false) -> PrecisionT {
        auto *arr = this->getData();
        const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        const auto gen_op = dispatcher.strToGeneratorOp(opName);
        return dispatcher.applyGenerator(getKernelForGenerator(gen_op), arr,
                                         this->getNumQubits(), opName, wires,
                                         adj);
    }

    /**
     * @brief Apply a single generator to the state-vector.
     *
     * @param opName Name of gate to apply.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires the gate applies to.
     * @param adj Indicates whether to use adjoint of operator.
     */
    [[nodiscard]] auto applyGenerator(
        const std::string &opName, const std::vector<size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        const std::vector<size_t> &wires, bool adj = false) -> PrecisionT {
        auto *arr = this->getData();
        const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        const auto generator_op = dispatcher.strToControlledGeneratorOp(opName);
        const auto kernel = getKernelForControlledGenerator(generator_op);
        return dispatcher.applyControlledGenerator(
            kernel, arr, this->getNumQubits(), opName, controlled_wires,
            controlled_values, wires, adj);
    }

    /**
     * @brief Apply a given controlled-matrix directly to the statevector.
     *
     * @param matrix Pointer to the array data (in row-major format).
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyControlledMatrix(
        const ComplexT *matrix, const std::vector<size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        const std::vector<size_t> &wires, bool inverse = false) {
        const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        auto *arr = this->getData();
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        PL_ABORT_IF_NOT(controlled_wires.size() == controlled_values.size(),
                        "`controlled_wires` must have the same size as "
                        "`controlled_values`.");
        const auto kernel = [n_wires = wires.size(), this]() {
            switch (n_wires) {
            case 1:
                return getKernelForControlledMatrix(
                    ControlledMatrixOperation::NCSingleQubitOp);
            case 2:
                return getKernelForControlledMatrix(
                    ControlledMatrixOperation::NCTwoQubitOp);
            default:
                return getKernelForControlledMatrix(
                    ControlledMatrixOperation::NCMultiQubitOp);
            }
        }();
        dispatcher.applyControlledMatrix(kernel, arr, this->getNumQubits(),
                                         matrix, controlled_wires,
                                         controlled_values, wires, inverse);
    }

    /**
     * @brief Apply a given controlled-matrix directly to the statevector.
     *
     * @param matrix Vector containing the statevector data (in row-major
     * format).
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void
    applyControlledMatrix(const std::vector<ComplexT> matrix,
                          const std::vector<size_t> &controlled_wires,
                          const std::vector<bool> &controlled_values,
                          const std::vector<size_t> &wires,
                          bool inverse = false) {
        applyControlledMatrix(matrix.data(), controlled_wires,
                              controlled_values, wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a given
     * kernel.
     *
     * @param kernel Kernel to run the operation
     * @param matrix Pointer to the array data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(Pennylane::Gates::KernelType kernel,
                            const ComplexT *matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        auto *arr = this->getData();

        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");

        dispatcher.applyMatrix(kernel, arr, this->getNumQubits(), matrix, wires,
                               inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a given
     * kernel.
     *
     * @param kernel Kernel to run the operation
     * @param matrix Matrix data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(Pennylane::Gates::KernelType kernel,
                            const std::vector<ComplexT> &matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(matrix.size() != exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");

        applyMatrix(kernel, matrix.data(), wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a
     * raw matrix pointer vector.
     *
     * @param matrix Pointer to the array data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(const ComplexT *matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        using Pennylane::Gates::MatrixOperation;

        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");

        const auto kernel = [n_wires = wires.size(), this]() {
            switch (n_wires) {
            case 1:
                return getKernelForMatrix(MatrixOperation::SingleQubitOp);
            case 2:
                return getKernelForMatrix(MatrixOperation::TwoQubitOp);
            default:
                return getKernelForMatrix(MatrixOperation::MultiQubitOp);
            }
        }();
        applyMatrix(kernel, matrix, wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param matrix Matrix data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    template <typename Alloc>
    inline void applyMatrix(const std::vector<ComplexT, Alloc> &matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(matrix.size() != exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");

        applyMatrix(matrix.data(), wires, inverse);
    }
};
} // namespace Pennylane::LightningQubit