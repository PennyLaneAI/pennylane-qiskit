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
 * @file DynamicDispatcher.hpp
 * Defines DynamicDispatcher class. Can be used to call a gate operation by
 * string.
 */

#pragma once

#include <complex>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "Error.hpp"
#include "GateIndices.hpp"
#include "KernelType.hpp"
#include "Macros.hpp"
#include "OpToMemberFuncPtr.hpp"
#include "Util.hpp" // PairHash, exp2

/// @cond DEV
namespace {
namespace GateConstant = Pennylane::Gates::Constant;
using Pennylane::Gates::GateOperation;
using Pennylane::Gates::GeneratorOperation;
using Pennylane::Gates::KernelType;
using Pennylane::Gates::MatrixOperation;
using Pennylane::Util::exp2;
using Pennylane::Util::lookup;
using Pennylane::Util::PairHash;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Internal {
constexpr auto generatorNamesWithoutPrefix() {
    constexpr std::string_view prefix{"Generator"};
    namespace GateConstant = Pennylane::Gates::Constant;
    std::array<std::pair<GeneratorOperation, std::string_view>,
               GateConstant::generator_names.size()>
        res{};
    for (size_t i = 0; i < GateConstant::generator_names.size(); i++) {
        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
        const auto [gntr_op, gntr_name] = GateConstant::generator_names[i];
        res[i].first = gntr_op;
        res[i].second = gntr_name.substr(prefix.size());
        // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
    }
    return res;
}

} // namespace Pennylane::LightningQubit::Internal
/// @endcond

namespace Pennylane::LightningQubit {
/**
 * @brief DynamicDispatcher class
 *
 * This is a singleton class that can call a gate/generator operation
 * dynamically. Currently, all gate operations (gates/generators/matrices) are
 * registered to this class when the library is loaded. As all functions besides
 * registration functions are already thread-safe, we can use this class
 * in multithreading environment without any problem.
 * In addition, adding mutex is not required unless kernel functions are
 * registered in multiple threads.
 */
template <typename PrecisionT> class DynamicDispatcher {
  public:
    using CFP_t = std::complex<PrecisionT>;

    using GateFunc = std::function<void(
        std::complex<PrecisionT> * /*data*/, size_t /*num_qubits*/,
        const std::vector<size_t> & /*wires*/, bool /*inverse*/,
        const std::vector<PrecisionT> & /*params*/)>;
    using ControlledGateFunc = std::function<void(
        std::complex<PrecisionT> * /*data*/, size_t /*num_qubits*/,
        const std::vector<size_t> & /*controlled_wires*/,
        const std::vector<bool> & /*controlled_values*/,
        const std::vector<size_t> & /*wires*/, bool /*inverse*/,
        const std::vector<PrecisionT> & /*params*/)>;

    using GeneratorFunc = Gates::GeneratorFuncPtrT<PrecisionT>;
    using ControlledGeneratorFunc =
        Gates::ControlledGeneratorFuncPtrT<PrecisionT>;

    using MatrixFunc = Gates::MatrixFuncPtrT<PrecisionT>;
    using ControlledMatrixFunc = Gates::ControlledMatrixFuncPtrT<PrecisionT>;

  private:
    std::unordered_map<std::string, GateOperation> str_to_gates_{};

    std::unordered_map<std::string, GeneratorOperation> str_to_gntrs_{};

    std::unordered_map<std::pair<GateOperation, KernelType>, GateFunc, PairHash>
        gate_kernels_{};

    std::unordered_map<std::pair<GeneratorOperation, KernelType>, GeneratorFunc,
                       PairHash>
        generator_kernels_{};

    std::unordered_map<std::pair<MatrixOperation, KernelType>, MatrixFunc,
                       PairHash>
        matrix_kernels_{};

    std::unordered_map<KernelType, std::string> kernel_names_{};

    std::unordered_map<std::string, ControlledGateOperation>
        str_to_controlled_gates_{};

    std::unordered_map<std::string, ControlledGeneratorOperation>
        str_to_controlled_gntrs_{};

    std::unordered_map<std::pair<ControlledGateOperation, KernelType>,
                       ControlledGateFunc, PairHash>
        controlled_gate_kernels_{};

    std::unordered_map<std::pair<ControlledGeneratorOperation, KernelType>,
                       ControlledGeneratorFunc, PairHash>
        controlled_generator_kernels_{};

    std::unordered_map<std::pair<ControlledMatrixOperation, KernelType>,
                       ControlledMatrixFunc, PairHash>
        controlled_matrix_kernels_{};

    DynamicDispatcher() {
        constexpr static auto gntr_names_without_prefix =
            Internal::generatorNamesWithoutPrefix();

        for (const auto &[gate_op, gate_name] : GateConstant::gate_names) {
            str_to_gates_.emplace(gate_name, gate_op);
        }
        for (const auto &[gntr_op, gntr_name] : gntr_names_without_prefix) {
            str_to_gntrs_.emplace(gntr_name, gntr_op);
        }
        for (const auto &[gate_op, gate_name] :
             GateConstant::controlled_gate_names) {
            str_to_controlled_gates_.emplace(gate_name, gate_op);
        }
        for (const auto &[gntr_op, gntr_name] :
             GateConstant::controlled_generator_names) {
            str_to_controlled_gntrs_.emplace(gntr_name, gntr_op);
        }
    }

  public:
    DynamicDispatcher(const DynamicDispatcher &) = delete;
    DynamicDispatcher(DynamicDispatcher &&) = delete;
    DynamicDispatcher &operator=(const DynamicDispatcher &) = delete;
    DynamicDispatcher &operator=(DynamicDispatcher &&) = delete;
    ~DynamicDispatcher() = default;

    /**
     * @brief Get the singleton instance
     */
    static DynamicDispatcher &getInstance() {
        static DynamicDispatcher singleton;
        return singleton;
    }

    /**
     * @brief Get all registered kernels
     */
    [[nodiscard]] auto registeredKernels() const -> std::vector<KernelType> {
        std::vector<KernelType> kernels;

        kernels.reserve(kernel_names_.size());
        for (const auto &[kernel, name] : kernel_names_) {
            kernels.emplace_back(kernel);
        }
        return kernels;
    }

    /**
     * @brief Check whether the kernel is registered to a dispatcher
     */
    [[nodiscard]] auto isRegisteredKernel(KernelType kernel) const {
        return kernel_names_.contains(kernel);
    }

    /**
     * @brief Register kernel name
     *
     * @param kernel Kernel
     * @param name Name of the kernel
     */
    void registerKernelName(KernelType kernel, std::string name) {
        kernel_names_.emplace(kernel, std::move(name));
    }

    /**
     * @brief Get registered name of the kernel
     *
     * @param kernel Kernel
     */
    [[nodiscard]] auto getKernelName(KernelType kernel) const -> std::string {
        return kernel_names_.at(kernel);
    }

    /**
     * @brief Get registered gates for the given kernel
     *
     * @param kernel Kernel
     */
    [[nodiscard]] auto registeredGatesForKernel(KernelType kernel) const
        -> std::unordered_set<GateOperation> {
        std::unordered_set<GateOperation> gates;

        for (const auto &[key, val] : gate_kernels_) {
            if (key.second == kernel) {
                gates.emplace(key.first);
            }
        }
        return gates;
    }

    /**
     * @brief Get registered controlled gates for the given kernel
     *
     * @param kernel Kernel
     */
    [[nodiscard]] auto
    registeredControlledGatesForKernel(KernelType kernel) const
        -> std::unordered_set<ControlledGateOperation> {
        std::unordered_set<ControlledGateOperation> gates;

        for (const auto &[key, val] : controlled_gate_kernels_) {
            if (key.second == kernel) {
                gates.emplace(key.first);
            }
        }
        return gates;
    }

    /**
     * @brief Get registered generators for the given kernel
     *
     * @param kernel Kernel
     */
    [[nodiscard]] auto registeredGeneratorsForKernel(KernelType kernel) const
        -> std::unordered_set<GeneratorOperation> {
        std::unordered_set<GeneratorOperation> gntrs;

        for (const auto &[key, val] : generator_kernels_) {
            if (key.second == kernel) {
                gntrs.emplace(key.first);
            }
        }
        return gntrs;
    }

    /**
     * @brief Get registered controlled generators for the given kernel
     *
     * @param kernel Kernel
     */
    [[nodiscard]] auto
    registeredControlledGeneratorsForKernel(KernelType kernel) const
        -> std::unordered_set<ControlledGeneratorOperation> {
        std::unordered_set<ControlledGeneratorOperation> generators;

        for (const auto &[key, val] : controlled_generator_kernels_) {
            if (key.second == kernel) {
                generators.emplace(key.first);
            }
        }
        return generators;
    }

    /**
     * @brief Get registered matrix operations for the given kernel
     *
     * @param kernel Kernel
     */
    [[nodiscard]] auto registeredMatricesForKernel(KernelType kernel) const
        -> std::unordered_set<MatrixOperation> {
        std::unordered_set<MatrixOperation> matrices;

        for (const auto &[key, val] : matrix_kernels_) {
            if (key.second == kernel) {
                matrices.emplace(key.first);
            }
        }
        return matrices;
    }

    /**
     * @brief Get registered controlled matrix operations for the given kernel
     *
     * @param kernel Kernel
     */
    [[nodiscard]] auto
    registeredControlledMatricesForKernel(KernelType kernel) const
        -> std::unordered_set<ControlledMatrixOperation> {
        std::unordered_set<ControlledMatrixOperation> matrices;

        for (const auto &[key, val] : controlled_matrix_kernels_) {
            if (key.second == kernel) {
                matrices.emplace(key.first);
            }
        }
        return matrices;
    }

    /**
     * @brief Gate name to gate operation
     *
     * @param gate_name Gate name
     */
    [[nodiscard]] auto strToGateOp(const std::string &gate_name) const
        -> GateOperation {
        return str_to_gates_.at(gate_name);
    }

    /**
     * @brief Gate name to controlled gate operation
     *
     * @param gate_name Gate name
     */
    [[nodiscard]] auto strToControlledGateOp(const std::string &gate_name) const
        -> ControlledGateOperation {
        return str_to_controlled_gates_.at(gate_name);
    }

    /**
     * @brief Returns true if the gate operation exists
     *
     * @param gate_name Gate name
     */
    [[nodiscard]] auto hasGateOp(const std::string &gate_name) const -> bool {
        return str_to_gates_.contains(gate_name);
    }

    /**
     * @brief Generator name to generator operation
     *
     * @param gntr_name Generator name without "Generator" prefix
     */
    [[nodiscard]] auto strToGeneratorOp(const std::string &gntr_name) const
        -> GeneratorOperation {
        return str_to_gntrs_.at(gntr_name);
    }

    /**
     * @brief Generator name to controlled generator operation
     *
     * @param gntr_name Generator name without "Generator" prefix
     */
    [[nodiscard]] auto
    strToControlledGeneratorOp(const std::string &gntr_name) const
        -> ControlledGeneratorOperation {
        return str_to_controlled_gntrs_.at(gntr_name);
    }

    /**
     * @brief Register a new gate operation for the operation. Can pass a custom
     * kernel
     */
    template <typename FunctionType>
    void registerGateOperation(GateOperation gate_op, KernelType kernel,
                               FunctionType &&func) {
        gate_kernels_.emplace(std::make_pair(gate_op, kernel),
                              std::forward<FunctionType>(func));
    }

    /**
     * @brief Register a new controlled gate operation for the operation. Can
     * pass a custom kernel
     */
    template <typename FunctionType>
    void registerControlledGateOperation(ControlledGateOperation gate_op,
                                         KernelType kernel,
                                         FunctionType &&func) {
        controlled_gate_kernels_.emplace(std::make_pair(gate_op, kernel),
                                         std::forward<FunctionType>(func));
    }

    /**
     * @brief Register a new gate generator for the operation. Can pass a custom
     * kernel
     */
    template <typename FunctionType>
    void registerGeneratorOperation(GeneratorOperation gntr_op,
                                    KernelType kernel, FunctionType &&func) {
        generator_kernels_.emplace(std::make_pair(gntr_op, kernel),
                                   std::forward<FunctionType>(func));
    }

    /**
     * @brief Register a new controlled gate generator for the operation. Can
     * pass a custom kernel
     */
    template <typename FunctionType>
    void
    registerControlledGeneratorOperation(ControlledGeneratorOperation gen_op,
                                         KernelType kernel,
                                         FunctionType &&func) {
        controlled_generator_kernels_.emplace(std::make_pair(gen_op, kernel),
                                              std::forward<FunctionType>(func));
    }

    /**
     * @brief Register a new matrix operation. Can pass a custom
     * kernel
     */
    void registerMatrixOperation(MatrixOperation mat_op, KernelType kernel,
                                 MatrixFunc func) {
        matrix_kernels_.emplace(std::make_pair(mat_op, kernel), func);
    }

    /**
     * @brief Register a new controlled matrix operation.
     */
    void registerControlledMatrixOperation(ControlledMatrixOperation mat_op,
                                           KernelType kernel,
                                           ControlledMatrixFunc func) {
        controlled_matrix_kernels_.emplace(std::make_pair(mat_op, kernel),
                                           func);
    }

    /**
     * @brief Check if a kernel function is registered for the given
     * gate operation and kernel.
     *
     * @param gate_op Gate operation
     * @param kernel Kernel
     */
    [[nodiscard]] bool isRegistered(GateOperation gate_op,
                                    KernelType kernel) const {
        return gate_kernels_.find(std::make_pair(gate_op, kernel)) !=
               gate_kernels_.cend();
    }

    /**
     * @brief Check if a kernel function is registered for the given
     * controlled gate operation and kernel.
     *
     * @param gate_op Gate operation
     * @param kernel Kernel
     */
    bool isRegistered(ControlledGateOperation gate_op,
                      KernelType kernel) const {
        return controlled_gate_kernels_.find(std::make_pair(gate_op, kernel)) !=
               controlled_gate_kernels_.cend();
    }

    /**
     * @brief Check if a kernel function is registered for the given
     * generator operation and kernel.
     *
     * @param gntr_op Generator operation
     * @param kernel Kernel
     */
    [[nodiscard]] bool isRegistered(GeneratorOperation gntr_op,
                                    KernelType kernel) const {
        return generator_kernels_.find(std::make_pair(gntr_op, kernel)) !=
               generator_kernels_.cend();
    }

    /**
     * @brief Check if a kernel function is registered for the given
     * controlled generator operation and kernel.
     *
     * @param gntr_op Generator operation
     * @param kernel Kernel
     */
    bool isRegistered(ControlledGeneratorOperation gen_op,
                      KernelType kernel) const {
        return controlled_generator_kernels_.find(std::make_pair(
                   gen_op, kernel)) != controlled_generator_kernels_.cend();
    }

    /**
     * @brief Check if a kernel function is registered for the given
     * matrix operation and kernel.
     *
     * @param mat_op Matrix operation
     * @param kernel Kernel
     */
    [[nodiscard]] bool isRegistered(MatrixOperation mat_op,
                                    KernelType kernel) const {
        return matrix_kernels_.find(std::make_pair(mat_op, kernel)) !=
               matrix_kernels_.cend();
    }

    /**
     * @brief Check if a kernel function is registered for the given
     * controlled matrix operation and kernel.
     *
     * @param mat_op Controlled matrix operation
     * @param kernel Kernel
     */
    bool isRegistered(ControlledMatrixOperation mat_op,
                      KernelType kernel) const {
        return controlled_matrix_kernels_.find(std::make_pair(
                   mat_op, kernel)) != controlled_matrix_kernels_.cend();
    }

    /**
     * @brief Apply a single gate to the state-vector using the given kernel.
     *
     * @param kernel Kernel to run the gate operation.
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param op_name Gate operation name.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(KernelType kernel, CFP_t *data, size_t num_qubits,
                        const std::string &op_name,
                        const std::vector<size_t> &wires, bool inverse,
                        const std::vector<PrecisionT> &params = {}) const {
        applyOperation(kernel, data, num_qubits, strToGateOp(op_name), wires,
                       inverse, params);
    }

    /**
     * @brief Apply a single gate to the state-vector using the given kernel.
     *
     * @param kernel Kernel to run the gate operation.
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param gate_op Gate operation.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(KernelType kernel, CFP_t *data, size_t num_qubits,
                        GateOperation gate_op, const std::vector<size_t> &wires,
                        bool inverse,
                        const std::vector<PrecisionT> &params = {}) const {
        const auto iter = gate_kernels_.find(std::make_pair(gate_op, kernel));
        PL_ABORT_IF(iter == gate_kernels_.cend(),
                    "Cannot find a registered kernel for a given gate "
                    "and kernel pair");
        (iter->second)(data, num_qubits, wires, inverse, params);
    }

    /**
     * @brief Apply a single controlled gate to the state-vector using the given
     * kernel.
     *
     * @param kernel Kernel to run the gate operation.
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param gate_op Gate operation.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyControlledGate(KernelType kernel, CFP_t *data, size_t num_qubits,
                             const std::string &op_name,
                             const std::vector<size_t> &controlled_wires,
                             const std::vector<bool> &controlled_values,
                             const std::vector<size_t> &wires, bool inverse,
                             const std::vector<PrecisionT> &params = {}) const {
        applyControlledGate(kernel, data, num_qubits,
                            strToControlledGateOp(op_name), controlled_wires,
                            controlled_values, wires, inverse, params);
    }

    /**
     * @brief Apply a single controlled gate to the state-vector using the given
     * kernel.
     *
     * @param kernel Kernel to run the gate operation.
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param gate_op Gate operation.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyControlledGate(KernelType kernel, CFP_t *data, size_t num_qubits,
                             const ControlledGateOperation op_name,
                             const std::vector<size_t> &controlled_wires,
                             const std::vector<bool> &controlled_values,
                             const std::vector<size_t> &wires, bool inverse,
                             const std::vector<PrecisionT> &params = {}) const {
        PL_ABORT_IF_NOT(controlled_wires.size() == controlled_values.size(),
                        "`controlled_wires` must have the same size as "
                        "`controlled_values`.");
        const auto iter =
            controlled_gate_kernels_.find(std::make_pair(op_name, kernel));
        PL_ABORT_IF(
            iter == controlled_gate_kernels_.cend(),
            "Cannot find a registered kernel for a given controlled gate "
            "and kernel pair");
        (iter->second)(data, num_qubits, controlled_wires, controlled_values,
                       wires, inverse, params);
    }

    /**
     * @brief Apply multiple gates to the state-vector using a registered kernel
     *
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param ops List of Gate operation names.
     * @param wires List of wires to apply each gate to.
     * @param inverse List of inverses
     * @param params List of parameters
     */
    void
    applyOperations(KernelType kernel, CFP_t *data, size_t num_qubits,
                    const std::vector<std::string> &ops,
                    const std::vector<std::vector<size_t>> &wires,
                    const std::vector<bool> &inverse,
                    const std::vector<std::vector<PrecisionT>> &params) const {
        const size_t numOperations = ops.size();
        PL_ABORT_IF(numOperations != wires.size() ||
                        numOperations != params.size(),
                    "Invalid arguments: number of operations, wires, and "
                    "parameters must all be equal");
        for (size_t i = 0; i < numOperations; i++) {
            applyOperation(kernel, data, num_qubits, ops[i], wires[i],
                           inverse[i], params[i]);
        }
    }

    /**
     * @brief Apply multiple (non-parameterized) gates to the state-vector
     * using a registered kernel
     *
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param ops List of Gate operation names.
     * @param wires List of wires to apply each gate to.
     * @param inverse List of inverses
     */
    void applyOperations(KernelType kernel, CFP_t *data, size_t num_qubits,
                         const std::vector<std::string> &ops,
                         const std::vector<std::vector<size_t>> &wires,
                         const std::vector<bool> &inverse) const {
        const size_t numOperations = ops.size();
        PL_ABORT_IF(numOperations != wires.size(),
                    "Invalid arguments: number of operations, wires, and "
                    "parameters must all be equal");
        for (size_t i = 0; i < numOperations; i++) {
            applyOperation(kernel, data, num_qubits, ops[i], wires[i],
                           inverse[i], {});
        }
    }

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param kernel Kernel to use for this operation
     * @param data Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param matrix Perfect square matrix in row-major order.
     * @param wires Wires the gate applies to.
     * @param inverse Indicate whether inverse should be taken.
     */
    void applyMatrix(KernelType kernel, CFP_t *data, size_t num_qubits,
                     const std::complex<PrecisionT> *matrix,
                     const std::vector<size_t> &wires, bool inverse) const {
        PL_ASSERT(num_qubits >= wires.size());

        const auto mat_op = [n_wires = wires.size()]() {
            switch (n_wires) {
            case 1:
                return MatrixOperation::SingleQubitOp;
            case 2:
                return MatrixOperation::TwoQubitOp;
            default:
                return MatrixOperation::MultiQubitOp;
            }
        }();

        const auto iter = matrix_kernels_.find(std::make_pair(mat_op, kernel));
        PL_ABORT_IF(iter == matrix_kernels_.end(),
                    std::string(lookup(GateConstant::matrix_names, mat_op)) +
                        " is not registered for the given kernel");
        (iter->second)(data, num_qubits, matrix, wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param kernel Kernel to use for this operation
     * @param data Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param matrix Perfect square matrix in row-major order.
     * @param wires Wires the gate applies to.
     * @param inverse Indicate whether inverse should be taken.
     */
    void applyMatrix(KernelType kernel, CFP_t *data, size_t num_qubits,
                     const std::vector<std::complex<PrecisionT>> &matrix,
                     const std::vector<size_t> &wires, bool inverse) const {
        PL_ABORT_IF_NOT(matrix.size() == exp2(2 * wires.size()),
                        "The size of matrix does not match with the given "
                        "number of wires");
        applyMatrix(kernel, data, num_qubits, matrix.data(), wires, inverse);
    }

    /**
     * @brief Apply a given matrix and controls directly to the statevector.
     *
     * @param kernel Kernel to use for this operation
     * @param data Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param matrix Perfect square matrix in row-major order.
     * @param wires Control wires.
     * @param wires Wires the gate applies to.
     * @param inverse Indicate whether inverse should be taken.
     */
    void applyControlledMatrix(KernelType kernel, CFP_t *data,
                               size_t num_qubits,
                               const std::complex<PrecisionT> *matrix,
                               const std::vector<size_t> &controlled_wires,
                               const std::vector<bool> &controlled_values,
                               const std::vector<size_t> &wires,
                               bool inverse) const {
        PL_ASSERT(num_qubits >= controlled_wires.size() + wires.size());
        PL_ABORT_IF_NOT(controlled_wires.size() == controlled_values.size(),
                        "`controlled_wires` must have the same size as "
                        "`controlled_values`.");
        const auto mat_op = [n_wires = wires.size()]() {
            switch (n_wires) {
            case 1:
                return ControlledMatrixOperation::NCSingleQubitOp;
            case 2:
                return ControlledMatrixOperation::NCTwoQubitOp;
            default:
                return ControlledMatrixOperation::NCMultiQubitOp;
            }
        }();

        const auto iter =
            controlled_matrix_kernels_.find(std::make_pair(mat_op, kernel));
        PL_ABORT_IF(
            iter == controlled_matrix_kernels_.end(),
            std::string(lookup(GateConstant::controlled_matrix_names, mat_op)) +
                " is not registered for the given kernel");
        (iter->second)(data, num_qubits, matrix, controlled_wires,
                       controlled_values, wires, inverse);
    }

    /**
     * @brief Apply a single generator to the state-vector using the given
     * kernel.
     *
     * @param kernel Kernel to run the gate operation.
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param gntr_op Generator operation.
     * @param wires Wires to apply gate to.
     * @param adj Indicates whether to use adjoint of gate.
     */
    auto applyGenerator(KernelType kernel, CFP_t *data, size_t num_qubits,
                        GeneratorOperation gntr_op,
                        const std::vector<size_t> &wires, bool adj) const
        -> PrecisionT {
        using Pennylane::Gates::Constant::generator_names;
        const auto iter =
            generator_kernels_.find(std::make_pair(gntr_op, kernel));
        PL_ABORT_IF(iter == generator_kernels_.cend(),
                    "Cannot find a registered kernel for a given generator "
                    "and kernel pair.");
        return (iter->second)(data, num_qubits, wires, adj);
    }

    /**
     * @brief Apply a single generator to the state-vector using the given
     * kernel.
     *
     * @param kernel Kernel to run the gate operation.
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param op_name Generator operation name.
     * @param wires Wires to apply gate to.
     * @param adj Indicates whether to use adjoint of gate.
     */
    auto applyGenerator(KernelType kernel, CFP_t *data, size_t num_qubits,
                        const std::string &op_name,
                        const std::vector<size_t> &wires, bool adj) const
        -> PrecisionT {
        return applyGenerator(kernel, data, num_qubits,
                              strToGeneratorOp(op_name), wires, adj);
    }

    /**
     * @brief Apply a single controlled generator to the state-vector using the
     * given kernel.
     *
     * @param kernel Kernel to run the gate operation.
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param gntr_op Generator operation.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to apply gate to.
     * @param adj Indicates whether to use adjoint of gate.
     */
    auto applyControlledGenerator(KernelType kernel, CFP_t *data,
                                  size_t num_qubits,
                                  const ControlledGeneratorOperation gntr_op,
                                  const std::vector<size_t> &controlled_wires,
                                  const std::vector<bool> &controlled_values,
                                  const std::vector<size_t> &wires,
                                  bool inverse) const -> PrecisionT {

        using Pennylane::Gates::Constant::controlled_generator_names;
        const auto iter =
            controlled_generator_kernels_.find(std::make_pair(gntr_op, kernel));
        PL_ABORT_IF(iter == controlled_generator_kernels_.cend(),
                    "Cannot find a registered kernel for a given controlled "
                    "generator "
                    "and kernel pair.");
        return (iter->second)(data, num_qubits, controlled_wires,
                              controlled_values, wires, inverse);
    }

    /**
     * @brief Apply a single controlled generator to the state-vector using the
     * given kernel.
     *
     * @param kernel Kernel to run the gate operation.
     * @param data Pointer to data.
     * @param num_qubits Number of qubits.
     * @param op_name Gate operation name.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to apply gate to.
     * @param adj Indicates whether to use adjoint of gate.
     */
    auto applyControlledGenerator(KernelType kernel, CFP_t *data,
                                  size_t num_qubits, const std::string &op_name,
                                  const std::vector<size_t> &controlled_wires,
                                  const std::vector<bool> &controlled_values,
                                  const std::vector<size_t> &wires,
                                  bool inverse) const -> PrecisionT {
        return applyControlledGenerator(
            kernel, data, num_qubits, strToControlledGeneratorOp(op_name),
            controlled_wires, controlled_values, wires, inverse);
    }
};
} // namespace Pennylane::LightningQubit

/// @cond DEV
namespace Pennylane::LightningQubit::Internal {
int registerAllAvailableKernels_Float();
int registerAllAvailableKernels_Double();

/**
 * @brief These functions are only used to register kernels to the dynamic
 * dispatcher.
 */
struct RegisterBeforeMain_Float {
    const static inline int dummy = registerAllAvailableKernels_Float();
};

struct RegisterBeforeMain_Double {
    const static inline int dummy = registerAllAvailableKernels_Double();
};
} // namespace Pennylane::LightningQubit::Internal
/// @endcond
