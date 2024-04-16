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
 * @file RegisterKernel.hpp
 * Register all gate and generator implementations
 */
#pragma once
#include <tuple>

#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "DynamicDispatcher.hpp"
#include "Error.hpp"
#include "GateIndices.hpp"
#include "GateOperation.hpp"
#include "OpToMemberFuncPtr.hpp"

/// @cond DEV
namespace {
using Pennylane::Util::lookup;
using Pennylane::Util::prepend_to_tuple;
} // namespace
/// @endcond

/// @cond DEV
namespace Pennylane::LightningQubit {
/**
 * @brief return a lambda function for the given kernel and gate operation
 *
 * As we want the lambda function to be stateless, kernel and gate_op are
 * template parameters (or the functions can be consteval in C++20).
 * In C++20, one also may use a template lambda function instead.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data.
 * @tparam ParamT Floating point type for parameters>
 * @tparam GateImplementation Gate implementation class.
 * @tparam gate_op Gate operation to make a functor.
 */
template <class PrecisionT, class ParamT, class GateImplementation,
          Pennylane::Gates::GateOperation gate_op>
constexpr auto gateOpToFunctor() {
    return [](std::complex<PrecisionT> *data, size_t num_qubits,
              const std::vector<size_t> &wires, bool inverse,
              const std::vector<PrecisionT> &params) {
        constexpr auto func_ptr =
            Gates::GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                         gate_op>::value;
        PL_ASSERT(params.size() ==
                  lookup(Pennylane::Gates::Constant::gate_num_params, gate_op));
        Gates::callGateOps(func_ptr, data, num_qubits, wires, inverse, params);
    };
}

/**
 * @brief return a lambda function for the given kernel and controlled gate
 * operation
 *
 * As we want the lambda function to be stateless, kernel and gate_op are
 * template parameters (or the functions can be consteval in C++20).
 * In C++20, one also may use a template lambda function instead.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data.
 * @tparam ParamT Floating point type for parameters>
 * @tparam GateImplementation Gate implementation class.
 * @tparam gate_op Gate operation to make a functor.
 */
template <class PrecisionT, class ParamT, class GateImplementation,
          Pennylane::Gates::ControlledGateOperation gate_op>
constexpr auto controlledGateOpToFunctor() {
    return [](std::complex<PrecisionT> *data, size_t num_qubits,
              const std::vector<size_t> &controlled_wires,
              const std::vector<bool> &controlled_values,
              const std::vector<size_t> &wires, bool inverse,
              const std::vector<PrecisionT> &params) {
        constexpr auto func_ptr = Gates::ControlledGateOpToMemberFuncPtr<
            PrecisionT, ParamT, GateImplementation, gate_op>::value;
        PL_ASSERT(params.size() ==
                  lookup(Pennylane::Gates::Constant::controlled_gate_num_params,
                         gate_op));
        Gates::callControlledGateOps(func_ptr, data, num_qubits,
                                     controlled_wires, controlled_values, wires,
                                     inverse, params);
    };
}

/**
 * @brief Internal recursion function for constructGateOpsFunctorTuple
 *
 * @return Tuple of gate operations and corresponding GateImplementation member
 * function pointers.
 */
template <class PrecisionT, class ParamT, class GateImplementation,
          size_t gate_idx>
constexpr auto constructGateOpsFunctorTupleIter() {
    if constexpr (gate_idx == GateImplementation::implemented_gates.size()) {
        return std::tuple{};
    } else if (gate_idx < GateImplementation::implemented_gates.size()) {
        constexpr auto gate_op =
            GateImplementation::implemented_gates[gate_idx];
        return prepend_to_tuple(
            std::pair{gate_op, gateOpToFunctor<PrecisionT, ParamT,
                                               GateImplementation, gate_op>()},
            constructGateOpsFunctorTupleIter<
                PrecisionT, ParamT, GateImplementation, gate_idx + 1>());
    }
}
/**
 * @brief Internal recursive function for
 * constructControlledGateOpsFunctorTuple
 */
template <class PrecisionT, class ParamT, class GateImplementation,
          size_t gate_idx>
constexpr auto constructControlledGateOpsFunctorTupleIter() {
    if constexpr (gate_idx ==
                  GateImplementation::implemented_controlled_gates.size()) {
        return std::tuple{};
    } else if (gate_idx <
               GateImplementation::implemented_controlled_gates.size()) {
        constexpr auto gate_op =
            GateImplementation::implemented_controlled_gates[gate_idx];
        return prepend_to_tuple(
            std::pair{gate_op,
                      controlledGateOpToFunctor<PrecisionT, ParamT,
                                                GateImplementation, gate_op>()},
            constructControlledGateOpsFunctorTupleIter<
                PrecisionT, ParamT, GateImplementation, gate_idx + 1>());
    }
}

/**
 * @brief Internal recursion function for constructGeneratorOpsFunctorTuple
 */
template <class PrecisionT, class GateImplementation, size_t gntr_idx>
constexpr auto constructGeneratorOpsFunctorTupleIter() {
    if constexpr (gntr_idx ==
                  GateImplementation::implemented_generators.size()) {
        return std::tuple{};
    } else if (gntr_idx < GateImplementation::implemented_generators.size()) {
        constexpr auto gntr_op =
            GateImplementation::implemented_generators[gntr_idx];
        return prepend_to_tuple(
            std::pair{gntr_op,
                      Gates::GeneratorOpToMemberFuncPtr<
                          PrecisionT, GateImplementation, gntr_op>::value},
            constructGeneratorOpsFunctorTupleIter<
                PrecisionT, GateImplementation, gntr_idx + 1>());
    }
}

/**
 * @brief Internal recursion function for
 * constructControlledGeneratorOpsFunctorTuple
 */
template <class PrecisionT, class GateImplementation, size_t gntr_idx>
constexpr auto constructControlledGeneratorOpsFunctorTupleIter() {
    if constexpr (gntr_idx ==
                  GateImplementation::implemented_controlled_generators
                      .size()) {
        return std::tuple{};
    } else if (gntr_idx <
               GateImplementation::implemented_controlled_generators.size()) {
        constexpr auto gntr_op =
            GateImplementation::implemented_controlled_generators[gntr_idx];
        return prepend_to_tuple(
            std::pair{gntr_op,
                      Gates::ControlledGeneratorOpToMemberFuncPtr<
                          PrecisionT, GateImplementation, gntr_op>::value},
            constructControlledGeneratorOpsFunctorTupleIter<
                PrecisionT, GateImplementation, gntr_idx + 1>());
    }
}

/**
 * @brief Internal recursive function for constructMatrixOpsFunctorTuple
 */
template <class PrecisionT, class GateImplementation, size_t mat_idx>
constexpr auto constructMatrixOpsFunctorTupleIter() {
    if constexpr (mat_idx == GateImplementation::implemented_matrices.size()) {
        return std::tuple{};
    } else if (mat_idx < GateImplementation::implemented_matrices.size()) {
        constexpr auto mat_op =
            GateImplementation::implemented_matrices[mat_idx];
        return prepend_to_tuple(
            std::pair{
                mat_op,
                Gates::MatrixOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                               mat_op>::value},
            constructMatrixOpsFunctorTupleIter<PrecisionT, GateImplementation,
                                               mat_idx + 1>());
    }
}

/**
 * @brief Internal recursive function for
 * constructControlledMatrixOpsFunctorTuple
 */
template <class PrecisionT, class GateImplementation, size_t mat_idx>
constexpr auto constructControlledMatrixOpsFunctorTupleIter() {
    if constexpr (mat_idx ==
                  GateImplementation::implemented_controlled_matrices.size()) {
        return std::tuple{};
    } else if (mat_idx <
               GateImplementation::implemented_controlled_matrices.size()) {
        constexpr auto mat_op =
            GateImplementation::implemented_controlled_matrices[mat_idx];
        return prepend_to_tuple(
            std::pair{mat_op,
                      Gates::ControlledMatrixOpToMemberFuncPtr<
                          PrecisionT, GateImplementation, mat_op>::value},
            constructControlledMatrixOpsFunctorTupleIter<
                PrecisionT, GateImplementation, mat_idx + 1>());
    }
}
/// @endcond

/**
 * @brief Tuple of gate operation and function pointer pairs.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam ParamT Floating point type of gate parameters
 * @tparam GateImplementation Gate implementation class.
 */
template <class PrecisionT, class ParamT, class GateImplementation>
constexpr auto gate_op_functor_tuple =
    constructGateOpsFunctorTupleIter<PrecisionT, ParamT, GateImplementation,
                                     0>();

/**
 * @brief Tuple of controlled gate operation and function pointer pairs.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam ParamT Floating point type of gate parameters
 * @tparam GateImplementation Gate implementation class.
 */
template <class PrecisionT, class ParamT, class GateImplementation>
constexpr auto controlled_gate_op_functor_tuple =
    constructControlledGateOpsFunctorTupleIter<PrecisionT, ParamT,
                                               GateImplementation, 0>();

/**
 * @brief Tuple of generator operation and function pointer pairs.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam GateImplementation Gate implementation class.
 */
template <class PrecisionT, class GateImplementation>
constexpr auto generator_op_functor_tuple =
    constructGeneratorOpsFunctorTupleIter<PrecisionT, GateImplementation, 0>();

/**
 * @brief Tuple of controlled generator operation and function pointer pairs.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam GateImplementation Gate implementation class.
 */
template <class PrecisionT, class GateImplementation>
constexpr auto controlled_generator_op_functor_tuple =
    constructControlledGeneratorOpsFunctorTupleIter<PrecisionT,
                                                    GateImplementation, 0>();

/**
 * @brief Tuple of matrix operation and function pointer pairs
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam GateImplementation Gate implementation class.
 */
template <class PrecisionT, class GateImplementation>
constexpr auto matrix_op_functor_tuple =
    constructMatrixOpsFunctorTupleIter<PrecisionT, GateImplementation, 0>();

/**
 * @brief Tuple of controlled matrix operation and function pointer pairs
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam GateImplementation Gate implementation class.
 */
template <class PrecisionT, class GateImplementation>
constexpr auto controlled_matrix_op_functor_tuple =
    constructControlledMatrixOpsFunctorTupleIter<PrecisionT, GateImplementation,
                                                 0>();

/**
 * @brief Register all implemented gates for a given kernel
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam ParamT Floating point type of gate parameters
 * @tparam GateImplementation Gate implementation class.
 */
template <class PrecisionT, class ParamT, class GateImplementation>
void registerAllImplementedGateOps() {
    auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    auto registerGateToDispatcher = [&dispatcher](
                                        const auto &gate_op_func_pair) {
        const auto &[gate_op, func] = gate_op_func_pair;
        dispatcher.registerGateOperation(gate_op, GateImplementation::kernel_id,
                                         func);
        return gate_op;
    };

    [[maybe_unused]] const auto registered_gate_ops = std::apply(
        [&registerGateToDispatcher](auto... elem) {
            return std::make_tuple(registerGateToDispatcher(elem)...);
        },
        gate_op_functor_tuple<PrecisionT, ParamT, GateImplementation>);
}

/**
 * @brief Register all implemented controlled gates for a given kernel
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam ParamT Floating point type of gate parameters
 * @tparam GateImplementation Gate implementation class.
 */
template <class PrecisionT, class ParamT, class GateImplementation>
void registerAllImplementedControlledGateOps() {
    auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    auto registerControlledGateToDispatcher =
        [&dispatcher](const auto &gate_op_func_pair) {
            const auto &[gate_op, func] = gate_op_func_pair;
            dispatcher.registerControlledGateOperation(
                gate_op, GateImplementation::kernel_id, func);
            return gate_op;
        };

    [[maybe_unused]] const auto registered_gate_ops = std::apply(
        [&registerControlledGateToDispatcher](auto... elem) {
            return std::make_tuple(registerControlledGateToDispatcher(elem)...);
        },
        controlled_gate_op_functor_tuple<PrecisionT, ParamT,
                                         GateImplementation>);
}

/**
 * @brief Register all implemented generators for a given kernel
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam GateImplementation Gate implementation class.
 */
template <class PrecisionT, class GateImplementation>
void registerAllImplementedGeneratorOps() {
    auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    auto registerGeneratorToDispatcher =
        [&dispatcher](const auto &gntr_op_func_pair) {
            const auto &[gntr_op, func] = gntr_op_func_pair;
            dispatcher.registerGeneratorOperation(
                gntr_op, GateImplementation::kernel_id, func);
            return gntr_op;
        };

    [[maybe_unused]] const auto registered_gntr_ops = std::apply(
        [&registerGeneratorToDispatcher](auto... elem) {
            return std::make_tuple(registerGeneratorToDispatcher(elem)...);
        },
        generator_op_functor_tuple<PrecisionT, GateImplementation>);
}

/**
 * @brief Register all implemented controlled generators for a given kernel
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam GateImplementation Gate implementation class.
 */
template <class PrecisionT, class GateImplementation>
void registerAllImplementedControlledGeneratorOps() {
    auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    auto registerControlledGeneratorToDispatcher =
        [&dispatcher](const auto &generator_op_func_pair) {
            const auto &[generator_op, func] = generator_op_func_pair;
            dispatcher.registerControlledGeneratorOperation(
                generator_op, GateImplementation::kernel_id, func);
            return generator_op;
        };

    [[maybe_unused]] const auto registered_generator_ops = std::apply(
        [&registerControlledGeneratorToDispatcher](auto... elem) {
            return std::make_tuple(
                registerControlledGeneratorToDispatcher(elem)...);
        },
        controlled_generator_op_functor_tuple<PrecisionT, GateImplementation>);
}

/**
 * @brief Register all implemented matrix operation for a given kernel
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam GateImplementation Gate implementation class.
 */
template <class PrecisionT, class GateImplementation>
void registerAllImplementedMatrixOps() {
    auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    auto registerMatrixToDispatcher = [&dispatcher](
                                          const auto &mat_op_func_pair) {
        const auto &[mat_op, func] = mat_op_func_pair;
        dispatcher.registerMatrixOperation(mat_op,
                                           GateImplementation::kernel_id, func);
        return mat_op;
    };

    [[maybe_unused]] const auto registered_mat_ops = std::apply(
        [&registerMatrixToDispatcher](auto... elem) {
            return std::make_tuple(registerMatrixToDispatcher(elem)...);
        },
        matrix_op_functor_tuple<PrecisionT, GateImplementation>);
}

/**
 * @brief Register all implemented controlled matrix operation for a given
 * kernel
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam GateImplementation Gate implementation class.
 */
template <class PrecisionT, class GateImplementation>
void registerAllImplementedControlledMatrixOps() {
    auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    auto registerControlledMatrixToDispatcher =
        [&dispatcher](const auto &mat_op_func_pair) {
            const auto &[mat_op, func] = mat_op_func_pair;
            dispatcher.registerControlledMatrixOperation(
                mat_op, GateImplementation::kernel_id, func);
            return mat_op;
        };

    [[maybe_unused]] const auto registered_mat_ops = std::apply(
        [&registerControlledMatrixToDispatcher](auto... elem) {
            return std::make_tuple(
                registerControlledMatrixToDispatcher(elem)...);
        },
        controlled_matrix_op_functor_tuple<PrecisionT, GateImplementation>);
}

/**
 * @brief Internal function to iterate over all available kernels in
 * the compile time
 */
template <class PrecisionT, class ParamT, class GateImplementation>
void registerKernel() {
    registerAllImplementedGateOps<PrecisionT, ParamT, GateImplementation>();
    registerAllImplementedGeneratorOps<PrecisionT, GateImplementation>();
    registerAllImplementedMatrixOps<PrecisionT, GateImplementation>();
    registerAllImplementedControlledGateOps<PrecisionT, ParamT,
                                            GateImplementation>();
    registerAllImplementedControlledGeneratorOps<PrecisionT,
                                                 GateImplementation>();
    registerAllImplementedControlledMatrixOps<PrecisionT, GateImplementation>();

    DynamicDispatcher<PrecisionT>::getInstance().registerKernelName(
        GateImplementation::kernel_id, std::string{GateImplementation::name});
}
} // namespace Pennylane::LightningQubit
/// @endcond
