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
 * @file Constant.hpp
 * Defines all constants for statevector
 */
#pragma once

#include <array>

#include "GateOperation.hpp"
#include "TypeList.hpp"

namespace Pennylane::Gates::Constant {
/**
 * @brief List of multi-qubit gates
 */
[[maybe_unused]] constexpr std::array multi_qubit_gates{GateOperation::MultiRZ};
[[maybe_unused]] constexpr std::array controlled_multi_qubit_gates{
    ControlledGateOperation::MultiRZ};
/**
 * @brief List of multi-qubit generators
 */
[[maybe_unused]] constexpr std::array multi_qubit_generators{
    GeneratorOperation::MultiRZ,
};
[[maybe_unused]] constexpr std::array controlled_multi_qubit_generators{
    ControlledGeneratorOperation::MultiRZ,
};
/**
 * @brief List of multi-qubit matrix operation
 */
[[maybe_unused]] constexpr std::array multi_qubit_matrix_ops{
    MatrixOperation::MultiQubitOp,
};

/**
 * @brief Gate names
 */
using GateView = typename std::pair<GateOperation, std::string_view>;
[[maybe_unused]] constexpr std::array gate_names = {
    GateView{GateOperation::Identity, "Identity"},
    GateView{GateOperation::PauliX, "PauliX"},
    GateView{GateOperation::PauliY, "PauliY"},
    GateView{GateOperation::PauliZ, "PauliZ"},
    GateView{GateOperation::Hadamard, "Hadamard"},
    GateView{GateOperation::S, "S"},
    GateView{GateOperation::T, "T"},
    GateView{GateOperation::PhaseShift, "PhaseShift"},
    GateView{GateOperation::RX, "RX"},
    GateView{GateOperation::RY, "RY"},
    GateView{GateOperation::RZ, "RZ"},
    GateView{GateOperation::Rot, "Rot"},
    GateView{GateOperation::CNOT, "CNOT"},
    GateView{GateOperation::CY, "CY"},
    GateView{GateOperation::CZ, "CZ"},
    GateView{GateOperation::IsingXX, "IsingXX"},
    GateView{GateOperation::IsingXY, "IsingXY"},
    GateView{GateOperation::IsingYY, "IsingYY"},
    GateView{GateOperation::IsingZZ, "IsingZZ"},
    GateView{GateOperation::SWAP, "SWAP"},
    GateView{GateOperation::ControlledPhaseShift, "ControlledPhaseShift"},
    GateView{GateOperation::CRX, "CRX"},
    GateView{GateOperation::CRY, "CRY"},
    GateView{GateOperation::CRZ, "CRZ"},
    GateView{GateOperation::CRot, "CRot"},
    GateView{GateOperation::SingleExcitation, "SingleExcitation"},
    GateView{GateOperation::SingleExcitationMinus, "SingleExcitationMinus"},
    GateView{GateOperation::SingleExcitationPlus, "SingleExcitationPlus"},
    GateView{GateOperation::Toffoli, "Toffoli"},
    GateView{GateOperation::CSWAP, "CSWAP"},
    GateView{GateOperation::DoubleExcitation, "DoubleExcitation"},
    GateView{GateOperation::DoubleExcitationMinus, "DoubleExcitationMinus"},
    GateView{GateOperation::DoubleExcitationPlus, "DoubleExcitationPlus"},
    GateView{GateOperation::MultiRZ, "MultiRZ"},
    GateView{GateOperation::GlobalPhase, "GlobalPhase"}};

using CGateView = typename std::pair<ControlledGateOperation, std::string_view>;
[[maybe_unused]] constexpr std::array controlled_gate_names = {
    CGateView{ControlledGateOperation::PauliX, "PauliX"},
    CGateView{ControlledGateOperation::PauliY, "PauliY"},
    CGateView{ControlledGateOperation::PauliZ, "PauliZ"},
    CGateView{ControlledGateOperation::Hadamard, "Hadamard"},
    CGateView{ControlledGateOperation::S, "S"},
    CGateView{ControlledGateOperation::T, "T"},
    CGateView{ControlledGateOperation::PhaseShift, "PhaseShift"},
    CGateView{ControlledGateOperation::RX, "RX"},
    CGateView{ControlledGateOperation::RY, "RY"},
    CGateView{ControlledGateOperation::RZ, "RZ"},
    CGateView{ControlledGateOperation::Rot, "Rot"},
    CGateView{ControlledGateOperation::SWAP, "SWAP"},
    CGateView{ControlledGateOperation::IsingXX, "IsingXX"},
    CGateView{ControlledGateOperation::IsingXY, "IsingXY"},
    CGateView{ControlledGateOperation::IsingYY, "IsingYY"},
    CGateView{ControlledGateOperation::IsingZZ, "IsingZZ"},
    CGateView{ControlledGateOperation::SingleExcitation, "SingleExcitation"},
    CGateView{ControlledGateOperation::SingleExcitationMinus,
              "SingleExcitationMinus"},
    CGateView{ControlledGateOperation::SingleExcitationPlus,
              "SingleExcitationPlus"},
    CGateView{ControlledGateOperation::DoubleExcitation, "DoubleExcitation"},
    CGateView{ControlledGateOperation::DoubleExcitationMinus,
              "DoubleExcitationMinus"},
    CGateView{ControlledGateOperation::DoubleExcitationPlus,
              "DoubleExcitationPlus"},
    CGateView{ControlledGateOperation::MultiRZ, "MultiRZ"},
    CGateView{ControlledGateOperation::GlobalPhase, "GlobalPhase"},
};

/**
 * @brief Generator names.
 *
 * Note that a name of generators must be "Generator" +
 * the name of the corresponding gate.
 */
using GeneratorView = typename std::pair<GeneratorOperation, std::string_view>;
[[maybe_unused]] constexpr std::array generator_names = {
    GeneratorView{GeneratorOperation::PhaseShift, "GeneratorPhaseShift"},
    GeneratorView{GeneratorOperation::RX, "GeneratorRX"},
    GeneratorView{GeneratorOperation::RY, "GeneratorRY"},
    GeneratorView{GeneratorOperation::RZ, "GeneratorRZ"},
    GeneratorView{GeneratorOperation::CRX, "GeneratorCRX"},
    GeneratorView{GeneratorOperation::CRY, "GeneratorCRY"},
    GeneratorView{GeneratorOperation::CRZ, "GeneratorCRZ"},
    GeneratorView{GeneratorOperation::IsingXX, "GeneratorIsingXX"},
    GeneratorView{GeneratorOperation::IsingXY, "GeneratorIsingXY"},
    GeneratorView{GeneratorOperation::IsingYY, "GeneratorIsingYY"},
    GeneratorView{GeneratorOperation::IsingZZ, "GeneratorIsingZZ"},
    GeneratorView{GeneratorOperation::ControlledPhaseShift,
                  "GeneratorControlledPhaseShift"},
    GeneratorView{GeneratorOperation::SingleExcitation,
                  "GeneratorSingleExcitation"},
    GeneratorView{GeneratorOperation::SingleExcitationMinus,
                  "GeneratorSingleExcitationMinus"},
    GeneratorView{GeneratorOperation::SingleExcitationPlus,
                  "GeneratorSingleExcitationPlus"},
    GeneratorView{GeneratorOperation::MultiRZ, "GeneratorMultiRZ"},
    GeneratorView{GeneratorOperation::DoubleExcitation,
                  "GeneratorDoubleExcitation"},
    GeneratorView{GeneratorOperation::DoubleExcitationMinus,
                  "GeneratorDoubleExcitationMinus"},
    GeneratorView{GeneratorOperation::DoubleExcitationPlus,
                  "GeneratorDoubleExcitationPlus"},
    GeneratorView{GeneratorOperation::GlobalPhase, "GeneratorGlobalPhase"},
};

using CGeneratorView =
    typename std::pair<ControlledGeneratorOperation, std::string_view>;
[[maybe_unused]] constexpr std::array controlled_generator_names = {
    CGeneratorView{ControlledGeneratorOperation::PhaseShift, "PhaseShift"},
    CGeneratorView{ControlledGeneratorOperation::RX, "RX"},
    CGeneratorView{ControlledGeneratorOperation::RY, "RY"},
    CGeneratorView{ControlledGeneratorOperation::RZ, "RZ"},
    CGeneratorView{ControlledGeneratorOperation::IsingXX, "IsingXX"},
    CGeneratorView{ControlledGeneratorOperation::IsingXY, "IsingXY"},
    CGeneratorView{ControlledGeneratorOperation::IsingYY, "IsingYY"},
    CGeneratorView{ControlledGeneratorOperation::IsingZZ, "IsingZZ"},
    CGeneratorView{ControlledGeneratorOperation::SingleExcitation,
                   "SingleExcitation"},
    CGeneratorView{ControlledGeneratorOperation::SingleExcitationMinus,
                   "SingleExcitationMinus"},
    CGeneratorView{ControlledGeneratorOperation::SingleExcitationPlus,
                   "SingleExcitationPlus"},
    CGeneratorView{ControlledGeneratorOperation::DoubleExcitation,
                   "DoubleExcitation"},
    CGeneratorView{ControlledGeneratorOperation::DoubleExcitationMinus,
                   "DoubleExcitationMinus"},
    CGeneratorView{ControlledGeneratorOperation::DoubleExcitationPlus,
                   "DoubleExcitationPlus"},
    CGeneratorView{ControlledGeneratorOperation::MultiRZ, "MultiRZ"},
    CGeneratorView{ControlledGeneratorOperation::GlobalPhase, "GlobalPhase"},
};

/**
 * @brief Matrix names.
 */
using MatrixView = typename std::pair<MatrixOperation, std::string_view>;
[[maybe_unused]] constexpr std::array matrix_names = {
    MatrixView{MatrixOperation::SingleQubitOp, "SingleQubitOp"},
    MatrixView{MatrixOperation::TwoQubitOp, "TwoQubitOp"},
    MatrixView{MatrixOperation::MultiQubitOp, "MultiQubitOp"},
};

using CMatrixView =
    typename std::pair<ControlledMatrixOperation, std::string_view>;
[[maybe_unused]] constexpr std::array controlled_matrix_names = {
    CMatrixView{ControlledMatrixOperation::NCSingleQubitOp, "NCSingleQubitOp"},
    CMatrixView{ControlledMatrixOperation::NCTwoQubitOp, "NCTwoQubitOp"},
    CMatrixView{ControlledMatrixOperation::NCMultiQubitOp, "NCMultiQubitOp"},
};

/**
 * @brief Number of wires for gates besides multi-qubit gates
 */
using GateNWires = typename std::pair<GateOperation, size_t>;
[[maybe_unused]] constexpr std::array gate_wires = {
    GateNWires{GateOperation::Identity, 1},
    GateNWires{GateOperation::PauliX, 1},
    GateNWires{GateOperation::PauliY, 1},
    GateNWires{GateOperation::PauliZ, 1},
    GateNWires{GateOperation::Hadamard, 1},
    GateNWires{GateOperation::S, 1},
    GateNWires{GateOperation::T, 1},
    GateNWires{GateOperation::PhaseShift, 1},
    GateNWires{GateOperation::RX, 1},
    GateNWires{GateOperation::RY, 1},
    GateNWires{GateOperation::RZ, 1},
    GateNWires{GateOperation::Rot, 1},
    GateNWires{GateOperation::CNOT, 2},
    GateNWires{GateOperation::CY, 2},
    GateNWires{GateOperation::CZ, 2},
    GateNWires{GateOperation::SWAP, 2},
    GateNWires{GateOperation::IsingXX, 2},
    GateNWires{GateOperation::IsingXY, 2},
    GateNWires{GateOperation::IsingYY, 2},
    GateNWires{GateOperation::IsingZZ, 2},
    GateNWires{GateOperation::ControlledPhaseShift, 2},
    GateNWires{GateOperation::CRX, 2},
    GateNWires{GateOperation::CRY, 2},
    GateNWires{GateOperation::CRZ, 2},
    GateNWires{GateOperation::CRot, 2},
    GateNWires{GateOperation::SingleExcitation, 2},
    GateNWires{GateOperation::SingleExcitationMinus, 2},
    GateNWires{GateOperation::SingleExcitationPlus, 2},
    GateNWires{GateOperation::Toffoli, 3},
    GateNWires{GateOperation::CSWAP, 3},
    GateNWires{GateOperation::DoubleExcitation, 4},
    GateNWires{GateOperation::DoubleExcitationMinus, 4},
    GateNWires{GateOperation::DoubleExcitationPlus, 4},
    GateNWires{GateOperation::GlobalPhase, 1},
};

using CGateNWires = typename std::pair<ControlledGateOperation, size_t>;
[[maybe_unused]] constexpr std::array controlled_gate_wires = {
    CGateNWires{ControlledGateOperation::PauliX, 1},
    CGateNWires{ControlledGateOperation::PauliY, 1},
    CGateNWires{ControlledGateOperation::PauliZ, 1},
    CGateNWires{ControlledGateOperation::Hadamard, 1},
    CGateNWires{ControlledGateOperation::S, 1},
    CGateNWires{ControlledGateOperation::T, 1},
    CGateNWires{ControlledGateOperation::PhaseShift, 1},
    CGateNWires{ControlledGateOperation::RX, 1},
    CGateNWires{ControlledGateOperation::RY, 1},
    CGateNWires{ControlledGateOperation::RZ, 1},
    CGateNWires{ControlledGateOperation::Rot, 1},
    CGateNWires{ControlledGateOperation::SWAP, 2},
    CGateNWires{ControlledGateOperation::IsingXX, 2},
    CGateNWires{ControlledGateOperation::IsingXY, 2},
    CGateNWires{ControlledGateOperation::IsingYY, 2},
    CGateNWires{ControlledGateOperation::IsingZZ, 2},
    CGateNWires{ControlledGateOperation::SingleExcitation, 2},
    CGateNWires{ControlledGateOperation::SingleExcitationMinus, 2},
    CGateNWires{ControlledGateOperation::SingleExcitationPlus, 2},
    CGateNWires{ControlledGateOperation::DoubleExcitation, 4},
    CGateNWires{ControlledGateOperation::DoubleExcitationMinus, 4},
    CGateNWires{ControlledGateOperation::DoubleExcitationPlus, 4},
    CGateNWires{ControlledGateOperation::GlobalPhase, 1},
};

/**
 * @brief Number of wires for generators besides multi-qubit gates
 */
using GeneratorNWires = typename std::pair<GeneratorOperation, size_t>;
[[maybe_unused]] constexpr std::array generator_wires = {
    GeneratorNWires{GeneratorOperation::PhaseShift, 1},
    GeneratorNWires{GeneratorOperation::RX, 1},
    GeneratorNWires{GeneratorOperation::RY, 1},
    GeneratorNWires{GeneratorOperation::RZ, 1},
    GeneratorNWires{GeneratorOperation::IsingXX, 2},
    GeneratorNWires{GeneratorOperation::IsingXY, 2},
    GeneratorNWires{GeneratorOperation::IsingYY, 2},
    GeneratorNWires{GeneratorOperation::IsingZZ, 2},
    GeneratorNWires{GeneratorOperation::CRX, 2},
    GeneratorNWires{GeneratorOperation::CRY, 2},
    GeneratorNWires{GeneratorOperation::CRZ, 2},
    GeneratorNWires{GeneratorOperation::SingleExcitation, 2},
    GeneratorNWires{GeneratorOperation::SingleExcitationMinus, 2},
    GeneratorNWires{GeneratorOperation::SingleExcitationPlus, 2},
    GeneratorNWires{GeneratorOperation::ControlledPhaseShift, 2},
    GeneratorNWires{GeneratorOperation::DoubleExcitation, 4},
    GeneratorNWires{GeneratorOperation::DoubleExcitationMinus, 4},
    GeneratorNWires{GeneratorOperation::DoubleExcitationPlus, 4},
    GeneratorNWires{GeneratorOperation::GlobalPhase, 1},
};

using CGeneratorNWires =
    typename std::pair<ControlledGeneratorOperation, size_t>;
[[maybe_unused]] constexpr std::array controlled_generator_wires = {
    CGeneratorNWires{ControlledGeneratorOperation::PhaseShift, 1},
    CGeneratorNWires{ControlledGeneratorOperation::RX, 1},
    CGeneratorNWires{ControlledGeneratorOperation::RY, 1},
    CGeneratorNWires{ControlledGeneratorOperation::RZ, 1},
    CGeneratorNWires{ControlledGeneratorOperation::IsingXX, 2},
    CGeneratorNWires{ControlledGeneratorOperation::IsingXY, 2},
    CGeneratorNWires{ControlledGeneratorOperation::IsingYY, 2},
    CGeneratorNWires{ControlledGeneratorOperation::IsingZZ, 2},
    CGeneratorNWires{ControlledGeneratorOperation::SingleExcitation, 2},
    CGeneratorNWires{ControlledGeneratorOperation::SingleExcitationMinus, 2},
    CGeneratorNWires{ControlledGeneratorOperation::SingleExcitationPlus, 2},
    CGeneratorNWires{ControlledGeneratorOperation::DoubleExcitation, 4},
    CGeneratorNWires{ControlledGeneratorOperation::DoubleExcitationMinus, 4},
    CGeneratorNWires{ControlledGeneratorOperation::DoubleExcitationPlus, 4},
    CGeneratorNWires{ControlledGeneratorOperation::GlobalPhase, 1},
};

/**
 * @brief Number of parameters for gates
 */
using GateNParams = typename std::pair<GateOperation, size_t>;
[[maybe_unused]] constexpr std::array gate_num_params = {
    GateNParams{GateOperation::Identity, 0},
    GateNParams{GateOperation::PauliX, 0},
    GateNParams{GateOperation::PauliY, 0},
    GateNParams{GateOperation::PauliZ, 0},
    GateNParams{GateOperation::Hadamard, 0},
    GateNParams{GateOperation::S, 0},
    GateNParams{GateOperation::T, 0},
    GateNParams{GateOperation::PhaseShift, 1},
    GateNParams{GateOperation::RX, 1},
    GateNParams{GateOperation::RY, 1},
    GateNParams{GateOperation::RZ, 1},
    GateNParams{GateOperation::Rot, 3},
    GateNParams{GateOperation::CNOT, 0},
    GateNParams{GateOperation::CY, 0},
    GateNParams{GateOperation::CZ, 0},
    GateNParams{GateOperation::SWAP, 0},
    GateNParams{GateOperation::IsingXX, 1},
    GateNParams{GateOperation::IsingXY, 1},
    GateNParams{GateOperation::IsingYY, 1},
    GateNParams{GateOperation::IsingZZ, 1},
    GateNParams{GateOperation::ControlledPhaseShift, 1},
    GateNParams{GateOperation::CRX, 1},
    GateNParams{GateOperation::CRY, 1},
    GateNParams{GateOperation::CRZ, 1},
    GateNParams{GateOperation::SingleExcitation, 1},
    GateNParams{GateOperation::SingleExcitationMinus, 1},
    GateNParams{GateOperation::SingleExcitationPlus, 1},
    GateNParams{GateOperation::CRot, 3},
    GateNParams{GateOperation::Toffoli, 0},
    GateNParams{GateOperation::DoubleExcitation, 1},
    GateNParams{GateOperation::DoubleExcitationMinus, 1},
    GateNParams{GateOperation::DoubleExcitationPlus, 1},
    GateNParams{GateOperation::CSWAP, 0},
    GateNParams{GateOperation::MultiRZ, 1},
    GateNParams{GateOperation::GlobalPhase, 1},
};

/**
 * @brief Number of parameters for gates
 */
using CGateNParams = typename std::pair<ControlledGateOperation, size_t>;
[[maybe_unused]] constexpr std::array controlled_gate_num_params = {
    CGateNParams{ControlledGateOperation::PauliX, 0},
    CGateNParams{ControlledGateOperation::PauliY, 0},
    CGateNParams{ControlledGateOperation::PauliZ, 0},
    CGateNParams{ControlledGateOperation::Hadamard, 0},
    CGateNParams{ControlledGateOperation::S, 0},
    CGateNParams{ControlledGateOperation::T, 0},
    CGateNParams{ControlledGateOperation::PhaseShift, 1},
    CGateNParams{ControlledGateOperation::RX, 1},
    CGateNParams{ControlledGateOperation::RY, 1},
    CGateNParams{ControlledGateOperation::RZ, 1},
    CGateNParams{ControlledGateOperation::Rot, 3},
    CGateNParams{ControlledGateOperation::SWAP, 0},
    CGateNParams{ControlledGateOperation::IsingXX, 1},
    CGateNParams{ControlledGateOperation::IsingXY, 1},
    CGateNParams{ControlledGateOperation::IsingYY, 1},
    CGateNParams{ControlledGateOperation::IsingZZ, 1},
    CGateNParams{ControlledGateOperation::SingleExcitation, 1},
    CGateNParams{ControlledGateOperation::SingleExcitationMinus, 1},
    CGateNParams{ControlledGateOperation::SingleExcitationPlus, 1},
    CGateNParams{ControlledGateOperation::DoubleExcitation, 1},
    CGateNParams{ControlledGateOperation::DoubleExcitationMinus, 1},
    CGateNParams{ControlledGateOperation::DoubleExcitationPlus, 1},
    CGateNParams{ControlledGateOperation::MultiRZ, 1},
    CGateNParams{ControlledGateOperation::GlobalPhase, 1},
};
} // namespace Pennylane::Gates::Constant
