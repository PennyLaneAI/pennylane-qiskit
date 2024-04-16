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
 * @file GateOperation.hpp
 * Defines possible operations.
 */
#pragma once
#include <cstdint>
#include <string>
#include <utility>

namespace Pennylane::Gates {
/**
 * @brief Enum class for all gate operations
 */
enum class GateOperation : uint32_t {
    BEGIN = 0,
    /* Single-qubit gates */
    Identity = 0,
    PauliX,
    PauliY,
    PauliZ,
    Hadamard,
    S,
    T,
    PhaseShift,
    RX,
    RY,
    RZ,
    Rot,
    /* Two-qubit gates */
    CNOT,
    CY,
    CZ,
    SWAP,
    IsingXX,
    IsingXY,
    IsingYY,
    IsingZZ,
    ControlledPhaseShift,
    CRX,
    CRY,
    CRZ,
    CRot,
    SingleExcitation,
    SingleExcitationMinus,
    SingleExcitationPlus,
    /* Three-qubit gates */
    Toffoli,
    CSWAP,
    /* Four-qubit gates */
    DoubleExcitation,
    DoubleExcitationMinus,
    DoubleExcitationPlus,
    /* Multi-qubit gates */
    MultiRZ,
    GlobalPhase,
    /* END (placeholder) */
    END
};

enum class ControlledGateOperation : uint32_t {
    BEGIN = 0,
    /* Single-qubit gates */
    PauliX = 0,
    PauliY,
    PauliZ,
    Hadamard,
    S,
    T,
    PhaseShift,
    RX,
    RY,
    RZ,
    Rot,
    /* Two-qubit gates */
    SWAP,
    IsingXX,
    IsingXY,
    IsingYY,
    IsingZZ,
    SingleExcitation,
    SingleExcitationMinus,
    SingleExcitationPlus,
    DoubleExcitation,
    DoubleExcitationMinus,
    DoubleExcitationPlus,
    /* Multi-qubit gates */
    MultiRZ,
    GlobalPhase,
    /* END (placeholder) */
    END
};

/**
 * @brief Enum class for all gate generators
 */
enum class GeneratorOperation : uint32_t {
    BEGIN = 0,
    /* Gate generators (only used internally for adjoint diff) */
    PhaseShift = 0,
    RX,
    RY,
    RZ,
    IsingXX,
    IsingXY,
    IsingYY,
    IsingZZ,
    CRX,
    CRY,
    CRZ,
    ControlledPhaseShift,
    SingleExcitation,
    SingleExcitationMinus,
    SingleExcitationPlus,
    DoubleExcitation,
    DoubleExcitationMinus,
    DoubleExcitationPlus,
    MultiRZ,
    GlobalPhase,
    /* END (placeholder) */
    END
};

enum class ControlledGeneratorOperation : uint32_t {
    BEGIN = 0,
    PhaseShift = 0,
    RX,
    RY,
    RZ,
    IsingXX,
    IsingXY,
    IsingYY,
    IsingZZ,
    SingleExcitation,
    SingleExcitationMinus,
    SingleExcitationPlus,
    DoubleExcitation,
    DoubleExcitationMinus,
    DoubleExcitationPlus,
    MultiRZ,
    GlobalPhase,
    /* END (placeholder) */
    END
};

/**
 * @brief Enum class for matrix operation
 */
enum class MatrixOperation : uint32_t {
    BEGIN = 0,
    SingleQubitOp = 0,
    TwoQubitOp,
    MultiQubitOp,
    /* END (placeholder) */
    END
};

enum class ControlledMatrixOperation : uint32_t {
    BEGIN = 0,
    NCSingleQubitOp = 0,
    NCTwoQubitOp,
    NCMultiQubitOp,
    /* END (placeholder) */
    END
};

} // namespace Pennylane::Gates
