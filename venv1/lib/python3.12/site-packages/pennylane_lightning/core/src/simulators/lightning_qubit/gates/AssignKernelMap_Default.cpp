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
#include "AssignKernelMap_Default.hpp"
#include "GateOperation.hpp"
#include "IntegerInterval.hpp" // full_domain, in_between_closed, larger_than, larger_than_equal_to, less_than, less_than_equal_to
#include "KernelMap.hpp"
#include "KernelType.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::LightningQubit::KernelMap;

using Pennylane::Gates::GateOperation;
using Pennylane::Gates::GeneratorOperation;
using Pennylane::Gates::KernelType;
using Pennylane::Gates::MatrixOperation;
using Util::full_domain;
using Util::in_between_closed;
using Util::larger_than;
using Util::larger_than_equal_to;
using Util::less_than;
using Util::less_than_equal_to;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::KernelMap::Internal {
constexpr static auto all_qubit_numbers = full_domain<size_t>();

void assignKernelsForGateOp_Default() {
    auto &instance = OperationKernelMap<GateOperation>::getInstance();

    instance.assignKernelForOp(GateOperation::Identity, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::PauliX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::PauliY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::PauliZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::Hadamard, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::S, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::T, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::PhaseShift, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::RX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::RY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::RZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::Rot, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    /* Two-qubit gates */
    instance.assignKernelForOp(GateOperation::CNOT, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::CY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::CZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::ControlledPhaseShift,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(GateOperation::SWAP, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::IsingXX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::IsingXY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::IsingYY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::IsingZZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::CRX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::CRY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::CRZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::CRot, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);

    /* Three-qubit gates */
    instance.assignKernelForOp(GateOperation::Toffoli, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::CSWAP, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);

    /* QChem gates */
    instance.assignKernelForOp(GateOperation::SingleExcitation, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::SingleExcitationMinus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(GateOperation::SingleExcitationPlus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(GateOperation::DoubleExcitation, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::DoubleExcitationPlus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(GateOperation::DoubleExcitationMinus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);

    /* Multi-qubit gates */
    instance.assignKernelForOp(GateOperation::MultiRZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GateOperation::GlobalPhase, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
}

void assignKernelsForGeneratorOp_Default() {
    auto &instance = OperationKernelMap<GeneratorOperation>::getInstance();

    instance.assignKernelForOp(GeneratorOperation::PhaseShift, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::RX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::RY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::RZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::IsingXX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::IsingXY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::IsingYY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::IsingZZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::CRX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::CRY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::CRZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::ControlledPhaseShift,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);

    instance.assignKernelForOp(GeneratorOperation::SingleExcitation,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::SingleExcitationMinus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::SingleExcitationPlus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::DoubleExcitation,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::DoubleExcitationPlus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::DoubleExcitationMinus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::MultiRZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::GlobalPhase, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
}
void assignKernelsForMatrixOp_Default() {
    auto &instance = OperationKernelMap<MatrixOperation>::getInstance();

    instance.assignKernelForOp(MatrixOperation::SingleQubitOp, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(MatrixOperation::TwoQubitOp, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(MatrixOperation::MultiQubitOp, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
}
void assignKernelsForControlledGateOp_Default() {
    auto &instance = OperationKernelMap<ControlledGateOperation>::getInstance();

    instance.assignKernelForOp(ControlledGateOperation::PauliX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::PauliY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::PauliZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::Hadamard, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::S, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::T, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::PhaseShift,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::RX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::RY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::RZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::Rot, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);

    instance.assignKernelForOp(ControlledGateOperation::SWAP, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::IsingXX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::IsingXY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::IsingYY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::IsingZZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::SingleExcitation,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::SingleExcitationMinus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::SingleExcitationPlus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::DoubleExcitation,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::DoubleExcitationMinus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::DoubleExcitationPlus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    /* Multi-qubit gates */
    instance.assignKernelForOp(ControlledGateOperation::MultiRZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGateOperation::GlobalPhase,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
}
void assignKernelsForControlledGeneratorOp_Default() {
    auto &instance =
        OperationKernelMap<ControlledGeneratorOperation>::getInstance();

    instance.assignKernelForOp(ControlledGeneratorOperation::PhaseShift,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(ControlledGeneratorOperation::RX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGeneratorOperation::RY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGeneratorOperation::RZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(ControlledGeneratorOperation::IsingXX,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(ControlledGeneratorOperation::IsingXY,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(ControlledGeneratorOperation::IsingYY,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(ControlledGeneratorOperation::IsingZZ,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(ControlledGeneratorOperation::SingleExcitation,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(
        ControlledGeneratorOperation::SingleExcitationMinus, all_threading,
        all_memory_model, all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(
        ControlledGeneratorOperation::SingleExcitationPlus, all_threading,
        all_memory_model, all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(ControlledGeneratorOperation::DoubleExcitation,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(
        ControlledGeneratorOperation::DoubleExcitationMinus, all_threading,
        all_memory_model, all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(
        ControlledGeneratorOperation::DoubleExcitationPlus, all_threading,
        all_memory_model, all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(ControlledGeneratorOperation::MultiRZ,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(ControlledGeneratorOperation::GlobalPhase,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
}
void assignKernelsForControlledMatrixOp_Default() {
    auto &instance =
        OperationKernelMap<ControlledMatrixOperation>::getInstance();

    instance.assignKernelForOp(ControlledMatrixOperation::NCSingleQubitOp,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(ControlledMatrixOperation::NCTwoQubitOp,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
    instance.assignKernelForOp(ControlledMatrixOperation::NCMultiQubitOp,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);
}
} // namespace Pennylane::LightningQubit::KernelMap::Internal
