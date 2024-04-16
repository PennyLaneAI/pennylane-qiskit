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
#include "KernelMap.hpp"

#include "GateOperation.hpp"
#include "IntegerInterval.hpp" // full_domain, in_between_closed, larger_than, larger_than_equal_to, less_than, less_than_equal_to
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
constexpr static auto leq_four = larger_than_equal_to<size_t>(4);

void assignKernelsForGateOp_AVX512(CPUMemoryModel memory_model) {
    auto &instance = OperationKernelMap<GateOperation>::getInstance();

    instance.assignKernelForOp(GateOperation::PauliX, all_threading,
                               memory_model, leq_four, KernelType::AVX512);
    instance.assignKernelForOp(GateOperation::PauliY, all_threading,
                               memory_model, leq_four, KernelType::AVX512);
    instance.assignKernelForOp(GateOperation::PauliZ, all_threading,
                               memory_model, leq_four, KernelType::AVX512);
    instance.assignKernelForOp(GateOperation::Hadamard, all_threading,
                               memory_model, leq_four, KernelType::AVX512);
    instance.assignKernelForOp(GateOperation::S, all_threading, memory_model,
                               leq_four, KernelType::AVX512);
    instance.assignKernelForOp(GateOperation::T, all_threading, memory_model,
                               leq_four, KernelType::AVX512);
    instance.assignKernelForOp(GateOperation::PhaseShift, all_threading,
                               memory_model, leq_four, KernelType::AVX512);
    instance.assignKernelForOp(GateOperation::RX, all_threading, memory_model,
                               leq_four, KernelType::AVX512);
    instance.assignKernelForOp(GateOperation::RY, all_threading, memory_model,
                               leq_four, KernelType::AVX512);
    instance.assignKernelForOp(GateOperation::RZ, all_threading, memory_model,
                               leq_four, KernelType::AVX512);
    instance.assignKernelForOp(GateOperation::Rot, all_threading, memory_model,
                               leq_four, KernelType::AVX512);
    /* Two-qubit gates */
    instance.assignKernelForOp(GateOperation::CZ, all_threading, memory_model,
                               leq_four, KernelType::AVX512);
    instance.assignKernelForOp(GateOperation::CNOT, all_threading, memory_model,
                               leq_four, KernelType::AVX512);
    instance.assignKernelForOp(GateOperation::SWAP, all_threading, memory_model,
                               leq_four, KernelType::AVX512);
    instance.assignKernelForOp(GateOperation::IsingXX, all_threading,
                               memory_model, leq_four, KernelType::AVX512);
    instance.assignKernelForOp(GateOperation::IsingYY, all_threading,
                               memory_model, leq_four, KernelType::AVX512);
    instance.assignKernelForOp(GateOperation::IsingZZ, all_threading,
                               memory_model, leq_four, KernelType::AVX512);
    /* Multi-qubit gates */
}

void assignKernelsForGeneratorOp_AVX512(CPUMemoryModel memory_model) {
    auto &instance = OperationKernelMap<GeneratorOperation>::getInstance();

    instance.assignKernelForOp(GeneratorOperation::RX, all_threading,
                               memory_model, leq_four, KernelType::AVX512);
    instance.assignKernelForOp(GeneratorOperation::RY, all_threading,
                               memory_model, leq_four, KernelType::AVX512);
    instance.assignKernelForOp(GeneratorOperation::RZ, all_threading,
                               memory_model, leq_four, KernelType::AVX512);
}
void assignKernelsForMatrixOp_AVX512(CPUMemoryModel memory_model) {
    auto &instance = OperationKernelMap<MatrixOperation>::getInstance();

    instance.assignKernelForOp(MatrixOperation::SingleQubitOp, all_threading,
                               memory_model, leq_four, KernelType::AVX512);
}
} // namespace Pennylane::LightningQubit::KernelMap::Internal
