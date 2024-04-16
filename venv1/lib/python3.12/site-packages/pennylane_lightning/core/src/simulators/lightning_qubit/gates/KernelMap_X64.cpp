
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
 * Assign kernel map
 */

#include "AssignKernelMap_AVX2.hpp"
#include "AssignKernelMap_AVX512.hpp"
#include "AssignKernelMap_Default.hpp"
#include "KernelMap.hpp"
#include "RuntimeInfo.hpp"

/// @cond DEV
namespace {
using Pennylane::Util::RuntimeInfo;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::KernelMap::Internal {
int assignKernelsForGateOp() {
    assignKernelsForGateOp_Default();

    if (RuntimeInfo::AVX2() && RuntimeInfo::FMA()) {
        assignKernelsForGateOp_AVX2(CPUMemoryModel::Aligned256);
        // LCOV_EXCL_START
        if (!RuntimeInfo::AVX512F()) {
            assignKernelsForGateOp_AVX2(CPUMemoryModel::Aligned512);
        }
        // LCOV_EXCL_STOP
    }
    // LCOV_EXCL_START
    if (RuntimeInfo::AVX512F()) {
        assignKernelsForGateOp_AVX512(CPUMemoryModel::Aligned512);
    }
    // LCOV_EXCL_STOP
    return 1;
}
int assignKernelsForGeneratorOp() {
    assignKernelsForGeneratorOp_Default();

    if (RuntimeInfo::AVX2() && RuntimeInfo::FMA()) {
        assignKernelsForGeneratorOp_AVX2(CPUMemoryModel::Aligned256);
        // LCOV_EXCL_START
        if (!RuntimeInfo::AVX512F()) {
            assignKernelsForGeneratorOp_AVX2(CPUMemoryModel::Aligned512);
        }
        // LCOV_EXCL_STOP
    }
    // LCOV_EXCL_START
    if (RuntimeInfo::AVX512F()) {
        assignKernelsForGeneratorOp_AVX512(CPUMemoryModel::Aligned512);
    }
    // LCOV_EXCL_STOP
    return 1;
}
int assignKernelsForMatrixOp() {
    assignKernelsForMatrixOp_Default();

    if (RuntimeInfo::AVX2() && RuntimeInfo::FMA()) {
        assignKernelsForMatrixOp_AVX2(CPUMemoryModel::Aligned256);
        // LCOV_EXCL_START
        if (!RuntimeInfo::AVX512F()) {
            assignKernelsForMatrixOp_AVX2(CPUMemoryModel::Aligned512);
        }
        // LCOV_EXCL_STOP
    }
    // LCOV_EXCL_START
    if (RuntimeInfo::AVX512F()) {
        assignKernelsForMatrixOp_AVX512(CPUMemoryModel::Aligned512);
    }
    // LCOV_EXCL_STOP
    return 1;
}
int assignKernelsForControlledGateOp() {
    assignKernelsForControlledGateOp_Default();
    return 1;
}
int assignKernelsForControlledGeneratorOp() {
    assignKernelsForControlledGeneratorOp_Default();
    return 1;
}
int assignKernelsForControlledMatrixOp() {
    assignKernelsForControlledMatrixOp_Default();
    return 1;
}
} // namespace Pennylane::LightningQubit::KernelMap::Internal
