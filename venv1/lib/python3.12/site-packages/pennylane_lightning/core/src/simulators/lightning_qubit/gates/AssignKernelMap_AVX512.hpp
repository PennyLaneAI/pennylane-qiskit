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

#include "CPUMemoryModel.hpp"

/// @cond DEV
namespace {
using Pennylane::Util::CPUMemoryModel;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::KernelMap::Internal {
void assignKernelsForGateOp_AVX512(CPUMemoryModel);
void assignKernelsForGeneratorOp_AVX512(CPUMemoryModel);
void assignKernelsForMatrixOp_AVX512(CPUMemoryModel);
} // namespace Pennylane::LightningQubit::KernelMap::Internal
