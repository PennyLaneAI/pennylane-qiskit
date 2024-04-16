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
 * Register all gate and generator implementations for X86
 */
#include "RegisterKernels_x64.hpp"
#include "DynamicDispatcher.hpp"
#include "RegisterKernel.hpp"
#include "RuntimeInfo.hpp"
#include "cpu_kernels/GateImplementationsLM.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"

namespace Pennylane::LightningQubit::Internal {
int registerAllAvailableKernels_Float() {
    using Pennylane::Util::RuntimeInfo;
    registerKernel<float, float, Gates::GateImplementationsLM>();
    registerKernel<float, float, Gates::GateImplementationsPI>();

    if (RuntimeInfo::AVX2() && RuntimeInfo::FMA()) {
        registerKernelsAVX2_Float();
    }
    if (RuntimeInfo::AVX512F()) {
        registerKernelsAVX512_Float();
    }
    return 1;
}

int registerAllAvailableKernels_Double() {
    using Pennylane::Util::RuntimeInfo;
    registerKernel<double, double, Gates::GateImplementationsLM>();
    registerKernel<double, double, Gates::GateImplementationsPI>();

    if (RuntimeInfo::AVX2() && RuntimeInfo::FMA()) {
        registerKernelsAVX2_Double();
    }
    if (RuntimeInfo::AVX512F()) {
        registerKernelsAVX512_Double();
    }
    return 1;
}
} // namespace Pennylane::LightningQubit::Internal
