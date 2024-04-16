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
 * Register all gate and generator implementations
 */
#include "RegisterKernel.hpp"
#include "RegisterKernels_x64.hpp"
#include "cpu_kernels/GateImplementationsAVX512.hpp"

namespace Pennylane::LightningQubit::Internal {
void registerKernelsAVX512_Float() {
    registerKernel<float, float, Gates::GateImplementationsAVX512>();
}
void registerKernelsAVX512_Double() {
    registerKernel<double, double, Gates::GateImplementationsAVX512>();
}
} // namespace Pennylane::LightningQubit::Internal
