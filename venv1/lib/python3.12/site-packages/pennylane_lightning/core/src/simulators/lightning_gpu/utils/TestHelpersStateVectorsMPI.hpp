// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
/**
 * @file
 * This file defines the necessary functionality to test over LQubit State
 * Vectors.
 */
#include "StateVectorCudaMPI.hpp"
#include "TypeList.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
} // namespace
/// @endcond

namespace Pennylane::LightningGPU::Util {
template <class StateVector> struct StateVectorMPIToName;

template <> struct StateVectorMPIToName<StateVectorCudaMPI<float>> {
    constexpr static auto name = "StateVectorCudaMPI<float>";
};
template <> struct StateVectorMPIToName<StateVectorCudaMPI<double>> {
    constexpr static auto name = "StateVectorCudaMPI<double>";
};

using TestStateVectorMPIBackends =
    Pennylane::Util::TypeList<StateVectorCudaMPI<float>,
                              StateVectorCudaMPI<double>, void>;
} // namespace Pennylane::LightningGPU::Util