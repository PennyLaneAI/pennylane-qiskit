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
#include "StateVectorKokkos.hpp"
#include "TypeList.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Util {
template <class StateVector> struct StateVectorToName;

template <> struct StateVectorToName<StateVectorKokkos<float>> {
    constexpr static auto name = "StateVectorKokkos<float>";
};
template <> struct StateVectorToName<StateVectorKokkos<double>> {
    constexpr static auto name = "StateVectorKokkos<double>";
};

using TestStateVectorBackends =
    Pennylane::Util::TypeList<StateVectorKokkos<float>,
                              StateVectorKokkos<double>, void>;
} // namespace Pennylane::LightningKokkos::Util