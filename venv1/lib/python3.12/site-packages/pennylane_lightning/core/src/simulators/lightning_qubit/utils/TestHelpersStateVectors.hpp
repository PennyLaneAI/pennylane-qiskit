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
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"
#include "TypeList.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Util {
template <class StateVector> struct StateVectorToName;

template <> struct StateVectorToName<StateVectorLQubitManaged<float>> {
    constexpr static auto name = "StateVectorLQubitManaged<float>";
};
template <> struct StateVectorToName<StateVectorLQubitManaged<double>> {
    constexpr static auto name = "StateVectorLQubitManaged<double>";
};
template <> struct StateVectorToName<StateVectorLQubitRaw<float>> {
    constexpr static auto name = "StateVectorLQubitRaw<float>";
};
template <> struct StateVectorToName<StateVectorLQubitRaw<double>> {
    constexpr static auto name = "StateVectorLQubitRaw<double>";
};

using TestStateVectorBackends = Pennylane::Util::TypeList<
    StateVectorLQubitManaged<float>, StateVectorLQubitManaged<double>,
    StateVectorLQubitRaw<float>, StateVectorLQubitRaw<double>, void>;
} // namespace Pennylane::LightningQubit::Util