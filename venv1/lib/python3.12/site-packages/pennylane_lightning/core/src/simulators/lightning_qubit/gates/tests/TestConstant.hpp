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
#include "Constant.hpp"
#include "ConstantTestHelpers.hpp" // count_unique, first_elems_of, second_elems_of
#include "ConstantUtil.hpp"        // array_has_elem
#include "GateOperation.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Gates;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Gates {
template <typename T, size_t size1, size_t size2>
constexpr auto are_mutually_disjoint(const std::array<T, size1> &arr1,
                                     const std::array<T, size2> &arr2) -> bool {
    return std::all_of(arr1.begin(), arr1.end(), [&arr2](const auto &elem) {
        return !Pennylane::Util::array_has_elem(arr2, elem);
    });
}

/*******************************************************************************
 * Check gate_names is well defined
 ******************************************************************************/

static_assert(Constant::gate_names.size() ==
                  static_cast<size_t>(GateOperation::END) -
                      static_cast<size_t>(GateOperation::BEGIN),
              "Constant gate_names must be defined for all gate operations.");
static_assert(Util::count_unique(Util::first_elems_of(Constant::gate_names)) ==
                  Constant::gate_names.size(),
              "First elements of gate_names must be distinct.");
static_assert(Util::count_unique(Util::second_elems_of(Constant::gate_names)) ==
                  Constant::gate_names.size(),
              "Second elements of gate_names must be distinct.");

/*******************************************************************************
 * Check generator_names is well defined
 ******************************************************************************/

constexpr auto check_generator_names_starts_with() -> bool {
    const auto &arr = Constant::generator_names;
    return std::all_of(arr.begin(), arr.end(), [](const auto &elem) {
        const auto &[gntr_op, gntr_name] = elem;
        return gntr_name.substr(0, 9) == "Generator";
    });
    return true;
}

static_assert(
    Constant::generator_names.size() ==
        static_cast<size_t>(GeneratorOperation::END) -
            static_cast<size_t>(GeneratorOperation::BEGIN),
    "Constant generator_names must be defined for all generator operations.");
static_assert(
    Util::count_unique(Util::first_elems_of(Constant::generator_names)) ==
        Constant::generator_names.size(),
    "First elements of generator_names must be distinct.");
static_assert(
    Util::count_unique(Util::second_elems_of(Constant::generator_names)) ==
        Constant::generator_names.size(),
    "Second elements of generator_names must be distinct.");
static_assert(check_generator_names_starts_with(),
              "Names of generators must start with \"Generator\"");

/*******************************************************************************
 * Check gate_wires is well defined
 ******************************************************************************/

static_assert(Constant::gate_wires.size() ==
                  static_cast<size_t>(GateOperation::END) -
                      static_cast<size_t>(GateOperation::BEGIN) -
                      Constant::multi_qubit_gates.size(),
              "Constant gate_wires must be defined for all gate operations "
              "acting on a fixed number of qubits.");
static_assert(
    are_mutually_disjoint(Util::first_elems_of(Constant::gate_wires),
                          Constant::multi_qubit_gates),
    "Constant gate_wires must not define values for multi-qubit gates.");
static_assert(Util::count_unique(Util::first_elems_of(Constant::gate_wires)) ==
                  Constant::gate_wires.size(),
              "First elements of gate_wires must be distinct.");

/*******************************************************************************
 * Check generator_wires is well defined
 ******************************************************************************/

static_assert(
    Constant::generator_wires.size() ==
        static_cast<size_t>(GeneratorOperation::END) -
            static_cast<size_t>(GeneratorOperation::BEGIN) -
            Constant::multi_qubit_generators.size(),
    "Constant generator_wires must be defined for all generator operations "
    "acting on a fixed number of qubits.");
static_assert(
    are_mutually_disjoint(Util::first_elems_of(Constant::generator_wires),
                          Constant::multi_qubit_generators),
    "Constant generator_wires must not define values for multi-qubit "
    "generators.");
static_assert(
    Util::count_unique(Util::first_elems_of(Constant::generator_wires)) ==
        Constant::generator_wires.size(),
    "First elements of generator_wires must be distinct.");
} // namespace Pennylane::LightningQubit::Gates
