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
#include "ConstantUtil.hpp" // lookup
#include "DynamicDispatcher.hpp"
#include "OpToMemberFuncPtr.hpp"
#include "TestHelpers.hpp"
#include "TestHelpersWires.hpp"
#include "TestKernels.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

#include <complex>
#include <type_traits>
#include <utility>

/**
 * @file
 * We test inverse of each gate operation here. For all gates in
 * implemented_gates, we test wether the state after applying an operation and
 * its inverse is the same as the initial state.
 *
 * Note the we only test generators only when it is included in
 * constexpr member variable implemented_generators.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::Util;
using namespace Pennylane::LightningQubit::Gates;
} // namespace
/// @endcond

template <typename PrecisionT, class RandomEngine>
void testInverseGateKernel(RandomEngine &re, KernelType kernel,
                           GateOperation gate_op, size_t num_qubits) {
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    const auto gate_name = lookup(Constant::gate_names, gate_op);
    const auto kernel_name = dispatcher.getKernelName(kernel);

    DYNAMIC_SECTION("Test inverse of " << gate_name << " for kernel "
                                       << kernel_name) {
        const auto ini_st =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        auto st = ini_st;

        const auto wires = createWires(gate_op, num_qubits);
        const auto params = createParams<PrecisionT>(gate_op);

        dispatcher.applyOperation(kernel, st.data(), num_qubits, gate_op, wires,
                                  false, params);
        dispatcher.applyOperation(kernel, st.data(), num_qubits, gate_op, wires,
                                  true, params);

        REQUIRE(st == approx(ini_st).margin(PrecisionT{1e-7}));
    }
}

template <typename PrecisionT, class RandomEngine>
void testInverseForAllGatesKernel(RandomEngine &re, KernelType kernel,
                                  size_t num_qubits) {
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    for (const auto gate_op : dispatcher.registeredGatesForKernel(kernel)) {
        testInverseGateKernel<PrecisionT>(re, kernel, gate_op, num_qubits);
    }
}

TEMPLATE_TEST_CASE("Test inverse of gate implementations",
                   "[GateImplementations_Inverse]", float, double) {
    using PrecisionT = TestType;
    std::mt19937 re(1337);

    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
    for (const auto kernel : dispatcher.registeredKernels()) {
        testInverseForAllGatesKernel<PrecisionT>(re, kernel, 5);
    }
}
