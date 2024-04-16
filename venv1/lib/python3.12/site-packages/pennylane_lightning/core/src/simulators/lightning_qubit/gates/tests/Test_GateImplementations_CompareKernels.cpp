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
#include <algorithm>
#include <complex>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "ConstantUtil.hpp" // lookup, array_has_elem
#include "DynamicDispatcher.hpp"
#include "KernelMap.hpp"
#include "KernelType.hpp"
#include "OpToMemberFuncPtr.hpp"
#include "TestHelpers.hpp" // PrecisionToName
#include "TestHelpersWires.hpp"
#include "TestKernels.hpp"
#include "Util.hpp" // for_each_enum

/**
 * @file Test_GateImplementations_Nonparam.cpp
 *
 * This file tests all gate operations (besides matrix) by comparing results
 * between different kernels (gate implementations).
 */
/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::LightningQubit::Util;
using namespace Pennylane::LightningQubit::Gates;
using Pennylane::Util::randomUnitary;
} // namespace
/// @endcond

template <typename PrecisionT>
auto kernelsImplementingGate(GateOperation gate_op) -> std::vector<KernelType> {
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
    auto kernels = dispatcher.registeredKernels();

    std::vector<KernelType> res;

    std::copy_if(
        kernels.begin(), kernels.end(), std::back_inserter(res),
        [&dispatcher, gate_op](KernelType kernel) {
            return dispatcher.registeredGatesForKernel(kernel).contains(
                gate_op);
        });
    return res;
}

template <typename PrecisionT>
auto kernelsImplementingMatrix(MatrixOperation mat_op)
    -> std::vector<KernelType> {
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
    auto kernels = dispatcher.registeredKernels();

    std::vector<KernelType> res;

    std::copy_if(
        kernels.begin(), kernels.end(), std::back_inserter(res),
        [&dispatcher, mat_op](KernelType kernel) {
            return dispatcher.registeredMatricesForKernel(kernel).contains(
                mat_op);
        });
    return res;
}

/**
 * @brief Apply the given gate using all implementing kernels and compare
 * the results.
 */
template <typename PrecisionT, class RandomEngine>
void testApplyGate(RandomEngine &re, GateOperation gate_op, size_t num_qubits) {
    using Pennylane::Gates::Constant::gate_names;

    const auto implementing_kernels =
        kernelsImplementingGate<PrecisionT>(gate_op);
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    std::ostringstream ss;
    ss << "Kernels implementing " << lookup(gate_names, gate_op) << " are ";
    for (KernelType kernel : implementing_kernels) {
        ss << dispatcher.getKernelName(kernel) << ", ";
    }

    INFO(ss.str() << "PrecisionT = " << PrecisionToName<PrecisionT>::value);

    const auto ini = createRandomStateVectorData<PrecisionT>(re, num_qubits);

    const auto params = createParams<PrecisionT>(gate_op);
    const auto all_wires = createAllWires(num_qubits, gate_op, true);

    const bool inverse = GENERATE(true, false);
    for (const auto &wires : all_wires) {
        std::ostringstream ss;
        ss << wires;
        auto wires_str = ss.str();
        INFO(wires_str);

        std::vector<TestVector<std::complex<PrecisionT>>> res;

        // Collect results from all implementing kernels
        for (auto kernel : implementing_kernels) {
            auto st = ini;
            dispatcher.applyOperation(kernel, st.data(), num_qubits, gate_op,
                                      wires, inverse, params);
            res.emplace_back(std::move(st));
        }

        // And compare them
        for (size_t i = 0; i < res.size() - 1; i++) {
            REQUIRE(res[i] ==
                    approx(res[i + 1]).margin(static_cast<PrecisionT>(1e-5)));
        }
    }
}

TEMPLATE_TEST_CASE("Test all kernels give the same results for gates",
                   "[GateImplementations_CompareKernels]", float, double) {
    /* We test all gate operations up to the number of qubits we give */
    constexpr size_t max_num_qubits = 5;
    std::mt19937 re{1337};
    for_each_enum<GateOperation>([&](GateOperation gate_op) {
        const size_t min_num_qubits = [=] {
            if (array_has_elem(Pennylane::Gates::Constant::multi_qubit_gates,
                               gate_op)) {
                return size_t{1};
            }
            return lookup(Pennylane::Gates::Constant::gate_wires, gate_op);
        }();
        for (size_t num_qubits = min_num_qubits; num_qubits <= max_num_qubits;
             num_qubits++) {
            testApplyGate<TestType>(re, gate_op, num_qubits);
        }
    });
}

template <typename PrecisionT, class RandomEngine>
void testMatrixOp(RandomEngine &re, size_t num_qubits, size_t num_wires,
                  bool inverse) {
    using Pennylane::Gates::Constant::matrix_names;
    PL_ASSERT(num_wires > 0);

    const auto mat_op = [num_wires]() -> MatrixOperation {
        switch (num_wires) {
        case 1:
            return MatrixOperation::SingleQubitOp;
        case 2:
            return MatrixOperation::TwoQubitOp;
        default:
            return MatrixOperation::MultiQubitOp;
        }
    }();

    const auto implementing_kernels =
        kernelsImplementingMatrix<PrecisionT>(mat_op);
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
    const auto op_name = lookup(matrix_names, mat_op);

    std::ostringstream ss;
    ss << "Test " << op_name << " with kernels: ";
    for (KernelType kernel : implementing_kernels) {
        ss << dispatcher.getKernelName(kernel) << ", ";
    }
    ss << "inverse: " << inverse;
    ss << ", num_qubits: " << num_qubits;

    const auto ini_st = createRandomStateVectorData<PrecisionT>(re, num_qubits);
    const auto matrix = randomUnitary<PrecisionT>(re, num_qubits);

    DYNAMIC_SECTION(ss.str()) {
        const auto all_wires =
            CombinationGenerator(num_qubits, num_wires).all_perms();
        for (const auto &wires : all_wires) {
            std::vector<TestVector<std::complex<PrecisionT>>> res;

            // Record result from each kerenl
            for (KernelType kernel : implementing_kernels) {
                auto st = ini_st;
                dispatcher.applyMatrix(kernel, st.data(), num_qubits,
                                       matrix.data(), wires, inverse);
                res.emplace_back(std::move(st));
            }

            // And compare them
            for (size_t idx = 0; idx < implementing_kernels.size() - 1; idx++) {
                REQUIRE(res[idx] == approx(res[idx + 1]).margin(1e-7));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Test all kernels give the same results for matrices",
                   "[Test_GateImplementations_CompareKernels]", float, double) {
    std::mt19937 re{1337};

    const size_t num_qubits = 5;

    for (bool inverse : {true, false}) {
        for (size_t num_wires = 1; num_wires <= 5; num_wires++) {
            testMatrixOp<TestType>(re, num_qubits, num_wires, inverse);
        }
    }
}
