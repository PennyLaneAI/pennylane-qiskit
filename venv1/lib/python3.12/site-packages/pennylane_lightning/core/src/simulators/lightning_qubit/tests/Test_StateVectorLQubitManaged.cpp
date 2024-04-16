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
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "CPUMemoryModel.hpp" // getBestAllocator
#include "LinearAlgebra.hpp"  //randomUnitary
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"
#include "TestHelpers.hpp" // createRandomStateVectorData, TestVector
#include "TestHelpersWires.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"

/**
 * @file
 *  Tests for the StateVectorLQubitManaged class.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using Pennylane::Util::createRandomStateVectorData;
using Pennylane::Util::randomUnitary;
using Pennylane::Util::TestVector;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("StateVectorLQubitManaged::StateVectorLQubitManaged",
                   "[StateVectorLQubitManaged]", float, double) {
    using PrecisionT = TestType;
    std::mt19937_64 re{1337};

    SECTION("StateVectorLQubitManaged<TestType> {size_t}") {
        REQUIRE(std::is_constructible_v<StateVectorLQubitManaged<TestType>,
                                        size_t>);
        const size_t num_qubits = 4;
        StateVectorLQubitManaged<PrecisionT> sv(num_qubits);

        REQUIRE(sv.getNumQubits() == 4);
        REQUIRE(sv.getLength() == 16);
        REQUIRE(sv.getDataVector().size() == 16);
    }
    SECTION("StateVectorLQubitManaged<TestType> {size_t}") {
        using TestVectorT = TestVector<std::complex<PrecisionT>>;
        REQUIRE(std::is_constructible_v<StateVectorLQubitManaged<TestType>,
                                        TestVectorT>);
        const size_t num_qubits = 5;
        TestVectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);
        StateVectorLQubitManaged<PrecisionT> sv(st_data);

        REQUIRE(sv.getNumQubits() == 5);
        REQUIRE(sv.getLength() == 32);
        REQUIRE(sv.getDataVector().size() == 32);
    }
    SECTION("StateVectorLQubitManaged<TestType> {const "
            "StateVectorLQubitRaw<TestType>&}") {
        REQUIRE(
            std::is_constructible_v<StateVectorLQubitManaged<TestType>,
                                    const StateVectorLQubitRaw<TestType> &>);
    }
    SECTION("Aligned 256bit statevector") {
        const auto memory_model = CPUMemoryModel::Aligned256;
        StateVectorLQubitManaged<PrecisionT> sv(4, Threading::SingleThread,
                                                memory_model);
        /* Even when we allocate 256 bit aligned memory it is possible that the
         * alignment happens to be 512 bit */
        REQUIRE(((getMemoryModel(sv.getDataVector().data()) ==
                  CPUMemoryModel::Aligned256) ||
                 (getMemoryModel(sv.getDataVector().data()) ==
                  CPUMemoryModel::Aligned512)));
    }

    SECTION("Aligned 512bit statevector") {
        const auto memory_model = CPUMemoryModel::Aligned512;
        StateVectorLQubitManaged<PrecisionT> sv(4, Threading::SingleThread,
                                                memory_model);
        REQUIRE((getMemoryModel(sv.getDataVector().data()) ==
                 CPUMemoryModel::Aligned512));
    }

    SECTION("updateData") {
        using TestVectorT = TestVector<std::complex<PrecisionT>>;
        const size_t num_qubits = 3;
        StateVectorLQubitManaged<PrecisionT> sv(num_qubits);

        TestVectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);
        sv.updateData(st_data);

        REQUIRE(sv.getDataVector() == approx(st_data));
    }
}

TEMPLATE_TEST_CASE("StateVectorLQubitManaged::setBasisState",
                   "[StateVectorLQubitManaged]", float, double) {
    using PrecisionT = TestType;
    using ComplexT = std::complex<PrecisionT>;
    using TestVectorT = TestVector<ComplexT>;
    std::mt19937_64 re{1337};

    const std::size_t num_qubits = 3;

    SECTION("Prepares a single computational basis state.") {
        TestVectorT init_state =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        TestVectorT expected_state(size_t{1U} << num_qubits, 0.0,
                                   getBestAllocator<ComplexT>());
        std::size_t index = GENERATE(0, 1, 2, 3, 4, 5, 6, 7);
        expected_state[index] = {1.0, 0.0};
        StateVectorLQubitManaged<PrecisionT> sv(init_state);
        sv.setBasisState(index);

        REQUIRE(sv.getDataVector() == approx(expected_state));
    }
}

TEMPLATE_TEST_CASE("StateVectorLQubitManaged::SetStateVector",
                   "[StateVectorLQubitManaged]", float, double) {
    using PrecisionT = TestType;
    using ComplexT = std::complex<PrecisionT>;
    using TestVectorT = TestVector<ComplexT>;
    std::mt19937_64 re{1337};

    const std::size_t num_qubits = 3;

    SECTION("Set state vector with values and indices") {
        TestVectorT init_state =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        auto expected_state = init_state;

        for (size_t i = 0; i < Pennylane::Util::exp2(num_qubits - 1); i++) {
            std::swap(expected_state[i * 2], expected_state[i * 2 + 1]);
        }

        std::vector<std::size_t> indices = {0, 2, 4, 6, 1, 3, 5, 7};

        std::vector<ComplexT> values = {
            init_state[1], init_state[3], init_state[5], init_state[7],
            init_state[0], init_state[2], init_state[4], init_state[6]};

        StateVectorLQubitManaged<PrecisionT> sv{num_qubits};
        sv.setStateVector(indices, values);

        REQUIRE(sv.getDataVector() == approx(expected_state));
    }
}
