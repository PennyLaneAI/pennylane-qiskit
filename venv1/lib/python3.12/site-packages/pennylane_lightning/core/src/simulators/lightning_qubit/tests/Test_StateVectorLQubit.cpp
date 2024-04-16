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
#include <limits> // numeric_limits
#include <random>
#include <type_traits>
#include <vector>

#include <catch2/catch.hpp>

#include "LinearAlgebra.hpp" //randomUnitary
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"
#include "TestHelpers.hpp" // createRandomStateVectorData
#include "TestHelpersWires.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"

/**
 * @file
 *  Tests for functionality:
 *      - defined in the intermediate base class StateVectorLQubit.
 *      - shared between all child classes.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using Pennylane::Util::randomUnitary;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("StateVectorLQubit::Constructibility",
                   "[Default Constructibility]", StateVectorLQubitRaw<>,
                   StateVectorLQubitManaged<>) {
    SECTION("StateVectorBackend<>") {
        REQUIRE(!std::is_constructible_v<TestType>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorLQubit::Constructibility",
                           "[General Constructibility]",
                           (StateVectorLQubitManaged, StateVectorLQubitRaw),
                           (float, double)) {
    using StateVectorT = TestType;
    using ComplexT = typename StateVectorT::ComplexT;

    SECTION("StateVectorBackend<TestType>") {
        REQUIRE(!std::is_constructible_v<StateVectorT>);
    }
    SECTION("StateVectorBackend<TestType> {ComplexT*, size_t}") {
        REQUIRE(std::is_constructible_v<StateVectorT, ComplexT *, size_t>);
    }
    SECTION("StateVectorBackend<TestType> {ComplexT*, size_t}: Fails if "
            "provided an inconsistent length.") {
        std::vector<ComplexT> st_data(14, 0.0);
        REQUIRE_THROWS_WITH(
            StateVectorT(st_data.data(), st_data.size()),
            Catch::Contains("The size of provided data must be a power of 2."));
    }
    SECTION(
        "StateVectorBackend<TestType> {const StateVectorBackend<TestType>&}") {
        REQUIRE(std::is_copy_constructible_v<StateVectorT>);
    }
    SECTION("StateVectorBackend<TestType> {StateVectorBackend<TestType>&&}") {
        REQUIRE(std::is_move_constructible_v<StateVectorT>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorLQubit::applyMatrix with a std::vector",
                           "[applyMatrix]",
                           (StateVectorLQubitManaged, StateVectorLQubitRaw),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using VectorT = TestVector<ComplexT>;
    std::mt19937_64 re{1337};

    SECTION("Test wrong matrix size") {
        std::vector<ComplexT> m(7, 0.0);
        const size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(st_data.data(), st_data.size());
        REQUIRE_THROWS_WITH(
            state_vector.applyMatrix(m, {0, 1}),
            Catch::Contains(
                "The size of matrix does not match with the given"));
    }

    SECTION("Test wrong number of wires") {
        std::vector<ComplexT> m(8, 0.0);
        const size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(st_data.data(), st_data.size());
        REQUIRE_THROWS_WITH(
            state_vector.applyMatrix(m, {0}),
            Catch::Contains(
                "The size of matrix does not match with the given"));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorLQubit::applyMatrix with a pointer",
                           "[applyMatrix]",
                           (StateVectorLQubitManaged, StateVectorLQubitRaw),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using VectorT = TestVector<ComplexT>;
    std::mt19937_64 re{1337};

    SECTION("Test wrong matrix") {
        std::vector<ComplexT> m(8, 0.0);
        const size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(st_data.data(), st_data.size());
        REQUIRE_THROWS_WITH(state_vector.applyMatrix(m.data(), {}),
                            Catch::Contains("must be larger than 0"));
    }

    SECTION("Test with different number of wires") {
        const size_t num_qubits = 5;
        for (size_t num_wires = 1; num_wires < num_qubits; num_wires++) {
            VectorT st_data_1 =
                createRandomStateVectorData<PrecisionT>(re, num_qubits);
            VectorT st_data_2 = st_data_1;

            StateVectorT state_vector_1(st_data_1.data(), st_data_1.size());
            StateVectorT state_vector_2(st_data_2.data(), st_data_2.size());

            std::vector<size_t> wires(num_wires);
            std::iota(wires.begin(), wires.end(), 0);

            const auto m = randomUnitary<PrecisionT>(re, num_wires);

            state_vector_1.applyMatrix(m, wires);

            Gates::GateImplementationsPI::applyMultiQubitOp<PrecisionT>(
                state_vector_2.getData(), num_qubits, m.data(), wires, false);

            PrecisionT eps = std::numeric_limits<PrecisionT>::epsilon() * 10E3;
            REQUIRE(isApproxEqual(
                state_vector_1.getData(), state_vector_1.getLength(),
                state_vector_2.getData(), state_vector_2.getLength(), eps));
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorLQubit::applyOperations",
                           "[applyOperations invalid arguments]",
                           (StateVectorLQubitManaged, StateVectorLQubitRaw),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using VectorT = TestVector<ComplexT>;
    std::mt19937_64 re{1337};

    SECTION("Test invalid arguments without parameters") {
        const size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(st_data.data(), st_data.size());

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"PauliX", "PauliY"}, {{0}},
                                         {false, false}),
            LightningException,
            "must all be equal"); // invalid wires
        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"PauliX", "PauliY"}, {{0}, {1}},
                                         {false}),
            LightningException, "must all be equal"); // invalid inverse
    }

    SECTION("Test invalid arguments with parameters") {
        const size_t num_qubits = 4;

        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(st_data.data(), st_data.size());

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"RX", "RY"}, {{0}}, {false, false},
                                         {{0.0}, {0.0}}),
            LightningException, "must all be equal"); // invalid wires

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"RX", "RY"}, {{0}, {1}}, {false},
                                         {{0.0}, {0.0}}),
            LightningException, "must all be equal"); // invalid wires

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"RX", "RY"}, {{0}, {1}},
                                         {false, false}, {{0.0}}),
            LightningException, "must all be equal"); // invalid parameters
    }
}
