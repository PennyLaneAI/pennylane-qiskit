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

#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp" // createRandomStateVectorData

/**
 * @file
 *  Tests for functionality for the class StateVectorKokkos.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::Util;
using Pennylane::Util::isApproxEqual;
using Pennylane::Util::randomUnitary;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("StateVectorLQubit::Constructibility",
                   "[Default Constructibility]", StateVectorKokkos<>) {
    SECTION("StateVectorBackend<>") {
        REQUIRE(!std::is_constructible_v<TestType>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorKokkos::Constructibility",
                           "[General Constructibility]", (StateVectorKokkos),
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

TEMPLATE_PRODUCT_TEST_CASE("StateVectorKokkos::applyMatrix with a std::vector",
                           "[applyMatrix]", (StateVectorKokkos),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using VectorT = TestVector<std::complex<PrecisionT>>;
    std::mt19937_64 re{1337};

    SECTION("Test wrong matrix size") {
        std::vector<ComplexT> m(7, 0.0);
        const size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);
        StateVectorT state_vector(reinterpret_cast<ComplexT *>(st_data.data()),
                                  st_data.size());
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

        StateVectorT state_vector(reinterpret_cast<ComplexT *>(st_data.data()),
                                  st_data.size());
        REQUIRE_THROWS_WITH(
            state_vector.applyMatrix(m, {0}),
            Catch::Contains(
                "The size of matrix does not match with the given"));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorKokkos::applyMatrix with a pointer",
                           "[applyMatrix]", (StateVectorKokkos),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using VectorT = TestVector<std::complex<PrecisionT>>;
    std::mt19937_64 re{1337};

    SECTION("Test wrong matrix") {
        std::vector<ComplexT> m(8, 0.0);
        const size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(reinterpret_cast<ComplexT *>(st_data.data()),
                                  st_data.size());
        REQUIRE_THROWS_WITH(state_vector.applyMatrix(m.data(), {}),
                            Catch::Contains("must be larger than 0"));
    }

    SECTION("Test with different number of wires") {
        using KokkosVector = typename StateVectorT::KokkosVector;
        const size_t num_qubits = 5;
        for (size_t num_wires = 1; num_wires < num_qubits; num_wires++) {
            VectorT st_data_1 =
                createRandomStateVectorData<PrecisionT>(re, num_qubits);
            VectorT st_data_2 = st_data_1;
            StateVectorT state_vector_1(
                reinterpret_cast<ComplexT *>(st_data_1.data()),
                st_data_1.size());
            StateVectorT state_vector_2(
                reinterpret_cast<ComplexT *>(st_data_2.data()),
                st_data_2.size());

            std::vector<size_t> wires(num_wires);
            std::iota(wires.begin(), wires.end(), 0);

            auto m = randomUnitary<PrecisionT>(re, num_wires);
            std::vector<ComplexT> mkvec(reinterpret_cast<ComplexT *>(m.data()),
                                        reinterpret_cast<ComplexT *>(m.data()) +
                                            m.size());
            KokkosVector mkview(reinterpret_cast<ComplexT *>(m.data()),
                                m.size());
            state_vector_1.applyMatrix(mkvec, wires);
            state_vector_2.applyMultiQubitOp(mkview, wires);

            PrecisionT eps = std::numeric_limits<PrecisionT>::epsilon() * 10E3;
            REQUIRE(isApproxEqual(
                state_vector_1.getData(), state_vector_1.getLength(),
                state_vector_2.getData(), state_vector_2.getLength(), eps));
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorKokkos::applyOperations",
                           "[applyOperations invalid arguments]",
                           (StateVectorKokkos), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using VectorT = TestVector<std::complex<PrecisionT>>;
    std::mt19937_64 re{1337};

    SECTION("Test invalid arguments without parameters") {
        const size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(reinterpret_cast<ComplexT *>(st_data.data()),
                                  st_data.size());

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"PauliX", "PauliY"}, {{0}},
                                         {false, false}),
            LightningException, "must all be equal"); // invalid wires
        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"PauliX", "PauliY"}, {{0}, {1}},
                                         {false}),
            LightningException, "must all be equal"); // invalid inverse
        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperation("PauliX", std::vector<std::size_t>{0},
                                        std::vector<bool>{false},
                                        std::vector<std::size_t>{1}),
            LightningException,
            "Controlled kernels not implemented."); // invalid controlled_wires
        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperation("PauliX", {}, std::vector<bool>{false},
                                        std::vector<std::size_t>{1}, false,
                                        {0.0}, std::vector<ComplexT>{}),
            LightningException,
            "`controlled_wires` must have the same size "
            "as"); // invalid controlled_wires
    }

    SECTION("Test invalid arguments with parameters") {
        const size_t num_qubits = 4;

        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(reinterpret_cast<ComplexT *>(st_data.data()),
                                  st_data.size());

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

    SECTION("Test invalid operation name and no matrix") {
        const size_t num_qubits = 3;

        StateVectorT state_vector(num_qubits);

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperation("GlobalPhaseShift", {0}, false, {0.0}),
            LightningException,
            "Operation does not exist for GlobalPhaseShift");
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::StateVectorKokkos",
                   "[StateVectorKokkos]", float, double) {
    using PrecisionT = TestType;
    using ComplexT = typename StateVectorKokkos<PrecisionT>::ComplexT;
    std::mt19937_64 re{1337};

    SECTION("StateVectorKokkos<TestType> {size_t}") {
        REQUIRE(std::is_constructible_v<StateVectorKokkos<TestType>, size_t>);
        const size_t num_qubits = 4;
        StateVectorKokkos<PrecisionT> sv(num_qubits);

        REQUIRE(sv.getNumQubits() == 4);
        REQUIRE(sv.getLength() == 16);
        REQUIRE(sv.getDataVector().size() == 16);
    }

    SECTION("StateVectorKokkos<TestType> {ComplexT *, size_t}") {
        using TestVectorT = TestVector<std::complex<PrecisionT>>;
        REQUIRE(std::is_constructible_v<StateVectorKokkos<TestType>, ComplexT *,
                                        size_t>);
        const size_t num_qubits = 5;
        TestVectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);
        StateVectorKokkos<PrecisionT> sv(
            reinterpret_cast<ComplexT *>(st_data.data()), st_data.size());

        REQUIRE(sv.getNumQubits() == 5);
        REQUIRE(sv.getLength() == 32);
        REQUIRE(sv.getDataVector().size() == 32);
    }

    SECTION("StateVectorKokkos<TestType> {ComplexT *, size_t}") {
        using TestVectorT = TestVector<std::complex<PrecisionT>>;
        REQUIRE(std::is_constructible_v<StateVectorKokkos<TestType>,
                                        std::vector<ComplexT>>);
        const size_t num_qubits = 5;
        TestVectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);
        std::vector<ComplexT> data_(st_data.data(),
                                    st_data.data() + st_data.size());
        StateVectorKokkos<PrecisionT> sv(data_);

        REQUIRE(sv.getNumQubits() == 5);
        REQUIRE(sv.getLength() == 32);
        REQUIRE(sv.getDataVector().size() == 32);
    }

    SECTION("StateVectorKokkos<TestType> {const "
            "StateVectorKokkos<TestType>&}") {
        REQUIRE(std::is_constructible_v<StateVectorKokkos<TestType>,
                                        const StateVectorKokkos<TestType> &>);
    }

    SECTION("updateData") {
        using TestVectorT = TestVector<std::complex<PrecisionT>>;
        const size_t num_qubits = 3;
        StateVectorKokkos<PrecisionT> sv(num_qubits);

        TestVectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);
        std::vector<ComplexT> data_(st_data.data(),
                                    st_data.data() + st_data.size());
        sv.updateData(data_);

        REQUIRE(sv.getDataVector() == data_);
        // REQUIRE(sv.getDataVector() == approx(st_data));
    }
}