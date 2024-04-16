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
#include <cstddef>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "MeasurementsKokkos.hpp"
#include "ObservablesKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp"

/**
 * @file
 *  Tests for StateVectorKokkos expectation value functionality.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::Observables;
using Pennylane::Util::createNonTrivialState;
using Pennylane::Util::write_CSR_vectors;
using std::size_t;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::NonExistent",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    const size_t num_qubits = 3;
    StateVectorKokkos<TestType> kokkos_sv{num_qubits};
    auto m = Measurements(kokkos_sv);

    SECTION("Using expval with non-existent operation") {
        PL_REQUIRE_THROWS_MATCHES(m.expval("XXX", {0}), LightningException,
                                  "Expval does not exist for named observable");
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValueIdentity",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    const size_t num_qubits = 3;
    auto ONE = TestType(1);
    StateVectorKokkos<TestType> kokkos_sv{num_qubits};
    auto m = Measurements(kokkos_sv);

    SECTION("Using expval") {
        kokkos_sv.applyOperation("Hadamard", {0}, false);
        kokkos_sv.applyOperation("CNOT", {0, 1}, false);
        kokkos_sv.applyOperation("CNOT", {1, 2}, false);
        auto ob = NamedObs<StateVectorKokkos<TestType>>("Identity", {0});
        auto res = m.expval(ob);
        CHECK(res == Approx(ONE));
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValuePauliX",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    {
        const size_t num_qubits = 3;

        auto ZERO = TestType(0);
        auto ONE = TestType(1);

        SECTION("Using expval") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyOperation("CNOT", {0, 1}, false);
            kokkos_sv.applyOperation("CNOT", {1, 2}, false);
            auto ob = NamedObs<StateVectorKokkos<TestType>>("PauliX", {0});
            auto res = m.expval(ob);
            CHECK(res == ZERO);
        }
        SECTION("Using expval: Plus states") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyOperation("Hadamard", {1}, false);
            kokkos_sv.applyOperation("Hadamard", {2}, false);
            auto ob = NamedObs<StateVectorKokkos<TestType>>("PauliX", {0});
            auto res = m.expval(ob);
            CHECK(res == Approx(ONE));
        }
        SECTION("Using expval: Minus states") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyOperation("PauliX", {0}, false);
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyOperation("PauliX", {1}, false);
            kokkos_sv.applyOperation("Hadamard", {1}, false);
            kokkos_sv.applyOperation("PauliX", {2}, false);
            kokkos_sv.applyOperation("Hadamard", {2}, false);
            auto ob = NamedObs<StateVectorKokkos<TestType>>("PauliX", {0});
            auto res = m.expval(ob);
            CHECK(res == -Approx(ONE));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValuePauliY",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    {
        const size_t num_qubits = 3;

        auto ZERO = TestType(0);
        auto ONE = TestType(1);
        auto PI = TestType(M_PI);

        SECTION("Using expval") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyOperation("CNOT", {0, 1}, false);
            kokkos_sv.applyOperation("CNOT", {1, 2}, false);
            auto ob = NamedObs<StateVectorKokkos<TestType>>("PauliY", {0});
            auto res = m.expval(ob);
            CHECK(res == ZERO);
        }
        SECTION("Using expval: Plus i states") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyOperation("RX", {0}, false, {-PI / 2});
            kokkos_sv.applyOperation("RX", {1}, false, {-PI / 2});
            kokkos_sv.applyOperation("RX", {2}, false, {-PI / 2});
            auto ob = NamedObs<StateVectorKokkos<TestType>>("PauliY", {0});
            auto res = m.expval(ob);
            CHECK(res == Approx(ONE));
        }
        SECTION("Using expval: Minus i states") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyOperation("RX", {0}, false, {PI / 2});
            kokkos_sv.applyOperation("RX", {1}, false, {PI / 2});
            kokkos_sv.applyOperation("RX", {2}, false, {PI / 2});
            auto ob = NamedObs<StateVectorKokkos<TestType>>("PauliY", {0});
            auto res = m.expval(ob);
            CHECK(res == -Approx(ONE));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValuePauliZ",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    {
        using StateVectorT = StateVectorKokkos<TestType>;
        using PrecisionT = StateVectorT::PrecisionT;

        // Defining the statevector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT kokkos_sv(statevector_data.data(),
                               statevector_data.size());

        SECTION("Using expval") {
            auto m = Measurements(kokkos_sv);
            auto ob = NamedObs<StateVectorKokkos<TestType>>("PauliZ", {1});
            auto res = m.expval(ob);
            PrecisionT ref = 0.77015115;
            REQUIRE(res == Approx(ref).margin(1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValueHadamard",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    {
        const size_t num_qubits = 3;
        auto INVSQRT2 = TestType(0.707106781186547524401);

        SECTION("Using expval") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyOperation("PauliX", {0});
            auto ob = NamedObs<StateVectorKokkos<TestType>>("Hadamard", {0});
            auto res = m.expval(ob);
            CHECK(res == -INVSQRT2);
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValueSingleQubitOp",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 3;

        auto INVSQRT2 = TestType(0.707106781186547524401);

        SECTION("Using expval") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);

            const TestType theta = M_PI / 2;
            const TestType c = std::cos(theta / 2);
            const TestType js = std::sin(-theta / 2);
            std::vector<ComplexT> matrix{c, {0, js}, {0, js}, c};

            auto ob = HermitianObs<StateVectorKokkos<TestType>>(matrix, {0});
            auto res = m.expval(ob);
            CHECK(res == INVSQRT2);
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValueTwoQubitOp",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    {
        const size_t num_qubits = 3;
        auto INVSQRT2 = TestType(0.707106781186547524401);

        SECTION("Using expval") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);

            const TestType theta = M_PI / 2;
            const TestType c = std::cos(theta / 2);
            const TestType js = std::sin(-theta / 2);
            std::vector<ComplexT> matrix(16);
            matrix[0] = c;
            matrix[1] = ComplexT{0, js};
            matrix[4] = ComplexT{0, js};
            matrix[5] = c;
            matrix[10] = ComplexT{1, 0};
            matrix[15] = ComplexT{1, 0};

            auto ob = HermitianObs<StateVectorKokkos<TestType>>(matrix, {0, 1});
            auto res = m.expval(ob);
            CHECK(res == INVSQRT2);
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::Hamiltonian_expval",
                   "[StateVectorKokkos_Expval]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    const size_t num_qubits = 3;
    SECTION("GetExpectationIdentity") {
        StateVectorKokkos<TestType> kokkos_sv{num_qubits};
        auto m = Measurements(kokkos_sv);
        std::vector<size_t> wires{0, 1, 2};

        kokkos_sv.applyOperation("Hadamard", {0}, false);
        kokkos_sv.applyOperation("CNOT", {0, 1}, false);
        kokkos_sv.applyOperation("CNOT", {1, 2}, false);

        size_t matrix_dim = static_cast<size_t>(1U) << num_qubits;
        std::vector<ComplexT> matrix(matrix_dim * matrix_dim);

        for (size_t i = 0; i < matrix.size(); i++) {
            if (i % matrix_dim == i / matrix_dim)
                matrix[i] = ComplexT{1, 0};
            else
                matrix[i] = ComplexT{0, 0};
        }

        auto results = m.expval(matrix, wires);
        ComplexT expected = {1, 0};
        CHECK(real(expected) == Approx(results).epsilon(1e-7));
    }

    SECTION("GetExpectationHermitianMatrix") {
        std::vector<ComplexT> init_state{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                         {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                         {0.3, 0.4}, {0.4, 0.5}};
        StateVectorKokkos<TestType> kokkos_sv{init_state.data(),
                                              init_state.size()};
        auto m = Measurements(kokkos_sv);
        std::vector<size_t> wires{0, 1, 2};
        std::vector<ComplexT> matrix{
            {0.5, 0.0},  {0.2, 0.5},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.5, 0.0},  {0.2, 0.5},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.5, 0.0},  {0.2, 0.5},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.5, 0.0},  {0.2, 0.5},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0}};

        auto results = m.expval(matrix, wires);
        ComplexT expected(1.263000, -1.011000);
        CHECK(real(expected) == Approx(results).epsilon(1e-7));
    }

    SECTION("Using expval") {
        std::vector<ComplexT> init_state{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                         {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                         {0.3, 0.4}, {0.4, 0.5}};
        StateVectorKokkos<TestType> kokkos_sv{init_state.data(),
                                              init_state.size()};
        std::vector<ComplexT> matrix{
            {0.5, 0.0},  {0.2, 0.5},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.5, 0.0},  {0.2, 0.5},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.5, 0.0},  {0.2, 0.5},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.5, 0.0},  {0.2, 0.5},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0}};

        auto m = Measurements(kokkos_sv);
        auto ob = HermitianObs<StateVectorKokkos<TestType>>(matrix, {0, 1, 2});
        auto res = m.expval(ob);
        ComplexT expected(1.263000, -1.011000);
        CHECK(real(expected) == Approx(res).epsilon(1e-7));
    }
}

TEMPLATE_TEST_CASE("Test expectation value of HamiltonianObs",
                   "[StateVectorKokkos_Expval]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    SECTION("Using expval") {
        std::vector<ComplexT> init_state{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                         {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                         {0.3, 0.4}, {0.4, 0.5}};
        StateVectorKokkos<TestType> kokkos_sv{init_state.data(),
                                              init_state.size()};
        auto m = Measurements(kokkos_sv);

        auto X0 = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
            "PauliX", std::vector<size_t>{0});
        auto Z1 = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
            "PauliZ", std::vector<size_t>{1});

        auto ob = Hamiltonian<StateVectorKokkos<TestType>>::create({0.3, 0.5},
                                                                   {X0, Z1});
        auto res = m.expval(*ob);
        auto expected = TestType(-0.086);
        CHECK(expected == Approx(res));
    }
}

TEMPLATE_TEST_CASE("Test expectation value of TensorProdObs",
                   "[StateVectorKokkos_Expval]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    SECTION("Using expval") {
        std::vector<ComplexT> init_state{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                         {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                         {0.3, 0.4}, {0.4, 0.5}};
        StateVectorKokkos<TestType> kokkos_sv{init_state.data(),
                                              init_state.size()};
        auto m = Measurements(kokkos_sv);

        auto X0 = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
            "PauliX", std::vector<size_t>{0});
        auto Z1 = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
            "PauliZ", std::vector<size_t>{1});

        auto ob = TensorProdObs<StateVectorKokkos<TestType>>::create({X0, Z1});
        auto res = m.expval(*ob);
        auto expected = TestType(-0.36);
        CHECK(expected == Approx(res));
    }
}

TEMPLATE_TEST_CASE("Test expectation value of NQubit Hermitian",
                   "[StateVectorKokkos_Expval]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    using VectorT = TestVector<std::complex<TestType>>;
    std::mt19937_64 re{1337};
    const size_t num_qubits = 7;
    VectorT sv_data = createRandomStateVectorData<TestType>(re, num_qubits);

    StateVectorKokkos<TestType> kokkos_sv(
        reinterpret_cast<ComplexT *>(sv_data.data()), sv_data.size());
    auto m = Measurements(kokkos_sv);

    auto X0 = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
        "PauliX", std::vector<size_t>{0});
    auto Y1 = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
        "PauliY", std::vector<size_t>{1});
    auto Z2 = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
        "PauliZ", std::vector<size_t>{2});
    auto X3 = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
        "PauliX", std::vector<size_t>{3});
    auto Y4 = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
        "PauliY", std::vector<size_t>{4});

    ComplexT j{0.0, 1.0};
    ComplexT u{1.0, 0.0};
    ComplexT z{0.0, 0.0};

    SECTION("3Qubit") {
        auto ob =
            TensorProdObs<StateVectorKokkos<TestType>>::create({X0, Y1, Z2});
        auto expected = m.expval(*ob);
        std::vector<ComplexT> matrix{z, z, z, z,  z, z,  -j, z, z,  z, z, z, z,
                                     z, z, j, z,  z, z,  z,  j, z,  z, z, z, z,
                                     z, z, z, -j, z, z,  z,  z, -j, z, z, z, z,
                                     z, z, z, z,  j, z,  z,  z, z,  j, z, z, z,
                                     z, z, z, z,  z, -j, z,  z, z,  z, z, z};
        SECTION("Hermitian") {
            auto hermitian =
                HermitianObs<StateVectorKokkos<TestType>>(matrix, {0, 1, 2});
            auto res = m.expval(hermitian);

            CHECK(expected == Approx(res));
        }
        SECTION("Matrix") {
            auto res = m.expval(matrix, {0, 1, 2});
            CHECK(expected == Approx(res));
        }
    }

    SECTION("4Qubit") {
        auto ob = TensorProdObs<StateVectorKokkos<TestType>>::create(
            {X0, Y1, Z2, X3});
        auto expected = m.expval(*ob);
        std::vector<ComplexT> matrix{
            z, z, z,  z, z, z, z, z,  z,  z, z, z, z, -j, z, z, z, z, z, z,
            z, z, z,  z, z, z, z, z,  -j, z, z, z, z, z,  z, z, z, z, z, z,
            z, z, z,  z, z, z, z, j,  z,  z, z, z, z, z,  z, z, z, z, z, z,
            z, z, j,  z, z, z, z, z,  z,  z, z, z, z, j,  z, z, z, z, z, z,
            z, z, z,  z, z, z, z, z,  j,  z, z, z, z, z,  z, z, z, z, z, z,
            z, z, z,  z, z, z, z, -j, z,  z, z, z, z, z,  z, z, z, z, z, z,
            z, z, -j, z, z, z, z, z,  z,  z, z, z, z, -j, z, z, z, z, z, z,
            z, z, z,  z, z, z, z, z,  -j, z, z, z, z, z,  z, z, z, z, z, z,
            z, z, z,  z, z, z, z, j,  z,  z, z, z, z, z,  z, z, z, z, z, z,
            z, z, j,  z, z, z, z, z,  z,  z, z, z, z, j,  z, z, z, z, z, z,
            z, z, z,  z, z, z, z, z,  j,  z, z, z, z, z,  z, z, z, z, z, z,
            z, z, z,  z, z, z, z, -j, z,  z, z, z, z, z,  z, z, z, z, z, z,
            z, z, -j, z, z, z, z, z,  z,  z, z, z, z, z,  z, z};
        SECTION("Hermitian") {
            auto hermitian =
                HermitianObs<StateVectorKokkos<TestType>>(matrix, {0, 1, 2, 3});
            auto res = m.expval(hermitian);

            CHECK(expected == Approx(res));
        }
        SECTION("Matrix") {
            auto res = m.expval(matrix, {0, 1, 2, 3});
            CHECK(expected == Approx(res));
        }
    }

    SECTION("5Qubit") {
        auto ob = TensorProdObs<StateVectorKokkos<TestType>>::create(
            {X0, Y1, Z2, X3, Y4});
        auto expected = m.expval(*ob);
        std::vector<ComplexT> matrix{
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  -u, z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  u, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, -u, z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  u, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, u,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            -u, z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, u, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  -u, z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  u,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, -u, z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  u, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, -u, z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  -u, z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  u, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, -u, z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  u, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  -u, z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  u, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, -u, z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  u,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  u,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, -u, z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  u, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, -u, z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  u, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, -u, z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  u, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  -u, z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  -u, z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  u, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, -u, z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  u,  z,  z,  z, z, z,  z, z,  z, z, z,
            z,  z, z,  z, z,  z,  z, z,  z,  z,  z,  z, z, z,  z, z,  z};
        SECTION("Hermitian") {
            auto hermitian = HermitianObs<StateVectorKokkos<TestType>>(
                matrix, {0, 1, 2, 3, 4});
            auto res = m.expval(hermitian);

            CHECK(expected == Approx(res));
        }
        SECTION("Matrix") {
            auto res = m.expval(matrix, {0, 1, 2, 3, 4});
            CHECK(expected == Approx(res));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::Hamiltonian_expval_Sparse",
                   "[StateVectorKokkos_Expval]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;

    SECTION("Sparse expval") {
        std::vector<ComplexT> init_state{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                         {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                         {0.3, 0.4}, {0.4, 0.5}};
        StateVectorKokkos<TestType> kokkos_sv{init_state.data(),
                                              init_state.size()};
        auto m = Measurements(kokkos_sv);

        std::vector<size_t> index_ptr = {0, 2, 4, 6, 8, 10, 12, 14, 16};
        std::vector<size_t> indices = {0, 3, 1, 2, 1, 2, 0, 3,
                                       4, 7, 5, 6, 5, 6, 4, 7};
        std::vector<ComplexT> values = {
            {3.1415, 0.0},  {0.0, -3.1415}, {3.1415, 0.0}, {0.0, 3.1415},
            {0.0, -3.1415}, {3.1415, 0.0},  {0.0, 3.1415}, {3.1415, 0.0},
            {3.1415, 0.0},  {0.0, -3.1415}, {3.1415, 0.0}, {0.0, 3.1415},
            {0.0, -3.1415}, {3.1415, 0.0},  {0.0, 3.1415}, {3.1415, 0.0}};

        auto result = m.expval(index_ptr.data(), index_ptr.size(),
                               indices.data(), values.data(), values.size());
        auto expected = TestType(3.1415);
        CHECK(expected == Approx(result).epsilon(1e-7));
    }

    SECTION("Testing Sparse Hamiltonian:") {
        using StateVectorT = StateVectorKokkos<TestType>;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;

        // Defining the statevector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measurements class.
        // This object attaches to the statevector allowing several
        // measurements.
        Measurements<StateVectorT> Measurer(statevector);
        const size_t num_qubits = 3;
        const size_t data_size = Pennylane::Util::exp2(num_qubits);

        std::vector<size_t> row_map;
        std::vector<size_t> entries;
        std::vector<ComplexT> values;
        write_CSR_vectors(row_map, entries, values, data_size);

        PrecisionT exp_values =
            Measurer.expval(row_map.data(), row_map.size(), entries.data(),
                            values.data(), values.size());
        PrecisionT exp_values_ref = 0.5930885;
        REQUIRE(exp_values == Approx(exp_values_ref).margin(1e-6));

        PrecisionT var_values =
            Measurer.var(row_map.data(), row_map.size(), entries.data(),
                         values.data(), values.size());
        PrecisionT var_values_ref = 2.4624654;
        REQUIRE(var_values == Approx(var_values_ref).margin(1e-6));
    }
}
