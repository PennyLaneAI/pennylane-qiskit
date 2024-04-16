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
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "MeasurementsKokkos.hpp"
#include "ObservablesKokkos.hpp"
#include "StateVectorKokkos.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::Observables;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("Test variance of NamedObs", "[StateVectorKokkos_Var]",
                   float, double) {
    const std::size_t num_qubits = 2;
    SECTION("var(PauliX[0])") {
        StateVectorKokkos<TestType> kokkos_sv{num_qubits};
        auto m = Measurements<StateVectorKokkos<TestType>>(kokkos_sv);

        kokkos_sv.applyOperation("RX", {0}, false, {0.7});
        kokkos_sv.applyOperation("RY", {0}, false, {0.7});
        kokkos_sv.applyOperation("RX", {1}, false, {0.5});
        kokkos_sv.applyOperation("RY", {1}, false, {0.5});

        auto ob = NamedObs<StateVectorKokkos<TestType>>("PauliX", {0});
        auto res = m.var(ob);
        auto expected = TestType(0.7572222074);
        CHECK(res == Approx(expected));
    }

    SECTION("var(PauliY[0])") {
        StateVectorKokkos<TestType> kokkos_sv{num_qubits};
        auto m = Measurements<StateVectorKokkos<TestType>>(kokkos_sv);

        kokkos_sv.applyOperation("RX", {0}, false, {0.7});
        kokkos_sv.applyOperation("RY", {0}, false, {0.7});
        kokkos_sv.applyOperation("RX", {1}, false, {0.5});
        kokkos_sv.applyOperation("RY", {1}, false, {0.5});

        auto ob = NamedObs<StateVectorKokkos<TestType>>("PauliY", {0});
        auto res = m.var(ob);
        auto expected = TestType(0.5849835715);
        CHECK(res == Approx(expected));
    }

    SECTION("var(PauliZ[1])") {
        StateVectorKokkos<TestType> kokkos_sv{num_qubits};
        auto m = Measurements<StateVectorKokkos<TestType>>(kokkos_sv);

        kokkos_sv.applyOperation("RX", {0}, false, {0.7});
        kokkos_sv.applyOperation("RY", {0}, false, {0.7});
        kokkos_sv.applyOperation("RX", {1}, false, {0.5});
        kokkos_sv.applyOperation("RY", {1}, false, {0.5});

        auto ob = NamedObs<StateVectorKokkos<TestType>>("PauliZ", {1});
        auto res = m.var(ob);
        auto expected = TestType(0.4068672016);
        CHECK(res == Approx(expected));
    }
}

TEMPLATE_TEST_CASE("Test variance of HermitianObs", "[StateVectorKokkos_Var]",
                   float, double) {
    const std::size_t num_qubits = 3;
    using StateVectorT = StateVectorKokkos<TestType>;
    using ComplexT = typename StateVectorT::ComplexT;
    SECTION("Using var") {
        StateVectorKokkos<TestType> kokkos_sv{num_qubits};
        auto m = Measurements<StateVectorKokkos<TestType>>(kokkos_sv);

        kokkos_sv.applyOperation("RX", {0}, false, {0.7});
        kokkos_sv.applyOperation("RY", {0}, false, {0.7});
        kokkos_sv.applyOperation("RX", {1}, false, {0.5});
        kokkos_sv.applyOperation("RY", {1}, false, {0.5});
        kokkos_sv.applyOperation("RX", {2}, false, {0.3});
        kokkos_sv.applyOperation("RY", {2}, false, {0.3});

        const TestType theta = M_PI / 2;
        const TestType c = std::cos(theta / 2);
        const TestType js = std::sin(-theta / 2);
        std::vector<ComplexT> matrix(16, 0);
        matrix[0] = c;
        matrix[1] = ComplexT{0, js};
        matrix[4] = ComplexT{0, js};
        matrix[5] = c;
        matrix[10] = ComplexT{1, 0};
        matrix[15] = ComplexT{1, 0};

        auto ob = HermitianObs<StateVectorKokkos<TestType>>(matrix, {0, 2});
        auto res = m.var(ob);
        auto expected = TestType(0.4103533486);
        CHECK(res == Approx(expected));
    }
}

TEMPLATE_TEST_CASE("Test variance of TensorProdObs", "[StateVectorKokkos_Var]",
                   float, double) {
    const std::size_t num_qubits = 3;
    SECTION("Using var") {
        StateVectorKokkos<TestType> kokkos_sv{num_qubits};
        auto m = Measurements<StateVectorKokkos<TestType>>(kokkos_sv);

        kokkos_sv.applyOperation("RX", {0}, false, {0.5});
        kokkos_sv.applyOperation("RY", {0}, false, {0.5});
        kokkos_sv.applyOperation("RX", {1}, false, {0.2});
        kokkos_sv.applyOperation("RY", {1}, false, {0.2});

        auto X0 = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
            "PauliX", std::vector<size_t>{0});
        auto Z1 = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
            "PauliZ", std::vector<size_t>{1});

        auto ob = TensorProdObs<StateVectorKokkos<TestType>>::create({X0, Z1});
        auto res = m.var(*ob);
        auto expected = TestType(0.836679);
        CHECK(expected == Approx(res));
    }
}

TEMPLATE_TEST_CASE("Test variance of HamiltonianObs", "[StateVectorKokkos_Var]",
                   float, double) {
    SECTION("Using var") {
        std::vector<Kokkos::complex<TestType>> init_state{
            {0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1}, {0.1, 0.2},
            {0.2, 0.2}, {0.3, 0.3}, {0.3, 0.4}, {0.4, 0.5}};
        StateVectorKokkos<TestType> kokkos_sv{init_state.data(),
                                              init_state.size()};
        auto m = Measurements<StateVectorKokkos<TestType>>(kokkos_sv);

        auto X0 = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
            "PauliX", std::vector<size_t>{0});
        auto Z1 = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
            "PauliZ", std::vector<size_t>{1});

        auto ob = Hamiltonian<StateVectorKokkos<TestType>>::create({0.3, 0.5},
                                                                   {X0, Z1});
        auto res = m.var(*ob);
        auto expected = TestType(0.224604);
        CHECK(expected == Approx(res));
    }
}
