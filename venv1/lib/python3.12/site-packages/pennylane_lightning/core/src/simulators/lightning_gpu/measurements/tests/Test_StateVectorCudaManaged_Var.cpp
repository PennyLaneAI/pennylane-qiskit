// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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

#include "MeasurementsGPU.hpp"
#include "ObservablesGPU.hpp"
#include "StateVectorCudaManaged.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU::Measures;
using namespace Pennylane::LightningGPU::Observables;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("Test variance of NamedObs", "[StateVectorCudaManaged_Var]",
                   float, double) {
    using StateVectorT = StateVectorCudaManaged<TestType>;
    const std::size_t num_qubits = 2;
    SECTION("var(PauliX[0])") {
        StateVectorT sv{num_qubits};
        sv.initSV();
        auto m = Measurements<StateVectorT>(sv);

        sv.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});

        auto ob = NamedObs<StateVectorT>("PauliX", {0});
        auto res = m.var(ob);
        auto expected = TestType(0.7572222074);
        CHECK(res == Approx(expected));
    }

    SECTION("var(PauliY[0])") {
        StateVectorT sv{num_qubits};
        sv.initSV();
        auto m = Measurements<StateVectorT>(sv);

        sv.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});

        auto ob = NamedObs<StateVectorT>("PauliY", {0});
        auto res = m.var(ob);
        auto expected = TestType(0.5849835715);
        CHECK(res == Approx(expected));
    }

    SECTION("var(PauliZ[1])") {
        StateVectorT sv{num_qubits};
        sv.initSV();
        auto m = Measurements<StateVectorT>(sv);

        sv.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});

        auto ob = NamedObs<StateVectorT>("PauliZ", {1});
        auto res = m.var(ob);
        auto expected = TestType(0.4068672016);
        CHECK(res == Approx(expected));
    }
}

TEMPLATE_TEST_CASE("Test variance of HermitianObs",
                   "[StateVectorCudaManaged_Var]", float, double) {
    const std::size_t num_qubits = 3;
    using StateVectorT = StateVectorCudaManaged<TestType>;
    using ComplexT = typename StateVectorT::ComplexT;
    SECTION("Using var") {
        StateVectorT sv{num_qubits};
        sv.initSV();
        auto m = Measurements<StateVectorT>(sv);

        sv.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}, {"RX"}, {"RY"}},
            {{0}, {0}, {1}, {1}, {2}, {2}},
            {{false}, {false}, {false}, {false}, {false}, {false}},
            {{0.7}, {0.7}, {0.5}, {0.5}, {0.3}, {0.3}});

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

        auto ob = HermitianObs<StateVectorT>(matrix, {0, 2});
        auto res = m.var(ob);
        auto expected = TestType(0.4103533486);
        CHECK(res == Approx(expected));
    }
}

TEMPLATE_TEST_CASE("Test variance of TensorProdObs",
                   "[StateVectorCudaManaged_Var]", float, double) {
    using StateVectorT = StateVectorCudaManaged<TestType>;
    const std::size_t num_qubits = 3;
    SECTION("Using var") {
        StateVectorT sv{num_qubits};
        sv.initSV();
        auto m = Measurements<StateVectorT>(sv);

        sv.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto X0 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliX", std::vector<size_t>{0});
        auto Z1 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});

        auto ob = TensorProdObs<StateVectorT>::create({X0, Z1});
        auto res = m.var(*ob);
        auto expected = TestType(0.836679);
        CHECK(expected == Approx(res));
    }
}

TEMPLATE_TEST_CASE("Test variance of HamiltonianObs",
                   "[StateVectorCudaManaged_Var]", float, double) {
    using StateVectorT = StateVectorCudaManaged<TestType>;
    SECTION("Using var") {
        std::vector<std::complex<TestType>> init_state{
            {0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1}, {0.1, 0.2},
            {0.2, 0.2}, {0.3, 0.3}, {0.3, 0.4}, {0.4, 0.5}};
        StateVectorT sv{init_state.data(), init_state.size()};
        auto m = Measurements<StateVectorT>(sv);

        auto X0 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliX", std::vector<size_t>{0});
        auto Z1 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});

        auto ob = Hamiltonian<StateVectorT>::create({0.3, 0.5}, {X0, Z1});
        auto res = m.var(*ob);
        auto expected = TestType(0.224604);
        CHECK(expected == Approx(res));
    }
}
