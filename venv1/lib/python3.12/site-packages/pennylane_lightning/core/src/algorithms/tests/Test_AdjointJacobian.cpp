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
#include <catch2/catch.hpp>

#include "Error.hpp" // LightningException
#include "JacobianData.hpp"
#include "TestHelpers.hpp" // PL_REQUIRE_THROWS_MATCHES, linspace

/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

#ifdef _ENABLE_PLQUBIT
constexpr bool BACKEND_FOUND = true;
constexpr bool SUPPORTS_CTRL = true;

#include "AdjointJacobianLQubit.hpp"
#include "ObservablesLQubit.hpp"
#include "TestHelpersStateVectors.hpp" // TestStateVectorBackends, StateVectorToName

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit::Util;
using namespace Pennylane::LightningQubit::Algorithms;
using namespace Pennylane::LightningQubit::Observables;
} // namespace
/// @endcond

#elif _ENABLE_PLKOKKOS == 1
constexpr bool BACKEND_FOUND = true;
constexpr bool SUPPORTS_CTRL = false;

#include "AdjointJacobianKokkos.hpp"
#include "ObservablesKokkos.hpp"
#include "TestHelpersStateVectors.hpp" // TestStateVectorBackends, StateVectorToName

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos::Util;
using namespace Pennylane::LightningKokkos::Algorithms;
using namespace Pennylane::LightningKokkos::Observables;
} // namespace
  /// @endcond

#elif _ENABLE_PLGPU == 1
constexpr bool BACKEND_FOUND = true;
constexpr bool SUPPORTS_CTRL = false;
#include "AdjointJacobianGPU.hpp"
#include "ObservablesGPU.hpp"
#include "TestHelpersStateVectors.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU::Algorithms;
using namespace Pennylane::LightningGPU::Observables;
} // namespace
  /// @endcond

#else
constexpr bool BACKEND_FOUND = false;
constexpr bool SUPPORTS_CTRL = false;
using TestStateVectorBackends = Pennylane::Util::TypeList<void>;

template <class StateVector> struct StateVectorToName {};
#endif

template <typename TypeList> void testAdjointJacobian() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;

        const std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};

        AdjointJacobian<StateVectorT> adj;

        DYNAMIC_SECTION("Throws an exception when size mismatches - "
                        << StateVectorToName<StateVectorT>::name) {
            const std::vector<size_t> tp{0, 1};
            const size_t num_qubits = 1;
            const size_t num_params = 3;
            const size_t num_obs = 1;
            const auto obs = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{0});
            std::vector<PrecisionT> jacobian(num_obs * tp.size() - 1, 0);

            auto ops = OpsData<StateVectorT>({"RX"}, {{0.742}}, {{0}}, {false});

            std::vector<ComplexT> cdata(1U << num_qubits);
            cdata[0] = ComplexT{1, 0};

            StateVectorT psi(cdata.data(), cdata.size());

            JacobianData<StateVectorT> tape{
                num_params, psi.getLength(), psi.getData(), {obs}, ops, tp};
            PL_REQUIRE_THROWS_MATCHES(
                adj.adjointJacobian(std::span{jacobian}, tape, psi, true),
                LightningException,
                "The size of preallocated jacobian must be same as");
        }

        DYNAMIC_SECTION("No trainable params - "
                        << StateVectorToName<StateVectorT>::name) {
            const std::vector<size_t> tp{};
            const size_t num_qubits = 1;
            const size_t num_params = 3;
            const size_t num_obs = 1;
            const auto obs = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{0});
            std::vector<PrecisionT> jacobian(num_obs * tp.size(), 0);

            for (const auto &p : param) {
                auto ops = OpsData<StateVectorT>({"RX"}, {{p}}, {{0}}, {false});

                std::vector<ComplexT> cdata(1U << num_qubits);
                cdata[0] = ComplexT{1, 0};

                StateVectorT psi(cdata.data(), cdata.size());

                JacobianData<StateVectorT> tape{
                    num_params, psi.getLength(), psi.getData(), {obs}, ops, tp};
                REQUIRE_NOTHROW(
                    adj.adjointJacobian(std::span{jacobian}, tape, psi, true));
            }
        }

        DYNAMIC_SECTION("Op=PhaseShift, Obs=Y - "
                        << StateVectorToName<StateVectorT>::name) {
            if (SUPPORTS_CTRL) {
                const std::vector<size_t> tp{0};
                const size_t num_qubits = GENERATE(2, 3, 4);

                const size_t num_params = 3;
                const size_t num_obs = 1;
                const auto obs = std::make_shared<NamedObs<StateVectorT>>(
                    "PauliY", std::vector<size_t>{num_qubits - 1});
                std::vector<PrecisionT> jacobian(num_obs * tp.size(), 0);

                for (const auto &p : param) {
                    std::vector<std::vector<size_t>> controls{
                        std::vector<size_t>(num_qubits - 1)};
                    std::iota(controls[0].begin(), controls[0].end(), 0);
                    std::vector<std::vector<bool>> control_values{
                        std::vector<bool>(num_qubits - 1, true)};
                    auto ops = OpsData<StateVectorT>(
                        {"PhaseShift"}, {{p}}, {{num_qubits - 1}}, {false},
                        {{}}, controls, control_values);

                    std::vector<ComplexT> cdata(1U << num_qubits);
                    cdata[cdata.size() - 2] =
                        Pennylane::Util::INVSQRT2<PrecisionT>();
                    cdata[cdata.size() - 1] =
                        Pennylane::Util::INVSQRT2<PrecisionT>();

                    StateVectorT psi(cdata.data(), cdata.size());
                    JacobianData<StateVectorT> tape{
                        num_params, psi.getLength(), psi.getData(), {obs}, ops,
                        tp};
                    adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

                    CAPTURE(jacobian);
                    CHECK(cos(p) == Approx(jacobian[0]));
                }
            }
        }

        DYNAMIC_SECTION("Op=RX, Obs=Z - "
                        << StateVectorToName<StateVectorT>::name) {
            const std::vector<size_t> tp{0};

            const size_t num_qubits = 1;
            const size_t num_params = 3;
            const size_t num_obs = 1;
            const auto obs = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{0});
            std::vector<PrecisionT> jacobian(num_obs * tp.size(), 0);

            for (const auto &p : param) {
                auto ops = OpsData<StateVectorT>({"RX"}, {{p}}, {{0}}, {false});

                std::vector<ComplexT> cdata(1U << num_qubits);
                cdata[0] = ComplexT{1, 0};

                StateVectorT psi(cdata.data(), cdata.size());

                JacobianData<StateVectorT> tape{
                    num_params, psi.getLength(), psi.getData(), {obs}, ops, tp};
                adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

                CAPTURE(jacobian);
                CHECK(-sin(p) == Approx(jacobian[0]));
            }
        }

        DYNAMIC_SECTION("Op=RY, Obs=X - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<size_t> tp{0};
            const size_t num_qubits = 1;
            const size_t num_params = 3;
            const size_t num_obs = 1;

            const auto obs = std::make_shared<NamedObs<StateVectorT>>(
                "PauliX", std::vector<size_t>{0});
            std::vector<PrecisionT> jacobian(num_obs * tp.size(), 0);

            for (const auto &p : param) {
                auto ops = OpsData<StateVectorT>({"RY"}, {{p}}, {{0}}, {false});

                std::vector<ComplexT> cdata(1U << num_qubits);
                cdata[0] = ComplexT{1, 0};

                StateVectorT psi(cdata.data(), cdata.size());

                JacobianData<StateVectorT> tape{
                    num_params, psi.getLength(), psi.getData(), {obs}, ops, tp};
                adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

                CAPTURE(jacobian);
                CHECK(cos(p) == Approx(jacobian[0]).margin(1e-7));
            }
        }

        DYNAMIC_SECTION("Op=RX, Obs=[Z,Z] - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<size_t> tp{0};
            const size_t num_qubits = 2;
            const size_t num_params = 1;
            const size_t num_obs = 2;
            std::vector<PrecisionT> jacobian(num_obs * tp.size(), 0);

            std::vector<ComplexT> cdata(1U << num_qubits);
            cdata[0] = ComplexT{1, 0};

            StateVectorT psi(cdata.data(), cdata.size());

            const auto obs1 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{0});
            const auto obs2 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{1});

            auto ops =
                OpsData<StateVectorT>({"RX"}, {{param[0]}}, {{0}}, {false});

            JacobianData<StateVectorT> tape{num_params,    psi.getLength(),
                                            psi.getData(), {obs1, obs2},
                                            ops,           tp};
            adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

            CAPTURE(jacobian);
            CHECK(-sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
            CHECK(0.0 == Approx(jacobian[1 * num_obs - 1]).margin(1e-7));
        }

        DYNAMIC_SECTION("Op=[RX,RX,RX], Obs=[Z,Z,Z] - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<size_t> tp{0, 1, 2};
            const size_t num_qubits = 3;
            const size_t num_params = 3;
            const size_t num_obs = 3;
            std::vector<PrecisionT> jacobian(num_obs * tp.size(), 0);

            std::vector<ComplexT> cdata(1U << num_qubits);
            cdata[0] = ComplexT{1, 0};

            StateVectorT psi(cdata.data(), cdata.size());

            const auto obs1 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{0});
            const auto obs2 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{1});
            const auto obs3 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{2});

            auto ops = OpsData<StateVectorT>(
                {"RX", "RX", "RX"}, {{param[0]}, {param[1]}, {param[2]}},
                {{0}, {1}, {2}}, {false, false, false});

            JacobianData<StateVectorT> tape{num_params,    psi.getLength(),
                                            psi.getData(), {obs1, obs2, obs3},
                                            ops,           tp};
            adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

            CAPTURE(jacobian);
            CHECK(-sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
            CHECK(-sin(param[1]) ==
                  Approx(jacobian[1 * num_params + 1]).margin(1e-7));
            CHECK(-sin(param[2]) ==
                  Approx(jacobian[2 * num_params + 2]).margin(1e-7));
        }

        DYNAMIC_SECTION("Op=[RX,RX,RX], Obs=[Z,Z,Z], TParams=[0,2] - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
            std::vector<size_t> t_params{0, 2};
            const size_t num_qubits = 3;
            const size_t num_params = 3;
            const size_t num_obs = 3;
            std::vector<PrecisionT> jacobian(num_obs * t_params.size(), 0);

            std::vector<ComplexT> cdata(1U << num_qubits);
            cdata[0] = ComplexT{1, 0};
            StateVectorT psi(cdata.data(), cdata.size());

            const auto obs1 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{0});
            const auto obs2 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{1});
            const auto obs3 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{2});

            auto ops = OpsData<StateVectorT>(
                {"RX", "RX", "RX"}, {{param[0]}, {param[1]}, {param[2]}},
                {{0}, {1}, {2}}, {false, false, false});

            JacobianData<StateVectorT> tape{num_params,    psi.getLength(),
                                            psi.getData(), {obs1, obs2, obs3},
                                            ops,           t_params};

            adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

            CAPTURE(jacobian);
            CHECK(-sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
            CHECK(0 == Approx(jacobian[1 * t_params.size() + 1]).margin(1e-7));
            CHECK(-sin(param[2]) ==
                  Approx(jacobian[2 * t_params.size() + 1]).margin(1e-7));
        }

        DYNAMIC_SECTION("Op=[RX,RX,RX], Obs=[ZZZ] - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
            std::vector<size_t> tp{0, 1, 2};
            const size_t num_qubits = 3;
            const size_t num_params = 3;
            const size_t num_obs = 1;
            std::vector<PrecisionT> jacobian(num_obs * tp.size(), 0);

            std::vector<ComplexT> cdata(1U << num_qubits);
            cdata[0] = ComplexT{1, 0};
            StateVectorT psi(cdata.data(), cdata.size());

            const auto obs = std::make_shared<TensorProdObs<StateVectorT>>(
                std::make_shared<NamedObs<StateVectorT>>(
                    "PauliZ", std::vector<size_t>{0}),
                std::make_shared<NamedObs<StateVectorT>>(
                    "PauliZ", std::vector<size_t>{1}),
                std::make_shared<NamedObs<StateVectorT>>(
                    "PauliZ", std::vector<size_t>{2}));
            auto ops = OpsData<StateVectorT>(
                {"RX", "RX", "RX"}, {{param[0]}, {param[1]}, {param[2]}},
                {{0}, {1}, {2}}, {false, false, false});

            JacobianData<StateVectorT> tape{
                num_params, psi.getLength(), psi.getData(), {obs}, ops, tp};

            adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

            CAPTURE(jacobian);

            // Computed with parameter shift
            CHECK(-0.1755096592645253 == Approx(jacobian[0]).margin(1e-7));
            CHECK(0.26478810666384334 == Approx(jacobian[1]).margin(1e-7));
            CHECK(-0.6312451595102775 == Approx(jacobian[2]).margin(1e-7));
        }

        DYNAMIC_SECTION("Op=Mixed, Obs=[XXX] - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
            std::vector<size_t> tp{0, 1, 2, 3, 4, 5};
            const size_t num_qubits = 3;
            const size_t num_params = 6;
            const size_t num_obs = 1;
            std::vector<PrecisionT> jacobian(num_obs * tp.size(), 0);

            std::vector<ComplexT> cdata(1U << num_qubits);
            cdata[0] = ComplexT{1, 0};
            StateVectorT psi(cdata.data(), cdata.size());

            const auto obs = std::make_shared<TensorProdObs<StateVectorT>>(
                std::make_shared<NamedObs<StateVectorT>>(
                    "PauliX", std::vector<size_t>{0}),
                std::make_shared<NamedObs<StateVectorT>>(
                    "PauliX", std::vector<size_t>{1}),
                std::make_shared<NamedObs<StateVectorT>>(
                    "PauliX", std::vector<size_t>{2}));
            std::vector<ComplexT> cnot{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0};
            auto ops = OpsData<StateVectorT>(
                {"RZ", "RY", "RZ", "QubitUnitary", "CNOT", "RZ", "RY", "RZ"},
                {{param[0]},
                 {param[1]},
                 {param[2]},
                 {},
                 {},
                 {param[0]},
                 {param[1]},
                 {param[2]}},
                {{0}, {0}, {0}, {0, 1}, {1, 2}, {1}, {1}, {1}},
                {false, false, false, false, false, false, false, false},
                std::vector<std::vector<ComplexT>>{
                    {}, {}, {}, cnot, {}, {}, {}, {}},
                std::vector<std::vector<size_t>>{
                    {}, {}, {}, {}, {}, {}, {}, {}},
                std::vector<std::vector<bool>>{{}, {}, {}, {}, {}, {}, {}, {}});

            JacobianData<StateVectorT> tape{
                num_params, psi.getLength(), psi.getData(), {obs}, ops, tp};

            adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

            CAPTURE(jacobian);

            // Computed with PennyLane using default.qubit.adjoint_jacobian
            CHECK(0.0 == Approx(jacobian[0]).margin(1e-7));
            CHECK(-0.674214427 == Approx(jacobian[1]).margin(1e-7));
            CHECK(0.275139672 == Approx(jacobian[2]).margin(1e-7));
            CHECK(0.275139672 == Approx(jacobian[3]).margin(1e-7));
            CHECK(-0.0129093062 == Approx(jacobian[4]).margin(1e-7));
            CHECK(0.323846156 == Approx(jacobian[5]).margin(1e-7));
        }

        DYNAMIC_SECTION("Decomposed Rot gate, non computational basis state - "
                        << StateVectorToName<StateVectorT>::name) {
            const std::vector<size_t> tp{0, 1, 2};
            const size_t num_params = 3;
            const size_t num_obs = 1;

            PrecisionT limit = 2 * M_PI;
            const std::vector<PrecisionT> thetas = linspace(-limit, limit, 7);

            std::vector<std::vector<PrecisionT>> expec_results{
                {0, -9.90819496e-01, 0},
                {-8.18996553e-01, 1.62526544e-01, 0},
                {-0.203949, 0.48593716, 0},
                {0, 1, 0},
                {-2.03948985e-01, 4.85937177e-01, 0},
                {-8.18996598e-01, 1.62526487e-01, 0},
                {0, -9.90819511e-01, 0}};

            const auto obs = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{0});

            for (size_t i = 0; i < thetas.size(); i++) {
                const PrecisionT theta = thetas[i];
                std::vector<PrecisionT> local_params{
                    theta, std::pow(theta, (PrecisionT)3),
                    Pennylane::Util::SQRT2<PrecisionT>() * theta};
                std::vector<PrecisionT> jacobian(num_obs * tp.size(), 0);

                std::vector<ComplexT> cdata{
                    Pennylane::Util::INVSQRT2<PrecisionT>(),
                    -Pennylane::Util::INVSQRT2<PrecisionT>()};
                StateVectorT psi(cdata.data(), cdata.size());

                auto ops = OpsData<StateVectorT>(
                    {"RZ", "RY", "RZ"},
                    {{local_params[0]}, {local_params[1]}, {local_params[2]}},
                    {{0}, {0}, {0}}, {false, false, false});

                JacobianData<StateVectorT> tape{
                    num_params, psi.getLength(), psi.getData(), {obs}, ops, tp};
                adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

                CAPTURE(theta);
                CAPTURE(jacobian);

                // Computed with PennyLane using default.qubit
                CHECK(expec_results[i][0] == Approx(jacobian[0]).margin(1e-4));
                CHECK(expec_results[i][1] == Approx(jacobian[1]).margin(1e-4));
                CHECK(expec_results[i][2] == Approx(jacobian[2]).margin(1e-4));
            }
        }

        DYNAMIC_SECTION("Mixed Ops, Obs and TParams - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
            const std::vector<size_t> t_params{1, 2, 3};
            const size_t num_obs = 1;

            PrecisionT limit = 2 * M_PI;
            const std::vector<PrecisionT> thetas = linspace(-limit, limit, 8);

            std::vector<PrecisionT> local_params{0.543, 0.54, 0.1,  0.5, 1.3,
                                                 -2.3,  0.5,  -0.5, 0.5};
            std::vector<PrecisionT> jacobian(num_obs * t_params.size(), 0);

            std::vector<ComplexT> cdata{Pennylane::Util::ONE<PrecisionT>(),
                                        Pennylane::Util::ZERO<PrecisionT>(),
                                        Pennylane::Util::ZERO<PrecisionT>(),
                                        Pennylane::Util::ZERO<PrecisionT>()};
            StateVectorT psi(cdata.data(), cdata.size());

            const auto obs = std::make_shared<TensorProdObs<StateVectorT>>(
                std::make_shared<NamedObs<StateVectorT>>(
                    "PauliX", std::vector<size_t>{0}),
                std::make_shared<NamedObs<StateVectorT>>(
                    "PauliZ", std::vector<size_t>{1}));

            auto ops = OpsData<StateVectorT>(
                {"Hadamard", "RX", "CNOT", "RZ", "RY", "RZ", "RZ", "RY", "RZ",
                 "RZ", "RY", "CNOT"},
                {{},
                 {local_params[0]},
                 {},
                 {local_params[1]},
                 {local_params[2]},
                 {local_params[3]},
                 {local_params[4]},
                 {local_params[5]},
                 {local_params[6]},
                 {local_params[7]},
                 {local_params[8]},
                 {}},
                {{0},
                 {0},
                 {0, 1},
                 {0},
                 {0},
                 {0},
                 {0},
                 {0},
                 {0},
                 {0},
                 {1},
                 {0, 1}},
                {false, false, false, false, false, false, false, false, false,
                 false, false, false});

            JacobianData<StateVectorT> tape{
                t_params.size(), psi.getLength(), psi.getData(), {obs}, ops,
                t_params};
            adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

            std::vector<PrecisionT> expected{-0.71429188, 0.04998561,
                                             -0.71904837};
            // Computed with PennyLane using default.qubit
            CHECK(expected[0] == Approx(jacobian[0]));
            CHECK(expected[1] == Approx(jacobian[1]));
            CHECK(expected[2] == Approx(jacobian[2]));
        }

        DYNAMIC_SECTION("Op=RX, Obs=Ham[Z0+Z1] - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
            std::vector<size_t> tp{0};
            const size_t num_qubits = 2;
            const size_t num_params = 1;
            const size_t num_obs = 1;
            std::vector<PrecisionT> jacobian(num_obs * tp.size(), 0);

            std::vector<ComplexT> cdata(1U << num_qubits);
            cdata[0] = ComplexT{1, 0};
            StateVectorT psi(cdata.data(), cdata.size());

            const auto obs1 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{0});
            const auto obs2 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{1});

            auto ham =
                Hamiltonian<StateVectorT>::create({0.3, 0.7}, {obs1, obs2});

            auto ops =
                OpsData<StateVectorT>({"RX"}, {{param[0]}}, {{0}}, {false});

            JacobianData<StateVectorT> tape{
                num_params, psi.getLength(), psi.getData(), {ham}, ops, tp};

            adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

            CAPTURE(jacobian);
            CHECK(-0.3 * sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
        }

        DYNAMIC_SECTION("Op=[RX,RX,RX], Obs=Ham[Z0+Z1+Z2], TParams=[0,2] - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
            std::vector<size_t> t_params{0, 2};
            const size_t num_qubits = 3;
            const size_t num_params = 3;
            const size_t num_obs = 1;
            std::vector<PrecisionT> jacobian(num_obs * t_params.size(), 0);

            std::vector<ComplexT> cdata(1U << num_qubits);
            cdata[0] = ComplexT{1, 0};
            StateVectorT psi(cdata.data(), cdata.size());

            auto obs1 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{0});
            auto obs2 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{1});
            auto obs3 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{2});

            auto ham = Hamiltonian<StateVectorT>::create({0.47, 0.32, 0.96},
                                                         {obs1, obs2, obs3});

            auto ops = OpsData<StateVectorT>(
                {"RX", "RX", "RX"}, {{param[0]}, {param[1]}, {param[2]}},
                {{0}, {1}, {2}}, {false, false, false});

            JacobianData<StateVectorT> tape{num_params,    psi.getLength(),
                                            psi.getData(), {ham},
                                            ops,           t_params};
            adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

            CAPTURE(jacobian);
            CHECK((-0.47 * sin(param[0]) == Approx(jacobian[0]).margin(1e-7)));
            CHECK((-0.96 * sin(param[2]) == Approx(jacobian[1]).margin(1e-7)));
        }

        DYNAMIC_SECTION("HermitianObs - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
            std::vector<size_t> t_params{0, 2};
            const size_t num_qubits = 3;
            const size_t num_params = 3;
            const size_t num_obs = 1;
            std::vector<PrecisionT> jacobian1(num_obs * t_params.size(), 0);
            std::vector<PrecisionT> jacobian2(num_obs * t_params.size(), 0);

            std::vector<ComplexT> cdata(1U << num_qubits);
            cdata[0] = ComplexT{1, 0};
            StateVectorT psi(cdata.data(), cdata.size());

            auto obs1 = std::make_shared<TensorProdObs<StateVectorT>>(
                std::make_shared<NamedObs<StateVectorT>>(
                    "PauliZ", std::vector<size_t>{0}),
                std::make_shared<NamedObs<StateVectorT>>(
                    "PauliZ", std::vector<size_t>{1}));
            auto obs2 = std::make_shared<HermitianObs<StateVectorT>>(
                std::vector<ComplexT>{1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                                      0, 0, 1},
                std::vector<size_t>{0, 1});

            auto ops = OpsData<StateVectorT>(
                {"RX", "RX", "RX"}, {{param[0]}, {param[1]}, {param[2]}},
                {{0}, {1}, {2}}, {false, false, false});

            JacobianData<StateVectorT> tape1{num_params,    psi.getLength(),
                                             psi.getData(), {obs1},
                                             ops,           t_params};

            JacobianData<StateVectorT> tape2{num_params,    psi.getLength(),
                                             psi.getData(), {obs2},
                                             ops,           t_params};
            adj.adjointJacobian(std::span{jacobian1}, tape1, psi, true);
            adj.adjointJacobian(std::span{jacobian2}, tape2, psi, true);

            CHECK((jacobian1 == PLApprox(jacobian2).margin(1e-7)));
        }

        testAdjointJacobian<typename TypeList::Next>();
    }
}

TEST_CASE("Algorithms::adjointJacobian", "[Algorithms]") {
    if constexpr (BACKEND_FOUND) {
        testAdjointJacobian<TestStateVectorBackends>();
    }
}