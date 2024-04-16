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
#include "TestHelpers.hpp"
#include <catch2/catch.hpp>

/// @cond DEV
namespace {
using Pennylane::Util::isApproxEqual;
} // namespace
/// @endcond
#include <algorithm>
#include <string>

#ifdef _ENABLE_PLQUBIT
constexpr bool BACKEND_FOUND = true;

#include "MeasurementsLQubit.hpp"
#include "ObservablesLQubit.hpp"
#include "TestHelpersStateVectors.hpp" // TestStateVectorBackends, StateVectorToName
#include "TestHelpersWires.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit::Measures;
using namespace Pennylane::LightningQubit::Observables;
using namespace Pennylane::LightningQubit::Util;
} // namespace
/// @endcond

#elif _ENABLE_PLKOKKOS == 1
constexpr bool BACKEND_FOUND = true;

#include "MeasurementsKokkos.hpp"
#include "ObservablesKokkos.hpp"
#include "TestHelpersStateVectors.hpp" // TestStateVectorBackends, StateVectorToName
#include "TestHelpersWires.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::Observables;
using namespace Pennylane::LightningKokkos::Util;
} // namespace
  /// @endcond

#elif _ENABLE_PLGPU == 1
constexpr bool BACKEND_FOUND = true;
#include "MeasurementsGPU.hpp"
#include "ObservablesGPU.hpp"
#include "TestHelpersStateVectors.hpp"
#include "TestHelpersWires.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU::Measures;
using namespace Pennylane::LightningGPU::Observables;
} // namespace
  /// @endcond

#else
constexpr bool BACKEND_FOUND = false;
using TestStateVectorBackends = Pennylane::Util::TypeList<void>;

template <class StateVector> struct StateVectorToName {};
#endif

template <typename TypeList> void testProbabilities() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;

        // Expected results calculated with Pennylane default.qubit:
        std::vector<std::pair<std::vector<size_t>, std::vector<PrecisionT>>>
            input = {
#ifdef _ENABLE_PLGPU
                // Bit index reodering conducted in the python layer
                // for L-GPU. Also L-GPU backend doesn't support
                // out of order wires for probability calculation
                {{2, 1, 0},
                 {0.67078706, 0.03062806, 0.0870997, 0.00397696, 0.17564072,
                  0.00801973, 0.02280642, 0.00104134}}
#else
                {{0, 1, 2},
                 {0.67078706, 0.03062806, 0.0870997, 0.00397696, 0.17564072,
                  0.00801973, 0.02280642, 0.00104134}},
                {{0, 2, 1},
                 {0.67078706, 0.0870997, 0.03062806, 0.00397696, 0.17564072,
                  0.02280642, 0.00801973, 0.00104134}},
                {{1, 0, 2},
                 {0.67078706, 0.03062806, 0.17564072, 0.00801973, 0.0870997,
                  0.00397696, 0.02280642, 0.00104134}},
                {{1, 2, 0},
                 {0.67078706, 0.0870997, 0.17564072, 0.02280642, 0.03062806,
                  0.00397696, 0.00801973, 0.00104134}},
                {{2, 0, 1},
                 {0.67078706, 0.17564072, 0.03062806, 0.00801973, 0.0870997,
                  0.02280642, 0.00397696, 0.00104134}},
                {{2, 1, 0},
                 {0.67078706, 0.17564072, 0.0870997, 0.02280642, 0.03062806,
                  0.00801973, 0.00397696, 0.00104134}},
                {{0, 1}, {0.70141512, 0.09107666, 0.18366045, 0.02384776}},
                {{0, 2}, {0.75788676, 0.03460502, 0.19844714, 0.00906107}},
                {{1, 2}, {0.84642778, 0.0386478, 0.10990612, 0.0050183}},
                {{2, 1}, {0.84642778, 0.10990612, 0.0386478, 0.0050183}},
                {{0}, {0.79249179, 0.20750821}},
                {{1}, {0.88507558, 0.11492442}},
                {{2}, {0.9563339, 0.0436661}}
#endif
            };

        // Defining the Statevector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measurements class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        std::vector<PrecisionT> probabilities;

        DYNAMIC_SECTION("Looping over different wire configurations - "
                        << StateVectorToName<StateVectorT>::name) {
            for (const auto &term : input) {
                probabilities = Measurer.probs(term.first);
                REQUIRE_THAT(term.second,
                             Catch::Approx(probabilities).margin(1e-6));
            }
        }

        testProbabilities<typename TypeList::Next>();
    }
}

TEST_CASE("Probabilities", "[MeasurementsBase]") {
    if constexpr (BACKEND_FOUND) {
        testProbabilities<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testProbabilitiesShots() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;

        // Expected results calculated with Pennylane default.qubit:
        std::vector<std::pair<std::vector<size_t>, std::vector<PrecisionT>>>
            input = {// prob shots only support in-order target wires for now
                     {{0, 1, 2},
                      {0.67078706, 0.03062806, 0.0870997, 0.00397696,
                       0.17564072, 0.00801973, 0.02280642, 0.00104134}},
                     {{0, 1}, {0.70141512, 0.09107666, 0.18366045, 0.02384776}},
                     {{0, 2}, {0.75788676, 0.03460502, 0.19844714, 0.00906107}},
                     {{1, 2}, {0.84642778, 0.0386478, 0.10990612, 0.0050183}},
                     {{0}, {0.79249179, 0.20750821}},
                     {{1}, {0.88507558, 0.11492442}},
                     {{2}, {0.9563339, 0.0436661}}};

        // Defining the Statevector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measurements class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        std::vector<PrecisionT> probabilities;
        DYNAMIC_SECTION(
            "Looping over different wire configurations - shots- fullsystem"
            << StateVectorToName<StateVectorT>::name) {
            size_t num_shots = 10000;
            probabilities = Measurer.probs(num_shots);
            REQUIRE_THAT(input[0].second,
                         Catch::Approx(probabilities).margin(5e-2));
        }

        DYNAMIC_SECTION(
            "Looping over different wire configurations - shots- sub system"
            << StateVectorToName<StateVectorT>::name) {
            for (const auto &term : input) {
                size_t num_shots = 10000;
                probabilities = Measurer.probs(term.first, num_shots);
                REQUIRE_THAT(term.second,
                             Catch::Approx(probabilities).margin(5e-2));
            }
        }

        testProbabilitiesShots<typename TypeList::Next>();
    }
}

TEST_CASE("Probabilities Shots", "[MeasurementsBase]") {
    if constexpr (BACKEND_FOUND) {
        testProbabilitiesShots<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testProbabilitiesObs() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;

        const size_t num_qubits = 3;

        // Defining the Statevector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        auto sv_data = createNonTrivialState<StateVectorT>();

        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        StateVectorT sv(sv_data.data(), sv_data.size());

        DYNAMIC_SECTION("Test PauliX"
                        << StateVectorToName<StateVectorT>::name) {
            for (size_t i = 0; i < num_qubits; i++) {
                NamedObs<StateVectorT> obs("PauliX", {i});
                Measurements<StateVectorT> Measurer_obs(statevector);

                sv.applyOperation("Hadamard", {i}, false);

                Measurements<StateVectorT> Measurer(sv);

                auto prob_obs = Measurer_obs.probs(obs);
                auto prob = Measurer.probs(std::vector<size_t>({i}));

                REQUIRE_THAT(prob_obs, Catch::Approx(prob).margin(1e-6));
            }
        }

        DYNAMIC_SECTION("Test PauliY"
                        << StateVectorToName<StateVectorT>::name) {
            for (size_t i = 0; i < num_qubits; i++) {
                NamedObs<StateVectorT> obs("PauliY", {i});
                Measurements<StateVectorT> Measurer_obs(statevector);

                sv.applyOperations({"PauliZ", "S", "Hadamard"}, {{i}, {i}, {i}},
                                   {false, false, false});

                Measurements<StateVectorT> Measurer(sv);

                auto prob_obs = Measurer_obs.probs(obs);
                auto prob = Measurer.probs(std::vector<size_t>({i}));

                REQUIRE_THAT(prob_obs, Catch::Approx(prob).margin(1e-6));
            }
        }

        DYNAMIC_SECTION("Test PauliZ"
                        << StateVectorToName<StateVectorT>::name) {
            for (size_t i = 0; i < num_qubits; i++) {
                NamedObs<StateVectorT> obs("PauliZ", {i});
                Measurements<StateVectorT> Measurer_obs(statevector);

                Measurements<StateVectorT> Measurer(sv);

                auto prob_obs = Measurer_obs.probs(obs);
                auto prob = Measurer.probs(std::vector<size_t>({i}));

                REQUIRE_THAT(prob_obs, Catch::Approx(prob).margin(1e-6));
            }
        }

        DYNAMIC_SECTION("Test Hadamard"
                        << StateVectorToName<StateVectorT>::name) {
            for (size_t i = 0; i < num_qubits; i++) {
                NamedObs<StateVectorT> obs("Hadamard", {i});
                Measurements<StateVectorT> Measurer_obs(statevector);
                const PrecisionT theta = -M_PI / 4.0;
                sv.applyOperation("RY", {i}, false, {theta});

                Measurements<StateVectorT> Measurer(sv);

                auto prob_obs = Measurer_obs.probs(obs);
                auto prob = Measurer.probs(std::vector<size_t>({i}));

                REQUIRE_THAT(prob_obs, Catch::Approx(prob).margin(1e-6));
            }
        }

        DYNAMIC_SECTION("Test Identity"
                        << StateVectorToName<StateVectorT>::name) {
            for (size_t i = 0; i < num_qubits; i++) {
                NamedObs<StateVectorT> obs("Identity", {i});
                Measurements<StateVectorT> Measurer_obs(statevector);

                Measurements<StateVectorT> Measurer(sv);

                auto prob_obs = Measurer_obs.probs(obs);
                auto prob = Measurer.probs(std::vector<size_t>({i}));

                REQUIRE_THAT(prob_obs, Catch::Approx(prob).margin(1e-6));
            }
        }

        DYNAMIC_SECTION("Test TensorProd XYZ"
                        << StateVectorToName<StateVectorT>::name) {
            auto X0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliX", std::vector<size_t>{0});
            auto Z1 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{1});
            auto Y2 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliY", std::vector<size_t>{2});
            auto obs = TensorProdObs<StateVectorT>::create({X0, Z1, Y2});

            Measurements<StateVectorT> Measurer_obs(statevector);

            sv.applyOperations({"Hadamard", "PauliZ", "S", "Hadamard"},
                               {{0}, {2}, {2}, {2}},
                               {false, false, false, false});

            Measurements<StateVectorT> Measurer(sv);

            auto prob_obs = Measurer_obs.probs(*obs);
            auto prob = Measurer.probs(std::vector<size_t>({0, 1, 2}));

            REQUIRE_THAT(prob_obs, Catch::Approx(prob).margin(1e-6));
        }

        DYNAMIC_SECTION("Test TensorProd YHI"
                        << StateVectorToName<StateVectorT>::name) {
            auto Y0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliY", std::vector<size_t>{0});
            auto H1 = std::make_shared<NamedObs<StateVectorT>>(
                "Hadamard", std::vector<size_t>{1});
            auto I2 = std::make_shared<NamedObs<StateVectorT>>(
                "Identity", std::vector<size_t>{2});
            auto obs = TensorProdObs<StateVectorT>::create({Y0, H1, I2});

            Measurements<StateVectorT> Measurer_obs(statevector);

            sv.applyOperations({"PauliZ", "S", "Hadamard"}, {{0}, {0}, {0}},
                               {false, false, false});
            const PrecisionT theta = -M_PI / 4.0;
            sv.applyOperation("RY", {1}, false, {theta});

            Measurements<StateVectorT> Measurer(sv);

            auto prob_obs = Measurer_obs.probs(*obs);
            auto prob = Measurer.probs(std::vector<size_t>({0, 1, 2}));

            REQUIRE_THAT(prob_obs, Catch::Approx(prob).margin(1e-6));
        }

        testProbabilitiesObs<typename TypeList::Next>();
    }
}

TEST_CASE("Probabilities Obs", "[MeasurementsBase]") {
    if constexpr (BACKEND_FOUND) {
        testProbabilitiesObs<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testProbabilitiesObsShots() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;

        // Defining the Statevector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        auto sv_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        StateVectorT sv(sv_data.data(), sv_data.size());

        DYNAMIC_SECTION("Test TensorProd XYZ"
                        << StateVectorToName<StateVectorT>::name) {
            auto X0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliX", std::vector<size_t>{0});
            auto Z1 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{1});
            auto Y2 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliY", std::vector<size_t>{2});
            auto obs = TensorProdObs<StateVectorT>::create({X0, Z1, Y2});

            Measurements<StateVectorT> Measurer_obs_shots(statevector);

            sv.applyOperations({"Hadamard", "PauliZ", "S", "Hadamard"},
                               {{0}, {2}, {2}, {2}},
                               {false, false, false, false});

            Measurements<StateVectorT> Measurer(sv);

            size_t num_shots = 10000;
            auto prob_obs_shots = Measurer_obs_shots.probs(*obs, num_shots);

#ifdef _ENABLE_PLGPU
            auto prob = Measurer.probs(std::vector<size_t>({2, 1, 0}));
#else
            auto prob = Measurer.probs(std::vector<size_t>({0, 1, 2}));
#endif

            REQUIRE_THAT(prob_obs_shots, Catch::Approx(prob).margin(5e-2));
        }

        DYNAMIC_SECTION("Test TensorProd YHI"
                        << StateVectorToName<StateVectorT>::name) {
            auto Y0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliY", std::vector<size_t>{0});
            auto H1 = std::make_shared<NamedObs<StateVectorT>>(
                "Hadamard", std::vector<size_t>{1});
            auto I2 = std::make_shared<NamedObs<StateVectorT>>(
                "Identity", std::vector<size_t>{2});
            auto obs = TensorProdObs<StateVectorT>::create({Y0, H1, I2});

            Measurements<StateVectorT> Measurer_obs_shots(statevector);

            sv.applyOperations({"PauliZ", "S", "Hadamard"}, {{0}, {0}, {0}},
                               {false, false, false});
            const PrecisionT theta = -M_PI / 4.0;
            sv.applyOperation("RY", {1}, false, {theta});

            Measurements<StateVectorT> Measurer(sv);

            size_t num_shots = 10000;
            auto prob_obs_shots = Measurer_obs_shots.probs(*obs, num_shots);
#ifdef _ENABLE_PLGPU
            auto prob = Measurer.probs(std::vector<size_t>({2, 1, 0}));
#else
            auto prob = Measurer.probs(std::vector<size_t>({0, 1, 2}));
#endif

            REQUIRE_THAT(prob_obs_shots, Catch::Approx(prob).margin(5e-2));
        }

        testProbabilitiesObsShots<typename TypeList::Next>();
    }
}

TEST_CASE("Probabilities Obs Shots", "[MeasurementsBase]") {
    if constexpr (BACKEND_FOUND) {
        testProbabilitiesObsShots<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testNamedObsExpval() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;

        // Defining the State Vector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::string> obs_name = {"PauliX", "PauliY", "PauliZ"};
        // Expected results calculated with Pennylane default.qubit:
        std::vector<std::vector<PrecisionT>> exp_values_ref = {
            {0.49272486, 0.42073549, 0.28232124},
            {-0.64421768, -0.47942553, -0.29552020},
            {0.58498357, 0.77015115, 0.91266780}};

        for (size_t ind_obs = 0; ind_obs < obs_name.size(); ind_obs++) {
            DYNAMIC_SECTION(obs_name[ind_obs]
                            << " - Varying wires"
                            << StateVectorToName<StateVectorT>::name) {
                for (size_t ind_wires = 0; ind_wires < wires_list.size();
                     ind_wires++) {
                    NamedObs<StateVectorT> obs(obs_name[ind_obs],
                                               wires_list[ind_wires]);
                    PrecisionT expected = exp_values_ref[ind_obs][ind_wires];
                    PrecisionT result = Measurer.expval(obs);
                    REQUIRE(expected == Approx(result).margin(1e-6));
                }
            }
        }
        testNamedObsExpval<typename TypeList::Next>();
    }
}

TEST_CASE("Expval - NamedObs", "[MeasurementsBase][Observables]") {
    if constexpr (BACKEND_FOUND) {
        testNamedObsExpval<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testNamedObsExpvalShot() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;

        // Defining the State Vector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::string> obs_name = {"PauliX", "PauliY", "PauliZ",
                                             "Hadamard", "Identity"};
        // Expected results calculated with Pennylane default.qubit:
        std::vector<std::vector<PrecisionT>> exp_values_ref = {
            {0.49272486, 0.42073549, 0.28232124},
            {-0.64421768, -0.47942553, -0.29552020},
            {0.58498357, 0.77015115, 0.91266780},
            {0.7620549436, 0.8420840225, 0.8449848566},
            {1.0, 1.0, 1.0}};
        for (size_t ind_obs = 0; ind_obs < obs_name.size(); ind_obs++) {
            DYNAMIC_SECTION(obs_name[ind_obs]
                            << " - Varying wires"
                            << StateVectorToName<StateVectorT>::name) {
                size_t num_shots = 20000;
                std::vector<size_t> shots_range = {};
                for (size_t ind_wires = 0; ind_wires < wires_list.size();
                     ind_wires++) {
                    NamedObs<StateVectorT> obs(obs_name[ind_obs],
                                               wires_list[ind_wires]);
                    PrecisionT expected = exp_values_ref[ind_obs][ind_wires];
                    PrecisionT result =
                        Measurer.expval(obs, num_shots, shots_range);
                    REQUIRE_THAT(result,
                                 Catch::Matchers::WithinRel(
                                     expected, static_cast<PrecisionT>(0.20)));
                }
            }
        }

        for (size_t ind_obs = 0; ind_obs < obs_name.size(); ind_obs++) {
            DYNAMIC_SECTION(obs_name[ind_obs]
                            << " - Varying wires-with shots_range"
                            << StateVectorToName<StateVectorT>::name) {
                size_t num_shots = 20000;
                std::vector<size_t> shots_range;
                for (size_t i = 0; i < num_shots; i += 2) {
                    shots_range.push_back(i);
                }
                for (size_t ind_wires = 0; ind_wires < wires_list.size();
                     ind_wires++) {
                    NamedObs<StateVectorT> obs(obs_name[ind_obs],
                                               wires_list[ind_wires]);
                    PrecisionT expected = exp_values_ref[ind_obs][ind_wires];
                    PrecisionT result =
                        Measurer.expval(obs, num_shots, shots_range);
                    REQUIRE_THAT(result,
                                 Catch::Matchers::WithinRel(
                                     expected, static_cast<PrecisionT>(0.20)));
                }
            }
        }
        testNamedObsExpvalShot<typename TypeList::Next>();
    }
}

TEST_CASE("Expval Shot- NamedObs", "[MeasurementsBase][Observables]") {
    if constexpr (BACKEND_FOUND) {
        testNamedObsExpvalShot<TestStateVectorBackends>();
    }
}

#ifdef PL_USE_LAPACK
template <typename TypeList> void testHermitianObsExpvalShot() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;
        using MatrixT = std::vector<ComplexT>;

        // Defining the State Vector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the Measurements class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer_shots(statevector);
        Measurements<StateVectorT> Measurer(statevector);

        DYNAMIC_SECTION("2x2 Hermitian matrix0"
                        << StateVectorToName<StateVectorT>::name) {
            MatrixT Hermitian_matrix{ComplexT{-3, 0}, ComplexT{2, 1},
                                     ComplexT{2, -1}, ComplexT{2, 0}};

            HermitianObs<StateVectorT> obs(Hermitian_matrix, {1});
            size_t num_shots = 80000;
            std::vector<size_t> shots_range = {};

            PrecisionT expected = Measurer.expval(obs);

            PrecisionT result_shots =
                Measurer_shots.expval(obs, num_shots, shots_range);

            REQUIRE_THAT(result_shots,
                         Catch::Matchers::WithinRel(
                             expected, static_cast<PrecisionT>(0.20)));
        }

        DYNAMIC_SECTION("2x2 Hermitian matrix1"
                        << StateVectorToName<StateVectorT>::name) {
            MatrixT Hermitian_matrix{ComplexT{0, 0}, ComplexT{0, -1},
                                     ComplexT{0, 1}, ComplexT{0, 0}};

            HermitianObs<StateVectorT> obs(Hermitian_matrix, {1});
            size_t num_shots = 20000;
            std::vector<size_t> shots_range = {};

            PrecisionT expected = Measurer.expval(obs);

            PrecisionT result_shots =
                Measurer_shots.expval(obs, num_shots, shots_range);

            REQUIRE_THAT(result_shots,
                         Catch::Matchers::WithinRel(
                             expected, static_cast<PrecisionT>(0.20)));
        }

        DYNAMIC_SECTION("2x2 Hermitian matrix2"
                        << StateVectorToName<StateVectorT>::name) {
            MatrixT Hermitian_matrix{ComplexT{0, 0}, ComplexT{1, 0},
                                     ComplexT{1, 0}, ComplexT{0, 0}};

            HermitianObs<StateVectorT> obs(Hermitian_matrix, {1});
            size_t num_shots = 20000;
            std::vector<size_t> shots_range = {};

            PrecisionT expected = Measurer.expval(obs);

            PrecisionT result_shots =
                Measurer_shots.expval(obs, num_shots, shots_range);

            REQUIRE_THAT(result_shots,
                         Catch::Matchers::WithinRel(
                             expected, static_cast<PrecisionT>(0.20)));
        }

        DYNAMIC_SECTION("2x2 Hermitian matrix3"
                        << StateVectorToName<StateVectorT>::name) {
            MatrixT Hermitian_matrix{ComplexT{1, 0}, ComplexT{0, 0},
                                     ComplexT{0, 0}, ComplexT{-1, 0}};

            HermitianObs<StateVectorT> obs(Hermitian_matrix, {1});
            size_t num_shots = 20000;
            std::vector<size_t> shots_range = {};

            PrecisionT expected = Measurer.expval(obs);

            PrecisionT result_shots =
                Measurer_shots.expval(obs, num_shots, shots_range);

            REQUIRE_THAT(result_shots,
                         Catch::Matchers::WithinRel(
                             expected, static_cast<PrecisionT>(0.20)));
        }

        testHermitianObsExpvalShot<typename TypeList::Next>();
    }
}

TEST_CASE("Expval Shot - HermitianObs ", "[MeasurementsBase][Observables]") {
    if constexpr (BACKEND_FOUND) {
        testHermitianObsExpvalShot<TestStateVectorBackends>();
    }
}
#endif

template <typename TypeList> void testHermitianObsExpval() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;
        using MatrixT = std::vector<ComplexT>;

        // Defining the State Vector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        const PrecisionT theta = M_PI / 2;
        const PrecisionT real_term = std::cos(theta);
        const PrecisionT imag_term = std::sin(theta);

        DYNAMIC_SECTION("Varying wires - 2x2 matrix - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
            // Expected results calculated with Pennylane default.qubit:
            std::vector<PrecisionT> exp_values_ref = {
                0.644217687237691, 0.4794255386042027, 0.29552020666133955};

            MatrixT Hermitian_matrix{real_term, ComplexT{0, imag_term},
                                     ComplexT{0, -imag_term}, real_term};

            for (size_t ind_wires = 0; ind_wires < wires_list.size();
                 ind_wires++) {
                HermitianObs<StateVectorT> obs(Hermitian_matrix,
                                               wires_list[ind_wires]);
                PrecisionT expected = exp_values_ref[ind_wires];
                PrecisionT result = Measurer.expval(obs);
                REQUIRE(expected == Approx(result).margin(1e-6));
            }
        }

        DYNAMIC_SECTION("Varying wires - 4x4 matrix - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<std::vector<size_t>> wires_list = {
                {0, 1}, {0, 2}, {1, 2}, {2, 1}};
            // Expected results calculated with Pennylane default.qubit:
            std::vector<PrecisionT> exp_values_ref = {
                0.5874490024807637, 0.44170554255359035, 0.3764821318486682,
                0.5021569932};

            MatrixT Hermitian_matrix(16);
            Hermitian_matrix[0] = real_term;
            Hermitian_matrix[1] = ComplexT{0, imag_term};
            Hermitian_matrix[4] = ComplexT{0, -imag_term};
            Hermitian_matrix[5] = real_term;
            Hermitian_matrix[10] = ComplexT{1.0, 0};
            Hermitian_matrix[15] = ComplexT{1.0, 0};

            for (size_t ind_wires = 0; ind_wires < wires_list.size();
                 ind_wires++) {
                HermitianObs<StateVectorT> obs(Hermitian_matrix,
                                               wires_list[ind_wires]);
                PrecisionT expected = exp_values_ref[ind_wires];
                PrecisionT result = Measurer.expval(obs);
                REQUIRE(expected == Approx(result).margin(1e-6));
            }
        }

        testHermitianObsExpval<typename TypeList::Next>();
    }
}

TEST_CASE("Expval - HermitianObs", "[MeasurementsBase][Observables]") {
    if constexpr (BACKEND_FOUND) {
        testHermitianObsExpval<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testTensorProdObsExpvalShot() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = StateVectorT::ComplexT;

        // Defining the State Vector that will be measured.
        std::vector<ComplexT> statevector_data{
            {0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1}, {0.1, 0.2},
            {0.2, 0.2}, {0.3, 0.3}, {0.3, 0.4}, {0.4, 0.5}};
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        DYNAMIC_SECTION(" - Without shots_range"
                        << StateVectorToName<StateVectorT>::name) {
            size_t num_shots = 20000;
            std::vector<size_t> shots_range = {};
            auto X0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliX", std::vector<size_t>{0});
            auto Z1 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{1});
            auto obs = TensorProdObs<StateVectorT>::create({X0, Z1});
            auto expected = PrecisionT(-0.36);
            auto result = Measurer.expval(*obs, num_shots, shots_range);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(
                                     expected, static_cast<PrecisionT>(0.20)));
        }

        DYNAMIC_SECTION(" - With Identity but no shots_range"
                        << StateVectorToName<StateVectorT>::name) {
            size_t num_shots = 20000;
            std::vector<size_t> shots_range = {};
            auto X0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliX", std::vector<size_t>{0});
            auto I1 = std::make_shared<NamedObs<StateVectorT>>(
                "Identity", std::vector<size_t>{1});
            auto obs = TensorProdObs<StateVectorT>::create({X0, I1});
            PrecisionT expected = Measurer.expval(*obs);
            PrecisionT result = Measurer.expval(*obs, num_shots, shots_range);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(
                                     expected, static_cast<PrecisionT>(0.20)));
        }

        DYNAMIC_SECTION(" With shots_range"
                        << StateVectorToName<StateVectorT>::name) {
            size_t num_shots = 20000;
            std::vector<size_t> shots_range;
            for (size_t i = 0; i < num_shots; i += 2) {
                shots_range.push_back(i);
            }
            auto X0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliX", std::vector<size_t>{0});
            auto Z1 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{1});
            auto obs = TensorProdObs<StateVectorT>::create({X0, Z1});
            auto expected = PrecisionT(-0.36);
            auto result = Measurer.expval(*obs, num_shots, shots_range);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(
                                     expected, static_cast<PrecisionT>(0.20)));
        }

        DYNAMIC_SECTION(" With Identity and shots_range"
                        << StateVectorToName<StateVectorT>::name) {
            size_t num_shots = 20000;
            std::vector<size_t> shots_range;
            for (size_t i = 0; i < num_shots; i += 2) {
                shots_range.push_back(i);
            }
            auto X0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliX", std::vector<size_t>{0});
            auto I1 = std::make_shared<NamedObs<StateVectorT>>(
                "Identity", std::vector<size_t>{1});
            auto obs = TensorProdObs<StateVectorT>::create({X0, I1});
            PrecisionT expected = Measurer.expval(*obs);
            PrecisionT result = Measurer.expval(*obs, num_shots, shots_range);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(
                                     expected, static_cast<PrecisionT>(0.20)));
        }

#ifdef PL_USE_LAPACK
        DYNAMIC_SECTION(" With Identity and shots_range"
                        << StateVectorToName<StateVectorT>::name) {
            size_t num_shots = 80000;
            std::vector<size_t> shots_range = {};
            auto X0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliX", std::vector<size_t>{0});

            std::vector<ComplexT> Hermitian_matrix1{
                ComplexT{3, 0}, ComplexT{2, 1}, ComplexT{2, -1},
                ComplexT{-3, 0}};
            auto H1 = std::make_shared<HermitianObs<StateVectorT>>(
                Hermitian_matrix1, std::vector<size_t>{1});

            auto obs = TensorProdObs<StateVectorT>::create({X0, H1});
            PrecisionT expected = Measurer.expval(*obs);
            PrecisionT result = Measurer.expval(*obs, num_shots, shots_range);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(
                                     expected, static_cast<PrecisionT>(0.20)));
        }
#endif

        testTensorProdObsExpvalShot<typename TypeList::Next>();
    }
}

TEST_CASE("Expval Shot- TensorProdObs", "[MeasurementsBase][Observables]") {
    if constexpr (BACKEND_FOUND) {
        testTensorProdObsExpvalShot<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testNamedObsVar() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;

        // Defining the State Vector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::string> obs_name = {"PauliX", "PauliY", "PauliZ"};
        // Expected results calculated with Pennylane default.qubit:
        std::vector<std::vector<PrecisionT>> exp_values_ref = {
            {0.7572222, 0.8229816, 0.9202947},
            {0.5849835, 0.7701511, 0.9126678},
            {0.6577942, 0.4068672, 0.1670374}};

        for (size_t ind_obs = 0; ind_obs < obs_name.size(); ind_obs++) {
            DYNAMIC_SECTION(obs_name[ind_obs]
                            << " - Varying wires"
                            << StateVectorToName<StateVectorT>::name) {
                for (size_t ind_wires = 0; ind_wires < wires_list.size();
                     ind_wires++) {
                    NamedObs<StateVectorT> obs(obs_name[ind_obs],
                                               wires_list[ind_wires]);
                    PrecisionT expected = exp_values_ref[ind_obs][ind_wires];
                    PrecisionT result = Measurer.var(obs);
                    REQUIRE(expected == Approx(result).margin(1e-6));
                }
            }

            DYNAMIC_SECTION(obs_name[ind_obs]
                            << " Shots - Varying wires"
                            << StateVectorToName<StateVectorT>::name) {
                for (size_t ind_wires = 0; ind_wires < wires_list.size();
                     ind_wires++) {
                    NamedObs<StateVectorT> obs(obs_name[ind_obs],
                                               wires_list[ind_wires]);
                    PrecisionT expected = exp_values_ref[ind_obs][ind_wires];
                    size_t num_shots = 20000;
                    PrecisionT result = Measurer.var(obs, num_shots);

                    REQUIRE_THAT(result,
                                 Catch::Matchers::WithinRel(
                                     expected, static_cast<PrecisionT>(0.20)));
                }
            }
        }
        testNamedObsVar<typename TypeList::Next>();
    }
}

TEST_CASE("Var - NamedObs", "[MeasurementsBase][Observables]") {
    if constexpr (BACKEND_FOUND) {
        testNamedObsVar<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testHermitianObsVar() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;
        using MatrixT = std::vector<ComplexT>;

        // Defining the State Vector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        const PrecisionT theta = M_PI / 2;
        const PrecisionT real_term = std::cos(theta);
        const PrecisionT imag_term = std::sin(theta);

        DYNAMIC_SECTION("Varying wires - 2x2 matrix - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
            // Expected results calculated with Pennylane default.qubit:
            std::vector<PrecisionT> exp_values_ref = {
                0.5849835714501204, 0.7701511529340699, 0.9126678074548389};

            MatrixT Hermitian_matrix{real_term, ComplexT{0, imag_term},
                                     ComplexT{0, -imag_term}, real_term};

            for (size_t ind_wires = 0; ind_wires < wires_list.size();
                 ind_wires++) {
                HermitianObs<StateVectorT> obs(Hermitian_matrix,
                                               wires_list[ind_wires]);
                PrecisionT expected = exp_values_ref[ind_wires];
                PrecisionT result = Measurer.var(obs);
                REQUIRE(expected == Approx(result).margin(1e-6));
            }
        }

        DYNAMIC_SECTION("Varying wires - 4x4 matrix - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<std::vector<size_t>> wires_list = {
                {0, 1}, {0, 2}, {1, 2}};
            // Expected results calculated with Pennylane default.qubit:
            std::vector<PrecisionT> exp_values_ref = {
                0.6549036423585175, 0.8048961865516002, 0.8582611741038356};

            MatrixT Hermitian_matrix(16);
            Hermitian_matrix[0] = real_term;
            Hermitian_matrix[1] = ComplexT{0, imag_term};
            Hermitian_matrix[4] = ComplexT{0, -imag_term};
            Hermitian_matrix[5] = real_term;
            Hermitian_matrix[10] = ComplexT{1.0, 0};
            Hermitian_matrix[15] = ComplexT{1.0, 0};

            for (size_t ind_wires = 0; ind_wires < wires_list.size();
                 ind_wires++) {
                HermitianObs<StateVectorT> obs(Hermitian_matrix,
                                               wires_list[ind_wires]);
                PrecisionT expected = exp_values_ref[ind_wires];
                PrecisionT result = Measurer.var(obs);
                REQUIRE(expected == Approx(result).margin(1e-6));
            }
        }

        testHermitianObsVar<typename TypeList::Next>();
    }
}

TEST_CASE("Var - HermitianObs", "[MeasurementsBase][Observables]") {
    if constexpr (BACKEND_FOUND) {
        testHermitianObsVar<TestStateVectorBackends>();
    }
}

#ifdef PL_USE_LAPACK
template <typename TypeList> void testHermitianObsShotVar() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;
        using MatrixT = std::vector<ComplexT>;

        // Defining the State Vector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        const PrecisionT theta = M_PI / 2;
        const PrecisionT real_term = std::cos(theta);
        const PrecisionT imag_term = std::sin(theta);

        DYNAMIC_SECTION("Varying wires - 2x2 matrix - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
            // Expected results calculated with Pennylane default.qubit:
            std::vector<PrecisionT> exp_values_ref = {
                0.5849835714501204, 0.7701511529340699, 0.9126678074548389};

            MatrixT Hermitian_matrix{real_term, ComplexT{0, imag_term},
                                     ComplexT{0, -imag_term}, real_term};

            for (size_t ind_wires = 0; ind_wires < wires_list.size();
                 ind_wires++) {
                HermitianObs<StateVectorT> obs(Hermitian_matrix,
                                               wires_list[ind_wires]);
                PrecisionT expected = exp_values_ref[ind_wires];
                size_t num_shots = 20000;
                PrecisionT result = Measurer.var(obs, num_shots);

                REQUIRE_THAT(result,
                             Catch::Matchers::WithinRel(
                                 expected, static_cast<PrecisionT>(0.20)));
            }
        }

        DYNAMIC_SECTION("Varying wires - 4x4 matrix - "
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<std::vector<size_t>> wires_list = {
                {0, 1}, {0, 2}, {1, 2}};
            // Expected results calculated with Pennylane default.qubit:
            std::vector<PrecisionT> exp_values_ref = {
                0.6549036423585175, 0.8048961865516002, 0.8582611741038356};

            MatrixT Hermitian_matrix(16);
            Hermitian_matrix[0] = real_term;
            Hermitian_matrix[1] = ComplexT{0, imag_term};
            Hermitian_matrix[4] = ComplexT{0, -imag_term};
            Hermitian_matrix[5] = real_term;
            Hermitian_matrix[10] = ComplexT{1.0, 0};
            Hermitian_matrix[15] = ComplexT{1.0, 0};

            for (size_t ind_wires = 0; ind_wires < wires_list.size();
                 ind_wires++) {
                HermitianObs<StateVectorT> obs(Hermitian_matrix,
                                               wires_list[ind_wires]);

                size_t num_shots = 20000;
                PrecisionT expected = exp_values_ref[ind_wires];
                PrecisionT result = Measurer.var(obs, num_shots);

                REQUIRE_THAT(result,
                             Catch::Matchers::WithinRel(
                                 expected, static_cast<PrecisionT>(0.20)));
            }
        }

        testHermitianObsShotVar<typename TypeList::Next>();
    }
}

TEST_CASE("Var - HermitianObs Shot", "[MeasurementsBase][Observables]") {
    if constexpr (BACKEND_FOUND) {
        testHermitianObsShotVar<TestStateVectorBackends>();
    }
}

#endif

template <typename TypeList> void testTensorProdObsVarShot() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = StateVectorT::ComplexT;

        // Defining the State Vector that will be measured.
        std::vector<ComplexT> statevector_data{
            {0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1}, {0.1, 0.2},
            {0.2, 0.2}, {0.3, 0.3}, {0.3, 0.4}, {0.4, 0.5}};
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        DYNAMIC_SECTION(" Without Identity"
                        << StateVectorToName<StateVectorT>::name) {
            size_t num_shots = 20000;
            auto X0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliX", std::vector<size_t>{0});
            auto Z1 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{1});
            auto obs = TensorProdObs<StateVectorT>::create({X0, Z1});
            auto expected = Measurer.var(*obs);
            auto result = Measurer.var(*obs, num_shots);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(
                                     expected, static_cast<PrecisionT>(0.20)));
        }

        DYNAMIC_SECTION(" full wires"
                        << StateVectorToName<StateVectorT>::name) {
            size_t num_shots = 20000;
            auto X2 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliX", std::vector<size_t>{2});
            auto Y1 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliY", std::vector<size_t>{1});
            auto Z0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{0});
            auto obs = TensorProdObs<StateVectorT>::create({X2, Y1, Z0});
            auto expected = Measurer.var(*obs);
            auto result = Measurer.var(*obs, num_shots);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(
                                     expected, static_cast<PrecisionT>(0.20)));
        }

#ifdef PL_USE_LAPACK
        DYNAMIC_SECTION("With Hermitian and NameObs"
                        << StateVectorToName<StateVectorT>::name) {
            using MatrixT = std::vector<ComplexT>;
            size_t num_shots = 20000;
            const PrecisionT theta = M_PI / 2;
            const PrecisionT real_term = std::cos(theta);
            const PrecisionT imag_term = std::sin(theta);

            MatrixT Hermitian_matrix(16);
            Hermitian_matrix[0] = real_term;
            Hermitian_matrix[1] = ComplexT{0, imag_term};
            Hermitian_matrix[4] = ComplexT{0, -imag_term};
            Hermitian_matrix[5] = real_term;
            Hermitian_matrix[10] = ComplexT{1.0, 0};
            Hermitian_matrix[15] = ComplexT{1.0, 0};

            auto Her = std::make_shared<HermitianObs<StateVectorT>>(
                Hermitian_matrix, std::vector<size_t>{0, 2});

            auto Y1 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliY", std::vector<size_t>{1});

            auto obs = TensorProdObs<StateVectorT>::create({Her, Y1});
            auto expected = Measurer.var(*obs);
            auto result = Measurer.var(*obs, num_shots);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(
                                     expected, static_cast<PrecisionT>(0.20)));
        }

        DYNAMIC_SECTION("With Hermitian and NameObs"
                        << StateVectorToName<StateVectorT>::name) {
            using MatrixT = std::vector<ComplexT>;
            size_t num_shots = 20000;

            MatrixT Hermitian_matrix(4, {0, 0});
            Hermitian_matrix[0] = 1;
            Hermitian_matrix[3] = -1;

            auto Her = std::make_shared<HermitianObs<StateVectorT>>(
                Hermitian_matrix, std::vector<size_t>{1});

            auto obs = TensorProdObs<StateVectorT>::create({Her});
            auto expected = Measurer.var(*obs);
            auto result = Measurer.var(*obs, num_shots);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(
                                     expected, static_cast<PrecisionT>(0.20)));
        }
#endif

        DYNAMIC_SECTION(" full wires with apply operations"
                        << StateVectorToName<StateVectorT>::name) {
            size_t num_shots = 20000;
            auto X2 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliX", std::vector<size_t>{2});
            auto Y1 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliY", std::vector<size_t>{1});
            auto Z0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{0});
            auto obs = TensorProdObs<StateVectorT>::create({X2, Y1, Z0});

            statevector.applyOperations({"Hadamard", "PauliZ", "S", "Hadamard"},
                                        {{0}, {1}, {2}, {2}},
                                        {false, false, false, false});

            Measurements<StateVectorT> Measurer0(statevector);

            auto expected = Measurer0.var(*obs);
            auto result = Measurer0.var(*obs, num_shots);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(
                                     expected, static_cast<PrecisionT>(0.20)));
        }

        DYNAMIC_SECTION(" With Identity"
                        << StateVectorToName<StateVectorT>::name) {
            size_t num_shots = 20000;
            auto X0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliX", std::vector<size_t>{0});
            auto I1 = std::make_shared<NamedObs<StateVectorT>>(
                "Identity", std::vector<size_t>{1});
            auto obs = TensorProdObs<StateVectorT>::create({X0, I1});
            PrecisionT expected = Measurer.var(*obs);
            PrecisionT result = Measurer.var(*obs, num_shots);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(
                                     expected, static_cast<PrecisionT>(0.20)));
        }

        testTensorProdObsVarShot<typename TypeList::Next>();
    }
}

TEST_CASE("Var Shot- TensorProdObs", "[MeasurementsBase][Observables]") {
    if constexpr (BACKEND_FOUND) {
        testTensorProdObsVarShot<TestStateVectorBackends>();
    }
}
template <typename TypeList> void testSamples() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;

        constexpr size_t twos[] = {
            1U << 0U,  1U << 1U,  1U << 2U,  1U << 3U,  1U << 4U,  1U << 5U,
            1U << 6U,  1U << 7U,  1U << 8U,  1U << 9U,  1U << 10U, 1U << 11U,
            1U << 12U, 1U << 13U, 1U << 14U, 1U << 15U, 1U << 16U, 1U << 17U,
            1U << 18U, 1U << 19U, 1U << 20U, 1U << 21U, 1U << 22U, 1U << 23U,
            1U << 24U, 1U << 25U, 1U << 26U, 1U << 27U, 1U << 28U, 1U << 29U,
            1U << 30U, 1U << 31U};

        // Defining the State Vector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measurements class.
        // This object attaches to the statevector allowing several
        // measurements.
        Measurements<StateVectorT> Measurer(statevector);

        std::vector<PrecisionT> expected_probabilities = {
            0.67078706, 0.03062806, 0.0870997,  0.00397696,
            0.17564072, 0.00801973, 0.02280642, 0.00104134};

        size_t num_qubits = 3;
        size_t N = std::pow(2, num_qubits);
        size_t num_samples = 100000;
        auto &&samples = Measurer.generate_samples(num_samples);

        std::vector<size_t> counts(N, 0);
        std::vector<size_t> samples_decimal(num_samples, 0);

        // convert samples to decimal and then bin them in counts
        for (size_t i = 0; i < num_samples; i++) {
            for (size_t j = 0; j < num_qubits; j++) {
                if (samples[i * num_qubits + j] != 0) {
                    samples_decimal[i] += twos[(num_qubits - 1 - j)];
                }
            }
            counts[samples_decimal[i]] += 1;
        }

        // compute estimated probabilities from histogram
        std::vector<PrecisionT> probabilities(counts.size());
        for (size_t i = 0; i < counts.size(); i++) {
            probabilities[i] = counts[i] / (PrecisionT)num_samples;
        }

        DYNAMIC_SECTION("No wires provided - "
                        << StateVectorToName<StateVectorT>::name) {
            REQUIRE_THAT(probabilities,
                         Catch::Approx(expected_probabilities).margin(.05));
        }
        testSamples<typename TypeList::Next>();
    }
}

TEST_CASE("Samples", "[MeasurementsBase]") {
    if constexpr (BACKEND_FOUND) {
        testSamples<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testSamplesCountsObs() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;

        // Defining the State Vector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        constexpr size_t twos[] = {
            1U << 0U,  1U << 1U,  1U << 2U,  1U << 3U,  1U << 4U,  1U << 5U,
            1U << 6U,  1U << 7U,  1U << 8U,  1U << 9U,  1U << 10U, 1U << 11U,
            1U << 12U, 1U << 13U, 1U << 14U, 1U << 15U, 1U << 16U, 1U << 17U,
            1U << 18U, 1U << 19U, 1U << 20U, 1U << 21U, 1U << 22U, 1U << 23U,
            1U << 24U, 1U << 25U, 1U << 26U, 1U << 27U, 1U << 28U, 1U << 29U,
            1U << 30U, 1U << 31U};

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::string> obs_name = {"PauliX", "PauliY", "PauliZ",
                                             "Hadamard", "Identity"};
        // Expected results calculated with Pennylane default.qubit:
        std::vector<std::vector<PrecisionT>> exp_values_ref = {
            {0.49272486, 0.42073549, 0.28232124},
            {-0.64421768, -0.47942553, -0.29552020},
            {0.58498357, 0.77015115, 0.91266780},
            {0.7620549436, 0.8420840225, 0.8449848566},
            {1.0, 1.0, 1.0}};
        for (size_t ind_obs = 0; ind_obs < obs_name.size(); ind_obs++) {
            DYNAMIC_SECTION(obs_name[ind_obs]
                            << " Sample Obs - Varying wires"
                            << StateVectorToName<StateVectorT>::name) {
                size_t num_shots = 20000;
                for (size_t ind_wires = 0; ind_wires < wires_list.size();
                     ind_wires++) {
                    NamedObs<StateVectorT> obs(obs_name[ind_obs],
                                               wires_list[ind_wires]);
                    PrecisionT expected = exp_values_ref[ind_obs][ind_wires];
                    auto samples = Measurer.sample(obs, num_shots);

                    PrecisionT result = 0.0;
                    for (auto &it : samples) {
                        result += it;
                    }
                    result /= num_shots;

                    REQUIRE_THAT(result,
                                 Catch::Matchers::WithinRel(
                                     expected, static_cast<PrecisionT>(0.20)));
                }
            }

            DYNAMIC_SECTION(obs_name[ind_obs]
                            << " Counts Obs - Varying wires"
                            << StateVectorToName<StateVectorT>::name) {
                size_t num_shots = 20000;
                for (size_t ind_wires = 0; ind_wires < wires_list.size();
                     ind_wires++) {
                    NamedObs<StateVectorT> obs(obs_name[ind_obs],
                                               wires_list[ind_wires]);
                    PrecisionT expected = exp_values_ref[ind_obs][ind_wires];
                    auto samples = Measurer.counts(obs, num_shots);

                    PrecisionT result = 0.0;
                    for (auto &it : samples) {
                        result += it.first * it.second;
                    }
                    result /= num_shots;

                    REQUIRE_THAT(result,
                                 Catch::Matchers::WithinRel(
                                     expected, static_cast<PrecisionT>(0.20)));
                }
            }
        }

        DYNAMIC_SECTION("samples() without obs"
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<PrecisionT> expected_probabilities = {
                0.67078706, 0.03062806, 0.0870997,  0.00397696,
                0.17564072, 0.00801973, 0.02280642, 0.00104134};

            size_t num_qubits = 3;
            size_t N = std::pow(2, num_qubits);
            size_t num_samples = 100000;
            auto &&samples = Measurer.sample(num_samples);

            std::vector<size_t> counts(N, 0);
            std::vector<size_t> samples_decimal(num_samples, 0);

            // convert samples to decimal and then bin them in counts
            for (size_t i = 0; i < num_samples; i++) {
                for (size_t j = 0; j < num_qubits; j++) {
                    if (samples[i * num_qubits + j] != 0) {
                        samples_decimal[i] += twos[(num_qubits - 1 - j)];
                    }
                }
                counts[samples_decimal[i]] += 1;
            }

            // compute estimated probabilities from histogram
            std::vector<PrecisionT> probabilities(counts.size());
            for (size_t i = 0; i < counts.size(); i++) {
                probabilities[i] = counts[i] / (PrecisionT)num_samples;
            }

            REQUIRE_THAT(probabilities,
                         Catch::Approx(expected_probabilities).margin(.05));
        }

        DYNAMIC_SECTION("counts() without obs"
                        << StateVectorToName<StateVectorT>::name) {
            std::vector<std::string> expected_keys = {
                "000", "001", "010", "011", "100", "101", "110", "111"};

            std::vector<PrecisionT> expected_probabilities = {
                0.67078706, 0.03062806, 0.0870997,  0.00397696,
                0.17564072, 0.00801973, 0.02280642, 0.00104134};

            size_t num_qubits = 3;
            size_t N = std::pow(2, num_qubits);
            size_t num_samples = 100000;

            auto &&counts_sample = Measurer.counts(num_samples);

            std::vector<size_t> counts(N, 0);

            // convert samples to decimal and then bin them in counts
            for (auto &it : counts_sample) {
                auto key = it.first;

                counts[key] = it.second;
            }

            // compute estimated probabilities from histogram
            std::vector<PrecisionT> probabilities(counts.size());
            for (size_t i = 0; i < counts.size(); i++) {
                probabilities[i] = counts[i] / (PrecisionT)num_samples;
            }

            REQUIRE_THAT(probabilities,
                         Catch::Approx(expected_probabilities).margin(.05));
        }

        testSamplesCountsObs<typename TypeList::Next>();
    }
}

TEST_CASE("Samples Obs", "[MeasurementsBase]") {
    if constexpr (BACKEND_FOUND) {
        testSamplesCountsObs<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testHamiltonianObsExpvalShot() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;

        // Defining the State Vector that will be measured.
        std::vector<ComplexT> statevector_data{
            {0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1}, {0.1, 0.2},
            {0.2, 0.2}, {0.3, 0.3}, {0.3, 0.4}, {0.4, 0.5}};

        std::vector<ComplexT> sv_data{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                      {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                      {0.3, 0.4}, {0.4, 0.5}};

        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        StateVectorT sv(sv_data.data(), sv_data.size());

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        auto X0 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliX", std::vector<size_t>{0});
        auto Z1 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});

        auto ob = Hamiltonian<StateVectorT>::create({0.3, 0.5}, {X0, Z1});

        DYNAMIC_SECTION("Without shots_range "
                        << StateVectorToName<StateVectorT>::name) {
            size_t num_shots = 20000;
            std::vector<size_t> shots_range = {};

            auto res = Measurer.expval(*ob, num_shots, shots_range);
            auto expected = PrecisionT(-0.086);

            REQUIRE_THAT(res, Catch::Matchers::WithinRel(
                                  expected, static_cast<PrecisionT>(0.20)));
        }

        DYNAMIC_SECTION("With shots_range "
                        << StateVectorToName<StateVectorT>::name) {
            size_t num_shots = 20000;
            std::vector<size_t> shots_range;
            for (size_t i = 0; i < num_shots; i += 2) {
                shots_range.push_back(i);
            }

            auto res = Measurer.expval(*ob, num_shots, shots_range);
            auto expected = PrecisionT(-0.086);

            REQUIRE_THAT(res, Catch::Matchers::WithinRel(
                                  expected, static_cast<PrecisionT>(0.20)));
        }

        DYNAMIC_SECTION("TensorProd with shots_range "
                        << StateVectorToName<StateVectorT>::name) {
            auto X0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliX", std::vector<size_t>{0});
            auto Z1 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{1});
            auto obs0 = TensorProdObs<StateVectorT>::create({X0, Z1});

            auto Y0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliY", std::vector<size_t>{0});
            auto H1 = std::make_shared<NamedObs<StateVectorT>>(
                "Hadamard", std::vector<size_t>{1});
            auto obs1 = TensorProdObs<StateVectorT>::create({Y0, H1});

            auto obs =
                Hamiltonian<StateVectorT>::create({0.1, 0.3}, {obs0, obs1});

            Measurements<StateVectorT> Measurer_analytic(sv);
            auto expected = Measurer_analytic.expval(*obs);

            size_t num_shots = 20000;
            auto res = Measurer.expval(*obs, num_shots, {});

            REQUIRE_THAT(res, Catch::Matchers::WithinRel(
                                  expected, static_cast<PrecisionT>(0.20)));
        }

#ifdef PL_USE_LAPACK
        DYNAMIC_SECTION("YHer" << StateVectorToName<StateVectorT>::name) {
            auto Y0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliY", std::vector<size_t>{0});

            std::vector<ComplexT> Hermitian_mat{ComplexT{3, 0}, ComplexT{2, 1},
                                                ComplexT{2, -1},
                                                ComplexT{-3, 0}};
            auto Her1 = std::make_shared<HermitianObs<StateVectorT>>(
                Hermitian_mat, std::vector<size_t>{1});

            auto ob = Hamiltonian<StateVectorT>::create({0.5, 0.5}, {Y0, Her1});

            size_t num_shots = 100000;

            auto res = Measurer.expval(*ob, num_shots, {});
            auto expected = Measurer.expval(*ob);

            REQUIRE_THAT(res, Catch::Matchers::WithinRel(
                                  expected, static_cast<PrecisionT>(0.20)));
        }
#endif

        testHamiltonianObsExpvalShot<typename TypeList::Next>();
    }
}

TEST_CASE("Expval Shot - HamiltonianObs ", "[MeasurementsBase][Observables]") {
    if constexpr (BACKEND_FOUND) {
        testHamiltonianObsExpvalShot<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testHamiltonianObsVarShot() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;

        // Defining the State Vector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        DYNAMIC_SECTION("YZ" << StateVectorToName<StateVectorT>::name) {
            auto Y0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliY", std::vector<size_t>{0});
            auto Z1 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{1});

            auto ob = Hamiltonian<StateVectorT>::create({0.5, 0.5}, {Y0, Z1});

            size_t num_shots = 20000;

            auto res = Measurer.var(*ob, num_shots);
            auto expected = Measurer.var(*ob);

            REQUIRE_THAT(res, Catch::Matchers::WithinRel(
                                  expected, static_cast<PrecisionT>(0.20)));
        }

        DYNAMIC_SECTION("YI" << StateVectorToName<StateVectorT>::name) {
            auto Y0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliY", std::vector<size_t>{0});
            auto I1 = std::make_shared<NamedObs<StateVectorT>>(
                "Identity", std::vector<size_t>{1});

            auto ob = Hamiltonian<StateVectorT>::create({0.5, 0.5}, {Y0, I1});

            size_t num_shots = 20000;

            auto res = Measurer.var(*ob, num_shots);
            auto expected = Measurer.var(*ob);

            REQUIRE_THAT(res, Catch::Matchers::WithinRel(
                                  expected, static_cast<PrecisionT>(0.20)));
        }

#ifdef PL_USE_LAPACK
        DYNAMIC_SECTION("YHer" << StateVectorToName<StateVectorT>::name) {
            using ComplexT = typename StateVectorT::ComplexT;
            auto Y0 = std::make_shared<NamedObs<StateVectorT>>(
                "PauliY", std::vector<size_t>{0});

            std::vector<ComplexT> Hermitian_mat1{ComplexT{3, 0}, ComplexT{2, 1},
                                                 ComplexT{2, -1},
                                                 ComplexT{-3, 0}};
            auto Her1 = std::make_shared<HermitianObs<StateVectorT>>(
                Hermitian_mat1, std::vector<size_t>{1});

            std::vector<ComplexT> Hermitian_mat2{ComplexT{2, 0}, ComplexT{1, 1},
                                                 ComplexT{1, -1},
                                                 ComplexT{-6, 0}};
            auto Her2 = std::make_shared<HermitianObs<StateVectorT>>(
                Hermitian_mat2, std::vector<size_t>{2});

            auto ob = Hamiltonian<StateVectorT>::create({0.5, 0.5, 1.0},
                                                        {Y0, Her1, Her2});

            size_t num_shots = 20000;

            auto res = Measurer.var(*ob, num_shots);
            auto expected = Measurer.var(*ob);

            REQUIRE_THAT(res, Catch::Matchers::WithinRel(
                                  expected, static_cast<PrecisionT>(0.20)));
        }
#endif

        testHamiltonianObsVarShot<typename TypeList::Next>();
    }
}

TEST_CASE("Var Shot - HamiltonianObs ", "[MeasurementsBase][Observables]") {
    if constexpr (BACKEND_FOUND) {
        testHamiltonianObsVarShot<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testSparseHObsMeasureShot() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using ComplexT = typename StateVectorT::ComplexT;

        // Defining the State Vector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT statevector(statevector_data.data(),
                                 statevector_data.size());

        // Initializing the measures class.
        // This object attaches to the statevector allowing several measures.
        Measurements<StateVectorT> Measurer(statevector);

        auto sparseH = SparseHamiltonian<StateVectorT>::create(
            {ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0},
             ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0},
             ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}},
            {7, 6, 5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5, 6, 7, 8}, {0, 1, 2});

        DYNAMIC_SECTION("Failed for expval "
                        << StateVectorToName<StateVectorT>::name) {
            size_t num_shots = 1000;
            std::vector<size_t> shots_range = {};
            REQUIRE_THROWS_WITH(
                Measurer.expval(*sparseH, num_shots, shots_range),
                Catch::Matchers::Contains("SparseHamiltonian observables do "
                                          "not support shot measurement."));
        }

        DYNAMIC_SECTION("Failed for var "
                        << StateVectorToName<StateVectorT>::name) {
            size_t num_shots = 1000;
            std::vector<size_t> shots_range = {};
            REQUIRE_THROWS_WITH(
                Measurer.var(*sparseH, num_shots),
                Catch::Matchers::Contains("SparseHamiltonian observables do "
                                          "not support shot measurement."));
        }

        testSparseHObsMeasureShot<typename TypeList::Next>();
    }
}

TEST_CASE("Measure Shot - SparseHObs ", "[MeasurementsBase][Observables]") {
    if constexpr (BACKEND_FOUND) {
        testSparseHObsMeasureShot<TestStateVectorBackends>();
    }
}