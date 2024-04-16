// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
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
#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos::Measures;
using Pennylane::Util::createNonTrivialState;
using Pennylane::Util::INVSQRT2;
}; // namespace
/// @endcond

TEMPLATE_PRODUCT_TEST_CASE("Expected Values", "[Measurements]",
                           (StateVectorKokkos), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    // Defining the statevector that will be measured.
    auto statevector_data = createNonTrivialState<StateVectorT>();
    StateVectorT statevector(statevector_data.data(), statevector_data.size());

    // Initializing the Measurements class.
    // This object attaches to the statevector allowing several measures.
    Measurements<StateVectorT> Measurer(statevector);

    SECTION("Testing single operation defined by a matrix:") {
        std::vector<ComplexT> PauliX = {0, 1, 1, 0};
        std::vector<size_t> wires_single = {0};
        PrecisionT exp_value = Measurer.expval(PauliX, wires_single);
        PrecisionT exp_values_ref = 0.492725;
        REQUIRE(exp_value == Approx(exp_values_ref).margin(1e-6));
    }

    SECTION("Testing single operation defined by its name:") {
        std::vector<size_t> wires_single = {0};
        PrecisionT exp_value = Measurer.expval("PauliX", wires_single);
        PrecisionT exp_values_ref = 0.492725;
        REQUIRE(exp_value == Approx(exp_values_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by a matrix:") {
        PrecisionT isqrt2 = INVSQRT2<PrecisionT>();
        std::vector<ComplexT> Identity = {1, 0, 0, 1};
        std::vector<ComplexT> Identity2 = {1, 0, 0, 0, 0, 1, 0, 0,
                                           0, 0, 1, 0, 0, 0, 0, 1};
        std::vector<ComplexT> PauliX = {0, 1, 1, 0};
        std::vector<ComplexT> PauliY = {0, ComplexT{0, -1}, ComplexT{0, 1}, 0};
        std::vector<ComplexT> PauliZ = {1, 0, 0, -1};
        std::vector<ComplexT> Hadamard = {isqrt2, isqrt2, isqrt2, -isqrt2};

        std::vector<PrecisionT> exp_values;
        std::vector<PrecisionT> exp_values_ref;
        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::vector<ComplexT>> operations_list;

        operations_list = {Identity, Identity, Identity};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {1.0, 1.0, 1.0};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {Identity2, Identity2};
        exp_values = Measurer.expval(operations_list, {{0, 1}, {1, 2}});
        exp_values_ref = {1.0, 1.0};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {PauliX, PauliX, PauliX};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.49272486, 0.42073549, 0.28232124};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {PauliY, PauliY, PauliY};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {-0.64421768, -0.47942553, -0.29552020};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {PauliZ, PauliZ, PauliZ};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.58498357, 0.77015115, 0.91266780};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {Hadamard, Hadamard, Hadamard};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.7620549436, 0.8420840225, 0.8449848566};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by its name:") {
        std::vector<PrecisionT> exp_values;
        std::vector<PrecisionT> exp_values_ref;
        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::string> operations_list;

        operations_list = {"Identity", "Identity", "Identity"};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {1.0, 1.0, 1.0};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {"PauliX", "PauliX", "PauliX"};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.49272486, 0.42073549, 0.28232124};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {"PauliY", "PauliY", "PauliY"};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {-0.64421768, -0.47942553, -0.29552020};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {"PauliZ", "PauliZ", "PauliZ"};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.58498357, 0.77015115, 0.91266780};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {"Hadamard", "Hadamard", "Hadamard"};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.7620549436, 0.8420840225, 0.8449848566};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Variances", "[Measurements]", (StateVectorKokkos),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    // Defining the State Vector that will be measured.
    auto statevector_data = createNonTrivialState<StateVectorT>();
    StateVectorT statevector(statevector_data.data(), statevector_data.size());

    // Initializing the measurements class.
    // This object attaches to the statevector allowing several measurements.
    Measurements<StateVectorT> Measurer(statevector);

    SECTION("Testing single operation defined by a matrix:") {
        std::vector<ComplexT> PauliX = {0, 1, 1, 0};
        std::vector<size_t> wires_single = {0};
        PrecisionT variance = Measurer.var(PauliX, wires_single);
        PrecisionT variances_ref = 0.7572222;
        REQUIRE(variance == Approx(variances_ref).margin(1e-6));
    }

    SECTION("Testing single operation defined by its name:") {
        std::vector<size_t> wires_single = {0};
        PrecisionT variance = Measurer.var("PauliX", wires_single);
        PrecisionT variances_ref = 0.7572222;
        REQUIRE(variance == Approx(variances_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by a matrix:") {
        std::vector<ComplexT> PauliX = {0, 1, 1, 0};
        std::vector<ComplexT> PauliY = {0, ComplexT{0, -1}, ComplexT{0, 1}, 0};
        std::vector<ComplexT> PauliZ = {1, 0, 0, -1};

        std::vector<PrecisionT> variances;
        std::vector<PrecisionT> variances_ref;
        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::vector<ComplexT>> operations_list;

        operations_list = {PauliX, PauliX, PauliX};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.7572222, 0.8229816, 0.9202947};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {PauliY, PauliY, PauliY};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.5849835, 0.7701511, 0.9126678};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {PauliZ, PauliZ, PauliZ};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.6577942, 0.4068672, 0.1670374};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by its name:") {
        std::vector<PrecisionT> variances;
        std::vector<PrecisionT> variances_ref;
        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::string> operations_list;

        operations_list = {"PauliX", "PauliX", "PauliX"};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.7572222, 0.8229816, 0.9202947};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {"PauliY", "PauliY", "PauliY"};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.5849835, 0.7701511, 0.9126678};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {"PauliZ", "PauliZ", "PauliZ"};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.6577942, 0.4068672, 0.1670374};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));
    }
}

TEMPLATE_TEST_CASE("Probabilities", "[Measures]", float, double) {
    // Probabilities calculated with Pennylane default.qubit:
    std::vector<std::pair<std::vector<size_t>, std::vector<TestType>>> input = {
        {{0, 1, 2},
         {0.67078706, 0.03062806, 0.0870997, 0.00397696, 0.17564072, 0.00801973,
          0.02280642, 0.00104134}},
        {{0, 2, 1},
         {0.67078706, 0.0870997, 0.03062806, 0.00397696, 0.17564072, 0.02280642,
          0.00801973, 0.00104134}},
        {{1, 0, 2},
         {0.67078706, 0.03062806, 0.17564072, 0.00801973, 0.0870997, 0.00397696,
          0.02280642, 0.00104134}},
        {{1, 2, 0},
         {0.67078706, 0.0870997, 0.17564072, 0.02280642, 0.03062806, 0.00397696,
          0.00801973, 0.00104134}},
        {{2, 0, 1},
         {0.67078706, 0.17564072, 0.03062806, 0.00801973, 0.0870997, 0.02280642,
          0.00397696, 0.00104134}},
        {{2, 1, 0},
         {0.67078706, 0.17564072, 0.0870997, 0.02280642, 0.03062806, 0.00801973,
          0.00397696, 0.00104134}},
        {{0, 1}, {0.70141512, 0.09107666, 0.18366045, 0.02384776}},
        {{0, 2}, {0.75788676, 0.03460502, 0.19844714, 0.00906107}},
        {{1, 2}, {0.84642778, 0.0386478, 0.10990612, 0.0050183}},
        {{2, 1}, {0.84642778, 0.10990612, 0.0386478, 0.0050183}},
        {{0}, {0.79249179, 0.20750821}},
        {{1}, {0.88507558, 0.11492442}},
        {{2}, {0.9563339, 0.0436661}}};

    // Defining the State Vector that will be measured.
    const std::size_t num_qubits = 3;
    auto statevector_data =
        createNonTrivialState<StateVectorKokkos<TestType>>(num_qubits);
    StateVectorKokkos<TestType> measure_sv(statevector_data.data(),
                                           statevector_data.size());

    SECTION("Looping over different wire configurations:") {
        auto m = Measurements(measure_sv);
        for (const auto &term : input) {
            auto probabilities = m.probs(term.first);
            REQUIRE_THAT(term.second,
                         Catch::Approx(probabilities).margin(1e-6));
        }
    }
}

TEST_CASE("Test tensor transposition", "[Measure]") {
    // Init Kokkos creating a StateVectorKokkos
    auto statevector_data = createNonTrivialState<StateVectorKokkos<double>>();
    StateVectorKokkos<double> statevector(statevector_data.data(),
                                          statevector_data.size());

    // Transposition axes and expected result.
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> input = {
        {{0, 1, 2}, {0, 1, 2, 3, 4, 5, 6, 7}},
        {{0, 2, 1}, {0, 2, 1, 3, 4, 6, 5, 7}},
        {{1, 0, 2}, {0, 1, 4, 5, 2, 3, 6, 7}},
        {{1, 2, 0}, {0, 4, 1, 5, 2, 6, 3, 7}},
        {{2, 0, 1}, {0, 2, 4, 6, 1, 3, 5, 7}},
        {{2, 1, 0}, {0, 4, 2, 6, 1, 5, 3, 7}},
        {{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
        {{0, 1, 3, 2}, {0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15}},
        {{0, 2, 1, 3}, {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15}},
        {{0, 2, 3, 1}, {0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15}},
        {{0, 3, 1, 2}, {0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15}},
        {{0, 3, 2, 1}, {0, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15}},
        {{1, 0, 2, 3}, {0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15}},
        {{1, 0, 3, 2}, {0, 2, 1, 3, 8, 10, 9, 11, 4, 6, 5, 7, 12, 14, 13, 15}},
        {{1, 2, 0, 3}, {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15}},
        {{1, 2, 3, 0}, {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15}},
        {{1, 3, 0, 2}, {0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15}},
        {{1, 3, 2, 0}, {0, 8, 2, 10, 1, 9, 3, 11, 4, 12, 6, 14, 5, 13, 7, 15}},
        {{2, 0, 1, 3}, {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15}},
        {{2, 0, 3, 1}, {0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15}},
        {{2, 1, 0, 3}, {0, 1, 8, 9, 4, 5, 12, 13, 2, 3, 10, 11, 6, 7, 14, 15}},
        {{2, 1, 3, 0}, {0, 8, 1, 9, 4, 12, 5, 13, 2, 10, 3, 11, 6, 14, 7, 15}},
        {{2, 3, 0, 1}, {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}},
        {{2, 3, 1, 0}, {0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15}},
        {{3, 0, 1, 2}, {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15}},
        {{3, 0, 2, 1}, {0, 4, 2, 6, 8, 12, 10, 14, 1, 5, 3, 7, 9, 13, 11, 15}},
        {{3, 1, 0, 2}, {0, 2, 8, 10, 4, 6, 12, 14, 1, 3, 9, 11, 5, 7, 13, 15}},
        {{3, 1, 2, 0}, {0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15}},
        {{3, 2, 0, 1}, {0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15}},
        {{3, 2, 1, 0}, {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15}}};

    using KokkosExecSpace = Kokkos::DefaultExecutionSpace;
    using UnmanagedSizeTHostView =
        Kokkos::View<size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    SECTION("Looping over different wire configurations:") {
        for (auto &term : input) {
            // Defining a tensor to be transposed.
            std::vector<size_t> indices(static_cast<std::size_t>(1U)
                                        << term.first.size());
            std::iota(indices.begin(), indices.end(), 0);

            std::vector<size_t> results(indices.size());

            Kokkos::View<size_t *> d_indices("d_indices", indices.size());
            Kokkos::View<size_t *> d_results("d_results", indices.size());
            Kokkos::View<size_t *> d_wires("d_wires", term.first.size());
            Kokkos::View<size_t *> d_trans_index("d_trans_index",
                                                 indices.size());

            Kokkos::deep_copy(d_indices, UnmanagedSizeTHostView(
                                             indices.data(), indices.size()));
            Kokkos::deep_copy(
                d_wires,
                UnmanagedSizeTHostView(term.first.data(), term.first.size()));

            using MDPolicyType_2D =
                Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>;

            MDPolicyType_2D mdpolicy_2d1(
                {{0, 0}}, {{static_cast<int>(indices.size()),
                            static_cast<int>(term.first.size())}});

            const int num_wires = term.first.size();

            Kokkos::parallel_for(
                "TransIndex", mdpolicy_2d1,
                getTransposedIndexFunctor(d_wires, d_trans_index, num_wires));

            Kokkos::parallel_for(
                "Transpose",
                Kokkos::RangePolicy<KokkosExecSpace>(0, indices.size()),
                getTransposedFunctor<size_t>(d_results, d_indices,
                                             d_trans_index));

            Kokkos::deep_copy(
                UnmanagedSizeTHostView(results.data(), results.size()),
                d_results);
            REQUIRE(term.second == results);
        }
    }
}
