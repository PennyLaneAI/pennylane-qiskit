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

#include "Gates.hpp"
#include "LinearAlgebraKokkos.hpp" // getRealOfComplexInnerProduct
#include "MeasurementsKokkos.hpp"
#include "ObservablesKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp"

/**
 * @file
 *  Tests for parametric gates functionality defined in the class
 * StateVectorKokkos.
 */

/// @cond DEV
namespace {
using namespace Pennylane::Gates;
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::Observables;
using Pennylane::LightningKokkos::Util::getRealOfComplexInnerProduct;
using Pennylane::Util::createNonTrivialState;
using Pennylane::Util::exp2;
using std::size_t;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyIsingXY",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 4;

        std::vector<ComplexT> ini_st{
            ComplexT{0.267462841882, 0.010768564798},
            ComplexT{0.228575129706, 0.010564590956},
            ComplexT{0.099492749900, 0.260849823392},
            ComplexT{0.093690204310, 0.189847108173},
            ComplexT{0.033390732374, 0.203836830144},
            ComplexT{0.226979395737, 0.081852150975},
            ComplexT{0.031235505729, 0.176933497281},
            ComplexT{0.294287602843, 0.145156781198},
            ComplexT{0.152742706049, 0.111628061129},
            ComplexT{0.012553863703, 0.120027860480},
            ComplexT{0.237156555364, 0.154658769755},
            ComplexT{0.117001120872, 0.228059505033},
            ComplexT{0.041495873225, 0.065934827444},
            ComplexT{0.089653239407, 0.221581340372},
            ComplexT{0.217892322429, 0.291261296999},
            ComplexT{0.292993251871, 0.186570798697},
        };

        std::vector<ComplexT> expected{
            ComplexT{0.267462849617, 0.010768564418},
            ComplexT{0.228575125337, 0.010564590804},
            ComplexT{0.099492751062, 0.260849833488},
            ComplexT{0.093690201640, 0.189847111702},
            ComplexT{0.015641822883, 0.225092900621},
            ComplexT{0.205574608177, 0.082808663337},
            ComplexT{0.006827173322, 0.211631480575},
            ComplexT{0.255280800811, 0.161572331669},
            ComplexT{0.119218164572, 0.115460377284},
            ComplexT{-0.000315789761, 0.153835664378},
            ComplexT{0.206786872079, 0.157633689097},
            ComplexT{0.093027614553, 0.271012980118},
            ComplexT{0.041495874524, 0.065934829414},
            ComplexT{0.089653238654, 0.221581339836},
            ComplexT{0.217892318964, 0.291261285543},
            ComplexT{0.292993247509, 0.186570793390},
        };

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});

            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applyOperation("IsingXY", {0, 1}, false, {0.312});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyRX",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        size_t num_qubits = 1;
        std::vector<TestType> angles = {0.1, 0.6};
        std::vector<std::vector<ComplexT>> expected_results{
            std::vector<ComplexT>{ComplexT{0.9987502603949663, 0.0},
                                  ComplexT{0.0, -0.04997916927067834}},
            std::vector<ComplexT>{ComplexT{0.9553364891256061, 0.0},
                                  ComplexT{0, -0.2955202066613395}},
            std::vector<ComplexT>{ComplexT{0.49757104789172696, 0.0},
                                  ComplexT{0, -0.867423225594017}}};

        std::vector<std::vector<ComplexT>> expected_results_adj{
            std::vector<ComplexT>{ComplexT{0.9987502603949663, 0.0},
                                  ComplexT{0.0, 0.04997916927067834}},
            std::vector<ComplexT>{ComplexT{0.9553364891256061, 0.0},
                                  ComplexT{0, 0.2955202066613395}},
            std::vector<ComplexT>{ComplexT{0.49757104789172696, 0.0},
                                  ComplexT{0, 0.867423225594017}}};

        SECTION("Apply adj directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});

                kokkos_sv.applyOperation("RX", {0}, true, {angles[index]});
                kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results_adj[index][j]) ==
                          Approx(imag(result_sv[j])));
                    CHECK(real(expected_results_adj[index][j]) ==
                          Approx(real(result_sv[j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyRY",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        size_t num_qubits = 1;
        const std::vector<TestType> angles{0.2, 0.7, 2.9};
        std::vector<std::vector<ComplexT>> expected_results{
            std::vector<ComplexT>{{0.8731983044562817, 0.04786268954660339},
                                  {0.0876120655431924, -0.47703040785184303}},
            std::vector<ComplexT>{{0.8243771119105122, 0.16439396602553008},
                                  {0.3009211363333468, -0.45035926880694604}},
            std::vector<ComplexT>{{0.10575112905629831, 0.47593196040758534},
                                  {0.8711876098966215, -0.0577721051072477}}};
        std::vector<std::vector<ComplexT>> expected_results_adj{
            std::vector<ComplexT>{{0.8731983044562817, -0.04786268954660339},
                                  {-0.0876120655431924, -0.47703040785184303}},
            std::vector<ComplexT>{{0.8243771119105122, -0.16439396602553008},
                                  {-0.3009211363333468, -0.45035926880694604}},
            std::vector<ComplexT>{{0.10575112905629831, -0.47593196040758534},
                                  {-0.8711876098966215, -0.0577721051072477}}};

        std::vector<ComplexT> ini_st{{0.8775825618903728, 0.0},
                                     {0.0, -0.47942553860420306}};

        SECTION("Apply adj directly") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});

                kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
                kokkos_sv.applyOperation("RY", {0}, true, {angles[index]});
                kokkos_sv.DeviceToHost(result_sv.data(), ini_st.size());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results_adj[index][j]) ==
                          Approx(imag(result_sv[j])));
                    CHECK(real(expected_results_adj[index][j]) ==
                          Approx(real(result_sv[j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyRZ",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        size_t num_qubits = 3;
        const std::vector<TestType> angles{0.2, 0.7, 2.9};
        const ComplexT coef(1.0 / (2 * std::sqrt(2)), 0);

        std::vector<std::vector<ComplexT>> rz_data;

        rz_data.reserve(angles.size());
        for (auto &a : angles) {
            rz_data.push_back(getRZ<Kokkos::complex, TestType>(a));
        }

        std::vector<std::vector<ComplexT>> expected_results = {
            {rz_data[0][0], rz_data[0][0], rz_data[0][0], rz_data[0][0],
             rz_data[0][3], rz_data[0][3], rz_data[0][3], rz_data[0][3]},
            {
                rz_data[1][0],
                rz_data[1][0],
                rz_data[1][3],
                rz_data[1][3],
                rz_data[1][0],
                rz_data[1][0],
                rz_data[1][3],
                rz_data[1][3],
            },
            {rz_data[2][0], rz_data[2][3], rz_data[2][0], rz_data[2][3],
             rz_data[2][0], rz_data[2][3], rz_data[2][0], rz_data[2][3]}};

        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperations(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{false}, {false}, {false}});
                kokkos_sv.applyOperation("RZ", {index}, false, {angles[index]});
                std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j]) * 2 * std::sqrt(2)) ==
                          Approx(real(expected_results[index][j])));
                    CHECK((imag(result_sv[j]) * 2 * std::sqrt(2)) ==
                          Approx(imag(expected_results[index][j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyPhaseShift",
                   "[StateVectorKokkosManaged_Param]", double) {
    const bool inverse = GENERATE(false, true);
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    const size_t num_qubits = 3;

    const std::vector<TestType> angles{0.3, 0.8, 2.4};
    const TestType sign = (inverse) ? -1.0 : 1.0;
    const ComplexT coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<ComplexT>> ps_data;
    ps_data.reserve(angles.size());
    for (auto &a : angles) {
        ps_data.push_back(getPhaseShift<Kokkos::complex, TestType>(a));
    }

    std::vector<std::vector<ComplexT>> expected_results = {
        {ps_data[0][0], ps_data[0][0], ps_data[0][0], ps_data[0][0],
         ps_data[0][3], ps_data[0][3], ps_data[0][3], ps_data[0][3]},
        {
            ps_data[1][0],
            ps_data[1][0],
            ps_data[1][3],
            ps_data[1][3],
            ps_data[1][0],
            ps_data[1][0],
            ps_data[1][3],
            ps_data[1][3],
        },
        {ps_data[2][0], ps_data[2][3], ps_data[2][0], ps_data[2][3],
         ps_data[2][0], ps_data[2][3], ps_data[2][0], ps_data[2][3]}};

    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyOperations(
                {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                {{false}, {false}, {false}});

            kokkos_sv.applyOperation("PhaseShift", {index}, inverse,
                                     {sign * angles[index]});
            std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
            kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

            for (size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK((real(result_sv[j]) * 2 * std::sqrt(2)) ==
                      Approx(real(expected_results[index][j])));
                CHECK((imag(result_sv[j]) * 2 * std::sqrt(2)) ==
                      Approx(imag(expected_results[index][j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyGlobalPhase",
                   "[StateVectorKokkosManaged_Param]", double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    std::mt19937_64 re{1337};
    const size_t num_qubits = 3;
    const bool inverse = GENERATE(false, true);
    const size_t index = GENERATE(0, 1, 2);
    const TestType param = 0.234;
    const ComplexT phase = Kokkos::exp(ComplexT{0, (inverse) ? param : -param});

    auto sv_data = createRandomStateVectorData<TestType>(re, num_qubits);
    StateVectorKokkos<TestType> kokkos_sv(
        reinterpret_cast<ComplexT *>(sv_data.data()), sv_data.size());
    kokkos_sv.applyOperation("GlobalPhase", {index}, inverse, {param});
    auto result_sv = kokkos_sv.getDataVector();
    for (size_t j = 0; j < exp2(num_qubits); j++) {
        ComplexT tmp = phase * ComplexT(sv_data[j]);
        CHECK((real(result_sv[j])) == Approx(real(tmp)));
        CHECK((imag(result_sv[j])) == Approx(imag(tmp)));
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyControlledGlobalPhase",
                   "[StateVectorKokkosManaged_Param]", double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    std::mt19937_64 re{1337};
    const size_t num_qubits = 3;
    const bool inverse = GENERATE(false, true);
    const size_t index = GENERATE(0, 1, 2);
    /* The `phase` array contains the diagonal entries of the controlled-phase
       operator. It can be created in Python using the following command

       ```
       global_phase_diagonal(-np.pi/2, wires=[0, 1, 2], controls=[0, 1],
       control_values=[0, 1])
       ```

       where the phase angle is chosen as `-np.pi/2` for simplicity.
    */
    const std::vector<ComplexT> phase = {{1.0, 0.}, {1.0, 0.}, {0.0, 1.},
                                         {0.0, 1.}, {1.0, 0.}, {1.0, 0.},
                                         {1.0, 0.}, {1.0, 0.}};

    auto sv_data = createRandomStateVectorData<TestType>(re, num_qubits);
    StateVectorKokkos<TestType> kokkos_sv(
        reinterpret_cast<ComplexT *>(sv_data.data()), sv_data.size());
    kokkos_sv.applyOperation("C(GlobalPhase)", {index}, inverse, {}, phase);
    auto result_sv = kokkos_sv.getDataVector();
    for (size_t j = 0; j < exp2(num_qubits); j++) {
        ComplexT tmp = (inverse) ? conj(phase[j]) : phase[j];
        tmp *= ComplexT(sv_data[j]);
        CHECK((real(result_sv[j])) == Approx(real(tmp)));
        CHECK((imag(result_sv[j])) == Approx(imag(tmp)));
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyControlledPhaseShift",
                   "[StateVectorKokkosManaged_Param]", double) {
    const bool inverse = GENERATE(false, true);
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    const size_t num_qubits = 3;

    const std::vector<TestType> angles{0.3, 2.4};
    const TestType sign = (inverse) ? -1.0 : 1.0;
    const ComplexT coef(1.0 / (2 * std::sqrt(2)), 0);

    std::vector<std::vector<ComplexT>> ps_data;
    ps_data.reserve(angles.size());
    for (auto &a : angles) {
        ps_data.push_back(getPhaseShift<Kokkos::complex, TestType>(a));
    }

    std::vector<std::vector<ComplexT>> expected_results = {
        {ps_data[0][0], ps_data[0][0], ps_data[0][0], ps_data[0][0],
         ps_data[0][0], ps_data[0][0], ps_data[0][3], ps_data[0][3]},
        {ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][3],
         ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][3]}};

    SECTION("Apply using dispatcher") {
        StateVectorKokkos<TestType> kokkos_sv{num_qubits};
        kokkos_sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                  {{0}, {1}, {2}}, {{false}, {false}, {false}});
        kokkos_sv.applyOperation("ControlledPhaseShift", {1, 2}, inverse,
                                 {sign * angles[1]});
        std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
        kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

        for (size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK((real(result_sv[j]) * 2 * std::sqrt(2)) ==
                  Approx(real(expected_results[1][j])));
            CHECK((imag(result_sv[j]) * 2 * std::sqrt(2)) ==
                  Approx(imag(expected_results[1][j])));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyRot",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    const size_t num_qubits = 3;
    const std::vector<std::vector<TestType>> angles{
        std::vector<TestType>{0.3, 0.8, 2.4},
        std::vector<TestType>{0.5, 1.1, 3.0},
        std::vector<TestType>{2.3, 0.1, 0.4}};

    std::vector<std::vector<ComplexT>> expected_results{
        std::vector<ComplexT>(0b1 << num_qubits),
        std::vector<ComplexT>(0b1 << num_qubits),
        std::vector<ComplexT>(0b1 << num_qubits)};

    for (size_t i = 0; i < angles.size(); i++) {
        const auto rot_mat = getRot<Kokkos::complex, TestType>(
            angles[i][0], angles[i][1], angles[i][2]);
        expected_results[i][0] = rot_mat[0];
        expected_results[i][0b1 << (num_qubits - i - 1)] = rot_mat[2];
    }

    SECTION("Apply using dispatcher") {
        for (size_t index = 0; index < num_qubits; index++) {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};

            kokkos_sv.applyOperation("Rot", {index}, false, angles[index]);
            std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
            kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

            for (size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK((real(result_sv[j])) ==
                      Approx(real(expected_results[index][j])));
                CHECK((imag(result_sv[j])) ==
                      Approx(imag(expected_results[index][j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyCRot",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    const bool inverse = GENERATE(false, true);

    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    const size_t num_qubits = 3;

    const std::vector<TestType> angles{0.3, 0.8, 2.4};
    std::vector<ComplexT> expected_results(8);
    const auto rot_mat = (inverse) ? getRot<Kokkos::complex, TestType>(
                                         -angles[2], -angles[1], -angles[0])
                                   : getRot<Kokkos::complex, TestType>(
                                         angles[0], angles[1], angles[2]);
    expected_results[0b1 << (num_qubits - 1)] = rot_mat[0];
    expected_results[(0b1 << num_qubits) - 2] = rot_mat[2];

    SECTION("Apply using dispatcher") {
        SECTION("CRot0,1 |100> -> |1>(a|0>+b|1>)|0>") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            kokkos_sv.applyOperation("PauliX", {0}, false);
            kokkos_sv.applyOperation("CRot", {0, 1}, inverse, angles);
            std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
            kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

            for (size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK((real(result_sv[j])) ==
                      Approx(real(expected_results[j])));
                CHECK((imag(result_sv[j])) ==
                      Approx(imag(expected_results[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyIsingXX",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    const size_t num_qubits = 3;

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<ComplexT>> expected_results{
        std::vector<ComplexT>(1 << num_qubits),
        std::vector<ComplexT>(1 << num_qubits),
        std::vector<ComplexT>(1 << num_qubits),
        std::vector<ComplexT>(1 << num_qubits)};
    expected_results[0][0] = ComplexT{0.9887710779360422, 0.0};
    expected_results[0][6] = ComplexT{0.0, -0.14943813247359922};

    expected_results[1][0] = ComplexT{0.9210609940028851, 0.0};
    expected_results[1][6] = ComplexT{0.0, -0.3894183423086505};

    expected_results[2][0] = ComplexT{0.9887710779360422, 0.0};
    expected_results[2][5] = ComplexT{0.0, -0.14943813247359922};

    expected_results[3][0] = ComplexT{0.9210609940028851, 0.0};
    expected_results[3][5] = ComplexT{0.0, -0.3894183423086505};

    SECTION("Apply directly") {
        bool adjoint = GENERATE(false, true);
        SECTION("IsingXX 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation("IsingXX", {0, 1}, adjoint,
                                         {angles[index]});
                std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(((adjoint) ? -1.0 : 1.0) *
                                 imag(expected_results[index][j])));
                }
            }
        }
        SECTION("IsingXX 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation("IsingXX", {0, 2}, adjoint,
                                         {angles[index]});
                std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(
                              expected_results[index + angles.size()][j])));
                    CHECK(
                        (imag(result_sv[j])) ==
                        Approx(
                            ((adjoint) ? -1.0 : 1.0) *
                            imag(expected_results[index + angles.size()][j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyIsingYY",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    const size_t num_qubits = 3;

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<ComplexT>> expected_results{
        std::vector<ComplexT>(1 << num_qubits),
        std::vector<ComplexT>(1 << num_qubits),
        std::vector<ComplexT>(1 << num_qubits),
        std::vector<ComplexT>(1 << num_qubits)};
    expected_results[0][0] = ComplexT{0.9887710779360422, 0.0};
    expected_results[0][6] = ComplexT{0.0, 0.14943813247359922};

    expected_results[1][0] = ComplexT{0.9210609940028851, 0.0};
    expected_results[1][6] = ComplexT{0.0, 0.3894183423086505};

    expected_results[2][0] = ComplexT{0.9887710779360422, 0.0};
    expected_results[2][5] = ComplexT{0.0, 0.14943813247359922};

    expected_results[3][0] = ComplexT{0.9210609940028851, 0.0};
    expected_results[3][5] = ComplexT{0.0, 0.3894183423086505};

    SECTION("Apply directly") {
        bool adjoint = GENERATE(false, true);
        SECTION("IsingYY 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation("IsingYY", {0, 1}, adjoint,
                                         {angles[index]});
                std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(((adjoint) ? -1.0 : 1.0) *
                                 imag(expected_results[index][j])));
                }
            }
        }
        SECTION("IsingYY 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation("IsingYY", {0, 2}, adjoint,
                                         {angles[index]});
                std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(
                              expected_results[index + angles.size()][j])));
                    CHECK(
                        (imag(result_sv[j])) ==
                        Approx(
                            ((adjoint) ? -1.0 : 1.0) *
                            imag(expected_results[index + angles.size()][j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyIsingZZ",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    const size_t num_qubits = 3;

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<ComplexT>> expected_results{
        std::vector<ComplexT>(1 << num_qubits, {0, 0}),
        std::vector<ComplexT>(1 << num_qubits, {0, 0})};
    expected_results[0][0] = ComplexT{0.9887710779360422, -0.14943813247359922};
    expected_results[1][0] = ComplexT{0.9210609940028851, -0.3894183423086505};

    SECTION("Apply directly") {
        bool adjoint = GENERATE(false, true);
        SECTION("IsingZZ 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation("IsingZZ", {0, 1}, adjoint,
                                         {angles[index]});
                std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(((adjoint) ? -1.0 : 1.0) *
                                 imag(expected_results[index][j])));
                }
            }
        }
        SECTION("IsingZZ 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation("IsingZZ", {0, 2}, adjoint,
                                         {angles[index]});
                std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(((adjoint) ? -1.0 : 1.0) *
                                 imag(expected_results[index][j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyMultiRZ",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    const size_t num_qubits = 3;

    const std::vector<TestType> angles{0.3, 0.8};

    std::vector<std::vector<ComplexT>> expected_results{
        std::vector<ComplexT>(1 << num_qubits),
        std::vector<ComplexT>(1 << num_qubits)};
    expected_results[0][0] = ComplexT{0.9887710779360422, -0.14943813247359922};
    expected_results[1][0] = ComplexT{0.9210609940028851, -0.3894183423086505};

    SECTION("Apply directly") {
        bool adjoint = GENERATE(false, true);
        SECTION("MultiRZ 0,1") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation("MultiRZ", {0, 1}, adjoint,
                                         {angles[index]});
                std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(((adjoint) ? -1.0 : 1.0) *
                                 imag(expected_results[index][j])));
                }
            }
        }
        SECTION("MultiRZ 0,2") {
            for (size_t index = 0; index < angles.size(); index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation("MultiRZ", {0, 2}, adjoint,
                                         {angles[index]});
                std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
                kokkos_sv.DeviceToHost(result_sv.data(), kokkos_sv.getLength());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK((real(result_sv[j])) ==
                          Approx(real(expected_results[index][j])));
                    CHECK((imag(result_sv[j])) ==
                          Approx(((adjoint) ? -1.0 : 1.0) *
                                 imag(expected_results[index][j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applySingleExcitation",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    const bool inverse = GENERATE(false, true);
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 3;

        std::vector<ComplexT> ini_st{
            ComplexT{0.125681356503, 0.252712197380},
            ComplexT{0.262591068130, 0.370189000494},
            ComplexT{0.129300299863, 0.371057794075},
            ComplexT{0.392248682814, 0.195795523118},
            ComplexT{0.303908059240, 0.082981563244},
            ComplexT{0.189140284321, 0.179512645957},
            ComplexT{0.173146612336, 0.092249594834},
            ComplexT{0.298857179897, 0.269627836165},
        };

        std::vector<ComplexT> expected{
            ComplexT{0.125681, 0.252712}, ComplexT{0.219798, 0.355848},
            ComplexT{0.1293, 0.371058},   ComplexT{0.365709, 0.181773},
            ComplexT{0.336159, 0.131522}, ComplexT{0.18914, 0.179513},
            ComplexT{0.223821, 0.117493}, ComplexT{0.298857, 0.269628},
        };

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
            const TestType angle =
                (inverse) ? -0.267030328057308 : 0.267030328057308;
            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applyOperation("SingleExcitation", {0, 2}, inverse,
                                     {angle});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applySingleExcitationMinus",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    const bool inverse = GENERATE(false, true);
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 3;

        std::vector<ComplexT> ini_st{
            ComplexT{0.125681356503, 0.252712197380},
            ComplexT{0.262591068130, 0.370189000494},
            ComplexT{0.129300299863, 0.371057794075},
            ComplexT{0.392248682814, 0.195795523118},
            ComplexT{0.303908059240, 0.082981563244},
            ComplexT{0.189140284321, 0.179512645957},
            ComplexT{0.173146612336, 0.092249594834},
            ComplexT{0.298857179897, 0.269627836165},
        };

        std::vector<ComplexT> expected{
            ComplexT{0.158204, 0.233733}, ComplexT{0.219798, 0.355848},
            ComplexT{0.177544, 0.350543}, ComplexT{0.365709, 0.181773},
            ComplexT{0.336159, 0.131522}, ComplexT{0.211353, 0.152737},
            ComplexT{0.223821, 0.117493}, ComplexT{0.33209, 0.227445},
        };

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
            const TestType angle =
                (inverse) ? -0.267030328057308 : 0.267030328057308;
            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applyOperation("SingleExcitationMinus", {0, 2}, inverse,
                                     {angle});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applySingleExcitationPlus",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    const bool inverse = GENERATE(false, true);
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 3;

        std::vector<ComplexT> ini_st{
            ComplexT{0.125681356503, 0.252712197380},
            ComplexT{0.262591068130, 0.370189000494},
            ComplexT{0.129300299863, 0.371057794075},
            ComplexT{0.392248682814, 0.195795523118},
            ComplexT{0.303908059240, 0.082981563244},
            ComplexT{0.189140284321, 0.179512645957},
            ComplexT{0.173146612336, 0.092249594834},
            ComplexT{0.298857179897, 0.269627836165},
        };

        std::vector<ComplexT> expected{
            ComplexT{0.090922, 0.267194},  ComplexT{0.219798, 0.355848},
            ComplexT{0.0787548, 0.384968}, ComplexT{0.365709, 0.181773},
            ComplexT{0.336159, 0.131522},  ComplexT{0.16356, 0.203093},
            ComplexT{0.223821, 0.117493},  ComplexT{0.260305, 0.307012},
        };

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
            const TestType angle =
                (inverse) ? -0.267030328057308 : 0.267030328057308;
            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_sv.applyOperation("SingleExcitationPlus", {0, 2}, inverse,
                                     {angle});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            for (size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(imag(expected[j]) == Approx(imag(result_sv[j])));
                CHECK(real(expected[j]) == Approx(real(result_sv[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyDoubleExcitation",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    std::vector<std::size_t> wires = {0, 1, 2, 3};
    std::pair<std::size_t, std::size_t> control =
        GENERATE(std::pair<std::size_t, std::size_t>{0, 0},
                 std::pair<std::size_t, std::size_t>{0, 1},
                 std::pair<std::size_t, std::size_t>{0, 2},
                 std::pair<std::size_t, std::size_t>{0, 3},
                 std::pair<std::size_t, std::size_t>{1, 2},
                 std::pair<std::size_t, std::size_t>{1, 3},
                 std::pair<std::size_t, std::size_t>{2, 3},
                 std::pair<std::size_t, std::size_t>{3, 3});
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 4;

        std::vector<ComplexT> ini_st{
            ComplexT{0.125681356503, 0.252712197380},
            ComplexT{0.262591068130, 0.370189000494},
            ComplexT{0.129300299863, 0.371057794075},
            ComplexT{0.392248682814, 0.195795523118},
            ComplexT{0.303908059240, 0.082981563244},
            ComplexT{0.189140284321, 0.179512645957},
            ComplexT{0.173146612336, 0.092249594834},
            ComplexT{0.298857179897, 0.269627836165},
            ComplexT{0.125681356503, 0.252712197380},
            ComplexT{0.262591068130, 0.370189000494},
            ComplexT{0.129300299863, 0.371057794075},
            ComplexT{0.392248682814, 0.195795523118},
            ComplexT{0.303908059240, 0.082981563244},
            ComplexT{0.189140284321, 0.179512645957},
            ComplexT{0.173146612336, 0.092249594834},
            ComplexT{0.298857179897, 0.269627836165},
        };

        std::vector<ComplexT> expected{
            ComplexT{0.125681, 0.252712},  ComplexT{0.262591, 0.370189},
            ComplexT{0.1293, 0.371058},    ComplexT{0.348302, 0.183007},
            ComplexT{0.303908, 0.0829816}, ComplexT{0.18914, 0.179513},
            ComplexT{0.173147, 0.0922496}, ComplexT{0.298857, 0.269628},
            ComplexT{0.125681, 0.252712},  ComplexT{0.262591, 0.370189},
            ComplexT{0.1293, 0.371058},    ComplexT{0.392249, 0.195796},
            ComplexT{0.353419, 0.108307},  ComplexT{0.18914, 0.179513},
            ComplexT{0.173147, 0.0922496}, ComplexT{0.298857, 0.269628},
        };

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            if (control.first == 3 && control.second == 3) {
                std::swap(wires[0], wires[2]);
                kokkos_sv.applyOperation("SWAP", {0, 2});
                std::swap(wires[1], wires[3]);
                kokkos_sv.applyOperation("SWAP", {1, 3});
            } else if (control.first != control.second) {
                std::swap(wires[control.first], wires[control.second]);
                kokkos_sv.applyOperation("SWAP",
                                         {control.first, control.second});
            }

            kokkos_sv.applyOperation("DoubleExcitation", wires, false,
                                     {0.267030328057308});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            StateVectorKokkos<TestType> kokkos_ref{num_qubits};
            std::vector<ComplexT> result_ref(kokkos_ref.getLength(), {0, 0});
            kokkos_ref.HostToDevice(expected.data(), expected.size());
            if (control.first == 3 && control.second == 3) {
                std::swap(wires[0], wires[2]);
                kokkos_ref.applyOperation("SWAP", {0, 2});
                std::swap(wires[1], wires[3]);
                kokkos_ref.applyOperation("SWAP", {1, 3});
            } else if (control.first != control.second) {
                kokkos_ref.applyOperation("SWAP",
                                          {control.first, control.second});
            }
            kokkos_ref.DeviceToHost(result_ref.data(), result_ref.size());

            for (size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(imag(result_ref[j]) == Approx(imag(result_sv[j])));
                CHECK(real(result_ref[j]) == Approx(real(result_sv[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyDoubleExcitationMinus",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    std::vector<std::size_t> wires = {0, 1, 2, 3};
    std::pair<std::size_t, std::size_t> control =
        GENERATE(std::pair<std::size_t, std::size_t>{0, 0},
                 std::pair<std::size_t, std::size_t>{0, 1},
                 std::pair<std::size_t, std::size_t>{0, 2},
                 std::pair<std::size_t, std::size_t>{0, 3},
                 std::pair<std::size_t, std::size_t>{1, 2},
                 std::pair<std::size_t, std::size_t>{1, 3},
                 std::pair<std::size_t, std::size_t>{2, 3},
                 std::pair<std::size_t, std::size_t>{3, 3});
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 4;

        std::vector<ComplexT> ini_st{
            ComplexT{0.125681356503, 0.252712197380},
            ComplexT{0.262591068130, 0.370189000494},
            ComplexT{0.129300299863, 0.371057794075},
            ComplexT{0.392248682814, 0.195795523118},
            ComplexT{0.303908059240, 0.082981563244},
            ComplexT{0.189140284321, 0.179512645957},
            ComplexT{0.173146612336, 0.092249594834},
            ComplexT{0.298857179897, 0.269627836165},
            ComplexT{0.125681356503, 0.252712197380},
            ComplexT{0.262591068130, 0.370189000494},
            ComplexT{0.129300299863, 0.371057794075},
            ComplexT{0.392248682814, 0.195795523118},
            ComplexT{0.303908059240, 0.082981563244},
            ComplexT{0.189140284321, 0.179512645957},
            ComplexT{0.173146612336, 0.092249594834},
            ComplexT{0.298857179897, 0.269627836165},
        };

        std::vector<ComplexT> expected{
            ComplexT{0.158204, 0.233733},  ComplexT{0.309533, 0.331939},
            ComplexT{0.177544, 0.350543},  ComplexT{0.348302, 0.183007},
            ComplexT{0.31225, 0.0417871},  ComplexT{0.211353, 0.152737},
            ComplexT{0.183886, 0.0683795}, ComplexT{0.33209, 0.227445},
            ComplexT{0.158204, 0.233733},  ComplexT{0.309533, 0.331939},
            ComplexT{0.177544, 0.350543},  ComplexT{0.414822, 0.141837},
            ComplexT{0.353419, 0.108307},  ComplexT{0.211353, 0.152737},
            ComplexT{0.183886, 0.0683795}, ComplexT{0.33209, 0.227445},
        };

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            if (control.first == 3 && control.second == 3) {
                std::swap(wires[0], wires[2]);
                kokkos_sv.applyOperation("SWAP", {0, 2});
                std::swap(wires[1], wires[3]);
                kokkos_sv.applyOperation("SWAP", {1, 3});
            } else if (control.first != control.second) {
                std::swap(wires[control.first], wires[control.second]);
                kokkos_sv.applyOperation("SWAP",
                                         {control.first, control.second});
            }

            kokkos_sv.applyOperation("DoubleExcitationMinus", wires, false,
                                     {0.267030328057308});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            StateVectorKokkos<TestType> kokkos_ref{num_qubits};
            std::vector<ComplexT> result_ref(kokkos_ref.getLength(), {0, 0});
            kokkos_ref.HostToDevice(expected.data(), expected.size());
            if (control.first == 3 && control.second == 3) {
                std::swap(wires[0], wires[2]);
                kokkos_ref.applyOperation("SWAP", {0, 2});
                std::swap(wires[1], wires[3]);
                kokkos_ref.applyOperation("SWAP", {1, 3});
            } else if (control.first != control.second) {
                kokkos_ref.applyOperation("SWAP",
                                          {control.first, control.second});
            }
            kokkos_ref.DeviceToHost(result_ref.data(), result_ref.size());

            for (size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(imag(result_ref[j]) == Approx(imag(result_sv[j])));
                CHECK(real(result_ref[j]) == Approx(real(result_sv[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::applyDoubleExcitationPlus",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    std::vector<std::size_t> wires = {0, 1, 2, 3};
    std::pair<std::size_t, std::size_t> control =
        GENERATE(std::pair<std::size_t, std::size_t>{0, 0},
                 std::pair<std::size_t, std::size_t>{0, 1},
                 std::pair<std::size_t, std::size_t>{0, 2},
                 std::pair<std::size_t, std::size_t>{0, 3},
                 std::pair<std::size_t, std::size_t>{1, 2},
                 std::pair<std::size_t, std::size_t>{1, 3},
                 std::pair<std::size_t, std::size_t>{2, 3},
                 std::pair<std::size_t, std::size_t>{3, 3});
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 4;

        std::vector<ComplexT> ini_st{
            ComplexT{0.125681356503, 0.252712197380},
            ComplexT{0.262591068130, 0.370189000494},
            ComplexT{0.129300299863, 0.371057794075},
            ComplexT{0.392248682814, 0.195795523118},
            ComplexT{0.303908059240, 0.082981563244},
            ComplexT{0.189140284321, 0.179512645957},
            ComplexT{0.173146612336, 0.092249594834},
            ComplexT{0.298857179897, 0.269627836165},
            ComplexT{0.125681356503, 0.252712197380},
            ComplexT{0.262591068130, 0.370189000494},
            ComplexT{0.129300299863, 0.371057794075},
            ComplexT{0.392248682814, 0.195795523118},
            ComplexT{0.303908059240, 0.082981563244},
            ComplexT{0.189140284321, 0.179512645957},
            ComplexT{0.173146612336, 0.092249594834},
            ComplexT{0.298857179897, 0.269627836165},
        };

        std::vector<ComplexT> expected{
            ComplexT{0.090922, 0.267194},  ComplexT{0.210975, 0.40185},
            ComplexT{0.0787548, 0.384968}, ComplexT{0.348302, 0.183007},
            ComplexT{0.290157, 0.122699},  ComplexT{0.16356, 0.203093},
            ComplexT{0.159325, 0.114478},  ComplexT{0.260305, 0.307012},
            ComplexT{0.090922, 0.267194},  ComplexT{0.210975, 0.40185},
            ComplexT{0.0787548, 0.384968}, ComplexT{0.362694, 0.246269},
            ComplexT{0.353419, 0.108307},  ComplexT{0.16356, 0.203093},
            ComplexT{0.159325, 0.114478},  ComplexT{0.260305, 0.307012},
        };

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
            kokkos_sv.HostToDevice(ini_st.data(), ini_st.size());
            if (control.first == 3 && control.second == 3) {
                std::swap(wires[0], wires[2]);
                kokkos_sv.applyOperation("SWAP", {0, 2});
                std::swap(wires[1], wires[3]);
                kokkos_sv.applyOperation("SWAP", {1, 3});
            } else if (control.first != control.second) {
                std::swap(wires[control.first], wires[control.second]);
                kokkos_sv.applyOperation("SWAP",
                                         {control.first, control.second});
            }

            kokkos_sv.applyOperation("DoubleExcitationPlus", wires, false,
                                     {0.267030328057308});
            kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

            StateVectorKokkos<TestType> kokkos_ref{num_qubits};
            std::vector<ComplexT> result_ref(kokkos_ref.getLength(), {0, 0});
            kokkos_ref.HostToDevice(expected.data(), expected.size());
            if (control.first == 3 && control.second == 3) {
                std::swap(wires[0], wires[2]);
                kokkos_ref.applyOperation("SWAP", {0, 2});
                std::swap(wires[1], wires[3]);
                kokkos_ref.applyOperation("SWAP", {1, 3});
            } else if (control.first != control.second) {
                kokkos_ref.applyOperation("SWAP",
                                          {control.first, control.second});
            }
            kokkos_ref.DeviceToHost(result_ref.data(), result_ref.size());

            for (size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(imag(result_ref[j]) == Approx(imag(result_sv[j])));
                CHECK(real(result_ref[j]) == Approx(real(result_sv[j])));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Sample", "[StateVectorKokkosManaged_Param]", float,
                   double) {
    constexpr uint32_t twos[] = {
        1U << 0U,  1U << 1U,  1U << 2U,  1U << 3U,  1U << 4U,  1U << 5U,
        1U << 6U,  1U << 7U,  1U << 8U,  1U << 9U,  1U << 10U, 1U << 11U,
        1U << 12U, 1U << 13U, 1U << 14U, 1U << 15U, 1U << 16U, 1U << 17U,
        1U << 18U, 1U << 19U, 1U << 20U, 1U << 21U, 1U << 22U, 1U << 23U,
        1U << 24U, 1U << 25U, 1U << 26U, 1U << 27U, 1U << 28U, 1U << 29U,
        1U << 30U, 1U << 31U};

    // Defining the State Vector that will be measured.
    const size_t num_qubits = 3;
    StateVectorKokkos<TestType> measure_sv{num_qubits};

    std::vector<std::string> gates;
    std::vector<std::vector<size_t>> wires;
    std::vector<bool> inv_op(num_qubits * 2, false);
    std::vector<std::vector<TestType>> phase;

    TestType initial_phase = 0.7;
    for (size_t n_qubit = 0; n_qubit < num_qubits; n_qubit++) {
        gates.emplace_back("RX");
        gates.emplace_back("RY");

        wires.push_back({n_qubit});
        wires.push_back({n_qubit});

        phase.push_back({initial_phase});
        phase.push_back({initial_phase});
        initial_phase -= 0.2;
    }

    measure_sv.applyOperations(gates, wires, inv_op, phase);

    std::vector<TestType> expected_probabilities = {
        0.67078706, 0.03062806, 0.0870997,  0.00397696,
        0.17564072, 0.00801973, 0.02280642, 0.00104134};

    size_t N = std::pow(2, num_qubits);
    size_t num_samples = 100000;

    auto m = Measures::Measurements(measure_sv);
    auto samples = m.generate_samples(num_samples);

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
    std::vector<TestType> probabilities(counts.size());
    for (size_t i = 0; i < counts.size(); i++) {
        probabilities[i] = counts[i] / (TestType)num_samples;
    }

    // compare estimated probabilities to real probabilities
    SECTION("No wires provided:") {
        REQUIRE_THAT(probabilities,
                     Catch::Approx(expected_probabilities).margin(.05));
    }
}

TEMPLATE_TEST_CASE("Test NQubit gate versus expectation value",
                   "[StateVectorKokkosManaged_Param]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    std::mt19937_64 re{1337};
    const size_t num_qubits = 7;
    auto sv_data = createRandomStateVectorData<TestType>(re, num_qubits);

    StateVectorKokkos<TestType> kokkos_sv(
        reinterpret_cast<ComplexT *>(sv_data.data()), sv_data.size());
    StateVectorKokkos<TestType> copy_sv{kokkos_sv};

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
        kokkos_sv.applyMatrix(matrix, {0, 1, 2});
        auto res = getRealOfComplexInnerProduct(copy_sv.getView(),
                                                kokkos_sv.getView());

        CHECK(expected == Approx(res));
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
        kokkos_sv.applyMatrix(matrix, {0, 1, 2, 3});
        auto res = getRealOfComplexInnerProduct(copy_sv.getView(),
                                                kokkos_sv.getView());

        CHECK(expected == Approx(res));
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
        kokkos_sv.applyMatrix(matrix, {0, 1, 2, 3, 4});
        auto res = getRealOfComplexInnerProduct(copy_sv.getView(),
                                                kokkos_sv.getView());

        CHECK(expected == Approx(res));
    }
}