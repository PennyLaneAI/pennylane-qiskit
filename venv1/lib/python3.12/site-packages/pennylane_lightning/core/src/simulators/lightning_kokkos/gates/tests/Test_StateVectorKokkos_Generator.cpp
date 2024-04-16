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
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup, array_has_elem, prepend_to_tuple, tuple_to_array
#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp"
#include "TestHelpersWires.hpp"

/**
 * @file
 *  Tests for generators functionality defined in the class StateVectorKokkos.
 */

/// @cond DEV
namespace {
using namespace Pennylane::Gates;
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::Util;
using std::size_t;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("StateVectorKokkos::applyGenerator - errors",
                   "[StateVectorKokkos_Generator]", float, double) {
    {
        const size_t num_qubits = 3;
        StateVectorKokkos<TestType> state_vector{num_qubits};
        PL_REQUIRE_THROWS_MATCHES(state_vector.applyGenerator("XXX", {0}),
                                  LightningException,
                                  "Generator does not exist for");
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyGenerator",
                   "[StateVectorKokkos_Generator]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    const size_t num_qubits = 4;
    const TestType ep = 1e-3;
    const TestType EP = 1e-4;

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

    std::unordered_map<std::string, GateOperation> str_to_gates_{};
    for (const auto &[gate_op, gate_name] : Constant::gate_names) {
        str_to_gates_.emplace(gate_name, gate_op);
    }

    const bool inverse = GENERATE(true, false);
    const std::string gate_name = GENERATE(
        "PhaseShift", "RX", "RY", "RZ", "ControlledPhaseShift", "CRX", "CRY",
        "CRZ", "IsingXX", "IsingXY", "IsingYY", "IsingZZ", "SingleExcitation",
        "SingleExcitationMinus", "SingleExcitationPlus", "DoubleExcitation",
        "DoubleExcitationMinus", "DoubleExcitationPlus", "MultiRZ",
        "GlobalPhase");
    {
        StateVectorKokkos<TestType> kokkos_gntr_sv{ini_st.data(),
                                                   ini_st.size()};
        StateVectorKokkos<TestType> kokkos_gate_svp{ini_st.data(),
                                                    ini_st.size()};
        StateVectorKokkos<TestType> kokkos_gate_svm{ini_st.data(),
                                                    ini_st.size()};

        const auto wires = createWires(str_to_gates_.at(gate_name), num_qubits);
        auto scale = kokkos_gntr_sv.applyGenerator(gate_name, wires, inverse);
        auto h = static_cast<TestType>(((inverse) ? -1.0 : 1.0) * ep);
        kokkos_gate_svp.applyOperation(gate_name, wires, inverse, {h});
        kokkos_gate_svm.applyOperation(gate_name, wires, inverse, {-h});

        auto result_gntr_sv = kokkos_gntr_sv.getDataVector();
        auto result_gate_svp = kokkos_gate_svp.getDataVector();
        auto result_gate_svm = kokkos_gate_svm.getDataVector();

        for (size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(-scale * imag(result_gntr_sv[j]) ==
                  Approx(0.5 *
                         (real(result_gate_svp[j]) - real(result_gate_svm[j])) /
                         ep)
                      .margin(EP));
            CHECK(scale * real(result_gntr_sv[j]) ==
                  Approx(0.5 *
                         (imag(result_gate_svp[j]) - imag(result_gate_svm[j])) /
                         ep)
                      .margin(EP));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyGeneratorDoubleExcitation",
                   "[StateVectorKokkos_Generator]", float, double) {
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
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<ComplexT> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                                 {0, 0});
            std::vector<ComplexT> result_gate_svp(kokkos_gate_svp.getLength(),
                                                  {0, 0});
            std::vector<ComplexT> result_gate_svm(kokkos_gate_svm.getLength(),
                                                  {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            if (control.first == 3 && control.second == 3) {
                std::swap(wires[0], wires[2]);
                std::swap(wires[1], wires[3]);
            } else if (control.first != control.second) {
                std::swap(wires[control.first], wires[control.second]);
            }

            auto scale =
                kokkos_gntr_sv.applyGenerator("DoubleExcitation", wires, false);
            kokkos_gate_svp.applyOperation("DoubleExcitation", wires, false,
                                           {ep});
            kokkos_gate_svm.applyOperation("DoubleExcitation", wires, false,
                                           {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyGeneratorDoubleExcitationMinus",
                   "[StateVectorKokkos_Generator]", float, double) {
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
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<ComplexT> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                                 {0, 0});
            std::vector<ComplexT> result_gate_svp(kokkos_gate_svp.getLength(),
                                                  {0, 0});
            std::vector<ComplexT> result_gate_svm(kokkos_gate_svm.getLength(),
                                                  {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            if (control.first == 3 && control.second == 3) {
                std::swap(wires[0], wires[2]);
                std::swap(wires[1], wires[3]);
            } else if (control.first != control.second) {
                std::swap(wires[control.first], wires[control.second]);
            }

            auto scale = kokkos_gntr_sv.applyGenerator("DoubleExcitationMinus",
                                                       wires, false);
            kokkos_gate_svp.applyOperation("DoubleExcitationMinus", wires,
                                           false, {ep});
            kokkos_gate_svm.applyOperation("DoubleExcitationMinus", wires,
                                           false, {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyGeneratorDoubleExcitationPlus",
                   "[StateVectorKokkos_Generator]", float, double) {
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
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

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

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_gntr_sv{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svp{num_qubits};
            StateVectorKokkos<TestType> kokkos_gate_svm{num_qubits};

            std::vector<ComplexT> result_gntr_sv(kokkos_gntr_sv.getLength(),
                                                 {0, 0});
            std::vector<ComplexT> result_gate_svp(kokkos_gate_svp.getLength(),
                                                  {0, 0});
            std::vector<ComplexT> result_gate_svm(kokkos_gate_svm.getLength(),
                                                  {0, 0});

            kokkos_gntr_sv.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svp.HostToDevice(ini_st.data(), ini_st.size());
            kokkos_gate_svm.HostToDevice(ini_st.data(), ini_st.size());

            if (control.first == 3 && control.second == 3) {
                std::swap(wires[0], wires[2]);
                std::swap(wires[1], wires[3]);
            } else if (control.first != control.second) {
                std::swap(wires[control.first], wires[control.second]);
            }

            auto scale = kokkos_gntr_sv.applyGenerator("DoubleExcitationPlus",
                                                       wires, false);
            kokkos_gate_svp.applyOperation("DoubleExcitationPlus", wires, false,
                                           {ep});
            kokkos_gate_svm.applyOperation("DoubleExcitationPlus", wires, false,
                                           {-ep});

            kokkos_gntr_sv.DeviceToHost(result_gntr_sv.data(),
                                        result_gntr_sv.size());
            kokkos_gate_svp.DeviceToHost(result_gate_svp.data(),
                                         result_gate_svp.size());
            kokkos_gate_svm.DeviceToHost(result_gate_svm.data(),
                                         result_gate_svm.size());

            for (size_t j = 0; j < exp2(num_qubits); j++) {
                CHECK(-scale * imag(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (real(result_gate_svp[j]) -
                              real(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
                CHECK(scale * real(result_gntr_sv[j]) ==
                      Approx(0.5 *
                             (imag(result_gate_svp[j]) -
                              imag(result_gate_svm[j])) /
                             ep)
                          .margin(EP));
            }
        }
    }
}
