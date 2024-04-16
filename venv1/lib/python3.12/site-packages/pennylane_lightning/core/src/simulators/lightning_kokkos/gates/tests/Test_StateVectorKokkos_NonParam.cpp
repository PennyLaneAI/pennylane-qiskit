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
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch.hpp>

#include "Gates.hpp" // getHadamard
#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp"

/**
 * @file
 *  Tests for non-parametric gates functionality defined in the class
 * StateVectorKokkos.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::Gates; // getHadamard, getCNOT,
                                  // getToffoli
using namespace Pennylane::Util;
using std::size_t;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("StateVectorKokkos::CopyConstructor",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        const size_t num_qubits = 3;
        StateVectorKokkos<TestType> kokkos_sv_1{num_qubits};
        kokkos_sv_1.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                    {{0}, {1}, {2}},
                                    {{false}, {false}, {false}});
        StateVectorKokkos<TestType> kokkos_sv_2{kokkos_sv_1};

        CHECK(kokkos_sv_1.getLength() == kokkos_sv_2.getLength());
        CHECK(kokkos_sv_1.getNumQubits() == kokkos_sv_2.getNumQubits());

        std::vector<Kokkos::complex<TestType>> kokkos_sv_1_host(
            kokkos_sv_1.getLength());
        std::vector<Kokkos::complex<TestType>> kokkos_sv_2_host(
            kokkos_sv_2.getLength());
        kokkos_sv_1.DeviceToHost(kokkos_sv_1_host.data(),
                                 kokkos_sv_1.getLength());
        kokkos_sv_2.DeviceToHost(kokkos_sv_2_host.data(),
                                 kokkos_sv_2.getLength());

        for (size_t i = 0; i < kokkos_sv_1_host.size(); i++) {
            CHECK(kokkos_sv_1_host[i] == kokkos_sv_2_host[i]);
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyNamedOperation",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        const size_t num_qubits = 3;
        StateVectorKokkos<TestType> state_vector{num_qubits};
        PL_REQUIRE_THROWS_MATCHES(state_vector.applyNamedOperation("XXX", {0}),
                                  LightningException,
                                  "Operation does not exist for");
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorKokkos::applyCY",
                           "[StateVectorKokkos_Nonparam]", (StateVectorKokkos),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    const bool inverse = GENERATE(true, false);

    SECTION("Apply::CY") {
        // Defining the statevector that will be measured.
        auto statevector_data = createNonTrivialState<StateVectorT>();
        StateVectorT sv1(statevector_data.data(), statevector_data.size());
        StateVectorT sv2(statevector_data.data(), statevector_data.size());

        const std::vector<std::size_t> wires{0, 1};
        std::vector<ComplexT> matrix = getCY<Kokkos::complex, PrecisionT>();
        sv1.applyOperation("CY", wires, inverse);
        sv2.applyMatrix(matrix, wires, inverse);

        auto result1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           sv1.getView());
        auto result2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           sv2.getView());

        for (size_t j = 0; j < sv1.getView().size(); j++) {
            CHECK(imag(result1[j]) == Approx(imag(result2[j])));
            CHECK(real(result1[j]) == Approx(real(result2[j])));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyHadamard",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    const bool inverse = GENERATE(true, false);
    {
        const size_t num_qubits = 3;
        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv(num_qubits);
                kokkos_sv.applyOperation("Hadamard", {index}, inverse);
                Kokkos::complex<TestType> expected(1.0 / std::sqrt(2), 0);
                auto result_subview = Kokkos::subview(kokkos_sv.getView(), 0);
                Kokkos::complex<TestType> result;
                Kokkos::deep_copy(result, result_subview);
                CHECK(expected.real() == Approx(result.real()));
            }
        }
        SECTION("Apply using matrix") {
            using ComplexT = StateVectorKokkos<TestType>::ComplexT;
            const auto isqrt2 = ComplexT{INVSQRT2<TestType>()};
            const std::vector<ComplexT> matrix = {isqrt2, isqrt2, isqrt2,
                                                  -isqrt2};
            for (size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv(num_qubits);
                kokkos_sv.applyOperation("Hadamard", {index}, inverse, {},
                                         matrix);
                Kokkos::complex<TestType> expected(1.0 / std::sqrt(2), 0);
                auto result_subview = Kokkos::subview(kokkos_sv.getView(), 0);
                Kokkos::complex<TestType> result;
                Kokkos::deep_copy(result, result_subview);
                CHECK(expected.real() == Approx(result.real()));
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyPauliX",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 3;

        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperation("PauliX", {index}, false);
                auto result_subview_0 = Kokkos::subview(kokkos_sv.getView(), 0);
                auto result_subview_1 = Kokkos::subview(
                    kokkos_sv.getView(),
                    0b1 << (kokkos_sv.getNumQubits() - index - 1));
                Kokkos::complex<TestType> result_0, result_1;
                Kokkos::deep_copy(result_0, result_subview_0);
                Kokkos::deep_copy(result_1, result_subview_1);

                CHECK(result_0 == ComplexT{ZERO<TestType>()});
                CHECK(result_1 == ComplexT{ONE<TestType>()});
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyPauliY",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                  {{0}, {1}, {2}}, {{false}, {false}, {false}});

        const auto p = ComplexT{HALF<TestType>()} *
                       ComplexT{INVSQRT2<TestType>()} *
                       ComplexT{IMAG<TestType>()};
        const auto m = ComplexT{NEGONE<TestType>()} * p;

        const std::vector<std::vector<ComplexT>> expected_results = {
            {m, m, m, m, p, p, p, p},
            {m, m, p, p, m, m, p, p},
            {m, p, m, p, m, p, m, p}};

        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperations(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{false}, {false}, {false}});

                kokkos_sv.applyOperation("PauliY", {index}, false);
                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    auto result_subview =
                        Kokkos::subview(kokkos_sv.getView(), j);
                    Kokkos::complex<TestType> result;
                    Kokkos::deep_copy(result, result_subview);

                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result)));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyPauliZ",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                  {{0}, {1}, {2}}, {{false}, {false}, {false}});

        const auto p =
            ComplexT{HALF<TestType>()} * ComplexT{INVSQRT2<TestType>()};
        const auto m = ComplexT{NEGONE<TestType>()} * p;

        const std::vector<std::vector<ComplexT>> expected_results = {
            {p, p, p, p, m, m, m, m},
            {p, p, m, m, p, p, m, m},
            {p, m, p, m, p, m, p, m}};

        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperations(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{false}, {false}, {false}});

                kokkos_sv.applyOperation("PauliZ", {index}, false);
                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    auto result_subview =
                        Kokkos::subview(kokkos_sv.getView(), j);
                    Kokkos::complex<TestType> result;
                    Kokkos::deep_copy(result, result_subview);

                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result)));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyS", "[StateVectorKokkos_Nonparam]",
                   float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                  {{0}, {1}, {2}}, {{false}, {false}, {false}});

        auto r = ComplexT{HALF<TestType>()} * ComplexT{INVSQRT2<TestType>()};
        auto i = r * ComplexT{IMAG<TestType>()};

        const std::vector<std::vector<ComplexT>> expected_results = {
            {r, r, r, r, i, i, i, i},
            {r, r, i, i, r, r, i, i},
            {r, i, r, i, r, i, r, i}};

        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperations(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{false}, {false}, {false}});
                kokkos_sv.applyOperation("S", {index}, false);
                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    auto result_subview =
                        Kokkos::subview(kokkos_sv.getView(), j);
                    Kokkos::complex<TestType> result;
                    Kokkos::deep_copy(result, result_subview);
                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result)));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyT", "[StateVectorKokkos_Nonparam]",
                   float, double) {
    const bool inverse = GENERATE(true, false);
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                                  {{0}, {1}, {2}},
                                  {{inverse}, {inverse}, {inverse}});

        auto r = ComplexT{HALF<TestType>()} * ComplexT{INVSQRT2<TestType>()};
        auto i = ComplexT{HALF<TestType>()} * ComplexT{HALF<TestType>()} *
                 (ComplexT{IMAG<TestType>()} + ComplexT{ONE<TestType>()});
        if (inverse) {
            i = conj(i);
        }

        const std::vector<std::vector<ComplexT>> expected_results = {
            {r, r, r, r, i, i, i, i},
            {r, r, i, i, r, r, i, i},
            {r, i, r, i, r, i, r, i}};

        SECTION("Apply using dispatcher") {
            for (size_t index = 0; index < num_qubits; index++) {
                StateVectorKokkos<TestType> kokkos_sv{num_qubits};
                kokkos_sv.applyOperations(
                    {{"Hadamard"}, {"Hadamard"}, {"Hadamard"}}, {{0}, {1}, {2}},
                    {{inverse}, {inverse}, {inverse}});
                kokkos_sv.applyOperation("T", {index}, inverse);

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    auto result_subview =
                        Kokkos::subview(kokkos_sv.getView(), j);
                    Kokkos::complex<TestType> result;
                    Kokkos::deep_copy(result, result_subview);
                    CHECK(imag(expected_results[index][j]) ==
                          Approx(imag(result)));
                    CHECK(real(expected_results[index][j]) ==
                          Approx(real(result)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyCNOT",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperation("Hadamard", {0}, false);

        auto ini_sv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          kokkos_sv.getView());

        SECTION("Apply using dispatcher") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            Kokkos::deep_copy(kokkos_sv.getView(), ini_sv);
            auto result = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, kokkos_sv.getView());
            for (size_t index = 1; index < num_qubits; index++) {
                kokkos_sv.applyOperation("CNOT", {index - 1, index}, false);
            }
            Kokkos::deep_copy(result, kokkos_sv.getView());
            CHECK(imag(ComplexT{INVSQRT2<TestType>()}) ==
                  Approx(imag(result[0])));
            CHECK(real(ComplexT{INVSQRT2<TestType>()}) ==
                  Approx(real(result[0])));
            CHECK(imag(ComplexT{INVSQRT2<TestType>()}) ==
                  Approx(imag(result[7])));
            CHECK(real(ComplexT{INVSQRT2<TestType>()}) ==
                  Approx(real(result[7])));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applySWAP",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                                  {{false}, {false}});

        auto ini_sv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          kokkos_sv.getView());

        auto z = ComplexT{ZERO<TestType>()};
        auto i = ComplexT{INVSQRT2<TestType>()};

        SECTION("Apply using dispatcher") {
            SECTION("SWAP0,1 |+10> -> 1+0>") {
                const std::vector<ComplexT> expected_results = {z, z, z, z,
                                                                i, z, i, z};

                StateVectorKokkos<TestType> svdat01{num_qubits};
                StateVectorKokkos<TestType> svdat10{num_qubits};
                Kokkos::deep_copy(svdat01.getView(), ini_sv);
                Kokkos::deep_copy(svdat10.getView(), ini_sv);

                svdat01.applyOperation("SWAP", {0, 1}, false);
                svdat10.applyOperation("SWAP", {1, 0}, false);

                auto sv01 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat01.getView());
                auto sv10 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat10.getView());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv01[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv01[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv10[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv10[j])));
                }
            }
            SECTION("SWAP0,2 |+10> -> |01+>") {
                const std::vector<ComplexT> expected_results = {z, z, i, i,
                                                                z, z, z, z};

                StateVectorKokkos<TestType> svdat02{num_qubits};
                StateVectorKokkos<TestType> svdat20{num_qubits};
                Kokkos::deep_copy(svdat02.getView(), ini_sv);
                Kokkos::deep_copy(svdat20.getView(), ini_sv);

                svdat02.applyOperation("SWAP", {0, 2}, false);
                svdat20.applyOperation("SWAP", {2, 0}, false);

                auto sv02 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat02.getView());
                auto sv20 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat20.getView());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv02[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv02[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv20[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv20[j])));
                }
            }
            SECTION("SWAP1,2 |+10> -> |+01>") {
                const std::vector<ComplexT> expected_results = {z, i, z, z,
                                                                z, i, z, z};

                StateVectorKokkos<TestType> svdat12{num_qubits};
                StateVectorKokkos<TestType> svdat21{num_qubits};
                Kokkos::deep_copy(svdat12.getView(), ini_sv);
                Kokkos::deep_copy(svdat21.getView(), ini_sv);

                svdat12.applyOperation("SWAP", {1, 2}, false);
                svdat21.applyOperation("SWAP", {2, 1}, false);

                auto sv12 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat12.getView());
                auto sv21 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat21.getView());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv12[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv12[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv21[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv21[j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyCZ", "[StateVectorKokkos_Nonparam]",
                   float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                                  {{false}, {false}});

        auto ini_sv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          kokkos_sv.getView());

        auto z = ComplexT{ZERO<TestType>()};
        auto i = ComplexT{INVSQRT2<TestType>()};

        SECTION("Apply using dispatcher") {
            SECTION("CZ0,1 |+10> -> 1+0>") {
                const std::vector<ComplexT> expected_results = {z, z, i,  z,
                                                                z, z, -i, z};

                StateVectorKokkos<TestType> svdat01{num_qubits};
                StateVectorKokkos<TestType> svdat10{num_qubits};
                Kokkos::deep_copy(svdat01.getView(), ini_sv);
                Kokkos::deep_copy(svdat10.getView(), ini_sv);

                svdat01.applyOperation("CZ", {0, 1}, false);
                svdat10.applyOperation("CZ", {1, 0}, false);

                auto sv01 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat01.getView());
                auto sv10 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat10.getView());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv01[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv01[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv10[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv10[j])));
                }
            }
            SECTION("CZ0,2 |+10> -> |01+>") {
                const std::vector<ComplexT> expected_results = {z, z, i, z,
                                                                z, z, i, z};

                StateVectorKokkos<TestType> svdat02{num_qubits};
                StateVectorKokkos<TestType> svdat20{num_qubits};
                Kokkos::deep_copy(svdat02.getView(), ini_sv);
                Kokkos::deep_copy(svdat20.getView(), ini_sv);

                svdat02.applyOperation("CZ", {0, 2}, false);
                svdat20.applyOperation("CZ", {2, 0}, false);

                auto sv02 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat02.getView());
                auto sv20 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat20.getView());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv02[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv02[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv20[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv20[j])));
                }
            }
            SECTION("CZ1,2 |+10> -> |+01>") {
                const std::vector<ComplexT> expected_results = {z, z, i, z,
                                                                z, z, i, z};

                StateVectorKokkos<TestType> svdat12{num_qubits};
                StateVectorKokkos<TestType> svdat21{num_qubits};
                Kokkos::deep_copy(svdat12.getView(), ini_sv);
                Kokkos::deep_copy(svdat21.getView(), ini_sv);

                svdat12.applyOperation("CZ", {1, 2}, false);
                svdat21.applyOperation("CZ", {2, 1}, false);

                auto sv12 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat12.getView());
                auto sv21 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat21.getView());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv12[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv12[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv21[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv21[j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyToffoli",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                                  {{false}, {false}});

        auto ini_sv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          kokkos_sv.getView());

        auto z = ComplexT{ZERO<TestType>()};
        auto i = ComplexT{INVSQRT2<TestType>()};

        SECTION("Apply using dispatcher") {
            SECTION("Toffoli [0,1,2],[1,0,2] |+10> -> +1+>") {
                const std::vector<ComplexT> expected_results = {z, z, i, z,
                                                                z, z, z, i};

                StateVectorKokkos<TestType> svdat012{num_qubits};
                StateVectorKokkos<TestType> svdat102{num_qubits};
                Kokkos::deep_copy(svdat012.getView(), ini_sv);
                Kokkos::deep_copy(svdat102.getView(), ini_sv);

                svdat012.applyOperation("Toffoli", {0, 1, 2}, false);
                svdat102.applyOperation("Toffoli", {1, 0, 2}, false);

                auto sv012 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat012.getView());
                auto sv102 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat102.getView());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv012[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv012[j])));
                    CHECK(imag(expected_results[j]) == Approx(imag(sv102[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv102[j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyMultiQubitOp",
                   "[StateVectorKokkos_Nonparam][Inverse]", float, double) {
    const bool inverse = GENERATE(true, false);
    size_t num_qubits = 3;
    StateVectorKokkos<TestType> sv_normal{num_qubits};
    StateVectorKokkos<TestType> sv_mq{num_qubits};
    using UnmanagedComplexHostView =
        Kokkos::View<Kokkos::complex<TestType> *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    SECTION("Single Qubit via applyOperation") {
        auto matrix = getHadamard<Kokkos::complex, TestType>();
        std::vector<size_t> wires = {0};
        sv_normal.applyOperation("Hadamard", wires, inverse);
        auto sv_normal_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_normal.getView());

        sv_mq.applyOperation("XXXXXXXX", wires, inverse, {}, matrix);
        auto sv_mq_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_mq.getView());

        for (size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(sv_normal_host[j]) == Approx(imag(sv_mq_host[j])));
            CHECK(real(sv_normal_host[j]) == Approx(real(sv_mq_host[j])));
        }
    }

    SECTION("Single Qubit") {
        auto matrix = getHadamard<Kokkos::complex, TestType>();
        std::vector<size_t> wires = {0};
        sv_normal.applyOperation("Hadamard", wires, inverse);
        auto sv_normal_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_normal.getView());

        Kokkos::View<Kokkos::complex<TestType> *> device_matrix("device_matrix",
                                                                matrix.size());
        Kokkos::deep_copy(device_matrix, UnmanagedComplexHostView(
                                             matrix.data(), matrix.size()));
        sv_mq.applyMultiQubitOp(device_matrix, wires, inverse);
        auto sv_mq_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_mq.getView());

        for (size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(sv_normal_host[j]) == Approx(imag(sv_mq_host[j])));
            CHECK(real(sv_normal_host[j]) == Approx(real(sv_mq_host[j])));
        }
    }

    SECTION("Two Qubit") {
        auto matrix = getCNOT<Kokkos::complex, TestType>();
        std::vector<size_t> wires = {0, 1};
        sv_normal.applyOperation("CNOT", wires, inverse);
        auto sv_normal_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_normal.getView());

        Kokkos::View<Kokkos::complex<TestType> *> device_matrix("device_matrix",
                                                                matrix.size());
        Kokkos::deep_copy(device_matrix, UnmanagedComplexHostView(
                                             matrix.data(), matrix.size()));
        sv_mq.applyMultiQubitOp(device_matrix, wires, inverse);
        auto sv_mq_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_mq.getView());

        for (size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(sv_normal_host[j]) == Approx(imag(sv_mq_host[j])));
            CHECK(real(sv_normal_host[j]) == Approx(real(sv_mq_host[j])));
        }
    }

    SECTION("Three Qubit") {
        auto matrix = getToffoli<Kokkos::complex, TestType>();
        std::vector<size_t> wires = {0, 1, 2};
        sv_normal.applyOperation("Toffoli", wires, inverse);
        auto sv_normal_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_normal.getView());

        Kokkos::View<Kokkos::complex<TestType> *> device_matrix("device_matrix",
                                                                matrix.size());
        Kokkos::deep_copy(device_matrix, UnmanagedComplexHostView(
                                             matrix.data(), matrix.size()));
        sv_mq.applyMultiQubitOp(device_matrix, wires, inverse);
        auto sv_mq_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, sv_mq.getView());

        for (size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(sv_normal_host[j]) == Approx(imag(sv_mq_host[j])));
            CHECK(real(sv_normal_host[j]) == Approx(real(sv_mq_host[j])));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::applyCSWAP",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const size_t num_qubits = 3;

        StateVectorKokkos<TestType> kokkos_sv{num_qubits};

        kokkos_sv.applyOperations({{"Hadamard"}, {"PauliX"}}, {{0}, {1}},
                                  {{false}, {false}});

        auto ini_sv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          kokkos_sv.getView());

        auto z = ComplexT{ZERO<TestType>()};
        auto i = ComplexT{INVSQRT2<TestType>()};

        SECTION("Apply using dispatcher") {
            SECTION("CSWAP [0,1,2]|+10> -> |010> + |101>") {
                const std::vector<ComplexT> expected_results = {z, z, i, z,
                                                                z, i, z, z};

                StateVectorKokkos<TestType> svdat012{num_qubits};
                Kokkos::deep_copy(svdat012.getView(), ini_sv);

                svdat012.applyOperation("CSWAP", {0, 1, 2}, false);

                auto sv012 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, svdat012.getView());

                for (size_t j = 0; j < exp2(num_qubits); j++) {
                    CHECK(imag(expected_results[j]) == Approx(imag(sv012[j])));
                    CHECK(real(expected_results[j]) == Approx(real(sv012[j])));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::SetStateVector",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    using PrecisionT = TestType;
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    const size_t num_qubits = 3;

    //`values[i]` on the host will be copy the `indices[i]`th element of the
    // state vector on the device.
    SECTION("Set state vector with values and their corresponding indices on "
            "the host") {
        std::vector<ComplexT> init_state{
            ComplexT{0.267462849617, 0.010768564418},
            ComplexT{0.228575125337, 0.010564590804},
            ComplexT{0.099492751062, 0.260849833488},
            ComplexT{0.093690201640, 0.189847111702},
            ComplexT{0.015641822883, 0.225092900621},
            ComplexT{0.205574608177, 0.082808663337},
            ComplexT{0.006827173322, 0.211631480575},
            ComplexT{0.255280800811, 0.161572331669},
        };
        auto expected_state = init_state;

        for (size_t i = 0; i < exp2(num_qubits - 1); i++) {
            std::swap(expected_state[i * 2], expected_state[i * 2 + 1]);
        }

        StateVectorKokkos<PrecisionT> kokkos_sv{num_qubits};
        std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
        kokkos_sv.HostToDevice(init_state.data(), init_state.size());

        // The setStates will shuffle the state vector values on the device with
        // the following indices and values setting on host. For example, the
        // values[i] is used to set the indices[i] th element of state vector on
        // the device. For example, values[2] (init_state[5]) will be copied to
        // indices[2]th or (4th) element of the state vector.
        std::vector<size_t> indices = {0, 2, 4, 6, 1, 3, 5, 7};

        std::vector<Kokkos::complex<PrecisionT>> values = {
            init_state[1], init_state[3], init_state[5], init_state[7],
            init_state[0], init_state[2], init_state[4], init_state[6]};

        kokkos_sv.setStateVector(indices, values);

        kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

        for (size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(expected_state[j]) == Approx(imag(result_sv[j])));
            CHECK(real(expected_state[j]) == Approx(real(result_sv[j])));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::SetIthStates",
                   "[StateVectorKokkos_Nonparam]", float, double) {
    using PrecisionT = TestType;
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    const size_t num_qubits = 3;

    SECTION(
        "Set Ith element of the state state on device with data on the host") {
        std::vector<ComplexT> init_state{
            ComplexT{0.267462849617, 0.010768564418},
            ComplexT{0.228575125337, 0.010564590804},
            ComplexT{0.099492751062, 0.260849833488},
            ComplexT{0.093690201640, 0.189847111702},
            ComplexT{0.015641822883, 0.225092900621},
            ComplexT{0.205574608177, 0.082808663337},
            ComplexT{0.006827173322, 0.211631480575},
            ComplexT{0.255280800811, 0.161572331669},
        };

        std::vector<ComplexT> expected_state{
            ComplexT{0.0, 0.0}, ComplexT{0.0, 0.0}, ComplexT{0.0, 0.0},
            ComplexT{1.0, 0.0}, ComplexT{0.0, 0.0}, ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0}, ComplexT{0.0, 0.0},
        };

        StateVectorKokkos<PrecisionT> kokkos_sv{num_qubits};
        std::vector<ComplexT> result_sv(kokkos_sv.getLength(), {0, 0});
        kokkos_sv.HostToDevice(init_state.data(), init_state.size());

        size_t index = 3;

        kokkos_sv.setBasisState(index);

        kokkos_sv.DeviceToHost(result_sv.data(), result_sv.size());

        for (size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(expected_state[j]) == Approx(imag(result_sv[j])));
            CHECK(real(expected_state[j]) == Approx(real(result_sv[j])));
        }
    }
}
