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

#include "Gates.hpp"
#include "StateVectorLQubitManaged.hpp"
#include "TestHelpers.hpp" // PrecisionToName, createProductState
#include "TestHelpersWires.hpp"
#include "TestKernels.hpp"
#include "Util.hpp" // ConstMult, INVSQRT2, IMAG, ZERO

/**
 * @file Test_GateImplementations_Nonparam.cpp
 *
 * This file contains tests for non-parameterized gates. List of such gates are
 * [PauliX, PauliY, PauliZ, Hadamard, S, T, CNOT, SWAP, CZ, Toffoli, CSWAP].
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::Util;
} // namespace
/// @endcond

/**
 * @brief Run test suit only when the gate is defined
 */
#define PENNYLANE_RUN_TEST(GATE_NAME)                                          \
    template <typename PrecisionT, class GateImplementation,                   \
              typename U = void>                                               \
    struct Apply##GATE_NAME##IsDefined {                                       \
        constexpr static bool value = false;                                   \
    };                                                                         \
    template <typename PrecisionT, class GateImplementation>                   \
    struct Apply##GATE_NAME##IsDefined<                                        \
        PrecisionT, GateImplementation,                                        \
        std::enable_if_t<std::is_pointer_v<                                    \
            decltype(&GateImplementation::template apply##GATE_NAME<           \
                     PrecisionT>)>>> {                                         \
        constexpr static bool value = true;                                    \
    };                                                                         \
    template <typename PrecisionT, typename TypeList>                          \
    void testApply##GATE_NAME##ForKernels() {                                  \
        if constexpr (!std::is_same_v<TypeList, void>) {                       \
            using GateImplementation = typename TypeList::Type;                \
            if constexpr (Apply##GATE_NAME##IsDefined<                         \
                              PrecisionT, GateImplementation>::value) {        \
                testApply##GATE_NAME<PrecisionT, GateImplementation>();        \
            } else {                                                           \
                SUCCEED("Member function apply" #GATE_NAME                     \
                        " is not defined for kernel "                          \
                        << GateImplementation::name);                          \
            }                                                                  \
            testApply##GATE_NAME##ForKernels<PrecisionT,                       \
                                             typename TypeList::Next>();       \
        }                                                                      \
    }                                                                          \
    TEMPLATE_TEST_CASE("GateImplementation::apply" #GATE_NAME,                 \
                       "[GateImplementations_Nonparam]", float, double) {      \
        using PrecisionT = TestType;                                           \
        testApply##GATE_NAME##ForKernels<PrecisionT, TestKernels>();           \
    }                                                                          \
    static_assert(true, "Require semicolon")

/*******************************************************************************
 * Single-qubit gates
 ******************************************************************************/
template <typename PrecisionT, class GateImplementation>
void testApplyIdentity() {
    using ComplexT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;
    for (size_t index = 0; index < num_qubits; index++) {
        auto st_pre = createZeroState<ComplexT>(num_qubits);
        auto st_post = createZeroState<ComplexT>(num_qubits);

        GateImplementation::applyIdentity(st_pre.data(), num_qubits, {index},
                                          false);
        CHECK(std::equal(st_pre.begin(), st_pre.end(), st_post.begin()));
    }
    for (size_t index = 0; index < num_qubits; index++) {
        auto st_pre = createZeroState<ComplexT>(num_qubits);
        auto st_post = createZeroState<ComplexT>(num_qubits);
        GateImplementation::applyHadamard(st_pre.data(), num_qubits, {index},
                                          false);
        GateImplementation::applyHadamard(st_post.data(), num_qubits, {index},
                                          false);

        GateImplementation::applyIdentity(st_pre.data(), num_qubits, {index},
                                          false);
        CHECK(std::equal(st_pre.begin(), st_pre.end(), st_post.begin()));
    }
}
PENNYLANE_RUN_TEST(Identity);

template <typename PrecisionT, class GateImplementation>
void testApplyPauliX() {
    using ComplexT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;
    DYNAMIC_SECTION(GateImplementation::name
                    << ", PauliX - " << PrecisionToName<PrecisionT>::value) {
        for (size_t index = 0; index < num_qubits; index++) {
            auto st = createZeroState<ComplexT>(num_qubits);

            GateImplementation::applyPauliX(st.data(), num_qubits, {index},
                                            false);

            std::string expected_str("000");
            expected_str[index] = '1';
            REQUIRE(st == approx(createProductState<PrecisionT>(expected_str)));
        }
    }
}
PENNYLANE_RUN_TEST(PauliX);

template <typename PrecisionT, class GateImplementation>
void testApplyPauliY() {
    using ComplexT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;

    constexpr ComplexT p =
        ConstMult(static_cast<PrecisionT>(0.5),
                  ConstMult(INVSQRT2<PrecisionT>(), IMAG<PrecisionT>()));
    constexpr ComplexT m = ConstMult(-1, p);

    const std::vector<std::vector<ComplexT>> expected_results = {
        {m, m, m, m, p, p, p, p},
        {m, m, p, p, m, m, p, p},
        {m, p, m, p, m, p, m, p}};

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = createPlusState<PrecisionT>(num_qubits);

        GateImplementation::applyPauliY(st.data(), num_qubits, {index}, false);

        CHECK(st == approx(expected_results[index]));
    }
}
PENNYLANE_RUN_TEST(PauliY);

template <typename PrecisionT, class GateImplementation>
void testApplyPauliZ() {
    using ComplexT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;

    constexpr ComplexT p(static_cast<PrecisionT>(0.5) * INVSQRT2<PrecisionT>());
    constexpr ComplexT m(ConstMult(-1, p));

    const std::vector<std::vector<ComplexT>> expected_results = {
        {p, p, p, p, m, m, m, m},
        {p, p, m, m, p, p, m, m},
        {p, m, p, m, p, m, p, m}};

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = createPlusState<PrecisionT>(num_qubits);
        GateImplementation::applyPauliZ(st.data(), num_qubits, {index}, false);

        CHECK(st == approx(expected_results[index]));
    }
}
PENNYLANE_RUN_TEST(PauliZ);

template <typename PrecisionT, class GateImplementation>
void testApplyHadamard() {
    using ComplexT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;
    for (size_t index = 0; index < num_qubits; index++) {
        auto st = createZeroState<ComplexT>(num_qubits);

        GateImplementation::applyHadamard(st.data(), num_qubits, {index},
                                          false);

        std::vector<char> expected_string;
        expected_string.resize(num_qubits);
        std::fill(expected_string.begin(), expected_string.end(), '0');
        expected_string[index] = '+';
        const auto expected = createProductState<PrecisionT>(
            std::string_view{expected_string.data(), num_qubits});
        CHECK(expected == approx(st));
    }
}
PENNYLANE_RUN_TEST(Hadamard);

template <typename PrecisionT, class GateImplementation> void testApplyS() {
    using ComplexT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;

    constexpr ComplexT r(static_cast<PrecisionT>(0.5) * INVSQRT2<PrecisionT>());
    constexpr ComplexT i(ConstMult(r, IMAG<PrecisionT>()));

    const std::vector<std::vector<ComplexT>> expected_results = {
        {r, r, r, r, i, i, i, i},
        {r, r, i, i, r, r, i, i},
        {r, i, r, i, r, i, r, i}};

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = createPlusState<PrecisionT>(num_qubits);

        GateImplementation::applyS(st.data(), num_qubits, {index}, false);

        CHECK(st == approx(expected_results[index]));
    }
}
PENNYLANE_RUN_TEST(S);

template <typename PrecisionT, class GateImplementation> void testApplyT() {
    using ComplexT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;
    // Test using |+++> state

    ComplexT r(1.0 / (2.0 * std::sqrt(2)), 0);
    ComplexT i(1.0 / 4, 1.0 / 4);

    const std::vector<std::vector<ComplexT>> expected_results = {
        {r, r, r, r, i, i, i, i},
        {r, r, i, i, r, r, i, i},
        {r, i, r, i, r, i, r, i}};

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = createPlusState<PrecisionT>(num_qubits);

        GateImplementation::applyT(st.data(), num_qubits, {index}, false);

        CHECK(st == approx(expected_results[index]));
    }
}
PENNYLANE_RUN_TEST(T);
/*******************************************************************************
 * Two-qubit gates
 ******************************************************************************/

template <typename PrecisionT, class GateImplementation> void testApplyCNOT() {
    const size_t num_qubits = 3;

    SECTION("CNOT0,1 |000> = |000>") {
        const auto ini_st = createProductState<PrecisionT>("000");
        auto st = ini_st;
        GateImplementation::applyCNOT(st.data(), num_qubits, {0, 1}, false);
        CHECK(st == ini_st);
    }

    SECTION("CNOT0,1 |100> = |110>") {
        const auto ini_st = createProductState<PrecisionT>("100");
        auto st = ini_st;
        GateImplementation::applyCNOT(st.data(), num_qubits, {0, 1}, false);
        CHECK(st == approx(createProductState<PrecisionT>("110")).margin(1e-7));
    }
    SECTION("CNOT1,2 |110> = |111>") {
        const auto ini_st = createProductState<PrecisionT>("110");
        auto st = ini_st;
        GateImplementation::applyCNOT(st.data(), num_qubits, {1, 2}, false);
        CHECK(st == approx(createProductState<PrecisionT>("111")).margin(1e-7));
    }

    SECTION("Generate GHZ state") {
        auto st = createProductState<PrecisionT>("+00");

        // Test using |+00> state to generate 3-qubit GHZ state
        for (size_t index = 1; index < num_qubits; index++) {
            GateImplementation::applyCNOT(st.data(), num_qubits,
                                          {index - 1, index}, false);
        }
        CHECK(st.front() == INVSQRT2<PrecisionT>());
        CHECK(st.back() == INVSQRT2<PrecisionT>());
    }
}
PENNYLANE_RUN_TEST(CNOT);

// NOLINTNEXTLINE: Avoiding complexity errors
template <typename PrecisionT, class GateImplementation> void testApplyCY() {
    using ComplexT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;
    auto ini_st_aligned = createProductState<PrecisionT>(
        "+10"); // Test using |+10> state using AlignedAllocator
    std::vector<ComplexT> ini_st{
        ini_st_aligned.begin(),
        ini_st_aligned
            .end()}; // Converted aligned data to default vector alignment
    CHECK(ini_st == std::vector<ComplexT>{
                        ZERO<PrecisionT>(), ZERO<PrecisionT>(),
                        std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0),
                        ZERO<PrecisionT>(), ZERO<PrecisionT>(),
                        ZERO<PrecisionT>(),
                        std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0),
                        ZERO<PrecisionT>()});

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CY 0,1 |+10> -> i|100> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexT> expected{
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(0, -INVSQRT2<PrecisionT>()),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>()};

        auto sv01 = ini_st;
        GateImplementation::applyCY(sv01.data(), num_qubits, {0, 1}, false);
        CHECK(sv01 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CY 0,2 |+10> -> |010> + i |111> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexT> expected{
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0.0),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(0.0, INVSQRT2<PrecisionT>())};

        auto sv02 = ini_st;

        GateImplementation::applyCY(sv02.data(), num_qubits, {0, 2}, false);
        CHECK(sv02 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CY 1,2 |+10> -> i|+11> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexT> expected{
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(0.0, INVSQRT2<PrecisionT>()),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(0.0, INVSQRT2<PrecisionT>())};

        auto sv12 = ini_st;

        GateImplementation::applyCY(sv12.data(), num_qubits, {1, 2}, false);
        CHECK(sv12 == expected);
    }
}
PENNYLANE_RUN_TEST(CY);

// NOLINTNEXTLINE: Avoiding complexity errors
template <typename PrecisionT, class GateImplementation> void testApplyCZ() {
    using ComplexT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;

    auto ini_st_aligned = createProductState<PrecisionT>(
        "+10"); // Test using |+10> state using AlignedAllocator
    std::vector<ComplexT> ini_st{
        ini_st_aligned.begin(),
        ini_st_aligned
            .end()}; // Converted aligned data to default vector alignment
    DYNAMIC_SECTION(GateImplementation::name
                    << ", CZ0,1 |+10> -> |-10> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexT> expected{
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(-INVSQRT2<PrecisionT>(), 0),
            ZERO<PrecisionT>()};

        auto sv01 = ini_st;
        auto sv10 = ini_st;

        GateImplementation::applyCZ(sv01.data(), num_qubits, {0, 1}, false);
        GateImplementation::applyCZ(sv10.data(), num_qubits, {1, 0}, false);

        CHECK(sv01 == expected);
        CHECK(sv10 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CZ0,2 |+10> -> |+10> - "
                    << PrecisionToName<PrecisionT>::value) {
        const auto &expected = ini_st;

        auto sv02 = ini_st;
        auto sv20 = ini_st;

        GateImplementation::applyCZ(sv02.data(), num_qubits, {0, 2}, false);
        GateImplementation::applyCZ(sv20.data(), num_qubits, {2, 0}, false);

        CHECK(sv02 == expected);
        CHECK(sv20 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CZ1,2 |+10> -> |+10> - "
                    << PrecisionToName<PrecisionT>::value) {
        const auto &expected = ini_st;

        auto sv12 = ini_st;
        auto sv21 = ini_st;

        GateImplementation::applyCZ(sv12.data(), num_qubits, {1, 2}, false);
        GateImplementation::applyCZ(sv21.data(), num_qubits, {2, 1}, false);

        CHECK(sv12 == expected);
        CHECK(sv21 == expected);
    }
}
PENNYLANE_RUN_TEST(CZ);

// NOLINTNEXTLINE: Avoiding complexity errors
template <typename PrecisionT, class GateImplementation> void testApplySWAP() {
    using ComplexT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;
    auto ini_st_aligned = createProductState<PrecisionT>(
        "+10"); // Test using |+10> state using AlignedAllocator
    std::vector<ComplexT> ini_st{
        ini_st_aligned.begin(),
        ini_st_aligned
            .end()}; // Converted aligned data to default vector alignment

    CHECK(ini_st ==
          std::vector<ComplexT>{ZERO<PrecisionT>(), ZERO<PrecisionT>(),
                                INVSQRT2<PrecisionT>(), ZERO<PrecisionT>(),
                                ZERO<PrecisionT>(), ZERO<PrecisionT>(),
                                INVSQRT2<PrecisionT>(), ZERO<PrecisionT>()});

    DYNAMIC_SECTION(GateImplementation::name
                    << ", SWAP0,1 |+10> -> |1+0> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexT> expected{
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0),
            ZERO<PrecisionT>()};
        auto sv01 = ini_st;
        auto sv10 = ini_st;

        GateImplementation::applySWAP(sv01.data(), num_qubits, {0, 1}, false);
        GateImplementation::applySWAP(sv10.data(), num_qubits, {1, 0}, false);

        CHECK(sv01 == expected);
        CHECK(sv10 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", SWAP0,2 |+10> -> |01+> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexT> expected{
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0),
            std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>()};

        auto sv02 = ini_st;
        auto sv20 = ini_st;

        GateImplementation::applySWAP(sv02.data(), num_qubits, {0, 2}, false);
        GateImplementation::applySWAP(sv20.data(), num_qubits, {2, 0}, false);

        CHECK(sv02 == expected);
        CHECK(sv20 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", SWAP1,2 |+10> -> |+01> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexT> expected{
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>()};

        auto sv12 = ini_st;
        auto sv21 = ini_st;

        GateImplementation::applySWAP(sv12.data(), num_qubits, {1, 2}, false);
        GateImplementation::applySWAP(sv21.data(), num_qubits, {2, 1}, false);

        CHECK(sv12 == expected);
        CHECK(sv21 == expected);
    }
}
PENNYLANE_RUN_TEST(SWAP);

/*******************************************************************************
 * Three-qubit gates
 ******************************************************************************/
template <typename PrecisionT, class GateImplementation>
void testApplyToffoli() {
    using ComplexT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;
    auto ini_st_aligned = createProductState<PrecisionT>(
        "+10"); // Test using |+10> state using AlignedAllocator
    std::vector<ComplexT> ini_st{
        ini_st_aligned.begin(),
        ini_st_aligned
            .end()}; // Converted aligned data to default vector alignment

    // Test using |+10> state
    DYNAMIC_SECTION(GateImplementation::name
                    << ", Toffoli 0,1,2 |+10> -> |010> + |111> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexT> expected{
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0)};

        auto sv012 = ini_st;

        GateImplementation::applyToffoli(sv012.data(), num_qubits, {0, 1, 2},
                                         false);

        CHECK(sv012 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", Toffoli 1,0,2 |+10> -> |010> + |111> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexT> expected{
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0)};

        auto sv102 = ini_st;

        GateImplementation::applyToffoli(sv102.data(), num_qubits, {1, 0, 2},
                                         false);

        CHECK(sv102 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", Toffoli 0,2,1 |+10> -> |+10> - "
                    << PrecisionToName<PrecisionT>::value) {
        const auto &expected = ini_st;

        auto sv021 = ini_st;

        GateImplementation::applyToffoli(sv021.data(), num_qubits, {0, 2, 1},
                                         false);

        CHECK(sv021 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", Toffoli 1,2,0 |+10> -> |+10> - "
                    << PrecisionToName<PrecisionT>::value) {
        const auto &expected = ini_st;

        auto sv120 = ini_st;
        GateImplementation::applyToffoli(sv120.data(), num_qubits, {1, 2, 0},
                                         false);
        CHECK(sv120 == expected);
    }
}
PENNYLANE_RUN_TEST(Toffoli);

template <typename PrecisionT, class GateImplementation> void testApplyCSWAP() {
    using ComplexT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;

    auto ini_st_aligned = createProductState<PrecisionT>(
        "+10"); // Test using |+10> state using AlignedAllocator
    std::vector<ComplexT> ini_st{
        ini_st_aligned.begin(),
        ini_st_aligned
            .end()}; // Converted aligned data to default vector alignment

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CSWAP 0,1,2 |+10> -> |010> + |101> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexT> expected{
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>()};

        auto sv012 = ini_st;
        GateImplementation::applyCSWAP(sv012.data(), num_qubits, {0, 1, 2},
                                       false);
        CHECK(sv012 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CSWAP 1,0,2 |+10> -> |01+> - "
                    << PrecisionToName<PrecisionT>::value) {
        std::vector<ComplexT> expected{
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0),
            std::complex<PrecisionT>(INVSQRT2<PrecisionT>(), 0),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>(),
            ZERO<PrecisionT>()};

        auto sv102 = ini_st;
        GateImplementation::applyCSWAP(sv102.data(), num_qubits, {1, 0, 2},
                                       false);
        CHECK(sv102 == expected);
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CSWAP 2,1,0 |+10> -> |+10> - "
                    << PrecisionToName<PrecisionT>::value) {
        const auto &expected = ini_st;

        auto sv210 = ini_st;
        GateImplementation::applyCSWAP(sv210.data(), num_qubits, {2, 1, 0},
                                       false);
        CHECK(sv210 == expected);
    }
}
PENNYLANE_RUN_TEST(CSWAP);

TEMPLATE_TEST_CASE("StateVectorLQubitManaged::applyOperation non-param "
                   "one-qubit with controls",
                   "[StateVectorLQubitManaged]", float, double) {
    using PrecisionT = TestType;
    std::mt19937 re{1337};
    const int num_qubits = 4;
    const auto margin = PrecisionT{1e-5};
    const size_t control = GENERATE(0, 1, 2, 3);
    const size_t wire = GENERATE(0, 1, 2, 3);
    StateVectorLQubitManaged<PrecisionT> sv0(num_qubits);
    StateVectorLQubitManaged<PrecisionT> sv1(num_qubits);

    DYNAMIC_SECTION("N-controlled PauliX - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire) {
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            sv0.applyOperation("CNOT", {control, wire});
            sv1.applyOperation("PauliX", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }

        if (control != 0 && wire != 0) {
            sv0.applyOperation("Toffoli", {0, control, wire});
            sv1.applyOperation("PauliX", std::vector<size_t>{0, control},
                               std::vector<bool>{true, true},
                               std::vector<size_t>{wire});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));

            sv0.applyOperation("Toffoli", {control, 0, wire});
            sv1.applyOperation("PauliX", std::vector<size_t>{control, 0},
                               std::vector<bool>{true, true},
                               std::vector<size_t>{wire});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled PauliY - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire) {
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            sv0.applyOperation("CY", {control, wire});
            sv1.applyOperation("PauliY", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled PauliZ - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire) {
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            sv0.applyOperation("CZ", {control, wire});
            sv1.applyOperation("PauliZ", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled Hadamard - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire) {
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            const auto matrix = getHadamard<std::complex, PrecisionT>();

            sv0.applyControlledMatrix(matrix, std::vector<size_t>{control},
                                      std::vector<bool>{true},
                                      std::vector<size_t>{wire});
            sv1.applyOperation("Hadamard", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }
    DYNAMIC_SECTION("N-controlled S - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire) {
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            const auto matrix = getS<std::complex, PrecisionT>();

            sv0.applyControlledMatrix(matrix, std::vector<size_t>{control},
                                      std::vector<bool>{true},
                                      std::vector<size_t>{wire});
            sv1.applyOperation("S", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled T - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire) {
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            const std::vector<std::complex<PrecisionT>> matrix =
                getT<std::complex, PrecisionT>();

            sv0.applyControlledMatrix(matrix, std::vector<size_t>{control},
                                      std::vector<bool>{true},
                                      std::vector<size_t>{wire});
            sv1.applyOperation("T", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorLQubitManaged::applyOperation non-param "
                   "two-qubit with controls",
                   "[StateVectorLQubitManaged]", float, double) {
    using PrecisionT = TestType;
    std::mt19937 re{1337};
    const int num_qubits = 4;
    const auto margin = PrecisionT{1e-5};
    const size_t control = GENERATE(0, 1, 2, 3);
    const size_t wire0 = GENERATE(0, 1, 2, 3);
    const size_t wire1 = GENERATE(0, 1, 2, 3);
    StateVectorLQubitManaged<PrecisionT> sv0(num_qubits);
    StateVectorLQubitManaged<PrecisionT> sv1(num_qubits);

    DYNAMIC_SECTION("N-controlled SWAP - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);
            sv0.applyOperation("CSWAP", {control, wire0, wire1});
            sv1.applyOperation("SWAP", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire0, wire1});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled SWAP with matrix- "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);
            const std::vector<std::complex<PrecisionT>> matrix =
                getSWAP<std::complex, PrecisionT>();
            sv0.applyControlledMatrix(matrix, std::vector<size_t>{control},
                                      std::vector<bool>{true},
                                      std::vector<size_t>{wire0, wire1});
            sv1.applyOperation("SWAP", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire0, wire1});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorLQubitManaged::controlled Toffoli",
                   "[StateVectorLQubitManaged]", float, double) {
    using PrecisionT = TestType;
    std::mt19937 re{1337};
    const int num_qubits = 6;
    const auto margin = PrecisionT{1e-5};
    const size_t control = GENERATE(0, 1, 2);
    StateVectorLQubitManaged<PrecisionT> sv0(num_qubits);
    StateVectorLQubitManaged<PrecisionT> sv1(num_qubits);

    auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
    sv0.updateData(st0);
    sv1.updateData(st0);
    const std::vector<std::complex<PrecisionT>> matrix =
        getToffoli<std::complex, PrecisionT>();
    sv0.applyControlledMatrix(matrix, std::vector<size_t>{control},
                              std::vector<bool>{true},
                              std::vector<size_t>{3, 4, 5});
    sv1.applyOperation("PauliX", std::vector<size_t>{control, 3, 4},
                       std::vector<bool>{true, true, true},
                       std::vector<size_t>{5});
    REQUIRE(sv0.getDataVector() == approx(sv1.getDataVector()).margin(margin));
}