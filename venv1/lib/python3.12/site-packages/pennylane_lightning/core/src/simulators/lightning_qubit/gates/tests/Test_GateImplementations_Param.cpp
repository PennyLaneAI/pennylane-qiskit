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
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "CPUMemoryModel.hpp"
#include "Gates.hpp"
#include "StateVectorLQubitManaged.hpp"
#include "TestHelpers.hpp" // PrecisionToName, createProductState
#include "TestHelpersWires.hpp"
#include "TestKernels.hpp"
#include "Util.hpp" // INVSQRT2
#if defined(_MSC_VER)
#pragma warning(disable : 4305)
#endif

/**
 * @file This file contains tests for parameterized gates. List of such gates is
 * [RX, RY, RZ, PhaseShift, Rot, ControlledPhaseShift, CRX, CRY, CRZ, CRot]
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::Util;
using namespace Pennylane::Gates;
} // namespace
/// @endcond

/**
 * @brief Run test suit only when the gate is defined
 */
#define PENNYLANE_RUN_TEST(GATE_NAME)                                          \
    template <typename PrecisionT, typename ParamT, class GateImplementation,  \
              typename U = void>                                               \
    struct Apply##GATE_NAME##IsDefined {                                       \
        constexpr static bool value = false;                                   \
    };                                                                         \
    template <typename PrecisionT, typename ParamT, class GateImplementation>  \
    struct Apply##GATE_NAME##IsDefined<                                        \
        PrecisionT, ParamT, GateImplementation,                                \
        std::enable_if_t<std::is_pointer_v<                                    \
            decltype(&GateImplementation::template apply##GATE_NAME<           \
                     PrecisionT, ParamT>)>>> {                                 \
        constexpr static bool value = true;                                    \
    };                                                                         \
    template <typename PrecisionT, typename ParamT, typename TypeList>         \
    void testApply##GATE_NAME##ForKernels() {                                  \
        if constexpr (!std::is_same_v<TypeList, void>) {                       \
            using GateImplementation = typename TypeList::Type;                \
            if constexpr (Apply##GATE_NAME##IsDefined<                         \
                              PrecisionT, ParamT,                              \
                              GateImplementation>::value) {                    \
                testApply##GATE_NAME<PrecisionT, ParamT,                       \
                                     GateImplementation>();                    \
            } else {                                                           \
                SUCCEED("Member function apply" #GATE_NAME                     \
                        " is not defined for kernel "                          \
                        << GateImplementation::name);                          \
            }                                                                  \
            testApply##GATE_NAME##ForKernels<PrecisionT, ParamT,               \
                                             typename TypeList::Next>();       \
        }                                                                      \
    }                                                                          \
    TEMPLATE_TEST_CASE("GateImplementation::apply" #GATE_NAME,                 \
                       "[GateImplementations_Param]", float, double) {         \
        using PrecisionT = TestType;                                           \
        using ParamT = TestType;                                               \
        testApply##GATE_NAME##ForKernels<PrecisionT, ParamT, TestKernels>();   \
    }                                                                          \
    static_assert(true, "Require semicolon")

/*******************************************************************************
 * Single-qubit gates
 ******************************************************************************/

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyPhaseShift() {
    using ComplexT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;

    // Test using |+++> state
    const auto isqrt2 = PrecisionT{INVSQRT2<PrecisionT>()};
    const std::vector<PrecisionT> angles{0.3, 0.8, 2.4};
    const ComplexT coef{isqrt2 / PrecisionT{2.0}, PrecisionT{0.0}};

    std::vector<std::vector<ComplexT>> ps_data;
    ps_data.reserve(angles.size());
    for (auto &a : angles) {
        ps_data.push_back(getPhaseShift<std::complex, PrecisionT>(a));
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

    for (auto &vec : expected_results) {
        scaleVector(vec, coef);
    }

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = createPlusState<PrecisionT>(num_qubits);

        GateImplementation::applyPhaseShift(st.data(), num_qubits, {index},
                                            false, {angles[index]});

        CHECK(st == approx(expected_results[index]));
    }
}
PENNYLANE_RUN_TEST(PhaseShift);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyRX() {
    using ComplexT = std::complex<PrecisionT>;
    const size_t num_qubits = 1;

    const std::vector<PrecisionT> angles{{0.1}, {0.6}};
    std::vector<std::vector<ComplexT>> expected_results{
        std::vector<ComplexT>{{0.9987502603949663, 0.0},
                              {0.0, -0.04997916927067834}},
        std::vector<ComplexT>{{0.9553364891256061, 0.0},
                              {0, -0.2955202066613395}},
        std::vector<ComplexT>{{0.49757104789172696, 0.0},
                              {0, -0.867423225594017}}};

    for (size_t index = 0; index < angles.size(); index++) {
        auto st = createZeroState<ComplexT>(num_qubits);

        GateImplementation::applyRX(st.data(), num_qubits, {0}, false,
                                    {angles[index]});

        CHECK(st == approx(expected_results[index]).epsilon(1e-7));
    }
}
PENNYLANE_RUN_TEST(RX);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyRY() {
    using ComplexT = std::complex<PrecisionT>;
    const size_t num_qubits = 1;

    const std::vector<PrecisionT> angles{0.2, 0.7, 2.9};
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

    const TestVector<ComplexT> init_state{
        {{0.8775825618903728, 0.0}, {0.0, -0.47942553860420306}},
        getBestAllocator<ComplexT>()};
    DYNAMIC_SECTION(GateImplementation::name
                    << ", RY - " << PrecisionToName<PrecisionT>::value) {
        for (size_t index = 0; index < angles.size(); index++) {
            auto st = init_state;
            GateImplementation::applyRY(st.data(), num_qubits, {0}, false,
                                        {angles[index]});
            CHECK(st == approx(expected_results[index]).epsilon(1e-5));
        }
    }
}
PENNYLANE_RUN_TEST(RY);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyRZ() {
    using ComplexT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;

    // Test using |+++> state
    const auto isqrt2 = PrecisionT{INVSQRT2<PrecisionT>()};

    const std::vector<PrecisionT> angles{0.2, 0.7, 2.9};
    const ComplexT coef{isqrt2 / PrecisionT{2.0}, PrecisionT{0.0}};

    std::vector<std::vector<ComplexT>> rz_data;
    rz_data.reserve(angles.size());
    for (auto &a : angles) {
        rz_data.push_back(getRZ<std::complex, PrecisionT>(a));
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

    for (auto &vec : expected_results) {
        scaleVector(vec, coef);
    }

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = createPlusState<PrecisionT>(num_qubits);

        GateImplementation::applyRZ(st.data(), num_qubits, {index}, false,
                                    {angles[index]});

        CHECK(st == approx(expected_results[index]));
    }

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = createPlusState<PrecisionT>(num_qubits);

        GateImplementation::applyRZ(st.data(), num_qubits, {index}, true,
                                    {-angles[index]});
        CHECK(st == approx(expected_results[index]));
    }
}
PENNYLANE_RUN_TEST(RZ);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyRot() {
    using ComplexT = std::complex<PrecisionT>;
    const size_t num_qubits = 3;
    auto ini_st = createZeroState<ComplexT>(num_qubits);

    const std::vector<std::vector<PrecisionT>> angles{
        std::vector<PrecisionT>{0.3, 0.8, 2.4},
        std::vector<PrecisionT>{0.5, 1.1, 3.0},
        std::vector<PrecisionT>{2.3, 0.1, 0.4}};

    std::vector<std::vector<ComplexT>> expected_results{
        std::vector<ComplexT>(1U << num_qubits),
        std::vector<ComplexT>(1U << num_qubits),
        std::vector<ComplexT>(1U << num_qubits)};

    for (size_t i = 0; i < angles.size(); i++) {
        const auto rot_mat = getRot<std::complex, PrecisionT>(
            angles[i][0], angles[i][1], angles[i][2]);
        expected_results[i][0] = rot_mat[0];
        expected_results[i][size_t{1U} << (num_qubits - i - 1)] = rot_mat[2];
    }

    for (size_t index = 0; index < num_qubits; index++) {
        auto st = createZeroState<ComplexT>(num_qubits);
        GateImplementation::applyRot(st.data(), num_qubits, {index}, false,
                                     angles[index][0], angles[index][1],
                                     angles[index][2]);

        CHECK(st == approx(expected_results[index]));
    }
}
PENNYLANE_RUN_TEST(Rot);

/*******************************************************************************
 * Two-qubit gates
 ******************************************************************************/
template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyIsingXX() {
    using ComplexT = std::complex<PrecisionT>;
    using std::cos;
    using std::sin;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXX0,1 |000> -> a|000> + b|110> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createZeroState<ComplexT>(num_qubits);
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{cos(angle / 2), 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, -sin(angle / 2)},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXX(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXX0,1 |100> -> a|100> + b|010> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("100");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, -sin(angle / 2)},
            ComplexT{0.0, 0.0},
            ComplexT{cos(angle / 2), 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXX(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXX0,1 |010> -> a|010> + b|100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("010");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{cos(angle / 2), 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, -sin(angle / 2)},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXX(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXX0,1 |110> -> a|110> + b|000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("110");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, -sin(angle / 2)},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{cos(angle / 2), 0.0},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXX(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXX0,2 - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st =
            TestVector<ComplexT>{{
                                     ComplexT{0.125681356503, 0.252712197380},
                                     ComplexT{0.262591068130, 0.370189000494},
                                     ComplexT{0.129300299863, 0.371057794075},
                                     ComplexT{0.392248682814, 0.195795523118},
                                     ComplexT{0.303908059240, 0.082981563244},
                                     ComplexT{0.189140284321, 0.179512645957},
                                     ComplexT{0.173146612336, 0.092249594834},
                                     ComplexT{0.298857179897, 0.269627836165},
                                 },
                                 getBestAllocator<ComplexT>()};
        const std::vector<size_t> wires = {0, 2};
        const ParamT angle = 0.267030328057308;
        std::vector<ComplexT> expected{
            ComplexT{0.148459317603, 0.225284945157},
            ComplexT{0.271300438716, 0.326438461763},
            ComplexT{0.164042082006, 0.327971890339},
            ComplexT{0.401037861022, 0.171003883572},
            ComplexT{0.350482432141, 0.047287216587},
            ComplexT{0.221097705423, 0.161184442326},
            ComplexT{0.197669694288, 0.039212892562},
            ComplexT{0.345592157995, 0.250015865318},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXX(st.data(), num_qubits, wires, false,
                                         angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(IsingXX);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyIsingXY() {
    using ComplexT = std::complex<PrecisionT>;
    using std::cos;
    using std::sin;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXY0,1 |000> -> a|000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createZeroState<ComplexT>(num_qubits);
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{1.0, 0.0}, ComplexT{0.0, 0.0}, ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0}, ComplexT{0.0, 0.0}, ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0}, ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXY(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXY0,1 |100> -> a|100> + b|010> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("100");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, sin(angle / 2)},
            ComplexT{0.0, 0.0},
            ComplexT{cos(angle / 2), 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXY(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXY0,1 |010> -> a|010> + b|100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("010");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{cos(angle / 2), 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, sin(angle / 2)},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXY(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXY0,1 |110> -> a|110> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("110");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0}, ComplexT{0.0, 0.0}, ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0}, ComplexT{0.0, 0.0}, ComplexT{0.0, 0.0},
            ComplexT{1.0, 0.0}, ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingXY(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingXY0,1 - "
                    << PrecisionToName<PrecisionT>::value) {
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

        const std::vector<size_t> wires = {0, 1};
        const ParamT angle = 0.312;

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

        auto st = ini_st;
        GateImplementation::applyIsingXY(st.data(), num_qubits, wires, false,
                                         angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(IsingXY);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyIsingYY() {
    using ComplexT = std::complex<PrecisionT>;
    using std::cos;
    using std::sin;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingYY0,1 |000> -> a|000> + b|110> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createZeroState<ComplexT>(num_qubits);
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{cos(angle / 2), 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, sin(angle / 2)},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingYY(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingYY0,1 |100> -> a|100> + b|010> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("100");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, -sin(angle / 2)},
            ComplexT{0.0, 0.0},
            ComplexT{cos(angle / 2), 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingYY(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingYY0,1 |010> -> a|010> + b|100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("010");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{cos(angle / 2), 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, -sin(angle / 2)},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingYY(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingYY0,1 |110> -> a|110> + b|000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("110");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, sin(angle / 2)},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{cos(angle / 2), 0.0},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingYY(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingYY0,1 - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        const auto ini_st =
            TestVector<ComplexT>{{ComplexT{0.276522701942, 0.192601873155},
                                  ComplexT{0.035951282872, 0.224882549474},
                                  ComplexT{0.142578003191, 0.016769549184},
                                  ComplexT{0.207510965432, 0.068085008177},
                                  ComplexT{0.231177902264, 0.039974505646},
                                  ComplexT{0.038587049391, 0.058503643276},
                                  ComplexT{0.023121176451, 0.294843178966},
                                  ComplexT{0.297936734810, 0.061981734524},
                                  ComplexT{0.140961289031, 0.061129422308},
                                  ComplexT{0.204531438234, 0.159178277448},
                                  ComplexT{0.143828437747, 0.031972463787},
                                  ComplexT{0.291528706380, 0.138875986482},
                                  ComplexT{0.297088897520, 0.179914971203},
                                  ComplexT{0.032991360504, 0.024025500927},
                                  ComplexT{0.121553926676, 0.263606060346},
                                  ComplexT{0.177173454285, 0.267447421480}},
                                 getBestAllocator<ComplexT>()};

        const std::vector<size_t> wires = {0, 1};
        const ParamT angle = 0.312;

        std::vector<ComplexT> expected{
            ComplexT{0.245211756573, 0.236421160261},
            ComplexT{0.031781919269, 0.227277526275},
            ComplexT{0.099890674345, 0.035451505339},
            ComplexT{0.163438308608, 0.094785319724},
            ComplexT{0.237868187763, 0.017588203228},
            ComplexT{0.062849689541, 0.026015566111},
            ComplexT{0.027807906892, 0.268916455494},
            ComplexT{0.315895675672, 0.015934827233},
            ComplexT{0.145460308037, 0.024469450691},
            ComplexT{0.211137338769, 0.151250126997},
            ComplexT{0.187891084547, 0.027991919467},
            ComplexT{0.297618553419, 0.090899723116},
            ComplexT{0.263557070771, 0.220692990352},
            ComplexT{-0.002348824386, 0.029319431141},
            ComplexT{0.117472403735, 0.282557065430},
            ComplexT{0.164443742376, 0.296440286247},
        };

        auto st = ini_st;
        GateImplementation::applyIsingYY(st.data(), num_qubits, wires, false,
                                         angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(IsingYY);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyIsingZZ() {
    using ComplexT = std::complex<PrecisionT>;
    using std::cos;
    using std::sin;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingZZ0,1 |000> -> |000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createZeroState<ComplexT>(num_qubits);
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{cos(angle / 2), -sin(angle / 2)},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingZZ(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingZZ0,1 |100> -> |100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("100");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{cos(angle / 2), sin(angle / 2)},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingZZ(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingZZ0,1 |010> -> |010> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("010");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{cos(angle / 2), sin(angle / 2)},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingZZ(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingZZ0,1 |110> -> |110> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("110");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{cos(angle / 2), -sin(angle / 2)},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applyIsingZZ(st.data(), num_qubits, {0, 1}, false,
                                         angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", IsingZZ0,1 - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        TestVector<ComplexT> ini_st{{ComplexT{0.267462841882, 0.010768564798},
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
                                     ComplexT{0.292993251871, 0.186570798697}},
                                    getBestAllocator<ComplexT>()};

        const std::vector<size_t> wires = {0, 1};
        const ParamT angle = 0.312;

        std::vector<ComplexT> expected{
            ComplexT{0.265888039508, -0.030917377350},
            ComplexT{0.227440863156, -0.025076966901},
            ComplexT{0.138812299373, 0.242224241539},
            ComplexT{0.122048663851, 0.172985266764},
            ComplexT{0.001315529800, 0.206549421962},
            ComplexT{0.211505899280, 0.116123534558},
            ComplexT{0.003366392733, 0.179637932181},
            ComplexT{0.268161243812, 0.189116978698},
            ComplexT{0.133544466595, 0.134003857126},
            ComplexT{-0.006247074818, 0.120520790080},
            ComplexT{0.210247652980, 0.189627242850},
            ComplexT{0.080147179284, 0.243468334233},
            ComplexT{0.051236139067, 0.058687025978},
            ComplexT{0.122991206449, 0.204961354585},
            ComplexT{0.260499076094, 0.253870909435},
            ComplexT{0.318422472324, 0.138783420076},
        };

        auto st = ini_st;
        GateImplementation::applyIsingZZ(st.data(), num_qubits, wires, false,
                                         angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(IsingZZ);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyControlledPhaseShift() {
    using ComplexT = std::complex<PrecisionT>;

    const size_t num_qubits = 3;

    // Test using |+++> state
    auto ini_st = createPlusState<PrecisionT>(num_qubits);

    const auto isqrt2 = INVSQRT2<PrecisionT>();

    const std::vector<PrecisionT> angles{0.3, 2.4};
    const ComplexT coef{isqrt2 / PrecisionT{2.0}, PrecisionT{0.0}};

    std::vector<std::vector<ComplexT>> ps_data;
    ps_data.reserve(angles.size());
    for (auto &a : angles) {
        ps_data.push_back(getPhaseShift<std::complex, PrecisionT>(a));
    }

    std::vector<std::vector<ComplexT>> expected_results = {
        {ps_data[0][0], ps_data[0][0], ps_data[0][0], ps_data[0][0],
         ps_data[0][0], ps_data[0][0], ps_data[0][3], ps_data[0][3]},
        {ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][3],
         ps_data[1][0], ps_data[1][0], ps_data[1][0], ps_data[1][3]}};

    for (auto &vec : expected_results) {
        scaleVector(vec, coef);
    }

    auto st = ini_st;

    GateImplementation::applyControlledPhaseShift(st.data(), num_qubits, {0, 1},
                                                  false, angles[0]);
    CAPTURE(st);
    CHECK(st == approx(expected_results[0]));
}
PENNYLANE_RUN_TEST(ControlledPhaseShift);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyCRX() {
    using ComplexT = std::complex<PrecisionT>;
    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRX0,1 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexT> ini_st{
            ComplexT{0.188018120185, 0.267344585187},
            ComplexT{0.172684792903, 0.187465336044},
            ComplexT{0.218892658302, 0.241508557821},
            ComplexT{0.107094509452, 0.233123916768},
            ComplexT{0.144398681914, 0.102112687699},
            ComplexT{0.266641428689, 0.096286886834},
            ComplexT{0.037126289559, 0.047222166486},
            ComplexT{0.136865047634, 0.203178369592},
            ComplexT{0.001562711889, 0.224933454573},
            ComplexT{0.009933412610, 0.080866505038},
            ComplexT{0.000948295069, 0.280652963863},
            ComplexT{0.109817299553, 0.150776413412},
            ComplexT{0.297480913626, 0.232588348025},
            ComplexT{0.247386444054, 0.077608200535},
            ComplexT{0.192650977126, 0.054764192471},
            ComplexT{0.033093927690, 0.243038790593},
        };

        const std::vector<size_t> wires = {0, 1};
        const ParamT angle = 0.312;

        std::vector<ComplexT> expected{
            ComplexT{0.188018120185, 0.267344585187},
            ComplexT{0.172684792903, 0.187465336044},
            ComplexT{0.218892658302, 0.241508557821},
            ComplexT{0.107094509452, 0.233123916768},
            ComplexT{0.144398681914, 0.102112687699},
            ComplexT{0.266641428689, 0.096286886834},
            ComplexT{0.037126289559, 0.047222166486},
            ComplexT{0.136865047634, 0.203178369592},
            ComplexT{0.037680529583, 0.175982985869},
            ComplexT{0.021870621269, 0.041448569986},
            ComplexT{0.009445384485, 0.247313095111},
            ComplexT{0.146244209335, 0.143803745197},
            ComplexT{0.328815969263, 0.229521152393},
            ComplexT{0.256946415396, 0.075122442730},
            ComplexT{0.233916049255, 0.053951837341},
            ComplexT{0.056117891609, 0.223025389250},
        };

        auto st = ini_st;
        GateImplementation::applyCRX(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRX0,2 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexT> ini_st{
            ComplexT{0.052996853820, 0.268704529517},
            ComplexT{0.082642978242, 0.195193762273},
            ComplexT{0.275869474800, 0.221416497403},
            ComplexT{0.198695648566, 0.006071386515},
            ComplexT{0.067983147697, 0.276232498024},
            ComplexT{0.136067312263, 0.055703741794},
            ComplexT{0.157173013237, 0.279061453647},
            ComplexT{0.104219108364, 0.247711145514},
            ComplexT{0.176998514444, 0.152305581694},
            ComplexT{0.055177054767, 0.009344289143},
            ComplexT{0.047003532929, 0.014270464770},
            ComplexT{0.067602001658, 0.237978418468},
            ComplexT{0.191357285454, 0.247486891611},
            ComplexT{0.059014417923, 0.240820754268},
            ComplexT{0.017675906958, 0.280795663824},
            ComplexT{0.149294381068, 0.236647612943},
        };

        const std::vector<size_t> wires = {0, 2};
        const ParamT angle = 0.312;

        std::vector<ComplexT> expected{
            ComplexT{0.052996853820, 0.268704529517},
            ComplexT{0.082642978242, 0.195193762273},
            ComplexT{0.275869474800, 0.221416497403},
            ComplexT{0.198695648566, 0.006071386515},
            ComplexT{0.067983147697, 0.276232498024},
            ComplexT{0.136067312263, 0.055703741794},
            ComplexT{0.157173013237, 0.279061453647},
            ComplexT{0.104219108364, 0.247711145514},
            ComplexT{0.177066334766, 0.143153236251},
            ComplexT{0.091481259734, -0.001272371824},
            ComplexT{0.070096171606, -0.013402737499},
            ComplexT{0.068232891172, 0.226515814342},
            ComplexT{0.232660238337, 0.241735302419},
            ComplexT{0.095065259834, 0.214700810780},
            ComplexT{0.055912814010, 0.247655060549},
            ComplexT{0.184897295154, 0.224604965678},
        };

        auto st = ini_st;
        GateImplementation::applyCRX(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRX1,3 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexT> ini_st{
            ComplexT{0.192438300910, 0.082027221475},
            ComplexT{0.217147770013, 0.101186506864},
            ComplexT{0.172631211937, 0.036301903892},
            ComplexT{0.006532319481, 0.086171029910},
            ComplexT{0.042291498813, 0.282934641945},
            ComplexT{0.231739267944, 0.188873888944},
            ComplexT{0.278594048803, 0.306941867941},
            ComplexT{0.126901023080, 0.220266540060},
            ComplexT{0.229998291616, 0.200076737619},
            ComplexT{0.016698938983, 0.160673755090},
            ComplexT{0.123754272868, 0.123889666882},
            ComplexT{0.128913058161, 0.104905508280},
            ComplexT{0.004957334386, 0.000151477546},
            ComplexT{0.286109480550, 0.287939421742},
            ComplexT{0.180882613126, 0.180408714716},
            ComplexT{0.169404192357, 0.128550443286},
        };

        const std::vector<size_t> wires = {1, 3};
        const ParamT angle = 0.312;

        std::vector<ComplexT> expected{
            ComplexT{0.192438300910, 0.082027221475},
            ComplexT{0.217147770013, 0.101186506864},
            ComplexT{0.172631211937, 0.036301903892},
            ComplexT{0.006532319481, 0.086171029910},
            ComplexT{0.071122903322, 0.243493995118},
            ComplexT{0.272884177375, 0.180009581467},
            ComplexT{0.309433364794, 0.283498205063},
            ComplexT{0.173048974802, 0.174307158347},
            ComplexT{0.229998291616, 0.200076737619},
            ComplexT{0.016698938983, 0.160673755090},
            ComplexT{0.123754272868, 0.123889666882},
            ComplexT{0.128913058161, 0.104905508280},
            ComplexT{0.049633717487, -0.044302629247},
            ComplexT{0.282658689673, 0.283672663198},
            ComplexT{0.198658723032, 0.151897953530},
            ComplexT{0.195376806318, 0.098886035231},
        };

        auto st = ini_st;
        GateImplementation::applyCRX(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(CRX);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyCRY() {
    using ComplexT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRY0,1 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexT> ini_st{
            ComplexT{0.024509081663, 0.005606762650},
            ComplexT{0.261792037054, 0.259257414596},
            ComplexT{0.168380715455, 0.096012484887},
            ComplexT{0.169761107379, 0.042890935442},
            ComplexT{0.012169527484, 0.082631086139},
            ComplexT{0.155790166500, 0.292998574950},
            ComplexT{0.150529463310, 0.282021216715},
            ComplexT{0.097100202708, 0.134938013786},
            ComplexT{0.062640753523, 0.251735121160},
            ComplexT{0.121654204141, 0.116964600258},
            ComplexT{0.152865184550, 0.084800955456},
            ComplexT{0.300145205424, 0.101098965771},
            ComplexT{0.288274703880, 0.038180155037},
            ComplexT{0.041378441702, 0.206525491532},
            ComplexT{0.033201995261, 0.096777018650},
            ComplexT{0.303210250465, 0.300817738868},
        };

        const std::vector<size_t> wires = {0, 1};
        const ParamT angle = 0.312;

        std::vector<ComplexT> expected{
            ComplexT{0.024509081663, 0.005606762650},
            ComplexT{0.261792037054, 0.259257414596},
            ComplexT{0.168380715455, 0.096012484887},
            ComplexT{0.169761107379, 0.042890935442},
            ComplexT{0.012169527484, 0.082631086139},
            ComplexT{0.155790166500, 0.292998574950},
            ComplexT{0.150529463310, 0.282021216715},
            ComplexT{0.097100202708, 0.134938013786},
            ComplexT{0.017091411508, 0.242746239557},
            ComplexT{0.113748028260, 0.083456799483},
            ComplexT{0.145850361424, 0.068735133269},
            ComplexT{0.249391258812, 0.053133825802},
            ComplexT{0.294506455875, 0.076828111036},
            ComplexT{0.059777143539, 0.222190141515},
            ComplexT{0.056549175144, 0.108777179774},
            ComplexT{0.346161234622, 0.312872353290},
        };

        auto st = ini_st;
        GateImplementation::applyCRY(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRY0,2 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexT> ini_st{
            ComplexT{0.102619838050, 0.054477528511},
            ComplexT{0.202715827962, 0.019268690848},
            ComplexT{0.009985085718, 0.046864154650},
            ComplexT{0.095353410397, 0.178365407785},
            ComplexT{0.265491448756, 0.075474015573},
            ComplexT{0.155542525434, 0.336145304405},
            ComplexT{0.264473386058, 0.073102790542},
            ComplexT{0.275654487087, 0.027356694914},
            ComplexT{0.040156237615, 0.323407814320},
            ComplexT{0.111584643322, 0.148005654537},
            ComplexT{0.143440399478, 0.139829784016},
            ComplexT{0.104105862006, 0.036845342185},
            ComplexT{0.254859090295, 0.077839069459},
            ComplexT{0.166580751989, 0.081673415646},
            ComplexT{0.322693919290, 0.244062536913},
            ComplexT{0.203101217204, 0.182142660415},
        };

        const std::vector<size_t> wires = {0, 2};
        const ParamT angle = 0.312;

        std::vector<ComplexT> expected{
            ComplexT{0.102619838050, 0.054477528511},
            ComplexT{0.202715827962, 0.019268690848},
            ComplexT{0.009985085718, 0.046864154650},
            ComplexT{0.095353410397, 0.178365407785},
            ComplexT{0.265491448756, 0.075474015573},
            ComplexT{0.155542525434, 0.336145304405},
            ComplexT{0.264473386058, 0.073102790542},
            ComplexT{0.275654487087, 0.027356694914},
            ComplexT{0.017382553849, 0.297755483640},
            ComplexT{0.094054909639, 0.140483782705},
            ComplexT{0.147937549133, 0.188379019063},
            ComplexT{0.120178355382, 0.059393264033},
            ComplexT{0.201627929216, 0.038974326513},
            ComplexT{0.133002468018, 0.052382480362},
            ComplexT{0.358372291916, 0.253192504889},
            ComplexT{0.226516213248, 0.192620277535},
        };

        auto st = ini_st;
        GateImplementation::applyCRY(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRY1,3 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexT> ini_st{
            ComplexT{0.058899496683, 0.031397556785},
            ComplexT{0.069961513798, 0.130434904124},
            ComplexT{0.217689437802, 0.274984586300},
            ComplexT{0.306390652950, 0.298990481245},
            ComplexT{0.209944539032, 0.220900665872},
            ComplexT{0.003587823096, 0.069341448987},
            ComplexT{0.114578641694, 0.136714993752},
            ComplexT{0.131460200149, 0.288466810023},
            ComplexT{0.153891247725, 0.128222510215},
            ComplexT{0.161391493466, 0.264248676428},
            ComplexT{0.102366240850, 0.123871730768},
            ComplexT{0.094155009506, 0.178235083697},
            ComplexT{0.137480035766, 0.038860712805},
            ComplexT{0.181542539134, 0.186931324992},
            ComplexT{0.130801257167, 0.165524479895},
            ComplexT{0.303475658073, 0.099907724058},
        };

        const std::vector<size_t> wires = {1, 3};
        const ParamT angle = 0.312;

        std::vector<ComplexT> expected{
            ComplexT{0.058899496683, 0.031397556785},
            ComplexT{0.069961513798, 0.130434904124},
            ComplexT{0.217689437802, 0.274984586300},
            ComplexT{0.306390652950, 0.298990481245},
            ComplexT{0.206837677400, 0.207444748683},
            ComplexT{0.036162925095, 0.102820314015},
            ComplexT{0.092762561137, 0.090236295654},
            ComplexT{0.147665692045, 0.306204998241},
            ComplexT{0.153891247725, 0.128222510215},
            ComplexT{0.161391493466, 0.264248676428},
            ComplexT{0.102366240850, 0.123871730768},
            ComplexT{0.094155009506, 0.178235083697},
            ComplexT{0.107604661198, 0.009345661471},
            ComplexT{0.200698008554, 0.190699066265},
            ComplexT{0.082062476397, 0.147991992696},
            ComplexT{0.320112783074, 0.124411723198},
        };

        auto st = ini_st;
        GateImplementation::applyCRY(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}

PENNYLANE_RUN_TEST(CRY);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyCRZ() {
    using ComplexT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRZ0,1 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexT> ini_st{
            ComplexT{0.264968228755, 0.059389110312},
            ComplexT{0.004927738580, 0.117198819444},
            ComplexT{0.192517901751, 0.061524928233},
            ComplexT{0.285160768924, 0.013212111581},
            ComplexT{0.278645646186, 0.212116779981},
            ComplexT{0.171786665640, 0.141260537212},
            ComplexT{0.199480649113, 0.218261452113},
            ComplexT{0.071007710848, 0.294720535623},
            ComplexT{0.169589173252, 0.010528306669},
            ComplexT{0.061973371011, 0.033143783035},
            ComplexT{0.177570977662, 0.116785656786},
            ComplexT{0.070266502325, 0.084338553411},
            ComplexT{0.053744021753, 0.146932844792},
            ComplexT{0.254428637803, 0.138916780809},
            ComplexT{0.260354050166, 0.267004004472},
            ComplexT{0.008910554792, 0.316282675508},
        };

        const std::vector<size_t> wires = {0, 1};
        const ParamT angle = 0.312;

        std::vector<ComplexT> expected{
            ComplexT{0.264968228755, 0.059389110312},
            ComplexT{0.004927738580, 0.117198819444},
            ComplexT{0.192517901751, 0.061524928233},
            ComplexT{0.285160768924, 0.013212111581},
            ComplexT{0.278645646186, 0.212116779981},
            ComplexT{0.171786665640, 0.141260537212},
            ComplexT{0.199480649113, 0.218261452113},
            ComplexT{0.071007710848, 0.294720535623},
            ComplexT{0.169165556003, -0.015948278519},
            ComplexT{0.066370291483, 0.023112625918},
            ComplexT{0.193559430151, 0.087778634862},
            ComplexT{0.082516747253, 0.072397233118},
            ComplexT{0.030262722499, 0.153498691785},
            ComplexT{0.229755796458, 0.176759943762},
            ComplexT{0.215708594452, 0.304212379961},
            ComplexT{-0.040337866447, 0.313826361773},
        };

        auto st = ini_st;
        GateImplementation::applyCRZ(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRZ0,2 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexT> ini_st{
            ComplexT{0.148770394604, 0.083378238599},
            ComplexT{0.274356796683, 0.083823071640},
            ComplexT{0.028016616540, 0.165919229565},
            ComplexT{0.123329104424, 0.295826835858},
            ComplexT{0.222343815006, 0.093160444663},
            ComplexT{0.288857659956, 0.138646598905},
            ComplexT{0.199272938656, 0.123099916175},
            ComplexT{0.182062963782, 0.098622669183},
            ComplexT{0.270467177482, 0.282942493365},
            ComplexT{0.147717133688, 0.038580110182},
            ComplexT{0.279040367487, 0.114344708857},
            ComplexT{0.229917326705, 0.222777886314},
            ComplexT{0.047595071834, 0.026542458656},
            ComplexT{0.133654136834, 0.275281854777},
            ComplexT{0.126723771272, 0.071649311030},
            ComplexT{0.040467231551, 0.098358909396},
        };

        const std::vector<size_t> wires = {0, 2};
        const ParamT angle = 0.312;

        std::vector<ComplexT> expected{
            ComplexT{0.148770394604, 0.083378238599},
            ComplexT{0.274356796683, 0.083823071640},
            ComplexT{0.028016616540, 0.165919229565},
            ComplexT{0.123329104424, 0.295826835858},
            ComplexT{0.222343815006, 0.093160444663},
            ComplexT{0.288857659956, 0.138646598905},
            ComplexT{0.199272938656, 0.123099916175},
            ComplexT{0.182062963782, 0.098622669183},
            ComplexT{0.311143020471, 0.237484672050},
            ComplexT{0.151917469671, 0.015161098089},
            ComplexT{0.257886371956, 0.156310134957},
            ComplexT{0.192512799579, 0.255794420869},
            ComplexT{0.051140958142, 0.018825391755},
            ComplexT{0.174801129192, 0.251173432304},
            ComplexT{0.114052908458, 0.090468071985},
            ComplexT{0.024693993739, 0.103451817578},
        };

        auto st = ini_st;
        GateImplementation::applyCRZ(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRZ1,3 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexT> ini_st{
            ComplexT{0.190769680625, 0.287992363388},
            ComplexT{0.098068639739, 0.098569855389},
            ComplexT{0.037728060139, 0.188330976218},
            ComplexT{0.091809561053, 0.200107659880},
            ComplexT{0.299856248683, 0.162326250675},
            ComplexT{0.064700651300, 0.038667789709},
            ComplexT{0.119630787356, 0.257575730461},
            ComplexT{0.061392768321, 0.055938727834},
            ComplexT{0.052661991695, 0.274401532393},
            ComplexT{0.238974614805, 0.213527036406},
            ComplexT{0.163750665141, 0.107235582319},
            ComplexT{0.260992375359, 0.008326988206},
            ComplexT{0.240406501616, 0.032737802983},
            ComplexT{0.152754313527, 0.107245249982},
            ComplexT{0.162638949527, 0.306372397719},
            ComplexT{0.231663044710, 0.107293515032},
        };

        const std::vector<size_t> wires = {1, 3};
        const ParamT angle = 0.312;

        std::vector<ComplexT> expected{
            ComplexT{0.190769680625, 0.287992363388},
            ComplexT{0.098068639739, 0.098569855389},
            ComplexT{0.037728060139, 0.188330976218},
            ComplexT{0.091809561053, 0.200107659880},
            ComplexT{0.321435301661, 0.113766991605},
            ComplexT{0.057907230634, 0.048250646420},
            ComplexT{0.158197104346, 0.235861099766},
            ComplexT{0.051956164721, 0.064797918341},
            ComplexT{0.052661991695, 0.274401532393},
            ComplexT{0.238974614805, 0.213527036406},
            ComplexT{0.163750665141, 0.107235582319},
            ComplexT{0.260992375359, 0.008326988206},
            ComplexT{0.242573571004, -0.005011228787},
            ComplexT{0.134236881868, 0.129676071390},
            ComplexT{0.208264445871, 0.277383118761},
            ComplexT{0.212179898392, 0.141983644728},
        };

        auto st = ini_st;
        GateImplementation::applyCRZ(st.data(), num_qubits, wires, false,
                                     angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(CRZ);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyCRot() {
    using ComplexT = std::complex<PrecisionT>;

    const std::vector<PrecisionT> angles{0.3, 0.8, 2.4};

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRot0,1 |000> -> |000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createZeroState<ComplexT>(num_qubits);

        auto st = createZeroState<ComplexT>(num_qubits);
        GateImplementation::applyCRot(st.data(), num_qubits, {0, 1}, false,
                                      angles[0], angles[1], angles[2]);

        CHECK(st == approx(ini_st));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRot0,1 |100> -> |1>(a|0>+b|1>)|0> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;

        auto st = createZeroState<ComplexT>(num_qubits);

        std::vector<ComplexT> expected_results(8);
        const auto rot_mat =
            getRot<std::complex, PrecisionT>(angles[0], angles[1], angles[2]);
        expected_results[size_t{1U} << (num_qubits - 1)] = rot_mat[0];
        expected_results[(size_t{1U} << num_qubits) - 2] = rot_mat[2];

        GateImplementation::applyPauliX(st.data(), num_qubits, {0}, false);

        GateImplementation::applyCRot(st.data(), num_qubits, {0, 1}, false,
                                      angles[0], angles[1], angles[2]);

        CHECK(st == approx(expected_results));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", CRot0,1 - " << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;

        std::vector<ComplexT> ini_st{
            ComplexT{0.234734234199, 0.088957328814},
            ComplexT{0.065109443398, 0.284054307559},
            ComplexT{0.272603451516, 0.101758170511},
            ComplexT{0.049922391489, 0.280849666080},
            ComplexT{0.012676439023, 0.283581988298},
            ComplexT{0.074837215146, 0.119865583718},
            ComplexT{0.220666349215, 0.083019197512},
            ComplexT{0.228645004012, 0.109144153614},
            ComplexT{0.186515011731, 0.009044330588},
            ComplexT{0.268705684298, 0.278878779206},
            ComplexT{0.007225255939, 0.104466710409},
            ComplexT{0.092186772555, 0.167323294042},
            ComplexT{0.198642540305, 0.317101356672},
            ComplexT{0.061416756317, 0.014463767792},
            ComplexT{0.109767506116, 0.244842265274},
            ComplexT{0.044108879936, 0.124327196075},
        };

        const std::vector<size_t> wires = {0, 1};
        const ParamT phi = 0.128;
        const ParamT theta = -0.563;
        const ParamT omega = 1.414;

        std::vector<ComplexT> expected{
            ComplexT{0.234734234199, 0.088957328814},
            ComplexT{0.065109443398, 0.284054307559},
            ComplexT{0.272603451516, 0.101758170511},
            ComplexT{0.049922391489, 0.280849666080},
            ComplexT{0.012676439023, 0.283581988298},
            ComplexT{0.074837215146, 0.119865583718},
            ComplexT{0.220666349215, 0.083019197512},
            ComplexT{0.228645004012, 0.109144153614},
            ComplexT{0.231541411002, -0.081215269214},
            ComplexT{0.387885772871, 0.005250582985},
            ComplexT{0.140096879751, 0.103289147066},
            ComplexT{0.206040689190, 0.073864544104},
            ComplexT{-0.115373527531, 0.318376165756},
            ComplexT{0.019345803102, -0.055678858513},
            ComplexT{-0.072480957773, 0.217744954736},
            ComplexT{-0.045461901445, 0.062632338099},
        };

        auto st = ini_st;
        GateImplementation::applyCRot(st.data(), num_qubits, wires, false, phi,
                                      theta, omega);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(CRot);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplySingleExcitation() {
    using ComplexT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitation0,1 |000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createZeroState<ComplexT>(num_qubits);
        ParamT angle = 0.312;
        auto st = ini_st;
        GateImplementation::applySingleExcitation(st.data(), num_qubits, {0, 1},
                                                  false, angle);
        CHECK(st == approx(ini_st));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitation0,1 |100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("100");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0},           ComplexT{0.0, 0.0},
            ComplexT{-0.1553680335, 0.0}, ComplexT{0.0, 0.0},
            ComplexT{0.9878566567, 0.0},  ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},           ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitation(st.data(), num_qubits, {0, 1},
                                                  false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitation0,1 |010> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("010");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0},          ComplexT{0.0, 0.0},
            ComplexT{0.9878566567, 0.0}, ComplexT{0.0, 0.0},
            ComplexT{0.1553680335, 0.0}, ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},          ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitation(st.data(), num_qubits, {0, 1},
                                                  false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitation0,1 |110> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("110");
        ParamT angle = 0.312;

        auto st = ini_st;
        GateImplementation::applySingleExcitation(st.data(), num_qubits, {0, 1},
                                                  false, angle);
        CHECK(st == approx(ini_st));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitation0,1 - "
                    << PrecisionToName<PrecisionT>::value) {
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
        const std::vector<size_t> wires = {0, 2};
        const ParamT angle = 0.267030328057308;
        std::vector<ComplexT> expected{
            ComplexT{0.125681, 0.252712}, ComplexT{0.219798, 0.355848},
            ComplexT{0.1293, 0.371058},   ComplexT{0.365709, 0.181773},
            ComplexT{0.336159, 0.131522}, ComplexT{0.18914, 0.179513},
            ComplexT{0.223821, 0.117493}, ComplexT{0.298857, 0.269628},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitation(st.data(), num_qubits, wires,
                                                  false, angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(SingleExcitation);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplySingleExcitationMinus() {
    using ComplexT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationMinus0,1 |000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createZeroState<ComplexT>(num_qubits);
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.9878566567, -0.1553680335},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitationMinus(st.data(), num_qubits,
                                                       {0, 1}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationMinus0,1 |100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("100");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0},           ComplexT{0.0, 0.0},
            ComplexT{-0.1553680335, 0.0}, ComplexT{0.0, 0.0},
            ComplexT{0.9878566567, 0.0},  ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},           ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitationMinus(st.data(), num_qubits,
                                                       {0, 1}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationMinus0,1 |010> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("010");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0},          ComplexT{0.0, 0.0},
            ComplexT{0.9878566567, 0.0}, ComplexT{0.0, 0.0},
            ComplexT{0.1553680335, 0.0}, ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},          ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitationMinus(st.data(), num_qubits,
                                                       {0, 1}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationMinus0,1 |110> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("110");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.9878566567, -0.1553680335},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitationMinus(st.data(), num_qubits,
                                                       {0, 1}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationMinus0,1 - "
                    << PrecisionToName<PrecisionT>::value) {
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
        const std::vector<size_t> wires = {0, 2};
        const ParamT angle = 0.267030328057308;
        std::vector<ComplexT> expected{
            ComplexT{0.158204, 0.233733}, ComplexT{0.219798, 0.355848},
            ComplexT{0.177544, 0.350543}, ComplexT{0.365709, 0.181773},
            ComplexT{0.336159, 0.131522}, ComplexT{0.211353, 0.152737},
            ComplexT{0.223821, 0.117493}, ComplexT{0.33209, 0.227445}};

        auto st = ini_st;
        GateImplementation::applySingleExcitationMinus(st.data(), num_qubits,
                                                       wires, false, angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(SingleExcitationMinus);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplySingleExcitationPlus() {
    using ComplexT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationPlus0,1 |000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createZeroState<ComplexT>(num_qubits);
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.9878566567, 0.1553680335},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitationPlus(st.data(), num_qubits,
                                                      {0, 1}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationPlus0,1 |100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("100");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0},           ComplexT{0.0, 0.0},
            ComplexT{-0.1553680335, 0.0}, ComplexT{0.0, 0.0},
            ComplexT{0.9878566567, 0.0},  ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},           ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitationPlus(st.data(), num_qubits,
                                                      {0, 1}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationPlus0,1 |010> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("010");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0},          ComplexT{0.0, 0.0},
            ComplexT{0.9878566567, 0.0}, ComplexT{0.0, 0.0},
            ComplexT{0.1553680335, 0.0}, ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},          ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitationPlus(st.data(), num_qubits,
                                                      {0, 1}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationPlus0,1 |110> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 3;
        const auto ini_st = createProductState<PrecisionT>("110");
        ParamT angle = 0.312;

        const std::vector<ComplexT> expected_results{
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.0, 0.0},
            ComplexT{0.9878566567, 0.1553680335},
            ComplexT{0.0, 0.0},
        };

        auto st = ini_st;
        GateImplementation::applySingleExcitationPlus(st.data(), num_qubits,
                                                      {0, 1}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", SingleExcitationPlus0,1 - "
                    << PrecisionToName<PrecisionT>::value) {
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
        const std::vector<size_t> wires = {0, 2};
        const ParamT angle = 0.267030328057308;
        std::vector<ComplexT> expected{
            ComplexT{0.090922, 0.267194},  ComplexT{0.219798, 0.355848},
            ComplexT{0.0787548, 0.384968}, ComplexT{0.365709, 0.181773},
            ComplexT{0.336159, 0.131522},  ComplexT{0.16356, 0.203093},
            ComplexT{0.223821, 0.117493},  ComplexT{0.260305, 0.307012}};

        auto st = ini_st;
        GateImplementation::applySingleExcitationPlus(st.data(), num_qubits,
                                                      wires, false, angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(SingleExcitationPlus);

/*******************************************************************************
 * Four-qubit gates
 ******************************************************************************/
template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyDoubleExcitation() {
    using ComplexT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitation0,1,2,3 |0000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createZeroState<ComplexT>(num_qubits);
        ParamT angle = 0.312;
        auto st = ini_st;
        GateImplementation::applyDoubleExcitation(st.data(), num_qubits,
                                                  {0, 1, 2, 3}, false, angle);
        CHECK(st == approx(ini_st));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitation0,1,2,3 |1100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createProductState<PrecisionT>("1100");
        ParamT angle = 0.312;

        std::vector<ComplexT> expected_results(16, ComplexT{});
        expected_results[3] = ComplexT{-0.1553680335, 0};
        expected_results[12] = ComplexT{0.9878566566949545, 0};

        auto st = ini_st;
        GateImplementation::applyDoubleExcitation(st.data(), num_qubits,
                                                  {0, 1, 2, 3}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitation0,1,2,3 |0011> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createProductState<PrecisionT>("0011");
        ParamT angle = 0.312;

        std::vector<ComplexT> expected_results(16, ComplexT{});
        expected_results[3] = ComplexT{0.9878566566949545, 0};
        expected_results[12] = ComplexT{0.15536803346720587, 0};

        auto st = ini_st;
        GateImplementation::applyDoubleExcitation(st.data(), num_qubits,
                                                  {0, 1, 2, 3}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitation0,1,2,3 - "
                    << PrecisionToName<PrecisionT>::value) {
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
        const std::vector<size_t> wires = {0, 1, 2, 3};
        const ParamT angle = 0.267030328057308;
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

        auto st = ini_st;
        GateImplementation::applyDoubleExcitation(st.data(), num_qubits, wires,
                                                  false, angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(DoubleExcitation);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyDoubleExcitationMinus() {
    using ComplexT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitationMinus0,1,2,3 |0000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createZeroState<ComplexT>(num_qubits);
        ParamT angle = 0.312;

        std::vector<ComplexT> expected_results(16, ComplexT{});
        expected_results[0] =
            ComplexT{0.9878566566949545, -0.15536803346720587};

        auto st = ini_st;
        GateImplementation::applyDoubleExcitationMinus(
            st.data(), num_qubits, {0, 1, 2, 3}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitationMinus0,1,2,3 |1100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createProductState<PrecisionT>("1100");
        ParamT angle = 0.312;

        std::vector<ComplexT> expected_results(16, ComplexT{});
        expected_results[3] = ComplexT{-0.1553680335, 0};
        expected_results[12] = ComplexT{0.9878566566949545, 0};

        auto st = ini_st;
        GateImplementation::applyDoubleExcitationMinus(
            st.data(), num_qubits, {0, 1, 2, 3}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitationMinus0,1,2,3 |0011> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createProductState<PrecisionT>("0011");
        ParamT angle = 0.312;

        std::vector<ComplexT> expected_results(16, ComplexT{});
        expected_results[3] = ComplexT{0.9878566566949545, 0};
        expected_results[12] = ComplexT{0.15536803346720587, 0};

        auto st = ini_st;
        GateImplementation::applyDoubleExcitationMinus(
            st.data(), num_qubits, {0, 1, 2, 3}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitationMinus0,1,2,3 - "
                    << PrecisionToName<PrecisionT>::value) {
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
        const std::vector<size_t> wires = {0, 1, 2, 3};
        const ParamT angle = 0.267030328057308;
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

        auto st = ini_st;
        GateImplementation::applyDoubleExcitationMinus(st.data(), num_qubits,
                                                       wires, false, angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(DoubleExcitationMinus);

template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyDoubleExcitationPlus() {
    using ComplexT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitationPlus0,1,2,3 |0000> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createZeroState<ComplexT>(num_qubits);
        ParamT angle = 0.312;

        std::vector<ComplexT> expected_results(16, ComplexT{});
        expected_results[0] = ComplexT{0.9878566566949545, 0.15536803346720587};

        auto st = ini_st;
        GateImplementation::applyDoubleExcitationPlus(
            st.data(), num_qubits, {0, 1, 2, 3}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitationPlus0,1,2,3 |1100> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createProductState<PrecisionT>("1100");
        ParamT angle = 0.312;

        std::vector<ComplexT> expected_results(16, ComplexT{});
        expected_results[3] = ComplexT{-0.1553680335, 0};
        expected_results[12] = ComplexT{0.9878566566949545, 0};

        auto st = ini_st;
        GateImplementation::applyDoubleExcitationPlus(
            st.data(), num_qubits, {0, 1, 2, 3}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitationPlus0,1,2,3 |0011> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const auto ini_st = createProductState<PrecisionT>("0011");
        ParamT angle = 0.312;

        std::vector<ComplexT> expected_results(16, ComplexT{});
        expected_results[3] = ComplexT{0.9878566566949545, 0};
        expected_results[12] = ComplexT{0.15536803346720587, 0};

        auto st = ini_st;
        GateImplementation::applyDoubleExcitationPlus(
            st.data(), num_qubits, {0, 1, 2, 3}, false, angle);
        REQUIRE(st == approx(expected_results).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", DoubleExcitationPlus0,1,2,3 - "
                    << PrecisionToName<PrecisionT>::value) {
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
        const std::vector<size_t> wires = {0, 1, 2, 3};
        const ParamT angle = 0.267030328057308;
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

        auto st = ini_st;
        GateImplementation::applyDoubleExcitationPlus(st.data(), num_qubits,
                                                      wires, false, angle);
        REQUIRE(st == approx(expected).margin(1e-5));
    }
}
PENNYLANE_RUN_TEST(DoubleExcitationPlus);

/*******************************************************************************
 * Multi-qubit gates
 ******************************************************************************/
template <typename PrecisionT, typename ParamT, class GateImplementation>
void testApplyMultiRZ() {
    using ComplexT = std::complex<PrecisionT>;

    DYNAMIC_SECTION(GateImplementation::name
                    << ", MultiRZ0 |++++> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const ParamT angle = M_PI;
        auto st = createPlusState<PrecisionT>(num_qubits);

        std::vector<ComplexT> expected{
            ComplexT{0, -0.25}, ComplexT{0, -0.25}, ComplexT{0, -0.25},
            ComplexT{0, -0.25}, ComplexT{0, -0.25}, ComplexT{0, -0.25},
            ComplexT{0, -0.25}, ComplexT{0, -0.25}, ComplexT{0, +0.25},
            ComplexT{0, +0.25}, ComplexT{0, +0.25}, ComplexT{0, +0.25},
            ComplexT{0, +0.25}, ComplexT{0, +0.25}, ComplexT{0, +0.25},
            ComplexT{0, +0.25},
        };

        GateImplementation::applyMultiRZ(st.data(), num_qubits, {0}, false,
                                         angle);

        REQUIRE(st == approx(expected).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", MultiRZ0 |++++> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const ParamT angle = M_PI;
        auto st = createPlusState<PrecisionT>(num_qubits);

        std::vector<ComplexT> expected{
            ComplexT{0, -0.25}, ComplexT{0, -0.25}, ComplexT{0, -0.25},
            ComplexT{0, -0.25}, ComplexT{0, -0.25}, ComplexT{0, -0.25},
            ComplexT{0, -0.25}, ComplexT{0, -0.25}, ComplexT{0, +0.25},
            ComplexT{0, +0.25}, ComplexT{0, +0.25}, ComplexT{0, +0.25},
            ComplexT{0, +0.25}, ComplexT{0, +0.25}, ComplexT{0, +0.25},
            ComplexT{0, +0.25},
        };

        GateImplementation::applyMultiRZ(st.data(), num_qubits, {0}, false,
                                         angle);

        REQUIRE(st == approx(expected).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", MultiRZ01 |++++> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const ParamT angle = M_PI;
        auto st = createPlusState<PrecisionT>(num_qubits);

        std::vector<ComplexT> expected{
            ComplexT{0, -0.25}, ComplexT{0, -0.25}, ComplexT{0, -0.25},
            ComplexT{0, -0.25}, ComplexT{0, +0.25}, ComplexT{0, +0.25},
            ComplexT{0, +0.25}, ComplexT{0, +0.25}, ComplexT{0, +0.25},
            ComplexT{0, +0.25}, ComplexT{0, +0.25}, ComplexT{0, +0.25},
            ComplexT{0, -0.25}, ComplexT{0, -0.25}, ComplexT{0, -0.25},
            ComplexT{0, -0.25},
        };

        GateImplementation::applyMultiRZ(st.data(), num_qubits, {0, 1}, false,
                                         angle);

        REQUIRE(st == approx(expected).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", MultiRZ012 |++++> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const ParamT angle = M_PI;
        auto st = createPlusState<PrecisionT>(num_qubits);

        std::vector<ComplexT> expected{
            ComplexT{0, -0.25}, ComplexT{0, -0.25}, ComplexT{0, +0.25},
            ComplexT{0, +0.25}, ComplexT{0, +0.25}, ComplexT{0, +0.25},
            ComplexT{0, -0.25}, ComplexT{0, -0.25}, ComplexT{0, +0.25},
            ComplexT{0, +0.25}, ComplexT{0, -0.25}, ComplexT{0, -0.25},
            ComplexT{0, -0.25}, ComplexT{0, -0.25}, ComplexT{0, +0.25},
            ComplexT{0, +0.25},
        };

        GateImplementation::applyMultiRZ(st.data(), num_qubits, {0, 1, 2},
                                         false, angle);

        REQUIRE(st == approx(expected).margin(1e-7));
    }
    DYNAMIC_SECTION(GateImplementation::name
                    << ", MultiRZ0123 |++++> - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        const ParamT angle = M_PI;
        auto st = createPlusState<PrecisionT>(num_qubits);

        std::vector<ComplexT> expected{
            ComplexT{0, -0.25}, ComplexT{0, +0.25}, ComplexT{0, +0.25},
            ComplexT{0, -0.25}, ComplexT{0, +0.25}, ComplexT{0, -0.25},
            ComplexT{0, -0.25}, ComplexT{0, +0.25}, ComplexT{0, +0.25},
            ComplexT{0, -0.25}, ComplexT{0, -0.25}, ComplexT{0, +0.25},
            ComplexT{0, -0.25}, ComplexT{0, +0.25}, ComplexT{0, +0.25},
            ComplexT{0, -0.25},
        };

        GateImplementation::applyMultiRZ(st.data(), num_qubits, {0, 1, 2, 3},
                                         false, angle);

        REQUIRE(st == approx(expected).margin(1e-7));
    }

    DYNAMIC_SECTION(GateImplementation::name
                    << ", MultiRZ013 - "
                    << PrecisionToName<PrecisionT>::value) {
        const size_t num_qubits = 4;
        std::vector<ComplexT> ini_st{
            ComplexT{0.029963367200, 0.181037777550},
            ComplexT{0.070992796807, 0.263183826811},
            ComplexT{0.086883003918, 0.090811332201},
            ComplexT{0.156989157753, 0.153911449950},
            ComplexT{0.193120178047, 0.257383787598},
            ComplexT{0.262262890778, 0.163282579388},
            ComplexT{0.110853627976, 0.247870990381},
            ComplexT{0.202098107411, 0.160525183734},
            ComplexT{0.025750679341, 0.172601520950},
            ComplexT{0.235737282225, 0.008347360496},
            ComplexT{0.085757778150, 0.248516366527},
            ComplexT{0.047549845173, 0.223003660220},
            ComplexT{0.086414423346, 0.250866254986},
            ComplexT{0.112429154107, 0.111787742027},
            ComplexT{0.240562329064, 0.010449374903},
            ComplexT{0.267984502939, 0.236708607552},
        };
        const std::vector<size_t> wires = {0, 1, 3};
        const ParamT angle = 0.6746272767672288;
        std::vector<ComplexT> expected{
            ComplexT{0.088189897518, 0.160919303534},
            ComplexT{-0.020109410195, 0.271847963971},
            ComplexT{0.112041208417, 0.056939635075},
            ComplexT{0.097204863997, 0.197194179664},
            ComplexT{0.097055284752, 0.306793234914},
            ComplexT{0.301522534529, 0.067284365065},
            ComplexT{0.022572982655, 0.270590123918},
            ComplexT{0.243835640173, 0.084594090888},
            ComplexT{-0.032823490356, 0.171397202432},
            ComplexT{0.225215396328, -0.070141071525},
            ComplexT{-0.001322233373, 0.262893576650},
            ComplexT{0.118674074836, 0.194699985129},
            ComplexT{0.164569740491, 0.208130081842},
            ComplexT{0.069096925107, 0.142696982805},
            ComplexT{0.230464206558, -0.069754376895},
            ComplexT{0.174543309361, 0.312059756876},
        };

        auto st = ini_st;

        GateImplementation::applyMultiRZ(st.data(), num_qubits, wires, false,
                                         angle);
        REQUIRE(st == approx(expected).margin(1e-7));
    }
}
PENNYLANE_RUN_TEST(MultiRZ);

TEMPLATE_TEST_CASE(
    "StateVectorLQubitManaged::applyOperation param one-qubit with controls",
    "[StateVectorLQubitManaged]", float, double) {
    using PrecisionT = TestType;
    std::mt19937 re{1337};
    const int num_qubits = 4;
    const auto margin = PrecisionT{1e-5};
    const size_t control = GENERATE(0, 1, 2, 3);
    const size_t wire = GENERATE(0, 1, 2, 3);
    StateVectorLQubitManaged<PrecisionT> sv0(num_qubits);
    StateVectorLQubitManaged<PrecisionT> sv1(num_qubits);

    DYNAMIC_SECTION("N-controlled PhaseShift - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        bool inverse = GENERATE(false, true);
        PrecisionT param = GENERATE(-1.5, -0.5, 0, 0.5, 1.5);
        if (control != wire) {
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            sv0.applyOperation("ControlledPhaseShift", {control, wire}, inverse,
                               {param});
            sv1.applyOperation("PhaseShift", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire}, inverse, {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled RX - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        bool inverse = GENERATE(false, true);
        PrecisionT param = GENERATE(-1.5, -0.5, 0, 0.5, 1.5);
        if (control != wire) {
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            sv0.applyOperation("CRX", {control, wire}, inverse, {param});
            sv1.applyOperation("RX", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire}, inverse, {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled RY - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        bool inverse = GENERATE(false, true);
        PrecisionT param = GENERATE(-1.5, -0.5, 0, 0.5, 1.5);
        if (control != wire) {
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            sv0.applyOperation("CRY", {control, wire}, inverse, {param});
            sv1.applyOperation("RY", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire}, inverse, {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled RZ - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        bool inverse = GENERATE(false, true);
        PrecisionT param = GENERATE(-1.5, -0.5, 0, 0.5, 1.5);
        if (control != wire) {
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            sv0.applyOperation("CRZ", {control, wire}, inverse, {param});
            sv1.applyOperation("RZ", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire}, inverse, {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled Rot - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        const bool inverse = GENERATE(false, true);
        const PrecisionT param = GENERATE(-1.5, -0.5, 0, 0.5, 1.5);
        if (control != wire) {
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);
            const std::vector<PrecisionT> params = {
                param, static_cast<PrecisionT>(2.0) * param,
                static_cast<PrecisionT>(3.0) * param};
            sv0.applyOperation("CRot", {control, wire}, inverse, params);
            sv1.applyOperation("Rot", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire}, inverse, params);
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }
}

TEMPLATE_TEST_CASE(
    "StateVectorLQubitManaged::applyOperation param two-qubits with controls",
    "[StateVectorLQubitManaged]", float, double) {
    using PrecisionT = TestType;
    using ComplexT = std::complex<TestType>;
    std::mt19937 re{1337};
    const int num_qubits = 4;
    const auto margin = PrecisionT{1e-5};
    const size_t control = GENERATE(0, 1, 2, 3);
    const size_t wire0 = GENERATE(0, 1, 2, 3);
    const size_t wire1 = GENERATE(0, 1, 2, 3);
    StateVectorLQubitManaged<PrecisionT> sv0(num_qubits);
    StateVectorLQubitManaged<PrecisionT> sv1(num_qubits);

    auto getControlledGate = [](std::vector<ComplexT> matrix) {
        std::vector<ComplexT> cmatrix(matrix.size() * 4);
        for (std::size_t i = 0; i < 4; i++) {
            cmatrix[i * 8 + i] = ComplexT{1.0};
        }
        for (std::size_t i = 0; i < 4; i++) {
            for (std::size_t j = 0; j < 4; j++) {
                cmatrix[(i + 4) * 8 + j + 4] = matrix[i * 4 + j];
            }
        }
        return cmatrix;
    };

    DYNAMIC_SECTION("N-controlled IsingXX - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        bool inverse = GENERATE(false, true);
        PrecisionT param = GENERATE(-1.5, -0.5, 0, 0.5, 1.5);
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            auto matrix = getIsingXX<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1}, inverse);
            sv1.applyOperation("IsingXX", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire0, wire1}, inverse,
                               {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled IsingXY - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        bool inverse = GENERATE(false, true);
        PrecisionT param = GENERATE(-1.5, -0.5, 0, 0.5, 1.5);
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            auto matrix = getIsingXY<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1}, inverse);
            sv1.applyOperation("IsingXY", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire0, wire1}, inverse,
                               {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled IsingYY - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        bool inverse = GENERATE(false, true);
        PrecisionT param = GENERATE(-1.5, -0.5, 0, 0.5, 1.5);
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            auto matrix = getIsingYY<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1}, inverse);
            sv1.applyOperation("IsingYY", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire0, wire1}, inverse,
                               {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled IsingZZ - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        bool inverse = GENERATE(false, true);
        PrecisionT param = GENERATE(-1.5, -0.5, 0, 0.5, 1.5);
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            auto matrix = getIsingZZ<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1}, inverse);
            sv1.applyOperation("IsingZZ", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire0, wire1}, inverse,
                               {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled SingleExcitation - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        bool inverse = GENERATE(false, true);
        PrecisionT param = GENERATE(-1.5, -0.5, 0, 0.5, 1.5);
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            auto matrix = getSingleExcitation<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1}, inverse);
            sv1.applyOperation("SingleExcitation", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire0, wire1}, inverse,
                               {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled SingleExcitationMinus - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        bool inverse = GENERATE(false, true);
        PrecisionT param = GENERATE(-1.5, -0.5, 0, 0.5, 1.5);
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            auto matrix =
                getSingleExcitationMinus<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1}, inverse);
            sv1.applyOperation(
                "SingleExcitationMinus", std::vector<size_t>{control},
                std::vector<bool>{true}, std::vector<size_t>{wire0, wire1},
                inverse, {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled SingleExcitationPlus - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        bool inverse = GENERATE(false, true);
        PrecisionT param = GENERATE(-1.5, -0.5, 0, 0.5, 1.5);
        if (control != wire0 && control != wire1 && wire0 != wire1) {
            auto matrix =
                getSingleExcitationPlus<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1}, inverse);
            sv1.applyOperation(
                "SingleExcitationPlus", std::vector<size_t>{control},
                std::vector<bool>{true}, std::vector<size_t>{wire0, wire1},
                inverse, {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }
}

TEMPLATE_TEST_CASE(
    "StateVectorLQubitManaged::applyOperation param four-qubits with controls",
    "[StateVectorLQubitManaged]", float, double) {
    using PrecisionT = TestType;
    using ComplexT = std::complex<TestType>;
    std::mt19937 re{1337};
    const int num_qubits = 5;
    const auto margin = PrecisionT{1e-5};
    const size_t control = GENERATE(0, 1, 2, 3, 4);
    const size_t wire0 = GENERATE(0, 1, 2, 3, 4);
    const size_t wire1 = GENERATE(0, 1, 2, 3, 4);
    const size_t wire2 = GENERATE(0, 1, 2, 3, 4);
    const size_t wire3 = GENERATE(0, 1, 2, 3, 4);
    StateVectorLQubitManaged<PrecisionT> sv0(num_qubits);
    StateVectorLQubitManaged<PrecisionT> sv1(num_qubits);

    auto getControlledGate = [](std::vector<ComplexT> matrix) {
        std::vector<ComplexT> cmatrix(matrix.size() * 4);
        for (std::size_t i = 0; i < 16; i++) {
            cmatrix[i * 32 + i] = ComplexT{1.0};
        }
        for (std::size_t i = 0; i < 16; i++) {
            for (std::size_t j = 0; j < 16; j++) {
                cmatrix[(i + 16) * 32 + j + 16] = matrix[i * 16 + j];
            }
        }
        return cmatrix;
    };

    DYNAMIC_SECTION("N-controlled DoubleExcitation - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << ", " << wire2
                    << ", " << wire3 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        bool inverse = GENERATE(false, true);
        PrecisionT param = GENERATE(-1.5, -0.5, 0, 0.5, 1.5);
        std::vector<std::size_t> wires = {control, wire0, wire1, wire2, wire3};
        std::sort(wires.begin(), wires.end());
        if (std::adjacent_find(wires.begin(), wires.end()) == wires.end()) {
            auto matrix = getDoubleExcitation<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1, wire2, wire3},
                            inverse);
            sv1.applyOperation("DoubleExcitation", std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire0, wire1, wire2, wire3},
                               inverse, {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled DoubleExcitationMinus - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << ", " << wire2
                    << ", " << wire3 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        bool inverse = GENERATE(false, true);
        PrecisionT param = GENERATE(-1.5, -0.5, 0, 0.5, 1.5);
        std::vector<std::size_t> wires = {control, wire0, wire1, wire2, wire3};
        std::sort(wires.begin(), wires.end());
        if (std::adjacent_find(wires.begin(), wires.end()) == wires.end()) {
            auto matrix =
                getDoubleExcitationMinus<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1, wire2, wire3},
                            inverse);
            sv1.applyOperation("DoubleExcitationMinus",
                               std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire0, wire1, wire2, wire3},
                               inverse, {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled DoubleExcitationPlus - "
                    << "controls = {" << control << "} "
                    << ", wires = {" << wire0 << ", " << wire1 << ", " << wire2
                    << ", " << wire3 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        bool inverse = GENERATE(false, true);
        PrecisionT param = GENERATE(-1.5, -0.5, 0, 0.5, 1.5);
        std::vector<std::size_t> wires = {control, wire0, wire1, wire2, wire3};
        std::sort(wires.begin(), wires.end());
        if (std::adjacent_find(wires.begin(), wires.end()) == wires.end()) {
            auto matrix =
                getDoubleExcitationPlus<std::complex, PrecisionT>(param);
            std::vector<ComplexT> cmatrix = getControlledGate(matrix);
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);

            sv0.applyMatrix(cmatrix, {control, wire0, wire1, wire2, wire3},
                            inverse);
            sv1.applyOperation("DoubleExcitationPlus",
                               std::vector<size_t>{control},
                               std::vector<bool>{true},
                               std::vector<size_t>{wire0, wire1, wire2, wire3},
                               inverse, {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }

    DYNAMIC_SECTION("N-controlled MultiRZ - "
                    << "controls = {" << control << ", " << wire0 << ", "
                    << wire1 << "} "
                    << ", wires = {" << wire2 << ", " << wire3 << "} - "
                    << PrecisionToName<PrecisionT>::value) {
        bool inverse = GENERATE(false, true);
        PrecisionT param = GENERATE(-1.5, -0.5, 0, 0.5, 1.5);
        std::vector<std::size_t> wires = {control, wire0, wire1, wire2, wire3};
        std::sort(wires.begin(), wires.end());
        const ComplexT e = std::exp(ComplexT{0, -0.5} * param);
        std::vector<ComplexT> matrix(16, 0.0);
        matrix[0] = e;
        matrix[5] = std::conj(e);
        matrix[10] = std::conj(e);
        matrix[15] = e;
        if (std::adjacent_find(wires.begin(), wires.end()) == wires.end()) {
            auto st0 = createRandomStateVectorData<PrecisionT>(re, num_qubits);
            sv0.updateData(st0);
            sv1.updateData(st0);
            sv0.applyControlledMatrix(matrix.data(), {control, wire0, wire1},
                                      std::vector<bool>{true, false, true},
                                      {wire2, wire3}, inverse);
            sv1.applyOperation(
                "MultiRZ", std::vector<size_t>{control, wire0, wire1},
                std::vector<bool>{true, false, true},
                std::vector<size_t>{wire2, wire3}, inverse, {param});
            REQUIRE(sv0.getDataVector() ==
                    approx(sv1.getDataVector()).margin(margin));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorLQubitManaged::applyGlobalPhase",
                   "[StateVectorLQubitManaged_Param]", double) {
    using ComplexT = StateVectorLQubitManaged<TestType>::ComplexT;
    std::mt19937_64 re{1337};

    const size_t num_qubits = 3;
    const bool inverse = GENERATE(false, true);
    const size_t index = GENERATE(0, 1, 2);
    const TestType param = 0.234;
    const ComplexT phase = std::exp(ComplexT{0, (inverse) ? param : -param});

    auto sv_data = createRandomStateVectorData<TestType>(re, num_qubits);
    StateVectorLQubitManaged<TestType> sv(
        reinterpret_cast<ComplexT *>(sv_data.data()), sv_data.size());
    sv.applyOperation("GlobalPhase", {index}, inverse, {param});
    auto result_sv = sv.getDataVector();
    for (size_t j = 0; j < exp2(num_qubits); j++) {
        ComplexT tmp = phase * ComplexT(sv_data[j]);
        CHECK((real(result_sv[j])) == Approx(real(tmp)));
        CHECK((imag(result_sv[j])) == Approx(imag(tmp)));
    }
}

TEMPLATE_TEST_CASE("StateVectorLQubitManaged::applyControlledGlobalPhase",
                   "[StateVectorLQubitManaged_Param]", double) {
    using ComplexT = StateVectorLQubitManaged<TestType>::ComplexT;
    std::mt19937_64 re{1337};

    const TestType pi2 = 1.5707963267948966;
    const size_t num_qubits = 3;
    const bool inverse = GENERATE(false, true);
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
    StateVectorLQubitManaged<TestType> sv(
        reinterpret_cast<ComplexT *>(sv_data.data()), sv_data.size());
    sv.applyOperation("GlobalPhase", {0, 1}, {0, 1}, {2}, inverse, {-pi2});
    auto result_sv = sv.getDataVector();
    for (size_t j = 0; j < exp2(num_qubits); j++) {
        ComplexT tmp = (inverse) ? conj(phase[j]) : phase[j];
        tmp *= ComplexT(sv_data[j]);
        CHECK((real(result_sv[j])) == Approx(real(tmp)));
        CHECK((imag(result_sv[j])) == Approx(imag(tmp)));
    }
}
