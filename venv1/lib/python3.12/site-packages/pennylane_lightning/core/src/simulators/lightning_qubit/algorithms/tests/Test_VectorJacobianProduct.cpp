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
#include <numbers>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch.hpp>

#include "AdjointJacobianLQubit.hpp"
#include "ConstantUtil.hpp"
#include "GateOperation.hpp"
#include "ObservablesLQubit.hpp"
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"
#include "TestHelpers.hpp"      // randomIntVector
#include "TestHelpersWires.hpp" // createWires
#include "Util.hpp"             // TestVector
#include "VectorJacobianProduct.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Algorithms;
using namespace Pennylane::Util;

using namespace Pennylane::LightningQubit::Algorithms;
using namespace Pennylane::LightningQubit::Observables;
using namespace Pennylane::LightningQubit::Util;
} // namespace
/// @endcond

#if !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif

/**
 * @brief
 *
 * @param length Size of the gate sequence
 * @param
 */
template <class StateVectorT, class RandomEngine>
auto createRandomOps(RandomEngine &re, size_t length, size_t wires)
    -> OpsData<StateVectorT> {
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using namespace Pennylane::LightningQubit::Gates;

    std::array gates_to_use = {Pennylane::Gates::GateOperation::RX,
                               Pennylane::Gates::GateOperation::RY,
                               Pennylane::Gates::GateOperation::RZ};

    std::vector<std::string> ops_names;
    std::vector<std::vector<PrecisionT>> ops_params;
    std::vector<std::vector<size_t>> ops_wires;
    std::vector<bool> ops_inverses;

    std::uniform_int_distribution<size_t> gate_dist(0, gates_to_use.size() - 1);
    std::uniform_real_distribution<PrecisionT> param_dist(0.0, 2 * M_PI);
    std::uniform_int_distribution<int> inverse_dist(0, 1);

    for (size_t i = 0; i < length; i++) {
        const auto gate_op = gates_to_use[gate_dist(re)];
        const auto gate_name = lookup(Constant::gate_names, gate_op);
        ops_names.emplace_back(gate_name);
        ops_params.emplace_back(std::vector<PrecisionT>{param_dist(re)});
        ops_inverses.emplace_back(inverse_dist(re));
        ops_wires.emplace_back(createWires(gate_op, wires));
    }

    return {ops_names,
            ops_params,
            ops_wires,
            ops_inverses,
            std::vector<std::vector<ComplexT>>(length),
            std::vector<std::vector<size_t>>(length),
            std::vector<std::vector<bool>>(length)};
}

TEMPLATE_PRODUCT_TEST_CASE("StateVector VJP", "[Algorithms]",
                           (StateVectorLQubitManaged, StateVectorLQubitRaw),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    using std::cos;
    using std::sin;
    using std::sqrt;

    VectorJacobianProduct<StateVectorT> vector_jacobian_product;

    constexpr static auto isqrt2 = INVSQRT2<PrecisionT>();

    SECTION("Do nothing if the tape does not have trainable parameters") {
        std::vector<ComplexT> vjp(1);
        OpsData<StateVectorT> ops_data{
            {"CNOT", "RX"},   // names
            {{}, {M_PI / 7}}, // params
            {{0, 1}, {1}},    // wires
            {false, false},   // inverses
        };

        auto dy = std::vector<ComplexT>(4);
        std::vector<ComplexT> ini_st{
            {isqrt2, 0.0}, {0.0, 0.0}, {isqrt2, 0.0}, {0.0, 0.0}};
        JacobianData<StateVectorT> jd{1, 4, ini_st.data(), {}, ops_data, {}};
        REQUIRE_NOTHROW(vector_jacobian_product(
            std::span{vjp}, jd, std::span<const ComplexT>{dy}, true));
    }

    SECTION("CNOT RX1") {
        const PrecisionT theta = std::numbers::pi_v<PrecisionT> / 7;
        OpsData<StateVectorT> ops_data{
            {"CNOT", "RX"}, // names
            {{}, {theta}},  // params
            {{0, 1}, {1}},  // wires
            {false, false}, // inverses
        };

        auto dy = std::vector<ComplexT>(4);

        std::vector<std::vector<ComplexT>> expected = {
            {{-isqrt2 / PrecisionT{2.0} * sin(theta / 2), 0.0}},
            {{0.0, -isqrt2 / PrecisionT{2.0} * cos(theta / 2)}},
            {{0.0, -isqrt2 / PrecisionT{2.0} * cos(theta / 2)}},
            {{-isqrt2 / PrecisionT{2.0} * sin(theta / 2), 0.0}},
        };

        SECTION("with apply_operations = true") {
            std::vector<ComplexT> ini_st{
                {isqrt2, 0.0}, {0.0, 0.0}, {isqrt2, 0.0}, {0.0, 0.0}};
            JacobianData<StateVectorT> jd{1,  4,        ini_st.data(),
                                          {}, ops_data, {0}};

            for (size_t i = 0; i < 4; i++) {
                std::fill(dy.begin(), dy.end(), ComplexT{0.0, 0.0});
                dy[i] = {1.0, 0.0};
                std::vector<ComplexT> vjp(1);
                vector_jacobian_product(std::span{vjp}, jd,
                                        std::span<const ComplexT>{dy}, true);

                REQUIRE(vjp == approx(expected[i]).margin(1e-5));
            }
        }

        SECTION("with apply_operations = false") {
            std::vector<ComplexT> final_st{{cos(theta / 2) * isqrt2, 0.0},
                                           {0.0, -isqrt2 * sin(theta / 2)},
                                           {0.0, -isqrt2 * sin(theta / 2)},
                                           {cos(theta / 2) * isqrt2, 0.0}};
            JacobianData<StateVectorT> jd{1,  4,        final_st.data(),
                                          {}, ops_data, {0}};

            for (size_t i = 0; i < 4; i++) {
                std::fill(dy.begin(), dy.end(), ComplexT{0.0, 0.0});
                dy[i] = {1.0, 0.0};
                std::vector<ComplexT> vjp(1);
                vector_jacobian_product(std::span{vjp}, jd,
                                        std::span<const ComplexT>{dy}, false);

                REQUIRE(vjp == approx(expected[i]).margin(1e-5));
            }
        }
    }

    SECTION("CNOT0,1 RX1 CNOT1,0 RX0 CNOT0,1 RX1 CNOT1,0 RX0") {
        std::vector<ComplexT> ini_st{
            {isqrt2, 0.0}, {0.0, 0.0}, {isqrt2, 0.0}, {0.0, 0.0}};

        OpsData<StateVectorT> ops_data{
            {"CNOT", "RX", "CNOT", "RX", "CNOT", "RX", "CNOT", "RX"}, // names
            {{}, {M_PI}, {}, {M_PI}, {}, {M_PI}, {}, {M_PI}},         // params
            {{0, 1}, {1}, {1, 0}, {0}, {0, 1}, {1}, {1, 0}, {0}},     // wires
            {false, false, false, false, false, false, false,
             false}, // inverses
        };

        std::vector<ComplexT> expected_der0 = {
            {0.0, -isqrt2 / 2.0},
            {0.0, 0.0},
            {0.0, 0.0},
            {0.0, -isqrt2 / 2.0},
        }; // For trainable_param == 0
        std::vector<ComplexT> expected_der1 = {
            {0.0, 0.0},
            {0.0, -isqrt2 / 2.0},
            {0.0, -isqrt2 / 2.0},
            {0.0, 0.0},
        }; // For trainable_param == 1

        SECTION("with apply_operations = true") {
            std::vector<ComplexT> ini_st{
                {isqrt2, 0.0}, {0.0, 0.0}, {isqrt2, 0.0}, {0.0, 0.0}};

            JacobianData<StateVectorT> jd{
                1, 4, ini_st.data(), {}, ops_data, {1, 2} // trainable params
            };

            auto dy = std::vector<ComplexT>(4);

            for (size_t i = 0; i < 4; i++) {
                std::fill(dy.begin(), dy.end(), ComplexT{0.0, 0.0});
                dy[i] = {1.0, 0.0};
                std::vector<ComplexT> vjp(2);
                vector_jacobian_product(std::span{vjp}, jd,
                                        std::span<const ComplexT>{dy}, true);

                REQUIRE(vjp[0] == approx(expected_der0[i]).margin(1e-5));
                REQUIRE(vjp[1] == approx(expected_der1[i]).margin(1e-5));
            }
        }

        SECTION("with apply_operations = false") {
            std::vector<ComplexT> final_st{
                {0.0, 0.0}, {isqrt2, 0.0}, {isqrt2, 0.0}, {0.0, 0.0}};

            JacobianData<StateVectorT> jd{
                4, 4, final_st.data(), {}, ops_data, {1, 2} // trainable params
            };

            auto dy = std::vector<ComplexT>(4);

            for (size_t i = 0; i < 4; i++) {
                std::fill(dy.begin(), dy.end(), ComplexT{0.0, 0.0});
                dy[i] = {1.0, 0.0};
                std::vector<ComplexT> vjp(2);
                vector_jacobian_product(std::span{vjp}, jd,
                                        std::span<const ComplexT>{dy}, false);

                REQUIRE(vjp[0] == approx(expected_der0[i]).margin(1e-5));
                REQUIRE(vjp[1] == approx(expected_der1[i]).margin(1e-5));
            }
        }
    }

    SECTION("Test complex dy") {
        OpsData<StateVectorT> ops_data1{
            {"CNOT", "RX"},   // names
            {{}, {M_PI / 7}}, // params
            {{0, 1}, {1}},    // wires
            {false, false},   // inverses
        };

        auto dy1 = std::vector<ComplexT>{
            {0.4, 0.4}, {0.4, 0.4}, {0.4, 0.4}, {0.4, 0.4}};

        OpsData<StateVectorT> ops_data2{
            {"CNOT", "RX"},    // names
            {{}, {-M_PI / 7}}, // params
            {{0, 1}, {1}},     // wires
            {false, false},    // inverses
        };

        auto dy2 = std::vector<ComplexT>{
            {0.4, -0.4}, {0.4, -0.4}, {0.4, -0.4}, {0.4, -0.4}};
        std::vector<ComplexT> ini_st{
            {isqrt2, 0.0}, {0.0, 0.0}, {isqrt2, 0.0}, {0.0, 0.0}};

        JacobianData<StateVectorT> jd1{1, 4, ini_st.data(), {}, ops_data1, {0}};
        JacobianData<StateVectorT> jd2{1, 4, ini_st.data(), {}, ops_data2, {0}};

        std::vector<ComplexT> vjp1(1);
        std::vector<ComplexT> vjp2(1);

        vector_jacobian_product(std::span{vjp1}, jd1,
                                std::span<const ComplexT>{dy1}, true);

        vector_jacobian_product(std::span{vjp2}, jd2,
                                std::span<const ComplexT>{dy2}, true);

        REQUIRE(vjp1[0] == approx(-std::conj(vjp2[0])));
    }

    SECTION("Test controlled complex dy") {
        OpsData<StateVectorT> ops_data1{
            {"PauliX", "RX"},       // names
            {{}, {M_PI / 7}},       // params
            {{1}, {1}},             // wires
            {false, false},         // inverses
            {{}, {}},               // matrices
            {{0, 2}, {2}},          // controlled wires
            {{true, true}, {true}}, // controlled values
        };

        auto dy1 = std::vector<ComplexT>{{0.4, 0.4}, {0.4, 0.4}, {0.4, 0.4},
                                         {0.4, 0.4}, {0.4, 0.4}, {0.4, 0.4},
                                         {0.4, 0.4}, {0.4, 0.4}};

        OpsData<StateVectorT> ops_data2{
            {"PauliX", "RX"},       // names
            {{}, {-M_PI / 7}},      // params
            {{1}, {1}},             // wires
            {false, false},         // inverses
            {{}, {}},               // matrices
            {{0, 2}, {2}},          // controlled wires
            {{true, true}, {true}}, // controlled values
        };

        auto dy2 = std::vector<ComplexT>{{0.4, -0.4}, {0.4, -0.4}, {0.4, -0.4},
                                         {0.4, -0.4}, {0.4, -0.4}, {0.4, -0.4},
                                         {0.4, -0.4}, {0.4, -0.4}};

        std::vector<ComplexT> ini_st{{isqrt2, 0.0}, {0.0, 0.0}, {isqrt2, 0.0},
                                     {0.0, 0.0},    {0.0, 0.0}, {0.0, 0.0},
                                     {0.0, 0.0},    {0.0, 0.0}};

        JacobianData<StateVectorT> jd1{1,  ini_st.size(), ini_st.data(),
                                       {}, ops_data1,     {0}};
        JacobianData<StateVectorT> jd2{1,  ini_st.size(), ini_st.data(),
                                       {}, ops_data2,     {0}};

        std::vector<ComplexT> vjp1(1);
        std::vector<ComplexT> vjp2(1);

        vector_jacobian_product(std::span{vjp1}, jd1,
                                std::span<const ComplexT>{dy1}, true);

        vector_jacobian_product(std::span{vjp2}, jd2,
                                std::span<const ComplexT>{dy2}, true);

        REQUIRE(vjp1[0] == approx(-std::conj(vjp2[0])));
    }

    SECTION(
        "Check the result is consistent with adjoint diff with observables") {
        using VectorT = TestVector<ComplexT>;
        AdjointJacobian<StateVectorT> adj;

        std::mt19937 re{1337};
        auto ops_data = createRandomOps<StateVectorT>(re, 10, 3);
        auto obs = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{0});

        const size_t num_params = [&]() {
            size_t r = 0;
            for (const auto &ops_params : ops_data.getOpsParams()) {
                if (!ops_params.empty()) {
                    ++r;
                }
            }
            return r;
        }();

        std::vector<size_t> trainable_params(num_params);
        std::iota(trainable_params.begin(), trainable_params.end(), 0);

        VectorT ini_st = createProductState<PrecisionT>("+++");

        StateVectorT sv(ini_st.data(), ini_st.size());

        for (size_t op_idx = 0; op_idx < ops_data.getOpsName().size();
             op_idx++) {
            sv.applyOperation(ops_data.getOpsName()[op_idx],
                              ops_data.getOpsWires()[op_idx], false,
                              ops_data.getOpsParams()[op_idx]);
        }

        std::vector<ComplexT> sv_data(sv.getData(),
                                      sv.getData() + sv.getLength());
        JacobianData<StateVectorT> jd{num_params, 8,        sv_data.data(),
                                      {obs},      ops_data, trainable_params};

        auto o_sv = sv;
        obs->applyInPlace(o_sv);

        std::vector<ComplexT> o_sv_data(o_sv.getData(),
                                        o_sv.getData() + o_sv.getLength());

        std::vector<PrecisionT> grad_vjp = [&]() {
            std::vector<ComplexT> vjp(num_params);
            vector_jacobian_product(std::span{vjp}, jd,
                                    std::span<const ComplexT>{o_sv_data},
                                    false);
            std::vector<PrecisionT> res(vjp.size());
            std::transform(vjp.begin(), vjp.end(), res.begin(),
                           [](const auto &x) { return 2 * std::real(x); });
            return res;
        }();

        std::vector<PrecisionT> jac(num_params);
        adj.adjointJacobian(std::span{jac}, jd, sv);

        REQUIRE(grad_vjp == approx(jac).margin(1e-5));
    }
}