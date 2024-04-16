#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include <catch2/catch.hpp>

#include "AdjointJacobianGPU.hpp"
#include "JacobianData.hpp"
#include "StateVectorCudaManaged.hpp"
#include "TestHelpers.hpp"
#include "TestHelpersStateVectors.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningGPU::Algorithms;
} // namespace
/// @endcond

/**
 * @brief Tests the constructability of the AdjointJacobianGPU.hpp classes.
 *
 */
TEMPLATE_TEST_CASE("AdjointJacobianGPU::AdjointJacobianGPU",
                   "[AdjointJacobianGPU]", float, double) {
    SECTION("AdjointJacobianGPU<TestType> {}") {
        REQUIRE(std::is_constructible<
                AdjointJacobian<StateVectorCudaManaged<TestType>>>::value);
    }
}

TEST_CASE("AdjointJacobianGPU::AdjointJacobianGPU Op=RX, Obs=Z",
          "[AdjointJacobianGPU]") {
    using StateVectorT = StateVectorCudaManaged<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    const std::vector<size_t> tp{0};
    {
        const size_t num_qubits = 1;
        const size_t num_obs = 1;
        const auto obs = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{0});

        std::vector<double> jacobian(num_obs * tp.size(), 0);

        for (const auto &p : param) {
            auto ops = OpsData<StateVectorT>({"RX"}, {{p}}, {{0}}, {false});

            StateVectorT psi(num_qubits);
            psi.initSV();

            JacobianData<StateVectorT> tape{
                param.size(), psi.getLength(), psi.getData(), {obs}, ops, tp};

            adj.adjointJacobian(std::span{jacobian}, tape, psi, true);
            CAPTURE(jacobian);
            CHECK(-sin(p) == Approx(jacobian[0]));
        }
    }
}

TEST_CASE("AdjointJacobianGPU::adjointJacobian Op=RY, Obs=X",
          "[AdjointJacobianGPU]") {
    using StateVectorT = StateVectorCudaManaged<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    const std::vector<size_t> tp{0};
    {
        const size_t num_qubits = 1;
        const size_t num_obs = 1;

        const auto obs = std::make_shared<NamedObs<StateVectorT>>(
            "PauliX", std::vector<size_t>{0});
        std::vector<double> jacobian(num_obs * tp.size(), 0);

        for (const auto &p : param) {
            auto ops = OpsData<StateVectorT>({"RY"}, {{p}}, {{0}}, {false});

            StateVectorT psi(num_qubits);
            psi.initSV();

            JacobianData<StateVectorT> tape{
                param.size(), psi.getLength(), psi.getData(), {obs}, ops, tp};

            adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

            CAPTURE(jacobian);
            CHECK(cos(p) == Approx(jacobian[0]).margin(1e-7));
        }
    }
}

TEST_CASE("AdjointJacobianGPU::adjointJacobian Op=[QubitStateVector, "
          "StatePrep, BasisState], Obs=[Z,Z]",
          "[AdjointJacobianGPU]") {
    const std::string test_ops =
        GENERATE("QubitStateVector", "StatePrep", "BasisState");
    using StateVectorT = StateVectorCudaManaged<double>;
    using ComplexT = StateVectorT::ComplexT;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0};
    {
        const size_t num_qubits = 2;
        const size_t num_obs = 2;
        std::vector<double> jacobian(num_obs * tp.size(), 0);
        std::vector<double> jacobian_ref(num_obs * tp.size(), 0);
        std::vector<ComplexT> matrix = {
            {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

        StateVectorT psi(num_qubits);
        psi.initSV();

        const auto obs1 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});

        auto ops = OpsData<StateVectorT>({test_ops}, {{param[0]}}, {{0}},
                                         {false}, {matrix});

        JacobianData<StateVectorT> tape{param.size(),  psi.getLength(),
                                        psi.getData(), {obs1, obs2},
                                        ops,           tp};

        // apply_operations should be set as false to cover if statement in
        // adjointJacobian when ops is "QubitStateVector" "StatePrep" or
        // "BasisState". If apply_operations is set as true, errors will be
        // thrown out since ops mentioned above is not supported in
        // apply_operation method of sv.
        adj.adjointJacobian(std::span{jacobian}, tape, psi, false);

        CAPTURE(jacobian);
        CHECK(jacobian == Pennylane::Util::approx(jacobian_ref).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobianGPU::adjointJacobian Op=RX, Obs=[Z,Z]",
          "[AdjointJacobianGPU]") {
    using StateVectorT = StateVectorCudaManaged<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0};
    {
        const size_t num_qubits = 2;
        const size_t num_obs = 2;
        std::vector<double> jacobian(num_obs * tp.size(), 0);

        StateVectorT psi(num_qubits);
        psi.initSV();

        const auto obs1 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});

        auto ops = OpsData<StateVectorT>({"RX"}, {{param[0]}}, {{0}}, {false});

        JacobianData<StateVectorT> tape{param.size(),  psi.getLength(),
                                        psi.getData(), {obs1, obs2},
                                        ops,           tp};

        adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

        CAPTURE(jacobian);
        CHECK(-sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
        CHECK(0.0 == Approx(jacobian[tp.size()]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobianGPU::AdjointJacobianGPU Op=[RX,RX,RX], Obs=[Z,Z,Z]",
          "[AdjointJacobianGPU]") {
    using StateVectorT = StateVectorCudaManaged<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 1, 2};
    {
        const size_t num_qubits = 3;
        const size_t num_obs = 3;
        std::vector<double> jacobian(num_obs * tp.size(), 0);

        StateVectorT psi(num_qubits);
        psi.initSV();

        const auto obs1 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});
        const auto obs3 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{2});

        auto ops = OpsData<StateVectorT>(
            {"RX", "RX", "RX"}, {{param[0]}, {param[1]}, {param[2]}},
            {{0}, {1}, {2}}, {false, false, false});

        JacobianData<StateVectorT> tape{param.size(),  psi.getLength(),
                                        psi.getData(), {obs1, obs2, obs3},
                                        ops,           tp};

        adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

        CAPTURE(jacobian);

        // Computed with parameter shift
        CHECK(-sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
        CHECK(-sin(param[1]) == Approx(jacobian[1 + tp.size()]).margin(1e-7));
        CHECK(-sin(param[2]) ==
              Approx(jacobian[2 + 2 * tp.size()]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobianGPU::AdjointJacobianGPU Op=[RX,RX,RX], Obs=[Z,Z,Z],"
          "TParams=[0,2]",
          "[AdjointJacobianGPU]") {
    using StateVectorT = StateVectorCudaManaged<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 2};
    {
        const size_t num_qubits = 3;
        const size_t num_obs = 3;
        std::vector<double> jacobian(num_obs * tp.size(), 0);

        StateVectorT psi(num_qubits);
        psi.initSV();

        const auto obs1 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});
        const auto obs3 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{2});
        auto ops = OpsData<StateVectorT>(
            {"RX", "RX", "RX"}, {{param[0]}, {param[1]}, {param[2]}},
            {{0}, {1}, {2}}, {false, false, false});

        JacobianData<StateVectorT> tape{param.size(),  psi.getLength(),
                                        psi.getData(), {obs1, obs2, obs3},
                                        ops,           tp};

        adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

        CAPTURE(jacobian);

        // Computed with parameter shift
        CHECK(-sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
        CHECK(0 == Approx(jacobian[1 + tp.size()]).margin(1e-7));
        CHECK(-sin(param[2]) ==
              Approx(jacobian[1 + 2 * tp.size()]).margin(1e-7));
    }
}

TEST_CASE("Algorithms::adjointJacobian Op=[RX,RX,RX], Obs=[ZZZ]",
          "[Algorithms]") {
    using StateVectorT = StateVectorCudaManaged<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 1, 2};
    {
        const size_t num_qubits = 3;
        const size_t num_obs = 1;
        std::vector<double> jacobian(num_obs * tp.size(), 0);

        StateVectorT psi(num_qubits);
        psi.initSV();

        const auto obs = std::make_shared<TensorProdObs<StateVectorT>>(
            std::make_shared<NamedObs<StateVectorT>>("PauliZ",
                                                     std::vector<size_t>{0}),
            std::make_shared<NamedObs<StateVectorT>>("PauliZ",
                                                     std::vector<size_t>{1}),
            std::make_shared<NamedObs<StateVectorT>>("PauliZ",
                                                     std::vector<size_t>{2}));
        auto ops = OpsData<StateVectorT>(
            {"RX", "RX", "RX"}, {{param[0]}, {param[1]}, {param[2]}},
            {{0}, {1}, {2}}, {false, false, false});

        JacobianData<StateVectorT> tape{
            param.size(), psi.getLength(), psi.getData(), {obs}, ops, tp};

        adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

        CAPTURE(jacobian);

        // Computed with parameter shift
        CHECK(-0.1755096592645253 == Approx(jacobian[0]).margin(1e-7));
        CHECK(0.26478810666384334 == Approx(jacobian[1]).margin(1e-7));
        CHECK(-0.6312451595102775 == Approx(jacobian[2]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobianGPU::adjointJacobian Op=Mixed, Obs=[XXX]",
          "[AdjointJacobianGPU]") {
    using StateVectorT = StateVectorCudaManaged<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 1, 2, 3, 4, 5};
    {
        const size_t num_qubits = 3;
        const size_t num_obs = 1;
        std::vector<double> jacobian(num_obs * tp.size(), 0);

        StateVectorT psi(num_qubits);
        psi.initSV();

        const auto obs = std::make_shared<TensorProdObs<StateVectorT>>(
            std::make_shared<NamedObs<StateVectorT>>("PauliX",
                                                     std::vector<size_t>{0}),
            std::make_shared<NamedObs<StateVectorT>>("PauliX",
                                                     std::vector<size_t>{1}),
            std::make_shared<NamedObs<StateVectorT>>("PauliX",
                                                     std::vector<size_t>{2}));
        auto ops = OpsData<StateVectorT>(
            {"RZ", "RY", "RZ", "CNOT", "CNOT", "RZ", "RY", "RZ"},
            {{param[0]},
             {param[1]},
             {param[2]},
             {},
             {},
             {param[0]},
             {param[1]},
             {param[2]}},
            {{0}, {0}, {0}, {0, 1}, {1, 2}, {1}, {1}, {1}},
            {false, false, false, false, false, false, false, false});

        JacobianData<StateVectorT> tape{
            param.size(), psi.getLength(), psi.getData(), {obs}, ops, tp};

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
}

TEST_CASE("AdjointJacobianGPU::adjointJacobian Decomposed Rot gate, non "
          "computational basis state",
          "[AdjointJacobianGPU]") {
    using StateVectorT = StateVectorCudaManaged<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    const std::vector<size_t> tp{0, 1, 2};
    {
        const size_t num_obs = 1;

        const auto thetas = Pennylane::Util::linspace(-2 * M_PI, 2 * M_PI, 7);
        std::unordered_map<double, std::vector<double>> expec_results{
            {thetas[0], {0.0, -9.90819496e-01, 0.0}},
            {thetas[1], {-8.18996553e-01, 1.62526544e-01, 0.0}},
            {thetas[2], {-0.203949, 0.48593716, 0.0}},
            {thetas[3], {0.0, 1.0, 0.0}},
            {thetas[4], {-2.03948985e-01, 4.85937177e-01, 0.0}},
            {thetas[5], {-8.18996598e-01, 1.62526487e-01, 0.0}},
            {thetas[6], {0.0, -9.90819511e-01, 0.0}}};

        for (const auto &theta : thetas) {
            std::vector<double> local_params{theta, std::pow(theta, 3),
                                             Pennylane::Util::SQRT2<double>() *
                                                 theta};
            std::vector<double> jacobian(num_obs * tp.size(), 0);

            std::vector<std::complex<double>> cdata{
                {Pennylane::Util::INVSQRT2<double>()},
                {-Pennylane::Util::INVSQRT2<double>()}};
            std::vector<std::complex<double>> new_data{cdata.begin(),
                                                       cdata.end()};
            StateVectorT psi(new_data.data(), new_data.size());

            const auto obs = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{0});

            auto ops = OpsData<StateVectorT>(
                {"RZ", "RY", "RZ"},
                {{local_params[0]}, {local_params[1]}, {local_params[2]}},
                {{0}, {0}, {0}}, {false, false, false});

            JacobianData<StateVectorT> tape{
                param.size(), psi.getLength(), psi.getData(), {obs}, ops, tp};

            adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

            CAPTURE(theta);
            CAPTURE(jacobian);

            // Computed with PennyLane using default.qubit
            CHECK(expec_results[theta][0] == Approx(jacobian[0]).margin(1e-7));
            CHECK(expec_results[theta][1] == Approx(jacobian[1]).margin(1e-7));
            CHECK(expec_results[theta][2] == Approx(jacobian[2]).margin(1e-7));
        }
    }
}

TEST_CASE("AdjointJacobianGPU::adjointJacobian Mixed Ops, Obs and TParams",
          "[AdjointJacobianGPU]") {
    using StateVectorT = StateVectorCudaManaged<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    {
        const std::vector<size_t> tp{1, 2, 3};
        const size_t num_obs = 1;

        const auto thetas = Pennylane::Util::linspace(-2 * M_PI, 2 * M_PI, 8);

        std::vector<double> local_params{0.543, 0.54, 0.1,  0.5, 1.3,
                                         -2.3,  0.5,  -0.5, 0.5};
        std::vector<double> jacobian(num_obs * tp.size(), 0);

        std::vector<std::complex<double>> cdata{
            {Pennylane::Util::ONE<double>()},
            {Pennylane::Util::ZERO<double>()},
            {Pennylane::Util::ZERO<double>()},
            {Pennylane::Util::ZERO<double>()}};
        std::vector<std::complex<double>> new_data{cdata.begin(), cdata.end()};
        StateVectorT psi(new_data.data(), new_data.size());

        const auto obs = std::make_shared<TensorProdObs<StateVectorT>>(
            std::make_shared<NamedObs<StateVectorT>>("PauliX",
                                                     std::vector<size_t>{0}),
            std::make_shared<NamedObs<StateVectorT>>("PauliZ",
                                                     std::vector<size_t>{1}));
        auto ops =
            OpsData<StateVectorT>({"Hadamard", "RX", "CNOT", "RZ", "RY", "RZ",
                                   "RZ", "RY", "RZ", "RZ", "RY", "CNOT"},
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
                                  std::vector<std::vector<std::size_t>>{{0},
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
                                  {false, false, false, false, false, false,
                                   false, false, false, false, false, false});

        JacobianData<StateVectorT> tape{
            param.size(), psi.getLength(), psi.getData(), {obs}, ops, tp};

        adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

        std::vector<double> expected{-0.71429188, 0.04998561, -0.71904837};
        // Computed with PennyLane using default.qubit
        CHECK(expected[0] == Approx(jacobian[0]));
        CHECK(expected[1] == Approx(jacobian[1]));
        CHECK(expected[2] == Approx(jacobian[2]));
    }
}

TEST_CASE("AdjointJacobianGPU::batchAdjointJacobian Mixed Ops, Obs and TParams",
          "[AdjointJacobianGPU]") {
    using StateVectorT = StateVectorCudaManaged<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    {
        const std::vector<size_t> tp{1, 2, 3};
        const size_t num_obs = 1;

        const auto thetas = Pennylane::Util::linspace(-2 * M_PI, 2 * M_PI, 8);

        std::vector<double> local_params{0.543, 0.54, 0.1,  0.5, 1.3,
                                         -2.3,  0.5,  -0.5, 0.5};
        std::vector<double> jacobian(num_obs * tp.size(), 0);

        std::vector<std::complex<double>> cdata{
            {Pennylane::Util::ONE<double>()},
            {Pennylane::Util::ZERO<double>()},
            {Pennylane::Util::ZERO<double>()},
            {Pennylane::Util::ZERO<double>()}};

        StateVectorT psi(cdata.data(), cdata.size());

        const auto obs = std::make_shared<TensorProdObs<StateVectorT>>(
            std::make_shared<NamedObs<StateVectorT>>("PauliX",
                                                     std::vector<size_t>{0}),
            std::make_shared<NamedObs<StateVectorT>>("PauliZ",
                                                     std::vector<size_t>{1}));
        auto ops =
            OpsData<StateVectorT>({"Hadamard", "RX", "CNOT", "RZ", "RY", "RZ",
                                   "RZ", "RY", "RZ", "RZ", "RY", "CNOT"},
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
                                  std::vector<std::vector<std::size_t>>{{0},
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
                                  {false, false, false, false, false, false,
                                   false, false, false, false, false, false});

        JacobianData<StateVectorT> tape{
            param.size(), psi.getLength(), psi.getData(), {obs}, ops, tp};

        adj.batchAdjointJacobian(std::span{jacobian}, tape, true);

        std::vector<double> expected{-0.71429188, 0.04998561, -0.71904837};
        // Computed with PennyLane using default.qubit
        CHECK(expected[0] == Approx(jacobian[0]));
        CHECK(expected[1] == Approx(jacobian[1]));
        CHECK(expected[2] == Approx(jacobian[2]));
    }
}

TEST_CASE("Algorithms::adjointJacobian Op=RX, Obs=Ham[Z0+Z1]", "[Algorithms]") {
    using StateVectorT = StateVectorCudaManaged<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0};
    {
        const size_t num_qubits = 2;
        const size_t num_obs = 1;
        std::vector<double> jacobian(num_obs * tp.size(), 0);

        StateVectorT psi(num_qubits);
        psi.initSV();

        const auto obs1 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});

        auto ham = Hamiltonian<StateVectorT>::create({0.3, 0.7}, {obs1, obs2});

        auto ops = OpsData<StateVectorT>({"RX"}, {{param[0]}}, {{0}}, {false});

        JacobianData<StateVectorT> tape{
            param.size(), psi.getLength(), psi.getData(), {ham}, ops, tp};

        adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

        CAPTURE(jacobian);
        CHECK(-0.3 * sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
    }
}

TEST_CASE(
    "AdjointJacobianGPU::AdjointJacobianGPU Op=[RX,RX,RX], Obs=Ham[Z0+Z1+Z2], "
    "TParams=[0,2]",
    "[AdjointJacobianGPU]") {
    using StateVectorT = StateVectorCudaManaged<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 2};
    {
        const size_t num_qubits = 3;
        const size_t num_obs = 1;
        std::vector<double> jacobian(num_obs * tp.size(), 0);

        StateVectorT psi(num_qubits);
        psi.initSV();

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

        JacobianData<StateVectorT> tape{
            param.size(), psi.getLength(), psi.getData(), {ham}, ops, tp};

        adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

        CAPTURE(jacobian);

        CHECK((-0.47 * sin(param[0]) == Approx(jacobian[0]).margin(1e-7)));
        CHECK((-0.96 * sin(param[2]) == Approx(jacobian[1]).margin(1e-7)));
    }
}

TEST_CASE("AdjointJacobianGPU::AdjointJacobianGPU Test HermitianObs",
          "[AdjointJacobianGPU]") {
    using StateVectorT = StateVectorCudaManaged<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 2};
    {
        const size_t num_qubits = 3;
        const size_t num_obs = 1;

        std::vector<double> jacobian1(num_obs * tp.size(), 0);
        std::vector<double> jacobian2(num_obs * tp.size(), 0);

        StateVectorT psi(num_qubits);
        psi.initSV();

        auto obs1 = std::make_shared<TensorProdObs<StateVectorT>>(
            std::make_shared<NamedObs<StateVectorT>>("PauliZ",
                                                     std::vector<size_t>{0}),
            std::make_shared<NamedObs<StateVectorT>>("PauliZ",
                                                     std::vector<size_t>{1}));
        auto obs2 = std::make_shared<HermitianObs<StateVectorT>>(
            std::vector<std::complex<double>>{1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1,
                                              0, 0, 0, 0, 1},
            std::vector<size_t>{0, 1});

        auto ops = OpsData<StateVectorT>(
            {"RX", "RX", "RX"}, {{param[0]}, {param[1]}, {param[2]}},
            {{0}, {1}, {2}}, {false, false, false});

        JacobianData<StateVectorT> tape1{
            param.size(), psi.getLength(), psi.getData(), {obs1}, ops, tp};

        JacobianData<StateVectorT> tape2{
            param.size(), psi.getLength(), psi.getData(), {obs2}, ops, tp};

        adj.adjointJacobian(std::span{jacobian1}, tape1, psi, true);
        adj.adjointJacobian(std::span{jacobian2}, tape2, psi, true);

        CHECK((jacobian1 == PLApprox(jacobian2).margin(1e-7)));
    }
}
