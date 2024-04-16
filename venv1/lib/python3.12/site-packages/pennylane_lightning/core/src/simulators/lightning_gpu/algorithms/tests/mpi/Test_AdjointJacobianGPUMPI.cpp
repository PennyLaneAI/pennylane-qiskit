// Copyright 2022-2023 Xanadu Quantum Technologies Inc.
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
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <catch2/catch.hpp>

#include "AdjointJacobianGPU.hpp"
#include "JacobianData.hpp"
#include "StateVectorCudaManaged.hpp"

#include "AdjointJacobianGPUMPI.hpp"
#include "JacobianDataMPI.hpp"
#include "MPIManager.hpp"
#include "StateVectorCudaMPI.hpp"

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
TEMPLATE_TEST_CASE("AdjointJacobianGPUMPI::AdjointJacobianGPUMPI",
                   "[AdjointJacobianGPUMPI]", float, double) {
    SECTION("AdjointJacobianGPUMPI<TestType> {}") {
        REQUIRE(std::is_constructible<
                AdjointJacobianMPI<StateVectorCudaMPI<TestType>>>::value);
    }
}

TEST_CASE("AdjointJacobianGPUMPI::adjointJacobianMPI Op=RX, Obs=[Z,Z]",
          "[AdjointJacobianGPUMPI]") {
    using StateVectorT = StateVectorCudaMPI<double>;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    AdjointJacobianMPI<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0};

    const size_t num_qubits = 2;
    const size_t num_obs = 2;
    std::vector<double> jacobian(num_obs * tp.size(), 0);
    std::vector<double> jacobian_serial(num_obs * tp.size(), 0);

    size_t mpi_buffersize = 1;

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    {
        StateVectorT psi(mpi_manager, dt_local, mpi_buffersize,
                         nGlobalIndexBits, nLocalIndexBits);
        psi.initSV();

        const auto obs1 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});

        auto ops = OpsData<StateVectorT>({"RX"}, {{param[0]}}, {{0}}, {false});

        JacobianDataMPI<StateVectorT> tape{
            param.size(), psi, {obs1, obs2}, ops, tp};

        adj.adjointJacobian(std::span{jacobian}, tape, psi, true);
        adj.adjointJacobian_serial(std::span{jacobian_serial}, tape, true);

        CAPTURE(jacobian);
        CAPTURE(jacobian_serial);
        CHECK(-sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
        CHECK(0.0 == Approx(jacobian[tp.size()]).margin(1e-7));
        CHECK(-sin(param[0]) == Approx(jacobian_serial[0]).margin(1e-7));
        CHECK(0.0 == Approx(jacobian_serial[tp.size()]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobianGPUMPI::adjointJacobianMPI Op=[QubitStateVector, "
          "StatePrep, BasisState], Obs=[Z,Z]",
          "[AdjointJacobianGPUMPI]") {
    const std::string test_ops =
        GENERATE("QubitStateVector", "StatePrep", "BasisState");
    using StateVectorT = StateVectorCudaMPI<double>;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    AdjointJacobianMPI<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0};

    const size_t num_qubits = 4;
    const size_t num_obs = 2;
    std::vector<double> jacobian(num_obs * tp.size(), 0);
    std::vector<double> jacobian_serial(num_obs * tp.size(), 0);
    std::vector<double> jacobian_ref(num_obs * tp.size(), 0);

    size_t mpi_buffersize = 1;

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    {
        StateVectorT psi(mpi_manager, dt_local, mpi_buffersize,
                         nGlobalIndexBits, nLocalIndexBits);
        psi.initSV();

        const auto obs1 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});

        auto ops =
            OpsData<StateVectorT>({test_ops}, {{param[0]}}, {{0}}, {false});

        JacobianDataMPI<StateVectorT> tape{
            param.size(), psi, {obs1, obs2}, ops, tp};

        adj.adjointJacobian(std::span{jacobian}, tape, psi, false);
        adj.adjointJacobian_serial(std::span{jacobian_serial}, tape, false);

        CAPTURE(jacobian);
        CAPTURE(jacobian_serial);
        CHECK(jacobian == Pennylane::Util::approx(jacobian_ref).margin(1e-7));
        CHECK(jacobian_serial ==
              Pennylane::Util::approx(jacobian_ref).margin(1e-7));
    }
}

TEST_CASE(
    "AdjointJacobianGPUMPI::AdjointJacobianGPUMPI Op=[RX,RX,RX], Obs=[Z,Z,Z]",
    "[AdjointJacobianGPUMPI]") {
    using StateVectorT = StateVectorCudaMPI<double>;
    AdjointJacobianMPI<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 1, 2};

    const size_t num_qubits = 3;
    const size_t num_obs = 3;
    std::vector<double> jacobian(num_obs * tp.size(), 0);
    std::vector<double> jacobian_serial(num_obs * tp.size(), 0);

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t mpi_buffersize = 1;

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    {
        StateVectorT psi(mpi_manager, dt_local, mpi_buffersize,
                         nGlobalIndexBits, nLocalIndexBits);
        psi.initSV();

        const auto obs1 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});
        const auto obs3 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliZ", std::vector<size_t>{2});

        auto ops = OpsData<StateVectorT>(
            {"RX", "RX", "RX"}, {{param[0]}, {param[1]}, {param[2]}},
            {{0}, {1}, {2}}, {false, false, false});

        JacobianDataMPI<StateVectorT> tape{
            param.size(), psi, {obs1, obs2, obs3}, ops, tp};

        adj.adjointJacobian(std::span{jacobian}, tape, psi, true);
        adj.adjointJacobian_serial(std::span{jacobian_serial}, tape, true);

        CAPTURE(jacobian);
        CAPTURE(jacobian_serial);

        // Computed with parameter shift
        CHECK(-sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
        CHECK(-sin(param[1]) == Approx(jacobian[1 + tp.size()]).margin(1e-7));
        CHECK(-sin(param[2]) ==
              Approx(jacobian[2 + 2 * tp.size()]).margin(1e-7));

        CHECK(-sin(param[0]) == Approx(jacobian_serial[0]).margin(1e-7));
        CHECK(-sin(param[1]) ==
              Approx(jacobian_serial[1 + tp.size()]).margin(1e-7));
        CHECK(-sin(param[2]) ==
              Approx(jacobian_serial[2 + 2 * tp.size()]).margin(1e-7));
    }
}

TEST_CASE(
    "AdjointJacobianGPUMPI::AdjointJacobianGPUMPI Op=[RX,RX,RX], Obs=[Z,Z,Z],"
    "TParams=[0,2]",
    "[AdjointJacobianGPUMPI]") {
    using StateVectorT = StateVectorCudaMPI<double>;
    AdjointJacobianMPI<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 2};

    const size_t num_qubits = 3;
    const size_t num_obs = 3;
    std::vector<double> jacobian(num_obs * tp.size(), 0);
    std::vector<double> jacobian_serial(num_obs * tp.size(), 0);

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t mpi_buffersize = 1;

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    {
        StateVectorT psi(mpi_manager, dt_local, mpi_buffersize,
                         nGlobalIndexBits, nLocalIndexBits);
        psi.initSV();

        const auto obs1 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});
        const auto obs3 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliZ", std::vector<size_t>{2});
        auto ops = OpsData<StateVectorT>(
            {"RX", "RX", "RX"}, {{param[0]}, {param[1]}, {param[2]}},
            {{0}, {1}, {2}}, {false, false, false});

        JacobianDataMPI<StateVectorT> tape{
            param.size(), psi, {obs1, obs2, obs3}, ops, tp};

        adj.adjointJacobian(std::span{jacobian}, tape, psi, true);
        adj.adjointJacobian_serial(std::span{jacobian_serial}, tape, true);

        CAPTURE(jacobian);
        CAPTURE(jacobian_serial);

        // Computed with parameter shift
        CHECK(-sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
        CHECK(0 == Approx(jacobian[1 + tp.size()]).margin(1e-7));
        CHECK(-sin(param[2]) ==
              Approx(jacobian[1 + 2 * tp.size()]).margin(1e-7));

        CHECK(-sin(param[0]) == Approx(jacobian_serial[0]).margin(1e-7));
        CHECK(0 == Approx(jacobian_serial[1 + tp.size()]).margin(1e-7));
        CHECK(-sin(param[2]) ==
              Approx(jacobian_serial[1 + 2 * tp.size()]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobianGPUMPI::adjointJacobian Op=[RX,RX,RX], Obs=[ZZZ]",
          "[AdjointJacobianGPUMPI]") {
    using StateVectorT = StateVectorCudaMPI<double>;
    AdjointJacobianMPI<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 1, 2};

    const size_t num_qubits = 3;
    const size_t num_obs = 1;
    std::vector<double> jacobian(num_obs * tp.size(), 0);
    std::vector<double> jacobian_serial(num_obs * tp.size(), 0);

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t mpi_buffersize = 1;

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    {
        StateVectorT psi(mpi_manager, dt_local, mpi_buffersize,
                         nGlobalIndexBits, nLocalIndexBits);
        psi.initSV();

        const auto obs = std::make_shared<TensorProdObsMPI<StateVectorT>>(
            std::make_shared<NamedObsMPI<StateVectorT>>("PauliZ",
                                                        std::vector<size_t>{0}),
            std::make_shared<NamedObsMPI<StateVectorT>>("PauliZ",
                                                        std::vector<size_t>{1}),
            std::make_shared<NamedObsMPI<StateVectorT>>(
                "PauliZ", std::vector<size_t>{2}));
        auto ops = OpsData<StateVectorT>(
            {"RX", "RX", "RX"}, {{param[0]}, {param[1]}, {param[2]}},
            {{0}, {1}, {2}}, {false, false, false});

        JacobianDataMPI<StateVectorT> tape{param.size(), psi, {obs}, ops, tp};

        adj.adjointJacobian(std::span{jacobian}, tape, psi, true);
        adj.adjointJacobian_serial(std::span{jacobian_serial}, tape, true);

        CAPTURE(jacobian);
        CAPTURE(jacobian_serial);

        // Computed with parameter shift
        CHECK(-0.1755096592645253 == Approx(jacobian[0]).margin(1e-7));
        CHECK(0.26478810666384334 == Approx(jacobian[1]).margin(1e-7));
        CHECK(-0.6312451595102775 == Approx(jacobian[2]).margin(1e-7));

        CHECK(-0.1755096592645253 == Approx(jacobian_serial[0]).margin(1e-7));
        CHECK(0.26478810666384334 == Approx(jacobian_serial[1]).margin(1e-7));
        CHECK(-0.6312451595102775 == Approx(jacobian_serial[2]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobianGPUMPI::adjointJacobian Op=Mixed, Obs=[XXX]",
          "[AdjointJacobianGPUMPI]") {
    using StateVectorT = StateVectorCudaMPI<double>;
    AdjointJacobianMPI<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 1, 2, 3, 4, 5};

    const size_t num_qubits = 3;
    const size_t num_obs = 1;
    std::vector<double> jacobian(num_obs * tp.size(), 0);
    std::vector<double> jacobian_serial(num_obs * tp.size(), 0);

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t mpi_buffersize = 1;

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    {
        StateVectorT psi(mpi_manager, dt_local, mpi_buffersize,
                         nGlobalIndexBits, nLocalIndexBits);
        psi.initSV();

        const auto obs = std::make_shared<TensorProdObsMPI<StateVectorT>>(
            std::make_shared<NamedObsMPI<StateVectorT>>("PauliX",
                                                        std::vector<size_t>{0}),
            std::make_shared<NamedObsMPI<StateVectorT>>("PauliX",
                                                        std::vector<size_t>{1}),
            std::make_shared<NamedObsMPI<StateVectorT>>(
                "PauliX", std::vector<size_t>{2}));
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

        JacobianDataMPI<StateVectorT> tape{param.size(), psi, {obs}, ops, tp};

        adj.adjointJacobian(std::span{jacobian}, tape, psi, true);
        adj.adjointJacobian_serial(std::span{jacobian_serial}, tape, true);

        CAPTURE(jacobian);
        CAPTURE(jacobian_serial);

        // Computed with PennyLane using default.qubit.adjoint_jacobian
        CHECK(0.0 == Approx(jacobian[0]).margin(1e-7));
        CHECK(-0.674214427 == Approx(jacobian[1]).margin(1e-7));
        CHECK(0.275139672 == Approx(jacobian[2]).margin(1e-7));
        CHECK(0.275139672 == Approx(jacobian[3]).margin(1e-7));
        CHECK(-0.0129093062 == Approx(jacobian[4]).margin(1e-7));
        CHECK(0.323846156 == Approx(jacobian[5]).margin(1e-7));

        CHECK(0.0 == Approx(jacobian_serial[0]).margin(1e-7));
        CHECK(-0.674214427 == Approx(jacobian_serial[1]).margin(1e-7));
        CHECK(0.275139672 == Approx(jacobian_serial[2]).margin(1e-7));
        CHECK(0.275139672 == Approx(jacobian_serial[3]).margin(1e-7));
        CHECK(-0.0129093062 == Approx(jacobian_serial[4]).margin(1e-7));
        CHECK(0.323846156 == Approx(jacobian_serial[5]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobianGPU::AdjointJacobianGPUMPI Op=[RX,RX,RX], "
          "Obs=Ham[Z0+Z1+Z2], "
          "TParams=[0,2]",
          "[AdjointJacobianGPU]") {
    using StateVectorT = StateVectorCudaMPI<double>;
    AdjointJacobianMPI<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 2};

    const size_t num_qubits = 3;
    const size_t num_obs = 1;
    std::vector<double> jacobian(num_obs * tp.size(), 0);
    std::vector<double> jacobian_serial(num_obs * tp.size(), 0);

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t mpi_buffersize = 1;

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);

    {
        StateVectorT psi(mpi_manager, dt_local, mpi_buffersize,
                         nGlobalIndexBits, nLocalIndexBits);
        psi.initSV();

        auto obs1 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliZ", std::vector<size_t>{0});
        auto obs2 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});
        auto obs3 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliZ", std::vector<size_t>{2});

        auto ham = HamiltonianMPI<StateVectorT>::create({0.47, 0.32, 0.96},
                                                        {obs1, obs2, obs3});

        auto ops = OpsData<StateVectorT>(
            {"RX", "RX", "RX"}, {{param[0]}, {param[1]}, {param[2]}},
            {{0}, {1}, {2}}, {false, false, false});

        JacobianDataMPI<StateVectorT> tape{param.size(), psi, {ham}, ops, tp};

        adj.adjointJacobian(std::span{jacobian}, tape, psi, true);
        adj.adjointJacobian_serial(std::span{jacobian_serial}, tape, true);

        CAPTURE(jacobian);
        CAPTURE(jacobian_serial);

        CHECK((-0.47 * sin(param[0]) == Approx(jacobian[0]).margin(1e-7)));
        CHECK((-0.96 * sin(param[2]) == Approx(jacobian[1]).margin(1e-7)));

        CHECK(
            (-0.47 * sin(param[0]) == Approx(jacobian_serial[0]).margin(1e-7)));
        CHECK(
            (-0.96 * sin(param[2]) == Approx(jacobian_serial[1]).margin(1e-7)));
    }
}

TEST_CASE("AdjointJacobianGPU::AdjointJacobianGPU Test HermitianObs",
          "[AdjointJacobianGPU]") {
    using StateVectorT = StateVectorCudaMPI<double>;
    AdjointJacobianMPI<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0, 2};

    const size_t num_qubits = 3;
    const size_t num_obs = 1;

    std::vector<double> jacobian1(num_obs * tp.size(), 0);
    std::vector<double> jacobian2(num_obs * tp.size(), 0);

    std::vector<double> jacobian1_serial(num_obs * tp.size(), 0);
    std::vector<double> jacobian2_serial(num_obs * tp.size(), 0);

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t mpi_buffersize = 1;

    int nGlobalIndexBits =
        std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) - 1;
    int nLocalIndexBits = num_qubits - nGlobalIndexBits;
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);

    {
        StateVectorT psi(mpi_manager, dt_local, mpi_buffersize,
                         nGlobalIndexBits, nLocalIndexBits);
        psi.initSV();

        auto obs1 = std::make_shared<TensorProdObsMPI<StateVectorT>>(
            std::make_shared<NamedObsMPI<StateVectorT>>("PauliZ",
                                                        std::vector<size_t>{0}),
            std::make_shared<NamedObsMPI<StateVectorT>>(
                "PauliZ", std::vector<size_t>{1}));
        auto obs2 = std::make_shared<HermitianObsMPI<StateVectorT>>(
            std::vector<std::complex<double>>{1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1,
                                              0, 0, 0, 0, 1},
            std::vector<size_t>{0, 1});

        auto ops = OpsData<StateVectorT>(
            {"RX", "RX", "RX"}, {{param[0]}, {param[1]}, {param[2]}},
            {{0}, {1}, {2}}, {false, false, false});

        JacobianDataMPI<StateVectorT> tape1{param.size(), psi, {obs1}, ops, tp};

        JacobianDataMPI<StateVectorT> tape2{param.size(), psi, {obs2}, ops, tp};

        adj.adjointJacobian(std::span{jacobian1}, tape1, psi, true);
        adj.adjointJacobian(std::span{jacobian2}, tape2, psi, true);

        adj.adjointJacobian_serial(std::span{jacobian1_serial}, tape1, true);
        adj.adjointJacobian_serial(std::span{jacobian2_serial}, tape2, true);

        CHECK((jacobian1 == PLApprox(jacobian2).margin(1e-7)));
        CHECK((jacobian1 == PLApprox(jacobian1_serial).margin(1e-7)));
        CHECK((jacobian1_serial == PLApprox(jacobian2_serial).margin(1e-7)));
    }
}
