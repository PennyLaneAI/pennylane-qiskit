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
#include <complex>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "MeasurementsGPU.hpp"
#include "MeasurementsGPUMPI.hpp"
#include "ObservablesGPU.hpp"
#include "ObservablesGPUMPI.hpp"
#include "StateVectorCudaMPI.hpp"
#include "StateVectorCudaManaged.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU::Measures;
using namespace Pennylane::LightningGPU::Observables;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("Test variance of NamedObs", "[StateVectorCudaMPI_Var]",
                   float, double) {
    using StateVectorT = StateVectorCudaMPI<TestType>;
    const std::size_t num_qubits = 2;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t mpi_buffersize = 1;

    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;

    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    mpi_manager.Barrier();

    SECTION("var(PauliX[0])") {
        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.initSV();

        auto m = MeasurementsMPI(sv);

        sv.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});

        auto ob = NamedObsMPI<StateVectorT>("PauliX", {0});
        auto res = m.var(ob);
        auto expected = TestType(0.7572222074);
        CHECK(res == Approx(expected));
    }

    SECTION("var(PauliY[0])") {
        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.initSV();

        auto m = MeasurementsMPI(sv);

        sv.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});

        auto ob = NamedObsMPI<StateVectorT>("PauliY", {0});
        auto res = m.var(ob);
        auto expected = TestType(0.5849835715);
        CHECK(res == Approx(expected));
    }

    SECTION("var(PauliZ[1])") {
        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.initSV();

        auto m = MeasurementsMPI(sv);

        sv.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.7}, {0.7}, {0.5}, {0.5}});

        auto ob = NamedObsMPI<StateVectorT>("PauliZ", {1});
        auto res = m.var(ob);
        auto expected = TestType(0.4068672016);
        CHECK(res == Approx(expected));
    }
}

TEMPLATE_TEST_CASE("Test variance of HermitianObs", "[StateVectorCudaMPI_Var]",
                   double) {
    const std::size_t num_qubits = 3;
    using StateVectorT = StateVectorCudaMPI<TestType>;
    using ComplexT = typename StateVectorT::ComplexT;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t mpi_buffersize = 1;

    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;

    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    mpi_manager.Barrier();

    SECTION("Using var") {
        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.initSV();

        auto m = MeasurementsMPI(sv);

        sv.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}, {"RX"}, {"RY"}},
            {{0}, {0}, {1}, {1}, {2}, {2}},
            {{false}, {false}, {false}, {false}, {false}, {false}},
            {{0.7}, {0.7}, {0.5}, {0.5}, {0.3}, {0.3}});

        const TestType theta = M_PI / 2;
        const TestType c = std::cos(theta / 2);
        const TestType js = std::sin(-theta / 2);
        std::vector<ComplexT> matrix(16, 0);
        matrix[0] = c;
        matrix[1] = ComplexT{0, js};
        matrix[4] = ComplexT{0, js};
        matrix[5] = c;
        matrix[10] = ComplexT{1, 0};
        matrix[15] = ComplexT{1, 0};

        auto ob = HermitianObsMPI<StateVectorT>(matrix, {0, 2});
        auto res = m.var(ob);
        auto expected = TestType(0.4103533486);
        CHECK(res == Approx(expected));
    }
}

TEMPLATE_TEST_CASE("Test variance of TensorProdObs", "[StateVectorCudaMPI_Var]",
                   double) {
    using StateVectorT = StateVectorCudaMPI<TestType>;
    const std::size_t num_qubits = 3;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t mpi_buffersize = 1;

    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;

    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    mpi_manager.Barrier();

    SECTION("Using var") {
        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.initSV();

        auto m = MeasurementsMPI(sv);

        sv.applyOperations(
            {{"RX"}, {"RY"}, {"RX"}, {"RY"}}, {{0}, {0}, {1}, {1}},
            {{false}, {false}, {false}, {false}}, {{0.5}, {0.5}, {0.2}, {0.2}});

        auto X0 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliX", std::vector<size_t>{0});
        auto Z1 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});

        auto ob = TensorProdObsMPI<StateVectorT>::create({X0, Z1});
        auto res = m.var(*ob);
        auto expected = TestType(0.836679);
        CHECK(expected == Approx(res));
    }
}

TEMPLATE_TEST_CASE("Test variance of HamiltonianObs",
                   "[StateVectorCudaMPI_Var]", float, double) {
    using StateVectorT = StateVectorCudaMPI<TestType>;
    const std::size_t num_qubits = 3;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t mpi_buffersize = 1;

    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;

    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    mpi_manager.Barrier();

    SECTION("Using var") {
        std::vector<std::complex<TestType>> init_state{
            {0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1}, {0.1, 0.2},
            {0.2, 0.2}, {0.3, 0.3}, {0.3, 0.4}, {0.4, 0.5}};

        auto local_init_sv = mpi_manager.scatter(init_state, 0);
        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.CopyHostDataToGpu(local_init_sv.data(), local_init_sv.size(), false);

        mpi_manager.Barrier();

        auto m = MeasurementsMPI(sv);

        auto X0 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliX", std::vector<size_t>{0});
        auto Z1 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});

        auto ob = HamiltonianMPI<StateVectorT>::create({0.3, 0.5}, {X0, Z1});
        auto res = m.var(*ob);
        auto expected = TestType(0.224604);
        CHECK(expected == Approx(res));
    }
}
