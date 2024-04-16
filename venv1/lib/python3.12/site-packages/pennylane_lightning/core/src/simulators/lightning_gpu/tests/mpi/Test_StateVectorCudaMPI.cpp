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
#include <limits> // numeric_limits
#include <random>
#include <type_traits>
#include <vector>

#include <catch2/catch.hpp>

#include "DevTag.hpp"
#include "MPIManager.hpp"
#include "StateVectorCudaMPI.hpp"
#include "TestHelpers.hpp" // createRandomStateVectorData
#include "mpi.h"

/**
 * @file
 *  Tests for functionality for the class StateVectorCudaMPI.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningGPU::MPI;
using namespace Pennylane::Util;

using Pennylane::Util::isApproxEqual;
using Pennylane::Util::randomUnitary;

std::mt19937_64 re{1337};
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("StateVectorCudaMPI::Constructibility",
                   "[Default Constructibility]", StateVectorCudaMPI<>) {
    SECTION("StateVectorBackend<>") {
        REQUIRE(!std::is_constructible_v<TestType>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorCudaMPI::Constructibility",
                           "[General Constructibility]", (StateVectorCudaMPI),
                           (float, double)) {
    using StateVectorT = TestType;
    using CFP_t = typename StateVectorT::CFP_t;

    SECTION("StateVectorBackend<TestType>") {
        REQUIRE(!std::is_constructible_v<StateVectorT>);
    }
    SECTION("StateVectorBackend<TestType> {MPIManager, DevTag<int>, size_t, "
            "size_t, size_t}") {
        REQUIRE(std::is_constructible_v<StateVectorT, MPIManager, DevTag<int>,
                                        size_t, size_t, size_t>);
    }
    SECTION("StateVectorBackend<TestType> {MPI_Comm, DevTag<int>, size_t, "
            "size_t, size_t}") {
        REQUIRE(std::is_constructible_v<StateVectorT, MPI_Comm, DevTag<int>,
                                        size_t, size_t, size_t>);
    }
    SECTION(
        "StateVectorBackend<TestType> {DevTag<int>, size_t, size_t, size_t}") {
        REQUIRE(std::is_constructible_v<StateVectorT, DevTag<int>, size_t,
                                        size_t, size_t>);
    }
    SECTION(
        "StateVectorBackend<TestType> {DevTag<int>, size_t, size_t, CFP_t}") {
        REQUIRE(std::is_constructible_v<StateVectorT, DevTag<int>, size_t,
                                        size_t, CFP_t *>);
    }
    SECTION("StateVectorBackend<TestType> {DevTag<int>, size_t, size_t}") {
        REQUIRE(
            std::is_constructible_v<StateVectorT, DevTag<int>, size_t, size_t>);
    }
    SECTION(
        "StateVectorBackend<TestType> {const StateVectorBackend<TestType>&}") {
        REQUIRE(std::is_copy_constructible_v<StateVectorT>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorCudaMPI::applyMatrix with a std::vector",
                           "[applyMatrix]", (StateVectorCudaMPI),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t mpi_buffersize = 128;
    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
    size_t subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<ComplexT> local_state(subSvLength);

    auto init_sv = Pennylane::Util::createRandomStateVectorData<PrecisionT>(
        re, num_qubits);

    mpi_manager.Scatter(init_sv.data(), local_state.data(), subSvLength, 0);
    mpi_manager.Barrier();

    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    mpi_manager.Barrier();

    SECTION("Test wrong matrix size") {
        std::vector<ComplexT> m(7, 0.0);

        StateVectorT state_vector(mpi_manager, dt_local, mpi_buffersize,
                                  nGlobalIndexBits, nLocalIndexBits);
        state_vector.CopyHostDataToGpu(local_state, false);
        REQUIRE_THROWS_WITH(
            state_vector.applyMatrix(m, {0, 1}),
            Catch::Contains(
                "The size of matrix does not match with the given"));
    }
    SECTION("Test wrong number of wires") {
        std::vector<ComplexT> m(8, 0.0);

        StateVectorT state_vector(mpi_manager, dt_local, mpi_buffersize,
                                  nGlobalIndexBits, nLocalIndexBits);
        state_vector.CopyHostDataToGpu(local_state, false);
        REQUIRE_THROWS_WITH(
            state_vector.applyMatrix(m, {0}),
            Catch::Contains(
                "The size of matrix does not match with the given"));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorCudaMPI::applyMatrix with a pointer",
                           "[applyMatrix]", (StateVectorCudaMPI),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t mpi_buffersize = 1;
    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
    size_t subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<ComplexT> local_state(subSvLength);

    auto init_sv = Pennylane::Util::createRandomStateVectorData<PrecisionT>(
        re, num_qubits);

    mpi_manager.Scatter(init_sv.data(), local_state.data(), subSvLength, 0);
    mpi_manager.Barrier();

    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    mpi_manager.Barrier();

    SECTION("Test wrong matrix") {
        std::vector<ComplexT> m(8, 0.0);

        StateVectorT state_vector(mpi_manager, dt_local, mpi_buffersize,
                                  nGlobalIndexBits, nLocalIndexBits);
        state_vector.CopyHostDataToGpu(local_state, false);
        REQUIRE_THROWS_WITH(state_vector.applyMatrix(m.data(), {}),
                            Catch::Contains("must be larger than 0"));
    }

    SECTION("Test a matrix represent PauliX") {
        std::vector<ComplexT> m = {
            {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

        StateVectorT state_vector(mpi_manager, dt_local, mpi_buffersize,
                                  nGlobalIndexBits, nLocalIndexBits);
        state_vector.CopyHostDataToGpu(local_state, false);
        StateVectorT state_vector_ref(mpi_manager, dt_local, mpi_buffersize,
                                      nGlobalIndexBits, nLocalIndexBits);
        state_vector_ref.CopyHostDataToGpu(local_state, false);
        state_vector.applyMatrix(m.data(), {1});
        state_vector_ref.applyPauliX({1}, false);

        CHECK(state_vector.getDataVector() ==
              Pennylane::Util::approx(state_vector_ref.getDataVector()));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorCudaMPI::applyOperations",
                           "[applyOperations invalid arguments]",
                           (StateVectorCudaMPI), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    const size_t num_qubits = 4;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t mpi_buffersize = 1;
    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
    size_t subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<ComplexT> local_state(subSvLength);

    auto init_sv = Pennylane::Util::createRandomStateVectorData<PrecisionT>(
        re, num_qubits);

    mpi_manager.Scatter(init_sv.data(), local_state.data(), subSvLength, 0);
    mpi_manager.Barrier();

    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    mpi_manager.Barrier();

    SECTION("Test invalid arguments without parameters") {
        StateVectorT state_vector(mpi_manager, dt_local, mpi_buffersize,
                                  nGlobalIndexBits, nLocalIndexBits);
        state_vector.CopyHostDataToGpu(local_state, false);

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"PauliX", "PauliY"}, {{0}},
                                         {false, false}),
            LightningException, "must all be equal"); // invalid wires
        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"PauliX", "PauliY"}, {{0}, {1}},
                                         {false}),
            LightningException, "must all be equal"); // invalid inverse
        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperation("PauliX", std::vector<std::size_t>{0},
                                        std::vector<bool>{false},
                                        std::vector<std::size_t>{1}, false,
                                        {0.0}, std::vector<ComplexT>{}),
            LightningException,
            "Controlled kernels not implemented."); // invalid controlled_wires
        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperation("PauliX", {}, std::vector<bool>{false},
                                        std::vector<std::size_t>{1}, false,
                                        {0.0}, std::vector<ComplexT>{}),
            LightningException,
            "`controlled_wires` must have the same size "
            "as"); // invalid controlled_wires
    }

    SECTION("Test invalid arguments with parameters") {
        StateVectorT state_vector(mpi_manager, dt_local, mpi_buffersize,
                                  nGlobalIndexBits, nLocalIndexBits);
        state_vector.CopyHostDataToGpu(local_state, false);

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"RX", "RY"}, {{0}}, {false, false},
                                         {{0.0}, {0.0}}),
            LightningException, "must all be equal"); // invalid wires

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"RX", "RY"}, {{0}, {1}}, {false},
                                         {{0.0}, {0.0}}),
            LightningException, "must all be equal"); // invalid wires

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"RX", "RY"}, {{0}, {1}},
                                         {false, false}, {{0.0}}),
            LightningException, "must all be equal"); // invalid parameters
    }
}