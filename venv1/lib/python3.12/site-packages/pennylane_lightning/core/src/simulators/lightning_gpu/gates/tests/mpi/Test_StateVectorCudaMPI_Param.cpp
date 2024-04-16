// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>
#include <mpi.h>

#include "cuGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

#include "StateVectorCudaMPI.hpp"
#include "StateVectorCudaManaged.hpp"

#include "MPIManager.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane;
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningGPU::MPI;

#define num_qubits 8
#define lsb_1qbit                                                              \
    { 0 }
#define msb_1qbit                                                              \
    { num_qubits - 1 }

#define lsb_2qbit                                                              \
    { 0, 1 }
#define msb_2qbit                                                              \
    { num_qubits - 2, num_qubits - 1 }
#define mlsb_2qbit                                                             \
    { 0, num_qubits - 1 }

#define lsb_3qbit                                                              \
    { 0, 1, 2 }
#define msb_3qbit                                                              \
    { num_qubits - 3, num_qubits - 2, num_qubits - 1 }
#define mlsb_3qbit                                                             \
    { 0, num_qubits - 2, num_qubits - 1 }

#define lsb_4qbit                                                              \
    { 0, 1, 2, 3 }
#define msb_4qbit                                                              \
    { num_qubits - 4, num_qubits - 3, num_qubits - 2, num_qubits - 1 }
#define mlsb_4qbit                                                             \
    { 0, 1, num_qubits - 2, num_qubits - 1 }

#define angle_1param                                                           \
    { 0.4 }
#define angle_3param                                                           \
    { 0.4, 0.3, 0.2 }

#define PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, NUM_QUBITS, GATE_METHOD,       \
                                      GATE_NAME, WIRE, ANGLE)                  \
    {                                                                          \
        const bool adjoint = GENERATE(true, false);                            \
        using cp_t = std::complex<TestType>;                                   \
        using PrecisionT = TestType;                                           \
        MPIManager mpi_manager(MPI_COMM_WORLD);                                \
        REQUIRE(mpi_manager.getSize() == 2);                                   \
        size_t mpi_buffersize = 1;                                             \
        size_t nGlobalIndexBits =                                              \
            std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;    \
        size_t nLocalIndexBits = NUM_QUBITS - nGlobalIndexBits;                \
        size_t subSvLength = 1 << nLocalIndexBits;                             \
        size_t svLength = 1 << NUM_QUBITS;                                     \
        mpi_manager.Barrier();                                                 \
        std::vector<cp_t> expected_sv(svLength);                               \
        std::vector<cp_t> local_state(subSvLength);                            \
        std::mt19937 re{1337};                                                 \
        auto init_sv =                                                         \
            Pennylane::Util::createRandomStateVectorData<PrecisionT>(          \
                re, (NUM_QUBITS));                                             \
        mpi_manager.Scatter(init_sv.data(), local_state.data(), subSvLength,   \
                            0);                                                \
        mpi_manager.Barrier();                                                 \
        int nDevices = 0;                                                      \
        cudaGetDeviceCount(&nDevices);                                         \
        REQUIRE(nDevices >= 2);                                                \
        int deviceId = mpi_manager.getRank() % nDevices;                       \
        cudaSetDevice(deviceId);                                               \
        DevTag<int> dt_local(deviceId, 0);                                     \
        mpi_manager.Barrier();                                                 \
        SECTION("Apply directly") {                                            \
            SECTION("Operation on target") {                                   \
                StateVectorCudaMPI<TestType> sv(                               \
                    mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,   \
                    nLocalIndexBits);                                          \
                sv.CopyHostDataToGpu(local_state, false);                      \
                sv.GATE_METHOD(WIRE, adjoint, ANGLE);                          \
                sv.CopyGpuDataToHost(local_state.data(), subSvLength);         \
                StateVectorCudaManaged<TestType> svdat{init_sv.data(),         \
                                                       svLength};              \
                if (mpi_manager.getRank() == 0) {                              \
                    svdat.GATE_METHOD(WIRE, adjoint, ANGLE);                   \
                    svdat.CopyGpuDataToHost(expected_sv.data(), svLength);     \
                }                                                              \
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);  \
                CHECK(local_state ==                                           \
                      Pennylane::Util::approx(expected_local_sv));             \
            }                                                                  \
        }                                                                      \
        SECTION("Apply using dispatcher") {                                    \
            SECTION("Operation on target") {                                   \
                StateVectorCudaMPI<TestType> sv(                               \
                    mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,   \
                    nLocalIndexBits);                                          \
                sv.CopyHostDataToGpu(local_state, false);                      \
                sv.applyOperation(GATE_NAME, WIRE, adjoint, ANGLE);            \
                sv.CopyGpuDataToHost(local_state.data(), subSvLength);         \
                StateVectorCudaManaged<TestType> svdat{init_sv.data(),         \
                                                       svLength};              \
                if (mpi_manager.getRank() == 0) {                              \
                    svdat.applyOperation(GATE_NAME, WIRE, adjoint, ANGLE);     \
                    svdat.CopyGpuDataToHost(expected_sv.data(), svLength);     \
                }                                                              \
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);  \
                CHECK(local_state ==                                           \
                      Pennylane::Util::approx(expected_local_sv));             \
            }                                                                  \
        }                                                                      \
    }

TEMPLATE_TEST_CASE("StateVectorCudaMPI::RX", "[StateVectorCudaMPI_Param]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyRX, "RX",
                                  lsb_1qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyRX, "RX",
                                  msb_1qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::RY", "[StateVectorCudaMPI_Param]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyRY, "RY",
                                  lsb_1qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyRY, "RY",
                                  msb_1qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::RZ", "[StateVectorCudaMPI_Param]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyRZ, "RZ",
                                  lsb_1qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyRZ, "RZ",
                                  msb_1qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::PhaseShift",
                   "[StateVectorCudaMPI_Param]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyPhaseShift,
                                  "PhaseShift", lsb_1qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyPhaseShift,
                                  "PhaseShift", msb_1qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::Rot", "[StateVectorCudaMPI_Param]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyRot, "Rot",
                                  lsb_1qbit, angle_3param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyRot, "Rot",
                                  msb_1qbit, angle_3param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::IsingXX", "[StateVectorCudaMPI_Param]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingXX, "IsingXX",
                                  lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingXX, "IsingXX",
                                  mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingXX, "IsingXX",
                                  msb_2qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::IsingXY", "[StateVectorCudaMPI_Param]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingXY, "IsingXY",
                                  lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingXY, "IsingXY",
                                  mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingXY, "IsingXY",
                                  msb_2qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::IsingYY", "[StateVectorCudaMPI_Param]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingYY, "IsingYY",
                                  lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingYY, "IsingYY",
                                  mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingYY, "IsingYY",
                                  msb_2qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::IsingZZ", "[StateVectorCudaMPI_Param]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingZZ, "IsingZZ",
                                  lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingZZ, "IsingZZ",
                                  mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyIsingZZ, "IsingZZ",
                                  msb_2qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::ControlledPhaseShift",
                   "[StateVectorCudaMPI_Param]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyControlledPhaseShift, "ControlledPhaseShift",
        lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyControlledPhaseShift, "ControlledPhaseShift",
        mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyControlledPhaseShift, "ControlledPhaseShift",
        msb_2qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::CRX", "[StateVectorCudaMPI_Param]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRX, "CRX",
                                  lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRX, "CRX",
                                  mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRX, "CRX",
                                  msb_2qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::CRY", "[StateVectorCudaMPI_Param]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRY, "CRY",
                                  lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRY, "CRY",
                                  mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRY, "CRY",
                                  msb_2qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::CRZ", "[StateVectorCudaMPI_Param]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRZ, "CRZ",
                                  lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRZ, "CRZ",
                                  mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRZ, "CRZ",
                                  msb_2qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::CRot", "[StateVectorCudaMPI_Param]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRot, "CRot",
                                  lsb_2qbit, angle_3param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRot, "CRot",
                                  mlsb_2qbit, angle_3param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyCRot, "CRot",
                                  msb_2qbit, angle_3param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::MultiRZ", "[StateVectorCudaMPI_Param]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyMultiRZ, "MultiRZ",
                                  lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyMultiRZ, "MultiRZ",
                                  mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyMultiRZ, "MultiRZ",
                                  msb_2qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::SingleExcitation",
                   "[StateVectorCudaMPI_Param]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applySingleExcitation,
                                  "SingleExcitation", lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applySingleExcitation,
                                  "SingleExcitation", mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applySingleExcitation,
                                  "SingleExcitation", msb_2qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::SingleExcitationMinus",
                   "[StateVectorCudaMPI_Param]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applySingleExcitationMinus,
        "SingleExcitationMinus", lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applySingleExcitationMinus,
        "SingleExcitationMinus", mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applySingleExcitationMinus,
        "SingleExcitationMinus", msb_2qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::SingleExcitationPlus",
                   "[StateVectorCudaMPI_Param]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applySingleExcitationPlus, "SingleExcitationPlus",
        lsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applySingleExcitationPlus, "SingleExcitationPlus",
        mlsb_2qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applySingleExcitationPlus, "SingleExcitationPlus",
        msb_2qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::DoubleExcitation",
                   "[StateVectorCudaMPI_Param]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyDoubleExcitation,
                                  "DoubleExcitation", lsb_4qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyDoubleExcitation,
                                  "DoubleExcitation", mlsb_4qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(TestType, num_qubits, applyDoubleExcitation,
                                  "DoubleExcitation", msb_4qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::DoubleExcitationMinus",
                   "[StateVectorCudaMPI_Param]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyDoubleExcitationMinus,
        "DoubleExcitationMinus", lsb_4qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyDoubleExcitationMinus,
        "DoubleExcitationMinus", mlsb_4qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyDoubleExcitationMinus,
        "DoubleExcitationMinus", msb_4qbit, angle_1param);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::DoubleExcitationPlus",
                   "[StateVectorCudaMPI_Param]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyDoubleExcitationPlus, "DoubleExcitationPlus",
        lsb_4qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyDoubleExcitationPlus, "DoubleExcitationPlus",
        mlsb_4qbit, angle_1param);
    PLGPU_MPI_TEST_GATE_OPS_PARAM(
        TestType, num_qubits, applyDoubleExcitationPlus, "DoubleExcitationPlus",
        msb_4qbit, angle_1param);
}

TEMPLATE_TEST_CASE("LightningGPUMPI:applyOperation", "[LightningGPUMPI_Param]",
                   float, double) {
    using StateVectorT = StateVectorCudaMPI<TestType>;
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

    SECTION("Catch failures caused by unsupported named gates") {
        std::string obs = "paulix";
        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.initSV();
        PL_CHECK_THROWS_MATCHES(sv.applyOperation(obs, {0}), LightningException,
                                "Currently unsupported gate: paulix");
    }
}