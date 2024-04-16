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

#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include <catch2/catch.hpp>

#include "MPIManager.hpp"
#include "StateVectorCudaMPI.hpp"
#include "StateVectorCudaManaged.hpp"
#include "mpi.h"

#include "TestHelpers.hpp"

using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningGPU::MPI;
// NOTE: the scaling factors are implicitly included in the Adjoint Jacobian
// evaluation, so excluded from the matrices here.

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

#define PLGPU_MPI_TEST_GENERATOR(TestType, NUM_QUBITS, GEN_METHOD, GEN_NAME,   \
                                 WIRE)                                         \
    {                                                                          \
        using cp_t = std::complex<TestType>;                                   \
        using PrecisionT = TestType;                                           \
        MPIManager mpi_manager(MPI_COMM_WORLD);                                \
        REQUIRE(mpi_manager.getSize() == 2);                                   \
        size_t mpi_buffersize = 1;                                             \
        size_t nGlobalIndexBits =                                              \
            std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;    \
        size_t nLocalIndexBits = (NUM_QUBITS)-nGlobalIndexBits;                \
        size_t subSvLength = 1 << nLocalIndexBits;                             \
        size_t svLength = 1 << (NUM_QUBITS);                                   \
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
            SECTION("Operation on target wire") {                              \
                StateVectorCudaMPI<TestType> sv(                               \
                    mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,   \
                    nLocalIndexBits);                                          \
                sv.CopyHostDataToGpu(local_state, false);                      \
                sv.GEN_METHOD(WIRE, false);                                    \
                sv.CopyGpuDataToHost(local_state.data(),                       \
                                     static_cast<std::size_t>(subSvLength));   \
                                                                               \
                StateVectorCudaManaged<TestType> svdat{init_sv.data(),         \
                                                       svLength};              \
                if (mpi_manager.getRank() == 0) {                              \
                    svdat.GEN_METHOD(WIRE, false);                             \
                    svdat.CopyGpuDataToHost(expected_sv.data(), svLength);     \
                }                                                              \
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);  \
                CHECK(local_state ==                                           \
                      Pennylane::Util::approx(expected_local_sv));             \
            }                                                                  \
        }                                                                      \
        SECTION("Apply using dispatcher") {                                    \
            SECTION("Operation on target wire") {                              \
                StateVectorCudaMPI<TestType> sv(                               \
                    mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,   \
                    nLocalIndexBits);                                          \
                sv.CopyHostDataToGpu(local_state, false);                      \
                sv.applyGenerator(GEN_NAME, WIRE, false);                      \
                sv.CopyGpuDataToHost(local_state.data(),                       \
                                     static_cast<std::size_t>(subSvLength));   \
                StateVectorCudaManaged<TestType> svdat{init_sv.data(),         \
                                                       svLength};              \
                if (mpi_manager.getRank() == 0) {                              \
                    svdat.applyGenerator(GEN_NAME, WIRE, false);               \
                    svdat.CopyGpuDataToHost(expected_sv.data(), svLength);     \
                }                                                              \
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);  \
                CHECK(local_state ==                                           \
                      Pennylane::Util::approx(expected_local_sv));             \
            }                                                                  \
        }                                                                      \
    }

TEMPLATE_TEST_CASE("Generators::applyGeneratorRX", "[GateGenerators]", float,
                   double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorRX, "RX",
                             lsb_1qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorRX, "RX",
                             msb_1qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorRY", "[GateGenerators]", float,
                   double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorRY, "RY",
                             lsb_1qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorRY, "RY",
                             msb_1qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorRZ", "[GateGenerators]", float,
                   double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorRZ, "RZ",
                             lsb_1qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorRZ, "RZ",
                             msb_1qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorPhaseShift", "[GateGenerators]",
                   float, double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorPhaseShift,
                             "PhaseShift", lsb_1qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorPhaseShift,
                             "PhaseShift", msb_1qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorIsingXX", "[GateGenerators]",
                   float, double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorIsingXX,
                             "IsingXX", lsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorIsingXX,
                             "IsingXX", mlsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorIsingXX,
                             "IsingXX", msb_2qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorIsingXY", "[GateGenerators]",
                   float, double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorIsingXY,
                             "IsingXY", lsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorIsingXY,
                             "IsingXY", mlsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorIsingXY,
                             "IsingXY", msb_2qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorIsingYY", "[GateGenerators]",
                   float, double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorIsingYY,
                             "IsingYY", lsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorIsingYY,
                             "IsingYY", mlsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorIsingYY,
                             "IsingYY", msb_2qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorIsingZZ", "[GateGenerators]",
                   float, double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorIsingZZ,
                             "IsingZZ", lsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorIsingZZ,
                             "IsingZZ", mlsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorIsingZZ,
                             "IsingZZ", msb_2qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorControlledPhaseShift",
                   "[GateGenerators]", float, double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorControlledPhaseShift,
                             "ControlledPhaseShift", lsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorControlledPhaseShift,
                             "ControlledPhaseShift", mlsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorControlledPhaseShift,
                             "ControlledPhaseShift", msb_2qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorCRX", "[GateGenerators]", float,
                   double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorCRX, "CRX",
                             lsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorCRX, "CRX",
                             mlsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorCRX, "CRX",
                             msb_2qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorCRY", "[GateGenerators]", float,
                   double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorCRY, "CRY",
                             lsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorCRY, "CRY",
                             mlsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorCRY, "CRY",
                             msb_2qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorCRZ", "[GateGenerators]", float,
                   double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorCRZ, "CRZ",
                             lsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorCRZ, "CRZ",
                             mlsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorCRZ, "CRZ",
                             msb_2qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorSingleExcitation",
                   "[GateGenerators]", float, double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorSingleExcitation, "SingleExcitation",
                             lsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorSingleExcitation, "SingleExcitation",
                             mlsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorSingleExcitation, "SingleExcitation",
                             msb_2qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorSingleExcitationMinus",
                   "[GateGenerators]", float, double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorSingleExcitationMinus,
                             "SingleExcitationMinus", lsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorSingleExcitationMinus,
                             "SingleExcitationMinus", mlsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorSingleExcitationMinus,
                             "SingleExcitationMinus", msb_2qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorSingleExcitationPlus",
                   "[GateGenerators]", float, double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorSingleExcitationPlus,
                             "SingleExcitationPlus", lsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorSingleExcitationPlus,
                             "SingleExcitationPlus", mlsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorSingleExcitationPlus,
                             "SingleExcitationPlus", msb_2qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorMultiRZ", "[GateGenerators]",
                   float, double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorMultiRZ,
                             "MultiRZ", lsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorMultiRZ,
                             "MultiRZ", mlsb_2qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits, applyGeneratorMultiRZ,
                             "MultiRZ", msb_2qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorDoubleExcitation",
                   "[GateGenerators]", float, double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorDoubleExcitation, "DoubleExcitation",
                             lsb_4qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorDoubleExcitation, "DoubleExcitation",
                             mlsb_4qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorDoubleExcitation, "DoubleExcitation",
                             msb_4qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorDoubleExcitationMinus",
                   "[GateGenerators]", float, double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorDoubleExcitationMinus,
                             "DoubleExcitationMinus", lsb_4qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorDoubleExcitationMinus,
                             "DoubleExcitationMinus", mlsb_4qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorDoubleExcitationMinus,
                             "DoubleExcitationMinus", msb_4qbit);
}

TEMPLATE_TEST_CASE("Generators::applyGeneratorDoubleExcitationPlus",
                   "[GateGenerators]", float, double) {
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorDoubleExcitationPlus,
                             "DoubleExcitationPlus", lsb_4qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorDoubleExcitationPlus,
                             "DoubleExcitationPlus", mlsb_4qbit);
    PLGPU_MPI_TEST_GENERATOR(TestType, num_qubits,
                             applyGeneratorDoubleExcitationPlus,
                             "DoubleExcitationPlus", msb_4qbit);
}
