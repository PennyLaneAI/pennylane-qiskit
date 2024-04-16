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

#include "DevTag.hpp"
#include "cuGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

#include "StateVectorCudaMPI.hpp"
#include "StateVectorCudaManaged.hpp"

#include "MPIManager.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane;
using namespace Pennylane::LightningGPU;

#define num_qubits 8
#define lsb_1qbit                                                              \
    { 0 }
#define msb_1qbit                                                              \
    { num_qubits - 1 }
#define lsb_2qbit                                                              \
    { 0, 1 }
#define msb_2qubit                                                             \
    { num_qubits - 2, num_qubits - 1 }
#define mlsb_2qubit                                                            \
    { 0, num_qubits - 1 }
#define lsb_3qbit                                                              \
    { 0, 1, 2 }
#define msb_3qubit                                                             \
    { num_qubits - 3, num_qubits - 2, num_qubits - 1 }
#define mlsb_3qubit                                                            \
    { 0, num_qubits - 2, num_qubits - 1 }

/**
 * @brief Tests the constructability of the StateVectorCudaMPI class.
 *
 */
TEMPLATE_TEST_CASE("StateVectorCudaMPI::StateVectorCudaMPI",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    SECTION("StateVectorCudaMPI<TestType> {MPIManager, DevTag, "
            "std::size_t, std::size_t, std::size_t}") {
        REQUIRE(std::is_constructible<StateVectorCudaMPI<TestType>, MPIManager,
                                      DevTag<int>, std::size_t, std::size_t,
                                      std::size_t>::value);
    }
    SECTION("StateVectorCudaMPI<TestType> {MPI_Comm, DevTag, "
            "std::size_t, std::size_t, std::size_t}") {
        REQUIRE(std::is_constructible<StateVectorCudaMPI<TestType>, MPI_Comm,
                                      DevTag<int>, std::size_t, std::size_t,
                                      std::size_t>::value);
    }
    SECTION("StateVectorCudaMPI<TestType> {DevTag, "
            "std::size_t, std::size_t, std::size_t}") {
        REQUIRE(std::is_constructible<StateVectorCudaMPI<TestType>, DevTag<int>,
                                      std::size_t, std::size_t,
                                      std::size_t>::value);
    }
    SECTION(
        "StateVectorCudaMPI<TestType> {DevTag, std::size_t, size_t, CFP_t}") {
        if (std::is_same_v<TestType, double>) {
            REQUIRE(std::is_constructible<StateVectorCudaMPI<TestType>,
                                          DevTag<int>, std::size_t, std::size_t,
                                          cuDoubleComplex *>::value);
        } else {
            REQUIRE(std::is_constructible<StateVectorCudaMPI<TestType>,
                                          DevTag<int>, std::size_t, std::size_t,
                                          cuFloatComplex *>::value);
        }
    }
    SECTION("StateVectorCudaMPI<TestType> {DevTag, "
            "std::size_t, std::size_t}") {
        REQUIRE(std::is_constructible<StateVectorCudaMPI<TestType>, DevTag<int>,
                                      std::size_t, std::size_t>::value);
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::SetStateVector",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t mpi_buffersize = 1;

    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
    size_t subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<cp_t> init_state(Pennylane::Util::exp2(num_qubits));
    std::vector<cp_t> expected_state(Pennylane::Util::exp2(num_qubits));
    std::vector<cp_t> local_state(subSvLength);

    using index_type =
        typename std::conditional<std::is_same<PrecisionT, float>::value,
                                  int32_t, int64_t>::type;

    std::vector<index_type> indices(Pennylane::Util::exp2(num_qubits));

    if (mpi_manager.getRank() == 0) {
        std::mt19937 re{1337};
        auto st = Pennylane::Util::createRandomStateVectorData<PrecisionT>(
            re, num_qubits);
        init_state.clear();
        init_state =
            std::vector<cp_t>(st.begin(), st.end(), init_state.get_allocator());
        expected_state = init_state;
        for (size_t i = 0; i < Pennylane::Util::exp2(num_qubits - 1); i++) {
            std::swap(expected_state[i * 2], expected_state[i * 2 + 1]);
            indices[i * 2] = i * 2 + 1;
            indices[i * 2 + 1] = i * 2;
        }
    }
    mpi_manager.Barrier();

    auto expected_local_state = mpi_manager.scatter<cp_t>(expected_state, 0);
    mpi_manager.Bcast<index_type>(indices, 0);
    mpi_manager.Bcast<cp_t>(init_state, 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);

    //`values[i]` on the host will be copy the `indices[i]`th element of the
    // state vector on the device.
    SECTION("Set state vector with values and their corresponding indices on "
            "the host") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, dt_local, mpi_buffersize,
                                          nGlobalIndexBits, nLocalIndexBits);
        // The setStates will shuffle the state vector values on the device with
        // the following indices and values setting on host. For example, the
        // values[i] is used to set the indices[i] th element of state vector on
        // the device. For example, values[2] (init_state[5]) will be copied to
        // indices[2]th or (4th) element of the state vector.

        sv.template setStateVector<index_type>(
            init_state.size(), init_state.data(), indices.data(), false);

        mpi_manager.Barrier();
        sv.CopyGpuDataToHost(local_state.data(),
                             static_cast<std::size_t>(subSvLength));
        mpi_manager.Barrier();

        CHECK(expected_local_state == Pennylane::Util::approx(local_state));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::SetIthStates",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t mpi_buffersize = 1;

    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
    size_t subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    int index;
    if (mpi_manager.getRank() == 0) {
        std::mt19937 re{1337};
        std::uniform_int_distribution<> distr(
            0, Pennylane::Util::exp2(num_qubits) - 1);
        index = distr(re);
    }
    mpi_manager.Bcast(index, 0);

    std::vector<cp_t> expected_state(Pennylane::Util::exp2(num_qubits), {0, 0});
    if (mpi_manager.getRank() == 0) {
        expected_state[index] = {1.0, 0};
    }

    auto expected_local_state = mpi_manager.scatter(expected_state, 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);

    SECTION(
        "Set Ith element of the state state on device with data on the host") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, dt_local, mpi_buffersize,
                                          nGlobalIndexBits, nLocalIndexBits);
        std::complex<PrecisionT> values = {1.0, 0};
        sv.setBasisState(values, index, false);

        std::vector<cp_t> h_sv0(subSvLength, {0.0, 0.0});
        sv.CopyGpuDataToHost(h_sv0.data(),
                             static_cast<std::size_t>(subSvLength));

        CHECK(expected_local_state == Pennylane::Util::approx(h_sv0));
    }
}

#define PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, NUM_QUBITS, GATE_METHOD,    \
                                         GATE_NAME, WIRE)                      \
    {                                                                          \
        const bool adjoint = GENERATE(true, false);                            \
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
                sv.GATE_METHOD(WIRE, adjoint);                                 \
                sv.CopyGpuDataToHost(local_state.data(),                       \
                                     static_cast<std::size_t>(subSvLength));   \
                                                                               \
                StateVectorCudaManaged<TestType> svdat{init_sv.data(),         \
                                                       svLength};              \
                if (mpi_manager.getRank() == 0) {                              \
                    svdat.GATE_METHOD(WIRE, adjoint);                          \
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
                sv.applyOperation(GATE_NAME, WIRE, adjoint);                   \
                sv.CopyGpuDataToHost(local_state.data(),                       \
                                     static_cast<std::size_t>(subSvLength));   \
                StateVectorCudaManaged<TestType> svdat{init_sv.data(),         \
                                                       svLength};              \
                if (mpi_manager.getRank() == 0) {                              \
                    svdat.applyOperation(GATE_NAME, WIRE, adjoint);            \
                    svdat.CopyGpuDataToHost(expected_sv.data(), svLength);     \
                }                                                              \
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);  \
                CHECK(local_state ==                                           \
                      Pennylane::Util::approx(expected_local_sv));             \
            }                                                                  \
        }                                                                      \
    }

TEMPLATE_TEST_CASE("StateVectorCudaMPI::Hadamard",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyHadamard,
                                     "Hadamard", lsb_1qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyHadamard,
                                     "Hadamard", {num_qubits - 1});
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::PauliX",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliX,
                                     "PauliX", lsb_1qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliX,
                                     "PauliX", {num_qubits - 1});
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::PauliY",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliY,
                                     "PauliY", lsb_1qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliY,
                                     "PauliY", {num_qubits - 1});
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::PauliZ",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliZ,
                                     "PauliZ", lsb_1qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliZ,
                                     "PauliZ", {num_qubits - 1});
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::S", "[StateVectorCudaMPI_Nonparam]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyS, "S",
                                     lsb_1qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyS, "S",
                                     {num_qubits - 1});
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::T", "[StateVectorCudaMPI_Nonparam]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyT, "T",
                                     lsb_1qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyT, "T",
                                     {num_qubits - 1});
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::CNOT", "[StateVectorCudaMPI_Nonparam]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCNOT, "CNOT",
                                     lsb_2qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCNOT, "CNOT",
                                     mlsb_2qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCNOT, "CNOT",
                                     msb_2qubit);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::SWAP", "[StateVectorCudaMPI_Nonparam]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applySWAP, "SWAP",
                                     lsb_2qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applySWAP, "SWAP",
                                     mlsb_2qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applySWAP, "SWAP",
                                     msb_2qubit);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::CY", "[StateVectorCudaMPI_Nonparam]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCY, "CY",
                                     lsb_2qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCY, "CY",
                                     mlsb_2qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCY, "CY",
                                     msb_2qubit);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::CZ", "[StateVectorCudaMPI_Nonparam]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCZ, "CZ",
                                     lsb_2qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCZ, "CZ",
                                     mlsb_2qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCZ, "CZ",
                                     msb_2qubit);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::Toffoli",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyToffoli,
                                     "Toffoli", lsb_3qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyToffoli,
                                     "Toffoli", mlsb_3qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyToffoli,
                                     "Toffoli", msb_3qubit);
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::CSWAP", "[StateVectorCudaMPI_Nonparam]",
                   float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCSWAP, "CSWAP",
                                     lsb_3qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCSWAP, "CSWAP",
                                     mlsb_3qubit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyCSWAP, "CSWAP",
                                     msb_3qubit);
}