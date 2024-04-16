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
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "MPIManager.hpp"
#include "MeasurementsGPU.hpp"
#include "MeasurementsGPUMPI.hpp"
#include "StateVectorCudaMPI.hpp"
#include "StateVectorCudaManaged.hpp"
#include "cuGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane::LightningGPU;
using namespace Pennylane::Util;

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningGPU::Measures;
using namespace Pennylane::LightningGPU::Observables;
using Pennylane::Util::createNonTrivialState;
using Pennylane::Util::write_CSR_vectors;
namespace cuUtil = Pennylane::LightningGPU::Util;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("[Identity]", "[StateVectorCudaMPI_Expval]", float, double) {
    using StateVectorT = StateVectorCudaMPI<TestType>;
    const size_t num_qubits = 3;
    auto ONE = TestType(1);

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

    StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                    nLocalIndexBits);
    sv.initSV();

    auto m = MeasurementsMPI(sv);

    SECTION("Using expval") {
        sv.applyOperations({{"Hadamard"}, {"CNOT"}, {"CNOT"}},
                           {{0}, {0, 1}, {1, 2}}, {{false}, {false}, {false}});
        auto ob = NamedObsMPI<StateVectorT>("Identity", {0});
        auto res = m.expval(ob);
        CHECK(res == Approx(ONE));
    }
}

TEMPLATE_TEST_CASE("[PauliX]", "[StateVectorCudaMPI_Expval]", float, double) {
    {
        using StateVectorT = StateVectorCudaMPI<TestType>;
        const size_t num_qubits = 3;

        auto ZERO = TestType(0);
        auto ONE = TestType(1);

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

        SECTION("Using expval") {
            StateVectorT sv(mpi_manager, dt_local, mpi_buffersize,
                            nGlobalIndexBits, nLocalIndexBits);
            sv.initSV();

            auto m = MeasurementsMPI(sv);
            sv.applyOperations({{"Hadamard"}, {"CNOT"}, {"CNOT"}},
                               {{0}, {0, 1}, {1, 2}},
                               {{false}, {false}, {false}});
            auto ob = NamedObsMPI<StateVectorT>("PauliX", {0});
            auto res = m.expval(ob);
            CHECK(res == ZERO);
        }

        SECTION("Using expval: Plus states") {
            StateVectorT sv(mpi_manager, dt_local, mpi_buffersize,
                            nGlobalIndexBits, nLocalIndexBits);
            sv.initSV();
            auto m = MeasurementsMPI(sv);
            sv.applyOperations({{"Hadamard"}, {"Hadamard"}, {"Hadamard"}},
                               {{0}, {1}, {2}}, {{false}, {false}, {false}});
            auto ob = NamedObsMPI<StateVectorT>("PauliX", {0});
            auto res = m.expval(ob);
            CHECK(res == Approx(ONE));
        }

        SECTION("Using expval: Minus states") {
            StateVectorT sv(mpi_manager, dt_local, mpi_buffersize,
                            nGlobalIndexBits, nLocalIndexBits);
            sv.initSV();
            auto m = MeasurementsMPI(sv);
            sv.applyOperations(
                {{"PauliX"},
                 {"Hadamard"},
                 {"PauliX"},
                 {"Hadamard"},
                 {"PauliX"},
                 {"Hadamard"}},
                {{0}, {0}, {1}, {1}, {2}, {2}},
                {{false}, {false}, {false}, {false}, {false}, {false}});
            auto ob = NamedObsMPI<StateVectorT>("PauliX", {0});
            auto res = m.expval(ob);
            CHECK(res == -Approx(ONE));
        }
    }
}

TEMPLATE_TEST_CASE("[PauliY]", "[StateVectorCudaMPI_Expval]", float, double) {
    {
        using StateVectorT = StateVectorCudaMPI<TestType>;
        const size_t num_qubits = 3;

        auto ZERO = TestType(0);
        auto ONE = TestType(1);
        auto PI = TestType(M_PI);

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

        SECTION("Using expval") {
            StateVectorT sv(mpi_manager, dt_local, mpi_buffersize,
                            nGlobalIndexBits, nLocalIndexBits);
            sv.initSV();
            auto m = MeasurementsMPI(sv);
            sv.applyOperations({{"Hadamard"}, {"CNOT"}, {"CNOT"}},
                               {{0}, {0, 1}, {1, 2}},
                               {{false}, {false}, {false}});
            auto ob = NamedObsMPI<StateVectorT>("PauliY", {0});
            auto res = m.expval(ob);
            CHECK(res == ZERO);
        }

        SECTION("Using expval: Plus i states") {
            StateVectorT sv(mpi_manager, dt_local, mpi_buffersize,
                            nGlobalIndexBits, nLocalIndexBits);
            sv.initSV();
            auto m = MeasurementsMPI(sv);
            sv.applyOperations({{"RX"}, {"RX"}, {"RX"}}, {{0}, {1}, {2}},
                               {{false}, {false}, {false}},
                               {{-PI / 2}, {-PI / 2}, {-PI / 2}});
            auto ob = NamedObsMPI<StateVectorT>("PauliY", {0});
            auto res = m.expval(ob);
            CHECK(res == Approx(ONE));
        }

        SECTION("Using expval: Minus i states") {
            StateVectorT sv(mpi_manager, dt_local, mpi_buffersize,
                            nGlobalIndexBits, nLocalIndexBits);
            sv.initSV();
            auto m = MeasurementsMPI(sv);
            sv.applyOperations({{"RX"}, {"RX"}, {"RX"}}, {{0}, {1}, {2}},
                               {{false}, {false}, {false}},
                               {{PI / 2}, {PI / 2}, {PI / 2}});
            auto ob = NamedObsMPI<StateVectorT>("PauliY", {0});
            auto res = m.expval(ob);
            CHECK(res == -Approx(ONE));
        }
    }
}

TEMPLATE_TEST_CASE("[PauliZ]", "[StateVectorCudaMPI_Expval]", float, double) {
    {
        using StateVectorT = StateVectorCudaMPI<TestType>;
        using PrecisionT = StateVectorT::PrecisionT;

        // Defining the statevector that will be measured.
        auto statevector_data =
            createNonTrivialState<StateVectorCudaManaged<TestType>>();
        size_t num_qubits = 3;

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

        auto sv_data_local = mpi_manager.scatter(statevector_data, 0);

        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.CopyHostDataToGpu(sv_data_local.data(), sv_data_local.size(), false);
        mpi_manager.Barrier();

        SECTION("Using expval") {
            auto m = MeasurementsMPI(sv);
            auto ob = NamedObsMPI<StateVectorT>("PauliZ", {1});
            auto res = m.expval(ob);
            PrecisionT ref = 0.77015115;
            REQUIRE(res == Approx(ref).margin(1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("[Hadamard]", "[StateVectorCudaMPI_Expval]", float, double) {
    {
        using StateVectorT = StateVectorCudaMPI<TestType>;
        const size_t num_qubits = 3;
        auto INVSQRT2 = TestType(0.707106781186547524401);

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

        SECTION("Using expval") {
            StateVectorT sv(mpi_manager, dt_local, mpi_buffersize,
                            nGlobalIndexBits, nLocalIndexBits);
            sv.initSV();
            auto m = MeasurementsMPI(sv);
            sv.applyOperation("PauliX", {0});
            auto ob = NamedObsMPI<StateVectorT>("Hadamard", {0});
            auto res = m.expval(ob);
            CHECK(res == Approx(-INVSQRT2).epsilon(1e-7));
        }
    }
}

TEMPLATE_TEST_CASE("Test expectation value of HamiltonianObs",
                   "[StateVectorCudaMPI_Expval]", float, double) {
    using StateVectorT = StateVectorCudaMPI<TestType>;
    using ComplexT = StateVectorT::ComplexT;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t num_qubits = 3;
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

    SECTION("Using expval") {
        std::vector<ComplexT> init_state{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                         {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                         {0.3, 0.4}, {0.4, 0.5}};
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
        auto res = m.expval(*ob);
        auto expected = TestType(-0.086);
        CHECK(expected == Approx(res));
    }
}

TEMPLATE_TEST_CASE("Test expectation value of TensorProdObs",
                   "[StateVectorCudaMPI_Expval]", float, double) {
    using StateVectorT = StateVectorCudaMPI<TestType>;
    using ComplexT = StateVectorT::ComplexT;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t num_qubits = 3;
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

    SECTION("Using expval") {
        std::vector<ComplexT> init_state{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                         {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                         {0.3, 0.4}, {0.4, 0.5}};
        auto local_init_sv = mpi_manager.scatter(init_state, 0);
        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.CopyHostDataToGpu(local_init_sv.data(), local_init_sv.size(), false);

        auto m = MeasurementsMPI(sv);

        auto X0 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliX", std::vector<size_t>{0});
        auto Z1 = std::make_shared<NamedObsMPI<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});

        auto ob = TensorProdObsMPI<StateVectorT>::create({X0, Z1});
        auto res = m.expval(*ob);
        auto expected = TestType(-0.36);
        CHECK(expected == Approx(res));
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaMPI::Hamiltonian_expval_Sparse",
                   "[StateVectorCudaMPI_Expval]", double) {
    using StateVectorT = StateVectorCudaMPI<TestType>;
    using ComplexT = StateVectorT::ComplexT;
    using IdxT = typename std::conditional<std::is_same<TestType, float>::value,
                                           int32_t, int64_t>::type;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t num_qubits = 3;
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

    SECTION("Sparse expval") {
        std::vector<ComplexT> init_state{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                         {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                         {0.3, 0.4}, {0.4, 0.5}};

        auto local_init_sv = mpi_manager.scatter(init_state, 0);
        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.CopyHostDataToGpu(local_init_sv.data(), local_init_sv.size(), false);
        auto m = MeasurementsMPI(sv);

        std::vector<IdxT> index_ptr = {0, 2, 4, 6, 8, 10, 12, 14, 16};
        std::vector<IdxT> indices = {0, 3, 1, 2, 1, 2, 0, 3,
                                     4, 7, 5, 6, 5, 6, 4, 7};
        std::vector<ComplexT> values = {
            {3.1415, 0.0},  {0.0, -3.1415}, {3.1415, 0.0}, {0.0, 3.1415},
            {0.0, -3.1415}, {3.1415, 0.0},  {0.0, 3.1415}, {3.1415, 0.0},
            {3.1415, 0.0},  {0.0, -3.1415}, {3.1415, 0.0}, {0.0, 3.1415},
            {0.0, -3.1415}, {3.1415, 0.0},  {0.0, 3.1415}, {3.1415, 0.0}};

        auto result = m.expval(
            index_ptr.data(), static_cast<int64_t>(index_ptr.size()),
            indices.data(), values.data(), static_cast<int64_t>(values.size()));
        auto expected = TestType(3.1415);
        CHECK(expected == Approx(result).epsilon(1e-7));
    }

    SECTION("Testing Sparse Hamiltonian:") {
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;

        // Defining the statevector that will be measured.
        auto statevector_data =
            createNonTrivialState<StateVectorCudaManaged<TestType>>();

        auto sv_data_local = mpi_manager.scatter(statevector_data, 0);

        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.CopyHostDataToGpu(sv_data_local.data(), sv_data_local.size(), false);
        mpi_manager.Barrier();
        // Initializing the measurements class.
        // This object attaches to the statevector allowing several
        // measurements.
        MeasurementsMPI<StateVectorT> Measurer(sv);
        size_t data_size = Pennylane::Util::exp2(num_qubits);

        std::vector<IdxT> row_map;
        std::vector<IdxT> entries;
        std::vector<ComplexT> values;
        write_CSR_vectors<ComplexT, IdxT>(row_map, entries, values,
                                          static_cast<IdxT>(data_size));

        PrecisionT exp_values = Measurer.expval(
            row_map.data(), static_cast<int64_t>(row_map.size()),
            entries.data(), values.data(), static_cast<int64_t>(values.size()));
        PrecisionT exp_values_ref = 0.5930885;
        REQUIRE(exp_values == Approx(exp_values_ref).margin(1e-6));

        mpi_manager.Barrier();

        PrecisionT var_values = Measurer.var(
            row_map.data(), static_cast<int64_t>(row_map.size()),
            entries.data(), values.data(), static_cast<int64_t>(values.size()));
        PrecisionT var_values_ref = 2.4624654;
        REQUIRE(var_values == Approx(var_values_ref).margin(1e-6));
    }
}
