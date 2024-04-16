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

#include "TestHelpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::LightningGPU::Measures;
using Pennylane::Util::createNonTrivialState;
}; // namespace
/// @endcond

TEMPLATE_TEST_CASE("Expected Values", "[MeasurementsMPI]", float, double) {
    using StateVectorT = StateVectorCudaMPI<TestType>;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

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

    // Initializing the Measurements class.
    // This object attaches to the statevector allowing several measures.
    SECTION("Testing single operation defined by a matrix:") {
        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.CopyHostDataToGpu(sv_data_local.data(), sv_data_local.size(), false);
        mpi_manager.Barrier();

        MeasurementsMPI<StateVectorT> Measurer(sv);
        mpi_manager.Barrier();
        const std::vector<ComplexT> PauliX = {ComplexT{0, 0}, ComplexT{1, 0},
                                              ComplexT{1, 0}, ComplexT{0, 0}};
        const std::vector<size_t> wires_single = {0};
        PrecisionT exp_value = Measurer.expval(PauliX, wires_single);
        PrecisionT exp_values_ref = 0.492725;
        REQUIRE(exp_value == Approx(exp_values_ref).margin(1e-6));
    }

    SECTION("Testing single operation defined by its name:") {
        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.CopyHostDataToGpu(sv_data_local.data(), sv_data_local.size(), false);
        mpi_manager.Barrier();

        MeasurementsMPI<StateVectorT> Measurer(sv);
        mpi_manager.Barrier();
        std::vector<size_t> wires_single = {0};
        PrecisionT exp_value = Measurer.expval("PauliX", wires_single);
        PrecisionT exp_values_ref = 0.492725;
        REQUIRE(exp_value == Approx(exp_values_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by a matrix:") {
        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.CopyHostDataToGpu(sv_data_local.data(), sv_data_local.size(), false);
        mpi_manager.Barrier();

        MeasurementsMPI<StateVectorT> Measurer(sv);
        mpi_manager.Barrier();
        std::vector<ComplexT> PauliX = {0, 1, 1, 0};
        std::vector<ComplexT> PauliY = {0, ComplexT{0, -1}, ComplexT{0, 1}, 0};
        std::vector<ComplexT> PauliZ = {1, 0, 0, -1};

        std::vector<PrecisionT> exp_values;
        std::vector<PrecisionT> exp_values_ref;
        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::vector<ComplexT>> operations_list;

        operations_list = {PauliX, PauliX, PauliX};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.49272486, 0.42073549, 0.28232124};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {PauliY, PauliY, PauliY};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {-0.64421768, -0.47942553, -0.29552020};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {PauliZ, PauliZ, PauliZ};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.58498357, 0.77015115, 0.91266780};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by its name:") {
        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.CopyHostDataToGpu(sv_data_local.data(), sv_data_local.size(), false);
        mpi_manager.Barrier();

        MeasurementsMPI<StateVectorT> Measurer(sv);
        mpi_manager.Barrier();
        std::vector<PrecisionT> exp_values;
        std::vector<PrecisionT> exp_values_ref;
        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::string> operations_list;

        operations_list = {"PauliX", "PauliX", "PauliX"};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.49272486, 0.42073549, 0.28232124};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {"PauliY", "PauliY", "PauliY"};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {-0.64421768, -0.47942553, -0.29552020};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));

        operations_list = {"PauliZ", "PauliZ", "PauliZ"};
        exp_values = Measurer.expval(operations_list, wires_list);
        exp_values_ref = {0.58498357, 0.77015115, 0.91266780};
        REQUIRE_THAT(exp_values, Catch::Approx(exp_values_ref).margin(1e-6));
    }

    SECTION("Catch failures caused by unsupported named gates") {
        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.CopyHostDataToGpu(sv_data_local.data(), sv_data_local.size(), false);
        mpi_manager.Barrier();

        MeasurementsMPI<StateVectorT> Measurer(sv);
        mpi_manager.Barrier();
        std::string obs = "paulix";
        PL_CHECK_THROWS_MATCHES(Measurer.expval(obs, {0}), LightningException,
                                "Currently unsupported observable: paulix");
    }

    SECTION("Catch failures caused by unsupported empty matrix gate") {
        StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);
        sv.CopyHostDataToGpu(sv_data_local.data(), sv_data_local.size(), false);
        mpi_manager.Barrier();

        MeasurementsMPI<StateVectorT> Measurer(sv);
        mpi_manager.Barrier();
        std::vector<ComplexT> gate_matrix = {};
        PL_CHECK_THROWS_MATCHES(Measurer.expval(gate_matrix, {0}),
                                LightningException,
                                "Currently unsupported observable");
    }
}

TEMPLATE_TEST_CASE("Pauliwords base on expval", "[MeasurementsMPI]", float,
                   double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    using StateVectorT = StateVectorCudaMPI<TestType>;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t numqubits = 4;
    size_t mpi_buffersize = 1;

    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = numqubits - nGlobalIndexBits;
    mpi_manager.Barrier();

    std::vector<cp_t> init_sv{{0.1653855288944372, 0.08360762242222763},
                              {0.0731293375604395, 0.13209080879903976},
                              {0.23742759434160687, 0.2613440813782711},
                              {0.16768740742688235, 0.2340607179431313},
                              {0.2247465091396771, 0.052469062762363974},
                              {0.1595307101966878, 0.018355977199570113},
                              {0.01433428625707798, 0.18836803047905595},
                              {0.20447553584586473, 0.02069817884076428},
                              {0.17324175995006008, 0.12834320562185453},
                              {0.021542232643170886, 0.2537776554975786},
                              {0.2917899745322105, 0.30227665008366594},
                              {0.17082687702494623, 0.013880922806771745},
                              {0.03801974084659355, 0.2233816291263903},
                              {0.1991010562067874, 0.2378546697582974},
                              {0.13833362414043807, 0.0571737109901294},
                              {0.1960850292216881, 0.22946370987301284}};

    auto local_state = mpi_manager.scatter(init_sv, 0);

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    mpi_manager.Barrier();

    StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                    nLocalIndexBits);
    sv.CopyHostDataToGpu(local_state, false);

    SECTION("Test getExpectationValuePauliWords (full wires)") {
        MeasurementsMPI<StateVectorT> Measurer(sv);
        mpi_manager.Barrier();

        std::vector<std::string> pauli_words = {"XYZI", "ZZXX"};
        std::vector<std::vector<size_t>> tgts = {{0, 1, 2, 3}, {0, 1, 2, 3}};
        std::vector<std::complex<PrecisionT>> coeffs = {{0.1, 0.0}, {0.2, 0.0}};

        auto expval_mpi = Measurer.expval(pauli_words, tgts, coeffs.data());

        CHECK(expval_mpi == Approx(0.0014895211).margin(1e-7));
    }

    SECTION("Test getExpectationValuePauliWords (global wires)") {
        MeasurementsMPI<StateVectorT> Measurer(sv);
        mpi_manager.Barrier();

        std::vector<std::string> pauli_words = {"X", "Y", "Z", "I"};
        std::vector<std::vector<size_t>> tgts = {{0}, {0}, {0}, {0}};
        std::vector<std::complex<PrecisionT>> coeffs = {
            {0.1, 0.0}, {0.2, 0.0}, {0.3, 0.0}, {0.4, 0.0}};

        auto expval_mpi = Measurer.expval(pauli_words, tgts, coeffs.data());

        CHECK(expval_mpi == Approx(0.4589167637).margin(1e-7));
    }

    SECTION("Test getExpectationValuePauliWords (local wires)") {
        MeasurementsMPI<StateVectorT> Measurer(sv);
        mpi_manager.Barrier();

        std::vector<std::string> pauli_words = {"X", "Y", "Z", "I"};
        std::vector<std::vector<size_t>> tgts = {
            {numqubits - 1}, {numqubits - 1}, {numqubits - 1}, {numqubits - 1}};
        std::vector<std::complex<PrecisionT>> coeffs = {
            {0.1, 0.0}, {0.2, 0.0}, {0.3, 0.0}, {0.4, 0.0}};

        auto expval_mpi = Measurer.expval(pauli_words, tgts, coeffs.data());

        CHECK(expval_mpi == Approx(0.4841317321).margin(1e-7));
    }

    SECTION("Test getExpectationValuePauliWords (mixed wires)") {
        MeasurementsMPI<StateVectorT> Measurer(sv);
        mpi_manager.Barrier();

        std::vector<std::string> pauli_words = {"X", "XY", "XYZ", "XYZI",
                                                "X", "Y",  "Z",   "I"};
        std::vector<std::vector<size_t>> tgts = {
            {0}, {0, 1}, {0, 1, 2}, {0, 1, 2, 3}, {3}, {3}, {3}, {3}};
        std::vector<std::complex<PrecisionT>> coeffs = {
            {0.1, 0.0}, {0.2, 0.0}, {0.3, 0.0}, {0.4, 0.0},
            {0.1, 0.0}, {0.2, 0.0}, {0.3, 0.0}, {0.4, 0.0}};

        auto expval_mpi = Measurer.expval(pauli_words, tgts, coeffs.data());

        CHECK(expval_mpi == Approx(0.4735548926).margin(1e-7));
    }
}

TEMPLATE_TEST_CASE("Variances", "[MeasurementsMPI]", double) {
    using StateVectorT = StateVectorCudaMPI<TestType>;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

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

    // Initializing the Measurements class.
    // This object attaches to the statevector allowing several measures.
    SECTION("Testing single operation defined by a matrix:") {
        MeasurementsMPI<StateVectorT> Measurer(sv);
        mpi_manager.Barrier();
        std::vector<ComplexT> PauliX = {0, 1, 1, 0};
        std::vector<size_t> wires_single = {0};
        PrecisionT variance = Measurer.var(PauliX, wires_single);
        PrecisionT variances_ref = 0.7572222;
        REQUIRE(variance == Approx(variances_ref).margin(1e-6));
    }

    SECTION("Testing single operation defined by its name:") {
        MeasurementsMPI<StateVectorT> Measurer(sv);
        mpi_manager.Barrier();
        std::vector<size_t> wires_single = {0};
        PrecisionT variance = Measurer.var("PauliX", wires_single);
        PrecisionT variances_ref = 0.7572222;
        REQUIRE(variance == Approx(variances_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by a matrix:") {
        MeasurementsMPI<StateVectorT> Measurer(sv);
        mpi_manager.Barrier();
        std::vector<ComplexT> PauliX = {0, 1, 1, 0};
        std::vector<ComplexT> PauliY = {0, ComplexT{0, -1}, ComplexT{0, 1}, 0};
        std::vector<ComplexT> PauliZ = {1, 0, 0, -1};

        std::vector<PrecisionT> variances;
        std::vector<PrecisionT> variances_ref;
        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::vector<ComplexT>> operations_list;

        operations_list = {PauliX, PauliX, PauliX};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.7572222, 0.8229816, 0.9202947};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {PauliY, PauliY, PauliY};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.5849835, 0.7701511, 0.9126678};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {PauliZ, PauliZ, PauliZ};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.6577942, 0.4068672, 0.1670374};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));
    }

    SECTION("Testing list of operators defined by its name:") {
        MeasurementsMPI<StateVectorT> Measurer(sv);
        mpi_manager.Barrier();
        std::vector<PrecisionT> variances;
        std::vector<PrecisionT> variances_ref;
        std::vector<std::vector<size_t>> wires_list = {{0}, {1}, {2}};
        std::vector<std::string> operations_list;

        operations_list = {"PauliX", "PauliX", "PauliX"};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.7572222, 0.8229816, 0.9202947};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {"PauliY", "PauliY", "PauliY"};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.5849835, 0.7701511, 0.9126678};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));

        operations_list = {"PauliZ", "PauliZ", "PauliZ"};
        variances = Measurer.var(operations_list, wires_list);
        variances_ref = {0.6577942, 0.4068672, 0.1670374};
        REQUIRE_THAT(variances, Catch::Approx(variances_ref).margin(1e-6));
    }
}

TEMPLATE_TEST_CASE("Probabilities", "[MeasuresMPI]", double) {
    using StateVectorT = StateVectorCudaMPI<TestType>;
    // Probabilities calculated with Pennylane default.qubit:
    std::vector<std::pair<std::vector<size_t>, std::vector<TestType>>> input = {
        {{2, 1, 0},
         {0.67078706, 0.03062806, 0.0870997, 0.00397696, 0.17564072, 0.00801973,
          0.02280642, 0.00104134}}};

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
    std::vector<TestType> expected_prob = {0.67078706, 0.03062806, 0.0870997,
                                           0.00397696, 0.17564072, 0.00801973,
                                           0.02280642, 0.00104134};
    auto prob_local = mpi_manager.scatter(expected_prob, 0);

    StateVectorT sv(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                    nLocalIndexBits);
    sv.CopyHostDataToGpu(sv_data_local.data(), sv_data_local.size(), false);
    mpi_manager.Barrier();

    SECTION("Looping over different wire configurations:") {
        auto m = MeasurementsMPI(sv);
        for (const auto &term : input) {
            auto probabilities = m.probs(term.first);
            REQUIRE_THAT(prob_local, Catch::Approx(probabilities).margin(1e-6));
        }
    }
}