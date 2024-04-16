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
#include <catch2/catch.hpp>

#include "MPIManager.hpp"
#include "ObservablesGPU.hpp"
#include "ObservablesGPUMPI.hpp"
#include "StateVectorCudaMPI.hpp"
#include "StateVectorCudaManaged.hpp"
#include "TestHelpers.hpp"
/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU::Observables;
using Pennylane::Util::LightningException;
} // namespace
/// @endcond

TEMPLATE_PRODUCT_TEST_CASE("NamedObsMPI", "[Observables]", (StateVectorCudaMPI),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using NamedObsT = NamedObsMPI<StateVectorT>;

    SECTION("Non-Default constructibility") {
        REQUIRE(!std::is_constructible_v<NamedObsT>);
    }

    SECTION("Constructibility") {
        REQUIRE(std::is_constructible_v<NamedObsT, std::string,
                                        std::vector<size_t>>);
    }

    SECTION("Constructibility - optional parameters") {
        REQUIRE(
            std::is_constructible_v<NamedObsT, std::string, std::vector<size_t>,
                                    std::vector<PrecisionT>>);
    }

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<NamedObsT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<NamedObsT>);
    }

    SECTION("NamedObs only accepts correct arguments") {
        REQUIRE_THROWS_AS(NamedObsT("PauliX", {}), LightningException);
        REQUIRE_THROWS_AS(NamedObsT("PauliX", {0, 3}), LightningException);

        REQUIRE_THROWS_AS(NamedObsT("RX", {0}), LightningException);
        REQUIRE_THROWS_AS(NamedObsT("RX", {0, 1, 2, 3}), LightningException);
        REQUIRE_THROWS_AS(
            NamedObsT("RX", {0}, std::vector<PrecisionT>{0.3, 0.4}),
            LightningException);
        REQUIRE_NOTHROW(
            NamedObsT("Rot", {0}, std::vector<PrecisionT>{0.3, 0.4, 0.5}));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("HermitianObsMPI", "[Observables]",
                           (StateVectorCudaMPI), (float, double)) {
    using StateVectorT = TestType;
    using ComplexT = typename StateVectorT::ComplexT;
    using MatrixT = std::vector<ComplexT>;
    using HermitianObsT = HermitianObsMPI<StateVectorT>;

    SECTION("Non-Default constructibility") {
        REQUIRE(!std::is_constructible_v<HermitianObsT>);
    }

    SECTION("Constructibility") {
        REQUIRE(std::is_constructible_v<HermitianObsT, MatrixT,
                                        std::vector<size_t>>);
    }

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<HermitianObsT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<HermitianObsT>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("TensorProdObs", "[Observables]",
                           (StateVectorCudaManaged), (float, double)) {
    using StateVectorT = TestType;
    using TensorProdObsT = TensorProdObsMPI<StateVectorT>;
    using NamedObsT = NamedObsMPI<StateVectorT>;
    using HermitianObsT = HermitianObsMPI<StateVectorT>;

    SECTION("Constructibility - NamedObs") {
        REQUIRE(
            std::is_constructible_v<TensorProdObsT,
                                    std::vector<std::shared_ptr<NamedObsT>>>);
    }

    SECTION("Constructibility - HermitianObs") {
        REQUIRE(std::is_constructible_v<
                TensorProdObsT, std::vector<std::shared_ptr<HermitianObsT>>>);
    }

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<TensorProdObsT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<TensorProdObsT>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("HamiltonianMPI", "[Observables]",
                           (StateVectorCudaMPI), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using TensorProdObsT = TensorProdObsMPI<StateVectorT>;
    using NamedObsT = NamedObsMPI<StateVectorT>;
    using HermitianObsT = HermitianObsMPI<StateVectorT>;
    using HamiltonianT = HamiltonianMPI<StateVectorT>;

    SECTION("Constructibility - NamedObs") {
        REQUIRE(
            std::is_constructible_v<HamiltonianT, std::vector<PrecisionT>,
                                    std::vector<std::shared_ptr<NamedObsT>>>);
    }

    SECTION("Constructibility - HermitianObs") {
        REQUIRE(std::is_constructible_v<
                HamiltonianT, std::vector<PrecisionT>,
                std::vector<std::shared_ptr<HermitianObsT>>>);
    }

    SECTION("Constructibility - TensorProdObsT") {
        REQUIRE(std::is_constructible_v<
                HamiltonianT, std::vector<PrecisionT>,
                std::vector<std::shared_ptr<TensorProdObsT>>>);
    }

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<HamiltonianT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<HamiltonianT>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("HamiltonianMPI::ApplyInPlace", "[Observables]",
                           (StateVectorCudaMPI), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using TensorProdObsT = TensorProdObsMPI<StateVectorT>;
    using NamedObsT = NamedObsMPI<StateVectorT>;
    using HamiltonianT = HamiltonianMPI<StateVectorT>;

    using TensorProdObs = TensorProdObs<StateVectorCudaManaged<PrecisionT>>;
    using NamedObs = NamedObs<StateVectorCudaManaged<PrecisionT>>;
    using Hamiltonian = Hamiltonian<StateVectorCudaManaged<PrecisionT>>;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    size_t num_qubits = 8;
    size_t mpi_buffersize = 1;
    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
    size_t subSvLength = 1 << nLocalIndexBits;
    size_t svLength = 1 << num_qubits;

    mpi_manager.Barrier();
    std::vector<ComplexT> expected_sv(svLength);
    std::vector<ComplexT> local_state(subSvLength);
    std::mt19937 re{1337};
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

    const auto h = PrecisionT{0.809}; // half of the golden ratio

    auto zz = std::make_shared<TensorProdObsT>(
        std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{0}),
        std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{1}));

    auto x1 = std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{0});
    auto x2 = std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{1});

    auto zz0 = std::make_shared<TensorProdObs>(
        std::make_shared<NamedObs>("PauliZ", std::vector<size_t>{0}),
        std::make_shared<NamedObs>("PauliZ", std::vector<size_t>{1}));

    auto x10 = std::make_shared<NamedObs>("PauliX", std::vector<size_t>{0});
    auto x20 = std::make_shared<NamedObs>("PauliX", std::vector<size_t>{1});

    auto ham = HamiltonianT::create({PrecisionT{1.0}, h, h}, {zz, x1, x2});
    auto ham0 = Hamiltonian::create({PrecisionT{1.0}, h, h}, {zz0, x10, x20});

    SECTION("ApplyInPlace", "[Apply Method]") {
        SECTION("Hamiltonian applies correctly to |+->") {
            StateVectorT sv_mpi(mpi_manager, dt_local, mpi_buffersize,
                                nGlobalIndexBits, nLocalIndexBits);

            sv_mpi.CopyHostDataToGpu(local_state, false);

            ham->applyInPlace(sv_mpi);

            sv_mpi.CopyGpuDataToHost(local_state.data(), local_state.size());

            std::vector<ComplexT> expected(svLength);
            std::vector<ComplexT> expected_local(subSvLength);

            if (mpi_manager.getRank() == 0) {
                StateVectorCudaManaged<PrecisionT> state_vector(init_sv.data(),
                                                                init_sv.size());
                ham0->applyInPlace(state_vector);
                state_vector.CopyGpuDataToHost(expected.data(),
                                               expected.size());
            }
            mpi_manager.Barrier();

            mpi_manager.Scatter(expected.data(), expected_local.data(),
                                subSvLength, 0);
            mpi_manager.Barrier();

            REQUIRE(isApproxEqual(expected_local.data(), expected_local.size(),
                                  local_state.data(), local_state.size()));
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Observables::HermitianHasherMPI", "[Observables]",
                           (StateVectorCudaMPI), (float, double)) {
    using StateVectorT = TestType;
    using ComplexT = typename StateVectorT::ComplexT;
    using TensorProdObsT = TensorProdObsMPI<StateVectorT>;
    using NamedObsT = NamedObsMPI<StateVectorT>;
    using HermitianT = HermitianObsMPI<StateVectorT>;
    using HamiltonianT = HamiltonianMPI<StateVectorT>;

    std::vector<ComplexT> hermitian_h{{0.7071067811865475, 0},
                                      {0.7071067811865475, 0},
                                      {0.7071067811865475, 0},
                                      {-0.7071067811865475, 0}};

    auto obs1 =
        std::make_shared<HermitianT>(hermitian_h, std::vector<size_t>{0});
    auto obs2 = std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{2});
    auto obs3 = std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{3});

    auto tp_obs1 = std::make_shared<TensorProdObsT>(obs1, obs2);
    auto tp_obs2 = std::make_shared<TensorProdObsT>(obs2, obs3);

    auto ham_1 =
        HamiltonianT::create({0.165, 0.13, 0.5423}, {obs1, obs2, obs2});
    auto ham_2 = HamiltonianT::create({0.8545, 0.3222}, {tp_obs1, tp_obs2});

    SECTION("HamiltonianGPU<TestType>::obsName") {
        std::ostringstream res1, res2;
        res1 << "Hamiltonian: { 'coeffs' : [0.165, 0.13, 0.5423], "
                "'observables' : [Hermitian"
             << MatrixHasher()(hermitian_h) << ", PauliX[2], PauliX[2]]}";
        res2 << "Hamiltonian: { 'coeffs' : [0.8545, 0.3222], 'observables' : "
                "[Hermitian"
             << MatrixHasher()(hermitian_h)
             << " @ PauliX[2], PauliX[2] @ PauliX[3]]}";

        CHECK(ham_1->getObsName() == res1.str());
        CHECK(ham_2->getObsName() == res2.str());
    }
}

TEMPLATE_PRODUCT_TEST_CASE("SparseHamiltonian::ApplyInPlace", "[Observables]",
                           (StateVectorCudaMPI), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    MPIManager mpi_manager(MPI_COMM_WORLD);

    const std::size_t num_qubits = 3;
    std::mt19937 re{1337};

    auto sparseH = SparseHamiltonianMPI<StateVectorT>::create(
        {ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0},
         ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0},
         ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}},
        {7, 6, 5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5, 6, 7, 8}, {0, 1, 2});

    size_t mpi_buffersize = 1;
    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
    size_t subSvLength = 1 << nLocalIndexBits;

    mpi_manager.Barrier();
    std::vector<ComplexT> expected_sv(subSvLength);
    std::vector<ComplexT> local_state(subSvLength);

    auto init_state = createRandomStateVectorData<PrecisionT>(re, num_qubits);

    mpi_manager.Scatter(init_state.data(), local_state.data(), subSvLength, 0);
    mpi_manager.Barrier();

    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    mpi_manager.Barrier();

    StateVectorT sv_mpi(mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,
                        nLocalIndexBits);

    sv_mpi.CopyHostDataToGpu(local_state, false);

    sparseH->applyInPlace(sv_mpi);

    std::reverse(init_state.begin(), init_state.end());
    mpi_manager.Scatter(init_state.data(), expected_sv.data(), subSvLength, 0);
    mpi_manager.Barrier();

    REQUIRE(isApproxEqual(sv_mpi.getDataVector().data(),
                          sv_mpi.getDataVector().size(), expected_sv.data(),
                          expected_sv.size()));
}