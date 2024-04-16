// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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

#include <complex>
#include <memory>
#include <vector>

#include "Error.hpp" // LightningException
#include "Observables.hpp"
#include "TestHelpers.hpp" // isApproxEqual, createZeroState, createProductState
#include "TypeList.hpp"
#include "Util.hpp" // TestVector
/**
 * @file
 *  Tests for Base Observable classes.
 */

/// @cond DEV
namespace {
using namespace Pennylane::Observables;

using Pennylane::Util::createProductState;
using Pennylane::Util::createZeroState;
using Pennylane::Util::isApproxEqual;
using Pennylane::Util::LightningException;
using Pennylane::Util::TestVector;
} // namespace
/// @endcond

#ifdef _ENABLE_PLGPU
constexpr bool BACKEND_FOUND = true;

#include "MPIManager.hpp"
#include "TestHelpersStateVectorsMPI.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU::Util;
} // namespace
  /// @endcond

#else
constexpr bool BACKEND_FOUND = false;
using TestStateVectoMPIBackends = Pennylane::Util::TypeList<void>;

template <class StateVector> struct StateVectorMPIToName {};
#endif

template <typename TypeList> void testNamedObsBase() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using NamedObsT = NamedObsBase<StateVectorT>;

        DYNAMIC_SECTION("Name of the Observable must be correct - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            REQUIRE(NamedObsT("PauliZ", {0}).getObsName() == "PauliZ[0]");
        }

        DYNAMIC_SECTION("Comparing objects names") {
            auto ob1 = NamedObsT("PauliX", {0});
            auto ob2 = NamedObsT("PauliX", {0});
            auto ob3 = NamedObsT("PauliZ", {0});

            REQUIRE(ob1 == ob2);
            REQUIRE(ob2 != ob3);
            REQUIRE(ob1 != ob3);
        }

        DYNAMIC_SECTION("Comparing objects wires") {
            auto ob1 = NamedObsT("PauliY", {0});
            auto ob2 = NamedObsT("PauliY", {0});
            auto ob3 = NamedObsT("PauliY", {1});

            REQUIRE(ob1 == ob2);
            REQUIRE(ob2 != ob3);
            REQUIRE(ob1 != ob3);
        }

        DYNAMIC_SECTION("Comparing objects parameters") {
            auto ob1 = NamedObsT("RZ", {0}, {0.4});
            auto ob2 = NamedObsT("RZ", {0}, {0.4});
            auto ob3 = NamedObsT("RZ", {0}, {0.1});

            REQUIRE(ob1 == ob2);
            REQUIRE(ob2 != ob3);
            REQUIRE(ob1 != ob3);
        }

        testNamedObsBase<typename TypeList::Next>();
    }
}

TEST_CASE("Methods implemented in the NamedObsBase class", "[NamedObsBase]") {
    if constexpr (BACKEND_FOUND) {
        testNamedObsBase<TestStateVectorMPIBackends>();
    }
}

template <typename TypeList> void testHermitianObsBase() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using ComplexT = typename StateVectorT::ComplexT;
        using HermitianObsT = HermitianObsBase<StateVectorT>;

        DYNAMIC_SECTION("HermitianObs only accepts correct arguments - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            auto ob1 =
                HermitianObsT{std::vector<ComplexT>{0.0, 0.0, 0.0, 0.0}, {0}};
            auto ob2 =
                HermitianObsT{std::vector<ComplexT>(16, ComplexT{}), {0, 1}};
            REQUIRE_THROWS_AS(
                HermitianObsT(std::vector<ComplexT>{0.0, 0.0, 0.0}, {0}),
                LightningException);
            REQUIRE_THROWS_AS(
                HermitianObsT(std::vector<ComplexT>{0.0, 0.0, 0.0, 0.0, 0.0},
                              {0, 1}),
                LightningException);
        }

        DYNAMIC_SECTION("getObsName - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            REQUIRE(
                HermitianObsT(std::vector<ComplexT>{1.0, 0.0, 2.0, 0.0}, {0})
                    .getObsName() == "Hermitian");
        }

        DYNAMIC_SECTION("Comparing objects matrices - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            auto ob1 =
                HermitianObsT{std::vector<ComplexT>{1.0, 0.0, 0.0, 0.0}, {0}};
            auto ob2 =
                HermitianObsT{std::vector<ComplexT>{1.0, 0.0, 0.0, 0.0}, {0}};
            auto ob3 =
                HermitianObsT{std::vector<ComplexT>{0.0, 1.0, 0.0, 0.0}, {0}};
            REQUIRE(ob1 == ob2);
            REQUIRE(ob1 != ob3);
            REQUIRE(ob2 != ob3);
        }

        DYNAMIC_SECTION("Comparing objects wires - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            auto ob1 =
                HermitianObsT{std::vector<ComplexT>{1.0, 0.0, -1.0, 0.0}, {0}};
            auto ob2 =
                HermitianObsT{std::vector<ComplexT>{1.0, 0.0, -1.0, 0.0}, {0}};
            auto ob3 =
                HermitianObsT{std::vector<ComplexT>{1.0, 0.0, -1.0, 0.0}, {1}};
            REQUIRE(ob1 == ob2);
            REQUIRE(ob1 != ob3);
            REQUIRE(ob2 != ob3);
        }

        testHermitianObsBase<typename TypeList::Next>();
    }
}

TEST_CASE("Methods implemented in the HermitianObsBase class",
          "[HermitianObsBase]") {
    if constexpr (BACKEND_FOUND) {
        testHermitianObsBase<TestStateVectorMPIBackends>();
    }
}

template <typename TypeList> void testTensorProdObsBase() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;
        using HermitianObsT = HermitianObsBase<StateVectorT>;
        using NamedObsT = NamedObsBase<StateVectorT>;
        using TensorProdObsT = TensorProdObsBase<StateVectorT>;

        DYNAMIC_SECTION("Overlapping wires throw an exception - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            auto ob1 = std::make_shared<HermitianObsT>(
                std::vector<ComplexT>(16, ComplexT{0.0, 0.0}),
                std::vector<size_t>{0, 1});
            auto ob2_1 =
                std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{1});
            auto ob2_2 =
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{2});
            auto ob2 = TensorProdObsT::create({ob2_1, ob2_2});

            REQUIRE_THROWS_AS(TensorProdObsT::create({ob1, ob2}),
                              LightningException);
        }

        DYNAMIC_SECTION(
            "Constructing an observable with non-overlapping wires - "
            << StateVectorMPIToName<StateVectorT>::name) {
            auto ob1 = std::make_shared<HermitianObsT>(
                std::vector<ComplexT>(16, ComplexT{0.0, 0.0}),
                std::vector<size_t>{0, 1});
            auto ob2_1 =
                std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{2});
            auto ob2_2 =
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{3});
            auto ob2 = TensorProdObsT::create({ob2_1, ob2_2});

            REQUIRE_NOTHROW(TensorProdObsT::create({ob1, ob2}));
        }

        DYNAMIC_SECTION("getObsName - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            auto ob = TensorProdObsT(
                std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{0}),
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{1}));
            REQUIRE(ob.getObsName() == "PauliX[0] @ PauliZ[1]");
        }

        DYNAMIC_SECTION("Compare tensor product observables"
                        << StateVectorMPIToName<StateVectorT>::name) {
            auto ob1 = TensorProdObsT{
                std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{0}),
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{1})};
            auto ob2 = TensorProdObsT{
                std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{0}),
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{1})};
            auto ob3 = TensorProdObsT{
                std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{0}),
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{2})};
            auto ob4 = TensorProdObsT{
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{0}),
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{1})};

            auto ob5 = TensorProdObsT{
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{0})};

            REQUIRE(ob1 == ob2);
            REQUIRE(ob1 != ob3);
            REQUIRE(ob1 != ob4);
            REQUIRE(ob1 != ob5);
        }

        DYNAMIC_SECTION("Tensor product applies to a statevector correctly"
                        << StateVectorMPIToName<StateVectorT>::name) {
            using VectorT = TestVector<ComplexT>;

            auto obs = TensorProdObsT{
                std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{0}),
                std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{2}),
            };

            SECTION("Test using |1+0>") {
                MPIManager mpi_manager(MPI_COMM_WORLD);
                REQUIRE(mpi_manager.getSize() == 2);

                const size_t num_qubits = 3;
                size_t mpi_buffersize = 1;

                int nGlobalIndexBits = std::bit_width(static_cast<unsigned int>(
                                           mpi_manager.getSize())) -
                                       1;
                int nLocalIndexBits = num_qubits - nGlobalIndexBits;
                size_t subSvLength = 1 << nLocalIndexBits;
                mpi_manager.Barrier();

                int nDevices = 0; // Number of GPU devices per node
                cudaGetDeviceCount(&nDevices);
                REQUIRE(nDevices >= 2);
                int deviceId = mpi_manager.getRank() % nDevices;
                cudaSetDevice(deviceId);
                DevTag<int> dt_local(deviceId, 0);

                VectorT st_data =
                    createProductState<PrecisionT, ComplexT>("1+0");

                std::vector<ComplexT> sv_data_local(subSvLength);
                mpi_manager.Scatter(st_data.data(), sv_data_local.data(),
                                    subSvLength, 0);

                StateVectorT sv(mpi_manager, dt_local, mpi_buffersize,
                                nGlobalIndexBits, nLocalIndexBits);
                sv.CopyHostDataToGpu(sv_data_local.data(), sv_data_local.size(),
                                     false);
                mpi_manager.Barrier();

                obs.applyInPlace(sv);

                VectorT expected =
                    createProductState<PrecisionT, ComplexT>("0+1");
                std::vector<ComplexT> expected_local(subSvLength);
                mpi_manager.Scatter(expected.data(), expected_local.data(),
                                    subSvLength, 0);

                REQUIRE(isApproxEqual(
                    sv.getDataVector().data(), sv.getDataVector().size(),
                    expected_local.data(), expected_local.size()));
            }

            SECTION("Test using |+-01>") {
                MPIManager mpi_manager(MPI_COMM_WORLD);
                REQUIRE(mpi_manager.getSize() == 2);

                const size_t num_qubits = 4;
                size_t mpi_buffersize = 1;

                int nGlobalIndexBits = std::bit_width(static_cast<unsigned int>(
                                           mpi_manager.getSize())) -
                                       1;
                int nLocalIndexBits = num_qubits - nGlobalIndexBits;
                size_t subSvLength = 1 << nLocalIndexBits;
                mpi_manager.Barrier();

                int nDevices = 0; // Number of GPU devices per node
                cudaGetDeviceCount(&nDevices);
                REQUIRE(nDevices >= 2);
                int deviceId = mpi_manager.getRank() % nDevices;
                cudaSetDevice(deviceId);
                DevTag<int> dt_local(deviceId, 0);

                VectorT st_data =
                    createProductState<PrecisionT, ComplexT>("+-01");

                std::vector<ComplexT> sv_data_local(subSvLength);
                mpi_manager.Scatter(st_data.data(), sv_data_local.data(),
                                    subSvLength, 0);

                StateVectorT sv(mpi_manager, dt_local, mpi_buffersize,
                                nGlobalIndexBits, nLocalIndexBits);
                sv.CopyHostDataToGpu(sv_data_local.data(), sv_data_local.size(),
                                     false);
                mpi_manager.Barrier();

                obs.applyInPlace(sv);

                VectorT expected =
                    createProductState<PrecisionT, ComplexT>("+-11");
                std::vector<ComplexT> expected_local(subSvLength);
                mpi_manager.Scatter(expected.data(), expected_local.data(),
                                    subSvLength, 0);

                REQUIRE(isApproxEqual(
                    sv.getDataVector().data(), sv.getDataVector().size(),
                    expected_local.data(), expected_local.size()));
            }
        }

        testTensorProdObsBase<typename TypeList::Next>();
    }
}

TEST_CASE("Methods implemented in the TensorProdObsBase class",
          "[TensorProdObsBase]") {
    if constexpr (BACKEND_FOUND) {
        testTensorProdObsBase<TestStateVectorMPIBackends>();
    }
}

template <typename TypeList> void testHamiltonianBase() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using NamedObsT = NamedObsBase<StateVectorT>;
        using TensorProdObsT = TensorProdObsBase<StateVectorT>;
        using HamiltonianT = HamiltonianBase<StateVectorT>;

        MPIManager mpi_manager(MPI_COMM_WORLD);
        REQUIRE(mpi_manager.getSize() == 2);

        const size_t num_qubits = 3;
        size_t mpi_buffersize = 1;

        int nGlobalIndexBits =
            std::bit_width(static_cast<unsigned int>(mpi_manager.getSize())) -
            1;
        int nLocalIndexBits = num_qubits - nGlobalIndexBits;
        mpi_manager.Barrier();

        int nDevices = 0; // Number of GPU devices per node
        cudaGetDeviceCount(&nDevices);
        REQUIRE(nDevices >= 2);
        int deviceId = mpi_manager.getRank() % nDevices;
        cudaSetDevice(deviceId);
        DevTag<int> dt_local(deviceId, 0);

        const auto h = PrecisionT{0.809}; // half of the golden ratio

        auto zz = std::make_shared<TensorProdObsT>(
            std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{0}),
            std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{1}));

        auto x1 = std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{0});
        auto x2 = std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{1});

        DYNAMIC_SECTION(
            "Hamiltonian constructor only accepts valid arguments - "
            << StateVectorMPIToName<StateVectorT>::name) {
            REQUIRE_NOTHROW(
                HamiltonianT::create({PrecisionT{1.0}, h, h}, {zz, x1, x2}));

            REQUIRE_THROWS_AS(
                HamiltonianT::create({PrecisionT{1.0}, h}, {zz, x1, x2}),
                LightningException);

            DYNAMIC_SECTION("getObsName - "
                            << StateVectorMPIToName<StateVectorT>::name) {
                auto X0 = std::make_shared<NamedObsT>("PauliX",
                                                      std::vector<size_t>{0});
                auto Z2 = std::make_shared<NamedObsT>("PauliZ",
                                                      std::vector<size_t>{2});

                REQUIRE(
                    HamiltonianT::create({0.3, 0.5}, {X0, Z2})->getObsName() ==
                    "Hamiltonian: { 'coeffs' : [0.3, 0.5], "
                    "'observables' : [PauliX[0], PauliZ[2]]}");
            }

            DYNAMIC_SECTION("Compare Hamiltonians - "
                            << StateVectorMPIToName<StateVectorT>::name) {
                auto X0 = std::make_shared<NamedObsT>("PauliX",
                                                      std::vector<size_t>{0});
                auto X1 = std::make_shared<NamedObsT>("PauliX",
                                                      std::vector<size_t>{1});
                auto X2 = std::make_shared<NamedObsT>("PauliX",
                                                      std::vector<size_t>{2});

                auto Y0 = std::make_shared<NamedObsT>("PauliY",
                                                      std::vector<size_t>{0});
                auto Y1 = std::make_shared<NamedObsT>("PauliY",
                                                      std::vector<size_t>{1});
                auto Y2 = std::make_shared<NamedObsT>("PauliY",
                                                      std::vector<size_t>{2});

                auto Z0 = std::make_shared<NamedObsT>("PauliZ",
                                                      std::vector<size_t>{0});
                auto Z1 = std::make_shared<NamedObsT>("PauliZ",
                                                      std::vector<size_t>{1});
                auto Z2 = std::make_shared<NamedObsT>("PauliZ",
                                                      std::vector<size_t>{2});

                auto ham1 = HamiltonianT::create(
                    {0.8, 0.5, 0.7},
                    {
                        std::make_shared<TensorProdObsT>(X0, Y1, Z2),
                        std::make_shared<TensorProdObsT>(Z0, X1, Y2),
                        std::make_shared<TensorProdObsT>(Y0, Z1, X2),
                    });

                auto ham2 = HamiltonianT::create(
                    {0.8, 0.5, 0.7},
                    {
                        std::make_shared<TensorProdObsT>(X0, Y1, Z2),
                        std::make_shared<TensorProdObsT>(Z0, X1, Y2),
                        std::make_shared<TensorProdObsT>(Y0, Z1, X2),
                    });

                auto ham3 = HamiltonianT::create(
                    {0.8, 0.5, 0.642},
                    {
                        std::make_shared<TensorProdObsT>(X0, Y1, Z2),
                        std::make_shared<TensorProdObsT>(Z0, X1, Y2),
                        std::make_shared<TensorProdObsT>(Y0, Z1, X2),
                    });

                auto ham4 = HamiltonianT::create(
                    {0.8, 0.5},
                    {
                        std::make_shared<TensorProdObsT>(X0, Y1, Z2),
                        std::make_shared<TensorProdObsT>(Z0, X1, Y2),
                    });

                auto ham5 = HamiltonianT::create(
                    {0.8, 0.5, 0.7},
                    {
                        std::make_shared<TensorProdObsT>(X0, Y1, Z2),
                        std::make_shared<TensorProdObsT>(Z0, X1, Y2),
                        std::make_shared<TensorProdObsT>(Y0, Z1, Y2),
                    });

                REQUIRE(*ham1 == *ham2);
                REQUIRE(*ham1 != *ham3);
                REQUIRE(*ham2 != *ham3);
                REQUIRE(*ham2 != *ham4);
                REQUIRE(*ham1 != *ham5);
            }

            DYNAMIC_SECTION("getWires - "
                            << StateVectorMPIToName<StateVectorT>::name) {
                auto Z0 = std::make_shared<NamedObsT>("PauliZ",
                                                      std::vector<size_t>{0});
                auto Z5 = std::make_shared<NamedObsT>("PauliZ",
                                                      std::vector<size_t>{5});
                auto Z9 = std::make_shared<NamedObsT>("PauliZ",
                                                      std::vector<size_t>{9});

                auto ham1 = HamiltonianT::create({0.8, 0.5, 0.7}, {Z0, Z5, Z9});

                REQUIRE(ham1->getWires() == std::vector<size_t>{0, 5, 9});
            }

            DYNAMIC_SECTION("applyInPlace must fail - "
                            << StateVectorMPIToName<StateVectorT>::name) {
                auto ham =
                    HamiltonianT::create({PrecisionT{1.0}, h, h}, {zz, x1, x2});

                StateVectorT sv_mpi(mpi_manager, dt_local, mpi_buffersize,
                                    nGlobalIndexBits, nLocalIndexBits);
                sv_mpi.initSV();

                REQUIRE_THROWS_AS(ham->applyInPlace(sv_mpi),
                                  LightningException);
            }
        }
        testHamiltonianBase<typename TypeList::Next>();
    }
}

TEST_CASE("Methods implemented in the HamiltonianBase class",
          "[HamiltonianBase]") {
    if constexpr (BACKEND_FOUND) {
        testHamiltonianBase<TestStateVectorMPIBackends>();
    }
}

template <typename TypeList> void testSparseHamiltonianBase() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;

        const std::size_t num_qubits = 3;
        std::mt19937 re{1337};

        MPIManager mpi_manager(MPI_COMM_WORLD);

        size_t mpi_buffersize = 1;
        size_t nGlobalIndexBits =
            std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
        size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
        size_t subSvLength = 1 << nLocalIndexBits;

        int nDevices = 0;
        cudaGetDeviceCount(&nDevices);
        int deviceId = mpi_manager.getRank() % nDevices;
        cudaSetDevice(deviceId);
        DevTag<int> dt_local(deviceId, 0);
        mpi_manager.Barrier();

        std::vector<ComplexT> expected_sv(subSvLength);
        std::vector<ComplexT> local_state(subSvLength);

        auto init_state =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        mpi_manager.Scatter(init_state.data(), local_state.data(), subSvLength,
                            0);
        mpi_manager.Barrier();

        DYNAMIC_SECTION("applyInPlace must fail - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            auto sparseH = SparseHamiltonianBase<StateVectorT>::create(
                {ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0},
                 ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0},
                 ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}},
                {7, 6, 5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5, 6, 7, 8},
                {0, 1, 2});

            StateVectorT sv_mpi(mpi_manager, dt_local, mpi_buffersize,
                                nGlobalIndexBits, nLocalIndexBits);

            sv_mpi.CopyHostDataToGpu(local_state, false);

            REQUIRE_THROWS_AS(sparseH->applyInPlace(sv_mpi),
                              LightningException);
        }

        DYNAMIC_SECTION("SparseHamiltonianBase - isEqual - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            auto sparseH0 = SparseHamiltonianBase<StateVectorT>::create(
                {ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0},
                 ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0},
                 ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}},
                {7, 6, 5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5, 6, 7, 8},
                {0, 1, 2});
            auto sparseH1 = SparseHamiltonianBase<StateVectorT>::create(
                {ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0},
                 ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0},
                 ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}},
                {7, 6, 5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5, 6, 7, 8},
                {0, 1, 2});
            auto sparseH2 = SparseHamiltonianBase<StateVectorT>::create(
                {ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0},
                 ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0},
                 ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}},
                {8, 6, 5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5, 6, 7, 8},
                {0, 1, 2});

            REQUIRE(*sparseH0 == *sparseH1);
            REQUIRE(*sparseH0 != *sparseH2);
        }

        testSparseHamiltonianBase<typename TypeList::Next>();
    }
}

TEST_CASE("Methods implemented in the SparseHamiltonianBase class",
          "[SparseHamiltonianBase]") {
    if constexpr (BACKEND_FOUND) {
        testSparseHamiltonianBase<TestStateVectorMPIBackends>();
    }
}
