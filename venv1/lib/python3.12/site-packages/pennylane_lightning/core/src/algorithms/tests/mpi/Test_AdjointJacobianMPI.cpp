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

#include "JacobianData.hpp"
#include "TestHelpers.hpp" //PLApprox
/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

#ifdef _ENABLE_PLGPU
constexpr bool BACKEND_FOUND = true;
#include "AdjointJacobianGPU.hpp"
#include "AdjointJacobianGPUMPI.hpp"
#include "JacobianDataMPI.hpp"
#include "MPIManager.hpp"
#include "ObservablesGPU.hpp"
#include "ObservablesGPUMPI.hpp"
#include "StateVectorCudaMPI.hpp"
#include "StateVectorCudaManaged.hpp"
#include "TestHelpersStateVectorsMPI.hpp"
#include "TestHelpersWires.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU::Algorithms;
using namespace Pennylane::LightningGPU::Observables;
} // namespace
  /// @endcond

#else
constexpr bool BACKEND_FOUND = false;
using TestStateVectorMPIBackends = Pennylane::Util::TypeList<void>;

template <class StateVector> struct StateVectorMPIToName {};
#endif

template <typename TypeList> void testAdjointJacobian() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;

        MPIManager mpi_manager(MPI_COMM_WORLD);
        REQUIRE(mpi_manager.getSize() == 2);

        const std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};

        AdjointJacobianMPI<StateVectorT> adj;

        DYNAMIC_SECTION("Op=[RX,RX,RX], Obs=[Z,Z,Z] - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            std::vector<size_t> tp{0, 1, 2};
            const size_t num_qubits = 3;
            const size_t num_params = 3;
            const size_t num_obs = 3;
            std::vector<PrecisionT> jacobian(num_obs * tp.size(), 0);

            size_t mpi_buffersize = 1;

            int nGlobalIndexBits = std::bit_width(static_cast<unsigned int>(
                                       mpi_manager.getSize())) -
                                   1;
            int nLocalIndexBits = num_qubits - nGlobalIndexBits;
            mpi_manager.Barrier();

            int nDevices = 0; // Number of GPU devices per node
            cudaGetDeviceCount(&nDevices);
            REQUIRE(nDevices >= 2);
            int deviceId = mpi_manager.getRank() % nDevices;
            cudaSetDevice(deviceId);
            DevTag<int> dt_local(deviceId, 0);

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
                num_params, psi, {obs1, obs2, obs3}, ops, tp};
            adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

            CAPTURE(jacobian);
            CHECK(-sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
            CHECK(-sin(param[1]) ==
                  Approx(jacobian[1 * num_params + 1]).margin(1e-7));
            CHECK(-sin(param[2]) ==
                  Approx(jacobian[2 * num_params + 2]).margin(1e-7));
        }

        DYNAMIC_SECTION("Op=[RX,RX,RX], Obs=[Z,Z,Z], TParams=[0,2] - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
            std::vector<size_t> t_params{0, 2};
            const size_t num_qubits = 3;
            const size_t num_params = 3;
            const size_t num_obs = 3;
            std::vector<PrecisionT> jacobian(num_obs * t_params.size(), 0);

            size_t mpi_buffersize = 1;

            int nGlobalIndexBits = std::bit_width(static_cast<unsigned int>(
                                       mpi_manager.getSize())) -
                                   1;
            int nLocalIndexBits = num_qubits - nGlobalIndexBits;
            mpi_manager.Barrier();

            int nDevices = 0; // Number of GPU devices per node
            cudaGetDeviceCount(&nDevices);
            REQUIRE(nDevices >= 2);
            int deviceId = mpi_manager.getRank() % nDevices;
            cudaSetDevice(deviceId);
            DevTag<int> dt_local(deviceId, 0);

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
                num_params, psi, {obs1, obs2, obs3}, ops, t_params};

            adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

            CAPTURE(jacobian);
            CHECK(-sin(param[0]) == Approx(jacobian[0]).margin(1e-7));
            CHECK(0 == Approx(jacobian[1 * t_params.size() + 1]).margin(1e-7));
            CHECK(-sin(param[2]) ==
                  Approx(jacobian[2 * t_params.size() + 1]).margin(1e-7));
        }

        DYNAMIC_SECTION("Op=[RX,RX,RX], Obs=[ZZZ] - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
            std::vector<size_t> tp{0, 1, 2};
            const size_t num_qubits = 3;
            const size_t num_params = 3;
            const size_t num_obs = 1;
            std::vector<PrecisionT> jacobian(num_obs * tp.size(), 0);

            size_t mpi_buffersize = 1;

            int nGlobalIndexBits = std::bit_width(static_cast<unsigned int>(
                                       mpi_manager.getSize())) -
                                   1;
            int nLocalIndexBits = num_qubits - nGlobalIndexBits;
            mpi_manager.Barrier();

            int nDevices = 0; // Number of GPU devices per node
            cudaGetDeviceCount(&nDevices);
            REQUIRE(nDevices >= 2);
            int deviceId = mpi_manager.getRank() % nDevices;
            cudaSetDevice(deviceId);
            DevTag<int> dt_local(deviceId, 0);

            StateVectorT psi(mpi_manager, dt_local, mpi_buffersize,
                             nGlobalIndexBits, nLocalIndexBits);
            psi.initSV();

            const auto obs = std::make_shared<TensorProdObsMPI<StateVectorT>>(
                std::make_shared<NamedObsMPI<StateVectorT>>(
                    "PauliZ", std::vector<size_t>{0}),
                std::make_shared<NamedObsMPI<StateVectorT>>(
                    "PauliZ", std::vector<size_t>{1}),
                std::make_shared<NamedObsMPI<StateVectorT>>(
                    "PauliZ", std::vector<size_t>{2}));
            auto ops = OpsData<StateVectorT>(
                {"RX", "RX", "RX"}, {{param[0]}, {param[1]}, {param[2]}},
                {{0}, {1}, {2}}, {false, false, false});

            JacobianDataMPI<StateVectorT> tape{num_params, psi, {obs}, ops, tp};

            adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

            CAPTURE(jacobian);

            // Computed with parameter shift
            CHECK(-0.1755096592645253 == Approx(jacobian[0]).margin(1e-7));
            CHECK(0.26478810666384334 == Approx(jacobian[1]).margin(1e-7));
            CHECK(-0.6312451595102775 == Approx(jacobian[2]).margin(1e-7));
        }

        DYNAMIC_SECTION("Op=Mixed, Obs=[XXX] - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
            std::vector<size_t> tp{0, 1, 2, 3, 4, 5};
            const size_t num_qubits = 3;
            const size_t num_params = 6;
            const size_t num_obs = 1;
            std::vector<PrecisionT> jacobian(num_obs * tp.size(), 0);

            size_t mpi_buffersize = 1;

            int nGlobalIndexBits = std::bit_width(static_cast<unsigned int>(
                                       mpi_manager.getSize())) -
                                   1;
            int nLocalIndexBits = num_qubits - nGlobalIndexBits;
            mpi_manager.Barrier();

            int nDevices = 0; // Number of GPU devices per node
            cudaGetDeviceCount(&nDevices);
            REQUIRE(nDevices >= 2);
            int deviceId = mpi_manager.getRank() % nDevices;
            cudaSetDevice(deviceId);
            DevTag<int> dt_local(deviceId, 0);

            StateVectorT psi(mpi_manager, dt_local, mpi_buffersize,
                             nGlobalIndexBits, nLocalIndexBits);
            psi.initSV();

            const auto obs = std::make_shared<TensorProdObsMPI<StateVectorT>>(
                std::make_shared<NamedObsMPI<StateVectorT>>(
                    "PauliX", std::vector<size_t>{0}),
                std::make_shared<NamedObsMPI<StateVectorT>>(
                    "PauliX", std::vector<size_t>{1}),
                std::make_shared<NamedObsMPI<StateVectorT>>(
                    "PauliX", std::vector<size_t>{2}));
            std::vector<ComplexT> cnot{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0};
            auto ops = OpsData<StateVectorT>(
                {"RZ", "RY", "RZ", "QubitUnitary", "CNOT", "RZ", "RY", "RZ"},
                {{param[0]},
                 {param[1]},
                 {param[2]},
                 {},
                 {},
                 {param[0]},
                 {param[1]},
                 {param[2]}},
                {{0}, {0}, {0}, {0, 1}, {1, 2}, {1}, {1}, {1}},
                {false, false, false, false, false, false, false, false},
                std::vector<std::vector<ComplexT>>{
                    {}, {}, {}, cnot, {}, {}, {}, {}},
                std::vector<std::vector<size_t>>{
                    {}, {}, {}, {}, {}, {}, {}, {}},
                std::vector<std::vector<bool>>{{}, {}, {}, {}, {}, {}, {}, {}});

            JacobianDataMPI<StateVectorT> tape{num_params, psi, {obs}, ops, tp};

            adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

            CAPTURE(jacobian);

            // Computed with PennyLane using default.qubit.adjoint_jacobian
            CHECK(0.0 == Approx(jacobian[0]).margin(1e-7));
            CHECK(-0.674214427 == Approx(jacobian[1]).margin(1e-7));
            CHECK(0.275139672 == Approx(jacobian[2]).margin(1e-7));
            CHECK(0.275139672 == Approx(jacobian[3]).margin(1e-7));
            CHECK(-0.0129093062 == Approx(jacobian[4]).margin(1e-7));
            CHECK(0.323846156 == Approx(jacobian[5]).margin(1e-7));
        }

        DYNAMIC_SECTION("Op=[RX,RX,RX], Obs=Ham[Z0+Z1+Z2], TParams=[0,2] - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
            std::vector<size_t> t_params{0, 2};
            const size_t num_qubits = 3;
            const size_t num_params = 3;
            const size_t num_obs = 1;
            std::vector<PrecisionT> jacobian(num_obs * t_params.size(), 0);

            size_t mpi_buffersize = 1;

            int nGlobalIndexBits = std::bit_width(static_cast<unsigned int>(
                                       mpi_manager.getSize())) -
                                   1;
            int nLocalIndexBits = num_qubits - nGlobalIndexBits;
            mpi_manager.Barrier();

            int nDevices = 0; // Number of GPU devices per node
            cudaGetDeviceCount(&nDevices);
            REQUIRE(nDevices >= 2);
            int deviceId = mpi_manager.getRank() % nDevices;
            cudaSetDevice(deviceId);
            DevTag<int> dt_local(deviceId, 0);

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

            JacobianDataMPI<StateVectorT> tape{
                num_params, psi, {ham}, ops, t_params};
            adj.adjointJacobian(std::span{jacobian}, tape, psi, true);

            CAPTURE(jacobian);
            CHECK((-0.47 * sin(param[0]) == Approx(jacobian[0]).margin(1e-7)));
            CHECK((-0.96 * sin(param[2]) == Approx(jacobian[1]).margin(1e-7)));
        }

        DYNAMIC_SECTION("HermitianObs - "
                        << StateVectorMPIToName<StateVectorT>::name) {
            std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
            std::vector<size_t> t_params{0, 2};
            const size_t num_qubits = 3;
            const size_t num_params = 3;
            const size_t num_obs = 1;
            std::vector<PrecisionT> jacobian1(num_obs * t_params.size(), 0);
            std::vector<PrecisionT> jacobian2(num_obs * t_params.size(), 0);

            size_t mpi_buffersize = 1;

            int nGlobalIndexBits = std::bit_width(static_cast<unsigned int>(
                                       mpi_manager.getSize())) -
                                   1;
            int nLocalIndexBits = num_qubits - nGlobalIndexBits;
            mpi_manager.Barrier();

            int nDevices = 0; // Number of GPU devices per node
            cudaGetDeviceCount(&nDevices);
            REQUIRE(nDevices >= 2);
            int deviceId = mpi_manager.getRank() % nDevices;
            cudaSetDevice(deviceId);
            DevTag<int> dt_local(deviceId, 0);

            StateVectorT psi(mpi_manager, dt_local, mpi_buffersize,
                             nGlobalIndexBits, nLocalIndexBits);
            psi.initSV();

            auto obs1 = std::make_shared<TensorProdObsMPI<StateVectorT>>(
                std::make_shared<NamedObsMPI<StateVectorT>>(
                    "PauliZ", std::vector<size_t>{0}),
                std::make_shared<NamedObsMPI<StateVectorT>>(
                    "PauliZ", std::vector<size_t>{1}));
            auto obs2 = std::make_shared<HermitianObsMPI<StateVectorT>>(
                std::vector<ComplexT>{1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                                      0, 0, 1},
                std::vector<size_t>{0, 1});

            auto ops = OpsData<StateVectorT>(
                {"RX", "RX", "RX"}, {{param[0]}, {param[1]}, {param[2]}},
                {{0}, {1}, {2}}, {false, false, false});

            JacobianDataMPI<StateVectorT> tape1{
                num_params, psi, {obs1}, ops, t_params};

            JacobianDataMPI<StateVectorT> tape2{
                num_params, psi, {obs2}, ops, t_params};
            adj.adjointJacobian(std::span{jacobian1}, tape1, psi, true);
            adj.adjointJacobian(std::span{jacobian2}, tape2, psi, true);

            CHECK((jacobian1 == PLApprox(jacobian2).margin(1e-7)));
        }

        testAdjointJacobian<typename TypeList::Next>();
    }
}

TEST_CASE("Algorithms::adjointJacobian", "[Algorithms]") {
    if constexpr (BACKEND_FOUND) {
        testAdjointJacobian<TestStateVectorMPIBackends>();
    }
}