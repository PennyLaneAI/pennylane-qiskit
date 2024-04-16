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
#include <complex>
#include <cstdio>
#include <vector>

#include <catch2/catch.hpp>

#include "MPILinearAlg.hpp"
#include "MPIManager.hpp"
#include "StateVectorCudaMPI.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp" // exp2

/**
 * @file
 *  Tests linear algebra functionality defined for the class StateVectorCudaMPI.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::Util;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("Linear Algebra::SparseMV", "[Linear Algebra]", float,
                   double) {
    using StateVectorT = StateVectorCudaMPI<TestType>;
    using ComplexT = StateVectorT::ComplexT;
    using CFP_t = StateVectorT::CFP_t;
    using IdxT = typename std::conditional<std::is_same<TestType, float>::value,
                                           int32_t, int64_t>::type;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    std::size_t num_qubits = 3;

    std::vector<ComplexT> state = {{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                   {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                   {0.3, 0.4}, {0.4, 0.5}};

    std::vector<ComplexT> result_refs = {{0.2, -0.1}, {-0.1, 0.2}, {0.2, 0.1},
                                         {0.1, 0.2},  {0.7, -0.2}, {-0.1, 0.6},
                                         {0.6, 0.1},  {0.2, 0.7}};

    std::vector<IdxT> indptr = {0, 2, 4, 6, 8, 10, 12, 14, 16};
    std::vector<IdxT> indices = {0, 3, 1, 2, 1, 2, 0, 3,
                                 4, 7, 5, 6, 5, 6, 4, 7};
    std::vector<ComplexT> values = {
        {1.0, 0.0},  {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0},
        {0.0, -1.0}, {1.0, 0.0},  {0.0, 1.0}, {1.0, 0.0},
        {1.0, 0.0},  {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0},
        {0.0, -1.0}, {1.0, 0.0},  {0.0, 1.0}, {1.0, 0.0}};

    size_t mpi_buffersize = 1;
    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
    size_t subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<ComplexT> local_state(subSvLength);
    std::vector<ComplexT> local_result_refs(subSvLength);

    mpi_manager.Scatter(state.data(), local_state.data(), subSvLength, 0);
    mpi_manager.Scatter(result_refs.data(), local_result_refs.data(),
                        subSvLength, 0);
    mpi_manager.Barrier();

    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    REQUIRE(nDevices >= 2);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);
    mpi_manager.Barrier();

    SECTION("Testing sparse matrix vector product:") {
        std::vector<ComplexT> local_result(local_state.size());

        StateVectorT sv_x(mpi_manager, dt_local, mpi_buffersize,
                          nGlobalIndexBits, nLocalIndexBits);
        StateVectorT sv_y(mpi_manager, dt_local, mpi_buffersize,
                          nGlobalIndexBits, nLocalIndexBits);
        sv_x.CopyHostDataToGpu(local_state, false);

        cuUtil::SparseMV_cuSparseMPI<IdxT, TestType, CFP_t>(
            mpi_manager, sv_x.getLength(), indptr.data(),
            static_cast<int64_t>(indptr.size()), indices.data(), values.data(),
            sv_x.getData(), sv_y.getData(),
            sv_x.getDataBuffer().getDevTag().getDeviceID(),
            sv_x.getDataBuffer().getDevTag().getStreamID(),
            sv_x.getCusparseHandle());

        mpi_manager.Barrier();

        sv_y.CopyGpuDataToHost(local_result.data(), local_result.size());

        for (std::size_t j = 0; j < local_result.size(); j++) {
            CHECK(imag(local_result[j]) == Approx(imag(local_result_refs[j])));
            CHECK(real(local_result[j]) == Approx(real(local_result_refs[j])));
        }
    }
}
