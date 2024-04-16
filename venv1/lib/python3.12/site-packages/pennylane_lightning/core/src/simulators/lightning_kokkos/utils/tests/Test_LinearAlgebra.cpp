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

#include "LinearAlgebraKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp" // exp2

/**
 * @file
 *  Tests linear algebra functionality defined for the class StateVectorKokkos.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::Util;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("Linear Algebra::SparseMV", "[Linear Algebra]", float,
                   double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;

    std::size_t num_qubits = 3;
    std::size_t data_size = exp2(num_qubits);

    std::vector<ComplexT> vectors = {{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                     {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                     {0.3, 0.4}, {0.4, 0.5}};

    const std::vector<ComplexT> result_refs = {
        {0.2, -0.1}, {-0.1, 0.2}, {0.2, 0.1}, {0.1, 0.2},
        {0.7, -0.2}, {-0.1, 0.6}, {0.6, 0.1}, {0.2, 0.7}};

    std::vector<size_t> indptr = {0, 2, 4, 6, 8, 10, 12, 14, 16};
    std::vector<size_t> indices = {0, 3, 1, 2, 1, 2, 0, 3,
                                   4, 7, 5, 6, 5, 6, 4, 7};
    std::vector<ComplexT> values = {
        {1.0, 0.0},  {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0},
        {0.0, -1.0}, {1.0, 0.0},  {0.0, 1.0}, {1.0, 0.0},
        {1.0, 0.0},  {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0},
        {0.0, -1.0}, {1.0, 0.0},  {0.0, 1.0}, {1.0, 0.0}};

    StateVectorKokkos<TestType> kokkos_vx{num_qubits};
    StateVectorKokkos<TestType> kokkos_vy{num_qubits};

    kokkos_vx.HostToDevice(vectors.data(), vectors.size());

    SECTION("Testing sparse matrix vector product:") {
        std::vector<ComplexT> result(data_size);
        Util::SparseMV_Kokkos<TestType>(
            kokkos_vx.getView(), kokkos_vy.getView(), indptr.data(),
            indptr.size(), indices.data(), values.data(), values.size());
        kokkos_vy.DeviceToHost(result.data(), result.size());

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(result[j]) == Approx(imag(result_refs[j])));
            CHECK(real(result[j]) == Approx(real(result_refs[j])));
        }
    }
}

TEMPLATE_TEST_CASE("Linear Algebra::axpy_Kokkos", "[Linear Algebra]", float,
                   double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;

    std::size_t num_qubits = 3;

    std::size_t data_size = exp2(num_qubits);

    ComplexT alpha = {2.0, 0.5};

    std::vector<ComplexT> v0 = {{0.0, 0.0}, {0.1, -0.1}, {0.1, 0.1},
                                {0.2, 0.1}, {0.2, 0.2},  {0.3, 0.3},
                                {0.4, 0.3}, {0.5, 0.4}};

    std::vector<ComplexT> v1 = {{-0.1, 0.2}, {0.2, -0.1}, {0.1, 0.2},
                                {0.2, 0.1},  {-0.2, 0.7}, {0.6, -0.1},
                                {0.1, 0.6},  {0.7, 0.2}};

    std::vector<ComplexT> result_refs = {
        {-0.1, 0.2}, {0.45, -0.25}, {0.25, 0.45}, {0.55, 0.4},
        {0.1, 1.2},  {1.05, 0.65},  {0.75, 1.4},  {1.5, 1.25}};

    StateVectorKokkos<TestType> kokkos_v0{num_qubits};
    StateVectorKokkos<TestType> kokkos_v1{num_qubits};

    kokkos_v0.HostToDevice(v0.data(), v0.size());
    kokkos_v1.HostToDevice(v1.data(), v1.size());

    SECTION("Testing imag of complex inner product:") {
        Util::axpy_Kokkos<TestType>(alpha, kokkos_v0.getView(),
                                    kokkos_v1.getView(), v0.size());
        std::vector<ComplexT> result(data_size);
        kokkos_v1.DeviceToHost(result.data(), result.size());

        for (std::size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(imag(result[j]) == Approx(imag(result_refs[j])));
            CHECK(real(result[j]) == Approx(real(result_refs[j])));
        }
    }
}
