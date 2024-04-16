// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "SparseLinAlg.hpp"
#include "TestHelpers.hpp" // write_CSR_vectors
#include "Util.hpp"        // exp2

#if defined(_MSC_VER)
#pragma warning(disable : 4305)
#endif

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::Util;

using namespace Pennylane::LightningQubit::Util;

using std::complex;
using std::size_t;
using std::string;
using std::vector;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("apply_Sparse_Matrix", "[Sparse]", float, double) {
    size_t num_qubits = 3;
    size_t data_size = Util::exp2(num_qubits);

    std::vector<std::vector<complex<TestType>>> vectors = {
        {0.33160916, 0.90944626, 0.81097291, 0.46112135, 0.42801563, 0.38077181,
         0.23550137, 0.57416324},
        {{0.26752544, 0.00484225},
         {0.49189265, 0.21231633},
         {0.28691029, 0.87552205},
         {0.13499786, 0.63862517},
         {0.31748372, 0.25701515},
         {0.96968437, 0.69821151},
         {0.53674213, 0.58564544},
         {0.02213429, 0.3050882}}};

    const std::vector<std::vector<complex<TestType>>> result_refs = {
        {-1.15200034, -0.23313581, -0.5595947, -0.7778672, -0.41387753,
         -0.28274519, -0.71943368, 0.00705271},
        {{-0.24650151, -0.51256229},
         {-0.06254307, -0.66804797},
         {-0.33998022, 0.02458055},
         {-0.46939616, -0.49391203},
         {-0.7871985, -1.07982153},
         {0.11545852, -0.14444908},
         {-0.45507653, -0.41765428},
         {-0.78213328, -0.28539948}}};

    std::vector<size_t> row_map;
    std::vector<size_t> entries;
    std::vector<complex<TestType>> values;
    write_CSR_vectors(row_map, entries, values, data_size);

    SECTION("Testing sparse matrix dense vector product:") {
        for (size_t vec = 0; vec < vectors.size(); vec++) {
            std::vector<complex<TestType>> result = apply_Sparse_Matrix(
                vectors[vec].data(), vectors[vec].size(), row_map.data(),
                row_map.size(), entries.data(), values.data(), values.size());
            REQUIRE(result_refs[vec] == approx(result).margin(1e-6));
        };
    }
}