
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

#include <cmath>
#include <complex>
#include <vector>

#include "TestHelpers.hpp"
#include "UtilLinearAlg.hpp"
#include <catch2/catch.hpp>

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::Util;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("Util::compute_diagonalizing_gates", "[Util][LinearAlgebra]",
                   float, double) {
    SECTION("For complex type") {
        std::vector<std::complex<TestType>> A{
            {-6.0, 0.0}, {2.0, 1.0}, {2.0, -1.0}, {0.0, 0.0}};
        std::vector<TestType> expectedEigenVals = {-6.741657, 0.741657};
        std::vector<std::complex<TestType>> expectedUnitaries = {
            {-0.94915323, 0.0},
            {0.2815786, 0.1407893},
            {0.31481445, 0.0},
            {0.84894846, 0.42447423}};
        size_t N = 2;
        size_t LDA = 2;
        std::vector<TestType> eigenVals;
        std::vector<std::complex<TestType>> Unitaries;
        compute_diagonalizing_gates(N, LDA, A, eigenVals, Unitaries);

        for (size_t i = 0; i < expectedEigenVals.size(); i++) {
            CHECK(eigenVals[i] == Approx(expectedEigenVals[i]).margin(1e-6));
        }

        for (size_t i = 0; i < Unitaries.size(); i++) {
            CHECK(Unitaries[i].real() ==
                  Approx(expectedUnitaries[i].real()).margin(1e-6));
            CHECK(Unitaries[i].imag() ==
                  Approx(expectedUnitaries[i].imag()).margin(1e-6));
        }
    }
}
