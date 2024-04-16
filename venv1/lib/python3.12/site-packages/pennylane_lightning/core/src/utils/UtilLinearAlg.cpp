// Copyright 2018-2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file
 * UtilLinearAlg functions explicit instantiation.
 */

#include <complex>
#include <vector>

#include "UtilLinearAlg.hpp"

template void Pennylane::Util::compute_diagonalizing_gates<float>(
    int n, int lda, const std::vector<std::complex<float>> &Ah,
    std::vector<float> &eigenVals, std::vector<std::complex<float>> &unitary);

template void Pennylane::Util::compute_diagonalizing_gates<double>(
    int n, int lda, const std::vector<std::complex<double>> &Ah,
    std::vector<double> &eigenVals, std::vector<std::complex<double>> &unitary);
