
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
 * [@file](https://github.com/file)
 * LAPACK wrapper functions declarations.
 */
#pragma once

#include <algorithm>
#include <complex>
#include <vector>

/// @cond DEV
namespace {
extern "C" {
// LAPACK routine for complex Hermitian eigensystems
extern void cheev_(const char *jobz, const char *uplo, const int *n,
                   std::complex<float> *a, const int *lda, float *w,
                   std::complex<float> *work, const int *lwork, float *rwork,
                   int *info);
extern void zheev_(const char *jobz, const char *uplo, const int *n,
                   std::complex<double> *a, const int *lda, double *w,
                   std::complex<double> *work, const int *lwork, double *rwork,
                   int *info);
}
} // namespace
/// @endcond

namespace Pennylane::Util {

/**
 * @brief Decompose Hermitian matrix into diagonal matrix and unitaries
 *
 * @tparam T Data type.
 *
 * @param n Number of columns.
 * @param lda Number of rows.
 * @param Ah Hermitian matrix to be decomposed.
 * @param eigenVals eigenvalue results.
 * @param unitaries unitary result.
 */
template <typename T>
void compute_diagonalizing_gates(int n, int lda,
                                 const std::vector<std::complex<T>> &Ah,
                                 std::vector<T> &eigenVals,
                                 std::vector<std::complex<T>> &unitary) {
    eigenVals.clear();
    eigenVals.resize(n);
    unitary = std::vector<std::complex<T>>(n * n, {0, 0});

    std::vector<std::complex<T>> ah(n * lda, {0.0, 0.0});

    // TODO optmize transpose
    for (size_t i = 0; i < static_cast<size_t>(n); i++) {
        for (size_t j = 0; j <= i; j++) {
            ah[j * n + i] = Ah[i * lda + j];
        }
    }

    char jobz = 'V'; // Enable both eigenvalues and eigenvectors computation
    char uplo = 'L'; // Upper triangle of matrix is stored
    std::vector<std::complex<T>> work_query(1); // Vector for optimal size query
    int lwork = -1;                             // Optimal workspace size query
    std::vector<T> rwork(3 * n - 2);            // Real workspace array
    int info;

    if constexpr (std::is_same<T, float>::value) {
        // Query optimal workspace size
        cheev_(&jobz, &uplo, &n, ah.data(), &lda, eigenVals.data(),
               work_query.data(), &lwork, rwork.data(), &info);
        // Allocate workspace
        lwork = static_cast<int>(work_query[0].real());
        std::vector<std::complex<T>> work_optimal(lwork, {0, 0});
        // Perform eigenvalue and eigenvector computation
        cheev_(&jobz, &uplo, &n, ah.data(), &lda, eigenVals.data(),
               work_optimal.data(), &lwork, rwork.data(), &info);
    } else {
        // Query optimal workspace size
        zheev_(&jobz, &uplo, &n, ah.data(), &lda, eigenVals.data(),
               work_query.data(), &lwork, rwork.data(), &info);
        // Allocate workspace
        lwork = static_cast<int>(work_query[0].real());
        std::vector<std::complex<T>> work_optimal(lwork, {0, 0});
        // Perform eigenvalue and eigenvector computation
        zheev_(&jobz, &uplo, &n, ah.data(), &lda, eigenVals.data(),
               work_optimal.data(), &lwork, rwork.data(), &info);
    }

    std::transform(ah.begin(), ah.end(), unitary.begin(),
                   [](std::complex<T> value) {
                       return std::complex<T>{value.real(), -value.imag()};
                   });
}
} // namespace Pennylane::Util
