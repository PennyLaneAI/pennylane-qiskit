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
/**
 * @file
 * Contains linear algebra utility functions.
 */
#pragma once

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <numeric>
#include <random>
#include <span>
#include <vector>

#include "Macros.hpp"
#include "TypeTraits.hpp" // remove_complex_t
#include "Util.hpp"       // ConstSum, ConstMult, ConstMultConj

/// @cond DEV

#if __has_include(<cblas.h>) && defined _ENABLE_BLAS
#include <cblas.h>
constexpr bool USE_CBLAS = true;
#else
constexpr bool USE_CBLAS = false;
#ifndef CBLAS_TRANSPOSE
using CBLAS_TRANSPOSE = enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
};
#endif

#ifndef CBLAS_LAYOUT
using CBLAS_LAYOUT = enum CBLAS_LAYOUT {
    CblasRowMajor = 101,
    CblasColMajor = 102
};
#endif
#endif

namespace {
using namespace Pennylane::Util;
} // namespace

/// @endcond

namespace Pennylane::LightningQubit::Util {
/**
 * @brief Transpose enum class
 */
enum class Trans : int {
    NoTranspose = CblasNoTrans,
    Transpose = CblasTrans,
    Adjoint = CblasConjTrans
};
/**
 * @brief Calculates the inner-product using OpenMP.
 *
 * @tparam T Floating point precision type.
 * @tparam NTERMS Number of terms proceeds by each thread
 * @param v1 Complex data array 1.
 * @param v2 Complex data array 2.
 * @param result Calculated inner-product of v1 and v2.
 * @param data_size Size of data arrays.
 */
template <class T,
          size_t NTERMS = (1U << 19U)> // NOLINT(readability-magic-numbers)
inline static void
omp_innerProd(const std::complex<T> *v1, const std::complex<T> *v2,
              std::complex<T> &result, const size_t data_size) {
#if defined(_OPENMP)
#pragma omp declare reduction(sm : std::complex<T> : omp_out =                 \
                                  ConstSum(omp_out, omp_in))                   \
    initializer(omp_priv = std::complex<T>{0, 0})

    size_t nthreads = data_size / NTERMS;
    if (nthreads < 1) {
        nthreads = 1;
    }

#pragma omp parallel for num_threads(nthreads) default(none)                   \
    shared(v1, v2, data_size) reduction(sm : result)
#endif
    for (size_t i = 0; i < data_size; i++) {
        result = ConstSum(result, ConstMult(*(v1 + i), *(v2 + i)));
    }
}

/**
 * @brief Calculates the inner-product using the best available method.
 *
 * @tparam T Floating point precision type.
 * @tparam STD_CROSSOVER Threshold for using OpenMP method
 * @param v1 Complex data array 1.
 * @param v2 Complex data array 2.
 * @param data_size Size of data arrays.
 * @return std::complex<T> Result of inner product operation.
 */
template <class T,
          size_t STD_CROSSOVER = (1U
                                  << 20U)> // NOLINT(readability-magic-numbers)
inline auto innerProd(const std::complex<T> *v1, const std::complex<T> *v2,
                      const size_t data_size) -> std::complex<T> {
    std::complex<T> result(0, 0);

    if constexpr (USE_CBLAS) {
        if constexpr (std::is_same_v<T, float>) {
            cblas_cdotu_sub(data_size, v1, 1, v2, 1, &result);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_zdotu_sub(data_size, v1, 1, v2, 1, &result);
        }
    } else {
        if (data_size < STD_CROSSOVER) {
            result = std::inner_product(
                v1, v1 + data_size, v2, std::complex<T>(), ConstSum<T>,
                static_cast<std::complex<T> (*)(
                    std::complex<T>, std::complex<T>)>(&ConstMult<T>));
        } else {
            omp_innerProd(v1, v2, result, data_size);
        }
    }
    return result;
}

/**
 * @brief Calculates the inner-product using OpenMP.
 * with the first dataset conjugated.
 *
 * @tparam T Floating point precision type.
 * @tparam NTERMS Number of terms proceeds by each thread
 * @param v1 Complex data array 1.
 * @param v2 Complex data array 2.
 * @param result Calculated inner-product of v1 and v2.
 * @param data_size Size of data arrays.
 */
template <class T,
          size_t NTERMS = (1U << 19U)> // NOLINT(readability-magic-numbers)
inline static void
omp_innerProdC(const std::complex<T> *v1, const std::complex<T> *v2,
               std::complex<T> &result, const size_t data_size) {
#if defined(_OPENMP)
#pragma omp declare reduction(sm : std::complex<T> : omp_out =                 \
                                  ConstSum(omp_out, omp_in))                   \
    initializer(omp_priv = std::complex<T>{0, 0})
#endif

#if defined(_OPENMP)
    size_t nthreads = data_size / NTERMS;
    if (nthreads < 1) {
        nthreads = 1;
    }
#endif

#if defined(_OPENMP)
#pragma omp parallel for num_threads(nthreads) default(none)                   \
    shared(v1, v2, data_size) reduction(sm : result)
#endif
    for (size_t i = 0; i < data_size; i++) {
        result = ConstSum(result, ConstMultConj(*(v1 + i), *(v2 + i)));
    }
}

/**
 * @brief Calculates the inner-product using the best available method
 * with the first dataset conjugated.
 *
 * @tparam T Floating point precision type.
 * @tparam STD_CROSSOVER Threshold for using OpenMP method
 * @param v1 Complex data array 1; conjugated before application.
 * @param v2 Complex data array 2.
 * @param data_size Size of data arrays.
 * @return std::complex<T> Result of inner product operation.
 */
template <class T,
          size_t STD_CROSSOVER = (1U
                                  << 20U)> // NOLINT(readability-magic-numbers)
inline auto innerProdC(const std::complex<T> *v1, const std::complex<T> *v2,
                       const size_t data_size) -> std::complex<T> {
    std::complex<T> result(0, 0);

    if constexpr (USE_CBLAS) {
        if constexpr (std::is_same_v<T, float>) {
            cblas_cdotc_sub(data_size, v1, 1, v2, 1, &result);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_zdotc_sub(data_size, v1, 1, v2, 1, &result);
        }
    } else {
        if (data_size < STD_CROSSOVER) {
            result =
                std::inner_product(v1, v1 + data_size, v2, std::complex<T>(),
                                   ConstSum<T>, ConstMultConj<T>);
        } else {
            omp_innerProdC(v1, v2, result, data_size);
        }
    }
    return result;
}

/**
 * @brief Calculates the inner-product using the best available method.
 *
 * @see innerProd(const std::complex<T> *v1, const std::complex<T> *v2,
 * const size_t data_size)
 */
template <class T, class AllocA, class AllocB>
inline auto innerProd(const std::vector<std::complex<T>, AllocA> &v1,
                      const std::vector<std::complex<T>, AllocB> &v2)
    -> std::complex<T> {
    return innerProd(v1.data(), v2.data(), v1.size());
}

/**
 * @brief Calculates the inner-product using the best available method with the
 * first dataset conjugated.
 *
 * @see innerProdC(const std::complex<T> *v1, const std::complex<T> *v2,
 * const size_t data_size)
 */
template <class T, class AllocA, class AllocB>
inline auto innerProdC(const std::vector<std::complex<T>, AllocA> &v1,
                       const std::vector<std::complex<T>, AllocB> &v2)
    -> std::complex<T> {
    return innerProdC(v1.data(), v2.data(), v1.size());
}

/**
 * @brief Calculates the matrix-vector product using OpenMP.
 *
 * @tparam T Floating point precision type.
 * @param mat Complex data array repr. a flatten (row-wise) matrix m * n.
 * @param v_in Complex data array repr. a vector of shape n * 1.
 * @param v_out Pre-allocated complex data array to store the result.
 * @param m Number of rows of `mat`.
 * @param n Number of columns of `mat`.
 * @param transpose Whether use a transposed version of `m_right`.
 * row-wise.
 */
template <class T>
inline static void
omp_matrixVecProd(const std::complex<T> *mat, const std::complex<T> *v_in,
                  std::complex<T> *v_out, size_t m, size_t n, Trans transpose) {
    if (!v_out) {
        return;
    }

    size_t row;
    size_t col;

    {
        switch (transpose) {
        case Trans::NoTranspose:
#if defined(_OPENMP)
#pragma omp parallel for default(none) private(row, col) firstprivate(m, n)    \
    shared(v_out, mat, v_in)
#endif
            for (row = 0; row < m; row++) {
                for (col = 0; col < n; col++) {
                    v_out[row] += mat[row * n + col] * v_in[col];
                }
            }
            break;
        case Trans::Transpose:
#if defined(_OPENMP)
#pragma omp parallel for default(none) private(row, col) firstprivate(m, n)    \
    shared(v_out, mat, v_in)
#endif
            for (row = 0; row < m; row++) {
                for (col = 0; col < n; col++) {
                    v_out[row] += mat[col * m + row] * v_in[col];
                }
            }
            break;
        case Trans::Adjoint:
#if defined(_OPENMP)
#pragma omp parallel for default(none) private(row, col) firstprivate(m, n)    \
    shared(v_out, mat, v_in)
#endif
            for (row = 0; row < m; row++) {
                for (col = 0; col < n; col++) {
                    v_out[row] += std::conj(mat[col * m + row]) * v_in[col];
                }
            }
            break;
        }
    }
}

/**
 * @brief Calculates the matrix-vector product using the best available method.
 *
 * @tparam T Floating point precision type.
 * @param mat Complex data array repr. a flatten (row-wise) matrix m * n.
 * @param v_in Complex data array repr. a vector of shape n * 1.
 * @param v_out Pre-allocated complex data array to store the result.
 * @param m Number of rows of `mat`.
 * @param n Number of columns of `mat`.
 * @param transpose Whether use a transposed version of `m_right`.
 */
template <class T>
inline void matrixVecProd(const std::complex<T> *mat,
                          const std::complex<T> *v_in, std::complex<T> *v_out,
                          size_t m, size_t n,
                          Trans transpose = Trans::NoTranspose) {
    if (!v_out) {
        return;
    }

    if constexpr (USE_CBLAS) {
        constexpr std::complex<T> co{1, 0};
        constexpr std::complex<T> cz{0, 0};
        const auto tr = static_cast<CBLAS_TRANSPOSE>(transpose);
        if constexpr (std::is_same_v<T, float>) {
            cblas_cgemv(CblasRowMajor, tr, m, n, &co, mat, m, v_in, 1, &cz,
                        v_out, 1);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_zgemv(CblasRowMajor, tr, m, n, &co, mat, m, v_in, 1, &cz,
                        v_out, 1);
        }
    } else {
        omp_matrixVecProd(mat, v_in, v_out, m, n, transpose);
    }
}

/**
 * @brief Calculates the matrix-vector product using the best available method.
 *
 * @see void matrixVecProd(const std::complex<T> *mat, const
 * std::complex<T> *v_in, std::complex<T> *v_out, size_t m, size_t n, size_t
 * nthreads = 1, bool transpose = false)
 */
template <class T>
inline auto matrixVecProd(const std::vector<std::complex<T>> &mat,
                          const std::vector<std::complex<T>> &v_in, size_t m,
                          size_t n, Trans transpose = Trans::NoTranspose)
    -> std::vector<std::complex<T>> {
    if (mat.size() != m * n) {
        throw std::invalid_argument(
            "Invalid number of rows and columns for the input matrix");
    }
    if (v_in.size() != n) {
        throw std::invalid_argument("Invalid size for the input vector");
    }

    std::vector<std::complex<T>> v_out(m);
    matrixVecProd(mat.data(), v_in.data(), v_out.data(), m, n, transpose);
    return v_out;
}

/**
 * @brief Calculates transpose of a matrix recursively and Cache-Friendly
 * using blocking and Cache-optimized techniques.
 *
 * @tparam T Floating point precision type.
 * @tparam BLOCKSIZE Size of submatrices in the blocking technique.
 * @param mat Data array repr. a flatten (row-wise) matrix m * n.
 * @param mat_t Pre-allocated data array to store the transpose of `mat`.
 * @param m Number of rows of `mat`.
 * @param n Number of columns of `mat`.
 * @param m1 Index of the first row.
 * @param m2 Index of the last row.
 * @param n1 Index of the first column.
 * @param n2 Index of the last column.
 */
template <class T, size_t BLOCKSIZE = 16> // NOLINT(readability-magic-numbers)
inline static void CFTranspose(const T *mat, T *mat_t, size_t m, size_t n,
                               size_t m1, size_t m2, size_t n1, size_t n2) {
    size_t r;
    size_t s;

    size_t r1;
    size_t s1;
    size_t r2;
    size_t s2;

    r1 = m2 - m1;
    s1 = n2 - n1;

    if (r1 >= s1 && r1 > BLOCKSIZE) {
        r2 = (m1 + m2) / 2;
        CFTranspose(mat, mat_t, m, n, m1, r2, n1, n2);
        m1 = r2;
        CFTranspose(mat, mat_t, m, n, m1, m2, n1, n2);
    } else if (s1 > BLOCKSIZE) {
        s2 = (n1 + n2) / 2;
        CFTranspose(mat, mat_t, m, n, m1, m2, n1, s2);
        n1 = s2;
        CFTranspose(mat, mat_t, m, n, m1, m2, n1, n2);
    } else {
        for (r = m1; r < m2; r++) {
            for (s = n1; s < n2; s++) {
                mat_t[s * m + r] = mat[r * n + s];
            }
        }
    }
}

/**
 * @brief Calculates transpose of a matrix recursively and Cache-Friendly
 * using blocking and Cache-optimized techniques.
 *
 * @tparam T Floating point precision type.
 * @tparam BLOCKSIZE Size of submatrices in the blocking techinque.
 * @param mat Data array repr. a flatten (row-wise) matrix m * n.
 * @param mat_t Pre-allocated data array to store the transpose of `mat`.
 * @param m Number of rows of `mat`.
 * @param n Number of columns of `mat`.
 * @param m1 Index of the first row.
 * @param m2 Index of the last row.
 * @param n1 Index of the first column.
 * @param n2 Index of the last column.
 */
template <class T, size_t BLOCKSIZE = 16> // NOLINT(readability-magic-numbers)
inline static void CFTranspose(const std::complex<T> *mat,
                               std::complex<T> *mat_t, size_t m, size_t n,
                               size_t m1, size_t m2, size_t n1, size_t n2) {
    size_t r;
    size_t s;

    size_t r1;
    size_t s1;
    size_t r2;
    size_t s2;

    r1 = m2 - m1;
    s1 = n2 - n1;

    if (r1 >= s1 && r1 > BLOCKSIZE) {
        r2 = (m1 + m2) / 2;
        CFTranspose(mat, mat_t, m, n, m1, r2, n1, n2);
        m1 = r2;
        CFTranspose(mat, mat_t, m, n, m1, m2, n1, n2);
    } else if (s1 > BLOCKSIZE) {
        s2 = (n1 + n2) / 2;
        CFTranspose(mat, mat_t, m, n, m1, m2, n1, s2);
        n1 = s2;
        CFTranspose(mat, mat_t, m, n, m1, m2, n1, n2);
    } else {
        for (r = m1; r < m2; r++) {
            for (s = n1; s < n2; s++) {
                mat_t[s * m + r] = mat[r * n + s];
            }
        }
    }
}

/**
 * @brief Transpose a matrix of shape m * n to n * m using the
 * best available method.
 *
 * @tparam T Floating point precision type.
 * @param mat Row-wise flatten matrix of shape m * n.
 * @param m Number of rows of `mat`.
 * @param n Number of columns of `mat`.
 * @return mat transpose of shape n * m.
 */
template <class T, class Allocator = std::allocator<T>>
inline auto Transpose(std::span<const T> mat, size_t m, size_t n,
                      Allocator allocator = std::allocator<T>())
    -> std::vector<T, Allocator> {
    if (mat.size() != m * n) {
        throw std::invalid_argument(
            "Invalid number of rows and columns for the input matrix");
    }

    std::vector<T, Allocator> mat_t(n * m, allocator);
    CFTranspose(mat.data(), mat_t.data(), m, n, 0, m, 0, n);
    return mat_t;
}

/**
 * @brief Transpose a matrix of shape m * n to n * m using the
 * best available method.
 *
 * This version may be merged with the above one when std::ranges is well
 * supported.
 *
 * @tparam T Floating point precision type.
 * @param mat Row-wise flatten matrix of shape m * n.
 * @param m Number of rows of `mat`.
 * @param n Number of columns of `mat`.
 * @return mat transpose of shape n * m.
 */
template <class T, class Allocator>
inline auto Transpose(const std::vector<T, Allocator> &mat, size_t m, size_t n)
    -> std::vector<T, Allocator> {
    return Transpose(std::span<const T>{mat}, m, n, mat.get_allocator());
}

/**
 * @brief Calculates vector-matrix product.
 *
 * @tparam T Floating point precision type.
 * @param v_in Data array repr. a vector of shape m * 1.
 * @param mat Data array repr. a flatten (row-wise) matrix m * n.
 * @param v_out Pre-allocated data array to store the result that is
 *              `mat_t \times v_in` where `mat_t` is transposed of `mat`.
 * @param m Number of rows of `mat`.
 * @param n Number of columns of `mat`.
 */
template <class T>
inline void vecMatrixProd(const T *v_in, const T *mat, T *v_out, size_t m,
                          size_t n) {
    if (!v_out) {
        return;
    }

    size_t i;
    size_t j;

    constexpr T z = static_cast<T>(0.0);
    bool allzero = true;
    for (j = 0; j < m; j++) {
        if (v_in[j] != z) {
            allzero = false;
            break;
        }
    }
    if (allzero) {
        return;
    }

    std::vector<T> mat_t(m * n);
    CFTranspose(mat, mat_t.data(), m, n, 0, m, 0, n);

    for (i = 0; i < n; i++) {
        T t = z;
        for (j = 0; j < m; j++) {
            t += mat_t[i * m + j] * v_in[j];
        }
        v_out[i] = t;
    }
}

/**
 * @brief Calculates the vector-matrix product using the best available method.
 *
 * @see inline void vecMatrixProd(const T *v_in,
 * const T *mat, T *v_out, size_t m, size_t n)
 */
template <class T, class AllocA, class AllocB>
inline auto vecMatrixProd(const std::vector<T, AllocA> &v_in,
                          const std::vector<T, AllocB> &mat, size_t m, size_t n)
    -> std::vector<T> {
    if (v_in.size() != m) {
        throw std::invalid_argument("Invalid size for the input vector");
    }
    if (mat.size() != m * n) {
        throw std::invalid_argument(
            "Invalid number of rows and columns for the input matrix");
    }

    std::vector<T, AllocA> v_out(n);
    vecMatrixProd(v_in.data(), mat.data(), v_out.data(), m, n);

    return v_out;
}

/**
 * @brief Calculates the vector-matrix product using the best available method.
 *
 * @see inline void vecMatrixProd(const T *v_in, const T *mat, T *v_out, size_t
 * m, size_t n)
 */
template <class T, class AllocA, class AllocB, class AllocC>
inline void
vecMatrixProd(std::vector<T, AllocA> &v_out, const std::vector<T, AllocB> &v_in,
              const std::vector<T, AllocC> &mat, size_t m, size_t n) {
    if (mat.size() != m * n) {
        throw std::invalid_argument(
            "Invalid number of rows and columns for the input matrix");
    }
    if (v_in.size() != m) {
        throw std::invalid_argument("Invalid size for the input vector");
    }
    if (v_out.size() != n) {
        throw std::invalid_argument("Invalid preallocated size for the result");
    }

    vecMatrixProd(v_in.data(), mat.data(), v_out.data(), m, n);
}

/**
 * @brief Calculates matrix-matrix product using OpenMP.
 *
 * @tparam T Floating point precision type.
 * @tparam STRIDE Size of stride in the cache-blocking technique
 * @param m_left Row-wise flatten matrix of shape m * k.
 * @param m_right Row-wise flatten matrix of shape k * n.
 * @param m_out Pre-allocated row-wise flatten matrix of shape m * n.
 * @param m Number of rows of `m_left`.
 * @param n Number of columns of `m_right`.
 * @param k Number of rows of `m_right`.
 * @param transpose Whether use a transposed version of `m_right`.
 *
 * @note Consider transpose=true, to get a better performance.
 *  To transpose a matrix efficiently, check Util::Transpose
 */

template <class T, size_t STRIDE = 2>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
inline void omp_matrixMatProd(const std::complex<T> *m_left,
                              const std::complex<T> *m_right,
                              std::complex<T> *m_out, size_t m, size_t n,
                              size_t k, Trans transpose) {
    if (!m_out) {
        return;
    }

    switch (transpose) {
    case Trans::Transpose:
#if defined(_OPENMP)
#pragma omp parallel for default(none) shared(m_left, m_right, m_out)          \
    firstprivate(m, n, k)
#endif
        for (size_t row = 0; row < m; row++) {
            for (size_t col = 0; col < n; col++) {
                for (size_t blk = 0; blk < k; blk++) {
                    m_out[row * n + col] +=
                        m_left[row * k + blk] * m_right[col * n + blk];
                }
            }
        }
        break;
    case Trans::Adjoint:
#if defined(_OPENMP)
#pragma omp parallel for default(none) shared(m_left, m_right, m_out)          \
    firstprivate(m, n, k)
#endif
        for (size_t row = 0; row < m; row++) {
            for (size_t col = 0; col < n; col++) {
                for (size_t blk = 0; blk < k; blk++) {
                    m_out[row * n + col] += m_left[row * k + blk] *
                                            std::conj(m_right[col * n + blk]);
                }
            }
        }
        break;
    case Trans::NoTranspose:
#if defined(_OPENMP)
#pragma omp parallel for default(none) shared(m_left, m_right, m_out)          \
    firstprivate(m, n, k)
#endif
        for (size_t row = 0; row < m; row += STRIDE) {
            size_t i;
            size_t j;
            size_t l;
            std::complex<T> t;
            for (size_t col = 0; col < n; col += STRIDE) {
                for (size_t blk = 0; blk < k; blk += STRIDE) {
                    // cache-blocking:
                    for (i = row; i < std::min(row + STRIDE, m); i++) {
                        for (j = col; j < std::min(col + STRIDE, n); j++) {
                            t = 0;
                            for (l = blk; l < std::min(blk + STRIDE, k); l++) {
                                t += m_left[i * k + l] * m_right[l * n + j];
                            }
                            m_out[i * n + j] += t;
                        }
                    }
                }
            }
        }
        break;
    }
}

/**
 * @brief Calculates matrix-matrix product using the best available method.
 *
 * @tparam T Floating point precision type.
 * @param m_left Row-wise flatten matrix of shape m * k.
 * @param m_right Row-wise flatten matrix of shape k * n.
 * @param m_out Pre-allocated row-wise flatten matrix of shape m * n.
 * @param m Number of rows of `m_left`.
 * @param n Number of columns of `m_right`.
 * @param k Number of rows of `m_right`.
 * @param transpose Whether use a transposed version of `m_right`.
 *
 * @note Consider transpose=true, to get a better performance.
 *  To transpose a matrix efficiently, check Util::Transpose
 */
template <class T>
inline void matrixMatProd(const std::complex<T> *m_left,
                          const std::complex<T> *m_right,
                          std::complex<T> *m_out, size_t m, size_t n, size_t k,
                          Trans transpose = Trans::NoTranspose) {
    if (!m_out) {
        return;
    }
    if constexpr (USE_CBLAS) {
        constexpr std::complex<T> co{1, 0};
        constexpr std::complex<T> cz{0, 0};
        const auto tr = static_cast<CBLAS_TRANSPOSE>(transpose);
        if constexpr (std::is_same_v<T, float>) {
            cblas_cgemm(CblasRowMajor, CblasNoTrans, tr, m, n, k, &co, m_left,
                        k, m_right, (transpose != Trans::NoTranspose) ? k : n,
                        &cz, m_out, n);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_zgemm(CblasRowMajor, CblasNoTrans, tr, m, n, k, &co, m_left,
                        k, m_right, (transpose != Trans::NoTranspose) ? k : n,
                        &cz, m_out, n);
        } else {
            static_assert(
                std::is_same_v<T, float> || std::is_same_v<T, double>,
                "This procedure only supports a single or double precision "
                "floating point types.");
        }
    } else {
        omp_matrixMatProd(m_left, m_right, m_out, m, n, k, transpose);
    }
}

/**
 * @brief Calculates the matrix-matrix product using the best available method.
 *
 * @see void matrixMatProd(const std::complex<T> *m_left, const std::complex<T>
 * *m_right, std::complex<T> *m_out, size_t m, size_t n, size_t k, size_t
 * nthreads = 1, bool transpose = false)
 *
 * @note consider transpose=true, to get a better performance.
 *  To transpose a matrix efficiently, check Util::Transpose
 */
template <class T>
inline auto matrixMatProd(const std::vector<std::complex<T>> m_left,
                          const std::vector<std::complex<T>> m_right, size_t m,
                          size_t n, size_t k,
                          Trans transpose = Trans::NoTranspose)
    -> std::vector<std::complex<T>> {
    if (m_left.size() != m * k) {
        throw std::invalid_argument(
            "Invalid number of rows and columns for the input left matrix");
    }
    if (m_right.size() != k * n) {
        throw std::invalid_argument(
            "Invalid number of rows and columns for the input right matrix");
    }

    std::vector<std::complex<T>> m_out(m * n);
    matrixMatProd(m_left.data(), m_right.data(), m_out.data(), m, n, k,
                  transpose);

    return m_out;
}

/**
 * @brief @rst
 * Calculate :math:`y += a*x` for a scalar :math:`a` and a vector :math:`x`
 * using OpenMP
 * @endrst
 *
 * @tparam STD_CROSSOVER The number of dimension after which OpenMP version
 * outperforms the standard method.
 *
 * @param dim Dimension of data
 * @param a Scalar to scale x
 * @param x Vector to add
 * @param y Vector to be added
 */
template <class T,
          size_t STD_CROSSOVER = 1U << 12U> // NOLINT(readability-magic-numbers)
void omp_scaleAndAdd(size_t dim, std::complex<T> a, const std::complex<T> *x,
                     std::complex<T> *y) {
    if (dim < STD_CROSSOVER) {
        for (size_t i = 0; i < dim; i++) {
            y[i] += a * x[i];
        }
    } else {
#if defined(_OPENMP)
#pragma omp parallel for default(none) firstprivate(a, dim, x, y)
#endif
        for (size_t i = 0; i < dim; i++) {
            y[i] += a * x[i];
        }
    }
}

/**
 * @brief @rst
 * Calculate :math:`y += a*x` for a scalar :math:`a` and a vector :math:`x`
 * using BLAS.
 * @endrst
 *
 * @param dim Dimension of data
 * @param a Scalar to scale x
 * @param x Vector to add
 * @param y Vector to be added
 */
template <class T>
void blas_scaleAndAdd(size_t dim, std::complex<T> a, const std::complex<T> *x,
                      std::complex<T> *y) {
    if constexpr (std::is_same_v<T, float>) {
        cblas_caxpy(dim, &a, x, 1, y, 1);
    } else if (std::is_same_v<T, double>) {
        cblas_zaxpy(dim, &a, x, 1, y, 1);
    } else {
        static_assert(
            std::is_same_v<T, float> || std::is_same_v<T, double>,
            "This procedure only supports a single or double precision "
            "floating point types.");
    }
}

/**
 * @brief @rst
 * Calculate :math:`y += a*x` for a scalar :math:`a` and a vector :math:`x`
 * using the best available method.
 * @endrst
 *
 *
 * @param dim Dimension of data
 * @param a Scalar to scale x
 * @param x Vector to add
 * @param y Vector to be added
 */
template <class T>
void scaleAndAdd(size_t dim, std::complex<T> a, const std::complex<T> *x,
                 std::complex<T> *y) {
    if constexpr (USE_CBLAS) {
        blas_scaleAndAdd(dim, a, x, y);
    } else {
        omp_scaleAndAdd(dim, a, x, y);
    }
}
/**
 * @brief @rst
 * Calculate :math:`y += a*x` for a scalar :math:`a` and a vector :math:`x`.
 * @endrst
 *
 * @param dim Dimension of data
 * @param a Scalar to scale x
 * @param x Vector to add
 * @param y Vector to be added
 */
template <class T>
void scaleAndAdd(std::complex<T> a, const std::vector<std::complex<T>> &x,
                 std::vector<std::complex<T>> &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Dimensions of vectors mismatch");
    }
    scaleAndAdd(x.size(), a, x.data(), y.data());
}
} // namespace Pennylane::LightningQubit::Util
