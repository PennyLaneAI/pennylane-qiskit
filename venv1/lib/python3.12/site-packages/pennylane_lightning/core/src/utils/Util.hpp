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
 * Defines utility functions for Bitwise operations.
 */
#pragma once

#include <cmath>
#include <complex>
#include <numbers>
#include <numeric> // transform_reduce
#include <set>
#include <type_traits> // is_same_v
#include <vector>

#include "Error.hpp"
#include "TypeTraits.hpp" // remove_complex_t

namespace Pennylane::Util {
/**
 * @brief Compile-time scalar real times complex number.
 *
 * @tparam U Precision of real value `a`.
 * @tparam T Precision of complex value `b` and result.
 * @param a Real scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr auto ConstMult(U a, std::complex<T> b)
    -> std::complex<T> {
    return {a * b.real(), a * b.imag()};
}

/**
 * @brief Compile-time scalar complex times complex.
 *
 * @tparam U Precision of complex value `a`.
 * @tparam T Precision of complex value `b` and result.
 * @param a Complex scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr auto ConstMult(std::complex<U> a, std::complex<T> b)
    -> std::complex<T> {
    return {a.real() * b.real() - a.imag() * b.imag(),
            a.real() * b.imag() + a.imag() * b.real()};
}
template <class T, class U = T>
inline static constexpr auto ConstMultConj(std::complex<U> a, std::complex<T> b)
    -> std::complex<T> {
    return {a.real() * b.real() + a.imag() * b.imag(),
            -a.imag() * b.real() + a.real() * b.imag()};
}

/**
 * @brief Compile-time scalar complex summation.
 *
 * @tparam T Precision of complex value `a` and result.
 * @tparam U Precision of complex value `b`.
 * @param a Complex scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr auto ConstSum(std::complex<U> a, std::complex<T> b)
    -> std::complex<T> {
    return a + b;
}

/**
 * @brief Return complex value 0.5+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0.5,0}
 */
template <class T> inline static constexpr auto HALF() -> std::complex<T> {
    return {0.5, 0}; // NOLINT(cppcoreguidelines-avoid-magic-numbers)
}

/**
 * @brief Return complex value -1+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{-1,0}
 */
template <class T> inline static constexpr auto NEGONE() -> std::complex<T> {
    return {-1, 0};
}

/**
 * @brief Return complex value 1+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{1,0}
 */
template <class T> inline static constexpr auto ONE() -> std::complex<T> {
    return {1, 0};
}

/**
 * @brief Return complex value 1+0i in the given precision.
 *
 * @tparam ComplexT Complex type.
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{1,0}
 */
template <template <class> class ComplexT, class T>
inline static constexpr auto ONE() -> ComplexT<T> {
    return {1, 0};
}

/**
 * @brief Return complex value 0+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,0}
 */
template <class T> inline static constexpr auto ZERO() -> std::complex<T> {
    return {0, 0};
}

/**
 * @brief Return complex value 0+0i in the given precision.
 *
 * @tparam ComplexT Complex type.
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,0}
 */
template <template <class> class ComplexT, class T>
inline static constexpr auto ZERO() -> ComplexT<T> {
    return {0, 0};
}

/**
 * @brief Return complex value 0+1i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,1}
 */
template <class T> inline static constexpr auto IMAG() -> std::complex<T> {
    return {0, 1};
}

/**
 * @brief Return complex value 0+1i in the given precision.
 *
 * @tparam ComplexT Complex type.
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,1}
 */
template <template <class> class ComplexT, class T>
inline static constexpr auto IMAG() -> ComplexT<T> {
    return {0, 1};
}

/**
 * @brief Returns sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T sqrt(2)
 */
template <class T> inline static constexpr auto SQRT2() -> T {
#if __cpp_lib_math_constants >= 201907L
    return std::numbers::sqrt2_v<T>;
#else
    if constexpr (std::is_same_v<T, float>) {
        return 0x1.6a09e6p+0F; // NOLINT: To be replaced in C++20
    } else {
        return 0x1.6a09e667f3bcdp+0; // NOLINT: To be replaced in C++20
    }
#endif
}

/**
 * @brief Returns sqrt(2) as a compile-time constant.
 *
 * @tparam ComplexT Complex type.
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T sqrt(2)
 */
template <template <class> class ComplexT, class T>
inline static constexpr auto SQRT2() -> ComplexT<T> {
#if __cpp_lib_math_constants >= 201907L
    return std::numbers::sqrt2_v<T>;
#else
    if constexpr (std::is_same_v<T, float>) {
        return 0x1.6a09e6p+0F; // NOLINT: To be replaced in C++20
    } else {
        return 0x1.6a09e667f3bcdp+0; // NOLINT: To be replaced in C++20
    }
#endif
}

/**
 * @brief Returns inverse sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T 1/sqrt(2)
 */
template <class T> inline static constexpr auto INVSQRT2() -> T {
    return {1 / SQRT2<T>()};
}

/**
 * @brief Returns inverse sqrt(2) as a compile-time constant.
 *
 * @tparam ComplexT Complex type.
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T 1/sqrt(2)
 */
template <template <class> class ComplexT, class T>
inline static constexpr auto INVSQRT2() -> ComplexT<T> {
    return static_cast<ComplexT<T>>(INVSQRT2<T>());
}

/**
 * @brief Calculates 2^n for some integer n > 0 using bitshifts.
 *
 * @param n the exponent
 * @return value of 2^n
 */
inline auto exp2(const size_t &n) -> size_t {
    return static_cast<size_t>(1) << n;
}

/**
 * @brief Log2 calculation.
 *
 * @param value Value to calculate for.
 * @return size_t
 */
inline auto log2(size_t value) -> size_t {
    return static_cast<size_t>(std::log2(value));
}

/**
 * @brief Calculates the decimal value for a qubit, assuming a big-endian
 * convention.
 *
 * @param qubitIndex the index of the qubit in the range [0, qubits)
 * @param qubits the number of qubits in the circuit
 * @return decimal value for the qubit at specified index
 */
inline auto maxDecimalForQubit(size_t qubitIndex, size_t qubits) -> size_t {
    PL_ASSERT(qubitIndex < qubits);
    return exp2(qubits - qubitIndex - 1);
}

/**
 * @brief Define a hash function for std::pair
 */
struct PairHash {
    /**
     * @brief A hash function for std::pair
     *
     * @tparam T The type of the first element of the pair
     * @tparam U The type of the first element of the pair
     * @param p A pair to compute hash
     */
    template <typename T, typename U>
    size_t operator()(const std::pair<T, U> &p) const {
        return std::hash<T>()(p.first) ^ std::hash<U>()(p.second);
    }
};

/**
 * @brief Iterate over all enum values (if BEGIN and END are defined).
 *
 * @tparam T enum type
 * @tparam Func function to execute
 */
template <class T, class Func> void for_each_enum(Func &&func) {
    for (auto e = T::BEGIN; e != T::END;
         e = static_cast<T>(std::underlying_type_t<T>(e) + 1)) {
        func(e);
    }
}
template <class T, class U, class Func> void for_each_enum(Func &&func) {
    for (auto e1 = T::BEGIN; e1 != T::END;
         e1 = static_cast<T>(std::underlying_type_t<T>(e1) + 1)) {
        for (auto e2 = U::BEGIN; e2 != U::END;
             e2 = static_cast<U>(std::underlying_type_t<U>(e2) + 1)) {
            func(e1, e2);
        }
    }
}

/**
 * @brief Streaming operator for vector data.
 *
 * @tparam T Vector data type.
 * @param os Output stream.
 * @param vec Vector data.
 * @return std::ostream&
 */
template <class T>
inline auto operator<<(std::ostream &os, const std::vector<T> &vec)
    -> std::ostream & {
    os << '[';
    if (!vec.empty()) {
        for (size_t i = 0; i < vec.size() - 1; i++) {
            os << vec[i] << ", ";
        }
        os << vec.back();
    }
    os << ']';
    return os;
}

/**
 * @brief Streaming operator for set data.
 *
 * @tparam T Vector data type.
 * @param os Output stream.
 * @param s Set data.
 * @return std::ostream&
 */
template <class T>
inline auto operator<<(std::ostream &os, const std::set<T> &s)
    -> std::ostream & {
    os << '{';
    for (const auto &e : s) {
        os << e << ",";
    }
    os << '}';
    return os;
}

/**
 * @brief @rst
 * Compute the squared norm of a real/complex vector :math:`\sum_k |v_k|^2`
 * @endrst
 *
 * @param data Data pointer
 * @param data_size Size of the data
 */
template <class T>
auto squaredNorm(const T *data, size_t data_size) -> remove_complex_t<T> {
    if constexpr (is_complex_v<T>) {
        // complex type
        using PrecisionT = remove_complex_t<T>;
        return std::transform_reduce(
            data, data + data_size, PrecisionT{}, std::plus<PrecisionT>(),
            static_cast<PrecisionT (*)(const std::complex<PrecisionT> &)>(
                &std::norm<PrecisionT>));
    } else {
        using PrecisionT = T;
        return std::transform_reduce(
            data, data + data_size, PrecisionT{}, std::plus<PrecisionT>(),
            static_cast<PrecisionT (*)(PrecisionT)>(std::norm));
    }
}

/**
 * @brief @rst
 * Compute the squared norm of a real/complex vector :math:`\sum_k |v_k|^2`
 * @endrst
 *
 * @param vec std::vector containing data
 */
template <class T, class Alloc>
auto squaredNorm(const std::vector<T, Alloc> &vec) -> remove_complex_t<T> {
    return squaredNorm(vec.data(), vec.size());
}

/**
 * @brief Determines the indices that would sort an array.
 *
 * @tparam T Vector data type.
 * @param arr Array to be inspected.
 * @param length Size of the array
 * @return a vector with indices that would sort the array.
 */
template <typename T>
inline auto sorting_indices(const T *arr, size_t length)
    -> std::vector<size_t> {
    std::vector<size_t> indices(length);
    iota(indices.begin(), indices.end(), 0);

    // indices will be sorted in accordance to the array provided.
    sort(indices.begin(), indices.end(),
         [&arr](size_t i1, size_t i2) { return arr[i1] < arr[i2]; });

    return indices;
}

/**
 * @brief Determines the indices that would sort a vector.
 *
 * @tparam T Array data type.
 * @param vec Vector to be inspected.
 * @return a vector with indices that would sort the vector.
 */
template <typename T>
inline auto sorting_indices(const std::vector<T> &vec) -> std::vector<size_t> {
    return sorting_indices(vec.data(), vec.size());
}

/**
 * @brief Generate indices for applying operations.
 *
 * This method will return the statevector indices participating in the
 * application of a gate to a given set of qubits.
 *
 * @param qubitIndices Indices of the qubits to apply operations.
 * @param num_qubits Number of qubits in register.
 * @return std::vector<size_t>
 */

inline auto
getIndicesAfterExclusion(const std::vector<size_t> &indicesToExclude,
                         size_t num_qubits) -> std::vector<size_t> {
    std::vector<size_t> indices;
    for (size_t i = 0; i < num_qubits; i++) {
        indices.emplace_back(i);
    }

    for (auto j : indicesToExclude) {
        for (size_t i = 0; i < indices.size(); i++) {
            if (j == indices[i]) {
                indices.erase(indices.begin() + static_cast<int>(i));
            }
        }
    }
    return indices;
}

/**
 * @brief Generate indices for applying operations.
 *
 * This method will return the statevector indices participating in the
 * application of a gate to a given set of qubits.
 *
 * @param qubitIndices Indices of the qubits to apply operations.
 * @param num_qubits Number of qubits in register.
 * @return std::vector<size_t>
 */

inline auto generateBitsPatterns(const std::vector<size_t> &qubitIndices,
                                 size_t num_qubits) -> std::vector<size_t> {
    std::vector<size_t> indices;
    indices.reserve(exp2(qubitIndices.size()));
    indices.emplace_back(0);

    for (size_t index_it0 = 0; index_it0 < qubitIndices.size(); index_it0++) {
        size_t index_it = qubitIndices.size() - 1 - index_it0;
        const size_t value =
            maxDecimalForQubit(qubitIndices[index_it], num_qubits);

        const size_t currentSize = indices.size();
        for (size_t j = 0; j < currentSize; j++) {
            indices.emplace_back(indices[j] + value);
        }
    }
    return indices;
}
/**
 * @brief Determines the transposed index of a tensor stored linearly.
 *  This function assumes each axis will have a length of 2 (|0>, |1>).
 *
 * @param ind index after transposition.
 * @param new_axes new axes distribution.
 * @return unsigned int with the new transposed index.
 */
inline auto transposed_state_index(size_t ind,
                                   const std::vector<size_t> &new_axes)
    -> size_t {
    size_t new_index = 0;
    const size_t max_axis = new_axes.size() - 1;
    // NOLINTNEXTLINE(modernize-loop-convert)
    for (auto axis = new_axes.rbegin(); axis != new_axes.rend(); ++axis) {
        new_index += (ind % 2) << (max_axis - *axis);
        ind /= 2;
    }
    return new_index;
}

/**
 * @brief Template for the transposition of state tensors,
 * axes are assumed to have a length of 2 (|0>, |1>).
 *
 * @tparam T Tensor data type.
 * @param tensor Tensor to be transposed.
 * @param new_axes new axes distribution.
 * @return Transposed Tensor.
 */
template <typename T>
auto transpose_state_tensor(const std::vector<T> &tensor,
                            const std::vector<size_t> &new_axes)
    -> std::vector<T> {
    std::vector<T> transposed_tensor(tensor.size());
    for (size_t ind = 0; ind < tensor.size(); ind++) {
        transposed_tensor[ind] = tensor[transposed_state_index(ind, new_axes)];
    }
    return transposed_tensor;
}

/**
 * @brief Kronecker product of two diagonal matrices. Only diagonal elements are
 * stored.
 *
 * @tparam T Data type.
 * @param diagA A vector containing the values of a diagonal matrix.
 * @param diagB A vector containing the values of a diagonal matrix.
 * @return kronAB A vector containing the diagonal values of the Kronecker
 * product.
 */
template <typename T>
auto kronProd(const std::vector<T> &diagA, const std::vector<T> &diagB)
    -> std::vector<T> {
    std::vector<T> result(diagA.size() * diagB.size(), 0);

    for (size_t i = 0; i < diagA.size(); i++) {
        for (size_t j = 0; j < diagB.size(); j++) {
            result[i * diagB.size() + j] = diagA[i] * diagB[j];
        }
    }

    return result;
}

/**
 * @brief Check if a matrix is a Hermitian matrix.
 *
 * @tparam T Data type.
 *
 * @param n Number of columns.
 * @param lda Number of rows.
 * @param mat A matrix to be checked.
 *
 * @return is_Hermitian Is the matrix a Hermitian matrix or not.
 */
template <typename T>
bool is_Hermitian(size_t n, size_t lda,
                  const std::vector<std::complex<T>> &mat) {
    // TODO OMP support
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < lda; j++) {
            if (mat[j + i * lda] != std::conj(mat[i + j * n])) {
                return false;
            }
        }
    }
    return true;
}

} // namespace Pennylane::Util
