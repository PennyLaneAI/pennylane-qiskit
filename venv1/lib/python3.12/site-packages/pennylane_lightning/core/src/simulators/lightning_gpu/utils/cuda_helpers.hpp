// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Adapted from JET: https://github.com/XanaduAI/jet.git

/**
 * @file cuda_helpers.hpp
 */

#pragma once
#include <algorithm>
#include <complex>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cusparse_v2.h>
#include <custatevec.h>

#include "DevTag.hpp"
#include "cuError.hpp"

namespace Pennylane::LightningGPU::Util {

// SFINAE check for existence of real() method in complex type
template <typename ComplexT>
constexpr auto is_cxx_complex(const ComplexT &t) -> decltype(t.real(), bool()) {
    return true;
}

// Catch-all fallback for CUDA complex types
constexpr bool is_cxx_complex(...) { return false; }

inline cuFloatComplex operator-(const cuFloatComplex &a) {
    return {-a.x, -a.y};
}
inline cuDoubleComplex operator-(const cuDoubleComplex &a) {
    return {-a.x, -a.y};
}

template <class CFP_t_T, class CFP_t_U = CFP_t_T>
inline static auto Div(const CFP_t_T &a, const CFP_t_U &b) -> CFP_t_T {
    if constexpr (std::is_same_v<CFP_t_T, cuComplex> ||
                  std::is_same_v<CFP_t_T, float2>) {
        return cuCdivf(a, b);
    } else if (std::is_same_v<CFP_t_T, cuDoubleComplex> ||
               std::is_same_v<CFP_t_T, double2>) {
        return cuCdiv(a, b);
    }
}

/**
 * @brief Conjugate function for CXX & CUDA complex types
 *
 * @tparam CFP_t Complex data type. Supports std::complex<float>,
 * std::complex<double>, cuFloatComplex, cuDoubleComplex
 * @param a The given complex number
 * @return CFP_t The conjugated complex number
 */
template <class CFP_t>
__host__ __device__ inline static constexpr auto Conj(CFP_t a) -> CFP_t {
    if constexpr (std::is_same_v<CFP_t, cuComplex> ||
                  std::is_same_v<CFP_t, float2>) {
        return cuConjf(a);
    } else {
        return cuConj(a);
    }
}

/**
 * @brief Multiplies two numbers for CXX & CUDA complex types
 *
 * @tparam CFP_t Complex data type. Supports std::complex<float>,
 * std::complex<double>, cuFloatComplex, cuDoubleComplex
 * @param a Complex number
 * @param b Complex number
 * @return CFP_t The multiplication result
 */
template <class CFP_t>
__host__ __device__ inline static constexpr auto Cmul(CFP_t a, CFP_t b)
    -> CFP_t {
    if constexpr (std::is_same_v<CFP_t, cuComplex> ||
                  std::is_same_v<CFP_t, float2>) {
        return cuCmulf(a, b);
    } else {
        return cuCmul(a, b);
    }
}

/**
 * @brief Compile-time scalar real times complex number.
 *
 * @tparam U Precision of real value `a`.
 * @tparam T Precision of complex value `b` and result.
 * @param a Real scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class Real_t, class CFP_t = cuDoubleComplex>
inline static constexpr auto ConstMultSC(Real_t a, CFP_t b) -> CFP_t {
    if constexpr (std::is_same_v<CFP_t, cuDoubleComplex>) {
        return make_cuDoubleComplex(a * b.x, a * b.y);
    } else {
        return make_cuFloatComplex(a * b.x, a * b.y);
    }
}

/**
 * @brief Utility to convert cuComplex types to std::complex types
 *
 * @tparam CFP_t cuFloatComplex or cuDoubleComplex types.
 * @param a CUDA compatible complex type.
 * @return std::complex converted a
 */
template <class CFP_t = cuDoubleComplex>
inline static constexpr auto cuToComplex(CFP_t a)
    -> std::complex<decltype(a.x)> {
    return std::complex<decltype(a.x)>{a.x, a.y};
}

/**
 * @brief Utility to convert std::complex types to cuComplex types
 *
 * @tparam CFP_t std::complex types.
 * @param a A std::complex type.
 * @return cuComplex converted a
 */
template <class CFP_t = std::complex<double>>
inline static constexpr auto complexToCu(CFP_t a) {
    if constexpr (std::is_same_v<CFP_t, std::complex<double>>) {
        return make_cuDoubleComplex(a.real(), a.imag());
    } else {
        return make_cuFloatComplex(a.real(), a.imag());
    }
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
template <class CFP_t_T, class CFP_t_U = CFP_t_T>
inline static constexpr auto ConstMult(CFP_t_T a, CFP_t_U b) -> CFP_t_T {
    if constexpr (is_cxx_complex(b)) {
        return {a.real() * b.real() - a.imag() * b.imag(),
                a.real() * b.imag() + a.imag() * b.real()};
    } else {
        return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
    }
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
template <class CFP_t_T, class CFP_t_U = CFP_t_T>
inline static constexpr auto ConstSum(CFP_t_T a, CFP_t_U b) -> CFP_t_T {
    if constexpr (std::is_same_v<CFP_t_T, cuComplex> ||
                  std::is_same_v<CFP_t_T, float2>) {
        return cuCaddf(a, b);
    } else {
        return cuCadd(a, b);
    }
}

/**
 * @brief Return complex value 1+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{1,0}
 */
template <class CFP_t> inline static constexpr auto ONE() -> CFP_t {
    return {1, 0};
}

/**
 * @brief Return complex value 0+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,0}
 */
template <class CFP_t> inline static constexpr auto ZERO() -> CFP_t {
    return {0, 0};
}

/**
 * @brief Return complex value 0+1i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,1}
 */
template <class CFP_t> inline static constexpr auto IMAG() -> CFP_t {
    return {0, 1};
}

/**
 * @brief Returns sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T sqrt(2)
 */
template <class CFP_t> inline static constexpr auto SQRT2() {
    if constexpr (std::is_same_v<CFP_t, float2> ||
                  std::is_same_v<CFP_t, cuFloatComplex>) {
        return CFP_t{0x1.6a09e6p+0F, 0}; // NOLINT: To be replaced in C++20
    } else if constexpr (std::is_same_v<CFP_t, double2> ||
                         std::is_same_v<CFP_t, cuDoubleComplex>) {
        return CFP_t{0x1.6a09e667f3bcdp+0,
                     0}; // NOLINT: To be replaced in C++20
    } else if constexpr (std::is_same_v<CFP_t, double>) {
        return 0x1.6a09e667f3bcdp+0; // NOLINT: To be replaced in C++20
    } else {
        return 0x1.6a09e6p+0F; // NOLINT: To be replaced in C++20
    }
}

/**
 * @brief Returns inverse sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T 1/sqrt(2)
 */
template <class CFP_t> inline static constexpr auto INVSQRT2() -> CFP_t {
    if constexpr (std::is_same_v<CFP_t, std::complex<float>> ||
                  std::is_same_v<CFP_t, std::complex<double>>) {
        return CFP_t(1 / M_SQRT2, 0);
    } else {
        return Div(CFP_t{1, 0}, SQRT2<CFP_t>());
    }
}

/**
 * If T is a supported data type for gates, this expression will
 * evaluate to `true`. Otherwise, it will evaluate to `false`.
 *
 * Supported data types are `float2`, `double2`, and their aliases.
 *
 * @tparam T candidate data type
 */
template <class T>
constexpr bool is_supported_data_type =
    std::is_same_v<T, cuComplex> || std::is_same_v<T, float2> ||
    std::is_same_v<T, cuDoubleComplex> || std::is_same_v<T, double2>;

/**
 * @brief Simple overloaded method to define CUDA data type.
 *
 * @param t
 * @return cuDoubleComplex
 */
inline cuDoubleComplex getCudaType(const double &t) {
    static_cast<void>(t);
    return {};
}
/**
 * @brief Simple overloaded method to define CUDA data type.
 *
 * @param t
 * @return cuFloatComplex
 */
inline cuFloatComplex getCudaType(const float &t) {
    static_cast<void>(t);
    return {};
}

/**
 * @brief Return the number of supported CUDA capable GPU devices.
 *
 * @return std::size_t
 */
inline int getGPUCount() {
    int result;
    PL_CUDA_IS_SUCCESS(cudaGetDeviceCount(&result));
    return result;
}

/**
 * @brief Return the current GPU device.
 *
 * @return int
 */
inline int getGPUIdx() {
    int result;
    PL_CUDA_IS_SUCCESS(cudaGetDevice(&result));
    return result;
}

inline static void deviceReset() { PL_CUDA_IS_SUCCESS(cudaDeviceReset()); }

/**
 * @brief Checks to see if the given GPU supports the
 * PennyLane-Lightning-GPU device. Minimum supported architecture is SM 7.0.
 *
 * @param device_number GPU device index
 * @return bool
 */
static bool isCuQuantumSupported(int device_number = 0) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_number);
    return deviceProp.major >= 7;
}

/**
 * @brief Get current GPU major.minor support
 *
 * @param device_number
 * @return std::pair<int,int>
 */
static std::pair<int, int> getGPUArch(int device_number = 0) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_number);
    return std::make_pair(deviceProp.major, deviceProp.minor);
}

inline static auto pauliStringToEnum(const std::string &pauli_word)
    -> std::vector<custatevecPauli_t> {
    // Map string rep to Pauli enums
    const std::unordered_map<std::string, custatevecPauli_t> pauli_map{
        std::pair<const std::string, custatevecPauli_t>{std::string("X"),
                                                        CUSTATEVEC_PAULI_X},
        std::pair<const std::string, custatevecPauli_t>{std::string("Y"),
                                                        CUSTATEVEC_PAULI_Y},
        std::pair<const std::string, custatevecPauli_t>{std::string("Z"),
                                                        CUSTATEVEC_PAULI_Z},
        std::pair<const std::string, custatevecPauli_t>{std::string("I"),
                                                        CUSTATEVEC_PAULI_I}};

    static constexpr std::size_t num_char = 1;

    std::vector<custatevecPauli_t> output;
    output.reserve(pauli_word.size());

    for (const auto ch : pauli_word) {
        auto out = pauli_map.at(std::string(num_char, ch));
        output.push_back(out);
    }
    return output;
}

inline static auto pauliStringToOpNames(const std::string &pauli_word)
    -> std::vector<std::string> {
    // Map string rep to Pauli
    const std::unordered_map<std::string, std::string> pauli_map{
        std::pair<const std::string, std::string>{std::string("X"),
                                                  std::string("PauliX")},
        std::pair<const std::string, std::string>{std::string("Y"),
                                                  std::string("PauliY")},
        std::pair<const std::string, std::string>{std::string("Z"),
                                                  std::string("PauliZ")},
        std::pair<const std::string, std::string>{std::string("I"),
                                                  std::string("Identity")}};

    static constexpr std::size_t num_char = 1;

    std::vector<std::string> output;
    output.reserve(pauli_word.size());

    for (const auto ch : pauli_word) {
        auto out = pauli_map.at(std::string(num_char, ch));
        output.push_back(out);
    }
    return output;
}

/**
 * Utility hash function for complex vectors representing matrices.
 */
struct MatrixHasher {
    template <class Precision = double>
    std::size_t
    operator()(const std::vector<std::complex<Precision>> &matrix) const {
        std::size_t hash_val = matrix.size();
        for (const auto &c_val : matrix) {
            hash_val ^= std::hash<Precision>()(c_val.real()) ^
                        std::hash<Precision>()(c_val.imag());
        }
        return hash_val;
    }
};

} // namespace Pennylane::LightningGPU::Util
