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
 * Define macros and compile-time constants.
 */
#pragma once
#include <array>
#include <string>

#if defined(PL_USE_OMP) && !(__has_include(<omp.h>) && defined(_OPENMP))
#undef PL_USE_OMP
#endif
/**
 * @brief Predefined macro variable to a string. Use std::format instead in
 * C++20. TODO: Replace stringify macro definition.
 */
// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define PL_TO_STR_INDIR(x) #x
#define PL_TO_STR(VAR) PL_TO_STR_INDIR(VAR)
// NOLINTEND(cppcoreguidelines-macro-usage)

#if defined(__GNUC__) || defined(__clang__)
#define PL_UNREACHABLE __builtin_unreachable()
#elif defined(_MSC_VER)
#define PL_UNREACHABLE __assume(false)
#else // Unsupported compiler
#define PL_UNREACHABLE
#endif

#if defined(__AVX2__)
#define PL_USE_AVX2 1
[[maybe_unused]] static constexpr bool use_avx2 = true;
#else
[[maybe_unused]] static constexpr bool use_avx2 = false;
#endif

#if defined(__AVX512F__)
#define PL_USE_AVX512F 1
[[maybe_unused]] static constexpr bool use_avx512f = true;
#else
[[maybe_unused]] static constexpr bool use_avx512f = false;
#endif

#if (_OPENMP >= 202011)
#define PL_UNROLL_LOOP _Pragma("omp unroll(8)")
#elif defined(__GNUC__)
#define PL_UNROLL_LOOP _Pragma("GCC unroll 8")
#elif defined(__clang__)
#define PL_UNROLL_LOOP _Pragma("unroll(8)")
#else
#define PL_UNROLL_LOOP
#endif

// Define force inline
#if defined(__GNUC__) || defined(__clang__)
#if NDEBUG
#define PL_FORCE_INLINE __attribute__((always_inline)) inline
#else
#define PL_FORCE_INLINE
#endif
#elif defined(_MSC_VER)
#if NDEBUG
#define PL_FORCE_INLINE __forceinline
#else
#define PL_FORCE_INLINE
#endif
#else
#if NDEBUG
#define PL_FORCE_INLINE inline
#else
#define PL_FORCE_INLINE
#endif
#endif

namespace Pennylane::Util {
/* Create constexpr values */
/// @cond DEV
#if defined(PL_USE_OMP)
[[maybe_unused]] static constexpr bool use_openmp = true;
#else
[[maybe_unused]] static constexpr bool use_openmp = false;
#endif
/// @endcond

enum class CPUArch { X86_64, PPC64, ARM, Unknown };

constexpr auto getCPUArchClangGCC() {
#if defined(__x86_64__)
    return CPUArch::X86_64;
#elif defined(__powerpc64__)
    return CPUArch::PPC64;
#elif defined(__arm__)
    return CPUArch::ARM;
#else
    return CPUArch::Unknown;
#endif
}

constexpr auto getCPUArchMSVC() {
#if defined(_M_AMD64)
    return CPUArch::X86_64;
#elif defined(_M_PPC)
    return CPUArch::PPC64;
#elif defined(_M_ARM)
    return CPUArch::ARM;
#else
    return CPUArch::Unknown;
#endif
}

/// @cond DEV
#if defined(__GNUC__) || defined(__clang__)
[[maybe_unused]] constexpr static auto cpu_arch = getCPUArchClangGCC();
#elif defined(_MSC_VER)
[[maybe_unused]] constexpr static auto cpu_arch = getCPUArchMSVC();
#else
[[maybe_unused]] constexpr static auto cpu_arch = CPUArch::Unknown;
#endif
/// @endcond

enum class OperatingSystem { Linux, Windows, MacOS, Unknown };

#if defined(__APPLE__)
[[maybe_unused]] constexpr static auto operating_system =
    OperatingSystem::MacOS;
#elif defined(__linux__)
[[maybe_unused]] constexpr static auto operating_system =
    OperatingSystem::Linux;
#elif defined(_MSC_VER)
[[maybe_unused]] constexpr static auto operating_system =
    OperatingSystem::Windows;
#else
[[maybe_unused]] constexpr static auto operating_system =
    OperatingSystem::Unknown;
#endif

enum class Compiler { GCC, Clang, MSVC, NVCC, NVHPC, Unknown };

/**
 * @brief When none of the specialized functions is called.
 */
template <Compiler compiler>
constexpr auto getCompilerVersion() -> std::string_view {
    return "Unknown version";
}
/**
 * @brief Create version string for GCC.
 *
 * This function raises an error when instantiated (invoked) if a compiler
 * does not define macros (i.e. other than GCC compatible compilers).
 */
template <>
constexpr auto getCompilerVersion<Compiler::GCC>() -> std::string_view {
    return PL_TO_STR(__GNUC__) "." PL_TO_STR(__GNUC_MINOR__) "." PL_TO_STR(
        __GNUC_PATCHLEVEL__);
}

/**
 * @brief Create version string for Clang.
 *
 * This function raises an error when instantiated (invoked) if a compiler
 * does not define macros (i.e. other than Clang).
 */
template <>
constexpr auto getCompilerVersion<Compiler::Clang>() -> std::string_view {
    return PL_TO_STR(__clang_major__) "." PL_TO_STR(
        __clang_minor__) "." PL_TO_STR(__clang_patchlevel__);
}

/**
 * @brief Create version string for MSVC.
 *
 * This function raises an error when instantiated (invoked) if a compiler
 * does not define macros (i.e. other than MSVC).
 */
template <>
constexpr auto getCompilerVersion<Compiler::MSVC>() -> std::string_view {
    return PL_TO_STR(_MSC_FULL_VER);
}

/**
 * @brief Create version string for NVCC.
 *
 * This function raises an error when instantiated (invoked) if a compiler
 * does not define macros (i.e. other than NVCC).
 *
 * See
 * https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-identification-macro
 * for related information
 */
template <>
constexpr auto getCompilerVersion<Compiler::NVCC>() -> std::string_view {
    return PL_TO_STR(__CUDACC_VER_MAJOR__) "." PL_TO_STR(
        __CUDACC_VER_MINOR__) "." PL_TO_STR(__CUDACC_VER_BUILD__);
}

/**
 * @brief Create version string for NVHPC (C/C++ compilers without CUDA from
 * NVIDIA).
 *
 * This function raises an error when instantiated (invoked) if a compiler
 * does not define macros (i.e. other than NVHPC).
 */
template <>
constexpr auto getCompilerVersion<Compiler::NVHPC>() -> std::string_view {
    return PL_TO_STR(__NVCOMPILER_MAJOR__) "." PL_TO_STR(
        __NVCOMPILER_MINOR__) "." PL_TO_STR(__NVCOMPILER_PATCHLEVEL__);
}
/// @cond DEV
#if defined(__NVCC__)
[[maybe_unused]] constexpr static auto compiler = Compiler::NVCC;
#elif defined(__NVCOMPILER)
[[maybe_unused]] constexpr static auto compiler = Compiler::NVHPC;
#elif defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER)
// All GCC compatible compilers define __GNUC__.
[[maybe_unused]] constexpr static auto compiler = Compiler::GCC;
#elif defined(__clang__)
[[maybe_unused]] constexpr static auto compiler = Compiler::Clang;
#elif defined(_MSC_VER)
[[maybe_unused]] constexpr static auto compiler = Compiler::MSVC;
#else
[[maybe_unused]] constexpr static auto compiler = Compiler::Unknown;
#endif
/// @endcond
[[maybe_unused]] constexpr std::array compiler_names = {
    std::pair{Compiler::NVCC, std::string_view{"NVCC"}},
    std::pair{Compiler::NVHPC, std::string_view{"NVHPC"}},
    std::pair{Compiler::GCC, std::string_view{"GCC"}},
    std::pair{Compiler::Clang, std::string_view{"Clang"}},
    std::pair{Compiler::MSVC, std::string_view{"MSVC"}},
    std::pair{Compiler::Unknown, std::string_view{"Unknown"}},
};
} // namespace Pennylane::Util
