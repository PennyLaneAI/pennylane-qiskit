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
 * Define memory models for CPU
 */
#pragma once
#include "Macros.hpp" // operating_system, OperatingSystem, cpu_arch
#include "Memory.hpp"
#include "RuntimeInfo.hpp"

#include <cstdint>
#include <memory>

namespace {
struct BitWidth {
    static constexpr uint32_t b64 = 64U;
    static constexpr uint32_t b32 = 32U;
};
} // namespace

// LCOV_EXCL_START
namespace Pennylane::Util {
/**
 * @brief Enum class for defining CPU memory alignments
 */
enum class CPUMemoryModel : uint8_t {
    Unaligned,
    Aligned256,
    Aligned512,
    END,
    BEGIN = Unaligned,
};

/**
 * @brief Compute alignment of a given data pointer
 *
 * @param ptr Pointer to data
 * @return CPUMemoryModel
 */
inline auto getMemoryModel(const void *ptr) -> CPUMemoryModel {
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
    if ((reinterpret_cast<uintptr_t>(ptr) % BitWidth::b64) == 0) {
        return CPUMemoryModel::Aligned512;
    }

    if ((reinterpret_cast<uintptr_t>(ptr) % BitWidth::b32) == 0) {
        return CPUMemoryModel::Aligned256;
    }
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

    return CPUMemoryModel::Unaligned;
}

/**
 * @brief Choose the best memory model to use using runtime/compile-time
 * information.
 *
 * @return CPUMemoryModel
 */
inline auto bestCPUMemoryModel() -> CPUMemoryModel {
    constexpr static bool is_unix =
        (operating_system == OperatingSystem::MacOS) ||
        (operating_system == OperatingSystem::Linux);
    if constexpr ((cpu_arch == CPUArch::X86_64) && is_unix) {
        // We enable AVX2/512 only for X86_64 arch with UNIX compatible OSs
        if (RuntimeInfo::AVX512F()) {
            // and the CPU support it as well
            return CPUMemoryModel::Aligned512;
        }
        if (RuntimeInfo::AVX2() && RuntimeInfo::FMA()) {
            return CPUMemoryModel::Aligned256;
        }
    }
    return CPUMemoryModel::Unaligned;
}

/**
 * @brief Return alignment of a given memory model.
 *
 * @tparam T Data type
 */
template <class T>
constexpr inline auto getAlignment(CPUMemoryModel memory_model) -> uint32_t {
    switch (memory_model) {
    case CPUMemoryModel::Aligned256:
        return BitWidth::b32;
    case CPUMemoryModel::Aligned512:
        return BitWidth::b64;
    default:
        return alignof(T);
    }
}

/**
 * @brief Get a corresponding allocator for standard library containers.
 *
 * @tparam T Data type
 */
template <class T>
constexpr auto getAllocator(CPUMemoryModel memory_model)
    -> Util::AlignedAllocator<T> {
    return Util::AlignedAllocator<T>{getAlignment<T>(memory_model)};
}

template <class T>
constexpr auto getBestAllocator() -> Util::AlignedAllocator<T> {
    return getAllocator<T>(bestCPUMemoryModel());
}
} // namespace Pennylane::Util
// LCOV_EXCL_STOP
