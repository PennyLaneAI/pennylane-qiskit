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
 * Runtime information based on cpuid
 */
#pragma once
#include <bitset>

namespace {
struct OffsetIndices {
    static constexpr int AVX = 28;
    static constexpr int AVX2 = 5;
    static constexpr int FMA = 12;
    static constexpr int AVX512F = 16;
};
constexpr int BitWidth = 32;

} // namespace

namespace Pennylane::Util {
/**
 * @brief This class is only usable in x86 or x86_64 architecture.
 */
class RuntimeInfo {
  private:
    /// @cond DEV
    struct InternalRuntimeInfo {
        InternalRuntimeInfo();

        std::string vendor{};
        std::string brand{};
        std::bitset<BitWidth> f_1_ecx{};
        std::bitset<BitWidth> f_1_edx{};
        std::bitset<BitWidth> f_7_ebx{};
        std::bitset<BitWidth> f_7_ecx{};
    };
    /// @endcond

    static const InternalRuntimeInfo &getInternalRuntimeInfo() {
        static InternalRuntimeInfo internal_runtime_info;
        return internal_runtime_info;
    }

  public:
    static inline bool AVX() {
        return getInternalRuntimeInfo().f_1_ecx[OffsetIndices::AVX];
    }
    static inline bool AVX2() {
        return getInternalRuntimeInfo().f_7_ebx[OffsetIndices::AVX2];
    }
    static inline bool FMA() {
        // NOLINTNEXTLINE(readability-magic-numbers)
        return getInternalRuntimeInfo().f_1_ecx[OffsetIndices::FMA];
    }
    static inline bool AVX512F() {
        // NOLINTNEXTLINE(readability-magic-numbers)
        return getInternalRuntimeInfo().f_7_ebx[OffsetIndices::AVX512F];
    }
    static const std::string &vendor() {
        return getInternalRuntimeInfo().vendor;
    }
    static const std::string &brand() { return getInternalRuntimeInfo().brand; }
};
} // namespace Pennylane::Util
