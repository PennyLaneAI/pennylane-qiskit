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
#include <cstdint>

#include "Macros.hpp"
#include "RuntimeInfo.hpp"

#include <array>
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#include <cpuid.h>
#elif defined(_MSC_VER) && defined(_WIN64)
#include <intrin.h>
#include <vector>
#endif

namespace Pennylane::Util {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
RuntimeInfo::InternalRuntimeInfo::InternalRuntimeInfo() {
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;

    __get_cpuid(0x00, &eax, &ebx, &ecx, &edx);

    const auto nids = eax;

    { // nids == 0 for vendor
        std::array<char, 0x20> tmp = {
            0,
        };
        *reinterpret_cast<int *>(tmp.data()) = ebx;
        *reinterpret_cast<int *>(tmp.data() + 4) = edx;
        *reinterpret_cast<int *>(tmp.data() + 8) = ecx;
        vendor = tmp.data();
    }

    if (nids >= 1) {
        eax = 1;
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
        f_1_ecx = ecx;
        f_1_edx = edx;
    }
    if (nids >= 7) { // NOLINT(readability-magic-numbers)
        // NOLINTNEXTLINE(readability-magic-numbers)
        __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
        f_7_ebx = ebx;
        f_7_ecx = ecx;
    }

    // NOLINTNEXTLINE(readability-magic-numbers)
    if (__get_cpuid_max(0x80000004, nullptr) != 0) {
        // NOLINTNEXTLINE(readability-magic-numbers)
        std::array<uint32_t, 12> tmp = {
            0,
        };
        auto *p = tmp.data();
        // NOLINTNEXTLINE(readability-magic-numbers)
        __get_cpuid(0x80000002, p + 0x0, p + 0x1, p + 0x2, p + 0x3);
        // NOLINTNEXTLINE(readability-magic-numbers)
        __get_cpuid(0x80000003, p + 0x4, p + 0x5, p + 0x6, p + 0x7);
        // NOLINTNEXTLINE(readability-magic-numbers)
        __get_cpuid(0x80000004, p + 0x8, p + 0x9, p + 0xa, p + 0xb);

        brand = reinterpret_cast<const char *>(p);
    }
}
#elif defined(_MSC_VER) && defined(_M_AMD64)
RuntimeInfo::InternalRuntimeInfo::InternalRuntimeInfo() {
    std::array<int, 4> cpui = {
        0,
    };
    __cpuid(cpui.data(), 0);

    const int nids = cpui[0];

    { // nids == 0 for vendor
        std::array<char, 0x20> tmp = {
            0,
        };
        *reinterpret_cast<int *>(tmp.data()) = cpui[1];
        *reinterpret_cast<int *>(tmp.data() + 4) = cpui[3];
        *reinterpret_cast<int *>(tmp.data() + 8) = cpui[2];
        vendor = tmp.data();
    }

    if (nids >= 1) {
        __cpuidex(cpui.data(), 1, 0);
        f_1_ecx = cpui[2];
        f_1_edx = cpui[3];
    }

    if (nids >= 7) {
        __cpuidex(cpui.data(), 7, 0);
        f_7_ebx = cpui[1];
        f_7_ecx = cpui[2];
    }

    // Calling __cpuid with 0x80000000 as the function_id argument
    // gets the number of the highest valid extended ID.
    __cpuid(cpui.data(), 0x80000000);
    const int nExIds_ = cpui[0];

    if (nExIds_ >= 0x80000004) {
        std::vector<std::array<int, 4>> tmp(3);
        __cpuidex(tmp[0].data(), 0x80000002, 0);
        __cpuidex(tmp[1].data(), 0x80000003, 0);
        __cpuidex(tmp[2].data(), 0x80000004, 0);

        char str[48];
        memset(str, 0, sizeof(str));
        memcpy(str, tmp[0].data(), sizeof(int) * 4);
        memcpy(str + 16, tmp[1].data(), sizeof(int) * 4);
        memcpy(str + 32, tmp[2].data(), sizeof(int) * 4);
        brand = str;
    }
}
#else
RuntimeInfo::InternalRuntimeInfo::InternalRuntimeInfo() = default;
#endif
} // namespace Pennylane::Util
