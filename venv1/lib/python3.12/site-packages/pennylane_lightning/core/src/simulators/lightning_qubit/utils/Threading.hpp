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
 * Define keys to select kernels
 */
#pragma once

#include "CPUMemoryModel.hpp"
#include "Macros.hpp"

#include <cstdint>

#if defined(PL_USE_OMP)
#include <omp.h>
#endif

/// @cond DEV
namespace {
using Pennylane::Util::CPUMemoryModel;
constexpr uint32_t ThreadBitShift = 8U;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Util {
enum class Threading : uint8_t {
    SingleThread,
    MultiThread,
    END,
    BEGIN = SingleThread,
};

/**
 * @brief Compute dispatch key using threading and memory information.
 *
 * @return Dispatch key
 */
constexpr uint32_t toDispatchKey(Threading threading,
                                 CPUMemoryModel memory_model) {
    /* Threading is in higher priority */
    return (static_cast<uint32_t>(threading) << ThreadBitShift) |
           static_cast<uint32_t>(memory_model);
}

/**
 * @brief Choose the best threading based on the current context.
 *
 * @return Threading
 */
inline auto bestThreading() -> Threading {
#ifdef PL_USE_OMP
    if (omp_in_parallel() != 0) {
        // We are already inside of the openmp parallel region (e.g.
        // inside adjoint diff).
        return Threading::SingleThread;
    }
    return Threading::MultiThread;
#endif
    return Threading::SingleThread;
}

} // namespace Pennylane::LightningQubit::Util
