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
 * @file GatePragmas.hpp
 * Defines macros for enabling various OpenMP options in the gate kernel
 * definitions.
 */
#pragma once

namespace Pennylane::LightningQubit::Gates::Pragmas {

// Defines utility macros to annotate gate-kernel loops with OpenMP parallel-for
// and OpenMP SIMD pragmas. Selectable at compile time.
#if defined PL_LQ_KERNEL_OMP && defined _OPENMP
#define PRAGMA_WRAP(S) _Pragma(#S)
#define PL_LOOP_PARALLEL(x) PRAGMA_WRAP(omp parallel for collapse(x))
#define PL_LOOP_PARALLEL_VA(x, ...) PRAGMA_WRAP(omp parallel for collapse(x) __VA_ARGS__)
#define PL_LOOP_SIMD PRAGMA_WRAP(omp simd)
#else
#define PL_LOOP_PARALLEL(N)
#define PL_LOOP_PARALLEL_VA(N, ...)
#define PL_LOOP_SIMD
#endif

}; // namespace Pennylane::LightningQubit::Gates::Pragmas
