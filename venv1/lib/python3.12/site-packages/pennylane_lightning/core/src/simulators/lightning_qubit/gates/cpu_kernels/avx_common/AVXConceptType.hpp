// Copyright 2023 Xanadu Quantum Technologies Inc.

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
 * Defines AVXConcept types
 */
#pragma once

#include "Macros.hpp"

#ifdef PL_USE_AVX2
#include "AVX2Concept.hpp"
#endif

#ifdef PL_USE_AVX512F
#include "AVX512Concept.hpp"
#endif

namespace Pennylane::LightningQubit::Gates::AVXCommon {
template <class PrecisionT, size_t packed_size> struct AVXConcept;

#ifdef PL_USE_AVX2
template <> struct AVXConcept<float, 8> {
    using Type = AVX2Concept<float>;
};
template <> struct AVXConcept<double, 4> {
    using Type = AVX2Concept<double>;
};
#endif

#ifdef PL_USE_AVX512F
template <> struct AVXConcept<float, 16> {
    using Type = AVX512Concept<float>;
};
template <> struct AVXConcept<double, 8> {
    using Type = AVX512Concept<double>;
};
#endif

template <class PrecisionT, size_t packed_size>
using AVXConceptType = typename AVXConcept<PrecisionT, packed_size>::Type;

} // namespace Pennylane::LightningQubit::Gates::AVXCommon
