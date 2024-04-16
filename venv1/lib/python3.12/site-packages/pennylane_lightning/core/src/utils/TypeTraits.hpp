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
 * Contains type traits
 */
#pragma once
#include <complex>
#include <limits>

namespace Pennylane::Util {
template <typename T> struct remove_complex {
    using type = T;
};
template <typename T> struct remove_complex<std::complex<T>> {
    using type = T;
};

template <typename T> using remove_complex_t = typename remove_complex<T>::type;

template <typename T> struct is_complex : std::false_type {};

template <typename T> struct is_complex<std::complex<T>> : std::true_type {};

template <typename T> constexpr bool is_complex_v = is_complex<T>::value;

/**
 * @brief Function return type
 *
 * Usage:
 * .. code-block::cpp
 *
 *     std::pair<int, int> g(std::tuple<int, int, int>);
 *     static_assert(std::is_same_v<FuncReturn<decltype(g)>::Type,
 * std::pair<int,int>>); // return type of g is std::pair<int, int>
 *
 */

template <class F> struct FuncReturn {
    // When instantiated
    static_assert(sizeof(F) == std::numeric_limits<size_t>::max(),
                  "The given type is not a function. Currently, lambda"
                  "functions are not supported.");
};
template <class R, class... A> struct FuncReturn<R (*)(A...)> {
    using Type = R;
};
template <class R, class... A> struct FuncReturn<R(A...)> {
    using Type = R;
};

} // namespace Pennylane::Util
