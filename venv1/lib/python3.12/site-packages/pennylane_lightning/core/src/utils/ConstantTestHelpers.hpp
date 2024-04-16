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
 * Contains utility functions for processing constants
 */
#pragma once

#include "TypeTraits.hpp"
#include "Util.hpp"

#include <algorithm>
#include <array>
#include <compare>
#include <cstdlib>
#include <stdexcept>

#if __has_include(<version>)
#include <version>
#endif

namespace Pennylane::LightningQubit::Util {
/**
 * @brief Extract first elements from the array of pairs.
 *
 * @tparam T Type of the first elements.
 * @tparam U Type of the second elements.
 * @tparam size Size of the array.
 * @param arr Array to extract.
 */
template <typename T, typename U, size_t size>
constexpr std::array<T, size>
first_elems_of(const std::array<std::pair<T, U>, size> &arr) {
    std::array<T, size> res = {
        T{},
    };
    std::transform(arr.begin(), arr.end(), res.begin(),
                   [](const auto &elem) { return std::get<0>(elem); });
    return res;
}
/**
 * @brief Extract second elements from the array of pairs.
 *
 * @tparam T Type of the first elements.
 * @tparam U Type of the second elements.
 * @tparam size Size of the array.
 * @param arr Array to extract.
 */
template <typename T, typename U, size_t size>
constexpr std::array<U, size>
second_elems_of(const std::array<std::pair<T, U>, size> &arr) {
    std::array<U, size> res = {
        U{},
    };
    std::transform(arr.begin(), arr.end(), res.begin(),
                   [](const auto &elem) { return std::get<1>(elem); });
    return res;
}

/**
 * @brief Count the number of unique elements in the array.
 *
 * This is O(n^2) version for all T
 *
 * @tparam T Type of array elements
 * @tparam size Size of the array
 * @return size_t
 */
template <typename T, size_t size>
constexpr size_t count_unique(const std::array<T, size> &arr) {
    size_t res = 0;

    for (size_t i = 0; i < size; i++) {
        bool counted = false;
        for (size_t j = 0; j < i; j++) {
            if (arr[j] == arr[i]) {
                counted = true;
                break;
            }
        }
        if (!counted) {
            res++;
        }
    }
    return res;
}

#if __cpp_lib_three_way_comparison >= 201907L
/**
 * @brief Count the number of unique elements in the array.
 *
 * This is a specialized version for partially ordered type T.
 *
 * @tparam T Type of array elements
 * @tparam size Size of the array
 * @return size_t
 */
template <std::three_way_comparable T, size_t size>
constexpr size_t count_unique(const std::array<T, size> &arr) {
    auto arr_cpd = arr;
    size_t dup_cnt = 0;
    std::sort(std::begin(arr_cpd), std::end(arr_cpd));
    for (size_t i = 0; i < size - 1; i++) {
        if (arr_cpd[i] == arr_cpd[i + 1]) {
            dup_cnt++;
        }
    }
    return size - dup_cnt;
}
#endif

} // namespace Pennylane::LightningQubit::Util
