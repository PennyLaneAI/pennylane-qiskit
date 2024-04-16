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

#include <algorithm>
#include <array>
#include <compare>
#include <cstdlib>
#include <stdexcept>
#include <tuple>

#if __has_include(<version>)
#include <version>
#endif

#include "TypeTraits.hpp"
#include "Util.hpp"

namespace Pennylane::Util {
/**
 * @brief Lookup key in array of pairs. For a constexpr map-like behavior.
 *
 * @tparam Key Type of keys
 * @tparam Value Type of values
 * @tparam size Size of std::array
 * @param arr Array to lookup
 * @param key Key to find
 */
template <typename Key, typename Value, size_t size>
constexpr auto lookup(const std::array<std::pair<Key, Value>, size> &arr,
                      const Key &key) -> Value {
    for (size_t idx = 0; idx < size; idx++) {
        if (std::get<0>(arr[idx]) == key) {
            return std::get<1>(arr[idx]);
        }
    }
    throw std::range_error("The given key does not exist.");
}

/**
 * @brief Check an array has an element.
 *
 * @tparam U Type of array elements.
 * @tparam size Size of array.
 * @param arr Array to check.
 * @param elem Element to find.
 */
template <typename U, size_t size>
constexpr auto array_has_elem(const std::array<U, size> &arr, const U &elem)
    -> bool {
    for (size_t idx = 0; idx < size; idx++) {
        if (arr[idx] == elem) {
            return true;
        }
    }
    return false;
}

/// @cond DEV
namespace Internal {
/**
 * @brief Helper function for prepend_to_tuple
 */
template <class T, class Tuple, std::size_t... I>
constexpr auto
prepend_to_tuple_helper(T &&elem, Tuple &&t,
                        [[maybe_unused]] std::index_sequence<I...> dummy) {
    return std::make_tuple(elem, std::get<I>(std::forward<Tuple>(t))...);
}
} // namespace Internal
/// @endcond

/**
 * @brief Prepend an element to a tuple
 * @tparam T Type of element
 * @tparam Tuple Type of the tuple (usually std::tuple)
 *
 * @param elem Element to prepend
 * @param t Tuple to add an element
 */
template <class T, class Tuple>
constexpr auto prepend_to_tuple(T &&elem, Tuple &&t) {
    return Internal::prepend_to_tuple_helper(
        std::forward<T>(elem), std::forward<Tuple>(t),
        std::make_index_sequence<
            std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
}

/**
 * @brief Transform a tuple to an array
 *
 * This function only works when all elements of the tuple are the same
 * type or convertible to the same type.
 *
 * @tparam T Type of the elements. This type usually needs to be specified.
 * @tparam Tuple Type of the tuple.
 * @param tuple Tuple to transform
 */
template <class Tuple> constexpr auto tuple_to_array(Tuple &&tuple) {
    using T = std::tuple_element_t<0, std::remove_cvref_t<Tuple>>;
    return std::apply(
        [](auto... n) { return std::array<T, sizeof...(n)>{n...}; },
        std::forward<Tuple>(tuple));
}

/// @cond DEV
namespace Internal {
/**
 * @brief Helper function for reverse_pairs
 */
template <class T, class U, size_t size, std::size_t... I>
constexpr auto
reverse_pairs_helper(const std::array<std::pair<T, U>, size> &arr,
                     [[maybe_unused]] std::index_sequence<I...> dummy) {
    return std::array{std::pair{arr[I].second, arr[I].first}...};
}
} // namespace Internal
/// @endcond

/**
 * @brief Swap positions of elements in each pair
 *
 * @tparam T Type of first elements
 * @tparam U Type of second elements
 * @tparam size Size of the array
 * @param arr Array to reverse
 * @return reversed array
 */
template <class T, class U, size_t size>
constexpr auto reverse_pairs(const std::array<std::pair<T, U>, size> &arr)
    -> std::array<std::pair<U, T>, size> {
    return Internal::reverse_pairs_helper(arr,
                                          std::make_index_sequence<size>{});
}

} // namespace Pennylane::Util
