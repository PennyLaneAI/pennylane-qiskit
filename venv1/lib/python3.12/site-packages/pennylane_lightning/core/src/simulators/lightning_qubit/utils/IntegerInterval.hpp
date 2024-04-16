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
 * Define IntegerInterval class
 */
#pragma once
#include <algorithm>
#include <limits>
#include <type_traits>

#include "Error.hpp"

namespace Pennylane::LightningQubit::Util {
/**
 * @brief Define integer interval [min_, max_)
 */
template <typename IntegerType> class IntegerInterval {
  private:
    static_assert(std::is_integral_v<IntegerType> &&
                  std::is_unsigned_v<IntegerType>);

    IntegerType min_;
    IntegerType max_;

  public:
    constexpr IntegerInterval(IntegerType min, IntegerType max)
        : min_{min}, max_{max} {
        PL_ASSERT(min < max);
    }
    bool operator()(IntegerType test_val) const {
        return (min_ <= test_val) && (test_val < max_);
    }

    [[nodiscard]] IntegerType min() const { return min_; }

    [[nodiscard]] IntegerType max() const { return max_; }
};

/**
 * @brief Create integer interval (from, inf)
 */
template <typename IntegerType>
constexpr auto larger_than(IntegerType from) -> IntegerInterval<IntegerType> {
    return IntegerInterval<IntegerType>{
        from + 1, std::numeric_limits<IntegerType>::max()};
}
/**
 * @brief Create integer interval [from, inf)
 */
template <typename IntegerType>
constexpr auto larger_than_equal_to(IntegerType from)
    -> IntegerInterval<IntegerType> {
    return IntegerInterval<IntegerType>{
        from, std::numeric_limits<IntegerType>::max()};
}
/**
 * @brief Create integer interval [0, to)
 */
template <typename IntegerType>
constexpr auto less_than(IntegerType to) -> IntegerInterval<IntegerType> {
    return IntegerInterval<IntegerType>{0, to};
}
/**
 * @brief Create integer interval [0, to]
 */
template <typename IntegerType>
constexpr auto less_than_equal_to(IntegerType to)
    -> IntegerInterval<IntegerType> {
    return IntegerInterval<IntegerType>{0, to + 1};
}

/**
 * @brief Create integer interval [from, to]
 */
template <typename IntegerType>
constexpr auto in_between_closed(IntegerType from, IntegerType to)
    -> IntegerInterval<IntegerType> {
    return IntegerInterval<IntegerType>{from, to + 1};
}

/**
 * @brief Create integer interval [0, inf)
 */
template <typename IntegerType>
constexpr auto full_domain() -> IntegerInterval<IntegerType> {
    return IntegerInterval<IntegerType>{
        0, std::numeric_limits<IntegerType>::max()};
}

/**
 * @brief
 * @rst
 * Test if the intersection of two intervals are empty, i.e.
 * :math:`I_1 \cap I_2 = \phi`.
 * @endrst
 */
template <typename IntegerType>
bool is_disjoint(const IntegerInterval<IntegerType> &interval1,
                 const IntegerInterval<IntegerType> &interval2) {
    return (interval1.max() <= interval2.min()) ||
           (interval2.max() <= interval1.min());
}
} // namespace Pennylane::LightningQubit::Util
