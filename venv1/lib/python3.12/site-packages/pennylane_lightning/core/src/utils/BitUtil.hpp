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
 * Defines utility functions for Bitwise operations.
 */
#pragma once
#include <algorithm> // sort
#include <array>
#include <bit>     // countr_zero, popcount, has_single_bit
#include <climits> // CHAR_BIT
#include <cstddef> // size_t
#include <vector>

namespace Pennylane::Util {
/**
 * @brief Faster log2 when the value is a power of 2.
 *
 * @param val Size of the state vector. Expected to be a power of 2.
 * @return size_t Log2(val), or the state vector's number of qubits.
 */
inline auto constexpr log2PerfectPower(size_t val) -> size_t {
    return static_cast<size_t>(std::countr_zero(val));
}

/**
 * @brief Verify if the value provided is a power of 2.
 *
 * @param value state vector size.
 * @return true
 * @return false
 */
inline auto constexpr isPerfectPowerOf2(size_t value) -> bool {
    return std::has_single_bit(value);
}

/**
 * @brief Fill ones from LSB to nbits. Runnable in a compile-time and for any
 * integer type.
 *
 * @tparam IntegerType Integer type to use
 * @param nbits Number of bits to fill
 */
template <class IntegerType = size_t>
inline auto constexpr fillTrailingOnes(size_t nbits) -> IntegerType {
    static_assert(std::is_integral_v<IntegerType> &&
                  std::is_unsigned_v<IntegerType>);

    return (nbits == 0) ? 0
                        : static_cast<IntegerType>(~IntegerType(0)) >>
                              static_cast<IntegerType>(
                                  CHAR_BIT * sizeof(IntegerType) - nbits);
}
/**
 * @brief Fill ones from MSB to pos
 *
 * @tparam IntegerType Integer type to use
 * @param pos Position up to which bit one is filled.
 */
template <class IntegerType = size_t>
inline auto constexpr fillLeadingOnes(size_t pos) -> size_t {
    static_assert(std::is_integral_v<IntegerType> &&
                  std::is_unsigned_v<IntegerType>);

    return (~IntegerType{0}) << pos;
}

/**
 * @brief Swap bits in i-th and j-th position in place
 */
inline auto constexpr bitswap(size_t bits, const size_t i, const size_t j)
    -> size_t {
    size_t x = ((bits >> i) ^ (bits >> j)) & 1U;
    return bits ^ ((x << i) | (x << j));
}

/**
 * @brief Return integers with leading/trailing ones at positions specified by a
 * list of target wires.
 *
 * @param wire_list Target wires.
 */
inline auto revWireParity(const std::vector<std::size_t> &wire_list)
    -> std::vector<std::size_t> {
    const std::size_t wire_size = wire_list.size();
    auto rev_wire = wire_list;
    std::sort(rev_wire.begin(), rev_wire.end());
    std::vector<std::size_t> parity(wire_size + 1);
    parity[0] = fillTrailingOnes(rev_wire[0]);
    for (std::size_t i = 1; i < wire_size; i++) {
        parity[i] = fillLeadingOnes(rev_wire[i - 1] + 1) &
                    fillTrailingOnes(rev_wire[i]);
    }
    parity[wire_size] = fillLeadingOnes(rev_wire[wire_size - 1] + 1);
    return parity;
}

template <std::size_t wire_size>
inline auto revWireParity(const std::array<std::size_t, wire_size> &wire_list)
    -> std::array<std::size_t, wire_size + 1> {
    auto rev_wire = wire_list;
    std::sort(rev_wire.begin(), rev_wire.end());
    std::array<std::size_t, wire_size + 1> parity{};
    parity[0] = fillTrailingOnes(rev_wire[0]);
    for (std::size_t i = 1; i < wire_size; i++) {
        parity[i] = fillLeadingOnes(rev_wire[i - 1] + 1) &
                    fillTrailingOnes(rev_wire[i]);
    }
    parity[wire_size] = fillLeadingOnes(rev_wire[wire_size - 1] + 1);
    return parity;
}

} // namespace Pennylane::Util