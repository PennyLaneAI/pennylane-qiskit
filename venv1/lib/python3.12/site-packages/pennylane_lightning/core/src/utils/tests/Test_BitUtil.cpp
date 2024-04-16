// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "BitUtil.hpp"
#include "TestHelpers.hpp"
#include <catch2/catch.hpp>
#include <cmath>

/// @cond DEV
namespace {
using namespace Pennylane;
} // namespace
/// @endcond

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("Utility bit operations", "[Util][BitUtil]") {
    SECTION("isPerfectPowerOf2") {
        size_t n = 1U;
        CHECK(Util::isPerfectPowerOf2(n));
        for (size_t k = 0; k < sizeof(size_t) - 2; k++) {
            n *= 2;
            CHECK(Util::isPerfectPowerOf2(n));
            CHECK(!Util::isPerfectPowerOf2(n + 1));
        }

        CHECK(!Util::isPerfectPowerOf2(0U));
        CHECK(!Util::isPerfectPowerOf2(124U));
        CHECK(!Util::isPerfectPowerOf2(1077U));
        CHECK(!Util::isPerfectPowerOf2(1000000000U));

        if constexpr (sizeof(size_t) == 8) {
            // if size_t is uint64_t
            CHECK(!Util::isPerfectPowerOf2(1234556789012345678U));
        }
    }

    SECTION("log2PerfectPower") {
        { // for uint32_t
            for (uint32_t c = 0; c < 32; c++) {
                uint32_t n = static_cast<uint32_t>(1U)
                             << static_cast<uint64_t>(c);
                CHECK(Util::log2PerfectPower(n) == c);
            }
        }
        { // for uint64_t
            for (uint32_t c = 0; c < 32; c++) {
                uint32_t n = static_cast<uint64_t>(1U)
                             << static_cast<uint64_t>(c);
                CHECK(Util::log2PerfectPower(n) == c);
            }
        }
    }

    SECTION("Bitswap") {
        CHECK(Util::bitswap(0B001101, 0, 1) == 0B001110);
        CHECK(Util::bitswap(0B001101, 0, 2) == 0B001101);
        CHECK(Util::bitswap(0B001101, 0, 3) == 0B001101);
        CHECK(Util::bitswap(0B001101, 0, 4) == 0B011100);
    }

    SECTION("fillTrailingOnes") {
        CHECK(Util::fillTrailingOnes<uint8_t>(4) == 0B1111);
        CHECK(Util::fillTrailingOnes<uint8_t>(6) == 0B111111);
        CHECK(Util::fillTrailingOnes<uint32_t>(17) == 0B1'1111'1111'1111'1111);
        CHECK(Util::fillTrailingOnes<uint64_t>(54) ==
              0x3F'FFFF'FFFF'FFFF); // 54 == 4*13 + 2
    }
}