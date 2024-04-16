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
#include <catch2/catch.hpp>

#include "ConstantTestHelpers.hpp"
#include "ConstantUtil.hpp"

#include "TestHelpers.hpp"

#if defined(_MSC_VER)
#pragma warning(disable : 4305)
#endif

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::LightningQubit;
} // namespace
/// @endcond

TEST_CASE("Utility array and tuples", "[Util][ConstantUtil]") {
    std::array<std::pair<int, std::string_view>, 5> test_pairs{
        std::pair(0, "Zero"),  std::pair(1, "One"),  std::pair(2, "Two"),
        std::pair(3, "Three"), std::pair(4, "Four"),
    };

    REQUIRE(reverse_pairs(test_pairs) ==
            std::array{
                std::pair<std::string_view, int>("Zero", 0),
                std::pair<std::string_view, int>("One", 1),
                std::pair<std::string_view, int>("Two", 2),
                std::pair<std::string_view, int>("Three", 3),
                std::pair<std::string_view, int>("Four", 4),
            });

    REQUIRE(reverse_pairs(test_pairs) !=
            std::array{
                std::pair<std::string_view, int>("Zero", 0),
                std::pair<std::string_view, int>("One", 1),
                std::pair<std::string_view, int>("Two", 0),
                std::pair<std::string_view, int>("Three", 3),
                std::pair<std::string_view, int>("Four", 4),
            });
}

TEST_CASE("Test utility functions for constants", "[Util][ConstantUtil]") {
    using namespace std::literals;

    SECTION("lookup") {
        constexpr std::array test_pairs = {
            std::pair{"Pennylane"sv, "-"sv},
            std::pair{"Lightning"sv, "is"sv},
            std::pair{"the"sv, "best"sv},
            std::pair{"QML"sv, "library"sv},
        };

        REQUIRE(lookup(test_pairs, "Pennylane"sv) == "-"sv);
        REQUIRE(lookup(test_pairs, "Lightning"sv) == "is"sv);
        REQUIRE(lookup(test_pairs, "the"sv) == "best"sv);
        REQUIRE(lookup(test_pairs, "QML"sv) == "library"sv);
        REQUIRE_THROWS(lookup(test_pairs, "bad"sv));
    }

    SECTION("count_unique") {
        constexpr std::array test_arr1 = {"This"sv, "is"sv, "a"sv, "test"sv,
                                          "arr"sv};
        constexpr std::array test_arr2 = {"This"sv, "is"sv,  "a"sv,
                                          "test"sv, "arr"sv, "is"sv};

        REQUIRE(Util::count_unique(test_arr1) == 5);
        REQUIRE(Util::count_unique(test_arr2) == 5);

        REQUIRE(Util::count_unique(std::array{nullptr, nullptr, nullptr}) == 1);
        REQUIRE(Util::count_unique(std::array{0, 0, 0}) == 1);
        REQUIRE(Util::count_unique(std::array{0, 1, 1}) == 2);
        REQUIRE(Util::count_unique(std::array{0, 1, 2}) == 3);
    }

    SECTION("lookup (constexpr context)") {
        enum class TestEnum { One, Two, Many };

        constexpr std::array test_pairs = {
            std::pair{TestEnum::One, uint32_t{1U}},
            std::pair{TestEnum::Two, uint32_t{2U}},
        };

        static_assert(lookup(test_pairs, TestEnum::One) == 1U);
        static_assert(lookup(test_pairs, TestEnum::Two) == 2U);
    }
}
