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
#include "TypeTraits.hpp"

#include <catch2/catch.hpp>

#include <tuple>
#include <type_traits>
#include <utility>

/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

TEST_CASE("Test remove_complex") {
    SECTION("remove_complex returns the floating point if the given type is "
            "std::complex") {
        STATIC_REQUIRE(
            std::is_same_v<remove_complex_t<std::complex<float>>, float>);
        STATIC_REQUIRE(
            std::is_same_v<remove_complex_t<std::complex<double>>, double>);
    }
    SECTION("remove_complex returns the same type if not") {
        STATIC_REQUIRE(std::is_same_v<remove_complex_t<float>, float>);
        STATIC_REQUIRE(std::is_same_v<remove_complex_t<double>, double>);
    }
}

TEST_CASE("Test is_complex") {
    SECTION("is_complex returns true if the given type is std::complex") {
        STATIC_REQUIRE(is_complex_v<std::complex<double>>);
        STATIC_REQUIRE(is_complex_v<std::complex<float>>);
    }
    SECTION("remove_complex returns false if not") {
        STATIC_REQUIRE(!is_complex_v<int>);
        STATIC_REQUIRE(!is_complex_v<long>);
        STATIC_REQUIRE(!is_complex_v<float>);
        STATIC_REQUIRE(!is_complex_v<double>);
    }
}

std::pair<int, int> g(std::tuple<int, int, int>);

TEST_CASE("Test FuncReturn") {
    SECTION("FuncReturn gives correct return types") {
        STATIC_REQUIRE(
            std::is_same_v<FuncReturn<decltype(g)>::Type,
                           std::pair<int, int>>); // return type of g is
                                                  // std::pair<int, int>

        using FuncPtr = std::pair<int, int> (*)(std::tuple<int, int, int>);
        STATIC_REQUIRE(
            std::is_same_v<FuncReturn<FuncPtr>::Type,
                           std::pair<int, int>>); // return type of g is
                                                  // std::pair<int, int>
    }
}
