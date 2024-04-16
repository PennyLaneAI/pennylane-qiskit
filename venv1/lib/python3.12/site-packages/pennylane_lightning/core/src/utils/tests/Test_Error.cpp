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
#include <cstring>
#include <exception>

#include <catch2/catch.hpp>

#include "Error.hpp"

/**
 * @brief Test LightningException class behaves correctly
 */
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("Error.hpp", "[Error]") {
    SECTION("Raw exception") {
        const auto e = Pennylane::Util::LightningException("Test exception e");
        auto e_mut =
            Pennylane::Util::LightningException("Test exception e_mut");

        REQUIRE_THROWS_WITH(throw e,
                            Catch::Matchers::Contains("Test exception e"));
        REQUIRE_THROWS_AS(throw e, Pennylane::Util::LightningException);

        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        const Pennylane::Util::LightningException e_copy(e);
        REQUIRE_THROWS_WITH(throw e_copy,
                            Catch::Matchers::Contains("Test exception e"));
        REQUIRE_THROWS_AS(throw e_copy, Pennylane::Util::LightningException);

        Pennylane::Util::LightningException e_move(std::move(e_mut));
        REQUIRE_THROWS_WITH(throw e_move,
                            Catch::Matchers::Contains("Test exception e_mut"));
        REQUIRE_THROWS_AS(throw e_move, Pennylane::Util::LightningException);

        REQUIRE(std::strcmp(e.what(), "Test exception e") == 0);
        REQUIRE(std::strcmp(e_copy.what(), "Test exception e") == 0);
        REQUIRE(std::strcmp(e_move.what(), "Test exception e_mut") == 0);
    }
    SECTION("Abort") {
        REQUIRE_THROWS_WITH(
            Pennylane::Util::Abort("Test abort", __FILE__, __LINE__, __func__),
            Catch::Matchers::Contains("Test abort"));
        REQUIRE_THROWS_AS(
            Pennylane::Util::Abort("Test abort", __FILE__, __LINE__, __func__),
            Pennylane::Util::LightningException);

        REQUIRE_THROWS_WITH(PL_ABORT("Test abort"),
                            Catch::Matchers::Contains("Test abort"));
        REQUIRE_THROWS_AS(PL_ABORT("Test abort"),
                          Pennylane::Util::LightningException);
    }
}
