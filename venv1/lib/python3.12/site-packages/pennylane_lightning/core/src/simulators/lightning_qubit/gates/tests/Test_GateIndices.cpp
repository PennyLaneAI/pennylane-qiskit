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
#include "GateIndices.hpp"

#include <catch2/catch.hpp>

#include <vector>

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::LightningQubit::Gates;
} // namespace
/// @endcond

TEST_CASE("generateBitPatterns", "[GateUtil]") {
    const size_t num_qubits = 4;
    SECTION("Qubit indices {}") {
        auto bit_pattern = generateBitPatterns({}, num_qubits);
        CHECK(bit_pattern == std::vector<size_t>{0});
    }
    SECTION("Qubit indices {i}") {
        for (size_t i = 0; i < num_qubits; i++) {
            std::vector<size_t> expected{0, size_t{1U} << (num_qubits - i - 1)};
            auto bit_pattern = generateBitPatterns({i}, num_qubits);
            CHECK(bit_pattern == expected);
        }
    }
    SECTION("Qubit indices {i,i+1,i+2}") {
        std::vector<size_t> expected_123{0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<size_t> expected_012{0, 2, 4, 6, 8, 10, 12, 14};
        auto bit_pattern_123 = generateBitPatterns({1, 2, 3}, num_qubits);
        auto bit_pattern_012 = generateBitPatterns({0, 1, 2}, num_qubits);

        CHECK(bit_pattern_123 == expected_123);
        CHECK(bit_pattern_012 == expected_012);
    }
    SECTION("Qubit indices {0,2,3}") {
        std::vector<size_t> expected{0, 1, 2, 3, 8, 9, 10, 11};
        auto bit_pattern = generateBitPatterns({0, 2, 3}, num_qubits);

        CHECK(bit_pattern == expected);
    }
    SECTION("Qubit indices {3,1,0}") {
        std::vector<size_t> expected{0, 8, 4, 12, 1, 9, 5, 13};
        auto bit_pattern = generateBitPatterns({3, 1, 0}, num_qubits);
        CHECK(bit_pattern == expected);
    }
}

TEST_CASE("getIndicesAfterExclusion", "[GateUtil]") {
    const size_t num_qubits = 4;
    SECTION("Qubit indices {}") {
        std::vector<size_t> expected{0, 1, 2, 3};
        auto indices = getIndicesAfterExclusion({}, num_qubits);
        CHECK(indices == expected);
    }
    SECTION("Qubit indices {i}") {
        for (size_t i = 0; i < num_qubits; i++) {
            std::vector<size_t> expected{0, 1, 2, 3};
            expected.erase(expected.begin() + i);

            auto indices = getIndicesAfterExclusion({i}, num_qubits);
            CHECK(indices == expected);
        }
    }
    SECTION("Qubit indices {i,i+1,i+2}") {
        std::vector<size_t> expected_123{0};
        std::vector<size_t> expected_012{3};
        auto indices_123 = getIndicesAfterExclusion({1, 2, 3}, num_qubits);
        auto indices_012 = getIndicesAfterExclusion({0, 1, 2}, num_qubits);

        CHECK(indices_123 == expected_123);
        CHECK(indices_012 == expected_012);
    }
    SECTION("Qubit indices {0,2,3}") {
        std::vector<size_t> expected{1};
        auto indices = getIndicesAfterExclusion({0, 2, 3}, num_qubits);

        CHECK(indices == expected);
    }
    SECTION("Qubit indices {3,1,0}") {
        std::vector<size_t> expected{2};
        auto indices = getIndicesAfterExclusion({3, 1, 0}, num_qubits);

        CHECK(indices == expected);
    }
}
