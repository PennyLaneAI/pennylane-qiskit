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
#include <complex>
#include <random>
#include <vector>

#include <catch2/catch.hpp>

#include "TestHelpers.hpp" // createZeroState, createRandomStateVectorData
#include "TypeList.hpp"

/**
 * @file
 *  Tests for functionality defined in the StateVectorBase class.
 */

/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

#ifdef _ENABLE_PLQUBIT
constexpr bool BACKEND_FOUND = true;

#include "TestHelpersStateVectors.hpp" // TestStateVectorBackends, StateVectorToName
#include "TestHelpersWires.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit::Util;
} // namespace
/// @endcond

#elif _ENABLE_PLKOKKOS == 1
constexpr bool BACKEND_FOUND = true;

#include "TestHelpersStateVectors.hpp" // TestStateVectorBackends, StateVectorToName
#include "TestHelpersWires.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos::Util;
} // namespace
  /// @endcond

#else
constexpr bool BACKEND_FOUND = false;
using TestStateVectorBackends = Pennylane::Util::TypeList<void>;

template <class StateVector> struct StateVectorToName {};
#endif

template <typename TypeList> void testStateVectorBase() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using ComplexT = typename StateVectorT::ComplexT;

        const size_t num_qubits = 4;
        auto st_data = createZeroState<ComplexT>(num_qubits);

        StateVectorT state_vector(st_data.data(), st_data.size());

        DYNAMIC_SECTION("Methods implemented in the base class - "
                        << StateVectorToName<StateVectorT>::name) {
            REQUIRE(state_vector.getNumQubits() == 4);
            REQUIRE(state_vector.getLength() == 16);
        }
        testStateVectorBase<typename TypeList::Next>();
    }
}

TEST_CASE("StateVectorBase", "[StateVectorBase]") {
    if constexpr (BACKEND_FOUND) {
        testStateVectorBase<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testApplyOperations() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        std::mt19937_64 re{1337};
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;

        const size_t num_qubits = 3;

        DYNAMIC_SECTION("Apply operations without parameters - "
                        << StateVectorToName<StateVectorT>::name) {
            auto st_data_1 =
                createRandomStateVectorData<PrecisionT>(re, num_qubits);
            auto st_data_2 = st_data_1;

            StateVectorT state_vector_1(
                reinterpret_cast<ComplexT *>(st_data_1.data()),
                st_data_1.size());
            StateVectorT state_vector_2(
                reinterpret_cast<ComplexT *>(st_data_2.data()),
                st_data_2.size());

            state_vector_1.applyOperations({"PauliX", "PauliY"}, {{0}, {1}},
                                           {false, false});

            state_vector_2.applyOperation("PauliX", {0}, false);
            state_vector_2.applyOperation("PauliY", {1}, false);

            REQUIRE(isApproxEqual(
                state_vector_1.getData(), state_vector_1.getLength(),
                state_vector_2.getData(), state_vector_2.getLength()));
        }

        DYNAMIC_SECTION("Apply 0-controlled operations without parameters - "
                        << StateVectorToName<StateVectorT>::name) {
            auto st_data_1 =
                createRandomStateVectorData<PrecisionT>(re, num_qubits);
            auto st_data_2 = st_data_1;

            StateVectorT state_vector_1(
                reinterpret_cast<ComplexT *>(st_data_1.data()),
                st_data_1.size());
            StateVectorT state_vector_2(
                reinterpret_cast<ComplexT *>(st_data_2.data()),
                st_data_2.size());

            state_vector_1.applyOperation("PauliX", {}, {}, {0}, false);
            state_vector_1.applyOperation("PauliY", {}, {}, {1}, false);

            state_vector_2.applyOperation("PauliX", {}, {}, {0}, false);
            state_vector_2.applyOperation("PauliY", {}, {}, {1}, false);

            REQUIRE(isApproxEqual(
                state_vector_1.getData(), state_vector_1.getLength(),
                state_vector_2.getData(), state_vector_2.getLength()));
        }

        DYNAMIC_SECTION("Apply operations with parameters - "
                        << StateVectorToName<StateVectorT>::name) {
            auto st_data_1 =
                createRandomStateVectorData<PrecisionT>(re, num_qubits);
            auto st_data_2 = st_data_1;

            StateVectorT state_vector_1(
                reinterpret_cast<ComplexT *>(st_data_1.data()),
                st_data_1.size());
            StateVectorT state_vector_2(
                reinterpret_cast<ComplexT *>(st_data_2.data()),
                st_data_2.size());

            state_vector_1.applyOperations({"RX", "RY"}, {{0}, {1}},
                                           {false, false}, {{0.1}, {0.2}});

            state_vector_2.applyOperation("RX", {0}, false, {0.1});
            state_vector_2.applyOperation("RY", {1}, false, {0.2});

            REQUIRE(isApproxEqual(
                state_vector_1.getData(), state_vector_1.getLength(),
                state_vector_2.getData(), state_vector_2.getLength()));
        }
        testApplyOperations<typename TypeList::Next>();
    }
}

TEST_CASE("StateVectorBase::applyOperations", "[applyOperations]") {
    if constexpr (BACKEND_FOUND) {
        testApplyOperations<TestStateVectorBackends>();
    }
}
