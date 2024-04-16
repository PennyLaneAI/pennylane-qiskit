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
#include "cpu_kernels/avx_common/SingleQubitGateHelper.hpp"

#include <catch2/catch.hpp>
#include <tuple>

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit::Gates::AVXCommon;
} // namespace
/// @endcond

template <typename PrecisionT, size_t packed_size>
struct MockSingleQubitGateWithoutParam {
    using Precision = PrecisionT;
    constexpr static size_t packed_size_ = packed_size;

    template <size_t rev_wire>
    static std::tuple<std::string, size_t, bool>
    applyInternal(std::complex<PrecisionT> *arr, const size_t num_qubits,
                  bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        return {"applyInternal", rev_wire, inverse};
    }

    static std::tuple<std::string, size_t, bool>
    applyExternal(std::complex<PrecisionT> *arr, const size_t num_qubits,
                  const size_t rev_wire, bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(rev_wire);
        static_cast<void>(inverse);
        return {"applyExternal", rev_wire, inverse};
    }
};

template <typename PrecisionT, size_t packed_size>
struct MockSingleQubitGateWithParam {
    using Precision = PrecisionT;
    constexpr static size_t packed_size_ = packed_size;

    template <size_t rev_wire, class ParamT>
    static std::tuple<std::string, size_t, bool>
    applyInternal(std::complex<PrecisionT> *arr, const size_t num_qubits,
                  bool inverse, ParamT angle) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        static_cast<void>(angle);
        return {"applyInternal", rev_wire, inverse};
    }

    template <class ParamT>
    static std::tuple<std::string, size_t, bool>
    applyExternal(std::complex<PrecisionT> *arr, const size_t num_qubits,
                  const size_t rev_wire, bool inverse, ParamT angle) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(rev_wire);
        static_cast<void>(inverse);
        static_cast<void>(angle);
        return {"applyExternal", rev_wire, inverse};
    }
};

/**
 * @brief Mock class that only `applyExternal` takes a parameter (which is
 * wrong).
 */
template <typename PrecisionT, size_t packed_size>
struct MockSingleQubitGateSomethingWrong {
    using Precision = PrecisionT;
    constexpr static size_t packed_size_ = packed_size;

    template <size_t rev_wire>
    static std::tuple<std::string, size_t, bool>
    applyInternal(std::complex<PrecisionT> *arr, const size_t num_qubits,
                  bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        return {"applyInternal", rev_wire, inverse};
    }

    template <class ParamT>
    static std::tuple<std::string, size_t, bool>
    applyExternal(std::complex<PrecisionT> *arr, const size_t num_qubits,
                  const size_t rev_wire, bool inverse, ParamT angle) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(rev_wire);
        static_cast<void>(inverse);
        static_cast<void>(angle);
        return {"applyExternal", rev_wire, inverse};
    }
};

TEMPLATE_TEST_CASE("Test SingleQubitGateHelper template functions",
                   "[SingleQubitGateHelper]", float, double) {
    STATIC_REQUIRE(HasInternalWithoutParam<
                   MockSingleQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasInternalWithParam<
                   MockSingleQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(HasExternalWithoutParam<
                   MockSingleQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasExternalWithParam<
                   MockSingleQubitGateWithoutParam<TestType, 4>>::value);

    STATIC_REQUIRE(!HasInternalWithoutParam<
                   MockSingleQubitGateWithParam<TestType, 4>>::value);
    STATIC_REQUIRE(
        HasInternalWithParam<MockSingleQubitGateWithParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasExternalWithoutParam<
                   MockSingleQubitGateWithParam<TestType, 4>>::value);
    STATIC_REQUIRE(
        HasExternalWithParam<MockSingleQubitGateWithParam<TestType, 4>>::value);

    STATIC_REQUIRE(HasInternalWithoutParam<
                   MockSingleQubitGateSomethingWrong<TestType, 4>>::value);
    STATIC_REQUIRE(!HasInternalWithParam<
                   MockSingleQubitGateSomethingWrong<TestType, 4>>::value);
    STATIC_REQUIRE(!HasExternalWithoutParam<
                   MockSingleQubitGateSomethingWrong<TestType, 4>>::value);
    STATIC_REQUIRE(HasExternalWithParam<
                   MockSingleQubitGateSomethingWrong<TestType, 4>>::value);

    // Test concepts
    STATIC_REQUIRE(SingleQubitGateWithoutParam<
                   MockSingleQubitGateWithoutParam<TestType, 4>>);
    STATIC_REQUIRE(
        SingleQubitGateWithParam<MockSingleQubitGateWithParam<TestType, 4>>);
}

TEMPLATE_TEST_CASE("Test SingleQubitGateWithoutParamHelper",
                   "[SingleQubitGateHelper]", float, double) {
    auto fallback = [](std::complex<TestType> *arr, size_t num_qubits,
                       const std::vector<size_t> &wires,
                       bool inverse) -> std::tuple<std::string, size_t, bool> {
        static_cast<void>(arr);
        return {"fallback", num_qubits - wires[0] - 1, inverse};
    };
    SECTION("Test SingleQubitGateWithoutParamHelper with packed_size = 4") {
        constexpr size_t packed_size = 4;
        std::vector<std::complex<TestType>> arr(
            16, std::complex<TestType>{0.0, 0.0});
        SingleQubitGateWithoutParamHelper<
            MockSingleQubitGateWithoutParam<TestType, packed_size>>
            func(fallback);
        // We pack 4 real numbers -> 2 complex numbers -> single qubit.
        // Thus only rev_wire = 0 calls the internal functions.

        for (bool inverse : {false, true}) {
            { // num_qubits= 4, wires = {0} -> rev_wires = 3
                const auto res = func(arr.data(), 4, {0}, inverse);
                REQUIRE(std::get<0>(res) == std::string("applyExternal"));
                REQUIRE(std::get<1>(res) == 3);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits= 4, wires = {1} -> rev_wires = 2
                const auto res = func(arr.data(), 4, {1}, inverse);
                REQUIRE(std::get<0>(res) == std::string("applyExternal"));
                REQUIRE(std::get<1>(res) == 2);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits= 4, wires = {2} -> rev_wires = 1
                const auto res = func(arr.data(), 4, {2}, inverse);
                REQUIRE(std::get<0>(res) == std::string("applyExternal"));
                REQUIRE(std::get<1>(res) == 1);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits= 4, wires = {3} -> rev_wires = 0
                const auto res = func(arr.data(), 4, {3}, inverse);
                REQUIRE(std::get<0>(res) == std::string("applyInternal"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits = 1 -> do not call fallback (as a single qubit
              // statevector fits into the packed data type)
                const auto res = func(arr.data(), 1, {0}, inverse);
                REQUIRE(std::get<0>(res) != std::string("fallback"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == inverse);
            }
        }
    }

    SECTION("Test SingleQubitGateWithoutParamHelper with packed_size = 8") {
        constexpr size_t packed_size = 8;
        std::vector<std::complex<TestType>> arr(
            16, std::complex<TestType>{0.0, 0.0});
        SingleQubitGateWithoutParamHelper<
            MockSingleQubitGateWithoutParam<TestType, packed_size>>
            func(fallback);
        // We pack 8 real numbers -> 4 complex numbers -> two qubits.
        // Thus rev_wire = 0 or 1 calls the internal functions.

        for (bool inverse : {false, true}) {
            { // num_qubits= 4, wires = {0} -> rev_wires = 3
                const auto res = func(arr.data(), 4, {0}, inverse);
                REQUIRE(std::get<0>(res) == std::string("applyExternal"));
                REQUIRE(std::get<1>(res) == 3);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits= 4, wires = {1} -> rev_wires = 2
                const auto res = func(arr.data(), 4, {1}, inverse);
                REQUIRE(std::get<0>(res) == std::string("applyExternal"));
                REQUIRE(std::get<1>(res) == 2);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits= 4, wires = {2} -> rev_wires = 1
                const auto res = func(arr.data(), 4, {2}, inverse);
                REQUIRE(std::get<0>(res) == std::string("applyInternal"));
                REQUIRE(std::get<1>(res) == 1);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits= 4, wires = {3} -> rev_wires = 0
                const auto res = func(arr.data(), 4, {3}, inverse);
                REQUIRE(std::get<0>(res) == std::string("applyInternal"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits = 1 -> call fallback
                const auto res = func(arr.data(), 1, {0}, inverse);
                REQUIRE(std::get<0>(res) == std::string("fallback"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits = 2 -> do not call fallback (as two qubits
              // statevector == 4 complex numbers fits to the packed data type)
                const auto res = func(arr.data(), 2, {0}, inverse);
                REQUIRE(std::get<0>(res) != std::string("fallback"));
                REQUIRE(std::get<1>(res) == 1);
                REQUIRE(std::get<2>(res) == inverse);
            }
        }
    }
}

TEMPLATE_TEST_CASE("Test SingleQubitGateWithParamHelper",
                   "[SingleQubitGateHelper]", float, double) {
    auto fallback =
        [](std::complex<TestType> *arr, size_t num_qubits,
           const std::vector<size_t> &wires, bool inverse,
           TestType angle) -> std::tuple<std::string, size_t, bool> {
        static_cast<void>(arr);
        static_cast<void>(angle);
        return {"fallback", num_qubits - wires[0] - 1, inverse};
    };
    SECTION("Test SingleQubitGateWithoutParamHelper with packed_size = 4") {
        constexpr size_t packed_size = 4;
        std::vector<std::complex<TestType>> arr(
            16, std::complex<TestType>{0.0, 0.0});
        SingleQubitGateWithParamHelper<
            MockSingleQubitGateWithParam<TestType, packed_size>, TestType>
            func(fallback);
        // We pack 4 real numbers -> 2 complex numbers -> single qubit.
        // Thus only rev_wire = 0 calls the internal functions.

        TestType angle = 0.312;

        for (bool inverse : {false, true}) {
            { // num_qubits= 4, wires = {0} -> rev_wires = 3
                const auto res = func(arr.data(), 4, {0}, inverse, angle);
                REQUIRE(std::get<0>(res) == std::string("applyExternal"));
                REQUIRE(std::get<1>(res) == 3);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits= 4, wires = {1} -> rev_wires = 2
                const auto res = func(arr.data(), 4, {1}, inverse, angle);
                REQUIRE(std::get<0>(res) == std::string("applyExternal"));
                REQUIRE(std::get<1>(res) == 2);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits= 4, wires = {2} -> rev_wires = 1
                const auto res = func(arr.data(), 4, {2}, inverse, angle);
                REQUIRE(std::get<0>(res) == std::string("applyExternal"));
                REQUIRE(std::get<1>(res) == 1);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits= 4, wires = {3} -> rev_wires = 0
                const auto res = func(arr.data(), 4, {3}, inverse, angle);
                REQUIRE(std::get<0>(res) == std::string("applyInternal"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits = 1 -> do not call fallback (as a single qubit
              // statevector fits into the packed data type)
                const auto res = func(arr.data(), 1, {0}, inverse, angle);
                REQUIRE(std::get<0>(res) != std::string("fallback"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == inverse);
            }
        }
    }

    SECTION("Test SingleQubitGateWithoutParamHelper with packed_size = 8") {
        constexpr size_t packed_size = 8;
        std::vector<std::complex<TestType>> arr(
            16, std::complex<TestType>{0.0, 0.0});
        SingleQubitGateWithParamHelper<
            MockSingleQubitGateWithParam<TestType, packed_size>, TestType>
            func(fallback);
        // We pack 8 real numbers -> 4 complex numbers -> two qubits.
        // Thus rev_wire = 0 or 1 calls the internal functions.

        TestType angle = 0.312;

        for (bool inverse : {false, true}) {
            { // num_qubits= 4, wires = {0} -> rev_wires = 3
                const auto res = func(arr.data(), 4, {0}, inverse, angle);
                REQUIRE(std::get<0>(res) == std::string("applyExternal"));
                REQUIRE(std::get<1>(res) == 3);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits= 4, wires = {1} -> rev_wires = 2
                const auto res = func(arr.data(), 4, {1}, inverse, angle);
                REQUIRE(std::get<0>(res) == std::string("applyExternal"));
                REQUIRE(std::get<1>(res) == 2);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits= 4, wires = {2} -> rev_wires = 1
                const auto res = func(arr.data(), 4, {2}, inverse, angle);
                REQUIRE(std::get<0>(res) == std::string("applyInternal"));
                REQUIRE(std::get<1>(res) == 1);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits= 4, wires = {3} -> rev_wires = 0
                const auto res = func(arr.data(), 4, {3}, inverse, angle);
                REQUIRE(std::get<0>(res) == std::string("applyInternal"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits = 1 -> call fallback
                const auto res = func(arr.data(), 1, {0}, inverse, angle);
                REQUIRE(std::get<0>(res) == std::string("fallback"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == inverse);
            }
            { // num_qubits = 2 -> do not call fallback (as two qubits
              // statevector == 4 complex numbers fits to the packed data type)
                const auto res = func(arr.data(), 2, {0}, inverse, angle);
                REQUIRE(std::get<0>(res) != std::string("fallback"));
                REQUIRE(std::get<1>(res) == 1);
                REQUIRE(std::get<2>(res) == inverse);
            }
        }
    }
}
