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
#include "cpu_kernels/avx_common/TwoQubitGateHelper.hpp"

#include <catch2/catch.hpp>
#include <tuple>

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit::Gates::AVXCommon;
} // namespace
/// @endcond

/**
 * Define mock classes. For symmetric gate, we do not have
 * ``applyExternalInternal`` member function.
 */

template <typename PrecisionT, size_t packed_size>
struct MockSymmetricTwoQubitGateWithoutParam {
    using Precision = PrecisionT;
    constexpr static size_t packed_size_ = packed_size;
    constexpr static bool symmetric = true;

    template <size_t rev_wire0, size_t rev_wire1>
    static std::tuple<std::string, size_t, size_t, bool>
    applyInternalInternal(std::complex<PrecisionT> *arr,
                          const size_t num_qubits, bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        return {"applyInternalInternal", rev_wire0, rev_wire1, inverse};
    }

    template <size_t rev_wire0>
    static std::tuple<std::string, size_t, size_t, bool>
    applyInternalExternal(std::complex<PrecisionT> *arr,
                          const size_t num_qubits, size_t rev_wire1,
                          bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        return {"applyInternalExternal", rev_wire0, rev_wire1, inverse};
    }

    static std::tuple<std::string, size_t, size_t, bool>
    applyExternalExternal(std::complex<PrecisionT> *arr,
                          const size_t num_qubits, size_t rev_wire0,
                          size_t rev_wire1, bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        return {"applyExternalExternal", rev_wire0, rev_wire1, inverse};
    }
};

template <typename PrecisionT, size_t packed_size>
struct MockSymmetricTwoQubitGateWithParam {
    using Precision = PrecisionT;
    constexpr static size_t packed_size_ = packed_size;
    constexpr static bool symmetric = true;

    template <size_t rev_wire0, size_t rev_wire1, typename ParamT>
    static std::tuple<std::string, size_t, size_t, bool>
    applyInternalInternal(std::complex<PrecisionT> *arr,
                          const size_t num_qubits, bool inverse, ParamT angle) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        static_cast<void>(angle);
        return {"applyInternalInternal", rev_wire0, rev_wire1, inverse};
    }

    template <size_t rev_wire0, typename ParamT>
    static std::tuple<std::string, size_t, size_t, bool>
    applyInternalExternal(std::complex<PrecisionT> *arr,
                          const size_t num_qubits, size_t rev_wire1,
                          bool inverse, ParamT angle) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        static_cast<void>(angle);
        return {"applyInternalExternal", rev_wire0, rev_wire1, inverse};
    }

    template <typename ParamT>
    static std::tuple<std::string, size_t, size_t, bool>
    applyExternalExternal(std::complex<PrecisionT> *arr,
                          const size_t num_qubits, size_t rev_wire0,
                          size_t rev_wire1, bool inverse, ParamT angle) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        static_cast<void>(angle);
        return {"applyExternalExternal", rev_wire0, rev_wire1, inverse};
    }
};

template <typename PrecisionT, size_t packed_size>
struct MockAsymmetricTwoQubitGateWithoutParam {
    using Precision = PrecisionT;
    constexpr static size_t packed_size_ = packed_size;
    constexpr static bool symmetric = false;

    template <size_t rev_wire0, size_t rev_wire1>
    static std::tuple<std::string, size_t, size_t, bool>
    applyInternalInternal(std::complex<PrecisionT> *arr,
                          const size_t num_qubits, bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        return {"applyInternalInternal", rev_wire0, rev_wire1, inverse};
    }

    template <size_t control>
    static std::tuple<std::string, size_t, size_t, bool>
    applyInternalExternal(std::complex<PrecisionT> *arr,
                          const size_t num_qubits, size_t target,
                          bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        return {"applyInternalExternal", control, target, inverse};
    }

    template <size_t target>
    static std::tuple<std::string, size_t, size_t, bool>
    applyExternalInternal(std::complex<PrecisionT> *arr,
                          const size_t num_qubits, size_t control,
                          bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        return {"applyExternalInternal", control, target, inverse};
    }

    static std::tuple<std::string, size_t, size_t, bool>
    applyExternalExternal(std::complex<PrecisionT> *arr,
                          const size_t num_qubits, size_t rev_wire0,
                          size_t rev_wire1, bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        return {"applyExternalExternal", rev_wire0, rev_wire1, inverse};
    }
};

template <typename PrecisionT, size_t packed_size>
struct MockAsymmetricTwoQubitGateWithParam {
    using Precision = PrecisionT;
    constexpr static size_t packed_size_ = packed_size;
    constexpr static bool symmetric = false;

    template <size_t rev_wire0, size_t rev_wire1, typename ParamT>
    static std::tuple<std::string, size_t, size_t, bool>
    applyInternalInternal(std::complex<PrecisionT> *arr,
                          const size_t num_qubits, bool inverse, ParamT angle) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        static_cast<void>(angle);
        return {"applyInternalInternal", rev_wire0, rev_wire1, inverse};
    }

    template <size_t control, typename ParamT>
    static std::tuple<std::string, size_t, size_t, bool>
    applyInternalExternal(std::complex<PrecisionT> *arr,
                          const size_t num_qubits, size_t target, bool inverse,
                          ParamT angle) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        static_cast<void>(angle);
        return {"applyInternalExternal", control, target, inverse};
    }

    template <size_t target, typename ParamT>
    static std::tuple<std::string, size_t, size_t, bool>
    applyExternalInternal(std::complex<PrecisionT> *arr,
                          const size_t num_qubits, size_t control, bool inverse,
                          ParamT angle) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        static_cast<void>(angle);
        return {"applyExternalInternal", control, target, inverse};
    }

    template <typename ParamT>
    static std::tuple<std::string, size_t, size_t, bool>
    applyExternalExternal(std::complex<PrecisionT> *arr,
                          const size_t num_qubits, size_t rev_wire0,
                          size_t rev_wire1, bool inverse, ParamT angle) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(inverse);
        static_cast<void>(angle);
        return {"applyExternalExternal", rev_wire0, rev_wire1, inverse};
    }
};

TEMPLATE_TEST_CASE("Test TwoQubitGateHelper template functions",
                   "[TwoQubitGateHelper]", float, double) {
    // Template functions detecting existing functions without params
    STATIC_REQUIRE(HasInternalInternalWithoutParam<
                   MockSymmetricTwoQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(HasInternalInternalWithoutParam<
                   MockAsymmetricTwoQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasInternalInternalWithoutParam<
                   MockSymmetricTwoQubitGateWithParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasInternalInternalWithoutParam<
                   MockAsymmetricTwoQubitGateWithParam<TestType, 4>>::value);

    STATIC_REQUIRE(HasInternalExternalWithoutParam<
                   MockSymmetricTwoQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(HasInternalExternalWithoutParam<
                   MockAsymmetricTwoQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasInternalExternalWithoutParam<
                   MockSymmetricTwoQubitGateWithParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasInternalExternalWithoutParam<
                   MockAsymmetricTwoQubitGateWithParam<TestType, 4>>::value);

    STATIC_REQUIRE(!HasExternalInternalWithoutParam<
                   MockSymmetricTwoQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(HasExternalInternalWithoutParam<
                   MockAsymmetricTwoQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasExternalInternalWithoutParam<
                   MockSymmetricTwoQubitGateWithParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasExternalInternalWithoutParam<
                   MockAsymmetricTwoQubitGateWithParam<TestType, 4>>::value);

    STATIC_REQUIRE(HasExternalExternalWithoutParam<
                   MockSymmetricTwoQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(HasExternalExternalWithoutParam<
                   MockAsymmetricTwoQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasExternalExternalWithoutParam<
                   MockSymmetricTwoQubitGateWithParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasExternalExternalWithoutParam<
                   MockAsymmetricTwoQubitGateWithParam<TestType, 4>>::value);

    // Template functions detecting existing functions with params
    STATIC_REQUIRE(!HasInternalInternalWithParam<
                   MockSymmetricTwoQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasInternalInternalWithParam<
                   MockAsymmetricTwoQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(HasInternalInternalWithParam<
                   MockSymmetricTwoQubitGateWithParam<TestType, 4>>::value);
    STATIC_REQUIRE(HasInternalInternalWithParam<
                   MockAsymmetricTwoQubitGateWithParam<TestType, 4>>::value);

    STATIC_REQUIRE(!HasInternalExternalWithParam<
                   MockSymmetricTwoQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasInternalExternalWithParam<
                   MockAsymmetricTwoQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(HasInternalExternalWithParam<
                   MockSymmetricTwoQubitGateWithParam<TestType, 4>>::value);
    STATIC_REQUIRE(HasInternalExternalWithParam<
                   MockAsymmetricTwoQubitGateWithParam<TestType, 4>>::value);

    STATIC_REQUIRE(!HasExternalInternalWithParam<
                   MockSymmetricTwoQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasExternalInternalWithParam<
                   MockAsymmetricTwoQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasExternalInternalWithParam<
                   MockSymmetricTwoQubitGateWithParam<TestType, 4>>::value);
    STATIC_REQUIRE(HasExternalInternalWithParam<
                   MockAsymmetricTwoQubitGateWithParam<TestType, 4>>::value);

    STATIC_REQUIRE(!HasExternalExternalWithParam<
                   MockSymmetricTwoQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(!HasExternalExternalWithParam<
                   MockAsymmetricTwoQubitGateWithoutParam<TestType, 4>>::value);
    STATIC_REQUIRE(HasExternalExternalWithParam<
                   MockSymmetricTwoQubitGateWithParam<TestType, 4>>::value);
    STATIC_REQUIRE(HasExternalExternalWithParam<
                   MockAsymmetricTwoQubitGateWithParam<TestType, 4>>::value);

    // Test concepts
    STATIC_REQUIRE(!SymmetricTwoQubitGateWithParam<
                   MockSymmetricTwoQubitGateWithoutParam<TestType, 4>>);
    STATIC_REQUIRE(!SymmetricTwoQubitGateWithParam<
                   MockAsymmetricTwoQubitGateWithoutParam<TestType, 4>>);
    STATIC_REQUIRE(SymmetricTwoQubitGateWithParam<
                   MockSymmetricTwoQubitGateWithParam<TestType, 4>>);
    STATIC_REQUIRE(!SymmetricTwoQubitGateWithParam<
                   MockAsymmetricTwoQubitGateWithParam<TestType, 4>>);

    STATIC_REQUIRE(!AsymmetricTwoQubitGateWithParam<
                   MockSymmetricTwoQubitGateWithoutParam<TestType, 4>>);
    STATIC_REQUIRE(!AsymmetricTwoQubitGateWithParam<
                   MockAsymmetricTwoQubitGateWithoutParam<TestType, 4>>);
    STATIC_REQUIRE(!AsymmetricTwoQubitGateWithParam<
                   MockSymmetricTwoQubitGateWithParam<TestType, 4>>);
    STATIC_REQUIRE(AsymmetricTwoQubitGateWithParam<
                   MockAsymmetricTwoQubitGateWithParam<TestType, 4>>);

    STATIC_REQUIRE(SymmetricTwoQubitGateWithoutParam<
                   MockSymmetricTwoQubitGateWithoutParam<TestType, 4>>);
    STATIC_REQUIRE(!SymmetricTwoQubitGateWithoutParam<
                   MockAsymmetricTwoQubitGateWithoutParam<TestType, 4>>);
    STATIC_REQUIRE(!SymmetricTwoQubitGateWithoutParam<
                   MockSymmetricTwoQubitGateWithParam<TestType, 4>>);
    STATIC_REQUIRE(!SymmetricTwoQubitGateWithoutParam<
                   MockAsymmetricTwoQubitGateWithParam<TestType, 4>>);

    STATIC_REQUIRE(SymmetricTwoQubitGateWithoutParam<
                   MockSymmetricTwoQubitGateWithoutParam<TestType, 4>>);
    STATIC_REQUIRE(!SymmetricTwoQubitGateWithoutParam<
                   MockAsymmetricTwoQubitGateWithoutParam<TestType, 4>>);
    STATIC_REQUIRE(!SymmetricTwoQubitGateWithoutParam<
                   MockSymmetricTwoQubitGateWithParam<TestType, 4>>);
    STATIC_REQUIRE(!SymmetricTwoQubitGateWithoutParam<
                   MockAsymmetricTwoQubitGateWithParam<TestType, 4>>);
}

std::pair<size_t, size_t> sort(size_t a, size_t b) {
    return {std::min(a, b), std::max(a, b)};
}

TEMPLATE_TEST_CASE("Test TwoQubitGateWithoutParamHelper",
                   "[TwoQubitGateHelper]", float, double) {
    auto fallback =
        [](std::complex<TestType> *arr, size_t num_qubits,
           const std::vector<size_t> &wires,
           bool inverse) -> std::tuple<std::string, size_t, size_t, bool> {
        static_cast<void>(arr);
        return {"fallback", num_qubits - wires[0] - 1,
                num_qubits - wires[1] - 1, inverse};
    };

    SECTION("Test TwoQubitGateWithoutParamHelper for symmetric gates with "
            "packed_size = 8") {
        constexpr size_t packed_size = 8;
        std::vector<std::complex<TestType>> arr(
            16, std::complex<TestType>{0.0, 0.0});
        TwoQubitGateWithoutParamHelper<
            MockSymmetricTwoQubitGateWithoutParam<TestType, packed_size>>
            func(fallback);

        // We pack 8 real numbers -> 4 complex numbers -> two qubits
        // rev_wire in {0, 1} is internal

        for (bool inverse : {false, true}) {
            { // num_qubits = 4, wires = {0, 1} -> rev_wires = {3, 2}
                const auto res = func(arr.data(), 4, {0, 1}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyExternalExternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{2, 3});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {1, 0} -> rev_wires = {2, 3}
                const auto res = func(arr.data(), 4, {1, 0}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyExternalExternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{2, 3});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {0, 3} -> rev_wires = {0, 3}
                const auto res = func(arr.data(), 4, {0, 3}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{0, 3});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {3, 0} -> rev_wires = {3, 0}
                const auto res = func(arr.data(), 4, {3, 0}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{0, 3});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {2, 3} -> rev_wires = {0, 1}
                const auto res = func(arr.data(), 4, {2, 3}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalInternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{0, 1});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {3, 2} -> rev_wires = {1, 0}
                const auto res = func(arr.data(), 4, {3, 2}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalInternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{0, 1});
                REQUIRE(std::get<3>(res) == inverse);
            }
        }
    }

    SECTION("Test TwoQubitGateWithoutParamHelper for symmetric gates with "
            "packed_size = 16") {
        constexpr size_t packed_size = 16;
        std::vector<std::complex<TestType>> arr(
            16, std::complex<TestType>{0.0, 0.0});
        TwoQubitGateWithoutParamHelper<
            MockSymmetricTwoQubitGateWithoutParam<TestType, packed_size>>
            func(fallback);

        // We pack 16 real numbers -> 8 complex numbers -> three qubits
        // rev_wire in {0, 1, 2} is internal

        for (bool inverse : {false, true}) {
            { // num_qubits = 4, wires = {0, 1} -> rev_wires = {2, 3}
                const auto res = func(arr.data(), 4, {0, 1}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{2, 3});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {1, 0} -> rev_wires = {3, 2}
                const auto res = func(arr.data(), 4, {1, 0}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{2, 3});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {0, 3} -> rev_wires = {0, 3}
                const auto res = func(arr.data(), 4, {0, 3}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{0, 3});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {3, 0} -> rev_wires = {3, 0}
                const auto res = func(arr.data(), 4, {3, 0}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{0, 3});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {1, 3} -> rev_wires = {0, 2}
                const auto res = func(arr.data(), 4, {1, 3}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalInternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{0, 2});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {3, 1} -> rev_wires = {0, 2}
                const auto res = func(arr.data(), 4, {3, 1}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalInternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{0, 2});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 2, wires = {0, 1} -> fallback
                const auto res = func(arr.data(), 2, {0, 1}, inverse);
                REQUIRE(std::get<0>(res) == std::string("fallback"));
                REQUIRE(std::get<1>(res) == 1);
                REQUIRE(std::get<2>(res) == 0);
                REQUIRE(std::get<3>(res) == inverse);
            }
        }
    }

    SECTION("Test TwoQubitGateWithoutParamHelper for asymmetric gates with "
            "packed_size = 8") {
        constexpr size_t packed_size = 8;
        std::vector<std::complex<TestType>> arr(
            16, std::complex<TestType>{0.0, 0.0});
        TwoQubitGateWithoutParamHelper<
            MockAsymmetricTwoQubitGateWithoutParam<TestType, packed_size>>
            func(fallback);

        // We pack 8 real numbers -> 4 complex numbers -> two qubits
        // rev_wire in {0, 1} is internal
        // The second wire is the target wire

        for (bool inverse : {false, true}) {
            { // num_qubits = 4, wires = {0, 1} -> rev_wires = {3, 2}
                const auto res = func(arr.data(), 4, {0, 1}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyExternalExternal"));
                REQUIRE(std::get<1>(res) == 3);
                REQUIRE(std::get<2>(res) == 2);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {1, 0} -> rev_wires = {2, 3}
                const auto res = func(arr.data(), 4, {1, 0}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyExternalExternal"));
                REQUIRE(std::get<1>(res) == 2);
                REQUIRE(std::get<2>(res) == 3);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {0, 3} -> rev_wires = {3, 0}
                const auto res = func(arr.data(), 4, {0, 3}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyExternalInternal"));
                REQUIRE(std::get<1>(res) == 3);
                REQUIRE(std::get<2>(res) == 0);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {3, 0} -> rev_wires = {0, 3}
                const auto res = func(arr.data(), 4, {3, 0}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == 3);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {2, 3} -> rev_wires = {1, 0}
                const auto res = func(arr.data(), 4, {2, 3}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalInternal"));
                REQUIRE(std::get<1>(res) == 1);
                REQUIRE(std::get<2>(res) == 0);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {3, 2} -> rev_wires = {0, 1}
                const auto res = func(arr.data(), 4, {3, 2}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalInternal"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == 1);
                REQUIRE(std::get<3>(res) == inverse);
            }
        }
    }

    SECTION("Test TwoQubitGateWithoutParamHelper for asymmetric gates with "
            "packed_size = 16") {
        constexpr size_t packed_size = 16;
        std::vector<std::complex<TestType>> arr(
            16, std::complex<TestType>{0.0, 0.0});
        TwoQubitGateWithoutParamHelper<
            MockAsymmetricTwoQubitGateWithoutParam<TestType, packed_size>>
            func(fallback);

        // We pack 16 real numbers -> 8 complex numbers -> three qubits
        // rev_wire in {0, 1, 2} is internal
        // The second wire is the target wire

        for (bool inverse : {false, true}) {
            { // num_qubits = 4, wires = {0, 1} -> rev_wires = {3, 2}
                const auto res = func(arr.data(), 4, {0, 1}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyExternalInternal"));
                REQUIRE(std::get<1>(res) == 3);
                REQUIRE(std::get<2>(res) == 2);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {1, 0} -> rev_wires = {2, 3}
                const auto res = func(arr.data(), 4, {1, 0}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(std::get<1>(res) == 2);
                REQUIRE(std::get<2>(res) == 3);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {0, 3} -> rev_wires = {3, 0}
                const auto res = func(arr.data(), 4, {0, 3}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyExternalInternal"));
                REQUIRE(std::get<1>(res) == 3);
                REQUIRE(std::get<2>(res) == 0);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {3, 0} -> rev_wires = {0, 3}
                const auto res = func(arr.data(), 4, {3, 0}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == 3);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {2, 3} -> rev_wires = {1, 0}
                const auto res = func(arr.data(), 4, {2, 3}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalInternal"));
                REQUIRE(std::get<1>(res) == 1);
                REQUIRE(std::get<2>(res) == 0);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {3, 2} -> rev_wires = {0, 1}
                const auto res = func(arr.data(), 4, {3, 2}, inverse);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalInternal"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == 1);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 2, wires = {0, 1} -> fallback
                const auto res = func(arr.data(), 2, {0, 1}, inverse);
                REQUIRE(std::get<0>(res) == std::string("fallback"));
                REQUIRE(std::get<1>(res) == 1);
                REQUIRE(std::get<2>(res) == 0);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 2, wires = {1, 0} -> fallback
                const auto res = func(arr.data(), 2, {1, 0}, inverse);
                REQUIRE(std::get<0>(res) == std::string("fallback"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == 1);
                REQUIRE(std::get<3>(res) == inverse);
            }
        }
    }
}

TEMPLATE_TEST_CASE("Test TwoQubitGateWithParamHelper", "[TwoQubitGateHelper]",
                   float, double) {
    auto fallback =
        [](std::complex<TestType> *arr, size_t num_qubits,
           const std::vector<size_t> &wires, bool inverse,
           TestType angle) -> std::tuple<std::string, size_t, size_t, bool> {
        static_cast<void>(arr);
        static_cast<void>(angle);
        return {"fallback", num_qubits - wires[0] - 1,
                num_qubits - wires[1] - 1, inverse};
    };

    const auto angle = static_cast<TestType>(0.312);

    SECTION("Test TwoQubitGateWithParamHelper for symmetric gates with "
            "packed_size = 8") {
        constexpr size_t packed_size = 8;
        std::vector<std::complex<TestType>> arr(
            16, std::complex<TestType>{0.0, 0.0});
        TwoQubitGateWithParamHelper<
            MockSymmetricTwoQubitGateWithParam<TestType, packed_size>, TestType>
            func(fallback);

        // We pack 8 real numbers -> 4 complex numbers -> two qubits
        // rev_wire in {0, 1} is internal

        for (bool inverse : {false, true}) {
            { // num_qubits = 4, wires = {0, 1} -> rev_wires = {3, 2}
                const auto res = func(arr.data(), 4, {0, 1}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyExternalExternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{2, 3});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {1, 0} -> rev_wires = {2, 3}
                const auto res = func(arr.data(), 4, {1, 0}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyExternalExternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{2, 3});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {0, 3} -> rev_wires = {0, 3}
                const auto res = func(arr.data(), 4, {0, 3}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{0, 3});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {3, 0} -> rev_wires = {3, 0}
                const auto res = func(arr.data(), 4, {3, 0}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{0, 3});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {2, 3} -> rev_wires = {0, 1}
                const auto res = func(arr.data(), 4, {2, 3}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalInternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{0, 1});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {3, 2} -> rev_wires = {1, 0}
                const auto res = func(arr.data(), 4, {3, 2}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalInternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{0, 1});
                REQUIRE(std::get<3>(res) == inverse);
            }
        }
    }

    SECTION("Test TwoQubitGateWithParamHelper for symmetric gates with "
            "packed_size = 16") {
        constexpr size_t packed_size = 16;
        std::vector<std::complex<TestType>> arr(
            16, std::complex<TestType>{0.0, 0.0});
        TwoQubitGateWithParamHelper<
            MockSymmetricTwoQubitGateWithParam<TestType, packed_size>, TestType>
            func(fallback);

        // We pack 16 real numbers -> 8 complex numbers -> three qubits
        // rev_wire in {0, 1, 2} is internal

        for (bool inverse : {false, true}) {
            { // num_qubits = 4, wires = {0, 1} -> rev_wires = {2, 3}
                const auto res = func(arr.data(), 4, {0, 1}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{2, 3});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {1, 0} -> rev_wires = {3, 2}
                const auto res = func(arr.data(), 4, {1, 0}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{2, 3});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {0, 3} -> rev_wires = {0, 3}
                const auto res = func(arr.data(), 4, {0, 3}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{0, 3});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {3, 0} -> rev_wires = {3, 0}
                const auto res = func(arr.data(), 4, {3, 0}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{0, 3});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {1, 3} -> rev_wires = {0, 2}
                const auto res = func(arr.data(), 4, {1, 3}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalInternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{0, 2});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {3, 1} -> rev_wires = {0, 2}
                const auto res = func(arr.data(), 4, {3, 1}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalInternal"));
                REQUIRE(sort(std::get<1>(res), std::get<2>(res)) ==
                        std::pair<size_t, size_t>{0, 2});
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 2, wires = {0, 1} -> fallback
                const auto res = func(arr.data(), 2, {0, 1}, inverse, angle);
                REQUIRE(std::get<0>(res) == std::string("fallback"));
                REQUIRE(std::get<1>(res) == 1);
                REQUIRE(std::get<2>(res) == 0);
                REQUIRE(std::get<3>(res) == inverse);
            }
        }
    }

    SECTION("Test TwoQubitGateWithParamHelper for asymmetric gates with "
            "packed_size = 8") {
        constexpr size_t packed_size = 8;
        std::vector<std::complex<TestType>> arr(
            16, std::complex<TestType>{0.0, 0.0});
        TwoQubitGateWithParamHelper<
            MockAsymmetricTwoQubitGateWithParam<TestType, packed_size>,
            TestType>
            func(fallback);

        // We pack 8 real numbers -> 4 complex numbers -> two qubits
        // rev_wire in {0, 1} is internal
        // The second wire is the target wire

        for (bool inverse : {false, true}) {
            { // num_qubits = 4, wires = {0, 1} -> rev_wires = {3, 2}
                const auto res = func(arr.data(), 4, {0, 1}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyExternalExternal"));
                REQUIRE(std::get<1>(res) == 3);
                REQUIRE(std::get<2>(res) == 2);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {1, 0} -> rev_wires = {2, 3}
                const auto res = func(arr.data(), 4, {1, 0}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyExternalExternal"));
                REQUIRE(std::get<1>(res) == 2);
                REQUIRE(std::get<2>(res) == 3);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {0, 3} -> rev_wires = {3, 0}
                const auto res = func(arr.data(), 4, {0, 3}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyExternalInternal"));
                REQUIRE(std::get<1>(res) == 3);
                REQUIRE(std::get<2>(res) == 0);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {3, 0} -> rev_wires = {0, 3}
                const auto res = func(arr.data(), 4, {3, 0}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == 3);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {2, 3} -> rev_wires = {1, 0}
                const auto res = func(arr.data(), 4, {2, 3}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalInternal"));
                REQUIRE(std::get<1>(res) == 1);
                REQUIRE(std::get<2>(res) == 0);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {3, 2} -> rev_wires = {0, 1}
                const auto res = func(arr.data(), 4, {3, 2}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalInternal"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == 1);
                REQUIRE(std::get<3>(res) == inverse);
            }
        }
    }

    SECTION("Test TwoQubitGateWithParamHelper for asymmetric gates with "
            "packed_size = 16") {
        constexpr size_t packed_size = 16;
        std::vector<std::complex<TestType>> arr(
            16, std::complex<TestType>{0.0, 0.0});
        TwoQubitGateWithParamHelper<
            MockAsymmetricTwoQubitGateWithParam<TestType, packed_size>,
            TestType>
            func(fallback);

        // We pack 16 real numbers -> 8 complex numbers -> three qubits
        // rev_wire in {0, 1, 2} is internal
        // The second wire is the target wire

        for (bool inverse : {false, true}) {
            { // num_qubits = 4, wires = {0, 1} -> rev_wires = {3, 2}
                const auto res = func(arr.data(), 4, {0, 1}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyExternalInternal"));
                REQUIRE(std::get<1>(res) == 3);
                REQUIRE(std::get<2>(res) == 2);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {1, 0} -> rev_wires = {2, 3}
                const auto res = func(arr.data(), 4, {1, 0}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(std::get<1>(res) == 2);
                REQUIRE(std::get<2>(res) == 3);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {0, 3} -> rev_wires = {3, 0}
                const auto res = func(arr.data(), 4, {0, 3}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyExternalInternal"));
                REQUIRE(std::get<1>(res) == 3);
                REQUIRE(std::get<2>(res) == 0);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {3, 0} -> rev_wires = {0, 3}
                const auto res = func(arr.data(), 4, {3, 0}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalExternal"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == 3);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {2, 3} -> rev_wires = {1, 0}
                const auto res = func(arr.data(), 4, {2, 3}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalInternal"));
                REQUIRE(std::get<1>(res) == 1);
                REQUIRE(std::get<2>(res) == 0);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 4, wires = {3, 2} -> rev_wires = {0, 1}
                const auto res = func(arr.data(), 4, {3, 2}, inverse, angle);
                REQUIRE(std::get<0>(res) ==
                        std::string("applyInternalInternal"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == 1);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 2, wires = {0, 1} -> fallback
                const auto res = func(arr.data(), 2, {0, 1}, inverse, angle);
                REQUIRE(std::get<0>(res) == std::string("fallback"));
                REQUIRE(std::get<1>(res) == 1);
                REQUIRE(std::get<2>(res) == 0);
                REQUIRE(std::get<3>(res) == inverse);
            }
            { // num_qubits = 2, wires = {1, 0} -> fallback
                const auto res = func(arr.data(), 2, {1, 0}, inverse, angle);
                REQUIRE(std::get<0>(res) == std::string("fallback"));
                REQUIRE(std::get<1>(res) == 0);
                REQUIRE(std::get<2>(res) == 1);
                REQUIRE(std::get<3>(res) == inverse);
            }
        }
    }
}
