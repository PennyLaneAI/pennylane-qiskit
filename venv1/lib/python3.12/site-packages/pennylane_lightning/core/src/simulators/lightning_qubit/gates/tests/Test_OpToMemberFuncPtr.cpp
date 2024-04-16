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
#include "Constant.hpp"
#include "ConstantTestHelpers.hpp" // count_unique, first_elems_of, second_elems_of
#include "ConstantUtil.hpp" // tuple_to_array, array_has_elem, prepend_to_tuple, lookup
#include "OpToMemberFuncPtr.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>
#include <tuple>

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::LightningQubit::Gates;
} // namespace
/// @endcond

template <typename EnumClass, uint32_t... I>
constexpr auto
allGateOpsHelper([[maybe_unused]] std::integer_sequence<uint32_t, I...> dummy) {
    return std::make_tuple(static_cast<EnumClass>(I)...);
}

template <typename EnumClass> constexpr auto allGateOps() {
    return Pennylane::Util::tuple_to_array(allGateOpsHelper<EnumClass>(
        std::make_integer_sequence<uint32_t,
                                   static_cast<uint32_t>(EnumClass::END)>{}));
}
template <class PrecisionT, class ParamT, class GateImplementation,
          uint32_t gate_idx>
constexpr bool testAllGatesImplementedIter() {
    if constexpr (gate_idx <
                  static_cast<uint32_t>(Pennylane::Gates::GateOperation::END) -
                      static_cast<uint32_t>(
                          Pennylane::Gates::GateOperation::BEGIN)) {
        constexpr auto gate_op = static_cast<GateOperation>(gate_idx);
        static_cast<void>(
            GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                  gate_op>::value);
        return testAllGatesImplementedIter<PrecisionT, ParamT,
                                           GateImplementation, gate_idx + 1>();
    } else {
        return true;
    }
}
template <class PrecisionT, class ParamT, class GateImplementation>
constexpr bool testAllGatesImplemented() {
    return testAllGatesImplementedIter<PrecisionT, ParamT, GateImplementation,
                                       static_cast<uint32_t>(
                                           GateOperation::BEGIN)>();
}

#define PENNYLANE_TESTS_DEFINE_GATE_OP_PARAM0(GATE_NAME)                       \
    template <class PrecisionT>                                                \
    static void apply##GATE_NAME(                                              \
        std::complex<PrecisionT> *arr, size_t num_qubits,                      \
        const std::vector<size_t> &wires, bool inverse) {                      \
        static_cast<void>(arr);                                                \
        static_cast<void>(num_qubits);                                         \
        static_cast<void>(wires);                                              \
        static_cast<void>(inverse);                                            \
    }
#define PENNYLANE_TESTS_DEFINE_GATE_OP_PARAM1(GATE_NAME)                       \
    template <class PrecisionT, class ParamT>                                  \
    static void apply##GATE_NAME(                                              \
        std::complex<PrecisionT> *arr, size_t num_qubits,                      \
        const std::vector<size_t> &wires, bool inverse, ParamT) {              \
        static_cast<void>(arr);                                                \
        static_cast<void>(num_qubits);                                         \
        static_cast<void>(wires);                                              \
        static_cast<void>(inverse);                                            \
    }
#define PENNYLANE_TESTS_DEFINE_GATE_OP_PARAM3(GATE_NAME)                       \
    template <class PrecisionT, class ParamT>                                  \
    static void apply##GATE_NAME(std::complex<PrecisionT> *arr,                \
                                 size_t num_qubits,                            \
                                 const std::vector<size_t> &wires,             \
                                 bool inverse, ParamT, ParamT, ParamT) {       \
        static_cast<void>(arr);                                                \
        static_cast<void>(num_qubits);                                         \
        static_cast<void>(wires);                                              \
        static_cast<void>(inverse);                                            \
    }
#define PENNYLANE_TESTS_DEFINE_GATE_OP(GATE_NAME, NUM_PARAMS)                  \
    PENNYLANE_TESTS_DEFINE_GATE_OP_PARAM##NUM_PARAMS(GATE_NAME)

#define PENNYLANE_TESTS_DEFINE_GENERATOR_OP(GENERATOR_NAME)                    \
    template <class PrecisionT>                                                \
    static PrecisionT applyGenerator##GENERATOR_NAME(                          \
        std::complex<PrecisionT> *arr, size_t num_qubits,                      \
        const std::vector<size_t> &wires, bool adj) {                          \
        static_cast<void>(arr);                                                \
        static_cast<void>(num_qubits);                                         \
        static_cast<void>(wires);                                              \
        static_cast<void>(adj);                                                \
        return PrecisionT{};                                                   \
    }

/**
 * @brief This class defines all dummy gates and generators to check
 * consistency of OpToMemberFuncPtr
 */
class DummyImplementation {
  public:
    constexpr static std::array implemented_gates = allGateOps<GateOperation>();
    constexpr static std::array implemented_generators =
        allGateOps<GeneratorOperation>();
    constexpr static std::string_view name = "Dummy";

    template <class PrecisionT>
    static void applyMatrix(std::complex<PrecisionT> *arr, size_t num_qubits,
                            const std::complex<PrecisionT> *matrix,
                            const std::vector<size_t> &wires, bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(matrix);
        static_cast<void>(wires);
        static_cast<void>(inverse);
    }

    PENNYLANE_TESTS_DEFINE_GATE_OP(Identity, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(PauliX, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(PauliY, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(PauliZ, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(Hadamard, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(S, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(T, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(PhaseShift, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(RX, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(RY, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(RZ, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(Rot, 3)
    PENNYLANE_TESTS_DEFINE_GATE_OP(CNOT, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(CY, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(CZ, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(SWAP, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(ControlledPhaseShift, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(IsingXX, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(IsingXY, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(IsingYY, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(IsingZZ, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(CRX, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(CRY, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(CRZ, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(SingleExcitation, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(SingleExcitationMinus, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(SingleExcitationPlus, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(CRot, 3)
    PENNYLANE_TESTS_DEFINE_GATE_OP(Toffoli, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(CSWAP, 0)
    PENNYLANE_TESTS_DEFINE_GATE_OP(DoubleExcitation, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(DoubleExcitationMinus, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(DoubleExcitationPlus, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(MultiRZ, 1)
    PENNYLANE_TESTS_DEFINE_GATE_OP(GlobalPhase, 1)

    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(PhaseShift)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(RX)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(RY)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(RZ)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(IsingXX)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(IsingXY)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(IsingYY)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(IsingZZ)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(CRX)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(CRY)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(CRZ)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(ControlledPhaseShift)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(SingleExcitation)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(SingleExcitationMinus)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(SingleExcitationPlus)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(DoubleExcitation)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(DoubleExcitationMinus)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(DoubleExcitationPlus)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(MultiRZ)
    PENNYLANE_TESTS_DEFINE_GENERATOR_OP(GlobalPhase)
};

static_assert(testAllGatesImplemented<float, float, DummyImplementation>(),
              "DummyImplementation must define all gate operations.");

struct ImplementedGates {
    constexpr static auto value = DummyImplementation::implemented_gates;
    constexpr static std::array<GateOperation, 0> ignore_list = {};

    template <typename PrecisionT, typename ParamT, GateOperation op>
    constexpr static auto func_ptr =
        GateOpToMemberFuncPtr<PrecisionT, ParamT, DummyImplementation,
                              op>::value;
};
struct ImplementedGenerators {
    constexpr static auto value = DummyImplementation::implemented_generators;
    constexpr static std::array<GeneratorOperation, 0> ignore_list = {};

    template <typename PrecisionT, typename ParamT, GeneratorOperation op>
    constexpr static auto func_ptr =
        GeneratorOpToMemberFuncPtr<PrecisionT, DummyImplementation, op>::value;
};

template <typename PrecisionT, typename ParamT, class ValueClass, size_t op_idx>
constexpr auto opFuncPtrPairsIter() {
    if constexpr (op_idx < ValueClass::value.size()) {
        constexpr auto op = ValueClass::value[op_idx];
        if constexpr (Pennylane::Util::array_has_elem(ValueClass::ignore_list,
                                                      op)) {
            return opFuncPtrPairsIter<PrecisionT, ParamT, ValueClass,
                                      op_idx + 1>();
        } else {
            const auto elem = std::pair{
                op, ValueClass::template func_ptr<PrecisionT, ParamT, op>};
            return Pennylane::Util::prepend_to_tuple(
                elem, opFuncPtrPairsIter<PrecisionT, ParamT, ValueClass,
                                         op_idx + 1>());
        }
    } else {
        return std::tuple{};
    }
}

/**
 * @brief Pairs of all implemented gate operations and the corresponding
 * function pointers.
 */
template <typename PrecisionT, typename ParamT>
constexpr static auto gate_op_func_ptr_pairs =
    opFuncPtrPairsIter<PrecisionT, ParamT, ImplementedGates,
                       static_cast<size_t>(GateOperation::BEGIN)>();

template <typename PrecisionT, typename ParamT>
constexpr static auto generator_op_func_ptr_pairs =
    opFuncPtrPairsIter<PrecisionT, ParamT, ImplementedGenerators,
                       static_cast<size_t>(GeneratorOperation::BEGIN)>();

template <typename PrecisionT, typename ParamT, size_t num_params,
          size_t tuple_idx>
constexpr auto gateOpFuncPtrPairsWithNumParamsIter() {
    if constexpr (tuple_idx <
                  std::tuple_size_v<
                      decltype(gate_op_func_ptr_pairs<PrecisionT, ParamT>)>) {
        constexpr auto elem =
            std::get<tuple_idx>(gate_op_func_ptr_pairs<PrecisionT, ParamT>);
        if constexpr (Pennylane::Util::lookup(Constant::gate_num_params,
                                              elem.first) == num_params) {
            return Pennylane::Util::prepend_to_tuple(
                elem, gateOpFuncPtrPairsWithNumParamsIter<
                          PrecisionT, ParamT, num_params, tuple_idx + 1>());
        } else {
            return gateOpFuncPtrPairsWithNumParamsIter<
                PrecisionT, ParamT, num_params, tuple_idx + 1>();
        }
    } else {
        return std::tuple{};
    }
}

template <typename PrecisionT, typename ParamT, size_t num_params>
constexpr auto gate_op_func_ptr_with_params = Pennylane::Util::tuple_to_array(
    gateOpFuncPtrPairsWithNumParamsIter<PrecisionT, ParamT, num_params, 0>());

template <typename PrecisionT, typename ParamT>
constexpr auto generator_op_func_ptr = Pennylane::Util::tuple_to_array(
    generator_op_func_ptr_pairs<PrecisionT, PrecisionT>);

template <typename T, typename U, size_t size>
auto testUniqueness(const std::array<std::pair<T, U>, size> &pairs) {
    REQUIRE(Util::count_unique(Util::first_elems_of(pairs)) == pairs.size());
    REQUIRE(Util::count_unique(Util::second_elems_of(pairs)) == pairs.size());
}

TEMPLATE_TEST_CASE("GateOpToMemberFuncPtr", "[GateOpToMemberFuncPtr]", float,
                   double) {
    // TODO: This can be done in compile time
    testUniqueness(gate_op_func_ptr_with_params<TestType, TestType, 0>);
    testUniqueness(gate_op_func_ptr_with_params<TestType, TestType, 1>);
    testUniqueness(gate_op_func_ptr_with_params<TestType, TestType, 3>);
}
TEMPLATE_TEST_CASE("GeneratorOpToMemberFuncPtr", "[GeneratorOpToMemberFuncPtr]",
                   float, double) {
    // TODO: This can be done in compile time
    testUniqueness(generator_op_func_ptr<TestType, TestType>);
}
