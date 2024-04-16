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
 * Defines template classes to extract member function pointers for
 * metaprogramming. Also defines some utility functions that calls such
 * pointers.
 */

#pragma once
#include <complex>
#include <limits>
#include <vector>

#include "Error.hpp"
#include "GateOperation.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Gates;
using Pennylane::Gates::GateOperation;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Gates {
/**
 * @brief Return a specific member function pointer for a given gate operation.
 * See specialized classes.
 */
template <class PrecisionT, class ParamT, class GateImplementation,
          GateOperation gate_op>
struct GateOpToMemberFuncPtr {
    // raises compile error when this struct is instantiated.
    static_assert(sizeof(PrecisionT) == std::numeric_limits<size_t>::max(),
                  "GateOpToMemberFuncPtr is not defined for the given gate. "
                  "When you define a new GateOperation, check that you also "
                  "have added the corresponding entry in "
                  "GateOpToMemberFuncPtr.");
    constexpr static auto value = nullptr;
};

template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::Identity> {
    constexpr static auto value =
        &GateImplementation::template applyIdentity<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::PauliX> {
    constexpr static auto value =
        &GateImplementation::template applyPauliX<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::PauliY> {
    constexpr static auto value =
        &GateImplementation::template applyPauliY<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::PauliZ> {
    constexpr static auto value =
        &GateImplementation::template applyPauliZ<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::Hadamard> {
    constexpr static auto value =
        &GateImplementation::template applyHadamard<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::S> {
    constexpr static auto value =
        &GateImplementation::template applyS<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::T> {
    constexpr static auto value =
        &GateImplementation::template applyT<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::PhaseShift> {
    constexpr static auto value =
        &GateImplementation::template applyPhaseShift<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::RX> {
    constexpr static auto value =
        &GateImplementation::template applyRX<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::RY> {
    constexpr static auto value =
        &GateImplementation::template applyRY<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::RZ> {
    constexpr static auto value =
        &GateImplementation::template applyRZ<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::Rot> {
    constexpr static auto value =
        &GateImplementation::template applyRot<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::CNOT> {
    constexpr static auto value =
        &GateImplementation::template applyCNOT<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::CY> {
    constexpr static auto value =
        &GateImplementation::template applyCY<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::CZ> {
    constexpr static auto value =
        &GateImplementation::template applyCZ<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::SWAP> {
    constexpr static auto value =
        &GateImplementation::template applySWAP<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::IsingXX> {
    constexpr static auto value =
        &GateImplementation::template applyIsingXX<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::IsingXY> {
    constexpr static auto value =
        &GateImplementation::template applyIsingXY<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::IsingYY> {
    constexpr static auto value =
        &GateImplementation::template applyIsingYY<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::IsingZZ> {
    constexpr static auto value =
        &GateImplementation::template applyIsingZZ<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::ControlledPhaseShift> {
    constexpr static auto value =
        &GateImplementation::template applyControlledPhaseShift<PrecisionT,
                                                                ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::CRX> {
    constexpr static auto value =
        &GateImplementation::template applyCRX<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::CRY> {
    constexpr static auto value =
        &GateImplementation::template applyCRY<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::CRZ> {
    constexpr static auto value =
        &GateImplementation::template applyCRZ<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::CRot> {
    constexpr static auto value =
        &GateImplementation::template applyCRot<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::SingleExcitation> {
    constexpr static auto value =
        &GateImplementation::template applySingleExcitation<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::SingleExcitationMinus> {
    constexpr static auto value =
        &GateImplementation::template applySingleExcitationMinus<PrecisionT,
                                                                 ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::SingleExcitationPlus> {
    constexpr static auto value =
        &GateImplementation::template applySingleExcitationPlus<PrecisionT,
                                                                ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::Toffoli> {
    constexpr static auto value =
        &GateImplementation::template applyToffoli<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::CSWAP> {
    constexpr static auto value =
        &GateImplementation::template applyCSWAP<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::DoubleExcitation> {
    constexpr static auto value =
        &GateImplementation::template applyDoubleExcitation<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::DoubleExcitationMinus> {
    constexpr static auto value =
        &GateImplementation::template applyDoubleExcitationMinus<PrecisionT,
                                                                 ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::DoubleExcitationPlus> {
    constexpr static auto value =
        &GateImplementation::template applyDoubleExcitationPlus<PrecisionT,
                                                                ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::MultiRZ> {
    constexpr static auto value =
        &GateImplementation::template applyMultiRZ<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                             GateOperation::GlobalPhase> {
    constexpr static auto value =
        &GateImplementation::template applyGlobalPhase<PrecisionT, ParamT>;
};

template <class PrecisionT, class ParamT, class GateImplementation,
          ControlledGateOperation gate_op>
struct ControlledGateOpToMemberFuncPtr {
    static_assert(sizeof(PrecisionT) == std::numeric_limits<size_t>::max(),
                  "Unrecognized matrix operation");
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::PauliX> {
    constexpr static auto value =
        &GateImplementation::template applyNCPauliX<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::PauliY> {
    constexpr static auto value =
        &GateImplementation::template applyNCPauliY<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::PauliZ> {
    constexpr static auto value =
        &GateImplementation::template applyNCPauliZ<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::Hadamard> {
    constexpr static auto value =
        &GateImplementation::template applyNCHadamard<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::S> {
    constexpr static auto value =
        &GateImplementation::template applyNCS<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::T> {
    constexpr static auto value =
        &GateImplementation::template applyNCT<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::PhaseShift> {
    constexpr static auto value =
        &GateImplementation::template applyNCPhaseShift<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::RX> {
    constexpr static auto value =
        &GateImplementation::template applyNCRX<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::RY> {
    constexpr static auto value =
        &GateImplementation::template applyNCRY<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::RZ> {
    constexpr static auto value =
        &GateImplementation::template applyNCRZ<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::Rot> {
    constexpr static auto value =
        &GateImplementation::template applyNCRot<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::SWAP> {
    constexpr static auto value =
        &GateImplementation::template applyNCSWAP<PrecisionT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::IsingXX> {
    constexpr static auto value =
        &GateImplementation::template applyNCIsingXX<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::IsingXY> {
    constexpr static auto value =
        &GateImplementation::template applyNCIsingXY<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::IsingYY> {
    constexpr static auto value =
        &GateImplementation::template applyNCIsingYY<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::IsingZZ> {
    constexpr static auto value =
        &GateImplementation::template applyNCIsingZZ<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<
    PrecisionT, ParamT, GateImplementation,
    ControlledGateOperation::SingleExcitation> {
    constexpr static auto value =
        &GateImplementation::template applyNCSingleExcitation<PrecisionT,
                                                              ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<
    PrecisionT, ParamT, GateImplementation,
    ControlledGateOperation::SingleExcitationMinus> {
    constexpr static auto value =
        &GateImplementation::template applyNCSingleExcitationMinus<PrecisionT,
                                                                   ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<
    PrecisionT, ParamT, GateImplementation,
    ControlledGateOperation::SingleExcitationPlus> {
    constexpr static auto value =
        &GateImplementation::template applyNCSingleExcitationPlus<PrecisionT,
                                                                  ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<
    PrecisionT, ParamT, GateImplementation,
    ControlledGateOperation::DoubleExcitation> {
    constexpr static auto value =
        &GateImplementation::template applyNCDoubleExcitation<PrecisionT,
                                                              ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<
    PrecisionT, ParamT, GateImplementation,
    ControlledGateOperation::DoubleExcitationMinus> {
    constexpr static auto value =
        &GateImplementation::template applyNCDoubleExcitationMinus<PrecisionT,
                                                                   ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<
    PrecisionT, ParamT, GateImplementation,
    ControlledGateOperation::DoubleExcitationPlus> {
    constexpr static auto value =
        &GateImplementation::template applyNCDoubleExcitationPlus<PrecisionT,
                                                                  ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::MultiRZ> {
    constexpr static auto value =
        &GateImplementation::template applyNCMultiRZ<PrecisionT, ParamT>;
};
template <class PrecisionT, class ParamT, class GateImplementation>
struct ControlledGateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                       ControlledGateOperation::GlobalPhase> {
    constexpr static auto value =
        &GateImplementation::template applyNCGlobalPhase<PrecisionT, ParamT>;
};

/**
 * @brief Return a specific member function pointer for a given generator
 * operation. See specialized classes.
 */
template <class PrecisionT, class GateImplementation,
          GeneratorOperation gntr_op>
struct GeneratorOpToMemberFuncPtr {
    // raises compile error when this struct is instantiated.
    static_assert(
        sizeof(GateImplementation) == std::numeric_limits<size_t>::max(),
        "GeneratorOpToMemberFuncPtr is not defined for the given generator. "
        "When you define a new GeneratorOperation, check that you also "
        "have added the corresponding entry in GeneratorOpToMemberFuncPtr.");
};

template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::RX> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorRX<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::RY> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorRY<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::RZ> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorRZ<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::PhaseShift> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorPhaseShift<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::IsingXX> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorIsingXX<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::IsingXY> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorIsingXY<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::IsingYY> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorIsingYY<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::IsingZZ> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorIsingZZ<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::CRX> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorCRX<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::CRY> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorCRY<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::CRZ> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorCRZ<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::ControlledPhaseShift> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorControlledPhaseShift<
            PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::SingleExcitation> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorSingleExcitation<
            PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::SingleExcitationMinus> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorSingleExcitationMinus<
            PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::SingleExcitationPlus> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorSingleExcitationPlus<
            PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::DoubleExcitation> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorDoubleExcitation<
            PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::DoubleExcitationMinus> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorDoubleExcitationMinus<
            PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::DoubleExcitationPlus> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorDoubleExcitationPlus<
            PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::MultiRZ> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorMultiRZ<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct GeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                  GeneratorOperation::GlobalPhase> {
    constexpr static auto value =
        &GateImplementation::template applyGeneratorGlobalPhase<PrecisionT>;
};

template <class PrecisionT, class GateImplementation,
          ControlledGeneratorOperation mat_op>
struct ControlledGeneratorOpToMemberFuncPtr {
    static_assert(sizeof(PrecisionT) == std::numeric_limits<size_t>::max(),
                  "Unrecognized generator operation");
};
template <class PrecisionT, class GateImplementation>
struct ControlledGeneratorOpToMemberFuncPtr<
    PrecisionT, GateImplementation, ControlledGeneratorOperation::PhaseShift> {
    constexpr static auto value =
        &GateImplementation::template applyNCGeneratorPhaseShift<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct ControlledGeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                            ControlledGeneratorOperation::RX> {
    constexpr static auto value =
        &GateImplementation::template applyNCGeneratorRX<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct ControlledGeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                            ControlledGeneratorOperation::RY> {
    constexpr static auto value =
        &GateImplementation::template applyNCGeneratorRY<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct ControlledGeneratorOpToMemberFuncPtr<PrecisionT, GateImplementation,
                                            ControlledGeneratorOperation::RZ> {
    constexpr static auto value =
        &GateImplementation::template applyNCGeneratorRZ<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct ControlledGeneratorOpToMemberFuncPtr<
    PrecisionT, GateImplementation, ControlledGeneratorOperation::IsingXX> {
    constexpr static auto value =
        &GateImplementation::template applyNCGeneratorIsingXX<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct ControlledGeneratorOpToMemberFuncPtr<
    PrecisionT, GateImplementation, ControlledGeneratorOperation::IsingXY> {
    constexpr static auto value =
        &GateImplementation::template applyNCGeneratorIsingXY<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct ControlledGeneratorOpToMemberFuncPtr<
    PrecisionT, GateImplementation, ControlledGeneratorOperation::IsingYY> {
    constexpr static auto value =
        &GateImplementation::template applyNCGeneratorIsingYY<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct ControlledGeneratorOpToMemberFuncPtr<
    PrecisionT, GateImplementation, ControlledGeneratorOperation::IsingZZ> {
    constexpr static auto value =
        &GateImplementation::template applyNCGeneratorIsingZZ<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct ControlledGeneratorOpToMemberFuncPtr<
    PrecisionT, GateImplementation,
    ControlledGeneratorOperation::SingleExcitation> {
    constexpr static auto value =
        &GateImplementation::template applyNCGeneratorSingleExcitation<
            PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct ControlledGeneratorOpToMemberFuncPtr<
    PrecisionT, GateImplementation,
    ControlledGeneratorOperation::SingleExcitationMinus> {
    constexpr static auto value =
        &GateImplementation::template applyNCGeneratorSingleExcitationMinus<
            PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct ControlledGeneratorOpToMemberFuncPtr<
    PrecisionT, GateImplementation,
    ControlledGeneratorOperation::SingleExcitationPlus> {
    constexpr static auto value =
        &GateImplementation::template applyNCGeneratorSingleExcitationPlus<
            PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct ControlledGeneratorOpToMemberFuncPtr<
    PrecisionT, GateImplementation,
    ControlledGeneratorOperation::DoubleExcitation> {
    constexpr static auto value =
        &GateImplementation::template applyNCGeneratorDoubleExcitation<
            PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct ControlledGeneratorOpToMemberFuncPtr<
    PrecisionT, GateImplementation,
    ControlledGeneratorOperation::DoubleExcitationMinus> {
    constexpr static auto value =
        &GateImplementation::template applyNCGeneratorDoubleExcitationMinus<
            PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct ControlledGeneratorOpToMemberFuncPtr<
    PrecisionT, GateImplementation,
    ControlledGeneratorOperation::DoubleExcitationPlus> {
    constexpr static auto value =
        &GateImplementation::template applyNCGeneratorDoubleExcitationPlus<
            PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct ControlledGeneratorOpToMemberFuncPtr<
    PrecisionT, GateImplementation, ControlledGeneratorOperation::MultiRZ> {
    constexpr static auto value =
        &GateImplementation::template applyNCGeneratorMultiRZ<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct ControlledGeneratorOpToMemberFuncPtr<
    PrecisionT, GateImplementation, ControlledGeneratorOperation::GlobalPhase> {
    constexpr static auto value =
        &GateImplementation::template applyNCGeneratorGlobalPhase<PrecisionT>;
};

/**
 * @brief Matrix operation to member function pointer
 */
template <class PrecisionT, class GateImplementation, MatrixOperation mat_op>
struct MatrixOpToMemberFuncPtr {
    static_assert(sizeof(PrecisionT) == std::numeric_limits<size_t>::max(),
                  "Unrecognized matrix operation");
};

template <class PrecisionT, class GateImplementation>
struct MatrixOpToMemberFuncPtr<PrecisionT, GateImplementation,
                               MatrixOperation::SingleQubitOp> {
    constexpr static auto value =
        &GateImplementation::template applySingleQubitOp<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct MatrixOpToMemberFuncPtr<PrecisionT, GateImplementation,
                               MatrixOperation::TwoQubitOp> {
    constexpr static auto value =
        &GateImplementation::template applyTwoQubitOp<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct MatrixOpToMemberFuncPtr<PrecisionT, GateImplementation,
                               MatrixOperation::MultiQubitOp> {
    constexpr static auto value =
        &GateImplementation::template applyMultiQubitOp<PrecisionT>;
};

template <class PrecisionT, class GateImplementation,
          ControlledMatrixOperation mat_op>
struct ControlledMatrixOpToMemberFuncPtr {
    static_assert(sizeof(PrecisionT) == std::numeric_limits<size_t>::max(),
                  "Unrecognized matrix operation");
};
template <class PrecisionT, class GateImplementation>
struct ControlledMatrixOpToMemberFuncPtr<
    PrecisionT, GateImplementation,
    ControlledMatrixOperation::NCSingleQubitOp> {
    constexpr static auto value =
        &GateImplementation::template applyNCSingleQubitOp<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct ControlledMatrixOpToMemberFuncPtr<
    PrecisionT, GateImplementation, ControlledMatrixOperation::NCTwoQubitOp> {
    constexpr static auto value =
        &GateImplementation::template applyNCTwoQubitOp<PrecisionT>;
};
template <class PrecisionT, class GateImplementation>
struct ControlledMatrixOpToMemberFuncPtr<
    PrecisionT, GateImplementation, ControlledMatrixOperation::NCMultiQubitOp> {
    constexpr static auto value =
        &GateImplementation::template applyNCMultiQubitOp<PrecisionT>;
};

/// @cond DEV
namespace Internal {
/**
 * @brief Gate operation pointer type for a statevector. See all specialized
 * types.
 */
template <class SVType, class ParamT, size_t num_params> struct GateMemFuncPtr {
    static_assert(num_params < 2 || num_params == 3,
                  "The given num_params is not supported.");
};
/**
 * @brief Function pointer type for a gate operation without parameters.
 */
template <class SVType, class ParamT> struct GateMemFuncPtr<SVType, ParamT, 0> {
    using Type = void (SVType::*)(const std::vector<size_t> &, bool);
};
/**
 * @brief Function pointer type for a gate operation with a single parameter.
 */
template <class SVType, class ParamT> struct GateMemFuncPtr<SVType, ParamT, 1> {
    using Type = void (SVType::*)(const std::vector<size_t> &, bool, ParamT);
};
/**
 * @brief Function pointer type for a gate operation with three parameters.
 */
template <class SVType, class ParamT> struct GateMemFuncPtr<SVType, ParamT, 3> {
    using Type = void (SVType::*)(const std::vector<size_t> &, bool, ParamT,
                                  ParamT, ParamT);
};

/**
 * @brief A convenient alias for GateMemFuncPtr.
 */
template <class SVType, class ParamT, size_t num_params>
using GateMemFuncPtrT =
    typename GateMemFuncPtr<SVType, ParamT, num_params>::Type;

/**
 * @brief Gate operation pointer type. See all specialized types.
 */
template <class PrecisionT, class ParamT, size_t num_params>
struct GateFuncPtr {
    static_assert(num_params < 2 || num_params == 3,
                  "The given num_params is not supported.");
};

/**
 * @brief Pointer type for a gate operation without parameters.
 */
template <class PrecisionT, class ParamT>
struct GateFuncPtr<PrecisionT, ParamT, 0> {
    using Type = void (*)(std::complex<PrecisionT> *, size_t,
                          const std::vector<size_t> &, bool);
};
/**
 * @brief Pointer type for a gate operation with a single parameter
 */
template <class PrecisionT, class ParamT>
struct GateFuncPtr<PrecisionT, ParamT, 1> {
    using Type = void (*)(std::complex<PrecisionT> *, size_t,
                          const std::vector<size_t> &, bool, ParamT);
};
/**
 * @brief Pointer type for a gate operation with three parameters
 */
template <class PrecisionT, class ParamT>
struct GateFuncPtr<PrecisionT, ParamT, 3> {
    using Type = void (*)(std::complex<PrecisionT> *, size_t,
                          const std::vector<size_t> &, bool, ParamT, ParamT,
                          ParamT);
};

/**
 * @brief Pointer type for a controlled gate operation
 */
template <class PrecisionT, class ParamT, size_t num_params>
struct ControlledGateFuncPtr {
    static_assert(num_params < 2 || num_params == 3,
                  "The given num_params is not supported.");
};
template <class PrecisionT, class ParamT>
struct ControlledGateFuncPtr<PrecisionT, ParamT, 0> {
    using Type = void (*)(std::complex<PrecisionT> *, size_t,
                          const std::vector<size_t> &,
                          const std::vector<bool> &,
                          const std::vector<size_t> &, bool);
};
template <class PrecisionT, class ParamT>
struct ControlledGateFuncPtr<PrecisionT, ParamT, 1> {
    using Type = void (*)(std::complex<PrecisionT> *, size_t,
                          const std::vector<size_t> &,
                          const std::vector<bool> &,
                          const std::vector<size_t> &, bool, ParamT);
};
template <class PrecisionT, class ParamT>
struct ControlledGateFuncPtr<PrecisionT, ParamT, 3> {
    using Type = void (*)(std::complex<PrecisionT> *, size_t,
                          const std::vector<size_t> &,
                          const std::vector<bool> &,
                          const std::vector<size_t> &, bool, ParamT, ParamT,
                          ParamT);
};

/**
 * @brief Pointer type for a generator operation
 */
template <class PrecisionT> struct GeneratorFuncPtr {
    using Type = PrecisionT (*)(std::complex<PrecisionT> *, size_t,
                                const std::vector<size_t> &, bool);
};

/**
 * @brief Pointer type for a controlled generator operation
 */
template <class PrecisionT> struct ControlledGeneratorFuncPtr {
    using Type = PrecisionT (*)(std::complex<PrecisionT> *, size_t,
                                const std::vector<size_t> &,
                                const std::vector<bool> &,
                                const std::vector<size_t> &, bool);
};

/**
 * @brief Pointer type for a matrix operation
 */
template <class PrecisionT> struct MatrixFuncPtr {
    using Type = void (*)(std::complex<PrecisionT> *, size_t,
                          const std::complex<PrecisionT> *,
                          const std::vector<size_t> &, bool);
};

/**
 * @brief Pointer type for a controlled matrix operation
 */
template <class PrecisionT> struct ControlledMatrixFuncPtr {
    using Type = void (*)(std::complex<PrecisionT> *, size_t,
                          const std::complex<PrecisionT> *,
                          const std::vector<size_t> &,
                          const std::vector<bool> &,
                          const std::vector<size_t> &, bool);
};

} // namespace Internal
/// @endcond

/**
 * @brief Convenient type alias for GateFuncPtr.
 */
template <class PrecisionT, class ParamT, size_t num_params>
using GateFuncPtrT =
    typename Internal::GateFuncPtr<PrecisionT, ParamT, num_params>::Type;

/**
 * @brief Convenient type alias for ControlledGateFuncPtrT.
 */
template <class PrecisionT, class ParamT, size_t num_params>
using ControlledGateFuncPtrT =
    typename Internal::ControlledGateFuncPtr<PrecisionT, ParamT,
                                             num_params>::Type;

/**
 * @brief Convenient type alias for GeneratorFuncPtr.
 */
template <class PrecisionT>
using GeneratorFuncPtrT = typename Internal::GeneratorFuncPtr<PrecisionT>::Type;

/**
 * @brief Convenient type alias for ControlledGeneratorFuncPtr.
 */
template <class PrecisionT>
using ControlledGeneratorFuncPtrT =
    typename Internal::ControlledGeneratorFuncPtr<PrecisionT>::Type;

/**
 * @brief Convenient type alias for MatrixfuncPtr.
 */
template <class PrecisionT>
using MatrixFuncPtrT = typename Internal::MatrixFuncPtr<PrecisionT>::Type;

/**
 * @brief Convenient type alias for ControlledMatrixFuncPtrT.
 */
template <class PrecisionT>
using ControlledMatrixFuncPtrT =
    typename Internal::ControlledMatrixFuncPtr<PrecisionT>::Type;

/**
 * @defgroup Call gate operation with provided arguments
 *
 * @tparam PrecisionT Floating point type for the state-vector.
 * @tparam ParamT Floating point type for the gate parameters.
 * @param func Function pointer for the gate operation.
 * @param data Data pointer the gate is applied to
 * @param num_qubits The number of qubits of the state-vector.
 * @param wires Wires the gate applies to.
 * @param inverse If true, we apply the inverse of the gate.
 * @param params The list of gate parameters.
 */
/// @{
/**
 * @brief Overload for a gate operation without parameters
 */
template <class PrecisionT, class ParamT>
inline void callGateOps(GateFuncPtrT<PrecisionT, ParamT, 0> func,
                        std::complex<PrecisionT> *data, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        [[maybe_unused]] const std::vector<ParamT> &params) {
    PL_ASSERT(params.empty());
    func(data, num_qubits, wires, inverse);
}

/**
 * @brief Overload for a gate operation for a single parameter
 */
template <class PrecisionT, class ParamT>
inline void callGateOps(GateFuncPtrT<PrecisionT, ParamT, 1> func,
                        std::complex<PrecisionT> *data, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        const std::vector<ParamT> &params) {
    PL_ASSERT(params.size() == 1);
    func(data, num_qubits, wires, inverse, params[0]);
}

/**
 * @brief Overload for a gate operation for three parameters
 */
template <class PrecisionT, class ParamT>
inline void callGateOps(GateFuncPtrT<PrecisionT, ParamT, 3> func,
                        std::complex<PrecisionT> *data, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        const std::vector<ParamT> &params) {
    PL_ASSERT(params.size() == 3);
    func(data, num_qubits, wires, inverse, params[0], params[1], params[2]);
}

/**
 * @brief Call a controlled gate operation.
 * @tparam PrecisionT Floating point type for the state-vector.
 */
template <class PrecisionT, class ParamT>
inline void
callControlledGateOps(ControlledGateFuncPtrT<PrecisionT, ParamT, 0> func,
                      std::complex<PrecisionT> *data, size_t num_qubits,
                      const std::vector<size_t> &controlled_wires,
                      const std::vector<bool> &controlled_values,
                      const std::vector<size_t> &wires, bool inverse,
                      [[maybe_unused]] const std::vector<ParamT> &params) {
    PL_ASSERT(params.empty());
    func(data, num_qubits, controlled_wires, controlled_values, wires, inverse);
}

template <class PrecisionT, class ParamT>
inline void
callControlledGateOps(ControlledGateFuncPtrT<PrecisionT, ParamT, 1> func,
                      std::complex<PrecisionT> *data, size_t num_qubits,
                      const std::vector<size_t> &controlled_wires,
                      const std::vector<bool> &controlled_values,
                      const std::vector<size_t> &wires, bool inverse,
                      const std::vector<ParamT> &params) {
    PL_ASSERT(params.size() == 1);
    func(data, num_qubits, controlled_wires, controlled_values, wires, inverse,
         params[0]);
}

template <class PrecisionT, class ParamT>
inline void
callControlledGateOps(ControlledGateFuncPtrT<PrecisionT, ParamT, 3> func,
                      std::complex<PrecisionT> *data, size_t num_qubits,
                      const std::vector<size_t> &controlled_wires,
                      const std::vector<bool> &controlled_values,
                      const std::vector<size_t> &wires, bool inverse,
                      const std::vector<ParamT> &params) {
    PL_ASSERT(params.size() == 3);
    func(data, num_qubits, controlled_wires, controlled_values, wires, inverse,
         params[0], params[1], params[2]);
}

/// @}
/**
 * @brief Call a generator operation.
 *
 * @tparam PrecisionT Floating point type for the state-vector.
 * @return Scaling factor
 */
template <class PrecisionT>
inline PrecisionT callGeneratorOps(GeneratorFuncPtrT<PrecisionT> func,
                                   std::complex<PrecisionT> *data,
                                   size_t num_qubits,
                                   const std::vector<size_t> &wires, bool adj) {
    return func(data, num_qubits, wires, adj);
}

/**
 * @brief Call a controlled generator operation.
 *
 * @tparam PrecisionT Floating point type for the state-vector.
 * @return Scaling factor
 */
template <class PrecisionT>
inline PrecisionT callGeneratorOps(GeneratorFuncPtrT<PrecisionT> func,
                                   std::complex<PrecisionT> *data,
                                   size_t num_qubits,
                                   const std::vector<size_t> &controlled_wires,
                                   const std::vector<size_t> &wires, bool adj) {
    return func(data, num_qubits, controlled_wires, wires, adj);
}

/**
 * @brief Call a matrix operation.
 * @tparam PrecisionT Floating point type for the state-vector.
 */
template <class PrecisionT>
inline void callMatrixOp(MatrixFuncPtrT<PrecisionT> func,
                         std::complex<PrecisionT> *data, size_t num_qubits,
                         const std::complex<PrecisionT *> matrix,
                         const std::vector<size_t> &wires, bool adj) {
    return func(data, num_qubits, matrix, wires, adj);
}

/**
 * @brief Call a controlled matrix operation.
 * @tparam PrecisionT Floating point type for the state-vector.
 */
template <class PrecisionT>
inline void callControlledMatrixOp(ControlledMatrixFuncPtrT<PrecisionT> func,
                                   std::complex<PrecisionT> *data,
                                   size_t num_qubits,
                                   const std::complex<PrecisionT *> matrix,
                                   const std::vector<size_t> &controlled_wires,
                                   const std::vector<bool> &controlled_values,
                                   const std::vector<size_t> &wires, bool adj) {
    return func(data, num_qubits, matrix, controlled_wires, controlled_values,
                wires, adj);
}

} // namespace Pennylane::LightningQubit::Gates
