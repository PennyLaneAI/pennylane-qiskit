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
 * Defines kernel functions for all AVX
 */
#pragma once
#include "avx_common/AVXGateKernels.hpp"
#include "avx_common/SingleQubitGateHelper.hpp"
#include "avx_common/TwoQubitGateHelper.hpp"

#include "Error.hpp"
#include "GateImplementationsLM.hpp"
#include "GateOperation.hpp"
#include "GatePragmas.hpp"
#include "Gates.hpp"
#include "KernelType.hpp"
#include "LinearAlgebra.hpp"

#include <immintrin.h>

#include <complex>
#include <vector>

/// @cond DEV
namespace {
using Pennylane::Gates::GateOperation;
using Pennylane::Gates::getRot;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Gates {
template <class Derived>
class GateImplementationsAVXCommon
    : public PauliGenerator<GateImplementationsAVXCommon<Derived>> {
  public:
    constexpr static std::array implemented_gates = {
        GateOperation::PauliX,     GateOperation::PauliY,
        GateOperation::PauliZ,     GateOperation::Hadamard,
        GateOperation::S,          GateOperation::T,
        GateOperation::PhaseShift, GateOperation::RX,
        GateOperation::RY,         GateOperation::RZ,
        GateOperation::Rot,        GateOperation::CNOT,
        GateOperation::CZ,         GateOperation::SWAP,
        GateOperation::IsingXX,    GateOperation::IsingYY,
        GateOperation::IsingZZ,    GateOperation::CY,
        GateOperation::IsingXY,    GateOperation::ControlledPhaseShift,
        GateOperation::CRY,        GateOperation::CRZ,
        GateOperation::CRX,
        /* CRot */
    };

    constexpr static std::array implemented_generators = {
        GeneratorOperation::PhaseShift, GeneratorOperation::RX,
        GeneratorOperation::RY,         GeneratorOperation::RZ,
        GeneratorOperation::IsingXX,    GeneratorOperation::IsingYY,
        GeneratorOperation::IsingZZ,
        /* IsingXY, CRX, CRY, CRZ, ControllPhaseShift */
    };

    constexpr static std::array<ControlledGateOperation, 0>
        implemented_controlled_gates{};

    constexpr static std::array<ControlledGeneratorOperation, 0>
        implemented_controlled_generators{};

    constexpr static std::array<ControlledMatrixOperation, 0>
        implemented_controlled_matrices{};

    template <class PrecisionT>
    static void applyPauliX(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        using ApplyPauliXAVX =
            AVXCommon::ApplyPauliX<PrecisionT,
                                   Derived::packed_bytes / sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 1);
        auto helper =
            AVXCommon::SingleQubitGateWithoutParamHelper<ApplyPauliXAVX>(
                &GateImplementationsLM::applyPauliX);
        helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT>
    static void applyPauliY(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        using ApplyPauliYAVX =
            AVXCommon::ApplyPauliY<PrecisionT,
                                   Derived::packed_bytes / sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 1);
        auto helper =
            AVXCommon::SingleQubitGateWithoutParamHelper<ApplyPauliYAVX>(
                &GateImplementationsLM::applyPauliY);
        helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT>
    static void applyPauliZ(std::complex<PrecisionT> *arr,
                            const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        using ApplyPauliZAVX =
            AVXCommon::ApplyPauliZ<PrecisionT,
                                   Derived::packed_bytes / sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 1);
        auto helper =
            AVXCommon::SingleQubitGateWithoutParamHelper<ApplyPauliZAVX>(
                &GateImplementationsLM::applyPauliZ);
        helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT>
    static void applyS(std::complex<PrecisionT> *arr, const size_t num_qubits,
                       const std::vector<size_t> &wires,
                       [[maybe_unused]] bool inverse) {
        using ApplySAVX = AVXCommon::ApplyS<PrecisionT, Derived::packed_bytes /
                                                            sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        auto helper = AVXCommon::SingleQubitGateWithoutParamHelper<ApplySAVX>(
            &GateImplementationsLM::applyS);
        helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT>
    static void applyT(std::complex<PrecisionT> *arr, const size_t num_qubits,
                       const std::vector<size_t> &wires,
                       [[maybe_unused]] bool inverse) {
        using ApplyTAVX = AVXCommon::ApplyT<PrecisionT, Derived::packed_bytes /
                                                            sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 1);
        auto helper = AVXCommon::SingleQubitGateWithoutParamHelper<ApplyTAVX>(
            &GateImplementationsLM::applyT);
        helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyPhaseShift(std::complex<PrecisionT> *arr,
                                const size_t num_qubits,
                                const std::vector<size_t> &wires, bool inverse,
                                ParamT angle) {
        using ApplyPhaseShiftAVX =
            AVXCommon::ApplyPhaseShift<PrecisionT, Derived::packed_bytes /
                                                       sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 1);
        auto helper =
            AVXCommon::SingleQubitGateWithParamHelper<ApplyPhaseShiftAVX,
                                                      ParamT>(
                &GateImplementationsLM::applyPhaseShift);
        helper(arr, num_qubits, wires, inverse, angle);
    }

    template <class PrecisionT>
    static void applyHadamard(std::complex<PrecisionT> *arr,
                              const size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse) {
        using ApplyHadamardAVX =
            AVXCommon::ApplyHadamard<PrecisionT, Derived::packed_bytes /
                                                     sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");
        PL_ASSERT(wires.size() == 1);
        auto helper =
            AVXCommon::SingleQubitGateWithoutParamHelper<ApplyHadamardAVX>(
                &GateImplementationsLM::applyHadamard);
        helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRX(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        using ApplyRXAVX =
            AVXCommon::ApplyRX<PrecisionT,
                               Derived::packed_bytes / sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");
        PL_ASSERT(wires.size() == 1);
        auto helper =
            AVXCommon::SingleQubitGateWithParamHelper<ApplyRXAVX, ParamT>(
                &GateImplementationsLM::applyRX);
        helper(arr, num_qubits, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRY(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        using ApplyRYAVX =
            AVXCommon::ApplyRY<PrecisionT,
                               Derived::packed_bytes / sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");
        PL_ASSERT(wires.size() == 1);
        auto helper =
            AVXCommon::SingleQubitGateWithParamHelper<ApplyRYAVX, ParamT>(
                &GateImplementationsLM::applyRY);
        helper(arr, num_qubits, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRZ(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        using ApplyRZAVX =
            AVXCommon::ApplyRZ<PrecisionT,
                               Derived::packed_bytes / sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");
        PL_ASSERT(wires.size() == 1);
        auto helper =
            AVXCommon::SingleQubitGateWithParamHelper<ApplyRZAVX, ParamT>(
                &GateImplementationsLM::applyRZ);
        helper(arr, num_qubits, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRot(std::complex<PrecisionT> *arr, const size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT phi, ParamT theta, ParamT omega) {
        PL_ASSERT(wires.size() == 1);

        const auto rotMat =
            (inverse) ? getRot<std::complex, PrecisionT>(-omega, -theta, -phi)
                      : getRot<std::complex, PrecisionT>(phi, theta, omega);

        Derived::applySingleQubitOp(arr, num_qubits, rotMat.data(), wires);
    }

    /* Two-qubit gates*/
    template <class PrecisionT>
    static void applyCZ(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse) {
        using ApplyCZAVX =
            AVXCommon::ApplyCZ<PrecisionT,
                               Derived::packed_bytes / sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 2);

        const AVXCommon::TwoQubitGateWithoutParamHelper<ApplyCZAVX> gate_helper(
            &GateImplementationsLM::applyCZ<PrecisionT>);

        gate_helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT>
    static void applyCY(std::complex<PrecisionT> *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse) {
        using ApplyCYAVX =
            AVXCommon::ApplyCY<PrecisionT,
                               Derived::packed_bytes / sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 2);

        const AVXCommon::TwoQubitGateWithoutParamHelper<ApplyCYAVX> gate_helper(
            &GateImplementationsLM::applyCY<PrecisionT>);

        gate_helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT>
    static void
    applySWAP(std::complex<PrecisionT> *arr, const size_t num_qubits,
              const std::vector<size_t> &wires, [[maybe_unused]] bool inverse) {
        using ApplySWAPAVX =
            AVXCommon::ApplySWAP<PrecisionT,
                                 Derived::packed_bytes / sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 2);

        const AVXCommon::TwoQubitGateWithoutParamHelper<ApplySWAPAVX>
            gate_helper(&GateImplementationsLM::applySWAP<PrecisionT>);

        gate_helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT>
    static void
    applyCNOT(std::complex<PrecisionT> *arr, const size_t num_qubits,
              const std::vector<size_t> &wires, [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 2);

        using ApplyCNOTAVX =
            AVXCommon::ApplyCNOT<PrecisionT,
                                 Derived::packed_bytes / sizeof(PrecisionT)>;

        static_assert(
            AVXCommon::AsymmetricTwoQubitGateWithoutParam<ApplyCNOTAVX>);

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 2);

        const AVXCommon::TwoQubitGateWithoutParamHelper<ApplyCNOTAVX>
            gate_helper(&GateImplementationsLM::applyCNOT<PrecisionT>);

        gate_helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingXX(std::complex<PrecisionT> *arr,
                             const size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse, ParamT angle) {
        PL_ASSERT(wires.size() == 2);

        using ApplyIsingXXAVX =
            AVXCommon::ApplyIsingXX<PrecisionT,
                                    Derived::packed_bytes / sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        const AVXCommon::TwoQubitGateWithParamHelper<ApplyIsingXXAVX, ParamT>
            gate_helper(
                &GateImplementationsLM::applyIsingXX<PrecisionT, ParamT>);

        gate_helper(arr, num_qubits, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingXY(std::complex<PrecisionT> *arr,
                             const size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse, ParamT angle) {
        PL_ASSERT(wires.size() == 2);

        using ApplyIsingXYAVX =
            AVXCommon::ApplyIsingXY<PrecisionT,
                                    Derived::packed_bytes / sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        const AVXCommon::TwoQubitGateWithParamHelper<ApplyIsingXYAVX, ParamT>
            gate_helper(
                &GateImplementationsLM::applyIsingXY<PrecisionT, ParamT>);

        gate_helper(arr, num_qubits, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingYY(std::complex<PrecisionT> *arr,
                             const size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse, ParamT angle) {
        PL_ASSERT(wires.size() == 2);

        using ApplyIsingYYAVX =
            AVXCommon::ApplyIsingYY<PrecisionT,
                                    Derived::packed_bytes / sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        const AVXCommon::TwoQubitGateWithParamHelper<ApplyIsingYYAVX, ParamT>
            gate_helper(
                &GateImplementationsLM::applyIsingYY<PrecisionT, ParamT>);

        gate_helper(arr, num_qubits, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingZZ(std::complex<PrecisionT> *arr,
                             const size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse, ParamT angle) {
        using ApplyIsingZZAVX =
            AVXCommon::ApplyIsingZZ<PrecisionT,
                                    Derived::packed_bytes / sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 2);

        const AVXCommon::TwoQubitGateWithParamHelper<ApplyIsingZZAVX, ParamT>
            gate_helper(
                &GateImplementationsLM::applyIsingZZ<PrecisionT, ParamT>);

        gate_helper(arr, num_qubits, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyControlledPhaseShift(std::complex<PrecisionT> *arr,
                                          const size_t num_qubits,
                                          const std::vector<size_t> &wires,
                                          [[maybe_unused]] bool inverse,
                                          ParamT angle) {
        using ApplyControlledPhaseShiftAVX =
            AVXCommon::ApplyControlledPhaseShift<
                PrecisionT, Derived::packed_bytes / sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 2);

        const AVXCommon::TwoQubitGateWithParamHelper<
            ApplyControlledPhaseShiftAVX, ParamT>
            gate_helper(
                &GateImplementationsLM::applyControlledPhaseShift<PrecisionT,
                                                                  ParamT>);

        gate_helper(arr, num_qubits, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyCRX(std::complex<PrecisionT> *arr, const size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT angle) {
        using ApplyCRXAVX =
            AVXCommon::ApplyCRX<PrecisionT,
                                Derived::packed_bytes / sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 2);

        const AVXCommon::TwoQubitGateWithParamHelper<ApplyCRXAVX, ParamT>
            gate_helper(&GateImplementationsLM::applyCRX<PrecisionT, ParamT>);

        gate_helper(arr, num_qubits, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyCRY(std::complex<PrecisionT> *arr, const size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT angle) {
        using ApplyCRYAVX =
            AVXCommon::ApplyCRY<PrecisionT,
                                Derived::packed_bytes / sizeof(PrecisionT)>;
        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 2);

        const AVXCommon::TwoQubitGateWithParamHelper<ApplyCRYAVX, ParamT>
            gate_helper(&GateImplementationsLM::applyCRY<PrecisionT, ParamT>);

        gate_helper(arr, num_qubits, wires, inverse, angle);
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyCRZ(std::complex<PrecisionT> *arr, const size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT angle) {
        using ApplyCRZAVX =
            AVXCommon::ApplyCRZ<PrecisionT,
                                Derived::packed_bytes / sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 2);

        const AVXCommon::TwoQubitGateWithParamHelper<ApplyCRZAVX, ParamT>
            gate_helper(&GateImplementationsLM::applyCRZ<PrecisionT, ParamT>);

        gate_helper(arr, num_qubits, wires, inverse, angle);
    }

    /* Generators */
    template <class PrecisionT>
    static auto applyGeneratorPhaseShift(std::complex<PrecisionT> *arr,
                                         const size_t num_qubits,
                                         const std::vector<size_t> &wires,
                                         [[maybe_unused]] bool inverse)
        -> PrecisionT {
        using ApplyGeneratorPhaseShiftAVX = AVXCommon::ApplyGeneratorPhaseShift<
            PrecisionT, Derived::packed_bytes / sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 1);

        const AVXCommon::SingleQubitGateWithoutParamHelper<
            ApplyGeneratorPhaseShiftAVX>
            gate_helper(
                &GateImplementationsLM::applyGeneratorPhaseShift<PrecisionT>);

        return gate_helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT>
    static auto applyGeneratorIsingXX(std::complex<PrecisionT> *arr,
                                      const size_t num_qubits,
                                      const std::vector<size_t> &wires,
                                      [[maybe_unused]] bool inverse)
        -> PrecisionT {
        using ApplyGeneratorIsingXXAVX = AVXCommon::ApplyGeneratorIsingXX<
            PrecisionT, Derived::packed_bytes / sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 2);

        const AVXCommon::TwoQubitGateWithoutParamHelper<
            ApplyGeneratorIsingXXAVX>
            gate_helper(
                &GateImplementationsLM::applyGeneratorIsingXX<PrecisionT>);

        return gate_helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT>
    static auto applyGeneratorIsingYY(std::complex<PrecisionT> *arr,
                                      const size_t num_qubits,
                                      const std::vector<size_t> &wires,
                                      [[maybe_unused]] bool inverse)
        -> PrecisionT {
        using ApplyGeneratorIsingYYAVX = AVXCommon::ApplyGeneratorIsingYY<
            PrecisionT, Derived::packed_bytes / sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 2);

        const AVXCommon::TwoQubitGateWithoutParamHelper<
            ApplyGeneratorIsingYYAVX>
            gate_helper(
                &GateImplementationsLM::applyGeneratorIsingYY<PrecisionT>);

        return gate_helper(arr, num_qubits, wires, inverse);
    }

    template <class PrecisionT>
    static auto applyGeneratorIsingZZ(std::complex<PrecisionT> *arr,
                                      const size_t num_qubits,
                                      const std::vector<size_t> &wires,
                                      [[maybe_unused]] bool inverse)
        -> PrecisionT {
        using ApplyGeneratorIsingZZAVX = AVXCommon::ApplyGeneratorIsingZZ<
            PrecisionT, Derived::packed_bytes / sizeof(PrecisionT)>;

        static_assert(std::is_same_v<PrecisionT, float> ||
                          std::is_same_v<PrecisionT, double>,
                      "Only float and double are supported.");

        PL_ASSERT(wires.size() == 2);

        const AVXCommon::TwoQubitGateWithoutParamHelper<
            ApplyGeneratorIsingZZAVX>
            gate_helper(
                &GateImplementationsLM::applyGeneratorIsingZZ<PrecisionT>);

        return gate_helper(arr, num_qubits, wires, inverse);
    }
};
} // namespace Pennylane::LightningQubit::Gates
