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
#include "GateImplementationsPI.hpp"

namespace Pennylane::LightningQubit::Gates {
template <class PrecisionT, class ParamT>
void GateImplementationsPI::applyDoubleExcitation(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool inverse,
    ParamT angle) {
    PL_ASSERT(wires.size() == 4);
    const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
    const PrecisionT c = std::cos(angle / 2);
    const PrecisionT s = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i0 = 3;
    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i1 = 12;

    for (const size_t &externalIndex : externalIndices) {
        std::complex<PrecisionT> *shiftedState = arr + externalIndex;
        const std::complex<PrecisionT> v3 = shiftedState[indices[i0]];
        const std::complex<PrecisionT> v12 = shiftedState[indices[i1]];

        shiftedState[indices[i0]] = c * v3 - s * v12;
        shiftedState[indices[i1]] = s * v3 + c * v12;
    }
}

template <class PrecisionT, class ParamT>
void GateImplementationsPI::applyDoubleExcitationMinus(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool inverse,
    ParamT angle) {
    PL_ASSERT(wires.size() == 4);
    const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

    const PrecisionT c = std::cos(angle / 2);
    const PrecisionT s = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
    const std::complex<PrecisionT> e =
        inverse ? std::exp(std::complex<PrecisionT>(0, angle / 2))
                : std::exp(-std::complex<PrecisionT>(0, angle / 2));

    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i0 = 3;
    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i1 = 12;

    for (const size_t &externalIndex : externalIndices) {
        std::complex<PrecisionT> *shiftedState = arr + externalIndex;
        const std::complex<PrecisionT> v3 = shiftedState[indices[i0]];
        const std::complex<PrecisionT> v12 = shiftedState[indices[i1]];
        // NOLINTBEGIN(readability-magic-numbers)
        shiftedState[indices[0]] *= e;
        shiftedState[indices[1]] *= e;
        shiftedState[indices[2]] *= e;
        shiftedState[indices[i0]] = c * v3 - s * v12;
        shiftedState[indices[4]] *= e;
        shiftedState[indices[5]] *= e;
        shiftedState[indices[6]] *= e;
        shiftedState[indices[7]] *= e;
        shiftedState[indices[8]] *= e;
        shiftedState[indices[9]] *= e;
        shiftedState[indices[10]] *= e;
        shiftedState[indices[11]] *= e;
        shiftedState[indices[i1]] = s * v3 + c * v12;
        shiftedState[indices[13]] *= e;
        shiftedState[indices[14]] *= e;
        shiftedState[indices[15]] *= e;
        // NOLINTEND(readability-magic-numbers)
    }
}

template <class PrecisionT, class ParamT>
void GateImplementationsPI::applyDoubleExcitationPlus(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool inverse,
    ParamT angle) {
    PL_ASSERT(wires.size() == 4);
    const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
    const PrecisionT c = std::cos(angle / 2);
    const PrecisionT s = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
    const std::complex<PrecisionT> e =
        inverse ? std::exp(-std::complex<PrecisionT>(0, angle / 2))
                : std::exp(std::complex<PrecisionT>(0, angle / 2));
    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i0 = 3;
    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i1 = 12;

    for (const size_t &externalIndex : externalIndices) {
        std::complex<PrecisionT> *shiftedState = arr + externalIndex;
        const std::complex<PrecisionT> v3 = shiftedState[indices[i0]];
        const std::complex<PrecisionT> v12 = shiftedState[indices[i1]];
        // NOLINTBEGIN(readability-magic-numbers)
        shiftedState[indices[0]] *= e;
        shiftedState[indices[1]] *= e;
        shiftedState[indices[2]] *= e;
        shiftedState[indices[i0]] = c * v3 - s * v12;
        shiftedState[indices[4]] *= e;
        shiftedState[indices[5]] *= e;
        shiftedState[indices[6]] *= e;
        shiftedState[indices[7]] *= e;
        shiftedState[indices[8]] *= e;
        shiftedState[indices[9]] *= e;
        shiftedState[indices[10]] *= e;
        shiftedState[indices[11]] *= e;
        shiftedState[indices[i1]] = s * v3 + c * v12;
        shiftedState[indices[13]] *= e;
        shiftedState[indices[14]] *= e;
        shiftedState[indices[15]] *= e;
        // NOLINTEND(readability-magic-numbers)
    }
}

template <class PrecisionT>
auto GateImplementationsPI::applyGeneratorDoubleExcitation(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> PrecisionT {
    PL_ASSERT(wires.size() == 4);
    const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i0 = 3;
    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i1 = 12;

    for (const size_t &externalIndex : externalIndices) {
        std::complex<PrecisionT> *shiftedState = arr + externalIndex;
        const std::complex<PrecisionT> v3 = shiftedState[indices[i0]];
        const std::complex<PrecisionT> v12 = shiftedState[indices[i1]];
        for (const size_t &i : indices) {
            shiftedState[i] = std::complex<PrecisionT>{};
        }

        shiftedState[indices[i0]] = -v12 * Pennylane::Util::IMAG<PrecisionT>();
        shiftedState[indices[i1]] = v3 * Pennylane::Util::IMAG<PrecisionT>();
    }
    // NOLINTNEXTLINE(readability-magic-numbers)
    return -static_cast<PrecisionT>(0.5);
}

template <class PrecisionT>
auto GateImplementationsPI::applyGeneratorDoubleExcitationMinus(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> PrecisionT {
    PL_ASSERT(wires.size() == 4);
    const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i0 = 3;
    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i1 = 12;

    for (const size_t &externalIndex : externalIndices) {
        std::complex<PrecisionT> *shiftedState = arr + externalIndex;

        shiftedState[indices[i0]] *= Pennylane::Util::IMAG<PrecisionT>();
        shiftedState[indices[i1]] *= -Pennylane::Util::IMAG<PrecisionT>();

        std::swap(shiftedState[indices[i0]], shiftedState[indices[i1]]);
    }
    // NOLINTNEXTLINE(readability-magic-numbers)
    return -static_cast<PrecisionT>(0.5);
}

template <class PrecisionT>
auto GateImplementationsPI::applyGeneratorDoubleExcitationPlus(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> PrecisionT {
    PL_ASSERT(wires.size() == 4);
    const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i0 = 3;
    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i1 = 12;

    for (const size_t &externalIndex : externalIndices) {
        std::complex<PrecisionT> *shiftedState = arr + externalIndex;
        for (const size_t &i : indices) {
            shiftedState[i] *= -1;
        }

        shiftedState[indices[i0]] *= -Pennylane::Util::IMAG<PrecisionT>();
        shiftedState[indices[i1]] *= Pennylane::Util::IMAG<PrecisionT>();

        std::swap(shiftedState[indices[i0]], shiftedState[indices[i1]]);
    }
    // NOLINTNEXTLINE(readability-magic-numbers)
    return -static_cast<PrecisionT>(0.5);
}
// Matrix operations
template void GateImplementationsPI::applySingleQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);
template void GateImplementationsPI::applySingleQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);

template void GateImplementationsPI::applyTwoQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);
template void GateImplementationsPI::applyTwoQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);

template void GateImplementationsPI::applyMultiQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);
template void GateImplementationsPI::applyMultiQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);

// Single-qubit gates
template void
GateImplementationsPI::applyIdentity<float>(std::complex<float> *, size_t,
                                            const std::vector<size_t> &, bool);
template void
GateImplementationsPI::applyIdentity<double>(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool);

template void
GateImplementationsPI::applyPauliX<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
template void
GateImplementationsPI::applyPauliX<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

template void
GateImplementationsPI::applyPauliY<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
template void
GateImplementationsPI::applyPauliY<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

template void
GateImplementationsPI::applyPauliZ<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
template void
GateImplementationsPI::applyPauliZ<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

template void
GateImplementationsPI::applyHadamard<float>(std::complex<float> *, size_t,
                                            const std::vector<size_t> &, bool);
template void
GateImplementationsPI::applyHadamard<double>(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool);

template void GateImplementationsPI::applyS<float>(std::complex<float> *,
                                                   size_t,
                                                   const std::vector<size_t> &,
                                                   bool);
template void GateImplementationsPI::applyS<double>(std::complex<double> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);

template void GateImplementationsPI::applyT<float>(std::complex<float> *,
                                                   size_t,
                                                   const std::vector<size_t> &,
                                                   bool);
template void GateImplementationsPI::applyT<double>(std::complex<double> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);

template void GateImplementationsPI::applyPhaseShift<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsPI::applyPhaseShift<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsPI::applyRX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsPI::applyRX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsPI::applyRY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsPI::applyRY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsPI::applyRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsPI::applyRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void
GateImplementationsPI::applyRot<float, float>(std::complex<float> *, size_t,
                                              const std::vector<size_t> &, bool,
                                              float, float, float);
template void
GateImplementationsPI::applyRot<double, double>(std::complex<double> *, size_t,
                                                const std::vector<size_t> &,
                                                bool, double, double, double);

// Two-qubit gates
template void
GateImplementationsPI::applyCNOT<float>(std::complex<float> *, size_t,
                                        const std::vector<size_t> &, bool);
template void
GateImplementationsPI::applyCNOT<double>(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool);

template void GateImplementationsPI::applyCY<float>(std::complex<float> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);
template void
GateImplementationsPI::applyCY<double>(std::complex<double> *, size_t,
                                       const std::vector<size_t> &, bool);

template void GateImplementationsPI::applyCZ<float>(std::complex<float> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);
template void
GateImplementationsPI::applyCZ<double>(std::complex<double> *, size_t,
                                       const std::vector<size_t> &, bool);

template void
GateImplementationsPI::applySWAP<float>(std::complex<float> *, size_t,
                                        const std::vector<size_t> &, bool);
template void
GateImplementationsPI::applySWAP<double>(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool);

template void GateImplementationsPI::applyIsingXX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsPI::applyIsingXX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsPI::applyIsingXY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsPI::applyIsingXY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsPI::applyIsingYY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsPI::applyIsingYY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsPI::applyIsingZZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsPI::applyIsingZZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsPI::applyControlledPhaseShift<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsPI::applyControlledPhaseShift<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsPI::applyCRX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsPI::applyCRX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsPI::applyCRY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsPI::applyCRY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsPI::applyCRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsPI::applyCRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void
GateImplementationsPI::applyCRot<float, float>(std::complex<float> *, size_t,
                                               const std::vector<size_t> &,
                                               bool, float, float, float);
template void
GateImplementationsPI::applyCRot<double, double>(std::complex<double> *, size_t,
                                                 const std::vector<size_t> &,
                                                 bool, double, double, double);

template void
GateImplementationsPI::applyToffoli<float>(std::complex<float> *, size_t,
                                           const std::vector<size_t> &, bool);
template void
GateImplementationsPI::applyToffoli<double>(std::complex<double> *, size_t,
                                            const std::vector<size_t> &, bool);

template void
GateImplementationsPI::applyCSWAP<float>(std::complex<float> *, size_t,
                                         const std::vector<size_t> &, bool);
template void
GateImplementationsPI::applyCSWAP<double>(std::complex<double> *, size_t,
                                          const std::vector<size_t> &, bool);

template void GateImplementationsPI::applyMultiRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsPI::applyMultiRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

/* QChem */
template void GateImplementationsPI::applyDoubleExcitation<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsPI::applyDoubleExcitation<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsPI::applyDoubleExcitationMinus<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsPI::applyDoubleExcitationMinus<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsPI::applyDoubleExcitationPlus<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsPI::applyDoubleExcitationPlus<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

/* Generators */
template auto GateImplementationsPI::applyGeneratorPhaseShift(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsPI::applyGeneratorPhaseShift(std::complex<double> *, size_t,
                                                const std::vector<size_t> &,
                                                bool) -> double;

template auto PauliGenerator<GateImplementationsPI>::applyGeneratorRX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto PauliGenerator<GateImplementationsPI>::applyGeneratorRX(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;

template auto PauliGenerator<GateImplementationsPI>::applyGeneratorRY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto PauliGenerator<GateImplementationsPI>::applyGeneratorRY(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;

template auto PauliGenerator<GateImplementationsPI>::applyGeneratorRZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto PauliGenerator<GateImplementationsPI>::applyGeneratorRZ(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;

template auto GateImplementationsPI::applyGeneratorIsingXX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsPI::applyGeneratorIsingXX(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;

template auto
GateImplementationsPI::applyGeneratorIsingXY(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsPI::applyGeneratorIsingXY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

template auto GateImplementationsPI::applyGeneratorIsingYY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsPI::applyGeneratorIsingYY(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;

template auto
GateImplementationsPI::applyGeneratorIsingZZ(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsPI::applyGeneratorIsingZZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

template auto GateImplementationsPI::applyGeneratorCRX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsPI::applyGeneratorCRX(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;

template auto GateImplementationsPI::applyGeneratorCRY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsPI::applyGeneratorCRY(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;

template auto GateImplementationsPI::applyGeneratorCRZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsPI::applyGeneratorCRZ(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;

template auto GateImplementationsPI::applyGeneratorControlledPhaseShift(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto GateImplementationsPI::applyGeneratorControlledPhaseShift(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;

/* QChem */
template auto GateImplementationsPI::applyGeneratorDoubleExcitation<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto GateImplementationsPI::applyGeneratorDoubleExcitation<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;

template auto GateImplementationsPI::applyGeneratorDoubleExcitationMinus<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsPI::applyGeneratorDoubleExcitationMinus<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;

template auto GateImplementationsPI::applyGeneratorDoubleExcitationPlus<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto GateImplementationsPI::applyGeneratorDoubleExcitationPlus<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
} // namespace Pennylane::LightningQubit::Gates
