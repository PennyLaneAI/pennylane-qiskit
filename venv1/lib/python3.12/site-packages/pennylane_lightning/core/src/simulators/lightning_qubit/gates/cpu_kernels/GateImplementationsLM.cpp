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

#include "GateImplementationsLM.hpp"
#include "Util.hpp" // exp2

/// @cond DEV
namespace {
using Pennylane::Util::exp2;
} // namespace
/// @endcond
namespace Pennylane::LightningQubit::Gates {
// Explicit instantiation starts

/* Matrix operations */

template void GateImplementationsLM::applySingleQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);
template void GateImplementationsLM::applySingleQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyTwoQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyTwoQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyMultiQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyMultiQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCSingleQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, const std::vector<bool> &,
    const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCSingleQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, const std::vector<bool> &,
    const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCTwoQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, const std::vector<bool> &,
    const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCTwoQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, const std::vector<bool> &,
    const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCMultiQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, const std::vector<bool> &,
    const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCMultiQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, const std::vector<bool> &,
    const std::vector<size_t> &, bool);

/* Single-qubit gates */

template void
GateImplementationsLM::applyIdentity<float>(std::complex<float> *, size_t,
                                            const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyIdentity<double>(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyPauliX<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyPauliX<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyPauliY<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyPauliY<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyPauliZ<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyPauliZ<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyHadamard<float>(std::complex<float> *, size_t,
                                            const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyHadamard<double>(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyS<float>(std::complex<float> *,
                                                   size_t,
                                                   const std::vector<size_t> &,
                                                   bool);
template void GateImplementationsLM::applyS<double>(std::complex<double> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);
template void GateImplementationsLM::applyT<float>(std::complex<float> *,
                                                   size_t,
                                                   const std::vector<size_t> &,
                                                   bool);
template void GateImplementationsLM::applyT<double>(std::complex<double> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);
template void GateImplementationsLM::applyPhaseShift<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyPhaseShift<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyRX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyRX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyRY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyRY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);
template void
GateImplementationsLM::applyRot<float, float>(std::complex<float> *, size_t,
                                              const std::vector<size_t> &, bool,
                                              float, float, float);
template void
GateImplementationsLM::applyRot<double, double>(std::complex<double> *, size_t,
                                                const std::vector<size_t> &,
                                                bool, double, double, double);

/* Two-qubit gates */

template void
GateImplementationsLM::applyCNOT<float>(std::complex<float> *, size_t,
                                        const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyCNOT<double>(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyCY<float>(std::complex<float> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);
template void
GateImplementationsLM::applyCY<double>(std::complex<double> *, size_t,
                                       const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyCZ<float>(std::complex<float> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);
template void
GateImplementationsLM::applyCZ<double>(std::complex<double> *, size_t,
                                       const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applySWAP<float>(std::complex<float> *, size_t,
                                        const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applySWAP<double>(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyCSWAP<float>(std::complex<float> *, size_t,
                                         const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyCSWAP<double>(std::complex<double> *, size_t,
                                          const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyToffoli<float>(std::complex<float> *, size_t,
                                           const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyToffoli<double>(std::complex<double> *, size_t,
                                            const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyIsingXX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyIsingXX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyIsingXY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyIsingXY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyIsingYY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyIsingYY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyIsingZZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyIsingZZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyControlledPhaseShift<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyControlledPhaseShift<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyCRX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyCRX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyCRY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyCRY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyCRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyCRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);
template void
GateImplementationsLM::applyCRot<float, float>(std::complex<float> *, size_t,
                                               const std::vector<size_t> &,
                                               bool, float, float, float);
template void
GateImplementationsLM::applyCRot<double, double>(std::complex<double> *, size_t,
                                                 const std::vector<size_t> &,
                                                 bool, double, double, double);
template void GateImplementationsLM::applyMultiRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyMultiRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyGlobalPhase<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyGlobalPhase<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

/* QChem functions */

template void GateImplementationsLM::applySingleExcitation<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);
template void GateImplementationsLM::applySingleExcitation<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);
template void GateImplementationsLM::applySingleExcitationMinus<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);
template void GateImplementationsLM::applySingleExcitationMinus<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);
template void GateImplementationsLM::applySingleExcitationPlus<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);
template void GateImplementationsLM::applySingleExcitationPlus<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);
template void GateImplementationsLM::applyDoubleExcitation<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);
template void GateImplementationsLM::applyDoubleExcitation<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);
template void GateImplementationsLM::applyDoubleExcitationMinus<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);
template void GateImplementationsLM::applyDoubleExcitationMinus<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);
template void GateImplementationsLM::applyDoubleExcitationPlus<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);
template void GateImplementationsLM::applyDoubleExcitationPlus<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);

/* N-controlled gates */

template void GateImplementationsLM::applyNCPauliX<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCPauliX<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCPauliY<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCPauliY<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCPauliZ<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCPauliZ<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCHadamard<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCHadamard<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCS<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCS<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCT<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCT<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyNCPhaseShift<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyNCPhaseShift<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyNCRX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyNCRX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyNCRY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyNCRY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyNCRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyNCRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyNCSingleExcitation<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyNCSingleExcitation<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyNCSingleExcitationMinus<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, float);
template void
GateImplementationsLM::applyNCSingleExcitationMinus<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyNCSingleExcitationPlus<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, float);
template void
GateImplementationsLM::applyNCSingleExcitationPlus<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyNCDoubleExcitation<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyNCDoubleExcitation<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyNCDoubleExcitationMinus<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, float);
template void
GateImplementationsLM::applyNCDoubleExcitationMinus<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, double);
template void GateImplementationsLM::applyNCDoubleExcitationPlus<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, float);
template void
GateImplementationsLM::applyNCDoubleExcitationPlus<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool, double);
// Generators
template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRX(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRY(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRZ(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLM::applyGeneratorPhaseShift(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLM::applyGeneratorPhaseShift(std::complex<double> *, size_t,
                                                const std::vector<size_t> &,
                                                bool) -> double;
template auto GateImplementationsLM::applyGeneratorCRX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLM::applyGeneratorCRX(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLM::applyGeneratorCRY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLM::applyGeneratorCRY(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLM::applyGeneratorCRZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLM::applyGeneratorCRZ(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;
template auto
GateImplementationsLM::applyGeneratorIsingXX(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLM::applyGeneratorIsingXX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLM::applyGeneratorIsingXY(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLM::applyGeneratorIsingXY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto GateImplementationsLM::applyGeneratorIsingYY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLM::applyGeneratorIsingYY(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
template auto
GateImplementationsLM::applyGeneratorIsingZZ(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLM::applyGeneratorIsingZZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto GateImplementationsLM::applyGeneratorControlledPhaseShift(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLM::applyGeneratorControlledPhaseShift(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLM::applyGeneratorMultiRZ(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLM::applyGeneratorMultiRZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

/* QChem */

template auto GateImplementationsLM::applyGeneratorSingleExcitation<float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> float;
template auto GateImplementationsLM::applyGeneratorSingleExcitation<double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> double;
template auto GateImplementationsLM::applyGeneratorSingleExcitationMinus<float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> float;
template auto
GateImplementationsLM::applyGeneratorSingleExcitationMinus<double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> double;
template auto GateImplementationsLM::applyGeneratorSingleExcitationPlus<float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> float;
template auto GateImplementationsLM::applyGeneratorSingleExcitationPlus<double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> double;
template auto GateImplementationsLM::applyGeneratorDoubleExcitation<float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> float;
template auto GateImplementationsLM::applyGeneratorDoubleExcitation<double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> double;
template auto GateImplementationsLM::applyGeneratorDoubleExcitationMinus<float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> float;
template auto
GateImplementationsLM::applyGeneratorDoubleExcitationMinus<double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> double;
template auto GateImplementationsLM::applyGeneratorDoubleExcitationPlus<float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> float;
template auto GateImplementationsLM::applyGeneratorDoubleExcitationPlus<double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> double;

/* N-controlled generators */

template auto GateImplementationsLM::applyNCGeneratorPhaseShift<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> float;
template auto GateImplementationsLM::applyNCGeneratorPhaseShift<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> double;
template auto GateImplementationsLM::applyNCGeneratorRX<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> float;
template auto GateImplementationsLM::applyNCGeneratorRX<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> double;
template auto GateImplementationsLM::applyNCGeneratorRY<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> float;
template auto GateImplementationsLM::applyNCGeneratorRY<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> double;
template auto GateImplementationsLM::applyNCGeneratorRZ<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> float;
template auto GateImplementationsLM::applyNCGeneratorRZ<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> double;
template auto GateImplementationsLM::applyNCGeneratorSingleExcitation<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> float;
template auto GateImplementationsLM::applyNCGeneratorSingleExcitation<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> double;
template auto
GateImplementationsLM::applyNCGeneratorSingleExcitationMinus<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLM::applyNCGeneratorSingleExcitationMinus<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> double;
template auto
GateImplementationsLM::applyNCGeneratorSingleExcitationPlus<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLM::applyNCGeneratorSingleExcitationPlus<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> double;
template auto GateImplementationsLM::applyNCGeneratorDoubleExcitation<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> float;
template auto GateImplementationsLM::applyNCGeneratorDoubleExcitation<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> double;
template auto
GateImplementationsLM::applyNCGeneratorDoubleExcitationMinus<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLM::applyNCGeneratorDoubleExcitationMinus<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> double;
template auto
GateImplementationsLM::applyNCGeneratorDoubleExcitationPlus<float>(
    std::complex<float> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLM::applyNCGeneratorDoubleExcitationPlus<double>(
    std::complex<double> *, size_t, const std::vector<size_t> &,
    const std::vector<bool> &, const std::vector<size_t> &, bool) -> double;
// Explicit instantiations ends
} // namespace Pennylane::LightningQubit::Gates
