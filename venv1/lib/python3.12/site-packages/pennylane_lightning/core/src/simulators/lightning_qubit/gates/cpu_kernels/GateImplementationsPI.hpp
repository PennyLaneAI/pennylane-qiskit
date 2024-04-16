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
 * Defines gate operations with precomputed indices
 */
#pragma once

/// @cond DEV
// Required for compilation with MSVC
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES // for C++
#endif
/// @endcond

#include <bit>
#include <complex>
#include <vector>

#include "BitUtil.hpp"
#include "GateIndices.hpp"
#include "GateOperation.hpp"
#include "GatePragmas.hpp"
#include "Gates.hpp"
#include "KernelType.hpp"
#include "LinearAlgebra.hpp"
#include "PauliGenerator.hpp"
#include "Util.hpp" // ZERO

/// @cond DEV
namespace {
using namespace Pennylane::Gates;
using namespace Pennylane::LightningQubit::Gates::Pragmas;
using Pennylane::Util::ZERO;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Gates {
/**
 * @brief Kernel functions for gate operations with precomputed indices
 *
 * For given wires, we first compute the indices the gate applies to and use
 * the computed indices to apply the operation.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data.
 * */
class GateImplementationsPI : public PauliGenerator<GateImplementationsPI> {
  public:
    constexpr static KernelType kernel_id = KernelType::PI;
    constexpr static std::string_view name = "PI";
    template <typename PrecisionT>
    constexpr static size_t required_alignment =
        std::alignment_of_v<PrecisionT>;
    template <typename PrecisionT>
    constexpr static uint32_t packed_bytes = sizeof(PrecisionT);

    constexpr static std::array implemented_gates = {
        GateOperation::Identity,
        GateOperation::PauliX,
        GateOperation::PauliY,
        GateOperation::PauliZ,
        GateOperation::Hadamard,
        GateOperation::S,
        GateOperation::T,
        GateOperation::PhaseShift,
        GateOperation::RX,
        GateOperation::RY,
        GateOperation::RZ,
        GateOperation::Rot,
        GateOperation::CNOT,
        GateOperation::CY,
        GateOperation::CZ,
        GateOperation::SWAP,
        GateOperation::IsingXX,
        GateOperation::IsingXY,
        GateOperation::IsingYY,
        GateOperation::IsingZZ,
        GateOperation::ControlledPhaseShift,
        GateOperation::CRX,
        GateOperation::CRY,
        GateOperation::CRZ,
        GateOperation::CRot,
        GateOperation::Toffoli,
        GateOperation::CSWAP,
        GateOperation::DoubleExcitation,
        GateOperation::DoubleExcitationMinus,
        GateOperation::DoubleExcitationPlus,
        GateOperation::MultiRZ,
    };

    constexpr static std::array implemented_generators = {
        GeneratorOperation::PhaseShift,
        GeneratorOperation::RX,
        GeneratorOperation::RY,
        GeneratorOperation::RZ,
        GeneratorOperation::IsingXX,
        GeneratorOperation::IsingXY,
        GeneratorOperation::IsingYY,
        GeneratorOperation::IsingZZ,
        GeneratorOperation::CRX,
        GeneratorOperation::CRY,
        GeneratorOperation::CRZ,
        GeneratorOperation::ControlledPhaseShift,
        GeneratorOperation::DoubleExcitation,
        GeneratorOperation::DoubleExcitationMinus,
        GeneratorOperation::DoubleExcitationPlus,
    };

    constexpr static std::array implemented_matrices = {
        MatrixOperation::SingleQubitOp,
        MatrixOperation::TwoQubitOp,
        MatrixOperation::MultiQubitOp,
    };

    constexpr static std::array<ControlledGateOperation, 0>
        implemented_controlled_gates{};

    constexpr static std::array<ControlledGeneratorOperation, 0>
        implemented_controlled_generators{};

    constexpr static std::array<ControlledMatrixOperation, 0>
        implemented_controlled_matrices{};

    /**
     * @brief Apply a single qubit gate to the statevector.
     *
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param matrix Perfect square matrix in row-major order.
     * @param wires Wires the gate applies to.
     * @param inverse Indicate whether inverse should be taken.
     */
    template <class PrecisionT>
    static inline void
    applySingleQubitOp(std::complex<PrecisionT> *arr, size_t num_qubits,
                       const std::complex<PrecisionT> *matrix,
                       const std::vector<size_t> &wires, bool inverse = false) {
        PL_ASSERT(wires.size() == 1);

        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        if (inverse) {
            for (const size_t &externalIndex : externalIndices) {
                std::complex<PrecisionT> *shiftedState = arr + externalIndex;
                const std::complex<PrecisionT> v0 = shiftedState[indices[0]];
                const std::complex<PrecisionT> v1 = shiftedState[indices[1]];
                shiftedState[indices[0]] =
                    std::conj(matrix[0B00]) * v0 +
                    std::conj(matrix[0B10]) *
                        v1; // NOLINT(readability-magic-numbers)
                shiftedState[indices[1]] =
                    std::conj(matrix[0B01]) * v0 +
                    std::conj(matrix[0B11]) *
                        v1; // NOLINT(readability-magic-numbers)
            }
        } else {
            for (const size_t &externalIndex : externalIndices) {
                std::complex<PrecisionT> *shiftedState = arr + externalIndex;
                const std::complex<PrecisionT> v0 = shiftedState[indices[0]];
                const std::complex<PrecisionT> v1 = shiftedState[indices[1]];
                shiftedState[indices[0]] =
                    matrix[0B00] * v0 +
                    matrix[0B01] * v1; // NOLINT(readability-magic-numbers)
                shiftedState[indices[1]] =
                    matrix[0B10] * v0 +
                    matrix[0B11] * v1; // NOLINT(readability-magic-numbers)
            }
        }
    }

    /**
     * @brief Apply a two qubit gate to the statevector.
     *
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param matrix Perfect square matrix in row-major order.
     * @param wires Wires the gate applies to.
     * @param inverse Indicate whether inverse should be taken.
     */
    template <class PrecisionT>
    static inline void
    applyTwoQubitOp(std::complex<PrecisionT> *arr, size_t num_qubits,
                    const std::complex<PrecisionT> *matrix,
                    const std::vector<size_t> &wires, bool inverse = false) {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        if (inverse) {
            for (const size_t &externalIndex : externalIndices) {
                std::complex<PrecisionT> *shiftedState = arr + externalIndex;

                const std::complex<PrecisionT> v00 = shiftedState[indices[0]];
                const std::complex<PrecisionT> v01 = shiftedState[indices[1]];
                const std::complex<PrecisionT> v10 = shiftedState[indices[2]];
                const std::complex<PrecisionT> v11 = shiftedState[indices[3]];

                // NOLINTBEGIN(readability-magic-numbers)
                shiftedState[indices[0]] = std::conj(matrix[0b0000]) * v00 +
                                           std::conj(matrix[0b0100]) * v01 +
                                           std::conj(matrix[0b1000]) * v10 +
                                           std::conj(matrix[0b1100]) * v11;
                shiftedState[indices[1]] = std::conj(matrix[0b0001]) * v00 +
                                           std::conj(matrix[0b0101]) * v01 +
                                           std::conj(matrix[0b1001]) * v10 +
                                           std::conj(matrix[0b1101]) * v11;
                shiftedState[indices[2]] = std::conj(matrix[0b0010]) * v00 +
                                           std::conj(matrix[0b0110]) * v01 +
                                           std::conj(matrix[0b1010]) * v10 +
                                           std::conj(matrix[0b1110]) * v11;
                shiftedState[indices[3]] = std::conj(matrix[0b0011]) * v00 +
                                           std::conj(matrix[0b0111]) * v01 +
                                           std::conj(matrix[0b1011]) * v10 +
                                           std::conj(matrix[0b1111]) * v11;
                // NOLINTEND(readability-magic-numbers)
            }
        } else {
            for (const size_t &externalIndex : externalIndices) {
                std::complex<PrecisionT> *shiftedState = arr + externalIndex;

                const std::complex<PrecisionT> v00 = shiftedState[indices[0]];
                const std::complex<PrecisionT> v01 = shiftedState[indices[1]];
                const std::complex<PrecisionT> v10 = shiftedState[indices[2]];
                const std::complex<PrecisionT> v11 = shiftedState[indices[3]];

                // NOLINTBEGIN(readability-magic-numbers)
                shiftedState[indices[0]] =
                    matrix[0b0000] * v00 + matrix[0b0001] * v01 +
                    matrix[0b0010] * v10 + matrix[0b0011] * v11;
                shiftedState[indices[1]] =
                    matrix[0b0100] * v00 + matrix[0b0101] * v01 +
                    matrix[0b0110] * v10 + matrix[0b0111] * v11;
                shiftedState[indices[2]] =
                    matrix[0b1000] * v00 + matrix[0b1001] * v01 +
                    matrix[0b1010] * v10 + matrix[0b1011] * v11;
                shiftedState[indices[3]] =
                    matrix[0b1100] * v00 + matrix[0b1101] * v01 +
                    matrix[0b1110] * v10 + matrix[0b1111] * v11;
                // NOLINTEND(readability-magic-numbers)
            }
        }
    }

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param arr Pointer to the statevector.
     * @param num_qubits Number of qubits.
     * @param matrix Perfect square matrix in row-major order.
     * @param wires Wires the gate applies to.
     * @param inverse Indicate whether inverse should be taken.
     */
    template <class PrecisionT>
    static void
    applyMultiQubitOp(std::complex<PrecisionT> *arr, size_t num_qubits,
                      const std::complex<PrecisionT> *matrix,
                      const std::vector<size_t> &wires, bool inverse) {
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        std::vector<std::complex<PrecisionT>> v(indices.size());
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            // Gather
            size_t pos = 0;
            for (const size_t &index : indices) {
                v[pos] = shiftedState[index];
                pos++;
            }

            // Apply + scatter
            if (inverse) {
                for (size_t i = 0; i < indices.size(); i++) {
                    size_t index = indices[i];
                    shiftedState[index] = 0;

                    for (size_t j = 0; j < indices.size(); j++) {
                        const size_t baseIndex = j * indices.size();
                        shiftedState[index] +=
                            std::conj(matrix[baseIndex + i]) * v[j];
                    }
                }
            } else {
                for (size_t i = 0; i < indices.size(); i++) {
                    size_t index = indices[i];
                    shiftedState[index] = 0;

                    const size_t baseIndex = i * indices.size();
                    for (size_t j = 0; j < indices.size(); j++) {
                        shiftedState[index] += matrix[baseIndex + j] * v[j];
                    }
                }
            }
        }
    }

    /* Single qubit operators */
    template <class PrecisionT>
    static void applyIdentity(std::complex<PrecisionT> *arr, size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 1);
        static_cast<void>(arr);        // No-op
        static_cast<void>(num_qubits); // No-op
        static_cast<void>(wires);      // No-op
    }

    template <class PrecisionT>
    static void applyPauliX(std::complex<PrecisionT> *arr, size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            std::swap(shiftedState[indices[0]], shiftedState[indices[1]]);
        }
    }

    template <class PrecisionT>
    static void applyPauliY(std::complex<PrecisionT> *arr, size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            std::complex<PrecisionT> v0 = shiftedState[indices[0]];
            shiftedState[indices[0]] =
                std::complex<PrecisionT>{shiftedState[indices[1]].imag(),
                                         -shiftedState[indices[1]].real()};
            shiftedState[indices[1]] =
                std::complex<PrecisionT>{-v0.imag(), v0.real()};
        }
    }

    template <class PrecisionT>
    static void applyPauliZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[1]] = -shiftedState[indices[1]];
        }
    }

    template <class PrecisionT>
    static void applyHadamard(std::complex<PrecisionT> *arr, size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;

            const std::complex<PrecisionT> v0 = shiftedState[indices[0]];
            const std::complex<PrecisionT> v1 = shiftedState[indices[1]];

            shiftedState[indices[0]] =
                Pennylane::Util::INVSQRT2<PrecisionT>() * (v0 + v1);
            shiftedState[indices[1]] =
                Pennylane::Util::INVSQRT2<PrecisionT>() * (v0 - v1);
        }
    }

    template <class PrecisionT>
    static void applyS(std::complex<PrecisionT> *arr, size_t num_qubits,
                       const std::vector<size_t> &wires, bool inverse) {
        PL_ASSERT(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        const std::complex<PrecisionT> shift =
            (inverse) ? -Pennylane::Util::IMAG<PrecisionT>()
                      : Pennylane::Util::IMAG<PrecisionT>();

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }

    template <class PrecisionT>
    static void applyT(std::complex<PrecisionT> *arr, size_t num_qubits,
                       const std::vector<size_t> &wires, bool inverse) {
        PL_ASSERT(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const std::complex<PrecisionT> shift =
            (inverse) ? std::conj(std::exp(std::complex<PrecisionT>(
                            0, static_cast<PrecisionT>(M_PI / 4))))
                      : std::exp(std::complex<PrecisionT>(
                            0, static_cast<PrecisionT>(M_PI / 4)));

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }

    /* Single qubit operators with a parameter */
    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyPhaseShift(std::complex<PrecisionT> *arr,
                                size_t num_qubits,
                                const std::vector<size_t> &wires, bool inverse,
                                ParamT angle) {
        PL_ASSERT(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        const std::complex<PrecisionT> s =
            inverse ? std::conj(std::exp(std::complex<PrecisionT>(0, angle)))
                    : std::exp(std::complex<PrecisionT>(0, angle));
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[1]] *= s;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRX(std::complex<PrecisionT> *arr, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        PL_ASSERT(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT js =
            (inverse) ? -std::sin(-angle / 2) : std::sin(-angle / 2);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            const std::complex<PrecisionT> v0 = shiftedState[indices[0]];
            const std::complex<PrecisionT> v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] =
                c * v0 + js * std::complex<PrecisionT>{-v1.imag(), v1.real()};
            shiftedState[indices[1]] =
                js * std::complex<PrecisionT>{-v0.imag(), v0.real()} + c * v1;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRY(std::complex<PrecisionT> *arr, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        PL_ASSERT(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT s =
            (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            const std::complex<PrecisionT> v0 = shiftedState[indices[0]];
            const std::complex<PrecisionT> v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = c * v0 - s * v1;
            shiftedState[indices[1]] = s * v0 + c * v1;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        ParamT angle) {
        PL_ASSERT(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const std::complex<PrecisionT> first =
            std::complex<PrecisionT>(std::cos(angle / 2), -std::sin(angle / 2));
        const std::complex<PrecisionT> second =
            std::complex<PrecisionT>(std::cos(angle / 2), std::sin(angle / 2));
        const std::complex<PrecisionT> shift1 =
            (inverse) ? std::conj(first) : first;
        const std::complex<PrecisionT> shift2 =
            (inverse) ? std::conj(second) : second;

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[0]] *= shift1;
            shiftedState[indices[1]] *= shift2;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyRot(std::complex<PrecisionT> *arr, size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT phi, ParamT theta, ParamT omega) {
        PL_ASSERT(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const auto rot = getRot<std::complex, PrecisionT>(phi, theta, omega);

        const std::complex<PrecisionT> t1 =
            (inverse) ? std::conj(rot[0]) : rot[0];
        const std::complex<PrecisionT> t2 = (inverse) ? -rot[1] : rot[1];
        const std::complex<PrecisionT> t3 = (inverse) ? -rot[2] : rot[2];
        const std::complex<PrecisionT> t4 =
            (inverse) ? std::conj(rot[3]) : rot[3];

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            const std::complex<PrecisionT> v0 = shiftedState[indices[0]];
            const std::complex<PrecisionT> v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = t1 * v0 + t2 * v1;
            shiftedState[indices[1]] = t3 * v0 + t4 * v1;
        }
    }

    /* Two qubit operators */
    template <class PrecisionT>
    static void applyCNOT(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            std::swap(shiftedState[indices[2]], shiftedState[indices[3]]);
        }
    }

    template <class PrecisionT>
    static void applyCY(std::complex<PrecisionT> *arr, size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            std::complex<PrecisionT> v2 = shiftedState[indices[2]];
            shiftedState[indices[2]] =
                std::complex<PrecisionT>{shiftedState[indices[3]].imag(),
                                         -shiftedState[indices[3]].real()};
            shiftedState[indices[3]] =
                std::complex<PrecisionT>{-v2.imag(), v2.real()};
        }
    }

    template <class PrecisionT>
    static void applyCZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[3]] *= -1;
        }
    }

    template <class PrecisionT>
    static void applySWAP(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            std::swap(shiftedState[indices[1]], shiftedState[indices[2]]);
        }
    }

    /* Two qubit operators with a parameter */
    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingXX(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires, bool inverse,
                             ParamT angle) {
        using ComplexT = std::complex<PrecisionT>;
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;

            const auto v0 = shiftedState[indices[0]];
            const auto v1 = shiftedState[indices[1]];
            const auto v2 = shiftedState[indices[2]];
            const auto v3 = shiftedState[indices[3]];

            shiftedState[indices[0]] = ComplexT{cr * real(v0) + sj * imag(v3),
                                                cr * imag(v0) - sj * real(v3)};
            shiftedState[indices[1]] = ComplexT{cr * real(v1) + sj * imag(v2),
                                                cr * imag(v1) - sj * real(v2)};
            shiftedState[indices[2]] = ComplexT{cr * real(v2) + sj * imag(v1),
                                                cr * imag(v2) - sj * real(v1)};
            shiftedState[indices[3]] = ComplexT{cr * real(v3) + sj * imag(v0),
                                                cr * imag(v3) - sj * real(v0)};
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingXY(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires, bool inverse,
                             ParamT angle) {
        using ComplexT = std::complex<PrecisionT>;
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;

            const auto v0 = shiftedState[indices[0]];
            const auto v1 = shiftedState[indices[1]];
            const auto v2 = shiftedState[indices[2]];
            const auto v3 = shiftedState[indices[3]];

            shiftedState[indices[0]] = ComplexT{real(v0), imag(v0)};
            shiftedState[indices[1]] = ComplexT{cr * real(v1) - sj * imag(v2),
                                                cr * imag(v1) + sj * real(v2)};
            shiftedState[indices[2]] = ComplexT{cr * real(v2) - sj * imag(v1),
                                                cr * imag(v2) + sj * real(v1)};
            shiftedState[indices[3]] = ComplexT{real(v3), imag(v3)};
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingYY(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires, bool inverse,
                             ParamT angle) {
        using ComplexT = std::complex<PrecisionT>;
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const PrecisionT cr = std::cos(angle / 2);
        const PrecisionT sj =
            inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;

            const auto v0 = shiftedState[indices[0]];
            const auto v1 = shiftedState[indices[1]];
            const auto v2 = shiftedState[indices[2]];
            const auto v3 = shiftedState[indices[3]];

            shiftedState[indices[0]] = ComplexT{cr * real(v0) - sj * imag(v3),
                                                cr * imag(v0) + sj * real(v3)};
            shiftedState[indices[1]] = ComplexT{cr * real(v1) + sj * imag(v2),
                                                cr * imag(v1) - sj * real(v2)};
            shiftedState[indices[2]] = ComplexT{cr * real(v2) + sj * imag(v1),
                                                cr * imag(v2) - sj * real(v1)};
            shiftedState[indices[3]] = ComplexT{cr * real(v3) - sj * imag(v0),
                                                cr * imag(v3) + sj * real(v0)};
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyIsingZZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires, bool inverse,
                             ParamT angle) {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const std::complex<PrecisionT> first =
            std::complex<PrecisionT>{std::cos(angle / 2), -std::sin(angle / 2)};
        const std::complex<PrecisionT> second =
            std::complex<PrecisionT>{std::cos(angle / 2), std::sin(angle / 2)};

        const std::array<std::complex<PrecisionT>, 2> shifts = {
            (inverse) ? std::conj(first) : first,
            (inverse) ? std::conj(second) : second};
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;

            shiftedState[indices[0]] *= shifts[0];
            shiftedState[indices[1]] *= shifts[1];
            shiftedState[indices[2]] *= shifts[1];
            shiftedState[indices[3]] *= shifts[0];
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyControlledPhaseShift(std::complex<PrecisionT> *arr,
                                          size_t num_qubits,
                                          const std::vector<size_t> &wires,
                                          bool inverse, ParamT angle) {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const std::complex<PrecisionT> s =
            inverse ? std::conj(std::exp(std::complex<PrecisionT>(0, angle)))
                    : std::exp(std::complex<PrecisionT>(0, angle));
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[3]] *= s;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyCRX(std::complex<PrecisionT> *arr, size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT angle) {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT js =
            (inverse) ? -std::sin(-angle / 2) : std::sin(-angle / 2);
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            const std::complex<PrecisionT> v0 = shiftedState[indices[2]];
            const std::complex<PrecisionT> v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] =
                c * v0 + js * std::complex<PrecisionT>{-v1.imag(), v1.real()};
            shiftedState[indices[3]] =
                js * std::complex<PrecisionT>{-v0.imag(), v0.real()} + c * v1;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyCRY(std::complex<PrecisionT> *arr, size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT angle) {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        const PrecisionT c = std::cos(angle / 2);
        const PrecisionT s =
            (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            const std::complex<PrecisionT> v0 = shiftedState[indices[2]];
            const std::complex<PrecisionT> v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = c * v0 - s * v1;
            shiftedState[indices[3]] = s * v0 + c * v1;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyCRZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         ParamT angle) {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        const std::complex<PrecisionT> m00 =
            (inverse) ? std::complex<PrecisionT>(std::cos(angle / 2),
                                                 std::sin(angle / 2))
                      : std::complex<PrecisionT>(std::cos(angle / 2),
                                                 -std::sin(angle / 2));
        const std::complex<PrecisionT> m11 =
            (inverse) ? std::complex<PrecisionT>(std::cos(angle / 2),
                                                 -std::sin(angle / 2))
                      : std::complex<PrecisionT>(std::cos(angle / 2),
                                                 std::sin(angle / 2));
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[2]] *= m00;
            shiftedState[indices[3]] *= m11;
        }
    }

    template <class PrecisionT, class ParamT = PrecisionT>
    static void applyCRot(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires, bool inverse,
                          ParamT phi, ParamT theta, ParamT omega) {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        const auto rot = getRot<std::complex, PrecisionT>(phi, theta, omega);

        const std::complex<PrecisionT> t1 =
            (inverse) ? std::conj(rot[0]) : rot[0];
        const std::complex<PrecisionT> t2 = (inverse) ? -rot[1] : rot[1];
        const std::complex<PrecisionT> t3 = (inverse) ? -rot[2] : rot[2];
        const std::complex<PrecisionT> t4 =
            (inverse) ? std::conj(rot[3]) : rot[3];
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            const std::complex<PrecisionT> v0 = shiftedState[indices[2]];
            const std::complex<PrecisionT> v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = t1 * v0 + t2 * v1;
            shiftedState[indices[3]] = t3 * v0 + t4 * v1;
        }
    }

    /* Three-qubit gate */
    template <class PrecisionT>
    static void applyToffoli(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 3);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        // Participating swapped indices
        static const size_t op_idx0 = 6;
        static const size_t op_idx1 = 7;
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            std::swap(shiftedState[indices[op_idx0]],
                      shiftedState[indices[op_idx1]]);
        }
    }

    template <class PrecisionT>
    static void applyCSWAP(std::complex<PrecisionT> *arr, size_t num_qubits,
                           const std::vector<size_t> &wires,
                           [[maybe_unused]] bool inverse) {
        PL_ASSERT(wires.size() == 3);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        // Participating swapped indices
        static const size_t op_idx0 = 5;
        static const size_t op_idx1 = 6;

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            std::swap(shiftedState[indices[op_idx0]],
                      shiftedState[indices[op_idx1]]);
        }
    }

    /* Four-qubit gates */
    template <class PrecisionT, class ParamT = PrecisionT>
    static void
    applyDoubleExcitation(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool inverse, ParamT angle);

    template <class PrecisionT, class ParamT = PrecisionT>
    static void
    applyDoubleExcitationMinus(std::complex<PrecisionT> *arr, size_t num_qubits,
                               const std::vector<size_t> &wires,
                               [[maybe_unused]] bool inverse, ParamT angle);

    template <class PrecisionT, class ParamT = PrecisionT>
    static void
    applyDoubleExcitationPlus(std::complex<PrecisionT> *arr, size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse, ParamT angle);

    /* Multi-qubit gates */
    template <class PrecisionT, class ParamT>
    static void applyMultiRZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse, ParamT angle) {
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        const std::complex<PrecisionT> first =
            std::complex<PrecisionT>{std::cos(angle / 2), -std::sin(angle / 2)};
        const std::complex<PrecisionT> second =
            std::complex<PrecisionT>{std::cos(angle / 2), std::sin(angle / 2)};

        const std::array<std::complex<PrecisionT>, 2> shifts = {
            (inverse) ? std::conj(first) : first,
            (inverse) ? std::conj(second) : second};

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            for (size_t k = 0; k < indices.size(); k++) {
                shiftedState[indices[k]] *= shifts[std::popcount(k) % 2];
            }
        }
    }

    /* Gate generators */
    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorPhaseShift(std::complex<PrecisionT> *arr, size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool adj) -> PrecisionT {
        PL_ASSERT(wires.size() == 1);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[0]] = std::complex<PrecisionT>{0.0, 0.0};
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return static_cast<PrecisionT>(1.0);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorCRX(std::complex<PrecisionT> *arr, size_t num_qubits,
                      const std::vector<size_t> &wires,
                      [[maybe_unused]] bool adj) -> PrecisionT {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[0]] = ZERO<PrecisionT>();
            shiftedState[indices[1]] = ZERO<PrecisionT>();

            std::swap(shiftedState[indices[2]], shiftedState[indices[3]]);
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorIsingXX(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool adj) -> PrecisionT {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            std::swap(shiftedState[indices[0]], shiftedState[indices[3]]);
            std::swap(shiftedState[indices[2]], shiftedState[indices[1]]);
        }

        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorIsingXY(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool adj) -> PrecisionT {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;

            std::swap(shiftedState[indices[2]], shiftedState[indices[1]]);
            shiftedState[indices[0]] = std::complex<PrecisionT>{0.0, 0.0};
            shiftedState[indices[3]] = std::complex<PrecisionT>{0.0, 0.0};
        }

        // NOLINTNEXTLINE(readability-magic-numbers)
        return static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorIsingYY(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool adj) -> PrecisionT {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            const auto v00 = shiftedState[indices[0]];
            shiftedState[indices[0]] = -shiftedState[indices[3]];
            shiftedState[indices[3]] = -v00;
            std::swap(shiftedState[indices[2]], shiftedState[indices[1]]);
        }

        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorIsingZZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool adj) -> PrecisionT {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;

            shiftedState[indices[1]] *= -1;
            shiftedState[indices[2]] *= -1;
        }

        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorCRY(std::complex<PrecisionT> *arr, size_t num_qubits,
                      const std::vector<size_t> &wires,
                      [[maybe_unused]] bool adj) -> PrecisionT {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            const auto v0 = shiftedState[indices[2]];
            shiftedState[indices[0]] = ZERO<PrecisionT>();
            shiftedState[indices[1]] = ZERO<PrecisionT>();
            shiftedState[indices[2]] =
                -Pennylane::Util::IMAG<PrecisionT>() * shiftedState[indices[3]];
            shiftedState[indices[3]] = Pennylane::Util::IMAG<PrecisionT>() * v0;
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorCRZ(std::complex<PrecisionT> *arr, size_t num_qubits,
                      const std::vector<size_t> &wires,
                      [[maybe_unused]] bool adj) -> PrecisionT {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[0]] = ZERO<PrecisionT>();
            shiftedState[indices[1]] = ZERO<PrecisionT>();
            shiftedState[indices[3]] *= -1;
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return -static_cast<PrecisionT>(0.5);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto applyGeneratorControlledPhaseShift(
        std::complex<PrecisionT> *arr, size_t num_qubits,
        const std::vector<size_t> &wires, [[maybe_unused]] bool adj)
        -> PrecisionT {
        PL_ASSERT(wires.size() == 2);
        const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

        for (const size_t &externalIndex : externalIndices) {
            std::complex<PrecisionT> *shiftedState = arr + externalIndex;
            shiftedState[indices[0]] = 0;
            shiftedState[indices[1]] = 0;
            shiftedState[indices[2]] = 0;
        }
        // NOLINTNEXTLINE(readability-magic-numbers)
        return static_cast<PrecisionT>(1);
    }

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorDoubleExcitation(std::complex<PrecisionT> *arr,
                                   size_t num_qubits,
                                   const std::vector<size_t> &wires,
                                   [[maybe_unused]] bool adj) -> PrecisionT;

    template <class PrecisionT>
    [[nodiscard]] static auto applyGeneratorDoubleExcitationMinus(
        std::complex<PrecisionT> *arr, size_t num_qubits,
        const std::vector<size_t> &wires, [[maybe_unused]] bool adj)
        -> PrecisionT;

    template <class PrecisionT>
    [[nodiscard]] static auto
    applyGeneratorDoubleExcitationPlus(std::complex<PrecisionT> *arr,
                                       size_t num_qubits,
                                       const std::vector<size_t> &wires,
                                       [[maybe_unused]] bool adj) -> PrecisionT;
};

// Matrix operations
extern template void GateImplementationsPI::applySingleQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);
extern template void GateImplementationsPI::applySingleQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);

extern template void GateImplementationsPI::applyTwoQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);
extern template void GateImplementationsPI::applyTwoQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);

extern template void GateImplementationsPI::applyMultiQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);
extern template void GateImplementationsPI::applyMultiQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);

// Single-qubit gates
extern template void
GateImplementationsPI::applyIdentity<float>(std::complex<float> *, size_t,
                                            const std::vector<size_t> &, bool);
extern template void
GateImplementationsPI::applyIdentity<double>(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool);

extern template void
GateImplementationsPI::applyPauliX<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
extern template void
GateImplementationsPI::applyPauliX<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

extern template void
GateImplementationsPI::applyPauliY<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
extern template void
GateImplementationsPI::applyPauliY<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

extern template void
GateImplementationsPI::applyPauliZ<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
extern template void
GateImplementationsPI::applyPauliZ<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

extern template void
GateImplementationsPI::applyHadamard<float>(std::complex<float> *, size_t,
                                            const std::vector<size_t> &, bool);
extern template void
GateImplementationsPI::applyHadamard<double>(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool);

extern template void
GateImplementationsPI::applyS<float>(std::complex<float> *, size_t,
                                     const std::vector<size_t> &, bool);
extern template void
GateImplementationsPI::applyS<double>(std::complex<double> *, size_t,
                                      const std::vector<size_t> &, bool);

extern template void
GateImplementationsPI::applyT<float>(std::complex<float> *, size_t,
                                     const std::vector<size_t> &, bool);
extern template void
GateImplementationsPI::applyT<double>(std::complex<double> *, size_t,
                                      const std::vector<size_t> &, bool);

extern template void GateImplementationsPI::applyPhaseShift<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsPI::applyPhaseShift<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsPI::applyRX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsPI::applyRX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsPI::applyRY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsPI::applyRY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsPI::applyRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsPI::applyRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void
GateImplementationsPI::applyRot<float, float>(std::complex<float> *, size_t,
                                              const std::vector<size_t> &, bool,
                                              float, float, float);
extern template void
GateImplementationsPI::applyRot<double, double>(std::complex<double> *, size_t,
                                                const std::vector<size_t> &,
                                                bool, double, double, double);

// Two-qubit gates
extern template void
GateImplementationsPI::applyCNOT<float>(std::complex<float> *, size_t,
                                        const std::vector<size_t> &, bool);
extern template void
GateImplementationsPI::applyCNOT<double>(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool);

extern template void
GateImplementationsPI::applyCY<float>(std::complex<float> *, size_t,
                                      const std::vector<size_t> &, bool);
extern template void
GateImplementationsPI::applyCY<double>(std::complex<double> *, size_t,
                                       const std::vector<size_t> &, bool);

extern template void
GateImplementationsPI::applyCZ<float>(std::complex<float> *, size_t,
                                      const std::vector<size_t> &, bool);
extern template void
GateImplementationsPI::applyCZ<double>(std::complex<double> *, size_t,
                                       const std::vector<size_t> &, bool);

extern template void
GateImplementationsPI::applySWAP<float>(std::complex<float> *, size_t,
                                        const std::vector<size_t> &, bool);
extern template void
GateImplementationsPI::applySWAP<double>(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool);

extern template void GateImplementationsPI::applyIsingXX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsPI::applyIsingXX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsPI::applyIsingXY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsPI::applyIsingXY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsPI::applyIsingYY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsPI::applyIsingYY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsPI::applyIsingZZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsPI::applyIsingZZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void
GateImplementationsPI::applyControlledPhaseShift<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void
GateImplementationsPI::applyControlledPhaseShift<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsPI::applyCRX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsPI::applyCRX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsPI::applyCRY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsPI::applyCRY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void GateImplementationsPI::applyCRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsPI::applyCRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

extern template void
GateImplementationsPI::applyCRot<float, float>(std::complex<float> *, size_t,
                                               const std::vector<size_t> &,
                                               bool, float, float, float);
extern template void
GateImplementationsPI::applyCRot<double, double>(std::complex<double> *, size_t,
                                                 const std::vector<size_t> &,
                                                 bool, double, double, double);

extern template void
GateImplementationsPI::applyToffoli<float>(std::complex<float> *, size_t,
                                           const std::vector<size_t> &, bool);
extern template void
GateImplementationsPI::applyToffoli<double>(std::complex<double> *, size_t,
                                            const std::vector<size_t> &, bool);

extern template void
GateImplementationsPI::applyCSWAP<float>(std::complex<float> *, size_t,
                                         const std::vector<size_t> &, bool);
extern template void
GateImplementationsPI::applyCSWAP<double>(std::complex<double> *, size_t,
                                          const std::vector<size_t> &, bool);

extern template void GateImplementationsPI::applyMultiRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
extern template void GateImplementationsPI::applyMultiRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

/* Generators */
extern template auto GateImplementationsPI::applyGeneratorPhaseShift(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto
GateImplementationsPI::applyGeneratorPhaseShift(std::complex<double> *, size_t,
                                                const std::vector<size_t> &,
                                                bool) -> double;

extern template auto PauliGenerator<GateImplementationsPI>::applyGeneratorRX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto PauliGenerator<GateImplementationsPI>::applyGeneratorRX(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;

extern template auto PauliGenerator<GateImplementationsPI>::applyGeneratorRY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto PauliGenerator<GateImplementationsPI>::applyGeneratorRY(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;

extern template auto PauliGenerator<GateImplementationsPI>::applyGeneratorRZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto PauliGenerator<GateImplementationsPI>::applyGeneratorRZ(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;

extern template auto GateImplementationsPI::applyGeneratorIsingXX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto
GateImplementationsPI::applyGeneratorIsingXX(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;

extern template auto
GateImplementationsPI::applyGeneratorIsingXY(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
extern template auto GateImplementationsPI::applyGeneratorIsingXY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

extern template auto GateImplementationsPI::applyGeneratorIsingYY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto
GateImplementationsPI::applyGeneratorIsingYY(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;

extern template auto
GateImplementationsPI::applyGeneratorIsingZZ(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
extern template auto GateImplementationsPI::applyGeneratorIsingZZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

extern template auto GateImplementationsPI::applyGeneratorCRX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto
GateImplementationsPI::applyGeneratorCRX(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;

extern template auto GateImplementationsPI::applyGeneratorCRY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto
GateImplementationsPI::applyGeneratorCRY(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;

extern template auto GateImplementationsPI::applyGeneratorCRZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto
GateImplementationsPI::applyGeneratorCRZ(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;

extern template auto GateImplementationsPI::applyGeneratorControlledPhaseShift(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
extern template auto GateImplementationsPI::applyGeneratorControlledPhaseShift(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
} // namespace Pennylane::LightningQubit::Gates
