# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
This module contains tests for converting Qiskit NoiseModels to PennyLane NoiseModels.
"""
from functools import reduce

import itertools as it
import pytest
import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires

# pylint:disable = wrong-import-position, unnecessary-lambda
qiksit = pytest.importorskip("qiskit", "1.0.0")
from qiskit_aer import noise
from qiskit.quantum_info.operators.channel import Kraus

from pennylane_qiskit.noise_models import (
    _build_noise_model_map,
    _build_qerror_op,
)


class TestLoadNoiseChannels:
    """Tests for the helper methods of :func:`load_noise_models()` function."""

    @staticmethod
    def _kraus_to_choi(krau_mats, optimize=False) -> np.ndarray:
        r"""Transforms Kraus representation of a channel to its Choi representation."""
        kraus_vecs = np.array([kraus.ravel(order="F") for kraus in krau_mats])
        return np.einsum("ij,ik->jk", kraus_vecs, kraus_vecs.conj(), optimize=optimize)

    @pytest.mark.parametrize(
        "qiskit_error, pl_channel",
        [
            (
                noise.amplitude_damping_error(0.123, 0.0),
                qml.AmplitudeDamping(0.123, wires=AnyWires),
            ),
            (noise.phase_damping_error(0.123), qml.PhaseDamping(0.123, wires=AnyWires)),
            (
                noise.phase_amplitude_damping_error(0.0345, 0.0),
                qml.AmplitudeDamping(0.0345, wires=AnyWires),
            ),
            (
                noise.phase_amplitude_damping_error(0.0, 0.0345),
                qml.PhaseDamping(0.0345, wires=AnyWires),
            ),
            (noise.reset_error(0.02789), qml.ResetError(0.02789, 0.0, wires=AnyWires)),
            (noise.reset_error(0.01364, 0.02789), qml.ResetError(0.01364, 0.02789, wires=AnyWires)),
            (
                noise.thermal_relaxation_error(0.25, 0.45, 1.0, 0.01),
                qml.ThermalRelaxationError(0.01, 0.25, 0.45, 1.0, wires=AnyWires),
            ),
            (
                noise.thermal_relaxation_error(0.45, 0.25, 1.0, 0.01),
                qml.ThermalRelaxationError(0.01, 0.45, 0.25, 1.0, wires=AnyWires),
            ),
            (
                noise.depolarizing_error(0.3264, 1),
                qml.DepolarizingChannel(0.3264 * 3 / 4, wires=AnyWires),
            ),
            (
                noise.pauli_error([("X", 0.1), ("I", 0.9)]),
                qml.BitFlip(0.1, wires=AnyWires),
            ),
            (
                noise.pauli_error([("Y", 0.178), ("I", 0.822)]),
                qml.PauliError("Y", 0.178, wires=AnyWires),
            ),
            (
                noise.coherent_unitary_error(qml.X(0).matrix()),
                qml.QubitChannel([qml.X(0).matrix()], wires=AnyWires),
            ),
            (
                noise.mixed_unitary_error(
                    [(qml.I(0).matrix(), 0.9), (qml.X(0).matrix(), 0.03), (qml.Y(0).matrix(), 0.07)]
                ),
                qml.QubitChannel(
                    [
                        np.sqrt(prob) * kraus_op(0).matrix()
                        for kraus_op, prob in [
                            (qml.X, 0.03),
                            (qml.Y, 0.07),
                            (qml.I, 0.9),
                        ]
                    ],
                    wires=AnyWires,
                ),
            ),
            (
                noise.depolarizing_error(0.2174, 2),
                qml.QubitChannel(
                    Kraus(noise.depolarizing_error(0.2174, 2)).data,
                    wires=[0, 1],
                ),
            ),
            (
                noise.phase_amplitude_damping_error(0.3451, 0.2356),
                qml.QubitChannel(
                    np.array(
                        [
                            [[-0.97035755 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -0.74564448 + 0.0j]],
                            [[-0.2416738 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.31450645 + 0.0j]],
                            [[0.0 + 0.0j, 0.58745213 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
                        ]
                    ),
                    wires=AnyWires,
                ),
            ),
            (
                noise.kraus_error(
                    [
                        [[-0.97035755 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -0.74564448 + 0.0j]],
                        [[-0.2416738 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.31450645 + 0.0j]],
                        [[0.0 + 0.0j, 0.58745213 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
                    ],
                ),
                qml.QubitChannel(
                    np.array(
                        [
                            [[-0.97035755 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -0.74564448 + 0.0j]],
                            [[-0.2416738 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.31450645 + 0.0j]],
                            [[0.0 + 0.0j, 0.58745213 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
                        ]
                    ),
                    wires=AnyWires,
                ),
            ),
        ],
    )
    def test_build_kraus_error_ops(self, qiskit_error, pl_channel):
        """Tests that a quantum error can be correctly converted into a PennyLane QubitChannel."""
        pl_op_from_qiskit = _build_qerror_op(qiskit_error)
        choi_mat1 = self._kraus_to_choi(
            pl_op_from_qiskit.compute_kraus_matrices(*pl_op_from_qiskit.data)
        )
        choi_mat2 = self._kraus_to_choi(pl_channel.compute_kraus_matrices(*pl_channel.data))
        assert np.allclose(choi_mat1, choi_mat2)

    @pytest.mark.parametrize(
        "depol1, depol2, exc_pop",
        [
            (0.123, 0.456, 0.414),
            (0.631, 0.729, 0.128),
            (0.384, 0.657, 0.902),
        ],
    )
    def test_build_model_map(self, depol1, depol2, exc_pop):
        """Tests that _build_noise_model_map constructs correct model map for a Qiskit noise model"""
        error_1 = noise.depolarizing_error(depol1, 1)
        error_2 = noise.depolarizing_error(depol2, 2)
        error_3 = noise.phase_amplitude_damping_error(0.14, 0.24, excited_state_population=exc_pop)

        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, ["rz", "sx", "x"])
        noise_model.add_all_qubit_quantum_error(error_2, ["cx"])
        noise_model.add_all_qubit_quantum_error(error_3, ["ry", "rx"])

        qerror_dmap, _ = _build_noise_model_map(noise_model)

        pl_channels = [
            qml.QubitChannel(
                qml.DepolarizingChannel.compute_kraus_matrices(3 * depol1 / 4),
                wires=AnyWires,
            ),
            qml.QubitChannel(
                [
                    np.sqrt(prob) * reduce(np.kron, prod, 1.0)
                    for prob, prod in zip(
                        [1 - 15 * depol2 / 16, *([depol2 / 16] * 15)],
                        it.product(
                            map(qml.matrix, [qml.I(0), qml.X(0), qml.Y(0), qml.Z(0)]), repeat=2
                        ),
                    )
                ],
                wires=AnyWires,
            ),
            qml.QubitChannel(
                qml.ThermalRelaxationError.compute_kraus_matrices(
                    exc_pop, 6.6302933312, 4.1837870638, 1.0
                ),
                wires=AnyWires,
            ),
        ]
        for key, channel in zip(list(qerror_dmap.keys()), pl_channels):
            choi_mat1 = self._kraus_to_choi(key.data)
            choi_mat2 = self._kraus_to_choi(channel.data)
            assert np.allclose(choi_mat1, choi_mat2)

        assert list(qerror_dmap.values()) == [
            {AnyWires: ["RZ", "SX", "X"]},
            {AnyWires: ["CNOT"]},
            {AnyWires: ["RY", "RX"]},
        ]

    @pytest.mark.parametrize(
        "combination, p_error",
        [
            (lambda err1, err2: err1.compose(err2), 0.052),
            (lambda err1, err2: err1.tensor(err2), 0.037),
            (lambda err1, err2: err1.expand(err2), 0.094),
        ],
    )
    def test_composition_error_ops(self, combination, p_error):
        """Tests that a combination of quantum errors can be correctly converted into a PennyLane QubitChannel."""

        bit_flip = noise.pauli_error([("X", p_error), ("I", 1 - p_error)])
        phase_flip = noise.pauli_error([("Z", p_error), ("I", 1 - p_error)])

        combined_error = combination(bit_flip, phase_flip)
        pl_op_from_qiskit = _build_qerror_op(combined_error)

        choi_mat1 = self._kraus_to_choi(Kraus(combined_error).data)
        choi_mat2 = self._kraus_to_choi(
            pl_op_from_qiskit.compute_kraus_matrices(*pl_op_from_qiskit.data)
        )
        assert np.allclose(choi_mat1, choi_mat2)
