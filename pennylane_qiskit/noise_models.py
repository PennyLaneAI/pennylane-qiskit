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
This module contains functions for converting Qiskit NoiseModel objects
into PennyLane NoiseModels.
"""
from collections import defaultdict
from typing import Tuple
from warnings import warn

import pennylane as qml
from pennylane.operation import AnyWires
from qiskit.quantum_info.operators.channel import Kraus

# pylint:disable = protected-access
qiskit_op_map = {
    "x": "X",
    "y": "Y",
    "z": "Z",
    "h": "Hadamard",
    "cx": "CNOT",
    "cz": "CZ",
    "swap": "SWAP",
    "iswap": "ISWAP",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "id": "Identity",
    "cswap": "CSWAP",
    "crx": "CRX",
    "cry": "CRY",
    "crz": "CRZ",
    "p": "PhaseShift",
    "ccx": "Toffoli",
    "qubitunitary": "QubitUnitary",
    "u1": "U1",
    "u2": "U2",
    "u3": "U3",
    "rzz": "IsingZZ",
    "ryy": "IsingYY",
    "rxx": "IsingXX",
    "s": "S",
    "t": "T",
    "sx": "SX",
    "cy": "CY",
    "ch": "CH",
    "cp": "CPhase",
    "ccz": "CCZ",
    "ecr": "ECR",
    "sdg": qml.adjoint(qml.S),
    "tdg": qml.adjoint(qml.T),
    "sxdg": qml.adjoint(qml.SX),
    "reset": qml.measure(AnyWires, reset=True),  # TODO: Improve reset support
}


def _build_qerror_op(error) -> qml.QubitChannel:
    """Builds a PennyLane error channel from a Qiksit ``QuantumError`` object.

    Args:
        error (QuantumError): Quantum error object

    Returns:
        qml.QubitChannel: an equivalent PennyLane error channel
    """
    try:
        kraus_matrices = Kraus(error).data
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Error {error} could not be converted.") from exc

    return qml.QubitChannel(K_list=kraus_matrices, wires=AnyWires)


def _build_noise_model_map(noise_model) -> Tuple[dict, dict]:
    """Builds a noise model map from a Qiskit noise model. This noise model map can be used
    to efficiently construct a PennyLane noise model.

    Args:
        noise_model (qiskit_aer.noise.NoiseModel): Qiskit's noise model

    Returns:
        (dict, dict): returns mappings for the given quantum errors and readout errors in the ``noise_model``.

    For plugin developers: noise model map tuple consists of following two (nested) mappings for
    quantum errors (qerror_dmap) and readout errors (rerror_dmap):
        * qerror_dmap: noise_operation -> wires -> target_gate

            .. code-block:: python

                qerror_dmap = {
                    noise_op1: {
                        AnyWires: [qiskit_op1, qiskit_op2],
                        (0, 1): [qiskit_op3],
                        (2,): [qiskit_op4]
                    },
                    noise_op2: {
                        AnyWires: [qiskit_op5],
                        (1, 2): [qiskit_op6, qiskit_op7]
                    }
                }

        * rerror_dmap: noise_operation -> wires -> target_measurement
    """
    qerror_dmap = defaultdict(lambda: defaultdict(list))

    # Add default quantum errors
    for gate_name, error in noise_model._default_quantum_errors.items():
        noise_op = _build_qerror_op(error)
        qerror_dmap[noise_op][AnyWires].append(qiskit_op_map[gate_name])

    # Add specific qubit errors
    for gate_name, qubit_dict in noise_model._local_quantum_errors.items():
        for qubits, error in qubit_dict.items():
            noise_op = _build_qerror_op(error)
            qerror_dmap[noise_op][qubits].append(qiskit_op_map[gate_name])

    # TODO: Add support for the readout error
    rerror_dmap = defaultdict(lambda: defaultdict(list))
    if noise_model._default_readout_error or noise_model._local_readout_errors:
        warn("Readout errors are not supported currently and will be skipped.")

    return qerror_dmap, rerror_dmap
