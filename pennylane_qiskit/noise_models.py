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

import numpy as np
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


def _build_qerror_op(error, **kwargs) -> qml.QubitChannel:
    """Builds an PennyLane error operation from a Qiksit's QuantumError object.

    Args:
        error (QuantumError): Quantum error object
        kwargs: Optional keyword arguments used for conversion.

    Returns:
        qml.QubitChannel: converted PennyLane quantum channel which is
            equivalent to the given Qiksit's ``QuantumError`` object
    """
    try:
        kraus_matrices = np.round(Kraus(error).data, decimals=kwargs.get("decimals", 10))
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Error {error} could not be converted.") from exc

    return qml.QubitChannel(K_list=kraus_matrices, wires=AnyWires)


def _build_noise_model_map(noise_model, **kwargs) -> Tuple[dict, dict]:
    """Builds noise model maps from which noise model can be constructed efficiently.

    Args:
        noise_model (qiskit_aer.noise.NoiseModel): Qiskit's noise model
        kwargs: Optional keyword arguments for providing extra information

    Keyword Arguments:
        quantum_error (bool): include quantum errors in the converted noise model. Default is ``True``.
        readout_error (bool): include readout errors in the converted noise model. Default is ``True``.
        decimals (int): number of decimal places to round the Kraus matrices. Default is ``10``.

    Returns:
        (dict, dict): returns mappings for ecountered quantum errors and readout errors.

    For plugin developers: noise model map tuple consists of following two mappings:
        * qerror_dmap: noise_operation -> wires -> target_gate
        * rerror_dmap: noise_operation -> wires -> target_measurement
    """
    qerror_dmap = defaultdict(lambda: defaultdict(list))

    # Add default quantum errors
    if kwargs.get("quantum_error", True):
        for gate_name, error in noise_model._default_quantum_errors.items():
            noise_op = _build_qerror_op(error, **kwargs)
            qerror_dmap[noise_op][AnyWires].append(qiskit_op_map[gate_name])

        # Add specific qubit errors
        for gate_name, qubit_dict in noise_model._local_quantum_errors.items():
            for qubits, error in qubit_dict.items():
                noise_op = _build_qerror_op(error, **kwargs)
                qerror_dmap[noise_op][qubits].append(qiskit_op_map[gate_name])

    # TODO: Add support for the readout error
    rerror_dmap = defaultdict(lambda: defaultdict(list))
    if kwargs.get("readout_error", True):
        if noise_model._default_readout_error or noise_model._local_readout_errors:
            warn("Readout errors are not supported currently and will be skipped.")

    return qerror_dmap, rerror_dmap
