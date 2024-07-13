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
import itertools as it
from collections import defaultdict
from functools import lru_cache, reduce
from typing import List, Tuple, Union
from warnings import warn

import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires
from qiskit.quantum_info.operators.channel import Choi, Kraus

# pylint:disable = protected-access

kraus_indice_map = {
    "PhaseDamping": {(0, 0), (0, 3), (3, 0), (3, 3)},
    "AmplitudeDamping": {(0, 0), (0, 3), (2, 2), (3, 0), (3, 3)},
    "ThermalRelaxation": {(0, 0), (0, 3), (1, 1), (2, 2), (3, 0), (3, 3)},
}
pauli_error_map = {"X": "BitFlip", "Z": "PhaseFlip", "Y": "PauliError"}
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
default_option_map = {"decimals": 10, "atol": 1e-8, "rtol": 1e-5}


def _kraus_to_choi(krau_op: Kraus, optimize=False) -> np.ndarray:
    r"""Transforms Kraus representation of a channel to its Choi representation.

    Quantum channels are generally defined by Kraus operators :math:`{K_i}`, which
    unfortunately do not provide a unique description of the channel. In contrast,
    the Choi matrix (\Lambda) computed from any such Kraus representation will
    always be unique and can be used unambiguosly to obtain represent a channel.

    .. math::

        \Lambda = \sum_{i, j} \vert i \rangle \langle j \vert \otimes \sum_k K_k \vert i \rangle \langle j K_k^\dagger

    Args:
        krau_op (Kraus): A Qiskit's Kraus operator that defines a channel.
        optimize (bool): Use intermediate ``einsum`` optimization.

    Returns:
        Choi matrix of the quantum channel defined by given Kraus operators.

    For plugin developers: This has a runtime cost of :math:`O(\#K * D^4)`, where :math:`\#K` are the
    number of Kraus operators, and :math:`D` is the dimensions of the transformed Hilbert space.
    """
    kraus_l, kraus_r = krau_op._data
    kraus_vecs1 = np.array([kraus.ravel(order="F") for kraus in kraus_l])
    kraus_vecs2 = kraus_vecs1
    if kraus_r is not None:  # pragma: no cover
        kraus_vecs2 = np.array([kraus.ravel(order="F") for kraus in kraus_r])
    return np.einsum("ij,ik->jk", kraus_vecs1, kraus_vecs2.conj(), optimize=optimize)


def _process_kraus_ops(
    kraus_mats: List[np.ndarray], **kwargs
) -> Tuple[bool, str, Union[float, np.ndarray, Kraus]]:
    """Checks parity for Qiskit's Kraus operations to the existing PennyLane channels.

    This helper method constructs an unique representation of the quantum channel via its Choi matrix and
    then uses it to map them to the following existing PennyLane Channels such as ``PhaseDamping``,
    ``AmplitudeDamping``, ``ThermalRelaxation`` and ``QubitChannel``.

    Args:
        kraus_mats (List(tensor)): list of Kraus operators defining a quantum channel
    """
    choi_matrix = _kraus_to_choi(Kraus(kraus_mats), optimize=kwargs.get("optimize", False))

    kdata = None
    if qml.math.shape(choi_matrix) == (4, 4):  # PennyLane channels are single-qubit
        decimals, atol, rtol = tuple(
            kwargs.get("options", {}).get(opt, dflt) for (opt, dflt) in default_option_map.items()
        )

        non_zero_indices = np.nonzero(choi_matrix.round(decimals))
        nz_indice = set(map(tuple, zip(*non_zero_indices)))
        nz_values = choi_matrix[non_zero_indices]

        # Note: Inequality here is to priortize thermal-relaxation errors over damping errors.
        if len(nz_values) <= 6 and nz_indice.issubset(kraus_indice_map["ThermalRelaxation"]):
            kdata = _process_thermal_relaxation(
                choi_matrix, decimals=decimals, atol=atol, rtol=rtol, **kwargs
            )
            if kwargs.get("thermal_relaxation", True) and kdata is not None:
                return kdata

        if len(nz_values) == 5 and nz_indice.issubset(kraus_indice_map["AmplitudeDamping"]):
            if np.allclose(
                [nz_values[0], sum(nz_values[[2, 4]])], np.ones(2), rtol=rtol, atol=atol
            ) and np.allclose(nz_values[[1, 3]], np.sqrt(nz_values[4]), rtol=rtol, atol=atol):
                return (True, "AmplitudeDamping", np.round(nz_values[2], decimals).real)

        if len(nz_values) == 4 and nz_indice.issubset(kraus_indice_map["PhaseDamping"]):
            if np.allclose(nz_values[[0, 3]], np.ones(2), rtol=rtol, atol=atol) and np.allclose(
                *nz_values[[1, 2]], rtol=rtol, atol=atol
            ):
                return (True, "PhaseDamping", np.round(1 - nz_values[1] ** 2, decimals).real)

    return (False, "QubitChannel", Kraus(Choi(choi_matrix)).data) if not kdata else kdata


def _extract_gate_time(gate_data: tuple[tuple[int], float], gate_wires: int) -> float:
    """Helper method to extract gate time for a quantum error"""
    tg = 1.0
    if gate_data is not None:
        gate_data[0] = (gate_data[0],) if isinstance(int, gate_data[0]) else gate_data[0]
        if gate_wires in gate_data[0]:
            tg = gate_data[1]
    return tg


def _process_thermal_relaxation(choi_matrix, **kwargs):
    """Computes parameters for thermal relaxation error from a Choi matrix of Kraus matrices"""
    nt_values = choi_matrix[tuple(zip(*sorted(kraus_indice_map["ThermalRelaxation"])))]
    decimals, atol, rtol = tuple(map(kwargs.get, default_option_map))

    kdata = None
    if np.allclose(
        nt_values[[(0, 2), (3, 5)]].sum(axis=1), 1.0, rtol=rtol, atol=atol
    ) and np.isclose(*nt_values[[1, 4]], rtol=rtol, atol=atol):
        tg = _extract_gate_time(
            gate_data=kwargs.get("gate_times", {}).get(kwargs.get("gate_name", None), None),
            gate_wires=kwargs.get("gate_wires", None),
        )
        if np.isclose(nt_values[[2, 3]].sum(), 0.0, rtol=rtol, atol=atol):
            pe, t1 = 0.0, np.inf
        else:
            pe = nt_values[2] / (nt_values[2] + nt_values[3])
            t1 = -tg / np.log(1 - (nt_values[3] + nt_values[2]))
        t2 = (
            np.inf
            if np.isclose(nt_values[1], 1.0, rtol=rtol, atol=atol)
            else (-tg / np.log(nt_values[1]))
        )
        kdata = (True, "ThermalRelaxationError", np.round([pe, t1, t2, tg], decimals).real)

    return kdata


@lru_cache
def _generate_product(items: Tuple, repeat: int = 1, matrix: bool = False) -> Tuple:
    """Helper method to generate product for Pauli terms and matrices efficiently."""
    return tuple(
        it.product(
            (
                items
                if not matrix
                else tuple(map(qml.matrix, tuple(getattr(qml, i)(0) for i in items)))
            ),
            repeat=repeat,
        )
    )  # TODO: Analyze speed gains with sparse matrices


def _process_depolarization(error_dict: dict, multi_pauli: bool = False) -> dict:
    """Checks parity for Qiskit's depolarization channel to that of PennyLane.

    Args:
        error_dict (dict): error dictionary for the quantum error
        multi_pauli (bool): accept multi-qubit Pauli errors

    Returns:
        dict: An updated error dictionary based on parity with depolarization channel.
    """
    error_wires, num_wires = error_dict["wires"], len(error_dict["wires"][0])

    if (
        len(set(error_dict["probs"])) == 2
        and set(error_wires) == {error_wires[0]}
        and (num_wires == 1 or multi_pauli)
        and list(it.chain(*error_dict["data"]))
        == ["".join(pauli) for pauli in _generate_product(("I", "X", "Y", "Z"), repeat=num_wires)]
    ):
        num_terms = 4**num_wires
        id_factor = num_terms / (num_terms - 1)
        prob_iden = error_dict["probs"][error_dict["data"].index(["I" * num_wires])]
        param = id_factor * (1 - prob_iden)
        error_dict["name"] = "DepolarizingChannel"
        error_dict["data"] = param / id_factor
        error_dict["probs"] = param
        return error_dict

    error_dict["name"] = "QubitChannel"
    kraus_ops = [
        reduce(np.kron, prod, 1.0)
        for prod in _generate_product(("I", "X", "Y", "Z"), repeat=num_wires, matrix=True)
    ]
    error_dict["data"] = [
        np.sqrt(prob) * kraus_op for prob, kraus_op in zip(error_dict["probs"], kraus_ops)
    ]
    return error_dict


def _process_reset(error_dict: dict, **kwargs) -> dict:
    """Checks parity of a qunatum error with ``Reset`` instruction to a PennyLane Channel.

    Args:
        error_dict (dict): error dictionary for the quantum error
        **kwargs: optional keyword arguments

    Returns:
        dict: An updated error dictionary based on parity with existing PennyLane channel.
    """
    error_probs = error_dict["probs"]

    if "Z" not in error_dict["name"]:
        error_dict["name"] = "ResetError"
        error_dict["data"] = error_probs[1:] + ([0.0] if len(error_probs[1:]) == 1 else [])
    else:  # uses t1 > t2
        error_dict["name"] = "ThermalRelaxationError"
        tg = _extract_gate_time(
            gate_data=kwargs.get("gate_times", {}).get(kwargs.get("gate_name", None), None),
            gate_wires=kwargs.get("gate_wires", None),
        )
        p0 = 1.0 if len(error_probs) == 3 else error_probs[2] / (error_probs[2] + error_probs[3])
        t1 = -tg / np.log(1 - error_probs[2] / p0)
        t2 = (1 / t1 - np.log(1 - 2 * error_probs[1] / (1 - error_probs[2] / p0)) / tg) ** -1
        decimals = kwargs.get("options", {}).get("decimals", default_option_map["decimals"])
        error_dict["data"] = list(np.round([1 - p0, t1, t2, tg], decimals=decimals))

    return error_dict


def _build_qerror_dict(error) -> dict[str, Union[float, int]]:
    """Builds error dictionary for post-processing from Qiskit's error object.

    Args:
        error (QuantumError): Quantum error object

    Returns:
        Tuple[bool, float]: A tuple representing whether the encountered quantum error
            is a depolarization error and the related probability.

    For plugin developers: the build dictionary representation help stores the following:
        * name - Qiskit's standard name for the encountered quantum error.
        * wires - Wires on which the error acts.
        * data - Data from the quantum error required by PennyLane for reconstruction.
        * probs - Probabilities for the instructions in a quantum error.
    """
    error_repr = error.to_dict()
    error_insts, error_probs = error_repr["instructions"], error_repr["probabilities"]
    if len(error_insts) != len(error_probs):  # pragma: no cover
        raise ValueError(
            f"Mismatch between instructions and provided probabilities, got {error_insts} and {error_probs}"
        )

    error_dict = {"name": [], "wires": [], "data": [None] * len(error_insts), "probs": error_probs}

    for idx, einst in enumerate(error_insts):
        e_name, e_wire, e_data = [], [], []
        for inst in einst:
            inst_name = inst["name"]
            e_name.append(inst_name[0].upper() if inst_name in ["id", "x", "y", "z"] else inst_name)
            e_wire.append(tuple(inst["qubits"]))
            e_data.append(error_probs[idx])

            if inst_name in ["pauli", "kraus", "unitary"]:
                e_data[-1] = inst["params"]

        if len(e_name) > 1 and e_name != ["reset", "X"]:
            error_dict["name"] = ["kraus"]
            error_dict["wires"] = [tuple(einst[0]["qubits"])]
            error_dict["data"] = [Kraus(error).data]
            break

        error_dict["name"].extend(e_name)
        error_dict["wires"].extend(e_wire)
        error_dict["data"][idx] = e_data[0]

    return error_dict


def _process_qerror_dict(error_dict: dict) -> dict[str, Union[float, int]]:
    """Helper method for post processing error dictionary for constructing PennyLane Channel."""
    if error_dict["name"] == "PauliError":
        error_dict["data"] = [error_dict["data"], error_dict["probs"]]

    if error_dict["name"] == "QubitChannel":
        error_dict["data"] = [error_dict["data"]]

    if not hasattr(error_dict["data"], "__iter__"):
        error_dict["data"] = [error_dict["data"]]

    if error_dict["wires"]:
        error_dict["wires"] = error_dict["wires"][0]

    error_dict.pop("probs", None)
    return error_dict


def _build_qerror_op(error, **kwargs) -> qml.operation.Operation:
    """Builds an PennyLane error operation from a Qiksit's QuantumError object.

    Args:
        error (QuantumError): Quantum error object
        kwargs: Optional keyword arguments used during conversion

    Returns:
        qml.operation.Channel: converted PennyLane quantum channel which is
            theoretically equivalent to the given Qiksit's QuantumError object
    """
    error_dict = _build_qerror_dict(error)

    error_probs = error_dict["probs"]
    sorted_name = sorted(error_dict["name"])

    if sorted_name[0] == "I" and len(sorted_name) == 2 and sorted_name[1] in ["X", "Y", "Z"]:
        prob_pauli = error_dict["probs"][error_dict["name"].index(sorted_name[1])]
        error_dict["name"] = pauli_error_map[sorted_name[1]]
        error_dict["data"] = prob_pauli if error_dict["name"] != "PauliError" else sorted_name[1]
        error_dict["probs"] = prob_pauli

    elif sorted_name == ["I", "X", "Y", "Z"]:
        error_dict["data"] = [["I"], ["X"], ["Y"], ["Z"]]
        error_dict = _process_depolarization(error_dict)

    elif set(sorted_name) == {"pauli"}:
        error_dict = _process_depolarization(error_dict, kwargs.get("multi_pauli", False))

    elif sorted_name[0] == "I" and sorted_name[-1] == "reset":
        error_dict = _process_reset(error_dict, **kwargs)

    elif sorted_name[0] == "kraus" and len(sorted_name) == 1:
        kflag, kname, kdata = _process_kraus_ops(error_dict["data"][0], **kwargs)
        error_dict["name"] = kname if kflag else "QubitChannel"
        error_dict["data"] = kdata

    elif "unitary" in sorted_name and (
        len(set(sorted_name)) == 1 or set(sorted_name) == {"unitary", "I"}
    ):
        error_dict["name"] = "QubitChannel"
        kraus_ops = [
            op[0] if isinstance(op, list) else qml.I(0).matrix() for op in error_dict["data"]
        ]
        error_dict["data"] = [
            np.sqrt(prob) * kraus_op for prob, kraus_op in zip(error_probs, kraus_ops)
        ]

    else:  # pragma: no cover
        raise ValueError(f"Error {error} could not be converted.")

    error_dict = _process_qerror_dict(error_dict=error_dict)
    return getattr(qml.ops, error_dict["name"])(*error_dict["data"], wires=AnyWires)


def _build_noise_model_map(noise_model, **kwargs) -> Tuple[dict, dict]:
    """Builds noise model maps from which noise model can be constructed efficiently.

    Args:
        noise_model (qiskit_aer.noise.NoiseModel): Qiskit's noise model
        kwargs: Optional keyword arguments for providing extra information

    Keyword Arguments:
        thermal_relaxation (bool): prefer conversion of ``QiskitErrors`` to thermal relaxation errors
            over damping errors. Default is ``False``.
        gate_times (Dict[Tuple(str, Tuple[int]), float]): a dictionary to provide gate times for building
            thermal relaxation error. Each key will be a tuple of instruction name and qubit indices and
            the corresponding value will be the time in seconds. If it is not provided or is incomplete,
            a default value of `1.0 s`` will be used for the specific constructions.
        multi_pauli (bool): assume depolarization channel to be multi-qubit. This is currently not
            supported with ``qml.DepolarizationChannel``, which is a single qubit channel.
        readout_error (bool): include readout error in the converted noise model. Default is ``True``.
        optimize (bool): controls if a contraction order optimization is used for ``einsum`` while
            transforming Kraus operators to a Choi matrix, wherever required. Default is ``False``.
        options (dict[str, Union[int, float]]): optional parameters related to tolerance and rounding:

            - decimals (int): number of decimal places to round the Kraus matrices. Default is ``10``.
            - atol (float): the relative tolerance parameter. Default value is ``1e-05``.
            - rtol (float): the absolute tolernace parameters. Defualt value is ``1e-08``.

    Returns:
        (dict, dict): returns mappings for ecountered quantum errors and readout errors.

    For plugin developers: noise model map tuple consists of following two mappings:
        * qerror_dmap: noise_operation -> wires -> target_gate
        * rerror_dmap: noise_operation -> wires -> target_measurement
    """
    qerror_dmap = defaultdict(lambda: defaultdict(list))

    # Add default quantum errors
    for gate_name, error in noise_model._default_quantum_errors.items():
        noise_op = _build_qerror_op(error, gate_name=gate_name, **kwargs)
        qerror_dmap[noise_op][AnyWires].append(qiskit_op_map[gate_name])

    # Add specific qubit errors
    for gate_name, qubit_dict in noise_model._local_quantum_errors.items():
        for qubits, error in qubit_dict.items():
            noise_op = _build_qerror_op(error, gate_name=gate_name, gate_wires=qubits, **kwargs)
            qerror_dmap[noise_op][qubits].append(qiskit_op_map[gate_name])

    # TODO: Add support for the readout error
    rerror_dmap = defaultdict(lambda: defaultdict(list))
    if kwargs.get("readout_error", True):
        if noise_model._default_readout_error or noise_model._local_readout_errors:
            warn("Readout errors are not supported currently and will be skipped.")

    return qerror_dmap, rerror_dmap
