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
from qiskit_aer.noise import NoiseModel, QuantumError

# pylint:disable = protected-access

kraus_indice_map = {
    "PhaseDamping": ((0, 0, 3, 3), (0, 3, 0, 3)),
    "AmplitudeDamping": ((0, 0, 2, 3, 3), (0, 3, 2, 0, 3)),
    "ThermalRelaxationError": ((0, 0, 1, 2, 3, 3), (0, 3, 1, 2, 0, 3)),
}
pauli_error_map = {"X": "BitFlip", "Z": "PhaseFlip", "Y": "PauliError"}
qiskit_op_map = {
    "cx": "CNOT",
    "sx": "SX",
    "id": "I",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "reset": qml.measure(AnyWires, reset=True),  # TODO: Improve reset support
}
default_option_map = [("decimals", 10), ("atol", 1e-8), ("rtol", 1e-5), ("optimize", False)]


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
    if kraus_r is not None:
        kraus_vecs2 = np.array([kraus.ravel(order="F") for kraus in kraus_r])
    return np.einsum("ij,ik->jk", kraus_vecs1, kraus_vecs2.conj(), optimize=optimize)


def _check_kraus_ops(
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

    if qml.math.shape(choi_matrix) == (4, 4):  # PennyLane channels are single-qubit
        decimals, atol, rtol = tuple(kwargs.get(opt, dflt) for (opt, dflt) in default_option_map)

        non_zero_indices = np.nonzero(choi_matrix.round(decimals))
        nz_values = choi_matrix[non_zero_indices]

        if len(nz_values) == 4 and np.allclose(
            non_zero_indices,
            kraus_indice_map["PhaseDamping"],
            rtol=rtol,
            atol=atol,
        ):
            if np.allclose(nz_values[[0, 3]], np.ones(2)) and np.isclose(*nz_values[[1, 2]]):
                return (True, "PhaseDamping", np.round(1 - nz_values[1] ** 2, decimals).real)

        if len(nz_values) == 5 and np.allclose(
            non_zero_indices,
            kraus_indice_map["AmplitudeDamping"],
            rtol=rtol,
            atol=atol,
        ):
            if np.allclose(
                [nz_values[0], sum(nz_values[[2, 4]])], np.ones(2), rtol=rtol, atol=atol
            ) and np.isclose(*nz_values[[1, 3]], rtol=rtol, atol=atol):
                return (True, "AmplitudeDamping", np.round(nz_values[2], decimals).real)

        if len(nz_values) == 6 and np.allclose(
            non_zero_indices,
            kraus_indice_map["ThermalRelaxationError"],
            rtol=rtol,
            atol=atol,
        ):
            if np.allclose(
                [sum(nz_values[[0, 2]]), sum(nz_values[[3, 5]])],
                np.ones(2),
                rtol=rtol,
                atol=atol,
            ) and np.isclose(
                *nz_values[[1, 4]], rtol=rtol, atol=atol
            ):  # uses t2 > t1
                tg = kwargs.get("gate_times", {}).get(kwargs["gate_name"], 1.0)
                pe = nz_values[2] / (nz_values[2] + nz_values[3])
                t1 = -tg / np.log(1 - nz_values[2] / pe)
                t2 = -tg / np.log(nz_values[1])
                return (True, "ThermalRelaxationError", np.round([pe, t1, t2, tg], decimals).real)

    return (False, "QubitChannel", Kraus(Choi(choi_matrix)).data)


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


def _check_depolarization(error_dict: dict) -> Tuple[bool, float]:
    """Checks parity for Qiskit's depolarization channel to that of PennyLane.

    Args:
        error_dict (dict): error dictionary for the quantum error

    Returns:
        (bool, float): A tuple representing whether the encountered quantum error
            is a depolarization error and the related probability.
    """
    error_wires, num_wires = error_dict["wires"], len(error_dict["wires"][0])

    if (
        len(set(error_dict["probs"])) == 2
        and set(error_wires) == {error_wires[0]}
        and list(it.chain(*error_dict["data"]))
        == ["".join(pauli) for pauli in _generate_product(("I", "X", "Y", "Z"), repeat=num_wires)]
    ):
        num_terms = 4**num_wires
        id_factor = num_terms / (num_terms - 1)
        prob_iden = error_dict["probs"][error_dict["data"].index(["I" * num_wires])]
        param = id_factor * (1 - prob_iden)
        return (True, param)

    return (False, 0.0)


def _build_qerror_dict(error: QuantumError) -> dict[str, Union[float, int]]:
    """Builds error dictionary for post-processing from Qiskit's error object.

    Args:
        error (dict): error dictionary for the quantum error

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


def _build_qerror_op(error: QuantumError, **kwargs) -> qml.operation.Operation:
    """Builds an PennyLane error operation from a Qiksit's QuantumError object.

    Args:
        error (QuantumError): Quantum error object
        kwargs: Optional keyword arguments used during conversion

    Returns:
        qml.operation.Channel: converted PennyLane quantum channel which is
            theoretically equivalent to the given Qiksit's QuantumError object
    """
    error_dict = _build_qerror_dict(error)

    error_wires, num_wires = error_dict["wires"], len(error_dict["wires"][0])
    error_probs = error_dict["probs"]
    sorted_name = sorted(error_dict["name"])

    if sorted_name[0] == "I" and sorted_name[1:] in ["X", "Y", "Z"] and len(sorted_name) == 2:
        prob_pauli = error_dict["probs"][error_dict["name"].index(sorted_name[1])]
        error_dict["name"] = pauli_error_map[sorted_name[1]]
        error_dict["data"] = prob_pauli
        error_dict["probs"] = prob_pauli

    elif sorted_name == ["I", "X", "Y", "Z"]:
        error_dict["data"] = [["I"], ["X"], ["Y"], ["Z"]]
        depol_flag, depol_param = _check_depolarization(error_dict)
        if depol_flag:
            error_dict["name"] = "DepolarizingChannel"
            error_dict["data"] = depol_param * 3 / 4
            error_dict["probs"] = depol_param
        else:
            error_dict["name"] = "QubitChannel"
            kraus_ops = [
                reduce(lambda mat1, mat2: mat1 @ mat2, prod, np.eye(int(2**num_wires)))
                for prod in _generate_product(("I", "X", "Y", "Z"), repeat=num_wires, matrix=True)
            ]
            error_dict["data"] = [
                np.sqrt(prob) * kraus_op for prob, kraus_op in zip(error_dict["probs"], kraus_ops)
            ]

    elif set(sorted_name) == {"pauli"}:
        depol_flag, depol_param = _check_depolarization(error_dict)
        if depol_flag and len(error_wires[0]) == 1:
            error_dict["name"] = "DepolarizingChannel"
            error_dict["probs"] = depol_param
        else:
            error_dict["name"] = "QubitChannel"  # TODO: Make PauliError multi-qubit channel
            kraus_ops = [
                reduce(lambda mat1, mat2: mat1 @ mat2, prod, np.eye(int(2**num_wires)))
                for prod in _generate_product(("I", "X", "Y", "Z"), repeat=num_wires, matrix=True)
            ]
            error_dict["data"] = [
                np.sqrt(prob) * kraus_op for prob, kraus_op in zip(error_dict["probs"], kraus_ops)
            ]

    elif sorted_name[0] == "I" and sorted_name[-1] == "reset":
        if "Z" not in error_dict["name"]:
            error_dict["name"] = "ResetError"
            error_dict["data"] = error_probs[1:]
        else:  # uses t1 > t2
            error_dict["name"] = "ThermalRelaxationError"
            tg = kwargs.get("gate_times", {}).get(kwargs["gate_name"], 1.0)
            p0 = (
                1.0 if len(error_probs) == 3 else error_probs[2] / (error_probs[2] + error_probs[3])
            )
            t1 = -tg / np.log(1 - error_probs[2] / p0)
            t2 = (1 / t1 - np.log(1 - 2 * error_probs[1] / (1 - error_probs[2] / p0)) / tg) ** -1
            error_dict["data"] = list(np.round([1 - p0, t1, t2, tg], kwargs.get("decimals", 10)))

    elif sorted_name[0] == "kraus" and len(sorted_name) == 1:
        kflag, kname, kdata = _check_kraus_ops(error_dict["data"][0], **kwargs)
        error_dict["name"] = kname if kflag else "QubitChannel"
        error_dict["data"] = kdata

    elif "unitary" in sorted_name and (
        len(set(sorted_name)) == 1 or set(sorted_name) == {"unitary", "I"}
    ):
        error_dict["name"] = "QubitChannel"
        kraus_ops = [
            op[0] if isinstance(op, list) else qml.I(0).matrix() * op for op in error_dict["data"]
        ]
        error_dict["data"] = [
            np.sqrt(prob) * kraus_op for prob, kraus_op in zip(error_probs, kraus_ops)
        ]

    else:  # pragma: no cover
        raise ValueError(f"Error {error} could not be converted.")

    if error_dict["name"] == "PauliError":
        error_dict["data"] = [error_dict["data"], error_dict["probs"]]

    if error_dict["name"] == "QubitChannel":
        error_dict["data"] = [error_dict["data"]]

    if not hasattr(error_dict["data"], "__iter__"):
        error_dict["data"] = [error_dict["data"]]

    if error_dict["wires"]:
        error_dict["wires"] = error_wires[0]

    error_dict.pop("probs", None)

    return getattr(qml.ops, error_dict["name"])(*error_dict["data"], wires=AnyWires)


def _build_noise_model_map(noise_model: NoiseModel, **kwargs) -> Tuple(dict, dict):
    """Builds noise model maps from which noise model can be constructed efficiently.

    Args:
        noise_model (NoiseModel): Qiskit's noise model
        kwargs: Optional keyword arguments for providing extra information

    Keyword Arguments:
        gate_times (Dict[str, float]): gate times for building thermal relaxation error.
            If not provided, the default value of ``1.0`` will be used for construction.
        decimals: number of decimal places to round the Kraus matrices for errors to.
            If not provided, the default value of ``10`` is used.
        atol: the relative tolerance parameter. Default value is ``1e-05``.
        rtol: the absolute tolernace parameters. Defualt value is ``1e-08``.
        optimize: controls if intermediate optimization is used while transforming Kraus
            operators to a Choi matrix, wherever required. Default is ``False``.

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
            noise_op = _build_qerror_op(error, gate_name=gate_name, **kwargs)
            qerror_dmap[noise_op][qubits].append(qiskit_op_map[gate_name])

    # TODO: Add support for the readout error
    rerror_dmap = defaultdict(lambda: defaultdict(list))
    if noise_model._default_readout_error or noise_model._local_readout_errors:
        warn(f"Readout errors {error} are not supported currently and will be skipped.")

    return qerror_dmap, rerror_dmap
