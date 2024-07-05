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

from qiskit.quantum_info.operators.channel import Choi, Kraus
import pennylane as qml
import numpy as np

# pylint:disable = protected-access

kraus_indice_map = {
    "PhaseDamping": ((0, 0, 3, 3), (0, 3, 0, 3)),
    "AmplitudeDamping": ((0, 0, 2, 3, 3), (0, 3, 2, 0, 3)),
    "ThermalRelaxationError": ((0, 0, 1, 2, 3, 3), (0, 3, 1, 2, 0, 3)),
}
pauli_error_map = {"X": "BitFlip", "Z": "PhaseFlip", "Y": "PauliError"}


def _kraus_to_choi(krau_op):
    """Transform Kraus representation of a channel to its Choi representation."""
    kraus_l, kraus_r = krau_op._data
    kraus_vecs1 = np.array([kraus.ravel(order="F") for kraus in kraus_l])
    kraus_vecs2 = kraus_vecs1
    if kraus_r is not None:
        kraus_vecs2 = np.array([kraus.ravel(order="F") for kraus in kraus_r])
    return np.einsum("ij,ik->jk", kraus_vecs1, kraus_vecs2.conj())


def _check_kraus_ops(kraus_mats):
    """Checks parity for Qiskit's Kraus operations to existing PL channels."""
    choi_matrix = _kraus_to_choi(Kraus(kraus_mats))  # Kraus-independent Choi matrix

    if qml.math.shape(choi_matrix) == (4, 4):  # PennyLane channels are single-qubit

        non_zero_indices = np.nonzero(choi_matrix.round(8))
        nz_values = choi_matrix[non_zero_indices]

        if len(nz_values) == 4 and np.allclose(non_zero_indices, kraus_indice_map["PhaseDamping"]):
            if np.allclose(nz_values[[0, 3]], np.ones(2)) and np.isclose(*nz_values[[1, 2]]):
                return (True, "PhaseDamping", np.round(1 - nz_values[1] ** 2, 10).real)

        if len(nz_values) == 5 and np.allclose(
            non_zero_indices, kraus_indice_map["AmplitudeDamping"]
        ):
            if np.allclose([nz_values[0], np.sum(nz_values[[2, 4]])], np.ones(2)) and np.isclose(
                *nz_values[[1, 3]]
            ):
                return (True, "AmplitudeDamping", np.round(nz_values[2], 10).real)

        if len(nz_values) == 6 and np.allclose(
            non_zero_indices, kraus_indice_map["ThermalRelaxationError"]
        ):
            if np.allclose(
                [np.sum(nz_values[[0, 2]]), np.sum(nz_values[[3, 5]])], np.ones(2)
            ) and np.isclose(*nz_values[[1, 4]]):
                pe = nz_values[2] / (nz_values[2] + nz_values[3])
                t1 = -1 / np.log(1 - nz_values[2] / pe).real
                t2 = -1 / np.log(nz_values[1])
                return (True, "ThermalRelaxationError", np.round([pe, t1, t2, 1.0], 10).real)

    return (False, "QubitChannel", Kraus(Choi(choi_matrix)).data)


def _check_depolarization(error_dict):
    """Checks parity for Qiskit's depolarization channel to that of PennyLane."""
    error_wires, num_wires = error_dict["wires"], len(error_dict["wires"][0])

    if (
        list(it.chain(*error_dict["data"]))
        == ["".join(tup) for tup in it.product(["I", "X", "Y", "Z"], repeat=num_wires)]
        and len(set(error_dict["probs"])) == 2
        and set(error_wires) == {error_wires[0]}
    ):
        num_terms = 4**num_wires
        id_factor = num_terms / (num_terms - 1)
        prob_iden = error_dict["probs"][error_dict["data"].index(["I" * num_wires])]
        param = id_factor * (1 - prob_iden)
        return (True, param)

    return (False, 0.0)


def _build_error(error):
    """Builds an error tuple from a Qiksit's QuantumError object"""

    error_repr = error.to_dict()
    error_insts, error_probs = error_repr["instructions"], error_repr["probabilities"]
    if len(error_insts) != len(error_probs):
        raise ValueError(
            f"Mismatch between instructions and the provided probabilities, got {error_insts} and {error_probs}"
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

    error_wires, num_wires = error_dict["wires"], len(error_dict["wires"][0])
    sorted_name = sorted(error_dict["name"])

    if sorted_name[0] == "I" and sorted_name[1:] in ["X", "Y", "Z"] and len(sorted_name) == 2:
        prob_pauli = error_dict["probs"][error_dict["name"].index(sorted_name[1])]
        error_dict["name"] = pauli_error_map[sorted_name[1]]
        error_dict["data"] = prob_pauli
        error_dict["probs"] = prob_pauli
        error_dict["wires"] = error_wires[0]

    elif sorted_name == ["I", "X", "Y", "Z"]:
        error_dict["data"] = [["I"], ["X"], ["Y"], ["Z"]]
        depol_flag, depol_param = _check_depolarization(error_dict)
        if depol_flag:
            error_dict["name"] = "DepolarizingChannel"
            error_dict["data"] = depol_param * 3 / 4
            error_dict["probs"] = depol_param
        else:
            error_dict["name"] = "QubitChannel"
            kraus_ops = (
                tup[0] @ tup[1]
                for tup in it.product(
                    list(map(qml.matrix, [qml.I(0), qml.X(0), qml.Y(0), qml.Z(0)])),
                    repeat=num_wires,
                )
            )
            error_dict["data"] = [
                np.sqrt(prob) * kraus_op for prob, kraus_op in zip(error_dict["probs"], kraus_ops)
            ]
        error_dict["wires"] = error_wires[0]

    elif set(sorted_name) == {"pauli"}:
        depol_flag, depol_param = _check_depolarization(error_dict)
        if depol_flag and len(error_wires[0]) == 1:
            error_dict["name"] = "DepolarizingChannel"
            error_dict["probs"] = depol_param
        else:
            error_dict["name"] = "QubitChannel"  # "PauliError"
            kraus_ops = (
                tup[0] @ tup[1]
                for tup in it.product(
                    list(map(qml.matrix, [qml.I(0), qml.X(0), qml.Y(0), qml.Z(0)])),
                    repeat=num_wires,
                )
            )
            error_dict["data"] = [
                np.sqrt(prob) * kraus_op for prob, kraus_op in zip(error_dict["probs"], kraus_ops)
            ]
        error_dict["wires"] = error_wires[0]

    elif sorted_name[0] == "I" and sorted_name[-1] == "reset":
        if "Z" not in error_dict["name"]:
            error_dict["name"] = "ResetError"
            error_dict["data"] = error_probs[1:]
        else:
            error_dict["name"] = "ThermalRelaxationError"
            pe = (
                0.0 if len(error_probs) == 3 else error_probs[3] / (error_probs[2] + error_probs[3])
            )
            t1 = -1 / np.log(1 - error_probs[2] / (1 - pe))
            t2 = (1 / t1 - np.log(1 - 2 * error_probs[1] / (1 - error_probs[2] / (1 - pe)))) ** -1
            error_dict["data"] = list(np.round([pe, t1, t2, 1.0], 10))
        error_dict["wires"] = error_wires[0]

    elif sorted_name[0] == "kraus" and len(sorted_name) == 1:
        kflag, kname, kdata = _check_kraus_ops(error_dict["data"][0])
        error_dict["name"] = kname if kflag else "QubitChannel"
        error_dict["data"] = kdata
        error_dict["wires"] = error_wires[0]

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
        error_dict["wires"] = error_wires[0]

    else:
        error_dict = {"name": [], "wires": [], "data": [], "probs": []}
        raise Warning(f"Error {error} could not be converted and will be skipped.")

    if error_dict["name"] == "PauliError":
        error_dict["data"] = [error_dict["data"], error_dict["probs"]]

    if error_dict["name"] == "QubitChannel":
        error_dict["data"] = [error_dict["data"]]

    if not hasattr(error_dict["data"], "__iter__"):
        error_dict["data"] = [error_dict["data"]]

    error_dict.pop("probs", None)

    return error_dict


def _build_noise_model(noise_model):
    error_list = []

    # Add default quantum errors
    for name, error in noise_model._default_quantum_errors.items():
        error_dict = _build_error(error)
        error_list.append(
            {
                "operations": [name],
                "noise_op": getattr(qml.ops, error_dict["name"])(
                    *error_dict["data"], wires=error_dict["wires"]
                ),
            }
        )

    # Add specific qubit errors
    for name, qubit_dict in noise_model._local_quantum_errors.items():
        for qubits, error in qubit_dict.items():
            error_dict = _build_error(error)
            error_list.append(
                {
                    "operations": [name],
                    "gate_qubits": [qubits],
                    "noise_op": getattr(qml.ops, error_dict["name"])(
                        *error_dict["data"], wires=error_dict["wires"]
                    ),
                }
            )

    # TODO: Add support for the readout error
    if noise_model._default_readout_error or noise_model._local_readout_errors:
        raise Warning(f"Readout errors {error} are not supported currently and will be skipped.")

    return error_list
