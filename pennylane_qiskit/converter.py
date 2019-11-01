# Copyright 2019 Xanadu Quantum Technologies Inc.

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
QuanctumCircuit converter module
================================

.. currentmodule:: pennylane_qiskit.load

This module contains functions for converting Qiskit QuantumCirducit objects
PennyLane templates.
"""
import warnings

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.exceptions import QiskitError

import pennylane.ops.qubit as pennylane_ops
from pennylane_qiskit.qiskit_device import QISKIT_OPERATION_MAP

# pylint: disable=too-many-instance-attributes

inv_map = {v.__name__: k for k, v in QISKIT_OPERATION_MAP.items()}


class WiresError(Exception):
    """Exception raised by  the pennylane_qiskit.converter when it encounters an illegal
    operation in the quantum circuit.
    """


def check_parameter_bound(param):
    """Utility function determining if a certain parameter in a QuantumCircuit has
    been bound.

    Args:
        param: the parameter to be checked

    Returns:
        param: the parameter after the check
    """
    if isinstance(param, Parameter):
        raise ValueError("The parameter {} was not bound correctly.".format(param))
    return param


def load(quantum_circuit: QuantumCircuit):
    """Returns a PennyLane template created based on the input QuantumCircuit or QASM string.
    Warnings are created for each of the QuantumCircuit instructions that were
    not incorporated in the PennyLane template.

    Args:
        quantum_circuit (qiskit.QuantumCircuit): the QuantumCircuit to be converted

    Returns:
        function: the new PennyLane template
    """

    def _function(params: dict = None, wires: list = None):
        """Returns a PennyLane template created based on the input QuantumCircuit.
        Warnings are created for each of the QuantumCircuit instructions that were
        not incorporated in the PennyLane template.

        Args:
            params (dict): specifies the parameters that need to bound in the
            QuantumCircuit

            wire_shift (list): the shift to be made with respect to the wires
            already specifiec in the QuantumCircuit
            e.g. if the QuantumCircuit acts on wires [0, 1] and if wire_shift == 1,
            then the returned Pennylane template will act on wires [1, 2]

        Returns:
            function: the new PennyLane template
        """

        if not isinstance(quantum_circuit, QuantumCircuit):
            raise ValueError("The circuit {} is not a valid Qiskit QuantumCircuit.".format(quantum_circuit))

        qc = quantum_circuit.bind_parameters(params) if params is not None else quantum_circuit

        # Wires from a qiskit circuit are unique w.r.t. a register name and a qubit index
        qc_wires = [(q.register.name, q.index) for q in quantum_circuit.qubits]

        if wires is None:
            wire_map = dict(zip(qc_wires, range(len(qc_wires))))
        elif len(qc_wires) == len(wires):
            wire_map = dict(zip(qc_wires, wires))
        else:
            raise WiresError("The specified number of wires - {} - does not match "
                             "the number of wires the loaded quantum circuit acts on.".format(len(wires)))

        # Processing the dictionary of parameters passed
        for op in qc.data:

            instruction_name = op[0].__class__.__name__

            operation_wires = [wire_map[(qubit.register.name, qubit.index)] for qubit in op[1]]

            if instruction_name in inv_map and inv_map[instruction_name] in pennylane_ops.ops:

                operation_name = inv_map[instruction_name]

                operation = getattr(pennylane_ops, operation_name)

                parameters = [check_parameter_bound(param) for param in op[0].params]

                if not parameters:
                    operation(wires=operation_wires)
                elif operation_name == 'QubitStateVector':
                    operation(np.array(parameters), wires=operation_wires)
                elif operation_name == 'QubitUnitary':
                    operation(*parameters, wires=operation_wires)
                elif len(parameters) == 1:
                    operation(float(*parameters), wires=operation_wires)
                else:
                    float_params = [float(param) for param in parameters]
                    operation(*float_params, wires=operation_wires)

            else:
                try:
                    operation_matrix = op[0].to_matrix()
                    pennylane_ops.QubitUnitary(operation_matrix, wires=operation_wires)
                except (AttributeError, QiskitError):
                    warnings.warn(__name__ + " The {} instruction is not supported by PennyLane,"
                                             " and has not been added to the template.".
                                  format(instruction_name),
                                  UserWarning)

    return _function


def load_qasm_from_file(file: str):
    """Returns a PennyLane template created based on the input qasm file.

        Args:
            file (str): the name of the file

        Returns:
            function: the new PennyLane template
    """
    return load(QuantumCircuit.from_qasm_file(file))


def load_qasm(qasm_string: str):
    """Returns a PennyLane template created based on the input qasm string.

        Args:
            qasm_string (str): the name of the qasm string

        Returns:
            function: the new PennyLane template
    """
    return load(QuantumCircuit.from_qasm_str(qasm_string))
