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

This module contains functions for converting Qiskit QuantumCircuit objects
into PennyLane circuit templates.
"""
import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.exceptions import QiskitError

import pennylane as qml
import pennylane.ops.qubit as pennylane_ops
from pennylane_qiskit.qiskit_device import QISKIT_OPERATION_MAP

# pylint: disable=too-many-instance-attributes

inv_map = {v.__name__: k for k, v in QISKIT_OPERATION_MAP.items()}


def _check_parameter_bound(param):
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


def _check_circuit_and_bind_parameters(quantum_circuit: QuantumCircuit, params: dict) -> QuantumCircuit:
    """Utility function for checking for a valid quantum circuit and then binding parameters.

    Args:
        quantum_circuit (QuantumCircuit): the quantum circuit to check and bind the parameters for
        params (dict): dictionary of the parameters in the circuit

    Returns:
        qc (QuantumCircuit): quantum circuit with bound parameters
    """
    if not isinstance(quantum_circuit, QuantumCircuit):
        raise ValueError("The circuit {} is not a valid Qiskit QuantumCircuit.".format(quantum_circuit))

    if params is None:
        return quantum_circuit

    for k, v in params.items():
        if isinstance(v, qml.variable.Variable):
            params.update({k: v.val})

    return quantum_circuit.bind_parameters(params)


def map_wires(wires: list, qc_wires: list) -> dict:
    """Utility function mapping the wires specified in a quantum circuit with the wires
    specified by the user for the template.

    Args:
        wires (list): wires specified for the template
        qc_wires (list): wires from the converted quantum circuit

    Returns:
        wire_map (dict): map from quantum circuit wires to the user defined wires
    """
    if wires is None:
        return dict(zip(qc_wires, range(len(qc_wires))))

    if len(qc_wires) == len(wires):
        return dict(zip(qc_wires, wires))

    raise qml.QuantumFunctionError("The specified number of wires - {} - does not match "
                                   "the number of wires the loaded quantum circuit acts on.".format(len(wires)))


def execute_supported_operation(operation_name: str, parameters: list, wires: list):
    """Utility function that executes an operation that is natively supported by PennyLane.

    Args:
        operation_name (str): wires specified for the template
        parameters (str): parameters of the operation that will be executed
        wires (list): wires of the operation
    """
    operation = getattr(pennylane_ops, operation_name)

    parameters = [_check_parameter_bound(param) for param in parameters]

    if not parameters:
        operation(wires=wires)
    elif operation_name == 'QubitStateVector':
        operation(np.array(parameters), wires=wires)
    elif operation_name == 'QubitUnitary':
        operation(*parameters, wires=wires)
    else:
        float_params = [float(param) for param in parameters]
        operation(*float_params, wires=wires)


def load(quantum_circuit: QuantumCircuit):
    """Loads a PennyLane template from a Qiskit QuantumCircuit.
    Warnings are created for each of the QuantumCircuit instructions that were
    not incorporated in the PennyLane template.

    Args:
        quantum_circuit (qiskit.QuantumCircuit): the QuantumCircuit to be converted

    Returns:
        function: the resulting PennyLane template
    """

    def _function(params: dict = None, wires: list = None):
        """Returns a PennyLane template created based on the input QuantumCircuit.
        Warnings are created for each of the QuantumCircuit instructions that were
        not incorporated in the PennyLane template.

        Args:
            params (dict): specifies the parameters that need to be bound in the QuantumCircuit
            wires (Sequence[int] or int): The wires the converted template acts on.
                Note that if the original QuantumCircuit acted on :math:`N` qubits,
                then this must be a list of length :math:`N`.

        Returns:
            function: the new PennyLane template
        """

        qc = _check_circuit_and_bind_parameters(quantum_circuit, params)

        # Wires from a qiskit circuit are unique w.r.t. a register name and a qubit index
        qc_wires = [(q.register.name, q.index) for q in quantum_circuit.qubits]

        wire_map = map_wires(wires, qc_wires)

        # Processing the dictionary of parameters passed
        for op in qc.data:

            instruction_name = op[0].__class__.__name__

            operation_wires = [wire_map[(qubit.register.name, qubit.index)] for qubit in op[1]]

            if instruction_name in inv_map and inv_map[instruction_name] in pennylane_ops.ops:

                execute_supported_operation(inv_map[instruction_name], op[0].params, operation_wires)

            elif instruction_name == 'SdgGate':

                sgate = getattr(pennylane_ops, 'S')
                sgate(wires=operation_wires).inv()

            elif instruction_name == 'TdgGate':

                tgate = getattr(pennylane_ops, 'T')
                tgate(wires=operation_wires).inv()

            else:
                try:
                    operation_matrix = op[0].to_matrix()
                    pennylane_ops.QubitUnitary(operation_matrix, wires=operation_wires)
                except (AttributeError, QiskitError):
                    warnings.warn(__name__ + ": The {} instruction is not supported by PennyLane,"
                                             " and has not been added to the template.".
                                  format(instruction_name),
                                  UserWarning)

    return _function


def load_qasm(qasm_string: str):
    """Loads a PennyLane template from a QASM string.
        Args:
            qasm_string (str): the name of the QASM string
        Returns:
            function: the new PennyLane template
    """
    return load(QuantumCircuit.from_qasm_str(qasm_string))


def load_qasm_from_file(file: str):
    """Loads a PennyLane template from a QASM file.
        Args:
            file (str): the name of the QASM file
        Returns:
            function: the new PennyLane template
    """
    return load(QuantumCircuit.from_qasm_file(file))
