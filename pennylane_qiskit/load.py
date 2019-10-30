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

This module contains a base class for constructing Qiskit devices for PennyLane.

Classes
-------

.. autosummary::
   QiskitDevice

Code details
~~~~~~~~~~~~
"""
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression
from pennylane_qiskit.qiskit_device import QISKIT_OPERATION_MAP
import pennylane.ops.qubit as pennylane_ops
import pennylane_qiskit.ops as plugin_ops
import warnings
import numpy as np

# pylint: disable=too-many-instance-attributes

_operation_map = QISKIT_OPERATION_MAP
inv_map = {v.__name__: k for k, v in _operation_map.items()}


def check_parameter_bound(param):
    if isinstance(param, Parameter):
        raise ValueError("The parameter {} was not bound correctly.".format(param))
    return param


def load(quantum_circuit: QuantumCircuit):
    """Returns a PennyLane template created based on the input qiskit.QuantumCircuit.
    Warnings are created for each of the QuantumCircuit instructions that were
    not incorporated in the PennyLane template.

    Args:
        quantum_circuit (qiskit.QuantumCircuit): the QuantumCircuit to be converted

    Returns:
        function: the new PennyLane template
    """

    if isinstance(quantum_circuit, str):
        quantum_circuit = QuantumCircuit._circuit_from_qasm(quantum_circuit)

    def _function(params: dict = None, wire_shift: int = 0):
        """Returns a PennyLane template created based on the input QuantumCircuit.
            Warnings are created for each of the QuantumCircuit instructions that were
            not incorporated in the PennyLane template.

            Args:
                params (dict): specifies the parameters that need to bound in the
                QuantumCircuit

                wire_shift (int): the shift to be made with respect to the wires
                already specifiec in the QuantumCircuit
                e.g. if the QuantumCircuit acts on wires [0, 1] and if wire_shift == 1,
                then the returned Pennylane template will act on wires [1, 2]

            Returns:
                function: the new PennyLane template
            """

        if not isinstance(quantum_circuit, QuantumCircuit):
            raise ValueError("The circuit {} is not a valid Qiskit QuantumCircuit.".format(quantum_circuit))

        if params is not None:
            qc = quantum_circuit.bind_parameters(params)
        else:
            qc = quantum_circuit

        # Processing the dictionary of parameters passed
        for op in qc.data:

            # Indexing a Qubit object to obtain a wire and shifting it with the start wire index
            wires = [wire_shift + qubit.index for qubit in op[1]]
            instruction_name = op[0].__class__.__name__

            print(plugin_ops)

            if instruction_name in inv_map and\
                    (inv_map[instruction_name] in pennylane_ops.ops or inv_map[instruction_name] in plugin_ops._ops):

                operation_name = inv_map[instruction_name]

                if operation_name in pennylane_ops.ops:
                    operation = getattr(pennylane_ops, operation_name)
                else:
                    operation = getattr(plugin_ops, operation_name)

                parameters = [check_parameter_bound(param) for param in op[0].params]

                if not parameters:
                    operation(wires=wires)
                elif operation_name == 'QubitStateVector':
                    operation(np.array(parameters), wires=wires)
                elif operation_name == 'QubitUnitary':
                    operation(*parameters, wires=wires)
                elif len(parameters) == 1:
                    operation(float(*parameters), wires=wires)
                else:
                    float_params = [float(param) for param in parameters]
                    operation(*float_params, wires=wires)

            else:
                warnings.warn(__name__ + " The {} instruction is not supported by PennyLane.".
                              format(instruction_name),
                              UserWarning)
    return _function


def load_qasm_from_file(file: str):
    return load(QuantumCircuit.from_qasm_file(file))


def load_qasm(string: str):
    return load(QuantumCircuit.from_qasm_str(string))
