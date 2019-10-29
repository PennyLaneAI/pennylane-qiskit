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
Base Qiskit device class
========================

.. currentmodule:: pennylane_qiskit.qiskit_device

This module contains a base class for constructing Qiskit devices for PennyLane.

Classes
-------

.. autosummary::
   QiskitDevice

Code details
~~~~~~~~~~~~
"""
from qiskit import QuantumCircuit
from pennylane_qiskit.qiskit_device import QISKIT_OPERATION_MAP
import pennylane.operation as ops

# pylint: disable=too-many-instance-attributes


class QuantumCircuitToTemplateConverter:

    def __init__(self, quantum_circuit: QuantumCircuit):
        self.circuit = quantum_circuit

        # Check for all the operations that are valid as per the conversion

        # unsupported gates which are essential to the quantum algorithm should instead raise an error


_operation_map = QISKIT_OPERATION_MAP
inv_map = {v.__name__: k for k, v in _operation_map.items()}
a = 3


# Defining the load function
def qiskit_to_pennylane(quantum_circuit: QuantumCircuit, params: dict = None):

    if not isinstance(quantum_circuit, QuantumCircuit):
        raise ValueError("The circuit {} is not a valid Qiskit QuantumCircuit.".format(quantum_circuit))

    operator_queue = []

    # TODO:
    # Check if parameters are needed -> if so, bind them
    # Check if parameters are needed -> if not passed, raise ERROR
    # Check if parameters are not needed -> if not passed, go on
    # Check if parameters are not needed -> if passed, , raise WARNING

    if len(quantum_circuit.parameters) > 0:
        if params is not None:
            quantum_circuit = quantum_circuit.bind_parameters(params)
        else:
            raise ValueError("Parameters required for circuit {}.".format(quantum_circuit.name))
    else:
        if params is not None:
            print("Parameters were passed although the  circuit {} does not require them.".format(quantum_circuit.name))

    # Processing the dictionary of parameters passed
    for op in quantum_circuit.data:

        #TODO:
        # What if multiple quantum registers are specified
        # Indexing a Qubit object to obtain a wire

        wires = [qubit.index for qubit in op[1]]
        operation = getattr(ops, op[0].name)

        # Call each operator with the specified parameters
        parameters = op[0].params

        if not parameters:
            operation(wires=wires)
        else:
            operation(*parameters, wires=wires)

    return operator_queue

def load(quantum_circuit: QuantumCircuit):
    queue = qml.converter()
    template = qiskit_to_pennylane(quantum_circuit)


    #[docs]    @staticmethod
    #def from_qasm_file(path):
    #    """Take in a QASM file and generate a QuantumCircuit object.

    #    Args:
    #      path (str): Path to the file for a QASM program
    #    Return:
    #      QuantumCircuit: The QuantumCircuit object for the input QASM
    #    """
    #    qasm = Qasm(filename=path)
    #    return _circuit_from_qasm(qasm)

    def _function(params, wires):

            pennylane_operator = inv_map[operator_name]
            result = getattr(qml.ops, pennylane_operator)()
            # First item of the list contains the gate

            queue =

            # Second the wires it acts on



            # convert operation to PennyLane operation
            # pass parameter dictionary & wire
    return _function

