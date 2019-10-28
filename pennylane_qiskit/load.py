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

# pylint: disable=too-many-instance-attributes


class QuantumCircuitToTemplateConverter:

    def __init__(self, quantum_circuit: QuantumCircuit):
        self.circuit = quantum_circuit

        # Check for all the operations that are valid as per the conversion

        # unsupported gates which are essential to the quantum algorithm should instead raise an error



_operation_map = {
    'BasisState': None,
    'QubitStateVector': None,
    'QubitUnitary': unitary,
    'PauliX': X,
    'PauliY': Y,
    'PauliZ': Z,
    'h': 'Hadamard',
    'S': S,
    'T': T,
    'CNOT': CNOT,
    'SWAP': SWAP,
    'CSWAP':CSWAP,
    'CZ': CZ,
    'PhaseShift': Rphi,
    'RX': Rotx,
    'RY': Roty,
    'RZ': Rotz,
    'Rot': Rot3,
    'CRX': CRotx,
    'CRY': CRoty,
    'CRZ': CRotz,
    'CRot': CRot3
    'unitary': 'Operation'
}

_observable_map = {
    'PauliX': X,
    'PauliY': Y,
    'PauliZ': Z,
    'Hadamard': H,
    'Hermitian': hermitian,
    'Identity': identity
}

# Defining the load function
def qiskit_to_pennylane(quantum_circuit: QuantumCircuit):

    if not isinstance(quantum_circuit, QuantumCircuit):
        raise ValueError("The circuit {} is not a valid Qiskit QuantumCircuit.".format(quantum_circuit))

    def _function(params, wires):

        # Processing the dictionary of parameters passed
        circuit_with_parameters = quantum_circuit.bind_parameters(params)
        for op in quantum_circuit.data:

            operator_name = op[0].name

            # First item of the list contains the gate
            pennylane_operator = _operation_map[operator_name]


            # Second the wires it acts on



            # convert operation to PennyLane operation
            # pass parameter dictionary & wire
    return _function


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