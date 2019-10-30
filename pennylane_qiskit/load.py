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
from qiskit.circuit import Parameter, ParameterExpression
from pennylane_qiskit.qiskit_device import QISKIT_OPERATION_MAP
import pennylane.ops.qubit as ops
import warnings
from sympy.core.numbers import Float

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
#def qiskit_to_pennylane(quantum_circuit: QuantumCircuit, params: dict = None):


#    return operator_queue

def load(quantum_circuit: QuantumCircuit):
    # queue = qml.converter()
    # template = qiskit_to_pennylane(quantum_circuit)

    def _function(start_wire_index: int = 0, params: dict = None):

        if not isinstance(quantum_circuit, QuantumCircuit):
            raise ValueError("The circuit {} is not a valid Qiskit QuantumCircuit.".format(quantum_circuit))
        # TODO:
        # Check if parameters are needed -> if so, bind them
        # Check if parameters are needed -> if not passed, raise ERROR
        # Check if parameters are not needed -> if not passed, go on
        # Check if parameters are not needed -> if passed, , raise WARNING

        operator_queue = []

        # Qiskit will raise an error, if parameters that are not present were to bound bound
        if params is not None:
            qc = quantum_circuit.bind_parameters(params)
        else:
            qc = quantum_circuit

        # Processing the dictionary of parameters passed

        for op in qc.data:

            # TODO:
            # What if multiple quantum registers are specified

            # Indexing a Qubit object to obtain a wire and shifting it with the start wire index

            # TODO:
            # Check what if more wires specified in qiskit

            wires = [start_wire_index + qubit.index for qubit in op[1]]
            instruction_name = op[0].__class__.__name__

            if instruction_name in inv_map and inv_map[instruction_name] in ops.ops:
                operation_name = inv_map[instruction_name]
                operation = getattr(ops, operation_name)


                # Check that the parameters were bound correctly

                # TODO:
                # Check if there could be a parameter of type other than ParameterExpression and float
                parameters = []
                for param in op[0].params:
                    if isinstance(param, Parameter):
                        raise ValueError("The parameter {} was not bound correctly.".format(param))
                    elif isinstance(param, ParameterExpression):

                        # Conversion of the ParameterExpression value to float
                        parameters.append(float(param._symbol_expr.evalf()))
                    
                    # Converted it to a float, if it can be
                    elif isinstance(float(param), float):
                        parameters.append(float(param))
                    else:
                        raise ValueError("Wrong type for {}.".format(param))

                if not parameters:
                    operation(wires=wires)
                else:
                    operation(*parameters, wires=wires)

            else:
                warnings.warn(__name__ + " The {} instruction is not supported by PennyLane.".
                              format(instruction_name),
                              UserWarning)
    return _function

