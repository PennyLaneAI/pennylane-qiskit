# Copyright 2018 Carsten Blank

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
Wrapper classes for qiskit Operations
===================

.. currentmodule:: pennylane_qiskit.ops

This module provides wrapper classes for `Operations` that are missing a class in qiskit.

"""
from typing import List, Tuple

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.extensions.standard import x, rx, ry, rz


class QiskitInstructions(object):

    def apply(self, qregs, param, circuit):
        # type: (List[Tuple[QuantumRegister, int]], List, QuantumCircuit) -> None
        pass


# class Toffoli(BasicProjectQGate): # pylint: disable=too-few-public-methods
#     """Class for the Toffoli gate.
#
#     Contrary to other gates, ProjectQ does not have a class for the Toffoli gate,
#     as it is implemented as a meta-gate.
#     For consistency we define this class, whose constructor is made to retun
#     a gate with the correct properties by overwriting __new__().
#     """
#     def __new__(*par): # pylint: disable=no-method-argument
#         return pq.ops.C(pq.ops.ZGate(), 2)
#
# class AllZGate(BasicProjectQGate): # pylint: disable=too-few-public-methods
#     """Class for the AllZ gate.
#
#     Contrary to other gates, ProjectQ does not have a class for the AllZ gate,
#     as it is implemented as a meta-gate.
#     For consistency we define this class, whose constructor is made to retun
#     a gate with the correct properties by overwriting __new__().
#     """
#     def __new__(*par): # pylint: disable=no-method-argument
#         return pq.ops.Tensor(pq.ops.ZGate())


class BasisState(QiskitInstructions):
    """Class for the BasisState preparation.

    qiskit does not currently have a dedicated gate for this, so we implement it here.
    """

    def apply(self, qregs, param, circuit):
        # type: (List[Tuple[QuantumRegister, int]], List, QuantumCircuit) -> None
        if len(param) == 0:
            raise Exception('Parameters are missing')
        for i, p in enumerate(param[0]):
            if p == 1:
                x(circuit, qregs[i])


class Rot(QiskitInstructions):
    """Class for the arbitrary single qubit rotation gate.

    ProjectQ does not currently have an arbitrary single qubit rotation gate,
    so we provide a class that return a suitable combination of rotation gates
    assembled into a single gate from the constructor of this class.
    """

    def apply(self, qregs, param, circuit):
        # type: (List[Tuple[QuantumRegister, int]], List, QuantumCircuit) -> None
        if len(param) == 0:
            raise Exception('Parameters are missing')
        for q in qregs:
            rx(circuit, param[0], q)
            ry(circuit, param[1], q)
            rz(circuit, param[2], q)


class QubitUnitary(QiskitInstructions):
    """Class for the arbitrary single qubit rotation gate.

    ProjectQ does not currently have an arbitrary single qubit rotation gate,
    so we provide a class that return a suitable combination of rotation gates
    assembled into a single gate from the constructor of this class.
    """

    def apply(self, qregs, param, circuit):
        # type: (List[Tuple[QuantumRegister, int]], List, QuantumCircuit) -> None
        if len(param) == 0:
            raise Exception('Parameters are missing')
        raise Exception("Not Implemented!")


class QubitStateVector(QiskitInstructions):
    """Class for the arbitrary single qubit rotation gate.

    ProjectQ does not currently have an arbitrary single qubit rotation gate,
    so we provide a class that return a suitable combination of rotation gates
    assembled into a single gate from the constructor of this class.
    """

    def apply(self, qregs, param, circuit):
        # type: (List[Tuple[QuantumRegister, int]], List, QuantumCircuit) -> None
        if len(param) == 0:
            raise Exception('Parameters are missing')
        raise Exception("Not Implemented!")
