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
import cmath
import math
from math import acos
from typing import List, Tuple, Optional

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.extensions import standard
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
#     For consistency we define this class, whose constructor is made to return
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
#     For consistency we define this class, whose constructor is made to return
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
                x.x(circuit, qregs[i])


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
            rz.rz(circuit, param[0], q)
            ry.ry(circuit, param[1], q)
            rz.rz(circuit, param[2], q)


class QubitUnitary(QiskitInstructions):

    tolerance = 1e-6
    """Class for the arbitrary single qubit rotation gate.

    ProjectQ does not currently have an arbitrary single qubit rotation gate,
    so we provide a class that return a suitable combination of rotation gates
    assembled into a single gate from the constructor of this class.
    """

    def apply(self, qregs, param, circuit):
        # type: (List[Tuple[QuantumRegister, int]], List, QuantumCircuit) -> None
        if len(param) == 0:
            raise Exception('Parameters are missing')
        parameters = param[0]
        if isinstance(parameters, np.ndarray):
            parameters = parameters.reshape((4,))
        if len(parameters) != 4:
            raise Exception('An array of 4 complex numbers must be given.')

        # We assume
        # [ a  b ]
        # [ c  d ]
        a = parameters[0]  # type: complex
        b = parameters[1]  # type: complex
        c = parameters[2]  # type: complex
        d = parameters[3]  # type: complex

        col1 = math.sqrt(abs(a) ** 2 + abs(c) ** 2)
        col2 = math.sqrt(abs(b) ** 2 + abs(d) ** 2)

        if abs(col1 - 1.0) > 1e-3 or abs(col2 - 1.0) > 1e-3:
            raise Exception('Not a unitary.')

        # We use the universal gate U (see https://qiskit.org/documentation/terra/summary_of_quantum_operations.html)
        # And assume that a ought to be without a global phase
        # Then we compute the angle theta
        global_phase = cmath.phase(a)
        theta = 2 * acos((a * cmath.exp(-1j*global_phase)).real)

        lam = None  # type: Optional[float]
        phi = None  # type: Optional[float]
        if abs(b) > self.tolerance:
            lam = cmath.phase(-b * cmath.exp(-1j*global_phase))
        if abs(c) > self.tolerance:
            phi = cmath.phase(c * cmath.exp(-1j*global_phase))

        lam_phi = cmath.phase(d * cmath.exp(-1j*global_phase))

        if lam is None and phi is None:
            lam = 0.0
            phi = lam_phi
        elif lam is None and phi is not None:
            lam = lam_phi - phi
        elif lam is not None and phi is None:
            phi = lam_phi - lam

        if abs(d * cmath.exp(-1j*global_phase) - cmath.exp(1.0j * lam + 1.0j * phi) * cmath.cos(theta / 2)) > self.tolerance:
            raise Exception('Not a unitary.')

        if isinstance(qregs, list):
            for q in qregs:
                standard.u3.u3(circuit, theta, phi, lam, q)
        else:
            standard.u3.u3(circuit, theta, phi, lam, qregs)


class QubitStateVector(QiskitInstructions):
    """Class for creating an arbitrary quantum state.
    """

    def apply(self, qregs, param, circuit):
        # type: (List[Tuple[QuantumRegister, int]], List, QuantumCircuit) -> None
        if len(param) == 0:
            raise Exception('Parameters are missing')
        if len(param) > 2 ** len(qregs):
            raise Exception("Too many parameters for the amount of qubits")
        from qiskit.extensions import initializer
        initializer.initialize(circuit, param[0], list(reversed(qregs)))
