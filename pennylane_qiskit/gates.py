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
Custom Qiskit gates
===================

**Module name:** :mod:`pennylane_qiskit.gates`

.. currentmodule:: pennylane_qiskit.gates

This module provides custom gates for PennyLane operations that are
missing a direct gate in Qiskit.

Classes
-------

.. autosummary::
    BasisState
    Rot

Code details
~~~~~~~~~~~~
"""
# pylint: disable=attribute-defined-outside-init,too-few-public-methods
from qiskit import QuantumRegister
from qiskit.circuit import Gate
from qiskit.extensions.standard import RYGate, RZGate, XGate


class BasisState(Gate):
    r"""Applies a PennyLane `BasisState` operation by decomposing
    it into a list of PauliX matrices.

    Args:
        n (array): prepares the basis state :math:`\ket{n}`, where ``n`` is an
            array of integers from the set :math:`\{0, 1\}`, i.e.,
            ``n = np.array([0, 1, 0])``, prepares the state :math:`|010\rangle`.
    """

    def __init__(self, n):
        super().__init__("basis_state", len(n), [n])

    def _define(self):
        q = QuantumRegister(self.num_qubits, "q")
        self.definition = [(XGate(), [q[w]], []) for w, p in enumerate(self.params[0]) if p == 1]


class Rot(Gate):
    r"""Applies a PennyLane ``Rot`` operation

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        omega (float): rotation angle :math:`\omega`
    """

    def __init__(self, phi, theta, omega):
        super().__init__("rot", 1, [phi, theta, omega])

    def _define(self):
        q = QuantumRegister(self.num_qubits, "q")
        self.definition = [
            (RZGate(self.params[0]), [q[0]], []),
            (RYGate(self.params[1]), [q[0]], []),
            (RZGate(self.params[2]), [q[0]], []),
        ]
