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
    QubitUnitary

Code details
~~~~~~~~~~~~
"""
import abc
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Gate
from qiskit.extensions.standard import IdGate, RXGate, RYGate, RZGate, U3Gate, XGate


class BasisState(Gate):
    r"""Applies a PennyLane `BasisState` operation by decomposing
    it into a list of PauliX matrices.

    Args:
        n (array): prepares the basis state :math:`\ket{n}`, where ``n`` is an
            array of integers from the set :math:`\{0, 1\}`, i.e.,
            if ``n = np.array([0, 1, 0])``, prepares the state :math:`|010\rangle`.
    """

    def __init__(self, n):
        super().__init__("basis_state", len(n), [n])

    def _define(self):
        q = QuantumRegister(self.num_qubits, "q")
        if np.any(self.params[0] == 1):
            self.definition = [
                (XGate(), [q[w]], []) for w, p in enumerate(self.params[0]) if p == 1
            ]
        else:
            self.definition = [(IdGate(), [q[0]], [])]


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


class QubitUnitary(Gate):
    """Applies a PennyLane ``QubitUnitary`` operation on a single qubit.

    The resulting unitary is mapped to the Qiskit ``u3`` gate.

    Args:
        U (array[complex]): square unitary matrix
    """

    TOL = 1e-6

    def __init__(self, U):
        super().__init__("qubit_unitary", 1, [U])

    def _define(self):
        q = QuantumRegister(self.num_qubits, "q")

        U = self.params[0]

        if U.shape != (2, 2):
            raise ValueError("Only a single qubit unitary is supported.")

        if not np.allclose(U @ U.conj().T, np.identity(2), atol=self.TOL, rtol=0):
            raise ValueError("Not a unitary.")

        # We assume
        # [ a  b ]
        # [ c  d ]

        a, b, c, d = U.flatten()

        # We use the universal single qubit gate U3
        # (see https://qiskit.org/documentation/terra/summary_of_quantum_operations.html)
        # with the assumption that element a has no global phase.
        global_phase = np.angle(a)
        theta = 2 * np.arccos((a * np.exp(-1j * global_phase)).real)

        lam = 0
        phi = 0

        if abs(b) > self.TOL:
            lam = np.angle(-b * np.exp(-1j * global_phase))

        if abs(c) > self.TOL:
            phi = np.angle(c * np.exp(-1j * global_phase))

        lam_phi = np.angle(d * np.exp(-1j * global_phase))

        if lam == 0:
            phi = lam_phi - phi
        else:
            phi = lam_phi - lam

        if not np.allclose(
            d * np.exp(-1j * global_phase),
            np.exp(1.0j * lam + 1.0j * phi) * np.cos(theta / 2),
            atol=self.TOL,
            rtol=0,
        ):
            raise RuntimeError("Error applying unitary.")

        self.definition = [(U3Gate(theta, phi, lam), [q[0]], [])]
