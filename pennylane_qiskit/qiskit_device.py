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
"""
import os

import numpy as np

import qiskit
from qiskit import extensions as ex
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

from qiskit.compiler import transpile, assemble
from qiskit.circuit.measure import measure
from qiskit.converters import dag_to_circuit, circuit_to_dag

from pennylane import Device, DeviceError

from .gates import BasisState, Rot, QubitUnitary
from ._version import __version__


QISKIT_OPERATION_MAP = {
    # native PennyLane operations also native to qiskit
    "PauliX": ex.XGate,
    "PauliY": ex.YGate,
    "PauliZ": ex.ZGate,
    "CNOT": ex.CnotGate,
    "CZ": ex.CzGate,
    "SWAP": ex.SwapGate,
    "RX": ex.RXGate,
    "RY": ex.RYGate,
    "RZ": ex.RZGate,
    "CRZ": ex.CrzGate,
    "PhaseShift": ex.U1Gate,
    "Hadamard": ex.HGate,
    "QubitStateVector": ex.Initialize,
    # operations not natively implemented in Qiskit but provided in gates.py
    "Rot": Rot,
    "BasisState": BasisState,
    "QubitUnitary": QubitUnitary,
    # additional operations not native to PennyLane but present in Qiskit
    "S": ex.SGate,
    "Sdg": ex.SdgGate,
    "T": ex.TGate,
    "Tdg": ex.TdgGate,
    "U1": ex.U1Gate,
    "U2": ex.U2Gate,
    "U3": ex.U3Gate,
    "CSWAP": ex.FredkinGate,
    "Toffoli": ex.ToffoliGate,
}


class QiskitDevice(Device):
    name = "Qiskit PennyLane plugin"
    short_name = "qiskit"
    pennylane_requires = ">=0.5.0"
    version = "0.1.0"
    plugin_version = __version__
    author = "Carsten Blank"

    _capabilities = {"model": "qubit"}

    _operation_map = QISKIT_OPERATION_MAP

    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard", "Hermitian"}

    _backend_kwargs = ["verbose", "backend"]
    _eigs = {}

    def __init__(self, wires, provider, backend, shots=1024, **kwargs):
        super().__init__(wires=wires, shots=shots)

        if "verbose" not in kwargs:
            kwargs["verbose"] = False

        self.provider = provider
        self.backend_name = backend
        self.compile_backend = kwargs.get("compile_backend")
        self.name = kwargs.get("name", "circuit")
        self.kwargs = kwargs

        self._capabilities["backend"] = [b.name() for b in self.provider.backends()]

        # Inner state
        self._reg = QuantumRegister(wires, "q")
        self._creg = ClassicalRegister(wires, "c")
        self._circuit = None
        self._current_job = None
        self._first_operation = True
        self.reset()

    @property
    def operations(self):
        return set(self._operation_map.keys())

    @property
    def backend(self):
        return self.provider.get_backend(self.backend_name)

    def apply(self, operation, wires, par):
        mapped_operation = self._operation_map[operation]

        if operation == "BasisState" and not self._first_operation:
            raise DeviceError(
                "Operation {} cannot be used after other Operations have already been applied "
                "on a {} device.".format(operation, self.short_name)
            )

        self._first_operation = False

        qregs = [(self._reg, i) for i in wires]

        if operation == "QubitStateVector":

            if len(par) > 2 ** len(qregs):
                raise ValueError("State vector must be of length 2**wires.")

            qregs = list(reversed(qregs))

        dag = circuit_to_dag(QuantumCircuit(self._reg, self._creg, name=""))
        gate = mapped_operation(*par)
        dag.apply_operation_back(gate, qargs=qregs)
        qc = dag_to_circuit(dag)
        self._circuit = self._circuit + qc

    def compile(self):
        compile_backend = self.compile_backend or self.backend
        compiled_circuits = transpile(self._circuit, backend=compile_backend)

        return assemble(experiments=compiled_circuits, backend=compile_backend, shots=self.shots)

    def run(self, qobj):
        self._current_job = self.backend.run(qobj, backend_options=self.kwargs)
        self._current_job.result()

    def pre_measure(self):
        # Add unitaries if a different expectation value is given
        for e in self.obs_queue:
            wire = [e.wires[0]]

            if e.name == "Identity":
                pass  # nothing to be done here! Will be taken care of later.

            elif e.name == "PauliZ":
                pass  # nothing to be done here! Will be taken care of later.

            elif e.name == "PauliX":
                # X = H.Z.H
                self.apply("Hadamard", wires=wire, par=[])

            elif e.name == "PauliY":
                # Y = (HS^)^.Z.(HS^) and S^=SZ
                self.apply("PauliZ", wires=wire, par=[])
                self.apply("S", wires=wire, par=[])
                self.apply("Hadamard", wires=wire, par=[])

            elif e.name == "Hadamard":
                # H = Ry(-pi/4)^.Z.Ry(-pi/4)
                self.apply("RY", wire, [-np.pi / 4])

            elif e.name == "Hermitian":
                # For arbitrary Hermitian matrix H, let U be the unitary matrix
                # that diagonalises it, and w_i be the eigenvalues.
                H = e.parameters[0]
                Hkey = tuple(H.flatten().tolist())

                if Hkey in self._eigs:
                    # retrieve eigenvectors
                    U = self._eigs[Hkey]["eigvec"]
                else:
                    # store the eigenvalues corresponding to H
                    # in a dictionary, so that they do not need to
                    # be calculated later
                    w, U = np.linalg.eigh(H)
                    self._eigs[Hkey] = {"eigval": w, "eigvec": U}

                # Perform a change of basis before measuring by applying U^ to the circuit
                self.apply("QubitUnitary", wire, [U.conj().T])

        # Add measurements if they are needed
        if self.backend_name not in ("statevector_simulator", "unitary_simulator"):
            for qr, cr in zip(self._reg, self._creg):
                measure(self._circuit, qr, cr)

        qobj = self.compile()
        self.run(qobj)

    def expval(self, expectation, wires, par):
        # Make wires lists.
        if isinstance(wires, int):
            wire = wires
        else:
            wire = wires[0]

        # Get the result of the job
        result = self._current_job.result()

        def to_probabilities(state):
            # Normalize the state in case some numerical errors have changed this!
            state = state / np.linalg.norm(state)
            probabilities = state.conj() * state
            return dict(
                [
                    ("{0:b}".format(i).zfill(self.num_wires), abs(p))
                    for i, p in enumerate(probabilities)
                ]
            )

        # Distinguish between three different calculations
        # As any different expectation value from PauliZ is already handled before
        # here we treat everything as PauliZ.
        if self.backend_name == "statevector_simulator":
            state = np.asarray(result.get_statevector())
            probabilities = to_probabilities(state)

        elif self.backend_name == "unitary_simulator":
            unitary = np.asarray(result.get_unitary())
            initial_state = np.zeros(shape=(self.num_wires ** 2,))
            initial_state[0] = 1
            state = unitary @ initial_state
            probabilities = to_probabilities(state)

        else:
            probabilities = dict(
                (state, count / self.shots) for state, count in result.get_counts().items()
            )

        # The first qubit measurement is right-most, so we need to reverse the measurement result
        zero = sum(
            p for (measurement, p) in probabilities.items() if measurement[::-1][wire] == "0"
        )
        one = sum(p for (measurement, p) in probabilities.items() if measurement[::-1][wire] == "1")

        expval = (1 - (2 * one) - (1 - 2 * zero)) / 2

        # for single qubit state probabilities |psi|^2 = (p0, p1),
        # we know that p0+p1=1 and that <Z>=p0-p1
        p0 = (1 + expval) / 2
        p1 = (1 - expval) / 2

        if expectation == "Identity":
            # <I> = \sum_i p_i
            return p0 + p1

        if expectation == "Hermitian":
            # <H> = \sum_i w_i p_i
            Hkey = tuple(par[0].flatten().tolist())
            w = self._eigs[Hkey]["eigval"]
            return w[0] * p0 + w[1] * p1

        return expval

    def reset(self):
        self._circuit = QuantumCircuit(self._reg, self._creg, name="temp")
        self._first_operation = True
