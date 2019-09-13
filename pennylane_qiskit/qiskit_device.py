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
# pylint: disable=too-many-instance-attributes
import abc
from collections import OrderedDict
import functools
import inspect
import itertools

import numpy as np

from qiskit import extensions as ex
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

from qiskit.compiler import transpile, assemble
from qiskit.circuit.measure import measure
from qiskit.converters import dag_to_circuit, circuit_to_dag

from pennylane import Device, DeviceError
from pennylane.operation import Sample

from .gates import BasisState, Rot
from ._version import __version__


@functools.lru_cache()
def z_eigs(n):
    r"""Returns the eigenvalues for :math:`Z^{\otimes n}`.

    Args:
        n (int): number of wires

    Returns:
        array[int]: eigenvalues of :math:`Z^{\otimes n}
    """
    if n == 1:
        return np.array([1, -1])
    return np.concatenate([z_eigs(n - 1), -z_eigs(n - 1)])


QISKIT_OPERATION_MAP = {
    # native PennyLane operations also native to qiskit
    "PauliX": ex.XGate,
    "PauliY": ex.YGate,
    "PauliZ": ex.ZGate,
    "Hadamard": ex.HGate,
    "CNOT": ex.CnotGate,
    "CZ": ex.CzGate,
    "SWAP": ex.SwapGate,
    "RX": ex.RXGate,
    "RY": ex.RYGate,
    "RZ": ex.RZGate,
    "CRZ": ex.CrzGate,
    "PhaseShift": ex.U1Gate,
    "QubitStateVector": ex.Initialize,
    # operations not natively implemented in Qiskit but provided in gates.py
    "Rot": Rot,
    "BasisState": BasisState,
    "QubitUnitary": ex.UnitaryGate,
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


class QiskitDevice(Device, abc.ABC):
    r"""Abstract Qiskit device for PennyLane.

    Args:
        wires (int): The number of qubits of the device
        provider (Provider): The Qiskit simulation provider
        backend (str): the desired backend
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
            For simulator devices, 0 means the exact EV is returned.

    Keyword Args:
        name (str): The name of the circuit. Default ``'circuit'``.
        compile_backend (BaseBackend): The backend used for compilation. If you wish
            to simulate a device compliant circuit, you can specify a backend here.
    """
    name = "Qiskit PennyLane plugin"
    pennylane_requires = ">=0.5.0"
    version = "0.5.0"
    plugin_version = __version__
    author = "Carsten Blank"

    _capabilities = {"model": "qubit"}
    _operation_map = QISKIT_OPERATION_MAP
    _state_backends = {"statevector_simulator", "unitary_simulator"}
    """set[str]: Set of backend names that define the backends
    that support returning the underlying quantum statevector"""

    operations = set(_operation_map.keys())
    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard", "Hermitian"}

    _eigs = {}

    def __init__(self, wires, provider, backend, shots=1024, **kwargs):
        super().__init__(wires=wires, shots=shots)

        if "verbose" not in kwargs:
            kwargs["verbose"] = False

        self.provider = provider
        self.backend_name = backend
        self._capabilities["backend"] = [b.name() for b in self.provider.backends()]

        # check that backend exists
        if backend not in self._capabilities["backend"]:
            raise ValueError(
                "Backend '{}' does not exist. Available backends "
                "are:\n {}".format(backend, self._capabilities["backend"])
            )

        # perform validation against backend
        b = self.backend
        if wires > b.configuration().n_qubits:
            raise ValueError(
                "Backend '{}' supports maximum {} wires".format(backend, b.configuration().n_qubits)
            )

        if backend not in self._state_backends and shots == 0:
            raise ValueError(
                "Number of shots for backend '{}' must be a positive integer.".format(backend)
            )

        # Inner state
        self._reg = QuantumRegister(wires, "q")
        self._creg = ClassicalRegister(wires, "c")
        self._circuit = None
        self._current_job = None
        self._first_operation = True
        self._state = None  # statevector of a simulator backend

        # job execution options
        self.memory = False  # do not return samples, just counts

        # determine if backend supports backend options and noise models,
        # and properly put together backend run arguments
        s = inspect.signature(b.run)
        self.run_args = {}
        self.compile_backend = None

        if "compile_backend" in kwargs:
            self.compile_backend = kwargs.pop("compile_backend")

        if "noise_model" in kwargs:
            if "noise_model" in s.parameters:
                self.run_args["noise_model"] = kwargs.pop("noise_model")
            else:
                raise ValueError("Backend {} does not support noisy simulations".format(backend))

        if "backend_options" in s.parameters:
            self.run_args["backend_options"] = kwargs

        self.reset()

    @property
    def backend(self):
        """The Qiskit simulation backend object"""
        return self.provider.get_backend(self.backend_name)

    def apply(self, operation, wires, par):
        mapped_operation = self._operation_map[operation]

        if operation == "BasisState" and not self._first_operation:
            raise DeviceError(
                "Operation {} cannot be used after other Operations have already been applied "
                "on a {} device.".format(operation, self.short_name)
            )

        self._first_operation = False

        qregs = [self._reg[i] for i in wires]

        if operation == "QubitStateVector":

            if len(par[0]) != 2 ** len(wires):
                raise ValueError("State vector must be of length 2**wires.")

            qregs = list(reversed(qregs))

        if operation == "QubitUnitary":

            if len(par[0]) != 2 ** len(wires):
                raise ValueError("Unitary matrix must be of shape (2**wires, 2**wires).")

            qregs = list(reversed(qregs))

        dag = circuit_to_dag(QuantumCircuit(self._reg, self._creg, name=""))
        gate = mapped_operation(*par)
        dag.apply_operation_back(gate, qargs=qregs)
        qc = dag_to_circuit(dag)
        self._circuit = self._circuit + qc

    def compile(self):
        """Compile the quantum circuit to target
        the provided compile_backend. If compile_backend is None,
        then the target is simply the backend."""
        compile_backend = self.compile_backend or self.backend
        compiled_circuits = transpile(self._circuit, backend=compile_backend)
        return assemble(
            experiments=compiled_circuits,
            backend=compile_backend,
            shots=self.shots if self.shots > 0 else 1,
            memory=self.memory,
        )

    def run(self, qobj):
        """Run the compiled circuit, and query the result."""
        self._current_job = self.backend.run(qobj, **self.run_args)
        result = self._current_job.result()

        if self.backend_name in self._state_backends:
            self._state = self._get_state(result)

    def _get_state(self, result):
        """Returns the statevector for state simulator backends.

        Args:
            result (qiskit.Result): result object

        Returns:
            array[float]: size ``(2**num_wires,)`` statevector
        """
        if self.backend_name == "statevector_simulator":
            state = np.asarray(result.get_statevector())

        elif self.backend_name == "unitary_simulator":
            unitary = np.asarray(result.get_unitary())
            initial_state = np.zeros([2 ** self.num_wires])
            initial_state[0] = 1

            state = unitary @ initial_state

        # reverse qubit order to match PennyLane convention
        return state.reshape([2] * self.num_wires).T.flatten()

    def rotate_basis(self, obs, wires, par):
        """Rotates the specified wires such that they
        are in the eigenbasis of the provided observable.

        Args:
            observable (str): the name of an observable
            wires (List[int]): wires the observable is measured on
            par (List[Any]): parameters of the observable
        """

    def pre_measure(self):
        for e in self.obs_queue:
            # Add unitaries if a different expectation value is given
            if e.return_type == Sample:
                self.memory = True  # make sure to return samples

            if e.name == "PauliX":
                # X = H.Z.H
                self.apply("Hadamard", wires=e.wires, par=[])

            elif e.name == "PauliY":
                # Y = (HS^)^.Z.(HS^) and S^=SZ
                self.apply("PauliZ", wires=e.wires, par=[])
                self.apply("S", wires=e.wires, par=[])
                self.apply("Hadamard", wires=e.wires, par=[])

            elif e.name == "Hadamard":
                # H = Ry(-pi/4)^.Z.Ry(-pi/4)
                self.apply("RY", e.wires, [-np.pi / 4])

            elif e.name == "Hermitian":
                # For arbitrary Hermitian matrix H, let U be the unitary matrix
                # that diagonalises it, and w_i be the eigenvalues.
                Hmat = e.parameters[0]
                Hkey = tuple(Hmat.flatten().tolist())

                if Hkey in self._eigs:
                    # retrieve eigenvectors
                    U = self._eigs[Hkey]["eigvec"]
                else:
                    # store the eigenvalues corresponding to H
                    # in a dictionary, so that they do not need to
                    # be calculated later
                    w, U = np.linalg.eigh(Hmat)
                    self._eigs[Hkey] = {"eigval": w, "eigvec": U}

                # Perform a change of basis before measuring by applying U^ to the circuit
                self.apply("QubitUnitary", e.wires, [U.conj().T])

        if self.backend_name not in self._state_backends:
            # Add measurements if they are needed
            for qr, cr in zip(self._reg, self._creg):
                measure(self._circuit, qr, cr)

        qobj = self.compile()
        self.run(qobj)

    def expval(self, observable, wires, par):
        if self.backend_name in self._state_backends and self.shots == 0:

            if observable == "Identity":
                return 1

            if observable == "Hermitian":
                Hkey = tuple(par[0].flatten().tolist())
                eigvals = self._eigs[Hkey]["eigval"]
            else:
                eigvals = np.array([1, -1])

            # exact expectation value
            prob = np.fromiter(self.probabilities(wires=wires).values(), dtype=np.float64)
            return (eigvals @ prob).real

        # estimate the ev
        return np.mean(self.sample(observable, wires, par))

    def var(self, observable, wires, par):
        if self.backend_name in self._state_backends and self.shots == 0:

            if observable == "Hermitian":
                Hkey = tuple(par[0].flatten().tolist())
                eigvals = self._eigs[Hkey]["eigval"]
            elif observable == "Identity":
                eigvals = np.array([1, 1])
            else:
                eigvals = np.array([1, -1])

            # exact variance value
            prob = np.fromiter(self.probabilities(wires=wires).values(), dtype=np.float64)
            return (eigvals**2 @ prob).real - (eigvals @ prob).real**2

        return np.var(self.sample(observable, wires, par))

    def sample(self, observable, wires, par, n=None):
        if n is None:
            n = self.shots

        if n == 0:
            raise ValueError("Calling sample with n = 0 is not possible.")

        if n < 0 or not isinstance(n, int):
            raise ValueError("The number of samples must be a positive integer.")

        if observable == "Identity":
            return np.ones([n])

        # branch out depending on the type of backend
        if self.backend_name in self._state_backends:
            # software simulator. Need to sample from probabilities.

            if observable == "Hermitian":
                Hkey = tuple(par[0].flatten().tolist())
                eigvals = self._eigs[Hkey]["eigval"]
            elif observable == "Identity":
                eigvals = np.array([1, 1])
            else:
                eigvals = np.array([1, -1])

            prob = np.fromiter(self.probabilities(wires=wires).values(), dtype=np.float64)
            return np.random.choice(eigvals, n, p=prob)

        # a hardware simulator
        if self.memory:
            # get the samples
            samples = self._current_job.result().get_memory()

            # reverse qubit order to match PennyLane convention
            samples = np.vstack([np.array([int(i) for i in s[::-1]]) for s in samples])

        else:
            # Need to convert counts into samples
            samples = np.vstack(
                [np.vstack([s] * int(self.shots * p)) for s, p in self.probabilities().items()]
            )

        if isinstance(observable, str) and observable in {"PauliX", "PauliY", "PauliZ", "Hadamard"}:
            return 1 - 2 * samples[:, wires[0]]

        Hkey = tuple(par[0].flatten().tolist())
        eigvals = self._eigs[Hkey]["eigval"]

        res = samples[:, np.array(wires)]
        samples = np.zeros([n])

        for w, b in zip(eigvals, itertools.product([0, 1], repeat=len(wires))):
            samples = np.where(np.all(res == b, axis=1), w, samples)

        return samples

    @property
    def state(self):
        return self._state

    def probabilities(self, wires=None):
        """Return the (marginal) probability of each computational basis
        state from the last run of the device.

        Args:
            wires (Sequence[int]): Sequence of wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.

        Returns:
            OrderedDict[tuple, float]: Dictionary mapping a tuple representing the state
            to the resulting probability. The dictionary should be sorted such that the
            state tuples are in lexicographical order.
        """
        # Note: Qiskit uses the convention that the first qubit is the
        # least significant qubit.
        if self._current_job is None:
            return None

        if self.backend_name in self._state_backends:
            # statevector simulator
            prob = np.abs(self.state.reshape([2] * self.num_wires)) ** 2
        else:
            # hardware simulator
            result = self._current_job.result()

            # sort the counts and reverse qubit order to match PennyLane convention
            nonzero_prob = {
                tuple(int(i) for i in s[::-1]): c / self.shots
                for s, c in result.get_counts().items()
            }

            if wires is None:
                # marginal probabilities not required
                return OrderedDict(tuple(sorted(nonzero_prob.items())))

            prob = np.zeros([2] * self.num_wires)

            for s, p in tuple(sorted(nonzero_prob.items())):
                prob[s] = p

        wires = wires or range(self.num_wires)
        wires = np.hstack(wires)

        basis_states = itertools.product(range(2), repeat=len(wires))
        inactive_wires = list(set(range(self.num_wires)) - set(wires))
        prob = np.apply_over_axes(np.sum, prob, inactive_wires).flatten()
        return OrderedDict(zip(basis_states, prob))

    def reset(self):
        self._circuit = QuantumCircuit(self._reg, self._creg, name="temp")
        self._first_operation = True
        self._state = None
