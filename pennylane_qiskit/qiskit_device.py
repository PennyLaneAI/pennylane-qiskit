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
This module contains a base class for constructing Qiskit devices for PennyLane.
"""
# pylint: disable=too-many-instance-attributes

import abc
import functools
import inspect
import itertools
import warnings
from collections import OrderedDict

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import extensions as ex
from qiskit.circuit.measure import measure
from qiskit.compiler import assemble, transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit

from pennylane import Device, QuantumFunctionError
from pennylane.operation import Sample

from ._version import __version__


@functools.lru_cache()
def pauli_eigs(n):
    r"""Returns the eigenvalues for :math:`A^{\otimes n}`,
    where :math:`A` is any operator that shares eigenvalues
    with the Pauli matrices.

    Args:
        n (int): number of wires

    Returns:
        array[int]: eigenvalues of :math:`Z^{\otimes n}`
    """
    if n == 1:
        return np.array([1, -1])
    return np.concatenate([pauli_eigs(n - 1), -pauli_eigs(n - 1)])


QISKIT_OPERATION_MAP = {
    # native PennyLane operations also native to qiskit
    "PauliX": ex.XGate,
    "PauliY": ex.YGate,
    "PauliZ": ex.ZGate,
    "Hadamard": ex.HGate,
    "CNOT": ex.CXGate,
    "CZ": ex.CZGate,
    "SWAP": ex.SwapGate,
    "RX": ex.RXGate,
    "RY": ex.RYGate,
    "RZ": ex.RZGate,
    "S": ex.SGate,
    "T": ex.TGate,
    # Adding the following for conversion compatibility
    "CSWAP": ex.CSwapGate,
    "CRX": ex.CRXGate,
    "CRY": ex.CRYGate,
    "CRZ": ex.CRZGate,
    "PhaseShift": ex.U1Gate,
    "QubitStateVector": ex.Initialize,
    "U2": ex.U2Gate,
    "U3": ex.U3Gate,
    "Toffoli": ex.CCXGate,
    "QubitUnitary": ex.UnitaryGate,
}

# Separate dictionary for the inverses as the operations dictionary needs
# to be invertable for the conversion functionality to work
QISKIT_OPERATION_INVERSES_MAP = {k + ".inv": v for k, v in QISKIT_OPERATION_MAP.items()}


class QiskitDevice(Device, abc.ABC):
    r"""Abstract Qiskit device for PennyLane.

    Args:
        wires (int): The number of qubits of the device
        provider (Provider): The Qiskit simulation provider
        backend (str): the desired backend
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.

    Keyword Args:
        name (str): The name of the circuit. Default ``'circuit'``.
        compile_backend (BaseBackend): The backend used for compilation. If you wish
            to simulate a device compliant circuit, you can specify a backend here.
        analytic (bool): For statevector backends, determines if the
            expectation values and variances are to be computed analytically.
            Default value is ``False``.
    """
    name = "Qiskit PennyLane plugin"
    pennylane_requires = ">=0.8.1"
    version = "0.9.0-dev"
    plugin_version = __version__
    author = "Xanadu"

    _capabilities = {"model": "qubit", "tensor_observables": True, "inverse_operations": True}
    _operation_map = {**QISKIT_OPERATION_MAP, **QISKIT_OPERATION_INVERSES_MAP}
    _state_backends = {"statevector_simulator", "unitary_simulator"}
    """set[str]: Set of backend names that define the backends
    that support returning the underlying quantum statevector"""

    operations = set(_operation_map.keys())
    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard", "Hermitian"}

    hw_analytic_warning_message = (
        "The analytic calculation of expectations and variances "
        "is only supported on statevector backends, not on the {}. "
        "The obtained result is based on sampling."
    )

    _eigs = {}

    def __init__(self, wires, provider, backend, shots=1024, **kwargs):
        super().__init__(wires=wires, shots=shots)

        self.analytic = kwargs.pop("analytic", False)

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

        # Inner state
        self._reg = QuantumRegister(wires, "q")
        self._creg = ClassicalRegister(wires, "c")
        self._circuit = None
        self._current_job = None
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

        qregs = [self._reg[i] for i in wires]

        if operation == "QubitStateVector":

            if self.backend_name == "unitary_simulator":
                raise QuantumFunctionError(
                    "The QubitStateVector operation is not supported on the unitary simulator backend."
                )

            if len(par[0]) != 2 ** len(wires):
                raise ValueError("State vector must be of length 2**wires.")

            qregs = list(reversed(qregs))

            # TODO: Once a fix is available in Qiskit-Aer, remove the following:
            par = (x.tolist() for x in par if isinstance(x, np.ndarray))

        if operation == "QubitUnitary":

            if len(par[0]) != 2 ** len(wires):
                raise ValueError("Unitary matrix must be of shape (2**wires, 2**wires).")

            qregs = list(reversed(qregs))

        dag = circuit_to_dag(QuantumCircuit(self._reg, self._creg, name=""))
        gate = mapped_operation(*par)

        if operation.endswith(".inv"):
            gate = gate.inverse()

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
            shots=self.shots,
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
        if obs == "PauliX":
            # X = H.Z.H
            self.apply("Hadamard", wires=wires, par=[])

        elif obs == "PauliY":
            # Y = (HS^)^.Z.(HS^) and S^=SZ
            self.apply("PauliZ", wires=wires, par=[])
            self.apply("S", wires=wires, par=[])
            self.apply("Hadamard", wires=wires, par=[])

        elif obs == "Hadamard":
            # H = Ry(-pi/4)^.Z.Ry(-pi/4)
            self.apply("RY", wires, [-np.pi / 4])

        elif obs == "Hermitian":
            # For arbitrary Hermitian matrix H, let U be the unitary matrix
            # that diagonalises it, and w_i be the eigenvalues.
            Hmat = par[0]
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
            self.apply("QubitUnitary", wires, [U.conj().T])

    def pre_measure(self):
        for e in self.obs_queue:
            # Add unitaries if a different expectation value is given
            # Exclude unitary_simulator as it does not support memory=True
            if (
                hasattr(e, "return_type")
                and e.return_type == Sample
                and self.backend_name != "unitary_simulator"
            ):
                self.memory = True  # make sure to return samples

            if isinstance(e.name, list):
                # tensor product
                for n, w, p in zip(e.name, e.wires, e.parameters):
                    self.rotate_basis(n, w, p)
            else:
                # single wire observable
                self.rotate_basis(e.name, e.wires, e.parameters)

        if self.backend_name not in self._state_backends:
            # Add measurements if they are needed
            for qr, cr in zip(self._reg, self._creg):
                measure(self._circuit, qr, cr)

        qobj = self.compile()
        self.run(qobj)

    def expval(self, observable, wires, par):
        if self.backend_name in self._state_backends and self.analytic:
            # exact expectation value
            eigvals = self.eigvals(observable, wires, par)
            prob = np.fromiter(self.probability(wires=wires).values(), dtype=np.float64)
            return (eigvals @ prob).real

        if self.analytic:
            # Raise a warning if backend is a hardware simulator
            warnings.warn(self.hw_analytic_warning_message.format(self.backend), UserWarning)

        # estimate the ev
        return np.mean(self.sample(observable, wires, par))

    def var(self, observable, wires, par):
        if self.backend_name in self._state_backends and self.analytic:
            # exact variance value
            eigvals = self.eigvals(observable, wires, par)
            prob = np.fromiter(self.probability(wires=wires).values(), dtype=np.float64)
            return (eigvals ** 2) @ prob - (eigvals @ prob).real ** 2

        if self.analytic:
            # Raise a warning if backend is a hardware simulator
            warnings.warn(self.hw_analytic_warning_message.format(self.backend), UserWarning)

        return np.var(self.sample(observable, wires, par))

    def sample(self, observable, wires, par):
        if observable == "Identity":
            return np.ones([self.shots])

        # branch out depending on the type of backend
        if self.backend_name in self._state_backends:
            # software simulator. Need to sample from probabilities.
            eigvals = self.eigvals(observable, wires, par)
            prob = np.fromiter(self.probability(wires=wires).values(), dtype=np.float64)
            return np.random.choice(eigvals, self.shots, p=prob)

        # a hardware simulator
        if self.memory:
            # get the samples
            samples = self._current_job.result().get_memory()

            # reverse qubit order to match PennyLane convention
            samples = np.vstack([np.array([int(i) for i in s[::-1]]) for s in samples])

        else:
            # Need to convert counts into samples
            samples = np.vstack(
                [np.vstack([s] * int(self.shots * p)) for s, p in self.probability().items()]
            )

        if isinstance(observable, str) and observable in {"PauliX", "PauliY", "PauliZ", "Hadamard"}:
            return 1 - 2 * samples[:, wires[0]]

        eigvals = self.eigvals(observable, wires, par)
        wires = np.hstack(wires)
        res = samples[:, np.array(wires)]
        samples = np.zeros([self.shots])

        for w, b in zip(eigvals, itertools.product([0, 1], repeat=len(wires))):
            samples = np.where(np.all(res == b, axis=1), w, samples)

        return samples

    @property
    def state(self):
        return self._state

    def probability(self, wires=None):
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

    def eigvals(self, observable, wires, par):
        """Determine the eigenvalues of observable(s).

        Args:
            observable (str, List[str]): the name of an observable,
                or a list of observables representing a tensor product
            wires (List[int]): wires the observable(s) is measured on
            par (List[Any]): parameters of the observable(s)

        Returns:
            array[float]: an array of size ``(len(wires),)`` containing the
            eigenvalues of the observable
        """
        # the standard observables all share a common eigenbasis {1, -1}
        # with the Pauli-Z gate/computational basis measurement
        standard_observables = {"PauliX", "PauliY", "PauliZ", "Hadamard"}

        # observable should be Z^{\otimes n}
        eigvals = pauli_eigs(len(wires))

        if isinstance(observable, list):
            # tensor product of observables

            # check if there are any non-standard observables (such as Identity, Hadamard)
            if set(observable) - standard_observables:
                # Tensor product of observables contains a mixture
                # of standard and non-standard observables
                eigvals = np.array([1])

                # group the observables into subgroups, depending on whether
                # they are in the standard observables or not.
                for k, g in itertools.groupby(
                    zip(observable, wires, par), lambda x: x[0] in standard_observables
                ):
                    if k:
                        # Subgroup g contains only standard observables.
                        # Determine the size of the subgroup, by transposing
                        # the list, flattening it, and determining the length.
                        n = len([w for sublist in list(zip(*g))[1] for w in sublist])
                        eigvals = np.kron(eigvals, pauli_eigs(n))
                    else:
                        # Subgroup g contains only non-standard observables.
                        for ns_obs in g:
                            # loop through all non-standard observables
                            if ns_obs[0] == "Hermitian":
                                # Hermitian observable has pre-computed eigenvalues
                                p = ns_obs[2]
                                Hkey = tuple(p[0].flatten().tolist())
                                eigvals = np.kron(eigvals, self._eigs[Hkey]["eigval"])

                            elif ns_obs[0] == "Identity":
                                # Identity observable has eigenvalues (1, 1)
                                eigvals = np.kron(eigvals, np.array([1, 1]))

        elif observable == "Hermitian":
            # single wire Hermitian observable
            Hkey = tuple(par[0].flatten().tolist())
            eigvals = self._eigs[Hkey]["eigval"]

        elif observable == "Identity":
            # single wire identity observable
            eigvals = np.ones(2 ** len(wires))

        return eigvals

    def reset(self):
        self._circuit = QuantumCircuit(self._reg, self._creg, name="temp")
        self._state = None
