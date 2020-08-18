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
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init


import abc
import inspect
import warnings

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import extensions as ex
from qiskit.circuit.measure import measure
from qiskit.compiler import assemble, transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit

from pennylane import QubitDevice, DeviceError

from ._version import __version__


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


class QiskitDevice(QubitDevice, abc.ABC):
    r"""Abstract Qiskit device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
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
    pennylane_requires = ">=0.11.0"
    version = "0.11.0"
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
        "The analytic calculation of expectations, variances and "
        "probabilities is only supported on statevector backends, not on the {}. "
        "Such statistics obtained from this device are estimates based on samples."
    )

    _eigs = {}

    def __init__(self, wires, provider, backend, shots=1024, **kwargs):
        super().__init__(wires=wires, shots=shots)

        # Keep track if the user specified analytic to be True
        user_specified_analytic = "analytic" in kwargs and kwargs["analytic"]
        self.analytic = kwargs.pop("analytic", False)

        if self.analytic and backend not in self._state_backends:

            if user_specified_analytic:
                # Raise a warning if the analytic attribute was set to True
                warnings.warn(self.hw_analytic_warning_message.format(backend), UserWarning)

            self.analytic = False

        if "verbose" not in kwargs:
            kwargs["verbose"] = False

        self.provider = provider
        self.backend_name = backend
        self._capabilities["backend"] = [b.name() for b in self.provider.backends()]

        # check that the backend exists
        if backend not in self._capabilities["backend"]:
            raise ValueError(
                "Backend '{}' does not exist. Available backends "
                "are:\n {}".format(backend, self._capabilities["backend"])
            )

        # perform validation against backend
        b = self.backend
        if len(self.wires) > b.configuration().n_qubits:
            raise ValueError(
                "Backend '{}' supports maximum {} wires".format(backend, b.configuration().n_qubits)
            )

        # Initialize inner state
        self.reset()

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

    @property
    def backend(self):
        """The Qiskit simulation backend object"""
        return self.provider.get_backend(self.backend_name)

    def reset(self):
        self._reg = QuantumRegister(self.num_wires, "q")
        self._creg = ClassicalRegister(self.num_wires, "c")
        self._circuit = QuantumCircuit(self._reg, self._creg, name="temp")

        self._current_job = None
        self._state = None  # statevector of a simulator backend

    def apply(self, operations, **kwargs):
        rotations = kwargs.get("rotations", [])

        applied_operations = self.apply_operations(operations)

        # Rotating the state for measurement in the computational basis
        rotation_circuits = self.apply_operations(rotations)
        applied_operations.extend(rotation_circuits)

        for circuit in applied_operations:
            self._circuit += circuit

        if self.backend_name not in self._state_backends:
            # Add measurements if they are needed
            for qr, cr in zip(self._reg, self._creg):
                measure(self._circuit, qr, cr)

        # These operations need to run for all devices
        qobj = self.compile()
        self.run(qobj)

    def apply_operations(self, operations):
        """Apply the circuit operations.

        This method serves as an auxiliary method to :meth:`~.QiskitDevice.apply`.

        Args:
            operations (List[pennylane.Operation]): operations to be applied

        Returns:
            list[QuantumCircuit]: a list of quantum circuit objects that
                specify the corresponding operations
        """
        circuits = []

        for operation in operations:
            # Apply the circuit operations
            device_wires = self.map_wires(operation.wires)
            par = operation.parameters
            operation = operation.name

            mapped_operation = self._operation_map[operation]

            self.qubit_unitary_check(operation, par, device_wires)
            self.qubit_state_vector_check(operation, par, device_wires)

            qregs = [self._reg[i] for i in device_wires.labels]

            if operation.split(".inv")[0] in ("QubitUnitary", "QubitStateVector"):
                # Need to revert the order of the quantum registers used in
                # Qiskit such that it matches the PennyLane ordering
                qregs = list(reversed(qregs))

            dag = circuit_to_dag(QuantumCircuit(self._reg, self._creg, name=""))
            gate = mapped_operation(*par)

            if operation.endswith(".inv"):
                gate = gate.inverse()

            dag.apply_operation_back(gate, qargs=qregs)
            circuit = dag_to_circuit(dag)
            circuits.append(circuit)

        return circuits

    def qubit_state_vector_check(self, operation, par, wires):
        """Input check for the the QubitStateVector operation."""
        if operation == "QubitStateVector":
            if self.backend_name == "unitary_simulator":
                raise DeviceError(
                    "The QubitStateVector operation "
                    "is not supported on the unitary simulator backend."
                )

            if len(par[0]) != 2 ** len(wires):
                raise ValueError("State vector must be of length 2**wires.")

    @staticmethod
    def qubit_unitary_check(operation, par, wires):
        """Input check for the the QubitUnitary operation."""
        if operation == "QubitUnitary":
            if len(par[0]) != 2 ** len(wires):
                raise ValueError(
                    "Unitary matrix must be of shape (2**wires,\
                        2**wires)."
                )

    def compile(self):
        """Compile the quantum circuit to target
        the provided compile_backend. If compile_backend is None,
        then the target is simply the backend."""
        compile_backend = self.compile_backend or self.backend
        compiled_circuits = transpile(self._circuit, backend=compile_backend)

        # Specify to have a memory for hw/hw simulators
        memory = str(compile_backend) not in self._state_backends

        return assemble(
            experiments=compiled_circuits, backend=compile_backend, shots=self.shots, memory=memory,
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

    def generate_samples(self):

        # branch out depending on the type of backend
        if self.backend_name in self._state_backends:
            # software simulator: need to sample from probabilities
            return super().generate_samples()

        # hardware or hardware simulator
        samples = self._current_job.result().get_memory()

        # reverse qubit order to match PennyLane convention
        return np.vstack([np.array([int(i) for i in s[::-1]]) for s in samples])

    @property
    def state(self):
        return self._state

    def analytic_probability(self, wires=None):
        if self._state is None:
            return None

        prob = self.marginal_prob(np.abs(self._state) ** 2, wires)
        return prob
