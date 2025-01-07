# Copyright 2019-2021 Xanadu Quantum Technologies Inc.

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
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.providers import Backend, BackendV2, QiskitBackendNotFoundError

from pennylane import DeviceError
from pennylane.devices import QubitDevice
from pennylane.measurements import SampleMP, CountsMP, ClassicalShadowMP, ShadowExpvalMP

from .converter import QISKIT_OPERATION_MAP
from ._version import __version__

SAMPLE_TYPES = (SampleMP, CountsMP, ClassicalShadowMP, ShadowExpvalMP)


def _get_backend_name(backend):
    try:
        return backend.name()  # BackendV1
    except TypeError:  # pragma: no cover
        return backend.name  # BackendV2


class QiskitDeviceLegacy(QubitDevice, abc.ABC):
    r"""Abstract Qiskit device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        provider (Provider | None): The Qiskit backend provider.
        backend (str | Backend): the desired backend. If a string, a provider must be given.
        shots (int or None): number of circuit evaluations/random samples used
            to estimate expectation values and variances of observables. For state vector backends,
            setting to ``None`` results in computing statistics like expectation values and variances analytically.

    Keyword Args:
        name (str): The name of the circuit. Default ``'circuit'``.
        compile_backend (BaseBackend): The backend used for compilation. If you wish
            to simulate a device compliant circuit, you can specify a backend here.
    """

    name = "Qiskit PennyLane plugin"
    pennylane_requires = ">=0.38.0"
    version = __version__
    plugin_version = __version__
    author = "Xanadu"

    _capabilities = {
        "model": "qubit",
        "tensor_observables": True,
        "inverse_operations": True,
    }
    _operation_map = QISKIT_OPERATION_MAP
    _state_backends = {
        "statevector_simulator",
        "simulator_statevector",
        "unitary_simulator",
        "aer_simulator_statevector",
        "aer_simulator_unitary",
    }
    """set[str]: Set of backend names that define the backends
    that support returning the underlying quantum statevector"""

    operations = set(_operation_map.keys())
    observables = {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Identity",
        "Hadamard",
        "Hermitian",
        "Projector",
    }

    analytic_warning_message = (
        "The analytic calculation of expectations, variances and "
        "probabilities is only supported on statevector backends, not on the {}. "
        "Such statistics obtained from this device are estimates based on samples."
    )

    _eigs = {}

    def __init__(self, wires, provider, backend, shots=1024, **kwargs):
        super().__init__(wires=wires, shots=shots)

        self.provider = provider

        if isinstance(backend, Backend):
            self._backend = backend
            self.backend_name = _get_backend_name(backend)
        elif provider is None:
            raise ValueError("Must pass a provider if the backend is not a Backend instance.")
        else:
            try:
                self._backend = provider.get_backend(backend)
            except QiskitBackendNotFoundError as e:
                available_backends = list(map(_get_backend_name, provider.backends()))
                raise ValueError(
                    f"Backend '{backend}' does not exist. Available backends "
                    f"are:\n {available_backends}"
                ) from e

            self.backend_name = _get_backend_name(self._backend)

        # Keep track if the user specified analytic to be True
        if shots is None and not self._is_state_backend:
            # Raise a warning if no shots were specified for a hardware device
            warnings.warn(self.analytic_warning_message.format(backend), UserWarning)

            self.shots = 1024

        self._capabilities["returns_state"] = self._is_state_backend

        # Perform validation against backend
        backend_qubits = (
            backend.num_qubits
            if isinstance(backend, BackendV2)
            else self.backend.configuration().n_qubits
        )
        if backend_qubits and len(self.wires) > int(backend_qubits):
            raise ValueError(f"Backend '{backend}' supports maximum {backend_qubits} wires")

        # Initialize inner state
        self.reset()

        self.process_kwargs(kwargs)

    def process_kwargs(self, kwargs):
        """Processing the keyword arguments that were provided upon device initialization.

        Args:
            kwargs (dict): keyword arguments to be set for the device
        """
        self.compile_backend = None
        if "compile_backend" in kwargs:
            self.compile_backend = kwargs.pop("compile_backend")

        if "noise_model" in kwargs:
            noise_model = kwargs.pop("noise_model")
            self.backend.set_options(noise_model=noise_model)

        # set transpile_args
        self.set_transpile_args(**kwargs)

        # Get further arguments for run
        self.run_args = {}

        # Specify to have a memory for hw/hw simulators
        compile_backend = self.compile_backend or self.backend
        memory = str(compile_backend) not in self._state_backends

        if memory:
            kwargs["memory"] = True

        # Consider the remaining kwargs as keyword arguments to run
        self.run_args.update(kwargs)

    @property
    def _is_state_backend(self):
        """Returns whether this device has a state backend."""
        return self.backend_name in self._state_backends or self.backend.options.get("method") in {
            "unitary",
            "statevector",
        }

    @property
    def _is_statevector_backend(self):
        """Returns whether this device has a statevector backend."""
        method = "statevector"
        return method in self.backend_name or self.backend.options.get("method") == method

    @property
    def _is_unitary_backend(self):
        """Returns whether this device has a unitary backend."""
        method = "unitary"
        return method in self.backend_name or self.backend.options.get("method") == method

    def set_transpile_args(self, **kwargs):
        """The transpile argument setter.

        Keyword Args:
            kwargs (dict): keyword arguments to be set for the Qiskit transpiler. For more details, see the
                `Qiskit transpiler documentation <https://qiskit.org/documentation/stubs/qiskit.compiler.transpile.html>`_
        """
        transpile_sig = inspect.signature(transpile).parameters
        self.transpile_args = {arg: kwargs[arg] for arg in transpile_sig if arg in kwargs}
        self.transpile_args.pop("circuits", None)
        self.transpile_args.pop("backend", None)

    @property
    def backend(self):
        """The Qiskit backend object.

        Returns:
            qiskit.providers.backend: Qiskit backend object.
        """
        return self._backend

    def reset(self):
        """Reset the Qiskit backend device"""
        # Reset only internal data, not the options that are determined on
        # device creation
        self._reg = QuantumRegister(self.num_wires, "q")
        self._creg = ClassicalRegister(self.num_wires, "c")
        self._circuit = QuantumCircuit(self._reg, self._creg, name="temp")

        self._current_job = None
        self._state = None  # statevector of a simulator backend

    def create_circuit_object(self, operations, **kwargs):
        """Builds the circuit objects based on the operations and measurements
        specified to apply.

        Args:
            operations (list[~.Operation]): operations to apply to the device

        Keyword args:
            rotations (list[~.Operation]): Operations that rotate the circuit
                pre-measurement into the eigenbasis of the observables.
        """
        rotations = kwargs.get("rotations", [])

        applied_operations = self.apply_operations(operations)

        # Rotating the state for measurement in the computational basis
        rotation_circuits = self.apply_operations(rotations)
        applied_operations.extend(rotation_circuits)

        for circuit in applied_operations:
            self._circuit &= circuit

        if not self._is_state_backend:
            # Add measurements if they are needed
            for qr, cr in zip(self._reg, self._creg):
                self._circuit.measure(qr, cr)
        elif "aer" in self.backend_name:
            self._circuit.save_state()

    def apply(self, operations, **kwargs):
        """Build the circuit object and apply the operations"""
        self.create_circuit_object(operations, **kwargs)

        # These operations need to run for all devices
        compiled_circuit = self.compile()
        self.run(compiled_circuit)

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

            for idx, p in enumerate(par):
                if isinstance(p, np.ndarray):
                    # Convert arrays so that Qiskit accepts the parameter
                    par[idx] = p.tolist()

            operation = operation.name

            mapped_operation = self._operation_map[operation]

            self.qubit_state_vector_check(operation)

            qregs = [self._reg[i] for i in device_wires.labels]

            if operation in ("QubitUnitary", "StatePrep"):
                # Need to revert the order of the quantum registers used in
                # Qiskit such that it matches the PennyLane ordering
                qregs = list(reversed(qregs))

            if operation in ("Barrier",):
                # Need to add the num_qubits for instantiating Barrier in Qiskit
                par = [len(self._reg)]

            dag = circuit_to_dag(QuantumCircuit(self._reg, self._creg, name=""))

            gate = mapped_operation(*par)

            dag.apply_operation_back(gate, qargs=qregs)
            circuit = dag_to_circuit(dag)
            circuits.append(circuit)

        return circuits

    def qubit_state_vector_check(self, operation):
        """Input check for the StatePrepBase operations.

        Args:
            operation (pennylane.Operation): operation to be checked

        Raises:
            DeviceError: If the operation is StatePrep
        """
        if operation == "StatePrep":
            if self._is_unitary_backend:
                raise DeviceError(
                    f"The {operation} operation "
                    "is not supported on the unitary simulator backend."
                )

    def compile(self):
        """Compile the quantum circuit to target the provided compile_backend.

        If compile_backend is None, then the target is simply the
        backend.
        """
        compile_backend = self.compile_backend or self.backend
        compiled_circuits = transpile(self._circuit, backend=compile_backend, **self.transpile_args)
        return compiled_circuits

    def run(self, qcirc):
        """Run the compiled circuit and query the result.

        Args:
            qcirc (qiskit.QuantumCircuit): the quantum circuit to be run on the backend
        """
        self._current_job = self.backend.run(qcirc, shots=self.shots, **self.run_args)
        result = self._current_job.result()

        if self._is_state_backend:
            self._state = self._get_state(result)

    def _get_state(self, result, experiment=None):
        """Returns the statevector for state simulator backends.

        Args:
            result (qiskit.Result): result object
            experiment (str or None): the name of the experiment to get the state for.

        Returns:
            array[float]: size ``(2**num_wires,)`` statevector
        """
        if self._is_statevector_backend:
            state = np.asarray(result.get_statevector(experiment))

        elif self._is_unitary_backend:
            unitary = np.asarray(result.get_unitary(experiment))
            initial_state = np.zeros([2**self.num_wires])
            initial_state[0] = 1

            state = unitary @ initial_state

        # reverse qubit order to match PennyLane convention
        return state.reshape([2] * self.num_wires).T.flatten()

    def generate_samples(self, circuit=None):
        r"""Returns the computational basis samples generated for all wires.

        Note that PennyLane uses the convention :math:`|q_0,q_1,\dots,q_{N-1}\rangle` where
        :math:`q_0` is the most significant bit.

        Args:
            circuit (str or None): the name of the circuit to get the state for

        Returns:
             array[complex]: array of samples in the shape ``(dev.shots, dev.num_wires)``
        """

        # branch out depending on the type of backend
        if self._is_state_backend:
            # software simulator: need to sample from probabilities
            return super().generate_samples()

        # hardware or hardware simulator
        samples = self._current_job.result().get_memory(circuit)
        # reverse qubit order to match PennyLane convention
        return np.vstack([np.array([int(i) for i in s[::-1]]) for s in samples])

    @property
    def state(self):
        """Get state of the device"""
        return self._state

    def analytic_probability(self, wires=None):
        """Get the analytic probability of the device"""
        if self._state is None:
            return None

        prob = self.marginal_prob(np.abs(self._state) ** 2, wires)
        return prob

    def compile_circuits(self, circuits):
        r"""Compiles multiple circuits one after the other.

        Args:
            circuits (list[.tapes.QuantumTape]): the circuits to be compiled

        Returns:
             list[QuantumCircuit]: the list of compiled circuits
        """
        # Compile each circuit object
        compiled_circuits = []

        for circuit in circuits:
            # We need to reset the device here, else it will
            # not start the next computation in the zero state
            self.reset()
            self.create_circuit_object(circuit.operations, rotations=circuit.diagonalizing_gates)

            compiled_circ = self.compile()
            compiled_circ.name = f"circ{len(compiled_circuits)}"
            compiled_circuits.append(compiled_circ)

        return compiled_circuits

    def batch_execute(self, circuits, timeout: int = None):
        """Batch execute the circuits on the device"""

        compiled_circuits = self.compile_circuits(circuits)

        if not compiled_circuits:
            # At least one circuit must always be provided to the backend.
            return []

        # Send the batch of circuit objects using backend.run
        self._current_job = self.backend.run(compiled_circuits, shots=self.shots, **self.run_args)

        try:
            result = self._current_job.result(timeout=timeout)
        except TypeError:  # pragma: no cover
            # timeout not supported
            result = self._current_job.result()

        # increment counter for number of executions of qubit device
        # pylint: disable=no-member
        self._num_executions += 1

        # Compute statistics using the state and/or samples
        results = []
        for circuit, circuit_obj in zip(circuits, compiled_circuits):
            # Update the tracker
            if self.tracker.active:
                self.tracker.update(executions=1, shots=self.shots)
                self.tracker.record()

            if self._is_state_backend:
                self._state = self._get_state(result, experiment=circuit_obj)

            # generate computational basis samples
            if self.shots is not None or any(
                isinstance(m, SAMPLE_TYPES) for m in circuit.measurements
            ):
                self._samples = self.generate_samples(circuit_obj)

            res = self.statistics(circuit)
            single_measurement = len(circuit.measurements) == 1
            res = res[0] if single_measurement else tuple(res)
            results.append(res)

        if self.tracker.active:
            self.tracker.update(batches=1, batch_len=len(circuits))
            self.tracker.record()

        return results
