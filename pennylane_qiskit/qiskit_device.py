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
import pennylane
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import extensions as ex
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit

from pennylane import QubitDevice, DeviceError

from ._version import __version__


QISKIT_OPERATION_MAP_SELF_ADJOINT = {
    # native PennyLane operations also native to qiskit
    "PauliX": ex.XGate,
    "PauliY": ex.YGate,
    "PauliZ": ex.ZGate,
    "Hadamard": ex.HGate,
    "CNOT": ex.CXGate,
    "CZ": ex.CZGate,
    "SWAP": ex.SwapGate,
    "ISWAP": ex.iSwapGate,
    "RX": ex.RXGate,
    "RY": ex.RYGate,
    "RZ": ex.RZGate,
    "Identity": ex.IGate,
    "CSWAP": ex.CSwapGate,
    "CRX": ex.CRXGate,
    "CRY": ex.CRYGate,
    "CRZ": ex.CRZGate,
    "PhaseShift": ex.PhaseGate,
    "QubitStateVector": ex.Initialize,
    "Toffoli": ex.CCXGate,
    "QubitUnitary": ex.UnitaryGate,
    "U1": ex.U1Gate,
    "U2": ex.U2Gate,
    "U3": ex.U3Gate,
    "IsingZZ": ex.RZZGate,
    "IsingYY": ex.RYYGate,
    "IsingXX": ex.RXXGate,
}

QISKIT_OPERATION_INVERSES_MAP_SELF_ADJOINT = {
    "Adjoint(" + k + ")": v for k, v in QISKIT_OPERATION_MAP_SELF_ADJOINT.items()
}

# Separate dictionary for the inverses as the operations dictionary needs
# to be invertible for the conversion functionality to work
QISKIT_OPERATION_MAP_NON_SELF_ADJOINT = {"S": ex.SGate, "T": ex.TGate, "SX": ex.SXGate}
QISKIT_OPERATION_INVERSES_MAP_NON_SELF_ADJOINT = {
    "Adjoint(S)": ex.SdgGate,
    "Adjoint(T)": ex.TdgGate,
    "Adjoint(SX)": ex.SXdgGate,
}

QISKIT_OPERATION_MAP = {
    **QISKIT_OPERATION_MAP_SELF_ADJOINT,
    **QISKIT_OPERATION_MAP_NON_SELF_ADJOINT,
}
QISKIT_OPERATION_INVERSES_MAP = {
    **QISKIT_OPERATION_INVERSES_MAP_SELF_ADJOINT,
    **QISKIT_OPERATION_INVERSES_MAP_NON_SELF_ADJOINT,
}


class QiskitDevice(QubitDevice, abc.ABC):
    r"""Abstract Qiskit device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        provider (Provider): The Qiskit simulation provider
        backend (str): the desired backend
        shots (int or None): number of circuit evaluations/random samples used
            to estimate expectation values and variances of observables. For state vector backends,
            setting to ``None`` results in computing statistics like expectation values and variances analytically.

    Keyword Args:
        name (str): The name of the circuit. Default ``'circuit'``.
        compile_backend (BaseBackend): The backend used for compilation. If you wish
            to simulate a device compliant circuit, you can specify a backend here.
    """
    name = "Qiskit PennyLane plugin"
    pennylane_requires = ">=0.30.0"
    version = __version__
    plugin_version = __version__
    author = "Xanadu"

    _capabilities = {
        "model": "qubit",
        "tensor_observables": True,
        "inverse_operations": True,
    }
    _operation_map = {**QISKIT_OPERATION_MAP, **QISKIT_OPERATION_INVERSES_MAP}
    _state_backends = {
        "statevector_simulator",
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

    hw_analytic_warning_message = (
        "The analytic calculation of expectations, variances and "
        "probabilities is only supported on statevector backends, not on the {}. "
        "Such statistics obtained from this device are estimates based on samples."
    )

    _eigs = {}

    def __init__(self, wires, provider, backend, shots=1024, **kwargs):
        super().__init__(wires=wires, shots=shots)

        # Keep track if the user specified analytic to be True
        if shots is None and backend not in self._state_backends:
            # Raise a warning if no shots were specified for a hardware device
            warnings.warn(self.hw_analytic_warning_message.format(backend), UserWarning)

            self.shots = 1024

        self._backend = None

        self.provider = provider
        self.backend_name = backend

        def _get_backend_name(name):
            return name if isinstance(name, str) else name()

        self._capabilities["backend"] = [
            _get_backend_name(b.name) for b in self.provider.backends()
        ]
        self._capabilities["returns_state"] = backend in self._state_backends

        # Check that the backend exists
        if backend not in self._capabilities["backend"]:
            raise ValueError(
                f"Backend '{backend}' does not exist. Available backends "
                f"are:\n {self._capabilities['backend']}"
            )

        # Perform validation against backend
        b = self.backend
        if len(self.wires) > int(b.configuration().n_qubits):
            raise ValueError(
                f"Backend '{backend}' supports maximum {b.configuration().n_qubits} wires"
            )

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
        """The Qiskit simulation backend object.

        Returns:
            qiskit.providers.backend: Qiskit backend object.
        """
        if self._backend is None:
            self._backend = self.provider.get_backend(self.backend_name)
        return self._backend

    def reset(self):
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

        if self.backend_name not in self._state_backends:
            # Add measurements if they are needed
            for qr, cr in zip(self._reg, self._creg):
                self._circuit.measure(qr, cr)
        elif "aer" in self.backend_name:
            self._circuit.save_state()

    def apply(self, operations, **kwargs):
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

            adjoint = operation.startswith("Adjoint(")
            split_op = operation.split("Adjoint(")

            if adjoint:
                if split_op[1] in ("QubitUnitary)", "QubitStateVector)"):
                    # Need to revert the order of the quantum registers used in
                    # Qiskit such that it matches the PennyLane ordering
                    qregs = list(reversed(qregs))
            else:
                if split_op[0] in ("QubitUnitary", "QubitStateVector"):
                    # Need to revert the order of the quantum registers used in
                    # Qiskit such that it matches the PennyLane ordering
                    qregs = list(reversed(qregs))

            dag = circuit_to_dag(QuantumCircuit(self._reg, self._creg, name=""))
            gate = mapped_operation(*par)

            dag.apply_operation_back(gate, qargs=qregs)
            circuit = dag_to_circuit(dag)
            circuits.append(circuit)

        return circuits

    def qubit_state_vector_check(self, operation):
        """Input check for the the QubitStateVector operation.

        Args:
            operation (pennylane.Operation): operation to be checked

        Raises:
            DeviceError: If the operation is QubitStateVector
        """
        if operation == "QubitStateVector":
            if "unitary" in self.backend_name:
                raise DeviceError(
                    "The QubitStateVector operation "
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

        if self.backend_name in self._state_backends:
            self._state = self._get_state(result)

    def _get_state(self, result, experiment=None):
        """Returns the statevector for state simulator backends.

        Args:
            result (qiskit.Result): result object
            experiment (str or None): the name of the experiment to get the state for.

        Returns:
            array[float]: size ``(2**num_wires,)`` statevector
        """
        if "statevector" in self.backend_name:
            state = np.asarray(result.get_statevector(experiment))

        elif "unitary" in self.backend_name:
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
        if self.backend_name in self._state_backends:
            # software simulator: need to sample from probabilities
            return super().generate_samples()

        # hardware or hardware simulator
        samples = self._current_job.result().get_memory(circuit)
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
        # pylint: disable=missing-function-docstring

        compiled_circuits = self.compile_circuits(circuits)

        # Send the batch of circuit objects using backend.run
        self._current_job = self.backend.run(compiled_circuits, shots=self.shots, **self.run_args)
        result = self._current_job.result(timeout=timeout)

        # increment counter for number of executions of qubit device
        self._num_executions += 1

        # Compute statistics using the state and/or samples
        results = []
        for circuit, circuit_obj in zip(circuits, compiled_circuits):
            # Update the tracker
            if self.tracker.active:
                self.tracker.update(executions=1, shots=self.shots)
                self.tracker.record()

            if self.backend_name in self._state_backends:
                self._state = self._get_state(result, experiment=circuit_obj)

            # generate computational basis samples
            if self.shots is not None or circuit.is_sampled:
                self._samples = self.generate_samples(circuit_obj)

            if not pennylane.active_return():
                res = self._statistics_legacy(circuit)
                res = np.asarray(res)
                results.append(res)
            else:
                res = self.statistics(circuit)
                single_measurement = len(circuit.measurements) == 1
                res = res[0] if single_measurement else tuple(res)
                results.append(res)

        if self.tracker.active:
            self.tracker.update(batches=1, batch_len=len(circuits))
            self.tracker.record()

        return results
