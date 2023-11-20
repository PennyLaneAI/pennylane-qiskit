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
This module contains a prototype base class for constructing Qiskit devices
for PennyLane with the new device API.
"""
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init


import abc
import inspect
import warnings

import numpy as np
import pennylane as qml

from typing import Union, Callable, Tuple, Optional, Sequence

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import extensions as ex
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.providers import Backend, QiskitBackendNotFoundError

from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime.constants import RunnerResult
from qiskit_ibm_runtime import Sampler, Estimator

from pennylane import transform
from pennylane import QubitDevice, DeviceError
from pennylane.transforms.core import TransformProgram
from pennylane.transforms import broadcast_expand
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.typing import Result, ResultBatch
from pennylane.devices import Device
from pennylane.devices.execution_config import ExecutionConfig, DefaultExecutionConfig
from pennylane.devices.preprocess import (
    decompose,
    validate_observables,
    validate_measurements,
    validate_device_wires,
    null_postprocessing
)
from pennylane.measurements import CountsMP, ExpectationMP, VarianceMP

from ._version import __version__

QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
Result_or_ResultBatch = Union[Result, ResultBatch]

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
    "StatePrep": ex.Initialize,
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

def accepted_sample_measurement(m: qml.measurements.MeasurementProcess) -> bool:
    """Specifies whether or not a measurement is accepted when sampling."""
    return isinstance(
        m,
        (
            qml.measurements.SampleMeasurement,
            qml.measurements.ClassicalShadowMP,
            qml.measurements.ShadowExpvalMP,
        ),
    )

@transform
def split_measurement_types(
    tape: qml.tape.QuantumTape,
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Split into seperate tapes based on measurement type. Counts will use the
    Qiskit Sampler, ExpectationValue and Variance will use the Estimator, and other
    strictly sample-based measurements will use the standard backend.run function"""

    use_sampler = [mp for mp in tape.measurements if isinstance(mp, CountsMP)]
    use_estimator = [mp for mp in tape.measurements if isinstance(mp, (ExpectationMP, VarianceMP))]
    other = [mp for mp in tape.measurements if not isinstance(mp, (CountsMP, ExpectationMP, VarianceMP))]

    output_tapes = []
    if use_sampler:
        output_tapes.append(qml.tape.QuantumScript(tape.operations, use_sampler, shots=tape.shots))
    if use_estimator:
        output_tapes.append(qml.tape.QuantumScript(tape.operations, use_estimator, shots=tape.shots))
    if other:
        output_tapes.append(qml.tape.QuantumScript(tape.operations, other, shots=tape.shots))

    return tuple(output_tapes), null_postprocessing

@transform
def validate_measurement_types(
        tape: qml.tape.QuantumTape,
    ) -> (Sequence[qml.tape.QuantumTape], Callable):

    measurement_types = set(type(mp) for mp in tape.measurements)

    if measurement_types.issubset({ExpectationMP, VarianceMP}):
        return (tape,), null_postprocessing
    elif measurement_types.issubset({CountsMP}):
        return (tape,), null_postprocessing
    else:
        if measurement_types.intersection({CountsMP, ExpectationMP, VarianceMP}):
            raise RuntimeError("Bad measurement combination")

    return (tape,), null_postprocessing




class QiskitDevice2(Device):
    r"""Abstract Qiskit device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        backend (str | Backend): the initialized Qiskit backend

    Keyword Args:
        shots (int or None): number of circuit evaluations/random samples used
            to estimate expectation values and variances of observables.
        use_primitives(bool): whether or not to use Qiskit Primitives and the Qiskit Runtime Session. Defaults to False.
    """
    name = "Qiskit PennyLane plugin"
    pennylane_requires = ">=0.30.0"
    version = __version__
    plugin_version = __version__
    author = "Xanadu"

    _operation_map = {**QISKIT_OPERATION_MAP, **QISKIT_OPERATION_INVERSES_MAP}

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

    short_name = "qiskit.remote2"

    def __init__(self, wires, backend, shots=1024, use_primitives=False, **kwargs):
        super().__init__(wires=wires, shots=shots)

        self._backend = backend

        self.runtime_service = QiskitRuntimeService(channel="ibm_quantum")
        self._use_primitives = use_primitives

        # Keep track if the user specified analytic to be True
        if shots is None:
            # Raise a warning if no shots were specified for a hardware device
            warnings.warn(f"The analytic calculation of expectations, variances and probabilities "
                          f"is only supported on statevector backends, not on the {backend}. Such "
                          f"statistics obtained from this device are estimates based on samples.",
                          UserWarning)

            self.shots = 1024

        # Perform validation against backend
        b = self.backend
        if len(self.wires) > int(b.configuration().n_qubits):
            raise ValueError(
                f"Backend '{backend}' supports maximum {b.configuration().n_qubits} wires"
            )

        self.reset()

    @property
    def backend(self):
        """The Qiskit backend object.

        Returns:
            qiskit.providers.backend: Qiskit backend object.
        """
        return self._backend

    @property
    def num_wires(self):
        return len(self.wires)

    def reset(self):
        # Reset only internal data, not the options that are determined on
        # device creation
        self._reg = QuantumRegister(self.num_wires, "q")
        self._creg = ClassicalRegister(self.num_wires, "c")
        self._circuit = QuantumCircuit(self._reg, self._creg, name="temp")

        self._current_job = None

    def stopping_condition(self, op: qml.operation.Operator) -> bool:
        """Specifies whether or not an observable is accepted by DefaultQubit."""
        return op.name in self.operations

    def observable_stopping_condition(self, obs: qml.operation.Operator) -> bool:
        """Specifies whether or not an observable is accepted by DefaultQubit."""
        return obs.name in self.observables

    def preprocess(
        self,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Tuple[TransformProgram, ExecutionConfig]:
        """This function defines the device transform program to be applied and an updated device configuration.

        Args:
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure describing the
                parameters needed to fully describe the execution.

        Returns:
            TransformProgram, ExecutionConfig: A transform program that when called returns QuantumTapes that the device
            can natively execute as well as a postprocessing function to be called after execution, and a configuration with
            unset specifications filled in.

        This device:

        * Supports any operations with explicit PennyLane to Qiskit gate conversions defined in the plugin
        * Does not intrinsically support parameter broadcasting

        """
        config = execution_config
        config.use_device_gradient = False

        transform_program = TransformProgram()

        transform_program.add_transform(validate_device_wires, self.wires, name=self.name)
        transform_program.add_transform(
            decompose, stopping_condition=self.stopping_condition, name=self.name
        )
        transform_program.add_transform(
            validate_measurements, sample_measurements=accepted_sample_measurement, name=self.name
        )
        transform_program.add_transform(
            validate_observables, stopping_condition=self.observable_stopping_condition, name=self.name
        )

        transform_program.add_transform(broadcast_expand)
        # transform_program.add_transform(split_measurement_types)
        transform_program.add_transform(validate_measurement_types)

        return transform_program, config

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

        for qr, cr in zip(self._reg, self._creg):
            self._circuit.measure(qr, cr)

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
            op_wires = operation.wires
            par = operation.parameters

            for idx, p in enumerate(par):
                if isinstance(p, np.ndarray):
                    # Convert arrays so that Qiskit accepts the parameter
                    par[idx] = p.tolist()

            operation = operation.name

            mapped_operation = self._operation_map[operation]

            qregs = [self._reg[i] for i in op_wires.labels]

            adjoint = operation.startswith("Adjoint(")
            split_op = operation.split("Adjoint(")

            if adjoint:
                if split_op[1] in ("QubitUnitary)", "QubitStateVector)", "StatePrep)"):
                    # Need to revert the order of the quantum registers used in
                    # Qiskit such that it matches the PennyLane ordering
                    qregs = list(reversed(qregs))
            else:
                if split_op[0] in ("QubitUnitary", "QubitStateVector", "StatePrep"):
                    # Need to revert the order of the quantum registers used in
                    # Qiskit such that it matches the PennyLane ordering
                    qregs = list(reversed(qregs))

            dag = circuit_to_dag(QuantumCircuit(self._reg, self._creg, name=""))
            gate = mapped_operation(*par)

            dag.apply_operation_back(gate, qargs=qregs)
            circuit = dag_to_circuit(dag)
            circuits.append(circuit)

        return circuits

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
            compiled_circ = transpile(self._circuit, backend=self.backend)
            compiled_circ.name = f"circ{len(compiled_circuits)}"
            compiled_circuits.append(compiled_circ)

        return compiled_circuits

    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:

        first_measurement = circuits[0].measurements[0]
        print(first_measurement)

        if isinstance(first_measurement, CountsMP) and self._use_primitives:
            print("results should come from sampler")
            results = self._execute_sampler(circuits)
        elif isinstance(first_measurement, (ExpectationMP, VarianceMP)) and self._use_primitives:
            print("results should come from estimator")
            results = self._execute_estimator(circuits)
        else:
            print("using old run function")
            compiled_circuits = self.compile_circuits(circuits)

            program_inputs = {"circuits": compiled_circuits, "shots": self.shots.total_shots}

            # for kwarg in self.kwargs:
            #     program_inputs[kwarg] = self.kwargs.get(kwarg)

            options = {"backend": self.backend.name}

            # Send circuits to the cloud for execution by the circuit-runner program.
            job = self.runtime_service.run(
                program_id="circuit-runner", options=options, inputs=program_inputs
            )
            self._current_job = job.result(decoder=RunnerResult)

            results = []

            for index, circuit in enumerate(circuits):
                self._samples = self.generate_samples(index)
                res = [mp.process_samples(self._samples, wire_order=self.wires) for mp in circuit.measurements]
                single_measurement = len(circuit.measurements) == 1
                res = res[0] if single_measurement else tuple(res)
                results.append(res)

        return results

    def _execute_sampler(self, circuits):
        """Execution for the Sampler primitive"""

        raise NotImplementedError()

        with Session(backend=self.backend):
            sampler = Sampler()

            for qc in circuits:
                result = sampler.run(qc).result()
                res.append(result.quasi_dists[0])

    def _execute_estimator(self, circuits):
        raise NotImplementedError()

    def generate_samples(self, circuit=None):
        r"""Returns the computational basis samples generated for all wires.

        Note that PennyLane uses the convention :math:`|q_0,q_1,\dots,q_{N-1}\rangle` where
        :math:`q_0` is the most significant bit.

        Args:
            circuit (int): position of the circuit in the batch.

        Returns:
             array[complex]: array of samples in the shape ``(dev.shots, dev.num_wires)``
        """
        counts = self._current_job.get_counts()
        # Batch of circuits
        if not isinstance(counts, dict):
            counts = self._current_job.get_counts()[circuit]

        samples = []
        for key, value in counts.items():
            for _ in range(0, value):
                samples.append(key)
        return np.vstack([np.array([int(i) for i in s[::-1]]) for s in samples])