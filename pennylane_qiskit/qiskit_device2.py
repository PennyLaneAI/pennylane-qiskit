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


import warnings
from typing import Union, Callable, Tuple, Sequence

import numpy as np
import pennylane as qml



from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import extensions as ex
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime.constants import RunnerResult
from qiskit_ibm_runtime import Sampler, Estimator


from pennylane import transform
from pennylane.transforms.core import TransformProgram
from pennylane.transforms import broadcast_expand
from pennylane.tape import QuantumTape
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
from pennylane.measurements import ProbabilityMP, ExpectationMP, VarianceMP

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

FULL_OPERATION_MAP = {**QISKIT_OPERATION_MAP, **QISKIT_OPERATION_INVERSES_MAP}

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

    use_sampler = [mp for mp in tape.measurements if isinstance(mp, ProbabilityMP)]
    use_estimator = [mp for mp in tape.measurements if isinstance(mp, (ExpectationMP, VarianceMP))]
    other = [mp for mp in tape.measurements if not isinstance(mp, (ProbabilityMP, ExpectationMP, VarianceMP))]

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
    """Temporary transform instead of split_measurement_types - until correct splitting of
    types is implemented, only allow certain groupings of types to catch invalid circuits before executing"""

    measurement_types = set(type(mp) for mp in tape.measurements)

    if measurement_types.issubset({ExpectationMP, VarianceMP}):
        return (tape,), null_postprocessing
    if measurement_types.issubset({ProbabilityMP}):
        return (tape,), null_postprocessing

    # combination of Probability, Expectation and Variance other than the ones above not allowed
    if measurement_types.intersection({ProbabilityMP, ExpectationMP, VarianceMP}):
        raise RuntimeError("Bad measurement combination")
    return (tape,), null_postprocessing


def circuit_to_qiskit(circuit, register_size, diagonalize=True, measure=True):
    """Builds the circuit objects based on the operations and measurements
    specified to apply.

    Args:
        operations (list[~.Operation]): operations to apply to the device

    Keyword args:
        rotations (list[~.Operation]): Operations that rotate the circuit
            pre-measurement into the eigenbasis of the observables.
    """

    reg = QuantumRegister(register_size, "q")
    creg = ClassicalRegister(register_size, "c")
    qc = QuantumCircuit(register_size, register_size, name="temp")

    for op in circuit.operations:
        qc &= operation_to_qiskit(op, register_size)

    # rotate the state for measurement in the computational basis
    if diagonalize:
        rotations = circuit.diagonalizing_gates
        for rot in rotations:
            qc &= operation_to_qiskit(rot, register_size)

    if measure:
        # barrier ensures we first do all operations, then do all measurements
        qc.barrier(reg)
        # we always measure the full register
        qc.measure(reg, creg)

    return qc

def operation_to_qiskit(operation, register_size=None):
    """Take a Pennylane operator and convert to a Qiskit circuit

    Args:
        operation (List[pennylane.Operation]): operation to be converted
        num_qubits (int): the total number of qubits on the device

    Returns:
        QuantumCircuit: a quantum circuit objects containing the translated operation
    """
    op_wires = operation.wires
    par = operation.parameters

    # make quantum and classical registers for the full register
    if register_size is None:
        register_size = len(op_wires)
    reg = QuantumRegister(register_size, "q")
    creg = ClassicalRegister(register_size, "c")

    for idx, p in enumerate(par):
        if isinstance(p, np.ndarray):
            # Convert arrays so that Qiskit accepts the parameter
            par[idx] = p.tolist()

    operation = operation.name

    mapped_operation = FULL_OPERATION_MAP[operation]

    qregs = [reg[i] for i in op_wires.labels]

    adjoint = operation.startswith("Adjoint(")
    split_op = operation.split("Adjoint(")

    # Need to revert the order of the quantum registers used in
    # Qiskit such that it matches the PennyLane ordering
    if adjoint:
        if split_op[1] in ("QubitUnitary)", "QubitStateVector)", "StatePrep)"):
            qregs = list(reversed(qregs))
    else:
        if split_op[0] in ("QubitUnitary", "QubitStateVector", "StatePrep"):
            qregs = list(reversed(qregs))

    dag = circuit_to_dag(QuantumCircuit(reg, creg, name=""))
    gate = mapped_operation(*par)

    dag.apply_operation_back(gate, qargs=qregs)
    circuit = dag_to_circuit(dag)

    return circuit

def mp_to_pauli(mp, register_size):
    """Convert a Pauli observable to a SparsePauliOp for measurement via Estimator

        Args:
            mp(Union[ExpectationMP, VarianceMP]): MeasurementProcess to be converted to a SparsePauliOp
            register_size(int): total size of the qubit register being measured
    """

    # ToDo: I believe this could be extended to cover expectation values of Hamiltonians

    observables = {"PauliX": "X",
                   "PauliY": "Y",
                   "PauliZ": "Z",
                   "Identity": "I"}

    pauli_string = ["I"] * register_size
    pauli_string[mp.wires[0]] = observables[mp.name]

    pauli_string = ('').join(pauli_string)

    return SparsePauliOp(pauli_string)





class QiskitDevice2(Device):
    r"""Hardware/hardware simulator Qiskit device for PennyLane.

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

    operations = set(FULL_OPERATION_MAP.keys())
    observables = {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Identity",
        "Hadamard",
        "Hermitian",
        "Projector",
    }

    @property
    def name(self):
        """The name of the device."""
        return "qiskit.remote2"

    def __init__(self, wires, backend, shots=1024, use_primitives=False, **kwargs):
        super().__init__(wires=wires, shots=shots)

        self._backend = backend

        self.runtime_service = QiskitRuntimeService(channel="ibm_quantum")
        self._use_primitives = use_primitives

        # Keep track if the user specified analytic to be True
        if shots is None:
            # Raise a warning if no shots were specified for a hardware device
            warnings.warn(f"The analytic calculation of expectations, variances and probabilities "
                          f"is only supported on statevector backends, not on the {backend.name}. Such "
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

    def compile_circuits(self, circuits):
        r"""Compiles multiple circuits one after the other.

        Args:
            circuits (list[QuantumCircuit]): the circuits to be compiled

        Returns:
             list[QuantumCircuit]: the list of compiled circuits
        """
        # Compile each circuit object
        compiled_circuits = []

        for circuit in circuits:
            compiled_circ = transpile(circuit, backend=self.backend)
            compiled_circ.name = f"circ{len(compiled_circuits)}"
            compiled_circuits.append(compiled_circ)

        return compiled_circuits

    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:

        first_measurement = circuits[0].measurements[0]

        if isinstance(first_measurement, (ExpectationMP, VarianceMP)) and self._use_primitives:
            # get results from Estimator
            results = self._execute_estimator(circuits)
        elif isinstance(first_measurement, ProbabilityMP) and self._use_primitives:
            # get results from Sampler
            results = self._execute_sampler(circuits)
        else:
            # the old run_service execution (this is the only option sample-based measurements)
            qcirc = [circuit_to_qiskit(circ, self.num_wires, diagonalize=True, measure=True) for circ in circuits]
            compiled_circuits = self.compile_circuits(qcirc)

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

        qcirc = [circuit_to_qiskit(circ, self.num_wires, diagonalize=True, measure=True) for circ in circuits]
        results = []

        with Session(backend=self.backend):
            sampler = Sampler()

            for qc in qcirc:
                result = sampler.run(qc).result()
                results.append(result.quasi_dists[0])

        return results

    def _execute_estimator(self, circuits):

        # initially we assume only one measurement per tape
        qcirc = [circuit_to_qiskit(circ, self.num_wires, diagonalize=False, measure=False) for circ in circuits]
        results = []

        with Session(backend=self.backend):
            estimator = Estimator()

            for circ, qc in zip(circuits, qcirc):
                pauli_observables = [mp_to_pauli(mp, self.num_wires) for mp in circ.observables]
                result = estimator.run([qc]*len(pauli_observables), pauli_observables).result()
                result = self._process_estimator_job(circ.measurements, result)
                results.append(result)
        # raise NotImplementedError
        return results

    def _process_estimator_job(self, measurements, job_result):

        expvals = job_result.values
        variances = [res["variance"] for res in job_result.metadata]

        result = []
        for i, mp in enumerate(measurements):
            if isinstance(mp, ExpectationMP):
                result.append(expvals[i])
            elif isinstance(mp, VarianceMP):
                result.append(variances[i])

        return result

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
