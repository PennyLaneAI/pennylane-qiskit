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

from qiskit.compiler import transpile

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
from .converter import FULL_OPERATION_MAP, circuit_to_qiskit, mp_to_pauli

QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
Result_or_ResultBatch = Union[Result, ResultBatch]



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

    estimator = []
    sampler = []
    no_prim = []

    for i, mp in enumerate(tape.measurements):
        if isinstance(mp, (ExpectationMP, VarianceMP)):
            estimator.append((mp, i))
        elif isinstance(mp, ProbabilityMP):
            sampler.append((mp, i))
        else:
            no_prim.append((mp, i))

    order_indices = [[i for mp, i in group] for group in [estimator, sampler, no_prim]]

    tapes = []
    if estimator:
        tapes.extend([qml.tape.QuantumScript(tape.operations, measurements=[mp for mp, i in estimator])])
    if sampler:
        tapes.extend([qml.tape.QuantumScript(tape.operations, measurements=[mp for mp, i in sampler])])
    if no_prim:
        tapes.extend([qml.tape.QuantumScript(tape.operations, measurements=[mp for mp, i in no_prim])])

    def reorder_fn(res):
        """re-order the output to the original shape and order"""
        result = {}

        for idx, r in zip(order_indices, res):
            result.update({k: v for k, v in zip(idx, r)})

        return tuple([result[i] for i in sorted(result.keys())])

    return tapes, reorder_fn


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

        if self._use_primitives:
            transform_program.add_transform(split_measurement_types)

        # transform_program.add_transform(broadcast_expand)
        # missing: split non-commuting, sum_expand, etc.

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

        if not self._use_primitives:
            results = self._execute_runtime_service(circuits)
            return results

        results = []

        estimator_circuits = [circ for circ in circuits if isinstance(circ.measurements[0], (ExpectationMP, VarianceMP))]
        sampler_circuits = [circ for circ in circuits if isinstance(circ.measurements[0], ProbabilityMP)]
        other_circuts = [circ for circ in circuits if not isinstance(circ.measurements[0], (ExpectationMP, VarianceMP, ProbabilityMP))]

        if estimator_circuits:
            # get results from Estimator
            result = self._execute_estimator(estimator_circuits)
            results.append(result)
        if sampler_circuits:
            # get results from Sampler
            result = self._execute_sampler(sampler_circuits)
            results.append(result)
        if other_circuts:
            # still use the old run_service execution (this is the only option sample-based measurements)
            result = self._execute_runtime_service(other_circuts)
            results.append(result)

        return results

    def _execute_runtime_service(self, circuits):
        """Execution using old runtime_service (can't use runtime sessions)"""

        print(f"using old runtime services, with circuits {circuits}")

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

        print(f"using sampler, with circuits {circuits}")

        qcirc = [circuit_to_qiskit(circ, self.num_wires, diagonalize=True, measure=True) for circ in circuits]
        results = []

        with Session(backend=self.backend):
            sampler = Sampler()

            for qc in qcirc:
                result = sampler.run(qc).result()
                results.append(result.quasi_dists[0])

        return results

    def _execute_estimator(self, circuits):

        print(f"using estimator, with circuits {circuits}")

        # initially we assume only one measurement per tape
        qcirc = [circuit_to_qiskit(circ, self.num_wires, diagonalize=False, measure=False) for circ in circuits]
        results = []

        with Session(backend=self.backend):
            estimator = Estimator()

            for circ, qc in zip(circuits, qcirc):
                pauli_observables = [mp_to_pauli(mp, self.num_wires) for mp in circ.observables]
                result = estimator.run([qc]*len(pauli_observables), pauli_observables).result()
                result = self._process_estimator_job(circ.measurements, result)
                results.extend(result)

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
