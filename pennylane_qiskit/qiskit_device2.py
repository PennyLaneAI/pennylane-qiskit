# Copyright 2019-2024 Xanadu Quantum Technologies Inc.

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
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init, missing-function-docstring


import warnings
import inspect
from typing import Union, Callable, Tuple, Sequence
from contextlib import contextmanager

import numpy as np
import pennylane as qml
from qiskit.compiler import transpile
from qiskit.providers import BackendV2

from qiskit_ibm_runtime import Session, SamplerV2 as Sampler, EstimatorV2 as Estimator

from pennylane import transform
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
)
from pennylane.measurements import ExpectationMP, VarianceMP

from ._version import __version__
from .converter import QISKIT_OPERATION_MAP, circuit_to_qiskit, mp_to_pauli

QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
Result_or_ResultBatch = Union[Result, ResultBatch]


# pylint: disable=protected-access
@contextmanager
def qiskit_session(device):
    """A context manager that creates a Qiskit Session and sets it as a session
    on the device while the context manager is active. Using the context manager
    will ensure the Session closes properly and is removed from the device after
    completing the tasks.

    Args:
        device (QiskitDevice2): the device that will create remote tasks using the session

    **Example:**

    .. code-block:: python

        import pennylane as qml
        from pennylane_qiskit import qiskit_session
        from qiskit_ibm_runtime import QiskitRuntimeService

        # get backend
        service = QiskitRuntimeService(channel="ibm_quantum")
        backend = service.least_busy(simulator=False, operational=True)

        # initialize device
        dev = qml.device('qiskit.remote', wires=2, backend=backend)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, 0)
            qml.CNOT([0, 1])
            return qml.expval(qml.PauliZ(1))

        angle = 0.1

        with qiskit_session(dev) as session:

            res = circuit(angle)[0] # you queue for the first execution

            # then this loop executes immediately after without queueing again
            while res > 0:
                angle += 0.3
                res = circuit(angle)[0]
    """
    # Code to acquire session:
    existing_session = device._session
    session = Session(backend=device.backend)
    device._session = session
    try:
        yield session
    finally:
        # Code to release session:
        session.close()
        device._session = existing_session


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
def split_execution_types(
    tape: qml.tape.QuantumTape,
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Split into separate tapes based on measurement type. Counts and sample-based measurements
    will use the Qiskit Sampler. ExpectationValue and Variance will use the Estimator, except
    when the measured observable does not have a `pauli_rep`. In that case, the Sampler will be
    used, and the raw samples will be processed to give an expectation value."""

    estimator = []
    sampler = []

    for i, mp in enumerate(tape.measurements):
        if isinstance(mp, (ExpectationMP, VarianceMP)):
            if mp.obs.pauli_rep:
                estimator.append((mp, i))
            else:
                warnings.warn(
                    f"The observable measured {mp.obs} does not have a `pauli_rep` "
                    "and will be run without using the Estimator primitive. Instead, "
                    "raw samples from the Sampler will be used."
                )
                sampler.append((mp, i))
        else:
            sampler.append((mp, i))

    order_indices = [[i for mp, i in group] for group in [estimator, sampler]]

    tapes = []
    if estimator:
        tapes.extend(
            [
                qml.tape.QuantumScript(
                    tape.operations,
                    measurements=[mp for mp, i in estimator],
                    shots=tape.shots,
                )
            ]
        )
    if sampler:
        tapes.extend(
            [
                qml.tape.QuantumScript(
                    tape.operations,
                    measurements=[mp for mp, i in sampler],
                    shots=tape.shots,
                )
            ]
        )

    def reorder_fn(res):
        """re-order the output to the original shape and order"""

        flattened_indices = [i for group in order_indices for i in group]
        flattened_results = [r for group in res for r in group]

        result = dict(zip(flattened_indices, flattened_results))

        result = tuple(result[i] for i in sorted(result.keys()))

        return result[0] if len(result) == 1 else result

    return tapes, reorder_fn


class QiskitDevice2(Device):
    r"""Hardware/simulator Qiskit device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['aux_wire', 'q1', 'q2']``).
        backend (Backend): the initialized Qiskit backend

    Keyword Args:
        shots (int or None): number of circuit evaluations/random samples used
            to estimate expectation values and variances of observables.
        session (Session): a Qiskit Session to use for device execution. If none is provided, a session will
            be created at each device execution.
        compile_backend (Union[Backend, None]): the backend to be used for compiling the circuit that will be
            sent to the backend device, to be set if the backend desired for compliation differs from the
            backend used for execution. Defaults to ``None``, which means the primary backend will be used.
        **kwargs: transpilation and runtime kwargs to be used for measurements without Qiskit Primitives.
            If any values are defined both in ``options`` and in the remaining ``kwargs``, the value
            provided in ``options`` will take precedence. These kwargs will be ignored for all Primitive-based
            measurements on the device.
    """

    operations = set(QISKIT_OPERATION_MAP.keys())
    observables = {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Identity",
        "Hadamard",
        "Hermitian",
        "Projector",
    }

    # pylint:disable = too-many-arguments
    def __init__(
        self,
        wires,
        backend,
        shots=1024,
        session=None,
        compile_backend=None,
        **kwargs,
    ):

        if shots is None:
            warnings.warn(
                "Expected an integer number of shots, but received shots=None. Defaulting "
                "to 1024 shots. The analytic calculation of results is not supported on "
                "this device. All statistics obtained from this device are estimates based "
                "on samples.",
                UserWarning,
            )

            shots = 1024

        super().__init__(wires=wires, shots=shots)

        self._backend = backend
        self._compile_backend = compile_backend if compile_backend else backend

        self._service = getattr(backend, "_service", None)
        self._session = session

        # initial kwargs are saved and referenced every time the kwargs used for transpilation and execution
        self._init_kwargs = kwargs
        # _kwargs are used instead of the Options for performing raw sample based measurements (using old Qiskit API)
        # the _kwargs are a combination of information from Options and _init_kwargs
        self._kwargs = kwargs

        # Perform validation against backend
        available_qubits = (
            backend.num_qubits
            if isinstance(backend, BackendV2)
            else backend.configuration().n_qubits
        )
        if len(self.wires) > int(available_qubits):
            raise ValueError(f"Backend '{backend}' supports maximum {available_qubits} wires")

        self.reset()
        # ToDo: process or update kwargs in some fashion here

    @property
    def backend(self):
        """The Qiskit backend object.

        Returns:
            qiskit.providers.Backend: Qiskit backend object.
        """
        return self._backend

    @property
    def compile_backend(self):
        """The ``compile_backend`` is a Qiskit backend object to be used for transpilation.
        Returns:
            qiskit.providers.backend: Qiskit backend object.
        """
        return self._compile_backend

    @property
    def service(self):
        """The QiskitRuntimeService service.

        Returns:
            qiskit.qiskit_ibm_runtime.QiskitRuntimeService
        """
        return self._service

    @property
    def session(self):
        """The QiskitRuntimeService session.

        Returns:
            qiskit.qiskit_ibm_runtime.Session
        """
        return self._session

    @property
    def num_wires(self):
        return len(self.wires)

    def update_session(self, session):
        self._session = session

    def reset(self):
        self._current_job = None

    def stopping_condition(self, op: qml.operation.Operator) -> bool:
        """Specifies whether or not an Operator is accepted by QiskitDevice2."""
        return op.name in self.operations

    def observable_stopping_condition(self, obs: qml.operation.Operator) -> bool:
        """Specifies whether or not an observable is accepted by QiskitDevice2."""
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
            decompose,
            stopping_condition=self.stopping_condition,
            name=self.name,
            skip_initial_state_prep=False,
        )
        transform_program.add_transform(
            validate_measurements,
            sample_measurements=accepted_sample_measurement,
            name=self.name,
        )
        transform_program.add_transform(
            validate_observables,
            stopping_condition=self.observable_stopping_condition,
            name=self.name,
        )

        transform_program.add_transform(broadcast_expand)
        # missing: split non-commuting, sum_expand, etc. [SC-62047]

        transform_program.add_transform(split_execution_types)

        return transform_program, config

    @staticmethod
    def get_transpile_args(kwargs):
        """The transpile argument setter.

        Keyword Args:
            kwargs (dict): keyword arguments to be set for the Qiskit transpiler. For more details, see the
                `Qiskit transpiler documentation <https://qiskit.org/documentation/stubs/qiskit.compiler.transpile.html>`_
        """

        transpile_sig = inspect.signature(transpile).parameters

        transpile_args = {arg: kwargs[arg] for arg in transpile_sig if arg in kwargs}
        transpile_args.pop("circuits", None)
        transpile_args.pop("backend", None)

        return transpile_args

    def compile_circuits(self, circuits):
        r"""Compiles multiple circuits one after the other.

        Args:
            circuits (list[QuantumCircuit]): the circuits to be compiled

        Returns:
             list[QuantumCircuit]: the list of compiled circuits
        """
        # Compile each circuit object
        compiled_circuits = []
        transpile_args = self.get_transpile_args(self._kwargs)

        for i, circuit in enumerate(circuits):
            compiled_circ = transpile(circuit, backend=self.compile_backend, **transpile_args)
            compiled_circ.name = f"circ{i}"
            compiled_circuits.append(compiled_circ)

        return compiled_circuits

    # pylint: disable=unused-argument, no-member
    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        session = self._session or Session(backend=self.backend)

        results = []

        if isinstance(circuits, QuantumScript):
            circuits = [circuits]

        @contextmanager
        def execute_circuits(session):
            try:
                for circ in circuits:
                    if circ.shots and len(circ.shots.shot_vector) > 1:
                        raise ValueError(
                            f"Setting shot vector {circ.shots.shot_vector} is not supported for {self.name}."
                            "Please use a single integer instead when specifying the number of shots."
                        )
                    if isinstance(circ.measurements[0], (ExpectationMP, VarianceMP)) and getattr(
                        circ.measurements[0].obs, "pauli_rep", None
                    ):
                        execute_fn = self._execute_estimator
                    else:
                        execute_fn = self._execute_sampler
                    results.append(execute_fn(circ, session))
                yield results
            finally:
                session.close()

        with execute_circuits(session) as results:
            return results

    def _execute_sampler(self, circuit, session):
        """Execution for the Sampler primitive"""

        qcirc = [circuit_to_qiskit(circuit, self.num_wires, diagonalize=True, measure=True)]
        sampler = Sampler(session=session)
        compiled_circuits = self.compile_circuits(qcirc)

        result = sampler.run(compiled_circuits).result()[0]
        classical_register_name = compiled_circuits[0].cregs[0].name
        self._current_job = getattr(result.data, classical_register_name)

        results = []

        # needs processing function to convert to the correct format for states, and
        # also handle instances where wires were specified in probs, and for multiple probs measurements
        # single_measurement = len(circuit.measurements) == 1
        # res = (res[0], ) if single_measurement else tuple(res)

        for index, circ in enumerate([circuit]):
            self._samples = self.generate_samples(index)
            res = [
                mp.process_samples(self._samples, wire_order=self.wires) for mp in circ.measurements
            ]
            single_measurement = len(circ.measurements) == 1
            res = res[0] if single_measurement else tuple(res)
            results.append(res)

        return tuple(results)

    def _execute_estimator(self, circuit, session):
        # the Estimator primitive takes care of diagonalization and measurements itself,
        # so diagonalizing gates and measurements are not included in the circuit
        qcirc = [circuit_to_qiskit(circuit, self.num_wires, diagonalize=False, measure=False)]
        estimator = Estimator(session=session)

        compiled_circuits = self.compile_circuits(qcirc)
        # split into one call per measurement
        # could technically be more efficient if there are some observables where we ask
        # for expectation value and variance on the same observable, but spending time on
        # that right now feels excessive

        pauli_observables = [mp_to_pauli(mp, self.num_wires) for mp in circuit.measurements]
        compiled_circuits *= len(pauli_observables)
        circ_and_obs = [
            (compiled_circuits[i], pauli_observables[i]) for i in range(len(pauli_observables))
        ]
        result = estimator.run(circ_and_obs).result()
        self._current_job = result
        result = self._process_estimator_job(circuit.measurements, result)

        return result

    @staticmethod
    def _process_estimator_job(measurements, job_result):
        """Estimator returns both expectation value and variance for each observable measured,
        along with some metadata. Extract the relevant number for each measurement process and
        return the requested results from the Estimator executions."""

        expvals = [res.data.evs.item() for res in job_result]
        variances = [
            res.data.stds.item() ** 2 * 4096 for res in job_result
        ]  # this 4096 is the # of shots Qiskit uses by default. It is hard-coded here.
        # ToDo: Track the # of shots and use that instead of hard-coding
        # to calculate the variance.

        result = []
        for i, mp in enumerate(measurements):
            if isinstance(mp, ExpectationMP):
                result.append(expvals[i])
            elif isinstance(mp, VarianceMP):
                result.append(variances[i])

        single_measurement = len(measurements) == 1
        result = (result[0],) if single_measurement else tuple(result)
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
            samples.extend([key] * value)
        return np.vstack([np.array([int(i) for i in s[::-1]]) for s in samples])
