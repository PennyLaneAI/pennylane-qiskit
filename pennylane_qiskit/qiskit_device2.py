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
# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init, missing-function-docstring


import warnings
import inspect
from typing import Union, Callable, Tuple, Sequence
from contextlib import contextmanager

import numpy as np
import pennylane as qml

from qiskit.compiler import transpile
from qiskit.providers import BackendV2

from qiskit_ibm_runtime import Session, Sampler, Estimator
from qiskit_ibm_runtime.constants import RunnerResult
from qiskit_ibm_runtime.options import Options

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
from pennylane.measurements import ProbabilityMP, ExpectationMP, VarianceMP

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

            # then this loop executes immediately after without qeueing again
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
        tapes.extend(
            [qml.tape.QuantumScript(tape.operations, measurements=[mp for mp, i in estimator])]
        )
    if sampler:
        tapes.extend(
            [qml.tape.QuantumScript(tape.operations, measurements=[mp for mp, i in sampler])]
        )
    if no_prim:
        tapes.extend(
            [qml.tape.QuantumScript(tape.operations, measurements=[mp for mp, i in no_prim])]
        )

    def reorder_fn(res):
        """re-order the output to the original shape and order"""

        flattened_indices = [i for group in order_indices for i in group]
        flattened_results = [r for group in res for r in group]

        result = dict(zip(flattened_indices, flattened_results))

        return tuple(result[i] for i in sorted(result.keys()))

    return tapes, reorder_fn


def qiskit_options_to_flat_dict(options):
    """Create a dictionary from a Qiskit Options object"""
    # this will break (or at least overwrite potentially relevant information)
    # if they name things in some categories on Options the same as things in
    # other categories, but at that point they've really departed from the kwarg API
    options_dict = {}
    for key, val in vars(options).items():
        if not hasattr(val, "__dict__"):
            if val is not None:
                options_dict[key] = val

        else:
            for k, v in vars(val).items():
                if v is not None:
                    options_dict[k] = v

    return options_dict


class QiskitDevice2(Device):
    r"""Hardware/hardware simulator Qiskit device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        backend (Backend): the initialized Qiskit backend

    Keyword Args:
        shots (int or None): number of circuit evaluations/random samples used
            to estimate expectation values and variances of observables.
        use_primitives(bool): whether or not to use Qiskit Primitives. Defaults to False. If True,
            getting expectation values and variance from the backend will use a Qiskit Estimator,
            and getting probabilities will use a Qiskit Sampler. Other measurement types will continue
            to return results from the backend without using a Primitive.
        options (Options): a Qiskit Options object for specifying handling the Qiskit task
            (transpiliation, error mitigation, execution, etc). Defaults to None. See Qiskit documentation
            for more details.
        session (Session): a Qiskit Session to use for device execution. If none is provided, a session will
            be created at each device execution.
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

    @property
    def name(self):
        """The name of the device."""
        return "qiskit.remote2"

    # pylint:disable = too-many-arguments
    def __init__(
        self, wires, backend, shots=1024, use_primitives=False, options=None, session=None, **kwargs
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

        # use device shots, even if shots are defined on the Options - to do this, we update the Options
        # shots is a required field in the Options object, and Primitive based measurements *will* use it
        if options and options.execution.shots != 4000:  # 4000 is the default value on Options
            warnings.warn(
                f"Setting shots via the Options is not supported on PennyLane devices. The shots {shots} "
                f"passed to the device will be used."
            )
        self.options = options or Options()
        self.options.execution.shots = shots
        super().__init__(wires=wires, shots=shots)

        self._backend = backend

        # ToDo: possibly things fail if this is not a QiskitRuntimeService - confirm and decide how to handle (SC 55725)
        self._service = backend._service
        self._use_primitives = use_primitives
        self._session = session

        # initial kwargs are saved and referenced every time the kwargs used for transpilation and execution
        self._init_kwargs = kwargs
        # _kwargs are used instead of the Options for performing raw sample based measurements (using old Qiskit API)
        # the _kwargs are a combination of information from Options and _init_kwargs
        self._kwargs = None
        if self.options.simulator.noise_model:
            self.backend.set_options(noise_model=self.options.simulator.noise_model)

        # Perform validation against backend
        available_qubits = backend.num_qubits if isinstance(backend, BackendV2) else backend.configuration().n_qubits
        if len(self.wires) > int(available_qubits):
            raise ValueError(
                f"Backend '{backend}' supports maximum {available_qubits} wires"
            )

        self.reset()
        self._update_kwargs()

    @property
    def backend(self):
        """The Qiskit backend object.

        Returns:
            qiskit.providers.backend: Qiskit backend object.
        """
        return self._backend

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
            decompose,
            stopping_condition=self.stopping_condition,
            name=self.name,
            skip_initial_state_prep=False,
        )
        transform_program.add_transform(
            validate_measurements, sample_measurements=accepted_sample_measurement, name=self.name
        )
        transform_program.add_transform(
            validate_observables,
            stopping_condition=self.observable_stopping_condition,
            name=self.name,
        )

        transform_program.add_transform(broadcast_expand)
        # missing: split non-commuting, sum_expand, etc.

        if self._use_primitives:
            transform_program.add_transform(split_measurement_types)

        return transform_program, config

    def _update_kwargs(self):
        """Combine the settings defined in options and the settings passed as kwargs, with
        the definition in options taking precedence if there there is conflicting information"""
        option_kwargs = qiskit_options_to_flat_dict(self.options)

        overlapping_kwargs = set(self._init_kwargs).intersection(set(option_kwargs))
        if overlapping_kwargs:
            warnings.warn(
                f"The keyword argument(s) {overlapping_kwargs} passed to the device are also "
                f"defined in the device Options. The definition in Options will be used."
            )
        if (
            option_kwargs["shots"] != 4000 and option_kwargs["shots"] != self.shots.total_shots
        ):  # 4000 is the default value on Options
            warnings.warn(
                f"Setting shots via the Options is not supported on PennyLane devices. The shots {self.shots} "
                f"passed to the device will be used."
            )
            self.options.execution.shots = self.shots.total_shots

        option_kwargs.pop("shots")
        kwargs = self._init_kwargs.copy()
        kwargs.update(option_kwargs)

        self._kwargs = kwargs

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

        for circuit in circuits:
            compiled_circ = transpile(circuit, backend=self.backend, **transpile_args)
            compiled_circ.name = f"circ{len(compiled_circuits)}"
            compiled_circuits.append(compiled_circ)

        return compiled_circuits

    # pylint: disable=unused-argument
    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        session = self._session or Session(backend=self.backend)

        if not self._use_primitives:
            results = self._execute_runtime_service(circuits, session=session)
            return results

        results = []

        if isinstance(circuits, (QuantumTape, QuantumScript)):
            circuits = [circuits]

        for circ in circuits:
            if isinstance(circ.measurements[0], (ExpectationMP, VarianceMP)):
                execute_fn = self._execute_estimator
            elif isinstance(circ.measurements[0], ProbabilityMP):
                execute_fn = self._execute_sampler
            else:
                execute_fn = self._execute_runtime_service
            results.append(execute_fn(circ, session))

        if self._session is None:
            # if this was not a session set on the device, but just one created for this execution, close
            session.close()

        return results

    def _execute_runtime_service(self, circuits, session):
        """Execution using old runtime_service (can't use runtime sessions)"""

        # update kwargs in case Options has been modified since last execution
        self._update_kwargs()

        # in case a single circuit is passed
        if isinstance(circuits, (QuantumTape, QuantumScript)):
            circuits = [circuits]

        qcirc = [
            circuit_to_qiskit(circ, self.num_wires, diagonalize=True, measure=True)
            for circ in circuits
        ]
        compiled_circuits = self.compile_circuits(qcirc)

        program_inputs = {"circuits": compiled_circuits, "shots": self.shots.total_shots}

        for kwarg in self._kwargs:
            program_inputs[kwarg] = self._kwargs.get(kwarg)

        backend_name = self.backend.name if isinstance(self.backend, BackendV2) else self.backend.configuration().backend_name

        options = {
            "backend": backend_name,
            "log_level": self.options.environment.log_level,
            "job_tags": self.options.environment.job_tags,
            "max_execution_time": self.options.max_execution_time,
        }

        # Send circuits to the cloud for execution by the circuit-runner program.
        job = self.service.run(
            program_id="circuit-runner",
            options=options,
            inputs=program_inputs,
            session_id=session.session_id,
        )
        self._current_job = job.result(decoder=RunnerResult)

        results = []

        for index, circuit in enumerate(circuits):
            self._samples = self.generate_samples(index)
            res = [
                mp.process_samples(self._samples, wire_order=self.wires)
                for mp in circuit.measurements
            ]
            single_measurement = len(circuit.measurements) == 1
            res = res[0] if single_measurement else tuple(res)
            results.append(res)

        return tuple(results)

    def _execute_sampler(self, circuit, session):
        """Execution for the Sampler primitive"""

        qcirc = circuit_to_qiskit(circuit, self.num_wires, diagonalize=True, measure=True)

        sampler = Sampler(session=session, options=self.options)

        result = sampler.run(qcirc).result()

        # needs processing function to convert to the correct format for states, and
        # also handle instances where wires were specified in probs, and for multiple probs measurements
        # single_measurement = len(circuit.measurements) == 1
        # res = (res[0], ) if single_measurement else tuple(res)

        return (result.quasi_dists[0],)

    def _execute_estimator(self, circuit, session):
        # the Estimator primitive takes care of diagonalization and measurements itself,
        # so diagonalizing gates and measurements are not included in the circuit
        qcirc = circuit_to_qiskit(circuit, self.num_wires, diagonalize=False, measure=False)

        estimator = Estimator(session=session, options=self.options)

        # split into one call per measurement
        # could technically be more efficient if there are some observables where we ask
        # for expectation value and variance on the same observable, but spending time on
        # that right now feels excessive
        pauli_observables = [mp_to_pauli(mp, self.num_wires) for mp in circuit.measurements]
        result = estimator.run([qcirc] * len(pauli_observables), pauli_observables).result()
        self._current_job = result
        result = self._process_estimator_job(circuit.measurements, result)

        return result

    @staticmethod
    def _process_estimator_job(measurements, job_result):
        """Estimator returns both expectation value and variance for each observable measured,
        along with some metadata. Extract the relevant number for each measurement process and
        return the requested results from the Estimator executions."""

        expvals = job_result.values
        variances = [res["variance"] for res in job_result.metadata]

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
            for _ in range(0, value):
                samples.append(key)
        return np.vstack([np.array([int(i) for i in s[::-1]]) for s in samples])
