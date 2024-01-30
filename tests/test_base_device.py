# Copyright 2021-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains tests for the base Qiskit device for the new PennyLane device API
"""

import numpy as np
import pytest
import inspect
from unittest.mock import patch, Mock

import pennylane as qml
from pennylane.tape.qscript import QuantumScript

from pennylane_qiskit import AerDevice
from pennylane_qiskit.qiskit_device2 import (
    QiskitDevice2,
    qiskit_session,
    accepted_sample_measurement,
    split_measurement_types,
    qiskit_options_to_flat_dict,
)
from pennylane_qiskit.converter import circuit_to_qiskit, mp_to_pauli, QISKIT_OPERATION_MAP

from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator
from qiskit_ibm_runtime.options import Options
from qiskit_ibm_runtime.constants import RunnerResult

# do not import Estimator (imported above) from qiskit.primitives - the identically
# named Estimator object has a different call signature than the remote device Estimator,
# and only runs local simulations. We need the Estimator from qiskit_ibm_runtime. They
# both use this EstimatorResults, however:
from qiskit.primitives import EstimatorResult

from qiskit import QuantumCircuit

from qiskit_aer.noise import NoiseModel


class Configuration:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.noise_model = None


class MockedBackend:
    def __init__(self, num_qubits=10):
        self._configuration = Configuration(num_qubits)
        self.options = self._configuration
        self._service = "SomeServiceProvider"

    def configuration(self):
        return self._configuration

    def set_options(self, noise_model):
        self.options.noise_model = noise_model


class MockSession:
    def __init__(self, backend, max_time=None):
        self.backend = backend
        self.max_time = max_time

try:
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.backend("ibmq_qasm_simulator")
except:
    backend = MockedBackend()

test_dev = QiskitDevice2(wires=5, backend=backend)


def options_for_testing():
    """Creates an Options object with defined values in multiple sub-categories"""
    options = Options()
    options.environment.job_tags = ["getting angle"]
    options.resilience.noise_amplifier = "placeholder"
    options.optimization_level = 2
    options.resilience_level = 1
    options.simulator.noise_model = "placeholder"
    return options


class TestDeviceInitialization:
    @pytest.mark.parametrize("use_primitives", [True, False])
    def test_use_primitives_kwarg(self, use_primitives):
        """Test the _use_primitives attribute is set on initialization"""
        dev = QiskitDevice2(wires=2, backend=backend, use_primitives=use_primitives)
        assert dev._use_primitives == use_primitives

    def test_no_shots_warns_and_defaults(self):
        """Test that initializing with shots=None raises a warning indicating that
        the device is sample based and will default to 1024 shots"""

        with pytest.warns(
            UserWarning,
            match="Expected an integer number of shots, but received shots=None",
        ):
            dev = QiskitDevice2(wires=2, backend=backend, shots=None)

        assert dev.shots.total_shots == 1024
        assert dev.options.execution.shots == 1024

    def test_kwargs_on_initialization(self, mocker):
        """Test that update_kwargs is called on intialization and combines the Options
        and kwargs as self._kwargs"""

        options = Options()
        options.environment.job_tags = ["my_tag"]

        spy = mocker.spy(QiskitDevice2, "_update_kwargs")

        dev = QiskitDevice2(
            wires=2,
            backend=backend,
            options=options,
            random_kwarg1=True,
            random_kwarg2="a",
        )

        spy.assert_called_once()

        # kwargs are updated to a combination of the information from the Options and kwargs
        assert dev._kwargs == {
            "random_kwarg1": True,
            "random_kwarg2": "a",
            "skip_transpilation": False,
            "init_qubits": True,
            "log_level": "WARNING",
            "job_tags": ["my_tag"],
        }

        # initial kwargs are saved without modification
        assert dev._init_kwargs == {"random_kwarg1": True, "random_kwarg2": "a"}

    def test_backend_wire_validation(self):
        """Test that the an error is raised if the number of device wires exceeds
        the number of wires available on the backend"""

        with pytest.raises(ValueError, match="supports maximum"):
            dev = QiskitDevice2(wires=500, backend=backend)

    def test_setting_simulator_noise_model(self):
        """Test that the simulator noise model saved on a passed Options
        object is used to set the backend noise model"""

        options = Options()
        options.simulator.noise_model = "PlaceholderForNoiseModel"

        new_backend = MockedBackend()
        dev1 = QiskitDevice2(wires=3, backend=backend)
        dev2 = QiskitDevice2(wires=3, backend=new_backend, options=options)

        assert dev1.backend.options.noise_model == None
        assert dev2.backend.options.noise_model == "PlaceholderForNoiseModel"


class TestQiskitSessionManagement:
    """Test using Qiskit sessions with the device"""

    def test_default_no_session_on_initialization(self):
        """Test that the default behaviour is no session at initialization"""

        dev = QiskitDevice2(wires=2, backend=backend)
        assert dev._session == None

    def test_initializing_with_session(self):
        """Test that you can initialize a device with an existing Qiskit session"""

        session = MockSession(backend=backend, max_time="1m")
        dev = QiskitDevice2(wires=2, backend=backend, session=session)
        assert dev._session == session

    @patch("pennylane_qiskit.qiskit_device2.Session")
    @pytest.mark.parametrize("initial_session", [None, MockSession(backend)])
    def test_using_session_context(self, mock_session, initial_session):
        """Test that you can add a session within a context manager"""

        dev = QiskitDevice2(wires=2, backend=backend, session=initial_session)

        assert dev._session == initial_session

        with qiskit_session(dev) as session:
            assert dev._session == session
            assert dev._session != initial_session

        assert dev._session == initial_session

    @pytest.mark.parametrize("initial_session", [None, MockSession(backend)])
    def test_update_session(self, initial_session):
        """Test that you can update the session stored on the device"""

        dev = QiskitDevice2(wires=2, backend=backend, session=initial_session)
        assert dev._session == initial_session

        new_session = MockSession(backend=backend, max_time="1m")
        dev.update_session(new_session)

        assert dev._session != initial_session
        assert dev._session == new_session


class TestDevicePreprocessing:
    """Tests the device preprocessing functions"""

    @pytest.mark.parametrize(
        "measurements, expectation",
        [
            (
                [
                    qml.expval(qml.PauliZ(1)),
                    qml.counts(),
                    qml.var(qml.PauliY(0)),
                    qml.probs(wires=[2]),
                ],
                [
                    [qml.expval(qml.PauliZ(1)), qml.var(qml.PauliY(0))],
                    [qml.probs(wires=[2])],
                    [qml.counts()],
                ],
            ),
            (
                [
                    qml.expval(qml.PauliZ(1)),
                    qml.expval(qml.PauliX(2)),
                    qml.var(qml.PauliY(0)),
                    qml.probs(wires=[2]),
                ],
                [
                    [
                        qml.expval(qml.PauliZ(1)),
                        qml.expval(qml.PauliX(2)),
                        qml.var(qml.PauliY(0)),
                    ],
                    [qml.probs(wires=[2])],
                ],
            ),
            (
                [
                    qml.expval(qml.PauliZ(1)),
                    qml.counts(),
                    qml.var(qml.PauliY(0)),
                ],
                [
                    [qml.expval(qml.PauliZ(1)), qml.var(qml.PauliY(0))],
                    [qml.counts()],
                ],
            ),
            (
                [
                    qml.expval(qml.PauliZ(1)),
                    qml.var(qml.PauliY(0)),
                ],
                [
                    [qml.expval(qml.PauliZ(1)), qml.var(qml.PauliY(0))],
                ],
            ),
            (
                [qml.counts(), qml.sample(wires=[1, 0])],
                [[qml.counts(), qml.sample(wires=[1, 0])]],
            ),
            (
                [qml.probs(wires=[2])],
                [[qml.probs(wires=[2])]],
            ),
        ],
    )
    def test_split_measurement_types(self, measurements, expectation):
        """Test that the split_measurement_types transform splits measurements into Estimator-based
        (expval, var), Sampler-based (probs) and raw-sample based (everything else)"""

        operations = [qml.PauliX(0), qml.PauliY(1), qml.Hadamard(2), qml.CNOT([2, 1])]
        qs = QuantumScript(operations, measurements=measurements)
        tapes, reorder_fn = split_measurement_types(qs)

        # operations not modified
        assert np.all([tape.operations == operations for tape in tapes])

        # measurements split as expected
        [tape.measurements for tape in tapes] == expectation

        # reorder_fn puts them back
        assert reorder_fn([tape.measurements for tape in tapes]) == tuple(qs.measurements)

    @pytest.mark.parametrize(
        "op, expected",
        [
            (qml.PauliX(0), True),
            (qml.CRX(0.1, wires=[0, 1]), True),
            (qml.sum(qml.PauliY(1), qml.PauliZ(0)), False),
            (qml.pow(qml.RX(1.1, 0), 3), False),
            (qml.adjoint(qml.S(0)), True),
            (qml.adjoint(qml.RX(1.2, 0)), False),
        ],
    )
    def test_stopping_conditions(self, op, expected):
        """Test that stopping_condition works"""
        res = test_dev.stopping_condition(op)
        assert res == expected

    @pytest.mark.parametrize(
        "obs, expected",
        [
            (qml.PauliX(0), True),
            (qml.Hadamard(3), True),
            (qml.prod(qml.PauliY(1), qml.PauliZ(0)), False),
        ],
    )
    def test_observable_stopping_condition(self, obs, expected):
        """Test that observable_stopping_condition works"""
        res = test_dev.observable_stopping_condition(obs)
        assert res == expected

    @pytest.mark.parametrize(
        "measurements,num_types",
        [
            ([qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])], 2),
            ([qml.expval(qml.PauliZ(0)), qml.sample(wires=[0, 1])], 2),
            ([qml.counts(), qml.probs(wires=[0, 1]), qml.sample()], 2),
            ([qml.var(qml.PauliZ(0)), qml.expval(qml.PauliX(1))], 1),
            ([qml.probs(wires=[0]), qml.counts(), qml.var(qml.PauliY(2))], 3),
        ],
    )
    def test_preprocess_splits_incompatible_primitive_measurements(self, measurements, num_types):
        """Test that the default behaviour for preprocess it to split the tapes based
        on meausrement type. Expval and Variance are one type (Estimator), Probs another (Sampler),
        and everything else a third (raw sample-based measurements)."""

        dev = QiskitDevice2(wires=5, backend=backend, use_primitives=True)
        qs = QuantumScript([], measurements=measurements, shots=qml.measurements.Shots(1000))

        program, _ = dev.preprocess()
        tapes, _ = program([qs])

        # measurements that are incompatible are split when use_primtives=True
        assert len(tapes) == num_types

    @pytest.mark.parametrize(
        "measurements",
        [
            [qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])],
            [qml.expval(qml.PauliZ(0)), qml.sample(wires=[0, 1])],
            [qml.counts(), qml.probs(wires=[0, 1]), qml.sample()],
        ],
    )
    def test_preprocess_measurements_without_primitives(self, measurements):
        """Test if Primitives are not being used that the preprocess does not split
        the tapes based on measurement type"""

        qs = QuantumScript([], measurements=measurements, shots=qml.measurements.Shots(1000))

        dev = QiskitDevice2(wires=5, backend=backend, use_primitives=False)
        program, _ = dev.preprocess()

        tapes, _ = program([qs])

        # measurements that are incompatible on the primitive-based device
        # are not split when use_primtives=False
        assert len(tapes) == 1

    def test_preprocess_decomposes_unsupported_operator(self):
        """Test that the device preprocess decomposes operators that
        aren't on the list of Qiskit-supported operators"""
        qs = QuantumScript(
            [qml.CosineWindow(wires=range(2))], measurements=[qml.expval(qml.PauliZ(0))]
        )

        # tape contains unsupported operations
        assert not np.all([op in QISKIT_OPERATION_MAP for op in qs.operations])

        program, _ = test_dev.preprocess()
        tapes, _ = program([qs])

        # tape no longer contained unsupporrted operations
        assert np.all([op.name in QISKIT_OPERATION_MAP for op in tapes[0].operations])

    def test_intial_state_prep_also_decomposes(self):
        """Test that the device preprocess decomposes
        unsupported operator even if they are state prep operators"""

        qs = QuantumScript(
            [qml.AmplitudeEmbedding(features=[0.5, 0.5, 0.5, 0.5], wires=range(2))],
            measurements=[qml.expval(qml.PauliZ(0))],
        )

        program, _ = test_dev.preprocess()
        tapes, _ = program([qs])

        assert np.all([op.name in QISKIT_OPERATION_MAP for op in tapes[0].operations])


class TestOptionsHandling:
    def test_qiskit_options_to_flat_dict(self):
        """Test that a Qiskit Options object is converted to an un-nested python dictionary"""

        options = options_for_testing()

        options_dict = qiskit_options_to_flat_dict(options)

        assert isinstance(options_dict, dict)
        # the values in the dict are not themselves dictionaries or convertable to dictionaries
        for val in options_dict.values():
            assert not hasattr(val, "__dict__")
            assert not isinstance(val, dict)

    @pytest.mark.parametrize("options", [None, options_for_testing()])
    def test_shots_kwarg_updates_default_options(self, options):
        """Check that the shots passed to the device are set on the device
        as well as updated on the Options object"""

        dev = QiskitDevice2(wires=2, backend=backend, shots=23, options=options)

        assert dev.shots.total_shots == 23
        assert dev.options.execution.shots == 23

    def test_warning_if_shots(self):
        """Test that a warning is raised if the user attempt to specify shots on
        Options instead of as a kwarg, and sets shots to the shots passed (defaults
        to 1024)."""

        options = options_for_testing()
        options.execution.shots = 1000

        with pytest.warns(
            UserWarning,
            match="Setting shots via the Options is not supported on PennyLane devices",
        ):
            dev = QiskitDevice2(wires=2, backend=backend, options=options)

        assert dev.shots.total_shots == 1024
        assert dev.options.execution.shots == 1024

        with pytest.warns(
            UserWarning,
            match="Setting shots via the Options is not supported on PennyLane devices",
        ):
            dev = QiskitDevice2(wires=2, backend=backend, shots=200, options=options)

        assert dev.shots.total_shots == 200
        assert dev.options.execution.shots == 200

    def test_update_kwargs_no_overlapping_options_passed(self):
        """Test that if there is no overlap between options defined as device kwargs and on Options,
        _update_kwargs creates a combined dictionary"""

        dev = QiskitDevice2(wires=2, backend=backend, random_kwarg1=True, random_kwarg2="a")

        assert dev._init_kwargs == {"random_kwarg1": True, "random_kwarg2": "a"}
        assert dev._kwargs == {
            "random_kwarg1": True,
            "random_kwarg2": "a",
            "skip_transpilation": False,
            "init_qubits": True,
            "log_level": "WARNING",
            "job_tags": [],
        }

        dev.options.environment.job_tags = ["my_tag"]
        dev.options.max_execution_time = "1m"

        dev._update_kwargs()

        # _init_kwargs are unchanged, _kwargs are updated
        assert dev._init_kwargs == {"random_kwarg1": True, "random_kwarg2": "a"}
        assert dev._kwargs == {
            "random_kwarg1": True,
            "random_kwarg2": "a",
            "max_execution_time": "1m",
            "skip_transpilation": False,
            "init_qubits": True,
            "log_level": "WARNING",
            "job_tags": ["my_tag"],
        }

    def test_update_kwargs_with_overlapping_options(self):
        """Test that if there is overlap between options defined as device kwargs and on Options,
        _update_kwargs creates a combined dictionary with Options taking precedence, and raises a
        warning"""

        dev = QiskitDevice2(wires=2, backend=backend, random_kwarg1=True, max_execution_time="1m")

        assert dev._init_kwargs == {"random_kwarg1": True, "max_execution_time": "1m"}
        assert dev._kwargs == {
            "random_kwarg1": True,
            "max_execution_time": "1m",
            "skip_transpilation": False,
            "init_qubits": True,
            "log_level": "WARNING",
            "job_tags": [],
        }

        dev.options.environment.job_tags = ["my_tag"]
        dev.options.max_execution_time = "30m"

        with pytest.warns(
            UserWarning,
            match="also defined in the device Options. The definition in Options will be used.",
        ):
            dev._update_kwargs()

        # _init_kwargs are unchanged, _kwargs are updated
        assert dev._init_kwargs == {"random_kwarg1": True, "max_execution_time": "1m"}
        assert dev._kwargs == {
            "random_kwarg1": True,
            "max_execution_time": "30m",  # definition from Options is used
            "skip_transpilation": False,
            "init_qubits": True,
            "log_level": "WARNING",
            "job_tags": ["my_tag"],
        }

    def test_update_kwargs_with_shots_set_on_options(self):
        """Test that if shots have been defined on Options, _update_kwargs raises a warning
        and ignores the shots as defined on Options"""

        dev = QiskitDevice2(wires=2, backend=backend, random_kwarg1=True)

        start_init_kwargs = dev._init_kwargs
        start_kwargs = dev._kwargs

        dev.options.execution.shots = 500

        with pytest.warns(
            UserWarning,
            match="Setting shots via the Options is not supported on PennyLane devices",
        ):
            assert dev.options.execution.shots == 500
            dev._update_kwargs()

        # _init_kwargs and _kwargs are unchanged, shots was ignored
        assert dev._init_kwargs == start_init_kwargs
        assert dev._kwargs == start_kwargs

        # the shots on the Options have been reset to the device shots
        assert dev.options.execution.shots == dev.shots.total_shots


class TestDeviceProperties:
    def test_name_property(self):
        """Test the backend property"""
        assert test_dev.name == "qiskit.remote2"

    def test_backend_property(self):
        """Test the backend property"""
        assert test_dev.backend == test_dev._backend
        assert test_dev.backend == backend

    def test_service_property(self):
        """Test the service property"""
        assert test_dev.service == test_dev._service

    def test_session_property(self):
        """Test the session property"""

        session = MockSession(backend=backend)
        dev = QiskitDevice2(wires=2, backend=backend, session=session)
        assert dev.session == dev._session
        assert dev.session == session

    def test_num_wires_property(self):
        """Test the num_wires property"""

        wires = [1, 2, 3]
        dev = QiskitDevice2(wires=wires, backend=backend)
        assert dev.num_wires == len(wires)


class TestMockedExecution:

    def test_get_transpile_args(self):
        """Test that get_transpile_args works as expected by filtering out
        kwargs that don't match the Qiskit transpile signature"""
        kwargs = {"random_kwarg": 3, "optimization_level": 3, "circuits": []}
        assert QiskitDevice2.get_transpile_args(kwargs) == {"optimization_level": 3}

    @patch("pennylane_qiskit.qiskit_device2.transpile")
    def test_compile_circuits(self, transpile_mock):
        """Tests compile_circuits with a mocked transpile function to avoid calling
        a remote backend. This renders it fairly useless."""

        transpile_mock.return_value = QuantumCircuit(2)

        # technically this doesn't matter due to the mock, but this is the correct input format for the function
        circuits = [QuantumScript([qml.PauliX(0)], measurements=[qml.expval(qml.PauliZ(0))]),
                    QuantumScript([qml.PauliX(0)], measurements=[qml.probs(wires=[0])]),
                    QuantumScript([qml.PauliX(0), qml.PauliZ(1)], measurements=[qml.counts()])]
        input_circuits = [circuit_to_qiskit(c, register_size=2) for c in circuits]

        compiled_circuits = test_dev.compile_circuits(input_circuits)

        assert len(compiled_circuits) == len(input_circuits)
        for i, circuit in enumerate(compiled_circuits):
            assert isinstance(circuit, QuantumCircuit)

    @pytest.mark.parametrize(
        "measurements, expectation",
        [
            ([qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))], (1, 0)),
            ([qml.var(qml.PauliX(0))], (1)),
            (
                [
                    qml.expval(qml.PauliX(0)),
                    qml.expval(qml.PauliZ(0)),
                    qml.var(qml.PauliX(0)),
                ],
                (0, 1, 1),
            ),
        ],
    )
    def test_process_estimator_job_mocked(self, measurements, expectation):
        """Test the process_estimator_job function with constructed return for
        Estimator (integration test that runs with a Token is below)"""

        values = np.array([np.random.ranf() for i in range(len(measurements))])
        metadata = [{'variance': np.random.ranf(), 'shots': 4000} for i in range(len(measurements))]

        result = EstimatorResult(values, metadata)
        processed_result = QiskitDevice2._process_estimator_job(measurements, result)

        assert isinstance(processed_result, tuple)
        assert len(processed_result) == len(measurements)

    @pytest.mark.parametrize("results, index", [({'00': 125, '10': 500, '01': 250, '11' : 125}, None),
                                                ([{}, {'00': 125, '10': 500, '01': 250, '11' : 125}], 1),
                                                ([{}, {}, {'00': 125, '10': 500, '01': 250, '11' : 125}], 2)])
    def test_generate_samples_mocked_single_result(self, results, index):
        """Test generate_samples with a Mocked return for the job result
        (integration test that runs with a Token is below)"""

        # create mocked Job with results dict
        def get_counts():
            return results

        mock_job = Mock()
        mock_job.configure_mock(get_counts=get_counts)
        test_dev._current_job = mock_job

        samples = test_dev.generate_samples(circuit=index)
        results_dict = results if index is None else results[index]

        assert len(samples) == sum(results_dict.values())
        assert len(samples[0]) == 2

        assert len(np.argwhere([np.allclose(s, [0, 0]) for s in samples])) == results_dict["00"]
        assert len(np.argwhere([np.allclose(s, [1, 1]) for s in samples])) == results_dict["11"]

        # order of samples is swapped compared to keys (Qiskit wire order convention is reverse of PennyLane)
        assert len(np.argwhere([np.allclose(s, [0, 1]) for s in samples])) == results_dict["10"]
        assert len(np.argwhere([np.allclose(s, [1, 0]) for s in samples])) == results_dict["01"]







    def test_execute_pipeline_no_primitives_mocked(self, mocker):
        """Test that a device **not** using Primitives only calls the _execute_runtime_service
        to execute, regardless of measurement type"""

        dev = QiskitDevice2(wires=5, backend=backend, use_primitives=False, session=MockSession())

        sampler_execute = mocker.spy(dev, "_execute_sampler")
        estimator_execute = mocker.spy(dev, "_execute_estimator")

        qs = QuantumScript(
            [qml.PauliX(0), qml.PauliY(1)],
            measurements=[
                qml.expval(qml.PauliZ(0)),
                qml.probs(wires=[0, 1]),
                qml.counts(),
                qml.sample(),
            ],
        )

        with patch.object(dev, "_execute_runtime_service", return_value="runtime_execute_res"):
            runtime_service_execute = mocker.spy(dev, "_execute_runtime_service")
            res = dev.execute(qs)

        runtime_service_execute.assert_called_once()
        sampler_execute.assert_not_called()
        estimator_execute.assert_not_called()

        assert res == "runtime_execute_res"

    def test_execute_pipeline_with_primitives_mocked(self, mocker):
        """Test that a device that **is** using Primitives calls the _execute_runtime_service
        to execute measurements that require raw samples, and the relevant primitive measurements
        on the other measurements"""

        dev = QiskitDevice2(wires=5, backend=backend, use_primitives=True, session=MockSession())

        qs = QuantumScript(
            [qml.PauliX(0), qml.PauliY(1)],
            measurements=[
                qml.expval(qml.PauliZ(0)),
                qml.probs(wires=[0, 1]),
                qml.counts(),
                qml.sample(),
            ],
        )
        tapes, reorder_fn = split_measurement_types(qs)

        with patch.object(dev, "_execute_runtime_service", return_value="runtime_execute_res"):
            with patch.object(dev, "_execute_sampler", return_value="sampler_execute_res"):
                with patch.object(dev, "_execute_estimator", return_value="estimator_execute_res"):
                    runtime_service_execute = mocker.spy(dev, "_execute_runtime_service")
                    sampler_execute = mocker.spy(dev, "_execute_sampler")
                    estimator_execute = mocker.spy(dev, "_execute_estimator")

                    res = dev.execute(tapes)

        runtime_service_execute.assert_called_once()
        sampler_execute.assert_called_once()
        estimator_execute.assert_called_once()

        assert res == ["estimator_execute_res", "sampler_execute_res", "runtime_execute_res"]


@pytest.mark.usefixtures("skip_if_no_account")
class TestExecution:

    @pytest.mark.parametrize("wire", [0, 1])
    @pytest.mark.parametrize(
        "angle,op,expectation",
        [
            (np.pi / 2, qml.RX, [0, -1, 0, 1, 0, 1]),
            (np.pi, qml.RX, [0, 0, -1, 1, 1, 0]),
            (np.pi / 2, qml.RY, [1, 0, 0, 0, 1, 1]),
            (np.pi, qml.RY, [0, 0, -1, 1, 1, 0]),
            (np.pi / 2, qml.RZ, [0, 0, 1, 1, 1, 0]),
        ],
    )
    def test_estimator_with_different_pauli_obs(self, mocker, wire, angle, op, expectation):
        """Test that the Estimator with various observables returns expected results.
        Essentially testing that the conversion to PauliOps in _execute_estimator behaves as
        expected. Iterating over wires ensures that the wire operated on and the wire measured
        correspond correctly (wire ordering convention in Qiskit and PennyLane don't match.)
        """

        dev = QiskitDevice2(wires=5, backend=backend, use_primitives=True)

        runtime_service_execute = mocker.spy(dev, "_execute_runtime_service")
        sampler_execute = mocker.spy(dev, "_execute_sampler")
        estimator_execute = mocker.spy(dev, "_execute_sampler")

        qs = QuantumScript(
            [op(angle, wire)],
            measurements=[
                qml.expval(qml.PauliX(wire)),
                qml.expval(qml.PauliY(wire)),
                qml.expval(qml.PauliZ(wire)),
                qml.var(qml.PauliX(wire)),
                qml.var(qml.PauliY(wire)),
                qml.var(qml.PauliZ(wire)),
            ],
        )

        res = dev.execute(qs)

        runtime_service_execute.assert_not_called()
        sampler_execute.assert_not_called()
        estimator_execute.assert_called_once()

        assert np.allclose(res, expectation, atol=0.1)

    @pytest.mark.parametrize(
        "measurements, expectation",
        [
            ([qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))], (1, 0)),
            ([qml.var(qml.PauliX(0))], (1)),
            (
                [
                    qml.expval(qml.PauliX(0)),
                    qml.expval(qml.PauliZ(0)),
                    qml.var(qml.PauliX(0)),
                ],
                (0, 1, 1),
            ),
        ],
    )
    def test_process_estimator_job(self, measurements, expectation):
        """for variance and for expval and for a combination"""

        # make PennyLane circuit
        qs = QuantumScript([], measurements=measurements)

        # convert to Qiskit circuit information
        qcirc = circuit_to_qiskit(qs, register_size=qs.num_wires, diagonalize=False, measure=False)
        pauli_observables = [mp_to_pauli(mp, qs.num_wires) for mp in qs.measurements]

        # run on simulator via Estimator
        estimator = Estimator(backend=backend)
        result = estimator.run([qcirc] * len(pauli_observables), pauli_observables).result()

        # confirm that the result is as expected - if the test fails at this point, its because the
        # Qiskit result format has changed
        assert isinstance(result, EstimatorResult)

        assert isinstance(result.values, np.ndarray)
        assert result.values.size == len(qs.measurements)

        assert isinstance(result.metadata, list)
        assert len(result.metadata) == len(qs.measurements)

        for data in result.metadata:
            assert isinstance(data, dict)
            assert list(data.keys()) == ["variance", "shots"]

        processed_result = QiskitDevice2._process_estimator_job(qs.measurements, result)
        assert isinstance(processed_result, tuple)
        assert np.allclose(processed_result, expectation, atol=0.05)

    @pytest.mark.parametrize("num_wires", [1, 3, 5])
    @pytest.mark.parametrize("num_shots", [50, 100])
    def test_generate_samples(self, num_wires, num_shots):
        qs = QuantumScript([], measurements=[qml.expval(qml.PauliX(0))])

        qcirc = circuit_to_qiskit(qs, register_size=num_wires, diagonalize=True, measure=True)
        compiled_circuits = test_dev.compile_circuits([qcirc])

        # Send circuits to the cloud for execution by the circuit-runner program
        job = test_dev.service.run(
            program_id="circuit-runner",
            options={"backend": backend.name},
            inputs={"circuits": compiled_circuits, "shots": num_shots},
        )

        test_dev._current_job = job.result(decoder=RunnerResult)

        samples = test_dev.generate_samples()

        assert len(samples) == num_shots
        assert len(samples[0]) == num_wires

        # we expect the samples to be orderd such that q0 has a 50% chance
        # of being excited, and everything else is in the ground state
        exp_res0 = np.zeros(num_wires)
        exp_res1 = np.zeros(num_wires)
        exp_res1[0] = 1

        # the two expected results are in samples
        assert exp_res1 in samples
        assert exp_res0 in samples

        # nothing else is in samples
        assert [s for s in samples if not s in np.array([exp_res0, exp_res1])] == []
