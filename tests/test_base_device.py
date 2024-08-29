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

from unittest.mock import patch, Mock
from flaky import flaky
import numpy as np
from pennylane import numpy as pnp
from pydantic_core import ValidationError
import pytest

import pennylane as qml
from pennylane.tape.qscript import QuantumScript
from qiskit_ibm_runtime import EstimatorV2 as Estimator, Session
from qiskit_ibm_runtime.fake_provider import FakeManila, FakeManilaV2
from qiskit_aer import AerSimulator

# do not import Estimator (imported above) from qiskit.primitives - the identically
# named Estimator object has a different call signature than the remote device Estimator,
# and only runs local simulations. We need the Estimator from qiskit_ibm_runtime. They
# both use this EstimatorResults, however:
from qiskit.providers import BackendV1, BackendV2

from qiskit import QuantumCircuit, transpile
from pennylane_qiskit.qiskit_device import (
    QiskitDevice,
    qiskit_session,
    split_execution_types,
)
from pennylane_qiskit.converter import (
    circuit_to_qiskit,
    QISKIT_OPERATION_MAP,
    mp_to_pauli,
)

# pylint: disable=protected-access, unused-argument, too-many-arguments, redefined-outer-name


# pylint: disable=too-few-public-methods
class Configuration:
    def __init__(self, n_qubits, backend_name):
        self.n_qubits = n_qubits
        self.backend_name = backend_name
        self.noise_model = None


class MockedBackend(BackendV2):
    def __init__(self, num_qubits=10, name="mocked_backend"):
        self._options = Configuration(num_qubits, name)
        self._service = "SomeServiceProvider"
        self.name = name
        self._target = Mock()
        self._target.num_qubits = num_qubits

    def set_options(self, noise_model):
        self.options.noise_model = noise_model

    def _default_options(self):
        return {}

    def max_circuits(self):
        return 10

    def run(self, *args, **kwargs):
        return None

    @property
    def target(self):
        return self._target


class MockedBackendLegacy(BackendV1):
    def __init__(self, num_qubits=10, name="mocked_backend_legacy"):
        self._configuration = Configuration(num_qubits, backend_name=name)
        self._service = "SomeServiceProvider"
        self._options = self._default_options()

    def configuration(self):
        return self._configuration

    def _default_options(self):
        return {}

    def run(self, *args, **kwargs):
        return None

    @property
    def options(self):
        return self._options


# pylint: disable=too-few-public-methods
class MockSession:
    def __init__(self, backend, max_time=None):
        self._backend = backend
        self._max_time = max_time
        self._args = "random"  # this is to satisfy a mock
        self._kwargs = "random"  # this is to satisfy a mock
        self.session_id = "123"

    def close(self):  # This is just to appease a test
        pass


mocked_backend = MockedBackend()
legacy_backend = MockedBackendLegacy()
aer_backend = AerSimulator()
test_dev = QiskitDevice(wires=5, backend=aer_backend)


class TestSupportForV1andV2:
    """Tests compatibility with BackendV1 and BackendV2"""

    @pytest.mark.parametrize(
        "backend",
        [legacy_backend, aer_backend, mocked_backend],
    )
    def test_v1_and_v2_mocked(self, backend):
        """Test that device initializes with no error mocked"""
        dev = QiskitDevice(wires=10, backend=backend)
        assert dev._backend == backend

    @pytest.mark.parametrize(
        "backend, shape",
        [
            (FakeManila(), (1024,)),
            (FakeManilaV2(), (1024,)),
        ],
    )
    def test_v1_and_v2_manila(self, backend, shape):
        """Test that device initializes and runs without error with V1 and V2 backends by Qiskit"""
        dev = QiskitDevice(wires=5, backend=backend)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0))

        res = circuit(np.pi / 2)

        assert np.shape(res) == shape
        assert dev._backend == backend


class TestDeviceInitialization:
    def test_compile_backend_kwarg(self):
        """Test that the compile_backend is set correctly if passed, and the main
        backend is used otherwise"""

        compile_backend = MockedBackend(name="compile_backend")
        main_backend = MockedBackend(name="main_backend")

        dev1 = QiskitDevice(wires=5, backend=main_backend)
        dev2 = QiskitDevice(wires=5, backend=main_backend, compile_backend=compile_backend)

        assert dev1._compile_backend == dev1._backend == main_backend

        assert dev2._compile_backend != dev2._backend
        assert dev2._compile_backend == compile_backend

    def test_no_shots_warns_and_defaults(self):
        """Test that initializing with shots=None raises a warning indicating that
        the device is sample based and will default to 1024 shots"""

        with pytest.warns(
            UserWarning,
            match="Expected an integer number of shots, but received shots=None",
        ):
            dev = QiskitDevice(wires=2, backend=aer_backend, shots=None)

        assert dev.shots.total_shots == 1024

    @pytest.mark.parametrize("backend", [aer_backend, legacy_backend])
    def test_backend_wire_validation(self, backend):
        """Test that an error is raised if the number of device wires exceeds
        the number of wires available on the backend, for both backend versions"""

        with pytest.raises(ValueError, match="supports maximum"):
            QiskitDevice(wires=500, backend=backend)

    def test_setting_simulator_noise_model(self):
        """Test that the simulator noise model saved on a passed Options
        object is used to set the backend noise model"""

        new_backend = MockedBackend()
        dev1 = QiskitDevice(wires=3, backend=aer_backend)
        dev2 = QiskitDevice(wires=3, backend=new_backend, noise_model={"placeholder": 1})

        assert dev1.backend.options.noise_model is None
        assert dev2.backend.options.noise_model == {"placeholder": 1}


class TestQiskitSessionManagement:
    """Test using Qiskit sessions with the device"""

    @pytest.mark.parametrize("backend", [aer_backend, FakeManila(), FakeManilaV2()])
    def test_default_no_session_on_initialization(self, backend):
        """Test that the default behaviour is no session at initialization"""

        dev = QiskitDevice(wires=2, backend=backend)
        assert dev._session is None

    @pytest.mark.parametrize("backend", [aer_backend, FakeManila(), FakeManilaV2()])
    def test_initializing_with_session(self, backend):
        """Test that you can initialize a device with an existing Qiskit session"""

        session = MockSession(backend=backend, max_time="1m")
        dev = QiskitDevice(wires=2, backend=backend, session=session)
        assert dev._session == session

    @patch("pennylane_qiskit.qiskit_device.Session")
    @pytest.mark.parametrize("initial_session", [None, MockSession(aer_backend)])
    def test_using_session_context(self, mock_session, initial_session):
        """Test that you can add a session within a context manager"""

        dev = QiskitDevice(wires=2, backend=aer_backend, session=initial_session)

        assert dev._session == initial_session

        with qiskit_session(dev) as session:
            assert dev._session == session
            assert dev._session != initial_session

        assert dev._session == initial_session

    def test_using_session_context_options(self):
        """Test that you can set session options using qiskit_session"""
        dev = QiskitDevice(wires=2, backend=aer_backend)

        assert dev._session is None

        with qiskit_session(dev, max_time=30) as session:
            assert dev._session == session
            assert dev._session is not None
            assert dev._session._max_time == 30

        assert dev._session is None

    def test_error_when_passing_unexpected_kwarg(self):
        """Test that we accept any keyword argument that the user wants to supply so that if
        Qiskit allows for more customization we can automatically accomodate those needs. Right
        now there are no such keyword arguments, so an error on Qiskit's side is raised."""

        dev = QiskitDevice(wires=2, backend=aer_backend)

        assert dev._session is None

        with pytest.raises(
            TypeError,  # Type error for wrong keyword argument differs across python versions
        ):
            with qiskit_session(dev, any_kwarg=30) as session:
                assert dev._session == session
                assert dev._session is not None

        assert dev._session is None

    def test_no_warning_when_using_initial_session_options(self):
        initial_session = Session(backend=aer_backend, max_time=30)
        dev = QiskitDevice(wires=2, backend=aer_backend, session=initial_session)

        assert dev._session == initial_session

        with qiskit_session(dev) as session:
            assert dev._session == session
            assert dev._session != initial_session
            assert dev._session._max_time == session._max_time
            assert dev._session._max_time != initial_session._max_time

        assert dev._session == initial_session
        assert dev._session._max_time == initial_session._max_time

    def test_warnings_when_overriding_session_context_options(self, recorder):
        """Test that warnings are raised when the session options try to override either the
        device's `backend` or `service`. Also ensures that the session options, even the
        default options, passed in from the `qiskit_session` take precedence, barring
        `backend` or `service`"""
        initial_session = Session(backend=aer_backend)
        dev = QiskitDevice(wires=2, backend=aer_backend, session=initial_session)

        assert dev._session == initial_session

        with pytest.warns(
            UserWarning,
            match="Using 'backend' set in device",
        ):
            with qiskit_session(dev, max_time=30, backend=FakeManilaV2()) as session:
                assert dev._session == session
                assert dev._session != initial_session
                assert dev._session._backend.name == "aer_simulator"

        with pytest.warns(
            UserWarning,
            match="Using 'service' set in device",
        ):
            with qiskit_session(dev, max_time=30, service="placeholder") as session:
                assert dev._session == session
                assert dev._session != initial_session
                assert dev._session._service != "placeholder"

        # device session should be unchanged by qiskit_session
        assert dev._session == initial_session

        max_time_session = Session(backend=aer_backend, max_time=60)
        dev = QiskitDevice(wires=2, backend=aer_backend, session=max_time_session)
        with qiskit_session(dev, max_time=30) as session:
            assert dev._session == session
            assert dev._session != initial_session
            assert dev._session._max_time == 30
            assert dev._session._max_time != 60

        assert dev._session == max_time_session
        assert dev._session._max_time == 60

    @pytest.mark.parametrize("initial_session", [None, MockSession(aer_backend)])
    def test_update_session(self, initial_session):
        """Test that you can update the session stored on the device"""

        dev = QiskitDevice(wires=2, backend=aer_backend, session=initial_session)
        assert dev._session == initial_session

        new_session = MockSession(backend=aer_backend, max_time="1m")
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
                    [qml.counts(), qml.probs(wires=[2])],
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
            (
                [
                    qml.expval(qml.Hadamard(0)),
                    qml.expval(qml.PauliX(0)),
                    qml.var(qml.PauliZ(0)),
                    qml.counts(),
                ],
                [
                    [qml.expval(qml.PauliX(0)), qml.var(qml.PauliZ(0))],
                    [qml.expval(qml.Hadamard(0)), qml.counts()],
                ],
            ),
        ],
    )
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_split_execution_types(self, measurements, expectation):
        """Test that the split_execution_types transform splits measurements into Estimator-based
        (expval, var) and Sampler-based (everything else)"""

        operations = [qml.PauliX(0), qml.PauliY(1), qml.Hadamard(2), qml.CNOT([2, 1])]
        qs = QuantumScript(operations, measurements=measurements)
        tapes, reorder_fn = split_execution_types(qs)

        # operations not modified
        assert np.all([tape.operations == operations for tape in tapes])

        # measurements split as expected
        assert [tape.measurements for tape in tapes] == expectation

        # reorder_fn puts them back
        assert (
            reorder_fn([tape.measurements for tape in tapes]) == qs.measurements[0]
            if len(qs.measurements) == 1
            else reorder_fn([tape.measurements for tape in tapes]) == tuple(qs.measurements)
        )

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
            (qml.prod(qml.PauliY(1), qml.PauliZ(0)), True),
            (qml.prod(qml.PauliY(1), qml.PauliZ(0)), True),
        ],
    )
    def test_observable_stopping_condition(self, obs, expected):
        """Test that observable_stopping_condition works"""
        res = test_dev.observable_stopping_condition(obs)
        assert res == expected

    @pytest.mark.parametrize(
        "measurements, num_tapes",
        [
            (
                [
                    qml.expval(qml.X(0) + qml.Y(0) + qml.Z(0)),
                ],
                3,
            ),
            (
                pytest.param(
                    [qml.var(qml.X(0) + qml.Y(0) + qml.Z(0))],
                    1,
                    marks=pytest.mark.xfail(reason="Split non commuting discussion pending"),
                )
            ),
            (
                [
                    qml.expval(qml.X(0)),
                    qml.expval(qml.Y(1)),
                    qml.expval(qml.Z(0) @ qml.Z(1)),
                    qml.expval(qml.X(0) @ qml.Z(1) + 0.5 * qml.Y(1) + qml.Z(0)),
                ],
                3,
            ),
            (
                [
                    qml.expval(
                        qml.prod(qml.X(0), qml.Z(0), qml.Z(0)) + 0.35 * qml.X(0) - 0.21 * qml.Z(0)
                    )
                ],
                2,
            ),
            (
                pytest.param(
                    [
                        qml.counts(qml.X(0)),
                        qml.counts(qml.Y(1)),
                        qml.counts(qml.Z(0) @ qml.Z(1)),
                        qml.counts(qml.X(0) @ qml.Z(1) + 0.5 * qml.Y(1) + qml.Z(0)),
                    ],
                    3,
                    marks=pytest.mark.xfail(reason="Split non commuting discussion pending"),
                )
            ),
            (
                pytest.param(
                    [
                        qml.sample(qml.X(0)),
                        qml.sample(qml.Y(1)),
                        qml.sample(qml.Z(0) @ qml.Z(1)),
                        qml.sample(qml.X(0) @ qml.Z(1) + 0.5 * qml.Y(1) + qml.Z(0)),
                    ],
                    3,
                    marks=pytest.mark.xfail(reason="Split non commuting discussion pending"),
                )
            ),
        ],
    )
    def test_preprocess_split_non_commuting(self, measurements, num_tapes):
        """Test that `split_non_commuting` works as expected in the preprocess function."""

        dev = QiskitDevice(wires=5, backend=aer_backend)
        qs = QuantumScript([], measurements=measurements, shots=qml.measurements.Shots(1000))

        program, _ = dev.preprocess()
        tapes, _ = program([qs])

        assert len(tapes) == num_tapes

    @pytest.mark.parametrize(
        "measurements,num_types",
        [
            ([qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])], 2),
            ([qml.expval(qml.PauliZ(0)), qml.sample(wires=[0, 1])], 2),
            ([qml.counts(), qml.probs(wires=[0, 1]), qml.sample()], 1),
            ([qml.var(qml.PauliZ(0)), qml.expval(qml.PauliX(1))], 1),
            ([qml.probs(wires=[0]), qml.counts(), qml.var(qml.PauliY(2))], 2),
        ],
    )
    def test_preprocess_splits_incompatible_primitive_measurements(self, measurements, num_types):
        """Test that the default behaviour for preprocess it to split the tapes based
        on measurement type. Expval and Variance are one type (Estimator), Probs and raw-sample based measurements
        are another type (Sampler)."""

        dev = QiskitDevice(wires=5, backend=aer_backend)
        qs = QuantumScript([], measurements=measurements, shots=qml.measurements.Shots(1000))

        program, _ = dev.preprocess()
        tapes, _ = program([qs])

        # measurements that are incompatible are split when use_primtives=True
        assert len(tapes) == num_types

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


class TestKwargsHandling:
    def test_warning_if_shots(self):
        """Test that a warning is raised if the user attempts to specify shots by using
        `default_shots`, and instead sets shots to the default amount of 1024."""

        with pytest.warns(
            UserWarning,
            match="default_shots was found in the keyword arguments",
        ):
            dev = QiskitDevice(wires=2, backend=aer_backend, default_shots=333)

        # Qiskit takes in `default_shots` to define the # of shots, therefore we use
        # the kwarg "default_shots" rather than shots to pass it to Qiskit.
        assert dev._kwargs["default_shots"] == 1024

        dev = QiskitDevice(wires=2, backend=aer_backend, shots=200)
        assert dev._kwargs["default_shots"] == 200

        with pytest.warns(
            UserWarning,
            match="default_shots was found in the keyword arguments",
        ):
            dev = QiskitDevice(wires=2, backend=aer_backend, options={"default_shots": 30})
        # resets to default since we reinitialize the device
        assert dev._kwargs["default_shots"] == 1024

    def test_warning_if_options_and_kwargs_overlap(self):
        """Test that a warning is raised if the user has options that overlap with the kwargs"""

        with pytest.warns(
            UserWarning,
            match="An overlap between",
        ):
            dev = QiskitDevice(
                wires=2,
                backend=aer_backend,
                options={"resilience_level": 1, "optimization_level": 1},
                resilience_level=2,
                random_sauce="spaghetti",
            )

        assert dev._kwargs["resilience_level"] == 1
        assert dev._transpile_args["optimization_level"] == 1

        # You can initialize the device with any kwarg, but you'll get a ValidationError
        # when you run the circuit
        assert dev._kwargs["random_sauce"] == "spaghetti"

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliX(0))

        with pytest.raises(ValidationError, match="Object has no attribute"):
            circuit()

    @pytest.mark.parametrize("backend", [aer_backend, FakeManila(), FakeManilaV2()])
    def test_options_and_kwargs_combine_into_unified_kwargs(self, backend):
        """Test that options set via the keyword argument options and options set via kwargs
        will combine into a single unified kwargs that is passed to the device"""

        dev = QiskitDevice(
            wires=5,
            backend=backend,
            options={"resilience_level": 1},
            execution={"init_qubits": False},
        )

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliX(0))

        circuit()
        assert dev._kwargs["resilience_level"] == 1
        assert dev._kwargs["execution"]["init_qubits"] is False

        circuit(shots=123)
        assert dev._kwargs["resilience_level"] == 1
        assert dev._kwargs["execution"]["init_qubits"] is False

    @pytest.mark.parametrize("backend", [aer_backend, FakeManila(), FakeManilaV2()])
    def test_no_error_is_raised_if_transpilation_options_are_passed(self, backend):
        """Tests that when transpilation options are passed in, they are properly
        handled without error"""

        dev = QiskitDevice(
            wires=5,
            backend=backend,
            options={"resilience_level": 1, "optimization_level": 1},
            seed_transpiler=42,
        )

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliX(0))

        circuit()
        assert dev._kwargs["resilience_level"] == 1
        assert not hasattr(dev._kwargs, "seed_transpiler")
        assert dev._transpile_args["seed_transpiler"] == 42

        # Make sure that running the circuit again doesn't change the optios
        circuit(shots=5)
        assert dev._kwargs["resilience_level"] == 1
        assert not hasattr(dev._kwargs, "seed_transpiler")
        assert dev._transpile_args["seed_transpiler"] == 42


class TestDeviceProperties:
    def test_name_property(self):
        """Test the backend property"""
        assert test_dev.name == "QiskitDevice"

    def test_backend_property(self):
        """Test the backend property"""
        assert test_dev.backend == test_dev._backend
        assert test_dev.backend == aer_backend

    @pytest.mark.parametrize("backend", [aer_backend, FakeManila(), FakeManilaV2()])
    def test_compile_backend_property(self, backend):
        """Test the compile_backend property"""

        compile_backend = MockedBackend(name="compile_backend")
        dev = QiskitDevice(wires=5, backend=backend, compile_backend=compile_backend)

        assert dev.compile_backend == dev._compile_backend
        assert dev.compile_backend == compile_backend

    def test_service_property(self):
        """Test the service property"""
        assert test_dev.service == test_dev._service

    def test_session_property(self):
        """Test the session property"""

        session = MockSession(backend=aer_backend)
        dev = QiskitDevice(wires=2, backend=aer_backend, session=session)
        assert dev.session == dev._session
        assert dev.session == session

    def test_num_wires_property(self):
        """Test the num_wires property"""

        wires = [1, 2, 3]
        dev = QiskitDevice(wires=wires, backend=aer_backend)
        assert dev.num_wires == len(wires)


class TestTrackerFunctionality:
    def test_tracker_batched(self):
        """Test that the tracker works for batched circuits"""
        dev = qml.device("default.qubit", wires=1, shots=10000)
        qiskit_dev = QiskitDevice(wires=1, backend=AerSimulator(), shots=10000)

        x = pnp.array(0.1, requires_grad=True)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.Z(0))

        @qml.qnode(qiskit_dev, diff_method="parameter-shift")
        def qiskit_circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.Z(0))

        with qml.Tracker(dev) as tracker:
            qml.grad(circuit)(x)

        with qml.Tracker(qiskit_dev) as qiskit_tracker:
            qml.grad(qiskit_circuit)(x)

        assert qiskit_tracker.history["batches"] == tracker.history["batches"]
        assert tracker.history["shots"] == qiskit_tracker.history["shots"]
        assert np.allclose(qiskit_tracker.history["results"], tracker.history["results"], atol=0.1)
        assert np.shape(qiskit_tracker.history["results"]) == np.shape(tracker.history["results"])
        assert qiskit_tracker.history["resources"][0] == tracker.history["resources"][0]
        assert "simulations" not in qiskit_dev.tracker.history
        assert "simulations" not in qiskit_dev.tracker.latest
        assert "simulations" not in qiskit_dev.tracker.totals

    def test_tracker_single_tape(self):
        """Test that the tracker works for a single tape"""
        dev = qml.device("default.qubit", wires=1, shots=10000)
        qiskit_dev = QiskitDevice(wires=1, backend=AerSimulator(), shots=10000)

        tape = qml.tape.QuantumTape([qml.S(0)], [qml.expval(qml.X(0))])
        with qiskit_dev.tracker:
            qiskit_out = qiskit_dev.execute(tape)

        with dev.tracker:
            pl_out = dev.execute(tape)

        assert (
            qiskit_dev.tracker.history["resources"][0].shots
            == dev.tracker.history["resources"][0].shots
        )
        assert np.allclose(pl_out, qiskit_out, atol=0.1)
        assert np.allclose(
            qiskit_dev.tracker.history["results"], dev.tracker.history["results"], atol=0.1
        )

        assert np.shape(qiskit_dev.tracker.history["results"]) == np.shape(
            dev.tracker.history["results"]
        )

        assert "simulations" not in qiskit_dev.tracker.history
        assert "simulations" not in qiskit_dev.tracker.latest
        assert "simulations" not in qiskit_dev.tracker.totals

    def test_tracker_split_by_measurement_type(self):
        """Test that the tracker works for as intended for circuits split by measurement type"""
        qiskit_dev = QiskitDevice(wires=5, backend=AerSimulator(), shots=10000)

        x = 0.1

        @qml.qnode(qiskit_dev)
        def qiskit_circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.Z(0)), qml.counts(qml.X(1))

        with qml.Tracker(qiskit_dev) as qiskit_tracker:
            qiskit_circuit(x)

        assert qiskit_tracker.totals["executions"] == 2
        assert qiskit_tracker.totals["shots"] == 20000
        assert "simulations" not in qiskit_dev.tracker.history
        assert "simulations" not in qiskit_dev.tracker.latest
        assert "simulations" not in qiskit_dev.tracker.totals

    def test_tracker_split_by_non_commute(self):
        """Test that the tracker works for as intended for circuits split by non commute"""
        qiskit_dev = QiskitDevice(wires=5, backend=AerSimulator(), shots=10000)

        x = 0.1

        @qml.qnode(qiskit_dev)
        def qiskit_circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.Z(0)), qml.expval(qml.X(0))

        with qml.Tracker(qiskit_dev) as qiskit_tracker:
            qiskit_circuit(x)

        assert qiskit_tracker.totals["executions"] == 2
        assert qiskit_tracker.totals["shots"] == 20000
        assert "simulations" not in qiskit_dev.tracker.history
        assert "simulations" not in qiskit_dev.tracker.latest
        assert "simulations" not in qiskit_dev.tracker.totals


class TestMockedExecution:
    def test_get_transpile_args(self):
        """Test that get_transpile_args works as expected by filtering out
        kwargs that don't match the Qiskit transpile signature"""

        # on a device
        transpile_args = {
            "random_kwarg": 3,
            "seed_transpiler": 42,
            "optimization_level": 3,
            "circuits": [],
        }
        compile_backend = MockedBackend(name="compile_backend")
        dev = QiskitDevice(
            wires=5, backend=aer_backend, compile_backend=compile_backend, **transpile_args
        )
        assert dev._transpile_args == {
            "optimization_level": 3,
            "seed_transpiler": 42,
        }

    @patch("pennylane_qiskit.qiskit_device.transpile")
    @pytest.mark.parametrize("compile_backend", [None, MockedBackend(name="compile_backend")])
    def test_compile_circuits(self, transpile_mock, compile_backend):
        """Tests compile_circuits with a mocked transpile function to avoid calling
        a remote backend. Confirm compile_backend and transpile_args are used."""

        transpile_args = {"seed_transpiler": 42, "optimization_level": 2}
        dev = QiskitDevice(
            wires=5, backend=aer_backend, compile_backend=compile_backend, **transpile_args
        )

        transpile_mock.return_value = QuantumCircuit(2)

        # technically this doesn't matter due to the mock, but this is the correct input format for the function
        circuits = [
            QuantumScript([qml.PauliX(0)], measurements=[qml.expval(qml.PauliZ(0))]),
            QuantumScript([qml.PauliX(0)], measurements=[qml.probs(wires=[0])]),
            QuantumScript([qml.PauliX(0), qml.PauliZ(1)], measurements=[qml.counts()]),
        ]
        input_circuits = [circuit_to_qiskit(c, register_size=2) for c in circuits]

        with patch.object(dev, "get_transpile_args", return_value=transpile_args):
            compiled_circuits = dev.compile_circuits(input_circuits)

        transpile_mock.assert_called_with(
            input_circuits[2], backend=dev.compile_backend, **dev._transpile_args
        )

        assert len(compiled_circuits) == len(input_circuits)
        for _, circuit in enumerate(compiled_circuits):
            assert isinstance(circuit, QuantumCircuit)

    @pytest.mark.parametrize(
        "results, index",
        [
            ({"00": 125, "10": 500, "01": 250, "11": 125}, None),
            ([{}, {"00": 125, "10": 500, "01": 250, "11": 125}], 1),
            ([{}, {}, {"00": 125, "10": 500, "01": 250, "11": 125}], 2),
        ],
    )
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

    @patch("pennylane_qiskit.qiskit_device.QiskitDevice._execute_estimator")
    def test_execute_pipeline_primitives_no_session(self, mocker):
        """Test that a Primitives-based device initialized with no Session creates one for the
        execution, and then returns the device session to None."""

        dev = QiskitDevice(wires=5, backend=aer_backend, session=None)

        assert dev._session is None

        qs = QuantumScript([qml.PauliX(0), qml.PauliY(1)], measurements=[qml.expval(qml.PauliZ(0))])

        with patch("pennylane_qiskit.qiskit_device.Session") as mock_session:
            dev.execute(qs)
            mock_session.assert_called_once()  # a session was created

        assert dev._session is None  # the device session is still None

    @pytest.mark.parametrize("backend", [aer_backend, legacy_backend, FakeManila(), FakeManilaV2()])
    def test_execute_pipeline_with_all_execute_types_mocked(self, mocker, backend):
        """Test that a device executes measurements that require raw samples via the sampler,
        and the relevant primitive measurements via the estimator"""

        dev = QiskitDevice(wires=5, backend=backend, session=MockSession(backend))

        qs = QuantumScript(
            [qml.PauliX(0), qml.PauliY(1)],
            measurements=[
                qml.expval(qml.PauliZ(0)),
                qml.probs(wires=[0, 1]),
                qml.counts(),
                qml.sample(),
            ],
        )
        tapes, _ = split_execution_types(qs)

        with patch.object(dev, "_execute_sampler", return_value="sampler_execute_res"):
            with patch.object(dev, "_execute_estimator", return_value="estimator_execute_res"):
                sampler_execute = mocker.spy(dev, "_execute_sampler")
                estimator_execute = mocker.spy(dev, "_execute_estimator")

                res = dev.execute(tapes)

        sampler_execute.assert_called_once()
        estimator_execute.assert_called_once()

        assert res == [
            "estimator_execute_res",
            "sampler_execute_res",
        ]

    @patch("pennylane_qiskit.qiskit_device.Estimator")
    @patch("pennylane_qiskit.qiskit_device.QiskitDevice._process_estimator_job")
    @pytest.mark.parametrize("session", [None, MockSession(aer_backend)])
    def test_execute_estimator_mocked(self, mocked_estimator, mocked_process_fn, session):
        """Test the _execute_estimator function using a mocked version of Estimator
        that returns a meaningless result."""

        qs = QuantumScript(
            [qml.PauliX(0)],
            measurements=[qml.expval(qml.PauliY(0)), qml.var(qml.PauliX(0))],
            shots=100,
        )
        result = test_dev._execute_estimator(qs, session)

        # to emphasize, this did nothing except appease CodeCov
        assert isinstance(result, Mock)

    def test_shot_vector_error_mocked(self):
        """Test that a device that executes a circuit with an array of shots raises the appropriate ValueError"""

        dev = QiskitDevice(wires=5, backend=aer_backend, session=MockSession(aer_backend))
        qs = QuantumScript(
            measurements=[
                qml.expval(qml.PauliX(0)),
            ],
            shots=[5, 10, 2],
        )

        with patch.object(dev, "_execute_estimator"):
            with pytest.raises(ValueError, match="Setting shot vector"):
                dev.execute(qs)


class TestExecution:

    @pytest.mark.parametrize("wire", [0, 1])
    @pytest.mark.parametrize(
        "angle, op, expectation",
        [
            (np.pi / 2, qml.RX, [0, -1, 0, 1, 0, 1]),
            (np.pi, qml.RX, [0, 0, -1, 1, 1, 0]),
            (np.pi / 2, qml.RY, [1, 0, 0, 0, 1, 1]),
            (np.pi, qml.RY, [0, 0, -1, 1, 1, 0]),
            (np.pi / 2, qml.RZ, [0, 0, 1, 1, 1, 0]),
        ],
    )
    @flaky(max_runs=10, min_passes=7)
    def test_estimator_with_different_pauli_obs(self, mocker, wire, angle, op, expectation):
        """Test that the Estimator with various observables returns expected results.
        Essentially testing that the conversion to PauliOps in _execute_estimator behaves as
        expected. Iterating over wires ensures that the wire operated on and the wire measured
        correspond correctly (wire ordering convention in Qiskit and PennyLane don't match.)
        """

        dev = QiskitDevice(wires=5, backend=aer_backend)

        sampler_execute = mocker.spy(dev, "_execute_sampler")
        estimator_execute = mocker.spy(dev, "_execute_estimator")

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

        sampler_execute.assert_not_called()
        estimator_execute.assert_called_once()

        assert np.allclose(res, expectation, atol=0.1)

    @pytest.mark.parametrize("wire", [0, 1, 2, 3])
    @pytest.mark.parametrize(
        "angle, op, multi_q_obs",
        [
            (
                np.pi / 2,
                qml.RX,
                qml.ops.LinearCombination([1, 3], [qml.X(3) @ qml.Y(1), qml.Z(0) * 3]),
            ),
            (
                np.pi,
                qml.RX,
                qml.ops.LinearCombination([1, 3], [qml.X(3) @ qml.Y(1), qml.Z(0) * 3])
                - 4 * qml.X(2),
            ),
            (np.pi / 2, qml.RY, qml.sum(qml.PauliZ(0), qml.PauliX(1))),
            (np.pi, qml.RY, qml.dot([2, 3], [qml.X(0), qml.Y(0)])),
            (
                np.pi / 2,
                qml.RZ,
                qml.Hamiltonian([1], [qml.X(0) @ qml.Y(2)]) - 3 * qml.Z(3) @ qml.Z(1),
            ),
        ],
    )
    @flaky(max_runs=10, min_passes=7)
    @pytest.mark.xfail(
        reason="Qiskit variances is different from PennyLane variances, discussion pending on resolution"
    )
    def test_estimator_with_various_multi_qubit_pauli_obs(
        self, mocker, wire, angle, op, multi_q_obs
    ):
        """Test that the Estimator with various multi-qubit observables returns expected results.
        Essentially testing that the conversion to PauliOps in _execute_estimator behaves as
        expected. Iterating over wires ensures that the wire operated on and the wire measured
        correspond correctly (wire ordering convention in Qiskit and PennyLane don't match.)
        """

        pl_dev = qml.device("default.qubit", wires=4)
        dev = QiskitDevice(wires=4, backend=aer_backend)

        sampler_execute = mocker.spy(dev, "_execute_sampler")
        estimator_execute = mocker.spy(dev, "_execute_estimator")

        qs = QuantumScript(
            [op(angle, wire)],
            measurements=[
                qml.expval(multi_q_obs),
                qml.var(multi_q_obs),
            ],
            shots=10000,
        )

        res = dev.execute(qs)
        expectation = pl_dev.execute(qs)

        sampler_execute.assert_not_called()
        estimator_execute.assert_called_once()

        assert np.allclose(res[0], expectation, atol=0.3)  ## atol is high due to high variance

    def test_tape_shots_used_for_estimator(self, mocker):
        """Tests that device uses tape shots rather than device shots for estimator"""
        dev = QiskitDevice(wires=5, backend=aer_backend, shots=2)

        estimator_execute = mocker.spy(dev, "_execute_estimator")

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliX(0))

        circuit(shots=[5])

        estimator_execute.assert_called_once()
        # calculates # of shots executed from precision
        assert int(np.ceil(1 / dev._current_job[0].metadata["target_precision"] ** 2)) == 5

        circuit()
        assert int(np.ceil(1 / dev._current_job[0].metadata["target_precision"] ** 2)) == 2

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
            ([qml.expval(0.5 * qml.Y(0) + 0.5 * qml.Y(0) - 1.5 * qml.X(0) - 0.5 * qml.Y(0))], (0)),
            (
                [
                    qml.expval(
                        qml.ops.LinearCombination(
                            [1, 3, 4], [qml.X(3) @ qml.Y(2), qml.Y(4) - qml.X(2), qml.Z(2) * 3]
                        )
                        + qml.X(4)
                    )
                ],
                (16),
            ),
        ],
    )
    @flaky(max_runs=10, min_passes=7)
    def test_process_estimator_job(self, measurements, expectation):
        """Tests that the estimator returns expected and accurate results for an ``expval`` and ``var`` for a variety of multi-qubit observables"""

        # make PennyLane circuit
        qs = QuantumScript([], measurements=measurements)

        # convert to Qiskit circuit information
        qcirc = circuit_to_qiskit(qs, register_size=qs.num_wires, diagonalize=False, measure=False)
        pauli_observables = [mp_to_pauli(mp, qs.num_wires) for mp in qs.measurements]

        # run on simulator via Estimator
        estimator = Estimator(backend=aer_backend)
        compiled_circuits = [transpile(qcirc, backend=aer_backend)]
        circ_and_obs = [(compiled_circuits[0], pauli_observables)]
        result = estimator.run(circ_and_obs).result()

        assert isinstance(result[0].data.evs, np.ndarray)
        assert result[0].data.evs.size == len(qs.measurements)

        assert isinstance(result[0].metadata, dict)

        processed_result = QiskitDevice._process_estimator_job(qs.measurements, result)
        assert isinstance(processed_result, tuple)
        assert np.allclose(processed_result, expectation, atol=0.1)

    @pytest.mark.parametrize("num_wires", [1, 3, 5])
    @pytest.mark.parametrize("num_shots", [50, 100])
    def test_generate_samples(self, num_wires, num_shots):
        qs = QuantumScript([], measurements=[qml.expval(qml.PauliX(0))])
        dev = QiskitDevice(wires=num_wires, backend=aer_backend, shots=num_shots)
        dev._execute_sampler(circuit=qs, session=Session(backend=aer_backend))

        samples = dev.generate_samples(0)

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

    def test_tape_shots_used_for_sampler(self, mocker):
        """Tests that device uses tape shots rather than device shots for sampler"""
        dev = QiskitDevice(wires=5, backend=aer_backend, shots=2)

        sampler_execute = mocker.spy(dev, "_execute_sampler")

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return qml.probs(wires=[0, 1])

        circuit(shots=[5])

        sampler_execute.assert_called_once()
        assert dev._current_job.num_shots == 5

        # Should reset to device shots if circuit ran again without shots defined
        circuit()
        assert dev._current_job.num_shots == 2

    def test_error_for_shot_vector(self):
        """Tests that a ValueError is raised if a shot vector is passed."""
        dev = QiskitDevice(wires=5, backend=aer_backend, shots=2)

        @qml.qnode(dev)
        def circuit():
            return qml.sample(qml.PauliX(0))

        with pytest.raises(ValueError, match="Setting shot vector"):
            circuit(shots=[5, 10, 2])

        # Should reset to device shots if circuit ran again without shots defined
        circuit()
        assert dev._current_job.num_shots == 2

    @pytest.mark.parametrize(
        "observable",
        [
            [qml.Hadamard(0), qml.PauliX(1)],
            [qml.PauliZ(0), qml.Hadamard(1)],
            [qml.PauliZ(0), qml.Hadamard(0)],
        ],
    )
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @flaky(max_runs=10, min_passes=7)
    def test_no_pauli_observable_gives_accurate_answer(self, mocker, observable):
        """Test that the device uses _sampler and _execute_estimator appropriately and
        provides an accurate answer for measurements with observables that don't have a pauli_rep.
        """

        dev = QiskitDevice(wires=5, backend=aer_backend)

        pl_dev = qml.device("default.qubit", wires=5)

        estimator_execute = mocker.spy(dev, "_execute_estimator")
        sampler_execute = mocker.spy(dev, "_execute_sampler")

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            qml.Hadamard(0)
            return qml.expval(observable[0]), qml.expval(observable[1])

        @qml.qnode(pl_dev)
        def pl_circuit():
            qml.X(0)
            qml.Hadamard(0)
            return qml.expval(observable[0]), qml.expval(observable[1])

        res = circuit()
        pl_res = pl_circuit()

        estimator_execute.assert_called_once()
        sampler_execute.assert_called_once()

        assert np.allclose(res, pl_res, atol=0.1)

    def test_warning_for_split_execution_types_when_observable_no_pauli(self):
        """Test that a warning is raised when device is passed a measurement on
        an observable that does not have a pauli_rep."""

        dev = QiskitDevice(wires=5, backend=aer_backend)

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            qml.Hadamard(0)
            return qml.expval(qml.Hadamard(0))

        with pytest.warns(
            UserWarning,
            match="The observable measured",
        ):
            circuit()

    @pytest.mark.parametrize("backend", [aer_backend, FakeManila(), FakeManilaV2()])
    def test_qiskit_probability_output_format(self, backend):
        """Test that the format and values of the Qiskit device's output for `qml.probs` is
        the same as pennylane's."""

        dev = qml.device("default.qubit", wires=[0, 1, 2, 3, 4])
        qiskit_dev = QiskitDevice(wires=[0, 1, 2, 3, 4], backend=backend)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            return [qml.probs(wires=[0, 1])]

        @qml.qnode(qiskit_dev)
        def qiskit_circuit():
            qml.Hadamard(0)
            return [qml.probs(wires=[0, 1])]

        res = circuit()
        qiskit_res = qiskit_circuit()

        assert np.shape(res) == np.shape(qiskit_res)

    @pytest.mark.parametrize("backend", [aer_backend, FakeManila(), FakeManilaV2()])
    def test_sampler_output_shape(self, backend):
        """Test that the shape of the results produced from the sampler for the Qiskit device
        is consistent with Pennylane"""
        dev = qml.device("default.qubit", wires=5, shots=1024)
        qiskit_dev = QiskitDevice(wires=5, backend=backend)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return [qml.sample(qml.X(0) @ qml.Y(1)), qml.sample(qml.X(0))]

        @qml.qnode(qiskit_dev)
        def qiskit_circuit(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return [qml.sample(qml.X(0) @ qml.Y(1)), qml.sample(qml.X(0))]

        res = circuit(np.pi / 2)
        qiskit_res = qiskit_circuit(np.pi / 2)

        assert np.shape(res) == np.shape(qiskit_res)

    @pytest.mark.parametrize("backend", [aer_backend, FakeManila(), FakeManilaV2()])
    def test_sampler_output_shape_multi_measurements(self, backend):
        """Test that the shape of the results produced from the sampler for the Qiskit device
        is consistent with Pennylane for circuits with multiple measurements"""
        dev = qml.device("default.qubit", wires=5, shots=10)
        qiskit_dev = QiskitDevice(wires=5, backend=backend, shots=10)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return (
                qml.sample(),
                qml.sample(qml.Y(0)),
                qml.expval(qml.X(1)),
                qml.var(qml.Y(0)),
                qml.counts(),
            )

        @qml.qnode(qiskit_dev)
        def qiskit_circuit(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return (
                qml.sample(),
                qml.sample(qml.Y(0)),
                qml.expval(qml.X(1)),
                qml.var(qml.Y(0)),
                qml.counts(),
            )

        res = circuit(np.pi / 2)
        qiskit_res = qiskit_circuit(np.pi / 2)

        assert np.shape(res[0]) == np.shape(qiskit_res[0])
        assert np.shape(res[1]) == np.shape(qiskit_res[1])
        assert len(res) == len(qiskit_res)

    @pytest.mark.parametrize(
        "observable",
        [
            lambda: [qml.expval(qml.Hadamard(0)), qml.expval(qml.Hadamard(0))],
            lambda: [
                qml.var(qml.Hadamard(0)),
                qml.var(qml.Hadamard(0)),
            ],
            lambda: [
                qml.expval(qml.X(0)),
                qml.expval(qml.Y(1)),
                qml.expval(0.5 * qml.Y(1)),
                qml.expval(qml.Z(0) @ qml.Z(1)),
                qml.expval(qml.X(0) @ qml.Z(1) + 0.5 * qml.Y(1) + qml.Z(0)),
                qml.expval(
                    qml.ops.LinearCombination(
                        [0.35, 0.46], [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.X(2)]
                    )
                ),
                qml.expval(
                    qml.ops.LinearCombination(
                        [1.0, 2.0, 3.0], [qml.X(0), qml.X(1), qml.Z(0)], grouping_type="qwc"
                    )
                ),
            ],
            lambda: [
                qml.expval(
                    qml.Hamiltonian([0.35, 0.46], [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Y(2)])
                )
            ],
            lambda: [qml.expval(qml.X(0) @ qml.Z(1) + qml.Z(0))],
            pytest.param(
                [qml.var(qml.X(0) + qml.Z(0))],
                marks=pytest.mark.xfail(reason="Qiskit itself is bugged when given Sum"),
            ),
            lambda: [
                qml.expval(qml.Hadamard(0)),
                qml.expval(qml.Hadamard(1)),
                qml.expval(qml.Hadamard(0) @ qml.Hadamard(1)),
                qml.expval(
                    qml.Hadamard(0) @ qml.Hadamard(1) + 0.5 * qml.Hadamard(1) + qml.Hadamard(0)
                ),
            ],
        ],
    )
    @flaky(max_runs=10, min_passes=7)
    def test_observables_that_need_split_non_commuting(self, observable):
        """Tests that observables that have non-commuting measurements are
        processed correctly when executed by the Estimator or, in the case of
        qml.Hadamard, executed by the Sampler via expval() or var"""
        qiskit_dev = QiskitDevice(wires=3, backend=aer_backend, shots=30000)

        @qml.qnode(qiskit_dev)
        def qiskit_circuit():
            qml.RX(np.pi / 3, 0)
            qml.RZ(np.pi / 3, 0)
            return observable()

        dev = qml.device("default.qubit", wires=3, shots=30000)

        @qml.qnode(dev)
        def circuit():
            qml.RX(np.pi / 3, 0)
            qml.RZ(np.pi / 3, 0)
            return observable()

        qiskit_res = qiskit_circuit()
        res = circuit()

        assert np.allclose(res, qiskit_res, atol=0.05)

    @pytest.mark.parametrize(
        "observable",
        [
            pytest.param(
                lambda: [qml.counts(qml.X(0) + qml.Y(0)), qml.counts(qml.X(0))],
                marks=pytest.mark.xfail(reason="Split non commuting discussion pending"),
            ),
            pytest.param(
                lambda: [
                    qml.counts(qml.X(0) @ qml.Z(1) + 0.5 * qml.Y(1) + qml.Z(0)),
                    qml.counts(0.5 * qml.Y(1)),
                ],
                marks=pytest.mark.xfail(reason="Split non commuting discussion pending"),
            ),
        ],
    )
    @flaky(max_runs=10, min_passes=7)
    def test_observables_that_need_split_non_commuting_counts(self, observable):
        """Tests that observables that have non-commuting measurents are processed
        correctly when executed by the Sampler via counts()"""
        qiskit_dev = QiskitDevice(wires=3, backend=aer_backend, shots=4000)

        @qml.qnode(qiskit_dev)
        def qiskit_circuit():
            qml.RX(np.pi / 3, 0)
            qml.RZ(np.pi / 3, 0)
            return observable()

        dev = qml.device("default.qubit", wires=3, shots=4000)

        @qml.qnode(dev)
        def circuit():
            qml.RX(np.pi / 3, 0)
            qml.RZ(np.pi / 3, 0)
            return observable()

        qiskit_res = qiskit_circuit()
        res = circuit()

        assert len(qiskit_res) == len(res)
        for res1, res2 in zip(qiskit_res, res):
            assert all(res1[key] - res2.get(key, 0) < 300 for key in res1)

    @pytest.mark.parametrize(
        "observable",
        [
            pytest.param(
                lambda: [qml.sample(qml.X(0) + qml.Y(0)), qml.sample(qml.X(0))],
                marks=pytest.mark.xfail(reason="Split non commuting discussion pending"),
            ),
            pytest.param(
                lambda: [qml.sample(qml.X(0) @ qml.Y(1)), qml.sample(qml.X(0))],
                marks=pytest.mark.xfail(reason="Split non commuting discussion pending"),
            ),
            pytest.param(
                lambda: [
                    qml.sample(qml.X(0) @ qml.Z(1) + 0.5 * qml.Y(1) + qml.Z(0)),
                    qml.sample(0.5 * qml.Y(1)),
                ],
                marks=pytest.mark.xfail(reason="Split non commuting discussion pending"),
            ),
            pytest.param(
                lambda: [
                    qml.sample(qml.X(0)),
                    qml.sample(qml.Y(1)),
                    qml.sample(0.5 * qml.Y(1)),
                    qml.sample(qml.Z(0) @ qml.Z(1)),
                    qml.sample(qml.X(0) @ qml.Z(1) + 0.5 * qml.Y(1) + qml.Z(0)),
                    qml.sample(
                        qml.ops.LinearCombination(
                            [0.35, 0.46], [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.X(2)]
                        )
                    ),
                ],
                marks=pytest.mark.xfail(reason="Split non commuting discussion pending"),
            ),
            pytest.param(
                lambda: [
                    qml.sample(
                        qml.ops.LinearCombination(
                            [1.0, 2.0, 3.0], [qml.X(0), qml.X(1), qml.Z(0)], grouping_type="qwc"
                        )
                    ),
                ],
                marks=pytest.mark.xfail(reason="Split non commuting discussion pending"),
            ),
            pytest.param(
                lambda: [
                    qml.sample(
                        qml.Hamiltonian([0.35, 0.46], [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Y(2)])
                    )
                ],
                marks=pytest.mark.xfail(reason="Split non commuting discussion pending"),
            ),
        ],
    )
    @flaky(max_runs=10, min_passes=7)
    def test_observables_that_need_split_non_commuting_samples(self, observable):
        """Tests that observables that have non-commuting measurents are processed
        correctly when executed by the Sampler via sample()"""
        qiskit_dev = QiskitDevice(wires=3, backend=aer_backend, shots=20000)

        @qml.qnode(qiskit_dev)
        def qiskit_circuit():
            qml.RX(np.pi / 3, 0)
            qml.RZ(np.pi / 3, 0)
            return observable()

        dev = qml.device("default.qubit", wires=3, shots=20000)

        @qml.qnode(dev)
        def circuit():
            qml.RX(np.pi / 3, 0)
            qml.RZ(np.pi / 3, 0)
            return observable()

        qiskit_res = qiskit_circuit()
        res = circuit()

        assert np.allclose(np.mean(qiskit_res, axis=1), np.mean(res, axis=1), atol=0.05)
