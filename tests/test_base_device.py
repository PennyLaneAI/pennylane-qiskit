import numpy as np
import pytest

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

from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime.options import Options

from qiskit_aer.noise import NoiseModel

try:
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.backend("ibmq_qasm_simulator")
    hw_backend = service.least_busy(simulator=False, operational=True)
    test_dev = dev = QiskitDevice2(wires=5, backend=backend)
except:
    pass


def test_options():
    """Creates an Options object with defined values in multiple sub-categories"""
    options = Options()
    options.environment.job_tags = ["getting angle"]
    options.resilience.noise_amplifier = "placeholder"
    options.optimization_level = 2
    options.resilience_level = 1
    options.simulator.noise_model = "placeholder"
    return options


@pytest.mark.usefixtures("skip_if_no_account")
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
            dev = QiskitDevice2(wires=500, backend=hw_backend)

    def test_setting_simulator_noise_model(self):
        """Test that the simulator noise model saved on a passed Options
        object is used to set the backend noise model"""

        options = Options()
        options.simulator.noise_model = NoiseModel.from_backend(hw_backend)

        new_backend = service.backend("ibmq_qasm_simulator")
        dev1 = QiskitDevice2(wires=3, backend=backend)
        dev2 = QiskitDevice2(wires=3, backend=new_backend, options=options)

        assert dev1.backend.options.noise_model == None
        assert isinstance(dev2.backend.options.noise_model, NoiseModel)


@pytest.mark.usefixtures("skip_if_no_account")
class TestQiskitSessionManagement:
    """Test using Qiskit sessions with the device"""

    def test_default_no_session_on_initialization(self):
        """Test that the default behaviour is no session at initialization"""

        dev = QiskitDevice2(wires=2, backend=backend)
        assert dev._session == None

    def test_initializing_with_session(self):
        """Test that you can initialize a device with an existing Qiskit session"""

        session = Session(backend=backend, max_time="1m")
        session2 = Session(backend=backend, max_time="1m")
        dev = QiskitDevice2(wires=2, backend=backend, session=session)

        assert dev._session == session
        assert dev._session != session2

    @pytest.mark.parametrize(
        "initial_session", [None, Session(backend=backend, max_time="1m")]
    )
    def test_using_session_context(self, initial_session):
        """Test that you can add a session within a context manager"""

        dev = QiskitDevice2(wires=2, backend=backend, session=initial_session)

        assert dev._session == initial_session

        with qiskit_session(dev) as session:
            assert dev._session == session
            assert dev._session != initial_session

        assert dev._session == initial_session

    @pytest.mark.parametrize(
        "initial_session", [None, Session(backend=backend, max_time="1m")]
    )
    def test_update_session(self, initial_session):
        """Test that you can update the session stored on the device"""

        dev = QiskitDevice2(wires=2, backend=backend, session=initial_session)
        assert dev._session == initial_session

        new_session = session2 = Session(backend=backend, max_time="1m")
        dev.update_session(new_session)

        assert dev._session != initial_session
        assert dev._session == new_session


class TestDevicePreprocessing:
    """Tests the device preprocessing functions"""

    @pytest.mark.parametrize(
        "mp, res",
        [(qml.measurements.CountsMP, True), (qml.measurements.StateMP, False)],
    )
    def test_accepted_sample_measurement(self, mp, res):
        """Test that the accepted_sample_measurement function
        for validate_measurements in preprocesing works as expected"""
        # this is not working the way I expected -
        # why is expval allowed through when I've defined this in this way??
        raise RuntimeError

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
            )
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
        assert reorder_fn([tape.measurements for tape in tapes]) == tuple(
            qs.measurements
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
    @pytest.mark.usefixtures("skip_if_no_account")
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
    @pytest.mark.usefixtures("skip_if_no_account")
    def test_observable_stopping_condition(self, obs, expected):
        """Test that observable_stopping_condition works"""
        res = test_dev.observable_stopping_condition(obs)
        assert res == expected

    @pytest.mark.usefixtures("skip_if_no_account")
    def test_device_preprocessing_state_measurement_raises_error(self):
        """Test that the preprocessing function on a tape with a state-based
        measurement raises an error"""
        raise RuntimeError

    def test_preprocess_splits_incompatible_primitive_measurements(self):
        raise RuntimeError

    def test_preprocess_measurements_without_primitives(self):
        raise RuntimeError

    def test_preprocess_decomposes_unsupported_observable(self):
        raise RuntimeError

    def test_preprocess_decomposes_unsupported_operator(self):
        raise RuntimeError


@pytest.mark.usefixtures("skip_if_no_account")
class TestOptionsHandling:
    def test_qiskit_options_to_flat_dict(self):
        """Test that a Qiskit Options object is converted to an un-nested python dictionary"""

        options = test_options()

        options_dict = qiskit_options_to_flat_dict(options)

        assert isinstance(options_dict, dict)
        # the values in the dict are not themselves dictionaries or convertable to dictionaries
        for val in options_dict.values():
            assert not hasattr(val, "__dict__")
            assert not isinstance(val, dict)

    @pytest.mark.parametrize("options", [None, test_options()])
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

        options = test_options()
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

        dev = QiskitDevice2(
            wires=2, backend=backend, random_kwarg1=True, random_kwarg2="a"
        )

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

        dev = QiskitDevice2(
            wires=2, backend=backend, random_kwarg1=True, max_execution_time="1m"
        )

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


@pytest.mark.usefixtures("skip_if_no_account")
class TestDeviceProperties:
    def test_name_property(self):
        """Test the backend property"""

        dev = QiskitDevice2(wires=2, backend=backend)
        assert dev.name == "qiskit.remote2"

    def test_backend_property(self):
        """Test the backend property"""

        dev = QiskitDevice2(wires=2, backend=backend)
        assert dev.backend == dev._backend
        assert dev.backend == backend

    def test_service_property(self):
        """Test the service property"""

        dev = QiskitDevice2(wires=2, backend=backend)
        assert dev.service == dev._service
        assert dev.service == service

    def test_session_property(self):
        """Test the session property"""

        session = Session(backend=backend)
        dev = QiskitDevice2(wires=2, backend=backend, session=session)
        assert dev.session == dev._session
        assert dev.session == session

    def test_num_wires_property(self):
        """Test the num_wires property"""

        wires = [1, 2, 3]
        dev = QiskitDevice2(wires=wires, backend=backend)
        assert dev.num_wires == len(wires)
