import numpy as np
import pytest

import pennylane as qml
from pennylane_qiskit import AerDevice
from pennylane_qiskit.qiskit_device2 import QiskitDevice2, qiskit_session, accepted_sample_measurement, qiskit_options_to_flat_dict

from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime.options import Options

try:
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.backend("ibmq_qasm_simulator")
except:
    pass

@pytest.mark.usefixtures("skip_if_no_account")
class TestDeviceInitialization:

    @pytest.mark.parametrize("use_primitives", [True, False])
    def test_use_primitives_kwarg(self, use_primitives):
        """Test the _use_primitives attribute is set on initialization"""
        dev = QiskitDevice2(wires=2, backend=backend, use_primitives=use_primitives)
        assert dev._use_primitives == use_primitives


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

    @pytest.mark.parametrize("initial_session", [None, Session(backend=backend, max_time="1m")])
    def test_using_session_context(self, initial_session):
        """Test that you can add a session within a context manager"""

        dev = QiskitDevice2(wires=2, backend=backend, session=initial_session)

        assert dev._session == initial_session

        with qiskit_session(dev) as session:
            assert dev._session == session
            assert dev._session != initial_session

        assert dev._session == initial_session

    @pytest.mark.parametrize("initial_session", [None, Session(backend=backend, max_time="1m")])
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

    @pytest.mark.parametrize("mp, res", [(qml.measurements.CountsMP, True), (qml.measurements.StateMP, False)])
    def test_accepted_sample_measurement(self, mp, res):
        """Test that the accepted_sample_measurement function
        for validate_measurements in preprocesing works as expected"""
        # this is not working the way I expected -
        # why is expval allowed through when I've defined this in this way??
        raise RuntimeError

    @pytest.mark.parametrize("tape, expectation", [(None, None)])
    def test_split_measurement_types(self, tape, expectation):
        """Test that the split_measurement_types transform """
        raise RuntimeError

    @pytest.mark.usefixtures("skip_if_no_account")
    def test_device_preprocessing_state_measurement_raises_error(self):
        """Test that the preprocessing function on a tape with a state-based measurement raises an error"""
        raise RuntimeError


class TestOptionsHandling:

    def test_qiskit_options_to_flat_dict(self):
        """Test that a Qiskit Options object is converted to an un-nested python dictionary"""

        options = Options()
        options.environment.job_tags = ["getting angle"]
        options.resilience.noise_amplifier = "placeholder"
        options.optimization_level = 2
        options.resilience_level = 1
        options.simulator.noise_model = "placeholder"

        options_dict = qiskit_options_to_flat_dict(options)

        assert isinstance(options_dict, dict)
        # the values in the dict are not themselves dictionaries or convertable to dictionaries
        for val in options_dict.values():
            assert not hasattr(val, "__dict__")
            assert not isinstance(val, dict)

@pytest.mark.usefixtures("skip_if_no_account")
class TestDeviceProperties:

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


