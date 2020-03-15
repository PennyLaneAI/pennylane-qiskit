import os

import numpy as np
import pennylane as qml
import pytest

from qiskit import IBMQ
from qiskit.providers.ibmq.exceptions import IBMQAccountError

from pennylane_qiskit import IBMQDevice
from pennylane_qiskit import ibmq as ibmq
from pennylane_qiskit import qiskit_device as qiskit_device


@pytest.fixture
def token():
    t = os.getenv("IBMQX_TOKEN_TEST", None)

    if t is None:
        pytest.skip("Skipping test, no IBMQ token available")

    yield t
    IBMQ.disable_account()


def test_load_from_env(token, monkeypatch):
    """test loading an IBMQ device from
    an env variable"""
    monkeypatch.setenv("IBMQX_TOKEN", token)
    dev = IBMQDevice(wires=1)
    assert dev.provider.credentials.is_ibmq()


def test_account_already_loaded(token):
    """Test loading an IBMQ device using
    an already loaded account"""
    IBMQ.enable_account(token)
    dev = IBMQDevice(wires=1)
    assert dev.provider.credentials.is_ibmq()

class MockQiskitDeviceInit:
    """A mocked version of the QiskitDevice __init__ method which
    is called on by the IBMQDevice"""

    def mocked_init(self, wires, provider, backend, shots, **kwargs):
        """Stores the provider which QiskitDevice.__init__ was
        called with."""
        self.provider = provider

def test_custom_provider(monkeypatch):
    """Tests that a custom provider can be passed when creating an IBMQ
    device."""
    mock_provider = "MockProvider"
    mock_qiskit_device = MockQiskitDeviceInit()

    with monkeypatch.context() as m:
        m.setattr(ibmq.QiskitDevice, "__init__",mock_qiskit_device.mocked_init)
        m.setattr(ibmq.IBMQ, "enable_account", lambda *args, **kwargs: None)

        # Here mocking to a value such that it is not None
        m.setattr(ibmq.IBMQ, "active_account", lambda *args, **kwargs: True)
        dev = IBMQDevice(wires=2, backend='ibmq_qasm_simulator', provider=mock_provider)

    assert mock_qiskit_device.provider == mock_provider

def mock_get_provider(*args, **kwargs):
    """A mock function for the get_provider Qiskit function to record the
    arguments which it was called with."""
    return (args, kwargs)

def test_default_provider(monkeypatch):
    """Tests that the default provider is used when no custom provider was
    specified."""
    mock_qiskit_device = MockQiskitDeviceInit()

    with monkeypatch.context() as m:
        m.setattr(ibmq.QiskitDevice, "__init__", mock_qiskit_device.mocked_init)
        m.setattr(ibmq.IBMQ, "get_provider", mock_get_provider)
        m.setattr(ibmq.IBMQ, "enable_account", lambda *args, **kwargs: None)

        # Here mocking to a value such that it is not None
        m.setattr(ibmq.IBMQ, "active_account", lambda *args, **kwargs: True)
        dev = IBMQDevice(wires=2, backend='ibmq_qasm_simulator')

    assert mock_qiskit_device.provider[0] == ()
    assert mock_qiskit_device.provider[1] == {'hub': 'ibm-q', 'group': 'open', 'project': 'main'}

def test_custom_provider_hub_group_project(monkeypatch):
    """Tests that the custom arguments passed during device instantiation are
    used when calling get_provider."""
    mock_qiskit_device = MockQiskitDeviceInit()

    custom_hub = "SomeHub"
    custom_group = "SomeGroup"
    custom_project = "SomeProject"

    with monkeypatch.context() as m:
        m.setattr(ibmq.QiskitDevice, "__init__", mock_qiskit_device.mocked_init)
        m.setattr(ibmq.IBMQ, "get_provider", mock_get_provider)
        m.setattr(ibmq.IBMQ, "enable_account", lambda *args, **kwargs: None)

        # Here mocking to a value such that it is not None
        m.setattr(ibmq.IBMQ, "active_account", lambda *args, **kwargs: True)
        dev = IBMQDevice(wires=2, backend='ibmq_qasm_simulator', hub=custom_hub, group=custom_group, project=custom_project)

    assert mock_qiskit_device.provider[0] == ()
    assert mock_qiskit_device.provider[1] == {'hub': custom_hub, 'group': custom_group, 'project': custom_project}


def test_load_from_disk(token):
    IBMQ.save_account(token)
    dev = IBMQDevice(wires=1)
    assert dev.provider.credentials.is_ibmq()
    IBMQ.delete_account()


def test_account_error():

    # Token is passed such that the test is skipped if no token was provided
    with pytest.raises(IBMQAccountError, match="No active IBM Q account"):
        IBMQDevice(wires=1)


@pytest.mark.parametrize("analytic", [False])
@pytest.mark.parametrize("shots", [8192])
def test_simple_circuit(token, tol, shots):
    IBMQ.enable_account(token)
    dev = IBMQDevice(wires=2, backend="ibmq_qasm_simulator", shots=shots)

    @qml.qnode(dev)
    def circuit(theta, phi):
        qml.RX(theta, wires=0)
        qml.RX(phi, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    theta = 0.432
    phi = 0.123

    res = circuit(theta, phi)
    expected = np.array([np.cos(theta), np.cos(theta) * np.cos(phi)])
    assert np.allclose(res, expected, **tol)
