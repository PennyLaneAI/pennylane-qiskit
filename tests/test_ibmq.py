import os

import numpy as np
import pennylane as qml
import pytest

from qiskit import IBMQ
from qiskit.providers.ibmq.exceptions import IBMQAccountError

from pennylane_qiskit import IBMQDevice


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


def test_load_from_disk(token):
    IBMQ.save_account(token)
    dev = IBMQDevice(wires=1)
    assert dev.provider.credentials.is_ibmq()
    IBMQ.delete_account()


def test_account_error(token):

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
