# Copyright 2021-2022 Xanadu Quantum Technologies Inc.

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
This module contains tests for PennyLane IBMQ devices.
"""
import numpy as np
import pennylane as qml
import pytest

from qiskit import IBMQ
from qiskit.providers.ibmq.exceptions import IBMQAccountError

from pennylane_qiskit import IBMQDevice
from pennylane_qiskit import ibmq as ibmq


def test_load_from_env(token, monkeypatch):
    """test loading an IBMQ device from
    an env variable"""
    monkeypatch.setenv("IBMQX_TOKEN", token)
    dev = IBMQDevice(wires=1)
    assert dev.provider.credentials.is_ibmq()


def test_load_from_env_multiple_device(token, monkeypatch):
    """Test creating multiple IBMQ devices when the environment variable
    for the IBMQ token was set."""
    monkeypatch.setenv("IBMQX_TOKEN", token)
    dev1 = IBMQDevice(wires=1)
    dev2 = IBMQDevice(wires=1)

    assert dev1.provider.credentials.is_ibmq()
    assert dev2.provider.credentials.is_ibmq()


def test_load_from_env_multiple_device_and_token(monkeypatch):
    """Test creating multiple devices when the different tokens are defined
    using an environment variable."""
    mock_provider = "MockProvider"
    mock_qiskit_device = MockQiskitDeviceInit()

    with monkeypatch.context() as m:
        m.setattr(ibmq.QiskitDevice, "__init__", mock_qiskit_device.mocked_init)

        creds = []

        def enable_account(new_creds):
            creds.append(new_creds)

        def active_account():
            if len(creds) != 0:
                return {"token": creds[-1]}

        m.setattr(ibmq.IBMQ, "enable_account", enable_account)
        m.setattr(ibmq.IBMQ, "disable_account", lambda: None)
        m.setattr(ibmq.IBMQ, "active_account", active_account)

        m.setenv("IBMQX_TOKEN", "TOKEN1")
        dev1 = IBMQDevice(wires=1, provider=mock_provider)
        # first login
        assert creds == ["TOKEN1"]
        dev1 = IBMQDevice(wires=1, provider=mock_provider)
        # same token, login is elided
        assert creds == ["TOKEN1"]

        m.setenv("IBMQX_TOKEN", "TOKEN2")
        dev2 = IBMQDevice(wires=1, provider=mock_provider)
        # second login
        assert creds == ["TOKEN1", "TOKEN2"]


def test_load_kwargs_takes_precedence(token, monkeypatch):
    """Test that with a potentially valid token stored as an environment
    variable, passing the token as a keyword argument takes precedence."""
    monkeypatch.setenv("IBMQX_TOKEN", "SomePotentiallyValidToken")
    dev = IBMQDevice(wires=1, ibmqx_token=token)
    assert dev.provider.credentials.is_ibmq()


def test_load_env_empty_string_has_short_error(monkeypatch):
    """Test that the empty string is treated as a missing token."""
    monkeypatch.setenv("IBMQX_TOKEN", "")
    with pytest.raises(IBMQAccountError, match="No active IBM Q account"):
        IBMQDevice(wires=1)


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
    monkeypatch.setenv("IBMQX_TOKEN", '1')

    with monkeypatch.context() as m:
        m.setattr(ibmq.QiskitDevice, "__init__", mock_qiskit_device.mocked_init)
        m.setattr(ibmq.IBMQ, "enable_account", lambda *args, **kwargs: None)


        # Here mocking to a value such that it is not None
        m.setattr(ibmq.IBMQ, "active_account", lambda *args, **kwargs: {"token": '1'})
        dev = IBMQDevice(wires=2, backend="ibmq_qasm_simulator", provider=mock_provider)

    assert mock_qiskit_device.provider == mock_provider


def mock_get_provider(*args, **kwargs):
    """A mock function for the get_provider Qiskit function to record the
    arguments which it was called with."""
    return (args, kwargs)


def test_default_provider(monkeypatch):
    """Tests that the default provider is used when no custom provider was
    specified."""
    mock_qiskit_device = MockQiskitDeviceInit()
    monkeypatch.setenv("IBMQX_TOKEN", '1')

    with monkeypatch.context() as m:
        m.setattr(ibmq.QiskitDevice, "__init__", mock_qiskit_device.mocked_init)
        m.setattr(ibmq.IBMQ, "get_provider", mock_get_provider)
        m.setattr(ibmq.IBMQ, "enable_account", lambda *args, **kwargs: None)

        # Here mocking to a value such that it is not None
        m.setattr(ibmq.IBMQ, "active_account", lambda *args, **kwargs: {"token": '1'})
        dev = IBMQDevice(wires=2, backend="ibmq_qasm_simulator")

    assert mock_qiskit_device.provider[0] == ()
    assert mock_qiskit_device.provider[1] == {"hub": "ibm-q", "group": "open", "project": "main"}


def test_custom_provider_hub_group_project(monkeypatch):
    """Tests that the custom arguments passed during device instantiation are
    used when calling get_provider."""
    mock_qiskit_device = MockQiskitDeviceInit()
    monkeypatch.setenv("IBMQX_TOKEN", '1')

    custom_hub = "SomeHub"
    custom_group = "SomeGroup"
    custom_project = "SomeProject"

    with monkeypatch.context() as m:
        m.setattr(ibmq.QiskitDevice, "__init__", mock_qiskit_device.mocked_init)
        m.setattr(ibmq.IBMQ, "get_provider", mock_get_provider)
        m.setattr(ibmq.IBMQ, "enable_account", lambda *args, **kwargs: None)

        # Here mocking to a value such that it is not None
        m.setattr(ibmq.IBMQ, "active_account", lambda *args, **kwargs: {"token": '1'})
        dev = IBMQDevice(
            wires=2,
            backend="ibmq_qasm_simulator",
            hub=custom_hub,
            group=custom_group,
            project=custom_project,
        )

    assert mock_qiskit_device.provider[0] == ()
    assert mock_qiskit_device.provider[1] == {
        "hub": custom_hub,
        "group": custom_group,
        "project": custom_project,
    }


def test_load_from_disk(token):
    """Test loading the account credentials and the device from disk."""
    IBMQ.save_account(token)
    dev = IBMQDevice(wires=1)
    assert dev.provider.credentials.is_ibmq()
    IBMQ.delete_account()


def test_account_error(monkeypatch):
    """Test that an error is raised if there is no active IBMQ account."""

    # Token is passed such that the test is skipped if no token was provided
    with pytest.raises(IBMQAccountError, match="No active IBM Q account"):
        with monkeypatch.context() as m:
            m.delenv("IBMQX_TOKEN", raising=False)
            IBMQDevice(wires=1)


@pytest.mark.parametrize("shots", [1000])
def test_simple_circuit(token, tol, shots):
    """Test executing a simple circuit submitted to IBMQ."""
    IBMQ.enable_account(token)
    dev = IBMQDevice(wires=2, backend="ibmq_qasm_simulator", shots=shots)

    @qml.qnode(dev)
    def circuit(theta, phi):
        qml.RX(theta, wires=0)
        qml.RX(phi, wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    theta = 0.432
    phi = 0.123

    res = circuit(theta, phi)
    expected = np.array([np.cos(theta), np.cos(theta) * np.cos(phi)])
    assert np.allclose(res, expected, **tol)


@pytest.mark.parametrize("shots", [1000])
def test_simple_circuit_with_batch_params(token, tol, shots, mocker):
    """Test that executing a simple circuit with batched parameters is
    submitted to IBMQ once."""
    IBMQ.enable_account(token)
    dev = IBMQDevice(wires=2, backend="ibmq_qasm_simulator", shots=shots)

    @qml.batch_params(all_operations=True)
    @qml.qnode(dev)
    def circuit(theta, phi):
        qml.RX(theta, wires=0)
        qml.RX(phi, wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    # Check that we run only once
    spy1 = mocker.spy(dev, "batch_execute")
    spy2 = mocker.spy(dev.backend, "run")

    # Batch the input parameters
    batch_dim = 3
    theta = np.linspace(0, 0.543, batch_dim)
    phi = np.linspace(0, 0.123, batch_dim)

    res = circuit(theta, phi)
    assert np.allclose(res[:, 0], np.cos(theta), **tol)
    assert np.allclose(res[:, 1], np.cos(theta) * np.cos(phi), **tol)

    # Check that IBMQBackend.run was called once
    assert spy1.call_count == 1
    assert spy2.call_count == 1


@pytest.mark.parametrize("shots", [1000])
def test_batch_execute_parameter_shift(token, tol, shots, mocker):
    """Test that devices provide correct result computing the gradient of a
    circuit using the parameter-shift rule and the batch execution pipeline."""
    IBMQ.enable_account(token)
    dev = IBMQDevice(wires=3, backend="ibmq_qasm_simulator", shots=shots)

    spy1 = mocker.spy(dev, "batch_execute")
    spy2 = mocker.spy(dev.backend, "run")

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(x, y):
        qml.RX(x, wires=[0])
        qml.RY(y, wires=[1])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliZ(2))

    x = qml.numpy.array(0.543, requires_grad=True)
    y = qml.numpy.array(0.123, requires_grad=True)

    res = qml.grad(circuit)(x, y)
    expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
    assert np.allclose(res, expected, **tol)

    # Check that QiskitDevice.batch_execute was called once
    assert spy1.call_count == 2

    # Check that run was called twice: for the partial derivatives and for
    # running the circuit
    assert spy2.call_count == 2


@pytest.mark.parametrize("shots", [1000])
def test_probability(token, tol, shots):
    """Test that the probs function works."""
    IBMQ.enable_account(token)
    dev = IBMQDevice(wires=2, backend="ibmq_qasm_simulator", shots=shots)
    dev_analytic = qml.device("default.qubit", wires=2, shots=None)

    x = [0.2, 0.5]

    def circuit(x):
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    prob = qml.QNode(circuit, dev)
    prob_analytic = qml.QNode(circuit, dev_analytic)

    # Calling the hardware only once
    hw_prob = prob(x)

    assert np.isclose(hw_prob.sum(), 1, **tol)
    assert np.allclose(prob_analytic(x), hw_prob, **tol)
    assert not np.array_equal(prob_analytic(x), hw_prob)


def test_track(token):
    """Test that the tracker works."""

    IBMQ.enable_account(token)
    dev = IBMQDevice(wires=1, backend="ibmq_qasm_simulator", shots=1)
    dev.tracker.active = True

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=0)
        return qml.probs(wires=0)

    circuit()

    assert "job_time" in dev.tracker.history
    if "job_time" in dev.tracker.history:
        assert "creating" in dev.tracker.history["job_time"][0]
        assert "validating" in dev.tracker.history["job_time"][0]
        assert "queued" in dev.tracker.history["job_time"][0]
        assert "running" in dev.tracker.history["job_time"][0]
        assert len(dev.tracker.history["job_time"][0]) == 4
