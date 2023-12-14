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
from functools import partial
import numpy as np
import pennylane as qml
import pytest
from unittest.mock import patch

from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_provider.exceptions import IBMAccountError
from qiskit_ibm_provider.job import IBMJobError, IBMCircuitJob

from pennylane_qiskit import IBMQDevice
from pennylane_qiskit import ibmq


class MockQiskitDeviceInit:
    """A mocked version of the QiskitDevice __init__ method which
    is called on by the IBMQDevice"""

    def mocked_init(self, wires, provider, backend, shots, **kwargs):
        """Stores the provider which QiskitDevice.__init__ was
        called with."""
        self.provider = provider


def test_multi_load_changing_token(monkeypatch):
    """Test multiple account loads with changing tokens."""
    with monkeypatch.context() as m:
        # unrelated mock
        m.setattr(ibmq.QiskitDevice, "__init__", lambda self, *a, **k: None)

        # mock save_account to save the token, saved_accounts lists those tokens
        tokens = []

        def saved_accounts():
            return {f"account-{i}": {"token": t} for i, t in enumerate(tokens)}

        def save_account(token=None, **kwargs):
            tokens.append(token)

        m.setattr(IBMProvider, "saved_accounts", saved_accounts)
        m.setattr(IBMProvider, "save_account", save_account)
        m.setattr(IBMProvider, "__init__", lambda self, *a, **k: None)

        m.setenv("IBMQX_TOKEN", "T1")
        IBMQDevice(wires=1)
        assert tokens == ["T1"]
        IBMQDevice(wires=1)
        assert tokens == ["T1"]

        m.setenv("IBMQX_TOKEN", "T2")
        IBMQDevice(wires=1)
        assert tokens == ["T1", "T2"]


def test_load_kwargs_takes_precedence(monkeypatch, mocker):
    """Test that with a potentially valid token stored as an environment
    variable, passing the token as a keyword argument takes precedence."""
    monkeypatch.setenv("IBMQX_TOKEN", "SomePotentiallyValidToken")
    mock = mocker.patch("qiskit_ibm_provider.IBMProvider.save_account")

    with monkeypatch.context() as m:
        m.setattr(IBMProvider, "__init__", lambda self, *a, **k: None)
        m.setattr(ibmq.QiskitDevice, "__init__", lambda self, *a, **k: None)
        IBMQDevice(wires=1, ibmqx_token="TheTrueToken")

    mock.assert_called_with(token="TheTrueToken", url=None, instance="ibm-q/open/main")


def test_custom_provider(monkeypatch):
    """Tests that a custom provider can be passed when creating an IBMQ
    device."""
    mock_provider = "MockProvider"
    mock_qiskit_device = MockQiskitDeviceInit()
    monkeypatch.setenv("IBMQX_TOKEN", "1")

    with monkeypatch.context() as m:
        m.setattr(IBMProvider, "__init__", lambda self, *a, **k: None)
        m.setattr(ibmq.QiskitDevice, "__init__", mock_qiskit_device.mocked_init)
        m.setattr(IBMProvider, "saved_accounts", lambda: {"my-account": {"token": "1"}})
        IBMQDevice(wires=2, backend="ibmq_qasm_simulator", provider=mock_provider)

    assert mock_qiskit_device.provider == mock_provider


def test_default_provider(monkeypatch):
    """Tests that the default provider is used when no custom provider was
    specified."""
    mock_qiskit_device = MockQiskitDeviceInit()
    monkeypatch.setenv("IBMQX_TOKEN", "1")

    def provider_init(self, instance=None):
        self.instance = instance

    with monkeypatch.context() as m:
        m.setattr(ibmq.QiskitDevice, "__init__", mock_qiskit_device.mocked_init)
        m.setattr(IBMProvider, "__init__", provider_init)
        m.setattr(IBMProvider, "saved_accounts", lambda: {"my-account": {"token": "1"}})
        IBMQDevice(wires=2, backend="ibmq_qasm_simulator")

    assert isinstance(mock_qiskit_device.provider, IBMProvider)
    assert mock_qiskit_device.provider.instance == "ibm-q/open/main"


def test_custom_provider_hub_group_project_url(monkeypatch, mocker):
    """Tests that the custom arguments passed during device instantiation are
    used when calling IBMProvider.save_account"""
    monkeypatch.setenv("IBMQX_TOKEN", "1")
    mock = mocker.patch("qiskit_ibm_provider.IBMProvider.save_account")

    custom_hub = "SomeHub"
    custom_group = "SomeGroup"
    custom_project = "SomeProject"
    instance = f"{custom_hub}/{custom_group}/{custom_project}"

    with monkeypatch.context() as m:
        m.setattr(ibmq.QiskitDevice, "__init__", lambda *a, **k: None)
        m.setattr(IBMProvider, "__init__", lambda self, *a, **k: None)
        IBMQDevice(
            wires=2,
            backend="ibmq_qasm_simulator",
            hub=custom_hub,
            group=custom_group,
            project=custom_project,
            ibmqx_url="example.com",
        )

    mock.assert_called_with(token="1", url="example.com", instance=instance)


@pytest.mark.usefixtures("skip_if_account_saved")
class TestMustNotHaveAccount:
    """Tests that require the user _not_ have an IBMQ account loaded."""

    def test_load_env_empty_string_has_short_error(self, monkeypatch):
        """Test that the empty string is treated as a missing token."""
        monkeypatch.setenv("IBMQX_TOKEN", "")
        with pytest.raises(IBMAccountError, match="No active IBM Q account"):
            IBMQDevice(wires=1)

    def test_account_error(self, monkeypatch):
        """Test that an error is raised if there is no active IBMQ account."""

        # Token is passed such that the test is skipped if no token was provided
        with pytest.raises(IBMAccountError, match="No active IBM Q account"):
            with monkeypatch.context() as m:
                m.delenv("IBMQX_TOKEN", raising=False)
                IBMQDevice(wires=1)


@pytest.mark.usefixtures("skip_if_no_account")
class TestIBMQWithRealAccount:
    """Tests that require an active IBMQ account."""

    def test_load_from_env_multiple_device(self):
        """Test creating multiple IBMQ devices when the environment variable
        for the IBMQ token was set."""
        dev1 = IBMQDevice(wires=1)
        dev2 = IBMQDevice(wires=1)
        assert dev1 is not dev2

    @pytest.mark.parametrize("shots", [1000])
    def test_simple_circuit(self, tol, shots):
        """Test executing a simple circuit submitted to IBMQ."""
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
    def test_simple_circuit_with_batch_params(self, tol, shots, mocker):
        """Test that executing a simple circuit with batched parameters is
        submitted to IBMQ once."""
        dev = IBMQDevice(wires=2, backend="ibmq_qasm_simulator", shots=shots)

        @partial(qml.batch_params, all_operations=True)
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
        assert np.allclose(res[0], np.cos(theta), **tol)
        assert np.allclose(res[1], np.cos(theta) * np.cos(phi), **tol)

        # Check that IBMQBackend.run was called once
        assert spy1.call_count == 1
        assert spy2.call_count == 1

    @pytest.mark.parametrize("shots", [1000])
    def test_batch_execute_parameter_shift(self, tol, shots, mocker):
        """Test that devices provide correct result computing the gradient of a
        circuit using the parameter-shift rule and the batch execution pipeline."""
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
    def test_probability(self, tol, shots):
        """Test that the probs function works."""
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

    def test_track(self):
        """Test that the tracker works."""
        dev = IBMQDevice(wires=1, backend="ibmq_qasm_simulator", shots=1)
        dev.tracker.active = True

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            return qml.probs(wires=0)

        circuit()

        assert "job_time" in dev.tracker.history
        assert set(dev.tracker.history["job_time"][0]) == {"queued", "running"}

    @patch(
        "qiskit_ibm_provider.job.ibm_circuit_job.IBMCircuitJob.time_per_step",
        return_value={"CREATING": "1683149330"},
    )
    @pytest.mark.parametrize("timeout", [None, 120])
    def test_track_fails_with_unexpected_metadata(self, mock_time_per_step, timeout, mocker):
        """Tests that the tracker fails when it doesn't get the required metadata."""
        batch_execute_spy = mocker.spy(ibmq.QiskitDevice, "batch_execute")
        wait_spy = mocker.spy(IBMCircuitJob, "wait_for_final_state")

        dev = IBMQDevice(wires=1, backend="ibmq_qasm_simulator", shots=1, timeout_secs=timeout)
        dev.tracker.active = True

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            return qml.probs(wires=0)

        with pytest.raises(IBMJobError, match="time_per_step had keys"):
            circuit()

        assert mock_time_per_step.call_count == 2
        batch_execute_spy.assert_called_with(dev, mocker.ANY, timeout=timeout)
        wait_spy.assert_called_with(mocker.ANY, timeout=timeout or 60)
