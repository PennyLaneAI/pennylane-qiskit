import os

import numpy as np
import pennylane as qml
import pytest

from qiskit import IBMQ

from pennylane_qiskit import IBMQCircuitRunnerDevice
from pennylane_qiskit import IBMQSamplerDevice


@pytest.fixture
def token():
    t = os.getenv("IBMQX_TOKEN_TEST", None)

    if t is None:
        pytest.skip("Skipping test, no IBMQ token available")

    yield t
    IBMQ.disable_account()


class TestCircuitRunner:
    def test_load_from_env(self, token, monkeypatch):
        """Test loading an IBMQ Circuit Runner Qiskit runtime device from an env variable."""
        monkeypatch.setenv("IBMQX_TOKEN", token)
        dev = IBMQCircuitRunnerDevice(wires=1)
        assert dev.provider.credentials.is_ibmq()

    def test_short_name(self, token):
        IBMQ.enable_account(token)
        dev = qml.device("qiskit.ibmq.circuitrunner", wires=1)
        return dev.provider.credentials.is_ibmq()

    @pytest.mark.parametrize("shots", [8000])
    def test_simple_circuit(self, token, tol, shots):
        """Test executing a simple circuit submitted to IBMQ."""
        IBMQ.enable_account(token)
        dev = IBMQCircuitRunnerDevice(wires=2, backend="ibmq_qasm_simulator", shots=shots)

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

    @pytest.mark.parametrize("shots", [8000])
    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "initial_layout": [0, 1],
                "layout_method": "trivial",
                "routing_method": "basic",
                "translation_method": "unroller",
                "seed_transpiler": 42,
                "optimization_level": 2,
                "init_qubits": True,
                "rep_delay": 0.01,
                "transpiler_options": {"approximation_degree": 1.0},
                "measurement_error_mmitigation": True,
            }
        ],
    )
    def test_kwargs_circuit(self, token, tol, shots, kwargs):
        """Test executing a simple circuit submitted to IBMQ."""
        IBMQ.enable_account(token)
        dev = IBMQCircuitRunnerDevice(wires=2, backend="ibmq_qasm_simulator", shots=shots, **kwargs)

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

    @pytest.mark.parametrize("shots", [8000])
    def test_batch_circuits(self, token, tol, shots):

        IBMQ.enable_account(token)
        dev = IBMQCircuitRunnerDevice(wires=2, backend="ibmq_qasm_simulator", shots=shots)

        # Batch the input parameters
        batch_dim = 3
        a = np.linspace(0, 0.543, batch_dim)
        b = np.linspace(0, 0.123, batch_dim)
        c = np.linspace(0, 0.987, batch_dim)

        @qml.batch_params
        @qml.qnode(dev)
        def circuit(x, y, z):
            """Reference QNode"""
            qml.PauliX(0)
            qml.Hadamard(wires=0)
            qml.Rot(x, y, z, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(a, b, c), np.cos(a) * np.sin(b), **tol)

    def test_track(token):
        """Test that the tracker works."""

        IBMQ.enable_account(token)
        dev = IBMQCircuitRunnerDevice(wires=1, backend="ibmq_qasm_simulator", shots=1)
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


class TestSampler:
    def test_load_from_env(self, token, monkeypatch):
        """Test loading an IBMQ Sampler Qiskit runtime device from an env variable."""
        monkeypatch.setenv("IBMQX_TOKEN", token)
        dev = IBMQSamplerDevice(wires=1)
        assert dev.provider.credentials.is_ibmq()

    def test_short_name(self, token):
        IBMQ.enable_account(token)
        dev = qml.device("qiskit.ibmq.sampler", wires=1)
        return dev.provider.credentials.is_ibmq()

    @pytest.mark.parametrize("shots", [8000])
    def test_simple_circuit(self, token, tol, shots):
        """Test executing a simple circuit submitted to IBMQ."""
        IBMQ.enable_account(token)
        dev = IBMQSamplerDevice(wires=2, backend="ibmq_qasm_simulator", shots=shots)

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

    @pytest.mark.parametrize("shots", [8000])
    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "return_mitigation_overhead": True,
                "run_config": {"seed_simulator": 42},
                "skip_transpilation": False,
                "transpile_config": {"approximation_degree": 1.0},
                "use_measurement_mitigation": True,
            }
        ],
    )
    def test_kwargs_circuit(self, token, tol, shots, kwargs):
        """Test executing a simple circuit submitted to IBMQ."""
        IBMQ.enable_account(token)
        dev = IBMQSamplerDevice(wires=2, backend="ibmq_qasm_simulator", shots=shots, **kwargs)

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

    @pytest.mark.parametrize("shots", [8000])
    def test_batch_circuits(self, token, tol, shots):

        IBMQ.enable_account(token)
        dev = IBMQSamplerDevice(wires=2, backend="ibmq_qasm_simulator", shots=shots)

        # Batch the input parameters
        batch_dim = 3
        a = np.linspace(0, 0.543, batch_dim)
        b = np.linspace(0, 0.123, batch_dim)
        c = np.linspace(0, 0.987, batch_dim)

        @qml.batch_params
        @qml.qnode(dev)
        def circuit(x, y, z):
            """Reference QNode"""
            qml.PauliX(0)
            qml.Hadamard(wires=0)
            qml.Rot(x, y, z, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(a, b, c), np.cos(a) * np.sin(b), **tol)

    def test_track(token):
        """Test that the tracker works."""

        IBMQ.enable_account(token)
        dev = IBMQSamplerDevice(wires=1, backend="ibmq_qasm_simulator", shots=1)
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


class TestCustomVQE:
    def test_simple_circuit(self):

        return True
