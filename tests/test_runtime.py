import os

import numpy as np
import pennylane as qml
import pytest

from qiskit import IBMQ

from pennylane_qiskit import IBMQCircuitRunnerDevice
from pennylane_qiskit import IBMQSamplerDevice
from pennylane_qiskit.vqe.vqe_runner import vqe_runner, upload_vqe_runner, delete_vqe_runner


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

    def test_track_circuit_runner(self, token):
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
            assert "total_time" in dev.tracker.history["job_time"][0]
            assert len(dev.tracker.history["job_time"][0]) == 1


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

    def test_track_sampler(self, token):
        """Test that the tracker works."""

        IBMQ.enable_account(token)
        dev = IBMQSamplerDevice(wires=1, backend="ibmq_qasm_simulator", shots=1)
        dev.tracker.active = True

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            return qml.probs(wires=0)

        circuit()

        assert len(dev.tracker.history) == 2


class TestCustomVQE:
    """Class to test the custom VQE program."""

    def test_simple_hamiltonian(self, token, tol):
        """Test a simple VQE problem with Hamiltonian and a circuit from PennyLane"""
        IBMQ.enable_account(token)
        tol = 1e-3

        def vqe_circuit(params, wires=0):
            qml.RX(params[0], wires=wires)
            qml.RY(params[1], wires=wires)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        program_id = upload_vqe_runner(hub='ibm-q-startup', group='xanadu', project='reservations')

        job = vqe_runner(program_id=program_id, backend="ibmq_qasm_simulator",
                         hamiltonian=hamiltonian, ansatz=vqe_circuit, x0=[3.97507603, 3.00854038],
                         optimizer="SPSA", optimizer_config={"maxiter": 20},
                         kwargs={'hub': 'ibm-q-startup', 'group': 'ibm-q-startup', 'project': 'reservations'})

        delete_vqe_runner(program_id=program_id)

        assert np.allclose(job.result()['fun'], -1.413, tol)
        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results
        assert "accepted" in job.intermediate_results
