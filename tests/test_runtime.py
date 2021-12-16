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
        dev = qml.device("qiskit.ibmq.circuit_runner", wires=1)
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

    @pytest.mark.parametrize("shots", [8000])
    def test_simple_hamiltonian(self, token, tol, shots):
        """Test a simple VQE problem with Hamiltonian and a circuit from PennyLane"""
        IBMQ.enable_account(token)
        tol = 1e-1

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)
        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        job = vqe_runner(
            program_id=program_id,
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz=vqe_circuit,
            x0=[3.97507603, 3.00854038],
            shots=shots,
            optimizer="SPSA",
            optimizer_config={"maxiter": 40},
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)

        assert np.allclose(job.result()["fun"], -1.43, tol)
        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results
        assert "accepted" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_simple_hamiltonian_scipy(self, token, tol, shots):
        """Test a simple VQE problem with Hamiltonian and a circuit from PennyLane."""
        IBMQ.enable_account(token)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)
        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        job = vqe_runner(
            program_id=program_id,
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz="EfficientSU2",
            x0=[3.97507603, 3.00854038],
            shots=shots,
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)

        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results
        assert "accepted" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_qnode(self, token, tol, shots):
        """Test that we cannot pass a Qnode as ansatz circuit"""
        IBMQ.enable_account(token)

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)
        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        with pytest.raises(qml.QuantumFunctionError, match="Must be a callable quantum function."):
            vqe_runner(
                program_id=program_id,
                backend="ibmq_qasm_simulator",
                hamiltonian=hamiltonian,
                ansatz=vqe_circuit,
                x0=[3.97507603, 3.00854038],
                shots=shots,
                optimizer="SPSA",
                optimizer_config={"maxiter": 40},
                kwargs={
                    "hub": "ibm-q-startup",
                    "group": "ibm-q-startup",
                    "project": "reservations",
                },
            )

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)

    @pytest.mark.parametrize("shots", [8000])
    def test_wrong_input(self, token, tol, shots):
        """Test that we can only give a single vector parameter to the ansatz circuit."""
        IBMQ.enable_account(token)

        def vqe_circuit(params, wire):
            qml.RX(params[0], wires=wire)
            qml.RY(params[1], wires=wire)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)
        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        with pytest.raises(qml.QuantumFunctionError, match="Param should be a single vector"):
            vqe_runner(
                program_id=program_id,
                backend="ibmq_qasm_simulator",
                hamiltonian=hamiltonian,
                ansatz=vqe_circuit,
                x0=[3.97507603, 3.00854038],
                shots=shots,
                optimizer="SPSA",
                optimizer_config={"maxiter": 40},
                kwargs={
                    "hub": "ibm-q-startup",
                    "group": "ibm-q-startup",
                    "project": "reservations",
                },
            )

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)

    @pytest.mark.parametrize("shots", [8000])
    def test_wrong_number_input_param(self, token, tol, shots):
        """Test that we need a certain number of parameters."""
        IBMQ.enable_account(token)

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            qml.RY(params[2], wires=0)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)
        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        with pytest.raises(qml.QuantumFunctionError, match="X0 has not enough parameters"):
            vqe_runner(
                program_id=program_id,
                backend="ibmq_qasm_simulator",
                hamiltonian=hamiltonian,
                ansatz=vqe_circuit,
                x0=[0, 0],
                shots=shots,
                optimizer="SPSA",
                optimizer_config={"maxiter": 40},
                kwargs={
                    "hub": "ibm-q-startup",
                    "group": "ibm-q-startup",
                    "project": "reservations",
                },
            )

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)

    @pytest.mark.parametrize("shots", [8000])
    def test_one_param(self, token, tol, shots):
        """Test that we can only give a single vector parameter to the ansatz circuit."""
        IBMQ.enable_account(token)

        def vqe_circuit(params):
            qml.RX(params, wires=0)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)
        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        job = vqe_runner(
            program_id=program_id,
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz=vqe_circuit,
            x0=[0.0],
            shots=shots,
            optimizer_config={"maxiter": 10},
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)

        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results
        assert "accepted" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_too_many_param(self, token, tol, shots):
        """Test that we handle the case where too many parameters were given."""
        IBMQ.enable_account(token)

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=0)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)
        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        job = vqe_runner(
            program_id=program_id,
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz=vqe_circuit,
            x0=[3.97507603, 3.00854038, 3.55637849],
            shots=shots,
            optimizer_config={"maxiter": 10},
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)

        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results
        assert "accepted" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_more_qubits_in_circuit_than_hamiltonian(self, token, tol, shots):
        """Test that we handle the case where there are more qubits in the circuit than the hamiltonian."""
        IBMQ.enable_account(token)

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)
        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        job = vqe_runner(
            program_id=program_id,
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz=vqe_circuit,
            x0=[3.97507603, 3.00854038],
            shots=shots,
            optimizer_config={"maxiter": 10},
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)

        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results
        assert "accepted" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_qubitunitary(self, token, tol, shots):
        """Test that we can handle a QubitUnitary operation."""
        IBMQ.enable_account(token)

        def vqe_circuit(params):
            qml.QubitUnitary(np.array([[1, 0], [0, 1]]), wires=0)
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)
        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        job = vqe_runner(
            program_id=program_id,
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz=vqe_circuit,
            x0=[3.97507603, 3.00854038],
            shots=shots,
            optimizer_config={"maxiter": 10},
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)

        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results
        assert "accepted" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_inverse(self, token, tol, shots):
        """Test that we can handle inverse operations."""
        IBMQ.enable_account(token)

        def vqe_circuit(params):
            qml.RX(params[0], wires=0).inv()
            qml.RX(params[1], wires=1)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)
        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        job = vqe_runner(
            program_id=program_id,
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz=vqe_circuit,
            x0=[3.97507603, 3.00854038],
            shots=shots,
            optimizer_config={"maxiter": 10},
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)

        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results
        assert "accepted" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_hamiltonian_format(self, token, tol, shots):
        """Test that we can handle inverse operations."""
        IBMQ.enable_account(token)

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)

        hamiltonian = qml.PauliZ(wires=0)
        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        with pytest.raises(qml.QuantumFunctionError, match="Hamiltonian required."):
            vqe_runner(
                program_id=program_id,
                backend="ibmq_qasm_simulator",
                hamiltonian=hamiltonian,
                ansatz=vqe_circuit,
                x0=[3.97507603, 3.00854038],
                shots=shots,
                optimizer_config={"maxiter": 10},
                kwargs={
                    "hub": "ibm-q-startup",
                    "group": "ibm-q-startup",
                    "project": "reservations",
                },
            )

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)

    @pytest.mark.parametrize("shots", [8000])
    def test_hamiltonian_tensor(self, token, tol, shots):
        """Test that we can handle tensor hamiltonians."""
        IBMQ.enable_account(token)

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)

        coeffs = [0.2, -0.543]
        obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(1)]
        hamiltonian = qml.Hamiltonian(coeffs, obs)

        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        job = vqe_runner(
            program_id=program_id,
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz=vqe_circuit,
            x0=[3.97507603, 3.00854038],
            shots=shots,
            optimizer_config={"maxiter": 10},
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)

        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results
        assert "accepted" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_not_auth_operation_hamiltonian(self, token, tol, shots):
        """Test that we can handle inverse operations."""
        IBMQ.enable_account(token)

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=0)

        H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.Hermitian(H, wires=0)]
        hamiltonian = qml.Hamiltonian(coeffs, obs)

        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        with pytest.raises(qml.QuantumFunctionError, match="Obs not accepted"):
            vqe_runner(
                program_id=program_id,
                backend="ibmq_qasm_simulator",
                hamiltonian=hamiltonian,
                ansatz=vqe_circuit,
                x0=[3.97507603, 3.00854038],
                shots=shots,
                optimizer_config={"maxiter": 10},
                kwargs={
                    "hub": "ibm-q-startup",
                    "group": "ibm-q-startup",
                    "project": "reservations",
                },
            )

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)

    @pytest.mark.parametrize("shots", [8000])
    def test_not_auth_operation_hamiltonian_tensor(self, token, tol, shots):
        """Test that we can handle inverse operations."""
        IBMQ.enable_account(token)

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)

        H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        coeffs = [1, 1]
        obs = [qml.PauliX(0) @ qml.Hermitian(H, wires=1)]
        hamiltonian = qml.Hamiltonian(coeffs, obs)

        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        with pytest.raises(qml.QuantumFunctionError, match="Obs not accepted"):
            vqe_runner(
                program_id=program_id,
                backend="ibmq_qasm_simulator",
                hamiltonian=hamiltonian,
                ansatz=vqe_circuit,
                x0=[3.97507603, 3.00854038],
                shots=shots,
                optimizer_config={"maxiter": 10},
                kwargs={
                    "hub": "ibm-q-startup",
                    "group": "ibm-q-startup",
                    "project": "reservations",
                },
            )

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)
