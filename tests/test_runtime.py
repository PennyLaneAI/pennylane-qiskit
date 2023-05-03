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
This module contains tests for PennyLane runtime programs.
"""

import numpy as np
import pennylane as qml
import pytest

from pennylane_qiskit import IBMQCircuitRunnerDevice, IBMQSamplerDevice
from pennylane_qiskit.vqe_runtime_runner import vqe_runner, hamiltonian_to_list_string


@pytest.mark.usefixtures("skip_if_no_account")
class TestCircuitRunner:
    """Test class for the circuit runner IBMQ runtime device."""

    def test_short_name(self):
        """Test that we can call the circuit runner using its shortname."""
        dev = qml.device("qiskit.ibmq.circuit_runner", wires=1)
        assert isinstance(dev, IBMQCircuitRunnerDevice)

    @pytest.mark.parametrize("shots", [8000])
    def test_simple_circuit(self, tol, shots):
        """Test executing a simple circuit submitted to IBMQ circuit runner runtime program."""

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
    def test_kwargs_circuit(self, tol, shots, kwargs):
        """Test executing a simple circuit submitted to IBMQ  circuit runner runtime program with kwargs."""

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
    def test_batch_circuits(self, tol, shots):
        """Test that we can send batched circuits to the circuit runner runtime program."""

        dev = IBMQCircuitRunnerDevice(wires=2, backend="ibmq_qasm_simulator", shots=shots)

        # Batch the input parameters
        batch_dim = 3
        a = np.linspace(0, 0.543, batch_dim)
        b = np.linspace(0, 0.123, batch_dim)
        c = np.linspace(0, 0.987, batch_dim)

        @qml.batch_params(all_operations=True)
        @qml.qnode(dev)
        def circuit(x, y, z):
            """Reference QNode"""
            qml.PauliX(0)
            qml.Hadamard(wires=0)
            qml.Rot(x, y, z, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(a, b, c), np.cos(a) * np.sin(b), **tol)

    def test_track_circuit_runner(self):
        """Test that the tracker works."""

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


@pytest.mark.usefixtures("skip_if_no_account")
class TestSampler:
    """Test class for the sampler IBMQ runtime device."""

    def test_short_name(self):
        dev = qml.device("qiskit.ibmq.sampler", wires=1)
        assert isinstance(dev, IBMQSamplerDevice)

    @pytest.mark.parametrize("shots", [8000])
    def test_simple_circuit(self, tol, shots):
        """Test executing a simple circuit submitted to IBMQ using the Sampler device."""

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
                "circuit_indices": [0],
                "run_options": {"seed_simulator": 42},
                "skip_transpilation": False,
            }
        ],
    )
    def test_kwargs_circuit(self, tol, shots, kwargs):
        """Test executing a simple circuit submitted to IBMQ using the Sampler device with kwargs."""

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
    def test_batch_circuits(self, tol, shots):
        """Test executing batched circuits submitted to IBMQ using the Sampler device."""

        dev = IBMQSamplerDevice(wires=1, backend="ibmq_qasm_simulator", shots=shots)

        # Batch the input parameters
        batch_dim = 3
        a = np.linspace(0, 0.543, batch_dim)
        b = np.linspace(0, 0.123, batch_dim)
        c = np.linspace(0, 0.987, batch_dim)

        @qml.batch_params(all_operations=True)
        @qml.qnode(dev)
        def circuit(x, y, z):
            """Reference QNode"""
            qml.PauliX(0)
            qml.Hadamard(wires=0)
            qml.Rot(x, y, z, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(a, b, c), np.cos(a) * np.sin(b), **tol)

    def test_track_sampler(self):
        """Test that the tracker works."""

        dev = IBMQSamplerDevice(wires=1, backend="ibmq_qasm_simulator", shots=1)
        dev.tracker.active = True

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            return qml.probs(wires=0)

        circuit()

        assert len(dev.tracker.history) == 2


@pytest.mark.usefixtures("skip_if_no_account")
class TestCustomVQE:
    """Class to test the custom VQE program."""

    @pytest.mark.parametrize("shots", [8000])
    def test_simple_hamiltonian(self, tol, shots):
        """Test a simple VQE problem with Hamiltonian and a circuit from PennyLane"""

        tol = 1e-1

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        job = vqe_runner(
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz=vqe_circuit,
            x0=[3.97507603, 3.00854038],
            shots=shots,
            optimizer="SPSA",
            optimizer_config={"maxiter": 40},
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        assert np.allclose(job.result()["optimal_value"], -1.43, tol)
        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_ansatz_qiskit(self, shots):
        """Test a simple VQE problem with an ansatz from Qiskit library."""

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        job = vqe_runner(
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz="EfficientSU2",
            x0=[0.0, 0.0, 0.0, 0.0],
            shots=shots,
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_ansatz_qiskit_invalid(self, shots):
        """Test a simple VQE problem with an invalid ansatz from Qiskit library."""

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        with pytest.raises(
            ValueError, match="Ansatz InEfficientSU2 not in n_local circuit library."
        ):
            vqe_runner(
                backend="ibmq_qasm_simulator",
                hamiltonian=hamiltonian,
                ansatz="InEfficientSU2",
                x0=[3.97507603, 3.00854038],
                shots=shots,
                kwargs={
                    "hub": "ibm-q-startup",
                    "group": "ibm-q-startup",
                    "project": "reservations",
                },
            )

    @pytest.mark.parametrize("shots", [8000])
    def test_qnode(self, shots):
        """Test that we cannot pass a QNode as ansatz circuit."""

        with qml.tape.QuantumTape() as vqe_tape:
            qml.RX(3.97507603, wires=0)
            qml.RY(3.00854038, wires=0)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        with pytest.raises(
            qml.QuantumFunctionError, match="The ansatz must be a callable quantum function."
        ):
            vqe_runner(
                backend="ibmq_qasm_simulator",
                hamiltonian=hamiltonian,
                ansatz=vqe_tape,
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

    @pytest.mark.parametrize("shots", [8000])
    def test_tape(self, shots):
        """Test that we cannot pass a tape as ansatz circuit."""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        with pytest.raises(
            qml.QuantumFunctionError, match="The ansatz must be a callable quantum function."
        ):
            vqe_runner(
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

    @pytest.mark.parametrize("shots", [8000])
    def test_wrong_input(self, shots):
        """Test that we can only give a single vector parameter to the ansatz circuit."""

        def vqe_circuit(params, wire):
            qml.RX(params[0], wires=wire)
            qml.RY(params[1], wires=wire)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        with pytest.raises(qml.QuantumFunctionError, match="Param should be a single vector."):
            vqe_runner(
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

    @pytest.mark.parametrize("shots", [8000])
    def test_wrong_number_input_param(self, shots):
        """Test that we need a certain number of parameters."""

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            qml.RY(params[2], wires=0)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        with pytest.raises(qml.QuantumFunctionError, match="Not enough parameters in X0."):
            vqe_runner(
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

    @pytest.mark.parametrize("shots", [8000])
    def test_one_param(self, shots):
        """Test that we can only give a single vector parameter to the ansatz circuit."""

        def vqe_circuit(params):
            qml.RX(params, wires=0)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        job = vqe_runner(
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz=vqe_circuit,
            x0=[0.0],
            shots=shots,
            optimizer_config={"maxiter": 10},
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_too_many_param(self, shots):
        """Test that we handle the case where too many parameters were given."""

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        with pytest.warns(
            UserWarning,
            match="In order to match the tape expansion, the number of parameters has been changed.",
        ):
            job = vqe_runner(
                backend="ibmq_qasm_simulator",
                hamiltonian=hamiltonian,
                ansatz=vqe_circuit,
                x0=[3.97507603, 3.00854038, 3.55637849],
                shots=shots,
                optimizer_config={"maxiter": 10},
                kwargs={
                    "hub": "ibm-q-startup",
                    "group": "ibm-q-startup",
                    "project": "reservations",
                },
            )

        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_more_qubits_in_circuit_than_hamiltonian(self, shots):
        """Test that we handle the case where there are more qubits in the circuit than the hamiltonian."""

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        job = vqe_runner(
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz=vqe_circuit,
            x0=[3.97507603, 3.00854038],
            shots=shots,
            optimizer_config={"maxiter": 10},
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_qubitunitary(self, shots):
        """Test that we can handle a QubitUnitary operation."""

        def vqe_circuit(params):
            qml.QubitUnitary(np.array([[1, 0], [0, 1]]), wires=0)
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        job = vqe_runner(
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz=vqe_circuit,
            x0=[3.97507603, 3.00854038],
            shots=shots,
            optimizer_config={"maxiter": 10},
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_inverse(self, shots):
        """Test that we can handle inverse operations."""

        def vqe_circuit(params):
            qml.adjoint(qml.RX(params[0], wires=0))
            qml.RX(params[1], wires=1)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        job = vqe_runner(
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz=vqe_circuit,
            x0=[3.97507603, 3.00854038],
            shots=shots,
            optimizer_config={"maxiter": 10},
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_hamiltonian_tensor(self, shots):
        """Test that we can handle tensor Hamiltonians."""

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)

        coeffs = [0.2, -0.543]
        obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliY(1)]
        hamiltonian = qml.Hamiltonian(coeffs, obs)

        job = vqe_runner(
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz=vqe_circuit,
            x0=[3.97507603, 3.00854038],
            shots=shots,
            optimizer_config={"maxiter": 10},
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results

    @pytest.mark.parametrize(
        "bad_op", [qml.Hermitian(np.array([[1, 0], [0, -1]]), wires=0), qml.Hadamard(1)]
    )
    @pytest.mark.parametrize("shots", [8000])
    def test_not_auth_operation_hamiltonian(self, shots, bad_op):
        """Test the observables in the Hamiltonian are I, X, Y, or Z."""

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=0)

        H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        coeffs = [1, 1]
        obs = [qml.PauliX(0), bad_op]
        hamiltonian = qml.Hamiltonian(coeffs, obs)

        with pytest.raises(qml.QuantumFunctionError, match="Observable is not accepted."):
            vqe_runner(
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

    @pytest.mark.parametrize("shots", [8000])
    def test_not_auth_operation_hamiltonian_tensor(self, shots):
        """Test the observables in the tensor Hamiltonian are I, X, Y, or Z."""

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)

        H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        coeffs = [1, 1]
        obs = [qml.PauliX(0) @ qml.Hermitian(H, wires=1), qml.PauliZ(wires=1)]
        hamiltonian = qml.Hamiltonian(coeffs, obs)

        with pytest.raises(qml.QuantumFunctionError, match="Observable is not accepted."):
            vqe_runner(
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

    @pytest.mark.parametrize("shots", [8000])
    def test_scipy_optimizer(self, tol, shots):
        """Test we can run a VQE problem with a SciPy optimizer."""

        tol = 1e-1

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        job = vqe_runner(
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz=vqe_circuit,
            x0=[3.97507603, 3.00854038],
            shots=shots,
            optimizer="Powell",
            optimizer_config={"maxiter": 10},
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        result = job.result()["optimal_value"]

        assert np.allclose(result, -1.43, tol)
        assert "parameters" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_scipy_optimizer(self, shots):
        """Test we can run a VQE problem with a SciPy optimizer."""

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        job = vqe_runner(
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz=vqe_circuit,
            x0=[3.97507603, 3.00854038],
            shots=shots,
            optimizer="COBYLA",
            optimizer_config={"maxiter": 10},
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        result = job.result()["optimal_value"]

        assert "parameters" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_simple_hamiltonian_with_untrainable_parameters(self, tol, shots):
        """Test a simple VQE problem with untrainable parameters."""

        tol = 1e-1

        def vqe_circuit(params):
            qml.RZ(0.1, wires=0)
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            qml.RZ(0.2, wires=0)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        job = vqe_runner(
            backend="ibmq_qasm_simulator",
            hamiltonian=hamiltonian,
            ansatz=vqe_circuit,
            x0=[3.97507603, 3.00854038],
            shots=shots,
            optimizer="SPSA",
            optimizer_config={"maxiter": 40},
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        assert np.allclose(job.result()["optimal_value"], -1.43, tol)
        assert isinstance(job.intermediate_results, dict)
        assert "nfev" in job.intermediate_results
        assert "parameters" in job.intermediate_results
        assert "function" in job.intermediate_results
        assert "step" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_invalid_function(self, shots):
        """Test that an invalid function cannot be passed."""

        def vqe_circuit(params):
            c = params[0] + params[1]

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        with pytest.raises(
            qml.QuantumFunctionError, match="Function contains no quantum operations."
        ):
            vqe_runner(
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

    @pytest.mark.parametrize("shots", [8000])
    def test_invalid_ansatz(self, shots):
        """Test that an invalid ansatz cannot be passed."""

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        with pytest.raises(ValueError, match="Input ansatz is not a quantum function or a string."):
            vqe_runner(
                backend="ibmq_qasm_simulator",
                hamiltonian=hamiltonian,
                ansatz=10,
                x0=[3.97507603, 3.00854038],
                shots=shots,
                optimizer_config={"maxiter": 10},
                kwargs={
                    "hub": "ibm-q-startup",
                    "group": "ibm-q-startup",
                    "project": "reservations",
                },
            )

    def test_qnspsa_disabled(self):
        """Tests that QNSPSA is rejected before launching a job."""
        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)

        with pytest.raises(ValueError, match="QNSPSA is not available for vqe_runner"):
            vqe_runner(
                backend="ibmq_qasm_simulator",
                hamiltonian=hamiltonian,
                x0=[3.97507603, 3.00854038],
                optimizer="QNSPSA",
            )


def test_hamiltonian_to_list_string():
    """Test the function that transforms a PennyLane Hamiltonian to a list string Hamiltonian."""
    coeffs = [1, 1]
    obs = [qml.PauliX(0) @ qml.PauliX(2), qml.PauliY(0) @ qml.PauliZ(1)]

    hamiltonian = qml.Hamiltonian(coeffs, obs)
    result = hamiltonian_to_list_string(hamiltonian, hamiltonian.wires)

    assert [("XIX", 1), ("YZI", 1)] == result


@pytest.mark.parametrize("shots", [8000])
def test_hamiltonian_format(shots):
    """Test that a PennyLane Hamiltonian is required."""

    def vqe_circuit(params):
        qml.RX(params[0], wires=0)
        qml.RX(params[1], wires=1)

    hamiltonian = qml.PauliZ(wires=0)

    with pytest.raises(
        qml.QuantumFunctionError, match="A PennyLane Hamiltonian object is required."
    ):
        vqe_runner(
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
