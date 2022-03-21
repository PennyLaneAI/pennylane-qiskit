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

from qiskit import IBMQ

from pennylane_qiskit import opstr_to_meas_circ
from pennylane_qiskit import IBMQCircuitRunnerDevice, IBMQSamplerDevice
from pennylane_qiskit.vqe_runtime_runner import (
    vqe_runner,
    upload_vqe_runner,
    delete_vqe_runner,
    hamiltonian_to_list_string,
)


class TestCircuitRunner:
    """Test class for the circuit runner IBMQ runtime device."""

    def test_load_from_env(self, token, monkeypatch):
        """Test loading an IBMQ Circuit Runner Qiskit runtime device from an env variable."""
        monkeypatch.setenv("IBMQX_TOKEN", token)
        dev = IBMQCircuitRunnerDevice(wires=1)
        assert dev.provider.credentials.is_ibmq()

    def test_short_name(self, token):
        """Test that we can call the circuit runner using its shortname."""
        IBMQ.enable_account(token)
        dev = qml.device("qiskit.ibmq.circuit_runner", wires=1)
        return dev.provider.credentials.is_ibmq()

    @pytest.mark.parametrize("shots", [8000])
    def test_simple_circuit(self, token, tol, shots):
        """Test executing a simple circuit submitted to IBMQ circuit runner runtime program."""
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
        """Test executing a simple circuit submitted to IBMQ  circuit runner runtime program with kwargs."""
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
        """Test that we can send batched circuits to the circuit runner runtime program."""
        IBMQ.enable_account(token)
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
    """Test class for the sampler IBMQ runtime device."""

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
        """Test executing a simple circuit submitted to IBMQ using the Sampler device."""
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
                "circuit_indices": [0],
                "run_options": {"seed_simulator": 42},
                "skip_transpilation": False,
            }
        ],
    )
    def test_kwargs_circuit(self, token, tol, shots, kwargs):
        """Test executing a simple circuit submitted to IBMQ using the Sampler device with kwargs."""
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
        """Test executing batched circuits submitted to IBMQ using the Sampler device."""
        IBMQ.enable_account(token)
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

    def test_hamiltonian_to_list_string(self):
        """Test the function that transforms a PennyLane Hamiltonian to a list string Hamiltonian."""
        coeffs = [1, 1]
        obs = [qml.PauliX(0) @ qml.PauliX(2), qml.Hadamard(0) @ qml.PauliZ(1)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)
        result = hamiltonian_to_list_string(hamiltonian, hamiltonian.wires)

        assert [(1, "XIX"), (1, "HZI")] == result

    def test_op_str_measurement_circ(self):
        """Test that the opstr_to_meas_circ function finds the necessary rotations before measurements in the
        circuit."""
        circ = opstr_to_meas_circ("HIHXZ")
        results = []
        for c in circ:
            if c:
                results.append((c.data[0][0].name, c.data[0][0].params))
            else:
                results.append(())
        assert [
            ("ry", [-0.7853981633974483]),
            (),
            ("ry", [-0.7853981633974483]),
            ("h", []),
            (),
        ] == results

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
    def test_ansatz_qiskit(self, token, tol, shots):
        """Test a simple VQE problem with an ansatz from Qiskit library."""
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
            x0=[0.0, 0.0, 0.0, 0.0],
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
    def test_ansatz_qiskit_invalid(self, token, tol, shots):
        """Test a simple VQE problem with an invalid ansatz from Qiskit library."""
        IBMQ.enable_account(token)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)
        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        with pytest.raises(
            ValueError, match="Ansatz InEfficientSU2 not in n_local circuit library."
        ):
            vqe_runner(
                program_id=program_id,
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

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)

    @pytest.mark.parametrize("shots", [8000])
    def test_qnode(self, token, tol, shots):
        """Test that we cannot pass a QNode as ansatz circuit."""
        IBMQ.enable_account(token)

        with qml.tape.QuantumTape() as vqe_tape:
            qml.RX(3.97507603, wires=0)
            qml.RY(3.00854038, wires=0)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)
        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        with pytest.raises(
            qml.QuantumFunctionError, match="The ansatz must be a callable quantum function."
        ):
            vqe_runner(
                program_id=program_id,
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

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)

    @pytest.mark.parametrize("shots", [8000])
    def test_tape(self, token, tol, shots):
        """Test that we cannot pass a tape as ansatz circuit."""
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

        with pytest.raises(
            qml.QuantumFunctionError, match="The ansatz must be a callable quantum function."
        ):
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

        with pytest.raises(qml.QuantumFunctionError, match="Param should be a single vector."):
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

        with pytest.raises(qml.QuantumFunctionError, match="Not enough parameters in X0."):
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
            qml.RY(params[1], wires=0)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)
        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        with pytest.warns(UserWarning) as record:
            job = vqe_runner(
                program_id=program_id,
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

        assert (
            record[-1].message.args[0]
            == "In order to match the tape expansion, the number of parameters has been changed."
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
        """Test that a PennyLane Hamiltonian is required."""
        IBMQ.enable_account(token)

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)

        hamiltonian = qml.PauliZ(wires=0)
        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        with pytest.raises(
            qml.QuantumFunctionError, match="A PennyLane Hamiltonian object is required."
        ):
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
        """Test that we can handle tensor Hamiltonians."""
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
        """Test the observables in the Hamiltonian are I, X, Y, Z or H."""
        IBMQ.enable_account(token)

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=0)

        H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.Hermitian(H, wires=0)]
        hamiltonian = qml.Hamiltonian(coeffs, obs)

        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        with pytest.raises(qml.QuantumFunctionError, match="Observable is not accepted."):
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
        """Test the observables in the tensor Hamiltonian are I, X, Y, Z or H."""
        IBMQ.enable_account(token)

        def vqe_circuit(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)

        H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        coeffs = [1, 1]
        obs = [qml.PauliX(0) @ qml.Hermitian(H, wires=1), qml.PauliZ(wires=1)]
        hamiltonian = qml.Hamiltonian(coeffs, obs)

        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        with pytest.raises(qml.QuantumFunctionError, match="Observable is not accepted."):
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
    def test_scipy_optimizer(self, token, tol, shots):
        """Test we can run a VQE problem with a SciPy optimizer."""
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
            optimizer="Powell",
            optimizer_config={"maxiter": 10},
            kwargs={"hub": "ibm-q-startup", "group": "ibm-q-startup", "project": "reservations"},
        )

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)
        result = job.result()["fun"]

        assert np.allclose(result, -1.43, tol)
        assert "parameters" in job.intermediate_results

    @pytest.mark.parametrize("shots", [8000])
    def test_simple_hamiltonian_with_untrainable_parameters(self, token, tol, shots):
        """Test a simple VQE problem with untrainable parameters."""
        IBMQ.enable_account(token)
        tol = 1e-1

        def vqe_circuit(params):
            qml.RZ(0.1, wires=0)
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            qml.RZ(0.2, wires=0)

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
    def test_invalid_function(self, token, tol, shots):
        """Test that an invalid function cannot be passed."""
        IBMQ.enable_account(token)

        def vqe_circuit(params):
            c = params[0] + params[1]

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)
        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        with pytest.raises(
            qml.QuantumFunctionError, match="Function contains no quantum operations."
        ):
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
    def test_invalid_ansatz(self, token, tol, shots):
        """Test that an invalid ansatz cannot be passed."""
        IBMQ.enable_account(token)

        coeffs = [1, 1]
        obs = [qml.PauliX(0), qml.PauliZ(0)]

        hamiltonian = qml.Hamiltonian(coeffs, obs)
        program_id = upload_vqe_runner(hub="ibm-q-startup", group="xanadu", project="reservations")

        with pytest.raises(ValueError, match="Input ansatz is not a quantum function or a string."):
            vqe_runner(
                program_id=program_id,
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

        provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")
        delete_vqe_runner(provider=provider, program_id=program_id)
