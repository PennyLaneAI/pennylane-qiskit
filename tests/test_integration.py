# Copyright 2021-2024 Xanadu Quantum Technologies Inc.

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
This module contains integration tests for PennyLane IBMQ devices.
"""
import sys

from functools import partial
import numpy as np

from conftest import state_backends

import pennylane as qml
from pennylane.numpy import tensor
import pytest
import qiskit
import qiskit_aer

from qiskit.providers import QiskitBackendNotFoundError
from qiskit.providers.basic_provider import BasicProvider
from pennylane_qiskit.qiskit_device_legacy import QiskitDeviceLegacy

# pylint: disable=protected-access, unused-argument, ungrouped-imports, too-many-arguments, too-few-public-methods


pldevices = [("qiskit.aer", qiskit_aer.Aer), ("qiskit.basicsim", BasicProvider())]


def check_provider_backend_compatibility(pldevice, backend_name):
    """Check the compatibility of provided backend"""
    dev_name, _ = pldevice
    if dev_name == "qiskit.aer" and backend_name == "basic_simulator":
        return (False, "basic_simulator is not supported on the AerDevice")

    if dev_name == "qiskit.basicsim" and backend_name != "basic_simulator":
        return (False, "Only the basic_simulator backend works with the BasicSimulatorDevice")
    return True, None


class TestDeviceIntegration:
    """Test the devices work correctly from the PennyLane frontend."""

    @pytest.mark.parametrize("d", pldevices)
    def test_load_device(self, d, backend):
        """Test that the qiskit device loads correctly"""

        # check compatibility between provider and backend, and skip if incompatible
        is_compatible, failure_msg = check_provider_backend_compatibility(d, backend)
        if not is_compatible:
            pytest.skip(failure_msg)

        dev = qml.device(d[0], wires=2, backend=backend, shots=1024)
        assert dev.num_wires == 2
        assert dev.shots.total_shots == 1024
        assert dev.short_name == d[0]
        assert dev.provider == d[1]

    @pytest.mark.parametrize("d", pldevices)
    def test_load_remote_device_with_backend_instance(self, d, backend):
        """Test that the qiskit.remote device loads correctly when passed a backend instance."""
        _, provider = d

        try:
            backend_instance = provider.get_backend(backend)
        except QiskitBackendNotFoundError:
            pytest.skip("Backend is not compatible with specified device")

        if backend_instance.configuration().n_qubits is None:
            pytest.skip("No qubits?")

        dev = qml.device(
            "qiskit.remote",
            wires=backend_instance.configuration().n_qubits,
            backend=backend_instance,
            shots=1024,
        )
        assert dev.num_wires == backend_instance.configuration().n_qubits
        assert dev.shots.total_shots == 1024
        assert dev.short_name == "qiskit.remote"

    def test_incorrect_backend(self):
        """Test that exception is raised if name is incorrect"""
        with pytest.raises(ValueError, match="Backend 'none' does not exist"):
            qml.device("qiskit.aer", wires=2, backend="none")

    def test_incorrect_backend_wires(self):
        """Test that exception is raised if number of wires is too large"""
        with pytest.raises(
            ValueError, match=r"Backend 'aer_simulator\_statevector' supports maximum"
        ):
            qml.device("qiskit.aer", wires=100, method="statevector")

    def test_args(self):
        """Test that the device requires correct arguments"""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qml.device("qiskit.aer")

        with pytest.raises(
            qml.DeviceError, match="specified number of shots needs to be at least 1"
        ):
            qml.device("qiskit.aer", wires=1, shots=0)

    @pytest.mark.parametrize("d", pldevices)
    @pytest.mark.parametrize("shots", [None, 8192])
    def test_one_qubit_circuit(self, shots, d, backend, tol):
        """Test that devices provide correct result for a simple circuit"""

        # check compatibility between provider and backend, and skip if incompatible
        is_compatible, failure_msg = check_provider_backend_compatibility(d, backend)
        if not is_compatible:
            pytest.skip(failure_msg)

        if backend not in state_backends and shots is None:
            pytest.skip("Hardware simulators do not support analytic mode")

        dev = qml.device(d[0], wires=1, backend=backend, shots=shots)

        a = 0.543
        b = 0.123
        c = 0.987

        @qml.qnode(dev)
        def circuit(x, y, z):
            """Reference QNode"""
            qml.BasisState(np.array([1]), wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(x, y, z, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(a, b, c), np.cos(a) * np.sin(b), **tol)

    @pytest.mark.parametrize("d", pldevices)
    @pytest.mark.parametrize("shots", [8192])
    def test_basis_state_and_rot(self, shots, d, backend, tol):
        """Integration test for the BasisState and Rot operations for non-analytic mode."""

        # check compatibility between provider and backend, and skip if incompatible
        is_compatible, failure_msg = check_provider_backend_compatibility(d, backend)
        if not is_compatible:
            pytest.skip(failure_msg)

        dev = qml.device(d[0], wires=1, backend=backend, shots=shots)

        a = 0
        b = 0
        c = np.pi
        expected = 1

        @qml.qnode(dev)
        def circuit(x, y, z):
            """Reference QNode"""
            qml.BasisState(np.array([0]), wires=0)
            qml.Rot(x, y, z, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(a, b, c), expected, **tol)

    def test_gradient_for_tensor_product(self):
        """Test that the gradient of a circuit containing a tensor product is
        computed without any errors."""
        n_qubits = 2
        depth = 2

        def ansatz(weights):
            weights = weights.reshape(depth, n_qubits)
            qml.RX(weights[0][0], wires=[0])
            qml.RZ(weights[0][1], wires=[0])
            qml.RX(weights[1][0], wires=[0])
            qml.RZ(weights[1][1], wires=[0])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        dev_qsk = qml.device(
            "qiskit.aer",
            wires=n_qubits,
            shots=1000,
            backend="qasm_simulator",
        )

        weights = np.random.random((depth, n_qubits)).flatten()

        # Want to get expectation value and gradient
        exp_sampled = qml.QNode(ansatz, dev_qsk, diff_method="parameter-shift")
        grad_shift = qml.grad(exp_sampled, argnum=0)
        exp_sampled(weights)
        grad_shift(weights)


class TestKeywordArguments:
    """Test keyword argument logic is correct"""

    @pytest.mark.parametrize("d", pldevices)
    def test_compile_backend(self, d):
        """Test that the compile backend argument is properly
        extracted"""
        dev = qml.device(d[0], wires=2, compile_backend="test value")
        assert dev.compile_backend == "test value"

    def test_noise_model_qasm_simulator(self, monkeypatch):
        """Test that the noise model argument is properly
        extracted if the backend supports it"""

        cache = []
        with monkeypatch.context() as m:
            m.setattr(
                qiskit_aer.AerSimulator, "set_options", lambda *args, **kwargs: cache.append(kwargs)
            )
            qml.device("qiskit.aer", wires=2, noise_model="test value")
        assert cache[-1] == {"noise_model": "test value"}

    def test_invalid_noise_model(self):
        """Test that the noise model argument causes an exception to be raised
        if the backend does not support it"""
        dev_name = pldevices[1][0]
        with pytest.raises(AttributeError, match="field noise_model is not valid for this backend"):
            qml.device(dev_name, wires=2, noise_model="test value")

    def test_overflow_kwargs(self):
        """Test all overflow kwargs are extracted for the AerDevice"""
        dev = qml.device("qiskit.aer", wires=2, k1="v1", k2="v2")
        assert dev.run_args["k1"] == "v1"
        assert dev.run_args["k2"] == "v2"


class TestLoadIntegration:
    """Integration tests for the PennyLane load function. This test ensures that the PennyLane-Qiskit
    specific load functions integrate properly with the PennyLane-Qiskit plugin."""

    # pylint: disable=implicit-str-concat
    hadamard_qasm = "OPENQASM 2.0;" 'include "qelib1.inc";' "qreg q[1];" "h q[0];"

    def test_load_qiskit_circuit(self):
        """Test that the default load function works correctly."""
        theta = qiskit.circuit.Parameter("θ")

        qc = qiskit.QuantumCircuit(2)
        qc.rx(theta, 0)

        my_template = qml.from_qiskit(qc)

        dev = qml.device("default.qubit", wires=2)

        angles = np.array([0.53896774, 0.79503606, 0.27826503, 0.0])

        @qml.qnode(dev)
        def loaded_quantum_circuit(angle):
            my_template({theta: angle})
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def quantum_circuit(angle):
            qml.RX(angle, wires=[0])
            return qml.expval(qml.PauliZ(0))

        for x in angles:
            assert np.allclose(loaded_quantum_circuit(x), quantum_circuit(x))

    def test_load_from_qasm_string(self):
        """Test that quantum circuits can be loaded from a qasm string."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def loaded_quantum_circuit():
            qml.from_qasm(TestLoadIntegration.hadamard_qasm, measurements=[])(wires=[0])
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def quantum_circuit():
            qml.Hadamard(wires=[0])
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(loaded_quantum_circuit(), quantum_circuit())

    @pytest.mark.skipif(sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6")
    def test_load_qasm_from_file(self, tmpdir):
        """Test that quantum circuits can be loaded from a qasm file."""
        apply_hadamard = tmpdir.join("hadamard.qasm")

        with open(apply_hadamard, "w", encoding="utf") as f:
            f.write(TestLoadIntegration.hadamard_qasm)

        with open(apply_hadamard, "r", encoding="utf") as f:
            hadamard = qml.from_qasm(f.read(), measurements=[])

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def loaded_quantum_circuit():
            hadamard(wires=[0])
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def quantum_circuit():
            qml.Hadamard(wires=[0])
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(loaded_quantum_circuit(), quantum_circuit())


class TestPLOperations:
    """Integration tests for checking certain PennyLane specific operations."""

    @pytest.mark.parametrize("shots", [None, 1000])
    def test_rotation(self, init_state, state_vector_device, shots, tol):
        """Test that the StatePrep and Rot operations are decomposed using a
        Qiskit device with statevector backend"""

        dev = state_vector_device(1)

        if dev._is_unitary_backend:
            pytest.skip("Test only runs for backends that are not the unitary simulator.")

        state = init_state(1)

        a = 0.542
        b = 1.3432
        c = -0.654

        I = np.eye(2)
        Y = np.array([[0, -1j], [1j, 0]])  #: Pauli-Y matrix
        Z = np.array([[1, 0], [0, -1]])  #: Pauli-Z matrix

        def ry(theta):
            return np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Y

        def rz(theta):
            return np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Z

        @qml.qnode(dev)
        def qubitstatevector_and_rot():
            qml.StatePrep(state, wires=[0])
            qml.Rot(a, b, c, wires=[0])
            return qml.expval(qml.Identity(0))

        qubitstatevector_and_rot()

        assert np.allclose(
            np.abs(dev.state) ** 2, np.abs(rz(c) @ ry(b) @ rz(a) @ state) ** 2, **tol
        )

    @pytest.mark.parametrize("shots", [None, 1000])
    def test_basisstate(self, init_state, state_vector_device, shots, tol):
        """Test that the Basisstate is decomposed using a Qiskit device with
        statevector backend"""

        dev = state_vector_device(2)
        state = np.array([1, 0])

        @qml.qnode(dev)
        def basisstate():
            qml.BasisState(state, wires=[0, 1])
            return qml.expval(qml.Identity(0))

        basisstate()

        expected_state = np.zeros(2**dev.num_wires)
        expected_state[2] = 1

        assert np.allclose(np.abs(dev.state) ** 2, np.abs(expected_state) ** 2, **tol)

    @pytest.mark.parametrize("shots", [None, 1000])
    def test_basisstate_init_all_zero_states(self, init_state, state_vector_device, shots, tol):
        """Test that the Basisstate that receives the all zero state is decomposed using
        a Qiskit device with statevector backend"""

        dev = state_vector_device(4)
        state = np.array([0, 0, 0, 0])

        @qml.qnode(dev)
        def basisstate():
            qml.BasisState(state, wires=[0, 1, 2, 3])
            return qml.expval(qml.Identity(0))

        basisstate()

        expected_state = np.zeros(2**dev.num_wires)
        expected_state[0] = 1

        assert np.allclose(np.abs(dev.state) ** 2, np.abs(expected_state) ** 2, **tol)

    @pytest.mark.parametrize("shots", [None, 1000])
    def test_adjoint(self, state_vector_device, shots, tol):
        """Test adjoint of an operation using Qiskit device with statevector backend."""
        dev = state_vector_device(1)

        if dev._is_unitary_backend:
            pytest.skip("Test only runs for backends that are not the unitary simulator.")

        x = 1.23

        @qml.qnode(dev)
        def rotate_back_and_forth():
            qml.RX(x, 0)
            qml.adjoint(qml.RX(x, 0))
            return qml.expval(qml.PauliZ(0))

        res = rotate_back_and_forth()

        assert np.allclose(res, 1)


class TestPLTemplates:
    """Integration tests for checking certain PennyLane templates."""

    def test_tensor_unwrapped_gradient_no_error(self, monkeypatch):
        """Tests that the gradient calculation of a circuit that contains a
        RandomLayers template taking a PennyLane tensor as differentiable
        argument executes without error.

        The main aim of the test is to check that unwrapping a single element
        tensor does not cause errors.
        """
        dev = qml.device("qiskit.aer", wires=4)

        @qml.qnode(dev)
        def circuit(phi):
            qml.templates.layers.RandomLayers(phi, wires=list(range(4)))
            return qml.expval(qml.PauliZ(0))

        phi = qml.numpy.tensor([[0.04439891, 0.14490549, 3.29725643, 2.51240058]])

        # Check that the jacobian executes without errors
        qml.jacobian(circuit)(phi)

    def test_single_gate_parameter(self, monkeypatch):
        """Test that when supplied a PennyLane tensor, a QNode passes an
        unwrapped tensor as the argument to a gate taking a single parameter"""
        dev = qml.device("qiskit.aer", wires=4)

        @qml.qnode(dev)
        def circuit(phi=None):
            for y in phi:
                for idx, x in enumerate(y):
                    qml.RX(x, wires=idx)
            return qml.expval(qml.PauliZ(0))

        phi = tensor([[0.04439891, 0.14490549, 3.29725643, 2.51240058]])
        circuit(phi)
        ops = circuit.tape.operations
        for i in range(phi.shape[1]):
            # Test each rotation applied
            assert ops[i].name == "RX"
            assert len(ops[i].parameters) == 1

            # Test that the gate parameter is a PennyLane tensor
            assert isinstance(ops[i].parameters[0], tensor)

    def test_multiple_gate_parameter(self):
        """Test that when supplied a PennyLane tensor, a QNode passes arguments
        as unwrapped tensors to a gate taking multiple parameters"""
        dev = qml.device("qiskit.aer", wires=1)

        @qml.qnode(dev)
        def circuit(phi=None):
            for idx, x in enumerate(phi):
                qml.Rot(*x, wires=idx)
            return qml.expval(qml.PauliZ(0))

        phi = tensor([[0.04439891, 0.14490549, 3.29725643]])

        circuit(phi)
        ops = circuit.tape.operations
        # Test the rotation applied
        assert ops[0].name == "Rot"
        assert len(ops[0].parameters) == 3

        # Test that the gate parameters are PennyLane tensors,
        assert isinstance(ops[0].parameters[0], tensor)

        assert isinstance(ops[0].parameters[1], tensor)

        assert isinstance(ops[0].parameters[2], tensor)


class TestInverses:
    """Integration tests checking that the inverse of the operations are applied."""

    def test_inverse_of_operation(self):
        """Test that the inverse of operations works as expected
        by comparing a simple circuit with default.qubit."""
        dev = qml.device("default.qubit", wires=2)

        dev2 = qml.device("qiskit.aer", backend="statevector_simulator", shots=None, wires=2)

        angles = np.array([0.53896774, 0.79503606, 0.27826503, 0.0])

        @qml.qnode(dev)
        def circuit_with_inverses(angle):
            qml.adjoint(qml.Hadamard(0))
            qml.adjoint(qml.RX(angle, wires=0))
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev2)
        def circuit_with_inverses_default_qubit(angle):
            qml.adjoint(qml.Hadamard(0))
            qml.adjoint(qml.RX(angle, wires=0))
            return qml.expval(qml.PauliZ(0))

        for x in angles:
            assert np.allclose(circuit_with_inverses(x), circuit_with_inverses_default_qubit(x))


class TestNoise:
    """Integration test for the noise models."""

    def test_noise_applied(self):
        """Test that the qiskit noise model is applied correctly"""
        noise_model = qiskit_aer.noise.NoiseModel()
        bit_flip = qiskit_aer.noise.pauli_error([("X", 1), ("I", 0)])

        # Create a noise model where the RX operation always flips the bit
        noise_model.add_all_qubit_quantum_error(bit_flip, ["z", "rz"])

        dev = qml.device("qiskit.aer", wires=2, noise_model=noise_model)

        @qml.qnode(dev)
        def circuit():
            qml.PauliZ(wires=[0])
            return qml.expval(qml.PauliZ(wires=0))

        assert circuit() == -1


class TestBatchExecution:
    """Test the devices work correctly with the batch execution pipeline."""

    @pytest.mark.parametrize("d", pldevices)
    @pytest.mark.parametrize("shots", [None, 8192])
    def test_one_qubit_circuit_batch_params(self, shots, d, backend, tol, mocker):
        """Test that devices provide correct result for a simple circuit using
        the batch_params transform."""

        # check compatibility between provider and backend, and skip if incompatible
        is_compatible, failure_msg = check_provider_backend_compatibility(d, backend)
        if not is_compatible:
            pytest.skip(failure_msg)

        if backend not in state_backends and shots is None:
            pytest.skip("Hardware simulators do not support analytic mode")

        dev = qml.device(d[0], wires=1, backend=backend, shots=shots)

        # Batch the input parameters
        batch_dim = 3
        a = np.linspace(0, 0.543, batch_dim)
        b = np.linspace(0, 0.123, batch_dim)
        c = np.linspace(0, 0.987, batch_dim)

        spy1 = mocker.spy(QiskitDeviceLegacy, "batch_execute")
        spy2 = mocker.spy(dev.backend, "run")

        @partial(qml.batch_params, all_operations=True)
        @qml.qnode(dev)
        def circuit(x, y, z):
            """Reference QNode"""
            qml.PauliX(0)
            qml.Hadamard(wires=0)
            qml.Rot(x, y, z, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(a, b, c), np.cos(a) * np.sin(b), **tol)

        # Check that QiskitDeviceLegacy.batch_execute was called
        assert spy1.call_count == 1
        assert spy2.call_count == 1

    @pytest.mark.parametrize("d", pldevices)
    @pytest.mark.parametrize("shots", [None, 8192])
    def test_batch_execute_parameter_shift(self, shots, d, backend, tol, mocker):
        """Test that devices provide correct result computing the gradient of a
        circuit using the parameter-shift rule and the batch execution pipeline."""

        # check compatibility between provider and backend, and skip if incompatible
        is_compatible, failure_msg = check_provider_backend_compatibility(d, backend)
        if not is_compatible:
            pytest.skip(failure_msg)

        if backend not in state_backends and shots is None:
            pytest.skip("Hardware simulators do not support analytic mode")

        dev = qml.device(d[0], wires=3, backend=backend, shots=shots)

        spy1 = mocker.spy(QiskitDeviceLegacy, "batch_execute")
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

        # Check that QiskitDeviceLegacy.batch_execute was called twice
        assert spy1.call_count == 2

        # Check that run was called twice: for the partial derivatives and for
        # running the circuit
        assert spy2.call_count == 2

    def test_tracker(self):
        """Tests the device tracker with batch execution."""
        dev = qml.device("qiskit.aer", shots=100, wires=3)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        x = tensor(0.1, requires_grad=True)

        with qml.Tracker(dev) as tracker:
            qml.grad(circuit)(x)

        expected = {
            "executions": [1, 1, 1],
            "shots": [100, 100, 100],
            "batches": [1, 1],
            "batch_len": [1, 2],
        }

        assert tracker.history == expected
