import sys

import numpy as np
import pennylane as qml
from pennylane.numpy import tensor
import pytest
import qiskit
import qiskit.providers.aer as aer

from pennylane_qiskit import AerDevice, BasicAerDevice

from conftest import state_backends

pldevices = [("qiskit.aer", qiskit.Aer), ("qiskit.basicaer", qiskit.BasicAer)]


class TestDeviceIntegration:
    """Test the devices work correctly from the PennyLane frontend."""

    @pytest.mark.parametrize("d", pldevices)
    def test_load_device(self, d, backend):
        """Test that the qiskit device loads correctly"""
        dev = qml.device(d[0], wires=2, backend=backend, shots=1024)
        assert dev.num_wires == 2
        assert dev.shots == 1024
        assert dev.short_name == d[0]
        assert dev.provider == d[1]

    def test_incorrect_backend(self):
        """Test that exception is raised if name is incorrect"""
        with pytest.raises(ValueError, match="Backend 'none' does not exist"):
            qml.device("qiskit.aer", wires=2, backend="none")

    def test_incorrect_backend_wires(self):
        """Test that exception is raised if number of wires is too large"""
        with pytest.raises(ValueError, match=r"Backend 'statevector\_simulator' supports maximum"):
            qml.device("qiskit.aer", wires=100, backend="statevector_simulator")

    def test_args(self):
        """Test that the device requires correct arguments"""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qml.device("qiskit.aer")

        with pytest.raises(qml.DeviceError, match="specified number of shots needs to be at least 1"):
            qml.device("qiskit.aer", backend="qasm_simulator", wires=1, shots=0)

    @pytest.mark.parametrize("d", pldevices)
    @pytest.mark.parametrize("analytic", [True, False])
    @pytest.mark.parametrize("shots", [8192])
    def test_one_qubit_circuit(self, shots, analytic, d, backend, tol):
        """Test that devices provide correct result for a simple circuit"""
        if backend not in state_backends and analytic:
            pytest.skip("Hardware simulators do not support analytic mode")

        dev = qml.device(d[0], wires=1, backend=backend, shots=shots, analytic=analytic)

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
    @pytest.mark.parametrize("analytic", [False])
    @pytest.mark.parametrize("shots", [8192])
    def test_one_qubit_circuit(self, shots, analytic, d, backend, tol):
        """Integration test for the Basisstate and Rot operations for when analytic
        is False"""
        dev = qml.device(d[0], wires=1, backend=backend, shots=shots, analytic=analytic)

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
            m.setattr(aer.QasmSimulator, "set_options", lambda *args, **kwargs: cache.append(kwargs))
            dev = qml.device("qiskit.aer", wires=2, noise_model="test value")
        assert cache[0] == {'noise_model': 'test value'}

    def test_invalid_noise_model(self):
        """Test that the noise model argument causes an exception to be raised
        if the backend does not support it"""
        with pytest.raises(ValueError, match="does not support noisy simulations"):
            dev = qml.device("qiskit.basicaer", wires=2, noise_model="test value")

    def test_overflow_kwargs(self):
        """Test all overflow kwargs are extracted for the AerDevice"""
        dev = qml.device('qiskit.aer', wires=2, k1="v1", k2="v2")
        assert dev.run_args["k1"] == "v1"
        assert dev.run_args["k2"] == "v2"


class TestLoadIntegration:
    """Integration tests for the PennyLane load function. This test ensures that the PennyLane-Qiskit
    specific load functions integrate properly with the PennyLane-Qiskit plugin."""

    hadamard_qasm = 'OPENQASM 2.0;' \
                    'include "qelib1.inc";' \
                    'qreg q[1];' \
                    'h q[0];'

    def test_load_qiskit_circuit(self):
        """Test that the default load function works correctly."""
        theta = qiskit.circuit.Parameter('θ')

        qc = qiskit.QuantumCircuit(2)
        qc.rx(theta, 0)

        my_template = qml.load(qc, format='qiskit')

        dev = qml.device('default.qubit', wires=2)

        angles = np.array([0.53896774, 0.79503606, 0.27826503, 0.])

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

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def loaded_quantum_circuit():
            qml.from_qasm(TestLoadIntegration.hadamard_qasm)(wires=[0])
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

        with open(apply_hadamard, "w") as f:
            f.write(TestLoadIntegration.hadamard_qasm)

        hadamard = qml.from_qasm_file(apply_hadamard)

        dev = qml.device('default.qubit', wires=2)

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

    @pytest.mark.parametrize("shots", [1000])
    @pytest.mark.parametrize("analytic", [True, False])
    def test_rotation(self, init_state, state_vector_device, shots, analytic, tol):
        """Test that the QubitStateVector and Rot operations are decomposed using a
        Qiskit device with statevector backend"""

        dev = state_vector_device(1)

        if dev.backend_name == "unitary_simulator":
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
            qml.QubitStateVector(state, wires=[0])
            qml.Rot(a, b, c, wires=[0])
            return qml.expval(qml.Identity(0))

        qubitstatevector_and_rot()

        assert np.allclose(np.abs(dev.state) ** 2, np.abs(rz(c) @ ry(b) @ rz(a) @ state) ** 2, **tol)

    @pytest.mark.parametrize("shots", [1000])
    @pytest.mark.parametrize("analytic", [True, False])
    def test_basisstate(self, init_state, state_vector_device, shots, analytic, tol):
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

    @pytest.mark.parametrize("shots", [1000])
    @pytest.mark.parametrize("analytic", [True, False])
    def test_basisstate_init_all_zero_states(self, init_state, state_vector_device, shots, analytic, tol):
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


class TestPLTemplates:
    """Integration tests for checking certain PennyLane templates."""

    def test_random_layers_tensor_unwrapped(self, monkeypatch):
        """Test that if random_layer() receives a one element PennyLane tensor,
        then it is unwrapped successfully.

        The test involves using RandomLayers, which then calls random_layer
        internally. Eventually each gate used by random_layer receives a single
        scalar.
        """
        dev = qml.device("qiskit.aer", wires=4)

        lst = []

        # Mock function that accumulates gate parameters
        mock_func = lambda par, wires: lst.append(par)

        with monkeypatch.context() as m:

            # Mock the gates used in RandomLayers
            m.setattr(qml.templates.layers.random, "RX", mock_func)
            m.setattr(qml.templates.layers.random, "RY", mock_func)
            m.setattr(qml.templates.layers.random, "RZ", mock_func)

            @qml.qnode(dev)
            def circuit(phi=None):
                qml.templates.layers.RandomLayers(phi, wires=list(range(4)))
                return qml.expval(qml.PauliZ(0))

            # RandomLayers loops over the random_layer function, with each call to random_layer
            # being passed a `np.tensor` scalar.
            phi = qml.numpy.tensor([[0.04439891, 0.14490549, 3.29725643, 2.51240058]])

            # Call the QNode, accumulate parameters
            circuit(phi=phi)

            # Check parameters
            assert all([isinstance(x, tensor) for x in lst])

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

        with qml._queuing.OperationRecorder() as rec:
            circuit(phi=phi)

        for i in range(phi.shape[1]):
            # Test each rotation applied
            assert rec.queue[i].name == "RX"
            assert len(rec.queue[i].parameters) == 1

            # Test that the gate parameter is a PennyLane tensor
            assert isinstance(rec.queue[i].parameters[0], tensor)

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


        with qml._queuing.OperationRecorder() as rec:
            circuit(phi=phi)

        # Test the rotation applied
        assert rec.queue[0].name == "Rot"
        assert len(rec.queue[0].parameters) == 3

        # Test that the gate parameters are PennyLane tensors,
        assert isinstance(rec.queue[0].parameters[0], tensor)

        assert isinstance(rec.queue[0].parameters[1], tensor)

        assert isinstance(rec.queue[0].parameters[2], tensor)

class TestInverses:
    """Integration tests checking that the inverse of the operations are applied."""

    def test_inverse_of_operation(self):
        """Test that the inverse of operations works as expected
        by comparing a simple circuit with default.qubit."""
        dev = qml.device('default.qubit', wires=2)

        dev2 = qml.device('qiskit.aer', backend='statevector_simulator', shots=5, wires=2, analytic=True)

        angles = np.array([0.53896774, 0.79503606, 0.27826503, 0.])

        @qml.qnode(dev)
        def circuit_with_inverses(angle):
            qml.Hadamard(0).inv()
            qml.RX(angle, wires=0).inv()
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev2)
        def circuit_with_inverses_default_qubit(angle):
            qml.Hadamard(0).inv()
            qml.RX(angle, wires=0).inv()
            return qml.expval(qml.PauliZ(0))

        for x in angles:
            assert np.allclose(circuit_with_inverses(x), circuit_with_inverses_default_qubit(x))

class TestNoise:
    """Integration test for the noise models."""

    def test_noise_applied(self):
        """Test that the qiskit noise model is applied correctly"""
        noise_model = aer.noise.NoiseModel()
        bit_flip = aer.noise.pauli_error([('X', 1), ('I', 0)])

        # Create a noise model where the RX operation always flips the bit
        noise_model.add_all_qubit_quantum_error(bit_flip, ["rx"])

        dev = qml.device('qiskit.aer', wires=2, noise_model=noise_model)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0, wires=[0])
            return qml.expval(qml.PauliZ(wires=0))

        assert circuit() == -1
