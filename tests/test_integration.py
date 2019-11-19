import sys

import numpy as np
import pennylane as qml
import pytest
import qiskit

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


class TestKeywordArguments:
    """Test keyword argument logic is correct"""

    @pytest.mark.parametrize("d", pldevices)
    def test_compile_backend(self, d):
        """Test that the compile backend argument is properly
        extracted"""
        dev = qml.device(d[0], wires=2, compile_backend="test value")
        assert dev.compile_backend == "test value"

    def test_noise_model(self):
        """Test that the noise model argument is properly
        extracted if the backend supports it"""
        dev = qml.device("qiskit.aer", wires=2, noise_model="test value")
        assert dev.run_args["noise_model"] == "test value"

    def test_invalid_noise_model(self):
        """Test that the noise model argument causes an exception to be raised
        if the backend does not support it"""
        with pytest.raises(ValueError, match="does not support noisy simulations"):
            dev = qml.device("qiskit.basicaer", wires=2, noise_model="test value")

    @pytest.mark.parametrize("d", pldevices)
    def test_overflow_backend_options(self, d):
        """Test all overflow backend options are extracted"""
        dev = qml.device(d[0], wires=2, k1="v1", k2="v2")
        assert dev.run_args["backend_options"]["k1"] == "v1"
        assert dev.run_args["backend_options"]["k2"] == "v2"


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
    def test_rotation(self, init_state, device, shots, analytic, tol):
        """Test that the QubitStateVector and Rot operations are decomposed using a Qiskit device"""
        dev = device(1)
        state = init_state(1)

        a = 0.542
        b = 1.3432
        c = -0.654

        @qml.qnode(dev)
        def qubitstatevector_and_rot():
            qml.QubitStateVector(state, wires=[0])
            qml.Rot(a, b, c, wires=[0])
            return qml.expval(qml.Identity(0))

        dev2 = qml.device('default.qubit', shots=shots, wires=2, analytic=analytic)

        @qml.qnode(dev2)
        def qubitstatevector_and_rot_default_qubit():
            qml.QubitStateVector(state, wires=[0])
            qml.Rot(a, b, c, wires=[0])
            return qml.expval(qml.Identity(0))

        assert np.allclose(qubitstatevector_and_rot(), qubitstatevector_and_rot_default_qubit(), **tol)
