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
