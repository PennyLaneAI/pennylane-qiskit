import pytest

import numpy as np
import pennylane as qml

from pennylane_qiskit import AerDevice, BasicAerDevice

from conftest import init_state, single_qubit, single_qubit_param, two_qubit, two_qubit_param, three_qubit, rot, U, U2, A


np.random.seed(42)


backends = ["statevector_simulator", "unitary_simulator"]


@pytest.fixture(params=backends)
def backend(request):
    return request.param


@pytest.fixture(params=[AerDevice, BasicAerDevice])
def device(request, backend):
    def _device(n, shots=0):
        return request.param(wires=n, backend=backend, shots=shots)
    return _device


class TestApply:
    """Test application of PennyLane operations."""

    def test_qubit_state_vector(self, init_state, device, tol):
        """Test PauliX application"""
        dev = device(1)
        state = init_state(1)

        dev.apply("QubitStateVector", [0], [state])
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state)**2
        expected = np.abs(state)**2
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,mat", single_qubit)
    def test_single_qubit_no_parameters(self, init_state, device, name, mat, tol):
        """Test PauliX application"""
        dev = device(1)
        state = init_state(1)

        dev.apply("QubitStateVector", [0], [state])
        dev.apply(name, [0], [])
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state)**2
        expected = np.abs(mat @ state)**2
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("name,func", single_qubit_param)
    def test_single_qubit_parameters(self, init_state, device, name, func, theta, tol):
        """Test PauliX application"""
        dev = device(1)
        state = init_state(1)

        dev.apply("QubitStateVector", [0], [state])
        dev.apply(name, [0], [theta])
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state)**2
        expected = np.abs(func(theta) @ state)**2
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_rotation(self, init_state, device, tol):
        """Test three axis rotation gate"""
        dev = device(1)
        state = init_state(1)

        a = 0.542
        b = 1.3432
        c = -0.654

        dev.apply("QubitStateVector", [0], [state])
        dev.apply("Rot", [0], [a, b, c])
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state)**2
        expected = np.abs(rot(a, b, c) @ state)**2
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,mat", two_qubit)
    def test_two_qubit_no_parameters(self, init_state, device, name, mat, tol):
        """Test PauliX application"""
        dev = device(2)
        state = init_state(2)

        dev.apply("QubitStateVector", [0, 1], [state])
        dev.apply(name, [0, 1], [])
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state)**2
        expected = np.abs(mat @ state)**2
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary(self, init_state, device, mat, tol):
        N = int(np.log2(len(mat)))
        dev = device(N)
        state = init_state(N)

        dev.apply("QubitStateVector", list(range(N)), [state])
        dev.apply("QubitUnitary", list(range(N)), [mat])
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state)**2
        expected = np.abs(mat @ state)**2
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("name, mat", three_qubit)
    def test_three_qubit_no_parameters(self, init_state, device, name, mat, tol):
        dev = device(3)
        state = init_state(3)

        dev.apply("QubitStateVector", [0, 1, 2], [state])
        dev.apply("QubitUnitary", [0, 1, 2], [mat])
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state)**2
        expected = np.abs(mat @ state)**2
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("name,func", two_qubit_param)
    def test_single_qubit_parameters(self, init_state, device, name, func, theta, tol):
        """Test PauliX application"""
        dev = device(2)
        state = init_state(2)

        dev.apply("QubitStateVector", [0, 1], [state])
        dev.apply(name, [0, 1], [theta])
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state)**2
        expected = np.abs(func(theta) @ state)**2
        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("shots", [0, 8192])
class TestExpval:
    """Test expectation values"""

    def test_identity_expectation(self, device, shots, tol):
        """Test that identity expectation value (i.e. the trace) is 1"""
        theta = 0.432
        phi = 0.123

        dev = device(2, shots=shots)
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("CNOT", wires=[0, 1], par=[])

        O = qml.Identity
        name = "Identity"

        dev._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
        res = dev.pre_measure()

        res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])

        # below are the analytic expectation values for this circuit (trace should always be 1)
        assert np.allclose(res, np.array([1, 1]), atol=tol)

    def test_pauliz_expectation(self, device, shots, tol):
        """Test that PauliZ expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2, shots=shots)
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("CNOT", wires=[0, 1], par=[])

        O = qml.PauliZ
        name = "PauliZ"

        dev._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
        res = dev.pre_measure()

        res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])

        # below are the analytic expectation values for this circuit
        assert np.allclose(res, np.array([np.cos(theta), np.cos(theta) * np.cos(phi)]), atol=0.03)

    def test_paulix_expectation(self, device, shots, tol):
        """Test that PauliX expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2, shots=shots)
        dev.apply("RY", wires=[0], par=[theta])
        dev.apply("RY", wires=[1], par=[phi])
        dev.apply("CNOT", wires=[0, 1], par=[])

        O = qml.PauliX
        name = "PauliX"

        dev._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
        dev.pre_measure()

        res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])
        # below are the analytic expectation values for this circuit
        assert np.allclose(res, np.array([np.sin(theta) * np.sin(phi), np.sin(phi)]), atol=0.03)

    def test_pauliy_expectation(self, device, shots, tol):
        """Test that PauliY expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2, shots=shots)
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("CNOT", wires=[0, 1], par=[])

        O = qml.PauliY
        name = "PauliY"

        dev._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
        dev.pre_measure()

        # below are the analytic expectation values for this circuit
        res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])
        assert np.allclose(res, np.array([0, -np.cos(theta) * np.sin(phi)]), atol=0.03)

    def test_hadamard_expectation(self, device, shots, tol):
        """Test that Hadamard expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2, shots=shots)
        dev.apply("RY", wires=[0], par=[theta])
        dev.apply("RY", wires=[1], par=[phi])
        dev.apply("CNOT", wires=[0, 1], par=[])

        O = qml.Hadamard
        name = "Hadamard"

        dev._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
        dev.pre_measure()

        res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])
        # below are the analytic expectation values for this circuit
        expected = np.array(
            [np.sin(theta) * np.sin(phi) + np.cos(theta), np.cos(theta) * np.cos(phi) + np.sin(phi)]
        ) / np.sqrt(2)
        assert np.allclose(res, expected, atol=0.03)

    def test_hermitian_expectation(self, device, shots, tol):
        """Test that arbitrary Hermitian expectation values are correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2, shots=shots)
        dev.apply("RY", wires=[0], par=[theta])
        dev.apply("RY", wires=[1], par=[phi])
        dev.apply("CNOT", wires=[0, 1], par=[])

        O = qml.Hermitian
        name = "Hermitian"

        dev._obs_queue = [O(A, wires=[0], do_queue=False), O(A, wires=[1], do_queue=False)]
        dev.pre_measure()

        res = np.array([dev.expval(name, [0], [A]), dev.expval(name, [1], [A])])

        # below are the analytic expectation values for this circuit with arbitrary
        # Hermitian observable A
        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]
        ev1 = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        ev2 = ((a - d) * np.cos(theta) * np.cos(phi) + 2 * re_b * np.sin(phi) + a + d) / 2
        expected = np.array([ev1, ev2])

        assert np.allclose(res, expected, atol=0.03)

    def test_multi_mode_hermitian_expectation(self, device, shots, tol):
        """Test that arbitrary multi-mode Hermitian expectation values are correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2, shots=shots)
        dev.apply("RY", wires=[0], par=[theta])
        dev.apply("RY", wires=[1], par=[phi])
        dev.apply("CNOT", wires=[0, 1], par=[])

        O = qml.Hermitian
        name = "Hermitian"

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        dev._obs_queue = [O(A, wires=[0, 1], do_queue=False)]
        dev.pre_measure()

        res = np.array([dev.expval(name, [0, 1], [A])])

        # below is the analytic expectation value for this circuit with arbitrary
        # Hermitian observable A
        expected = 0.5 * (
            6 * np.cos(theta) * np.sin(phi)
            - np.sin(theta) * (8 * np.sin(phi) + 7 * np.cos(phi) + 3)
            - 2 * np.sin(phi)
            - 6 * np.cos(phi)
            - 6
        )

        assert np.allclose(res, expected, atol=0.1)


@pytest.mark.parametrize("shots", [0, 100000])
class TestTensorExpval:
    """Test tensor expectation values"""

    def test_paulix_pauliy(self, device, shots, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        theta = 0.432
        phi = 0.123
        varphi = -0.543

        dev = device(3, shots=shots)
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("RX", wires=[2], par=[varphi])
        dev.apply("CNOT", wires=[0, 1], par=[])
        dev.apply("CNOT", wires=[1, 2], par=[])

        dev._obs_queue = [
            qml.PauliX(wires=[0], do_queue=False) \
            @ qml.PauliY(wires=[2], do_queue=False)
        ]
        res = dev.pre_measure()

        res = dev.expval(["PauliX", "PauliY"], [[0], [2]], [[], [], []])
        expected = np.sin(theta)*np.sin(phi)*np.sin(varphi)

        # below are the analytic expectation values for this circuit (trace should always be 1)
        assert np.allclose(res, expected, atol=0.03)

    def test_pauliz_hadamard(self, device, shots, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        theta = 0.432
        phi = 0.123
        varphi = -0.543

        dev = device(3, shots=shots)
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("RX", wires=[2], par=[varphi])
        dev.apply("CNOT", wires=[0, 1], par=[])
        dev.apply("CNOT", wires=[1, 2], par=[])

        dev._obs_queue = [
            qml.PauliZ(wires=[0], do_queue=False) \
            @ qml.Hadamard(wires=[1], do_queue=False) \
            @ qml.PauliY(wires=[2], do_queue=False)
        ]
        res = dev.pre_measure()

        res = dev.expval(["PauliZ", "Hadamard", "PauliY"], [[0], [1], [2]], [[], [], []])
        expected = -(np.cos(varphi)*np.sin(phi) + np.sin(varphi)*np.cos(theta))/np.sqrt(2)

        # below are the analytic expectation values for this circuit (trace should always be 1)
        assert np.allclose(res, expected, atol=0.03)

    def test_hermitian(self, device, shots, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        theta = 0.432
        phi = 0.123
        varphi = -0.543

        dev = device(3, shots=shots)
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("RX", wires=[2], par=[varphi])
        dev.apply("CNOT", wires=[0, 1], par=[])
        dev.apply("CNOT", wires=[1, 2], par=[])

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        dev._obs_queue = [
            qml.PauliZ(wires=[0], do_queue=False) \
            @ qml.Hermitian(A, wires=[1, 2], do_queue=False)
        ]
        res = dev.pre_measure()

        res = dev.expval(["PauliZ", "Hermitian"], [[0], [1, 2]], [[], [A]])
        expected = 0.5*(-6*np.cos(theta)*(np.cos(varphi)+1)-2*np.sin(varphi)*(np.cos(theta)+np.sin(phi)-2*np.cos(phi))+3*np.cos(varphi)*np.sin(phi)+np.sin(phi))

        # below are the analytic expectation values for this circuit (trace should always be 1)
        assert np.allclose(res, expected, atol=0.1)
