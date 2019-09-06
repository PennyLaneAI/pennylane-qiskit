import pytest

import numpy as np
import pennylane as qml

from conftest import U, U2, A


np.random.seed(42)


@pytest.mark.parametrize("shots", [0, 8192])
class TestExpval:
    """Test expectation values"""

    def test_identity_expectation(self, device, shots, tol):
        """Test that identity expectation value (i.e. the trace) is 1"""
        theta = 0.432
        phi = 0.123

        dev = device(2)
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("CNOT", wires=[0, 1], par=[])

        O = qml.Identity
        name = "Identity"

        dev._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
        res = dev.pre_measure()

        res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])

        assert np.allclose(res, np.array([1, 1]), **tol)

    def test_pauliz_expectation(self, device, shots, tol):
        """Test that PauliZ expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2)
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("CNOT", wires=[0, 1], par=[])

        O = qml.PauliZ
        name = "PauliZ"

        dev._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
        res = dev.pre_measure()

        res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])

        assert np.allclose(res, np.array([np.cos(theta), np.cos(theta) * np.cos(phi)]), **tol)

    def test_paulix_expectation(self, device, shots, tol):
        """Test that PauliX expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2)
        dev.apply("RY", wires=[0], par=[theta])
        dev.apply("RY", wires=[1], par=[phi])
        dev.apply("CNOT", wires=[0, 1], par=[])

        O = qml.PauliX
        name = "PauliX"

        dev._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
        dev.pre_measure()

        res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])
        assert np.allclose(res, np.array([np.sin(theta) * np.sin(phi), np.sin(phi)]), **tol)

    def test_pauliy_expectation(self, device, shots, tol):
        """Test that PauliY expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2)
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("CNOT", wires=[0, 1], par=[])

        O = qml.PauliY
        name = "PauliY"

        dev._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
        dev.pre_measure()

        res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])
        assert np.allclose(res, np.array([0, -np.cos(theta) * np.sin(phi)]), **tol)

    def test_hadamard_expectation(self, device, shots, tol):
        """Test that Hadamard expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2)
        dev.apply("RY", wires=[0], par=[theta])
        dev.apply("RY", wires=[1], par=[phi])
        dev.apply("CNOT", wires=[0, 1], par=[])

        O = qml.Hadamard
        name = "Hadamard"

        dev._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
        dev.pre_measure()

        res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])
        expected = np.array(
            [np.sin(theta) * np.sin(phi) + np.cos(theta), np.cos(theta) * np.cos(phi) + np.sin(phi)]
        ) / np.sqrt(2)
        assert np.allclose(res, expected, **tol)

    def test_hermitian_expectation(self, device, shots, tol):
        """Test that arbitrary Hermitian expectation values are correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2)
        dev.apply("RY", wires=[0], par=[theta])
        dev.apply("RY", wires=[1], par=[phi])
        dev.apply("CNOT", wires=[0, 1], par=[])

        O = qml.Hermitian
        name = "Hermitian"

        dev._obs_queue = [O(A, wires=[0], do_queue=False), O(A, wires=[1], do_queue=False)]
        dev.pre_measure()

        res = np.array([dev.expval(name, [0], [A]), dev.expval(name, [1], [A])])

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]
        ev1 = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        ev2 = ((a - d) * np.cos(theta) * np.cos(phi) + 2 * re_b * np.sin(phi) + a + d) / 2
        expected = np.array([ev1, ev2])

        assert np.allclose(res, expected, **tol)

    def test_multi_mode_hermitian_expectation(self, device, shots, tol):
        """Test that arbitrary multi-mode Hermitian expectation values are correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2)
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

        assert np.allclose(res, expected, **tol)
