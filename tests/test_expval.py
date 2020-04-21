import pytest

import numpy as np
import pennylane as qml

from conftest import U, U2, A


np.random.seed(42)

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


@pytest.mark.parametrize("theta, phi", list(zip(THETA, PHI)))
@pytest.mark.parametrize("analytic", [True, False])
@pytest.mark.parametrize("shots", [8192])
@pytest.mark.usefixtures("skip_unitary")
class TestExpval:
    """Test expectation values"""

    def test_identity_expectation(self, theta, phi, device, shots, tol):
        """Test that identity expectation value (i.e. the trace) is 1"""
        dev = device(2)

        O1 = qml.Identity(wires=[0])
        O2 = qml.Identity(wires=[1])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.CNOT(wires=[0, 1])
            ],
            rotations=[*O1.diagonalizing_gates(), *O2.diagonalizing_gates()]
        )

        dev._samples = dev.generate_samples()

        res = np.array([dev.expval(O1), dev.expval(O2)])
        assert np.allclose(res, np.array([1, 1]), **tol)

    def test_pauliz_expectation(self, theta, phi, device, shots, tol):
        """Test that PauliZ expectation value is correct"""
        dev = device(2)
        O1 = qml.PauliZ(wires=[0])
        O2 = qml.PauliZ(wires=[1])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.CNOT(wires=[0, 1])
            ],
            rotations=[*O1.diagonalizing_gates(), *O2.diagonalizing_gates()]
        )

        dev._samples = dev.generate_samples()

        res = np.array([dev.expval(O1), dev.expval(O2)])
        assert np.allclose(res, np.array([np.cos(theta), np.cos(theta) * np.cos(phi)]), **tol)

    def test_paulix_expectation(self, theta, phi, device, shots, tol):
        """Test that PauliX expectation value is correct"""
        dev = device(2)
        O1 = qml.PauliX(wires=[0])
        O2 = qml.PauliX(wires=[1])

        dev.apply(
            [
                qml.RY(theta, wires=[0]),
                qml.RY(phi, wires=[1]),
                qml.CNOT(wires=[0, 1])
            ],
            rotations=[*O1.diagonalizing_gates(), *O2.diagonalizing_gates()]
        )

        dev._samples = dev.generate_samples()

        res = np.array([dev.expval(O1), dev.expval(O2)])
        assert np.allclose(res, np.array([np.sin(theta) * np.sin(phi), np.sin(phi)]), **tol)

    def test_pauliy_expectation(self, theta, phi, device, shots, tol):
        """Test that PauliY expectation value is correct"""
        dev = device(2)
        O1 = qml.PauliY(wires=[0])
        O2 = qml.PauliY(wires=[1])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.CNOT(wires=[0, 1])
            ],
            rotations=[*O1.diagonalizing_gates(), *O2.diagonalizing_gates()]
        )

        dev._samples = dev.generate_samples()

        res = np.array([dev.expval(O1), dev.expval(O2)])
        assert np.allclose(res, np.array([0, -np.cos(theta) * np.sin(phi)]), **tol)

    def test_hadamard_expectation(self, theta, phi, device, shots, tol):
        """Test that Hadamard expectation value is correct"""
        dev = device(2)
        O1 = qml.Hadamard(wires=[0])
        O2 = qml.Hadamard(wires=[1])

        dev.apply(
            [
                qml.RY(theta, wires=[0]),
                qml.RY(phi, wires=[1]),
                qml.CNOT(wires=[0, 1])
            ],
            rotations=[*O1.diagonalizing_gates(), *O2.diagonalizing_gates()]
        )

        dev._samples = dev.generate_samples()

        res = np.array([dev.expval(O1), dev.expval(O2)])
        expected = np.array(
            [np.sin(theta) * np.sin(phi) + np.cos(theta), np.cos(theta) * np.cos(phi) + np.sin(phi)]
        ) / np.sqrt(2)
        assert np.allclose(res, expected, **tol)

    def test_hermitian_expectation(self, theta, phi, device, shots, tol):
        """Test that arbitrary Hermitian expectation values are correct"""
        dev = device(2)
        O1 = qml.Hermitian(A, wires=[0])
        O2 = qml.Hermitian(A, wires=[1])

        dev.apply(
            [
                qml.RY(theta, wires=[0]),
                qml.RY(phi, wires=[1]),
                qml.CNOT(wires=[0, 1])
            ],
            rotations=[*O1.diagonalizing_gates(), *O2.diagonalizing_gates()]
        )
        dev._samples = dev.generate_samples()

        res = np.array([dev.expval(O1), dev.expval(O2)])

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]
        ev1 = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        ev2 = ((a - d) * np.cos(theta) * np.cos(phi) + 2 * re_b * np.sin(phi) + a + d) / 2
        expected = np.array([ev1, ev2])

        assert np.allclose(res, expected, **tol)

    def test_multi_qubit_hermitian_expectation(self, theta, phi, device, shots, tol):
        """Test that arbitrary multi-qubit Hermitian expectation values are
        correct"""
        dev = device(2)

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )
        O1 = qml.Hermitian(A, wires=[0, 1])

        dev.apply(
            [
                qml.RY(theta, wires=[0]),
                qml.RY(phi, wires=[1]),
                qml.CNOT(wires=[0, 1])
            ],
            rotations=[*O1.diagonalizing_gates()]
        )
        dev._samples = dev.generate_samples()

        res = np.array([dev.expval(O1)])

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


@pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
@pytest.mark.parametrize("analytic", [True, False])
@pytest.mark.parametrize("shots", [8192])
class TestTensorExpval:
    """Test tensor expectation values"""

    def test_paulix_pauliy(self, theta, phi, varphi, device, shots, tol):
        """Test that a tensor product involving PauliX and PauliY works
        correctly"""
        dev = device(3)
        obs = qml.PauliX(0) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            rotations=obs.diagonalizing_gates()
        )

        dev._samples = dev.generate_samples()
        res = dev.expval(obs)

        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)

        assert np.allclose(res, expected, **tol)

    def test_pauliz_identity(self, theta, phi, varphi, device, shots, tol):
        """Test that a tensor product involving PauliZ and Identity works
        correctly"""
        dev = device(3)
        obs = qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            rotations=obs.diagonalizing_gates()
        )

        dev._samples = dev.generate_samples()
        res = dev.expval(obs)

        expected = np.cos(varphi)*np.cos(phi)

        assert np.allclose(res, expected, **tol)

    def test_pauliz_hadamard_pauliy(self, theta, phi, varphi, device, shots, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard
        works correctly"""
        dev = device(3)
        obs = qml.PauliZ(0) @ qml.Hadamard(1) @ qml.PauliY(2)

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            rotations=obs.diagonalizing_gates()
        )

        dev._samples = dev.generate_samples()
        res = dev.expval(obs)
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)

        assert np.allclose(res, expected, **tol)

    def test_hermitian(self, theta, phi, varphi, device, shots, tol):
        """Test that a tensor product involving qml.Hermitian works
        correctly"""
        dev = device(3)
        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        obs = qml.PauliZ(0) @ qml.Hermitian(A, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            rotations=obs.diagonalizing_gates()
        )

        dev._samples = dev.generate_samples()
        res = dev.expval(obs)
        expected = 0.5 * (
            -6 * np.cos(theta) * (np.cos(varphi) + 1)
            - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi))
            + 3 * np.cos(varphi) * np.sin(phi)
            + np.sin(phi)
        )

        assert np.allclose(res, expected, **tol)

    def test_hermitian_hermitian(self, theta, phi, varphi, device, shots, tol):
        """Test that a tensor product involving two Hermitian matrices works
        correctly"""
        dev = device(3)
        A1 = np.array([[1, 2],
                       [2, 4]])

        A2 = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )
        obs = qml.Hermitian(A1, wires=[0]) @ qml.Hermitian(A2, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            rotations=obs.diagonalizing_gates()
        )

        dev._samples = dev.generate_samples()
        res = dev.expval(obs)
        expected = 0.25 * (
            -30
            + 4 * np.cos(phi) * np.sin(theta)
            + 3 * np.cos(varphi) * (-10 + 4 * np.cos(phi) * np.sin(theta) - 3 * np.sin(phi))
            - 3 * np.sin(phi)
            - 2 * (5 + np.cos(phi) * (6 + 4 * np.sin(theta)) + (-3 + 8 * np.sin(theta)) * np.sin(phi))
            * np.sin(varphi)
            + np.cos(theta)
            * (
                18
                + 5 * np.sin(phi)
                + 3 * np.cos(varphi) * (6 + 5 * np.sin(phi))
                + 2 * (3 + 10 * np.cos(phi) - 5 * np.sin(phi)) * np.sin(varphi)
            )
        )

        assert np.allclose(res, expected, **tol)

    def test_hermitian_identity_expectation(self, theta, phi, varphi, device, shots, tol):
        """Test that a tensor product involving an Hermitian matrix and the
        identity works correctly"""
        dev = device(2)

        obs = qml.Hermitian(A, wires=[0]) @ qml.Identity(1)

        dev.apply(
            [
                qml.RY(theta, wires=[0]),
                qml.RY(phi, wires=[1]),
                qml.CNOT(wires=[0, 1]),
            ],
            rotations=obs.diagonalizing_gates()
        )

        dev._samples = dev.generate_samples()
        res = dev.expval(obs)

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]
        expected = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2

        assert np.allclose(res, expected, **tol)
