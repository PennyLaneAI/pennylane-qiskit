import pytest

import numpy as np
import pennylane as qml

from pennylane_qiskit import AerDevice, BasicAerDevice

from conftest import U, U2, A


np.random.seed(42)

THETA = np.linspace(0.11, 1, 3)
PHI = np.linspace(0.32, 1, 3)
VARPHI = np.linspace(0.02, 1, 3)


@pytest.mark.parametrize("analytic", [False])
@pytest.mark.parametrize("shots", [8192])
class TestSample:
    """Tests for the sample return type"""

    def test_sample_values(self, device, shots, tol):
        """Tests if the samples returned by sample have
        the correct values
        """
        dev = device(1)
        par = 1.5708

        observable = qml.PauliZ(wires=[0])

        dev.apply(
            [
                qml.RX(par, wires=[0]),
            ],
            rotations=[*observable.diagonalizing_gates()]
        )

        dev._samples = dev.generate_samples()

        s1 = dev.sample(observable)

        # s1 should only contain 1 and -1
        assert np.allclose(s1 ** 2, 1, **tol)

    @pytest.mark.parametrize("theta", THETA)
    def test_sample_values_hermitian(self, theta, device, shots, tol):
        """Tests if the samples of a Hermitian observable returned by sample have
        the correct values
        """
        dev = device(1)

        A = np.array([[1, 2j], [-2j, 0]])

        observable = qml.Hermitian(A, wires=[0])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
            ],
            rotations=[*observable.diagonalizing_gates()]
        )

        dev._samples = dev.generate_samples()

        s1 = dev.sample(observable)

        # s1 should only contain the eigenvalues of
        # the hermitian matrix
        eigvals = np.linalg.eigvalsh(A)
        assert set(np.round(s1, 8)).issubset(set(np.round(eigvals, 8)))

        # the analytic mean is 2*sin(theta)+0.5*cos(theta)+0.5
        assert np.allclose(np.mean(s1), 2 * np.sin(theta) + 0.5 * np.cos(theta) + 0.5, **tol)

        # the analytic variance is 0.25*(sin(theta)-4*cos(theta))^2
        assert np.allclose(np.var(s1), 0.25 * (np.sin(theta) - 4 * np.cos(theta)) ** 2, **tol)

    @pytest.mark.parametrize("theta", THETA)
    def test_sample_values_hermitian_multi_qubit(self, theta, device, shots, tol):
        """Tests if the samples of a multi-qubit Hermitian observable returned by sample have
        the correct values
        """
        dev = device(2)

        A = np.array(
            [
                [1, 2j, 1 - 2j, 0.5j],
                [-2j, 0, 3 + 4j, 1],
                [1 + 2j, 3 - 4j, 0.75, 1.5 - 2j],
                [-0.5j, 1, 1.5 + 2j, -1],
            ]
        )

        observable = qml.Hermitian(A, wires=[0, 1])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RY(2 * theta, wires=[1]),
                qml.CNOT(wires=[0, 1])
            ],
            rotations=[*observable.diagonalizing_gates()]
        )

        dev._samples = dev.generate_samples()

        s1 = dev.sample(observable)

        # s1 should only contain the eigenvalues of
        # the hermitian matrix
        eigvals = np.linalg.eigvalsh(A)
        assert set(np.round(s1, 8)).issubset(set(np.round(eigvals, 8)))

        # make sure the mean matches the analytic mean
        expected = (
            88 * np.sin(theta)
            + 24 * np.sin(2 * theta)
            - 40 * np.sin(3 * theta)
            + 5 * np.cos(theta)
            - 6 * np.cos(2 * theta)
            + 27 * np.cos(3 * theta)
            + 6
        ) / 32
        assert np.allclose(np.mean(s1), expected, **tol)


@pytest.mark.parametrize("theta, phi, varphi", list(zip(THETA, PHI, VARPHI)))
@pytest.mark.parametrize("analytic", [False])
@pytest.mark.parametrize("shots", [8192])
class TestTensorSample:
    """Test tensor expectation values"""

    def test_paulix_pauliy(self, theta, phi, varphi, device, shots, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = device(3)

        observable = qml.PauliX(wires=[0]) @ qml.PauliY(wires=[2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            rotations=[*observable.diagonalizing_gates()]
        )

        dev._samples = dev.generate_samples()

        s1 = dev.sample(observable)

        # s1 should only contain 1 and -1
        assert np.allclose(s1 ** 2, 1, **tol)

        mean = np.mean(s1)
        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)
        assert np.allclose(mean, expected, **tol)

        var = np.var(s1)
        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16
        assert np.allclose(var, expected, **tol)

    def test_pauliz_hadamard_pauliy(self, theta, phi, varphi, device, shots, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = device(3)

        observable = qml.PauliZ(wires=[0]) @ qml.Hadamard(wires=[1]) @ qml.PauliY(wires=[2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            rotations=[*observable.diagonalizing_gates()]
        )

        dev._samples = dev.generate_samples()

        s1 = dev.sample(observable)

        # s1 should only contain 1 and -1
        assert np.allclose(s1 ** 2, 1, **tol)

        mean = np.mean(s1)
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)
        assert np.allclose(mean, expected, **tol)

        var = np.var(s1)
        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4
        assert np.allclose(var, expected, **tol)

    def test_hermitian(self, theta, phi, varphi, device, shots, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = device(3)

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )
        observable = qml.PauliZ(wires=[0]) @ qml.Hermitian(A, wires=[1, 2])

        dev.apply(
            [
                qml.RX(theta, wires=[0]),
                qml.RX(phi, wires=[1]),
                qml.RX(varphi, wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2])
            ],
            rotations=[*observable.diagonalizing_gates()]
        )

        dev._samples = dev.generate_samples()

        s1 = dev.sample(observable)

        # s1 should only contain the eigenvalues of
        # the hermitian matrix tensor product Z
        Z = np.diag([1, -1])
        eigvals = np.linalg.eigvalsh(np.kron(Z, A))
        assert set(np.round(s1, 8)).issubset(set(np.round(eigvals, 8)))

        mean = np.mean(s1)
        expected = 0.5 * (
            -6 * np.cos(theta) * (np.cos(varphi) + 1)
            - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi))
            + 3 * np.cos(varphi) * np.sin(phi)
            + np.sin(phi)
        )
        assert np.allclose(mean, expected, **tol)

        var = np.var(s1)
        expected = (
            1057
            - np.cos(2 * phi)
            + 12 * (27 + np.cos(2 * phi)) * np.cos(varphi)
            - 2 * np.cos(2 * varphi) * np.sin(phi) * (16 * np.cos(phi) + 21 * np.sin(phi))
            + 16 * np.sin(2 * phi)
            - 8 * (-17 + np.cos(2 * phi) + 2 * np.sin(2 * phi)) * np.sin(varphi)
            - 8 * np.cos(2 * theta) * (3 + 3 * np.cos(varphi) + np.sin(varphi)) ** 2
            - 24 * np.cos(phi) * (np.cos(phi) + 2 * np.sin(phi)) * np.sin(2 * varphi)
            - 8
            * np.cos(theta)
            * (
                4
                * np.cos(phi)
                * (
                    4
                    + 8 * np.cos(varphi)
                    + np.cos(2 * varphi)
                    - (1 + 6 * np.cos(varphi)) * np.sin(varphi)
                )
                + np.sin(phi)
                * (
                    15
                    + 8 * np.cos(varphi)
                    - 11 * np.cos(2 * varphi)
                    + 42 * np.sin(varphi)
                    + 3 * np.sin(2 * varphi)
                )
            )
        ) / 16
        assert np.allclose(var, expected, **tol)
