import pytest

import numpy as np
import pennylane as qml

from pennylane_qiskit import AerDevice, BasicAerDevice

from conftest import U, U2, A, Tensor


np.random.seed(42)

THETA = np.linspace(0.11, 2*np.pi-0.13, 3)
PHI = np.linspace(0.32, 2*np.pi-0.11, 3)
VARPHI = np.linspace(0.02, 2*np.pi-0.12, 3)


@pytest.mark.parametrize("theta, phi", zip(THETA, PHI))
@pytest.mark.parametrize("analytic", [True, False])
@pytest.mark.parametrize("shots", [8192])
class TestVar:
    """Tests for the variance"""

    def test_var(self, theta, phi, device, shots, tol):
        """Tests for variance calculation"""
        dev = device(2)

        # test correct variance for <Z> of a rotated state
        dev.apply("RX", wires=[0], par=[phi])
        dev.apply("RY", wires=[0], par=[theta])

        dev._obs_queue = [qml.PauliZ(wires=[0], do_queue=False)]
        dev.pre_measure()

        var = dev.var("PauliZ", [0], [])
        expected = 0.25 * (3 - np.cos(2 * theta) - 2 * np.cos(theta) ** 2 * np.cos(2 * phi))

        assert np.allclose(var, expected, **tol)

    def test_var_hermitian(self, theta, phi, device, shots, tol):
        """Tests for variance calculation using an arbitrary Hermitian observable"""
        dev = device(2)

        # test correct variance for <H> of a rotated state
        H = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        dev.apply("RX", wires=[0], par=[phi])
        dev.apply("RY", wires=[0], par=[theta])

        dev._obs_queue = [qml.Hermitian(H, wires=[0], do_queue=False)]
        dev.pre_measure()

        var = dev.var("Hermitian", [0], [H])
        expected = 0.5 * (
            2 * np.sin(2 * theta) * np.cos(phi) ** 2
            + 24 * np.sin(phi) * np.cos(phi) * (np.sin(theta) - np.cos(theta))
            + 35 * np.cos(2 * phi)
            + 39
        )

        assert np.allclose(var, expected, **tol)


@pytest.mark.parametrize("theta, phi, varphi", zip(THETA, PHI, VARPHI))
@pytest.mark.parametrize("analytic", [True, False])
@pytest.mark.parametrize("shots", [8192])
class TestTensorVar:
    """Tests for variance of tensor observables"""

    def test_paulix_pauliy(self, theta, phi, varphi, device, shots, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = device(3)
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("RX", wires=[2], par=[varphi])
        dev.apply("CNOT", wires=[0, 1], par=[])
        dev.apply("CNOT", wires=[1, 2], par=[])

        dev._obs_queue = [
            Tensor(["PauliX", "PauliY"], [[0], [2]], [[], []], qml.operation.Variance)
        ]
        dev.pre_measure()

        res = dev.var(["PauliX", "PauliY"], [[0], [2]], [[], [], []])

        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16

        assert np.allclose(res, expected, **tol)

    def test_pauliz_hadamard(self, theta, phi, varphi, device, shots, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = device(3)
        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RX", wires=[1], par=[phi])
        dev.apply("RX", wires=[2], par=[varphi])
        dev.apply("CNOT", wires=[0, 1], par=[])
        dev.apply("CNOT", wires=[1, 2], par=[])

        dev._obs_queue = [
            Tensor(["PauliZ", "Hadamard", "PauliY"], [[0], [1], [2]], [[], [], []], qml.operation.Variance)
        ]
        dev.pre_measure()

        res = dev.var(["PauliZ", "Hadamard", "PauliY"], [[0], [1], [2]], [[], [], []])

        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4

        assert np.allclose(res, expected, **tol)

    def test_hermitian(self, theta, phi, varphi, device, shots, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        dev = device(3)
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

        dev._obs_queue = [Tensor(["PauliZ", "Hermitian"], [[0], [1, 2]], [[], [A]], qml.operation.Variance)]
        dev.pre_measure()

        res = dev.var(["PauliZ", "Hermitian"], [[0], [1, 2]], [[], [A]])

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

        assert np.allclose(res, expected, **tol)
