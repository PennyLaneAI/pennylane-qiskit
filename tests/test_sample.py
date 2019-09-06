import pytest

import numpy as np
import pennylane as qml

from pennylane_qiskit import AerDevice, BasicAerDevice

from conftest import U, U2, A


np.random.seed(42)


@pytest.mark.parametrize("shots", [8192])
class TestSample:
    """Tests for the sample return type"""

    def test_sample_values(self, device, shots, tol):
        """Tests if the samples returned by sample have
        the correct values
        """
        dev = device(1)

        dev.apply("RX", wires=[0], par=[1.5708])

        dev._obs_queue = [qml.PauliZ(wires=[0], do_queue=False)]

        for idx in range(len(dev._obs_queue)):
            dev._obs_queue[idx].return_type = qml.operation.Sample

        dev.pre_measure()

        s1 = dev.sample("PauliZ", [0], [], 10)

        # s1 should only contain 1 and -1
        assert np.allclose(s1 ** 2, 1, **tol)

    def test_sample_values_hermitian(self, device, shots, tol):
        """Tests if the samples of a Hermitian observable returned by sample have
        the correct values
        """
        dev = device(1)

        theta = 0.543
        A = np.array([[1, 2j], [-2j, 0]])

        dev.apply("RX", wires=[0], par=[theta])

        dev._obs_queue = [qml.Hermitian(A, wires=[0], do_queue=False)]

        for idx in range(len(dev._obs_queue)):
            dev._obs_queue[idx].return_type = qml.operation.Sample

        dev.pre_measure()

        s1 = dev.sample("Hermitian", [0], [A])

        # s1 should only contain the eigenvalues of
        # the hermitian matrix
        eigvals = np.linalg.eigvalsh(A)
        assert np.allclose(sorted(list(set(s1))), sorted(eigvals), **tol)

        # the analytic mean is 2*sin(theta)+0.5*cos(theta)+0.5
        assert np.allclose(np.mean(s1), 2 * np.sin(theta) + 0.5 * np.cos(theta) + 0.5, **tol)

        # the analytic variance is 0.25*(sin(theta)-4*cos(theta))^2
        assert np.allclose(np.var(s1), 0.25 * (np.sin(theta) - 4 * np.cos(theta)) ** 2, **tol)

    def test_sample_values_hermitian_multi_qubit(self, device, shots, tol):
        """Tests if the samples of a multi-qubit Hermitian observable returned by sample have
        the correct values
        """
        dev = device(2)
        theta = 0.543

        A = np.array(
            [
                [1, 2j, 1 - 2j, 0.5j],
                [-2j, 0, 3 + 4j, 1],
                [1 + 2j, 3 - 4j, 0.75, 1.5 - 2j],
                [-0.5j, 1, 1.5 + 2j, -1],
            ]
        )

        dev.apply("RX", wires=[0], par=[theta])
        dev.apply("RY", wires=[1], par=[2 * theta])
        dev.apply("CNOT", wires=[0, 1], par=[])

        dev._obs_queue = [qml.Hermitian(A, wires=[0, 1], do_queue=False)]

        for idx in range(len(dev._obs_queue)):
            dev._obs_queue[idx].return_type = qml.operation.Sample

        dev.pre_measure()

        s1 = dev.sample("Hermitian", [0, 1], [A])

        # s1 should only contain the eigenvalues of
        # the hermitian matrix
        eigvals = np.linalg.eigvalsh(A)
        assert np.allclose(sorted(list(set(s1))), sorted(eigvals), **tol)

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

    def test_sample_exception_analytic_mode(self, device, shots):
        """Tests if the sampling raises an error for sample size n=0
        """
        dev = device(1)
        dev._obs_queue = [qml.PauliZ(wires=[0], do_queue=False)]

        for idx in range(len(dev._obs_queue)):
            dev._obs_queue[idx].return_type = qml.operation.Sample

        dev.pre_measure()

        with pytest.raises(ValueError, match="Calling sample with n = 0 is not possible"):
            dev.sample("PauliZ", [0], [], n=0)

        if shots != 0:
            pytest.skip()

        # self.def.shots = 0, so this should also fail
        with pytest.raises(ValueError, match="Calling sample with n = 0 is not possible"):
            dev.sample("PauliZ", [0], [])

    def test_sample_exception_wrong_n(self, device, shots):
        """Tests if the sampling raises an error for sample size n<0
        or non-integer n
        """
        dev = device(1)
        dev._obs_queue = [qml.PauliZ(wires=[0], do_queue=False)]

        for idx in range(len(dev._obs_queue)):
            dev._obs_queue[idx].return_type = qml.operation.Sample

        dev.pre_measure()

        with pytest.raises(ValueError, match="The number of samples must be a positive integer"):
            dev.sample("PauliZ", [0], [], n=-12)

        # self.def.shots = 0, so this should also fail
        with pytest.raises(ValueError, match="The number of samples must be a positive integer"):
            dev.sample("PauliZ", [0], [], n=12.3)
