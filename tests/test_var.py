import pytest

import numpy as np
import pennylane as qml

from pennylane_qiskit import AerDevice, BasicAerDevice

from conftest import U, U2, A


np.random.seed(42)


@pytest.mark.parametrize("shots", [0, 8192])
class TestVar:
    """Tests for the variance"""

    def test_var(self, device, shots, tol):
        """Tests for variance calculation"""
        dev = device(2)
        dev.active_wires = {0}

        phi = 0.543
        theta = 0.6543

        # test correct variance for <Z> of a rotated state
        dev.apply("RX", wires=[0], par=[phi])
        dev.apply("RY", wires=[0], par=[theta])

        dev._obs_queue = [qml.PauliZ(wires=[0], do_queue=False)]
        dev.pre_measure()

        var = dev.var("PauliZ", [0], [])
        expected = 0.25 * (3 - np.cos(2 * theta) - 2 * np.cos(theta) ** 2 * np.cos(2 * phi))

        assert np.allclose(var, expected, **tol)

    def test_var_hermitian(self, device, shots, tol):
        """Tests for variance calculation using an arbitrary Hermitian observable"""
        dev = device(2)
        dev.active_wires = {0}

        phi = 0.543
        theta = 0.6543

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
