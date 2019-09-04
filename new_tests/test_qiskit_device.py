import pytest

import numpy as np

from pennylane_qiskit.qiskit_device import z_eigs


Z = np.diag([1, -1])


class TestZEigs:
    r"""Test that eigenvalues of Z^{\otimes n} are correctly generated"""

    def test_one(self):
        """Test that eigs(Z) = [1, -1]"""
        assert np.all(z_eigs(1) == np.array([1, -1]))

    @pytest.mark.parametrize("n", [2, 3, 6])
    def test_multiple(self, n):
        r"""Test that eigs(Z^{\otimes n}) is correct"""
        res = z_eigs(n)
        Zn = np.kron(Z, Z)

        for _ in range(n-2):
            Zn = np.kron(Zn, Z)

        expected = np.diag(Zn)
        assert np.all(res == expected)
