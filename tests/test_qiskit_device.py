import pytest

import numpy as np

from pennylane_qiskit.qiskit_device import pauli_eigs
from pennylane_qiskit import AerDevice


Z = np.diag([1, -1])


class TestZEigs:
    r"""Test that eigenvalues of Z^{\otimes n} are correctly generated"""

    def test_one(self):
        """Test that eigs(Z) = [1, -1]"""
        assert np.all(pauli_eigs(1) == np.array([1, -1]))

    @pytest.mark.parametrize("n", [2, 3, 6])
    def test_multiple(self, n):
        r"""Test that eigs(Z^{\otimes n}) is correct"""
        res = pauli_eigs(n)
        Zn = np.kron(Z, Z)

        for _ in range(n - 2):
            Zn = np.kron(Zn, Z)

        expected = np.diag(Zn)
        assert np.all(res == expected)


class TestProbabilities:
    """Tests for the probability function"""

    def test_probability_no_results(self):
        """Test that the probabilities function returns
        None if no job has yet been run."""
        dev = AerDevice(backend="statevector_simulator", wires=1, analytic=True)
        assert dev.probabilities() is None


class TestAnalyticWarningHWSimulator:
    """Tests that a warning is raised if the analytic attribute is true on
    hardware simulators"""

    def test_warning_raised_for_analytic_expval(self, backend, recorder):

        import pennylane as qml

        dev = qml.device("qiskit.basicaer", backend=backend, wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.warns(UserWarning) as record:
            with recorder:
                circuit()

        # check that only one warning was raised
        assert len(record) == 1
        # check that the message matches
        assert record[0].message.args[0] == "The analytic calculation of expectations and variances "\
                                            "is only supported on statevector backends, not on the {}. "\
                                            "The obtained result is based on sampling.".format(dev.backend)
