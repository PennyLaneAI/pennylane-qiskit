import numpy as np
import pytest

import pennylane as qml
from pennylane_qiskit import AerDevice, BasicAerDevice


class TestProbabilities:
    """Tests for the probability function"""

    def test_probability_no_results(self):
        """Test that the probabilities function returns
        None if no job has yet been run."""
        dev = AerDevice(backend="statevector_simulator", wires=1, analytic=True)
        assert dev.probability() is None

@pytest.mark.parametrize("qiskit_device", [AerDevice, BasicAerDevice])
@pytest.mark.parametrize("wires", [1,2])
@pytest.mark.transpile_args_test
class TestTranspilationOptionInitialization:
    """Tests for passing the transpilation options to qiskit at time of device init"""

    def test_no_transpilation_options(self, qiskit_device, wires):
        """Test that the transpilation options must me {} if not provided."""
        dev = qiskit_device(wires=wires)
        assert dev.transpile_args == {}

    def test_with_transpilation_options(self, qiskit_device, wires):
        """test that the transpilation options are set as expected during device init"""
        dev = qiskit_device(wires=wires, abc=123, optimization_level=2)
        assert dev.transpile_args == {'optimization_level':2}

class TestAnalyticWarningHWSimulator:
    """Tests the warnings for when the analytic attribute of a device is set to true"""

    def test_warning_raised_for_hardware_backend_analytic_expval(self, hardware_backend, recorder):
        """Tests that a warning is raised if the analytic attribute is true on
            hardware simulators when calculating the expectation"""

        with pytest.warns(UserWarning) as record:
            dev = qml.device("qiskit.basicaer", backend=hardware_backend, wires=2, analytic=True)

        # check that only one warning was raised
        assert len(record) == 1
        # check that the message matches
        assert record[0].message.args[0] == "The analytic calculation of "\
                "expectations, variances and probabilities is only supported on "\
                "statevector backends, not on the {}. Such statistics obtained from this "\
                "device are estimates based on samples.".format(dev.backend)

    def test_no_warning_raised_for_software_backend_analytic_expval(self, statevector_backend, recorder, recwarn):
        """Tests that no warning is raised if the analytic attribute is true on
            statevector simulators when calculating the expectation"""

        dev = qml.device("qiskit.basicaer", backend=statevector_backend, wires=2, analytic=True)

        # check that no warnings were raised
        assert len(recwarn) == 0
