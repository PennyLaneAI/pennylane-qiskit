import numpy as np
import pytest

import pennylane as qml
from pennylane_qiskit import AerDevice
import qiskit.providers.aer.noise as noise

test_transpile_options = [
    {},
    {"optimization_level": 2},
    {"optimization_level": 2, "seed_transpiler": 22},
]

test_device_options = [{}, {"optimization_level": 3}, {"optimization_level": 1}]


class TestProbabilities:
    """Tests for the probability function"""

    def test_probability_no_results(self):
        """Test that the probabilities function returns
        None if no job has yet been run."""
        dev = AerDevice(backend="aer_simulator_statevector", wires=1, shots=None)
        assert dev.probability() is None


@pytest.mark.parametrize("wires", [1, 2, 3])
@pytest.mark.parametrize("shots", [None])
@pytest.mark.parametrize("device_options", test_device_options)
class TestTranspilationOptionInitialization:
    """Tests for passing the transpilation options to qiskit at time of device
    initialization."""

    def test_device_with_transpilation_options(self, device, wires, device_options):
        """Test that the transpilation options must be persisted if provided."""
        dev = device(wires, device_options)
        assert dev.transpile_args == device_options

    @pytest.mark.parametrize("transpile_options", test_transpile_options)
    def test_transpilation_option_update(self, device, wires, device_options, transpile_options):
        """Test that the transpilation options are updated as expected."""
        dev = device(wires, device_options)
        assert dev.transpile_args == device_options
        dev.set_transpile_args(**transpile_options)
        assert dev.transpile_args == transpile_options


class TestAnalyticWarningHWSimulator:
    """Tests the warnings for when the analytic attribute of a device is set to true"""

    def test_warning_raised_for_hardware_backend_analytic_expval(self, hardware_backend, recorder):
        """Tests that a warning is raised if the analytic attribute is true on
        hardware simulators when calculating the expectation"""
        if "aer" in hardware_backend:
            pytest.skip("Not supported on basicaer")

        with pytest.warns(UserWarning) as record:
            dev = qml.device("qiskit.basicaer", backend=hardware_backend, wires=2, shots=None)

        # check that only one warning was raised
        assert len(record) == 1
        # check that the message matches
        assert (
            record[0].message.args[0] == "The analytic calculation of "
            "expectations, variances and probabilities is only supported on "
            "statevector backends, not on the {}. Such statistics obtained from this "
            "device are estimates based on samples.".format(dev.backend)
        )

    def test_no_warning_raised_for_software_backend_analytic_expval(
        self, statevector_backend, recorder, recwarn
    ):
        """Tests that no warning is raised if the analytic attribute is true on
        statevector simulators when calculating the expectation"""
        if "aer" in statevector_backend:
            pytest.skip("Not supported on basicaer")

        dev = qml.device("qiskit.basicaer", backend=statevector_backend, wires=2, shots=None)

        # check that no warnings were raised
        assert len(recwarn) == 0


class TestAerBackendOptions:
    """Test the backend options of Aer backends."""

    def test_backend_options_cleaned(self):
        """Test that the backend options are cleared upon new Aer device
        initialization."""
        noise_model = noise.NoiseModel()
        bit_flip = noise.pauli_error([("X", 1), ("I", 0)])

        # Create a noise model where the RX operation always flips the bit
        noise_model.add_all_qubit_quantum_error(bit_flip, ["rx"])

        dev = qml.device("qiskit.aer", wires=2, noise_model=noise_model)
        assert dev.backend.options.get("noise_model") is not None

        dev2 = qml.device("qiskit.aer", wires=2)
        assert dev2.backend.options.get("noise_model") is None
