import numpy as np
import pytest

import pennylane as qml
from pennylane_qiskit import AerDevice
from pennylane_qiskit.qiskit_device import QiskitDevice
from qiskit_aer import noise

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

    def test_warning_raised_for_hardware_backend_analytic_expval(self, recorder):
        """Tests that a warning is raised if the analytic attribute is true on
        hardware simulators when calculating the expectation"""

        with pytest.warns(UserWarning) as record:
            dev = qml.device("qiskit.aer", backend="aer_simulator", wires=2, shots=None)

        # check that only one warning was raised
        assert len(record) == 1
        # check that the message matches
        assert (
            record[0].message.args[0] == "The analytic calculation of "
            "expectations, variances and probabilities is only supported on "
            "statevector backends, not on the {}. Such statistics obtained from this "
            "device are estimates based on samples.".format(dev.backend.name)
        )

    @pytest.mark.parametrize("method", ["unitary", "statevector"])
    def test_no_warning_raised_for_software_backend_analytic_expval(
        self, method, recorder, recwarn
    ):
        """Tests that no warning is raised if the analytic attribute is true on
        statevector simulators when calculating the expectation"""

        dev = qml.device("qiskit.aer", backend="aer_simulator", method=method, wires=2, shots=None)

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


@pytest.mark.parametrize("shots", [None])
class TestBatchExecution:
    """Tests for the batch_execute method."""

    with qml.tape.QuantumTape() as tape1:
        qml.PauliX(wires=0)
        qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))

    with qml.tape.QuantumTape() as tape2:
        qml.PauliX(wires=0)
        qml.expval(qml.PauliZ(wires=0))

    @pytest.mark.parametrize("n_tapes", [1, 2, 3])
    def test_calls_to_execute(self, device, n_tapes, mocker):
        """Tests that only the device's dedicated batch execute method is
        called and not the general execute method."""

        dev = device(2)
        spy = mocker.spy(QiskitDevice, "execute")

        tapes = [self.tape1] * n_tapes
        dev.batch_execute(tapes)

        # Check that QiskitDevice.execute was not called
        assert spy.call_count == 0

    @pytest.mark.parametrize("n_tapes", [1, 2, 3])
    def test_calls_to_reset(self, n_tapes, mocker, device):
        """Tests that the device's reset method is called the correct number of
        times."""

        dev = device(2)
        spy = mocker.spy(QiskitDevice, "reset")

        tapes = [self.tape1] * n_tapes
        dev.batch_execute(tapes)

        assert spy.call_count == n_tapes

    def test_result(self, device, tol):
        """Tests that the result has the correct shape and entry types."""
        dev = device(2)
        tapes = [self.tape1, self.tape2]
        res = dev.batch_execute(tapes)

        # We're calling device methods directly, need to reset before the next
        # execution
        dev.reset()
        tape1_expected = dev.execute(self.tape1)

        dev.reset()
        tape2_expected = dev.execute(self.tape2)

        assert len(res) == 2
        assert isinstance(res[0], tuple)
        assert len(res[0]) == 2
        assert np.allclose(qml.math.stack(res[0]), tape1_expected, atol=0)

        assert isinstance(res[1], np.ndarray)
        assert np.allclose(res[1], tape2_expected, atol=0)

    def test_result_no_tapes(self, device):
        """Tests that the result is correct when there are no tapes to execute."""
        dev = device(2)
        res = dev.batch_execute([])

        # We're calling device methods directly, need to reset before the next
        # execution
        dev.reset()
        assert not res

    def test_result_empty_tape(self, device, tol):
        """Tests that the result has the correct shape and entry types for
        empty tapes."""
        dev = device(2)

        empty_tape = qml.tape.QuantumTape()
        tapes = [empty_tape] * 3
        res = dev.batch_execute(tapes)

        # We're calling device methods directly, need to reset before the next
        # execution
        dev.reset()
        assert len(res) == 3
        assert np.allclose(res[0], dev.execute(empty_tape), atol=0)

    def test_num_executions_recorded(self, device):
        """Tests that the number of executions are recorded correctly.."""
        dev = device(2)
        tapes = [self.tape1, self.tape2]
        res = dev.batch_execute(tapes)
        assert dev.num_executions == 1

    def test_barrier_tape(self, device, tol):
        """Tests that the barriers are accounted for during conversion."""
        dev = device(2)

        @qml.qnode(dev)
        def barrier_func():
            qml.Barrier([0, 1])
            return qml.state()

        res = barrier_func()
        assert barrier_func.tape.operations[0] == qml.Barrier([0, 1])
        assert np.allclose(res, dev.batch_execute([barrier_func.tape]), atol=0)
