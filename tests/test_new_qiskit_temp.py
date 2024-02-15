import pytest
import pennylane as qml
import qiskit


def test_error_is_raised_if_initalizing_device():
    """Test that when Qiskit 1.0 is installed, an error is raised if you try
    to initialize a device. This is a temporary test and will be removed along
    with the error one everything is compatible with 1.0"""
    if qiskit.__version__ != "1.0.0":
        pass

    else:
        with pytest.raises(
            RuntimeError,
            match="The devices in the PennyLane Qiskit plugin are currently only compatible with version of Qiskit below 0.46",
        ):
            qml.device("qiskit.aer", wires=2)
