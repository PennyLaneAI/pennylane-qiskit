import pytest
import pennylane as qml
import qiskit

from unittest.mock import Mock


@pytest.mark.parametrize(
    "device_name",
    [
        "qiskit.aer",
        "qiskit.basicaer",
        "qiskit.remote",
        "qiskit.ibmq",
        "qiskit.ibmq.circuit_runner",
        "qiskit.ibmq.sampler",
    ],
)
def test_error_is_raised_if_initalizing_device(monkeypatch, device_name):
    """Test that when Qiskit 1.0 is installed, an error is raised if you try
    to initialize a device. This is a temporary test and will be removed along
    with the error one everything is compatible with 1.0"""

    # make it not fail in the normal tests (running 0.46)
    if qiskit.__version__ != "1.0.0":
        monkeypatch.setattr(qiskit, "__version__", "1.0.0")

    # test that the correct error is actually raised in Qiskit 1.0 (rather than fx an import error)
    with pytest.raises(
        RuntimeError,
        match="The devices in the PennyLane Qiskit plugin are currently only compatible with versions of Qiskit below 0.46",
    ):
        if device_name in ["qiskit.aer", "qiskit.basicaer"]:
            qml.device(device_name, wires=2)
        else:
            # use a Mock backend to avoid call to the remote service
            qml.device(device_name, wires=2, backend=Mock())

