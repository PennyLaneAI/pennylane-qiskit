import pytest
import pennylane as qml
import qiskit

from semantic_version import Version
from unittest.mock import Mock

from pennylane_qiskit import BasicSimulatorDevice


@pytest.mark.skipif(
    Version(qiskit.__version__) < Version("1.0.0"),
    reason="versions below 1.0 are compatible with BasicAer",
)
def test_error_is_raised_if_initalizing_basicaer_device(monkeypatch):
    """Test that when Qiskit 1.0 is installed, an error is raised if you try
    to initialize the 'qiskit.basicaer' device."""

    # test that the correct error is actually raised in Qiskit 1.0 (rather than fx an import error)
    with pytest.raises(
        RuntimeError,
        match="Qiskit has discontinued the BasicAer device",
    ):
        qml.device("qiskit.basicaer", wires=2)


@pytest.mark.skipif(
    Version(qiskit.__version__) >= Version("1.0.0"),
    reason="versions 1.0 and above are compatible with BasicSimulator",
)
def test_error_is_raised_if_initalizing_basic_simulator_device(monkeypatch):
    """Test that when a version of Qiskit below 1.0 is installed, an error is raised if you try
    to initialize the BasicSimulatorDevice."""

    # test that the correct error is actually raised in Qiskit 1.0 (rather than fx an import error)
    with pytest.raises(
        RuntimeError,
        match="device is not compatible with version of Qiskit prior to 1.0",
    ):
        BasicSimulatorDevice(wires=2)
