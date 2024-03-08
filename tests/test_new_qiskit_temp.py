import pytest
import pennylane as qml
import qiskit

from unittest.mock import Mock


def test_error_is_raised_if_initalizing_device(monkeypatch):
    """Test that when Qiskit 1.0 is installed, an error is raised if you try
    to initialize the 'qiskit.basicaer' device."""

    # skip in the 0.X.X tests
    pytest.skipif(Version(qiskit.__version__) < Version("1.0.0"))

    # test that the correct error is actually raised in Qiskit 1.0 (rather than fx an import error)
    with pytest.raises(
        RuntimeError,
        match="Qiskit has discontinued the BasicAer device",
    ):
        qml.device("qiskit.basicaer", wires=2)

