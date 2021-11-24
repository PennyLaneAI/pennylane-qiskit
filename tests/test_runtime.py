import os

import numpy as np
import pennylane as qml
import pytest

from qiskit import IBMQ
from qiskit.providers.ibmq.exceptions import IBMQAccountError

from pennylane_qiskit import IBMQDevice
from pennylane_qiskit import IBMQCircuitRunnerDevice
from pennylane_qiskit import IBMQSamplerDevice
from pennylane_qiskit import qiskit_device as qiskit_device


@pytest.fixture
def token():
    t = os.getenv("IBMQX_TOKEN_TEST", None)

    if t is None:
        pytest.skip("Skipping test, no IBMQ token available")

    yield t
    IBMQ.disable_account()


class TestCircuitRunner:

    def test_load_from_env(self, token, monkeypatch):
        """Test loading an IBMQ Circuit Runner Qiskit runtime device from an env variable."""
        monkeypatch.setenv("IBMQX_TOKEN", token)
        dev = IBMQCircuitRunnerDevice(wires=1)
        assert dev.provider.credentials.is_ibmq()

    def test_simple_circuit(self):

        return True

class TestSampler:

    def test_load_from_env(self, token, monkeypatch):
        """Test loading an IBMQ Sampler Qiskit runtime device from an env variable."""
        monkeypatch.setenv("IBMQX_TOKEN", token)
        dev = IBMQSamplerDevice(wires=1)
        assert dev.provider.credentials.is_ibmq()

    def test_simple_circuit(self):

        return True

class TestCustomVQE:

    def test_simple_circuit(self):

        return True
