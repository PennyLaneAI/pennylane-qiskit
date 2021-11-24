import os

import numpy as np
import pennylane as qml
import pytest

from qiskit import IBMQ
from qiskit.providers.ibmq.exceptions import IBMQAccountError

from pennylane_qiskit import IBMQDevice
from pennylane_qiskit import ibmq as ibmq
from pennylane_qiskit import qiskit_device as qiskit_device


@pytest.fixture
def token():
    t = os.getenv("IBMQX_TOKEN_TEST", None)

    if t is None:
        pytest.skip("Skipping test, no IBMQ token available")

    yield t
    IBMQ.disable_account()


class TestCircuitRunner:
    def test_simple_circuit(self, token):

        return True

class TestSampler:
    def test_simple_circuit(self, token):

        return True

class TestCustomVQE:

    def test_simple_circuit(self, token):

        return True
