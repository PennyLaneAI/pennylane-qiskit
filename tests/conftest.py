# Copyright 2021-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
This module contains tests for PennyLane runtime programs.
"""

import os
import pytest
import numpy as np
import qiskit

import pennylane as qml
from semantic_version import Version
from qiskit_ibm_provider import IBMProvider
from pennylane_qiskit import AerDevice, BasicAerDevice, BasicSimulatorDevice

np.random.seed(42)

U = np.array(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ]
)

U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / np.sqrt(3)
A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])


if Version(qiskit.__version__) < Version("1.0.0"):
    test_devices = [AerDevice, BasicAerDevice]
    hw_backends = ["qasm_simulator", "aer_simulator"]
    state_backends = [
        "statevector_simulator",
        "unitary_simulator",
    ]
else:
    test_devices = [AerDevice, BasicSimulatorDevice]
    hw_backends = ["qasm_simulator", "aer_simulator", "basic_simulator"]
    state_backends = [
        "statevector_simulator",
        "unitary_simulator",
        "aer_simulator_statevector",
        "aer_simulator_unitary",
    ]


@pytest.fixture
def skip_if_no_account():
    t = os.getenv("IBMQX_TOKEN", None)
    try:
        IBMProvider(token=t)
    except Exception:
        missing = "token" if t else "account"
        pytest.skip(f"Skipping test, no IBMQ {missing} available")


@pytest.fixture
def skip_if_account_saved():
    if IBMProvider.saved_accounts():
        pytest.skip("Skipping test, IBMQ will load an account successfully")


@pytest.fixture
def tol(shots):
    if shots is None:
        return {"atol": 0.01, "rtol": 0}

    return {"atol": 0.05, "rtol": 0.1}


@pytest.fixture
def init_state(scope="session"):
    def _init_state(n):
        state = np.random.random([2**n]) + np.random.random([2**n]) * 1j
        state /= np.linalg.norm(state)
        return state

    return _init_state


@pytest.fixture
def skip_unitary(backend):
    if "unitary" in backend:
        pytest.skip("This test does not support the unitary simulator backend.")


@pytest.fixture
def run_only_for_unitary(backend):
    if "unitary" not in backend:
        pytest.skip("This test only supports the unitary simulator.")


@pytest.fixture(params=state_backends + hw_backends)
def backend(request):
    return request.param


@pytest.fixture(params=state_backends)
def statevector_backend(request):
    return request.param


@pytest.fixture(params=hw_backends)
def hardware_backend(request):
    return request.param


@pytest.fixture(params=test_devices)
def device(request, backend, shots):
    print("getting a device")
    if backend not in state_backends:
        if shots is None:
            pytest.skip("Hardware simulators do not support analytic mode")

    if backend == "aer_simulator" and not issubclass(request.param, AerDevice):
        print("I should be skipping this test")
        pytest.skip("Only the AerDevice can use the aer_simulator backend")

    if issubclass(request.param, BasicSimulatorDevice) and backend != "basic_simulator":
        pytest.skip("BasicSimulator is the only supported backend for the BasicSimulatorDevice")

    if backend == "basic_simulator" and not issubclass(request.param, BasicSimulatorDevice):
        pytest.skip("BasicSimulator is the only supported backend for the BasicSimulatorDevice")

    def _device(n, device_options=None):
        if device_options is None:
            device_options = {}
        return request.param(wires=n, backend=backend, shots=shots, **device_options)

    return _device


@pytest.fixture(params=test_devices)
def state_vector_device(request, statevector_backend, shots):

    if backend == "aer_simulator" and not issubclass(request.param, AerDevice):
        pytest.skip("Only the AerDevice can use the aer_simulator backend")

    if issubclass(request.param, BasicSimulatorDevice) and backend != "basic_simulator":
        pytest.skip("BasicSimulator is the only supported backend for the BasicSimulatorDevice")

    if backend == "basic_simulator" and not issubclass(request.param, BasicSimulatorDevice):
        pytest.skip("BasicSimulator is the only supported backend for the BasicSimulatorDevice")

    def _device(n):
        return request.param(wires=n, backend=statevector_backend, shots=shots)

    return _device


@pytest.fixture(scope="function")
def mock_device(monkeypatch):
    """A mock instance of the abstract Device class"""

    with monkeypatch.context() as m:
        dev = qml.Device
        m.setattr(dev, "__abstractmethods__", frozenset())
        yield qml.Device()


@pytest.fixture(scope="function")
def recorder():
    return qml.tape.OperationRecorder()


@pytest.fixture(scope="function")
def qubit_device_single_wire():
    return qml.device("default.qubit", wires=1)


@pytest.fixture(scope="function")
def qubit_device_2_wires():
    return qml.device("default.qubit", wires=2)
