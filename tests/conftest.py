import pytest
import numpy as np

import pennylane as qml
from pennylane_qiskit import AerDevice, BasicAerDevice

import contextlib
import io


np.random.seed(42)

U = np.array(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ]
)

U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / np.sqrt(3)
A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])


state_backends = [
    "statevector_simulator",
    "unitary_simulator",
    "aer_simulator_statevector",
    "aer_simulator_unitary",
]
hw_backends = ["qasm_simulator", "aer_simulator"]


@pytest.fixture
def tol(shots):
    if shots is None:
        return {"atol": 0.01, "rtol": 0}

    return {"atol": 0.05, "rtol": 0.1}


@pytest.fixture
def init_state(scope="session"):
    def _init_state(n):
        state = np.random.random([2 ** n]) + np.random.random([2 ** n]) * 1j
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


@pytest.fixture(params=[AerDevice, BasicAerDevice])
def device(request, backend, shots):
    if backend not in state_backends and shots is None:
        pytest.skip("Hardware simulators do not support analytic mode")

    if (issubclass(request.param, AerDevice) and "aer" not in backend) or (
        issubclass(request.param, BasicAerDevice) and "aer" in backend
    ):
        pytest.skip("Only the AerSimulator is supported on AerDevice")

    def _device(n, device_options=None):
        if device_options is None:
            device_options = {}
        return request.param(wires=n, backend=backend, shots=shots, **device_options)

    return _device


@pytest.fixture(params=[AerDevice, BasicAerDevice])
def state_vector_device(request, statevector_backend, shots):

    if (issubclass(request.param, AerDevice) and "aer" not in statevector_backend) or (
        issubclass(request.param, BasicAerDevice) and "aer" in statevector_backend
    ):
        pytest.skip("Only the AerSimulator is supported on AerDevice")

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
