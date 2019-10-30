import pytest
import numpy as np

import pennylane as qml
from pennylane_qiskit import AerDevice, BasicAerDevice

import contextlib
import io


# TODO: once in PennyLane, modify the usage of Recorder and  OperationRecorder
class Recorder:
    """Recorder class used by the :class:`~.OperationRecorder`.

    The Recorder class is a very minimal QNode, that simply
    acts as a QNode context for operator queueing."""
    # pylint: disable=too-few-public-methods
    def __init__(self, old_context):
        self.old_context = old_context
        self.ops = []
        self.queue = []
        self.ev = []
        self.num_wires = 1

    def _append_op(self, op):
        """:class:`~.Operator` objects call this method
        and append themselves upon initialization."""
        self.ops.append(op)

        if isinstance(op, qml.operation.Observable):
            if op.return_type is None:
                self.queue.append(op)
            else:
                self.ev.append(op)
        else:
            self.queue.append(op)

        # this ensure the recorder does not interfere with
        # any QNode contexts
        if self.old_context:
            self.old_context._append_op(op)


class OperationRecorder:
    """A template and quantum function inspector,
    allowing easy introspection of operators that have been
    applied without requiring a QNode.

    **Example**:

    The OperationRecorder is a context manager. Executing templates
    or quantum functions stores resulting applied operators in the
    recorder, which can then be printed.

    >>> weights = qml.init.strong_ent_layers_normal(n_layers=1, n_wires=2)
    >>>
    >>> with qml.utils.OperationRecorder() as rec:
    >>>    qml.templates.layers.StronglyEntanglingLayers(*weights, wires=[0, 1])
    >>>
    >>> print(rec)
    Operations
    ==========
    Rot(-0.10832656163640327, 0.14429091013664083, -0.010835826725765343, wires=[0])
    Rot(-0.11254523669444501, 0.0947222564914006, -0.09139600968423377, wires=[1])
    CNOT(wires=[0, 1])
    CNOT(wires=[1, 0])

    Alternatively, the :attr:`~.OperationRecorder.queue` attribute can be used
    to directly accessed the applied :class:`~.Operation` and :class:`~.Observable`
    objects.
    """
    def __init__(self):
        self.rec = None
        """~.Recorder: a very minimal QNode, that simply
        acts as a QNode context for operator queueing"""

        self.queue = None
        """List[~.Operators]: list of operations applied within
        the OperatorRecorder context."""

        self.old_context = None

    def __enter__(self):
        self.rec = Recorder(qml.QNode._current_context)

        # store the old context to be returned later
        self.old_context = qml.QNode._current_context

        # set the recorder as the QNode context
        qml.QNode._current_context = self.rec

        self.queue = None

        return self

    def __exit__(self, *args, **kwargs):
        self.queue = self.rec.ops
        qml.QNode._current_context = self.old_context

    def __str__(self):
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            qml.QNode.print_applied(self.rec)
            output = buf.getvalue()

        return output


np.random.seed(42)

U = np.array(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ]
)

U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / np.sqrt(3)
A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])


state_backends = ["statevector_simulator", "unitary_simulator"]
hw_backends = ["qasm_simulator"]


@pytest.fixture
def tol(analytic):
    if analytic:
        return {"atol": 0.01, "rtol": 0}

    return {"atol": 0.05, "rtol": 0.1}


@pytest.fixture
def init_state(scope="session"):
    def _init_state(n):
        state = np.random.random([2 ** n]) + np.random.random([2 ** n]) * 1j
        state /= np.linalg.norm(state)
        return state

    return _init_state


@pytest.fixture(params=state_backends + hw_backends)
def backend(request):
    return request.param


@pytest.fixture(params=[AerDevice, BasicAerDevice])
def device(request, backend, shots, analytic):
    if backend not in state_backends and analytic == True:
        pytest.skip("Hardware simulators do not support analytic mode")

    def _device(n):
        return request.param(wires=n, backend=backend, shots=shots, analytic=analytic)

    return _device


@pytest.fixture(scope="function")
def mock_device(monkeypatch):
    """A mock instance of the abstract Device class"""

    with monkeypatch.context() as m:
        dev = qml.Device
        m.setattr(dev, '__abstractmethods__', frozenset())
        yield qml.Device()


@pytest.fixture(scope="function")
def recorder():
    return OperationRecorder()
