# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the :class:`~.RemoteDevice` class, a PennyLane device that allows
evaluation and differentiation on any Qiskit backend using Pennylane.
"""

from .qiskit_device import QiskitDevice


class RemoteDevice(QiskitDevice):
    """A PennyLane device for any Qiskit backend.

    Args:
        wires (int or Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['aux_wire', 'q1', 'q2']``).
        backend (Backend): the initialized Qiskit backend

    Keyword Args:
        shots (Union[int, None]): number of circuit evaluations/random samples used
            to estimate expectation values and variances of observables.
        session (Session): a Qiskit Session to use for device execution. If none is provided,
            a session will be created at each device execution.
        compile_backend (Union[Backend, None]): the backend to be used for compiling the circuit
            that will be sent to the backend device, to be set if the backend desired for
            compilation differs from the backend used for execution. Defaults to ``None``,
            which means the primary backend will be used.
        **kwargs: transpilation and runtime keyword arguments to be used for measurements with
            Primitives. If an `options` dictionary is defined amongst the kwargs, and there are
            settings that overlap with those in kwargs, the settings in `options` will take
            precedence over kwargs. Keyword arguments accepted by both the transpiler and at
            runtime (e.g. ``optimization_level``) will be passed to the transpiler rather
            than to the Primitive.

    **Example:**

    .. code-block:: python

        import pennylane as qml
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        backend = service.least_busy(n_qubits=127, simulator=False, operational=True)
        dev = qml.device("qiskit.remote", wires=127, backend=backend)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

    >>> circuit(np.pi/3, shots=1024)
    0.529296875

    This device also supports the use of local simulators such as ``AerSimulator`` or
    fake backends such as ``FakeManila``.

    .. code-block:: python

        import pennylane as qml
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        dev = qml.device("qiskit.remote", wires=5, backend=backend)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

    >>> circuit(np.pi/3, shots=1024)
    0.49755859375

    We can also change the number of shots, either when initializing the device or when we execute
    the circuit. Note that the shots number specified on circuit execution will override whatever
    was set on device initialization.

    .. code-block:: python

        dev = qml.device("qiskit.remote", wires=5, backend=backend, shots=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(1))

    >>> circuit(np.pi/3) # this will run with 2 shots
    array([-1.,  1.])

    >>> circuit(np.pi/3, shots=5) # this will run with 5 shots
    array([-1., -1.,  1.,  1.,  1.])

    >>> circuit(np.pi/3) # this will run with 2 shots
    array([-1.,  1.])

    Internally, the device uses the `EstimatorV2 <https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.EstimatorV2/>`_
    and the `SamplerV2 <https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.SamplerV2>`_
    runtime primitives to execute the measurements. To set options for
    `transpilation <https://docs.quantum.ibm.com/run/configure-runtime-compilation>`_ or
    `runtime <https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/options>`_, simply pass
    the keyword arguments into the device. If you wish to change options other than ``shots``,
    PennyLane requires you to re-initialize the device to do so.

    .. code-block:: python

        import pennylane as qml
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2

        backend = FakeManilaV2()
        dev = qml.device(
            "qiskit.remote",
            wires=5,
            backend=backend,
            resilience_level=1,
            optimization_level=1,
            seed_transpiler=42,
        )
        # to change options, re-initialize the device
        dev = qml.device(
            "qiskit.remote",
            wires=5,
            backend=backend,
            resilience_level=1,
            optimization_level=2,
            seed_transpiler=24,
        )
    """

    short_name = "qiskit.remote"

    def __init__(self, wires, backend, shots=None, **kwargs):
        super().__init__(wires, backend=backend, shots=shots, **kwargs)
