# Copyright 2018 Carsten Blank

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
Devices
=======

.. currentmodule:: pennylane_qiskit.devices

This plugin offers access to the following qiskit backends by providing
corresponding PennyLane devices:

.. autosummary::
   :nosignatures:

   AerQiskitDevice
   IbmQQiskitDevice

See below for a description of the devices and the supported Operations and Expectations.

AerQiskitDevice
#################

.. autoclass:: AerQiskitDevice

IbmQQiskitDevice
##################

.. autoclass:: IbmQQiskitDevice

"""
from time import sleep
from typing import Dict, Sequence, Union, Any, List

import qiskit
from pennylane import Device, DeviceError
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.backends import BaseProvider, BaseBackend, BaseJob, JobStatus
from qiskit.extensions.standard import (x, y, z)
from qiskit.result import Result
from qiskit.unroll import CircuitBackend

from ._version import __version__
from .pqops import BasisState, QiskitInstructions, Rot, QubitUnitary, QubitStateVector

QISKIT_OPERATION_MAP = {
    # native PennyLane operations also native to qiskit
    'PauliX': 'x',
    'PauliY': 'y',
    'PauliZ': 'z',
    'CNOT': 'cx',
    'CZ': 'cz',
    'SWAP': 'swap',
    'RX': 'rx',
    'RY': 'ry',
    'RZ': 'rz',
    'PhaseShift': 'u1',
    'Hadamard': 'h',

    # operations not natively implemented in qiskit but provided in pqops.py
    'Rot': Rot(),
    'BasisState': BasisState(),
    'QubitStateVector': QubitStateVector(),
    'QubitUnitary': QubitUnitary(),

    # additional operations not native to PennyLane but present in qiskit
    'S': 's',
    'T': 't'
}  # type: Dict[str, Union[str, QiskitInstructions]]


class QiskitDevice(Device):
    name = 'Qiskit PennyLane plugin'
    short_name = 'qiskit'
    pennylane_requires = '0.1.0'
    version = '0.1.0'
    plugin_version = __version__
    author = 'Carsten Blank'
    _capabilities = {
        'model': 'qubit'
    }  # type: Dict[str, any]
    _operation_map = QISKIT_OPERATION_MAP
    _expectation_map = {key: val for key, val in _operation_map.items()
                        if val in [x, y, z]}
    _backend_kwargs = ['num_runs', 'verbose', 'backend']

    def __init__(self, wires, backend, shots=1024, **kwargs):
        super().__init__(wires=wires, shots=shots)

        # translate some arguments
        for key, val in {'log': 'verbose'}.items():
            if key in kwargs:
                kwargs[val] = kwargs[key]

        # clean some arguments
        if 'num_runs' in kwargs and isinstance(kwargs['num_runs'], int) and kwargs['num_runs'] > 0:
            self.shots = kwargs['num_runs']
        else:
            kwargs['num_runs'] = self.shots

        kwargs['backend'] = backend
        self.backend = kwargs['backend']
        self.compile_backend = kwargs['compile_backend'] if 'compile_backend' in kwargs else self.backend
        self.name = kwargs.get('name', 'circuit')
        self.kwargs = kwargs

        # Inner state
        self._reg = QuantumRegister(wires, "q")
        self._creg = ClassicalRegister(wires, "c")
        self._provider = None  # type: BaseProvider
        self._dagcircuit = CircuitBackend()  # type: CircuitBackend
        self._current_job = None  # type: BaseJob
        self._first_operation = True
        self.reset()

    @property
    def operations(self):
        return set(self._operation_map.keys())

    @property
    def expectations(self):
        return set(self._expectation_map.keys())

    def apply(self, operation, wires, par):
        # type: (Any, Sequence[int], List) -> None
        """Apply a quantum operation.

        For plugin developers: this function should apply the operation on the device.

        Args:
            operation (str): name of the operation
            wires (Sequence[int]): subsystems the operation is applied on
            par (tuple): parameters for the operation
        """
        operation = self._operation_map[operation]

        if isinstance(operation, BasisState) and not self._first_operation:
            raise DeviceError("Operation {} cannot be used after other Operations have already been applied "
                              "on a {} device.".format(operation, self.short_name))
        self._first_operation = False

        if isinstance(operation, str):
            qureg = [("q", i) for i in wires]
            import qiskit.qasm._node as node
            real_params = [node.Real(p) for p in par]
            self._dagcircuit.start_gate(operation, real_params, qureg)
            self._dagcircuit.end_gate(operation, real_params, qureg)
        elif isinstance(operation, QiskitInstructions):
            op = operation  # type: QiskitInstructions
            qregs = [(self._reg, i) for i in wires]
            op.apply(qregs=qregs, param=list(par), circuit=self._dagcircuit.circuit)
        else:
            raise ValueError("The operation is not of an expected type. This is a software bug!")

    def pre_expval(self):
        compile_backend = self._provider.get_backend(self.compile_backend)  # type: BaseBackend
        for qr, cr in zip(self._reg, self._creg):
            self._dagcircuit.circuit.measure(qr, cr)
        qobj = qiskit.compile(circuits=self._dagcircuit.circuit, backend=compile_backend, shots=self.shots)
        backend = self._provider.get_backend(self.backend)  #type: BaseBackend
        try:
            self._current_job = backend.run(qobj)  # type: BaseJob
            sleep(0.1)
            not_done = [JobStatus.INITIALIZING, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.VALIDATING]
            while self._current_job.status() in not_done:
                sleep(2)
        except Exception as ex:
            raise Exception("Error during job execution: {}!".format(ex))

    def expval(self, expectation, wires, par):
        result = self._current_job.result()  # type: Result

        probabilities = dict((state[::-1], count/self.shots) for state, count in result.get_counts().items())

        expval = None

        if expectation == 'PauliZ':
            if isinstance(wires, int):
                wire = wires
            else:
                wire = wires[0]

            zero = sum(p for (state, p) in probabilities.items() if state[wire] == '0')
            one = sum(p for (state, p) in probabilities.items() if state[wire] == '1')

            expval = (1-(2*one)-(1-2*zero))/2

        return expval

    def reset(self):
        self._dagcircuit = CircuitBackend()  #type: CircuitBackend
        self._dagcircuit.new_qreg(name="q", size=self.num_wires)
        self._dagcircuit.new_creg(name="c", size=self.num_wires)
        self._dagcircuit.set_basis(list(self._dagcircuit.circuit.definitions.keys()))
        for name, definition in self._dagcircuit.circuit.definitions.items():
            self._dagcircuit.define_gate(name, definition)
        self._first_operation = True


class AerQiskitDevice(QiskitDevice):
    """A PennyLane :code:`qiskit.aer` device for the `Qiskit Local Simulator` backend.

    Args:
       wires (int): The number of qubits of the device

    Keyword Args:
      backend (str): the desired backend to run the code on. Default is :code:`qasm_simulator`.

    This device can, for example, be instantiated from PennyLane as follows:

    .. code-block:: python

        import pennylane as qml
        dev = qml.device('qiskit.aer', wires=XXX)

    Supported PennyLane Operations:
      :class:`pennylane.PauliX`,
      :class:`pennylane.PauliY`,
      :class:`pennylane.PauliZ`,
      :class:`pennylane.CNOT`,
      :class:`pennylane.CZ`,
      :class:`pennylane.SWAP`,
      :class:`pennylane.RX`,
      :class:`pennylane.RY`,
      :class:`pennylane.RZ`,
      :class:`pennylane.PhaseShift`,
      :class:`pennylane.QubitStateVector`,
      :class:`pennylane.Hadamard`,
      :class:`pennylane.Rot`,
      :class:`pennylane.QubitUnitary`,
      :class:`pennylane.BasisState`

    Supported PennyLane Expectations:
      :class:`pennylane.PauliZ`

    Extra Operations:
      :class:`pennylane_pq.S <pennylane_qiskit.ops.S>`,
      :class:`pennylane_pq.T <pennylane_qiskit.ops.T>`

    ..

    """
    short_name = 'qiskit.aer'

    def __init__(self, wires, shots=1024, **kwargs):
        backend = kwargs.get('backend', 'qasm_simulator')
        super().__init__(wires, backend=backend, shots=shots, **kwargs)
        self._provider = qiskit.Aer
        self._capabilities['backend'] = [b.name() for b in self._provider.available_backends()]


class IbmQQiskitDevice(QiskitDevice):
    """A PennyLane :code:`qiskit.ibm` device for the `Qiskit Local Simulator` backend.

    Args:
       wires (int): The number of qubits of the device
       ibmqx_token (str): The IBMQ API token

    Keyword Args:
      backend (str): the desired backend to run the code on. Default is :code:`ibmq_qasm_simulator`.

    This device can, for example, be instantiated from PennyLane as follows:

    .. code-block:: python

        import pennylane as qml
        dev = qml.device('qiskit.ibm', wires=XXX, ibmqx_token='XXX')

    Supported PennyLane Operations:
      :class:`pennylane.PauliX`,
      :class:`pennylane.PauliY`,
      :class:`pennylane.PauliZ`,
      :class:`pennylane.CNOT`,
      :class:`pennylane.CZ`,
      :class:`pennylane.SWAP`,
      :class:`pennylane.RX`,
      :class:`pennylane.RY`,
      :class:`pennylane.RZ`,
      :class:`pennylane.PhaseShift`,
      :class:`pennylane.QubitStateVector`,
      :class:`pennylane.Hadamard`,
      :class:`pennylane.Rot`,
      :class:`pennylane.QubitUnitary`,
      :class:`pennylane.BasisState`

    Supported PennyLane Expectations:
      :class:`pennylane.PauliZ`

    Extra Operations:
      :class:`pennylane_pq.S <pennylane_qiskit.ops.S>`,
      :class:`pennylane_pq.T <pennylane_qiskit.ops.T>`

    ..
    """
    short_name = 'qiskit.ibmq'
    _backend_kwargs = ['num_runs', 'verbose', 'backend', 'ibmqx_token']

    def __init__(self, wires, shots=1024, **kwargs):
        if 'ibmqx_token' not in kwargs:
            raise ValueError("IBMQX Token is missing!")
        backend = kwargs.get('backend', 'ibmq_qasm_simulator')
        super().__init__(wires, backend=backend, shots=shots, **kwargs)
        self._provider = qiskit.IBMQ
        if kwargs['ibmqx_token'] not in map(lambda e: e['token'], self._provider.active_accounts()):
            self._provider.enable_account(kwargs['ibmqx_token'])
        self._capabilities['backend'] = [b.name() for b in self._provider.available_backends()]
