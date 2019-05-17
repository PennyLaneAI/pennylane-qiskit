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
    BasicAerQiskitDevice
    IbmQQiskitDevice

See below for a description of the devices and the supported Operations and Expectations.

AerQiskitDevice
###############

.. autoclass:: AerQiskitDevice

BasicAerQiskitDevice
####################

.. autoclass:: BasicAerQiskitDevice

IbmQQiskitDevice
################

.. autoclass:: IbmQQiskitDevice

"""
import os
import inspect
from typing import Dict, Sequence, Any, List, Union, Optional, Type

import qiskit
import qiskit.compiler
from pennylane import Device, DeviceError
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Instruction, Gate
from qiskit.circuit.measure import measure
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.extensions import XGate, RXGate, U1Gate, HGate, RYGate, RZGate, CzGate, CnotGate, YGate, ZGate, SGate, \
    TGate, U2Gate, U3Gate, SwapGate
from qiskit.extensions.standard import (x, y, z)
from qiskit.providers import BaseProvider, BaseJob, BaseBackend, JobStatus
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.result import Result

from ._version import __version__
from .qiskitops import BasisState, Rot, QubitStateVector, QubitUnitary, QiskitInstructions

QISKIT_OPERATION_MAP = {
    # native PennyLane operations also native to qiskit
    'PauliX': XGate,
    'PauliY': YGate,
    'PauliZ': ZGate,
    'CNOT': CnotGate,
    'CZ': CzGate,
    'SWAP': SwapGate,
    'RX': RXGate,
    'RY': RYGate,
    'RZ': RZGate,
    'PhaseShift': U1Gate,
    'Hadamard': HGate,

    # operations not natively implemented in qiskit but provided in pqops.py
    'Rot': Rot(),
    'BasisState': BasisState(),
    'QubitStateVector': QubitStateVector(),
    'QubitUnitary': QubitUnitary(),

    # additional operations not native to PennyLane but present in qiskit
    'S': SGate,
    'T': TGate,
    'U1': U1Gate,
    'U2': U2Gate,
    'U3': U3Gate
}  # type: Dict[str, Union[Type[Gate], QiskitInstructions]]


class QiskitDevice(Device):
    name = 'Qiskit PennyLane plugin'
    short_name = 'qiskit'
    pennylane_requires = '>=0.1.0'
    version = '0.1.0'
    plugin_version = __version__
    author = 'Carsten Blank'
    _capabilities = {
        'model': 'qubit'
    }  # type: Dict[str, any]
    _operation_map = QISKIT_OPERATION_MAP
    _expectation_map = {key: val for key, val in _operation_map.items()
                        if val in [x, y, z]}
    _backend_kwargs = ['verbose', 'backend']

    def __init__(self, wires, backend, shots=1024, **kwargs):
        super().__init__(wires=wires, shots=shots)

        if 'verbose' not in kwargs:
            kwargs['verbose'] = False

        kwargs['backend'] = backend
        self.backend = kwargs['backend']
        self.compile_backend = kwargs['compile_backend'] if 'compile_backend' in kwargs else self.backend
        self.name = kwargs.get('name', 'circuit')
        self.kwargs = kwargs

        # Inner state
        self._reg = QuantumRegister(wires, "q")
        self._creg = ClassicalRegister(wires, "c")
        self._provider = None  # type: Optional[BaseProvider]
        self._circuit = None  # type: Optional[QuantumCircuit]
        self._current_job = None  # type: Optional[BaseJob]
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

        Args:
            operation (str): name of the operation
            wires (Sequence[int]): subsystems the operation is applied on
            par (tuple): parameters for the operation
        """

        mapped_operation = self._operation_map[operation]

        if isinstance(mapped_operation, BasisState) and not self._first_operation:
            raise DeviceError("Operation {} cannot be used after other Operations have already been applied "
                              "on a {} device.".format(operation, self.short_name))
        self._first_operation = False

        qregs = [(self._reg, i) for i in wires]

        if inspect.isclass(mapped_operation):
            dag = circuit_to_dag(QuantumCircuit(self._reg, self._creg, name=''))
            instruction = mapped_operation(*par)
            if isinstance(instruction, Gate):
                dag.apply_operation_back(instruction, qargs=qregs)
                qc = dag_to_circuit(dag)
                self._circuit = self._circuit + qc
            else:
                raise ValueError("Class not known and cannot be instantiated: ".format(type(instruction)))
        elif isinstance(mapped_operation, QiskitInstructions):
            op = mapped_operation  # type: QiskitInstructions
            op.apply(qregs=qregs, param=list(par), circuit=self._circuit)
        else:
            raise ValueError("The operation is not of an expected type. This is a software bug!")

    def pre_expval(self):
        compile_backend = self._provider.get_backend(self.compile_backend)  # type: BaseBackend

        for qr, cr in zip(self._reg, self._creg):
            measure(self._circuit, qr, cr)

        compiled_circuits = qiskit.compiler.transpile(self._circuit, backend=compile_backend)
        qobj = qiskit.compiler.assemble(experiments=compiled_circuits, backend=compile_backend, shots=self.shots)
        backend = self._provider.get_backend(self.backend)  # type: BaseBackend

        try:
            if isinstance(backend, AerBackend) and (isinstance(self, BasicAerQiskitDevice) or
                                                    isinstance(self, AerQiskitDevice)):
                self._current_job = backend.run(qobj, noise_model=self._noise_model)
            else:
                self._current_job = backend.run(qobj)  # type: BaseJob

            not_done = [JobStatus.INITIALIZING, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.VALIDATING]
            self._current_job.result()  # call result here once and discard it to trigger the actual computation

        except Exception as ex:
            raise Exception("Error during job execution: {}!".format(ex))

    def expval(self, expectation, wires, par):
        result = self._current_job.result()  # type: Result

        probabilities = dict((state[::-1], count / self.shots) for state, count in result.get_counts().items())

        expval = None

        if expectation == 'PauliZ':
            if isinstance(wires, int):
                wire = wires
            else:
                wire = wires[0]

            zero = sum(p for (state, p) in probabilities.items() if state[wire] == '0')
            one = sum(p for (state, p) in probabilities.items() if state[wire] == '1')

            expval = (1 - (2 * one) - (1 - 2 * zero)) / 2

        return expval

    def reset(self):
        self._circuit = QuantumCircuit(self._reg, self._creg, name='temp')
        self._first_operation = True


class BasicAerQiskitDevice(QiskitDevice):
    """A PennyLane :code:`qiskit.basicaer` device for the `Qiskit Local Simulator` backend.

    Args:
       wires (int): The number of qubits of the device
       noise_model (NoiseModel, optional): NoiseModel Object from qiskit.providers.aer.noise. Defaults to None

    Keyword Args:
      backend (str): the desired backend to run the code on. Default is :code:`qasm_simulator`.

    This device can, for example, be instantiated from PennyLane as follows:

    .. code-block:: python

        import pennylane as qml
        dev = qml.device('qiskit.basicaer', wires=XXX)

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
      :class:`pennylane_qiskit.S <pennylane_qiskit.ops.S>`,
      :class:`pennylane_qiskit.T <pennylane_qiskit.ops.T>`
      :class:`pennylane_qiskit.U1 <pennylane_qiskit.ops.U1>`,
      :class:`pennylane_qiskit.U2 <pennylane_qiskit.ops.U2>`
      :class:`pennylane_qiskit.U3 <pennylane_qiskit.ops.U3>`,

    """
    short_name = 'qiskit.basicaer'

    def __init__(self, wires, shots=1024, backend='qasm_simulator', noise_model=None, **kwargs):
        super().__init__(wires, backend=backend, shots=shots, **kwargs)
        self._provider = qiskit.BasicAer
        self._noise_model = noise_model
        self._capabilities['backend'] = [b.name() for b in self._provider.backends()]


class AerQiskitDevice(QiskitDevice):
    """A PennyLane :code:`qiskit.aer` device for the `Qiskit Simulator Aer (local)` backend.

    Args:
       wires (int): The number of qubits of the device

    Keyword Args:
      backend (str): the desired backend to run the code on. Default is :code:`qasm_simulator`.
      noise_model (NoiseModel, optional): NoiseModel Object from qiskit.providers.aer.noise. Defaults to None

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
      :class:`pennylane_qiskit.S <pennylane_qiskit.ops.S>`,
      :class:`pennylane_qiskit.T <pennylane_qiskit.ops.T>`
      :class:`pennylane_qiskit.U1 <pennylane_qiskit.ops.U1>`,
      :class:`pennylane_qiskit.U2 <pennylane_qiskit.ops.U2>`
      :class:`pennylane_qiskit.U3 <pennylane_qiskit.ops.U3>`,

    """
    short_name = 'qiskit.basicaer'

    def __init__(self, wires, shots=1024, backend='qasm_simulator', noise_model=None, **kwargs):
        super().__init__(wires, backend=backend, shots=shots, **kwargs)
        self._provider = qiskit.Aer
        self._noise_model = noise_model
        self._capabilities['backend'] = [b.name() for b in self._provider.backends()]


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
      :class:`pennylane_qiskit.S <pennylane_qiskit.ops.S>`,
      :class:`pennylane_qiskit.T <pennylane_qiskit.ops.T>`
      :class:`pennylane_qiskit.U1 <pennylane_qiskit.ops.U1>`,
      :class:`pennylane_qiskit.U2 <pennylane_qiskit.ops.U2>`
      :class:`pennylane_qiskit.U3 <pennylane_qiskit.ops.U3>`,

    """
    short_name = 'qiskit.ibmq'
    _backend_kwargs = ['verbose', 'backend', 'ibmqx_token']

    def __init__(self, wires, backend='ibmq_qasm_simulator', shots=1024, **kwargs):
        token_from_env = os.getenv('IBMQX_TOKEN')
        if 'ibmqx_token' not in kwargs and token_from_env is None:
            raise ValueError("IBMQX Token is missing!")
        token = token_from_env or kwargs['ibmqx_token']
        super().__init__(wires=wires, backend=backend, shots=shots, **kwargs)
        self._provider = qiskit.IBMQ
        if token not in map(lambda e: e['token'], self._provider.active_accounts()):
            self._provider.enable_account(token)
        self._capabilities['backend'] = [b.name() for b in self._provider.backends()]
