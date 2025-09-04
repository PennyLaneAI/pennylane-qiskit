# Copyright 2019 Xanadu Quantum Technologies Inc.

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
This module contains the :class:`~.BasicAerDevice` class, a PennyLane device that allows
evaluation and differentiation of Qiskit Terra's BasicAer simulator
using PennyLane.
"""
from qiskit.providers.basic_provider import BasicProvider

from .qiskit_device_legacy import QiskitDeviceLegacy


class BasicSimulatorDevice(QiskitDeviceLegacy):
    """A PennyLane device for the native Python Qiskit simulator.

    For more information on the ``BasicSimulator`` backend options and transpile options, please visit the
    `BasicProvider documentation <https://docs.quantum.ibm.com/api/qiskit/providers_basic_provider>`_.
    These options can be passed to this plugin device as keyword arguments.

    Args:
        wires (int or Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['aux_wire', 'q1', 'q2']``).
        backend (str): the desired backend
        shots (int or None): number of circuit evaluations/random samples used
            to estimate expectation values and variances of observables. For statevector backends,
            setting to ``None`` results in computing statistics like expectation values and variances analytically.
    """

    short_name = "qiskit.basicsim"

    analytic_warning_message = (
        "The plugin does not currently support analytic calculation of expectations, variances "
        "and probabilities with the BasicProvider backend {}. Such statistics obtained from this "
        "device are estimates based on samples."
    )

    def __init__(self, wires, shots=None, backend="basic_simulator", **kwargs):
        super().__init__(wires, provider=BasicProvider(), backend=backend, shots=shots, **kwargs)
