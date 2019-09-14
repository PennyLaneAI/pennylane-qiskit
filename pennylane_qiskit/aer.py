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
Aer Device
==========

**Module name:** :mod:`pennylane_qiskit.aer`

.. currentmodule:: pennylane_qiskit.aer

This module contains the :class:`~.AerDevice` class, a PennyLane device that allows
evaluation and differentiation of Qiskit Aer's C++ simulator
using PennyLane.

Classes
-------

.. autosummary::
   AerDevice

Code details
~~~~~~~~~~~~
"""
import qiskit

from .qiskit_device import QiskitDevice


class AerDevice(QiskitDevice):
    """A PennyLane device for the C++ Qiskit Aer simulator.

    Please refer to the `Qiskit documentation <https://qiskit.org/documentation/autodoc/qiskit.providers.aer.backends.html>`_ for
    for further information to the noise model and backend options.

    A range of :code:`backend_options` can be given as kwargs that will be passed to the simulator.

    For details on the backends, please check out

        * `qasm_simulator <https://qiskit.org/documentation/autodoc/qiskit.providers.basicaer.qasm_simulator.html>`_
        * `statevector_simulator  <https://qiskit.org/documentation/autodoc/qiskit.providers.basicaer.statevector_simulator.html>`_
        * `unitary_simulator  <https://qiskit.org/documentation/autodoc/qiskit.providers.basicaer.unitary_simulator.html>`_

    Args:
        wires (int): The number of qubits of the device
        backend (str): the desired backend
        shots (int): number of circuit evaluations/random samples used
            to estimate expectation values and variances of observables

    Keyword Args:
        name (str): The name of the circuit. Default ``'circuit'``.
        compile_backend (BaseBackend): The backend used for compilation. If you wish
            to simulate a device compliant circuit, you can specify a backend here.
        noise_model (NoiseModel): NoiseModel Object from ``qiskit.providers.aer.noise``
    """

    # pylint: disable=too-many-arguments

    short_name = "qiskit.aer"

    def __init__(self, wires, shots=1024, backend="qasm_simulator", **kwargs):
        super().__init__(wires, provider=qiskit.Aer, backend=backend, shots=shots, **kwargs)
