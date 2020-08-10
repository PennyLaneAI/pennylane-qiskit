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
This module contains the :class:`~.AerDevice` class, a PennyLane device that allows
evaluation and differentiation of Qiskit Aer's C++ simulator
using PennyLane.
"""
import qiskit

from .qiskit_device import QiskitDevice


class AerDevice(QiskitDevice):
    """A PennyLane device for the C++ Qiskit Aer simulator.

    Please refer to the `Qiskit documentation <https://qiskit.org/documentation/>`_ for
    for further information to the noise model and backend options.

    A range of :code:`backend_options` can be given as kwargs that will be passed to the simulator.

    For more information on backends, please visit the
    `Aer provider documentation <https://qiskit.org/documentation/apidoc/aer_provider.html>`_.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        backend (str): the desired backend
        shots (int): number of circuit evaluations/random samples used
            to estimate expectation values and variances of observables

    Keyword Args:
        name (str): The name of the circuit. Default ``'circuit'``.
        compile_backend (BaseBackend): The backend used for compilation. If you wish
            to simulate a device compliant circuit, you can specify a backend here.
        analytic (bool): For statevector backends, determines if the
            expectation values and variances are to be computed analytically.
            Default value is ``False``.
        noise_model (NoiseModel): NoiseModel Object from ``qiskit.providers.aer.noise``
    """

    # pylint: disable=too-many-arguments

    short_name = "qiskit.aer"

    def __init__(self, wires, shots=1024, backend="qasm_simulator", **kwargs):
        super().__init__(wires, provider=qiskit.Aer, backend=backend, shots=shots, **kwargs)
