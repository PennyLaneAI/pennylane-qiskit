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
import qiskit_aer

from .qiskit_device_legacy import QiskitDeviceLegacy


class AerDevice(QiskitDeviceLegacy):
    """A PennyLane device for the C++ Qiskit Aer simulator.

    Please refer to the `Qiskit documentation <https://qiskit.org/documentation/>`_ for
    further information on the noise model, backend options and transpile options.

    A range of :code:`backend_options` that will be passed to the simulator and
    a range of transpile options can be given as kwargs.

    For more information on backends, please visit the
    `qiskit_aer documentation <https://qiskit.org/ecosystem/aer/index.html>`_.

    Args:
        wires (int or Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        backend (str): the desired backend
        method (str): The desired simulation method. A list of supported simulation
            methods can be returned using ``qiskit_aer.AerSimulator().available_methods()``, or by referring
            to the ``AerSimulator`` `documentation <https://qiskit.org/ecosystem/aer/stubs/qiskit_aer.AerSimulator.html>`__.
        shots (int or None): number of circuit evaluations/random samples used
            to estimate expectation values and variances of observables. For statevector backends,
            setting to ``None`` results in computing statistics like expectation values and variances analytically.

    Keyword Args:
        name (str): The name of the circuit. Default ``'circuit'``.
        compile_backend (BaseBackend): The backend used for compilation. If you wish
            to simulate a device compliant circuit, you can specify a backend here.
        noise_model (NoiseModel): NoiseModel Object from ``qiskit_aer.noise``
    """

    short_name = "qiskit.aer"

    def __init__(self, wires, shots=None, backend="aer_simulator", method="automatic", **kwargs):
        if method != "automatic":
            backend += "_" + method

        super().__init__(wires, provider=qiskit_aer.Aer, backend=backend, shots=shots, **kwargs)
