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
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        provider (Provider | None): provider to lookup the backend on (ignored if a backend instance is passed).
        backend (str | Backend): the desired backend. Either a name to look up on a provider, or a
            BackendV1 or BackendV2 instance.
        shots (int or None): number of circuit evaluations/random samples used
            to estimate expectation values and variances of observables. For statevector backends,
            setting to ``None`` results in computing statistics like expectation values and variances analytically.

    Keyword Args:
        name (str): The name of the circuit. Default ``'circuit'``.
    """

    short_name = "qiskit.remote"

    def __init__(self, wires, backend, provider=None, shots=1024, **kwargs):
        super().__init__(wires, provider=provider, backend=backend, shots=shots, **kwargs)
