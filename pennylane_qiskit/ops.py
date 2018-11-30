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
Operations
##########

.. currentmodule:: pennylane_qiskit.ops

In addition to the suitable default operations native to PennyLane,
the devices of the qiskit plugin support a number of additional operations
that can be used alongside the native PennyLane operations when defining
quantum functions:

.. autosummary::
   S
   T

.. note::
    For convenience, and to mirror the behavior of the operations built into
    PennyLane, the operations defined here are also accessible directly under
    the top-level :code:`pennylane_qiskit` context, i.e., you can use
    :code:`pennylane_qiskit.S([0])` instead of :code:`pennylane_qiskit.ops.S([0])`
    when defining a :code:`QNode` using the :code:`qnode` decorator.

"""

from pennylane.operation import Operation


class S(Operation):
    r"""S gate.

    .. math:: S = \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 1
    par_domain = None


class T(Operation):
    r"""T gate.

    .. math:: T = \begin{bmatrix}1&0\\0&\exp(i \pi / 4)\end{bmatrix}

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 1
    par_domain = None
