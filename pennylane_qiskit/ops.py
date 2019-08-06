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
r"""
Custom operations
=================

**Module name:** :mod:`pennylane_qiskit.ops`

.. currentmodule:: pennylane_qiskit.ops

In addition to the suitable default operations native to PennyLane,
the devices of the Qiskit plugin support a number of additional operations
that can be used alongside the native PennyLane operations when defining
quantum functions.

.. note::

    For convenience, and to mirror the behavior of the operations built into
    PennyLane, the operations defined here are also accessible directly under
    the top-level :code:`pennylane_qiskit` context, i.e., you can use
    :code:`pennylane_qiskit.S([0])` instead of :code:`pennylane_qiskit.ops.S([0])`
    when defining a :code:`QNode` using the :code:`qnode` decorator.

Operations
----------

.. autosummary::
   S
   Sdg
   T
   Tdg
   U1
   U2
   U3
   CSWAP
   Toffoli

Code details
~~~~~~~~~~~~
"""
# pylint: disable=too-few-public-methods
from pennylane.operation import Operation


class S(Operation):
    r"""S(wires)
    S gate.

    .. math:: S = \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 1
    par_domain = None


class Sdg(Operation):
    r"""Sdg(wires)
    :math:`S^\dagger` gate.

    .. math:: S^\dagger = \begin{bmatrix} 1 & 0 \\ 0 & -i \end{bmatrix}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 1
    par_domain = None


class T(Operation):
    r"""T(wires)
    T gate.

    .. math:: T = \begin{bmatrix}1&0\\0&e^{i \pi / 4}\end{bmatrix}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 1
    par_domain = None


class Tdg(Operation):
    r"""Tdg(wires)
    :math:`T^\dagger` gate.

    .. math:: T = \begin{bmatrix}1&0\\0&e^{-i \pi / 4}\end{bmatrix}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 1
    par_domain = None


class U1(Operation):
    r"""U1(lambda, wires)
    U1 gate.

    .. math:: u_1 = \begin{bmatrix}1&0\\0&\exp(i \lambda)\end{bmatrix}

    .. note::

        This is identical to the PennyLane ``PhaseShift`` operation.

    Args:
        lambda (float): quantum phase :math:`\lambda`
        wires (Sequence[int] or int): the subsystem the gate acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = "R"


class U2(Operation):
    r"""U2(phi, lambda, wires)
    U2 gate.

    .. math::

        u_2 = \begin{bmatrix} 1 & -\exp(i \lambda)
        \\ \exp(i \phi) & \exp(i (\phi + \lambda)) \end{bmatrix}

    Args:
        phi (float): azimuthal angle :math:`\phi`
        lambda (float): quantum phase :math:`\lambda`
        wires (Sequence[int] or int): the subsystem the gate acts on
    """
    num_params = 2
    num_wires = 1
    par_domain = "R"


class U3(Operation):
    r"""U3(theta, phi, lambda, wires)
    U3 gate.

    .. math::

        u_3 = \begin{bmatrix} \cos(\theta/2) & -\exp(i \lambda)\sin(\theta/2) \\
        \exp(i \phi)\sin(\theta/2) & \exp(i (\phi + \lambda))\cos(\theta/2) \end{bmatrix}

    Args:
        theta (float): polar angle :math:`\theta`
        phi (float): azimuthal angle :math:`\phi`
        lambda (float): quantum phase :math:`\lambda`
        wires (Sequence[int] or int): the subsystem the gate acts on
    """
    num_params = 3
    num_wires = 1
    par_domain = "R"
    grad_method = "F"


class CSWAP(Operation):
    r"""CSWAP(wires)
    Controlled-swap gate.

    .. math::

        CSWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
             0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
             0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
             0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
             0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
             0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
             0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
             0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 3
    par_domain = None


class Toffoli(Operation):
    r"""Toffoli(wires)
    Toffoli (controlled-controlled-X) gate.

    .. math::

        Toffoli =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0
        \end{pmatrix}

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 3
    par_domain = None
