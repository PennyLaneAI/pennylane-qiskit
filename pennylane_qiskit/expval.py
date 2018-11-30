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
Expectations
############

.. currentmodule:: pennylane_qiskit.expval

In addition to the suitable default operations native to PennyLane,
the devices of the ProjectQ plugin support a number of additional expectations
that can be used alongside the native PennyLane expectations when defining
quantum functions:

.. autosummary::
   .. AllPauliZ
"""

# from pennylane.operation import Expectation

# class AllPauliZ(Expectation):
#     r"""Measure Pauli Z on all qubits.

#     .. math:: AllPauliZ = \sigma_z \otimes\dots\otimes \sigma_z

#     """
#     num_params = 0
#     num_wires = 0
#     par_domain = None
