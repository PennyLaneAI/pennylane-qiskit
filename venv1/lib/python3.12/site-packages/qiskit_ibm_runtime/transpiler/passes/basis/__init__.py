# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
==========================================================
Basis (:mod:`qiskit_ibm_runtime.transpiler.passes.basis`)
==========================================================

.. currentmodule:: qiskit_ibm_runtime.transpiler.passes.basis

Passes to layout circuits to IBM backend's instruction sets.
"""

from .convert_id_to_delay import ConvertIdToDelay
