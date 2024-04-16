# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
===========================================================
Visualizations (:mod:`qiskit_ibm_provider.visualization`)
===========================================================

.. currentmodule:: qiskit_ibm_provider.visualization

Note:
    To use these tools locally, you'll need to install the
    additional dependencies for the visualization functions::

        pip install qiskit_ibm_provider[visualization]

Interactive Visualizations
==========================

.. autosummary::
   :toctree: ../stubs/

   iplot_gate_map
   iplot_error_map
"""

from .interactive import iplot_error_map, iplot_gate_map
