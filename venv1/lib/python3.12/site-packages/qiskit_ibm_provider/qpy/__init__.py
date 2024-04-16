# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
###########################################################
QPY serialization (:mod:`qiskit_ibm_runtime.qpy`)
###########################################################

.. currentmodule:: qiskit_ibm_runtime.qpy

*********
Using QPY
*********

This is downstream fork of the :mod:`qiskit.qpy` module.

Using QPY is defined to be straightforward and mirror the user API of the
serializers in Python's standard library, ``pickle`` and ``json``. There are
2 user facing functions: :func:`qiskit.circuit.qpy_serialization.dump` and
:func:`qiskit.circuit.qpy_serialization.load` which are used to dump QPY data
to a file object and load circuits from QPY data in a file object respectively.
For example::

    from qiskit.circuit import QuantumCircuit
    from qiskit import qpy

    qc = QuantumCircuit(2, name='Bell', metadata={'test': True})
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    with open('bell.qpy', 'wb') as fd:
        qpy.dump(qc, fd)

    with open('bell.qpy', 'rb') as fd:
        new_qc = qpy.load(fd)[0]

API documentation
=================

.. autosummary::
   :toctree: ../stubs/

   load
   dump

QPY Compatibility
=================

The QPY format is designed to be backwards compatible moving forward. This means
you should be able to load a QPY with any newer Qiskit version than the one
that generated it. However, loading a QPY file with an older Qiskit version is
not supported and may not work.

For example, if you generated a QPY file using qiskit-terra 0.18.1 you could
load that QPY file with qiskit-terra 0.19.0 and a hypothetical qiskit-terra
0.29.0. However, loading that QPY file with 0.18.0 is not supported and may not
work.

**********
QPY Format
**********

https://docs.quantum.ibm.com/api/qiskit/qpy#qpy-format
"""

from .interface import dump, load

# For backward compatibility. Provide, Runtime, Experiment call these private functions.
from .binary_io import (
    _write_instruction,
    _read_instruction,
    _write_parameter_expression,
    _read_parameter_expression,
    _read_parameter_expression_v3,
    _write_parameter,
    _read_parameter,
)
