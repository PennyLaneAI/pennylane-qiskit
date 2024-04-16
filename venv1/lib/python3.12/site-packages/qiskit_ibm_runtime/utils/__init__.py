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
==============================================================================
Utilities (:mod:`qiskit_ibm_runtime.utils`)
==============================================================================

.. currentmodule:: qiskit_ibm_runtime.utils

Utility functions related to the IBM Runtime Services.

Conversion
==========
.. autosummary::
    :toctree: ../stubs/

    seconds_to_duration
    utc_to_local

Misc Functions
==============
.. autosummary::
    :toctree: ../stubs/

    to_python_identifier
"""

from .converters import (
    utc_to_local,
    local_to_utc,
    seconds_to_duration,
    duration_difference,
)
from .utils import (
    to_python_identifier,
    is_crn,
    get_runtime_api_base_url,
    resolve_crn,
    are_circuits_dynamic,
    validate_no_dd_with_dynamic_circuits,
)
from .json import RuntimeEncoder, RuntimeDecoder, to_base64_string
from . import pubsub
