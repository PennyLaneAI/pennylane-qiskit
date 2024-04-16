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
===================================================
IBM Quantum Provider (:mod:`qiskit_ibm_provider`)
===================================================

.. currentmodule:: qiskit_ibm_provider

Modules representing the Qiskit IBM Quantum Provider.

Logging
=====================

The qiskit-ibm-provider uses the ``qiskit_ibm_provider`` logger.

Two environment variables can be used to control the logging:

    * ``QISKIT_IBM_PROVIDER_LOG_LEVEL``: Specifies the log level to use, for the Qiskit
      IBM provider modules. If an invalid level is set, the log level defaults to ``WARNING``.
      The valid log levels are ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, and ``CRITICAL``
      (case-insensitive). If the environment variable is not set, then the parent logger's level
      is used, which also defaults to ``WARNING``.
    * ``QISKIT_IBM_PROVIDER_LOG_FILE``: Specifies the name of the log file to use. If specified,
      messages will be logged to the file only. Otherwise messages will be logged to the standard
      error (usually the screen).

For more advanced use, you can modify the logger itself. For example, to manually set the level
to ``WARNING``::

    import logging
    logging.getLogger('qiskit_ibm_provider').setLevel(logging.WARNING)

Functions
=========
.. autosummary::
    :toctree: ../stubs/

    least_busy

Classes
=======
.. autosummary::
    :toctree: ../stubs/

    IBMProvider
    IBMBackend
    IBMBackendService
    Session

Exceptions
==========
.. autosummary::
    :toctree: ../stubs/

    IBMError
    IBMProviderError
    IBMProviderValueError
    IBMBackendError
    IBMBackendApiError
    IBMBackendApiProtocolError
    IBMBackendValueError
    IBMProviderError
"""

import logging
from typing import List, Optional, Union
from datetime import datetime, timedelta

from qiskit.providers import Backend  # type: ignore[attr-defined]

from .ibm_provider import IBMProvider
from .ibm_backend import IBMBackend
from .session import Session
from .job.ibm_job import IBMJob
from .exceptions import *
from .ibm_backend_service import IBMBackendService
from .utils.utils import setup_logger
from .version import __version__

# Setup the logger for the IBM Quantum Provider package.
logger = logging.getLogger(__name__)
setup_logger(logger)

# Constants used by the IBM Quantum logger.
QISKIT_IBM_PROVIDER_LOGGER_NAME = "qiskit_ibm_provider"
"""The name of the IBM Quantum logger."""
QISKIT_IBM_PROVIDER_LOG_LEVEL = "QISKIT_IBM_PROVIDER_LOG_LEVEL"
"""The environment variable name that is used to set the level for the IBM Quantum logger."""
QISKIT_IBM_PROVIDER_LOG_FILE = "QISKIT_IBM_PROVIDER_LOG_FILE"
"""The environment variable name that is used to set the file for the IBM Quantum logger."""


def least_busy(backends: List[Backend]) -> Backend:
    """Return the least busy backend from a list.

    Return the least busy available backend for those that
    have a ``pending_jobs`` in their ``status``. Note that local
    backends may not have this attribute.

    Args:
        backends: The backends to choose from.

    Returns:
        The backend with the fewest number of pending jobs.

    Raises:
        IBMError: If the backends list is empty, or if none of the backends
            is available, or if a backend in the list
            does not have the ``pending_jobs`` attribute in its status.
    """
    if not backends:
        raise IBMError(
            "Unable to find the least_busy backend from an empty list."
        ) from None
    try:
        candidates = []
        for back in backends:
            backend_status = back.status()
            if not backend_status.operational or backend_status.status_msg != "active":
                continue
            candidates.append(back)
        if not candidates:
            raise IBMError("No backend matches the criteria.")
        return min(candidates, key=lambda b: b.status().pending_jobs)
    except AttributeError as ex:
        raise IBMError(
            "A backend in the list does not have the `pending_jobs` "
            "attribute in its status."
        ) from ex
