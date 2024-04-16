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

"""Options related to the execution environment."""

from typing import Optional, Callable, List, Literal

from .utils import primitive_dataclass

LogLevelType = Literal[
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]


@primitive_dataclass
class EnvironmentOptions:
    """Options related to the execution environment.

    Args:
        log_level: logging level to set in the execution environment. The valid
            log levels are: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, and ``CRITICAL``.
            Default: ``WARNING``.

        callback: Callback function to be invoked for any interim results and final result.
            The callback function will receive 2 positional parameters:

                1. Job ID
                2. Job result.

        job_tags: Tags to be assigned to the job. The tags can subsequently be used
            as a filter in the :meth:`qiskit_ibm_runtime.qiskit_runtime_service.jobs()`
            function call.
    """

    log_level: LogLevelType = "WARNING"
    callback: Optional[Callable] = None
    job_tags: Optional[List] = None
