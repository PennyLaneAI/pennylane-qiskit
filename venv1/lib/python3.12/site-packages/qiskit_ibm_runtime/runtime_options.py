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

"""Runtime options that control the execution environment."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Optional, List

from qiskit.providers.backend import Backend

from .exceptions import IBMInputValueError
from .utils.utils import validate_job_tags


@dataclass(init=False)
class RuntimeOptions:
    """Class for representing generic runtime execution options."""

    backend: Optional[str | Backend] = None
    image: Optional[str] = None
    log_level: Optional[str] = None
    instance: Optional[str] = None
    job_tags: Optional[List[str]] = None
    max_execution_time: Optional[int] = None
    session_time: Optional[int] = None

    def __init__(
        self,
        backend: Optional[str | Backend] = None,
        image: Optional[str] = None,
        log_level: Optional[str] = None,
        instance: Optional[str] = None,
        job_tags: Optional[List[str]] = None,
        max_execution_time: Optional[int] = None,
        session_time: Optional[int] = None,
    ) -> None:
        """RuntimeOptions constructor.

        Args:
            backend: target backend to run on. This is required for ``ibm_quantum`` channel.
            image: the runtime image used to execute the program, specified in
                the form of ``image_name:tag``. Not all accounts are
                authorized to select a different image.
            log_level: logging level to set in the execution environment. The valid
                log levels are: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, and ``CRITICAL``.
                The default level is ``WARNING``.
            instance: The hub/group/project to use, in that format. This is only supported
                for ``ibm_quantum`` channel. If ``None``, a hub/group/project that provides
                access to the target backend is randomly selected.
            job_tags: Tags to be assigned to the job. The tags can subsequently be used
                as a filter in the :meth:`jobs()` function call.
            max_execution_time: Maximum execution time in seconds, which is based
                on system execution time (not wall clock time). System execution time is the
                amount of time that the system is dedicated to processing your job. If a job exceeds
                this time limit, it is forcibly cancelled. Simulator jobs continue to use wall
                clock time.
            session_time: Length of session in seconds.
        """
        self.backend = backend
        self.image = image
        self.log_level = log_level
        self.instance = instance
        self.job_tags = job_tags
        self.max_execution_time = max_execution_time
        self.session_time = session_time

    def validate(self, channel: str) -> None:
        """Validate options.

        Args:
            channel: channel type.

        Raises:
            IBMInputValueError: If one or more option is invalid.
        """
        if self.image and not re.match(
            "[a-zA-Z0-9]+([/.\\-_][a-zA-Z0-9]+)*:[a-zA-Z0-9]+([.\\-_][a-zA-Z0-9]+)*$",
            self.image,
        ):
            raise IBMInputValueError('"image" needs to be in form of image_name:tag')

        if channel == "ibm_quantum" and not self.backend:
            raise IBMInputValueError(
                '"backend" is required field in "options" for "ibm_quantum" channel.'
            )

        if self.instance and channel != "ibm_quantum":
            raise IBMInputValueError('"instance" is only supported for "ibm_quantum" channel.')

        if self.log_level and not isinstance(logging.getLevelName(self.log_level.upper()), int):
            raise IBMInputValueError(
                f"{self.log_level} is not a valid log level. The valid log levels are: `DEBUG`, "
                f"`INFO`, `WARNING`, `ERROR`, and `CRITICAL`."
            )

        if self.job_tags:
            validate_job_tags(self.job_tags)

    def get_backend_name(self) -> str:
        """Get backend name."""
        if isinstance(self.backend, str):
            return self.backend
        if self.backend:
            return self.backend.name if self.backend.version == 2 else self.backend.name()
        return None
