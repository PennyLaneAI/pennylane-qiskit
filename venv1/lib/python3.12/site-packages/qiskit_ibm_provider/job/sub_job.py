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

"""IBM Quantum sub job."""

import threading
from concurrent.futures import Future
from typing import Optional, Union

from qiskit.qobj import QasmQobj, PulseQobj
from qiskit.result import Result

from .exceptions import IBMJobFailureError, IBMJobInvalidStateError
from .ibm_circuit_job import IBMCircuitJob
from .utils import auto_retry


class SubJob:
    """Representation of a sub-job that belongs to an ``IBMCompositeJob``.

    This class is mainly used to keep track of the indexes of each
    ``IBMCircuitJob`` inside an ``IBMCompositeJob``.
    """

    def __init__(
        self,
        start_index: int,
        end_index: int,
        job_index: int,
        total: int,
        qobj: Optional[Union[QasmQobj, PulseQobj]] = None,
        job: IBMCircuitJob = None,
    ) -> None:
        """SubJob constructor.

        Args:
            start_index: Circuit start index.
            end_index: Circuit end index.
            job_index: Job index.
            total: Total number of jobs.
            qobj: Qobj for this job.
            job: Circuit job.
        """
        self.start_index = start_index
        self.end_index = end_index
        self.job_index = job_index
        self.total_jobs = total
        self._qobj = qobj
        self._job = job
        self.event = threading.Event()
        self._submit_error: Optional[Exception] = None
        self.future: Optional[Future] = None

    def format_tag(self, tag_template: str) -> str:
        """Format the the job tag using indexes.

        Args:
            tag_template: Tag template to use.

        Returns:
            Formatted tag.
        """
        return tag_template.format(
            job_index=self.job_index,
            total_jobs=self.total_jobs,
            start_index=self.start_index,
            end_index=self.end_index,
        )

    @property
    def qobj(self) -> Optional[Union[QasmQobj, PulseQobj]]:
        """Return the Qobj for this job.

        Returns:
            The Qobj for this job, or ``None`` if the job does not have a Qobj.
        """
        if self._qobj:
            return self._qobj
        if self.job:
            return self.job._get_qobj()
        return None

    @property
    def job(self) -> Optional[IBMCircuitJob]:
        """Return the ``IBMCircuitJob`` instance represented by this subjob.

        Returns:
            The corresponding ``IBMCircuitJob`` instance, or ``None`` if job
            has not been submitted.
        """
        if self.future and not self.future.done():
            return None
        return self._job

    @job.setter
    def job(self, job: IBMCircuitJob) -> None:
        """Sets the ``IBMCircuitJob`` instance.

        Args:
            job: The ``IBMCircuitJob`` instance.
        """
        self._job = job

    @property
    def submit_error(self) -> Optional[Exception]:
        """Return job submit error.

        Returns:
            Job submit errors, if any.
        """
        if self.future and not self.future.done():
            return None
        return self._submit_error

    @submit_error.setter
    def submit_error(self, error: Exception) -> None:
        """Set job submit error.

        Args:
            error: Job submit error to set.
        """
        self._submit_error = error

    def reset(self) -> None:
        """Clear job and error data."""
        self.future = None
        self.job = None
        self.submit_error = None
        self.event.clear()

    def result(self, refresh: bool, partial: bool) -> Optional[Result]:
        """Return job result.

        Args:
            refresh: If ``True``, re-query the server for the result.
            partial: If ``True``, return partial results if possible.

        Returns:
            Job result or ``None`` if job result is not available.
        """
        if not self.job:
            return None
        try:
            return auto_retry(self.job.result, refresh=refresh, partial=partial)
        except (IBMJobFailureError, IBMJobInvalidStateError):
            return None

    def __repr__(self) -> str:
        job_id = self.job.job_id() if self.job else None
        return (
            f"<{self.__class__.__name__}> {self.job_index} (job ID {job_id}) "
            f" for circuits {self.start_index}-{self.end_index}"
        )
