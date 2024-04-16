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

"""IBM Quantum composite job."""

import copy
import logging
import re
import threading
import time
import traceback
import uuid
from collections import defaultdict
from concurrent import futures
from datetime import datetime as python_datetime
from functools import wraps
from typing import Dict, Optional, Tuple, Any, List, Callable, Union

from qiskit.assembler.disassemble import disassemble
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.compiler import assemble
from qiskit.providers.jobstatus import JOB_FINAL_STATES, JobStatus
from qiskit.providers.models import BackendProperties
from qiskit.qobj import QasmQobj, PulseQobj
from qiskit.result import Result
from qiskit.result.models import ExperimentResult

from qiskit_ibm_provider import ibm_backend  # pylint: disable=unused-import
from .constants import (
    IBM_COMPOSITE_JOB_ID_PREFIX,
    IBM_COMPOSITE_JOB_INDEX_PREFIX,
    IBM_COMPOSITE_JOB_TAG_PREFIX,
)
from .exceptions import (
    IBMJobApiError,
    IBMJobFailureError,
    IBMJobTimeoutError,
    IBMJobInvalidStateError,
)
from .ibm_circuit_job import IBMCircuitJob
from .ibm_job import IBMJob
from .queueinfo import QueueInfo
from .sub_job import SubJob
from .utils import auto_retry, JOB_STATUS_TO_INT, JobStatusQueueInfo, last_job_stat_pos
from ..api.clients import AccountClient
from ..exceptions import IBMBackendJobLimitError
from ..utils.utils import validate_job_tags, api_status_to_job_status

logger = logging.getLogger(__name__)


def _requires_submit(func):  # type: ignore
    """Decorator used by ``IBMCompositeJob`` to wait for all jobs to be submitted."""

    @wraps(func)
    def _wrapper(self, *args, **kwargs):  # type: ignore
        self.block_for_submit()
        return func(self, *args, **kwargs)

    return _wrapper


class IBMCompositeJob(IBMJob):
    """Representation of a set of jobs that execute on an IBM Quantum backend.

    An ``IBMCompositeJob`` instance is returned when you call
    :meth:`IBMBackend.run()<qiskit_ibm_provider.ibm_backend.IBMBackend.run()>`
    to submit a list of circuits whose length exceeds the maximum allowed by
    the backend or by the ``max_circuits_per_job`` parameter.

    This ``IBMCompositeJob`` instance manages all the sub-jobs for you and can
    be used like a traditional job instance. For example, you can continue to
    use methods like :meth:`status()` and :meth:`result()` to get the job
    status and result, respectively.

    You can also retrieve a previously executed ``IBMCompositeJob`` using the
    :meth:`~qiskit_ibm_provider.IBMBackendService.job` and
    :meth:`~qiskit_ibm_provider.IBMBackendService.jobs` methods, like you would with
    traditional jobs.

    ``IBMCompositeJob`` also allows you to re-run failed jobs, using the
    :meth:`rerun_failed()` method. This method will re-submit all failed or
    cancelled sub-jobs. Any circuits that failed to be submitted (e.g. due to
    server error) will only be re-submitted if the circuits are known. That is,
    if this ``IBMCompositeJob`` was returned by
    :meth:`qiskit_ibm_provider.IBMBackend.run` and not retrieved from the server.

    Some of the methods in this class are blocking, which means control may
    not be returned immediately. :meth:`result()` is an example
    of a blocking method, and control will return only after all sub-jobs finish.

    ``IBMCompositeJob`` uses job tags to identify sub-jobs. It is therefore
    important to preserve these tags. All tags used internally by ``IBMCompositeJob``
    start with ``ibm_composite_job_``.
    """

    _id_suffix = "_"
    _index_tag = (
        IBM_COMPOSITE_JOB_INDEX_PREFIX
        + "{job_index}:{total_jobs}:{start_index}:{end_index}"
    )
    _index_pattern = re.compile(
        rf"{IBM_COMPOSITE_JOB_INDEX_PREFIX}"
        r"(?P<job_index>\d+):(?P<total_jobs>\d+):"
        r"(?P<start_index>\d+):(?P<end_index>\d+)"
    )

    _executor = futures.ThreadPoolExecutor()
    """Threads used for asynchronous processing."""

    def __init__(
        self,
        backend: "ibm_backend.IBMBackend",
        api_client: AccountClient,
        job_id: Optional[str] = None,
        creation_date: Optional[python_datetime] = None,
        jobs: Optional[List[IBMCircuitJob]] = None,
        circuits_list: Optional[List[List[QuantumCircuit]]] = None,
        run_config: Optional[Dict] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        client_version: Optional[Dict] = None,
    ) -> None:
        """IBMCompositeJob constructor.

        Args:
            backend: The backend instance used to run this job.
            api_client: Object for connecting to the server.
            job_id: Job ID.
            creation_date: Job creation date.
            jobs: A list of sub-jobs.
            circuits_list: Circuits for this job.
            run_config: Runtime configuration for this job.
            name: Job name.
            tags: Job tags.
            client_version: Client used for the job.

        Raises:
            IBMJobInvalidStateError: If one or more subjobs is missing.
        """
        if jobs is None and circuits_list is None:
            raise IBMJobInvalidStateError(
                '"jobs" and "circuits_list" cannot both be None.'
            )

        self._job_id = (
            job_id or IBM_COMPOSITE_JOB_ID_PREFIX + uuid.uuid4().hex + self._id_suffix
        )
        tags = tags or []
        filtered_tags = [
            tag for tag in tags if not tag.startswith(IBM_COMPOSITE_JOB_TAG_PREFIX)
        ]
        super().__init__(
            backend=backend,
            api_client=api_client,
            job_id=self._job_id,
            name=name,
            tags=filtered_tags,
        )
        self._status = JobStatus.INITIALIZING
        self._creation_date = creation_date
        self._client_version = client_version

        # Properties used for job submit.
        self._sub_jobs: List[SubJob] = []

        # Properties used for caching.
        self._user_cancelled = False
        self._job_error_msg: Optional[str] = None
        self._properties: Optional[List] = None
        self._queue_info = None
        self._qobj = None
        self._result: Optional[Result] = None
        self._circuits = None

        # Properties used for wait_for_final_state callback.
        self._callback_lock = threading.Lock()
        self._user_callback: Optional[Callable] = None
        self._user_wait_value: Optional[float] = None
        # A tuple of status and queue position.
        self._last_reported_stat: Tuple[Optional[JobStatus], Optional[int]] = (
            None,
            None,
        )
        self._last_reported_time = 0.0
        self._job_statuses: Dict[str, JobStatusQueueInfo] = {}

        if circuits_list is not None:
            self._circuits = [circ for sublist in circuits_list for circ in sublist]
            self._submit_circuits(circuits_list, run_config)
        else:
            # Validate the jobs.
            total = 0
            for job in jobs:
                job_idx, total, start, end = self._find_circuit_indexes(job)
                self._sub_jobs.append(
                    SubJob(
                        start_index=start,
                        end_index=end,
                        job_index=job_idx,
                        total=total,
                        job=job,
                    )
                )
            missing = set(range(total)) - {
                sub_job.job_index for sub_job in self._sub_jobs
            }
            if len(missing) > 0:
                raise IBMJobInvalidStateError(
                    f"Composite job {self.job_id()} is missing jobs at "
                    f"indexes {', '.join([str(idx) for idx in missing])}."
                )
            self._sub_jobs.sort(key=lambda sj: sj.job_index)

    @classmethod
    def from_jobs(
        cls, job_id: str, jobs: List[IBMCircuitJob], api_client: AccountClient
    ) -> "IBMCompositeJob":
        """Return an instance of this class.

        The input job ID is used to query for sub-job information from the server.

        Args:
            job_id: Job ID.
            jobs: A list of circuit jobs that belong to this composite job.
            api_client: Client to use to communicate with the server.

        Returns:
            An instance of this class.
        """
        logger.debug(
            "Restoring IBMCompositeJob from jobs %s", [job.job_id() for job in jobs]
        )
        ref_job = jobs[0]
        return cls(
            backend=ref_job.backend(),
            api_client=api_client,
            job_id=job_id,
            creation_date=ref_job.creation_date(),
            jobs=jobs,
            circuits_list=None,
            run_config=None,
            name=ref_job.name(),
            tags=ref_job.tags(),
            client_version=ref_job.client_version,
        )

    def _submit_circuits(
        self,
        circuit_lists: List[List[QuantumCircuit]],
        run_config: Dict,
    ) -> None:
        """Assemble and submit circuits.

        Args:
            circuit_lists: List of circuits to submit.
            run_config: Configuration used for assembling.
        """
        # Assemble all circuits first before submitting them with threads to
        # avoid conflicts with terra assembler.
        exp_index = 0
        logger.debug("Assembling all circuits.")
        for idx, circs in enumerate(circuit_lists):
            qobj = assemble(circs, backend=self.backend(), **run_config)
            self._sub_jobs.append(
                SubJob(
                    start_index=exp_index,
                    end_index=exp_index + len(circs) - 1,
                    job_index=idx,
                    total=len(circuit_lists),
                    qobj=qobj,
                )
            )
            exp_index += len(circs)
        self._sub_jobs[0].event.set()  # Tell the first job it can now be submitted.

        for sub_job in self._sub_jobs:
            sub_job.future = self._executor.submit(self._async_submit, sub_job=sub_job)

    def _async_submit(self, sub_job: SubJob) -> None:
        """Submit a Qobj asynchronously.

        Args:
            sub_job: A sub job.

        Raises:
            Exception: If submit failed.
        """
        tags = self._tags.copy()
        tags.append(sub_job.format_tag(self._index_tag))
        tags.append(self.job_id())
        job: Optional[IBMCircuitJob] = None
        logger.debug(
            "Submitting job %s for circuits %s-%s.",
            sub_job.job_index,
            sub_job.start_index,
            sub_job.end_index,
        )
        sub_job.event.wait()

        try:
            while job is None:
                if self._user_cancelled:
                    return  # Abandon submit if user cancelled.
                try:
                    job = auto_retry(
                        self.backend()._submit_job,
                        qobj=sub_job.qobj,
                        job_name=self._name,
                        job_tags=tags,
                        composite_job_id=self.job_id(),
                    )
                except IBMBackendJobLimitError:
                    oldest_running = self._provider.backend.jobs(
                        limit=1,
                        descending=False,
                        status=list(set(JobStatus) - set(JOB_FINAL_STATES)),
                    )
                    if oldest_running:
                        oldest_running = oldest_running[0]
                        logger.warning(
                            "Job limit reached, waiting for job %s to finish "
                            "before submitting the next one.",
                            oldest_running.job_id(),
                        )
                        try:
                            # Set a timeout in case the job is stuck.
                            oldest_running.wait_for_final_state(timeout=300)
                        except Exception as err:  # pylint: disable=broad-except
                            # Don't kill the submit if unable to wait for old job.
                            logger.debug(
                                "An error occurred while waiting for "
                                "job %s to finish: %s",
                                oldest_running.job_id(),
                                err,
                            )
                except Exception as err:  # pylint: disable=broad-except
                    sub_job.submit_error = err
                    logger.debug(
                        "An error occurred submitting sub-job %s: %s",
                        sub_job.job_index,
                        traceback.format_exc(),
                    )
                    raise

            if self._user_cancelled:
                job.cancel()
            sub_job.job = job
            logger.debug(
                "Job %s submitted for circuits %s-%s.",
                job.job_id(),
                sub_job.start_index,
                sub_job.end_index,
            )
        finally:
            try:
                # Wake up the next submit.
                next(
                    sub_job.event
                    for sub_job in self._sub_jobs
                    if not sub_job.event.is_set()
                ).set()
            except StopIteration:
                pass

    @_requires_submit
    def properties(
        self, refresh: bool = False
    ) -> Optional[Union[List[BackendProperties], BackendProperties]]:
        """Return the backend properties for this job.

         Args:
            refresh: If ``True``, re-query the server for the backend properties.
                Otherwise, return a cached version.

        Note:
            This method blocks until all sub-jobs are submitted.

        Returns:
            The backend properties used for this job, or ``None`` if
            properties are not available. A list of backend properties is
            returned if the sub-jobs used different properties.

        Raises:
            IBMJobApiError: If an unexpected error occurred when communicating
                with the server.
        """
        if self._properties is None:
            self._properties = []
            properties_ts = []
            for job in self._get_circuit_jobs():
                props = job.properties(refresh)
                if props.last_update_date not in properties_ts:
                    self._properties.append(props)
                    properties_ts.append(props.last_update_date)

        if not self._properties:
            return None
        if len(self._properties) == 1:
            return self._properties[0]
        return self._properties

    def result(
        self,
        timeout: Optional[float] = None,
        wait: float = 5,
        partial: bool = False,
        refresh: bool = False,
    ) -> Result:
        """Return the result of the job.

        Note:
            This method blocks until all sub-jobs finish.

        Note:
            Some IBM Quantum job results can only be read once. A
            second attempt to query the server for the same job will fail,
            since the job has already been "consumed".

            The first call to this method in an ``IBMCompositeJob`` instance will
            query the server and consume any available job results. Subsequent
            calls to that instance's ``result()`` will also return the results, since
            they are cached. However, attempting to retrieve the results again in
            another instance or session might fail due to the job results
            having been consumed.

        Note:
            When `partial=True`, this method will attempt to retrieve partial
            results of failed jobs. In this case, precaution should
            be taken when accessing individual experiments, as doing so might
            cause an exception. The ``success`` attribute of the returned
            :class:`~qiskit.result.Result` instance can be used to verify
            whether it contains partial results.

            For example, if one of the circuits in the job failed, trying to
            get the counts of the unsuccessful circuit would raise an exception
            since there are no counts to return::

                try:
                    counts = result.get_counts("failed_circuit")
                except QiskitError:
                    print("Circuit execution failed!")

        If the job failed, you can use :meth:`error_message()` to get more information.

        Args:
            timeout: Number of seconds to wait for job.
            wait: Time in seconds between queries.
            partial: If ``True``, return partial results if possible. Partial results
                refer to experiments within a sub-job, not individual sub-jobs.
                That is, this method will still block until all sub-jobs finish
                even if `partial` is set to ``True``.
            refresh: If ``True``, re-query the server for the result. Otherwise
                return the cached value.

        Returns:
            Job result.

        Raises:
            IBMJobInvalidStateError: If the job was cancelled.
            IBMJobFailureError: If the job failed.
            IBMJobApiError: If an unexpected error occurred when communicating
                with the server.
        """
        # pylint: disable=arguments-differ
        self.wait_for_final_state(wait=wait, timeout=timeout)
        if self._status == JobStatus.DONE or partial:
            self._result = self._gather_results(refresh=refresh, partial=partial)
            if self._result is not None:
                return self._result

        if self._status is JobStatus.CANCELLED:
            raise IBMJobInvalidStateError(
                "Unable to retrieve result for job {}. "
                "Job was cancelled.".format(self.job_id())
            )
        error_message = self.error_message()
        if "\n" in error_message:
            error_message = ". Use the error_message() method to get more details."
        else:
            error_message = ": " + error_message
        raise IBMJobFailureError(
            "Unable to retrieve result for job {}. Job has failed{}".format(
                self.job_id(), error_message
            )
        )

    def cancel(self) -> bool:
        """Attempt to cancel the job.

        Note:
            Depending on the state the job is in, it might be impossible to
            cancel the job.

        Returns:
            ``True`` if the job is cancelled, else ``False``.

        Raises:
            IBMJobApiError: If an unexpected error occurred when communicating
                with the server.
        """
        self._user_cancelled = True
        # Wake up all pending job submits.
        for sub_job in self._sub_jobs:
            sub_job.event.set()

        all_cancelled = []
        for job in self._get_circuit_jobs():
            try:
                all_cancelled.append(job.cancel())
            except IBMJobApiError as err:
                if "Error code: 3209" not in str(err):
                    raise

        return all(all_cancelled)

    @_requires_submit
    def update_name(self, name: str) -> str:
        """Update the name associated with this job.

        Note:
            This method blocks until all sub-jobs are submitted.

        Args:
            name: The new `name` for this job.

        Returns:
            The new name associated with this job.

        Raises:
            IBMJobApiError: If an unexpected error occurred when communicating
                with the server or updating the job name.
            IBMJobInvalidStateError: If the input job name is not a string.
        """
        if not isinstance(name, str):
            raise IBMJobInvalidStateError(
                '"{}" of type "{}" is not a valid job name. '
                "The job name needs to be a string.".format(name, type(name))
            )

        self._name = name
        for job in self._get_circuit_jobs():
            auto_retry(job.update_name, name)

        return self._name

    @_requires_submit
    def update_tags(self, new_tags: List[str]) -> List[str]:
        """Update the tags associated with this job.

        Note:
            This method blocks until all sub-jobs are submitted.

        Args:
            new_tags: New tags to assign to the job.

        Returns:
            The new tags associated with this job.

        Raises:
            IBMJobApiError: If an unexpected error occurred when communicating
                with the server or updating the job tags.
            IBMJobInvalidStateError: If none of the input parameters are specified or
                if any of the input parameters are invalid.
        """
        validate_job_tags(new_tags, IBMJobInvalidStateError)
        new_tags_set = set(new_tags)

        for job in self._get_circuit_jobs():
            tags_to_update = new_tags_set.union(  # type: ignore[attr-defined]
                {
                    tag
                    for tag in job.tags()
                    if tag.startswith(IBM_COMPOSITE_JOB_TAG_PREFIX)
                }
            )
            auto_retry(job.update_tags, list(tags_to_update))
        self._tags = new_tags
        return self._tags

    def status(self) -> JobStatus:
        """Query the server for the latest job status.

        Note:
            This method is not designed to be invoked repeatedly in a loop for
            an extended period of time. Doing so may cause the server to reject
            your request.
            Use :meth:`wait_for_final_state()` if you want to wait for the job to finish.

        Note:
            If the job failed, you can use :meth:`error_message()` to get
            more information.

        Note:
            Since this job contains multiple sub-jobs, the returned status is mapped
            in the following order:

                * INITIALIZING - if any sub-job is being initialized.
                * VALIDATING - if any sub-job is being validated.
                * QUEUED - if any sub-job is queued.
                * RUNNING - if any sub-job is still running.
                * ERROR - if any sub-job incurred an error.
                * CANCELLED - if any sub-job is cancelled.
                * DONE - if all sub-jobs finished.

        Returns:
            The status of the job.

        Raises:
            IBMJobApiError: If an unexpected error occurred when communicating
                with the server.
        """
        self._update_status_queue_info_error()
        return self._status

    def report(self, detailed: bool = True) -> str:
        """Return a report on current sub-job statuses.

        Args:
            detailed: If ``True``, return a detailed report. Otherwise return a
                summary report.

        Returns:
            A report on sub-job statuses.
        """
        report = [f"Composite Job {self.job_id()}:", "  Summary report:"]

        status_counts: Dict[JobStatus, int] = defaultdict(int)
        status_by_id = {}  # Used to save status to keep things in sync.
        for job in self._get_circuit_jobs():
            status = job.status()
            status_counts[status] += 1
            status_by_id[job.job_id()] = status
        status_counts[JobStatus.ERROR] += len(
            [sub_job for sub_job in self._sub_jobs if sub_job.submit_error]
        )

        # Summary report.
        count_report = []
        # Format header.
        for stat in [
            "Total",
            "Successful",
            "Failed",
            "Cancelled",
            "Running",
            "Pending",
        ]:
            count_report.append(" " * 4 + stat + " jobs: {}")
        max_text = max(len(text) for text in count_report)
        count_report = [text.rjust(max_text) for text in count_report]
        # Format counts.
        count_report[0] = count_report[0].format(len(self._sub_jobs))
        non_pending_count = 0
        for idx, stat in enumerate(
            [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED, JobStatus.RUNNING], 1
        ):
            non_pending_count += status_counts[stat]
            count_report[idx] = count_report[idx].format(status_counts[stat])
        count_report[-1] = count_report[-1].format(
            len(self._sub_jobs) - non_pending_count
        )
        report += count_report

        # Detailed report.
        if detailed:
            report.append("\n  Detail report:")
            for sub_job in self._sub_jobs:
                report.append(
                    " " * 4 + f"Circuits {sub_job.start_index}-{sub_job.end_index}:"
                )
                report.append(" " * 6 + f"Job index: {sub_job.job_index}")
                if sub_job.job and sub_job.job.job_id() in status_by_id:
                    report.append(" " * 6 + f"Job ID: {sub_job.job.job_id()}")
                    report.append(
                        " " * 6 + f"Status: {status_by_id[sub_job.job.job_id()]}"
                    )
                elif sub_job.submit_error:
                    report.append(" " * 6 + f"Status: {sub_job.submit_error}")
                else:
                    report.append(" " * 6 + "Status: Job not yet submitted.")

        return "\n".join(report)

    def error_message(self) -> Optional[str]:
        """Provide details about the reason of failure.

        Note:
            This method blocks until the job finishes.

        Returns:
            An error report if the job failed or ``None`` otherwise.
        """
        self.wait_for_final_state()
        if self._status != JobStatus.ERROR:
            return None

        if not self._job_error_msg:
            self._update_status_queue_info_error()

        return self._job_error_msg

    def queue_position(self, refresh: bool = False) -> Optional[int]:
        """Return the position of the job in the server queue.

        This method returns the queue position of the sub-job that is
        last in queue.

        Note:
            The position returned is within the scope of the provider
            and may differ from the global queue position.

        Args:
            refresh: If ``True``, re-query the server to get the latest value.
                Otherwise return the cached value.

        Returns:
            Position in the queue or ``None`` if position is unknown or not applicable.
        """
        if refresh:
            self._update_status_queue_info_error()
        if self._status != JobStatus.QUEUED:
            self._queue_info = None
            return None
        return self._queue_info.queue_position if self._queue_info else None

    def queue_info(self) -> Optional[QueueInfo]:
        """Return queue information for this job.

        This method returns the queue information of the sub-job that is
        last in queue.

        The queue information may include queue position, estimated start and
        end time, and dynamic priorities for the hub, group, and project. See
        :class:`QueueInfo` for more information.

        Note:
            The queue information is calculated after the job enters the queue.
            Therefore, some or all of the information may not be immediately
            available, and this method may return ``None``.

        Returns:
            A :class:`QueueInfo` instance that contains queue information for
            this job, or ``None`` if queue information is unknown or not
            applicable.
        """
        self._update_status_queue_info_error()
        if self._status != JobStatus.QUEUED:
            self._queue_info = None
        return self._queue_info

    def creation_date(self) -> Optional[python_datetime]:
        """Return job creation date, in local time.

        Returns:
            The job creation date as a datetime object, in local time, or
            ``None`` if job submission hasn't finished or failed.
        """
        if not self._creation_date:
            circuit_jobs = self._get_circuit_jobs()
            if not circuit_jobs:
                return None
            self._creation_date = min(job.creation_date() for job in circuit_jobs)
        return self._creation_date

    def time_per_step(self) -> Optional[Dict]:
        """Return the date and time information on each step of the job processing.

        The output dictionary contains the date and time information on each
        step of the job processing, in local time. The keys of the dictionary
        are the names of the steps, and the values are the date and time data,
        as a datetime object with local timezone info.
        For example::

            {'CREATING': datetime(2020, 2, 13, 15, 19, 25, 717000, tzinfo=tzlocal(),
             'CREATED': datetime(2020, 2, 13, 15, 19, 26, 467000, tzinfo=tzlocal(),
             'VALIDATING': datetime(2020, 2, 13, 15, 19, 26, 527000, tzinfo=tzlocal()}

        Returns:
            Date and time information on job processing steps, in local time,
            or ``None`` if the information is not yet available.
        """
        output = None
        creation_date = self.creation_date()
        if creation_date is not None:
            output = {"CREATING": creation_date}
        if self._has_pending_submit():
            return output

        timestamps = defaultdict(list)
        for job in self._get_circuit_jobs():
            job_timestamps = job.time_per_step()
            if job_timestamps is None:
                continue
            for key, val in job_timestamps.items():
                timestamps[key].append(val)

        self._update_status_queue_info_error()
        for key, val in timestamps.items():
            if (
                JOB_STATUS_TO_INT[api_status_to_job_status(key)]
                > JOB_STATUS_TO_INT[self._status]
            ):
                continue
            if key == "CREATING":
                continue
            if key in ["TRANSPILING", "VALIDATING", "QUEUED", "RUNNING"]:
                output[key] = sorted(val)[0]
            else:
                output[key] = sorted(val, reverse=True)[0]

        return output

    def scheduling_mode(self) -> Optional[str]:
        """Return the scheduling mode the job is in.

        The scheduling mode indicates how the job is scheduled to run. For example,
        ``fairshare`` indicates the job is scheduled using a fairshare algorithm.

        ``fairshare`` is returned if any of the sub-jobs has scheduling mode of
        ``fairshare``.

        This information is only available if the job status is ``RUNNING`` or ``DONE``.

        Returns:
            The scheduling mode the job is in or ``None`` if the information
            is not available.
        """
        if self._has_pending_submit():
            return None

        mode = None
        for job in self._get_circuit_jobs():
            job_mode = job.scheduling_mode()
            if job_mode == "fairshare":
                return "fairshare"
            if job_mode:
                mode = job_mode
        return mode

    @property
    def client_version(self) -> Dict[str, str]:
        """Return version of the client used for this job.

        Returns:
            Client version in dictionary format, where the key is the name
                of the client and the value is the version. An empty dictionary
                is returned if the information is not yet known.
        """
        circuit_jobs = self._get_circuit_jobs()
        if not self._client_version and circuit_jobs:
            self._client_version = circuit_jobs[0].client_version

        return self._client_version

    def refresh(self) -> None:
        """Obtain the latest job information from the server.

        This method may add additional attributes to this job instance, if new
        information becomes available.

        Raises:
            IBMJobApiError: If an unexpected error occurred when communicating
                with the server.
        """
        for job in self._get_circuit_jobs():
            if job.status() not in JOB_FINAL_STATES:
                job.refresh()

    def circuits(self) -> List[QuantumCircuit]:
        """Return the circuits for this job.

        Returns:
            The circuits for this job.
        """
        if not self._circuits:
            qobj = self._get_qobj()
            self._circuits, _, _ = disassemble(qobj)

        return self._circuits

    def backend_options(self) -> Dict[str, Any]:
        """Return the backend configuration options used for this job.

        Options that are not applicable to the job execution are not returned.
        Some but not all of the options with default values are returned.
        You can use :attr:`qiskit_ibm_provider.IBMBackend.options` to see
        all backend options.

        Returns:
            Backend options used for this job.
        """
        qobj = self._get_qobj()
        _, options, _ = disassemble(qobj)
        return options

    def header(self) -> Dict:
        """Return the user header specified for this job.

        Returns:
            User header specified for this job. An empty dictionary
            is returned if the header cannot be retrieved.
        """
        qobj = self._get_qobj()
        _, _, header = disassemble(qobj)
        return header

    @_requires_submit
    def wait_for_final_state(
        self,
        timeout: Optional[float] = None,
        wait: Optional[float] = None,
        callback: Optional[Callable] = None,
    ) -> None:
        """Wait until the job progresses to a final state such as ``DONE`` or ``ERROR``.

        Args:
            timeout: Seconds to wait for the job. If ``None``, wait indefinitely.
            wait: Seconds to wait between invoking the callback function. If ``None``,
                the callback function is invoked only if job status or queue position
                has changed.
            callback: Callback function invoked after each querying iteration.
                The following positional arguments are provided to the callback function:

                    * job_id: Job ID
                    * job_status: Status of the job from the last query.
                    * job: This ``IBMCompositeJob`` instance.

                In addition, the following keyword arguments are also provided:

                    * queue_info: A :class:`QueueInfo` instance with job queue information,
                      or ``None`` if queue information is unknown or not applicable.
                      You can use the ``to_dict()`` method to convert the
                      :class:`QueueInfo` instance to a dictionary, if desired.

        Raises:
            IBMJobTimeoutError: if the job does not reach a final state before the
                specified timeout.
        """
        if self._status in JOB_FINAL_STATES:
            return

        self._user_callback = callback
        self._user_wait_value = wait
        status_callback = self._status_callback if callback else None

        # We need to monitor all jobs to give the most up-to-date information
        # to the user callback function. Websockets are preferred to avoid
        # excessive requests.
        job_futures = []
        for job in self._get_circuit_jobs():
            job_futures.append(
                self._executor.submit(
                    job.wait_for_final_state,
                    timeout=timeout,
                    wait=wait,
                    callback=status_callback,
                )
            )
        future_stats = futures.wait(job_futures, timeout=timeout)
        self._update_status_queue_info_error()
        if future_stats[1]:
            raise IBMJobTimeoutError(
                f"Timeout waiting for job {self.job_id()}"
            ) from None
        for fut in future_stats[0]:
            exception = fut.exception()
            if exception is not None:
                raise exception from None

    def sub_jobs(self, block_for_submit: bool = True) -> List[IBMCircuitJob]:
        """Return all submitted sub-jobs.

        Args:
            block_for_submit: ``True`` if this method should block until
                all sub-jobs are submitted. ``False`` if the method should
                return immediately with submitted sub-jobs, if any.

        Returns:
            All submitted sub-jobs.
        """
        if block_for_submit:
            self.block_for_submit()
        return self._get_circuit_jobs()

    def sub_job(self, circuit_index: int) -> Optional[IBMCircuitJob]:
        """Retrieve the job used to submit the specified circuit.

        Args:
            circuit_index: Index of the circuit whose job is to be returned.

        Returns:
            The Job submitted for the circuit, or ``None`` if the job has
            not been submitted or the submit failed.

        Raises:
            IBMJobInvalidStateError: If the circuit index is out of range.
        """
        last_index = self._sub_jobs[-1].end_index
        if circuit_index > last_index:
            raise IBMJobInvalidStateError(
                f"Circuit index {circuit_index} greater than circuit count {last_index}."
            )

        for sub_job in self._sub_jobs:
            if sub_job.end_index >= circuit_index >= sub_job.start_index:
                return sub_job.job

        return None

    def rerun_failed(self) -> None:
        """Re-submit all failed sub-jobs.

        Note:
            All sub-jobs that are in "ERROR" or "CANCELLED" states will
            be re-submitted.
            Sub-jobs that failed to be submitted will only be re-submitted if the
            circuits are known. That is, if this ``IBMCompositeJob`` was
            returned by :meth:`qiskit_ibm_provider.IBMBackend.run` and not
            retrieved from the server.
        """
        for sub_job in self._sub_jobs:
            if sub_job.submit_error is not None or (
                sub_job.job
                and sub_job.job.status() in [JobStatus.ERROR, JobStatus.CANCELLED]
            ):
                sub_job.reset()
                sub_job.future = self._executor.submit(
                    self._async_submit, sub_job=sub_job
                )
        try:
            next(
                sub_job.event
                for sub_job in self._sub_jobs
                if not sub_job.event.is_set()
            ).set()
        except StopIteration:
            pass
        self._status = JobStatus.INITIALIZING

    def block_for_submit(self) -> None:
        """Block until all sub-jobs are submitted."""
        futures.wait(
            [sub_job.future for sub_job in self._sub_jobs if sub_job.future is not None]
        )

    def _status_callback(
        self,
        job_id: str,
        job_status: JobStatus,
        job: IBMCircuitJob,  # pylint: disable=unused-argument
        queue_info: QueueInfo,
    ) -> None:
        """Callback function used when a sub-job status changes.

        Args:
            job_id: Sub-job ID.
            job_status: Sub-job status.
            job: Sub-job.
            queue_info: Sub-job queue info.
        """
        with self._callback_lock:
            self._job_statuses[job_id] = JobStatusQueueInfo(job_status, queue_info)
            status, queue_info = last_job_stat_pos(list(self._job_statuses.values()))
            pos = queue_info.position if queue_info else None

            report = False
            cur_time = time.time()
            if self._user_wait_value is None:
                if self._last_reported_stat != (status, pos):
                    report = True
            elif cur_time - self._last_reported_time >= self._user_wait_value:
                report = True

            self._last_reported_stat = (status, pos)
            self._last_reported_time = cur_time
            if report and self._user_callback:
                logger.debug(
                    "Invoking callback function, job status=%s, queue_info=%s",
                    status,
                    queue_info,
                )
                self._user_callback(self.job_id(), status, self, queue_info=queue_info)

    def _update_status_queue_info_error(self) -> None:
        """Update the status, queue information, and error message of this composite job."""
        # pylint: disable=too-many-return-statements

        if self._has_pending_submit():
            self._status = JobStatus.INITIALIZING
            return

        if self._status in [JobStatus.CANCELLED, JobStatus.DONE]:
            return
        if self._status is JobStatus.ERROR and self._job_error_msg:
            return

        statuses: Dict[JobStatus, List[SubJob]] = defaultdict(list)
        for sub_job in self._sub_jobs:
            if sub_job.job:
                statuses[sub_job.job.status()].append(sub_job)
            elif sub_job.submit_error:
                statuses[JobStatus.ERROR].append(sub_job)
            else:
                statuses[JobStatus.INITIALIZING].append(sub_job)

        for stat in [JobStatus.INITIALIZING, JobStatus.VALIDATING]:
            if stat in statuses:
                self._status = stat
                return

        if JobStatus.QUEUED in statuses:
            self._status = JobStatus.QUEUED
            # Sort by queue position. Put `None` to the end.
            last_queued = sorted(
                statuses[JobStatus.QUEUED],
                key=lambda sub_j: (
                    sub_j.job.queue_position() is None,
                    sub_j.job.queue_position(),
                ),
            )[-1]
            self._queue_info = last_queued.job.queue_info()
            return

        if JobStatus.RUNNING in statuses:
            self._status = JobStatus.RUNNING
            return

        if JobStatus.ERROR in statuses:
            self._status = JobStatus.ERROR
            self._build_error_report(statuses.get(JobStatus.ERROR, []))
            return

        for stat in [JobStatus.CANCELLED, JobStatus.DONE]:
            if stat in statuses:
                self._status = stat
                return

        bad_stats = {k: v for k, v in statuses.items() if k not in JobStatus}
        raise IBMJobInvalidStateError("Invalid job status found: " + str(bad_stats))

    def _build_error_report(self, failed_jobs: List[SubJob]) -> None:
        """Build the error report.

        Args:
            failed_jobs: A list of failed jobs.
        """
        error_list = []
        for sub_job in failed_jobs:
            error_text = f"Circuits {sub_job.start_index}-{sub_job.end_index}: "
            if sub_job.submit_error:
                error_text += f"Job submit failed: {sub_job.submit_error}"
            elif sub_job.job:
                error_text += (
                    f"Job {sub_job.job.job_id()} failed: {sub_job.job.error_message()}"
                )
            error_list.append(error_text)

        if len(error_list) > 1:
            self._job_error_msg = "The following circuits failed:\n{}".format(
                "\n".join(error_list)
            )
        else:
            self._job_error_msg = error_list[0]

    def _find_circuit_indexes(self, job: IBMCircuitJob) -> List[int]:
        """Find the circuit indexes of the input job.

        Args:
            job: The circuit job.

        Returns:
            A list of job index, total jobs, start index, and end index.

        Raises:
            IBMJobInvalidStateError: If a sub-job is missing proper tags.
        """
        index_tag = [
            tag for tag in job.tags() if tag.startswith(IBM_COMPOSITE_JOB_INDEX_PREFIX)
        ]
        match = None
        if index_tag:
            match = re.match(self._index_pattern, index_tag[0])
        if match is None:
            raise IBMJobInvalidStateError(
                f"Job {job.job_id()} in composite job {self.job_id()}"
                f" is missing proper tags."
            )
        output = []
        for name in ["job_index", "total_jobs", "start_index", "end_index"]:
            output.append(int(match.group(name)))
        return output

    def _has_pending_submit(self) -> bool:
        """Return whether there are pending job submit futures.

        Returns:
            Whether there are pending job submit futures.
        """
        sub_fut = [
            sub_job.future for sub_job in self._sub_jobs if sub_job.future is not None
        ]
        return not all(fut.done() for fut in sub_fut)

    def _gather_results(self, refresh: bool, partial: bool) -> Optional[Result]:
        """Retrieve the job result response.

        Args:
            refresh: If ``True``, re-query the server for the result.
               Otherwise return the cached value.
            partial: Whether partial result should be collected.

        Returns:
            Combined job result, or ``None`` if not all sub-jobs have finished.

        Raises:
            IBMJobApiError: If an unexpected error occurred when communicating
                with the server.
        """
        if self._result and not refresh:
            return self._result

        job_results = [
            sub_job.result(refresh=refresh, partial=partial)
            for sub_job in self._sub_jobs
        ]

        if not partial and any(result is None for result in job_results):
            return None
        try:
            good_result = next(res for res in job_results if res is not None).to_dict()
        except StopIteration:
            return None

        ref_expr_result = good_result["results"][0]
        template_expr_result = {"success": False, "data": {}, "status": "ERROR"}
        for key in ["shots", "meas_level", "seed", "meas_return"]:
            template_expr_result[key] = ref_expr_result.get(key, None)

        good_result["job_id"] = self.job_id()
        good_result["success"] = self._status == JobStatus.DONE
        good_result["results"] = []
        good_result["status"] = (
            "PARTIAL COMPLETED" if self._status != JobStatus.DONE else "COMPLETED"
        )
        combined_result = Result.from_dict(good_result)

        for idx, result in enumerate(job_results):
            if result is not None:
                combined_result.results.extend(result.results)
            else:
                # Get experiment header from Qobj if possible.
                sub_job = self._sub_jobs[idx]
                qobj = sub_job.qobj
                experiments = qobj.experiments if qobj else None

                expr_results = []
                for circ_idx in range(sub_job.end_index - sub_job.start_index + 1):
                    if experiments:
                        template_expr_result["header"] = experiments[
                            circ_idx
                        ].header.to_dict()
                    expr_results.append(
                        ExperimentResult.from_dict(template_expr_result)
                    )
                combined_result.results.extend(expr_results)

        return combined_result

    def _get_qobj(self) -> Optional[Union[QasmQobj, PulseQobj]]:
        """Return the Qobj for this job.

        Returns:
            The Qobj for this job, or ``None`` if the job does not have a Qobj.

        Raises:
            IBMJobApiError: If an unexpected error occurred when retrieving
                job information from the server.
        """
        if not self._qobj:
            self._qobj = copy.deepcopy(self._sub_jobs[0].qobj)
            for idx in range(1, len(self._get_circuit_jobs())):
                self._qobj.experiments.extend(self._sub_jobs[idx].qobj.experiments)
        return self._qobj

    def _get_circuit_jobs(self) -> List[IBMCircuitJob]:
        """Get all circuit jobs.

        Returns:
            A list of circuit jobs.
        """
        return [sub_job.job for sub_job in self._sub_jobs if sub_job.job]

    def submit(self) -> None:
        """Unsupported method.

        Note:
            This method is not supported, please use
            :meth:`~qiskit_ibm_provider.ibm_backend.IBMBackend.run`
            to submit a job.

        Raises:
            NotImplementedError: Upon invocation.
        """
        raise NotImplementedError(
            "job.submit() is not supported. Please use "
            "IBMBackend.run() to submit a job."
        )
