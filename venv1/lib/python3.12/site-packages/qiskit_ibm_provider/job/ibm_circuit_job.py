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

"""IBM Quantum job."""

import json
import logging
import time
import queue
from concurrent import futures
from datetime import datetime
from typing import Dict, Optional, Any, List
import re
import requests

import dateutil.parser
from qiskit.providers.jobstatus import JOB_FINAL_STATES, JobStatus
from qiskit.circuit.quantumcircuit import QuantumCircuit

from qiskit.result import Result

from qiskit_ibm_provider import ibm_backend  # pylint: disable=unused-import
from .constants import IBM_COMPOSITE_JOB_TAG_PREFIX, IBM_MANAGED_JOB_ID_PREFIX
from .exceptions import (
    IBMJobError,
    IBMJobApiError,
    IBMJobFailureError,
    IBMJobTimeoutError,
    IBMJobInvalidStateError,
)
from .ibm_job import IBMJob
from .queueinfo import QueueInfo
from .utils import build_error_report, api_to_job_error
from ..api.clients import (
    AccountClient,
    RuntimeClient,
    RuntimeWebsocketClient,
    WebsocketClientCloseCode,
)
from ..api.exceptions import ApiError, RequestsApiError
from ..apiconstants import ApiJobStatus, ApiJobKind
from ..utils.converters import utc_to_local
from ..utils.json_decoder import decode_result
from ..utils.json import RuntimeDecoder
from ..utils.utils import validate_job_tags, api_status_to_job_status

logger = logging.getLogger(__name__)


class IBMCircuitJob(IBMJob):
    """Representation of a job that executes on an IBM Quantum backend.

    The job may be executed on a simulator or a real device. A new ``IBMCircuitJob``
    instance is returned when you call
    :meth:`IBMBackend.run()<qiskit_ibm_provider.ibm_backend.IBMBackend.run()>`
    to submit a job to a particular backend.

    If the job is successfully submitted, you can inspect the job's status by
    calling :meth:`status()`. Job status can be one of the
    :class:`~qiskit.providers.JobStatus` members.
    For example::

        from qiskit.providers.jobstatus import JobStatus

        job = backend.run(...)

        try:
            job_status = job.status()  # Query the backend server for job status.
            if job_status is JobStatus.RUNNING:
                print("The job is still running")
        except IBMJobApiError as ex:
            print("Something wrong happened!: {}".format(ex))

    Note:
        An error may occur when querying the remote server to get job information.
        The most common errors are temporary network failures
        and server errors, in which case an
        :class:`~qiskit_ibm_provider.job.IBMJobApiError`
        is raised. These errors usually clear quickly, so retrying the operation is
        likely to succeed.

    Some of the methods in this class are blocking, which means control may
    not be returned immediately. :meth:`result()` is an example
    of a blocking method::

        job = backend.run(...)

        try:
            job_result = job.result()  # It will block until the job finishes.
            print("The job finished with result {}".format(job_result))
        except JobError as ex:
            print("Something wrong happened!: {}".format(ex))

    Job information retrieved from the server is attached to the ``IBMCircuitJob``
    instance as attributes. Given that Qiskit and the server can be updated
    independently, some of these attributes might be deprecated or experimental.
    Supported attributes can be retrieved via methods. For example, you
    can use :meth:`creation_date()` to retrieve the job creation date,
    which is a supported attribute.
    """

    _executor = futures.ThreadPoolExecutor()
    """Threads used for asynchronous processing."""

    def __init__(
        self,
        backend: "ibm_backend.IBMBackend",
        api_client: AccountClient,
        job_id: str,
        creation_date: Optional[str] = None,
        status: Optional[str] = None,
        runtime_client: RuntimeClient = None,  # TODO: make mandatory after completely switching
        kind: Optional[str] = None,
        name: Optional[str] = None,
        time_per_step: Optional[dict] = None,
        result: Optional[dict] = None,
        error: Optional[dict] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        run_mode: Optional[str] = None,
        client_info: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """IBMCircuitJob constructor.

        Args:
            backend: The backend instance used to run this job.
            api_client: Object for connecting to the server.
            job_id: Job ID.
            creation_date: Job creation date.
            status: Job status returned by the server.
            runtime_client: Object for connecting to the runtime server
            kind: Job type.
            name: Job name.
            time_per_step: Time spent for each processing step.
            result: Job result.
            error: Job error.
            tags: Job tags.
            run_mode: Scheduling mode the job runs in.
            client_info: Client information from the API.
            kwargs: Additional job attributes.
        """
        super().__init__(
            backend=backend,
            api_client=api_client,
            job_id=job_id,
            name=name,
            session_id=session_id,
            tags=tags,
        )
        self._runtime_client = runtime_client
        self._creation_date = None
        if creation_date is not None:
            self._creation_date = dateutil.parser.isoparse(creation_date)
        self._api_status = status
        self._kind = ApiJobKind(kind) if kind else None
        self._time_per_step = time_per_step
        self._error = error
        self._run_mode = run_mode
        self._status = None
        self._params: Dict[str, Any] = None
        self._queue_info: QueueInfo = None
        if status is not None:
            self._status = api_status_to_job_status(status)
        self._client_version = self._extract_client_version(client_info)
        self._set_result(result)
        self._usage_estimation: Dict[str, Any] = {}

        # Properties used for caching.
        self._cancelled = False
        self._job_error_msg = None  # type: Optional[str]
        self._refreshed = False

        self._ws_client_future = None  # type: Optional[futures.Future]
        self._result_queue = queue.Queue()  # type: queue.Queue
        self._ws_client = RuntimeWebsocketClient(
            websocket_url=self._api_client._params.get_runtime_api_base_url().replace(
                "https", "wss"
            ),
            client_params=self._api_client._params,
            job_id=job_id,
            message_queue=self._result_queue,
        )

    def result(  # type: ignore[override]
        self,
        timeout: Optional[float] = None,
        refresh: bool = False,
    ) -> Result:
        """Return the result of the job.

        Note:
            Some IBM Quantum job results can only be read once. A
            second attempt to query the server for the same job will fail,
            since the job has already been "consumed".

            The first call to this method in an ``IBMCircuitJob`` instance will
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

            For example, if one of the experiments in the job failed, trying to
            get the counts of the unsuccessful experiment would raise an exception
            since there are no counts to return::

                try:
                    counts = result.get_counts("failed_experiment")
                except QiskitError:
                    print("Experiment failed!")

        If the job failed, you can use :meth:`error_message()` to get more information.

        Args:
            timeout: Number of seconds to wait for job.
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
        if self._result is None or refresh:
            self.wait_for_final_state(timeout=timeout)
            if self._status is JobStatus.CANCELLED:
                raise IBMJobInvalidStateError(
                    "Unable to retrieve result for job {}. "
                    "Job was cancelled.".format(self.job_id())
                )
            if self._status == JobStatus.ERROR:
                error_message = self.error_message()
                raise IBMJobFailureError(f"Job failed: " f"{error_message}")
            self._retrieve_result(refresh=refresh)
        return self._result

    def cancel(self) -> bool:
        """Attempt to cancel the job.

        Note:
            Depending on the state the job is in, it might be impossible to
            cancel the job.

        Returns:
            ``True`` if the job is cancelled, else ``False``.

        Raises:
            IBMJobInvalidStateError: If the job is in a state that cannot be cancelled.
            IBMJobError: If unable to cancel job.
        """
        try:
            self._runtime_client.job_cancel(self.job_id())
            self._cancelled = True
            logger.debug(
                'Job %s cancel status is "%s".',
                self.job_id(),
                self._cancelled,
            )
            self._ws_client.disconnect(WebsocketClientCloseCode.CANCEL)
            self._status = JobStatus.CANCELLED
            return self._cancelled
        except RequestsApiError as ex:
            if ex.status_code == 409:
                raise IBMJobInvalidStateError(
                    f"Job cannot be cancelled: {ex}"
                ) from None
            raise IBMJobError(f"Failed to cancel job: {ex}") from None

    def update_tags(self, new_tags: List[str]) -> List[str]:
        """Update the tags associated with this job.

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
        # Tags prefix that denotes a job belongs to a jobset or composite job.
        filter_tags = (IBM_MANAGED_JOB_ID_PREFIX, IBM_COMPOSITE_JOB_TAG_PREFIX)
        tags_to_keep = set(filter(lambda x: x.startswith(filter_tags), self._tags))

        tags_to_update = set(new_tags)
        validate_job_tags(new_tags, IBMJobInvalidStateError)
        tags_to_update = tags_to_update.union(tags_to_keep)

        with api_to_job_error():
            response = self._runtime_client.update_tags(
                job_id=self.job_id(), tags=list(tags_to_update)
            )

        if response.status_code == 204:
            with api_to_job_error():
                api_response = self._runtime_client.job_get(self.job_id())
            self._tags = api_response.pop("tags", [])
            return self._tags
        else:
            raise IBMJobApiError(
                "An unexpected error occurred when updating the "
                "tags for job {}. The tags were not updated for "
                "the job.".format(self.job_id())
            )

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

        Returns:
            The status of the job.

        Raises:
            IBMJobApiError: If an unexpected error occurred when communicating
                with the server.
        """
        if self._status is not None and self._status in JOB_FINAL_STATES:
            return self._status

        with api_to_job_error():
            api_response = self._runtime_client.job_get(self.job_id())["state"]
            # response state possibly has two values: status and reason
            # reason is not used in the current interface
            self._api_status = api_response["status"]
            self._status = api_status_to_job_status(self._api_status)

        return self._status

    def error_message(self) -> Optional[str]:
        """Provide details about the reason of failure.

        Returns:
            An error report if the job failed or ``None`` otherwise.
        """
        # pylint: disable=attribute-defined-outside-init
        if self._status in [JobStatus.DONE, JobStatus.CANCELLED]:
            return None
        if self._job_error_msg is not None:
            return self._job_error_msg

        # First try getting error message from the runtime job data
        response = self._runtime_client.job_get(job_id=self.job_id())
        if api_status_to_job_status(response["state"]["status"]) != JobStatus.ERROR:
            return None
        reason = response["state"].get("reason")
        reason_code = response["state"].get("reason_code")
        # If there is a meaningful reason, return it
        if reason is not None and reason != "Error":
            if reason_code:
                self._job_error_msg = f"Error code {reason_code}; {reason}"
            else:
                self._job_error_msg = reason
            return self._job_error_msg

        # Now try parsing a meaningful reason from the results, if possible
        api_result = self._download_external_result(
            self._runtime_client.job_results(self.job_id())
        )
        reason = self._parse_result_for_errors(api_result)
        if reason is not None:
            self._job_error_msg = reason
            return self._job_error_msg

        # We don't really know the error; return the data to the user
        self._job_error_msg = "Unknown error; job result was\n" + api_result
        return self._job_error_msg

    def queue_position(self, refresh: bool = False) -> Optional[int]:
        """Return the position of the job in the server queue.

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
            api_metadata = self._runtime_client.job_metadata(self.job_id())
            self._queue_info = QueueInfo(
                position_in_queue=api_metadata.get("position_in_queue"),
                status=self._api_status,
                estimated_start_time=api_metadata.get("estimated_start_time"),
                estimated_completion_time=api_metadata.get("estimated_completion_time"),
            )

        if self._queue_info:
            return self._queue_info.position
        return None

    def queue_info(self) -> Optional[QueueInfo]:
        """Return queue information for this job.

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
        # Get latest queue information.
        api_metadata = self._runtime_client.job_metadata(self.job_id())
        self._queue_info = QueueInfo(
            position_in_queue=api_metadata.get("position_in_queue"),
            status=self._api_status,
            estimated_start_time=api_metadata.get("estimated_start_time"),
            estimated_completion_time=api_metadata.get("estimated_completion_time"),
        )
        # Return queue information only if it has any useful information.
        if self._queue_info and any(
            value is not None
            for attr, value in self._queue_info.__dict__.items()
            if not attr.startswith("_") and attr != "job_id"
        ):
            return self._queue_info
        return None

    def creation_date(self) -> datetime:
        """Return job creation date, in local time.

        Returns:
            The job creation date as a datetime object, in local time.
        """
        if self._creation_date is None:
            self.refresh()
        creation_date_local_dt = utc_to_local(self._creation_date)
        return creation_date_local_dt

    def job_id(self) -> str:
        """Return the job ID assigned by the server.

        Returns:
            Job ID.
        """
        return self._job_id

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
        if not self._time_per_step or self._status not in JOB_FINAL_STATES:
            self.refresh()

        # Note: By default, `None` should be returned if no time per step info is available.
        time_per_step_local = None
        if self._time_per_step:
            time_per_step_local = {}
            for step_name, time_data_utc in self._time_per_step.items():
                time_per_step_local[step_name] = (
                    utc_to_local(time_data_utc) if time_data_utc else None
                )

        return time_per_step_local

    @property
    def client_version(self) -> Dict[str, str]:
        """Return version of the client used for this job.

        Returns:
            Client version in dictionary format, where the key is the name
                of the client and the value is the version.
        """
        if not self._client_version and not self._refreshed:
            self.refresh()
        return self._client_version

    @property
    def usage_estimation(self) -> Dict[str, Any]:
        """Return usage estimation information for this job.

        Returns:
            ``quantum_seconds`` which is the estimated quantum time
            of the job in seconds. Quantum time represents the time that
            the QPU complex is occupied exclusively by the job.
        """
        if not self._usage_estimation:
            self.refresh()

        return self._usage_estimation

    def refresh(self) -> None:
        """Obtain the latest job information from the server.

        This method may add additional attributes to this job instance, if new
        information becomes available.

        Raises:
            IBMJobApiError: If an unexpected error occurred when communicating
                with the server.
        """
        # TODO: Change to use runtime response data as much as possible
        with api_to_job_error():
            api_response = self._runtime_client.job_get(self.job_id())
            api_metadata = self._runtime_client.job_metadata(self.job_id())

        try:
            api_response.pop("id")
            self._creation_date = dateutil.parser.isoparse(api_response.pop("created"))
            self._api_status = api_response.pop("state")["status"]
        except (KeyError, TypeError) as err:
            raise IBMJobApiError(
                "Unexpected return value received " "from the server: {}".format(err)
            ) from err
        self._usage_estimation = {
            "quantum_seconds": api_response.pop("estimated_running_time_seconds", None),
        }
        self._time_per_step = api_metadata.get("timestamps", None)
        self._tags = api_response.pop("tags", [])
        self._status = api_status_to_job_status(self._api_status)
        self._params = api_response.get("params", {})
        self._client_version = self._extract_client_version(
            api_metadata.get("qiskit_version", None)
        )
        if self._status == JobStatus.DONE:
            api_result = self._download_external_result(
                self._runtime_client.job_results(self.job_id())
            )
            self._set_result(api_result)
        self._refreshed = True

    def backend_options(self) -> Dict:
        """Return the backend configuration options used for this job.

        Options that are not applicable to the job execution are not returned.
        Some but not all of the options with default values are returned.
        You can use :attr:`qiskit_ibm_provider.IBMBackend.options` to see
        all backend options.

        Returns:
            Backend options used for this job. An empty dictionary
            is returned if the options cannot be retrieved.
        """
        self._get_params()
        if self._params:
            return {
                k: v
                for (k, v) in self._params.items()
                if k not in ["header", "circuits"]
            }
        return {}

    def header(self) -> Dict:
        """Return the user header specified for this job.

        Returns:
            User header specified for this job. An empty dictionary
            is returned if the header cannot be retrieved.
        """
        self._get_params()
        if self._params:
            return self._params.get("header")
        return {}

    def circuits(self) -> List[QuantumCircuit]:
        """Return the circuits for this job.

        Returns:
            The circuits or for this job. An empty list
            is returned if the circuits cannot be retrieved (for example, if
            the job uses an old format that is no longer supported).
        """
        self._get_params()
        if self._params:
            circuits = self._params["circuits"]
            if isinstance(circuits, list):
                return circuits
            return [circuits]
        return []

    def _get_params(self) -> None:
        """Retrieve job parameters"""
        if not self._params:
            with api_to_job_error():
                if self._provider._runtime_client.job_type(self.job_id()) == "IQX":
                    raise IBMJobError(
                        f"{self.job_id()} is a legacy job. Retrieving parameters of legacy "
                        f"jobs is not supported from qiskit-ibm-provider"
                    ) from None
                api_response = self._runtime_client.job_get(self.job_id())
                self._params = api_response.get("params", {})

    def wait_for_final_state(  # pylint: disable=arguments-differ
        self,
        timeout: Optional[float] = None,
        wait: int = 3,
    ) -> None:
        """Use the websocket server to wait for the final the state of a job. The server
            will remain open if the job is still running and the connection will be terminated
            once the job completes. Then update and return the status of the job.

        Args:
            timeout: Seconds to wait for the job. If ``None``, wait indefinitely.

        Raises:
            IBMJobTimeoutError: If the job does not complete within given timeout.
        """
        try:
            start_time = time.time()
            if self._is_streaming():
                self._ws_client_future.result(timeout)
            # poll for status after stream has closed until status is final
            # because status doesn't become final as soon as stream closes
            status = self.status()
            while status not in JOB_FINAL_STATES:
                elapsed_time = time.time() - start_time
                if timeout is not None and elapsed_time >= timeout:
                    raise IBMJobTimeoutError(
                        f"Timed out waiting for job to complete after {timeout} secs."
                    )
                time.sleep(wait)
                status = self.status()
        except futures.TimeoutError:
            raise IBMJobTimeoutError(
                f"Timed out waiting for job to complete after {timeout} secs."
            )

    def _is_streaming(self) -> bool:
        """Return whether job results are being streamed.

        Returns:
            Whether job results are being streamed.
        """
        if self._ws_client_future is None:
            return False

        if self._ws_client_future.done():
            return False

        return True

    def _download_external_result(self, response: Any) -> Any:
        """Download result from external URL.

        Args:
            response: Response to check for url keyword, if available, download result from given URL
        """
        try:
            result_url_json = json.loads(response)
            if "url" in result_url_json:
                url = result_url_json["url"]
                result_response = requests.get(url, timeout=10)
                return result_response.text
            return response
        except json.JSONDecodeError:
            return response

    def _retrieve_result(self, refresh: bool = False) -> None:
        """Retrieve the job result response.

        Args:
            refresh: If ``True``, re-query the server for the result.
               Otherwise return the cached value.

        Raises:
            IBMJobApiError: If an unexpected error occurred when communicating
                with the server.
        """
        if self._api_status in (
            ApiJobStatus.ERROR_CREATING_JOB.value,
            ApiJobStatus.ERROR_VALIDATING_JOB.value,
            ApiJobStatus.ERROR_TRANSPILING_JOB.value,
        ):
            # No results if job was never executed.
            return

        if not self._result or refresh:  # type: ignore[has-type]
            try:
                if self._provider._runtime_client.job_type(self.job_id()) == "IQX":
                    api_result = self._api_client.job_result(self.job_id())
                else:
                    api_result = self._download_external_result(
                        self._runtime_client.job_results(self.job_id())
                    )

                self._set_result(api_result)
            except ApiError as err:
                if self._status not in (JobStatus.ERROR, JobStatus.CANCELLED):
                    raise IBMJobApiError(
                        "Unable to retrieve result for "
                        "job {}: {}".format(self.job_id(), str(err))
                    ) from err

    def _parse_result_for_errors(self, raw_data: str) -> str:
        """Checks whether the job result contains errors

        Args:
            raw_data: Raw result data.

        returns:
            The error message, if found

        """
        result = re.search("JobError: '(.*)'", raw_data)
        if result is not None:
            return result.group(1)
        else:
            index = raw_data.rfind("Traceback")
            if index != -1:
                return "Unknown error; " + raw_data[index:]
        return None

    def _set_result(self, raw_data: str) -> None:
        """Set the job result.

        Args:
            raw_data: Raw result data.

        Raises:
            IBMJobInvalidStateError: If result is in an unsupported format.
            IBMJobApiError: If an unexpected error occurred when communicating
                with the server.
        """
        if raw_data is None:
            self._result = None
            return
        # TODO: check whether client version can be extracted from runtime data
        # raw_data["client_version"] = self.client_version
        try:
            data_dict = decode_result(raw_data, RuntimeDecoder)
            self._result = Result.from_dict(data_dict)
        except (KeyError, TypeError) as err:
            if not self._kind:
                raise IBMJobInvalidStateError(
                    "Unable to retrieve result for job {}. Job result "
                    "is in an unsupported format.".format(self.job_id())
                ) from err
            raise IBMJobApiError(
                "Unable to retrieve result for "
                "job {}: {}".format(self.job_id(), str(err))
            ) from err

    def _check_for_error_message(self, result_response: Dict[str, Any]) -> None:
        """Retrieves the error message from the result response.

        Args:
            result_response: Dictionary of the result response.
        """
        if result_response.get("results", None):
            # If individual errors given
            self._job_error_msg = build_error_report(result_response["results"])
        elif "error" in result_response:
            self._job_error_msg = self._format_message_from_error(
                result_response["error"]
            )

    def _format_message_from_error(self, error: Dict) -> str:
        """Format message from the error field.

        Args:
            The error field.

        Returns:
            A formatted error message.

        Raises:
            IBMJobApiError: If invalid data received from the server.
        """
        try:
            return "{}. Error code: {}.".format(error["message"], error["code"])
        except KeyError as ex:
            raise IBMJobApiError(
                "Failed to get error message for job {}. Invalid error "
                "data received: {}".format(self.job_id(), error)
            ) from ex

    def _extract_client_version(self, data: str) -> Dict:
        """Extract client version from API.

        Args:
            data: API client version.

        Returns:
            Extracted client version.

        Additional info:
            The runtime client returns the version as a string, e.g.
            "0.1.0,0.21.2"
            Where the numbers represent versions of qiskit-ibm-provider and qiskit-terra
        """
        if data is not None:
            if "," not in data:  # sometimes only the metapackage version is returned
                return {"qiskit": data}
            client_components = ["qiskit-ibm-provider", "qiskit-terra"]
            return dict(zip(client_components, data.split(",")))
        return {}

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
