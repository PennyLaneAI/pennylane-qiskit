# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base runtime job class."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Callable, Dict, Type, Union, Sequence, List, Tuple
import json
import logging
from concurrent import futures
import traceback
import queue
from datetime import datetime
import requests

from qiskit.providers.backend import Backend
from qiskit.providers.models import BackendProperties
from qiskit.providers.jobstatus import JobStatus as RuntimeJobStatus

# pylint: disable=unused-import,cyclic-import

from qiskit_ibm_runtime import qiskit_runtime_service

from .utils import utc_to_local
from .utils.utils import validate_job_tags
from .utils.queueinfo import QueueInfo
from .constants import DEFAULT_DECODERS, API_TO_JOB_ERROR_MESSAGE
from .exceptions import (
    IBMError,
    IBMApiError,
    IBMRuntimeError,
)
from .utils.result_decoder import ResultDecoder
from .api.clients import RuntimeClient, RuntimeWebsocketClient, WebsocketClientCloseCode
from .api.exceptions import RequestsApiError
from .api.client_parameters import ClientParameters

logger = logging.getLogger(__name__)


class BaseRuntimeJob(ABC):
    """Base Runtime Job class."""

    _POISON_PILL = "_poison_pill"
    """Used to inform streaming to stop."""

    _executor = futures.ThreadPoolExecutor(thread_name_prefix="runtime_job")

    JOB_FINAL_STATES: Tuple[Any, ...] = ()
    ERROR: Union[str, RuntimeJobStatus] = None

    def __init__(
        self,
        backend: Backend,
        api_client: RuntimeClient,
        client_params: ClientParameters,
        job_id: str,
        program_id: str,
        service: "qiskit_runtime_service.QiskitRuntimeService",
        params: Optional[Dict] = None,
        creation_date: Optional[str] = None,
        user_callback: Optional[Callable] = None,
        result_decoder: Optional[Union[Type[ResultDecoder], Sequence[Type[ResultDecoder]]]] = None,
        image: Optional[str] = "",
        session_id: Optional[str] = None,
        tags: Optional[List] = None,
        version: Optional[int] = None,
    ) -> None:
        """RuntimeJob constructor.

        Args:
            backend: The backend instance used to run this job.
            api_client: Object for connecting to the server.
            client_params: Parameters used for server connection.
            job_id: Job ID.
            program_id: ID of the program this job is for.
            params: Job parameters.
            creation_date: Job creation date, in UTC.
            user_callback: User callback function.
            result_decoder: A :class:`ResultDecoder` subclass used to decode job results.
            image: Runtime image used for this job: image_name:tag.
            service: Runtime service.
            session_id: Job ID of the first job in a runtime session.
            tags: Tags assigned to the job.
            version: Primitive version.
        """
        self._backend = backend
        self._job_id = job_id
        self._api_client = api_client
        self._interim_results: Optional[Any] = None
        self._params = params or {}
        self._creation_date = creation_date
        self._program_id = program_id
        self._reason: Optional[str] = None
        self._error_message: Optional[str] = None
        self._image = image
        self._final_interim_results = False
        self._service = service
        self._session_id = session_id
        self._tags = tags
        self._usage_estimation: Dict[str, Any] = {}
        self._version = version
        self._queue_info: QueueInfo = None
        self._user_callback = user_callback
        self._status: Union[RuntimeJobStatus, str] = None

        decoder = result_decoder or DEFAULT_DECODERS.get(program_id, None) or ResultDecoder
        if isinstance(decoder, Sequence):
            self._interim_result_decoder, self._final_result_decoder = decoder
        else:
            self._interim_result_decoder = self._final_result_decoder = decoder

        # Used for streaming
        self._ws_client_future = None  # type: Optional[futures.Future]
        self._result_queue = queue.Queue()  # type: queue.Queue
        self._ws_client = RuntimeWebsocketClient(
            websocket_url=client_params.get_runtime_api_base_url().replace("https", "wss"),
            client_params=client_params,
            job_id=job_id,
            message_queue=self._result_queue,
        )

    def job_id(self) -> str:
        """Return a unique id identifying the job."""
        return self._job_id

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

    def cancel_result_streaming(self) -> None:
        """Cancel result streaming."""
        if not self._is_streaming():
            return
        self._ws_client.disconnect(WebsocketClientCloseCode.CANCEL)

    def metrics(self) -> Dict[str, Any]:
        """Return job metrics.

        Returns:
            Job metrics, which includes timestamp information.

        Raises:
            IBMRuntimeError: If a network error occurred.
        """
        try:
            return self._api_client.job_metadata(self.job_id())
        except RequestsApiError as err:
            raise IBMRuntimeError(f"Failed to get job metadata: {err}") from None

    def update_tags(self, new_tags: List[str]) -> List[str]:
        """Update the tags associated with this job.

        Args:
            new_tags: New tags to assign to the job.

        Returns:
            The new tags associated with this job.

        Raises:
            IBMApiError: If an unexpected error occurred when communicating
                with the server or updating the job tags.
        """
        tags_to_update = set(new_tags)
        validate_job_tags(new_tags)

        response = self._api_client.update_tags(job_id=self.job_id(), tags=list(tags_to_update))

        if response.status_code == 204:
            api_response = self._api_client.job_get(self.job_id())
            self._tags = api_response.pop("tags", [])
            return self._tags
        else:
            raise IBMApiError(
                "An unexpected error occurred when updating the "
                "tags for job {}. The tags were not updated for "
                "the job.".format(self.job_id())
            )

    def properties(self, refresh: bool = False) -> Optional[BackendProperties]:
        """Return the backend properties for this job.

        Args:
            refresh: If ``True``, re-query the server for the backend properties.
                Otherwise, return a cached version.

        Returns:
            The backend properties used for this job, at the time the job was run,
            or ``None`` if properties are not available.
        """

        return self._backend.properties(refresh, self.creation_date)

    def error_message(self) -> Optional[str]:
        """Returns the reason if the job failed.

        Returns:
            Error message string or ``None``.
        """
        self._set_status_and_error_message()
        return self._error_message

    def _set_status_and_error_message(self) -> None:
        """Fetch and set status and error message."""
        if self._status not in self.JOB_FINAL_STATES:
            response = self._api_client.job_get(job_id=self.job_id())
            self._set_status(response)
            self._set_error_message(response)

    def _set_status(self, job_response: Dict) -> None:
        """Set status.

        Args:
            job_response: Job response from runtime API.

        Raises:
            IBMError: If an unknown status is returned from the server.
        """
        try:
            reason = job_response["state"].get("reason")
            reason_code = job_response["state"].get("reason_code")
            if reason:
                # TODO remove this in https://github.com/Qiskit/qiskit-ibm-runtime/issues/989
                if reason.upper() == "RAN TOO LONG":
                    self._reason = reason.upper()
                else:
                    self._reason = reason
                if reason_code:
                    self._reason = f"Error code {reason_code}; {self._reason}"
            self._status = self._status_from_job_response(job_response)
        except KeyError:
            raise IBMError(f"Unknown status: {job_response['state']['status']}")

    def _set_error_message(self, job_response: Dict) -> None:
        """Set error message if the job failed.

        Args:
            job_response: Job response from runtime API.
        """
        if self._status == self.ERROR:
            self._error_message = self._error_msg_from_job_response(job_response)
        else:
            self._error_message = None

    @abstractmethod
    def _status_from_job_response(self, response: Dict) -> str:
        """Returns the job status from an API response."""
        return response["state"]["status"].upper()

    def _error_msg_from_job_response(self, response: Dict) -> str:
        """Returns the error message from an API response.

        Args:
            response: Job response from the runtime API.

        Returns:
            Error message.
        """
        status = response["state"]["status"].upper()

        job_result_raw = self._download_external_result(
            self._api_client.job_results(job_id=self.job_id())
        )
        index = job_result_raw.rfind("Traceback")
        if index != -1:
            job_result_raw = job_result_raw[index:]

        if status == "CANCELLED" and self._reason == "RAN TOO LONG":
            error_msg = API_TO_JOB_ERROR_MESSAGE["CANCELLED - RAN TOO LONG"]
            return error_msg.format(self.job_id(), job_result_raw)
        else:
            error_msg = API_TO_JOB_ERROR_MESSAGE["FAILED"]
            return error_msg.format(self.job_id(), self._reason or job_result_raw)

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

    def _start_websocket_client(self) -> None:
        """Start websocket client to stream results."""
        try:
            logger.debug("Start websocket client for job %s", self.job_id())
            self._ws_client.job_results()
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "An error occurred while streaming results from the server for job %s:\n%s",
                self.job_id(),
                traceback.format_exc(),
            )
        finally:
            self._result_queue.put_nowait(self._POISON_PILL)

    def _stream_results(
        self,
        result_queue: queue.Queue,
        user_callback: Callable,
        decoder: Optional[Type[ResultDecoder]] = None,
    ) -> None:
        """Stream results.

        Args:
            result_queue: Queue used to pass websocket messages.
            user_callback: User callback function.
            decoder: A :class:`ResultDecoder` (sub)class used to decode job results.
        """
        logger.debug("Start result streaming for job %s", self.job_id())
        _decoder = decoder or self._interim_result_decoder
        while True:
            try:
                response = result_queue.get()
                if response == self._POISON_PILL:
                    self._empty_result_queue(result_queue)
                    return

                response = self._download_external_result(response)

                user_callback(self.job_id(), _decoder.decode(response))
            except Exception:  # pylint: disable=broad-except
                logger.warning(
                    "An error occurred while streaming results for job %s:\n%s",
                    self.job_id(),
                    traceback.format_exc(),
                )

    @staticmethod
    def _empty_result_queue(result_queue: queue.Queue) -> None:
        """Empty the result queue.

        Args:
            result_queue: Result queue to empty.
        """
        try:
            while True:
                result_queue.get_nowait()
        except queue.Empty:
            pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}('{self._job_id}', '{self._program_id}')>"

    @property
    def image(self) -> str:
        """Return the runtime image used for the job.

        Returns:
            Runtime image: image_name:tag or "" if the default
            image is used.
        """
        return self._image

    @property
    def inputs(self) -> Dict:
        """Job input parameters.

        Returns:
            Input parameters used in this job.
        """
        if not self._params:
            response = self._api_client.job_get(job_id=self.job_id(), exclude_params=False)
            self._params = response.get("params", {})
        return self._params

    @property
    def program_id(self) -> str:
        """Program ID.

        Returns:
            ID of the program this job is for.
        """
        return self._program_id

    @property
    def creation_date(self) -> Optional[datetime]:
        """Job creation date in local time.

        Returns:
            The job creation date as a datetime object, in local time, or
            ``None`` if creation date is not available.
        """
        if not self._creation_date:
            response = self._api_client.job_get(job_id=self.job_id())
            self._creation_date = response.get("created", None)

        if not self._creation_date:
            return None
        creation_date_local_dt = utc_to_local(self._creation_date)
        return creation_date_local_dt

    @property
    def session_id(self) -> str:
        """Session ID.

        Returns:
            Session ID. None if the backend is a simulator.
        """
        if not self._session_id:
            response = self._api_client.job_get(job_id=self.job_id())
            self._session_id = response.get("session_id", None)
        return self._session_id

    @property
    def tags(self) -> List:
        """Job tags.

        Returns:
            Tags assigned to the job that can be used for filtering.
        """
        return self._tags

    @property
    def usage_estimation(self) -> Dict[str, Any]:
        """Return the usage estimation infromation for this job.

        Returns:
            ``quantum_seconds`` which is the estimated system execution time
            of the job in seconds. Quantum time represents the time that
            the system is dedicated to processing your job.
        """
        if not self._usage_estimation:
            response = self._api_client.job_get(job_id=self.job_id())
            self._usage_estimation = {
                "quantum_seconds": response.pop("estimated_running_time_seconds", None),
            }

        return self._usage_estimation

    @abstractmethod
    def in_final_state(self) -> bool:
        """Return whether the job is in a final job state such as ``DONE`` or ``ERROR``."""
        pass

    @abstractmethod
    def errored(self) -> bool:
        """Return whether the job has failed."""
        pass
