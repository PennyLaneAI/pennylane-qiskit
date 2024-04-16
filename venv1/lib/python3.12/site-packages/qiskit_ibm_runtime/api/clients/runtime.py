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

"""Client for accessing IBM Quantum runtime service."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime as python_datetime
from requests import Response

from qiskit_ibm_runtime.api.session import RetrySession

from .backend import BaseBackendClient
from ..rest.runtime import Runtime
from ..client_parameters import ClientParameters
from ...utils.hgp import from_instance_format

logger = logging.getLogger(__name__)


class RuntimeClient(BaseBackendClient):
    """Client for accessing runtime service."""

    def __init__(
        self,
        params: ClientParameters,
    ) -> None:
        """RuntimeClient constructor.

        Args:
            params: Connection parameters.
        """
        self._session = RetrySession(
            base_url=params.get_runtime_api_base_url(),
            auth=params.get_auth_handler(),
            **params.connection_parameters(),
        )
        self._api = Runtime(self._session)

    def program_run(
        self,
        program_id: str,
        backend_name: Optional[str],
        params: Dict,
        image: Optional[str],
        hgp: Optional[str],
        log_level: Optional[str],
        session_id: Optional[str],
        job_tags: Optional[List[str]] = None,
        max_execution_time: Optional[int] = None,
        start_session: Optional[bool] = False,
        session_time: Optional[int] = None,
        channel_strategy: Optional[str] = None,
    ) -> Dict:
        """Run the specified program.

        Args:
            program_id: Program ID.
            backend_name: Name of the backend to run the program.
            params: Parameters to use.
            image: The runtime image to use.
            hgp: Hub/group/project to use.
            log_level: Log level to use.
            session_id: Job ID of the first job in a runtime session.
            job_tags: Tags to be assigned to the job.
            max_execution_time: Maximum execution time in seconds.
            start_session: Set to True to explicitly start a runtime session. Defaults to False.
            session_time: Length of session in seconds.
            channel_strategy: Error mitigation strategy.

        Returns:
            JSON response.
        """
        hgp_dict = {}
        if hgp:
            hub, group, project = from_instance_format(hgp)
            hgp_dict = {"hub": hub, "group": group, "project": project}
        return self._api.program_run(
            program_id=program_id,
            backend_name=backend_name,
            params=params,
            image=image,
            log_level=log_level,
            session_id=session_id,
            job_tags=job_tags,
            max_execution_time=max_execution_time,
            start_session=start_session,
            session_time=session_time,
            channel_strategy=channel_strategy,
            **hgp_dict,
        )

    def job_get(self, job_id: str, exclude_params: bool = True) -> Dict:
        """Get job data.

        Args:
            job_id: Job ID.

        Returns:
            JSON response.
        """
        response = self._api.program_job(job_id).get(exclude_params=exclude_params)
        logger.debug("Runtime job get response: %s", response)
        return response

    def jobs_get(
        self,
        limit: int = None,
        skip: int = None,
        backend_name: str = None,
        pending: bool = None,
        program_id: str = None,
        hub: str = None,
        group: str = None,
        project: str = None,
        job_tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        created_after: Optional[python_datetime] = None,
        created_before: Optional[python_datetime] = None,
        descending: bool = True,
    ) -> Dict:
        """Get job data for all jobs.

        Args:
            limit: Number of results to return.
            skip: Number of results to skip.
            backend_name: Name of the backend to retrieve jobs from.
            pending: Returns 'QUEUED' and 'RUNNING' jobs if True,
                returns 'DONE', 'CANCELLED' and 'ERROR' jobs if False.
            program_id: Filter by Program ID.
            hub: Filter by hub - hub, group, and project must all be specified.
            group: Filter by group - hub, group, and project must all be specified.
            project: Filter by project - hub, group, and project must all be specified.
            job_tags: Filter by tags assigned to jobs. Matched jobs are associated with all tags.
            session_id: Job ID of the first job in a runtime session.
            created_after: Filter by the given start date, in local time. This is used to
                find jobs whose creation dates are after (greater than or equal to) this
                local date/time.
            created_before: Filter by the given end date, in local time. This is used to
                find jobs whose creation dates are before (less than or equal to) this
                local date/time.
            descending: If ``True``, return the jobs in descending order of the job
                creation date (i.e. newest first) until the limit is reached.

        Returns:
            JSON response.
        """
        return self._api.jobs_get(
            limit=limit,
            skip=skip,
            backend_name=backend_name,
            pending=pending,
            program_id=program_id,
            hub=hub,
            group=group,
            project=project,
            job_tags=job_tags,
            session_id=session_id,
            created_after=created_after,
            created_before=created_before,
            descending=descending,
        )

    def job_results(self, job_id: str) -> str:
        """Get the results of a program job.

        Args:
            job_id: Program job ID.

        Returns:
            Job result.
        """
        return self._api.program_job(job_id).results()

    def job_interim_results(self, job_id: str) -> str:
        """Get the interim results of a program job.

        Args:
            job_id: Program job ID.

        Returns:
            Job interim results.
        """
        return self._api.program_job(job_id).interim_results()

    def job_cancel(self, job_id: str) -> None:
        """Cancel a job.

        Args:
            job_id: Runtime job ID.
        """
        self._api.program_job(job_id).cancel()

    def job_delete(self, job_id: str) -> None:
        """Delete a job.

        Args:
            job_id: Runtime job ID.
        """
        self._api.program_job(job_id).delete()

    def job_logs(self, job_id: str) -> str:
        """Get the job logs.

        Args:
            job_id: Program job ID.

        Returns:
            Job logs.
        """
        return self._api.program_job(job_id).logs()

    def job_metadata(self, job_id: str) -> Dict[str, Any]:
        """Get job metadata.

        Args:
            job_id: Program job ID.

        Returns:
            Job metadata.
        """
        return self._api.program_job(job_id).metadata()

    def create_session(
        self,
        backend: Optional[str] = None,
        instance: Optional[str] = None,
        max_time: Optional[int] = None,
        channel: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a session.

        Args:
            mode: Execution mode.
        """
        return self._api.runtime_session(session_id=None).create(
            backend, instance, max_time, channel, mode
        )

    def cancel_session(self, session_id: str) -> None:
        """Close all jobs in the runtime session.

        Args:
            session_id: Session ID.
        """
        self._api.runtime_session(session_id=session_id).cancel()

    def close_session(self, session_id: str) -> None:
        """Update session so jobs can no longer be submitted."""
        self._api.runtime_session(session_id=session_id).close()

    def session_details(self, session_id: str) -> Dict[str, Any]:
        """Get session details.

        Args:
            session_id: Session ID.

        Returns:
            Session details.
        """
        return self._api.runtime_session(session_id=session_id).details()

    def list_backends(
        self, hgp: Optional[str] = None, channel_strategy: Optional[str] = None
    ) -> List[str]:
        """Return IBM backends available for this service instance.

        Args:
            hgp: Filter by hub/group/project.
            channel_strategy: Filter by channel strategy.

        Returns:
            IBM backends available for this service instance.
        """
        return self._api.backends(hgp=hgp, channel_strategy=channel_strategy)["devices"]

    def is_qctrl_enabled(self) -> bool:
        """Returns a boolean of whether or not the instance has q-ctrl enabled.

        Returns:
            Boolean value.
        """
        return self._api.is_qctrl_enabled()

    def backend_configuration(self, backend_name: str) -> Dict[str, Any]:
        """Return the configuration of the IBM backend.

        Args:
            backend_name: The name of the IBM backend.

        Returns:
            Backend configuration.
        """
        return self._api.backend(backend_name).configuration()

    def backend_status(self, backend_name: str) -> Dict[str, Any]:
        """Return the status of the IBM backend.

        Args:
            backend_name: The name of the IBM backend.

        Returns:
            Backend status.
        """
        return self._api.backend(backend_name).status()

    def backend_properties(
        self, backend_name: str, datetime: Optional[python_datetime] = None
    ) -> Dict[str, Any]:
        """Return the properties of the IBM backend.

        Args:
            backend_name: The name of the IBM backend.
            datetime: Date and time for additional filtering of backend properties.

        Returns:
            Backend properties.

        Raises:
            NotImplementedError: If `datetime` is specified.
        """
        return self._api.backend(backend_name).properties(datetime=datetime)

    def backend_pulse_defaults(self, backend_name: str) -> Dict:
        """Return the pulse defaults of the IBM backend.

        Args:
            backend_name: The name of the IBM backend.

        Returns:
            Backend pulse defaults.
        """
        return self._api.backend(backend_name).pulse_defaults()

    def update_tags(self, job_id: str, tags: list) -> Response:
        """Update the tags of the job.

        Args:
            job_id: The ID of the job.
            tags: The new tags to be assigned to the job.

        Returns:
            API Response.
        """
        return self._api.program_job(job_id).update_tags(tags)
