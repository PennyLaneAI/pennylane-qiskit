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
from typing import Any, Dict, List, Optional, Union
from datetime import datetime as python_datetime
from requests import Response

from .base import BaseClient
from ..rest.runtime import Runtime
from ..client_parameters import ClientParameters
from ..session import RetrySession
from ...utils.hgp import from_instance_format

logger = logging.getLogger(__name__)


class RuntimeClient(BaseClient):
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
            params.url, auth=params.get_auth_handler(), **params.connection_parameters()
        )
        self._api = Runtime(self._session)

    def list_backends(self, hgp: str) -> List[str]:
        """Return a list of backends.

        Args:
            hgp: hub, group, and project.

        Returns:
            The list of backends from the given hgp.
        """
        return self._api.backend().backends(hgp=hgp)

    def backend_properties(
        self, backend: str, datetime: Optional[python_datetime] = None
    ) -> Dict[str, Any]:
        """Return the properties of the backend.

        Args:
            backend: The name of the backend.

        Returns:
            Backend properties.
        """
        return self._api.backend(backend).properties(datetime=datetime)

    def backend_pulse_defaults(self, backend: str) -> Dict:
        """Return the pulse defaults of the backend.

        Args:
            backend: The name of the backend.

        Returns:
            Backend pulse defaults.
        """
        return self._api.backend(backend).pulse_defaults()

    def backend_status(self, backend: str) -> Dict[str, Any]:
        """Return the status of the backend.

        Args:
            backend: The name of the backend.

        Returns:
            Backend status.
        """
        return self._api.backend(backend).status()

    def backend_configuration(self, backend: str) -> Dict[str, Any]:
        """Return the configuration of the backend.

        Args:
            backend: The name of the backend.

        Returns:
            Backend configuration.
        """
        return self._api.backend(backend).configuration()

    def program_run(
        self,
        program_id: str,
        backend_name: Optional[str],
        params: Dict,
        image: Optional[str] = None,
        hgp: Optional[str] = None,
        log_level: Optional[str] = None,
        session_id: Optional[str] = None,
        job_tags: Optional[List[str]] = None,
        max_execution_time: Optional[int] = None,
        session_time: Optional[Union[int, str]] = None,
        start_session: Optional[bool] = False,
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
            session_time: max_time: (EXPERIMENTAL setting, can break between releases without warning)
                Maximum amount of time, a runtime session can be open before being
                forcibly closed. Can be specified as seconds (int) or a string like "2h 30m 40s".

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
            **hgp_dict
        )

    def job_get(self, job_id: str, exclude_params: bool = None) -> Dict:
        """Get job data.

        Args:
            job_id: Job ID.

        Returns:
            JSON response.
        """
        response = self._api.program_job(job_id).get(exclude_params=exclude_params)
        logger.debug("Runtime job get response: %s", response)
        return response

    def job_type(self, job_id: str) -> str:
        """Get job type.

        Args:
            job_id: Job ID.

        Returns:
            Job type, either "IQX" or "RUNTIME".
        """
        return self._api.program_job(job_id).job_type()

    def jobs_get(
        self,
        limit: int = None,
        skip: int = None,
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
        backend: str = None,
    ) -> Dict:
        """Get job data for all jobs.

        Args:
            limit: Number of results to return.
            skip: Number of results to skip.
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
            backend=backend,
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

    def job_status(self, job_id: str) -> Dict[str, Any]:
        """Return the status of the job.

        Args:
            job_id: The ID of the job.

        Returns:
            Job status.
        """
        return self.job_get(job_id)["state"]

    def update_tags(self, job_id: str, tags: list) -> Response:
        """Update the tags of the job.

        Args:
            job_id: The ID of the job.

        Returns:
            API Response.
        """
        return self._api.program_job(job_id).update_tags(tags)

    def create_session(
        self,
        backend: Optional[str] = None,
        instance: Optional[str] = None,
        max_time: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new runtime session.

        Args:
            backend: The name of the backend to use.
            instance: The instance to use.
            mode: The mode to use.

        Returns:
            The created session.
        """
        return self._api.runtime_session().create(backend, instance, max_time, mode)

    def close_session(self, session_id: str) -> None:
        """Close session

        Args:
            session_id (str): the id of the session to close
        """
        self._api.runtime_session(session_id=session_id).close()

    def cancel_session(self, session_id: str) -> None:
        """Cancel session

        Args:
            session_id (str): the id of the session to cancel
        """
        self._api.runtime_session(session_id=session_id).cancel()
