# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Client for accessing an individual IBM Quantum account."""

import logging
import json
from typing import List, Dict, Any, Optional

from .base import BaseClient
from ..client_parameters import ClientParameters
from ..rest import Api
from ..session import RetrySession

logger = logging.getLogger(__name__)


class AccountClient(BaseClient):
    """Client for accessing an individual IBM Quantum account."""

    def __init__(self, params: ClientParameters) -> None:
        """AccountClient constructor.

        Args:
            params: Parameters used for server connection.
        """
        self._session = RetrySession(
            params.url, auth=params.get_auth_handler(), **params.connection_parameters()
        )
        self._params = params
        self.base_api = Api(self._session)

    # Old iqx api
    def job_get(self, job_id: str) -> Dict[str, Any]:
        """Return information about the job.
        Args:
            job_id: The ID of the job.
        Returns:
            Job information.
        """
        return self.base_api.job(job_id).get()

    def list_jobs(
        self,
        limit: int = 10,
        skip: int = 0,
        descending: bool = True,
        extra_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return a list of job data, with filtering and pagination.
        In order to reduce the amount of data transferred, the server only
        sends back a subset of the total information for each job.
        Args:
            limit: Maximum number of items to return.
            skip: Offset for the items to return.
            descending: Whether the jobs should be in descending order.
            extra_filter: Additional filtering passed to the query.
        Returns:
            A list of job data.
        """
        return self.base_api.jobs(
            limit=limit, skip=skip, descending=descending, extra_filter=extra_filter
        )

    def job_result(self, job_id: str) -> str:
        """Retrieve and return the job result.
        Args:
            job_id: The ID of the job.
        Returns:
            Job result.
        """

        job_api = self.base_api.job(job_id)

        # Get the download URL.
        download_url = job_api.result_url()["url"]

        # Download the result from object storage.
        result_response = job_api.get_object_storage(download_url)

        return json.dumps(result_response)
