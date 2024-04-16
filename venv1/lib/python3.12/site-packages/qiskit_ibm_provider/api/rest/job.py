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

"""Job REST adapter."""

import logging
from json.decoder import JSONDecodeError

from typing import Dict, Any

from .base import RestAdapterBase
from ..session import RetrySession
from ..exceptions import ApiIBMProtocolError
from .utils.data_mapper import map_job_response, map_job_status_response

logger = logging.getLogger(__name__)


class Job(RestAdapterBase):
    """Rest adapter for job related endpoints."""

    URL_MAP = {
        "cancel": "/cancel",
        "self": "/v/1",
        "status": "/status",
        "properties": "/properties",
        "delete": "",
        "result_url": "/resultDownloadUrl",
    }

    def __init__(
        self, session: RetrySession, job_id: str, url_prefix: str = ""
    ) -> None:
        """Job constructor.

        Args:
            session: Session to be used in the adapter.
            job_id: ID of the job.
            url_prefix: Prefix to use in the URL.
        """
        self.job_id = job_id
        super().__init__(session, "{}/Jobs/{}".format(url_prefix, job_id))

    def get(self) -> Dict[str, Any]:
        """Return job information.

        Returns:
            JSON response of job information.
        """
        url = self.get_url("self")

        response = self.session.get(url).json()

        if "calibration" in response:
            response["_properties"] = response.pop("calibration")
        response = map_job_response(response)

        return response

    def cancel(self) -> Dict[str, Any]:
        """Cancel a job.

        Returns:
            JSON response.
        """
        url = self.get_url("cancel")
        return self.session.post(url).json()

    def properties(self) -> Dict[str, Any]:
        """Return the backend properties of a job.

        Returns:
            JSON response.
        """
        url = self.get_url("properties")
        return self.session.get(url).json()

    def status(self) -> Dict[str, Any]:
        """Return the status of a job.

        Returns:
            JSON response of job status.

        Raises:
            ApiIBMProtocolError: If an unexpected result is received from the server.
        """
        url = self.get_url("status")
        raw_response = self.session.get(url)
        try:
            api_response = raw_response.json()
        except JSONDecodeError as err:
            raise ApiIBMProtocolError(
                "Unrecognized return value received from the server: {!r}. This could be caused"
                " by too many requests.".format(raw_response.content)
            ) from err
        return map_job_status_response(api_response)

    def delete(self) -> None:
        """Mark job for deletion."""
        url = self.get_url("delete")
        self.session.delete(url)

    def result_url(self) -> Dict[str, Any]:
        """Return an object storage URL for downloading results.

        Returns:
            JSON response.
        """
        url = self.get_url("result_url")
        return self.session.get(url).json()

    def get_object_storage(self, url: str) -> Dict[str, Any]:
        """Get via object_storage.
        Args:
            url: Object storage URL.
        Returns:
            JSON response.
        """
        logger.debug("Downloading from object storage.")
        response = self.session.get(url, bare=True, timeout=600).json()
        return response
