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

"""Program Job REST adapter."""

import json
from typing import Dict
from requests import Response

from .base import RestAdapterBase
from ..session import RetrySession
from ...utils.json import RuntimeDecoder


class ProgramJob(RestAdapterBase):
    """Rest adapter for program job related endpoints."""

    URL_MAP = {
        "self": "",
        "results": "/results",
        "cancel": "/cancel",
        "logs": "/logs",
        "interim_results": "/interim_results",
        "metrics": "/metrics",
        "tags": "/tags",
    }

    def __init__(self, session: RetrySession, job_id: str, url_prefix: str = "") -> None:
        """ProgramJob constructor.

        Args:
            session: Session to be used in the adapter.
            job_id: ID of the program job.
            url_prefix: Prefix to use in the URL.
        """
        super().__init__(session, "{}/jobs/{}".format(url_prefix, job_id))

    def get(self, exclude_params: bool = None) -> Dict:
        """Return program job information.

        Args:
            exclude_params: If ``True``, the params will not be included in the response.

        Returns:
            JSON response.
        """
        payload = {}
        if exclude_params:
            payload["exclude_params"] = "true"
        return self.session.get(self.get_url("self"), params=payload).json(cls=RuntimeDecoder)

    def delete(self) -> None:
        """Delete program job."""
        self.session.delete(self.get_url("self"))

    def interim_results(self) -> str:
        """Return program job interim results.

        Returns:
            Interim results.
        """
        response = self.session.get(self.get_url("interim_results"))
        return response.text

    def results(self) -> str:
        """Return program job results.

        Returns:
            Job results.
        """
        response = self.session.get(self.get_url("results"))
        return response.text

    def cancel(self) -> None:
        """Cancel the job."""
        self.session.post(self.get_url("cancel"))

    def logs(self) -> str:
        """Retrieve job logs.

        Returns:
            Job logs.
        """
        return self.session.get(self.get_url("logs")).text

    def metadata(self) -> Dict:
        """Retrieve job metadata.

        Returns:
            Job Metadata.
        """
        return self.session.get(self.get_url("metrics")).json()

    def update_tags(self, tags: list) -> Response:
        """Update job tags.

        Returns:
            API Response.
        """
        return self.session.put(self.get_url("tags"), data=json.dumps({"tags": tags}))
