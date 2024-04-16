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

"""IBM Cloud Backend REST adapter."""

from typing import Dict, Any, Optional
from datetime import datetime as python_datetime

from qiskit_ibm_runtime.api.rest.base import RestAdapterBase
from ..session import RetrySession


class CloudBackend(RestAdapterBase):
    """Rest adapter for IBM Cloud backend related endpoints."""

    URL_MAP = {
        "configuration": "/configuration",
        "properties": "/properties",
        "pulse_defaults": "/defaults",
        "status": "/status",
    }

    def __init__(self, session: RetrySession, backend_name: str, url_prefix: str = "") -> None:
        """Backend constructor.

        Args:
            session: Session to be used in the adaptor.
            backend_name: Name of the backend.
            url_prefix: Base URL.
        """
        self.backend_name = backend_name
        super().__init__(session, "{}/backends/{}".format(url_prefix, backend_name))

    def configuration(self) -> Dict[str, Any]:
        """Return backend configuration.

        Returns:
            JSON response of backend configuration.
        """
        url = self.get_url("configuration")
        return self.session.get(url).json()

    def properties(self, datetime: Optional[python_datetime] = None) -> Dict[str, Any]:
        """Return backend properties.

        Returns:
            JSON response of backend properties.
        """
        url = self.get_url("properties")

        params = {}
        if datetime:
            params["updated_before"] = datetime.isoformat()

        response = self.session.get(url, params=params).json()
        # Adjust name of the backend.
        if response:
            response["backend_name"] = self.backend_name
        return response

    def pulse_defaults(self) -> Dict[str, Any]:
        """Return backend pulse defaults.

        Returns:
            JSON response of pulse defaults.
        """
        url = self.get_url("pulse_defaults")
        return self.session.get(url).json()

    def status(self) -> Dict[str, Any]:
        """Return backend status.

        Returns:
            JSON response of backend status.
        """
        url = self.get_url("status")
        response = self.session.get(url).json()
        # Adjust fields according to the specs (BackendStatus).
        ret = {
            "backend_name": self.backend_name,
            "backend_version": response.get("backend_version", "0.0.0"),
            "status_msg": response.get("status", ""),
            "operational": bool(response.get("state", False)),
        }
        # 'pending_jobs' is required, and should be >= 0.
        if "length_queue" in response:
            ret["pending_jobs"] = max(response["length_queue"], 0)
        else:
            ret["pending_jobs"] = 0
        return ret
