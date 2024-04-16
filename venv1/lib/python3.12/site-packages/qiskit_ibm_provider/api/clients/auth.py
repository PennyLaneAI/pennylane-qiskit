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

"""Client for accessing IBM Quantum authentication services."""

from typing import Dict, List, Optional, Any, Union
from requests.exceptions import RequestException

from ..auth import QuantumAuth
from ..exceptions import AuthenticationLicenseError, RequestsApiError
from ..rest import Api
from ..session import RetrySession
from ..client_parameters import ClientParameters

from .base import BaseClient


class AuthClient(BaseClient):
    """Client for accessing IBM Quantum authentication services."""

    def __init__(self, client_params: ClientParameters) -> None:
        """AuthClient constructor.

        Args:
            client_params: Parameters used for server connection.
        """
        self.api_token = client_params.token
        self.auth_url = client_params.url
        self._service_urls = {}  # type: ignore[var-annotated]

        self.auth_api = Api(
            RetrySession(self.auth_url, **client_params.connection_parameters())
        )
        self.base_api = self._init_service_clients(
            **client_params.connection_parameters()
        )

    def _init_service_clients(self, **request_kwargs: Any) -> Api:
        """Initialize the clients used for communicating with the API.

        Args:
            **request_kwargs: Arguments for the request ``Session``.

        Returns:
            Client for the API server.
        """
        # Request an access token.
        self.access_token = self._request_access_token()
        self.auth_api.session.auth = QuantumAuth(access_token=self.access_token)
        self._service_urls = self.user_urls()

        # Create the api server client, using the access token.
        base_api = Api(
            RetrySession(
                self._service_urls["http"],
                auth=QuantumAuth(access_token=self.access_token),
                **request_kwargs,
            )
        )

        return base_api

    def _request_access_token(self) -> str:
        """Request a new access token from the API authentication service.

        Returns:
            A new access token.

        Raises:
            AuthenticationLicenseError: If the user hasn't accepted the license agreement.
            RequestsApiError: If the request failed.
        """
        try:
            response = self.auth_api.login(self.api_token)
            return response["id"]
        except RequestsApiError as ex:
            # Get the original exception that raised.
            original_exception = ex.__cause__

            if isinstance(original_exception, RequestException):
                # Get the response from the original request exception.
                error_response = (
                    # pylint: disable=no-member
                    original_exception.response
                )
                if error_response is not None and error_response.status_code == 401:
                    try:
                        error_code = error_response.json()["error"]["name"]
                        if error_code == "ACCEPT_LICENSE_REQUIRED":
                            message = error_response.json()["error"]["message"]
                            raise AuthenticationLicenseError(message)
                    except (ValueError, KeyError):
                        # the response did not contain the expected json.
                        pass
            raise

    # User account-related public functions.

    def user_urls(self) -> Dict[str, str]:
        """Retrieve the API URLs from the authentication service.

        Returns:
            A dict with the base URLs for the services. Currently
            supported keys are:

                * ``http``: The API URL for HTTP communication.
                * ``ws``: The API URL for websocket communication.
                * ``services`: The API URL for additional services.
        """
        response = self.auth_api.user_info()
        return response["urls"]

    def user_hubs(self) -> List[Dict[str, str]]:
        """Retrieve the hub/group/project sets available to the user.

        The first entry in the list will be the default set, as indicated by
        the ``isDefault`` field from the API.

        Returns:
            A list of dictionaries with the hub, group, and project values keyed by
            ``hub``, ``group``, and ``project``, respectively.
        """
        response = self.base_api.hubs()

        hubs = []  # type: ignore[var-annotated]
        for hub in response:
            hub_name = hub["name"]
            for group_name, group in hub["groups"].items():
                for project_name, project in group["projects"].items():
                    entry = {
                        "hub": hub_name,
                        "group": group_name,
                        "project": project_name,
                    }

                    # Move to the top if it is the default h/g/p.
                    if project.get("isDefault"):
                        hubs.insert(0, entry)
                    else:
                        hubs.append(entry)

        return hubs

    # Miscellaneous public functions.

    def api_version(self) -> Dict[str, Union[str, bool]]:
        """Return the version of the API.

        Returns:
            API version.
        """
        return self.base_api.version()

    def current_access_token(self) -> Optional[str]:
        """Return the current access token.

        Returns:
            The access token in use.
        """
        return self.access_token

    def current_service_urls(self) -> Dict:
        """Return the current service URLs.

        Returns:
            A dict with the base URLs for the services, in the same
            format as :meth:`user_urls()`.
        """
        return self._service_urls
