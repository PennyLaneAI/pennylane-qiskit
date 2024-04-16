# coding: utf-8

# Copyright 2023 IBM All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional

from requests import Request

from .authenticator import Authenticator
from ..token_managers.mcsp_token_manager import MCSPTokenManager


class MCSPAuthenticator(Authenticator):
    """The MCSPAuthenticator uses an apikey to obtain an access token from the MCSP token server.
    When the access token expires, a new access token is obtained from the token server.
    The access token will be added to outbound requests via the Authorization header
    of the form:    "Authorization: Bearer <access-token>"

    Keyword Args:
        url: The base endpoint URL for the MCSP token service [required].
        apikey: The API key used to obtain an access token [required].
        disable_ssl_verification:  A flag that indicates whether verification of the server's SSL
            certificate should be disabled or not. Defaults to False.
        headers: Default headers to be sent with every MCSP token request. Defaults to None.
        proxies: Dictionary for mapping request protocol to proxy URL.
        proxies.http (optional): The proxy endpoint to use for HTTP requests.
        proxies.https (optional): The proxy endpoint to use for HTTPS requests.

    Attributes:
        token_manager (MCSPTokenManager): Retrieves and manages MCSP tokens from the endpoint specified by the url.

    Raises:
        TypeError: The `disable_ssl_verification` is not a bool.
        ValueError: The apikey and/or url are not valid for MCSP token exchange requests.
    """

    def __init__(
        self,
        apikey: str,
        url: str,
        *,
        disable_ssl_verification: bool = False,
        headers: Optional[Dict[str, str]] = None,
        proxies: Optional[Dict[str, str]] = None,
    ) -> None:
        # Check the type of `disable_ssl_verification`. Must be a bool.
        if not isinstance(disable_ssl_verification, bool):
            raise TypeError('disable_ssl_verification must be a bool')

        self.token_manager = MCSPTokenManager(
            apikey=apikey,
            url=url,
            disable_ssl_verification=disable_ssl_verification,
            headers=headers,
            proxies=proxies,
        )

        self.validate()

    def authentication_type(self) -> str:
        """Returns this authenticator's type ('mcsp')."""
        return Authenticator.AUTHTYPE_MCSP

    def validate(self) -> None:
        """Validate apikey and url for token requests.

        Raises:
            ValueError: The apikey and/or url are not valid for token requests.
        """
        if self.token_manager.apikey is None:
            raise ValueError('The apikey shouldn\'t be None.')

        if self.token_manager.url is None:
            raise ValueError('The url shouldn\'t be None.')

    def authenticate(self, req: Request) -> None:
        """Adds MCSP authentication information to the request.

        The MCSP bearer token will be added to the request's headers in the form:
            Authorization: Bearer <bearer-token>

        Args:
            req:  The request to add MCSP authentication information to. Must contain a key to a dictionary
            called headers.
        """
        headers = req.get('headers')
        bearer_token = self.token_manager.get_token()
        headers['Authorization'] = 'Bearer {0}'.format(bearer_token)

    def set_disable_ssl_verification(self, status: bool = False) -> None:
        """Set the flag that indicates whether verification of the server's SSL certificate should be
        disabled or not. Defaults to False.

        Args:
            status: Set to true in order to disable SSL certificate verification. Defaults to False.

        Raises:
            TypeError: The `status` is not a bool.
        """
        self.token_manager.set_disable_ssl_verification(status)

    def set_headers(self, headers: Dict[str, str]) -> None:
        """Default headers to be sent with every MCSP token request.

        Args:
            headers: The headers to be sent with every MCSP token request.
        """
        self.token_manager.set_headers(headers)

    def set_proxies(self, proxies: Dict[str, str]) -> None:
        """Sets the proxies the token manager will use to communicate with MCSP on behalf of the host.

        Args:
            proxies: Dictionary for mapping request protocol to proxy URL.
            proxies.http (optional): The proxy endpoint to use for HTTP requests.
            proxies.https (optional): The proxy endpoint to use for HTTPS requests.
        """
        self.token_manager.set_proxies(proxies)
