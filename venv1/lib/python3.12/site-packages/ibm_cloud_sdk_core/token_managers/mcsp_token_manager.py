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

import json
from typing import Dict, Optional

from .jwt_token_manager import JWTTokenManager


class MCSPTokenManager(JWTTokenManager):
    """The MCSPTokenManager accepts a user-supplied apikey and performs the necessary interactions with
    the Multi-Cloud Saas Platform (MCSP) token service to obtain an MCSP access token (a bearer token).
    When the access token expires, a new access token is obtained from the token server.

    Keyword Arguments:
        apikey: The apikey for authentication [required].
        url: The endpoint for JWT token requests [required].
        disable_ssl_verification: Disable ssl verification. Defaults to False.
        headers: Headers to be sent with every service token request. Defaults to None.
        proxies: Proxies to use for making request. Defaults to None.
        proxies.http (optional): The proxy endpoint to use for HTTP requests.
        proxies.https (optional): The proxy endpoint to use for HTTPS requests.
    """

    TOKEN_NAME = 'token'
    OPERATION_PATH = '/siusermgr/api/1.0/apikeys/token'

    def __init__(
        self,
        apikey: str,
        url: str,
        *,
        disable_ssl_verification: bool = False,
        headers: Optional[Dict[str, str]] = None,
        proxies: Optional[Dict[str, str]] = None,
    ) -> None:
        self.apikey = apikey
        self.headers = headers
        if self.headers is None:
            self.headers = {}
        self.headers['Content-Type'] = 'application/json'
        self.headers['Accept'] = 'application/json'
        self.proxies = proxies
        super().__init__(url, disable_ssl_verification=disable_ssl_verification, token_name=self.TOKEN_NAME)

    def request_token(self) -> dict:
        """Makes a request for a token."""
        response = self._request(
            method='POST',
            headers=self.headers,
            url=self.url + self.OPERATION_PATH,
            data=json.dumps({"apikey": self.apikey}),
            proxies=self.proxies,
        )
        return response

    def set_headers(self, headers: Dict[str, str]) -> None:
        """Headers to be sent with every MCSP token request.

        Args:
            headers: The headers to be sent with every MCSP token request.
        """
        if isinstance(headers, dict):
            self.headers = headers
        else:
            raise TypeError('headers must be a dictionary')

    def set_proxies(self, proxies: Dict[str, str]) -> None:
        """Sets the proxies the token manager will use to communicate with MCSP on behalf of the host.

        Args:
            proxies: Proxies to use for making request. Defaults to None.
            proxies.http (optional): The proxy endpoint to use for HTTP requests.
            proxies.https (optional): The proxy endpoint to use for HTTPS requests.
        """
        if isinstance(proxies, dict):
            self.proxies = proxies
        else:
            raise TypeError('proxies must be a dictionary')
