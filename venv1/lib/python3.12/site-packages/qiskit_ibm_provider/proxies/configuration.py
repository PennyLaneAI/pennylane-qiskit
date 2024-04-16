# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Proxy related classes and functions."""


from dataclasses import dataclass
from typing import Optional, Dict, Any
from urllib.parse import urlparse

from requests_ntlm import HttpNtlmAuth


@dataclass
class ProxyConfiguration:
    """Class for representing a proxy configuration.

    Args
        urls: a dictionary mapping protocol or protocol and host to the URL of the proxy. Refer to
            https://docs.python-requests.org/en/latest/api/#requests.Session.proxies for details.
        username_ntlm: username used to enable NTLM user authentication.
        password_ntlm: password used to enable NTLM user authentication.
    """

    urls: Optional[Dict[str, str]] = None
    username_ntlm: Optional[str] = None
    password_ntlm: Optional[str] = None

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        if not any(
            [
                isinstance(self.username_ntlm, str)
                and isinstance(self.password_ntlm, str),
                self.username_ntlm is None and self.password_ntlm is None,
            ]
        ):
            raise ValueError(
                f"Invalid proxy configuration for NTLM authentication. None or both of username and "
                f"password must be provided. Got username_ntlm={self.username_ntlm}, "
                f"password_ntlm={self.password_ntlm}."
            )

        if self.urls is not None and not isinstance(self.urls, dict):
            raise ValueError(
                f"Invalid proxy configuration. Expected `urls` to contain a dictionary mapping protocol "
                f"or protocol and host to the URL of the proxy. Got {self.urls}"
            )

    def to_dict(self) -> dict:
        """Transform configuration to dictionary."""

        return {k: v for k, v in self.__dict__.items() if v is not None}

    def to_request_params(self) -> dict:
        """Transform configuration to request parameters.

        Returns:
            A dictionary with proxy configuration parameters in the format
            expected by ``requests``. The following keys can be present:
            ``proxies``and ``auth``.
        """

        request_kwargs = {}
        if self.urls:
            request_kwargs["proxies"] = self.urls

        if self.username_ntlm and self.password_ntlm:
            request_kwargs["auth"] = HttpNtlmAuth(
                self.username_ntlm, self.password_ntlm
            )

        return request_kwargs

    def to_ws_params(self, ws_url: str) -> dict:
        """Extract proxy information for websocket.

        Args:
            ws_url: Websocket URL.

        Returns:
            A dictionary with proxy configuration parameters in the format expected by websocket.
            The following keys can be present: ``http_proxy_host``and ``http_proxy_port``,
            ``proxy_type``, ``http_proxy_auth``.
        """
        out: Any = {}

        if self.urls:
            proxies = self.urls
            url_parts = urlparse(ws_url)
            proxy_keys = [
                ws_url,
                "wss",
                "https://" + url_parts.hostname,
                "https",
                "all://" + url_parts.hostname,
                "all",
            ]
            for key in proxy_keys:
                if key in proxies:
                    proxy_parts = urlparse(proxies[key], scheme="http")
                    out["http_proxy_host"] = proxy_parts.hostname
                    out["http_proxy_port"] = proxy_parts.port
                    out["proxy_type"] = (
                        "http"
                        if proxy_parts.scheme.startswith("http")
                        else proxy_parts.scheme
                    )
                    if proxy_parts.username and proxy_parts.password:
                        out["http_proxy_auth"] = (
                            proxy_parts.username,
                            proxy_parts.password,
                        )
                    break

        if self.username_ntlm and self.password_ntlm:
            out["http_proxy_auth"] = (self.username_ntlm, self.password_ntlm)

        return out
