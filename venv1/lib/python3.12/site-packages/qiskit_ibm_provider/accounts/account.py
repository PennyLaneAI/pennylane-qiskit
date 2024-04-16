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

"""Account related classes and functions."""

import logging
from typing import Optional
from urllib.parse import urlparse

from requests.auth import AuthBase
from typing_extensions import Literal

from .exceptions import InvalidAccountError
from ..api.auth import QuantumAuth
from ..proxies import ProxyConfiguration
from ..utils.hgp import from_instance_format

ChannelType = Optional[Literal["ibm_quantum"]]

IBM_QUANTUM_API_URL = "https://auth.quantum-computing.ibm.com/api"
logger = logging.getLogger(__name__)


class Account:
    """Class that represents an account."""

    def __init__(
        self,
        channel: ChannelType,
        token: str,
        url: Optional[str] = None,
        instance: Optional[str] = None,
        proxies: Optional[ProxyConfiguration] = None,
        verify: Optional[bool] = True,
    ):
        """Account constructor.

        Args:
            channel: Channel type, ``ibm_quantum``.
            token: Account token to use.
            url: Authentication URL.
            instance: Service instance to use.
            proxies: Proxy configuration.
            verify: Whether to verify server's TLS certificate.
        """
        resolved_url = url or IBM_QUANTUM_API_URL

        self.channel = channel
        self.token = token
        self.url = resolved_url
        self.instance = instance
        self.proxies = proxies
        self.verify = verify

    def to_saved_format(self) -> dict:
        """Returns a dictionary that represents how the account is saved on disk."""
        result = {k: v for k, v in self.__dict__.items() if v is not None}
        if self.proxies:
            result["proxies"] = self.proxies.to_dict()
        return result

    @classmethod
    def from_saved_format(cls, data: dict) -> "Account":
        """Creates an account instance from data saved on disk."""
        proxies = data.get("proxies")
        return cls(
            channel=data.get("channel"),
            url=data.get("url"),
            token=data.get("token"),
            instance=data.get("instance"),
            proxies=ProxyConfiguration(**proxies) if proxies else None,
            verify=data.get("verify", True),
        )

    def get_auth_handler(self) -> AuthBase:
        """Returns the respective authentication handler."""
        return QuantumAuth(access_token=self.token)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Account):
            return False
        return all(
            [
                self.channel == other.channel,
                self.token == other.token,
                self.url == other.url,
                self.instance == other.instance,
                self.proxies == other.proxies,
                self.verify == other.verify,
            ]
        )

    def validate(self) -> "Account":
        """Validates the account instance.

        Raises:
            InvalidAccountError: if the account is invalid

        Returns:
            This Account instance.
        """

        self._assert_valid_channel(self.channel)
        self._assert_valid_token(self.token)
        self._assert_valid_url(self.url)
        self._assert_valid_instance(self.instance)
        self._assert_valid_proxies(self.proxies)
        return self

    @staticmethod
    def _assert_valid_channel(channel: ChannelType) -> None:
        """Assert that the channel parameter is valid."""
        if channel != "ibm_quantum":
            raise InvalidAccountError(
                f"Invalid `channel` value. Expected "
                f"{'ibm_quantum'}, got '{channel}'."
            )

    @staticmethod
    def _assert_valid_token(token: str) -> None:
        """Assert that the token is valid."""
        if not (isinstance(token, str) and len(token) > 0):
            raise InvalidAccountError(
                f"Invalid `token` value. Expected a non-empty string, got '{token}'."
            )

    @staticmethod
    def _assert_valid_url(url: str) -> None:
        """Assert that the URL is valid."""
        try:
            urlparse(url)
        except:
            raise InvalidAccountError(
                f"Invalid `url` value. Failed to parse '{url}' as URL."
            )

    @staticmethod
    def _assert_valid_proxies(config: ProxyConfiguration) -> None:
        """Assert that the proxy configuration is valid."""
        if config is not None:
            config.validate()

    @staticmethod
    def _assert_valid_instance(instance: str) -> None:
        """Assert that the instance name is valid for the given account type."""
        if instance is not None:
            try:
                from_instance_format(instance)
            except:
                raise InvalidAccountError(
                    f"Invalid `instance` value. Expected hub/group/project format, got {instance}"
                )
