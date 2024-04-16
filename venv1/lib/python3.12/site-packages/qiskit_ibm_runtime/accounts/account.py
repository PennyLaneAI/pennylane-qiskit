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

from abc import abstractmethod
import logging
from typing import Optional, Literal
from urllib.parse import urlparse

from requests.auth import AuthBase
from ..proxies import ProxyConfiguration
from ..utils.hgp import from_instance_format

from .exceptions import InvalidAccountError, CloudResourceNameResolutionError
from ..api.auth import QuantumAuth, CloudAuth
from ..utils import resolve_crn

AccountType = Optional[Literal["cloud", "legacy"]]
ChannelType = Optional[Literal["ibm_cloud", "ibm_quantum"]]

IBM_QUANTUM_API_URL = "https://auth.quantum-computing.ibm.com/api"
IBM_CLOUD_API_URL = "https://cloud.ibm.com"
logger = logging.getLogger(__name__)


class Account:
    """Class that represents an account. This is an abstract class."""

    def __init__(
        self,
        token: str,
        instance: Optional[str] = None,
        proxies: Optional[ProxyConfiguration] = None,
        verify: Optional[bool] = True,
        channel_strategy: Optional[str] = None,
    ):
        """Account constructor.

        Args:
            channel: Channel type, ``ibm_cloud`` or ``ibm_quantum``.
            token: Account token to use.
            url: Authentication URL.
            instance: Service instance to use.
            proxies: Proxy configuration.
            verify: Whether to verify server's TLS certificate.
            channel_strategy: Error mitigation strategy.
        """
        self.channel: str = None
        self.url: str = None
        self.token = token
        self.instance = instance
        self.proxies = proxies
        self.verify = verify
        self.channel_strategy = channel_strategy

    def to_saved_format(self) -> dict:
        """Returns a dictionary that represents how the account is saved on disk."""
        result = {k: v for k, v in self.__dict__.items() if v is not None}
        if self.proxies:
            result["proxies"] = self.proxies.to_dict()
        return result

    @classmethod
    def from_saved_format(cls, data: dict) -> "Account":
        """Creates an account instance from data saved on disk."""
        channel = data.get("channel")
        proxies = data.get("proxies")
        proxies = ProxyConfiguration(**proxies) if proxies else None
        url = data.get("url")
        token = data.get("token")
        instance = data.get("instance")
        verify = data.get("verify", True)
        channel_strategy = data.get("channel_strategy")
        return cls.create_account(
            channel=channel,
            url=url,
            token=token,
            instance=instance,
            proxies=proxies,
            verify=verify,
            channel_strategy=channel_strategy,
        )

    @classmethod
    def create_account(
        cls,
        channel: str,
        token: str,
        url: Optional[str] = None,
        instance: Optional[str] = None,
        proxies: Optional[ProxyConfiguration] = None,
        verify: Optional[bool] = True,
        channel_strategy: Optional[str] = None,
    ) -> "Account":
        """Creates an account for a specific channel."""
        if channel == "ibm_quantum":
            return QuantumAccount(
                url=url,
                token=token,
                instance=instance,
                proxies=proxies,
                verify=verify,
                channel_strategy=channel_strategy,
            )
        elif channel == "ibm_cloud":
            return CloudAccount(
                url=url,
                token=token,
                instance=instance,
                proxies=proxies,
                verify=verify,
                channel_strategy=channel_strategy,
            )
        else:
            raise InvalidAccountError(
                f"Invalid `channel` value. Expected one of "
                f"{['ibm_cloud', 'ibm_quantum']}, got '{channel}'."
            )

    def resolve_crn(self) -> None:
        """Resolves the corresponding unique Cloud Resource Name (CRN) for the given non-unique service
        instance name and updates the ``instance`` attribute accordingly.
        Relevant for "ibm_cloud" channel only."""
        pass

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
        self._assert_valid_channel_strategy(self.channel_strategy)
        return self

    @staticmethod
    def _assert_valid_channel_strategy(channel_strategy: str) -> None:
        """Assert that the channel strategy is valid."""
        # add more strategies as they are implemented
        strategies = ["q-ctrl", "default"]
        if channel_strategy and channel_strategy not in strategies:
            raise InvalidAccountError(
                f"Invalid `channel_strategy` value. Expected one of "
                f"{strategies}, got '{channel_strategy}'."
            )

    @staticmethod
    def _assert_valid_channel(channel: ChannelType) -> None:
        """Assert that the channel parameter is valid."""
        if not (channel in ["ibm_cloud", "ibm_quantum"]):
            raise InvalidAccountError(
                f"Invalid `channel` value. Expected one of "
                f"['ibm_cloud', 'ibm_quantum'], got '{channel}'."
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
            raise InvalidAccountError(f"Invalid `url` value. Failed to parse '{url}' as URL.")

    @staticmethod
    def _assert_valid_proxies(config: ProxyConfiguration) -> None:
        """Assert that the proxy configuration is valid."""
        if config is not None:
            config.validate()

    @staticmethod
    @abstractmethod
    def _assert_valid_instance(instance: str) -> None:
        """Assert that the instance name is valid for the given account type."""
        pass


class QuantumAccount(Account):
    """Class that represents an account with channel 'ibm_quantum.'"""

    def __init__(
        self,
        token: str,
        url: Optional[str] = None,
        instance: Optional[str] = None,
        proxies: Optional[ProxyConfiguration] = None,
        verify: Optional[bool] = True,
        channel_strategy: Optional[str] = None,
    ):
        """Account constructor.

        Args:
            token: Account token to use.
            url: Authentication URL.
            instance: Service instance to use.
            proxies: Proxy configuration.
            verify: Whether to verify server's TLS certificate.
            channel_strategy: Error mitigation strategy.
        """
        super().__init__(token, instance, proxies, verify, channel_strategy)
        resolved_url = url or IBM_QUANTUM_API_URL
        self.channel = "ibm_quantum"
        self.url = resolved_url

    def get_auth_handler(self) -> AuthBase:
        """Returns the Quantum authentication handler."""
        return QuantumAuth(access_token=self.token)

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


class CloudAccount(Account):
    """Class that represents an account with channel 'ibm_cloud'."""

    def __init__(
        self,
        token: str,
        url: Optional[str] = None,
        instance: Optional[str] = None,
        proxies: Optional[ProxyConfiguration] = None,
        verify: Optional[bool] = True,
        channel_strategy: Optional[str] = None,
    ):
        """Account constructor.

        Args:
            token: Account token to use.
            url: Authentication URL.
            instance: Service instance to use.
            proxies: Proxy configuration.
            verify: Whether to verify server's TLS certificate.
            channel_strategy: Error mitigation strategy.
        """
        super().__init__(token, instance, proxies, verify, channel_strategy)
        resolved_url = url or IBM_CLOUD_API_URL
        self.channel = "ibm_cloud"
        self.url = resolved_url

    def get_auth_handler(self) -> AuthBase:
        """Returns the Cloud authentication handler."""
        return CloudAuth(api_key=self.token, crn=self.instance)

    def resolve_crn(self) -> None:
        """Resolves the corresponding unique Cloud Resource Name (CRN) for the given non-unique service
        instance name and updates the ``instance`` attribute accordingly.

        No-op if ``instance`` attribute is set to a Cloud Resource Name (CRN).

        Raises:
            CloudResourceNameResolutionError: if CRN value cannot be resolved.
        """
        crn = resolve_crn(
            channel="ibm_cloud",
            url=self.url,
            token=self.token,
            instance=self.instance,
        )
        if len(crn) == 0:
            raise CloudResourceNameResolutionError(
                f"Failed to resolve CRN value for the provided service name {self.instance}."
            )
        if len(crn) > 1:
            # handle edge-case where multiple service instances with the same name exist
            logger.warning(
                "Multiple CRN values found for service name %s: %s. Using %s.",
                self.instance,
                crn,
                crn[0],
            )

        # overwrite with CRN value
        self.instance = crn[0]

    @staticmethod
    def _assert_valid_instance(instance: str) -> None:
        """Assert that the instance name is valid for the given account type."""
        if not (isinstance(instance, str) and len(instance) > 0):
            raise InvalidAccountError(
                f"Invalid `instance` value. Expected a non-empty string, got '{instance}'."
            )
