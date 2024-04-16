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

"""Represent IBM Quantum account client parameters."""

from typing import Dict, Optional, Any, Union
from ..proxies import ProxyConfiguration

from ..utils import get_runtime_api_base_url
from ..api.auth import QuantumAuth, CloudAuth

TEMPLATE_IBM_HUBS = "{prefix}/Network/{hub}/Groups/{group}/Projects/{project}"
"""str: Template for creating an IBM Quantum URL with hub/group/project information."""


class ClientParameters:
    """IBM Quantum account client parameters."""

    def __init__(
        self,
        channel: str,
        token: str,
        url: str = None,
        instance: Optional[str] = None,
        proxies: Optional[ProxyConfiguration] = None,
        verify: bool = True,
    ) -> None:
        """ClientParameters constructor.

        Args:
            channel: Channel type. ``ibm_cloud`` or ``ibm_quantum``.
            token: IBM Quantum API token.
            url: IBM Quantum URL (gets replaced with a new-style URL with hub, group, project).
            instance: Service instance to use.
            proxies: Proxy configuration.
            verify: If ``False``, ignores SSL certificates errors.
        """
        self.token = token
        self.instance = instance
        self.channel = channel
        self.url = url
        self.proxies = proxies
        self.verify = verify

    def get_auth_handler(self) -> Union[CloudAuth, QuantumAuth]:
        """Returns the respective authentication handler."""
        if self.channel == "ibm_cloud":
            return CloudAuth(api_key=self.token, crn=self.instance)

        return QuantumAuth(access_token=self.token)

    def get_runtime_api_base_url(self) -> str:
        """Returns the Runtime API base url."""
        return get_runtime_api_base_url(self.url, self.instance)

    def connection_parameters(self) -> Dict[str, Any]:
        """Construct connection related parameters.

        Returns:
            A dictionary with connection-related parameters in the format
            expected by ``requests``. The following keys can be present:
            ``proxies``, ``verify``, and ``auth``.
        """
        request_kwargs: Any = {"verify": self.verify}

        if self.proxies:
            request_kwargs.update(self.proxies.to_request_params())

        return request_kwargs
