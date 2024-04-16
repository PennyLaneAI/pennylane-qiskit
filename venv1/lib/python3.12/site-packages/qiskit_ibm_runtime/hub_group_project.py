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

"""A hub, group and project in an IBM Quantum account."""

import logging
from typing import Any, List

from qiskit_ibm_runtime import (  # pylint: disable=unused-import
    ibm_backend,
    qiskit_runtime_service,
)

from .api.client_parameters import ClientParameters
from .api.clients import RuntimeClient
from .utils.hgp import from_instance_format

logger = logging.getLogger(__name__)


class HubGroupProject:
    """Represents a hub/group/project with IBM Quantum backends and services associated with it."""

    def __init__(
        self,
        client_params: ClientParameters,
        instance: str,
        service: "qiskit_runtime_service.QiskitRuntimeService",
    ) -> None:
        """HubGroupProject constructor

        Args:
            client_params: Parameters used for server connection.
            instance: Hub/group/project.
        """
        self._service = service
        self._runtime_client = RuntimeClient(client_params)
        # Initialize the internal list of backends.
        # self._backends: Dict[str, "ibm_backend.IBMBackend"] = {}
        self._backends: List[str] = []
        self._hub, self._group, self._project = from_instance_format(instance)

    @property
    def backends(self) -> List[str]:
        """Gets the backends for the hub/group/project.

        Returns:
            A list of backend names.
        """
        if not self._backends:
            self._backends = self._discover_remote_backends()
        return self._backends

    @backends.setter
    def backends(self, value: List[str]) -> None:
        """Sets the value for the hub/group/project's backends.

        Args:
            value: the backends
        """
        self._backends = value

    def _discover_remote_backends(self) -> List[str]:
        """Return the remote backends available for this hub/group/project.

        Returns:
            A list of backends.
        """
        backends = self._runtime_client.list_backends(self.name)
        return backends or []

    def has_backend(self, name: str) -> bool:
        """Determine if the hgp can access the backend."""
        return name in self._backends

    @property
    def name(self) -> str:
        """Returns the unique id.

        Returns:
            An ID uniquely represents this h/g/p.
        """
        return f"{self._hub}/{self._group}/{self._project}"

    def __repr__(self) -> str:
        hgp_info = "hub='{}', group='{}', project='{}'".format(
            self._hub, self._group, self._project
        )
        return "<{}({})>".format(self.__class__.__name__, hgp_info)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, HubGroupProject):
            return False
        return self.name == other.name
