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

"""IBM Quantum job."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime as python_datetime
from typing import Dict, Optional, Any, List, Union

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.providers.job import JobV1 as Job
from qiskit.providers.models import BackendProperties
from qiskit.qobj import QasmQobj, PulseQobj
from qiskit.result import Result

from qiskit_ibm_provider import ibm_backend  # pylint: disable=unused-import
from .queueinfo import QueueInfo
from ..api.clients import AccountClient

logger = logging.getLogger(__name__)


class IBMJob(Job, ABC):
    """Abstract base class for all IBM Quantum job."""

    _data = {}  # type: Dict

    def __init__(
        self,
        backend: "ibm_backend.IBMBackend",
        api_client: AccountClient,
        job_id: str,
        name: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """IBMJob constructor.

        Args:
            backend: The backend instance used to run this job.
            api_client: Object for connecting to the server.
            job_id: Job ID.
            name: Job name.
            session_id: Job ID of the first job in a runtime session.
            tags: Job tags.
            kwargs: Additional job attributes.
        """
        Job.__init__(self, backend, job_id)
        self._api_client = api_client
        self._name = name
        self._session_id = session_id
        self._tags = tags or []
        self._provider = backend.provider

        self._data = {}
        for key, value in kwargs.items():
            # Append suffix to key to avoid conflicts.
            self._data[key + "_"] = value

    def properties(self, refresh: bool = False) -> Optional[BackendProperties]:
        """Return the backend properties for this job.

        Args:
            refresh: If ``True``, re-query the server for the backend properties.
                Otherwise, return a cached version.

        Returns:
            The backend properties used for this job, at the time the job was run,
            or ``None`` if properties are not available.
        """
        return self._backend.properties(refresh, self.creation_date())

    @abstractmethod
    def result(
        self,
        timeout: Optional[float] = None,
        wait: float = 5,
        partial: bool = False,
        refresh: bool = False,
    ) -> Result:
        """Return the result of the job.

        Args:
            timeout: Number of seconds to wait for job.
            wait: Time in seconds between queries.
            partial: If ``True``, return partial results if possible.
            refresh: If ``True``, re-query the server for the result. Otherwise
                return the cached value.

        Returns:
            Job result.
        """
        # pylint: disable=arguments-differ
        pass

    @abstractmethod
    def cancel(self) -> bool:
        """Attempt to cancel the job.

        Returns:
            ``True`` if the job is cancelled, else ``False``.
        """
        pass

    def update_name(self, name: str) -> str:
        """Update the name associated with this job.

        Args:
            name: The new `name` for this job.

        Returns:
            The new name associated with this job.
        """
        pass

    @abstractmethod
    def update_tags(self, new_tags: List[str]) -> List[str]:
        """Update the tags associated with this job.

        Args:
            new_tags: New tags to assign to the job.

        Returns:
            The new tags associated with this job.
        """
        pass

    @abstractmethod
    def error_message(self) -> Optional[str]:
        """Provide details about the reason of failure.

        Returns:
            An error report if the job failed or ``None`` otherwise.
        """
        pass

    @abstractmethod
    def queue_position(self, refresh: bool = False) -> Optional[int]:
        """Return the position of the job in the server queue.

        Args:
            refresh: If ``True``, re-query the server to get the latest value.
                Otherwise return the cached value.

        Returns:
            Position in the queue or ``None`` if position is unknown or not applicable.
        """
        pass

    @abstractmethod
    def queue_info(self) -> Optional[QueueInfo]:
        """Return queue information for this job.

        Returns:
            A :class:`QueueInfo` instance that contains queue information for
            this job, or ``None`` if queue information is unknown or not
            applicable.
        """
        pass

    @abstractmethod
    def creation_date(self) -> python_datetime:
        """Return job creation date, in local time.

        Returns:
            The job creation date as a datetime object, in local time.
        """
        pass

    def time_per_step(self) -> Optional[Dict]:
        """Return the date and time information on each step of the job processing.

        Returns:
            Date and time information on job processing steps, in local time,
            or ``None`` if the information is not yet available.
        """
        pass

    def scheduling_mode(self) -> Optional[str]:
        """Return the scheduling mode the job is in.

        Returns:
            The scheduling mode the job is in or ``None`` if the information
            is not available.
        """
        pass

    @abstractmethod
    def refresh(self) -> None:
        """Obtain the latest job information from the server."""
        pass

    def circuits(self) -> List[QuantumCircuit]:
        """Return the circuits for this job.

        Returns:
            The circuits for this job. An empty list
            is returned if the circuits cannot be retrieved (for example, if
            the job uses an old format that is no longer supported).
        """
        pass

    @abstractmethod
    def backend_options(self) -> Dict[str, Any]:
        """Return the backend configuration options used for this job.

        Returns:
            Backend options used for this job.
        """
        pass

    @abstractmethod
    def header(self) -> Dict:
        """Return the user header specified for this job.

        Returns:
            User header specified for this job.
        """
        pass

    def name(self) -> Optional[str]:
        """Return the name assigned to this job.

        Returns:
            Job name or ``None`` if no name was assigned to this job.
        """
        return self._name

    def tags(self) -> List[str]:
        """Return the tags assigned to this job.

        Returns:
            Tags assigned to this job.
        """
        return self._tags.copy()

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError("Attribute {} is not defined.".format(name)) from None

    def _get_qobj(self) -> Optional[Union[QasmQobj, PulseQobj]]:
        """Return the Qobj for this job.

        Returns:
            The Qobj for this job, or ``None`` if the job does not have a Qobj.
        """
        pass

    def __repr__(self) -> str:
        return "<{}('{}')>".format(self.__class__.__name__, self.job_id())
