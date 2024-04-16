# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Client for accessing backend information."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime as python_datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseBackendClient(ABC):
    """Client for accessing backend information."""

    @abstractmethod
    def backend_status(self, backend_name: str) -> Dict[str, Any]:
        """Return the status of the backend.

        Args:
            backend_name: The name of the backend.

        Returns:
            Backend status.
        """
        pass

    @abstractmethod
    def backend_properties(
        self, backend_name: str, datetime: Optional[python_datetime] = None
    ) -> Dict[str, Any]:
        """Return the properties of the backend.

        Args:
            backend_name: The name of the backend.
            datetime: Date and time for additional filtering of backend properties.

        Returns:
            Backend properties.
        """
        pass

    @abstractmethod
    def backend_pulse_defaults(self, backend_name: str) -> Dict:
        """Return the pulse defaults of the backend.

        Args:
            backend_name: The name of the backend.

        Returns:
            Backend pulse defaults.
        """
        pass
