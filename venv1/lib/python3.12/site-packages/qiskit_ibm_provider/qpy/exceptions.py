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

"""Exception for errors raised by the pulse module."""
from typing import Any
from qiskit.qpy.exceptions import QpyError
from ..exceptions import IBMError


class IBMQpyError(QpyError, IBMError):
    """Errors raised by the qpy module."""

    def __init__(self, *message: Any):
        """Set the error message."""
        super().__init__(*message)
        self.message = " ".join(message)

    def __str__(self) -> str:
        """Return the message."""
        return repr(self.message)
