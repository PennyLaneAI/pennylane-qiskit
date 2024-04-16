# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Methods for checking if we are inside a Session context manager"""

from contextvars import ContextVar
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .session import Session

# Default session
_DEFAULT_SESSION: ContextVar[Optional["Session"]] = ContextVar("_DEFAULT_SESSION", default=None)
_IN_SESSION_CM: ContextVar[bool] = ContextVar("_IN_SESSION_CM", default=False)


def set_cm_session(session: Optional["Session"]) -> None:
    """Set the context manager session."""
    _DEFAULT_SESSION.set(session)
    _IN_SESSION_CM.set(session is not None)


def get_cm_session() -> "Session":
    """Return the context managed session."""
    return _DEFAULT_SESSION.get()
