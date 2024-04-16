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

"""Exceptions related to the IBM Runtime service."""

from qiskit.exceptions import QiskitError
from qiskit.providers.exceptions import JobTimeoutError, JobError


class IBMError(QiskitError):
    """Base class for errors raised by the runtime service modules."""

    pass


class IBMAccountError(IBMError):
    """Account related errors."""

    pass


class IBMBackendError(IBMError):
    """Base class for errors raised by the backend modules."""

    pass


class IBMBackendApiProtocolError(IBMBackendError):
    """Errors raised when an unexpected value is received from the server."""

    pass


class IBMBackendValueError(IBMBackendError, ValueError):
    """Value errors raised by the backend modules."""

    pass


class IBMBackendApiError(IBMBackendError):
    """Errors that occur unexpectedly when querying the server."""

    pass


class IBMInputValueError(IBMError):
    """Error raised due to invalid input value."""

    pass


class IBMNotAuthorizedError(IBMError):
    """Error raised when a service is invoked from an unauthorized account."""

    pass


class IBMApiError(IBMError):
    """Error raised when a server error encountered."""

    pass


class IBMRuntimeError(IBMError):
    """Base class for errors raised by the runtime service modules."""

    pass


class RuntimeDuplicateProgramError(IBMRuntimeError):
    """Error raised when a program being uploaded already exists."""

    pass


class RuntimeProgramNotFound(IBMRuntimeError):
    """Error raised when a program is not found."""

    pass


class RuntimeJobFailureError(JobError):
    """Error raised when a runtime job failed."""

    pass


class RuntimeJobNotFound(IBMRuntimeError):
    """Error raised when a job is not found."""

    pass


class RuntimeInvalidStateError(IBMRuntimeError):
    """Errors raised when the state is not valid for the operation."""

    pass


class RuntimeJobTimeoutError(JobTimeoutError):
    """Error raised when waiting for job times out."""

    pass


class RuntimeJobMaxTimeoutError(IBMRuntimeError):
    """Error raised when a job times out."""

    pass
