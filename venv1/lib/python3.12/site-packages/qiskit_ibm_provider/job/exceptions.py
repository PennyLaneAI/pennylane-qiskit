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

"""Exceptions related to IBM Quantum jobs."""

from qiskit.providers.exceptions import JobError, JobTimeoutError

from ..exceptions import IBMError


class IBMJobError(JobError, IBMError):
    """Base class for errors raised by the job modules."""

    pass


class IBMJobApiError(IBMJobError):
    """Errors that occur unexpectedly when querying the server."""

    pass


class IBMJobFailureError(IBMJobError):
    """Errors raised when a job failed."""

    pass


class IBMJobInvalidStateError(IBMJobError):
    """Errors raised when a job is not in a valid state for the operation."""

    pass


class IBMJobTimeoutError(JobTimeoutError, IBMJobError):
    """Errors raised when a job operation times out."""

    pass


class IBMJobNotFoundError(IBMJobError):
    """Errors raised when a job cannot be found."""

    pass
