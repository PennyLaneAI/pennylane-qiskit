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

"""Exceptions for the ``Accounts`` module."""

from ..exceptions import IBMAccountError


class AccountsError(IBMAccountError):
    """Base class for errors raised during account management."""


class InvalidAccountError(AccountsError):
    """Errors raised when the account is invalid."""


class AccountNotFoundError(AccountsError):
    """Errors raised when the account is not found."""


class AccountAlreadyExistsError(AccountsError):
    """Errors raised when the account already exists."""


class CloudResourceNameResolutionError(AccountsError):
    """Errors raised when the Cloud Resource Name (CRN) cannot be resolved for a given service name."""
