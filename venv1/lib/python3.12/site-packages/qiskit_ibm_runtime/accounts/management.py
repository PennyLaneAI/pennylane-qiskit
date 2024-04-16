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

"""Account management related classes and functions."""

import os
from typing import Optional, Dict

from ..proxies import ProxyConfiguration
from .exceptions import AccountNotFoundError
from .account import Account, ChannelType
from .storage import save_config, read_config, delete_config

_DEFAULT_ACCOUNT_CONFIG_JSON_FILE = os.path.join(
    os.path.expanduser("~"), ".qiskit", "qiskit-ibm.json"
)
_DEFAULT_ACCOUNT_NAME = "default"
_DEFAULT_ACCOUNT_NAME_IBM_QUANTUM = "default-ibm-quantum"
_DEFAULT_ACCOUNT_NAME_IBM_CLOUD = "default-ibm-cloud"
_DEFAULT_CHANNEL_TYPE: ChannelType = "ibm_cloud"
_CHANNEL_TYPES = [_DEFAULT_CHANNEL_TYPE, "ibm_quantum"]


class AccountManager:
    """Class that bundles account management related functionality."""

    @classmethod
    def save(
        cls,
        token: Optional[str] = None,
        url: Optional[str] = None,
        instance: Optional[str] = None,
        channel: Optional[ChannelType] = None,
        filename: Optional[str] = None,
        name: Optional[str] = _DEFAULT_ACCOUNT_NAME,
        proxies: Optional[ProxyConfiguration] = None,
        verify: Optional[bool] = None,
        overwrite: Optional[bool] = False,
        channel_strategy: Optional[str] = None,
        set_as_default: Optional[bool] = None,
    ) -> None:
        """Save account on disk."""
        channel = channel or os.getenv("QISKIT_IBM_CHANNEL") or _DEFAULT_CHANNEL_TYPE
        name = name or cls._get_default_account_name(channel)
        filename = filename if filename else _DEFAULT_ACCOUNT_CONFIG_JSON_FILE
        filename = os.path.expanduser(filename)
        config = Account.create_account(
            channel=channel,
            token=token,
            url=url,
            instance=instance,
            proxies=proxies,
            verify=verify,
            channel_strategy=channel_strategy,
        )
        return save_config(
            filename=filename,
            name=name,
            overwrite=overwrite,
            config=config
            # avoid storing invalid accounts
            .validate().to_saved_format(),
            set_as_default=set_as_default,
        )

    @staticmethod
    def list(
        default: Optional[bool] = None,
        channel: Optional[ChannelType] = None,
        filename: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Account]:
        """List all accounts in a given filename, or in the default account file."""
        filename = filename if filename else _DEFAULT_ACCOUNT_CONFIG_JSON_FILE
        filename = os.path.expanduser(filename)

        def _matching_name(account_name: str) -> bool:
            return name is None or name == account_name

        def _matching_channel(account: Account) -> bool:
            return channel is None or account.channel == channel

        def _matching_default(account_name: str) -> bool:
            default_accounts = [
                _DEFAULT_ACCOUNT_NAME,
                _DEFAULT_ACCOUNT_NAME_IBM_QUANTUM,
                _DEFAULT_ACCOUNT_NAME_IBM_CLOUD,
            ]
            if default is None:
                return True
            elif default is False:
                return account_name not in default_accounts
            else:
                return account_name in default_accounts

        # load all accounts
        all_accounts = map(
            lambda kv: (
                kv[0],
                Account.from_saved_format(kv[1]),
            ),
            read_config(filename=filename).items(),
        )

        # filter based on input parameters
        filtered_accounts = dict(
            list(
                filter(
                    lambda kv: _matching_channel(kv[1])
                    and _matching_default(kv[0])
                    and _matching_name(kv[0]),
                    all_accounts,
                )
            )
        )

        return filtered_accounts

    @classmethod
    def get(
        cls,
        filename: Optional[str] = None,
        name: Optional[str] = None,
        channel: Optional[ChannelType] = None,
    ) -> Optional[Account]:
        """Read account from disk.

        Args:
            filename: Full path of the file from which to get the account.
            name: Account name.
            channel: Channel type.
            Order of precedence for selecting the account:
            1. If name is specified, get account with that name
            2. If the environment variables define an account, get that one
            3. If the channel parameter is defined,
               a. get the account of this channel type defined as "is_default_account"
               b. get the account of this channel type with default name
               c. get any account of this channel type
            4. If the channel is defined in "QISKIT_IBM_CHANNEL"
               a. get the account of this channel type defined as "is_default_account"
               b. get the account of this channel type with default name
               c. get any account of this channel type
            5. If a default account is defined in the json file, get that account
            6. Get any account that is defined in the json file with
               preference for _DEFAULT_CHANNEL_TYPE.


        Returns:
            Account information.

        Raises:
            AccountNotFoundError: If the input value cannot be found on disk.
        """
        filename = filename if filename else _DEFAULT_ACCOUNT_CONFIG_JSON_FILE
        filename = os.path.expanduser(filename)
        if name:
            saved_account = read_config(filename=filename, name=name)
            if not saved_account:
                raise AccountNotFoundError(f"Account with the name {name} does not exist on disk.")
            return Account.from_saved_format(saved_account)

        channel_ = channel or os.getenv("QISKIT_IBM_CHANNEL") or _DEFAULT_CHANNEL_TYPE
        env_account = cls._from_env_variables(channel_)
        if env_account is not None:
            return env_account

        all_config = read_config(filename=filename)
        # Get the default account for the given channel.
        # If channel == None, get the default account, for any channel, if it exists
        saved_account = cls._get_default_account(all_config, channel)

        if saved_account is not None:
            return Account.from_saved_format(saved_account)

        # Get the default account from the channel defined in the environment variable
        account = cls._get_default_account(all_config, channel=channel_)
        if account is not None:
            return Account.from_saved_format(account)

        # check for any account
        for channel_type in _CHANNEL_TYPES:
            account_name = cls._get_default_account_name(channel=channel_type)
            if account_name in all_config:
                return Account.from_saved_format(all_config[account_name])

        raise AccountNotFoundError("Unable to find account.")

    @classmethod
    def delete(
        cls,
        filename: Optional[str] = None,
        name: Optional[str] = None,
        channel: Optional[ChannelType] = None,
    ) -> bool:
        """Delete account from disk."""
        filename = filename if filename else _DEFAULT_ACCOUNT_CONFIG_JSON_FILE
        filename = os.path.expanduser(filename)
        name = name or cls._get_default_account_name(channel)
        return delete_config(
            filename=filename,
            name=name,
        )

    @classmethod
    def _from_env_variables(cls, channel: Optional[ChannelType]) -> Optional[Account]:
        """Read account from environment variable."""
        token = os.getenv("QISKIT_IBM_TOKEN")
        url = os.getenv("QISKIT_IBM_URL")
        if not (token and url):
            return None
        return Account.create_account(
            token=token,
            url=url,
            instance=os.getenv("QISKIT_IBM_INSTANCE"),
            channel=channel,
        )

    @classmethod
    def _get_default_account(
        cls, all_config: dict, channel: Optional[str] = None
    ) -> Optional[dict]:
        default_channel_account = None
        any_channel_account = None

        for account_name in all_config:
            account = all_config[account_name]
            if channel:
                if account.get("channel") == channel and account.get("is_default_account"):
                    return account
                if account.get(
                    "channel"
                ) == channel and account_name == cls._get_default_account_name(channel):
                    default_channel_account = account
                if account.get("channel") == channel:
                    any_channel_account = account
            else:
                if account.get("is_default_account"):
                    return account

        if default_channel_account:
            return default_channel_account
        elif any_channel_account:
            return any_channel_account
        return None

    @classmethod
    def _get_default_account_name(cls, channel: ChannelType) -> str:
        return (
            _DEFAULT_ACCOUNT_NAME_IBM_QUANTUM
            if channel == "ibm_quantum"
            else _DEFAULT_ACCOUNT_NAME_IBM_CLOUD
        )
