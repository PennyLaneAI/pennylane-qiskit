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
from .exceptions import AccountNotFoundError
from .account import Account, ChannelType
from ..proxies import ProxyConfiguration
from .storage import save_config, read_config, delete_config


class AccountManager:
    """Class that bundles account management related functionality."""

    _default_account_config_json_file = os.path.join(
        os.path.expanduser("~"), ".qiskit", "qiskit-ibm.json"
    )
    _default_account_name = "default"
    _default_account_name_legacy = "default-legacy"
    _default_account_name_ibm_quantum = "default-ibm-quantum"
    _default_channel_type = "ibm_quantum"

    @classmethod
    def save(
        cls,
        token: Optional[str] = None,
        url: Optional[str] = None,
        instance: Optional[str] = None,
        channel: Optional[ChannelType] = None,
        name: Optional[str] = None,
        proxies: Optional[ProxyConfiguration] = None,
        verify: Optional[bool] = None,
        overwrite: Optional[bool] = False,
    ) -> None:
        """Save account on disk."""
        name = name or cls._default_account_name_ibm_quantum
        return save_config(
            filename=cls._default_account_config_json_file,
            name=name,
            overwrite=overwrite,
            config=Account(
                token=token,
                url=url,
                instance=instance,
                channel=channel,
                proxies=proxies,
                verify=verify,
            )
            # avoid storing invalid accounts
            .validate().to_saved_format(),
        )

    @classmethod
    def list(
        cls,
        default: Optional[bool] = None,
        channel: Optional[ChannelType] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Account]:
        """List all accounts saved on disk."""

        def _matching_name(account_name: str) -> bool:
            return name is None or name == account_name

        def _matching_channel(account: Account) -> bool:
            return channel is None or account.channel == channel

        def _matching_default(account_name: str) -> bool:
            default_accounts = [
                cls._default_account_name,
                cls._default_account_name_ibm_quantum,
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
            read_config(filename=cls._default_account_config_json_file).items(),
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
        cls, name: Optional[str] = None, channel: Optional[ChannelType] = None
    ) -> Optional[Account]:
        """Read account from disk.

        Args:
            name: Account name. Takes precedence.
            channel: Channel type.

        Returns:
            Account information.

        Raises:
            AccountNotFoundError: If the input value cannot be found on disk.
        """
        if name:
            saved_account = read_config(
                filename=cls._default_account_config_json_file, name=name
            )
            if not saved_account:
                raise AccountNotFoundError(
                    f"Account with the name {name} does not exist on disk."
                )
            return Account.from_saved_format(saved_account)

        channel_ = channel or cls._default_channel_type
        env_account = cls._from_env_variables(channel_)
        if env_account is not None:
            return env_account

        if channel:
            saved_account = read_config(
                filename=cls._default_account_config_json_file,
                name=cls._default_account_name_ibm_quantum,
            )
            if saved_account is None:
                raise AccountNotFoundError(f"No default {channel} account saved.")
            return Account.from_saved_format(saved_account)

        all_config = read_config(filename=cls._default_account_config_json_file)
        account_name = cls._default_account_name_ibm_quantum
        if account_name in all_config:
            return Account.from_saved_format(all_config[account_name])

        raise AccountNotFoundError("Unable to find account.")

    @classmethod
    def delete(
        cls,
        name: Optional[str] = None,
    ) -> bool:
        """Delete account from disk."""
        name = name or cls._default_account_name_ibm_quantum
        return delete_config(name=name, filename=cls._default_account_config_json_file)

    @classmethod
    def _from_env_variables(cls, channel: Optional[ChannelType]) -> Optional[Account]:
        """Read account from environment variable."""
        token = os.getenv("QISKIT_IBM_TOKEN")
        url = os.getenv("QISKIT_IBM_URL")
        if not (token and url):
            return None
        return Account(
            token=token,
            url=url,
            instance=os.getenv("QISKIT_IBM_INSTANCE"),
            channel=channel,
        )
