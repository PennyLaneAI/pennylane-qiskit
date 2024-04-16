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

"""Qiskit runtime service."""

import json
import logging
import traceback
import warnings
from datetime import datetime
from collections import OrderedDict
from typing import Dict, Callable, Optional, Union, List, Any, Type, Sequence

from qiskit.providers.backend import BackendV2 as Backend
from qiskit.providers.provider import ProviderV1 as Provider
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.providers.providerutils import filter_backends
from qiskit.providers.models import (
    PulseBackendConfiguration,
    QasmBackendConfiguration,
)

from qiskit_ibm_runtime import ibm_backend
from .proxies import ProxyConfiguration
from .utils.hgp import to_instance_format, from_instance_format
from .utils.backend_decoder import configuration_from_server_data


from .utils.utils import validate_job_tags
from .accounts import AccountManager, Account, ChannelType
from .api.clients import AuthClient, VersionClient
from .api.clients.runtime import RuntimeClient
from .api.exceptions import RequestsApiError
from .constants import QISKIT_IBM_RUNTIME_API_URL
from .exceptions import IBMNotAuthorizedError, IBMInputValueError, IBMAccountError
from .exceptions import (
    IBMRuntimeError,
    RuntimeProgramNotFound,
    RuntimeJobNotFound,
)
from .hub_group_project import HubGroupProject  # pylint: disable=cyclic-import
from .utils.result_decoder import ResultDecoder
from .runtime_job import RuntimeJob
from .runtime_job_v2 import RuntimeJobV2
from .utils import RuntimeDecoder, to_python_identifier
from .api.client_parameters import ClientParameters
from .runtime_options import RuntimeOptions
from .ibm_backend import IBMBackend

logger = logging.getLogger(__name__)

SERVICE_NAME = "runtime"


class QiskitRuntimeService(Provider):
    """Class for interacting with the Qiskit Runtime service."""

    global_service = None

    def __init__(
        self,
        channel: Optional[ChannelType] = None,
        token: Optional[str] = None,
        url: Optional[str] = None,
        filename: Optional[str] = None,
        name: Optional[str] = None,
        instance: Optional[str] = None,
        proxies: Optional[dict] = None,
        verify: Optional[bool] = None,
        channel_strategy: Optional[str] = None,
    ) -> None:
        """QiskitRuntimeService constructor

        An account is selected in the following order:

            - Account with the input `name`, if specified.
            - Default account for the `channel` type, if `channel` is specified but `token` is not.
            - Account defined by the input `channel` and `token`, if specified.
            - Account defined by the `default_channel` if defined in filename
            - Account defined by the environment variables, if defined.
            - Default account for the ``ibm_cloud`` account, if one is available.
            - Default account for the ``ibm_quantum`` account, if one is available.

        `instance`, `proxies`, and `verify` can be used to overwrite corresponding
        values in the loaded account.

        Args:
            channel: Channel type. ``ibm_cloud`` or ``ibm_quantum``.
            token: IBM Cloud API key or IBM Quantum API token.
            url: The API URL.
                Defaults to https://cloud.ibm.com (ibm_cloud) or
                https://auth.quantum-computing.ibm.com/api (ibm_quantum).
            filename: Full path of the file where the account is created.
                Default: _DEFAULT_ACCOUNT_CONFIG_JSON_FILE
            name: Name of the account to load.
            instance: The service instance to use.
                For ``ibm_cloud`` runtime, this is the Cloud Resource Name (CRN) or the service name.
                For ``ibm_quantum`` runtime, this is the hub/group/project in that format.
            proxies: Proxy configuration. Supported optional keys are
                ``urls`` (a dictionary mapping protocol or protocol and host to the URL of the proxy,
                documented at https://docs.python-requests.org/en/latest/api/#requests.Session.proxies),
                ``username_ntlm``, ``password_ntlm`` (username and password to enable NTLM user
                authentication)
            verify: Whether to verify the server's TLS certificate.
            channel_strategy: Error mitigation strategy.

        Returns:
            An instance of QiskitRuntimeService.

        Raises:
            IBMInputValueError: If an input is invalid.
        """
        super().__init__()

        self._account = self._discover_account(
            token=token,
            url=url,
            instance=instance,
            channel=channel,
            filename=filename,
            name=name,
            proxies=ProxyConfiguration(**proxies) if proxies else None,
            verify=verify,
            channel_strategy=channel_strategy,
        )

        self._client_params = ClientParameters(
            channel=self._account.channel,
            token=self._account.token,
            url=self._account.url,
            instance=self._account.instance,
            proxies=self._account.proxies,
            verify=self._account.verify,
        )

        self._channel_strategy = channel_strategy or self._account.channel_strategy
        self._channel = self._account.channel
        self._backends: Dict[str, "ibm_backend.IBMBackend"] = {}
        self._backend_configs: Dict[str, Any] = {}

        if self._channel == "ibm_cloud":
            self._api_client = RuntimeClient(self._client_params)
            # TODO: We can make the backend discovery lazy
            self._backends = self._discover_cloud_backends()
            QiskitRuntimeService.global_service = self
            self._validate_channel_strategy()
            return
        else:
            auth_client = self._authenticate_ibm_quantum_account(self._client_params)
            # Update client parameters to use authenticated values.
            self._client_params.url = auth_client.current_service_urls()["services"]["runtime"]
            if self._client_params.url == "https://api.de.quantum-computing.ibm.com/runtime":
                warnings.warn(
                    "Features in versions of qiskit-ibm-runtime greater than and including "
                    "0.13.0 may not be supported in this environment"
                )
            self._client_params.token = auth_client.current_access_token()
            self._api_client = RuntimeClient(self._client_params)
            self._hgps = self._initialize_hgps(auth_client)
            for hgp in self._hgps.values():
                for backend_name in hgp.backends:
                    if backend_name not in self._backends:
                        self._backends[backend_name] = None
            self._current_instance = self._account.instance
            if not self._current_instance:
                self._current_instance = self._get_hgp().name
                logger.info("Default instance: %s", self._current_instance)
        QiskitRuntimeService.global_service = self

        # TODO - it'd be nice to allow some kind of autocomplete, but `service.ibmq_foo`
        # just seems wrong since backends are not runtime service instances.
        # self._discover_backends()

    def _discover_account(
        self,
        token: Optional[str] = None,
        url: Optional[str] = None,
        instance: Optional[str] = None,
        channel: Optional[ChannelType] = None,
        filename: Optional[str] = None,
        name: Optional[str] = None,
        proxies: Optional[ProxyConfiguration] = None,
        verify: Optional[bool] = None,
        channel_strategy: Optional[str] = None,
    ) -> Account:
        """Discover account."""
        account = None
        verify_ = verify or True
        if channel_strategy:
            if channel_strategy not in ["q-ctrl", "default"]:
                raise ValueError(f"{channel_strategy} is not a valid channel strategy.")
            if channel and channel != "ibm_cloud":
                raise ValueError(
                    f"The channel strategy {channel_strategy} is "
                    "only supported on the ibm_cloud channel."
                )
        if name:
            if filename:
                if any([channel, token, url]):
                    logger.warning(
                        "Loading account from file %s with name %s. Any input "
                        "'channel', 'token' or 'url' are ignored.",
                        filename,
                        name,
                    )
            else:
                if any([channel, token, url]):
                    logger.warning(
                        "Loading account with name %s. Any input "
                        "'channel', 'token' or 'url' are ignored.",
                        name,
                    )
            account = AccountManager.get(filename=filename, name=name)
        elif channel:
            if channel and channel not in ["ibm_cloud", "ibm_quantum"]:
                raise ValueError("'channel' can only be 'ibm_cloud' or 'ibm_quantum'")
            if token:
                account = Account.create_account(
                    channel=channel,
                    token=token,
                    url=url,
                    instance=instance,
                    proxies=proxies,
                    verify=verify_,
                    channel_strategy=channel_strategy,
                )
            else:
                if url:
                    logger.warning("Loading default %s account. Input 'url' is ignored.", channel)
                account = AccountManager.get(filename=filename, name=name, channel=channel)
        elif any([token, url]):
            # Let's not infer based on these attributes as they may change in the future.
            raise ValueError(
                "'channel' is required if 'token', or 'url' is specified but 'name' is not."
            )

        # channel is not defined yet, get it from the AccountManager
        if account is None:
            account = AccountManager.get(filename=filename)

        if instance:
            account.instance = instance
        if proxies:
            account.proxies = proxies
        if verify is not None:
            account.verify = verify

        # resolve CRN if needed
        self._resolve_crn(account)

        # ensure account is valid, fail early if not
        account.validate()

        return account

    def _validate_channel_strategy(self) -> None:
        """Raise an error if the passed in channel_strategy and
        instance do not match.

        """
        qctrl_enabled = self._api_client.is_qctrl_enabled()
        if self._channel_strategy == "q-ctrl":
            if not qctrl_enabled:
                raise IBMNotAuthorizedError(
                    "The instance passed in is not compatible with Q-CTRL channel strategy. "
                    "Please switch to or create an instance with the Q-CTRL strategy enabled. "
                    "See https://cloud.ibm.com/docs/quantum-computing?"
                    "topic=quantum-computing-get-started for more information"
                )
        else:
            if qctrl_enabled:
                raise IBMNotAuthorizedError(
                    "The instance passed in is only compatible with Q-CTRL performance "
                    "management strategy. "
                    "To use this instance, set channel_strategy='q-ctrl'."
                )

    def _discover_cloud_backends(self) -> Dict[str, "ibm_backend.IBMBackend"]:
        """Return the remote backends available for this service instance.

        Returns:
            A dict of the remote backend instances, keyed by backend name.
        """
        ret = OrderedDict()  # type: ignore[var-annotated]
        backends_list = self._api_client.list_backends(channel_strategy=self._channel_strategy)
        for backend_name in backends_list:
            raw_config = self._api_client.backend_configuration(backend_name=backend_name)
            config = configuration_from_server_data(
                raw_config=raw_config, instance=self._account.instance
            )
            if not config:
                continue
            ret[config.backend_name] = ibm_backend.IBMBackend(
                configuration=config,
                service=self,
                api_client=self._api_client,
            )
        return ret

    def _resolve_crn(self, account: Account) -> None:
        account.resolve_crn()

    def _authenticate_ibm_quantum_account(self, client_params: ClientParameters) -> AuthClient:
        """Authenticate against IBM Quantum and populate the hub/group/projects.

        Args:
            client_params: Parameters used for server connection.

        Raises:
            IBMInputValueError: If the URL specified is not a valid IBM Quantum authentication URL.
            IBMNotAuthorizedError: If the account is not authorized to use runtime.

        Returns:
            Authentication client.
        """
        version_info = self._check_api_version(client_params)
        # Check the URL is a valid authentication URL.
        if not version_info["new_api"] or "api-auth" not in version_info:
            raise IBMInputValueError(
                "The URL specified ({}) is not an IBM Quantum authentication URL. "
                "Valid authentication URL: {}.".format(
                    client_params.url, QISKIT_IBM_RUNTIME_API_URL
                )
            )
        auth_client = AuthClient(client_params)
        service_urls = auth_client.current_service_urls()
        if not service_urls.get("services", {}).get(SERVICE_NAME):
            raise IBMNotAuthorizedError(
                "This account is not authorized to use ``ibm_quantum`` runtime service."
            )
        return auth_client

    def _initialize_hgps(
        self,
        auth_client: AuthClient,
    ) -> Dict:
        """Authenticate against IBM Quantum and populate the hub/group/projects.

        Args:
            auth_client: Authentication data.

        Raises:
            IBMInputValueError: If the URL specified is not a valid IBM Quantum authentication URL.
            IBMAccountError: If no hub/group/project could be found for this account.
            IBMInputValueError: If instance parameter is not found in hgps.

        Returns:
            The hub/group/projects for this account.
        """
        # pylint: disable=unsubscriptable-object
        hgps: OrderedDict[str, HubGroupProject] = OrderedDict()
        service_urls = auth_client.current_service_urls()
        user_hubs = auth_client.user_hubs()
        for hub_info in user_hubs:
            # Build credentials.
            hgp_params = ClientParameters(
                channel=self._account.channel,
                token=auth_client.current_access_token(),
                url=service_urls["services"]["runtime"],
                instance=to_instance_format(
                    hub_info["hub"], hub_info["group"], hub_info["project"]
                ),
                proxies=self._account.proxies,
                verify=self._account.verify,
            )

            # Build the hgp.
            try:
                hgp = HubGroupProject(
                    client_params=hgp_params, instance=hgp_params.instance, service=self
                )
                hgps[hgp.name] = hgp
            except Exception:  # pylint: disable=broad-except
                # Catch-all for errors instantiating the hgp.
                logger.warning(
                    "Unable to instantiate hub/group/project for %s: %s",
                    hub_info,
                    traceback.format_exc(),
                )
        if not hgps:
            raise IBMAccountError(
                "No hub/group/project that supports Qiskit Runtime could "
                "be found for this account."
            )
        # Move open hgp to end of the list
        if len(hgps) > 1:
            open_key, open_val = hgps.popitem(last=False)
            hgps[open_key] = open_val

        default_hgp = self._account.instance
        if default_hgp:
            if default_hgp in hgps:
                # Move user selected hgp to front of the list
                hgps.move_to_end(default_hgp, last=False)
            else:
                raise IBMInputValueError(
                    f"Hub/group/project {default_hgp} could not be found for this account."
                )
        return hgps

    @staticmethod
    def _check_api_version(params: ClientParameters) -> Dict[str, Union[bool, str]]:
        """Check the version of the remote server in a set of client parameters.

        Args:
            params: Parameters used for server connection.

        Returns:
            A dictionary with version information.
        """
        version_finder = VersionClient(url=params.url, **params.connection_parameters())
        return version_finder.version()

    def _get_hgp(
        self,
        instance: Optional[str] = None,
        backend_name: Optional[Any] = None,
    ) -> HubGroupProject:
        """Return an instance of `HubGroupProject`.

        This function also allows to find the `HubGroupProject` that contains a backend
        `backend_name`.

        Args:
            instance: The hub/group/project to use.
            backend_name: Name of the IBM Quantum backend.

        Returns:
            An instance of `HubGroupProject` that matches the specified criteria or the default.

        Raises:
            IBMInputValueError: If no hub/group/project matches the specified criteria,
                or if the input value is in an incorrect format.
            QiskitBackendNotFoundError: If backend cannot be found.
        """
        if instance:
            _ = from_instance_format(instance)  # Verify format
            if instance not in self._hgps:
                raise IBMInputValueError(
                    f"Hub/group/project {instance} " "could not be found for this account."
                )
            if backend_name and not self._hgps[instance].has_backend(backend_name):
                raise QiskitBackendNotFoundError(
                    f"Backend {backend_name} cannot be found in " f"hub/group/project {instance}"
                )
            return self._hgps[instance]

        if not backend_name:
            return list(self._hgps.values())[0]

        for hgp in self._hgps.values():
            if hgp.has_backend(backend_name):
                return hgp

        error_message = (
            f"Backend {backend_name} cannot be found in any " f"hub/group/project for this account."
        )
        if not isinstance(backend_name, str):
            error_message += (
                f" {backend_name} is of type {type(backend_name)} but should "
                f"instead be initialized through the {self}."
            )

        raise QiskitBackendNotFoundError(error_message)

    def _discover_backends(self) -> None:
        """Discovers the remote backends for this account, if not already known."""
        for backend in self._backends.values():
            backend_name = to_python_identifier(backend.name)
            # Append _ if duplicate
            while backend_name in self.__dict__:
                backend_name += "_"
            setattr(self, backend_name, backend)

    # pylint: disable=arguments-differ
    def backends(
        self,
        name: Optional[str] = None,
        min_num_qubits: Optional[int] = None,
        instance: Optional[str] = None,
        dynamic_circuits: Optional[bool] = None,
        filters: Optional[Callable[[List["ibm_backend.IBMBackend"]], bool]] = None,
        **kwargs: Any,
    ) -> List["ibm_backend.IBMBackend"]:
        """Return all backends accessible via this account, subject to optional filtering.

        Args:
            name: Backend name to filter by.
            min_num_qubits: Minimum number of qubits the backend has to have.
            instance: This is only supported for ``ibm_quantum`` runtime and is in the
                hub/group/project format.
            dynamic_circuits: Filter by whether the backend supports dynamic circuits.
            filters: More complex filters, such as lambda functions.
                For example::

                    QiskitRuntimeService.backends(
                        filters=lambda b: b.max_shots > 50000)
                    QiskitRuntimeService.backends(
                        filters=lambda x: ("rz" in x.basis_gates )

            **kwargs: Simple filters that require a specific value for an attribute in
                backend configuration or status.
                Examples::

                    # Get the operational real backends
                    QiskitRuntimeService.backends(simulator=False, operational=True)

                    # Get the backends with at least 127 qubits
                    QiskitRuntimeService.backends(min_num_qubits=127)

                    # Get the backends that support OpenPulse
                    QiskitRuntimeService.backends(open_pulse=True)

                For the full list of backend attributes, see the `IBMBackend` class documentation
                <https://docs.quantum.ibm.com/api/qiskit/providers_models>

        Returns:
            The list of available backends that match the filter.

        Raises:
            IBMInputValueError: If an input is invalid.
            QiskitBackendNotFoundError: If the backend is not in any instance.
        """
        backends: List[IBMBackend] = []
        instance_filter = instance if instance else self._account.instance
        if self._channel == "ibm_quantum":
            if name:
                if name not in self._backends:
                    raise QiskitBackendNotFoundError("No backend matches the criteria.")
                if not self._backends[name] or instance_filter != self._backends[name]._instance:
                    self._set_backend_config(name)
                    self._backends[name] = self._create_backend_obj(
                        self._backend_configs[name],
                        instance_filter,
                    )
                if self._backends[name]:
                    backends.append(self._backends[name])
            elif instance_filter:
                hgp = self._get_hgp(instance=instance_filter)
                for backend_name in hgp.backends:
                    if (
                        not self._backends[backend_name]
                        or instance_filter != self._backends[backend_name]._instance
                    ):
                        self._set_backend_config(backend_name, instance_filter)
                        self._backends[backend_name] = self._create_backend_obj(
                            self._backend_configs[backend_name], instance_filter
                        )
                    if self._backends[backend_name]:
                        backends.append(self._backends[backend_name])
            else:
                for backend_name, backend_config in self._backends.items():
                    if not backend_config:
                        self._set_backend_config(backend_name)
                        self._backends[backend_name] = self._create_backend_obj(
                            self._backend_configs[backend_name]
                        )
                    if self._backends[backend_name]:
                        backends.append(self._backends[backend_name])

        else:
            if instance:
                raise IBMInputValueError(
                    "The 'instance' keyword is only supported for ``ibm_quantum`` runtime."
                )
            backends = list(self._backends.values())

        if name:
            kwargs["backend_name"] = name
        if min_num_qubits:
            backends = list(
                filter(lambda b: b.configuration().n_qubits >= min_num_qubits, backends)
            )

        if dynamic_circuits is not None:
            backends = list(
                filter(
                    lambda b: ("qasm3" in getattr(b.configuration(), "supported_features", []))
                    == dynamic_circuits,
                    backends,
                )
            )
        return filter_backends(backends, filters=filters, **kwargs)

    def _set_backend_config(self, backend_name: str, instance: Optional[str] = None) -> None:
        """Retrieve backend configuration and add to backend_configs.
        Args:
            backend_name: backend name that will be returned.
            instance: the current h/g/p.
        """
        if backend_name not in self._backend_configs:
            raw_config = self._api_client.backend_configuration(backend_name)
            config = configuration_from_server_data(raw_config=raw_config, instance=instance)
            self._backend_configs[backend_name] = config

    def _create_backend_obj(
        self,
        config: Union[QasmBackendConfiguration, PulseBackendConfiguration],
        instance: Optional[str] = None,
    ) -> IBMBackend:
        """Given a backend configuration return the backend object.
        Args:
            config: backend configuration.
            instance: the current h/g/p.
        Returns:
            A backend object.
        Raises:
            QiskitBackendNotFoundError: if the backend is not in the hgp passed in.
        """
        if config:
            if not instance:
                for hgp in list(self._hgps.values()):
                    if config.backend_name in hgp.backends:
                        instance = to_instance_format(hgp._hub, hgp._group, hgp._project)
                        break

            elif config.backend_name not in self._get_hgp(instance=instance).backends:
                hgps_with_backend = []
                for hgp in list(self._hgps.values()):
                    if config.backend_name in hgp.backends:
                        hgps_with_backend.append(
                            to_instance_format(hgp._hub, hgp._group, hgp._project)
                        )
                raise QiskitBackendNotFoundError(
                    f"Backend {config.backend_name} is not in "
                    f"{instance}. Please try a different instance. "
                    f"{config.backend_name} is in the following instances you have access to: "
                    f"{hgps_with_backend}"
                )

            return ibm_backend.IBMBackend(
                instance=instance,
                configuration=config,
                service=self,
                api_client=self._api_client,
            )
        return None

    def active_account(self) -> Optional[Dict[str, str]]:
        """Return the IBM Quantum account currently in use for the session.

        Returns:
            A dictionary with information about the account currently in the session.
        """
        return self._account.to_saved_format()

    @staticmethod
    def delete_account(
        filename: Optional[str] = None,
        name: Optional[str] = None,
        channel: Optional[ChannelType] = None,
    ) -> bool:
        """Delete a saved account from disk.

        Args:
            filename: Name of file from which to delete the account.
            name: Name of the saved account to delete.
            channel: Channel type of the default account to delete.
                Ignored if account name is provided.

        Returns:
            True if the account was deleted.
            False if no account was found.
        """
        return AccountManager.delete(filename=filename, name=name, channel=channel)

    @staticmethod
    def save_account(
        token: Optional[str] = None,
        url: Optional[str] = None,
        instance: Optional[str] = None,
        channel: Optional[ChannelType] = None,
        filename: Optional[str] = None,
        name: Optional[str] = None,
        proxies: Optional[dict] = None,
        verify: Optional[bool] = None,
        overwrite: Optional[bool] = False,
        channel_strategy: Optional[str] = None,
        set_as_default: Optional[bool] = None,
    ) -> None:
        """Save the account to disk for future use.

        Args:
            token: IBM Cloud API key or IBM Quantum API token.
            url: The API URL.
                Defaults to https://cloud.ibm.com (ibm_cloud) or
                https://auth.quantum-computing.ibm.com/api (ibm_quantum).
            instance: The CRN (ibm_cloud) or hub/group/project (ibm_quantum).
            channel: Channel type. `ibm_cloud` or `ibm_quantum`.
            filename: Full path of the file where the account is saved.
            name: Name of the account to save.
            proxies: Proxy configuration. Supported optional keys are
                ``urls`` (a dictionary mapping protocol or protocol and host to the URL of the proxy,
                documented at https://docs.python-requests.org/en/latest/api/#requests.Session.proxies),
                ``username_ntlm``, ``password_ntlm`` (username and password to enable NTLM user
                authentication)
            verify: Verify the server's TLS certificate.
            overwrite: ``True`` if the existing account is to be overwritten.
            channel_strategy: Error mitigation strategy.
            set_as_default: If ``True``, the account is saved in filename,
                as the default account.
        """

        AccountManager.save(
            token=token,
            url=url,
            instance=instance,
            channel=channel,
            filename=filename,
            name=name,
            proxies=ProxyConfiguration(**proxies) if proxies else None,
            verify=verify,
            overwrite=overwrite,
            channel_strategy=channel_strategy,
            set_as_default=set_as_default,
        )

    @staticmethod
    def saved_accounts(
        default: Optional[bool] = None,
        channel: Optional[ChannelType] = None,
        filename: Optional[str] = None,
        name: Optional[str] = None,
    ) -> dict:
        """List the accounts saved on disk.

        Args:
            default: If set to True, only default accounts are returned.
            channel: Channel type. `ibm_cloud` or `ibm_quantum`.
            filename: Name of file whose accounts are returned.
            name: If set, only accounts with the given name are returned.

        Returns:
            A dictionary with information about the accounts saved on disk.

        Raises:
            ValueError: If an invalid account is found on disk.
        """
        return dict(
            map(
                lambda kv: (kv[0], Account.to_saved_format(kv[1])),
                AccountManager.list(
                    default=default, channel=channel, filename=filename, name=name
                ).items(),
            ),
        )

    def backend(
        self,
        name: str = None,
        instance: Optional[str] = None,
    ) -> Backend:
        """Return a single backend matching the specified filtering.

        Args:
            name: Name of the backend.
            instance: This is only supported for ``ibm_quantum`` runtime and is in the
                hub/group/project format. If an instance is not given, among the providers
                with access to the backend, a premium provider will be prioritized.
                For users without access to a premium provider, the default open provider will be used.

        Returns:
            Backend: A backend matching the filtering.

        Raises:
            QiskitBackendNotFoundError: if no backend could be found.
        """
        # pylint: disable=arguments-differ, line-too-long
        backends = self.backends(name, instance=instance)
        if not backends:
            cloud_msg_url = ""
            if self._channel == "ibm_cloud":
                cloud_msg_url = (
                    " Learn more about available backends here "
                    "https://cloud.ibm.com/docs/quantum-computing?topic=quantum-computing-choose-backend "
                )
            raise QiskitBackendNotFoundError("No backend matches the criteria." + cloud_msg_url)
        return backends[0]

    def get_backend(self, name: str = None, **kwargs: Any) -> Backend:
        return self.backend(name, **kwargs)

    def run(
        self,
        program_id: str,
        inputs: Dict,
        options: Optional[Union[RuntimeOptions, Dict]] = None,
        callback: Optional[Callable] = None,
        result_decoder: Optional[Union[Type[ResultDecoder], Sequence[Type[ResultDecoder]]]] = None,
        session_id: Optional[str] = None,
        start_session: Optional[bool] = False,
    ) -> Union[RuntimeJob, RuntimeJobV2]:
        """Execute the runtime program.

        Args:
            program_id: Program ID.
            inputs: Program input parameters. These input values are passed
                to the runtime program.
            options: Runtime options that control the execution environment.
                See :class:`RuntimeOptions` for all available options.

            callback: Callback function to be invoked for any interim results and final result.
                The callback function will receive 2 positional parameters:

                    1. Job ID
                    2. Job result.

            result_decoder: A :class:`ResultDecoder` subclass used to decode job results.
                If more than one decoder is specified, the first is used for interim results and
                the second final results. If not specified, a program-specific decoder or the default
                ``ResultDecoder`` is used.
            session_id: Job ID of the first job in a runtime session.
            start_session: Set to True to explicitly start a runtime session. Defaults to False.

        Returns:
            A ``RuntimeJob`` instance representing the execution.

        Raises:
            IBMInputValueError: If input is invalid.
            RuntimeProgramNotFound: If the program cannot be found.
            IBMRuntimeError: An error occurred running the program.
        """
        qrt_options: RuntimeOptions = options
        if options is None:
            qrt_options = RuntimeOptions()
        elif isinstance(options, Dict):
            qrt_options = RuntimeOptions(**options)

        qrt_options.validate(channel=self.channel)

        hgp_name = None
        if self._channel == "ibm_quantum":
            # Find the right hgp
            hgp = self._get_hgp(
                instance=qrt_options.instance, backend_name=qrt_options.get_backend_name()
            )
            hgp_name = hgp.name
            if hgp_name != self._current_instance:
                self._current_instance = hgp_name
                logger.info("Instance selected: %s", self._current_instance)
        backend = self.backend(name=qrt_options.get_backend_name(), instance=hgp_name)
        status = backend.status()
        if status.operational is True and status.status_msg != "active":
            warnings.warn(
                f"The backend {backend.name} currently has a status of {status.status_msg}."
            )

        version = inputs.get("version", 1) if inputs else 1
        try:
            response = self._api_client.program_run(
                program_id=program_id,
                backend_name=qrt_options.get_backend_name(),
                params=inputs,
                image=qrt_options.image,
                hgp=hgp_name,
                log_level=qrt_options.log_level,
                session_id=session_id,
                job_tags=qrt_options.job_tags,
                max_execution_time=qrt_options.max_execution_time,
                start_session=start_session,
                session_time=qrt_options.session_time,
                channel_strategy=None
                if self._channel_strategy == "default"
                else self._channel_strategy,
            )
            if self._channel == "ibm_quantum":
                messages = response.get("messages")
                if messages:
                    warning_message = messages[0].get("data")
                    warnings.warn(warning_message)

        except RequestsApiError as ex:
            if ex.status_code == 404:
                raise RuntimeProgramNotFound(f"Program not found: {ex.message}") from None
            raise IBMRuntimeError(f"Failed to run program: {ex}") from None
        backend = (
            self.backend(name=response["backend"], instance=hgp_name)
            if response["backend"]
            else qrt_options.get_backend_name()
        )

        if version == 2:
            job = RuntimeJobV2(
                backend=backend,
                api_client=self._api_client,
                client_params=self._client_params,
                job_id=response["id"],
                program_id=program_id,
                user_callback=callback,
                result_decoder=result_decoder,
                image=qrt_options.image,
                service=self,
                version=version,
            )
        else:
            job = RuntimeJob(
                backend=backend,
                api_client=self._api_client,
                client_params=self._client_params,
                job_id=response["id"],
                program_id=program_id,
                user_callback=callback,
                result_decoder=result_decoder,
                image=qrt_options.image,
                service=self,
                version=version,
            )
        return job

    def job(self, job_id: str) -> Union[RuntimeJob, RuntimeJobV2]:
        """Retrieve a runtime job.

        Args:
            job_id: Job ID.

        Returns:
            Runtime job retrieved.

        Raises:
            RuntimeJobNotFound: If the job doesn't exist.
            IBMRuntimeError: If the request failed.
        """
        try:
            response = self._api_client.job_get(job_id, exclude_params=True)
        except RequestsApiError as ex:
            if ex.status_code == 404:
                raise RuntimeJobNotFound(f"Job not found: {ex.message}") from None
            raise IBMRuntimeError(f"Failed to delete job: {ex}") from None
        return self._decode_job(response)

    def jobs(
        self,
        limit: Optional[int] = 10,
        skip: int = 0,
        backend_name: Optional[str] = None,
        pending: bool = None,
        program_id: str = None,
        instance: Optional[str] = None,
        job_tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        descending: bool = True,
    ) -> List[Union[RuntimeJob, RuntimeJobV2]]:
        """Retrieve all runtime jobs, subject to optional filtering.

        Args:
            limit: Number of jobs to retrieve. ``None`` means no limit.
            skip: Starting index for the job retrieval.
            backend_name: Name of the backend to retrieve jobs from.
            pending: Filter by job pending state. If ``True``, 'QUEUED' and 'RUNNING'
                jobs are included. If ``False``, 'DONE', 'CANCELLED' and 'ERROR' jobs
                are included.
            program_id: Filter by Program ID.
            instance: This is only supported for ``ibm_quantum`` runtime and is in the
                hub/group/project format.
            job_tags: Filter by tags assigned to jobs. Matched jobs are associated with all tags.
            session_id: Job ID of the first job in a runtime session.
            created_after: Filter by the given start date, in local time. This is used to
                find jobs whose creation dates are after (greater than or equal to) this
                local date/time.
            created_before: Filter by the given end date, in local time. This is used to
                find jobs whose creation dates are before (less than or equal to) this
                local date/time.
            descending: If ``True``, return the jobs in descending order of the job
                creation date (i.e. newest first) until the limit is reached.

        Returns:
            A list of runtime jobs.

        Raises:
            IBMInputValueError: If an input value is invalid.
        """
        hub = group = project = None
        if instance:
            if self._channel == "ibm_cloud":
                raise IBMInputValueError(
                    "The 'instance' keyword is only supported for ``ibm_quantum`` runtime."
                )
            hub, group, project = from_instance_format(instance)
        if job_tags:
            validate_job_tags(job_tags)

        job_responses = []  # type: List[Dict[str, Any]]
        current_page_limit = limit or 20
        offset = skip

        while True:
            jobs_response = self._api_client.jobs_get(
                limit=current_page_limit,
                skip=offset,
                backend_name=backend_name,
                pending=pending,
                program_id=program_id,
                hub=hub,
                group=group,
                project=project,
                job_tags=job_tags,
                session_id=session_id,
                created_after=created_after,
                created_before=created_before,
                descending=descending,
            )
            job_page = jobs_response["jobs"]
            # count is the total number of jobs that would be returned if
            # there was no limit or skip
            count = jobs_response["count"]

            job_responses += job_page

            if len(job_responses) == count - skip:
                # Stop if there are no more jobs returned by the server.
                break

            if limit:
                if len(job_responses) >= limit:
                    # Stop if we have reached the limit.
                    break
                current_page_limit = limit - len(job_responses)
            else:
                current_page_limit = 20

            offset += len(job_page)

        return [self._decode_job(job) for job in job_responses]

    def delete_job(self, job_id: str) -> None:
        """Delete a runtime job.

        Note that this operation cannot be reversed.

        Args:
            job_id: ID of the job to delete.

        Raises:
            RuntimeJobNotFound: If the job doesn't exist.
            IBMRuntimeError: If the request failed.
        """
        try:
            self._api_client.job_delete(job_id)
        except RequestsApiError as ex:
            if ex.status_code == 404:
                raise RuntimeJobNotFound(f"Job not found: {ex.message}") from None
            raise IBMRuntimeError(f"Failed to delete job: {ex}") from None

    def _decode_job(self, raw_data: Dict) -> Union[RuntimeJob, RuntimeJobV2]:
        """Decode job data received from the server.

        Args:
            raw_data: Raw job data received from the server.

        Returns:
            Decoded job data.
        """
        instance = None
        if self._channel == "ibm_quantum":
            hub = raw_data.get("hub")
            group = raw_data.get("group")
            project = raw_data.get("project")
            if all([hub, group, project]):
                instance = to_instance_format(hub, group, project)
        # Try to find the right backend
        try:
            if "backend" in raw_data:
                backend = self.backend(raw_data["backend"], instance=instance)
            else:
                backend = None
        except QiskitBackendNotFoundError:
            backend = ibm_backend.IBMRetiredBackend.from_name(
                backend_name=raw_data["backend"],
                api=None,
            )

        params = raw_data.get("params", {})
        if isinstance(params, list):
            if len(params) > 0:
                params = params[0]
            else:
                params = {}
        if not isinstance(params, str):
            params = json.dumps(params)

        decoded = json.loads(params, cls=RuntimeDecoder)
        return RuntimeJob(
            backend=backend,
            api_client=self._api_client,
            client_params=self._client_params,
            service=self,
            job_id=raw_data["id"],
            program_id=raw_data.get("program", {}).get("id", ""),
            params=decoded,
            creation_date=raw_data.get("created", None),
            session_id=raw_data.get("session_id"),
            tags=raw_data.get("tags"),
        )

    def least_busy(
        self,
        min_num_qubits: Optional[int] = None,
        instance: Optional[str] = None,
        filters: Optional[Callable[[List["ibm_backend.IBMBackend"]], bool]] = None,
        **kwargs: Any,
    ) -> ibm_backend.IBMBackend:
        """Return the least busy available backend.

        Args:
            min_num_qubits: Minimum number of qubits the backend has to have.
            instance: This is only supported for ``ibm_quantum`` runtime and is in the
                hub/group/project format.
            filters: Filters can be defined as for the :meth:`backends` method.
                An example to get the operational backends with 5 qubits::

                    QiskitRuntimeService.least_busy(n_qubits=5, operational=True)

        Returns:
            The backend with the fewest number of pending jobs.

        Raises:
            QiskitBackendNotFoundError: If no backend matches the criteria.
        """
        backends = self.backends(
            min_num_qubits=min_num_qubits, instance=instance, filters=filters, **kwargs
        )
        candidates = []
        for back in backends:
            backend_status = back.status()
            if not backend_status.operational or backend_status.status_msg != "active":
                continue
            candidates.append(back)
        if not candidates:
            raise QiskitBackendNotFoundError("No backend matches the criteria.")
        return min(candidates, key=lambda b: b.status().pending_jobs)

    def instances(self) -> List[str]:
        """Return the IBM Quantum instances list currently in use for the session.

        Returns:
            A list with instances currently in the session.
        """
        if self._channel == "ibm_quantum":
            return list(self._hgps.keys())
        return []

    @property
    def channel(self) -> str:
        """Return the channel type used.

        Returns:
            The channel type used.
        """
        return self._channel

    def __repr__(self) -> str:
        return "<{}>".format(self.__class__.__name__)
