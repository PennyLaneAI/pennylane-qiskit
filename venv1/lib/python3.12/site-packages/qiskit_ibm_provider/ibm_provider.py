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

"""Provider for a single IBM Quantum account."""

import logging
import traceback
import warnings
from datetime import datetime
from collections import OrderedDict
from typing import Dict, List, Optional, Any, Callable, Union
from typing_extensions import Literal

from qiskit.providers import ProviderV1 as Provider  # type: ignore[attr-defined]
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.providers.exceptions import QiskitBackendNotFoundError

from .accounts import AccountManager, Account
from .api.client_parameters import ClientParameters
from .api.clients import AuthClient, VersionClient, RuntimeClient
from .apiconstants import QISKIT_IBM_API_URL
from .exceptions import IBMAccountError
from .exceptions import (
    IBMInputValueError,
)

from .hub_group_project import HubGroupProject  # pylint: disable=cyclic-import
from .ibm_backend import IBMBackend  # pylint: disable=cyclic-import
from .ibm_backend_service import IBMBackendService  # pylint: disable=cyclic-import
from .job import IBMJob  # pylint: disable=cyclic-import
from .proxies.configuration import ProxyConfiguration
from .utils.hgp import to_instance_format, from_instance_format

logger = logging.getLogger(__name__)


class IBMProvider(Provider):
    """Provides access to the IBM Quantum services available to an account.

    Authenticate against IBM Quantum for use from saved credentials or during session.

    Credentials can be saved to disk by calling the `save_account()` method::

        from qiskit_ibm_provider import IBMProvider
        IBMProvider.save_account(token=<INSERT_IBM_QUANTUM_TOKEN>)

    You can set the default project using the `hub`, `group`, and `project` keywords
    in `save_account()`. Once credentials are saved you can simply instantiate the
    provider like below to load the saved account and default project::

        from qiskit_ibm_provider import IBMProvider
        provider = IBMProvider()

    Instead of saving credentials to disk, you can also set the environment
    variables QISKIT_IBM_TOKEN, QISKIT_IBM_URL and QISKIT_IBM_INSTANCE
    and then instantiate the provider as below::

        from qiskit_ibm_provider import IBMProvider
        provider = IBMProvider()

    You can also enable an account just for the current session by instantiating
    the provider with the API token::

        from qiskit_ibm_provider import IBMProvider
        provider = IBMProvider(token=<INSERT_IBM_QUANTUM_TOKEN>)

    `token` is the only required attribute that needs to be set using one of the above methods.
    If no `url` is set, it defaults to 'https://auth.quantum-computing.ibm.com/api'.

    Note:
        The hub/group/project is selected based on the below selection order,
        in decreasing order of priority.

        * The hub/group/project you explicity specify when calling a service.
          Ex: `provider.get_backend()`, etc.
        * The hub/group/project required for the service.
        * The default hub/group/project you set using `save_account()`.
        * A premium hub/group/project in your account.
        * An open access hub/group/project.

    The IBMProvider offers the :class:`~qiskit_ibm_provider.ibm_backend_service.IBMBackendService`
    which gives access to the IBM Quantum devices and simulators.

    You can obtain an instance of the service as an attribute of the ``IBMProvider`` instance.
    For example::

        backend_service = provider.backend

    Since :class:`~qiskit_ibm_provider.ibm_backend_service.IBMBackendService`
    is the main service, some of the backend-related methods are available
    through this class for convenience.

    The :meth:`backends()` method returns all the backends available to this account::

        backends = provider.backends()

    The :meth:`get_backend()` method returns a backend that matches the filters
    passed as argument. An example of retrieving a backend that matches a
    specified name::

        simulator_backend = provider.get_backend('ibmq_qasm_simulator')

    IBMBackend's are uniquely identified by their name. If you invoke :meth:`get_backend()` twice,
    you will get the same IBMBackend instance, and any previously updated options will be reset
    to the default values.

    It is also possible to use the ``backend`` attribute to reference a backend.
    As an example, to retrieve the same backend from the example above::

        simulator_backend = provider.backend.ibmq_qasm_simulator

    Note:
        The ``backend`` attribute can be used to autocomplete the names of
        backends available to this account. To autocomplete, press ``tab``
        after ``provider.backend.``. This feature may not be available
        if an error occurs during backend discovery. Also note that
        this feature is only available in interactive sessions, such as
        in Jupyter Notebook and the Python interpreter.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        url: Optional[str] = None,
        name: Optional[str] = None,
        instance: Optional[str] = None,
        proxies: Optional[dict] = None,
        verify: Optional[bool] = None,
    ) -> None:
        """IBMProvider constructor

        Args:
            token: IBM Quantum API token.
            url: The API URL.
                Defaults to https://auth.quantum-computing.ibm.com/api.
            name: Name of the account to load.
            instance: Provider in the hub/group/project format.
            proxies: Proxy configuration. Supported optional keys are
                ``urls`` (a dictionary mapping protocol or protocol and host to the URL of the proxy,
                documented at https://docs.python-requests.org/en/latest/api/#requests.Session.proxies),
                ``username_ntlm``, ``password_ntlm`` (username and password to enable NTLM user
                authentication)
            verify: Whether to verify the server's TLS certificate.

        Returns:
            An instance of IBMProvider

        Raises:
            IBMInputValueError: If an input is invalid.

        """
        super().__init__()
        self._account = self._discover_account(
            token=token,
            url=url,
            instance=instance,
            name=name,
            proxies=ProxyConfiguration(**proxies) if proxies else None,
            verify=verify,
        )

        self._client_params = ClientParameters(
            token=self._account.token,
            url=self._account.url,
            instance=self._account.instance,
            proxies=self._account.proxies,
            verify=self._account.verify,
        )
        self._auth_client = self._authenticate_ibm_quantum_account(self._client_params)
        self._client_params.url = self._auth_client.current_service_urls()["services"][
            "runtime"
        ]
        self._client_params.token = self._auth_client.current_access_token()
        self._runtime_client = RuntimeClient(self._client_params)

        self._hgps = self._initialize_hgps(self._auth_client)
        self._initialize_services()

    @staticmethod
    def _discover_account(
        token: Optional[str] = None,
        url: Optional[str] = None,
        instance: Optional[str] = None,
        name: Optional[str] = None,
        proxies: Optional[ProxyConfiguration] = None,
        verify: Optional[bool] = None,
    ) -> Account:
        """Discover account."""
        verify_ = verify or True
        if name:
            if any([token, url]):
                logger.warning(
                    "Loading account with name %s. Any input 'token', 'url' are ignored.",
                    name,
                )
            account = AccountManager.get(name=name)
        else:
            if token:
                account = Account(
                    channel="ibm_quantum",
                    token=token,
                    url=url,
                    instance=instance,
                    proxies=proxies,
                    verify=verify_,
                )
            else:
                if url:
                    logger.warning(
                        "Loading default ibm_quantum account. Input 'url' is ignored."
                    )
                account = AccountManager.get(channel="ibm_quantum")

        if account is None:
            account = AccountManager.get()

        if instance:
            account.instance = instance
        if proxies:
            account.proxies = proxies
        if verify is not None:
            account.verify = verify

        # ensure account is valid, fail early if not
        account.validate()

        return account

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
                token=auth_client.current_access_token(),
                url=service_urls["http"],
                instance=to_instance_format(
                    hub_info["hub"], hub_info["group"], hub_info["project"]
                ),
                proxies=self._account.proxies,
                verify=self._account.verify,
            )

            # Build the hgp.
            try:
                hgp = HubGroupProject(
                    client_params=hgp_params,
                    instance=hgp_params.instance,
                    provider=self,
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
                warnings.warn(
                    f"Default hub/group/project {default_hgp} not "
                    "found for the account and is ignored."
                )
        return hgps

    def _authenticate_ibm_quantum_account(
        self, client_params: ClientParameters
    ) -> AuthClient:
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
                    client_params.url, QISKIT_IBM_API_URL
                )
            )
        return AuthClient(client_params)

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
        backend_name: Optional[str] = None,
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
                    f"Hub/group/project {instance} "
                    "could not be found for this account."
                )
            if backend_name and not self._hgps[instance].backend(backend_name):
                raise QiskitBackendNotFoundError(
                    f"Backend {backend_name} cannot be found in "
                    f"hub/group/project {instance}"
                )
            return self._hgps[instance]

        if not backend_name:
            return list(self._hgps.values())[0]

        for hgp in self._hgps.values():
            if hgp.backend(backend_name):
                return hgp

        raise QiskitBackendNotFoundError(
            f"Backend {backend_name} cannot be found in any"
            f"hub/group/project for this account."
        )

    def _get_hgps(
        self,
    ) -> List[HubGroupProject]:
        """Return a list of `HubGroupProject` instances.

        Returns:
            A list of `HubGroupProject` instancess.
        """
        hgps = [hgp for key, hgp in self._hgps.items()]
        return hgps

    def _initialize_services(self) -> None:
        """Initialize all services."""
        self._backend = None
        hgps = self._get_hgps()
        for hgp in hgps:
            # Initialize backend service
            if not self._backend:
                self._backend = IBMBackendService(self, hgp)
            if self._backend:
                break
        self._services = {"backend": self._backend}

    @property
    def backend(self) -> IBMBackendService:
        """Return the backend service.

        Returns:
            The backend service instance.
        """
        return self._backend

    def active_account(self) -> Optional[Dict[str, str]]:
        """Return the IBM Quantum account currently in use for the session.

        Returns:
            A dictionary with information about the account currently in the session.
        """
        hgps = self._get_hgps()
        active_account_dict = self._account.to_saved_format()
        active_account_dict.update({"instance": hgps[0].name})
        return active_account_dict

    def instances(self) -> List[str]:
        """Return the IBM Quantum instances list currently in use for the session.

        Returns:
            A list with instances currently in the session.
        """
        return [hgp.name for hgp in self._get_hgps()]

    @staticmethod
    def delete_account(name: Optional[str] = None) -> bool:
        """Delete a saved account from disk.

        Args:
            name: Name of the saved account to delete.

        Returns:
            True if the account was deleted.
            False if no account was found.
        """
        return AccountManager.delete(name=name)

    @staticmethod
    def save_account(
        token: Optional[str] = None,
        url: Optional[str] = None,
        instance: Optional[str] = None,
        name: Optional[str] = None,
        proxies: Optional[dict] = None,
        verify: Optional[bool] = None,
        overwrite: Optional[bool] = False,
    ) -> None:
        """Save the account to disk for future use.

        Args:
            token: IBM Quantum API token.
            url: The API URL.
                Defaults to https://auth.quantum-computing.ibm.com/api
            instance: The hub/group/project.
            name: Name of the account to save.
            proxies: Proxy configuration. Supported optional keys are
                ``urls`` (a dictionary mapping protocol or protocol and host to the URL of the proxy,
                documented at https://docs.python-requests.org/en/latest/api/#requests.Session.proxies),
                ``username_ntlm``, ``password_ntlm`` (username and password to enable NTLM user
                authentication)
            verify: Verify the server's TLS certificate.
            overwrite: ``True`` if the existing account is to be overwritten.
        """

        AccountManager.save(
            token=token,
            url=url,
            instance=instance,
            channel="ibm_quantum",
            name=name,
            proxies=ProxyConfiguration(**proxies) if proxies else None,
            verify=verify,
            overwrite=overwrite,
        )

    @staticmethod
    def saved_accounts(
        default: Optional[bool] = None,
        name: Optional[str] = None,
    ) -> dict:
        """List the accounts saved on disk.

        Args:
            default: If set to True, only default accounts are returned.
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
                    default=default, channel="ibm_quantum", name=name
                ).items(),
            ),
        )

    def backends(
        self,
        name: Optional[str] = None,
        filters: Optional[Callable[[List[IBMBackend]], bool]] = None,
        min_num_qubits: Optional[int] = None,
        instance: Optional[str] = None,
        **kwargs: Any,
    ) -> List[IBMBackend]:
        """Return all backends accessible via this account, subject to optional filtering.

        Args:
            name: Backend name to filter by.
            min_num_qubits: Minimum number of qubits the backend must have.
            instance: The provider in the hub/group/project format.
            filters: More complex filters, such as lambda functions.
                For example::

                    IBMProvider.backends(filters=lambda b: b.max_shots > 50000)
                    IBMProvider.backends(filters=lambda x: ("rz" in x.basis_gates )

            **kwargs: Simple filters that require a specific value for an attribute in
                backend configuration, backend status, or provider credentials.
                Examples::

                    # Get the operational real backends
                    IBMProvider.backends(simulator=False, operational=True)
                    # Get the backends with at least 127 qubits
                    IBMProvider.backends(min_num_qubits=127)
                    # Get the backends that support OpenPulse
                    IBMProvider.backends(open_pulse=True)

                For the full list of backend attributes, see the `IBMBackend` class documentation
                <https://docs.quantum.ibm.com/api/qiskit/providers_models>

        Returns:
            The list of available backends that match the filter.
        """
        # pylint: disable=arguments-differ
        return self._backend.backends(
            name=name,
            filters=filters,
            min_num_qubits=min_num_qubits,
            instance=instance or self._account.instance,
            **kwargs,
        )

    def jobs(
        self,
        limit: Optional[int] = 10,
        skip: int = 0,
        backend_name: Optional[str] = None,
        status: Optional[Literal["pending", "completed"]] = None,
        start_datetime: Optional[datetime] = None,
        end_datetime: Optional[datetime] = None,
        job_tags: Optional[List[str]] = None,
        descending: bool = True,
        instance: Optional[str] = None,
        legacy: bool = False,
    ) -> List[IBMJob]:
        """Return a list of jobs, subject to optional filtering.

        Retrieve jobs that match the given filters and paginate the results
        if desired. Note that the server has a limit for the number of jobs
        returned in a single call. As a result, this function might involve
        making several calls to the server.

        Args:
            limit: Number of jobs to retrieve. ``None`` means no limit. Note that the
                number of sub-jobs within a composite job count towards the limit.
            skip: Starting index for the job retrieval.
            backend_name: Name of the backend to retrieve jobs from.
            status: Filter jobs with either "pending" or "completed" status.
            start_datetime: Filter by the given start date, in local time. This is used to
                find jobs whose creation dates are after (greater than or equal to) this
                local date/time.
            end_datetime: Filter by the given end date, in local time. This is used to
                find jobs whose creation dates are before (less than or equal to) this
                local date/time.
            job_tags: Filter by tags assigned to jobs. Matched jobs are associated with all tags.
            descending: If ``True``, return the jobs in descending order of the job
                creation date (i.e. newest first) until the limit is reached.
            instance: The provider in the hub/group/project format.
            legacy: If ``True``, only retrieve jobs run from the deprecated ``qiskit-ibmq-provider``.
            Otherwise, only retrieve jobs run from ``qiskit-ibm-provider``.

        Returns:
            A list of ``IBMJob`` instances.

        """

        return self._backend.jobs(
            limit=limit,
            skip=skip,
            backend_name=backend_name,
            status=status,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            job_tags=job_tags,
            descending=descending,
            instance=instance or self._account.instance,
            legacy=legacy,
        )

    def retrieve_job(self, job_id: str) -> IBMJob:
        """Return a single job.

        Args:
            job_id: The ID of the job to retrieve.

        Returns:
            The job with the given id.
        """
        return self._backend.retrieve_job(job_id=job_id)

    def get_backend(
        self,
        name: str = None,
        instance: Optional[str] = None,
        **kwargs: Any,
    ) -> Backend:
        """Return a single backend matching the specified filtering.

        Args:
            name (str): Name of the backend.
            instance: The provider in the hub/group/project format.
            **kwargs: Dict used for filtering.

        Returns:
            Backend: a backend matching the filtering.

        Raises:
            QiskitBackendNotFoundError: if no backend could be found or
                more than one backend matches the filtering criteria.
            IBMProviderValueError: If only one or two parameters from `hub`, `group`,
                `project` are specified.
        """
        # pylint: disable=arguments-differ
        backends = self.backends(name, instance=instance, **kwargs)
        if len(backends) > 1:
            raise QiskitBackendNotFoundError(
                "More than one backend matches the criteria"
            )
        if not backends:
            raise QiskitBackendNotFoundError("No backend matches the criteria")
        backends[0]._options = IBMBackend._default_options()
        return backends[0]

    def __repr__(self) -> str:
        return "<{}>".format(self.__class__.__name__)
