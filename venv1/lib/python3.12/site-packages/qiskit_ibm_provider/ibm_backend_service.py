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

"""Backend namespace for an IBM Quantum account."""

import logging
from datetime import datetime
from typing import Dict, List, Callable, Optional, Any, Union
from typing_extensions import Literal

from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.providers.jobstatus import JobStatus
from qiskit.providers.providerutils import filter_backends
from qiskit.providers.models import (
    PulseBackendConfiguration,
    QasmBackendConfiguration,
)

# pylint: disable=unused-import
from qiskit_ibm_provider import (
    ibm_backend,
    ibm_provider,
)
from .api.exceptions import ApiError
from .api.clients import AccountClient
from .apiconstants import ApiJobStatus
from .exceptions import (
    IBMBackendValueError,
    IBMBackendApiError,
    IBMBackendApiProtocolError,
    IBMInputValueError,
)
from .hub_group_project import HubGroupProject
from .ibm_backend import IBMBackend, IBMRetiredBackend
from .job import IBMJob, IBMCircuitJob
from .job.exceptions import IBMJobNotFoundError
from .utils.hgp import from_instance_format
from .utils.backend_decoder import configuration_from_server_data
from .utils.converters import local_to_utc
from .utils.utils import (
    to_python_identifier,
    validate_job_tags,
    filter_data,
    api_status_to_job_status,
)
from .utils.hgp import to_instance_format

logger = logging.getLogger(__name__)
PAGE_SIZE = 50


class IBMBackendService:
    """Backend namespace for an IBM Quantum account.

    Represent a namespace that provides backend related services for the IBM
    Quantum backends available to this account. An instance of
    this class is used as a callable attribute to the :class:`IBMProvider`
    class. This allows a convenient way to query for all backends or to access
    a specific backend::

        backends = provider.backends()  # Invoke backends() to get the backends.
        sim_backend = provider.backend.ibmq_qasm_simulator  # Get a specific backend instance.

    Also, you are able to retrieve jobs from an account without specifying the backend name.
    For example, to retrieve the ten most recent jobs you have submitted, regardless of the
    backend they were submitted to, you could do::

        most_recent_jobs = provider.backend.jobs(limit=10)

    It is also possible to retrieve a single job without specifying the backend name::

        job = provider.backend.retrieve_job(<JOB_ID>)
    """

    def __init__(
        self, provider: "ibm_provider.IBMProvider", hgp: HubGroupProject
    ) -> None:
        """IBMBackendService constructor.

        Args:
            provider: IBM Quantum account provider.
            hgp: default hub/group/project to use for the service.
        """
        super().__init__()
        self._provider = provider
        self._default_hgp = hgp
        self._backends: Dict[str, IBMBackend] = {}
        self._backend_configs: Dict[str, Any] = {}
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        """Initialize the internal list of backends."""
        # Add backends from user selected hgp followed by backends
        # from other hgps if not already added
        for hgp in self._provider._get_hgps():
            for name in hgp.backends:
                if name not in self._backends:
                    self._backends[name] = None

    def _discover_backends(self) -> None:
        """Discovers the remote backends for this account, if not already known."""
        for backend in self._backends.values():
            backend_name = to_python_identifier(backend.name)
            # Append _ if duplicate
            while backend_name in self.__dict__:
                backend_name += "_"
            setattr(self, backend_name, backend)

    def backends(
        self,
        name: Optional[str] = None,
        filters: Optional[Callable[[List[IBMBackend]], bool]] = None,
        min_num_qubits: Optional[int] = None,
        instance: Optional[str] = None,
        dynamic_circuits: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[IBMBackend]:
        """Return all backends accessible via this account, subject to optional filtering.

        Args:
            name: Backend name to filter by.
            min_num_qubits: Minimum number of qubits the backend must have.
            instance: The provider in the hub/group/project format.
            dynamic_circuits: Filter by whether the backend supports dynamic circuits.
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

        Raises:
            IBMBackendValueError: If only one or two parameters from `hub`, `group`,
                `project` are specified.
            QiskitBackendNotFoundError: If the backend is not found in any instance.
        """
        backends: List[IBMBackend] = []
        if name:
            if name not in self._backends:
                raise QiskitBackendNotFoundError("No backend matches the criteria")
            if not self._backends[name] or instance != self._backends[name]._instance:
                self._set_backend_config(name)
                self._backends[name] = self._create_backend_obj(
                    self._backend_configs[name], instance, self._provider._get_hgps()
                )
            if self._backends[name]:
                backends.append(self._backends[name])
        elif instance:
            hgp = self._provider._get_hgp(instance=instance)
            for backend_name in hgp.backends.keys():
                if (
                    not self._backends[backend_name]
                    or instance != self._backends[backend_name]._instance
                ):
                    self._set_backend_config(backend_name, instance)
                    self._backends[backend_name] = self._create_backend_obj(
                        self._backend_configs[backend_name], instance
                    )
                if self._backends[backend_name]:
                    backends.append(self._backends[backend_name])
        else:
            hgps = self._provider._get_hgps()
            for backend_name, backend_config in self._backends.items():
                if not backend_config:
                    self._set_backend_config(backend_name)
                    self._backends[backend_name] = self._create_backend_obj(
                        self._backend_configs[backend_name], hgps=hgps
                    )
                if self._backends[backend_name]:
                    backends.append(self._backends[backend_name])
        # Special handling of the `name` parameter, to support alias resolution.
        if name:
            aliases = self._aliased_backend_names()
            aliases.update(self._deprecated_backend_names())
            name = aliases.get(name, name)
            kwargs["backend_name"] = name
        if min_num_qubits:
            backends = list(
                filter(lambda b: b.configuration().n_qubits >= min_num_qubits, backends)
            )
        if dynamic_circuits is not None:
            backends = list(
                filter(
                    lambda b: (
                        "qasm3" in getattr(b.configuration(), "supported_features", [])
                    )
                    == dynamic_circuits,
                    backends,
                )
            )

        return filter_backends(backends, filters=filters, **kwargs)

    def jobs(
        self,
        limit: Optional[int] = 10,
        skip: int = 0,
        backend_name: Optional[str] = None,
        status: Optional[
            Union[
                Literal["pending", "completed"],
                List[Union[JobStatus, str]],
                JobStatus,
                str,
            ]
        ] = None,
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
            status: Filter jobs with either "pending" or "completed" status. You can also specify by
            exact status. For example, `status=JobStatus.RUNNING` or `status="RUNNING"`
                or `status=["RUNNING", "ERROR"]`.
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
            legacy: If ``True``, only retrieve jobs run from the archived ``qiskit-ibmq-provider``.
            Otherwise, only retrieve jobs run from ``qiskit-ibm-provider``.

        Returns:
            A list of ``IBMJob`` instances.

        Raises:
            IBMBackendValueError: If a keyword value is not recognized.
            TypeError: If the input `start_datetime` or `end_datetime` parameter value
                is not valid.
        """
        # Build the filter for the query.

        api_filter = {}  # type: Dict[str, Any]
        all_job_statuses = [status.name for status in JobStatus]
        if isinstance(status, JobStatus):
            status = status.name
        if isinstance(status, str):
            status = status.upper()
        if isinstance(status, list):
            status = [x.name if isinstance(x, JobStatus) else x.upper() for x in status]
            if status in (["INITIALIZING"], ["VALIDATING"]):
                return []
            elif all(x in ["DONE", "CANCELLED", "ERROR"] for x in status):
                api_filter["pending"] = False
            elif all(x in ["QUEUED", "RUNNING"] for x in status):
                api_filter["pending"] = True
        elif status in all_job_statuses:
            if status in ["INITIALIZING", "VALIDATING"]:
                return []
            elif status in ["DONE", "CANCELLED", "ERROR"]:
                api_filter["pending"] = False
            elif status in ["QUEUED", "RUNNING"]:
                api_filter["pending"] = True
        if backend_name:
            api_filter["backend"] = backend_name
        if status == "PENDING":
            api_filter["pending"] = True
        if status == "COMPLETED":
            api_filter["pending"] = False
        if start_datetime:
            api_filter["created_after"] = local_to_utc(start_datetime).isoformat()
        if end_datetime:
            api_filter["created_before"] = local_to_utc(end_datetime).isoformat()
        if job_tags:
            validate_job_tags(job_tags, IBMBackendValueError)
            api_filter["job_tags"] = job_tags
        if instance:
            hub, group, project = from_instance_format(instance)
            api_filter["hub"] = hub
            api_filter["group"] = group
            api_filter["project"] = project
        # Retrieve all requested jobs.
        filter_by_status = (
            status
            and status not in ["PENDING", "COMPLETED"]
            and (
                status in all_job_statuses or all(x in all_job_statuses for x in status)
            )
        )
        job_list = []
        while True:
            job_responses = self._get_jobs(
                api_filter=api_filter,
                limit=limit,
                skip=skip,
                descending=descending,
                legacy=legacy,
            )
            if len(job_responses) == 0:
                break
            for job_info in job_responses:
                if (
                    job_info.get("program", {}).get("id")
                    in [
                        "circuit-runner",
                        "qasm3-runner",
                    ]
                    or legacy
                ):
                    if filter_by_status:
                        if legacy:
                            job_info_status = job_info["status"].upper()
                        else:
                            job_info_status = job_info["state"]["status"].upper()
                        job_status = api_status_to_job_status(job_info_status).name
                        if (isinstance(status, str) and job_status != status) or (
                            isinstance(status, list) and job_status not in status
                        ):
                            continue
                    job = self._restore_circuit_job(
                        job_info, raise_error=False, legacy=legacy
                    )
                    if job is None:
                        logger.warning(
                            'Discarding job "%s" because it contains invalid data.',
                            job_info.get("job_id", ""),
                        )
                        continue
                    job_list.append(job)
                    if limit and len(job_list) == limit:
                        return job_list
            skip += len(job_responses)
        return job_list

    def _get_jobs(
        self,
        api_filter: Dict,
        limit: Optional[int] = 10,
        skip: int = 0,
        descending: bool = True,
        legacy: bool = False,
    ) -> List:
        """Retrieve the requested number of jobs from the server using pagination.

        Args:
            api_filter: Filter used for querying.
            limit: Number of jobs to retrieve. ``None`` means no limit.
            skip: Starting index for the job retrieval.
            descending: If ``True``, return the jobs in descending order of the job
                creation date (i.e. newest first) until the limit is reached.
            legacy: Filter to only retrieve jobs run from the archived ``qiskit-ibmq-provider``.

        Returns:
            A list of raw API response.
        """
        # Retrieve the requested number of jobs, using pagination. The server
        # might limit the number of jobs per request.
        job_responses: List[Dict[str, Any]] = []
        current_page_limit = (
            limit if (limit is not None and limit <= PAGE_SIZE) else PAGE_SIZE
        )
        while True:
            if legacy:
                job_page = self._default_hgp._api_client.list_jobs(
                    limit=current_page_limit,
                    skip=skip,
                    descending=descending,
                    extra_filter=api_filter,
                )
            else:
                job_page = self._provider._runtime_client.jobs_get(
                    limit=current_page_limit,
                    skip=skip,
                    descending=descending,
                    **api_filter,
                )["jobs"]
            if logger.getEffectiveLevel() is logging.DEBUG:
                filtered_data = [filter_data(job) for job in job_page]
                logger.debug("jobs() response data is %s", filtered_data)
            if not job_page:
                # Stop if there are no more jobs returned by the server.
                break
            job_responses += job_page
            if limit:
                if len(job_responses) >= limit:
                    # Stop if we have reached the limit.
                    break
                current_page_limit = limit - len(job_responses)
            else:
                current_page_limit = PAGE_SIZE
            skip = len(job_responses)
        return job_responses

    def _restore_circuit_job(
        self, job_info: Dict, raise_error: bool, legacy: bool = False
    ) -> Optional[IBMCircuitJob]:
        """Restore a circuit job from the API response.

        Args:
            job_info: Job info in dictionary format.
            raise_error: Whether to raise an exception if `job_info` is in
                an invalid format.
            legacy: Filter to only retrieve jobs run from the archived ``qiskit-ibmq-provider``.

        Returns:
            Circuit job restored from the data, or ``None`` if format is invalid.

        Raises:
            IBMBackendApiProtocolError: If unexpected return value received
                 from the server.
        """
        if legacy:
            job_id = job_info.get("job_id", "")
            backend_name = job_info.get("_backend_info", {}).get("name", "unknown")
            instance = None
            job_params = job_info
        else:
            job_id = job_info["id"]
            job_params = {
                "job_id": job_id,
                "creation_date": job_info["created"],
                "status": job_info["status"],
                "runtime_client": self._provider._runtime_client,
                "tags": job_info.get("tags"),
            }
            # Recreate the backend used for this job.
            backend_name = job_info.get("backend")
            instance = to_instance_format(
                job_info["hub"], job_info["group"], job_info["project"]
            )
        try:
            backend = self._provider.get_backend(backend_name, instance)
        except QiskitBackendNotFoundError:
            backend = IBMRetiredBackend.from_name(
                backend_name=backend_name,
                provider=self._provider,
                api=self._default_hgp._api_client,
            )
        try:
            job = IBMCircuitJob(
                backend=backend, api_client=self._default_hgp._api_client, **job_params
            )
            return job
        except TypeError as ex:
            if raise_error:
                raise IBMBackendApiProtocolError(
                    f"Unexpected return value received from the server "
                    f"when retrieving job {job_id}: {ex}"
                ) from ex
        return None

    def _merge_logical_filters(self, cur_filter: Dict, new_filter: Dict) -> None:
        """Merge the logical operators in the input filters.

        Args:
            cur_filter: Current filter.
            new_filter: New filter to be merged into ``cur_filter``.

        Returns:
            ``cur_filter`` with ``new_filter``'s logical operators merged into it.
        """
        logical_operators_to_expand = ["or", "and"]
        for key in logical_operators_to_expand:
            if key in new_filter:
                if key in cur_filter:
                    cur_filter[key].extend(new_filter[key])
                else:
                    cur_filter[key] = new_filter[key]

    def _update_creation_date_filter(
        self,
        cur_dt_filter: Dict[str, Any],
        gte_dt: Optional[str] = None,
        lte_dt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Use the new start and end datetime in the creation date filter.

        Args:
            cur_dt_filter: Current creation date filter.
            gte_dt: New start datetime.
            lte_dt: New end datetime.

        Returns:
            Updated creation date filter.
        """
        if not gte_dt:
            gt_list = [
                cur_dt_filter.pop(gt_op)
                for gt_op in ["gt", "gte"]
                if gt_op in cur_dt_filter
            ]
            if "between" in cur_dt_filter and len(cur_dt_filter["between"]) > 0:
                gt_list.append(cur_dt_filter.pop("between")[0])
            gte_dt = max(gt_list) if gt_list else None
        if not lte_dt:
            lt_list = [
                cur_dt_filter.pop(lt_op)
                for lt_op in ["lt", "lte"]
                if lt_op in cur_dt_filter
            ]
            if "between" in cur_dt_filter and len(cur_dt_filter["between"]) > 1:
                lt_list.append(cur_dt_filter.pop("between")[1])
            lte_dt = min(lt_list) if lt_list else None
        new_dt_filter = {}  # type: Dict[str, Union[str, List[str]]]
        if gte_dt and lte_dt:
            new_dt_filter["between"] = [gte_dt, lte_dt]
        elif gte_dt:
            new_dt_filter["gte"] = gte_dt
        elif lte_dt:
            new_dt_filter["lte"] = lte_dt
        return new_dt_filter

    def _get_status_db_filter(
        self, status_arg: Union[JobStatus, str, List[Union[JobStatus, str]]]
    ) -> Dict[str, Any]:
        """Return the db filter to use when retrieving jobs based on a status or statuses.

        Returns:
            The status db filter used to query the api when retrieving jobs that match
            a given status or list of statuses.

        Raises:
            IBMBackendError: If a status value is not recognized.
        """
        _final_status_filter = None
        if isinstance(status_arg, list):
            _final_status_filter = {"or": []}
            for status in status_arg:
                status_filter = self._get_status_filter(status)
                _final_status_filter["or"].append(status_filter)
        else:
            status_filter = self._get_status_filter(status_arg)
            _final_status_filter = status_filter
        return _final_status_filter

    def _get_status_filter(self, status: Union[JobStatus, str]) -> Dict[str, Any]:
        """Return the db filter to use when retrieving jobs based on a status.

        Returns:
            The status db filter used to query the api when retrieving jobs
            that match a given status.

        Raises:
            IBMBackendValueError: If the status value is not recognized.
        """
        if isinstance(status, str):
            try:
                status = JobStatus[status.upper()]
            except KeyError:
                raise IBMBackendValueError(
                    '"{}" is not a valid status value. Valid values are {}'.format(
                        status, ", ".join(job_status.name for job_status in JobStatus)
                    )
                ) from None
        _status_filter = {}  # type: Dict[str, Any]
        if status == JobStatus.INITIALIZING:
            _status_filter = {
                "status": {
                    "inq": [ApiJobStatus.CREATING.value, ApiJobStatus.CREATED.value]
                }
            }
        elif status == JobStatus.VALIDATING:
            _status_filter = {
                "status": {
                    "inq": [ApiJobStatus.VALIDATING.value, ApiJobStatus.VALIDATED.value]
                }
            }
        elif status == JobStatus.RUNNING:
            _status_filter = {"status": ApiJobStatus.RUNNING.value}
        elif status == JobStatus.QUEUED:
            _status_filter = {"status": ApiJobStatus.QUEUED.value}
        elif status == JobStatus.CANCELLED:
            _status_filter = {"status": ApiJobStatus.CANCELLED.value}
        elif status == JobStatus.DONE:
            _status_filter = {"status": ApiJobStatus.COMPLETED.value}
        elif status == JobStatus.ERROR:
            _status_filter = {"status": {"regexp": "^ERROR"}}
        else:
            raise IBMBackendValueError(
                '"{}" is not a valid status value. Valid values are {}'.format(
                    status, ", ".join(job_status.name for job_status in JobStatus)
                )
            )
        return _status_filter

    def retrieve_job(self, job_id: str) -> IBMJob:
        """Return a single job.

        Args:
            job_id: The ID of the job to retrieve.

        Returns:
            The job with the given id.

        Raises:
            IBMBackendApiError: If an unexpected error occurred when retrieving
                the job.
            IBMBackendApiProtocolError: If unexpected return value received
                 from the server.
            IBMJobNotFoundError: If job cannot be found.
            IBMInputValueError: If job exists but was run from a different service.
        """
        try:
            legacy = False
            if self._provider._runtime_client.job_type(job_id) == "IQX":
                legacy = True
                job_info = self._default_hgp._api_client.job_get(job_id)
            else:
                job_info = self._provider._runtime_client.job_get(
                    job_id, exclude_params=True
                )
                if job_info.get("program", {}).get("id") not in [
                    "circuit-runner",
                    "qasm3-runner",
                ]:
                    raise IBMInputValueError(
                        f"Job {job_id} was not run with qiskit-ibm-provider, "
                        f"please try retrieving job results from qiskit-ibm-runtime"
                    )
        except ApiError as ex:
            if "Error code: 3250." in str(ex):
                raise IBMJobNotFoundError(f"Job {job_id} not found.")
            raise IBMBackendApiError(
                "Failed to get job {}: {}".format(job_id, str(ex))
            ) from ex
        job = self._restore_circuit_job(job_info, raise_error=True, legacy=legacy)
        return job

    def _set_backend_config(
        self, backend_name: str, instance: Optional[str] = None
    ) -> None:
        """Retrieve backend configuration and add to backend_configs.

        Args:
            backend_name: backend name that will be returned.
            instance: the current h/g/p.
        """
        if backend_name not in self._backend_configs:
            raw_config = self._provider._runtime_client.backend_configuration(
                backend_name
            )
            config = configuration_from_server_data(
                raw_config=raw_config, instance=instance
            )
            self._backend_configs[backend_name] = config

    def _create_backend_obj(
        self,
        config: Union[QasmBackendConfiguration, PulseBackendConfiguration],
        instance: Optional[str] = None,
        hgps: Optional[List] = None,
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
                for hgp in hgps:
                    if config.backend_name in hgp.backends:
                        instance = to_instance_format(
                            hgp._hub, hgp._group, hgp._project
                        )
                        break

            elif (
                config.backend_name
                not in self._provider._get_hgp(instance=instance).backends
            ):
                raise QiskitBackendNotFoundError(
                    f"Backend {config.backend_name} is not in "
                    f"{instance}: please try a different hub/group/project."
                )

            return ibm_backend.IBMBackend(
                instance=instance,
                configuration=config,
                api_client=AccountClient(self._provider._client_params),
                provider=self._provider,
            )
        return None

    @staticmethod
    def _deprecated_backend_names() -> Dict[str, str]:
        """Returns deprecated backend names."""
        return {
            "ibmqx_qasm_simulator": "ibmq_qasm_simulator",
            "ibmqx_hpc_qasm_simulator": "ibmq_qasm_simulator",
            "real": "ibmqx1",
        }

    @staticmethod
    def _aliased_backend_names() -> Dict[str, str]:
        """Returns aliased backend names."""
        return {
            "ibmq_5_yorktown": "ibmqx2",
            "ibmq_5_tenerife": "ibmqx4",
            "ibmq_16_rueschlikon": "ibmqx5",
            "ibmq_20_austin": "QS1_1",
        }
