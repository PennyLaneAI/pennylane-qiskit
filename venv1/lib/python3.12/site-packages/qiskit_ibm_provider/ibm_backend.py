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

"""Module for interfacing with an IBM Quantum Backend."""

import copy
import logging
import warnings
from dataclasses import asdict
from datetime import datetime as python_datetime
from typing import Iterable, Dict, List, Union, Optional, Any

from qiskit.circuit import QuantumCircuit
from qiskit.providers.backend import BackendV2 as Backend
from qiskit.providers.models import (
    BackendStatus,
    BackendProperties,
    PulseDefaults,
    GateConfig,
    QasmBackendConfiguration,
    PulseBackendConfiguration,
)
from qiskit.providers.options import Options
from qiskit.pulse.channels import (
    AcquireChannel,
    ControlChannel,
    DriveChannel,
    MeasureChannel,
)

from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.target import Target

from qiskit_ibm_provider import (  # pylint: disable=unused-import
    ibm_provider,
)
from .session import Session
from .api.clients import AccountClient
from .exceptions import (
    IBMBackendError,
    IBMBackendValueError,
    IBMBackendApiError,
    IBMBackendApiProtocolError,
)

from .job import IBMJob, IBMCircuitJob
from .transpiler.passes.basis.convert_id_to_delay import (
    ConvertIdToDelay,
)
from .utils import validate_job_tags, are_circuits_dynamic
from .utils.options import QASM2Options, QASM3Options
from .utils.pubsub import Publisher
from .utils.converters import local_to_utc
from .utils.json_decoder import (
    defaults_from_server_data,
    properties_from_server_data,
    target_from_server_data,
)
from .api.exceptions import RequestsApiError


logger = logging.getLogger(__name__)


QOBJRUNNERPROGRAMID = "circuit-runner"
QASM3RUNNERPROGRAMID = "qasm3-runner"


class IBMBackend(Backend):
    """Backend class interfacing with an IBM Quantum device.

    You can run experiments on a backend using the :meth:`run()` method. The
    :meth:`run()` method takes one or more :class:`~qiskit.circuit.QuantumCircuit`
    and returns an :class:`~qiskit_ibm_provider.job.IBMJob`
    instance that represents the submitted job. Each job has a unique job ID, which
    can later be used to retrieve the job. An example of this flow::

        from qiskit import transpile
        from qiskit_ibm_provider import IBMProvider
        from qiskit.circuit.random import random_circuit

        provider = IBMProvider()
        backend = provider.backend.ibmq_vigo
        qx = random_circuit(n_qubits=5, depth=4)
        transpiled = transpile(qx, backend=backend)
        job = backend.run(transpiled)
        retrieved_job = provider.backend.retrieve_job(job.job_id())

    Note:

        * Unlike :meth:`qiskit.execute`, the :meth:`run` method does not transpile
          the circuits for you, so be sure to do so before submitting them.

        * You should not instantiate the ``IBMBackend`` class directly. Instead, use
          the methods provided by an :class:`IBMProvider` instance to retrieve and handle
          backends.

    Other methods return information about the backend. For example, the :meth:`status()` method
    returns a :class:`BackendStatus<qiskit.providers.models.BackendStatus>` instance.
    The instance contains the ``operational`` and ``pending_jobs`` attributes, which state whether
    the backend is operational and also the number of jobs in the server queue for the backend,
    respectively::

        status = backend.status()
        is_operational = status.operational
        jobs_in_queue = status.pending_jobs

    Here is list of attributes available on the ``IBMBackend`` class:
        * name: backend name.
        * backend_version: backend version in the form X.Y.Z.
        * num_qubits: number of qubits.
        * target: A :class:`qiskit.transpiler.Target` object for the backend.
        * basis_gates: list of basis gates names on the backend.
        * gates: list of basis gates on the backend.
        * local: backend is local or remote.
        * simulator: backend is a simulator.
        * conditional: backend supports conditional operations.
        * open_pulse: backend supports open pulse.
        * memory: backend supports memory.
        * max_shots: maximum number of shots supported.
        * coupling_map (list): The coupling map for the device
        * supported_instructions (List[str]): Instructions supported by the backend.
        * dynamic_reprate_enabled (bool): whether delay between programs can be set dynamically
          (ie via ``rep_delay``). Defaults to False.
        * rep_delay_range (List[float]): 2d list defining supported range of repetition
          delays for backend in μs. First entry is lower end of the range, second entry is
          higher end of the range. Optional, but will be specified when
          ``dynamic_reprate_enabled=True``.
        * default_rep_delay (float): Value of ``rep_delay`` if not specified by user and
          ``dynamic_reprate_enabled=True``.
        * n_uchannels: Number of u-channels.
        * u_channel_lo: U-channel relationship on device los.
        * meas_levels: Supported measurement levels.
        * qubit_lo_range: Qubit lo ranges for each qubit with form (min, max) in GHz.
        * meas_lo_range: Measurement lo ranges for each qubit with form (min, max) in GHz.
        * dt: Qubit drive channel timestep in nanoseconds.
        * dtm: Measurement drive channel timestep in nanoseconds.
        * rep_times: Supported repetition times (program execution time) for backend in μs.
        * meas_kernels: Supported measurement kernels.
        * discriminators: Supported discriminators.
        * hamiltonian: An optional dictionary with fields characterizing the system hamiltonian.
        * channel_bandwidth (list): Bandwidth of all channels
          (qubit, measurement, and U)
        * acquisition_latency (list): Array of dimension
          n_qubits x n_registers. Latency (in units of dt) to write a
          measurement result from qubit n into register slot m.
        * conditional_latency (list): Array of dimension n_channels
          [d->u->m] x n_registers. Latency (in units of dt) to do a
          conditional operation on channel n from register slot m
        * meas_map (list): Grouping of measurement which are multiplexed
        * max_circuits (int): The maximum number of experiments per job
        * sample_name (str): Sample name for the backend
        * n_registers (int): Number of register slots available for feedback
          (if conditional is True)
        * register_map (list): An array of dimension n_qubits X
          n_registers that specifies whether a qubit can store a
          measurement in a certain register slot.
        * configurable (bool): True if the backend is configurable, if the
          backend is a simulator
        * credits_required (bool): True if backend requires credits to run a
          job.
        * online_date (datetime): The date that the device went online
        * display_name (str): Alternate name field for the backend
        * description (str): A description for the backend
        * tags (list): A list of string tags to describe the backend
        * version: version of ``Backend`` class (Ex: 1, 2)
        * channels: An optional dictionary containing information of each channel -- their
          purpose, type, and qubits operated on.
        * parametric_pulses (list): A list of pulse shapes which are supported on the backend.
          For example: ``['gaussian', 'constant']``
        * processor_type (dict): Processor type for this backend. A dictionary of the
          form ``{"family": <str>, "revision": <str>, segment: <str>}`` such as
          ``{"family": "Canary", "revision": "1.0", segment: "A"}``.

            * family: Processor family of this backend.
            * revision: Revision version of this processor.
            * segment: Segment this processor belongs to within a larger chip.
    """

    id_warning_issued = False

    def __init__(
        self,
        configuration: Union[QasmBackendConfiguration, PulseBackendConfiguration],
        provider: "ibm_provider.IBMProvider",
        api_client: AccountClient,
        instance: Optional[str] = None,
    ) -> None:
        """IBMBackend constructor.

        Args:
            configuration: Backend configuration.
            provider: IBM Quantum account provider.
            api_client: IBM Quantum client used to communicate with the server.
        """
        super().__init__(
            provider=provider,
            name=configuration.backend_name,
            online_date=configuration.online_date,
            backend_version=configuration.backend_version,
        )
        self._instance = instance
        self._api_client = api_client
        self._configuration = configuration
        self._properties = None
        self._defaults = None
        self._target = None
        self._max_circuits = configuration.max_experiments
        self._session: Session = None
        if not self._configuration.simulator:
            self.options.set_validator("noise_model", type(None))
            self.options.set_validator("seed_simulator", type(None))
        if hasattr(configuration, "max_shots"):
            self.options.set_validator("shots", (1, configuration.max_shots))
        if hasattr(configuration, "rep_delay_range"):
            self.options.set_validator(
                "rep_delay",
                (configuration.rep_delay_range[0], configuration.rep_delay_range[1]),
            )

    def __getattr__(self, name: str) -> Any:
        """Gets attribute from self or configuration
        This magic method executes when user accesses an attribute that
        does not yet exist on IBMBackend class.
        """
        # Prevent recursion since these properties are accessed within __getattr__
        if name in ["_properties", "_defaults", "_target", "_configuration"]:
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(
                    self.__class__.__name__, name
                )
            )
        try:
            return super().__getattribute__(name)
        except AttributeError:
            pass
        # If attribute is still not available on IBMBackend class,
        # fallback to check if the attribute is available in configuration
        try:
            return self._configuration.__getattribute__(name)
        except AttributeError:
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(
                    self.__class__.__name__, name
                )
            )

    def _get_target(
        self,
        *,
        datetime: Optional[python_datetime] = None,
        refresh: bool = False,
    ) -> Target:
        """Gets target from configuration, properties and pulse defaults."""
        if datetime:
            if not isinstance(datetime, python_datetime):
                raise TypeError("'{}' is not of type 'datetime'.")
            datetime = local_to_utc(datetime)

        if datetime or refresh or self._target is None:
            client = getattr(self.provider, "_runtime_client")
            api_properties = client.backend_properties(self.name, datetime=datetime)
            api_pulse_defaults = client.backend_pulse_defaults(self.name)
            target = target_from_server_data(
                configuration=self._configuration,
                pulse_defaults=api_pulse_defaults,
                properties=api_properties,
            )
            if datetime:
                # Don't cache result.
                return target
            self._target = target
        return self._target

    @classmethod
    def _default_options(cls) -> Options:
        """Default runtime options."""
        return Options(**{**asdict(QASM3Options()), **asdict(QASM2Options())})

    @property
    def dtm(self) -> float:
        """Return the system time resolution of output signals
        Returns:
            dtm: The output signal timestep in seconds.
        """
        return self._configuration.dtm

    @property
    def max_circuits(self) -> int:
        """The maximum number of circuits
        The maximum number of circuits that can be
        run in a single job. If there is no limit this will return None.
        """
        return self._max_circuits

    @property
    def meas_map(self) -> List[List[int]]:
        """Return the grouping of measurements which are multiplexed
        This is required to be implemented if the backend supports Pulse
        scheduling.
        Returns:
            meas_map: The grouping of measurements which are multiplexed
        """
        return self._configuration.meas_map

    @property
    def target(self) -> Target:
        """A :class:`qiskit.transpiler.Target` object for the backend.
        Returns:
            Target
        """
        return self._get_target()

    def target_history(self, datetime: Optional[python_datetime] = None) -> Target:
        """A :class:`qiskit.transpiler.Target` object for the backend.
        Returns:
            Target with properties found on `datetime`
        """
        return self._get_target(datetime=datetime)

    def run(
        self,
        circuits: Union[QuantumCircuit, str, List[Union[QuantumCircuit, str]]],
        dynamic: bool = None,
        job_tags: Optional[List[str]] = None,
        init_circuit: Optional[QuantumCircuit] = None,
        init_num_resets: Optional[int] = None,
        header: Optional[Dict] = None,
        shots: Optional[Union[int, float]] = None,
        memory: Optional[bool] = None,
        meas_level: Optional[Union[int, MeasLevel]] = None,
        meas_return: Optional[Union[str, MeasReturnType]] = None,
        rep_delay: Optional[float] = None,
        init_qubits: Optional[bool] = None,
        use_measure_esp: Optional[bool] = None,
        noise_model: Optional[Any] = None,
        seed_simulator: Optional[int] = None,
        **run_config: Dict,
    ) -> IBMJob:
        """Run on the backend.
        If a keyword specified here is also present in the ``options`` attribute/object,
        the value specified here will be used for this run.

        Args:
            circuits: An individual or a
                list of :class:`~qiskit.circuits.QuantumCircuit`.
            dynamic: Whether the circuit is dynamic (uses in-circuit conditionals)
            job_tags: Tags to be assigned to the job. The tags can subsequently be used
                as a filter in the :meth:`jobs()` function call.
            init_circuit: A quantum circuit to execute for initializing qubits before each circuit.
                If specified, ``init_num_resets`` is ignored. Applicable only if ``dynamic=True``
                is specified.
            init_num_resets: The number of qubit resets to insert before each circuit execution.

            The following parameters are applicable only if ``dynamic=False`` is specified or
            defaulted to.

            header: User input that will be attached to the job and will be
                copied to the corresponding result header. Headers do not affect the run.
                This replaces the old ``Qobj`` header.
            shots: Number of repetitions of each circuit, for sampling. Default: 4000
                or ``max_shots`` from the backend configuration, whichever is smaller.
            memory: If ``True``, per-shot measurement bitstrings are returned as well
                (provided the backend supports it). For OpenPulse jobs, only
                measurement level 2 supports this option.
            meas_level: Level of the measurement output for pulse experiments. See
                `OpenPulse specification <https://arxiv.org/pdf/1809.03452.pdf>`_ for details:

                * ``0``, measurements of the raw signal (the measurement output pulse envelope)
                * ``1``, measurement kernel is selected (a complex number obtained after applying the
                  measurement kernel to the measurement output signal)
                * ``2`` (default), a discriminator is selected and the qubit state is stored (0 or 1)

            meas_return: Level of measurement data for the backend to return. For ``meas_level`` 0 and 1:

                * ``single`` returns information from every shot.
                * ``avg`` returns average measurement output (averaged over number of shots).

            rep_delay: Delay between programs in seconds. Only supported on certain
                backends (if ``backend.configuration().dynamic_reprate_enabled=True``).
                If supported, ``rep_delay`` must be from the range supplied
                by the backend (``backend.configuration().rep_delay_range``). Default is given by
                ``backend.configuration().default_rep_delay``.
            init_qubits: Whether to reset the qubits to the ground state for each shot.
                Default: ``True``.
            use_measure_esp: Whether to use excited state promoted (ESP) readout for measurements
                which are the terminal instruction to a qubit. ESP readout can offer higher fidelity
                than standard measurement sequences. See
                `here <https://arxiv.org/pdf/2008.08571.pdf>`_.
                Default: ``True`` if backend supports ESP readout, else ``False``. Backend support
                for ESP readout is determined by the flag ``measure_esp_enabled`` in
                ``backend.configuration()``.
            noise_model: Noise model. (Simulators only)
            seed_simulator: Random seed to control sampling. (Simulators only)
            **run_config: Extra arguments used to configure the run.

        Returns:
            The job to be executed.

        Raises:
            IBMBackendApiError: If an unexpected error occurred while submitting
                the job.
            IBMBackendApiProtocolError: If an unexpected value received from
                 the server.
            IBMBackendValueError:
                - If an input parameter value is not valid.
                - If ESP readout is used and the backend does not support this.
        """
        # pylint: disable=arguments-differ
        validate_job_tags(job_tags, IBMBackendValueError)
        if not isinstance(circuits, List):
            circuits = [circuits]
        self._check_circuits_attributes(circuits)

        if (
            use_measure_esp
            and getattr(self.configuration(), "measure_esp_enabled", False) is False
        ):
            raise IBMBackendValueError(
                "ESP readout not supported on this device. Please make sure the flag "
                "'use_measure_esp' is unset or set to 'False'."
            )
        actually_dynamic = are_circuits_dynamic(circuits)
        if dynamic is False and actually_dynamic:
            warnings.warn(
                "Parameter 'dynamic' is False, but the circuit contains dynamic constructs."
            )
        dynamic = dynamic or actually_dynamic

        if dynamic and "qasm3" not in getattr(
            self.configuration(), "supported_features", []
        ):
            warnings.warn(f"The backend {self.name} does not support dynamic circuits.")

        status = self.status()
        if status.operational is True and status.status_msg != "active":
            warnings.warn(f"The backend {self.name} is currently paused.")

        program_id = str(run_config.get("program_id", ""))
        if not program_id:
            if dynamic:
                program_id = QASM3RUNNERPROGRAMID
            else:
                program_id = QOBJRUNNERPROGRAMID
        else:
            run_config.pop("program_id", None)

        image: Optional[str] = run_config.get("image", None)  # type: ignore
        if image is not None:
            image = str(image)

        if isinstance(init_circuit, bool):
            warnings.warn(
                "init_circuit does not accept boolean values. "
                "A quantum circuit should be passed in instead."
            )

        if isinstance(shots, float):
            shots = int(shots)
        if not self.configuration().simulator:
            circuits = self._deprecate_id_instruction(circuits)

        run_config_dict = self._get_run_config(
            program_id=program_id,
            init_circuit=init_circuit,
            init_num_resets=init_num_resets,
            header=header,
            shots=shots,
            memory=memory,
            meas_level=meas_level,
            meas_return=meas_return,
            rep_delay=rep_delay,
            init_qubits=init_qubits,
            use_measure_esp=use_measure_esp,
            noise_model=noise_model,
            seed_simulator=seed_simulator,
            **run_config,
        )

        run_config_dict["circuits"] = circuits
        if not program_id.startswith(QASM3RUNNERPROGRAMID):
            # Transpiling in circuit-runner is deprecated.
            run_config_dict["skip_transpilation"] = True

        return self._runtime_run(
            program_id=program_id,
            inputs=run_config_dict,
            backend_name=self.name,
            job_tags=job_tags,
            image=image,
        )

    def _runtime_run(
        self,
        program_id: str,
        inputs: Dict,
        backend_name: str,
        job_tags: Optional[List[str]] = None,
        image: Optional[str] = None,
    ) -> IBMCircuitJob:
        """Runs the runtime program and returns the corresponding job object"""
        hgp_name = self._instance or self.provider._get_hgp().name

        session_id = None
        if self._session:
            if not self._session.active:
                raise RuntimeError(f"The session {self._session.session_id} is closed.")
            session_id = self._session.session_id

        try:
            response = self.provider._runtime_client.program_run(
                program_id=program_id,
                backend_name=backend_name,
                params=inputs,
                hgp=hgp_name,
                job_tags=job_tags,
                session_id=session_id,
                start_session=False,
                image=image,
            )
        except RequestsApiError as ex:
            raise IBMBackendApiError("Error submitting job: {}".format(str(ex))) from ex
        try:
            job = IBMCircuitJob(
                backend=self,
                api_client=self._api_client,
                runtime_client=self.provider._runtime_client,
                job_id=response["id"],
                session_id=session_id,
            )
            logger.debug("Job %s was successfully submitted.", job.job_id())
        except TypeError as err:
            logger.debug("Invalid job data received: %s", response)
            raise IBMBackendApiProtocolError(
                "Unexpected return value received from the server "
                "when submitting job: {}".format(str(err))
            ) from err
        Publisher().publish("ibm.job.start", job)
        return job

    def _get_run_config(self, program_id: str, **kwargs: Any) -> Dict:
        """Return the consolidated runtime configuration."""
        # Check if is a QASM3 like program id.
        if program_id.startswith(QASM3RUNNERPROGRAMID):
            fields = asdict(QASM3Options()).keys()
            run_config_dict = QASM3Options().to_transport_dict()
        else:
            fields = asdict(QASM2Options()).keys()
            run_config_dict = QASM2Options().to_transport_dict()

        backend_options = self._options.__dict__
        for key, val in kwargs.items():
            if val is not None:
                run_config_dict[key] = val
                if key not in fields and not self.configuration().simulator:
                    warnings.warn(  # type: ignore[unreachable]
                        f"{key} is not a recognized runtime option and may be ignored by the backend.",
                        stacklevel=4,
                    )
            elif backend_options.get(key) is not None and key in fields:
                run_config_dict[key] = backend_options[key]
        return run_config_dict

    def properties(
        self, refresh: bool = False, datetime: Optional[python_datetime] = None
    ) -> Optional[BackendProperties]:
        """Return the backend properties, subject to optional filtering.

        This data describes qubits properties (such as T1 and T2),
        gates properties (such as gate length and error), and other general
        properties of the backend.

        The schema for backend properties can be found in
        `Qiskit/ibm-quantum-schemas
        <https://github.com/Qiskit/ibm-quantum-schemas/blob/main/schemas/backend_properties_schema.json>`__.

        Args:
            refresh: If ``True``, re-query the server for the backend properties.
                Otherwise, return a cached version.
            datetime: By specifying `datetime`, this function returns an instance
                of the :class:`BackendProperties<qiskit.providers.models.BackendProperties>`
                whose timestamp is closest to, but older than, the specified `datetime`.

        Returns:
            The backend properties or ``None`` if the backend properties are not
            currently available.

        Raises:
            TypeError: If an input argument is not of the correct type.
        """
        # pylint: disable=arguments-differ
        if self._configuration.simulator:
            # Simulators do not have backend properties.
            return None
        if not isinstance(refresh, bool):
            raise TypeError(
                "The 'refresh' argument needs to be a boolean. "
                "{} is of type {}".format(refresh, type(refresh))
            )
        if datetime and not isinstance(datetime, python_datetime):
            raise TypeError("'{}' is not of type 'datetime'.")

        if datetime:
            datetime = local_to_utc(datetime)

        if datetime or refresh or self._properties is None:
            api_properties = self.provider._runtime_client.backend_properties(
                self.name, datetime=datetime
            )
            if not api_properties:
                return None
            backend_properties = properties_from_server_data(api_properties)
            if datetime:  # Don't cache result.
                return backend_properties
            self._properties = backend_properties
        return self._properties

    def status(self) -> BackendStatus:
        """Return the backend status.

        Note:
            If the returned :class:`~qiskit.providers.models.BackendStatus`
            instance has ``operational=True`` but ``status_msg="internal"``,
            then the backend is accepting jobs but not processing them.

        Returns:
            The status of the backend.

        Raises:
            IBMBackendApiProtocolError: If the status for the backend cannot be formatted properly.
        """
        api_status = self.provider._runtime_client.backend_status(self.name)

        try:
            return BackendStatus.from_dict(api_status)
        except TypeError as ex:
            raise IBMBackendApiProtocolError(
                "Unexpected return value received from the server when "
                "getting backend status: {}".format(str(ex))
            ) from ex

    def defaults(self, refresh: bool = False) -> Optional[PulseDefaults]:
        """Return the pulse defaults for the backend.

        The schema for default pulse configuration can be found in
        `Qiskit/ibm-quantum-schemas
        <https://github.com/Qiskit/ibm-quantum-schemas/blob/main/schemas/default_pulse_configuration_schema.json>`__.

        Args:
            refresh: If ``True``, re-query the server for the backend pulse defaults.
                Otherwise, return a cached version.

        Returns:
            The backend pulse defaults or ``None`` if the backend does not support pulse.
        """
        if refresh or self._defaults is None:
            api_defaults = self.provider._runtime_client.backend_pulse_defaults(
                self.name
            )
            if api_defaults:
                self._defaults = defaults_from_server_data(api_defaults)
            else:
                self._defaults = None

        return self._defaults

    def configuration(
        self,
    ) -> Union[QasmBackendConfiguration, PulseBackendConfiguration]:
        """Return the backend configuration.

        Backend configuration contains fixed information about the backend, such
        as its name, number of qubits, basis gates, coupling map, quantum volume, etc.

        The schema for backend configuration can be found in
        `Qiskit/ibm-quantum-schemas
        <https://github.com/Qiskit/ibm-quantum-schemas/blob/main/schemas/backend_configuration_schema.json>`__.

        Returns:
            The configuration for the backend.
        """
        return self._configuration

    def drive_channel(self, qubit: int) -> DriveChannel:
        """Return the drive channel for the given qubit.

        Returns:
            DriveChannel: The Qubit drive channel
        """
        return self._configuration.drive(qubit=qubit)

    def measure_channel(self, qubit: int) -> MeasureChannel:
        """Return the measure stimulus channel for the given qubit.

        Returns:
            MeasureChannel: The Qubit measurement stimulus line
        """
        return self._configuration.measure(qubit=qubit)

    def acquire_channel(self, qubit: int) -> AcquireChannel:
        """Return the acquisition channel for the given qubit.

        Returns:
            AcquireChannel: The Qubit measurement acquisition line.
        """
        return self._configuration.acquire(qubit=qubit)

    def control_channel(self, qubits: Iterable[int]) -> List[ControlChannel]:
        """Return the secondary drive channel for the given qubit.

        This is typically utilized for controlling multiqubit interactions.
        This channel is derived from other channels.

        Args:
            qubits: Tuple or list of qubits of the form
                ``(control_qubit, target_qubit)``.

        Returns:
            List[ControlChannel]: The Qubit measurement acquisition line.
        """
        return self._configuration.control(qubits=qubits)

    def __repr__(self) -> str:
        return "<{}('{}')>".format(self.__class__.__name__, self.name)

    def _deprecate_id_instruction(
        self, circuits: List[QuantumCircuit]
    ) -> List[QuantumCircuit]:
        """Raise a DeprecationWarning if any circuit contains an 'id' instruction.

        Additionally, if 'delay' is a 'supported_instruction', replace each 'id'
        instruction (in-place) with the equivalent ('sx'-length) 'delay' instruction.

        Args:
            circuits: The individual or list of :class:`~qiskit.circuits.QuantumCircuit`
                 passed to :meth:`IBMBackend.run()<IBMBackend.run>`. Modified in-place.

        Returns:
            A modified copy of the original circuit where 'id' instructions are replaced with
            'delay' instructions. A copy is used so the original circuit is not modified.
            If there are no 'id' instructions or 'delay' is not supported, return the original circuit.
        """

        id_support = "id" in getattr(self.configuration(), "basis_gates", [])
        delay_support = "delay" in getattr(
            self.configuration(), "supported_instructions", []
        )

        if not delay_support:
            return circuits

        circuit_has_id = any(
            instr.name == "id"
            for circuit in circuits
            if isinstance(circuit, QuantumCircuit)
            for instr, qargs, cargs in circuit.data
        )
        if not circuit_has_id:
            return circuits
        if not self.id_warning_issued:
            if id_support and delay_support:
                warnings.warn(
                    "Support for the 'id' instruction has been deprecated "
                    "from IBM hardware backends. Any 'id' instructions "
                    "will be replaced with their equivalent 'delay' instruction. "
                    "Please use the 'delay' instruction instead.",
                    DeprecationWarning,
                    stacklevel=4,
                )
            else:
                warnings.warn(
                    "Support for the 'id' instruction has been removed "
                    "from IBM hardware backends. Any 'id' instructions "
                    "will be replaced with their equivalent 'delay' instruction. "
                    "Please use the 'delay' instruction instead.",
                    DeprecationWarning,
                    stacklevel=4,
                )

            self.id_warning_issued = True

        # Make sure we don't mutate user's input circuits
        circuits = copy.deepcopy(circuits)
        # Convert id gates to delays.
        pm = PassManager(  # pylint: disable=invalid-name
            ConvertIdToDelay(self.target.durations())
        )
        circuits = pm.run(circuits)

        return circuits

    @classmethod
    def get_translation_stage_plugin(cls) -> str:
        """Return the default translation stage plugin name for IBM backends."""
        return "ibm_dynamic_circuits"

    def _check_circuits_attributes(self, circuits: List[QuantumCircuit]) -> None:
        """Check that circuits can be executed on backend.
        Raises:
            IBMBackendValueError:
                - If one of the circuits contains more qubits than on the backend."""

        if len(circuits) > self._max_circuits:
            raise IBMBackendValueError(
                f"Number of circuits, {len(circuits)} exceeds the "
                f"maximum for this backend, {self._max_circuits})"
            )
        for circ in circuits:
            if isinstance(circ, QuantumCircuit):
                if circ.num_qubits > self._configuration.num_qubits:
                    raise IBMBackendValueError(
                        f"Circuit contains {circ.num_qubits} qubits, "
                        f"but backend has only {self.num_qubits}."
                    )
                self._check_faulty(circ)

    def _check_faulty(self, circuit: QuantumCircuit) -> None:
        """Check if the input circuit uses faulty qubits or edges.

        Args:
            circuit: Circuit to check.

        Raises:
            ValueError: If an instruction operating on a faulty qubit or edge is found.
        """
        if not self.properties():
            return

        faulty_qubits = self.properties().faulty_qubits()
        faulty_gates = self.properties().faulty_gates()
        faulty_edges = [
            tuple(gate.qubits) for gate in faulty_gates if len(gate.qubits) > 1
        ]

        for instr in circuit.data:
            if instr.operation.name == "barrier":
                continue
            qubit_indices = tuple(circuit.find_bit(x).index for x in instr.qubits)

            for circ_qubit in qubit_indices:
                if circ_qubit in faulty_qubits:
                    raise ValueError(
                        f"Circuit {circuit.name} contains instruction "
                        f"{instr} operating on a faulty qubit {circ_qubit}."
                    )

            if len(qubit_indices) == 2 and qubit_indices in faulty_edges:
                raise ValueError(
                    f"Circuit {circuit.name} contains instruction "
                    f"{instr} operating on a faulty edge {qubit_indices}"
                )

    def open_session(self, max_time: Optional[Union[int, str]] = None) -> Session:
        """Open session"""
        if not self._configuration.simulator:
            new_session = self.provider._runtime_client.create_session(
                self.name, self._instance, max_time
            )
            self._session = Session(max_time=max_time, session_id=new_session.get("id"))
        else:
            self._session = Session()
        return self._session

    @property
    def session(self) -> Session:
        """Return session"""
        return self._session

    def cancel_session(self) -> None:
        """Cancel session. All pending jobs will be cancelled."""
        if self._session:
            self._session.cancel()
            if self._session.session_id:
                self.provider._runtime_client.cancel_session(self._session.session_id)
        self._session = None

    def close_session(self) -> None:
        """Close the session so new jobs will no longer be accepted, but existing
        queued or running jobs will run to completion. The session will be terminated once there
        are no more pending jobs."""
        if self._session:
            self._session.cancel()
            if self._session.session_id:
                self.provider._runtime_client.close_session(self._session.session_id)
        self._session = None


class IBMRetiredBackend(IBMBackend):
    """Backend class interfacing with an IBM Quantum device no longer available."""

    def __init__(
        self,
        configuration: Union[QasmBackendConfiguration, PulseBackendConfiguration],
        provider: "ibm_provider.IBMProvider",
        api_client: AccountClient,
    ) -> None:
        """IBMRetiredBackend constructor.

        Args:
            configuration: Backend configuration.
            provider: IBM Quantum account provider.
            credentials: IBM Quantum credentials.
            api_client: IBM Quantum client used to communicate with the server.
        """
        super().__init__(configuration, provider, api_client)
        self._status = BackendStatus(
            backend_name=self.name,
            backend_version=self.configuration().backend_version,
            operational=False,
            pending_jobs=0,
            status_msg="This backend is no longer available.",
        )

    @classmethod
    def _default_options(cls) -> Options:
        """Default runtime options."""
        return super()._default_options()

    def properties(
        self, refresh: bool = False, datetime: Optional[python_datetime] = None
    ) -> None:
        """Return the backend properties."""
        return None

    def defaults(self, refresh: bool = False) -> None:
        """Return the pulse defaults for the backend."""
        return None

    def status(self) -> BackendStatus:
        """Return the backend status."""
        return self._status

    def run(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        """Run a Circuit."""
        # pylint: disable=arguments-differ
        raise IBMBackendError(
            "This backend ({}) is no longer available.".format(self.name)
        )

    @classmethod
    def from_name(
        cls,
        backend_name: str,
        provider: "ibm_provider.IBMProvider",
        api: AccountClient,
    ) -> "IBMRetiredBackend":
        """Return a retired backend from its name."""
        configuration = QasmBackendConfiguration(
            backend_name=backend_name,
            backend_version="0.0.0",
            online_date="2019-10-16T04:00:00Z",
            n_qubits=1,
            basis_gates=[],
            simulator=False,
            local=False,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=1,
            gates=[GateConfig(name="TODO", parameters=[], qasm_def="TODO")],
            coupling_map=[[0, 1]],
            max_experiments=300,
        )
        return cls(configuration, provider, api)
