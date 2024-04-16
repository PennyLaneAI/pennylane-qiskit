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

import logging
from typing import Iterable, Union, Optional, Any, List, Dict
from datetime import datetime as python_datetime
from copy import deepcopy
from dataclasses import asdict
import warnings

from qiskit import QuantumCircuit
from qiskit.qobj.utils import MeasLevel, MeasReturnType

from qiskit.providers.backend import BackendV2 as Backend
from qiskit.providers.options import Options
from qiskit.providers.models import (
    BackendStatus,
    BackendProperties,
    PulseDefaults,
    GateConfig,
    QasmBackendConfiguration,
    PulseBackendConfiguration,
)
from qiskit.pulse.channels import (
    AcquireChannel,
    ControlChannel,
    DriveChannel,
    MeasureChannel,
)
from qiskit.transpiler.target import Target

# temporary until we unite the 2 Session classes
from .provider_session import (
    Session as ProviderSession,
)

from .utils.utils import validate_job_tags
from . import qiskit_runtime_service  # pylint: disable=unused-import,cyclic-import
from .runtime_job import RuntimeJob

from .api.clients import RuntimeClient
from .exceptions import IBMBackendApiProtocolError, IBMBackendValueError, IBMBackendApiError
from .utils.backend_converter import (
    convert_to_target,
)
from .utils.default_session import get_cm_session as get_cm_primitive_session
from .utils.backend_decoder import (
    defaults_from_server_data,
    properties_from_server_data,
)
from .utils.deprecation import issue_deprecation_msg
from .utils.options import QASM2Options, QASM3Options
from .api.exceptions import RequestsApiError
from .utils import local_to_utc, are_circuits_dynamic

from .utils.pubsub import Publisher


logger = logging.getLogger(__name__)

QOBJRUNNERPROGRAMID = "circuit-runner"
QASM3RUNNERPROGRAMID = "qasm3-runner"


class IBMBackend(Backend):
    """Backend class interfacing with an IBM Quantum backend.

    Note:

        * You should not instantiate the ``IBMBackend`` class directly. Instead, use
          the methods provided by an :class:`QiskitRuntimeService` instance to retrieve and handle
          backends.

    This class represents an IBM Quantum backend. Its attributes and methods provide
    information about the backend. For example, the :meth:`status()` method
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
        service: "qiskit_runtime_service.QiskitRuntimeService",
        api_client: RuntimeClient,
        instance: Optional[str] = None,
    ) -> None:
        """IBMBackend constructor.

        Args:
            configuration: Backend configuration.
            service: Instance of QiskitRuntimeService.
            api_client: IBM client used to communicate with the server.
        """
        super().__init__(
            name=configuration.backend_name,
            online_date=configuration.online_date,
            backend_version=configuration.backend_version,
        )
        self._instance = instance
        self._service = service
        self._api_client = api_client
        self._configuration = configuration
        self._properties = None
        self._defaults = None
        self._target = None
        self._max_circuits = configuration.max_experiments
        self._session: ProviderSession = None
        if (
            not self._configuration.simulator
            and hasattr(self.options, "noise_model")
            and hasattr(self.options, "seed_simulator")
        ):
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
                "'{}' object has no attribute '{}'".format(self.__class__.__name__, name)
            )
        # Lazy load properties and pulse defaults and construct the target object.
        self._get_properties()
        self._get_defaults()
        self._convert_to_target()
        # Check if the attribute now is available on IBMBackend class due to above steps
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
                "'{}' object has no attribute '{}'".format(self.__class__.__name__, name)
            )

    def _get_properties(self, datetime: Optional[python_datetime] = None) -> None:
        """Gets backend properties and decodes it"""
        if datetime:
            datetime = local_to_utc(datetime)
        if datetime or not self._properties:
            api_properties = self._api_client.backend_properties(self.name, datetime=datetime)
            if api_properties:
                backend_properties = properties_from_server_data(api_properties)
                self._properties = backend_properties

    def _get_defaults(self) -> None:
        """Gets defaults if pulse backend and decodes it"""
        if not self._defaults and isinstance(self._configuration, PulseBackendConfiguration):
            api_defaults = self._api_client.backend_pulse_defaults(self.name)
            if api_defaults:
                self._defaults = defaults_from_server_data(api_defaults)

    def _convert_to_target(self, refresh: bool = False) -> None:
        """Converts backend configuration, properties and defaults to Target object"""
        if refresh or not self._target:
            self._target = convert_to_target(
                configuration=self._configuration,
                properties=self._properties,
                defaults=self._defaults,
            )

    @classmethod
    def _default_options(cls) -> Options:
        """Default runtime options."""
        return Options(
            shots=4000,
            memory=False,
            meas_level=MeasLevel.CLASSIFIED,
            meas_return=MeasReturnType.AVERAGE,
            memory_slots=None,
            memory_slot_size=100,
            rep_time=None,
            rep_delay=None,
            init_qubits=True,
            use_measure_esp=None,
            # Simulator only
            noise_model=None,
            seed_simulator=None,
        )

    @property
    def service(self) -> "qiskit_runtime_service.QiskitRuntimeService":
        """Return the ``service`` object

        Returns:
            service: instance of QiskitRuntimeService
        """
        return self._service

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

        The maximum number of circuits (or Pulse schedules) that can be
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
        self._get_properties()
        self._get_defaults()
        self._convert_to_target()
        return self._target

    def target_history(self, datetime: Optional[python_datetime] = None) -> Target:
        """A :class:`qiskit.transpiler.Target` object for the backend.

        Returns:
            Target with properties found on `datetime`
        """
        self._get_properties(datetime=datetime)
        self._get_defaults()
        self._convert_to_target(refresh=True)
        return self._target

    def properties(
        self, refresh: bool = False, datetime: Optional[python_datetime] = None
    ) -> Optional[BackendProperties]:
        """Return the backend properties, subject to optional filtering.

        This data describes qubits properties (such as T1 and T2),
        gates properties (such as gate length and error), and other general
        properties of the backend.

        The schema for backend properties can be found in
        `Qiskit/ibm-quantum-schemas/backend_properties
        <https://github.com/Qiskit/ibm-quantum-schemas/blob/main/schemas/backend_properties_schema.json>`_.

        Args:
            refresh: If ``True``, re-query the server for the backend properties.
                Otherwise, return a cached version.
            datetime: By specifying `datetime`, this function returns an instance
                of the :class:`BackendProperties<qiskit.providers.models.BackendProperties>`
                whose timestamp is closest to, but older than, the specified `datetime`.
                Note that this is only supported using ``ibm_quantum`` runtime.

        Returns:
            The backend properties or ``None`` if the backend properties are not
            currently available.

        Raises:
            TypeError: If an input argument is not of the correct type.
            NotImplementedError: If `datetime` is specified when cloud runtime is used.
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
        if datetime:
            if not isinstance(datetime, python_datetime):
                raise TypeError("'{}' is not of type 'datetime'.")
            datetime = local_to_utc(datetime)
        if datetime or refresh or self._properties is None:
            api_properties = self._api_client.backend_properties(self.name, datetime=datetime)
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
        api_status = self._api_client.backend_status(self.name)

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
        `Qiskit/ibm-quantum-schemas/default_pulse_configuration
        <https://github.com/Qiskit/ibm-quantum-schemas/blob/main/schemas/default_pulse_configuration_schema.json>`_.

        Args:
            refresh: If ``True``, re-query the server for the backend pulse defaults.
                Otherwise, return a cached version.

        Returns:
            The backend pulse defaults or ``None`` if the backend does not support pulse.
        """
        if refresh or self._defaults is None:
            api_defaults = self._api_client.backend_pulse_defaults(self.name)
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
        `Qiskit/ibm-quantum-schemas/backend_configuration
        <https://github.com/Qiskit/ibm-quantum-schemas/blob/main/schemas/backend_configuration_schema.json>`_.

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
        """Return the secondary drive channel for the given qubit

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

    def __call__(self) -> "IBMBackend":
        # For backward compatibility only, can be removed later.
        return self

    def _check_circuits_attributes(self, circuits: List[Union[QuantumCircuit, str]]) -> None:
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
                self.check_faulty(circ)

    def check_faulty(self, circuit: QuantumCircuit) -> None:
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
        faulty_edges = [tuple(gate.qubits) for gate in faulty_gates if len(gate.qubits) > 1]

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

    def __deepcopy__(self, _memo: dict = None) -> "IBMBackend":
        cpy = IBMBackend(
            configuration=deepcopy(self.configuration()),
            service=self._service,
            api_client=deepcopy(self._api_client),
            instance=self._instance,
        )
        cpy.name = self.name
        cpy.description = self.description
        cpy.online_date = self.online_date
        cpy.backend_version = self.backend_version
        cpy._coupling_map = self._coupling_map
        cpy._defaults = deepcopy(self._defaults, _memo)
        cpy._target = deepcopy(self._target, _memo)
        cpy._max_circuits = self._max_circuits
        cpy._options = deepcopy(self._options, _memo)
        return cpy

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
    ) -> RuntimeJob:
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
            header: User input that will be attached to the job and will be
                copied to the corresponding result header. Headers do not affect the run.
                This replaces the old ``Qobj`` header. This parameter is applicable only
                if ``dynamic=False`` is specified or defaulted to.
            shots: Number of repetitions of each circuit, for sampling. Default: 4000
                or ``max_shots`` from the backend configuration, whichever is smaller.
                This parameter is applicable only if ``dynamic=False`` is specified or defaulted to.
            memory: If ``True``, per-shot measurement bitstrings are returned as well
                (provided the backend supports it). For OpenPulse jobs, only
                measurement level 2 supports this option. This parameter is applicable only if
                ``dynamic=False`` is specified or defaulted to.
            meas_level: Level of the measurement output for pulse experiments. See
                `OpenPulse specification <https://arxiv.org/pdf/1809.03452.pdf>`_ for details:

                * ``0``, measurements of the raw signal (the measurement output pulse envelope)
                * ``1``, measurement kernel is selected (a complex number obtained after applying the
                  measurement kernel to the measurement output signal)
                * ``2`` (default), a discriminator is selected and the qubit state is stored (0 or 1)

                This parameter is applicable only if ``dynamic=False`` is specified or defaulted to.
            meas_return: Level of measurement data for the backend to return. For ``meas_level`` 0 and 1:

                * ``single`` returns information from every shot.
                * ``avg`` returns average measurement output (averaged over number of shots).

                This parameter is applicable only if ``dynamic=False`` is specified or defaulted to.
            rep_delay: Delay between programs in seconds. Only supported on certain
                backends (if ``backend.configuration().dynamic_reprate_enabled=True``).
                If supported, ``rep_delay`` must be from the range supplied
                by the backend (``backend.configuration().rep_delay_range``). Default is given by
                ``backend.configuration().default_rep_delay``. This parameter is applicable only if
                ``dynamic=False`` is specified or defaulted to.
            init_qubits: Whether to reset the qubits to the ground state for each shot.
                Default: ``True``. This parameter is applicable only if ``dynamic=False`` is specified
                or defaulted to.
            use_measure_esp: Whether to use excited state promoted (ESP) readout for measurements
                which are the terminal instruction to a qubit. ESP readout can offer higher fidelity
                than standard measurement sequences. See
                `here <https://arxiv.org/pdf/2008.08571.pdf>`_.
                Default: ``True`` if backend supports ESP readout, else ``False``. Backend support
                for ESP readout is determined by the flag ``measure_esp_enabled`` in
                ``backend.configuration()``. This parameter is applicable only if ``dynamic=False`` is
                specified or defaulted to.
            noise_model: Noise model (Simulators only). This parameter is applicable
                only if ``dynamic=False`` is specified or defaulted to.
            seed_simulator: Random seed to control sampling (Simulators only). This parameter
                is applicable only if ``dynamic=False`` is specified or defaulted to.
            **run_config: Extra arguments used to configure the run. This parameter is applicable
                only if ``dynamic=False`` is specified or defaulted to.

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
        issue_deprecation_msg(
            msg="backend.run() and related sessions methods are deprecated ",
            version="0.23",
            remedy="More details can be found in the primitives migration "
            "guide https://docs.quantum.ibm.com/api/migration-guides/qiskit-runtime.",
            period="6 months",
        )
        validate_job_tags(job_tags)
        if not isinstance(circuits, List):
            circuits = [circuits]
        self._check_circuits_attributes(circuits)

        if use_measure_esp and getattr(self.configuration(), "measure_esp_enabled", False) is False:
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

        if dynamic and "qasm3" not in getattr(self.configuration(), "supported_features", []):
            warnings.warn(f"The backend {self.name} does not support dynamic circuits.")

        status = self.status()
        if status.operational is True and status.status_msg != "active":
            warnings.warn(f"The backend {self.name} is currently paused.")

        program_id = str(run_config.get("program_id", ""))
        if program_id:
            run_config.pop("program_id", None)
        else:
            program_id = QASM3RUNNERPROGRAMID if dynamic else QOBJRUNNERPROGRAMID

        image: Optional[str] = run_config.get("image", None)  # type: ignore
        if image is not None:
            image = str(image)

        if isinstance(init_circuit, bool):
            raise IBMBackendApiError(
                "init_circuit does not accept boolean values. "
                "A quantum circuit should be passed in instead."
            )

        if isinstance(shots, float):
            shots = int(shots)

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
    ) -> RuntimeJob:
        """Runs the runtime program and returns the corresponding job object"""
        hgp_name = None
        if self._service._channel == "ibm_quantum":
            hgp_name = self._instance or self._service._get_hgp().name

        # Check if initialized within a Primitive session. If so, issue a warning.
        if get_cm_primitive_session():
            warnings.warn(
                "A Primitive session is open but Backend.run() jobs will not be run within this session"
            )
        session_id = None
        if self._session:
            if not self._session.active:
                raise RuntimeError(f"The session {self._session.session_id} is closed.")
            session_id = self._session.session_id

        log_level = getattr(self.options, "log_level", None)  # temporary
        try:
            response = self._api_client.program_run(
                program_id=program_id,
                backend_name=backend_name,
                params=inputs,
                hgp=hgp_name,
                log_level=log_level,
                job_tags=job_tags,
                session_id=session_id,
                start_session=False,
                image=image,
            )
        except RequestsApiError as ex:
            raise IBMBackendApiError("Error submitting job: {}".format(str(ex))) from ex
        try:
            job = RuntimeJob(
                backend=self,
                api_client=self._api_client,
                client_params=self._service._client_params,
                job_id=response["id"],
                program_id=program_id,
                session_id=session_id,
                service=self.service,
                tags=job_tags,
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

    def open_session(self, max_time: Optional[Union[int, str]] = None) -> ProviderSession:
        """Open session"""
        issue_deprecation_msg(
            msg="backend.run() and related sessions methods are deprecated ",
            version="0.23",
            remedy="More details can be found in the primitives migration guide "
            "https://docs.quantum.ibm.com/api/migration-guides/qiskit-runtime.",
            period="6 months",
        )
        if not self._configuration.simulator:
            new_session = self._service._api_client.create_session(
                self.name, self._instance, max_time, self._service.channel
            )
            self._session = ProviderSession(max_time=max_time, session_id=new_session.get("id"))
        else:
            self._session = ProviderSession()
        return self._session

    @property
    def session(self) -> ProviderSession:
        """Return session"""
        issue_deprecation_msg(
            msg="backend.run() and related sessions methods are deprecated ",
            version="0.23",
            remedy="More details can be found in the primitives migration "
            "guide https://docs.quantum.ibm.com/api/migration-guides/qiskit-runtime.",
            period="6 months",
        )
        return self._session

    def cancel_session(self) -> None:
        """Cancel session. All pending jobs will be cancelled."""
        issue_deprecation_msg(
            msg="backend.run() and related sessions methods are deprecated ",
            version="0.23",
            remedy="More details can be found in the primitives migration "
            "guide https://docs.quantum.ibm.com/api/migration-guides/qiskit-runtime.",
            period="6 months",
        )
        if self._session:
            self._session.cancel()
            if self._session.session_id:
                self._api_client.close_session(self._session.session_id)

        self._session = None

    def close_session(self) -> None:
        """Close the session so new jobs will no longer be accepted, but existing
        queued or running jobs will run to completion. The session will be terminated once there
        are no more pending jobs."""
        issue_deprecation_msg(
            msg="backend.run() and related sessions methods are deprecated ",
            version="0.23",
            remedy="More details can be found in the primitives migration "
            "guide https://docs.quantum.ibm.com/api/migration-guides/qiskit-runtime.",
            period="6 months",
        )
        if self._session:
            self._session.cancel()
            if self._session.session_id:
                self._api_client.close_session(self._session.session_id)
        self._session = None


class IBMRetiredBackend(IBMBackend):
    """Backend class interfacing with an IBM Quantum device no longer available."""

    def __init__(
        self,
        configuration: Union[QasmBackendConfiguration, PulseBackendConfiguration],
        service: "qiskit_runtime_service.QiskitRuntimeService",
        api_client: Optional[RuntimeClient] = None,
    ) -> None:
        """IBMRetiredBackend constructor.

        Args:
            configuration: Backend configuration.
            service: Instance of QiskitRuntimeService.
            api_client: IBM Quantum client used to communicate with the server.
        """
        super().__init__(configuration, service, api_client)
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
        return Options(shots=4000)

    def properties(self, refresh: bool = False, datetime: Optional[python_datetime] = None) -> None:
        """Return the backend properties."""
        return None

    def defaults(self, refresh: bool = False) -> None:
        """Return the pulse defaults for the backend."""
        return None

    def status(self) -> BackendStatus:
        """Return the backend status."""
        return self._status

    @classmethod
    def from_name(
        cls,
        backend_name: str,
        api: Optional[RuntimeClient] = None,
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
        return cls(configuration, api)
