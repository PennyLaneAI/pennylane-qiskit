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

"""Qiskit Runtime flexible session."""

from __future__ import annotations

from typing import Dict, Optional, Type, Union, Callable, Any
from types import TracebackType
from functools import wraps
import warnings

from qiskit.providers.backend import BackendV1, BackendV2

from qiskit_ibm_runtime import QiskitRuntimeService
from .exceptions import IBMInputValueError
from .runtime_job import RuntimeJob
from .runtime_job_v2 import RuntimeJobV2
from .utils.result_decoder import ResultDecoder
from .ibm_backend import IBMBackend
from .utils.default_session import set_cm_session
from .utils.deprecation import issue_deprecation_msg
from .utils.converters import hms_to_seconds
from .fake_provider.local_service import QiskitRuntimeLocalService


def _active_session(func):  # type: ignore
    """Decorator used to ensure the session is active."""

    @wraps(func)
    def _wrapper(self, *args, **kwargs):  # type: ignore
        if not self._active:
            raise RuntimeError("The session is closed.")
        return func(self, *args, **kwargs)

    return _wrapper


class Session:
    """Class for creating a Qiskit Runtime session.

    A Qiskit Runtime ``session`` allows you to group a collection of iterative calls to
    the quantum computer. A session is started when the first job within the session
    is started. Subsequent jobs within the session are prioritized by the scheduler.

    You can open a Qiskit Runtime session using this ``Session`` class and submit jobs
    to one or more primitives.

    For example::

        from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import Session, SamplerV2 as Sampler

        service = QiskitRuntimeService()
        backend = service.least_busy(operational=True, simulator=False)

        # Bell Circuit
        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(qr, cr, name="bell")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr, cr)

        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa_circuit = pm.run(qc)

        with Session(backend=backend) as session:
            sampler = Sampler(session=session)
            job = sampler.run([isa_circuit])
            pub_result = job.result()[0]
            print(f"Sampler job ID: {job.job_id()}")
            print(f"Counts: {pub_result.data.cr.get_counts()}")
    """

    def __init__(
        self,
        service: Optional[QiskitRuntimeService] = None,
        backend: Optional[Union[str, BackendV1, BackendV2]] = None,
        max_time: Optional[Union[int, str]] = None,
    ):  # pylint: disable=line-too-long
        """Session constructor.

        Args:
            service: Optional instance of the ``QiskitRuntimeService`` class.
                If ``None``, the service associated with the backend, if known, is used.
                Otherwise ``QiskitRuntimeService()`` is used to initialize
                your default saved account.
            backend: Optional instance of ``Backend`` class or string name of backend.
                If not specified, a backend will be selected automatically (IBM Cloud channel only).

            max_time:
                Maximum amount of time, a runtime session can be open before being
                forcibly closed. Can be specified as seconds (int) or a string like "2h 30m 40s".
                This value must be less than the
                `system imposed maximum
                <https://docs.quantum.ibm.com/run/max-execution-time>`_.

        Raises:
            ValueError: If an input value is invalid.
        """
        self._service: Optional[QiskitRuntimeService | QiskitRuntimeLocalService] = None
        self._backend: Optional[BackendV1 | BackendV2] = None
        self._instance = None
        self._active = True
        self._session_id = None

        self._service = service
        if isinstance(backend, IBMBackend):
            self._service = self._service or backend.service
            self._backend = backend
        elif isinstance(backend, (BackendV1, BackendV2)):
            self._service = QiskitRuntimeLocalService()
            self._backend = backend
        else:
            if not self._service:
                self._service = (
                    QiskitRuntimeService()
                    if QiskitRuntimeService.global_service is None
                    else QiskitRuntimeService.global_service
                )
            if isinstance(backend, str):
                self._backend = self._service.backend(backend)
            elif backend is None:
                if self._service.channel == "ibm_quantum":
                    raise ValueError('"backend" is required for ``ibm_quantum`` channel.')
                issue_deprecation_msg(
                    "Not providing a backend is deprecated",
                    "0.21.0",
                    "Passing in a backend will be required, please provide a backend.",
                )
            else:
                raise ValueError(f"Invalid backend type {type(backend)}")

        self._max_time = (
            max_time
            if max_time is None or isinstance(max_time, int)
            else hms_to_seconds(max_time, "Invalid max_time value: ")
        )

        if isinstance(self._backend, IBMBackend):
            self._instance = self._backend._instance
            if not self._backend.configuration().simulator:
                self._session_id = self._create_session()

        if not self._session_id:
            warnings.warn(
                "Session is not supported in local testing mode or when using a simulator."
            )

    def _create_session(self) -> Optional[str]:
        """Create a session."""
        if isinstance(self._service, QiskitRuntimeService):
            session = self._service._api_client.create_session(
                self.backend(), self._instance, self._max_time, self._service.channel
            )
            return session.get("id")
        return None

    @_active_session
    def run(
        self,
        program_id: str,
        inputs: Dict,
        options: Optional[Dict] = None,
        callback: Optional[Callable] = None,
        result_decoder: Optional[Type[ResultDecoder]] = None,
    ) -> Union[RuntimeJob, RuntimeJobV2]:
        """Run a program in the session.

        Args:
            program_id: Program ID.
            inputs: Program input parameters. These input values are passed
                to the runtime program.
            options: Runtime options that control the execution environment.
                See :class:`qiskit_ibm_runtime.RuntimeOptions` for all available options.
            callback: Callback function to be invoked for any interim results and final result.

        Returns:
            Submitted job.
        """

        options = options or {}

        if "instance" not in options:
            options["instance"] = self._instance

        options["backend"] = self._backend

        if isinstance(self._service, QiskitRuntimeService):
            job = self._service.run(
                program_id=program_id,  # type: ignore[arg-type]
                options=options,
                inputs=inputs,
                session_id=self._session_id,
                start_session=False,
                callback=callback,
                result_decoder=result_decoder,
            )

            if self._backend is None:
                self._backend = job.backend()
        else:
            job = self._service.run(  # type: ignore[call-arg]
                program_id=program_id,  # type: ignore[arg-type]
                options=options,
                inputs=inputs,
            )

        return job

    def cancel(self) -> None:
        """Cancel all pending jobs in a session."""
        self._active = False
        if self._session_id and isinstance(self._service, QiskitRuntimeService):
            self._service._api_client.cancel_session(self._session_id)

    def close(self) -> None:
        """Close the session so new jobs will no longer be accepted, but existing
        queued or running jobs will run to completion. The session will be terminated once there
        are no more pending jobs."""
        self._active = False
        if self._session_id and isinstance(self._service, QiskitRuntimeService):
            self._service._api_client.close_session(self._session_id)

    def backend(self) -> Optional[str]:
        """Return backend for this session.

        Returns:
            Backend for this session. None if unknown.
        """
        if self._backend:
            return self._backend.name if self._backend.version == 2 else self._backend.name()
        return None

    def status(self) -> Optional[str]:
        """Return current session status.

        Returns:
            The current status of the session, including:
            Pending: Session is created but not active.
            It will become active when the next job of this session is dequeued.
            In progress, accepting new jobs: session is active and accepting new jobs.
            In progress, not accepting new jobs: session is active and not accepting new jobs.
            Closed: max_time expired or session was explicitly closed.
            None: status details are not available.
        """
        details = self.details()
        if details:
            state = details["state"]
            accepting_jobs = details["accepting_jobs"]
            if state in ["open", "inactive"]:
                return "Pending"
            if state == "active" and accepting_jobs:
                return "In progress, accepting new jobs"
            if state == "active" and not accepting_jobs:
                return "In progress, not accepting new jobs"
            return state.capitalize()

        return None

    def details(self) -> Optional[Dict[str, Any]]:
        """Return session details.

        Returns:
            A dictionary with the sessions details, including:
            id: id of the session.
            backend_name: backend used for the session.
            interactive_timeout: The maximum idle time (in seconds) between jobs that
            is allowed to occur before the session is deactivated.
            max_time: Maximum allowed time (in seconds) for the session, subject to plan limits.
            active_timeout: The maximum time (in seconds) a session can stay active.
            state: State of the session - open, active, inactive, or closed.
            accepting_jobs: Whether or not the session is accepting jobs.
            last_job_started: Timestamp of when the last job in the session started.
            last_job_completed: Timestamp of when the last job in the session completed.
            started_at: Timestamp of when the session was started.
            closed_at: Timestamp of when the session was closed.
            activated_at: Timestamp of when the session state was changed to active.
            mode: Execution mode of the session.
            usage_time: The usage time, in seconds, of this Session or Batch.
            Usage is defined as the time a quantum system is committed to complete a job.
        """
        if self._session_id and isinstance(self._service, QiskitRuntimeService):
            response = self._service._api_client.session_details(self._session_id)
            if response:
                return {
                    "id": response.get("id"),
                    "backend_name": response.get("backend_name"),
                    "interactive_timeout": response.get("interactive_ttl"),
                    "max_time": response.get("max_ttl"),
                    "active_timeout": response.get("active_ttl"),
                    "state": response.get("state"),
                    "accepting_jobs": response.get("accepting_jobs"),
                    "last_job_started": response.get("last_job_started"),
                    "last_job_completed": response.get("last_job_completed"),
                    "started_at": response.get("started_at"),
                    "closed_at": response.get("closed_at"),
                    "activated_at": response.get("activated_at"),
                    "mode": response.get("mode"),
                    "usage_time": response.get("elapsed_time"),
                }
        return None

    @property
    def session_id(self) -> Optional[str]:
        """Return the session ID.

        Returns:
            Session ID. None if the backend is a simulator.
        """
        return self._session_id

    @property
    def service(self) -> QiskitRuntimeService:
        """Return service associated with this session.

        Returns:
            :class:`qiskit_ibm_runtime.QiskitRuntimeService` associated with this session.
        """
        return self._service

    @classmethod
    def from_id(
        cls,
        session_id: str,
        service: Optional[QiskitRuntimeService] = None,
    ) -> "Session":
        """Construct a Session object with a given session_id

        Args:
            session_id: the id of the session to be created. This must be an already
                existing session id.
            service: instance of the ``QiskitRuntimeService`` class.
                If ``None``, ``QiskitRuntimeService()`` is used to initialize your default saved account.

         Raises:
            IBMInputValueError: If given `session_id` does not exist.

        Returns:
            A new Session with the given ``session_id``

        """
        if not service:
            warnings.warn(
                (
                    "The `service` parameter will be required in a future release no sooner than "
                    "3 months after the release of qiskit-ibm-runtime 0.23.0 ."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            service = QiskitRuntimeService()

        response = service._api_client.session_details(session_id)
        backend = response.get("backend_name")
        mode = response.get("mode")
        class_name = "dedicated" if cls.__name__.lower() == "session" else cls.__name__.lower()
        if mode != class_name:
            raise IBMInputValueError(
                f"Input ID {session_id} has execution mode {mode} instead of {class_name}."
            )

        session = cls(service, backend)
        session._session_id = session_id
        return session

    def __enter__(self) -> "Session":
        set_cm_session(self)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        set_cm_session(None)
        self.close()
