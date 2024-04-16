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

from typing import Optional, Type, Union
from types import TracebackType
from contextvars import ContextVar

from qiskit_ibm_provider.utils.converters import hms_to_seconds


class Session:
    """Class for creating a flexible Qiskit Runtime session.

    A Qiskit Runtime ``session`` allows you to group a collection of iterative calls to
    the quantum computer. A session is started when the first job within the session
    is started. Subsequent jobs within the session are prioritized by the scheduler.
    Data used within a session, such as transpiled circuits, is also cached to avoid
    unnecessary overhead.

    You can open a Qiskit Runtime session using this ``Session`` class
    and submit one or more jobs.

    For example::
        from qiskit_ibm_provider import IBMProvider

        # Bell Circuit
        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(qr, cr, name="bell")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr, cr)

        backend = IBMProvider().get_backend("ibmq_qasm_simulator")
        backend.open_session()

        job = backend.run(qc)
        print(f"Job ID: {job.job_id()}")
        print(f"Result: {job.result()}")
        # Close the session only if all jobs are finished and
        # you don't need to run more in the session.
        backend.cancel_session()

    Session can also be used as a context manager::

        with backend.open_session() as session:
            job = backend.run(qc)
            assert job.job_id() == session.session_id

    """

    def __init__(
        self,
        max_time: Optional[Union[int, str]] = None,
        session_id: Optional[str] = None,
    ):
        """Session constructor.

        Args:
            max_time: (EXPERIMENTAL setting, can break between releases without warning)
                Maximum amount of time, a runtime session can be open before being
                forcibly closed. Can be specified as seconds (int) or a string like "2h 30m 40s".
                This value must be in between 300 seconds and the
                `system imposed maximum
                <https://docs.quantum.ibm.com/run/max-execution-time>`_.

        Raises:
            ValueError: If an input value is invalid.
        """
        self._instance = None
        self._session_id = session_id
        self._active = True

        self._max_time = (
            max_time
            if max_time is None or isinstance(max_time, int)
            else hms_to_seconds(max_time, "Invalid max_time value: ")
        )

    @property
    def session_id(self) -> str:
        """Return the session ID.

        Returns:
            Session ID. None until a job runs in the session.
        """
        return self._session_id

    @property
    def active(self) -> bool:
        """Return the status of the session.

        Returns:
            True if the session is active, False otherwise.
        """
        return self._active

    def cancel(self) -> None:
        """Set the session._active status to False"""
        self._active = False

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


# Default session
_DEFAULT_SESSION: ContextVar[Optional[Session]] = ContextVar(
    "_DEFAULT_SESSION", default=None
)
_IN_SESSION_CM: ContextVar[bool] = ContextVar("_IN_SESSION_CM", default=False)


def set_cm_session(session: Optional[Session]) -> None:
    """Set the context manager session."""
    _DEFAULT_SESSION.set(session)
    _IN_SESSION_CM.set(session is not None)


def get_cm_session() -> Session:
    """Return the context managed session."""
    return _DEFAULT_SESSION.get()
