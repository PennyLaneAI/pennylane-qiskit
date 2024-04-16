# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qiskit Runtime batch mode."""

from typing import Optional, Union

from qiskit.providers.backend import BackendV1, BackendV2

from qiskit_ibm_runtime import QiskitRuntimeService
from .session import Session


class Batch(Session):
    """Class for running jobs in batch execution mode.

    Similar to a ``session``, a Qiskit Runtime ``batch`` groups a collection of
    iterative calls to the quantum computer. Batch mode can shorten processing time if all jobs
    can be provided at the outset. To submit iterative jobs, use sessions instead.

    Using batch mode has these benefits:
        - The jobs' classical computation, such as compilation, is run in parallel.
          Thus, running multiple jobs in a batch is significantly faster than running them serially.

        - There is minimal delay between job, which can help avoid drift.

    You can open a Qiskit Runtime batch by using this ``Batch`` class, then submit jobs
    to one or more primitives.

    For example::

        from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import Batch, SamplerV2 as Sampler

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

        with Batch(backend=backend) as batch:
            sampler = Sampler(batch)
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
    ):
        """Batch constructor.

        Args:
            service: Optional instance of the ``QiskitRuntimeService`` class.
                If ``None``, the service associated with the backend, if known, is used.
                Otherwise ``QiskitRuntimeService()`` is used to initialize
                your default saved account.
            backend: Optional instance of ``Backend`` class or backend string name.

            max_time:
                Maximum amount of time a runtime session can be open before being
                forcibly closed. Can be specified as seconds (int) or a string like "2h 30m 40s".
                This value must be less than the
                `system imposed maximum
                <https://docs.quantum.ibm.com/run/max-execution-time>`_.

        Raises:
            ValueError: If an input value is invalid.
        """
        super().__init__(service=service, backend=backend, max_time=max_time)

    def _create_session(self) -> Optional[str]:
        """Create a session."""
        if isinstance(self._service, QiskitRuntimeService):
            session = self._service._api_client.create_session(
                self.backend(), self._instance, self._max_time, self._service.channel, "batch"
            )
            return session.get("id")
        return None
