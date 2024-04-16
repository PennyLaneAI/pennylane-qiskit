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

"""Estimator primitive."""

from __future__ import annotations
import os
from typing import Optional, Dict, Sequence, Any, Union, Iterable
import logging

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.primitives import BaseEstimator
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.primitives.containers import EstimatorPubLike
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from .runtime_job import RuntimeJob
from .runtime_job_v2 import RuntimeJobV2
from .ibm_backend import IBMBackend
from .options import Options
from .options.estimator_options import EstimatorOptions
from .base_primitive import BasePrimitiveV1, BasePrimitiveV2
from .utils.qctrl import validate as qctrl_validate
from .utils.qctrl import validate_v2 as qctrl_validate_v2


# pylint: disable=unused-import,cyclic-import
from .session import Session

logger = logging.getLogger(__name__)


class Estimator:
    """Base class for Qiskit Runtime Estimator."""

    version = 0


class EstimatorV2(BasePrimitiveV2[EstimatorOptions], Estimator, BaseEstimatorV2):
    r"""Class for interacting with Qiskit Runtime Estimator primitive service.

    Qiskit Runtime Estimator primitive service estimates expectation values of quantum circuits and
    observables.

    The :meth:`run` can be used to submit circuits, observables, and parameters
    to the Estimator primitive.

    Following construction, an estimator is used by calling its :meth:`run` method
    with a list of PUBs (Primitive Unified Blocs). Each PUB contains four values that, together,
    define a computation unit of work for the estimator to complete:

    * a single :class:`~qiskit.circuit.QuantumCircuit`, possibly parametrized, whose final state we
      define as :math:`\psi(\theta)`,

    * one or more observables (specified as any :class:`~.ObservablesArrayLike`, including
      :class:`~.Pauli`, :class:`~.SparsePauliOp`, ``str``) that specify which expectation values to
      estimate, denoted :math:`H_j`, and

    * a collection parameter value sets to bind the circuit against, :math:`\theta_k`.

    * an optional target precision for expectation value estimates.

    Here is an example of how the estimator is used.

    .. code-block:: python

        from qiskit.circuit.library import RealAmplitudes
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

        service = QiskitRuntimeService()
        backend = service.least_busy(operational=True, simulator=False)

        psi = RealAmplitudes(num_qubits=2, reps=2)
        hamiltonian = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
        theta = [0, 1, 1, 2, 3, 5]

        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa_psi = pm.run(psi)
        isa_observables = hamiltonian.apply_layout(isa_psi.layout)

        estimator = Estimator(backend=backend)

        # calculate [ <psi(theta1)|hamiltonian|psi(theta)> ]
        job = estimator.run([(isa_psi, isa_observables, [theta])])
        pub_result = job.result()[0]
        print(f"Expectation values: {pub_result.data.evs}")
    """

    _options_class = EstimatorOptions

    version = 2

    def __init__(
        self,
        backend: Optional[Union[str, IBMBackend]] = None,
        session: Optional[Session] = None,
        options: Optional[Union[Dict, EstimatorOptions]] = None,
    ):
        """Initializes the Estimator primitive.

        Args:
            backend: Backend to run the primitive. This can be a backend name or an :class:`IBMBackend`
                instance. If a name is specified, the default account (e.g. ``QiskitRuntimeService()``)
                is used.

            session: Session in which to call the primitive.

                If both ``session`` and ``backend`` are specified, ``session`` takes precedence.
                If neither is specified, and the primitive is created inside a
                :class:`qiskit_ibm_runtime.Session` context manager, then the session is used.
                Otherwise if IBM Cloud channel is used, a default backend is selected.

            options: Estimator options, see :class:`EstimatorOptions` for detailed description.

        Raises:
            NotImplementedError: If "q-ctrl" channel strategy is used.
        """
        BaseEstimatorV2.__init__(self)
        Estimator.__init__(self)
        BasePrimitiveV2.__init__(self, backend=backend, session=session, options=options)

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> RuntimeJobV2:
        """Submit a request to the estimator primitive.

        Args:
            pubs: An iterable of pub-like (primitive unified bloc) objects, such as
                tuples ``(circuit, observables)`` or ``(circuit, observables, parameter_values)``.
            precision: The target precision for expectation value estimates of each
                run Estimator Pub that does not specify its own precision. If None
                the estimator's default precision value will be used.

        Returns:
            Submitted job.

        """
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        return self._run(coerced_pubs)  # type: ignore[arg-type]

    def _validate_options(self, options: dict) -> None:
        """Validate that program inputs (options) are valid

        Raises:
            ValidationError: if validation fails.
            ValueError: if validation fails.
        """

        if self._service._channel_strategy == "q-ctrl":
            qctrl_validate_v2(options)
            return

        if (
            options.get("resilience", {}).get("pec_mitigation", False) is True
            and self._backend is not None
            and self._backend.configuration().simulator is True
            and not options["simulator"]["coupling_map"]
        ):
            raise ValueError(
                "When the backend is a simulator and pec_mitigation is enabled, "
                "a coupling map is required."
            )

    @classmethod
    def _program_id(cls) -> str:
        """Return the program ID."""
        return "estimator"


class EstimatorV1(BasePrimitiveV1, Estimator, BaseEstimator):
    """Class for interacting with Qiskit Runtime Estimator primitive service.

    Qiskit Runtime Estimator primitive service estimates expectation values of quantum circuits and
    observables.

    The :meth:`run` can be used to submit circuits, observables, and parameters
    to the Estimator primitive.

    You are encouraged to use :class:`~qiskit_ibm_runtime.Session` to open a session,
    during which you can invoke one or more primitives. Jobs submitted within a session
    are prioritized by the scheduler.

    Example::

        from qiskit.circuit.library import RealAmplitudes
        from qiskit.quantum_info import SparsePauliOp

        from qiskit_ibm_runtime import QiskitRuntimeService, Estimator

        service = QiskitRuntimeService(channel="ibm_cloud")

        psi1 = RealAmplitudes(num_qubits=2, reps=2)

        H1 = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
        H2 = SparsePauliOp.from_list([("IZ", 1)])
        H3 = SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)])

        with Session(service=service, backend="ibmq_qasm_simulator") as session:
            estimator = Estimator(session=session)

            theta1 = [0, 1, 1, 2, 3, 5]

            # calculate [ <psi1(theta1)|H1|psi1(theta1)> ]
            psi1_H1 = estimator.run(circuits=[psi1], observables=[H1], parameter_values=[theta1])
            print(psi1_H1.result())

            # calculate [ <psi1(theta1)|H2|psi1(theta1)>, <psi1(theta1)|H3|psi1(theta1)> ]
            psi1_H23 = estimator.run(
                circuits=[psi1, psi1],
                observables=[H2, H3],
                parameter_values=[theta1]*2
            )
            print(psi1_H23.result())
    """

    version = 1

    def __init__(
        self,
        backend: Optional[Union[str, IBMBackend]] = None,
        session: Optional[Session] = None,
        options: Optional[Union[Dict, Options]] = None,
    ):
        """Initializes the Estimator primitive.

        Args:
            backend: Backend to run the primitive. This can be a backend name or an :class:`IBMBackend`
                instance. If a name is specified, the default account (e.g. ``QiskitRuntimeService()``)
                is used.

            session: Session in which to call the primitive.

                If both ``session`` and ``backend`` are specified, ``session`` takes precedence.
                If neither is specified, and the primitive is created inside a
                :class:`qiskit_ibm_runtime.Session` context manager, then the session is used.
                Otherwise if IBM Cloud channel is used, a default backend is selected.

            options: Primitive options, see :class:`Options` for detailed description.
                The ``backend`` keyword is still supported but is deprecated.
        """
        # `self._options` in this class is a Dict.
        # The base class, however, uses a `_run_options` which is an instance of
        # qiskit.providers.Options. We largely ignore this _run_options because we use
        # a nested dictionary to categorize options.
        BaseEstimator.__init__(self)
        Estimator.__init__(self)
        BasePrimitiveV1.__init__(self, backend=backend, session=session, options=options)

    def run(  # pylint: disable=arguments-differ
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | str] | BaseOperator | str,
        parameter_values: Sequence[float] | Sequence[Sequence[float]] | None = None,
        **kwargs: Any,
    ) -> RuntimeJob:
        """Submit a request to the estimator primitive.

        Args:
            circuits: a (parameterized) :class:`~qiskit.circuit.QuantumCircuit` or
                a list of (parameterized) :class:`~qiskit.circuit.QuantumCircuit`.

            observables: Observable objects.

            parameter_values: Concrete parameters to be bound.

            **kwargs: Individual options to overwrite the default primitive options.
                These include the runtime options in :class:`qiskit_ibm_runtime.RuntimeOptions`.

        Returns:
            Submitted job.
            The result of the job is an instance of :class:`qiskit.primitives.EstimatorResult`.

        Raises:
            ValueError: Invalid arguments are given.
        """
        # To bypass base class merging of options.
        user_kwargs = {"_user_kwargs": kwargs}
        return super().run(
            circuits=circuits,
            observables=observables,
            parameter_values=parameter_values,
            **user_kwargs,
        )

    def _run(  # pylint: disable=arguments-differ
        self,
        circuits: tuple[QuantumCircuit, ...],
        observables: tuple[SparsePauliOp, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **kwargs: Any,
    ) -> RuntimeJob:
        """Submit a request to the estimator primitive.

        Args:
            circuits: a (parameterized) :class:`~qiskit.circuit.QuantumCircuit` or
                a list of (parameterized) :class:`~qiskit.circuit.QuantumCircuit`.

            observables: A list of observable objects.

            parameter_values: An optional list of concrete parameters to be bound.

            **kwargs: Individual options to overwrite the default primitive options.
                These include the runtime options in :class:`~qiskit_ibm_runtime.RuntimeOptions`.

        Returns:
            Submitted job
        """
        inputs = {
            "circuits": circuits,
            "observables": observables,
            "parameters": [circ.parameters for circ in circuits],
            "parameter_values": parameter_values,
        }
        return self._run_primitive(
            primitive_inputs=inputs, user_kwargs=kwargs.get("_user_kwargs", {})
        )

    def _validate_options(self, options: dict) -> None:
        """Validate that program inputs (options) are valid
        Raises:
            ValueError: if resilience_level is out of the allowed range.
            ValueError: if resilience_level==3, backend is simulator and no coupling map
        """
        if os.getenv("QISKIT_RUNTIME_SKIP_OPTIONS_VALIDATION"):
            return

        if self._service._channel_strategy == "q-ctrl":
            qctrl_validate(options)
            return

        if not options.get("resilience_level") in list(
            range(Options._MAX_RESILIENCE_LEVEL_ESTIMATOR + 1)
        ):
            raise ValueError(
                f"resilience_level can only take the values "
                f"{list(range(Options._MAX_RESILIENCE_LEVEL_ESTIMATOR + 1))} in Estimator"
            )

        if (
            options.get("resilience_level") == 3
            and self._backend
            and self._backend.configuration().simulator
        ):
            if not options.get("simulator").get("coupling_map"):
                raise ValueError(
                    "When the backend is a simulator and resilience_level == 3,"
                    "a coupling map is required."
                )
        Options.validate_options(options)

    @classmethod
    def _program_id(cls) -> str:
        """Return the program ID."""
        return "estimator"
