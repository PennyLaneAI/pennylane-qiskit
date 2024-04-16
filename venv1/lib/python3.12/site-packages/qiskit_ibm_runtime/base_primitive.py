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

"""Base class for Qiskit Runtime primitives."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Union, TypeVar, Generic, Type
import copy
import logging
from dataclasses import asdict, replace
import warnings

from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.providers.options import Options as TerraOptions
from qiskit.providers.backend import BackendV1, BackendV2

from .provider_session import get_cm_session as get_cm_provider_session

from .options import Options
from .options.options import BaseOptions, OptionsV2
from .options.utils import merge_options, set_default_error_levels
from .runtime_job import RuntimeJob
from .runtime_job_v2 import RuntimeJobV2
from .ibm_backend import IBMBackend
from .utils.default_session import get_cm_session
from .utils.deprecation import issue_deprecation_msg
from .utils.utils import validate_isa_circuits, is_simulator, validate_no_dd_with_dynamic_circuits
from .constants import DEFAULT_DECODERS
from .qiskit_runtime_service import QiskitRuntimeService
from .fake_provider.local_service import QiskitRuntimeLocalService


# pylint: disable=unused-import,cyclic-import
from .session import Session

logger = logging.getLogger(__name__)
OptionsT = TypeVar("OptionsT", bound=BaseOptions)


class BasePrimitiveV2(ABC, Generic[OptionsT]):
    """Base class for Qiskit Runtime primitives."""

    _options_class: Type[OptionsT] = OptionsV2  # type: ignore[assignment]
    version = 2

    def __init__(
        self,
        backend: Optional[Union[str, BackendV1, BackendV2]] = None,
        session: Optional[Session] = None,
        options: Optional[Union[Dict, OptionsT]] = None,
    ):
        """Initializes the primitive.

        Args:

            backend: Backend to run the primitive. This can be a backend name or a ``Backend``
                instance. If a name is specified, the default account (e.g. ``QiskitRuntimeService()``)
                is used.

            session: Session in which to call the primitive.

                If both ``session`` and ``backend`` are specified, ``session`` takes precedence.
                If neither is specified, and the primitive is created inside a
                :class:`qiskit_ibm_runtime.Session` context manager, then the session is used.
                Otherwise if IBM Cloud channel is used, a default backend is selected.

            options: Primitive options, see :class:`qiskit_ibm_runtime.options.EstimatorOptions`
                and :class:`qiskit_ibm_runtime.options.SamplerOptions` for detailed description
                on estimator and sampler options, respectively.

        Raises:
            ValueError: Invalid arguments are given.
        """
        self._session: Optional[Session] = None
        self._service: QiskitRuntimeService | QiskitRuntimeLocalService = None
        self._backend: Optional[BackendV1 | BackendV2] = None

        self._set_options(options)

        if isinstance(session, Session):
            self._session = session
            self._service = self._session.service
            self._backend = self._session._backend
            return
        elif session is not None:  # type: ignore[unreachable]
            raise ValueError("session must be of type Session or None")

        if isinstance(backend, IBMBackend):  # type: ignore[unreachable]
            self._service = backend.service
            self._backend = backend
        elif isinstance(backend, (BackendV1, BackendV2)):
            self._service = QiskitRuntimeLocalService()
            self._backend = backend
        elif isinstance(backend, str):
            self._service = (
                QiskitRuntimeService()
                if QiskitRuntimeService.global_service is None
                else QiskitRuntimeService.global_service
            )
            self._backend = self._service.backend(backend)
        elif get_cm_session():
            self._session = get_cm_session()
            self._service = self._session.service
            self._backend = self._service.backend(
                name=self._session.backend(), instance=self._session._instance
            )
        else:
            self._service = (
                QiskitRuntimeService()
                if QiskitRuntimeService.global_service is None
                else QiskitRuntimeService.global_service
            )
            if self._service.channel != "ibm_cloud":
                raise ValueError(
                    "A backend or session must be specified when not using ibm_cloud channel."
                )
            issue_deprecation_msg(
                "Not providing a backend is deprecated",
                "0.22.0",
                "Passing in a backend will be required, please provide a backend.",
                3,
            )

    def _run(self, pubs: Union[list[EstimatorPub], list[SamplerPub]]) -> RuntimeJobV2:
        """Run the primitive.

        Args:
            pubs: Inputs PUBs to pass to the primitive.

        Returns:
            Submitted job.
        """
        primitive_inputs = {"pubs": pubs}
        options_dict = asdict(self.options)
        self._validate_options(options_dict)
        primitive_options = self._options_class._get_program_inputs(options_dict)
        primitive_inputs.update(primitive_options)
        runtime_options = self._options_class._get_runtime_options(options_dict)

        validate_no_dd_with_dynamic_circuits([pub.circuit for pub in pubs], self.options)
        if self._backend:
            for pub in pubs:
                if getattr(self._backend, "target", None) and not is_simulator(self._backend):
                    validate_isa_circuits([pub.circuit], self._backend.target)

                if isinstance(self._backend, IBMBackend):
                    self._backend.check_faulty(pub.circuit)

        logger.info("Submitting job using options %s", primitive_options)

        if self._session:
            return self._session.run(
                program_id=self._program_id(),
                inputs=primitive_inputs,
                options=runtime_options,
                callback=options_dict.get("environment", {}).get("callback", None),
                result_decoder=DEFAULT_DECODERS.get(self._program_id()),
            )

        if self._backend:
            runtime_options["backend"] = self._backend
            if "instance" not in runtime_options and isinstance(self._backend, IBMBackend):
                runtime_options["instance"] = self._backend._instance

        if isinstance(self._service, QiskitRuntimeService):
            return self._service.run(
                program_id=self._program_id(),
                options=runtime_options,
                inputs=primitive_inputs,
                callback=options_dict.get("environment", {}).get("callback", None),
                result_decoder=DEFAULT_DECODERS.get(self._program_id()),
            )

        return self._service.run(
            program_id=self._program_id(),  # type: ignore[arg-type]
            options=runtime_options,
            inputs=primitive_inputs,
        )

    @property
    def session(self) -> Optional[Session]:
        """Return session used by this primitive.

        Returns:
            Session used by this primitive, or ``None`` if session is not used.
        """
        return self._session

    @property
    def options(self) -> OptionsT:
        """Return options"""
        return self._options

    def _set_options(self, options: Optional[Union[Dict, OptionsT]] = None) -> None:
        """Set options."""
        if options is None:
            self._options = self._options_class()
        elif isinstance(options, dict):
            default_options = self._options_class()
            self._options = self._options_class(**merge_options(default_options, options))
        elif isinstance(options, self._options_class):
            self._options = replace(options)
        else:
            raise TypeError(
                f"Invalid 'options' type. It can only be a dictionary of {self._options_class}"
            )

    @abstractmethod
    def _validate_options(self, options: dict) -> None:
        """Validate that program inputs (options) are valid

        Raises:
            ValueError: if resilience_level is out of the allowed range.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def _program_id(cls) -> str:
        """Return the program ID."""
        raise NotImplementedError()


class BasePrimitiveV1(ABC):
    """Base class for Qiskit Runtime primitives."""

    version = 1

    def __init__(
        self,
        backend: Optional[Union[str, BackendV1, BackendV2]] = None,
        session: Optional[Session] = None,
        options: Optional[Union[Dict, Options]] = None,
    ):
        """Initializes the primitive.

        Args:

            backend: Backend to run the primitive. This can be a backend name or a ``Backend``
                instance. If a name is specified, the default account (e.g. ``QiskitRuntimeService()``)
                is used.

            session: Session in which to call the primitive.

                If both ``session`` and ``backend`` are specified, ``session`` takes precedence.
                If neither is specified, and the primitive is created inside a
                :class:`qiskit_ibm_runtime.Session` context manager, then the session is used.
                Otherwise if IBM Cloud channel is used, a default backend is selected.

            options: Primitive options, see :class:`Options` for detailed description.
                The ``backend`` keyword is still supported but is deprecated.

        Raises:
            ValueError: Invalid arguments are given.
        """
        # `self._options` in this class is a Dict.
        # The base class, however, uses a `_run_options` which is an instance of
        # qiskit.providers.Options. We largely ignore this _run_options because we use
        # a nested dictionary to categorize options.
        self._session: Optional[Session] = None
        self._service: QiskitRuntimeService | QiskitRuntimeLocalService = None
        self._backend: Optional[BackendV1 | BackendV2] = None

        issue_deprecation_msg(
            "The Sampler and Estimator V1 primitives have been deprecated",
            "0.23.0",
            "Please use the V2 Primitives. See the `V2 migration guide "
            "<https://docs.quantum.ibm.com/api/migration-guides/v2-primitives>`_. for more details",
            3,
        )

        if options is None:
            self._options = asdict(Options())
        elif isinstance(options, Options):
            self._options = asdict(copy.deepcopy(options))
        else:
            options_copy = copy.deepcopy(options)
            default_options = asdict(Options())
            self._options = merge_options(default_options, options_copy)

        if isinstance(session, Session):
            self._session = session
            self._service = self._session.service
            self._backend = self._session._backend
            return
        elif session is not None:  # type: ignore[unreachable]
            raise ValueError("session must be of type Session or None")

        if isinstance(backend, IBMBackend):  # type: ignore[unreachable]
            self._service = backend.service
            self._backend = backend
        elif isinstance(backend, (BackendV1, BackendV2)):
            self._service = QiskitRuntimeLocalService()
            self._backend = backend
        elif isinstance(backend, str):
            self._service = (
                QiskitRuntimeService()
                if QiskitRuntimeService.global_service is None
                else QiskitRuntimeService.global_service
            )
            self._backend = self._service.backend(backend)
        elif get_cm_session():
            self._session = get_cm_session()
            self._service = self._session.service
            self._backend = self._service.backend(
                name=self._session.backend(), instance=self._session._instance
            )
        else:
            self._service = (
                QiskitRuntimeService()
                if QiskitRuntimeService.global_service is None
                else QiskitRuntimeService.global_service
            )
            if self._service.channel != "ibm_cloud":
                raise ValueError(
                    "A backend or session must be specified when not using ibm_cloud channel."
                )
            issue_deprecation_msg(
                "Not providing a backend is deprecated",
                "0.21.0",
                "Passing in a backend will be required, please provide a backend.",
                3,
            )
        # Check if initialized within a IBMBackend session. If so, issue a warning.
        if get_cm_provider_session():
            warnings.warn(
                "A Backend.run() session is open but Primitives will not be run within this session"
            )

    def _run_primitive(self, primitive_inputs: Dict, user_kwargs: Dict) -> RuntimeJob:
        """Run the primitive.

        Args:
            primitive_inputs: Inputs to pass to the primitive.
            user_kwargs: Individual options to overwrite the default primitive options.

        Returns:
            Submitted job.
        """
        # TODO: Don't check service / backend
        if (
            self._backend  # pylint: disable=too-many-boolean-expressions
            and isinstance(self._backend, IBMBackend)
            and isinstance(self._backend.service, QiskitRuntimeService)
            and not self._backend.simulator
            and getattr(self._backend, "target", None)
            and self._service._channel_strategy != "q-ctrl"
        ):
            validate_isa_circuits(primitive_inputs["circuits"], self._backend.target)

        combined = Options._merge_options(self._options, user_kwargs)

        if self._backend:
            combined = set_default_error_levels(
                combined,
                self._backend,
                Options._DEFAULT_OPTIMIZATION_LEVEL,
                Options._DEFAULT_RESILIENCE_LEVEL,
            )
        else:
            combined["optimization_level"] = Options._DEFAULT_OPTIMIZATION_LEVEL
            combined["resilience_level"] = Options._DEFAULT_RESILIENCE_LEVEL

        self._validate_options(combined)

        combined = Options._set_default_resilience_options(combined)
        combined = Options._remove_none_values(combined)

        primitive_inputs.update(Options._get_program_inputs(combined))

        if (
            isinstance(self._backend, IBMBackend)
            and combined["transpilation"]["skip_transpilation"]
        ):
            for circ in primitive_inputs["circuits"]:
                self._backend.check_faulty(circ)

        logger.info("Submitting job using options %s", combined)

        runtime_options = Options._get_runtime_options(combined)
        if self._session:
            return self._session.run(
                program_id=self._program_id(),
                inputs=primitive_inputs,
                options=runtime_options,
                callback=combined.get("environment", {}).get("callback", None),
                result_decoder=DEFAULT_DECODERS.get(self._program_id()),
            )

        if self._backend:
            runtime_options["backend"] = self._backend
            if "instance" not in runtime_options and isinstance(self._backend, IBMBackend):
                runtime_options["instance"] = self._backend._instance

        if isinstance(self._service, QiskitRuntimeService):
            return self._service.run(
                program_id=self._program_id(),  # type: ignore[arg-type]
                options=runtime_options,
                inputs=primitive_inputs,
                callback=combined.get("environment", {}).get("callback", None),
                result_decoder=DEFAULT_DECODERS.get(self._program_id()),
            )
        return self._service.run(  # type: ignore[call-arg]
            program_id=self._program_id(),  # type: ignore[arg-type]
            options=runtime_options,
            inputs=primitive_inputs,
        )

    @property
    def session(self) -> Optional[Session]:
        """Return session used by this primitive.

        Returns:
            Session used by this primitive, or ``None`` if session is not used.
        """
        return self._session

    @property
    def options(self) -> TerraOptions:
        """Return options values for the sampler.
        Returns:
            options
        """
        return TerraOptions(**self._options)

    def set_options(self, **fields: Any) -> None:
        """Set options values for the sampler.

        Args:
            **fields: The fields to update the options
        """
        self._options = merge_options(  # pylint: disable=attribute-defined-outside-init
            self._options, fields
        )

    @abstractmethod
    def _validate_options(self, options: dict) -> None:
        """Validate that program inputs (options) are valid

        Raises:
            ValueError: if resilience_level is out of the allowed range.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def _program_id(cls) -> str:
        """Return the program ID."""
        raise NotImplementedError()
