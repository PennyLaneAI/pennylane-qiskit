# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qiskit runtime service."""

from __future__ import annotations

import logging
import copy
from typing import Dict, Union, Literal
import warnings
from dataclasses import asdict

from qiskit.utils import optionals
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.primitives import BackendSampler, BackendEstimator
from qiskit.primitives.primitive_job import PrimitiveJob

from ..runtime_options import RuntimeOptions
from ..ibm_backend import IBMBackend
from ..qiskit.primitives import BackendEstimatorV2, BackendSamplerV2  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


class QiskitRuntimeLocalService:
    """Class for local testing mode."""

    def __init__(
        self,
    ) -> None:
        """QiskitRuntimeLocalService constructor.


        Returns:
            An instance of QiskitRuntimeService.

        """
        self._channel_strategy = None

    def run(
        self,
        program_id: Literal["sampler", "estimator"],
        inputs: Dict,
        options: Union[RuntimeOptions, Dict],
    ) -> PrimitiveJob:
        """Execute the runtime program.

        Args:
            program_id: Program ID.
            inputs: Program input parameters. These input values are passed
                to the runtime program.
            options: Runtime options that control the execution environment.
                See :class:`RuntimeOptions` for all available options.

        Returns:
            A job representing the execution.

        Raises:
            ValueError: If input is invalid.
            NotImplementedError: If using V2 primitives.
        """
        if isinstance(options, Dict):
            qrt_options = copy.deepcopy(options)
        else:
            qrt_options = asdict(options)

        backend = qrt_options.pop("backend", None)

        if program_id not in ["sampler", "estimator"]:
            raise ValueError("Only sampler and estimator are supported in local testing mode.")
        if isinstance(backend, IBMBackend):
            raise ValueError(
                "Local testing mode is not supported when a cloud-based backend is used."
            )
        if isinstance(backend, str):
            raise ValueError(
                "Passing a backend name is not supported in local testing mode. "
                "Please pass a backend instance."
            )

        inputs = copy.deepcopy(inputs)
        primitive_version = inputs.pop("version", 1)
        if primitive_version == 1:
            primitive_inputs = {
                "circuits": inputs.pop("circuits"),
                "parameter_values": inputs.pop("parameter_values"),
            }
            if program_id == "estimator":
                primitive_inputs["observables"] = inputs.pop("observables")
            inputs.pop("parameters", None)

            if optionals.HAS_AER:
                # pylint: disable=import-outside-toplevel
                from qiskit_aer.backends.aerbackend import AerBackend

                if isinstance(backend, AerBackend):
                    return self._run_aer_primitive_v1(
                        primitive=program_id, options=inputs, inputs=primitive_inputs
                    )

            return self._run_backend_primitive_v1(
                backend=backend,
                primitive=program_id,
                options=inputs,
                inputs=primitive_inputs,
            )
        else:
            primitive_inputs = {"pubs": inputs.pop("pubs")}
            return self._run_backend_primitive_v2(
                backend=backend,
                primitive=program_id,
                options=inputs.get("options", {}),
                inputs=primitive_inputs,
            )

    def _run_aer_primitive_v1(
        self, primitive: Literal["sampler", "estimator"], options: dict, inputs: dict
    ) -> PrimitiveJob:
        """Run V1 Aer primitive.

        Args:
            primitive: Name of the primitive.
            options: Primitive options to use.
            inputs: Primitive inputs.

        Returns:
            The job object of the result of the primitive.
        """
        # pylint: disable=import-outside-toplevel
        from qiskit_aer.primitives import Sampler, Estimator

        # TODO: issue warning if extra options are used
        options_copy = copy.deepcopy(options)
        transpilation_options = options_copy.get("transpilation_settings", {})
        skip_transpilation = transpilation_options.pop("skip_transpilation", False)
        optimization_level = transpilation_options.pop("optimization_settings", {}).pop(
            "level", None
        )
        transpilation_options["optimization_level"] = optimization_level
        input_run_options = options_copy.get("run_options", {})
        run_options = {
            "shots": input_run_options.pop("shots", None),
            "seed_simulator": input_run_options.pop("seed_simulator", None),
        }
        backend_options = {"noise_model": input_run_options.pop("noise_model", None)}

        if primitive == "sampler":
            primitive_inst = Sampler(
                backend_options=backend_options,
                transpile_options=transpilation_options,
                run_options=run_options,
                skip_transpilation=skip_transpilation,
            )
        else:
            primitive_inst = Estimator(
                backend_options=backend_options,
                transpile_options=transpilation_options,
                run_options=run_options,
                skip_transpilation=skip_transpilation,
            )
        return primitive_inst.run(**inputs)

    def _run_backend_primitive_v1(
        self,
        backend: BackendV1 | BackendV2,
        primitive: Literal["sampler", "estimator"],
        options: dict,
        inputs: dict,
    ) -> PrimitiveJob:
        """Run V1 backend primitive.

        Args:
            backend: The backend to run the primitive on.
            primitive: Name of the primitive.
            options: Primitive options to use.
            inputs: Primitive inputs.

        Returns:
            The job object of the result of the primitive.
        """
        options_copy = copy.deepcopy(options)
        transpilation_options = options_copy.get("transpilation_settings", {})
        skip_transpilation = transpilation_options.pop("skip_transpilation", False)
        optimization_level = transpilation_options.pop("optimization_settings", {}).get("level")
        transpilation_options["optimization_level"] = optimization_level
        input_run_options = options.get("run_options", {})
        run_options = {
            "shots": input_run_options.get("shots"),
            "seed_simulator": input_run_options.get("seed_simulator"),
            "noise_model": input_run_options.get("noise_model"),
        }
        if primitive == "sampler":
            primitive_inst = BackendSampler(backend=backend, skip_transpilation=skip_transpilation)
        else:
            primitive_inst = BackendEstimator(backend=backend)

        primitive_inst.set_transpile_options(**transpilation_options)
        return primitive_inst.run(**inputs, **run_options)

    def _run_backend_primitive_v2(
        self,
        backend: BackendV1 | BackendV2,
        primitive: Literal["sampler", "estimator"],
        options: dict,
        inputs: dict,
    ) -> PrimitiveJob:
        """Run V2 backend primitive.

        Args:
            backend: The backend to run the primitive on.
            primitive: Name of the primitive.
            options: Primitive options to use.
            inputs: Primitive inputs.

        Returns:
            The job object of the result of the primitive.
        """
        options_copy = copy.deepcopy(options)

        prim_options = {}
        if seed_simulator := options_copy.pop("simulator", {}).pop("seed_simulator", None):
            prim_options["seed_simulator"] = seed_simulator
        if primitive == "sampler":
            if default_shots := options_copy.pop("default_shots", None):
                prim_options["default_shots"] = default_shots
            primitive_inst = BackendSamplerV2(backend=backend, options=prim_options)
        else:
            if default_precision := options_copy.pop("default_precision", None):
                prim_options["default_precision"] = default_precision
            primitive_inst = BackendEstimatorV2(backend=backend, options=prim_options)

        if options_copy:
            warnings.warn(f"Options {options_copy} have no effect in local testing mode.")

        return primitive_inst.run(**inputs)
