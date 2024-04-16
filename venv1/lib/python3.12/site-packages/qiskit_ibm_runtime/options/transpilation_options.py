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

"""Transpilation options."""

from typing import List, Union, Literal

from pydantic import field_validator

from .utils import Unset, UnsetType, skip_unset_validation, primitive_dataclass


LayoutMethodType = Literal[
    "trivial",
    "dense",
    "noise_adaptive",
    "sabre",
]
RoutingMethodType = Literal[
    "basic",
    "lookahead",
    "stochastic",
    "sabre",
    "none",
]
MAX_OPTIMIZATION_LEVEL: int = 1


@primitive_dataclass
class TranspilationOptions:
    """Transpilation options.

    Args:

        skip_transpilation: Whether to skip transpilation. Default is False.

        initial_layout: Initial position of virtual qubits on physical qubits.
            See ``qiskit.compiler.transpile`` for more information.

        layout_method: Name of layout selection pass. One of
            'trivial', 'dense', 'noise_adaptive', 'sabre'.

        routing_method: Name of routing pass.
            One of 'basic', 'lookahead', 'stochastic', 'sabre', 'none'.

        approximation_degree: heuristic dial used for circuit approximation
            (1.0=no approximation, 0.0=maximal approximation)
    """

    skip_transpilation: bool = False
    initial_layout: Union[UnsetType, dict, List] = Unset  # TODO: Support Layout
    layout_method: Union[UnsetType, LayoutMethodType] = Unset
    routing_method: Union[UnsetType, RoutingMethodType] = Unset
    approximation_degree: Union[UnsetType, float] = Unset

    @field_validator("approximation_degree")
    @classmethod
    @skip_unset_validation
    def _validate_approximation_degree(cls, degree: float) -> float:
        """Validate approximation_degree."""
        if not 0.0 <= degree <= 1.0:
            raise ValueError(
                "approximation_degree must be between 0.0 (maximal approximation) "
                "and 1.0 (no approximation)"
            )
        return degree
