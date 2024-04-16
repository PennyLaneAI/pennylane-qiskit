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

"""Sampler options."""

from typing import Union

from pydantic import Field

from .utils import Dict, Unset, UnsetType
from .sampler_execution_options import SamplerExecutionOptionsV2
from .options import OptionsV2
from .utils import primitive_dataclass
from .dynamical_decoupling_options import DynamicalDecouplingOptions
from .twirling_options import TwirlingOptions


@primitive_dataclass
class SamplerOptions(OptionsV2):
    """Options for V2 Sampler.

    Args:
        default_shots: The default number of shots to use if none are specified in the PUBs
            or in the run method. Default: 4096.

        dynamical_decoupling: Suboptions for dynamical decoupling. See
            :class:`DynamicalDecouplingOptions` for all available options.

        execution: Execution time options. See :class:`ExecutionOptionsV2` for all available options.

        twirling: Pauli twirling options. See :class:`TwirlingOptions` for all available options.

        experimental: Experimental options.
    """

    # Sadly we cannot use pydantic's built in validation because it won't work on Unset.
    default_shots: Union[UnsetType, int] = Unset
    dynamical_decoupling: Union[DynamicalDecouplingOptions, Dict] = Field(
        default_factory=DynamicalDecouplingOptions
    )
    execution: Union[SamplerExecutionOptionsV2, Dict] = Field(
        default_factory=SamplerExecutionOptionsV2
    )
    twirling: Union[TwirlingOptions, Dict] = Field(default_factory=TwirlingOptions)
    experimental: Union[UnsetType, dict] = Unset
