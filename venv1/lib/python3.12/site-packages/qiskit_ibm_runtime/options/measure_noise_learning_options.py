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

"""Options for measurement noise learning."""

from typing import Union, Literal

from .utils import Unset, UnsetType, primitive_dataclass, make_constraint_validator


@primitive_dataclass
class MeasureNoiseLearningOptions:
    """Options for measurement noise learning.

    .. note::
        These options are only used when the resilience level or options specify a
        technique that requires measurement noise learning.

    Args:
        num_randomizations: The number of random circuits to draw for the measurement
            learning experiment. Default: 32.

        shots_per_randomization: The number of shots to use for the learning experiment
            per random circuit. If "auto", the value will be chosen automatically
            based on the input PUBs. Default: "auto".
    """

    num_randomizations: Union[UnsetType, int] = Unset
    shots_per_randomization: Union[UnsetType, int, Literal["auto"]] = Unset

    _ge1 = make_constraint_validator("num_randomizations", "shots_per_randomization", ge=1)
