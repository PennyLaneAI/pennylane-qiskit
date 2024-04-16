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

"""Probabalistic error cancellation mitigation options."""

from typing import Union, Literal

from .utils import Unset, UnsetType, primitive_dataclass, make_constraint_validator


@primitive_dataclass
class PecOptions:
    """Probabalistic error cancellation mitigation options.

    Args:
        max_overhead: The maximum circuit sampling overhead allowed, or
            ``None`` for no maximum. Default: 100.

        noise_gain: The amount by which to scale the noise, where:

            * A value of 0 corresponds to removing the full learned noise.
            * A value of 1 corresponds to no removal of the learned noise.
            * A value between 0 and 1 corresponds to partially removing the learned noise.
            * A value greater than one corresponds to amplifying the learned noise.

            If "auto", the value in the range ``[0, 1)`` will be chosen automatically
            for each input PUB based on the learned noise strength, ``max_overhead``,
            and the depth of the PUB. Default: "auto".
    """

    max_overhead: Union[UnsetType, float, None] = Unset
    noise_gain: Union[UnsetType, float, Literal["auto"]] = Unset

    _gt0 = make_constraint_validator("max_overhead", gt=0)
    _ge0 = make_constraint_validator("noise_gain", ge=0)
