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

"""Options for dynamical decoupling."""

from typing import Union, Literal

from .utils import Unset, UnsetType, primitive_dataclass


@primitive_dataclass
class DynamicalDecouplingOptions:
    """Options for dynamical decoupling (DD).

    Args:
        enable: Whether to enable DD as specified by the other options in this class.
            Default: False.

        sequence_type: Which dynamical decoupling sequence to use. Default: "XX".

            * ``"XX"``: use the sequence ``tau/2 - (+X) - tau - (+X) - tau/2``
            * ``"XpXm"``: use the sequence ``tau/2 - (+X) - tau - (-X) - tau/2``
            * ``"XY4"``: : use the sequence
              ``tau/2 - (+X) - tau - (+Y) - tau (-X) - tau - (-Y) - tau/2``

        extra_slack_distribution: Where to put extra timing delays due to rounding issues.
            Rounding issues arise because the discrete time step ``dt`` of the system cannot
            be divided. This option takes following values. Default: "middle".

            * ``"middle"``: Put the extra slack to the interval at the middle of the sequence.
            * ``"edges"``: Divide the extra slack as evenly as possible into intervals at
              beginning and end of the sequence.

        scheduling_method: Whether to schedule gates as soon as ("asap") or
            as late as ("alap") possible. Default: "alap".
    """

    enable: Union[UnsetType, bool] = Unset
    sequence_type: Union[UnsetType, Literal["XX", "XpXm", "XY4"]] = Unset
    extra_slack_distribution: Union[UnsetType, Literal["middle", "edges"]] = Unset
    scheduling_method: Union[UnsetType, Literal["alap", "asap"]] = Unset
