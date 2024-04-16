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

"""Options for learning layer noise."""

from typing import Union, List

from pydantic import ValidationInfo, field_validator

from .utils import (
    Unset,
    UnsetType,
    primitive_dataclass,
    make_constraint_validator,
    skip_unset_validation,
)


@primitive_dataclass
class LayerNoiseLearningOptions:
    """Options for learning layer noise.

    .. note::
        These options are only used when the resilience level or options specify a
        technique that requires layer noise learning.

    Args:
        max_layers_to_learn: The max number of unique layers to learn.
            A ``None`` value indicates that there is no limit.
            If there are more unique layers present, then some layers will not be learned or
            mitigated. The learned layers are prioritized based on the number of times they
            occur in a set of run estimator PUBs, and for equally occurring layers are
            further sorted by the number of two-qubit gates in the layer. Default: 4.

        shots_per_randomization: The total number of shots to use per random learning circuit.
            A learning circuit is a random circuit at a specific learning depth with a specific
            measurement basis that is executed on hardware. Default: 128.

        num_randomizations: The number of random circuits to use per learning circuit
            configuration. A configuration is a measurement basis and depth setting.
            For example, if your experiment has six depths, and nine required measurement bases,
            then setting this value to 32 will result in a total of ``32 * 9 * 6`` circuits
            that need to be executed (at :attr:`~shots_per_randomization` each). Default: 32.

        layer_pair_depths: The circuit depths (measured in number of pairs) to use in learning
            experiments. Pairs are used as the unit because we exploit the order-2 nature of
            our entangling gates in the noise learning implementation. A value of ``3``
            would correspond to 6 layers of the layer of interest, for example.
            Default: (0, 1, 2, 4, 16, 32).
    """

    max_layers_to_learn: Union[UnsetType, int, None] = Unset
    shots_per_randomization: Union[UnsetType, int] = Unset
    num_randomizations: Union[UnsetType, int] = Unset
    layer_pair_depths: Union[UnsetType, List[int]] = Unset

    _ge0 = make_constraint_validator("max_layers_to_learn", ge=0)
    _ge1 = make_constraint_validator("shots_per_randomization", "num_randomizations", ge=1)

    @field_validator("layer_pair_depths", mode="after")
    @classmethod
    @skip_unset_validation
    def _nonnegative_list(cls, value: List[int], info: ValidationInfo) -> List[int]:
        if any(i < 0 for i in value):
            raise ValueError(f"`{cls.__name__}.{info.field_name}` option value must all be >= 0")
        return value
