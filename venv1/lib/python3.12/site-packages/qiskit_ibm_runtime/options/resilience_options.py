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

"""Resilience options."""

from typing import Sequence, Literal, Union, Optional
from dataclasses import asdict

from pydantic import model_validator, Field

from .utils import Unset, UnsetType, Dict, primitive_dataclass
from .measure_noise_learning_options import MeasureNoiseLearningOptions
from .zne_options import ZneOptions
from .pec_options import PecOptions
from .layer_noise_learning_options import LayerNoiseLearningOptions


NoiseAmplifierType = Literal[
    "LocalFoldingAmplifier",
]
ExtrapolatorType = Literal[
    "LinearExtrapolator",
    "QuadraticExtrapolator",
    "CubicExtrapolator",
    "QuarticExtrapolator",
]


@primitive_dataclass
class ResilienceOptionsV2:
    """Resilience options for V2 Estimator.

    Args:
        measure_mitigation: Whether to enable measurement error mitigation method.
            Further suboptions are available in :attr:`~measure_noise_learning`.
            Default: True.

        measure_noise_learning: Additional measurement noise learning options.
            See :class:`MeasureNoiseLearningOptions` for all options.

        zne_mitigation: Whether to turn on Zero Noise Extrapolation error mitigation method.
            Further suboptions are available in :attr:`~zne`.
            Default: False.

        zne: Additional zero noise extrapolation mitigation options.
            See :class:`ZneOptions` for all options.

        pec_mitigation: Whether to turn on Probabilistic Error Cancellation error mitigation method.
            Further suboptions are available in :attr:`~pec`.
            Default: False.

        pec: Additional probabalistic error cancellation mitigation options.
            See :class:`PecOptions` for all options.

        layer_noise_learning: Layer noise learning options.
            See :class:`LayerNoiseLearningOptions` for all options.
    """

    measure_mitigation: Union[UnsetType, bool] = Unset
    measure_noise_learning: Union[MeasureNoiseLearningOptions, Dict] = Field(
        default_factory=MeasureNoiseLearningOptions
    )
    zne_mitigation: Union[UnsetType, bool] = Unset
    zne: Union[ZneOptions, Dict] = Field(default_factory=ZneOptions)
    pec_mitigation: Union[UnsetType, bool] = Unset
    pec: Union[PecOptions, Dict] = Field(default_factory=PecOptions)
    layer_noise_learning: Union[LayerNoiseLearningOptions, Dict] = Field(
        default_factory=LayerNoiseLearningOptions
    )

    @model_validator(mode="after")
    def _validate_options(self) -> "ResilienceOptionsV2":
        """Validate the model."""
        if not self.measure_mitigation and any(asdict(self.measure_noise_learning).values()):
            raise ValueError(
                "'measure_noise_learning' options are set, but 'measure_mitigation' is not set to True."
            )

        if not self.zne_mitigation and any(asdict(self.zne).values()):
            raise ValueError("'zne' options are set, but 'zne_mitigation' is not set to True.")

        if not self.pec_mitigation and any(asdict(self.pec).values()):
            raise ValueError("'pec' options are set, but 'pec_mitigation' is not set to True.")

        # Validate not ZNE+PEC
        if self.pec_mitigation is True and self.zne_mitigation is True:
            raise ValueError(
                "pec_mitigation and zne_mitigation`options cannot be "
                "simultaneously enabled. Set one of them to False."
            )

        return self


@primitive_dataclass
class ResilienceOptions:
    """Resilience options for V1 primitives.

    Args:
        noise_factors: An list of real valued noise factors that determine by what amount the
            circuits' noise is amplified.
            Only applicable for ``resilience_level=2``.
            Default: ``None``, and (1, 3, 5) if resilience level is 2.

        noise_amplifier: A noise amplification strategy. Currently only
        ``"LocalFoldingAmplifier"`` is supported Only applicable for ``resilience_level=2``.
            Default: "LocalFoldingAmplifier".

        extrapolator: An extrapolation strategy. One of ``"LinearExtrapolator"``,
            ``"QuadraticExtrapolator"``, ``"CubicExtrapolator"``, ``"QuarticExtrapolator"``.
            Note that ``"CubicExtrapolator"`` and ``"QuarticExtrapolator"`` require more
            noise factors than the default.
            Only applicable for ``resilience_level=2``.
            Default: ``None``, and ``LinearExtrapolator`` if resilience level is 2.
    """

    noise_amplifier: Optional[NoiseAmplifierType] = None
    noise_factors: Optional[Sequence[float]] = None
    extrapolator: Optional[ExtrapolatorType] = None

    @model_validator(mode="after")
    def _validate_options(self) -> "ResilienceOptions":
        """Validate the model."""
        required_factors = {
            "QuarticExtrapolator": 5,
            "CubicExtrapolator": 4,
        }
        req_len = required_factors.get(self.extrapolator, None)
        if req_len and len(self.noise_factors) < req_len:
            raise ValueError(f"{self.extrapolator} requires at least {req_len} noise_factors.")

        return self
