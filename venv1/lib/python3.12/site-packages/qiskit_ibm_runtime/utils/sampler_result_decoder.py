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

"""Sampler result decoder."""

from typing import Dict, Union
from math import sqrt

from qiskit.result import QuasiDistribution
from qiskit.primitives import SamplerResult
from qiskit.primitives import PrimitiveResult

from .result_decoder import ResultDecoder


class SamplerResultDecoder(ResultDecoder):
    """Class used to decode sampler results."""

    @classmethod
    def decode(cls, raw_result: str) -> Union[SamplerResult, PrimitiveResult]:
        """Convert the result to SamplerResult."""
        decoded: Dict = super().decode(raw_result)

        if isinstance(decoded, PrimitiveResult):
            return decoded

        # TODO: Handle V2 result that is returned in dict format

        # V1 result
        quasi_dists = []
        for quasi, meta in zip(decoded["quasi_dists"], decoded["metadata"]):
            shots = meta.get("shots", float("inf"))
            overhead = meta.get("readout_mitigation_overhead", 1.0)

            # M3 mitigation overhead is gamma^2
            # https://github.com/Qiskit-Partners/mthree/blob/423d7e83a12491c59c9f58af46b75891bc622949/mthree/mitigation.py#L457
            #
            # QuasiDistribution stddev_upper_bound is gamma / sqrt(shots)
            # https://github.com/Qiskit/qiskit-terra/blob/ff267b5de8b83aef86e2c9ac6c7f918f58500505/qiskit/result/mitigation/local_readout_mitigator.py#L288
            stddev = sqrt(overhead / shots)
            quasi_dists.append(QuasiDistribution(quasi, shots=shots, stddev_upper_bound=stddev))
        return SamplerResult(
            quasi_dists=quasi_dists,
            metadata=decoded["metadata"],
        )
