# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Estimator result decoder."""

from typing import Dict, Union
import numpy as np

from qiskit.primitives import EstimatorResult
from qiskit.primitives.containers import PrimitiveResult

from .result_decoder import ResultDecoder


class EstimatorResultDecoder(ResultDecoder):
    """Class used to decode estimator results"""

    @classmethod
    def decode(  # type: ignore # pylint: disable=arguments-differ
        cls, raw_result: str
    ) -> Union[EstimatorResult, PrimitiveResult]:
        """Convert the result to EstimatorResult."""
        decoded: Dict = super().decode(raw_result)
        if isinstance(decoded, PrimitiveResult):
            return decoded
        else:
            return EstimatorResult(
                values=np.asarray(decoded["values"]),
                metadata=decoded["metadata"],
            )
