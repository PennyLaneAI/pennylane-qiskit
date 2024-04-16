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

"""Constant values."""

from qiskit.providers.jobstatus import JobStatus

from .utils.result_decoder import ResultDecoder
from .utils.estimator_result_decoder import EstimatorResultDecoder
from .utils.sampler_result_decoder import SamplerResultDecoder
from .utils.runner_result import RunnerResult


QISKIT_IBM_RUNTIME_API_URL = "https://auth.quantum-computing.ibm.com/api"

API_TO_JOB_STATUS = {
    "QUEUED": JobStatus.QUEUED,
    "RUNNING": JobStatus.RUNNING,
    "COMPLETED": JobStatus.DONE,
    "FAILED": JobStatus.ERROR,
    "CANCELLED": JobStatus.CANCELLED,
}

API_TO_JOB_ERROR_MESSAGE = {
    "FAILED": "Job {} has failed:\n{}",
    "CANCELLED - RAN TOO LONG": "Job {} ran longer than maximum execution time. "
    "Job was cancelled:\n{}",
}

DEFAULT_DECODERS = {
    "sampler": [ResultDecoder, SamplerResultDecoder],
    "estimator": [ResultDecoder, EstimatorResultDecoder],
    "circuit-runner": RunnerResult,
    "qasm3-runner": RunnerResult,
}
