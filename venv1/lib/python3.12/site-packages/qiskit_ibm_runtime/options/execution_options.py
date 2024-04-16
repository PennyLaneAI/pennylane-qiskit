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

"""Execution options."""

from typing import Union

from .utils import Unset, UnsetType, primitive_dataclass


@primitive_dataclass
class ExecutionOptionsV2:
    """Execution options for V2 primitives.

    Args:

        init_qubits: Whether to reset the qubits to the ground state for each shot.
            Default: ``True``.

        rep_delay: The repetition delay. This is the delay between a measurement and
            the subsequent quantum circuit. This is only supported on backends that have
            ``backend.dynamic_reprate_enabled=True``. It must be from the
            range supplied by ``backend.rep_delay_range``.
            Default is given by ``backend.default_rep_delay``.
    """

    init_qubits: Union[UnsetType, bool] = Unset
    rep_delay: Union[UnsetType, float] = Unset


@primitive_dataclass
class ExecutionOptions:
    """Execution options for V1 primitives.

    Args:
        shots: Number of repetitions of each circuit, for sampling. Default: 4000.

        init_qubits: Whether to reset the qubits to the ground state for each shot.
            Default: ``True``.
    """

    shots: int = 4000
    init_qubits: bool = True
