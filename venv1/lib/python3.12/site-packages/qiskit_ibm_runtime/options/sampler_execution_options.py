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

"""Sampler Execution options."""
from typing import Literal, Union
from .execution_options import ExecutionOptionsV2
from .utils import primitive_dataclass, Unset, UnsetType


@primitive_dataclass
class SamplerExecutionOptionsV2(ExecutionOptionsV2):
    r"""Extension of :class:~.ExecutionOptionsV2` for the sampler primitive."""

    meas_type: Union[UnsetType, Literal["classified", "kerneled", "avg_kerneled"]] = Unset
    r"""How to process and return measurement results.

    This option sets the return type of all classical registers in all :class:`~.PubResult`\s.
    If a sampler pub with shape ``pub_shape`` has a circuit that contains a classical register 
    with size ``creg_size``, then the returned data associated with this register will have one of
    the following formats depending on the value of this option.

    * ``"classified"``: A :class:`~.BitArray` of shape ``pub_shape`` over ``num_shots`` with a 
      number of bits equal to ``creg_size``.

    * ``"kerneled"``: A complex NumPy array of shape ``(*pub_shape, num_shots, creg_size)``, where
      each entry represents an IQ data point (resulting from kerneling the measurement trace) in 
      arbitrary units.

    * ``"avg_kerneled"``: A complex NumPy array of shape ``(*pub_shape, creg_size)``, where
      each entry represents an IQ data point (resulting from kerneling the measurement trace and 
      averaging over shots) in arbitrary units. This option is equivalent to selecting
      ``"kerneled"`` and then averaging over the shots axis, but requires less data bandwidth.

    See `Here <https://pubs.aip.org/aip/rsi/article/88/10/104703/836456>`_ for 
    a description of kerneling.
    """
