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

"""Utilities related to Qobj."""

from typing import Dict, Union

from qiskit.qobj import QasmQobj, PulseQobj

from .json_decoder import decode_pulse_qobj


def dict_to_qobj(qobj_dict: Dict) -> Union[QasmQobj, PulseQobj]:
    """Convert a Qobj in dictionary format to an instance.

    Args:
        qobj_dict: Qobj in dictionary format.

    Returns:
        The corresponding QasmQobj or PulseQobj instance.
    """
    if qobj_dict["type"] == "PULSE":
        decode_pulse_qobj(qobj_dict)  # Convert to proper types.
        return PulseQobj.from_dict(qobj_dict)
    return QasmQobj.from_dict(qobj_dict)
