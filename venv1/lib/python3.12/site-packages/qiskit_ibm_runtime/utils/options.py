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

"""Backend run options."""

from dataclasses import asdict, dataclass
from typing import Dict, Union, Any, Optional
from qiskit.circuit import QuantumCircuit
from qiskit.qobj.utils import MeasLevel, MeasReturnType


@dataclass
class CommonOptions:
    """Options common for both paths."""

    shots: int = 4000
    meas_level: Union[int, MeasLevel] = MeasLevel.CLASSIFIED
    init_qubits: bool = True
    rep_delay: Optional[float] = None
    memory: bool = False
    meas_return: Union[str, MeasReturnType] = MeasReturnType.AVERAGE

    def to_transport_dict(self) -> Dict[str, Any]:
        """Remove None values so runtime defaults are used."""
        dict_ = asdict(self)
        for key in list(dict_.keys()):
            if dict_[key] is None:
                del dict_[key]
        return dict_


@dataclass
class QASM3Options(CommonOptions):
    """Options for the QASM3 path."""

    init_circuit: Optional[QuantumCircuit] = None
    init_num_resets: Optional[int] = None


@dataclass
class QASM2Options(CommonOptions):
    """Options for the QASM2 path."""

    header: Optional[Dict] = None
    init_qubits: bool = True
    use_measure_esp: Optional[bool] = None
    # Simulator only
    noise_model: Any = None
    seed_simulator: Optional[int] = None
