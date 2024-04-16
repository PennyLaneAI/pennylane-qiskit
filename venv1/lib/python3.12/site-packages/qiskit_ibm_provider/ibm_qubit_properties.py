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

"""Module for Qubit Properties of an IBM Quantum Backend."""

from qiskit.providers.backend import QubitProperties


class IBMQubitProperties(QubitProperties):
    """A representation of the properties of a qubit on an IBM backend."""

    __slots__ = (  # pylint: disable=redefined-slots-in-subclass
        "t1",
        "t2",
        "frequency",
        "anharmonicity",
        "operational",
    )

    def __init__(  # type: ignore[no-untyped-def]
        self,
        t1=None,
        t2=None,
        frequency=None,
        anharmonicity=None,
        operational=True,
    ):
        """Create a new ``IBMQubitProperties`` object

        Args:
            t1: The T1 time for a qubit in secs
            t2: The T2 time for a qubit in secs
            frequency: The frequency of a qubit in Hz
            anharmonicity: The anharmonicity of a qubit in Hz
            operational: A boolean value representing if this qubit is operational.
        """
        super().__init__(t1=t1, t2=t2, frequency=frequency)
        self.anharmonicity = anharmonicity
        self.operational = operational

    def __repr__(self):  # type: ignore[no-untyped-def]
        return (
            f"IBMQubitProperties(t1={self.t1}, t2={self.t2}, frequency={self.frequency}, "
            f"anharmonicity={self.anharmonicity})"
        )
