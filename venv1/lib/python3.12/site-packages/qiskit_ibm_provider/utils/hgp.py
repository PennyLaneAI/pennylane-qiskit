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

"""Hub/group/project utility functions."""

from typing import Tuple
from ..exceptions import IBMInputValueError


def from_instance_format(instance: str) -> Tuple[str, str, str]:
    """Convert the input instance to [hub, group, project].

    Args:
        instance: Service instance in hub/group/project format.

    Returns:
        Hub, group, and project.

    Raises:
        IBMInputValueError: If input is not in the correct format.
    """
    try:
        hub, group, project = instance.split("/")
        return hub, group, project
    except (ValueError, AttributeError):
        raise IBMInputValueError(
            f"Input instance value {instance} is not in the"
            f"correct hub/group/project format."
        )


def to_instance_format(hub: str, group: str, project: str) -> str:
    """Convert input to hub/group/project format."""
    return f"{hub}/{group}/{project}"
