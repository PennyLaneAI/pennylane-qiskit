# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qctrl validation functions and helpers."""

import logging
from typing import Any, Optional, Dict, List

from ..options import Options
from ..options import EnvironmentOptions, ExecutionOptions, TranspilationOptions, SimulatorOptions
from ..options.utils import UnsetType

logger = logging.getLogger(__name__)


def validate(options: Dict[str, Any]) -> None:
    """Validates the options for qctrl"""

    # Raise error on bad options.
    _raise_if_error_in_options(options)
    # Override options and warn.
    _warn_and_clean_options(options)

    # Default validation otherwise.
    TranspilationOptions(**options.get("transpilation", {}))
    execution_time = options.get("max_execution_time")
    if execution_time is not None:
        if execution_time > Options._MAX_EXECUTION_TIME:
            raise ValueError(
                f"max_execution_time must be below " f"{Options._MAX_EXECUTION_TIME} seconds."
            )

    EnvironmentOptions(**options.get("environment", {}))
    ExecutionOptions(**options.get("execution", {}))
    SimulatorOptions(**options.get("simulator", {}))


def _raise_if_error_in_options(options: Dict[str, Any]) -> None:
    """Checks for settings that produce errors and raise a ValueError"""

    # Fail on resilience_level set to 0
    resilience_level = options.get("resilience_level", 1)
    _check_argument(
        resilience_level > 0,
        description=(
            "Q-CTRL Primitives do not support resilience level 0. Please "
            "set resilience_level to 1 and re-try"
        ),
        arguments={},
    )

    optimization_level = options.get("optimization_level", 3)
    _check_argument(
        optimization_level > 0,
        description="Q-CTRL Primitives do not support optimization level 0. Please\
        set optimization_level to 3 and re-try",
        arguments={},
    )


def _warn_and_clean_options(options: Dict[str, Any]) -> None:
    """
    Validate and update transpilation settings
    """
    # Issue a warning and override if any of these setting is not None
    # or a different value than the default below
    expected_options = {
        "optimization_level": 3,
        "resilience_level": 1,
        "transpilation": {"approximation_degree": 0, "skip_transpilation": False},
        "resilience": {
            "noise_amplifier": None,
            "noise_factors": None,
            "extrapolator": None,
        },
    }

    # Collect keys with mis-matching values
    different_keys = _validate_values(expected_options, options)
    # Override options
    _update_values(expected_options, options)
    if different_keys:
        logger.warning(
            "The following settings cannot be customized and will be overwritten: %s",
            ",".join(sorted(different_keys)),
        )


def validate_v2(options: Dict[str, Any]) -> None:
    """Validates the options for qctrl"""

    # Raise error on bad options.
    _raise_if_error_in_options_v2(options)
    # Override options and warn.
    _warn_and_clean_options_v2(options)

    # Default validation otherwise.
    TranspilationOptions(**options.get("transpilation", {}))
    execution_time = options.get("max_execution_time")
    if execution_time is not None and not isinstance(execution_time, UnsetType):
        if execution_time > Options._MAX_EXECUTION_TIME:
            raise ValueError(
                f"max_execution_time must be below " f"{Options._MAX_EXECUTION_TIME} seconds."
            )

    EnvironmentOptions(**options.get("environment", {}))
    # ExecutionOptions(**options.get("execution", {}))
    SimulatorOptions(**options.get("simulator", {}))


def _raise_if_error_in_options_v2(options: Dict[str, Any]) -> None:
    """Checks for settings that produce errors and raise a ValueError"""

    # Fail on resilience_level set to 0
    resilience_level = options.get("resilience_level", 1)
    if isinstance(resilience_level, UnsetType):
        resilience_level = 1
    _check_argument(
        resilience_level > 0,
        description=(
            "Q-CTRL Primitives do not support resilience level 0. Please "
            "set resilience_level to 1 and re-try"
        ),
        arguments={},
    )

    optimization_level = options.get("optimization_level", 1)
    if isinstance(optimization_level, UnsetType):
        optimization_level = 1
    _check_argument(
        optimization_level > 0,
        description="Q-CTRL Primitives do not support optimization level 0. Please\
        set optimization_level to 1 and re-try",
        arguments={},
    )


def _warn_and_clean_options_v2(options: Dict[str, Any]) -> None:
    """
    Validate and update transpilation settings
    """
    # Issue a warning and override if any of these setting is not None
    # or a different value than the default below
    expected_options = {
        "optimization_level": 1,
        "resilience_level": 1,
        "resilience": {
            "measure_mitigation": None,
            "measure_noise_learning": None,
            "zne_mitigation": None,
            "zne": None,
            "pec_mitigation": None,
            "pec": None,
            "layer_noise_learning": None,
        },
        "twirling": None,
        "dynamical_decoupling": None,
    }

    # Collect keys with mis-matching values
    different_keys = _validate_values(expected_options, options)
    # Override options
    _update_values(expected_options, options)
    if different_keys:
        logger.warning(
            "The following settings cannot be customized and will be overwritten: %s",
            ",".join(sorted(different_keys)),
        )


def _validate_values(
    expected_options: Dict[str, Any], current_options: Optional[Dict[str, Any]]
) -> List[str]:
    """Validates expected_options and current_options have the same values if the
    keys of expected_options are present in current_options"""

    if current_options is None:
        return []

    different_keys = []
    for expected_key, expected_value in expected_options.items():
        if isinstance(expected_value, dict):
            different_keys.extend(
                _validate_values(expected_value, current_options.get(expected_key, None))
            )
        else:
            current_value = current_options.get(expected_key, None)
            if (current_value not in (None, UnsetType)) and expected_value != current_value:
                different_keys.append(expected_key)
    return different_keys


def _update_values(
    expected_options: Dict[str, Any], current_options: Optional[Dict[str, Any]]
) -> None:

    if current_options is None:
        return

    for expected_key, expected_value in expected_options.items():
        if isinstance(expected_value, dict):
            _update_values(expected_value, current_options.get(expected_key, None))
        else:
            if expected_key in current_options:
                current_options[expected_key] = expected_value


def _check_argument(
    condition: bool,
    description: str,
    arguments: Dict[str, str],
) -> None:
    if not condition:
        error_str = f"{description} arguments={arguments}"
        raise ValueError(error_str)
