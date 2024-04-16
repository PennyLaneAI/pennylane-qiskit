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

"""Utility functions for options."""

from __future__ import annotations

from typing import Optional, Union, Callable, TYPE_CHECKING, Any
import functools
import copy
from dataclasses import is_dataclass, asdict
from numbers import Real

from pydantic import ConfigDict, ValidationInfo, field_validator
from pydantic.dataclasses import dataclass

from qiskit.providers.backend import Backend

from ..utils.utils import is_simulator

if TYPE_CHECKING:
    from ..options.options import BaseOptions


def set_default_error_levels(
    options: dict,
    backend: Backend,
    default_optimization_level: int,
    default_resilience_level: int,
) -> dict:
    """Set default resilience and optimization levels.

    Args:
        options: user passed in options.
        backend: backend the job will run on.
        default_optimization_level: the default optimization level from the options class
        default_resilience_level: the default resilience level from the options class

    Returns:
        options with correct error level defaults.
    """
    is_sim = is_simulator(backend)

    if options.get("optimization_level") is None:
        if is_sim and not options.get("simulator", {}).get("noise_model"):
            options["optimization_level"] = 1
        else:
            options["optimization_level"] = default_optimization_level

    if options.get("resilience_level") is None:
        if is_sim and not options.get("simulator", {}).get("noise_model"):
            options["resilience_level"] = 0
        else:
            options["resilience_level"] = default_resilience_level
    return options


def remove_dict_unset_values(in_dict: dict) -> None:
    """Remove Unset values."""
    for key, val in list(in_dict.items()):
        if isinstance(val, UnsetType):
            del in_dict[key]
        elif isinstance(val, dict):
            remove_dict_unset_values(val)


def remove_empty_dict(in_dict: dict) -> None:
    """Remove empty dictionaries."""
    for key, val in list(in_dict.items()):
        if isinstance(val, dict):
            if val:
                remove_empty_dict(val)
            if not val:
                del in_dict[key]


def _to_obj(cls_, data):  # type: ignore
    if data is None:
        return cls_()
    if isinstance(data, cls_):
        return data
    if isinstance(data, dict):
        return cls_(**data)
    raise TypeError(
        f"{data} has an unspported type {type(data)}. It can only be {cls_} or a dictionary."
    )


def merge_options(
    old_options: Union[dict, "BaseOptions"], new_options: Optional[dict] = None
) -> dict:
    """Merge current options with the new ones.

    Args:
        new_options: New options to merge.

    Returns:
        Merged dictionary.

    Raises:
        TypeError: if input type is invalid.
    """

    def _update_options(old: dict, new: dict, matched: Optional[dict] = None) -> None:
        if not new and not matched:
            return
        matched = matched or {}

        for key, val in old.items():
            if isinstance(val, dict):
                new_matched = new.pop(key, {})
                _update_options(val, new, new_matched)
            elif key in new.keys():
                old[key] = new.pop(key)
            elif key in matched.keys():
                old[key] = matched.pop(key)

        # Add new keys.
        for key, val in matched.items():
            old[key] = val

        # Clear the matched dict so it's not reused
        matched.clear()

    if is_dataclass(old_options):
        combined = asdict(old_options)
    elif isinstance(old_options, dict):
        combined = copy.deepcopy(old_options)
    else:
        raise TypeError("'old_options' can only be a dictionary or dataclass.")

    if not new_options:
        return combined
    new_options_copy = copy.deepcopy(new_options)

    # First update values of the same key.
    _update_options(combined, new_options_copy)

    # Add new keys.
    combined.update(new_options_copy)

    return combined


def skip_unset_validation(func: Callable) -> Callable:
    """Decorator used to skip unset value"""

    @functools.wraps(func)
    def wrapper(cls: Any, val: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(val, UnsetType):
            return val
        return func(cls, val, *args, **kwargs)

    return wrapper


class Dict:
    """Fake Dict type.

    This class is used to show dictionary as an acceptable type in docs without
    attaching all the dictionary attributes in Jupyter's auto-complete.
    """

    pass


class UnsetType:
    """Class used to represent an unset field."""

    def __repr__(self) -> str:
        return "Unset"

    def __new__(cls) -> "UnsetType":
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False


Unset = UnsetType()


primitive_dataclass = dataclass(
    config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True, extra="forbid")
)


def make_constraint_validator(
    *field_names: str,
    ge: Real | None = None,
    gt: Real | None = None,
    le: Real | None = None,
    lt: Real | None = None,
) -> Callable:
    """Make a field validator that performs the give constraint if the value is numeric.
    This differs to the one built-in to ``pydantic.Field`` in that it ignores non-Real types,
    which lets us apply this to fields with annotations like ``int | Literal["auto"]``.
    Args:
        field_names: The field names to check.
        ge: A number the value must be greater than or equal to.
        gt: A number the value must be strictly greater than.
        le: A number the value must be less than or equal to.
        lt: A number the value must be strictly less than.
    Returns:
        A new field validator.
    """

    @field_validator(*field_names, mode="before")  # type: ignore[misc]
    @classmethod
    @skip_unset_validation
    def validator(cls: Any, value: Any, validation_info: ValidationInfo) -> Any:
        if isinstance(value, Real):
            if ge is not None and (value < ge):
                raise ValueError(
                    f"{cls.__name__}.{validation_info.field_name} must be >={ge}, but is =={value}."
                )
            if gt is not None and (value <= gt):
                raise ValueError(
                    f"{cls.__name__}.{validation_info.field_name} must be >{gt}, but is =={value}."
                )
            if le is not None and (value > le):
                raise ValueError(
                    f"{cls.__name__}.{validation_info.field_name} must be <={le}, but is =={value}."
                )
            if lt is not None and (value >= lt):
                raise ValueError(
                    f"{cls.__name__}.{validation_info.field_name} must be <{lt}, but is =={value}."
                )
        return value

    return validator
