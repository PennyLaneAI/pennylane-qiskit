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
# pylint: disable=method-hidden
# pylint: disable=too-many-return-statements

"""Utility functions for the runtime service."""

import base64
import copy
import functools
import importlib
import inspect
import io
import json
import re
import warnings
import zlib

from datetime import date
from typing import Any, Callable, Dict, List, Union, Tuple

import dateutil.parser
import numpy as np

try:
    import scipy.sparse

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import qiskit_aer

    HAS_AER = True
except ImportError:
    HAS_AER = False

from qiskit.circuit import (
    Instruction,
    Parameter,
    ParameterExpression,
    ParameterVector,
    QuantumCircuit,
    QuantumRegister,
)
from qiskit.result import Result
from qiskit.version import __version__ as _terra_version_string
from qiskit.utils import optionals

from qiskit.qpy import (
    _write_parameter_expression,
    _read_parameter_expression,
    _read_parameter_expression_v3,
    load,
    dump,
)

from qiskit.qpy.binary_io.value import _write_parameter, _read_parameter

_TERRA_VERSION = tuple(
    int(x)
    for x in re.match(r"\d+\.\d+\.\d", _terra_version_string).group(0).split(".")[:3]
)


def to_base64_string(data: str) -> str:
    """Convert string to base64 string.

    Args:
        data: string to convert

    Returns:
        data as base64 string
    """
    return base64.b64encode(data.encode("utf-8")).decode("utf-8")


def _serialize_and_encode(
    data: Any, serializer: Callable, compress: bool = True, **kwargs: Any
) -> str:
    """Serialize the input data and return the encoded string.

    Args:
        data: Data to be serialized.
        serializer: Function used to serialize data.
        compress: Whether to compress the serialized data.
        kwargs: Keyword arguments to pass to the serializer.

    Returns:
        String representation.
    """
    with io.BytesIO() as buff:
        serializer(buff, data, **kwargs)
        buff.seek(0)
        serialized_data = buff.read()

    if compress:
        serialized_data = zlib.compress(serialized_data)
    return base64.standard_b64encode(serialized_data).decode("utf-8")


def _decode_and_deserialize(
    data: str, deserializer: Callable, decompress: bool = True
) -> Any:
    """Decode and deserialize input data.

    Args:
        data: Data to be deserialized.
        deserializer: Function used to deserialize data.
        decompress: Whether to decompress.

    Returns:
        Deserialized data.
    """
    buff = io.BytesIO()
    decoded = base64.standard_b64decode(data)
    if decompress:
        decoded = zlib.decompress(decoded)

    with io.BytesIO() as buff:
        buff.write(decoded)
        buff.seek(0)
        return deserializer(buff)


def _deserialize_from_settings(mod_name: str, class_name: str, settings: Dict) -> Any:
    """Deserialize an object from its settings.

    Args:
        mod_name: Name of the module.
        class_name: Name of the class.
        settings: Object settings.

    Returns:
        Deserialized object.

    Raises:
        ValueError: If unable to find the class.
    """
    mod = importlib.import_module(mod_name)
    for name, clz in inspect.getmembers(mod, inspect.isclass):
        if name == class_name:
            return clz(**settings)
    raise ValueError(f"Unable to find class {class_name} in module {mod_name}")


def _set_int_keys_flag(obj: Dict) -> Union[Dict, List]:
    """Recursively sets '__int_keys__' flag if dictionary uses integer keys

    Args:
        obj: dictionary

    Returns:
        obj with the '__int_keys__' flag set if dictionary uses integer key
    """
    if isinstance(obj, dict):
        for k, val in list(obj.items()):
            if isinstance(k, int):
                obj["__int_keys__"] = True
            _set_int_keys_flag(val)
    return obj


def _cast_strings_keys_to_int(obj: Dict) -> Dict:
    """Casts string to int keys in dictionary when '__int_keys__' flag is set

    Args:
        obj: dictionary

    Returns:
        obj with string keys cast to int keys and '__int_keys__' flags removed
    """
    if isinstance(obj, dict):
        int_keys: List[int] = []
        for k, val in list(obj.items()):
            if "__int_keys__" in obj:
                try:
                    int_keys.append(int(k))
                except ValueError:
                    pass
            _cast_strings_keys_to_int(val)
        while len(int_keys) > 0:
            key = int_keys.pop()
            obj[key] = obj[str(key)]
            obj.pop(str(key))
        if "__int_keys__" in obj:
            del obj["__int_keys__"]
    return obj


class RuntimeEncoder(json.JSONEncoder):
    """JSON Encoder used by runtime service."""

    def default(self, obj: Any) -> Any:  # pylint: disable=arguments-differ
        if isinstance(obj, date):
            return {"__type__": "datetime", "__value__": obj.isoformat()}
        if isinstance(obj, complex):
            return {"__type__": "complex", "__value__": [obj.real, obj.imag]}
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                return {"__type__": "ndarray", "__value__": obj.tolist()}
            value = _serialize_and_encode(obj, np.save, allow_pickle=False)
            return {"__type__": "ndarray", "__value__": value}
        if isinstance(obj, np.int64):
            return obj.item()
        if isinstance(obj, np.number):
            # Maybe we should encode the numpy data type here for better accuracy.
            return {"__type__": type(obj.item()).__name__, "__value__": obj.item()}
        if isinstance(obj, set):
            return {"__type__": "set", "__value__": list(obj)}
        if isinstance(obj, Result):
            return {"__type__": "Result", "__value__": obj.to_dict()}
        if hasattr(obj, "to_json"):
            return {"__type__": "to_json", "__value__": obj.to_json()}
        if isinstance(obj, QuantumCircuit):
            kwargs: Dict[str, object] = {"use_symengine": bool(optionals.HAS_SYMENGINE)}
            if _TERRA_VERSION[0] >= 1:
                # NOTE: This can be updated only after the server side has
                # updated to a newer qiskit version.
                kwargs["version"] = 10
            value = _serialize_and_encode(
                data=obj,
                serializer=lambda buff, data: dump(
                    data, buff, RuntimeEncoder, **kwargs
                ),  # type: ignore[no-untyped-call]
            )
            return {"__type__": "QuantumCircuit", "__value__": value}
        if isinstance(obj, Parameter):
            value = _serialize_and_encode(
                data=obj,
                serializer=_write_parameter,
                compress=False,
            )
            return {"__type__": "Parameter", "__value__": value}
        if isinstance(obj, ParameterExpression):
            value = _serialize_and_encode(
                data=obj,
                serializer=_write_parameter_expression,
                compress=False,
                use_symengine=bool(optionals.HAS_SYMENGINE),
            )
            return {"__type__": "ParameterExpression", "__value__": value}
        if isinstance(obj, Instruction):
            kwargs = {"use_symengine": bool(optionals.HAS_SYMENGINE)}
            if _TERRA_VERSION[0] >= 1:
                # NOTE: This can be updated only after the server side has
                # updated to a newer qiskit version.
                kwargs["version"] = 10
            # Append instruction to empty circuit
            quantum_register = QuantumRegister(obj.num_qubits)
            quantum_circuit = QuantumCircuit(quantum_register)
            quantum_circuit.append(obj, quantum_register)
            value = _serialize_and_encode(
                data=quantum_circuit,
                serializer=lambda buff, data: dump(
                    data, buff, **kwargs
                ),  # type: ignore[no-untyped-call]
            )
            return {"__type__": "Instruction", "__value__": value}
        if HAS_AER and isinstance(obj, qiskit_aer.noise.NoiseModel):
            return {"__type__": "NoiseModel", "__value__": obj.to_dict()}
        if hasattr(obj, "settings"):
            return {
                "__type__": "settings",
                "__module__": obj.__class__.__module__,
                "__class__": obj.__class__.__name__,
                "__value__": _set_int_keys_flag(copy.deepcopy(obj.settings)),
            }
        if callable(obj):
            warnings.warn(
                f"Callable {obj} is not JSON serializable and will be set to None."
            )
            return None
        if HAS_SCIPY and isinstance(obj, scipy.sparse.spmatrix):
            value = _serialize_and_encode(obj, scipy.sparse.save_npz, compress=False)
            return {"__type__": "spmatrix", "__value__": value}
        return super().default(obj)


class RuntimeDecoder(json.JSONDecoder):
    """JSON Decoder used by runtime service."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.pop("encoding", None)
        super().__init__(object_hook=self.object_hook, *args, **kwargs)
        self.__parameter_vectors: Dict[str, Tuple[ParameterVector, set]] = {}
        self.__read_parameter_expression = (
            functools.partial(
                _read_parameter_expression_v3,
                vectors=self.__parameter_vectors,
            )
            if _TERRA_VERSION >= (0, 19, 1)
            else _read_parameter_expression
        )

    def object_hook(self, obj: Any) -> Any:
        """Called to decode object."""
        if "__type__" in obj:
            obj_type = obj["__type__"]
            obj_val = obj["__value__"]

            if obj_type == "datetime":
                return dateutil.parser.parse(obj_val)
            if obj_type == "complex":
                return obj_val[0] + 1j * obj_val[1]
            if obj_type == "ndarray":
                if isinstance(obj_val, list):
                    return np.array(obj_val)
                return _decode_and_deserialize(obj_val, np.load)
            if obj_type == "set":
                return set(obj_val)
            if obj_type == "QuantumCircuit":
                return _decode_and_deserialize(
                    data=obj_val,
                    deserializer=lambda buff: load(
                        buff, metadata_deserializer=RuntimeDecoder
                    ),
                )[
                    0
                ]  # type: ignore[no-untyped-call]
            if obj_type == "Parameter":
                return _decode_and_deserialize(obj_val, _read_parameter, False)
            if obj_type == "ParameterExpression":
                return _decode_and_deserialize(
                    obj_val, self.__read_parameter_expression, False  # type: ignore[arg-type]
                )
            if obj_type == "Instruction":
                # Standalone instructions are encoded as the sole instruction in a QPY serialized circuit
                # to deserialize load qpy circuit and return first instruction object in that circuit.
                circuit = _decode_and_deserialize(obj_val, load)[0]
                return circuit.data[0][0]
            if obj_type == "settings":
                return _deserialize_from_settings(
                    mod_name=obj["__module__"],
                    class_name=obj["__class__"],
                    settings=_cast_strings_keys_to_int(obj_val),
                )
            if obj_type == "Result":
                return Result.from_dict(obj_val)
            if obj_type == "spmatrix":
                return _decode_and_deserialize(obj_val, scipy.sparse.load_npz, False)
            if obj_type == "to_json":
                return obj_val
            if obj_type == "NoiseModel":
                if HAS_AER:
                    return qiskit_aer.noise.NoiseModel.from_dict(obj_val)
                warnings.warn("Qiskit Aer is needed to restore noise model.")
                return obj_val
        return obj
