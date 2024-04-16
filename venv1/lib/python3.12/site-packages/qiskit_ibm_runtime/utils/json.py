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
from qiskit.circuit.parametertable import ParameterView
from qiskit.result import Result
from qiskit.version import __version__ as _terra_version_string
from qiskit.utils import optionals
from qiskit.qpy import (
    _write_parameter_expression,
    _read_parameter_expression_v3,
    load,
    dump,
)
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.qpy.binary_io.value import _write_parameter, _read_parameter
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.primitives.containers import (
    BitArray,
    DataBin,
    make_data_bin,
    PubResult,
    PrimitiveResult,
)

_TERRA_VERSION = tuple(
    int(x) for x in re.match(r"\d+\.\d+\.\d", _terra_version_string).group(0).split(".")[:3]
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


def _decode_and_deserialize(data: str, deserializer: Callable, decompress: bool = True) -> Any:
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
        if isinstance(obj, set):
            return {"__type__": "set", "__value__": list(obj)}
        if isinstance(obj, Result):
            return {"__type__": "Result", "__value__": obj.to_dict()}
        if hasattr(obj, "to_json"):
            return {"__type__": "to_json", "__value__": obj.to_json()}
        if isinstance(obj, QuantumCircuit):
            kwargs: Dict[str, object] = {"use_symengine": bool(optionals.HAS_SYMENGINE)}
            if _TERRA_VERSION[0] >= 1:
                kwargs["version"] = 11
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
        if isinstance(obj, ParameterView):
            return obj.data
        if isinstance(obj, Instruction):
            kwargs = {"use_symengine": bool(optionals.HAS_SYMENGINE)}
            if _TERRA_VERSION[0] >= 1:
                kwargs["version"] = 11
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
        if isinstance(obj, ObservablesArray):
            return {"__type__": "ObservablesArray", "__value__": obj.tolist()}
        if isinstance(obj, BindingsArray):
            out_val = {"shape": obj.shape}
            encoded_data = {}
            for key, val in obj.data.items():
                encoded_data[json.dumps(key, cls=RuntimeEncoder)] = val
            out_val["data"] = encoded_data
            return {"__type__": "BindingsArray", "__value__": out_val}
        if isinstance(obj, BitArray):
            out_val = {"array": obj.array, "num_bits": obj.num_bits}
            return {"__type__": "BitArray", "__value__": out_val}
        if isinstance(obj, DataBin):
            out_val = {
                "field_names": obj._FIELDS,
                "field_types": [str(field_type) for field_type in obj._FIELD_TYPES],
                "shape": obj._SHAPE,
                "fields": {field_name: getattr(obj, field_name) for field_name in obj._FIELDS},
            }
            return {"__type__": "DataBin", "__value__": out_val}
        if isinstance(obj, EstimatorPub):
            return (
                obj.circuit,
                obj.observables.tolist(),
                obj.parameter_values.as_array(obj.circuit.parameters),
                obj.precision,
            )
        if isinstance(obj, SamplerPub):
            return (
                obj.circuit,
                obj.parameter_values.as_array(obj.circuit.parameters),
                obj.shots,
            )
        if isinstance(obj, PubResult):
            out_val = {"data": obj.data, "metadata": obj.metadata}
            return {"__type__": "PubResult", "__value__": out_val}
        if isinstance(obj, PrimitiveResult):
            out_val = {"pub_results": obj._pub_results, "metadata": obj.metadata}
            return {"__type__": "PrimitiveResult", "__value__": out_val}
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
            warnings.warn(f"Callable {obj} is not JSON serializable and will be set to None.")
            return None
        if HAS_SCIPY and isinstance(obj, scipy.sparse.spmatrix):
            value = _serialize_and_encode(obj, scipy.sparse.save_npz, compress=False)
            return {"__type__": "spmatrix", "__value__": value}
        return super().default(obj)


class RuntimeDecoder(json.JSONDecoder):
    """JSON Decoder used by runtime service."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)
        self.__parameter_vectors: Dict[str, Tuple[ParameterVector, set]] = {}
        self.__read_parameter_expression = functools.partial(
            _read_parameter_expression_v3,
            vectors=self.__parameter_vectors,
            use_symengine=bool(optionals.HAS_SYMENGINE),
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
                return _decode_and_deserialize(obj_val, load)[0]
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
            if obj_type == "settings" and obj["__module__"].startswith(
                "qiskit.quantum_info.operators"
            ):
                return _deserialize_from_settings(
                    mod_name=obj["__module__"],
                    class_name=obj["__class__"],
                    settings=_cast_strings_keys_to_int(obj_val),
                )
            if obj_type == "Result":
                return Result.from_dict(obj_val)
            if obj_type == "spmatrix":
                return _decode_and_deserialize(obj_val, scipy.sparse.load_npz, False)
            if obj_type == "ObservablesArray":
                return ObservablesArray(obj_val)
            if obj_type == "BindingsArray":
                ba_kwargs = {"shape": obj_val.get("shape", None)}
                data = obj_val.get("data", None)
                if isinstance(data, dict):
                    decoded_data = {}
                    for key, val in data.items():
                        # Convert to tuple or it can't be a key
                        decoded_key = tuple(json.loads(key, cls=RuntimeDecoder))
                        decoded_data[decoded_key] = val
                    ba_kwargs["data"] = decoded_data
                elif data:
                    raise ValueError(f"Unexpected data type {type(data)} in BindingsArray.")
                return BindingsArray(**ba_kwargs)
            if obj_type == "BitArray":
                return BitArray(**obj_val)
            if obj_type == "DataBin":
                field_names = obj_val["field_names"]
                field_types = [
                    globals().get(field_type, field_type) for field_type in obj_val["field_types"]
                ]
                shape = obj_val["shape"]
                if shape is not None and isinstance(shape, list):
                    shape = tuple(shape)
                data_bin_cls = make_data_bin(
                    zip(field_names, field_types) if field_names and field_types else None,
                    shape=shape,
                )
                return data_bin_cls(**obj_val["fields"])
            if obj_type == "PubResult":
                return PubResult(**obj_val)
            if obj_type == "PrimitiveResult":
                return PrimitiveResult(**obj_val)
            if obj_type == "to_json":
                return obj_val
            if obj_type == "NoiseModel":
                if HAS_AER:
                    return qiskit_aer.noise.NoiseModel.from_dict(obj_val)
                warnings.warn("Qiskit Aer is needed to restore noise model.")
                return obj_val
        return obj
