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

"""General utility functions."""

from __future__ import annotations
import copy
import keyword
import logging
import os
import re
from queue import Queue
from threading import Condition
from typing import List, Optional, Any, Dict, Union, Tuple, Sequence
from urllib.parse import urlparse

import requests
from ibm_cloud_sdk_core.authenticators import (  # pylint: disable=import-error
    IAMAuthenticator,
)
from ibm_platform_services import ResourceControllerV2  # pylint: disable=import-error
from qiskit.circuit import QuantumCircuit, ControlFlowOp
from qiskit.transpiler import Target
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit_ibm_runtime.exceptions import IBMInputValueError


def is_simulator(backend: BackendV1 | BackendV2) -> bool:
    """Return true if the backend is a simulator.

    Args:
        backend: Backend to check.

    Returns:
        True if backend is a simulator.
    """
    if hasattr(backend, "configuration"):
        return getattr(backend.configuration(), "simulator", False)
    return getattr(backend, "simulator", False)


def is_isa_circuit(circuit: QuantumCircuit, target: Target) -> str:
    """Checks if the circuit is an ISA circuit, meaning that it has a layout and that it
    only uses instructions that exist in the target.

    Args:
        circuit: A single QuantumCircuit
        target: The backend target

    Returns:
        Message on why the circuit is not an ISA circuit, if applicable.
    """
    if circuit.num_qubits > target.num_qubits:
        return (
            f"The circuit has {circuit.num_qubits} qubits "
            f"but the target system requires {target.num_qubits} qubits."
        )

    for instruction in circuit.data:
        name = instruction.operation.name
        qargs = tuple(circuit.find_bit(x).index for x in instruction.qubits)
        if (
            not target.instruction_supported(name, qargs)
            and name != "barrier"
            and not circuit.has_calibration_for(instruction)
        ):
            return (
                f"The instruction {name} on qubits {qargs} is not supported by the target system."
            )
    return ""


def validate_isa_circuits(circuits: Sequence[QuantumCircuit], target: Target) -> None:
    """Validate if all circuits are ISA circuits

    Args:
        circuits: A list of QuantumCircuits.
        target: The backend target
    """
    for circuit in circuits:
        message = is_isa_circuit(circuit, target)
        if message:
            raise IBMInputValueError(
                message
                + " Circuits that do not match the target hardware definition are no longer "
                "supported after March 4, 2024. See the transpilation documentation "
                "(https://docs.quantum.ibm.com/transpile) for instructions to transform circuits and "
                "the primitive examples (https://docs.quantum.ibm.com/run/primitives-examples) to see "
                "this coupled with operator transformations."
            )


def are_circuits_dynamic(circuits: List[QuantumCircuit], qasm_default: bool = True) -> bool:
    """Checks if the input circuits are dynamic."""
    for circuit in circuits:
        if isinstance(circuit, str):
            return qasm_default  # currently do not verify QASM inputs
        for inst in circuit:
            if (
                isinstance(inst.operation, ControlFlowOp)
                or getattr(inst.operation, "condition", None) is not None
            ):
                return True
    return False


def validate_no_dd_with_dynamic_circuits(circuits: List[QuantumCircuit], options: Any) -> None:
    """Validate that if dynamical decoupling options are enabled,
    no circuit in the pubs is dynamic

    Args:
        circuits: A list of QuantumCircuits.
        options: The runtime options
    """
    if not hasattr(options, "dynamical_decoupling") or not options.dynamical_decoupling.enable:
        return
    if are_circuits_dynamic(circuits, False):
        raise IBMInputValueError(
            "Dynamical decoupling currently cannot be used with dynamic circuits"
        )


def validate_job_tags(job_tags: Optional[List[str]]) -> None:
    """Validates input job tags.

    Args:
        job_tags: Job tags to be validated.

    Raises:
        IBMInputValueError: If the job tags are invalid.
    """
    if job_tags and (
        not isinstance(job_tags, list) or not all(isinstance(tag, str) for tag in job_tags)
    ):
        raise IBMInputValueError("job_tags needs to be a list of strings.")


def get_iam_api_url(cloud_url: str) -> str:
    """Computes the IAM API URL for the given IBM Cloud URL."""
    parsed_url = urlparse(cloud_url)
    return f"{parsed_url.scheme}://iam.{parsed_url.hostname}"


def get_resource_controller_api_url(cloud_url: str) -> str:
    """Computes the Resource Controller API URL for the given IBM Cloud URL."""
    parsed_url = urlparse(cloud_url)
    return f"{parsed_url.scheme}://resource-controller.{parsed_url.hostname}"


def resolve_crn(channel: str, url: str, instance: str, token: str) -> List[str]:
    """Resolves the Cloud Resource Name (CRN) for the given cloud account."""
    if channel != "ibm_cloud":
        raise ValueError("CRN value can only be resolved for cloud accounts.")

    if is_crn(instance):
        # no need to resolve CRN value by name
        return [instance]
    else:
        with requests.Session() as session:
            # resolve CRN value based on the provided service name
            authenticator = IAMAuthenticator(token, url=get_iam_api_url(url))
            client = ResourceControllerV2(authenticator=authenticator)
            client.set_service_url(get_resource_controller_api_url(url))
            client.set_http_client(session)
            list_response = client.list_resource_instances(name=instance)
            result = list_response.get_result()
            row_count = result["rows_count"]
            if row_count == 0:
                return []
            else:
                return list(map(lambda resource: resource["crn"], result["resources"]))


def is_crn(locator: str) -> bool:
    """Check if a given value is a CRN (Cloud Resource Name).

    Args:
        locator: The value to check.

    Returns:
        Whether the input is a CRN.
    """
    return isinstance(locator, str) and locator.startswith("crn:")


def get_runtime_api_base_url(url: str, instance: str) -> str:
    """Computes the Runtime API base URL based on the provided input parameters.

    Args:
        url: The URL.
        instance: The instance.

    Returns:
        Runtime API base URL
    """

    # ibm_quantum: no need to resolve runtime API URL
    api_host = url

    # cloud: compute runtime API URL based on crn and URL
    if is_crn(instance) and not _is_experimental_runtime_url(url):
        parsed_url = urlparse(url)
        api_host = (
            f"{parsed_url.scheme}://{_location_from_crn(instance)}"
            f".quantum-computing.{parsed_url.hostname}"
        )

    return api_host


def _is_experimental_runtime_url(url: str) -> bool:
    """Checks if the provided url points to an experimental runtime cluster.
    This type of URLs is used for internal development purposes only.

    Args:
        url: The URL.
    """
    return isinstance(url, str) and "experimental" in url


def _location_from_crn(crn: str) -> str:
    """Computes the location from a given CRN.

    Args:
        crn: A CRN (format: https://cloud.ibm.com/docs/account?topic=account-crn#format-crn)

    Returns:
        The location.
    """
    pattern = "(.*?):(.*?):(.*?):(.*?):(.*?):(.*?):.*"
    return re.search(pattern, crn).group(6)


def to_python_identifier(name: str) -> str:
    """Convert a name to a valid Python identifier.

    Args:
        name: Name to be converted.

    Returns:
        Name that is a valid Python identifier.
    """
    # Python identifiers can only contain alphanumeric characters
    # and underscores and cannot start with a digit.
    pattern = re.compile(r"\W|^(?=\d)", re.ASCII)
    if not name.isidentifier():
        name = re.sub(pattern, "_", name)

    # Convert to snake case
    name = re.sub("((?<=[a-z0-9])[A-Z]|(?!^)(?<!_)[A-Z](?=[a-z]))", r"_\1", name).lower()

    while keyword.iskeyword(name):
        name += "_"

    return name


def setup_logger(logger: logging.Logger) -> None:
    """Setup the logger for the runtime modules with the appropriate level.

    It involves:
        * Use the `QISKIT_IBM_RUNTIME_LOG_LEVEL` environment variable to
          determine the log level to use for the runtime modules. If an invalid
          level is set, the log level defaults to ``WARNING``. The valid log levels
          are ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, and ``CRITICAL``
          (case-insensitive). If the environment variable is not set, then the parent
          logger's level is used, which also defaults to `WARNING`.
        * Use the `QISKIT_IBM_RUNTIME_LOG_FILE` environment variable to specify the
          filename to use when logging messages. If a log file is specified, the log
          messages will not be logged to the screen. If a log file is not specified,
          the log messages will only be logged to the screen and not to a file.
    """
    log_level = os.getenv("QISKIT_IBM_RUNTIME_LOG_LEVEL", "")
    log_file = os.getenv("QISKIT_IBM_RUNTIME_LOG_FILE", "")

    # Setup the formatter for the log messages.
    log_fmt = "%(module)s.%(funcName)s:%(levelname)s:%(asctime)s: %(message)s"
    formatter = logging.Formatter(log_fmt)

    # Set propagate to `False` since handlers are to be attached.
    logger.propagate = False

    # Log messages to a file (if specified), otherwise log to the screen (default).
    if log_file:
        # Setup the file handler.
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        # Setup the stream handler, for logging to console, with the given format.
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Set the logging level after formatting, if specified.
    if log_level:
        # Default to `WARNING` if the specified level is not valid.
        level = logging.getLevelName(log_level.upper())
        if not isinstance(level, int):
            logger.warning(
                '"%s" is not a valid log level. The valid log levels are: '
                "`DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.",
                log_level,
            )
            level = logging.WARNING
        logger.debug('The logger is being set to level "%s"', level)
        logger.setLevel(level)


def filter_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the data with certain fields filtered.

    Data to be filtered out includes hub/group/project information.

    Args:
        data: Original data to be filtered.

    Returns:
        Filtered data.
    """
    if not isinstance(data, dict):
        return data

    data_to_filter = copy.deepcopy(data)
    keys_to_filter = ["hubInfo"]
    _filter_value(data_to_filter, keys_to_filter)  # type: ignore[arg-type]
    return data_to_filter


def _filter_value(data: Dict[str, Any], filter_keys: List[Union[str, Tuple[str, str]]]) -> None:
    """Recursive function to filter out the values of the input keys.

    Args:
        data: Data to be filtered
        filter_keys: A list of keys whose values are to be filtered out. Each
            item in the list can be a string or a tuple. A tuple indicates nested
            keys, such as ``{'backend': {'name': ...}}`` and must have a length
            of 2.
    """
    for key, value in data.items():
        for filter_key in filter_keys:
            if isinstance(filter_key, str) and key == filter_key:
                data[key] = "..."
            elif key == filter_key[0] and filter_key[1] in value:
                data[filter_key[0]][filter_key[1]] = "..."
            elif isinstance(value, dict):
                _filter_value(value, filter_keys)


class RefreshQueue(Queue):
    """A queue that replaces the oldest item with the new item being added when full.

    A FIFO queue with a bounded size. Once the queue is full, when a new item
    is being added, the oldest item on the queue is discarded to make space for
    the new item.
    """

    def __init__(self, maxsize: int):
        """RefreshQueue constructor.

        Args:
            maxsize: Maximum size of the queue.
        """
        self.condition = Condition()
        super().__init__(maxsize=maxsize)

    def put(self, item: Any) -> None:  # type: ignore[override]
        """Put `item` into the queue.

        If the queue is full, the oldest item is replaced by `item`.

        Args:
            item: Item to put into the queue.
        """
        # pylint: disable=arguments-differ

        with self.condition:
            if self.full():
                super().get(block=False)
            super().put(item, block=False)
            self.condition.notify()

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        """Remove and return an item from the queue.

        Args:
            block: If ``True``, block if necessary until an item is available.
            timeout: Block at most `timeout` seconds before raising the
                ``queue.Empty`` exception if no item was available. If
                ``None``, block indefinitely until an item is available.

        Returns:
            An item from the queue.

        Raises:
            queue.Empty: If `block` is ``False`` and no item is available, or
                if `block` is ``True`` and no item is available before `timeout`
                is reached.
        """
        with self.condition:
            if block and self.empty():
                self.condition.wait(timeout)
            return super().get(block=False)

    def notify_all(self) -> None:
        """Wake up all threads waiting for items on the queued."""
        with self.condition:
            self.condition.notify_all()


class CallableStr(str):
    """A callable string."""

    def __call__(self) -> str:
        return self
