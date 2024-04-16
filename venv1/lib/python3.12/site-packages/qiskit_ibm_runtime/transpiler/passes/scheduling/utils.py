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

"""Utility functions for scheduling passes."""

import warnings
from typing import Callable, Generator, Optional, Tuple, Union
from functools import lru_cache

from qiskit.circuit import ControlFlowOp, Measure, Reset, Parameter
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.instruction_durations import (
    InstructionDurations,
    InstructionDurationsType,
)
from qiskit.transpiler.target import Target
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.providers import Backend, BackendV1


BlockOrderingCallableType = Callable[[DAGCircuit], Generator[DAGOpNode, None, None]]


def block_order_op_nodes(dag: DAGCircuit) -> Generator[DAGOpNode, None, None]:
    """Yield nodes such that they are sorted into groups of blocks that minimize synchronization.

    Measurements are also grouped.
    """

    def _is_grouped_measure(node: DAGOpNode) -> bool:
        """Does this node need to be grouped?"""
        return isinstance(node.op, (Reset, Measure))

    def _is_block_trigger(node: DAGOpNode) -> bool:
        """Does this node trigger the end of a block?"""
        return isinstance(node.op, ControlFlowOp)

    @lru_cache(maxsize=8192)
    def _emit(
        node: DAGOpNode,
        grouped_measure: Tuple[DAGOpNode],
        block_triggers: Tuple[DAGOpNode],
    ) -> bool:
        """Should we emit this node?"""
        for measure in grouped_measure:
            if dag.is_predecessor(node, measure):
                return True
        for block_trigger in block_triggers:
            if dag.is_predecessor(node, block_trigger):
                return True

        return _is_grouped_measure(node) or _is_block_trigger(node)

    # Begin processing nodes in order
    next_nodes = dag.topological_op_nodes()
    while next_nodes:
        curr_nodes = next_nodes  # Setup the next iteration nodes
        next_nodes_set = set()  # Nodes that will make it into the next iteration
        next_nodes = []  # Nodes to process in order in the next iteration
        to_push = []  # Do we push this to the very last block?
        yield_measures = []  # Measures/resets we will yield first
        yield_block_triggers = []  # Followed by block triggers (conditionals)
        block_break = False  # Did we encounter a block trigger in this iteration?
        for node in curr_nodes:
            # If we have added this node to the next set of nodes
            # skip for now.
            if node in next_nodes_set:
                next_nodes.append(node)
                continue

            # If this nodes is a measurement
            # push on the measurements to process
            if _is_grouped_measure(node):
                block_break = True
                node_descendants = dag.descendants(node)
                next_nodes_set |= set(node_descendants)
                yield_measures.append(node)
            # If this node is a block push this onto
            # the block trigger list.
            elif _is_block_trigger(node):
                block_break = True
                node_descendants = dag.descendants(node)
                next_nodes_set |= set(node_descendants)
                yield_block_triggers.append(node)
            # Otherwise we push onto the final list of blocks to emit
            # as part of the final block.
            else:
                to_push.append(node)

        new_to_push = []
        for node in to_push:
            node_descendants = dag.descendants(node)
            if any(
                _emit(descendant, tuple(yield_measures), tuple(yield_block_triggers))
                for descendant in node_descendants
                if isinstance(descendant, DAGOpNode)
            ):
                yield node
            else:
                new_to_push.append(node)

        to_push = new_to_push

        # First emit the measurements which will feed
        for node in yield_measures:
            yield node
        # Into the block triggers we will emit.
        for node in yield_block_triggers:
            yield node

        # We're at the last block and emit the final nodes
        if not block_break:
            for node in to_push:
                yield node
            break
        # Otherwise emit the final nodes
        # Add to the front of the list to be processed next
        to_push.extend(next_nodes)
        next_nodes = to_push

    _emit.cache_clear()


InstrKey = Union[
    Tuple[str, None, None],
    Tuple[str, Tuple[int], None],
    Tuple[str, Tuple[int], Tuple[Parameter]],
]


class DynamicCircuitInstructionDurations(InstructionDurations):
    """For dynamic circuits the IBM Qiskit backend currently
    reports instruction durations that differ compared with those
    required for the legacy Qobj-based path. For now we use this
    class to report updated InstructionDurations.
    TODO: This would be mitigated by a specialized Backend/Target for
    dynamic circuit backends.
    """

    MEASURE_PATCH_CYCLES = 160
    MEASURE_PATCH_ODD_OFFSET = 64

    def __init__(
        self,
        instruction_durations: Optional[InstructionDurationsType] = None,
        dt: float = None,
        enable_patching: bool = True,
    ):
        """Dynamic circuit instruction durations."""
        self._enable_patching = enable_patching
        super().__init__(instruction_durations=instruction_durations, dt=dt)

    @classmethod
    def from_backend(cls, backend: Backend) -> "DynamicCircuitInstructionDurations":
        """Construct a :class:`DynamicInstructionDurations` object from the backend.
        Args:
            backend: backend from which durations (gate lengths) and dt are extracted.
        Returns:
            DynamicInstructionDurations: The InstructionDurations constructed from backend.
        """
        if isinstance(backend, BackendV1):
            return super(DynamicCircuitInstructionDurations, cls).from_backend(backend)

        # Get durations from target if BackendV2
        return cls.from_target(backend.target)

    @classmethod
    def from_target(cls, target: Target) -> "DynamicCircuitInstructionDurations":
        """Construct a :class:`DynamicInstructionDurations` object from the target.
        Args:
            target: target from which durations (gate lengths) and dt are extracted.
        Returns:
            DynamicInstructionDurations: The InstructionDurations constructed from backend.
        """

        instruction_durations_dict = target.durations().duration_by_name_qubits
        instruction_durations = []
        for instr_key, instr_value in instruction_durations_dict.items():
            instruction_durations += [(*instr_key, *instr_value)]
        try:
            dt = target.dt
        except AttributeError:
            dt = None
        return cls(instruction_durations, dt=dt)

    def update(
        self, inst_durations: Optional[InstructionDurationsType], dt: float = None
    ) -> "DynamicCircuitInstructionDurations":
        """Update self with inst_durations (inst_durations overwrite self). Overrides the default
        durations for certain hardcoded instructions.

        Args:
            inst_durations: Instruction durations to be merged into self (overwriting self).
            dt: Sampling duration in seconds of the target backend.

        Returns:
            InstructionDurations: The updated InstructionDurations.

        Raises:
            TranspilerError: If the format of instruction_durations is invalid.
        """

        # First update as normal
        super().update(inst_durations, dt=dt)

        if not self._enable_patching or inst_durations is None:
            return self

        # Then update required instructions. This code is ugly
        # because the InstructionDurations code is handling too many
        # formats in update and this code must also.
        if isinstance(inst_durations, InstructionDurations):
            for key in inst_durations.keys():
                self._patch_instruction(key)
        else:
            for name, qubits, _, parameters, _ in inst_durations:
                if isinstance(qubits, int):
                    qubits = [qubits]

                if isinstance(parameters, (int, float)):
                    parameters = [parameters]

                if qubits is None:
                    key = (name, None, None)
                elif parameters is None:
                    key = (name, tuple(qubits), None)
                else:
                    key = (name, tuple(qubits), tuple(parameters))

                self._patch_instruction(key)

        return self

    def _patch_instruction(self, key: InstrKey) -> None:
        """Dispatcher logic for instruction patches"""
        name = key[0]
        if name == "measure":
            self._patch_measurement(key)
        elif name == "reset":
            self._patch_reset(key)

    def _convert_and_patch_key(self, key: InstrKey) -> None:
        """Convert duration to dt and patch key"""
        prev_duration, unit = self._get_duration(key)
        if unit != "dt":
            prev_duration = self._convert_unit(prev_duration, unit, "dt")
            # raise TranspilerError('Can currently only patch durations of "dt".')
        odd_cycle_correction = self._get_odd_cycle_correction()
        new_duration = prev_duration + self.MEASURE_PATCH_CYCLES + odd_cycle_correction
        if unit != "dt":  # convert back to original unit
            new_duration = self._convert_unit(new_duration, "dt", unit)
        self._patch_key(key, new_duration, unit)

    def _patch_measurement(self, key: InstrKey) -> None:
        """Patch measurement duration by extending duration by 160dt as temporarily
        required by the dynamic circuit backend.
        """
        self._convert_and_patch_key(key)
        # Enforce patching of reset on measurement update
        self._patch_reset(("reset", key[1], key[2]))

    def _patch_reset(self, key: InstrKey) -> None:
        """Patch reset duration by extending duration by measurement patch as temporarily
        required by the dynamic circuit backend.
        """
        # We patch the reset to be the duration of the measurement if it
        # is available as it currently
        # triggers the end of scheduling after the measurement pulse
        measure_key = ("measure", key[1], key[2])
        try:
            measure_duration, unit = self._get_duration(measure_key)
            self._patch_key(key, measure_duration, unit)
        except KeyError:
            # Fall back to reset key if measure not available
            self._convert_and_patch_key(key)

    def _get_duration(self, key: InstrKey) -> Tuple[int, str]:
        """Handling for the complicated structure of this class.

        TODO: This class implementation should be simplified in Qiskit. Too many edge cases.
        """
        if key[1] is None and key[2] is None:
            duration = self.duration_by_name[key[0]]
        elif key[2] is None:
            duration = self.duration_by_name_qubits[(key[0], key[1])]
        else:
            duration = self.duration_by_name_qubits_params[key]
        return duration

    def _patch_key(self, key: InstrKey, duration: int, unit: str) -> None:
        """Handling for the complicated structure of this class.

        TODO: This class implementation should be simplified in Qiskit. Too many edge cases.
        """
        if key[1] is None and key[2] is None:
            self.duration_by_name[key[0]] = (duration, unit)
        elif key[2] is None:
            self.duration_by_name_qubits[(key[0], key[1])] = (duration, unit)

        self.duration_by_name_qubits_params[key] = (duration, unit)

    def _get_odd_cycle_correction(self) -> int:
        """Determine the amount of the odd cycle correction to apply
        For devices with short gates with odd lenghts we add an extra 16dt to the measurement

        TODO: Eliminate the need for this correction
        """
        key_pulse = "sx"
        key_qubit = 0
        try:
            key_duration = self.get(key_pulse, key_qubit, "dt")
        except TranspilerError:
            warnings.warn(
                f"No {key_pulse} gate found for {key_qubit} for detection of "
                "short odd gate lengths, default measurement timing will be used."
            )
            key_duration = 160  # keyPulse gate not found

        if key_duration < 160 and key_duration % 32:
            return self.MEASURE_PATCH_ODD_OFFSET
        return 0
