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

"""Scheduler for dynamic circuit backends."""

from abc import abstractmethod
from typing import Dict, List, Optional, Union, Set, Tuple
import itertools

import qiskit
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.scheduling.time_unit_conversion import TimeUnitConversion

from qiskit.circuit import Barrier, Clbit, ControlFlowOp, Measure, Qubit, Reset
from qiskit.circuit.bit import Bit
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler.exceptions import TranspilerError

from .utils import BlockOrderingCallableType, block_order_op_nodes


class BaseDynamicCircuitAnalysis(TransformationPass):
    """Base class for scheduling analysis

    This is a scheduler designed to work for the unique scheduling constraints of the dynamic circuits
    backends due to the limitations imposed by hardware. This is expected to evolve over time as the
    dynamic circuit backends also change.

    The primary differences are that:

    * Resets and control-flow currently trigger the end of a "quantum block". The period between the end
        of the block and the next is *nondeterministic*
        ie., we do not know when the next block will begin (as we could be evaluating a classical
        function of nondeterministic length) and therefore the
        next block starts at a *relative* t=0.
    * During a measurement it is possible to apply gates in parallel on disjoint qubits.
    * Measurements and resets on disjoint qubits happen simultaneously and are part of the same block.
    """

    def __init__(
        self,
        durations: qiskit.transpiler.instruction_durations.InstructionDurations,
        block_ordering_callable: Optional[BlockOrderingCallableType] = None,
    ) -> None:
        """Scheduler for dynamic circuit backends.

        Args:
            durations: Durations of instructions to be used in scheduling.
            block_ordering_callable: A callable used to produce an ordering of the nodes to minimize
                the number of blocks needed. If not provided, :func:`~block_order_op_nodes` will be
                used.
        """
        self._durations = durations
        self._block_ordering_callable = (
            block_order_op_nodes if block_ordering_callable is None else block_ordering_callable
        )

        self._dag: Optional[DAGCircuit] = None
        self._block_dag: Optional[DAGCircuit] = None
        self._wire_map: Optional[Dict[Bit, Bit]] = None
        self._node_mapped_wires: Optional[Dict[DAGNode, List[Bit]]] = None
        self._node_block_dags: Dict[DAGNode, DAGCircuit] = {}
        # Mapping of control-flow nodes to their containing blocks
        self._block_idx_dag_map: Dict[int, DAGCircuit] = {}
        # Mapping of block indices to the respective DAGCircuit

        self._current_block_idx = 0
        self._max_block_t1: Optional[Dict[int, int]] = None
        # Track as we build to avoid extra pass
        self._control_flow_block = False
        self._node_start_time: Optional[Dict[DAGNode, Tuple[int, int]]] = None
        self._node_stop_time: Optional[Dict[DAGNode, Tuple[int, int]]] = None
        self._bit_stop_times: Optional[Dict[int, Dict[Union[Qubit, Clbit], int]]] = None
        # Dictionary of blocks each containing a dictionary with the key for each bit
        # in the block and its value being the final time of the bit within the block.
        self._current_block_measures: Set[DAGNode] = set()
        self._current_block_measures_has_reset: bool = False
        self._node_tied_to: Optional[Dict[DAGNode, Set[DAGNode]]] = None
        # Nodes that the scheduling of this node is tied to.
        self._bit_indices: Optional[Dict[Qubit, int]] = None

        self._time_unit_converter = TimeUnitConversion(durations)

        super().__init__()

    @property
    def _current_block_bit_times(self) -> Dict[Union[Qubit, Clbit], int]:
        return self._bit_stop_times[self._current_block_idx]

    def _visit_block(self, block: DAGCircuit, wire_map: Dict[Qubit, Qubit]) -> None:
        # Push the previous block dag onto the stack
        prev_block_dag = self._block_dag
        self._block_dag = block
        prev_wire_map, self._wire_map = self._wire_map, wire_map

        # We must run this on the individual block
        # as the current implementation does not recurse
        # into the circuit structure.
        self._time_unit_converter.run(block)
        self._begin_new_circuit_block()

        for node in self._block_ordering_callable(block):
            self._visit_node(node)

        # Final flush
        self._flush_measures()

        # Pop the previous block dag off the stack restoring it
        self._block_dag = prev_block_dag
        self._wire_map = prev_wire_map

    def _visit_node(self, node: DAGNode) -> None:
        if isinstance(node.op, ControlFlowOp):
            self._visit_control_flow_op(node)
        elif node.op.condition_bits:
            raise TranspilerError(
                "c_if control-flow is not supported by this pass. "
                'Please apply "ConvertConditionsToIfOps" to convert these '
                "conditional operations to new-style Qiskit control-flow."
            )
        else:
            if isinstance(node.op, Measure):
                self._visit_measure(node)
            elif isinstance(node.op, Reset):
                self._visit_reset(node)
            else:
                self._visit_generic(node)

    def _visit_control_flow_op(self, node: DAGNode) -> None:
        # TODO: This is a hack required to tie nodes of control-flow
        # blocks across the scheduler and block_base_padder. This is
        # because the current control flow nodes store the block as a
        # circuit which is not hashable. For processing we are currently
        # required to convert each circuit block to a dag which is inefficient
        # and causes node relationships stored in analysis to be lost between
        # passes as we are constantly recreating the block dags.
        # We resolve this here by caching these dags in the property set.
        self._node_block_dags[node] = node_block_dags = []

        t0 = max(  # pylint: disable=invalid-name
            self._current_block_bit_times[bit] for bit in self._map_wires(node)
        )

        # Duration is 0 as we do not schedule across terminator
        t1 = t0  # pylint: disable=invalid-name
        self._update_bit_times(node, t0, t1)

        for block in node.op.blocks:
            self._control_flow_block = True

            new_dag = circuit_to_dag(block)
            inner_wire_map = {
                inner: outer
                for outer, inner in zip(self._map_wires(node), new_dag.qubits + new_dag.clbits)
            }
            node_block_dags.append(new_dag)
            self._visit_block(new_dag, inner_wire_map)

        # Begin new block for exit to "then" block.
        self._begin_new_circuit_block()

    @abstractmethod
    def _visit_measure(self, node: DAGNode) -> None:
        raise NotImplementedError

    @abstractmethod
    def _visit_reset(self, node: DAGNode) -> None:
        raise NotImplementedError

    @abstractmethod
    def _visit_generic(self, node: DAGNode) -> None:
        raise NotImplementedError

    def _init_run(self, dag: DAGCircuit) -> None:
        """Setup for initial run."""

        self._dag = dag
        self._block_dag = None
        self._wire_map = {wire: wire for wire in dag.wires}
        self._node_mapped_wires = {}
        self._node_block_dags = {}
        self._block_idx_dag_map = {}

        self._current_block_idx = 0
        self._max_block_t1 = {}
        self._control_flow_block = False

        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("ASAP schedule runs on physical circuits only")

        self._node_start_time = {}
        self._node_stop_time = {}
        self._bit_stop_times = {0: {q: 0 for q in dag.qubits + dag.clbits}}
        self._current_block_measures = set()
        self._current_block_measures_has_reset = False
        self._node_tied_to = {}
        self._bit_indices = {q: index for index, q in enumerate(dag.qubits)}

    def _get_duration(self, node: DAGNode, dag: Optional[DAGCircuit] = None) -> int:
        if node.op.condition_bits or isinstance(node.op, ControlFlowOp):
            # As we cannot currently schedule through conditionals model
            # as zero duration to avoid padding.
            return 0

        indices = [self._bit_indices[qarg] for qarg in self._map_qubits(node)]

        # Fall back to current block dag if not specified.
        dag = dag or self._block_dag

        if dag.has_calibration_for(node):
            # If node has calibration, this value should be the highest priority
            cal_key = tuple(indices), tuple(float(p) for p in node.op.params)
            duration = dag.calibrations[node.op.name][cal_key].duration
            node.op.duration = duration
        else:
            # map to outer dag to get the appropriate durations
            duration = self._durations.get(node.op, indices, unit="dt")

        if isinstance(duration, ParameterExpression):
            raise TranspilerError(
                f"Parameterized duration ({duration}) "
                f"of {node.op.name} on qubits {indices} is not bounded."
            )
        if duration is None:
            raise TranspilerError(f"Duration of {node.op.name} on qubits {indices} is not found.")

        return duration

    def _update_bit_times(  # pylint: disable=invalid-name
        self, node: DAGNode, t0: int, t1: int, update_cargs: bool = True
    ) -> None:
        self._max_block_t1[self._current_block_idx] = max(
            self._max_block_t1.get(self._current_block_idx, 0), t1
        )

        update_bits = self._map_wires(node) if update_cargs else self._map_qubits(node)
        for bit in update_bits:
            self._current_block_bit_times[bit] = t1

        self._node_start_time[node] = (self._current_block_idx, t0)
        self._node_stop_time[node] = (self._current_block_idx, t1)

    def _begin_new_circuit_block(self) -> None:
        """Create a new timed circuit block completing the previous block."""
        self._current_block_idx += 1
        self._block_idx_dag_map[self._current_block_idx] = self._block_dag
        self._control_flow_block = False
        self._bit_stop_times[self._current_block_idx] = {
            self._wire_map[wire]: 0 for wire in self._block_dag.wires
        }
        self._flush_measures()

    def _flush_measures(self) -> None:
        """Flush currently accumulated measurements by resetting block measures."""
        for node in self._current_block_measures:
            self._node_tied_to[node] = self._current_block_measures.copy()

        self._current_block_measures = set()
        self._current_block_measures_has_reset = False

    def _current_block_measure_qargs(self) -> Set[Qubit]:
        return set(
            qarg for measure in self._current_block_measures for qarg in self._map_qubits(measure)
        )

    def _check_flush_measures(self, node: DAGNode) -> None:
        if self._current_block_measure_qargs() & set(self._map_qubits(node)):
            if self._current_block_measures_has_reset:
                # If a reset is included we must trigger the end of a block.
                self._begin_new_circuit_block()
            else:
                # Otherwise just trigger a measurement flush
                self._flush_measures()

    def _map_wires(self, node: DAGNode) -> List[Qubit]:
        """Map the wires from the current node to the top-level block's wires.

        TODO: We should have an easier approach to wire mapping from the transpiler.
        """
        if node not in self._node_mapped_wires:
            self._node_mapped_wires[node] = wire_map = [
                self._wire_map[q] for q in node.qargs + node.cargs
            ]
            return wire_map

        return self._node_mapped_wires[node]

    def _map_qubits(self, node: DAGNode) -> List[Qubit]:
        """Map the qubits from the current node to the top-level block's qubits.

        TODO: We should have an easier approach to wire mapping from the transpiler.
        """
        return [wire for wire in self._map_wires(node) if isinstance(wire, Qubit)]


class ASAPScheduleAnalysis(BaseDynamicCircuitAnalysis):
    """Dynamic circuits as-soon-as-possible (ASAP) scheduling analysis pass.

    This is a scheduler designed to work for the unique scheduling constraints of the dynamic circuits
    backends due to the limitations imposed by hardware. This is expected to evolve over time as the
    dynamic circuit backends also change.

    In its current form this is similar to Qiskit's ASAP scheduler in which instructions
    start as early as possible.

    The primary differences are that:

    * Resets and control-flow currently trigger the end of a "quantum block". The period between the end
        of the block and the next is *nondeterministic*
        ie., we do not know when the next block will begin (as we could be evaluating a classical
        function of nondeterministic length) and therefore the
        next block starts at a *relative* t=0.
    * During a measurement it is possible to apply gates in parallel on disjoint qubits.
    * Measurements and resets on disjoint qubits happen simultaneously and are part of the same block.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the ALAPSchedule pass on `dag`.
        Args:
            dag (DAGCircuit): DAG to schedule.
        Raises:
            TranspilerError: if the circuit is not mapped on physical qubits.
            TranspilerError: if conditional bit is added to non-supported instruction.
        Returns:
            The scheduled DAGCircuit.
        """
        self._init_run(dag)

        # Trivial wire map at the top-level
        wire_map = {wire: wire for wire in dag.wires}
        # Top-level dag is the entry block
        self._visit_block(dag, wire_map)

        self.property_set["node_start_time"] = self._node_start_time
        self.property_set["node_block_dags"] = self._node_block_dags
        return dag

    def _visit_measure(self, node: DAGNode) -> None:
        """Visit a measurement node.

        Measurement currently triggers the end of a deterministically scheduled block
        of instructions in IBM dynamic circuits hardware.
        This means that it is possible to schedule *up to* a measurement (and during its pulses)
        but the measurement will be followed by a period of indeterminism.
        All measurements on disjoint qubits that topologically follow another
        measurement will be collected and performed in parallel. A measurement on a qubit
        intersecting with the set of qubits to be measured in parallel will trigger the
        end of a scheduling block with said measurement occurring in a following block
        which begins another grouping sequence. This behavior will change in future
        backend software updates."""

        current_block_measure_qargs = self._current_block_measure_qargs()
        # We handle a set of qubits here as _visit_reset currently calls
        # this method and a reset may have multiple qubits.
        measure_qargs = set(self._map_qubits(node))

        t0q = max(
            self._current_block_bit_times[q] for q in measure_qargs
        )  # pylint: disable=invalid-name

        # If the measurement qubits overlap, we need to flush measurements and start a
        # new scheduling block.
        if current_block_measure_qargs & measure_qargs:
            if self._current_block_measures_has_reset:
                # If a reset is included we must trigger the end of a block.
                self._begin_new_circuit_block()
                t0q = 0
            else:
                # Otherwise just trigger a measurement flush
                self._flush_measures()
        else:
            # Otherwise we need to increment all measurements to start at the same time within the block.
            t0q = max(  # pylint: disable=invalid-name
                itertools.chain(
                    [t0q],
                    (self._node_start_time[measure][1] for measure in self._current_block_measures),
                )
            )

        # Insert this measure into the block
        self._current_block_measures.add(node)

        for measure in self._current_block_measures:
            t0 = t0q  # pylint: disable=invalid-name
            bit_indices = {bit: index for index, bit in enumerate(self._block_dag.qubits)}
            measure_duration = self._durations.get(
                Measure(),
                [bit_indices[qarg] for qarg in self._map_qubits(measure)],
                unit="dt",
            )
            t1 = t0 + measure_duration  # pylint: disable=invalid-name
            self._update_bit_times(measure, t0, t1)

    def _visit_reset(self, node: DAGNode) -> None:
        """Visit a reset node.

        Reset currently triggers the end of a pulse block in IBM dynamic circuits hardware
        as conditional reset is performed internally using a c_if. This means that it is
        possible to schedule *up to* a reset (and during its measurement pulses)
        but the reset will be followed by a period of conditional indeterminism.
        All resets on disjoint qubits will be collected on the same qubits to be run simultaneously.
        """
        # Process as measurement
        self._current_block_measures_has_reset = True
        self._visit_measure(node)
        # Then set that we are now a conditional node.
        self._control_flow_block = True

    def _visit_generic(self, node: DAGNode) -> None:
        """Visit a generic node such as a gate or barrier."""
        op_duration = self._get_duration(node)

        # If the measurement qubits overlap, we need to flush the measurement group
        self._check_flush_measures(node)

        t0 = max(  # pylint: disable=invalid-name
            self._current_block_bit_times[bit] for bit in self._map_wires(node)
        )

        t1 = t0 + op_duration  # pylint: disable=invalid-name
        self._update_bit_times(node, t0, t1)


class ALAPScheduleAnalysis(BaseDynamicCircuitAnalysis):
    """Dynamic circuits as-late-as-possible (ALAP) scheduling analysis pass.

    This is a scheduler designed to work for the unique scheduling constraints of the dynamic circuits
    backends due to the limitations imposed by hardware. This is expected to evolve over time as the
    dynamic circuit backends also change.

    In its current form this is similar to Qiskit's ALAP scheduler in which instructions
    start as late as possible.

    The primary differences are that:

    * Resets and control-flow currently trigger the end of a "quantum block". The period between the end
        of the block and the next is *nondeterministic*
        ie., we do not know when the next block will begin (as we could be evaluating a classical
        function of nondeterministic length) and therefore the
        next block starts at a *relative* t=0.
    * During a measurement it is possible to apply gates in parallel on disjoint qubits.
    * Measurements and resets on disjoint qubits happen simultaneously and are part of the same block.
    """

    def run(self, dag: DAGCircuit) -> None:
        """Run the ASAPSchedule pass on `dag`.
        Args:
            dag (DAGCircuit): DAG to schedule.
        Raises:
            TranspilerError: if the circuit is not mapped on physical qubits.
            TranspilerError: if conditional bit is added to non-supported instruction.
        Returns:
            The scheduled DAGCircuit.
        """
        self._init_run(dag)

        # Trivial wire map at the top-level
        wire_map = {wire: wire for wire in dag.wires}
        # Top-level dag is the entry block
        self._visit_block(dag, wire_map)
        self._push_block_durations()
        self.property_set["node_start_time"] = self._node_start_time
        self.property_set["node_block_dags"] = self._node_block_dags
        return dag

    def _visit_measure(self, node: DAGNode) -> None:
        """Visit a measurement node.

        Measurement currently triggers the end of a deterministically scheduled block
        of instructions in IBM dynamic circuits hardware.
        This means that it is possible to schedule *up to* a measurement (and during its pulses)
        but the measurement will be followed by a period of indeterminism.
        All measurements on disjoint qubits that topologically follow another
        measurement will be collected and performed in parallel. A measurement on a qubit
        intersecting with the set of qubits to be measured in parallel will trigger the
        end of a scheduling block with said measurement occurring in a following block
        which begins another grouping sequence. This behavior will change in future
        backend software updates."""

        current_block_measure_qargs = self._current_block_measure_qargs()
        # We handle a set of qubits here as _visit_reset currently calls
        # this method and a reset may have multiple qubits.
        measure_qargs = set(self._map_qubits(node))

        t0q = max(
            self._current_block_bit_times[q] for q in measure_qargs
        )  # pylint: disable=invalid-name

        # If the measurement qubits overlap, we need to flush measurements and start a
        # new scheduling block.
        if current_block_measure_qargs & measure_qargs:
            if self._current_block_measures_has_reset:
                # If a reset is included we must trigger the end of a block.
                self._begin_new_circuit_block()
                t0q = 0
            else:
                # Otherwise just trigger a measurement flush
                self._flush_measures()
        else:
            # Otherwise we need to increment all measurements to start at the same time within the block.
            t0q = max(  # pylint: disable=invalid-name
                itertools.chain(
                    [t0q],
                    (self._node_start_time[measure][1] for measure in self._current_block_measures),
                )
            )

        # Insert this measure into the block
        self._current_block_measures.add(node)

        for measure in self._current_block_measures:
            t0 = t0q  # pylint: disable=invalid-name
            bit_indices = {bit: index for index, bit in enumerate(self._block_dag.qubits)}
            measure_duration = self._durations.get(
                Measure(),
                [bit_indices[qarg] for qarg in self._map_qubits(measure)],
                unit="dt",
            )
            t1 = t0 + measure_duration  # pylint: disable=invalid-name
            self._update_bit_times(measure, t0, t1)

    def _visit_reset(self, node: DAGNode) -> None:
        """Visit a reset node.

        Reset currently triggers the end of a pulse block in IBM dynamic circuits hardware
        as conditional reset is performed internally using a c_if. This means that it is
        possible to schedule *up to* a reset (and during its measurement pulses)
        but the reset will be followed by a period of conditional indeterminism.
        All resets on disjoint qubits will be collected on the same qubits to be run simultaneously.
        """
        # Process as measurement
        self._current_block_measures_has_reset = True
        self._visit_measure(node)
        # Then set that we are now a conditional node.
        self._control_flow_block = True

    def _visit_generic(self, node: DAGNode) -> None:
        """Visit a generic node such as a gate or barrier."""

        # If True we are coming from a conditional block.
        # start a new block for the unconditional operations.
        if self._control_flow_block:
            self._begin_new_circuit_block()

        op_duration = self._get_duration(node)

        # If the measurement qubits overlap, we need to flush the measurement group
        self._check_flush_measures(node)

        t0 = max(  # pylint: disable=invalid-name
            self._current_block_bit_times[bit] for bit in self._map_wires(node)
        )

        t1 = t0 + op_duration  # pylint: disable=invalid-name
        self._update_bit_times(node, t0, t1)

    def _push_block_durations(self) -> None:
        """After scheduling of each block, pass over and push the times of all nodes."""

        # Store the next available time to push to for the block by bit
        block_bit_times = {}
        # Iterated nodes starting at the first, from the node with the
        # last time, preferring barriers over non-barriers

        def order_ops(item: Tuple[DAGNode, Tuple[int, int]]) -> Tuple[int, int, bool, int]:
            """Iterated nodes ordering by channel, time and preferring that barriers are processed
            first."""
            return (
                item[1][0],
                -item[1][1],
                not isinstance(item[0].op, Barrier),
                self._get_duration(item[0], dag=self._block_idx_dag_map[item[1][0]]),
            )

        iterate_nodes = sorted(self._node_stop_time.items(), key=order_ops)

        new_node_start_time = {}
        new_node_stop_time = {}

        def _calculate_new_times(
            block: int, node: DAGNode, block_bit_times: Dict[int, Dict[Qubit, int]]
        ) -> int:
            max_block_time = min(block_bit_times[block][bit] for bit in self._map_qubits(node))

            t0 = self._node_start_time[node][1]  # pylint: disable=invalid-name
            t1 = self._node_stop_time[node][1]  # pylint: disable=invalid-name
            # Determine how much to shift by
            node_offset = max_block_time - t1
            new_t0 = t0 + node_offset
            return new_t0

        scheduled = set()

        def _update_time(
            block: int,
            node: DAGNode,
            new_time: int,
            block_bit_times: Dict[int, Dict[Qubit, int]],
        ) -> None:
            scheduled.add(node)

            new_node_start_time[node] = (block, new_time)
            new_node_stop_time[node] = (
                block,
                new_time + self._get_duration(node, dag=self._block_idx_dag_map[block]),
            )

            # Update available times by bit
            for bit in self._map_qubits(node):
                block_bit_times[block][bit] = new_time

        for node, (
            block,
            _,
        ) in iterate_nodes:  # pylint: disable=invalid-name
            # skip already scheduled
            if node in scheduled:
                continue
            # Start with last time as the time to push to
            if block not in block_bit_times:
                block_bit_times[block] = {q: self._max_block_t1[block] for q in self._dag.wires}

            # Calculate the latest available time to push to collectively for tied nodes
            tied_nodes = self._node_tied_to.get(node, None)
            if tied_nodes is not None:
                # Take the minimum time that will be schedulable
                # self._node_tied_to includes the node itself.
                new_times = [
                    _calculate_new_times(block, tied_node, block_bit_times)
                    for tied_node in self._node_tied_to[node]
                ]
                new_time = min(new_times)
                for tied_node in tied_nodes:
                    _update_time(block, tied_node, new_time, block_bit_times)

            else:
                new_t0 = _calculate_new_times(block, node, block_bit_times)
                _update_time(block, node, new_t0, block_bit_times)

        self._node_start_time = new_node_start_time
        self._node_stop_time = new_node_stop_time
