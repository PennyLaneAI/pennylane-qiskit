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

"""Padding pass to fill timeslots for IBM (dynamic circuit) backends."""

from typing import Dict, Iterable, List, Optional, Union, Set

from qiskit.circuit import (
    Qubit,
    Clbit,
    ControlFlowOp,
    Gate,
    IfElseOp,
    Instruction,
    Measure,
)
from qiskit.circuit.bit import Bit
from qiskit.circuit.library import Barrier
from qiskit.circuit.delay import Delay
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.converters import dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from .utils import block_order_op_nodes, BlockOrderingCallableType


class BlockBasePadder(TransformationPass):
    """The base class of padding pass.

    This pass requires one of scheduling passes to be executed before itself.
    Since there are multiple scheduling strategies, the selection of scheduling
    pass is left in the hands of the pass manager designer.
    Once a scheduling analysis pass is run, ``node_start_time`` is generated
    in the :attr:`property_set`.  This information is represented by a python dictionary of
    the expected instruction execution times keyed on the node instances.
    The padding pass expects all ``DAGOpNode`` in the circuit to be scheduled.

    This base class doesn't define any sequence to interleave, but it manages
    the location where the sequence is inserted, and provides a set of information necessary
    to construct the proper sequence. Thus, a subclass of this pass just needs to implement
    :meth:`_pad` method, in which the subclass constructs a circuit block to insert.
    This mechanism removes lots of boilerplate logic to manage whole DAG circuits.

    Note that padding pass subclasses should define interleaving sequences satisfying:

        - Interleaved sequence does not change start time of other nodes
        - Interleaved sequence should have total duration of the provided ``time_interval``.

    Any manipulation violating these constraints may prevent this base pass from correctly
    tracking the start time of each instruction,
    which may result in violation of hardware alignment constraints.
    """

    def __init__(
        self,
        schedule_idle_qubits: bool = False,
        block_ordering_callable: Optional[BlockOrderingCallableType] = None,
    ) -> None:

        self._node_start_time = None
        self._node_block_dags = None
        self._idle_after: Optional[Dict[Qubit, int]] = None
        self._root_dag = None
        self._dag = None
        self._block_dag = None
        self._prev_node: Optional[DAGNode] = None
        self._wire_map: Optional[Dict[Bit, Bit]] = None
        self._block_duration = 0
        self._current_block_idx = 0
        self._conditional_block = False
        self._bit_indices: Optional[Dict[Qubit, int]] = None
        # Nodes that the scheduling of this node is tied to.

        self._last_node_to_touch: Optional[Dict[Qubit, DAGNode]] = None
        # Last node to touch a bit

        self._fast_path_nodes: Set[DAGNode] = set()

        self._dirty_qubits: Set[Qubit] = set()
        # Qubits that are dirty in the circuit.
        self._schedule_idle_qubits = schedule_idle_qubits
        self._idle_qubits: Set[Qubit] = set()

        # Block ordering callable
        self._block_ordering_callable = (
            block_order_op_nodes if block_ordering_callable is None else block_ordering_callable
        )

        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the padding pass on ``dag``.

        Args:
            dag: DAG to be checked.

        Returns:
            DAGCircuit: DAG with idle time filled with instructions.

        Raises:
            TranspilerError: When a particular node is not scheduled, likely some transform pass
                is inserted before this node is called.
        """
        if not self._schedule_idle_qubits:
            self._idle_qubits = set(wire for wire in dag.idle_wires() if isinstance(wire, Qubit))
        self._pre_runhook(dag)

        self._init_run(dag)

        # Trivial wire map at the top-level
        wire_map = {wire: wire for wire in dag.wires}
        # Top-level dag is the entry block
        new_dag = self._visit_block(dag, wire_map)

        return new_dag

    def _init_run(self, dag: DAGCircuit) -> None:
        """Setup for initial run."""
        self._node_start_time = self.property_set["node_start_time"].copy()
        self._node_block_dags = self.property_set["node_block_dags"]
        self._idle_after = {bit: 0 for bit in dag.qubits}
        self._current_block_idx = 0
        self._conditional_block = False
        self._block_duration = 0

        # Prepare DAG to pad
        self._root_dag = dag
        self._dag = self._empty_dag_like(dag)
        self._block_dag = self._dag
        self._bit_indices = {q: index for index, q in enumerate(dag.qubits)}
        self._last_node_to_touch = {}
        self._fast_path_nodes = set()
        self._dirty_qubits = set()

        self.property_set["node_start_time"].clear()
        self._prev_node = None
        self._wire_map = {}

    def _empty_dag_like(
        self,
        dag: DAGCircuit,
        pad_wires: bool = True,
        wire_map: Optional[Dict[Qubit, Qubit]] = None,
        ignore_idle: bool = False,
    ) -> DAGCircuit:
        """Create an empty dag like the input dag."""
        new_dag = DAGCircuit()

        # Ensure *all* registers are included from the input circuit
        # so that they are scheduled in sub-blocks

        # The top-level QuantumCircuit has the full registers available
        # Control flow blocks do not get the full register added to the
        # block but just the bits. When testing for equivalency the register
        # information is taken into account. To work around this we try to
        # while enabling generic handling of QuantumCircuits we
        # add the register if available and otherwise add the bits directly.
        # We need this work around as otherwise the padded circuit will
        # not be equivalent to one written manually as bits will not
        # be defined on registers like in the test case.

        source_wire_dag = self._root_dag if pad_wires else dag

        # trivial wire map if not provided, or if the top-level dag is used
        if not wire_map or pad_wires:
            wire_map = {wire: wire for wire in source_wire_dag.wires}
        if dag.qregs and self._schedule_idle_qubits or not ignore_idle:
            for qreg in source_wire_dag.qregs.values():
                new_dag.add_qreg(qreg)
        else:
            new_dag.add_qubits(
                [
                    wire_map[qubit]
                    for qubit in source_wire_dag.qubits
                    if qubit not in self._idle_qubits or not ignore_idle
                ]
            )

        # Don't add root cargs as these will not be padded.
        # Just focus on current block dag.
        if dag.cregs:
            for creg in dag.cregs.values():
                new_dag.add_creg(creg)
        else:
            new_dag.add_clbits(dag.clbits)

        new_dag.name = dag.name
        new_dag.metadata = dag.metadata
        new_dag.unit = self.property_set["time_unit"] or "dt"
        if new_dag.unit != "dt":
            raise TranspilerError(
                'All blocks must have time units of "dt". '
                "Please run TimeUnitConversion pass prior to padding."
            )

        new_dag.calibrations = dag.calibrations
        new_dag.global_phase = dag.global_phase
        return new_dag

    def _pre_runhook(self, dag: DAGCircuit) -> None:
        """Extra routine inserted before running the padding pass.

        Args:
            dag: DAG circuit on which the sequence is applied.

        Raises:
            TranspilerError: If the whole circuit or instruction is not scheduled.
        """
        if "node_start_time" not in self.property_set:
            raise TranspilerError(
                f"The input circuit {dag.name} is not scheduled. Call one of scheduling passes "
                f"before running the {self.__class__.__name__} pass."
            )

    def _pad(
        self,
        block_idx: int,
        qubit: Qubit,
        t_start: int,
        t_end: int,
        next_node: DAGNode,
        prev_node: DAGNode,
        enable_dd: bool = False,
    ) -> None:
        """Interleave instruction sequence in between two nodes.

        .. note::
            If a DAGOpNode is added here, it should update node_start_time property
            in the property set so that the added node is also scheduled.
            This is achieved by adding operation via :meth:`_apply_scheduled_op`.

        .. note::

            This method doesn't check if the total duration of new DAGOpNode added here
            is identical to the interval (``t_end - t_start``).
            A developer of the pass must guarantee this is satisfied.
            If the duration is greater than the interval, your circuit may be
            compiled down to the target code with extra duration on the backend compiler,
            which is then played normally without error. However, the outcome of your circuit
            might be unexpected due to erroneous scheduling.

        Args:
            block_idx: Execution block index for this node.
            qubit: The wire that the sequence is applied on.
            t_start: Absolute start time of this interval.
            t_end: Absolute end time of this interval.
            next_node: Node that follows the sequence.
            prev_node: Node ahead of the sequence.
        """
        raise NotImplementedError

    def _get_node_duration(self, node: DAGNode) -> int:
        """Get the duration of a node."""
        if node.op.condition_bits or isinstance(node.op, ControlFlowOp):
            # As we cannot currently schedule through conditionals model
            # as zero duration to avoid padding.
            return 0

        indices = [self._bit_indices[qarg] for qarg in self._map_wires(node.qargs)]

        if self._block_dag.has_calibration_for(node):
            # If node has calibration, this value should be the highest priority
            cal_key = tuple(indices), tuple(float(p) for p in node.op.params)
            duration = self._block_dag.calibrations[node.op.name][cal_key].duration
        else:
            duration = self._durations.get(node.op, indices, unit="dt")

        if isinstance(duration, ParameterExpression):
            raise TranspilerError(
                f"Parameterized duration ({duration}) "
                f"of {node.op.name} on qubits {indices} is not bounded."
            )
        if duration is None:
            raise TranspilerError(f"Duration of {node.op.name} on qubits {indices} is not found.")

        return duration

    def _needs_block_terminating_barrier(self, prev_node: DAGNode, curr_node: DAGNode) -> bool:
        # Only barrier if not in fast-path nodes
        is_fast_path_node = curr_node in self._fast_path_nodes

        def _is_terminating_barrier(node: DAGNode) -> bool:
            return (
                isinstance(node.op, (Barrier, ControlFlowOp))
                and len(node.qargs) == self._block_dag.num_qubits()
            )

        return not (
            prev_node is None
            or (isinstance(prev_node.op, ControlFlowOp) and isinstance(curr_node.op, ControlFlowOp))
            or _is_terminating_barrier(prev_node)
            or _is_terminating_barrier(curr_node)
            or is_fast_path_node
        )

    def _add_block_terminating_barrier(
        self, block_idx: int, time: int, current_node: DAGNode, force: bool = False
    ) -> None:
        """Add a block terminating barrier to prevent topological ordering slide by.

        TODO: Fix by ensuring control-flow is a block terminator in the core circuit IR.
        """
        # Only add a barrier to the end if a viable barrier is not already present on all qubits
        # Only barrier if not in fast-path nodes
        needs_terminating_barrier = True
        if not force:
            needs_terminating_barrier = self._needs_block_terminating_barrier(
                self._prev_node, current_node
            )

        if needs_terminating_barrier:
            # Terminate with a barrier to ensure topological ordering does not slide past
            if self._schedule_idle_qubits:
                barrier = Barrier(self._block_dag.num_qubits())
                qubits = self._block_dag.qubits
            else:
                barrier = Barrier(self._block_dag.num_qubits() - len(self._idle_qubits))
                qubits = [x for x in self._block_dag.qubits if x not in self._idle_qubits]

            barrier_node = self._apply_scheduled_op(
                block_idx,
                time,
                barrier,
                qubits,
                [],
            )
            barrier_node.op.duration = 0

    def _visit_block(
        self,
        block: DAGCircuit,
        wire_map: Dict[Qubit, Qubit],
        pad_wires: bool = True,
        ignore_idle: bool = False,
    ) -> DAGCircuit:
        # Push the previous block dag onto the stack
        prev_node = self._prev_node
        self._prev_node = None
        prev_wire_map, self._wire_map = self._wire_map, wire_map

        prev_block_dag = self._block_dag
        self._block_dag = new_block_dag = self._empty_dag_like(
            block, pad_wires, wire_map=wire_map, ignore_idle=ignore_idle
        )

        self._block_duration = 0
        self._conditional_block = False

        for node in self._block_ordering_callable(block):
            enable_dd_node = False
            # add DD if node is a named barrier
            if (
                isinstance(node, DAGOpNode)
                and isinstance(node.op, Barrier)
                and getattr(self, "_dd_barrier", None)
                and node.op.label
                and self._dd_barrier in node.op.label
            ):
                enable_dd_node = True
            self._visit_node(node, enable_dd=enable_dd_node)

        # Terminate the block to pad it after scheduling.
        prev_block_duration = self._block_duration
        prev_block_idx = self._current_block_idx
        self._terminate_block(self._block_duration, self._current_block_idx)
        new_block_dag.duration = prev_block_duration

        # Edge-case: Add a barrier if the final node is a fast-path
        if self._prev_node in self._fast_path_nodes:
            self._add_block_terminating_barrier(
                prev_block_duration, prev_block_idx, self._prev_node, force=True
            )

        # Pop the previous block dag off the stack restoring it
        self._block_dag = prev_block_dag
        self._prev_node = prev_node
        self._wire_map = prev_wire_map

        return new_block_dag

    def _visit_node(self, node: DAGNode, enable_dd: bool = False) -> None:
        if isinstance(node.op, ControlFlowOp):
            if isinstance(node.op, IfElseOp):
                self._visit_if_else_op(node)
            else:
                self._visit_control_flow_op(node)
        elif node in self._node_start_time:
            if isinstance(node.op, Delay):
                self._visit_delay(node)
            else:
                self._visit_generic(node, enable_dd=enable_dd)
        else:
            raise TranspilerError(
                f"Operation {repr(node)} is likely added after the circuit is scheduled. "
                "Schedule the circuit again if you transformed it."
            )
        self._prev_node = node

    def _visit_if_else_op(self, node: DAGNode) -> None:
        """check if is fast-path eligible otherwise fall back
        to standard ControlFlowOp handling."""

        if self._will_use_fast_path(node):
            self._fast_path_nodes.add(node)
        self._visit_control_flow_op(node)

    def _will_use_fast_path(self, node: DAGNode) -> bool:
        """Check if this conditional operation will be scheduled on the fastpath.
        This will happen if
        1. This operation is a direct descendent of a current measurement block to be flushed
        2. The operation only operates on the qubit that is measured.
        """
        # Verify IfElseOp has a direct measurement predecessor
        condition_bits = node.op.condition_bits
        # Fast-path valid only with a single bit.
        if not condition_bits or len(condition_bits) > 1:
            return False

        bit = condition_bits[0]
        last_node, last_node_dag = self._last_node_to_touch.get(bit, (None, None))

        last_node_in_block = last_node_dag is self._block_dag

        if not (
            last_node_in_block
            and isinstance(last_node.op, Measure)
            and set(self._map_wires(node.qargs)) == set(self._map_wires(last_node.qargs))
        ):
            return False

        # Fast path contents are limited to gates and delays
        for block in node.op.blocks:
            if not all(isinstance(inst.operation, (Gate, Delay)) for inst in block.data):
                return False
        return True

    def _visit_control_flow_op(self, node: DAGNode) -> None:
        """Visit a control-flow node to pad."""

        # Control-flow terminator ends scheduling of block currently
        block_idx, t0 = self._node_start_time[node]  # pylint: disable=invalid-name
        self._terminate_block(t0, block_idx)
        self._add_block_terminating_barrier(block_idx, t0, node)

        # Only pad non-fast path nodes
        fast_path_node = node in self._fast_path_nodes

        # TODO: This is a hack required to tie nodes of control-flow
        # blocks across the scheduler and block_base_padder. This is
        # because the current control flow nodes store the block as a
        # circuit which is not hashable. For processing we are currently
        # required to convert each circuit block to a dag which is inefficient
        # and causes node relationships stored in analysis to be lost between
        # passes as we are constantly recreating the block dags.
        # We resolve this here by extracting the cached dag blocks that were
        # stored by the scheduling pass.
        new_node_block_dags = []
        for block_idx, _ in enumerate(node.op.blocks):
            block_dag = self._node_block_dags[node][block_idx]
            inner_wire_map = {
                inner: outer
                for outer, inner in zip(
                    self._map_wires(node.qargs + node.cargs),
                    block_dag.qubits + block_dag.clbits,
                )
            }
            new_node_block_dags.append(
                self._visit_block(
                    block_dag,
                    pad_wires=not fast_path_node,
                    wire_map=inner_wire_map,
                    ignore_idle=True,
                )
            )

        # Build new control-flow operation containing scheduled blocks
        # and apply to the DAG.
        new_control_flow_op = node.op.replace_blocks(
            dag_to_circuit(block) for block in new_node_block_dags
        )
        # Enforce that this control-flow operation contains all wires since it has now been padded
        # such that each qubit is scheduled within each block. Don't added all cargs as these will not
        # be padded.
        if fast_path_node:
            padded_qubits = node.qargs
        elif not self._schedule_idle_qubits:
            padded_qubits = [q for q in self._block_dag.qubits if q not in self._idle_qubits]
        else:
            padded_qubits = self._block_dag.qubits
        self._apply_scheduled_op(
            block_idx,
            t0,
            new_control_flow_op,
            padded_qubits,
            self._map_wires(node.cargs),
        )

    def _visit_delay(self, node: DAGNode) -> None:
        """The padding class considers a delay instruction as idle time
        rather than instruction. Delay node is not added so that
        we can extract non-delay predecessors.
        """
        block_idx, t0 = self._node_start_time[node]  # pylint: disable=invalid-name
        # Trigger the end of a block
        if block_idx > self._current_block_idx:
            self._terminate_block(self._block_duration, self._current_block_idx)
            self._add_block_terminating_barrier(block_idx, t0, node)

        self._conditional_block = bool(node.op.condition_bits)

        self._current_block_idx = block_idx

        t1 = t0 + self._get_node_duration(node)  # pylint: disable=invalid-name
        self._block_duration = max(self._block_duration, t1)

    def _visit_generic(self, node: DAGNode, enable_dd: bool = False) -> None:
        """Visit a generic node to pad."""
        # Note: t0 is the relative time with respect to the current block specified
        # by block_idx.
        block_idx, t0 = self._node_start_time[node]  # pylint: disable=invalid-name

        # Trigger the end of a block
        if block_idx > self._current_block_idx:
            self._terminate_block(self._block_duration, self._current_block_idx)
            self._add_block_terminating_barrier(block_idx, t0, node)

        # This block will not be padded as it is conditional.
        # See TODO below.
        self._conditional_block = bool(node.op.condition_bits)

        # Now set the current block index.
        self._current_block_idx = block_idx

        t1 = t0 + self._get_node_duration(node)  # pylint: disable=invalid-name
        self._block_duration = max(self._block_duration, t1)

        for bit in self._map_wires(node.qargs):
            if bit in self._idle_qubits:
                continue
            # Fill idle time with some sequence
            if t0 - self._idle_after.get(bit, 0) > 0:
                # Find previous node on the wire, i.e. always the latest node on the wire
                prev_node = next(self._block_dag.predecessors(self._block_dag.output_map[bit]))

                self._pad(
                    block_idx=block_idx,
                    qubit=bit,
                    t_start=self._idle_after[bit],
                    t_end=t0,
                    next_node=node,
                    prev_node=prev_node,
                    enable_dd=enable_dd,
                )

            self._idle_after[bit] = t1

        if not isinstance(node.op, (Barrier, Delay)):
            self._dirty_qubits |= set(self._map_wires(node.qargs))

        new_node = self._apply_scheduled_op(
            block_idx,
            t0,
            node.op,
            self._map_wires(node.qargs),
            self._map_wires(node.cargs),
        )
        self._last_node_to_touch.update(
            {bit: (new_node, self._block_dag) for bit in new_node.qargs + new_node.cargs}
        )

    def _terminate_block(self, block_duration: int, block_idx: int) -> None:
        """Terminate the end of a block scheduling region."""
        # Update all other qubits as not idle so that delays are *not*
        # inserted. This is because we need the delays to be inserted in
        # the conditional circuit block.
        self._block_duration = 0
        self._pad_until_block_end(block_duration, block_idx)
        self._idle_after = {bit: 0 for bit in self._block_dag.qubits}

    def _pad_until_block_end(self, block_duration: int, block_idx: int) -> None:
        # Add delays until the end of circuit.
        for bit in self._block_dag.qubits:
            if bit in self._idle_qubits:
                continue
            idle_after = self._idle_after.get(bit, 0)
            if block_duration - idle_after > 0:
                node = self._block_dag.output_map[bit]
                prev_node = next(self._block_dag.predecessors(node))
                self._pad(
                    block_idx=block_idx,
                    qubit=bit,
                    t_start=idle_after,
                    t_end=block_duration,
                    next_node=node,
                    prev_node=prev_node,
                )

    def _apply_scheduled_op(
        self,
        block_idx: int,
        t_start: int,
        oper: Instruction,
        qubits: Union[Qubit, Iterable[Qubit]],
        clbits: Union[Clbit, Iterable[Clbit]] = (),
    ) -> DAGNode:
        """Add new operation to DAG with scheduled information.

        This is identical to apply_operation_back + updating the node_start_time propety.

        Args:
            block_idx: Execution block index for this node.
            t_start: Start time of new node.
            oper: New operation that is added to the DAG circuit.
            qubits: The list of qubits that the operation acts on.
            clbits: The list of clbits that the operation acts on.

        Returns:
            The DAGNode applied to.
        """
        if isinstance(qubits, Qubit):
            qubits = [qubits]
        if isinstance(clbits, Clbit):
            clbits = [clbits]

        new_node = self._block_dag.apply_operation_back(oper, qubits, clbits)
        self.property_set["node_start_time"][new_node] = (block_idx, t_start)
        return new_node

    def _map_wires(self, wires: Iterable[Bit]) -> List[Bit]:
        """Map the wires from the current block to the top-level block's wires.

        TODO: We should have an easier approach to wire mapping from the transpiler.
        """
        return [self._wire_map[w] for w in wires]
