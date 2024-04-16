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

"""Padding pass to insert Delay into empty timeslots for dynamic circuit backends."""

from typing import Optional

from qiskit.circuit import Qubit
from qiskit.circuit.delay import Delay
from qiskit.dagcircuit import DAGNode, DAGOutNode
from qiskit.transpiler.instruction_durations import InstructionDurations

from .block_base_padder import BlockBasePadder
from .utils import BlockOrderingCallableType


class PadDelay(BlockBasePadder):
    """Padding idle time with Delay instructions.

    Consecutive delays will be merged in the output of this pass.

    .. code-block::python

        durations = InstructionDurations([("x", None, 160), ("cx", None, 800)])

        qc = QuantumCircuit(2)
        qc.delay(100, 0)
        qc.x(1)
        qc.cx(0, 1)

    The ASAP-scheduled circuit output may become

    .. parsed-literal::

             ┌────────────────┐
        q_0: ┤ Delay(160[dt]) ├──■──
             └─────┬───┬──────┘┌─┴─┐
        q_1: ──────┤ X ├───────┤ X ├
                   └───┘       └───┘

    Note that the additional idle time of 60dt on the ``q_0`` wire coming from the duration difference
    between ``Delay`` of 100dt (``q_0``) and ``XGate`` of 160 dt (``q_1``) is absorbed in
    the delay instruction on the ``q_0`` wire, i.e. in total 160 dt.

    See :class:`BlockBasePadder` pass for details.
    """

    def __init__(
        self,
        durations: InstructionDurations,
        fill_very_end: bool = True,
        schedule_idle_qubits: bool = False,
        block_ordering_callable: Optional[BlockOrderingCallableType] = None,
    ):
        """Create new padding delay pass.

        Args:
            durations: Durations of instructions to be used in scheduling.
            fill_very_end: Set ``True`` to fill the end of circuit with delay.
            schedule_idle_qubits: Set to true if you'd like a delay inserted on idle qubits.
                This is useful for timeline visualizations, but may cause issues for execution
                on large backends.
            block_ordering_callable: A callable used to produce an ordering of the nodes to minimize
                the number of blocks needed. If not provided, :func:`~block_order_op_nodes` will be
                used.
        """
        super().__init__(
            schedule_idle_qubits=schedule_idle_qubits,
            block_ordering_callable=block_ordering_callable,
        )
        self._durations = durations
        self.fill_very_end = fill_very_end

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
        if not self.fill_very_end and isinstance(next_node, DAGOutNode):
            return

        time_interval = t_end - t_start
        self._apply_scheduled_op(block_idx, t_start, Delay(time_interval, "dt"), qubit)
