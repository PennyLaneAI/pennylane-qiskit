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

"""Dynamical decoupling insertion pass for IBM (dynamic circuit) backends."""

import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import rustworkx as rx
from qiskit.circuit import Qubit, Gate
from qiskit.circuit.delay import Delay
from qiskit.circuit.library.standard_gates import IGate, UGate, U3Gate
from qiskit.circuit.reset import Reset
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGInNode, DAGOpNode
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes.optimization import Optimize1qGates
from qiskit.transpiler import CouplingMap

try:
    from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer
except ImportError:
    from qiskit.synthesis import OneQubitEulerDecomposer

from .block_base_padder import BlockBasePadder
from .utils import BlockOrderingCallableType


class PadDynamicalDecoupling(BlockBasePadder):
    """Dynamical decoupling insertion pass for IBM dynamic circuit backends.

    This pass works on a scheduled, physical circuit. It scans the circuit for
    idle periods of time (i.e. those containing delay instructions) and inserts
    a DD sequence of gates in those spots. These gates amount to the identity,
    so do not alter the logical action of the circuit, but have the effect of
    mitigating decoherence in those idle periods.
    As a special case, the pass allows a length-1 sequence (e.g. [XGate()]).
    In this case the DD insertion happens only when the gate inverse can be
    absorbed into a neighboring gate in the circuit (so we would still be
    replacing Delay with something that is equivalent to the identity).
    This can be used, for instance, as a Hahn echo.
    This pass ensures that the inserted sequence preserves the circuit exactly
    (including global phase).

    .. jupyter-execute::

        import numpy as np
        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import XGate
        from qiskit.transpiler import PassManager, InstructionDurations
        from qiskit.visualization import timeline_drawer

        from qiskit_ibm_runtime.transpiler.passes.scheduling import ALAPScheduleAnalysis
        from qiskit_ibm_runtime.transpiler.passes.scheduling import PadDynamicalDecoupling

        circ = QuantumCircuit(4)
        circ.h(0)
        circ.cx(0, 1)
        circ.cx(1, 2)
        circ.cx(2, 3)
        circ.measure_all()
        durations = InstructionDurations(
            [("h", 0, 50), ("cx", [0, 1], 700), ("reset", None, 10),
             ("cx", [1, 2], 200), ("cx", [2, 3], 300),
             ("x", None, 50), ("measure", None, 1000)]
        )

    .. jupyter-execute::

        # balanced X-X sequence on all qubits
        dd_sequence = [XGate(), XGate()]
        pm = PassManager([ALAPScheduleAnalysis(durations),
                          PadDynamicalDecoupling(durations, dd_sequence)])
        circ_dd = pm.run(circ)
        circ_dd.draw()

    .. jupyter-execute::

        # Uhrig sequence on qubit 0
        n = 8
        dd_sequence = [XGate()] * n
        def uhrig_pulse_location(k):
            return np.sin(np.pi * (k + 1) / (2 * n + 2)) ** 2
        spacings = []
        for k in range(n):
            spacings.append(uhrig_pulse_location(k) - sum(spacings))
        spacings.append(1 - sum(spacings))
        pm = PassManager(
            [
                ALAPScheduleAnalysis(durations),
                PadDynamicalDecoupling(durations, dd_sequence, qubits=[0], spacings=spacings),
            ]
        )
        circ_dd = pm.run(circ)
        circ_dd.draw()

    .. note::

        You need to call
        :class:`~qiskit_ibm_runtime.transpiler.passes.scheduling.ALAPScheduleAnalysis`
        before running dynamical decoupling to guarantee your circuit satisfies acquisition
        alignment constraints for dynamic circuit backends.
    """

    def __init__(
        self,
        durations: InstructionDurations,
        dd_sequences: Union[List[Gate], List[List[Gate]]],
        qubits: Optional[List[int]] = None,
        spacings: Optional[Union[List[List[float]], List[float]]] = None,
        skip_reset_qubits: bool = True,
        pulse_alignment: int = 16,
        extra_slack_distribution: str = "middle",
        sequence_min_length_ratios: Optional[Union[int, List[int]]] = None,
        insert_multiple_cycles: bool = False,
        coupling_map: CouplingMap = None,
        alt_spacings: Optional[Union[List[List[float]], List[float]]] = None,
        schedule_idle_qubits: bool = False,
        dd_barrier: Optional[str] = None,
        block_ordering_callable: Optional[BlockOrderingCallableType] = None,
    ):
        """Dynamical decoupling initializer.

        Args:
            durations: Durations of instructions to be used in scheduling.
            dd_sequences: Sequence of gates to apply in idle spots.
                Alternatively a list of gate sequences may be supplied that
                will preferentially be inserted if there is a delay of sufficient
                duration. This may be tuned by the optionally supplied
                ``sequence_min_length_ratios``.
            qubits: Physical qubits on which to apply DD.
                If None, all qubits will undergo DD (when possible).
            spacings: A list of lists of spacings between the DD gates.
                The available slack will be divided according to this.
                The list length must be one more than the length of dd_sequence,
                and the elements must sum to 1. If None, a balanced spacing
                will be used [d/2, d, d, ..., d, d, d/2]. This spacing only
                applies to the first subcircuit, if a ``coupling_map`` is
                specified
            skip_reset_qubits: If True, does not insert DD on idle periods that
                immediately follow initialized/reset qubits
                (as qubits in the ground state are less susceptible to decoherence).
            pulse_alignment: The hardware constraints for gate timing allocation.
                This is usually provided from ``backend.configuration().timing_constraints``.
                If provided, the delay length, i.e. ``spacing``, is implicitly adjusted to
                satisfy this constraint.
            extra_slack_distribution: The option to control the behavior of DD sequence generation.
                The duration of the DD sequence should be identical to an idle time in the
                scheduled quantum circuit, however, the delay in between gates comprising the sequence
                should be integer number in units of dt, and it might be further truncated
                when ``pulse_alignment`` is specified. This sometimes results in the duration of
                the created sequence being shorter than the idle time
                that you want to fill with the sequence, i.e. `extra slack`.
                This option takes following values.

                    * "middle": Put the extra slack to the interval at the middle of the sequence.
                    * "edges": Divide the extra slack as evenly as possible into
                      intervals at beginning and end of the sequence.
            sequence_min_length_ratios: List of minimum delay length to DD sequence ratio to satisfy
                in order to insert the DD sequence. For example if the X-X dynamical decoupling sequence
                is 320dt samples long and the available delay is 384dt it has a ratio of 384dt/320dt=1.2.
                From the perspective of dynamical decoupling this is likely to add more control noise
                than decoupling error rate reductions. The defaults value is 2.0.
            insert_multiple_cycles: If the available duration exceeds
                2*sequence_min_length_ratio*duration(dd_sequence) enable the insertion of multiple
                rounds of the dynamical decoupling sequence in that delay.
            coupling_map: directed graph representing the coupling map for the device. Specifying a
                coupling map partitions the device into subcircuits, in order to apply DD sequences
                with different pulse spacings within each. Currently support 2 subcircuits.
            alt_spacings: A list of lists of spacings between the DD gates, for the second subcircuit,
                as determined by the coupling map. If None, a balanced spacing that is staggered with
                respect to the first subcircuit will be used [d, d, d, ..., d, d, 0].
            schedule_idle_qubits: Set to true if you'd like a delay inserted on idle qubits.
                This is useful for timeline visualizations, but may cause issues
                for execution on large backends.
            dd_barrier: only apply DD to delays terminating with a barrier
                whose label contains the specified string
            block_ordering_callable: A callable used to produce an ordering of the nodes to minimize the
                number of blocks needed. If not provided, :func:`~block_order_op_nodes` will be used.
        Raises:
            TranspilerError: When invalid DD sequence is specified.
            TranspilerError: When pulse gate with the duration which is
                non-multiple of the alignment constraint value is found.
            TranspilerError: When the coupling map is not supported (i.e., if degree > 3)
        """

        super().__init__(
            schedule_idle_qubits=schedule_idle_qubits,
            block_ordering_callable=block_ordering_callable,
        )
        self._durations = durations

        # Enforce list of DD sequences
        if dd_sequences:
            try:
                iter(dd_sequences[0])
            except TypeError:
                dd_sequences = [dd_sequences]
        self._dd_sequences = dd_sequences
        self._qubits = qubits
        self._skip_reset_qubits = skip_reset_qubits
        self._alignment = pulse_alignment
        self._coupling_map = coupling_map
        self._coupling_coloring = None
        self._dd_barrier = dd_barrier

        if spacings is not None:
            try:
                iter(spacings[0])  # type: ignore
            except TypeError:
                spacings = [spacings]  # type: ignore
        if alt_spacings is not None:
            try:
                iter(alt_spacings[0])  # type: ignore
            except TypeError:
                alt_spacings = [alt_spacings]  # type: ignore
        self._spacings = spacings
        self._alt_spacings = alt_spacings

        if self._spacings and len(self._spacings) != len(self._dd_sequences):
            raise TranspilerError("Number of sequence spacings must equal number of DD sequences.")

        if self._alt_spacings:
            if not self._coupling_map:
                warnings.warn(
                    "Alternate spacings are ignored because a coupling map was not provided"
                )
            elif len(self._alt_spacings) != len(self._dd_sequences):
                raise TranspilerError(
                    "Number of alternate sequence spacings must equal number of DD sequences."
                )

        self._extra_slack_distribution = extra_slack_distribution

        self._dd_sequence_lengths: Dict[Qubit, List[List[Gate]]] = {}
        self._sequence_phase = 0

        if sequence_min_length_ratios is None:
            # Use 2.0 as a sane default
            self._sequence_min_length_ratios = [2.0 for _ in self._dd_sequences]
        else:
            try:
                iter(sequence_min_length_ratios)  # type: ignore
            except TypeError:
                sequence_min_length_ratios = [sequence_min_length_ratios]  # type: ignore
            self._sequence_min_length_ratios = sequence_min_length_ratios  # type: ignore

        if len(self._sequence_min_length_ratios) != len(self._dd_sequences):
            raise TranspilerError("Number of sequence lengths must equal number of DD sequences.")

        self._insert_multiple_cycles = insert_multiple_cycles

    def _pre_runhook(self, dag: DAGCircuit) -> None:
        super()._pre_runhook(dag)

        if self._coupling_map:
            physical_qubits = [dag.qubits.index(q) for q in dag.qubits]
            subgraph = self._coupling_map.graph.subgraph(physical_qubits)
            self._coupling_coloring = rx.graph_greedy_color(subgraph.to_undirected())
            if any(c > 1 for c in self._coupling_coloring.values()):
                raise TranspilerError(
                    "This circuit topology is not supported for staggered dynamical decoupling."
                    "The maximum connectivity is 3 nearest neighbors per qubit."
                )

        spacings_required = self._spacings is None
        if spacings_required:
            self._spacings = []  # type: ignore
        alt_spacings_required = self._alt_spacings is None and self._coupling_map is not None
        if alt_spacings_required:
            self._alt_spacings = []  # type: ignore

        for seq_idx, seq in enumerate(self._dd_sequences):
            num_pulses = len(self._dd_sequences[seq_idx])

            # Check if physical circuit is given
            if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
                raise TranspilerError("DD runs on physical circuits only.")

            # Set default spacing otherwise validate user input
            if spacings_required:
                mid = 1 / num_pulses
                end = mid / 2
                self._spacings.append([end] + [mid] * (num_pulses - 1) + [end])  # type: ignore
            else:
                if sum(self._spacings[seq_idx]) != 1 or any(  # type: ignore
                    a < 0 for a in self._spacings[seq_idx]  # type: ignore
                ):
                    raise TranspilerError(
                        "The spacings must be given in terms of fractions "
                        "of the slack period and sum to 1."
                    )

            if self._coupling_map:
                if alt_spacings_required:
                    mid = 1 / num_pulses
                    self._alt_spacings.append([mid] * num_pulses + [0])  # type: ignore
                else:
                    if sum(self._alt_spacings[seq_idx]) != 1 or any(  # type: ignore
                        a < 0 for a in self._alt_spacings[seq_idx]  # type: ignore
                    ):
                        raise TranspilerError(
                            "The spacings must be given in terms of fractions "
                            "of the slack period and sum to 1."
                        )

            # Check if DD sequence is identity
            if num_pulses != 1:
                if num_pulses % 2 != 0:
                    raise TranspilerError(
                        "DD sequence must contain an even number of gates (or 1)."
                    )
                # TODO: this check should use the quantum info package in Qiskit.
                noop = np.eye(2)
                for gate in self._dd_sequences[seq_idx]:
                    noop = noop.dot(gate.to_matrix())
                if not matrix_equal(noop, IGate().to_matrix(), ignore_phase=True):
                    raise TranspilerError("The DD sequence does not make an identity operation.")
                self._sequence_phase = np.angle(noop[0][0])

            # Precompute qubit-wise DD sequence length for performance
            for qubit in dag.qubits:
                seq_length_ = []
                if qubit not in self._dd_sequence_lengths:
                    self._dd_sequence_lengths[qubit] = []

                physical_index = dag.qubits.index(qubit)
                if (
                    self._qubits
                    and physical_index not in self._qubits
                    or qubit in self._idle_qubits
                ):
                    continue

                for index, gate in enumerate(seq):
                    try:
                        # Check calibration.
                        gate_length = dag.calibrations[gate.name][(physical_index, gate.params)]
                        if gate_length % self._alignment != 0:
                            # This is necessary to implement lightweight scheduling logic for this pass.
                            # Usually the pulse alignment constraint and pulse data chunk size take
                            # the same value, however, we can intentionally violate this pattern
                            # at the gate level. For example, we can create a schedule consisting of
                            # a pi-pulse of 32 dt followed by a post buffer, i.e. delay, of 4 dt
                            # on the device with 16 dt constraint. Note that the pi-pulse length
                            # is multiple of 16 dt but the gate length of 36 is not multiple of it.
                            # Such pulse gate should be excluded.
                            raise TranspilerError(
                                f"Pulse gate {gate.name} with length non-multiple of {self._alignment} "
                                f"is not acceptable in {self.__class__.__name__} pass."
                            )
                    except KeyError:
                        gate_length = self._durations.get(gate, physical_index)
                    seq_length_.append(gate_length)
                    # Update gate duration.
                    # This is necessary for current timeline drawer, i.e. scheduled.

                    if hasattr(
                        gate, "to_mutable"
                    ):  # TODO this check can be removed after Qiskit 1.0, as it is always True
                        gate = gate.to_mutable()
                        seq[index] = gate
                    gate.duration = gate_length
                self._dd_sequence_lengths[qubit].append(seq_length_)

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
        # This routine takes care of the pulse alignment constraint for the DD sequence.
        # Note that the alignment constraint acts on the t0 of the DAGOpNode.
        # Now this constrained scheduling problem is simplified to the problem of
        # finding a delay amount which is a multiple of the constraint value by assuming
        # that the duration of every DAGOpNode is also a multiple of the constraint value.
        #
        # For example, given the constraint value of 16 and XY4 with 160 dt gates.
        # Here we assume current interval is 992 dt.
        #
        # relative spacing := [0.125, 0.25, 0.25, 0.25, 0.125]
        # slack = 992 dt - 4 x 160 dt = 352 dt
        #
        # unconstrained sequence: 44dt-X1-88dt-Y2-88dt-X3-88dt-Y4-44dt
        # constrained sequence  : 32dt-X1-80dt-Y2-80dt-X3-80dt-Y4-32dt + extra slack 48 dt
        #
        # Now we evenly split extra slack into start and end of the sequence.
        # The distributed slack should be multiple of 16.
        # Start = +16, End += 32
        #
        # final sequence       : 48dt-X1-80dt-Y2-80dt-X3-80dt-Y4-64dt / in total 992 dt
        #
        # Now we verify t0 of every node starts from multiple of 16 dt.
        #
        # X1:  48 dt (3 x 16 dt)
        # Y2:  48 dt + 160 dt + 80 dt = 288 dt (18 x 16 dt)
        # Y3: 288 dt + 160 dt + 80 dt = 528 dt (33 x 16 dt)
        # Y4: 368 dt + 160 dt + 80 dt = 768 dt (48 x 16 dt)
        #
        # As you can see, constraints on t0 are all satified without explicit scheduling.
        time_interval = t_end - t_start

        if self._qubits and self._block_dag.qubits.index(qubit) not in self._qubits:
            # Target physical qubit is not the target of this DD sequence.
            self._apply_scheduled_op(
                block_idx, t_start, Delay(time_interval, self._block_dag.unit), qubit
            )
            return

        if not self._skip_reset_qubits and qubit not in self._dirty_qubits:
            # mark all qubits as dirty if skip_reset_qubits is False
            self._dirty_qubits.update([qubit])

        if (
            not isinstance(prev_node, DAGInNode)
            and self._skip_reset_qubits
            and isinstance(prev_node.op, Reset)
            and qubit in prev_node.qargs
        ):
            self._dirty_qubits.remove(qubit)

        if qubit not in self._dirty_qubits or (self._dd_barrier and not enable_dd):
            # Previous node is the start edge or reset, i.e. qubit is ground state;
            # or dd to be applied before named barrier only
            self._apply_scheduled_op(
                block_idx, t_start, Delay(time_interval, self._block_dag.unit), qubit
            )
            return

        for sequence_idx, _ in enumerate(self._dd_sequences):
            dd_sequence = self._dd_sequences[sequence_idx]
            seq_lengths = self._dd_sequence_lengths[qubit][sequence_idx]
            seq_length = np.sum(seq_lengths)
            seq_ratio = self._sequence_min_length_ratios[sequence_idx]
            spacings = self._spacings[sequence_idx]
            alt_spacings = (
                np.asarray(self._alt_spacings[sequence_idx]) if self._coupling_map else None
            )

            # Verify the delay duration exceeds the minimum time to insert
            if time_interval / seq_length <= seq_ratio:
                continue

            if self._insert_multiple_cycles:
                num_sequences = max(int(time_interval // (seq_length * seq_ratio)), 1)
                if (num_sequences % 2 == 1) and len(dd_sequence) == 1:
                    warnings.warn(
                        "Sequence would result in an odd number of DD cycles with original DD "
                        "sequence of length 1. This may result in non-identity sequence insertion "
                        "and so are defaulting to 1 cycle insertion."
                    )
                    num_sequences = 1
            else:
                num_sequences = 1

            # multiple dd sequences may be inserted
            if num_sequences > 1:
                dd_sequence = list(dd_sequence) * num_sequences
                seq_lengths = seq_lengths * num_sequences
                seq_length = np.sum(seq_lengths)
                spacings = spacings * num_sequences

            spacings = np.asarray(spacings) / num_sequences
            slack = time_interval - seq_length
            sequence_gphase = self._sequence_phase

            if slack <= 0:
                continue

            if len(dd_sequence) == 1:
                # Special case of using a single gate for DD
                u_inv = dd_sequence[0].inverse().to_matrix()
                theta, phi, lam, phase = OneQubitEulerDecomposer().angles_and_phase(u_inv)
                if isinstance(next_node, DAGOpNode) and isinstance(next_node.op, (UGate, U3Gate)):
                    # Absorb the inverse into the successor (from left in circuit)
                    theta_r, phi_r, lam_r = next_node.op.params
                    next_node.op.params = Optimize1qGates.compose_u3(
                        theta_r, phi_r, lam_r, theta, phi, lam
                    )
                    sequence_gphase += phase
                elif isinstance(prev_node, DAGOpNode) and isinstance(prev_node.op, (UGate, U3Gate)):
                    # Absorb the inverse into the predecessor (from right in circuit)
                    theta_l, phi_l, lam_l = prev_node.op.params
                    prev_node.op.params = Optimize1qGates.compose_u3(
                        theta, phi, lam, theta_l, phi_l, lam_l
                    )
                    sequence_gphase += phase
                else:
                    # Don't do anything if there's no single-qubit gate to absorb the inverse
                    self._apply_scheduled_op(
                        block_idx,
                        t_start,
                        Delay(time_interval, self._block_dag.unit),
                        qubit,
                    )
                    return

            def _constrained_length(values: np.array) -> np.array:
                return self._alignment * np.floor(values / self._alignment)

            if self._coupling_map:
                if self._coupling_coloring[self._dag.qubits.index(qubit)] == 0:
                    sub_spacings = spacings
                else:
                    sub_spacings = alt_spacings
            else:
                sub_spacings = spacings

            # (1) Compute DD intervals satisfying the constraint
            taus = _constrained_length(slack * sub_spacings)
            extra_slack = slack - np.sum(taus)
            # (2) Distribute extra slack
            if self._extra_slack_distribution == "middle":
                mid_ind = int((len(taus) - 1) / 2)
                to_middle = _constrained_length(extra_slack)
                taus[mid_ind] += to_middle
                if extra_slack - to_middle:
                    # If to_middle is not a multiple value of the pulse alignment,
                    # it is truncated to the nearest multiple value and
                    # the rest of slack is added to the end.
                    taus[-1] += extra_slack - to_middle
            elif self._extra_slack_distribution == "edges":
                to_begin_edge = _constrained_length(extra_slack / 2)
                taus[0] += to_begin_edge
                taus[-1] += extra_slack - to_begin_edge
            else:
                raise TranspilerError(
                    f"Option extra_slack_distribution = {self._extra_slack_distribution} is invalid."
                )

            # (3) Construct DD sequence with delays
            idle_after = t_start
            dd_ind = 0
            # Interleave delays with DD sequence operations
            for tau_idx, tau in enumerate(taus):
                if tau > 0:
                    self._apply_scheduled_op(
                        block_idx, idle_after, Delay(tau, self._dag.unit), qubit
                    )
                    idle_after += tau

                # Detect if we are on a sequence boundary
                # If so skip insert of sequence to allow delays to combine
                # There are two cases.
                # 1. The number of delays to be inserted is equal to the number of gates.
                # 2. There is an extra delay inserted after the last operation.
                # The condition below handles both.
                seq_length = int(len(taus) / num_sequences)
                if len(dd_sequence) == len(taus) or tau_idx % seq_length != (seq_length - 1):
                    gate = dd_sequence[dd_ind]
                    gate_length = seq_lengths[dd_ind]
                    self._apply_scheduled_op(block_idx, idle_after, gate, qubit)
                    idle_after += gate_length
                    dd_ind += 1

            self._block_dag.global_phase = self._block_dag.global_phase + sequence_gphase
            return

        # DD could not be applied, delay instead
        self._apply_scheduled_op(
            block_idx, t_start, Delay(time_interval, self._block_dag.unit), qubit
        )
        return
