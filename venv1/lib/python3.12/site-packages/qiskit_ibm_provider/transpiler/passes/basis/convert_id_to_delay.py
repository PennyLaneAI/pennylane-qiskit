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

"""Pass to convert Id gate operations to a delay instruction."""

from typing import Dict

from qiskit.converters import dag_to_circuit, circuit_to_dag

from qiskit.circuit import ControlFlowOp
from qiskit.circuit import Delay
from qiskit.circuit.library import IGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.instruction_durations import InstructionDurations


class ConvertIdToDelay(TransformationPass):
    """Convert :class:`qiskit.circuit.library.standard_gates.IGate` to
    a delay of the corresponding length.
    """

    def __init__(self, durations: InstructionDurations, gate: str = "sx"):
        """Convert :class:`qiskit.circuit.library.IGate` to a
        Convert :class:`qiskit.circuit.Delay`.

        Args:
            duration: Duration of the delay to replace the identity gate with.
            gate: Single qubit gate to extract duration from.
        """
        self.durations = durations
        self.gate = gate
        self._cached_durations: Dict[int, int] = {}

        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        self._run_inner(dag)
        return dag

    def _run_inner(self, dag: DAGCircuit) -> bool:
        """Run the pass on one :class:`.DAGCircuit`, mutating it.  Returns ``True`` if the circuit
        was modified and ``False`` if not."""
        modified = False
        qubit_index_map = {bit: index for index, bit in enumerate(dag.qubits)}
        for node in dag.op_nodes():
            if isinstance(node.op, ControlFlowOp):
                modified_blocks = False
                new_dags = []
                for block in node.op.blocks:
                    new_dag = circuit_to_dag(block)
                    modified_blocks |= self._run_inner(new_dag)
                    new_dags.append(new_dag)
                if not modified_blocks:
                    continue
                dag.substitute_node(
                    node,
                    node.op.replace_blocks(dag_to_circuit(block) for block in new_dags),
                    inplace=True,
                )
            elif isinstance(node.op, IGate):
                delay_op = Delay(self._get_duration(qubit_index_map[node.qargs[0]]))
                dag.substitute_node(node, delay_op, inplace=True)

                modified = True

        return modified

    def _get_duration(self, qubit: int) -> int:
        """Get the duration of a gate in dt."""
        duration = self._cached_durations.get(qubit, None)
        if duration:
            return duration

        duration = self.durations.get(self.gate, qubit)
        self._cached_durations[qubit] = duration

        return duration
