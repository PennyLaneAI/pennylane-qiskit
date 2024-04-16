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

"""Converters for migration from IBM Quantum BackendV1 to BackendV2."""

from typing import Any, Dict, List

from qiskit.transpiler.target import Target, InstructionProperties
from qiskit.utils.units import apply_prefix
from qiskit.circuit.library.standard_gates import (
    IGate,
    SXGate,
    XGate,
    CXGate,
    RZGate,
    ECRGate,
    CZGate,
)
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.circuit import IfElseOp, WhileLoopOp, ForLoopOp, SwitchCaseOp
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.delay import Delay
from qiskit.circuit.gate import Gate
from qiskit.circuit.measure import Measure
from qiskit.circuit.reset import Reset
from qiskit.providers.models import (
    BackendConfiguration,
    BackendProperties,
    PulseDefaults,
)

from ..ibm_qubit_properties import IBMQubitProperties


def convert_to_target(
    configuration: BackendConfiguration,
    properties: BackendProperties = None,
    defaults: PulseDefaults = None,
) -> Target:
    """Uses configuration, properties and pulse defaults
    to construct and return Target class.
    """
    name_mapping = {
        "id": IGate(),
        "sx": SXGate(),
        "x": XGate(),
        "cx": CXGate(),
        "rz": RZGate(Parameter("λ")),
        "reset": Reset(),
        "ecr": ECRGate(),
        "cz": CZGate(),
        "if_else": IfElseOp,
        "while_loop": WhileLoopOp,
        "for_loop": ForLoopOp,
        "switch_case": SwitchCaseOp,
    }
    custom_gates = {}
    target = None
    faulty_qubits = set()
    # Parse from properties if it exists
    if properties is not None:
        faulty_qubits = set(properties.faulty_qubits())
        qubit_properties = qubit_props_list_from_props(properties=properties)
        target = Target(num_qubits=configuration.n_qubits, qubit_properties=qubit_properties)
        # Parse instructions
        gates: Dict[str, Any] = {}
        for gate in properties.gates:
            name = gate.gate
            if name in name_mapping:
                if name not in gates:
                    gates[name] = {}
            elif name not in custom_gates:
                custom_gate = Gate(name, len(gate.qubits), [])
                custom_gates[name] = custom_gate
                gates[name] = {}

            qubits = tuple(gate.qubits)
            if any(not properties.is_qubit_operational(qubit) for qubit in qubits):
                continue
            if not properties.is_gate_operational(name, gate.qubits):
                continue
            gate_props = {}
            for param in gate.parameters:
                if param.name == "gate_error":
                    gate_props["error"] = param.value
                if param.name == "gate_length":
                    gate_props["duration"] = apply_prefix(param.value, param.unit)
            gates[name][qubits] = InstructionProperties(**gate_props)
        for gate, props in gates.items():
            if gate in name_mapping:
                inst = name_mapping.get(gate)
            else:
                inst = custom_gates[gate]
            target.add_instruction(inst, props)
        # Create measurement instructions:
        measure_props = {}
        for qubit, _ in enumerate(properties.qubits):
            if not properties.is_qubit_operational(qubit):
                continue
            measure_props[(qubit,)] = InstructionProperties(
                duration=properties.readout_length(qubit),
                error=properties.readout_error(qubit),
            )
        target.add_instruction(Measure(), measure_props)
    # Parse from configuration because properties doesn't exist
    else:
        target = Target(num_qubits=configuration.n_qubits)
        for gate in configuration.gates:
            name = gate.name
            gate_props = (
                {tuple(x): None for x in gate.coupling_map}  # type: ignore[misc]
                if hasattr(gate, "coupling_map")
                else {None: None}
            )
            gate_len = len(gate.coupling_map[0]) if hasattr(gate, "coupling_map") else 0
            if name in name_mapping:
                target.add_instruction(name_mapping[name], gate_props)
            else:
                custom_gate = Gate(name, gate_len, [])
                target.add_instruction(custom_gate, gate_props)
        target.add_instruction(Measure())
    # parse global configuration properties
    if hasattr(configuration, "dt"):
        target.dt = configuration.dt
    if hasattr(configuration, "timing_constraints"):
        target.granularity = configuration.timing_constraints.get("granularity")
        target.min_length = configuration.timing_constraints.get("min_length")
        target.pulse_alignment = configuration.timing_constraints.get("pulse_alignment")
        target.acquire_alignment = configuration.timing_constraints.get("acquire_alignment")
    supported_instructions = set(getattr(configuration, "supported_instructions", []))
    control_flow_ops = CONTROL_FLOW_OP_NAMES.intersection(supported_instructions)
    for op in control_flow_ops:
        target.add_instruction(name_mapping[op], name=op)
    # If pulse defaults exists use that as the source of truth
    if defaults is not None:
        inst_map = defaults.instruction_schedule_map
        for inst in inst_map.instructions:
            for qarg in inst_map.qubits_with_instruction(inst):
                sched = inst_map.get(inst, qarg)
                if inst in target:
                    try:
                        qarg = tuple(qarg)
                    except TypeError:
                        qarg = (qarg,)
                    if inst == "measure":
                        for qubit in qarg:
                            if qubit in faulty_qubits:
                                continue
                            target[inst][(qubit,)].calibration = sched
                    else:
                        if any(qubit in faulty_qubits for qubit in qarg):
                            continue
                        target[inst][qarg].calibration = sched
    if "delay" not in target:
        target.add_instruction(
            Delay(Parameter("t")),
            {(bit,): None for bit in range(target.num_qubits) if bit not in faulty_qubits},
        )
    return target


def qubit_props_list_from_props(
    properties: BackendProperties,
) -> List[IBMQubitProperties]:
    """Uses BackendProperties to construct
    and return a list of IBMQubitProperties.
    """
    qubit_props: List[IBMQubitProperties] = []
    for qubit, _ in enumerate(properties.qubits):
        try:
            t_1 = properties.t1(qubit)
        except Exception:  # pylint: disable=broad-except
            t_1 = None
        try:
            t_2 = properties.t2(qubit)
        except Exception:  # pylint: disable=broad-except
            t_2 = None
        try:
            frequency = properties.frequency(qubit)
        except Exception:  # pylint: disable=broad-except
            t_2 = None
        try:
            anharmonicity = properties.qubit_property(qubit, "anharmonicity")[0]
        except Exception:  # pylint: disable=broad-except
            anharmonicity = None
        qubit_props.append(
            IBMQubitProperties(  # type: ignore[no-untyped-call]
                t1=t_1,
                t2=t_2,
                frequency=frequency,
                anharmonicity=anharmonicity,
            )
        )
    return qubit_props
