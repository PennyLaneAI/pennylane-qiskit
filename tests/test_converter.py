# Copyright 2021-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
This module contains tests for converting circuits for PennyLane IBMQ devices.
"""
import sys
from typing import cast

import itertools as it
import functools as ft

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import library as lib
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.classical import expr
from qiskit.circuit.library import DraperQFTAdder
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.channel import Kraus

import pennylane as qml
from pennylane import I, X, Y, Z
from pennylane import numpy as np
from pennylane.measurements import MidMeasureMP
from pennylane.noise import op_in, wires_in, partial_wires
from pennylane.operation import AnyWires
from pennylane.tape.qscript import QuantumScript
from pennylane.wires import Wires
from pennylane_qiskit.converter import (
    load,
    load_pauli_op,
    load_noise_model,
    load_qasm,
    load_qasm_from_file,
    map_wires,
    circuit_to_qiskit,
    operation_to_qiskit,
    mp_to_pauli,
    _format_params_dict,
    _check_parameter_bound,
)


# pylint: disable=protected-access, unused-argument, too-many-arguments

THETA = np.linspace(0.11, 3, 5)
PHI = np.linspace(0.32, 3, 5)
VARPHI = np.linspace(0.02, 3, 5)


class TestConverterQiskitToPennyLane:
    """Tests the converter function that allows converting QuantumCircuit objects
    to Pennylane templates."""

    def test_quantum_circuit_init_by_specifying_rotation_in_circuit(self, recorder):
        """Tests the load method for a QuantumCircuit initialized using separately defined
        quantum and classical registers."""

        angle = 0.5

        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)

        qc = QuantumCircuit(qr, cr)
        qc.rz(angle, [0])

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit()

        assert len(recorder.queue) == 1
        assert recorder.queue[0].name == "RZ"
        assert recorder.queue[0].parameters == [angle]
        assert recorder.queue[0].wires == Wires([0])

    def test_quantum_circuit_by_passing_parameters(self, recorder):
        """Tests the load method for a QuantumCircuit initialized by passing the number
        of registers required."""

        theta = Parameter("θ")
        angle = 0.5

        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit(params={theta: angle})

        assert len(recorder.queue) == 1
        assert recorder.queue[0].name == "RZ"
        assert recorder.queue[0].parameters == [angle]
        assert recorder.queue[0].wires == Wires([0])

    def test_loaded_quantum_circuit_and_further_pennylane_operations(self, recorder):
        """Tests that a loaded quantum circuit can be used around other PennyLane
        templates in a circuit."""

        theta = Parameter("θ")
        angle = 0.5

        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])

        quantum_circuit = load(qc)

        with recorder:
            qml.PauliZ(0)
            quantum_circuit(params={theta: angle})
            qml.Hadamard(0)

        assert len(recorder.queue) == 3
        assert recorder.queue[0].name == "PauliZ"
        assert recorder.queue[0].parameters == []
        assert recorder.queue[0].wires == Wires([0])
        assert recorder.queue[1].name == "RZ"
        assert recorder.queue[1].parameters == [angle]
        assert recorder.queue[1].wires == Wires([0])
        assert recorder.queue[2].name == "Hadamard"
        assert recorder.queue[2].parameters == []
        assert recorder.queue[2].wires == Wires([0])

    def test_quantum_circuit_with_multiple_parameters(self, recorder):
        """Tests loading a circuit with multiple parameters."""

        angle1 = 0.5
        angle2 = 0.3

        phi = Parameter("φ")
        theta = Parameter("θ")

        qc = QuantumCircuit(3, 1)
        qc.rx(phi, 1)
        qc.rz(theta, 0)

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit(params={phi: angle1, theta: angle2})

        assert len(recorder.queue) == 2
        assert recorder.queue[0].name == "RX"
        assert recorder.queue[0].parameters == [angle1]
        assert recorder.queue[0].wires == Wires([1])
        assert recorder.queue[1].name == "RZ"
        assert recorder.queue[1].parameters == [angle2]
        assert recorder.queue[1].wires == Wires([0])

    def test_quantum_circuit_with_gate_requiring_multiple_parameters(self, recorder):
        """Tests loading a circuit containing a gate that requires
        multiple parameters."""

        angle1 = 0.5
        angle2 = 0.3
        angle3 = 0.1

        phi = Parameter("φ")
        lam = Parameter("λ")
        theta = Parameter("θ")

        qc = QuantumCircuit(3, 1)
        qc.u(phi, lam, theta, [0])

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit(params={phi: angle1, lam: angle2, theta: angle3})

        assert recorder.queue[0].name == "U3"
        assert len(recorder.queue[0].parameters) == 3
        assert recorder.queue[0].parameters == [0.5, 0.3, 0.1]
        assert recorder.queue[0].wires == Wires([0])

    def test_longer_parameter_expression(self):
        """Tests parameter expression with arbitrary operations and length"""

        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")

        a_val = 0.1
        b_val = 0.2
        c_val = 0.3

        qc = QuantumCircuit(1, 1)
        qc.rx(a * np.cos(b) + c, [0])

        quantum_circuit = load(qc)

        with qml.tape.QuantumTape() as tape:
            quantum_circuit(params={a: a_val, b: b_val, c: c_val}, wires=(0,))

        recorded_op = tape.operations[0]
        assert isinstance(recorded_op, qml.RX)
        assert recorded_op.parameters == a_val * np.cos(b_val) + c_val

    def test_quantum_circuit_loaded_multiple_times_with_different_arguments(self, recorder):
        """Tests that a loaded quantum circuit can be called multiple times with
        different arguments."""

        theta = Parameter("θ")
        angle1 = 0.5
        angle2 = -0.5
        angle3 = 0

        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit(params={theta: angle1})
            quantum_circuit(params={theta: angle2})
            quantum_circuit(params={theta: angle3})

        assert len(recorder.queue) == 3
        assert recorder.queue[0].name == "RZ"
        assert recorder.queue[0].parameters == [angle1]
        assert recorder.queue[0].wires == Wires([0])
        assert recorder.queue[1].name == "RZ"
        assert recorder.queue[1].parameters == [angle2]
        assert recorder.queue[1].wires == Wires([0])
        assert recorder.queue[2].name == "RZ"
        assert recorder.queue[2].parameters == [angle3]
        assert recorder.queue[2].wires == Wires([0])

    def test_quantum_circuit_with_bound_parameters(self, recorder):
        """Tests loading a quantum circuit that already had bound parameters."""

        theta = Parameter("θ")

        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])
        qc_1 = qc.assign_parameters({theta: 0.5})

        quantum_circuit = load(qc_1)

        with recorder:
            quantum_circuit()

        assert len(recorder.queue) == 1
        assert recorder.queue[0].name == "RZ"
        assert recorder.queue[0].parameters == [0.5]
        assert recorder.queue[0].wires == Wires([0])

    def test_pass_parameters_to_bind(self, recorder):
        """Tests parameter binding by passing parameters when loading a quantum circuit."""

        theta = Parameter("θ")

        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit(params={theta: 0.5})

        assert len(recorder.queue) == 1
        assert recorder.queue[0].name == "RZ"
        assert recorder.queue[0].parameters == [0.5]
        assert recorder.queue[0].wires == Wires([0])

    def test_unused_parameters_are_ignored(self, recorder):
        """Tests that unused parameters are ignored during assignment."""
        a, b, c = [Parameter(var) for var in "abc"]
        v = ParameterVector("v", 2)

        qc = QuantumCircuit(1)
        qc.rz(a, [0])

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit(params={a: 0.1, b: 0.2, c: 0.3, v: [0.4, 0.5]})

        assert len(recorder.queue) == 1
        assert recorder.queue[0].name == "RZ"
        assert recorder.queue[0].parameters == [0.1]
        assert recorder.queue[0].wires == Wires([0])

    def test_unused_parameter_vector_items_are_ignored(self, recorder):
        """Tests that unused parameter vector items are ignored during assignment."""
        a, b = [Parameter(var) for var in "ab"]
        v = ParameterVector("v", 3)

        qc = QuantumCircuit(1)
        qc.rz(v[1], [0])

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit(params={a: 0.1, b: 0.2, v: [0.3, 0.4, 0.5]})

        assert len(recorder.queue) == 1
        assert recorder.queue[0].name == "RZ"
        assert recorder.queue[0].parameters == [0.4]
        assert recorder.queue[0].wires == Wires([0])

    def test_quantum_circuit_error_not_qiskit_circuit_passed(self, recorder):
        """Tests the load method raises a ValueError, if something
        that is not a QuantumCircuit was passed."""

        qc = qml.PauliZ(0)

        quantum_circuit = load(qc)

        with pytest.raises(ValueError):
            with recorder:
                quantum_circuit()

    def test_wires_error_too_few_wires_specified(self, recorder):
        """Tests that an error is raised when too few wires were specified."""

        only_two_wires = [0, 1]
        three_wires_for_the_operation = [0, 1, 2]

        qr1 = QuantumRegister(2)
        qr2 = QuantumRegister(1)

        qc = QuantumCircuit(qr1, qr2)

        qc.cswap(*three_wires_for_the_operation)

        quantum_circuit = load(qc)

        with pytest.raises(
            qml.QuantumFunctionError,
            match=f"The specified number of wires - {len(only_two_wires)} - does not match the"
            " number of wires the loaded quantum circuit acts on.",
        ):
            with recorder:
                quantum_circuit(wires=only_two_wires)

    def test_wires_error_too_many_wires_specified(self, recorder):
        """Tests that an error is raised when too many wires were specified."""

        more_than_three_wires = [4, 13, 123, 321]
        three_wires_for_the_operation = [0, 1, 2]

        qr1 = QuantumRegister(2)
        qr2 = QuantumRegister(1)

        qc = QuantumCircuit(qr1, qr2)

        qc.cswap(*three_wires_for_the_operation)

        quantum_circuit = load(qc)

        with pytest.raises(
            qml.QuantumFunctionError,
            match=f"The specified number of wires - {len(more_than_three_wires)} - does not match the"
            " number of wires the loaded quantum circuit acts on.",
        ):
            with recorder:
                quantum_circuit(wires=more_than_three_wires)

    def test_wires_two_different_quantum_registers(self, recorder):
        """Tests loading a circuit with the three-qubit operations supported by PennyLane."""

        three_wires = [0, 1, 2]

        qr1 = QuantumRegister(2)
        qr2 = QuantumRegister(1)

        qc = QuantumCircuit(qr1, qr2)

        qc.cswap(*three_wires)

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert recorder.queue[0].name == "CSWAP"
        assert recorder.queue[0].parameters == []
        assert recorder.queue[0].wires == Wires(three_wires)

    def test_wires_quantum_circuit_init_with_two_different_quantum_registers(self, recorder):
        """Tests that the wires is correct even if the quantum circuit was initiliazed with two
        separate quantum registers."""

        three_wires = [0, 1, 2]

        qr1 = QuantumRegister(2)
        qr2 = QuantumRegister(1)

        qc = QuantumCircuit(qr1, qr2)

        qc.cswap(*three_wires)

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit(wires=three_wires)

        assert recorder.queue[0].name == "CSWAP"
        assert recorder.queue[0].parameters == []
        assert recorder.queue[0].wires == Wires(three_wires)

    def test_wires_pass_different_wires_than_for_circuit(self, recorder):
        """Tests that custom wires can be passed to the loaded template."""

        three_wires = [4, 7, 1]

        qr1 = QuantumRegister(2)
        qr2 = QuantumRegister(1)

        qc = QuantumCircuit(qr1, qr2)

        qc.cswap(*[0, 1, 2])

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit(wires=three_wires)

        assert recorder.queue[0].name == "CSWAP"
        assert recorder.queue[0].parameters == []
        assert recorder.queue[0].wires == Wires(three_wires)


class TestConverterGatesQiskitToPennyLane:
    """Tests over specific gate related tests"""

    @pytest.mark.parametrize(
        "qiskit_operation, pennylane_name",
        [(QuantumCircuit.crx, "CRX"), (QuantumCircuit.crz, "CRZ"), (QuantumCircuit.cry, "CRY")],
    )
    def test_controlled_rotations(self, qiskit_operation, pennylane_name, recorder):
        """Tests loading a circuit with two qubit controlled rotations (except
        for CRY)."""

        q2 = QuantumRegister(2)
        qc = QuantumCircuit(q2)

        qiskit_operation(qc, 0.5, q2[0], q2[1])

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit()

        assert len(recorder.queue) == 1
        assert recorder.queue[0].name == pennylane_name
        assert recorder.queue[0].parameters == [0.5]
        assert recorder.queue[0].wires == Wires([0, 1])

    @pytest.mark.parametrize(
        "qiskit_operation, pennylane_name",
        [
            (QuantumCircuit.rxx, "IsingXX"),
            (QuantumCircuit.ryy, "IsingYY"),
            (QuantumCircuit.rzz, "IsingZZ"),
        ],
    )
    def test_controlled_rotations_ising(self, qiskit_operation, pennylane_name, recorder):
        """Tests loading a circuit with two qubit Ising operations."""

        q2 = QuantumRegister(2)
        qc = QuantumCircuit(q2)

        qiskit_operation(qc, 0.5, q2[0], q2[1])

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit()

        assert len(recorder.queue) == 1
        assert recorder.queue[0].name == pennylane_name
        assert recorder.queue[0].parameters == [0.5]
        assert recorder.queue[0].wires == Wires([0, 1])

    def test_one_qubit_operations_supported_by_pennylane(self, recorder):
        """Tests loading a circuit with the one-qubit operations supported by PennyLane."""

        single_wire = [0]

        qc = QuantumCircuit(1, 1)
        qc.x(single_wire)
        qc.y(single_wire)
        qc.z(single_wire)
        qc.h(single_wire)
        qc.s(single_wire)
        qc.t(single_wire)
        qc.sx(single_wire)
        qc.id(single_wire)

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert len(recorder.queue) == 8

        assert recorder.queue[0].name == "PauliX"
        assert recorder.queue[0].parameters == []
        assert recorder.queue[0].wires == Wires(single_wire)

        assert recorder.queue[1].name == "PauliY"
        assert recorder.queue[1].parameters == []
        assert recorder.queue[1].wires == Wires(single_wire)

        assert recorder.queue[2].name == "PauliZ"
        assert recorder.queue[2].parameters == []
        assert recorder.queue[2].wires == Wires(single_wire)

        assert recorder.queue[3].name == "Hadamard"
        assert recorder.queue[3].parameters == []
        assert recorder.queue[3].wires == Wires(single_wire)

        assert recorder.queue[4].name == "S"
        assert recorder.queue[4].parameters == []
        assert recorder.queue[4].wires == Wires(single_wire)

        assert recorder.queue[5].name == "T"
        assert recorder.queue[5].parameters == []
        assert recorder.queue[5].wires == Wires(single_wire)

        assert recorder.queue[6].name == "SX"
        assert recorder.queue[6].parameters == []
        assert recorder.queue[6].wires == Wires(single_wire)

        assert recorder.queue[7].name == "Identity"
        assert recorder.queue[7].parameters == []
        assert recorder.queue[7].wires == Wires(single_wire)

    def test_one_qubit_parametrized_operations_supported_by_pennylane(self, recorder):
        """Tests loading a circuit with the one-qubit parametrized operations supported by PennyLane."""

        single_wire = [0]
        angle = 0.3333

        phi = 0.3
        lam = 0.4
        theta = 0.2

        q_reg = QuantumRegister(1)
        qc = QuantumCircuit(q_reg)

        qc.p(angle, single_wire)
        qc.rx(angle, single_wire)
        qc.ry(angle, single_wire)
        qc.rz(angle, single_wire)
        qc.u(phi, lam, theta, [0])

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert recorder.queue[0].name == "PhaseShift"
        assert recorder.queue[0].parameters == [angle]
        assert recorder.queue[0].wires == Wires(single_wire)

        assert recorder.queue[1].name == "RX"
        assert recorder.queue[1].parameters == [angle]
        assert recorder.queue[1].wires == Wires(single_wire)

        assert recorder.queue[2].name == "RY"
        assert recorder.queue[2].parameters == [angle]
        assert recorder.queue[2].wires == Wires(single_wire)

        assert recorder.queue[3].name == "RZ"
        assert recorder.queue[3].parameters == [angle]
        assert recorder.queue[3].wires == Wires(single_wire)

        assert recorder.queue[4].name == "U3"
        assert len(recorder.queue[4].parameters) == 3
        assert recorder.queue[4].parameters == [0.3, 0.4, 0.2]
        assert recorder.queue[4].wires == Wires([0])

    def test_two_qubit_operations_supported_by_pennylane(self, recorder):
        """Tests loading a circuit with the two-qubit operations supported by PennyLane."""

        two_wires = [0, 1]

        qc = QuantumCircuit(2, 1)

        qc.cx(*two_wires)
        qc.cz(*two_wires)
        qc.swap(*two_wires)
        qc.iswap(*two_wires)

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert len(recorder.queue) == 4

        assert recorder.queue[0].name == "CNOT"
        assert recorder.queue[0].parameters == []
        assert recorder.queue[0].wires == Wires(two_wires)

        assert recorder.queue[1].name == "CZ"
        assert recorder.queue[1].parameters == []
        assert recorder.queue[1].wires == Wires(two_wires)

        assert recorder.queue[2].name == "SWAP"
        assert recorder.queue[2].parameters == []
        assert recorder.queue[2].wires == Wires(two_wires)

        assert recorder.queue[3].name == "ISWAP"
        assert recorder.queue[3].parameters == []
        assert recorder.queue[3].wires == Wires(two_wires)

    def test_two_qubit_parametrized_operations_supported_by_pennylane(self, recorder):
        """Tests loading a circuit with the two-qubit parametrized operations supported by PennyLane."""

        two_wires = [0, 1]
        angle = 0.3333

        qc = QuantumCircuit(2, 1)

        qc.crz(angle, *two_wires)
        qc.rzz(angle, *two_wires)
        qc.ryy(angle, *two_wires)
        qc.rxx(angle, *two_wires)

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert len(recorder.queue) == 4

        assert recorder.queue[0].name == "CRZ"
        assert recorder.queue[0].parameters == [angle]
        assert recorder.queue[0].wires == Wires(two_wires)

        assert recorder.queue[1].name == "IsingZZ"
        assert recorder.queue[1].parameters == [angle]
        assert recorder.queue[1].wires == Wires(two_wires)

        assert recorder.queue[2].name == "IsingYY"
        assert recorder.queue[2].parameters == [angle]
        assert recorder.queue[2].wires == Wires(two_wires)

        assert recorder.queue[3].name == "IsingXX"
        assert recorder.queue[3].parameters == [angle]
        assert recorder.queue[3].wires == Wires(two_wires)

    def test_three_qubit_operations_supported_by_pennylane(self, recorder):
        """Tests loading a circuit with the three-qubit operations supported by PennyLane."""

        three_wires = [0, 1, 2]

        qc = QuantumCircuit(3, 1)
        qc.cswap(*three_wires)
        qc.ccx(*three_wires)
        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit()

        assert recorder.queue[0].name == "CSWAP"
        assert recorder.queue[0].parameters == []
        assert recorder.queue[0].wires == Wires(three_wires)

        assert recorder.queue[1].name == "Toffoli"
        assert len(recorder.queue[1].parameters) == 0
        assert recorder.queue[1].wires == Wires(three_wires)

    def test_operations_adjoint_ops(self, recorder):
        """Tests loading a circuit with the operations Sdg, Tdg, and SXdg gates."""

        qc = QuantumCircuit(3, 1)

        qc.sdg([0])
        qc.tdg([0])
        qc.sxdg([0])
        qc.append(lib.GlobalPhaseGate(1.2))

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert recorder.queue[0].name == "Adjoint(S)"
        assert len(recorder.queue[0].parameters) == 0
        assert recorder.queue[0].wires == Wires([0])

        assert recorder.queue[1].name == "Adjoint(T)"
        assert len(recorder.queue[1].parameters) == 0
        assert recorder.queue[1].wires == Wires([0])

        assert recorder.queue[2].name == "Adjoint(SX)"
        assert len(recorder.queue[2].parameters) == 0
        assert recorder.queue[2].wires == Wires([0])

        assert recorder.queue[3].name == "Adjoint(GlobalPhase)"
        assert recorder.queue[3].parameters == [1.2]
        assert recorder.queue[3].wires == Wires([])

    def test_controlled_gates(self, recorder):
        """Tests loading a circuit with controlled gates."""

        qc = QuantumCircuit(3)
        qc.cy(0, 1)
        qc.ch(1, 2)
        qc.cp(1.2, 2, 1)
        qc.ccz(0, 2, 1)
        qc.barrier()
        qc.ecr(1, 0)

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit()

        assert len(recorder.queue) == 6

        assert recorder.queue[0].name == "CY"
        assert len(recorder.queue[0].parameters) == 0
        assert recorder.queue[0].wires == Wires([0, 1])

        assert recorder.queue[1].name == "CH"
        assert len(recorder.queue[1].parameters) == 0
        assert recorder.queue[1].wires == Wires([1, 2])

        assert recorder.queue[2].name == "ControlledPhaseShift"
        assert recorder.queue[2].parameters == [1.2]
        assert recorder.queue[2].wires == Wires([2, 1])

        assert recorder.queue[3].name == "CCZ"
        assert len(recorder.queue[3].parameters) == 0
        assert recorder.queue[3].wires == Wires([0, 2, 1])

        assert recorder.queue[4].name == "Barrier"
        assert len(recorder.queue[4].parameters) == 0
        assert recorder.queue[4].wires == Wires([0, 1, 2])

        assert recorder.queue[5].name == "ECR"
        assert len(recorder.queue[5].parameters) == 0
        assert recorder.queue[5].wires == Wires([1, 0])

    def test_operation_transformed_into_qubit_unitary(self, recorder):
        """Tests loading a circuit with operations that can be converted,
        but not natively supported by PennyLane."""

        qc = QuantumCircuit(3, 1)

        qc.cs([0], [1])

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert recorder.queue[0].name == "QubitUnitary"
        assert len(recorder.queue[0].parameters) == 1
        assert np.array_equal(recorder.queue[0].parameters[0], lib.CSGate().to_matrix())
        assert recorder.queue[0].wires == Wires([0, 1])


class TestCheckParameterBound:
    """Tests for the :func:`_check_parameter_bound()` function."""

    def test_parameter_vector_element_is_unbound(self):
        """Tests that no exception is raised if the vector associated with a parameter vector
        element exists in the dictionary of unbound parameters.
        """
        param_vec = ParameterVector("θ", 2)
        param = cast(ParameterVectorElement, param_vec[1])
        _check_parameter_bound(param=param, unbound_params={param_vec: [0.1, 0.2]})

    def test_parameter_vector_element_is_not_unbound(self):
        """Tests that a ValueError is raised if the vector associated with a parameter vector
        element is missing from the dictionary of unbound parameters.
        """
        param_vec = ParameterVector("θ", 2)
        param = cast(ParameterVectorElement, param_vec[1])

        match = r"The vector of parameter θ\[1\] was not bound correctly\."
        with pytest.raises(ValueError, match=match):
            _check_parameter_bound(param=param, unbound_params={})

    def test_parameter_is_unbound(self):
        """Tests that no exception is raised if the checked parameter exists in the dictionary of
        unbound parameters.
        """
        param = Parameter("θ")
        _check_parameter_bound(param=param, unbound_params={param: 0.1})

    def test_parameter_is_not_unbound(self):
        """Tests that a ValueError is raised if the checked parameter is missing in the dictionary
        of unbound parameters.
        """
        param = Parameter("θ")
        with pytest.raises(ValueError, match=r"The parameter θ was not bound correctly\."):
            _check_parameter_bound(param=param, unbound_params={})


class TestConverterUtils:
    """Tests the utility functions used by the converter function."""

    def test_map_wires(self, recorder):
        """Tests the map_wires function for wires of a quantum circuit."""

        wires = [0]
        qc = QuantumCircuit(1)
        qc_wires = [hash(q) for q in qc.qubits]

        assert map_wires(qc_wires=wires, wires=qc_wires) == {0: hash(qc.qubits[0])}

    def test_map_wires_instantiate_quantum_circuit_with_registers(self, recorder):
        """Tests the map_wires function for wires of a quantum circuit instantiated
        using quantum registers."""

        wires = [0, 1, 2]
        qr1 = QuantumRegister(1)
        qr2 = QuantumRegister(1)
        qr3 = QuantumRegister(1)
        qc = QuantumCircuit(qr1, qr2, qr3)
        qc_wires = [hash(q) for q in qc.qubits]

        mapped_wires = map_wires(qc_wires=wires, wires=qc_wires)

        assert len(mapped_wires) == len(wires)
        assert list(mapped_wires.keys()) == wires
        for q in qc.qubits:
            assert hash(q) in mapped_wires.values()

    def test_map_wires_provided_non_standard_order(self, recorder):
        """Tests the map_wires function for wires of non-standard order."""

        wires = [1, 2, 0]
        qc = QuantumCircuit(3)
        qc_wires = [hash(q) for q in qc.qubits]

        mapped_wires = map_wires(qc_wires=wires, wires=qc_wires)

        for q in qc.qubits:
            assert hash(q) in mapped_wires.values()

        assert len(mapped_wires) == len(wires)
        assert set(mapped_wires.keys()) == set(wires)
        assert mapped_wires[0] == qc_wires[2]
        assert mapped_wires[1] == qc_wires[0]
        assert mapped_wires[2] == qc_wires[1]

    def test_map_wires_exception_mismatch_in_number_of_wires(self, recorder):
        """Tests that the map_wires function raises an exception if there is a mismatch between
        wires."""

        wires = [0, 1, 2]
        qc = QuantumCircuit(1)
        qc_wires = [hash(q) for q in qc.qubits]

        with pytest.raises(
            qml.QuantumFunctionError,
            match=f"The specified number of wires - {len(wires)} - does not match ",
        ):
            map_wires(qc_wires, wires)

    def test_format_params_dict_old_interface(self):
        """Test the old interface for setting the value of Qiskit Parameters -
        passing a dictionary of the form ``{Parameter("name"): value, ...}`` as
        either an arg or with the params kwarg"""

        theta = Parameter("θ")
        phi = Parameter("φ")
        rotation_angle1 = 0.5
        rotation_angle2 = 0.3

        qc = QuantumCircuit(2)
        qc.rz(theta, [0])
        qc.rx(phi, [0])

        # works if params was passed as a kwarg, args is ()
        params = {theta: rotation_angle1, phi: rotation_angle2}
        output_params = _format_params_dict(qc, params)
        assert output_params == params

        # works if params was passed as an arg and params=None
        output_params2 = _format_params_dict(qc, None, params)
        assert output_params2 == params

    @pytest.mark.parametrize(
        "args, kwargs",
        [
            ((0.5, 0.3, 0.4), {}),
            ((), {"a": 0.5, "b": 0.3, "c": 0.4}),
            ((0.5, 0.3), {"c": 0.4}),
            ((0.5, 0.4), {"b": 0.3}),
            ((0.3,), {"a": 0.5, "c": 0.4}),
        ],
    )
    def test_format_params_dict_new_interface(self, args, kwargs):
        """Test the new interface for setting the value of Qiskit Parameters -
        passing either ordered args, or keyword arguments where the argument
        matches the Parameter name, or some combination of the two.

        The kwargs are passed as a dictionary to this function, and the args
        as a tuple. This tests the new case where `params=None`"""

        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")

        qc = QuantumCircuit(2)
        qc.rz(a, [0])
        qc.rx(b, [0])
        qc.ry(c, [0])

        params = _format_params_dict(qc, None, *args, **kwargs)
        assert params == {a: 0.5, b: 0.3, c: 0.4}


class TestConverterWarningsAndErrors:
    """Tests that the converter.load function emits warnings and errors."""

    def test_template_not_supported(self, recorder):
        """Tests that a warning is raised if an unsupported instruction was reached."""
        qc = DraperQFTAdder(3)

        quantum_circuit = load(qc)
        params = []

        with pytest.warns(UserWarning) as record:
            with recorder:
                quantum_circuit(*params)

        # check that the message matches
        assert (
            record[-1].message.args[0]
            == "pennylane_qiskit.converter: The Gate instruction is not supported by"
            " PennyLane, and has not been added to the template."
        )

    @pytest.mark.parametrize("invalid_param", ["wires", "params"])
    def test_params_and_wires_not_valid_param_names(self, invalid_param):
        """Test that ambiguous parameter names 'wires' and 'params' in the Qiskit
        QuantumCircuit raise an error"""

        parameter = Parameter(invalid_param)

        qc = QuantumCircuit(2, 2)
        qc.rx(parameter, 0)
        qc.rx(0.3, 1)
        qc.measure_all()

        with pytest.raises(RuntimeError, match="this argument is reserved"):
            load(qc)(0.3)

    def test_kwarg_does_not_match_params(self):
        """Test that if a parameter kwarg doesn't match any of the Parameter
        names in a QuantumCircuit, an error is raised"""

        parameter = Parameter("name1")
        parameter2 = Parameter("a")

        qc = QuantumCircuit(2, 2)
        qc.rx(parameter, 0)
        qc.rx(parameter2, 1)
        qc.measure_all()

        # works with correct name
        load(qc)(0.2, name1=0.3)

        # raises error with incorrect name
        with pytest.raises(
            TypeError,
            match="Got unexpected parameter keyword argument 'name2'. Circuit contains parameters: a, name1",
        ):
            load(qc)(0.2, name2=0.3)

    def test_kwarg_does_not_match_params_parametervector(self):
        """Test that if a parameter kwarg when the Parameter is a kwarg compares to the ParameterVector
        name, and raises an error if it does not match"""

        parameter = ParameterVector("vectorname", 2)

        qc = QuantumCircuit(2, 2)
        qc.rx(parameter[0], 0)
        qc.rx(parameter[1], 1)
        qc.measure_all()

        # works with correct name
        load(qc)(vectorname=[0.3, 0.4])

        # raises error with incorrect name
        with pytest.raises(
            TypeError,
            match="Got unexpected parameter keyword argument 'wrong_vectorname'. Circuit contains parameters: vectorname",
        ):
            load(qc)(wrong_vectorname=[0.3, 0.4])

    def test_too_many_args(self):
        """Test that if too many positional arguments are passed to define ``Parameter`` values,
        a clear error is raised"""

        a = Parameter("a")
        b = Parameter("b")
        c = ParameterVector("c", 2)

        qc = QuantumCircuit(2, 2)
        qc.rx(a * c[0], 0)
        qc.rx(b * c[1], 1)
        qc.measure_all()

        with pytest.raises(TypeError, match="Expected 3 positional arguments but 4 were given"):
            load(qc)(0.2, 0.4, [0.1, 0.3], 0.5)

        with pytest.raises(TypeError, match="Expected 1 positional argument but 2 were given"):
            load(qc)(0.4, 0.5, a=0.2, c=[0.1, 0.3])

        with pytest.raises(TypeError, match="Expected 1 positional argument but 2 were given"):
            load(qc)([0.1, 0.3], 0.5, a=0.2, b=0.4)

    def test_missing_argument(self):
        """Test that if calling with missing arguments, a clear error is raised"""

        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")

        qc = QuantumCircuit(2, 2)
        qc.rx(a, 0)
        qc.rx(b * c, 1)
        qc.measure_all()

        with pytest.raises(
            TypeError, match="Missing 1 required argument to define Parameter value for: c"
        ):
            load(qc)(0.2, 0.3)

        with pytest.raises(
            TypeError, match="Missing 1 required argument to define Parameter value for: b"
        ):
            load(qc)(0.2, c=0.4)

        with pytest.raises(
            TypeError, match="Missing 2 required arguments to define Parameter values for: a, c"
        ):
            load(qc)(b=0.3)

        with pytest.raises(
            TypeError, match="Missing 1 required argument to define Parameter value for: c"
        ):
            load(qc)({a: 0.2, b: 0.3})

    def test_missing_argument_with_parametervector(self):
        """Test that if calling with missing arguments, a clear error is raised correctly
        when ParameterVectors are included in the circuit"""

        a = Parameter("a")
        b = ParameterVector("v", 2)
        c = Parameter("c")

        qc = QuantumCircuit(2, 2)
        qc.rx(a, 0)
        qc.rx(b[0] * b[1], 1)
        qc.ry(c, 0)
        qc.measure_all()

        # loads with correct arguments
        load(qc)(0.2, 0.4, [0.1, 0.3])

        with pytest.raises(
            TypeError, match="Missing 1 required argument to define Parameter value for: v"
        ):
            load(qc)(0.2, c=0.4)

        with pytest.raises(
            TypeError, match="Missing 2 required arguments to define Parameter values for: c, v"
        ):
            load(qc)(a=0.2)

        with pytest.raises(
            TypeError, match="Missing 2 required arguments to define Parameter values for: a, v"
        ):
            load(qc)({c: 0.3})

    def test_no_parameters_raises_error(self, recorder):
        """Tests the load method for a QuantumCircuit raises a TypeError if no
        parameters are passed to a loaded function that requires parameters"""

        theta = Parameter("θ")

        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])

        quantum_circuit = load(qc)

        with pytest.raises(TypeError, match="Missing required argument to define Parameter value"):
            quantum_circuit()


class TestConverterQasm:
    """Tests that the converter.load function allows conversion from qasm."""

    qft_qasm = (
        "OPENQASM 2.0;"
        'include "qelib1.inc";'
        "qreg q[4];"
        "creg c[4];"
        "x q[0]; "
        "x q[2];"
        "barrier q;"
        "h q[0];"
        "h q[1];"
        "h q[2];"
        "h q[3];"
        "measure q -> c;"
    )

    @pytest.mark.skipif(sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6")
    def test_qasm_from_file(self, tmpdir, recorder):
        """Tests that a QuantumCircuit object is deserialized from a qasm file."""
        qft_qasm = tmpdir.join("qft.qasm")

        with open(qft_qasm, "w", encoding="utf") as f:
            f.write(TestConverterQasm.qft_qasm)

        quantum_circuit = load_qasm_from_file(qft_qasm)

        with recorder:
            quantum_circuit()

        # only 2 x X, a Barrier, and 4 x H queued (not 4 x qml.measure)
        assert len(recorder.queue) == 7

        assert recorder.queue[0].name == "PauliX"
        assert recorder.queue[0].parameters == []
        assert recorder.queue[0].wires == Wires([0])

        assert recorder.queue[1].name == "PauliX"
        assert recorder.queue[1].parameters == []
        assert recorder.queue[1].wires == Wires([2])

        assert recorder.queue[2].name == "Barrier"
        assert recorder.queue[2].parameters == []
        assert recorder.queue[2].wires == Wires([0, 1, 2, 3])

        assert recorder.queue[3].name == "Hadamard"
        assert recorder.queue[3].parameters == []
        assert recorder.queue[3].wires == Wires([0])

        assert recorder.queue[4].name == "Hadamard"
        assert recorder.queue[4].parameters == []
        assert recorder.queue[4].wires == Wires([1])

        assert recorder.queue[5].name == "Hadamard"
        assert recorder.queue[5].parameters == []
        assert recorder.queue[5].wires == Wires([2])

        assert recorder.queue[6].name == "Hadamard"
        assert recorder.queue[6].parameters == []
        assert recorder.queue[6].wires == Wires([3])

    def test_qasm_file_not_found_error(self):
        """Tests that an error is propagated, when a non-existing file is specified for parsing."""
        qft_qasm = "some_qasm_file.qasm"

        with pytest.raises(FileNotFoundError):
            load_qasm_from_file(qft_qasm)

    def test_qasm_(self, recorder):
        """Tests that a QuantumCircuit object is deserialized from a qasm string."""
        qasm_string = (
            'include "qelib1.inc";'
            "qreg q[4];"
            "creg c[4];"
            "x q[0];"
            "cx q[2],q[0];"
            "measure q -> c;"
        )

        quantum_circuit = load_qasm(qasm_string)

        with recorder:
            quantum_circuit(params={})

        # X and CNOT queued with 4 x qml.measure
        assert len(recorder.queue) == 6

        assert recorder.queue[0].name == "PauliX"
        assert recorder.queue[0].parameters == []
        assert recorder.queue[0].wires == Wires([0])

        assert recorder.queue[1].name == "CNOT"
        assert recorder.queue[1].parameters == []
        assert recorder.queue[1].wires == Wires([2, 0])

        for i in range(2, 6):
            assert recorder.queue[i].name == "MidMeasureMP"
            assert recorder.queue[i].wires == Wires([i - 2])

    def test_qasm_measure(self):
        """Tests that measurements specified as an argument are added to the converted circuit."""
        qasm_string = (
            'include "qelib1.inc";' + "qreg q[2];" + "creg c[2];" + "h q[0];" + "cx q[0], q[1];"
        )
        dev = qml.device("default.qubit")
        measurements = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
        loaded_circuit = load_qasm(qasm_string, measurements=measurements)

        # loaded circuit with measurements
        @qml.qnode(dev)
        def quantum_circuit1():
            return loaded_circuit()

        # native pennylane measurements
        @qml.qnode(dev)
        def quantum_circuit2():
            load_qasm(qasm_string)()
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

        assert quantum_circuit1() == quantum_circuit2()

    def test_qasm_mid_circuit_measure(self):
        """Tests that the QASM primitive measure is correctly converted."""
        qasm_string = (
            'include "qelib1.inc";'
            "qreg q[2];"
            "creg c[2];"
            "h q[0];"
            "measure q[0] -> c[0];"
            "rz(0.24) q[0];"
            "cx q[0], q[1];"
            "measure q -> c;"
        )
        dev = qml.device("default.qubit")
        loaded_circuit = load_qasm(qasm_string)

        # loaded circuit
        @qml.qnode(dev)
        def quantum_circuit1():
            mid_measure, _, m1 = loaded_circuit()
            qml.cond(mid_measure == 0, qml.RX)(np.pi / 2, 0)
            return qml.expval(mid_measure), qml.expval(m1)

        # native pennylane circuit
        @qml.qnode(dev)
        def quantum_circuit2():
            qml.Hadamard(0)
            mid_measure = qml.measure(0)
            qml.RZ(0.24, 0)
            qml.CNOT([0, 1])
            qml.measure([0])
            m1 = qml.measure([1])
            qml.cond(mid_measure == 0, qml.RX)(np.pi / 2, 0)
            return qml.expval(mid_measure), qml.expval(m1)

        assert quantum_circuit1() == quantum_circuit2()


class TestConverterIntegration:
    def test_use_loaded_circuit_in_qnode(self, qubit_device_2_wires):
        """Tests loading a converted template in a QNode."""

        angle = 0.5

        qc = QuantumCircuit(2)
        qc.rz(angle, [0])

        quantum_circuit = load(qc)

        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit():
            quantum_circuit()
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qubit_device_2_wires)
        def circuit_native_pennylane():
            qml.RZ(angle, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit() == circuit_native_pennylane()

    def test_load_circuit_inside_of_qnode(self, qubit_device_2_wires):
        """Tests loading a QuantumCircuit inside of the QNode circuit
        definition."""

        theta = Parameter("θ")
        angle = 0.5

        qc = QuantumCircuit(2)
        qc.rz(theta, [0])

        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit():
            load(qc)({theta: angle})
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qubit_device_2_wires)
        def circuit_native_pennylane():
            qml.RZ(angle, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit() == circuit_native_pennylane()

    def test_passing_parameter_into_qnode_old_interface(self, qubit_device_2_wires):
        """Tests passing a circuit parameter into the QNode."""

        theta = Parameter("θ")
        rotation_angle = 0.5

        qc = QuantumCircuit(2)
        qc.rz(theta, [0])

        # with params dict as arg
        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit(angle):
            load(qc)({theta: angle})
            return qml.expval(qml.PauliZ(0))

        # with params dict as kwarg
        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit2(angle):
            load(qc)(params={theta: angle})
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qubit_device_2_wires)
        def circuit_native_pennylane(angle):
            qml.RZ(angle, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit(rotation_angle) == circuit_native_pennylane(
            rotation_angle
        )
        assert circuit_loaded_qiskit_circuit2(rotation_angle) == circuit_native_pennylane(
            rotation_angle
        )

    def test_one_parameter_in_qc_one_passed_into_qnode(self, qubit_device_2_wires):
        """Tests passing a parameter by pre-defining it and then
        passing another to the QNode."""

        theta = Parameter("theta")
        phi = Parameter("phi")
        rotation_angle1 = 0.5
        rotation_angle2 = 0.3

        qc = QuantumCircuit(2)
        qc.rz(theta, [0])
        qc.rx(phi, [0])

        # as args
        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit(angle):
            load(qc)(rotation_angle2, angle)  # order is phi, theta (alphabetical)
            return qml.expval(qml.PauliZ(0))

        # as kwargs
        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit2(angle):
            load(qc)(theta=angle, phi=rotation_angle2)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qubit_device_2_wires)
        def circuit_native_pennylane(angle):
            qml.RZ(angle, wires=0)
            qml.RX(rotation_angle2, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit(rotation_angle1) == circuit_native_pennylane(
            rotation_angle1
        )
        assert circuit_loaded_qiskit_circuit2(rotation_angle1) == circuit_native_pennylane(
            rotation_angle1
        )

    def test_initialize_with_qubit_state_vector(self, qubit_device_single_wire):
        """Tests the QuantumCircuit.initialize method in a QNode."""

        prob_amplitudes = [1 / np.sqrt(2), 1 / np.sqrt(2)]

        qreg = QuantumRegister(2)
        qc = QuantumCircuit(qreg)
        qc.initialize(prob_amplitudes, [qreg[0]])

        @qml.qnode(qubit_device_single_wire)
        def circuit_loaded_qiskit_circuit():
            load(qc)()
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qubit_device_single_wire)
        def circuit_native_pennylane():
            qml.StatePrep(np.array(prob_amplitudes), wires=[0])
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit() == circuit_native_pennylane()

    @pytest.mark.parametrize("shots", [None])
    @pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
    def test_gradient(self, theta, phi, varphi, shots, tol):
        """Tests that the gradient of a circuit is calculated correctly."""
        qc = QuantumCircuit(3)
        qiskit_params = [Parameter(f"param_{i}") for i in range(3)]

        qc.rx(qiskit_params[0], 0)
        qc.rx(qiskit_params[1], 1)
        qc.rx(qiskit_params[2], 2)
        qc.cx(0, 1)
        qc.cx(1, 2)

        # convert to a PennyLane circuit
        qc_pl = qml.from_qiskit(qc)

        dev = qml.device("default.qubit", wires=3, shots=shots)

        @qml.qnode(dev)
        def circuit(params):
            qiskit_param_mapping = dict(map(list, zip(qiskit_params, params)))
            qc_pl(qiskit_param_mapping)
            return qml.expval(qml.PauliX(0) @ qml.PauliY(2))

        dcircuit = qml.grad(circuit, 0)
        res = dcircuit([theta, phi, varphi])
        expected = [
            np.cos(theta) * np.sin(phi) * np.sin(varphi),
            np.sin(theta) * np.cos(phi) * np.sin(varphi),
            np.sin(theta) * np.sin(phi) * np.cos(varphi),
        ]

        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("shots", [None])
    def test_gradient_with_parameter_vector(self, shots, tol):
        """Tests that the gradient of a circuit with a parameter vector is calculated correctly."""
        qiskit_circuit = QuantumCircuit(1)

        theta_param = ParameterVector("θ", 2)
        theta_val = np.array([np.pi / 4, np.pi / 16])

        qiskit_circuit.rx(theta_param[0], 0)
        qiskit_circuit.rx(theta_param[1] * 4, 0)

        pl_circuit_loader = qml.from_qiskit(qiskit_circuit)

        dev = qml.device("default.qubit", wires=1, shots=shots)

        @qml.qnode(dev)
        def circuit(theta):
            pl_circuit_loader(params={theta_param: theta})
            return qml.expval(qml.PauliZ(0))

        have_gradient = qml.grad(circuit)(theta_val)
        want_gradient = [-1, -4]
        assert np.allclose(have_gradient, want_gradient, **tol)

    @pytest.mark.parametrize("shots", [None])
    def test_gradient_with_parameter_expressions(self, shots, tol):
        """Tests that the gradient of a circuit with parameter expressions is calculated correctly."""
        qiskit_circuit = QuantumCircuit(1)

        theta_param = ParameterVector("θ", 3)
        theta_val = np.array([3 * np.pi / 16, np.pi / 64, np.pi / 96])

        phi_param = Parameter("φ")
        phi_val = np.array(np.pi / 8)

        # Apply an instruction with a regular parameter.
        qiskit_circuit.rx(phi_param, 0)
        # Apply an instruction with a parameter vector element.
        qiskit_circuit.rx(theta_param[0], 0)
        # Apply an instruction with a parameter expression involving one parameter.
        qiskit_circuit.rx(theta_param[1] + theta_param[1], 0)
        # Apply an instruction with a parameter expression involving two parameters.
        qiskit_circuit.rx(3 * theta_param[2] + phi_param, 0)

        pl_circuit_loader = qml.from_qiskit(qiskit_circuit)

        dev = qml.device("default.qubit", wires=1, shots=shots)

        @qml.qnode(dev)
        def circuit(phi, theta):
            pl_circuit_loader(params={phi_param: phi, theta_param: theta})
            return qml.expval(qml.PauliZ(0))

        have_phi_gradient, have_theta_gradient = qml.grad(circuit)(phi_val, theta_val)
        want_phi_gradient, want_theta_gradient = [-2], [-1, -2, -3]
        assert np.allclose(have_phi_gradient, want_phi_gradient, **tol)
        assert np.allclose(have_theta_gradient, want_theta_gradient, **tol)

    @pytest.mark.parametrize("shots", [None])
    def test_differentiable_param_is_array(self, shots, tol):
        """Test that extracting the differentiable parameters works correctly
        for arrays"""
        qc = QuantumCircuit(3)
        qiskit_params = [Parameter(f"param_{i}".format(i)) for i in range(3)]

        theta = 0.53
        phi = -1.23
        varphi = 0.8654
        params = [qml.numpy.tensor(theta), qml.numpy.tensor(phi), qml.numpy.tensor(varphi)]

        qc.rx(qiskit_params[0], 0)
        qc.rx(qiskit_params[1], 1)
        qc.rx(qiskit_params[2], 2)
        qc.cx(0, 1)
        qc.cx(1, 2)

        # convert to a PennyLane circuit
        qc_pl = qml.from_qiskit(qc)

        dev = qml.device("default.qubit", wires=3, shots=shots)

        @qml.qnode(dev)
        def circuit(params):
            qiskit_param_mapping = dict(map(list, zip(qiskit_params, params)))
            qc_pl(qiskit_param_mapping)
            return qml.expval(qml.PauliX(0) @ qml.PauliY(2))

        dcircuit = qml.grad(circuit, 0)
        res = dcircuit(params)
        expected = [
            np.cos(theta) * np.sin(phi) * np.sin(varphi),
            np.sin(theta) * np.cos(phi) * np.sin(varphi),
            np.sin(theta) * np.sin(phi) * np.cos(varphi),
        ]

        assert np.allclose(res, expected, **tol)

    def test_parameter_expression(self):
        """Tests the output and the gradient of a QNode that contains loaded Qiskit gates taking functions of parameters as argument"""

        a = Parameter("a")
        b = Parameter("b")

        qc = QuantumCircuit(2)
        qc.rx(a + np.cos(b), 0)
        qc.ry(a * b, 1)
        qc.cx(0, 1)

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(a_val, b_val):
            load(qc)({a: a_val, b: b_val}, wires=(0, 1))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        x = np.array(0.1, requires_grad=True)
        y = np.array(0.2, requires_grad=True)

        res = circuit(x, y)
        res_expected = [np.cos(x + np.cos(y)), np.sin(x * y)]

        assert np.allclose(res, res_expected)

        def cost(x, y):
            return qml.math.stack(circuit(x, y))

        jac = qml.jacobian(cost)(x, y)

        jac_expected = [
            [-np.sin(x + np.cos(y)), np.cos(x * y) * y],
            [np.sin(x + np.cos(y)) * np.sin(y), np.cos(x * y) * x],
        ]

        assert np.allclose(jac, jac_expected)

    def test_quantum_circuit_with_single_measurement(self, qubit_device_single_wire):
        """Tests loading a converted template in a QNode with a single measurement."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()

        measurement = qml.expval(qml.PauliZ(0))
        quantum_circuit = load(qc, measurements=measurement)

        @qml.qnode(qubit_device_single_wire)
        def circuit_loaded_qiskit_circuit():
            return quantum_circuit()

        @qml.qnode(qubit_device_single_wire)
        def circuit_native_pennylane():
            qml.Hadamard(0)
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit() == circuit_native_pennylane()

    def test_quantum_circuit_with_multiple_measurements(self, qubit_device_2_wires):
        """Tests loading a converted template in a QNode with multiple measurements."""

        angle = 0.543

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)
        qc.z(0).c_if(0, 1)
        qc.rz(angle, [0])
        qc.cx(0, 1)
        qc.measure_all()

        measurements = [qml.expval(qml.PauliZ(0)), qml.vn_entropy([1])]
        quantum_circuit = load(qc, measurements=measurements)

        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit():
            return quantum_circuit()

        @qml.qnode(qubit_device_2_wires)
        def circuit_native_pennylane():
            qml.Hadamard(0)
            m0 = qml.measure(0)
            qml.cond(m0, qml.PauliZ)(0)
            qml.RZ(angle, wires=0)
            qml.CNOT([0, 1])
            return [qml.expval(qml.PauliZ(0)), qml.vn_entropy([1])]

        assert circuit_loaded_qiskit_circuit() == circuit_native_pennylane()

        quantum_circuit = load(qc, measurements=None)

        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit2():
            meas = quantum_circuit()
            return [qml.expval(m) for m in meas]

        @qml.qnode(qubit_device_2_wires)
        def circuit_native_pennylane2():
            qml.Hadamard(0)
            m0 = qml.measure(0)
            qml.RZ(angle, wires=0)
            qml.CNOT([0, 1])
            return [qml.expval(m) for m in [m0, qml.measure(0), qml.measure(1)]]

        assert circuit_loaded_qiskit_circuit2() == circuit_native_pennylane2()

    def test_diff_meas_circuit(self):
        """Tests mid-measurements are recognized and returned correctly."""

        angle = 0.543

        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.measure(0, 0)
        qc.rx(angle, [0])
        qc.cx(0, 1)
        qc.measure(1, 1)

        qc1 = QuantumCircuit(3, 3)
        qc1.h(0)
        qc1.measure(2, 2)
        qc1.rx(angle, [0])
        qc1.cx(0, 1)
        qc1.measure(1, 1)

        qtemp, qtemp1 = load(qc), load(qc1)
        assert qtemp()[0] == qml.measure(0) and qtemp1()[0] == qml.measure(2)

        qtemp2 = load(qc, measurements=[qml.expval(qml.PauliZ(0))])
        assert qtemp()[0] != qtemp2()[0] and qtemp2()[0] == qml.expval(qml.PauliZ(0))


class TestConverterPennyLaneCircuitToQiskit:
    def test_circuit_to_qiskit(self):
        """Test that a simple PennyLane circuit is converted to the expected Qiskit circuit"""

        qscript = QuantumScript([qml.Hadamard(1), qml.CNOT([1, 0])])
        qc = circuit_to_qiskit(qscript, len(qscript.wires), diagonalize=False, measure=False)

        operation_names = [instruction.operation.name for instruction in qc.data]

        assert operation_names == ["h", "cx"]

    def test_circuit_to_qiskit_with_parameterized_gate(self):
        """Test that a simple PennyLane circuit is converted to the expected Qiskit circuit"""
        angle = 1.2

        qscript = QuantumScript([qml.Hadamard(1), qml.CNOT([1, 0]), qml.RX(angle, 2)])
        qc = circuit_to_qiskit(qscript, len(qscript.wires), diagonalize=False, measure=False)

        operation_names = [instruction.operation.name for instruction in qc.data]
        operation_params = [instruction.operation.params for instruction in qc.data]

        assert operation_names == ["h", "cx", "rx"]
        assert operation_params == [[], [], [angle]]

    @pytest.mark.parametrize("operations", [[], [qml.PauliX(0), qml.PauliY(1)], [qml.Hadamard(0)]])
    @pytest.mark.parametrize("register_size", [2, 5])
    def test_circuit_to_qiskit_register_size(self, operations, register_size):
        """Test that the regsiter_size determines the shape of the Qiskit
        QuantumCircuit register"""

        qc = circuit_to_qiskit(QuantumScript(operations), register_size)

        # there is a single classical and a single quantum register
        assert len(qc.cregs) == len(qc.qregs) == 1

        # the register contains qubits equal to the register size
        assert len(qc.qubits) == register_size

    @pytest.mark.parametrize(
        "operations, final_op_name",
        [([qml.PauliX(0), qml.PauliY(1)], "y"), ([[qml.CNOT([0, 1]), qml.Hadamard(1)], "h"])],
    )
    @pytest.mark.parametrize("measure", [True, False])
    def test_circuit_to_qiskit_measure_kwarg(self, operations, final_op_name, measure):
        """Test that measurements are added to the circuit if and only if measure=True"""

        qc = circuit_to_qiskit(QuantumScript(operations), 2, measure=measure)
        final_instruction = qc.data[-1]

        if measure:
            assert final_instruction.operation.name == "measure"

    @pytest.mark.parametrize("diagonalize", [True, False])
    def test_circuit_to_qiskit_diagonalize_kwarg(self, diagonalize):
        """Test that diagonalizing gates are included in the circuit if diagonalize=True"""

        qscript = QuantumScript(
            [qml.Hadamard(1), qml.CNOT([1, 0])], measurements=[qml.expval(qml.PauliY(1))]
        )
        assert qscript.diagonalizing_gates == [qml.PauliZ(1), qml.S(1), qml.Hadamard(1)]

        qc = circuit_to_qiskit(qscript, 2, diagonalize=diagonalize, measure=True)

        # get list of instruction names up to the barrier (played right before measurements)
        instructions = []
        for instruction in qc.data:
            if instruction.operation.name == "barrier":
                break
            instructions.append(instruction.operation.name)

        # check length of instructions matches length of expected gates
        expected_gates = qscript.operations
        if diagonalize:
            expected_gates += qscript.diagonalizing_gates

        assert len(instructions) == len(expected_gates)

    def test_circuit_to_qiskit_measurements_with_overlapping_wires(self):
        """Test that diagonalizing gates work for circuits with
        measurements on overlapping wires"""

        measurements = [qml.sample(qml.X(0) @ qml.Y(1)), qml.sample(qml.X(0))]
        tape = qml.tape.QuantumScript(measurements=measurements)

        qc = circuit_to_qiskit(tape, 2, diagonalize=True, measure=True)

        # get list of instruction names up to the barrier (played right before measurements)
        instructions = []
        for instruction in qc.data:
            if instruction.operation.name == "barrier":
                break
            instructions.append(instruction.operation.name)

        # manually diagonalized test case since Qiskit transpiles whatever we had before
        # and that results is different from PL's diagonalization
        expected_gates = ["ry", "rx"]

        assert len(instructions) == len(expected_gates)
        assert instructions == expected_gates


class TestConverterGatePennyLaneToQiskit:
    def test_non_parameteric_operation_to_qiskit(self):
        """Test that a non-parameteric operation is correctly converted to a
        Qiskit circuit with a single operation"""

        op = qml.PauliX(0)

        qc = operation_to_qiskit(op, QuantumRegister(1))
        ops = [instruction.operation.name for instruction in qc.data]
        qubits = [instruction.qubits for instruction in qc.data][0]
        wires = [qc.find_bit(q).index for q in qubits]

        assert ops == ["x"]
        assert wires == [0]

    def test_parameteric_operation_to_qiskit(self):
        """Test that a parameteric operation is correctly converted to a
        Qiskit circuit with a single operation"""

        op = qml.RX(1.23, 2)

        qc = operation_to_qiskit(op, QuantumRegister(3))
        ops = [instruction.operation.name for instruction in qc.data]
        qubits = [instruction.qubits for instruction in qc.data][0]
        wires = [qc.find_bit(q).index for q in qubits]
        params = [instruction.operation.params for instruction in qc.data]

        assert ops == ["rx"]
        assert wires == [2]
        assert params == [[1.23]]

    # ToDo: add custom wire label support? Or have we already mapped to integers here? Story #55168
    @pytest.mark.parametrize("op_wires", ([0, 1], [2, 4]))
    def test_multi_wire_operation_to_qiskit(self, op_wires):
        """Test that an operation with multiple wires is correctly converted to a
        Qiskit circuit with a single operation"""

        op = qml.CNOT(op_wires)

        qc = operation_to_qiskit(op, QuantumRegister(5))
        ops = [instruction.operation.name for instruction in qc.data]
        qubits = [instruction.qubits for instruction in qc.data][0]
        qc_wires = [qc.find_bit(q).index for q in qubits]

        assert ops == ["cx"]
        assert qc_wires == op_wires

    @pytest.mark.parametrize(
        "op",
        [
            qml.QubitUnitary(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], wires=[0, 1]
            ),
            qml.StatePrep(np.array([1, 0, 0, 0]), wires=[0, 1]),
        ],
    )
    def test_state_prep_ops_have_reversed_register(self, op):
        """Tests that the wire order is reversed when applying matrix-based operators from PennyLane,
        because the Qiskit convention for inferring wire order for matrices is the reverse of the
        PennyLane convention"""

        qc = operation_to_qiskit(op, reg=QuantumRegister(3))
        qubits = qc[0].qubits
        wires = [qc.find_bit(q).index for q in qubits]

        # wires on the qiskit circuit are the PL wires reversed
        assert Wires(wires) == op.wires[::-1]

    def test_with_predefined_creg(self):
        """Test that it also works if passing in an already existing classical register"""

        creg = ClassicalRegister(3)

        op = qml.RX(1.23, 2)

        qc1 = operation_to_qiskit(op, QuantumRegister(3), creg=creg)
        qc2 = operation_to_qiskit(op, QuantumRegister(3), creg=None)

        ops1 = [instruction.operation.name for instruction in qc1.data]
        params1 = [instruction.operation.params for instruction in qc1.data]
        ops2 = [instruction.operation.name for instruction in qc2.data]
        params2 = [instruction.operation.params for instruction in qc2.data]

        qubits1 = [instruction.qubits for instruction in qc1.data][0]
        wires1 = [qc1.find_bit(q).index for q in qubits1]
        qubits2 = [instruction.qubits for instruction in qc2.data][0]
        wires2 = [qc2.find_bit(q).index for q in qubits2]

        assert ops1 == ops2 == ["rx"]
        assert wires1 == wires2 == [2]
        assert params1 == params2 == [[1.23]]


# pylint:disable=too-few-public-methods
class TestConverterUtilsPennyLaneToQiskit:
    @pytest.mark.parametrize("measurement_type", [qml.expval, qml.var])
    @pytest.mark.parametrize(
        "operator, expected",
        [
            (qml.X(0), SparsePauliOp("IIIIX")),
            (qml.I(1), SparsePauliOp("IIIII")),
            (Y(0), SparsePauliOp("IIIIY")),
            (qml.PauliZ(0), SparsePauliOp("IIIIZ")),
            (
                X(0) + I(0) + 2 * Y(1) + I(1),
                SparsePauliOp("IIIIX")
                + SparsePauliOp("IIIII")
                + 2 * SparsePauliOp("IIIYI")
                + SparsePauliOp("IIIII"),
            ),
            (
                qml.X(0) + qml.X(0) + qml.Y(1) + qml.Z(2),
                SparsePauliOp("IIIIX")
                + SparsePauliOp("IIIIX")
                + SparsePauliOp("IIIYI")
                + SparsePauliOp("IIZII"),
            ),
            (
                qml.sum(X(0) + X(0) + Y(1) + Z(2)),
                SparsePauliOp("IIIIX")
                + SparsePauliOp("IIIIX")
                + SparsePauliOp("IIIYI")
                + SparsePauliOp("IIZII"),
            ),
            (
                (qml.X(0) + 2 * qml.Y(1)),
                SparsePauliOp("IIIIX") + 2 * SparsePauliOp("IIIYI"),
            ),
            (
                qml.sum(X(0) + qml.s_prod(2, Y(1))),
                SparsePauliOp("IIIIX") + 2 * SparsePauliOp("IIIYI"),
            ),
            (qml.X(0) + qml.Y(0), SparsePauliOp("IIIIX") + SparsePauliOp("IIIIY")),
            (
                0.5 * X(0) + 3 * (X(2) + qml.PauliY(1)),
                0.5 * SparsePauliOp("IIIIX")
                + 3 * (SparsePauliOp("IIXII") + SparsePauliOp("IIIYI")),
            ),
            (
                0.5 * X(0) + 0.5 * qml.Y(0) - 1.5 * qml.X(0) - 0.5 * qml.Y(0),
                0.5 * SparsePauliOp("IIIIX")
                + 0.5 * SparsePauliOp("IIIIY")
                - 1.5 * SparsePauliOp("IIIIX")
                - 0.5 * SparsePauliOp("IIIIY"),
            ),
            (
                qml.ops.LinearCombination(
                    [1, 3, 4],
                    [X(3) @ Y(2), Y(4) - X(2), Z(2) * 3],
                )
                + qml.X(4),
                1 * SparsePauliOp("IXIII") @ SparsePauliOp("IIYII")
                + 3 * (SparsePauliOp("YIIII") - SparsePauliOp("IIXII"))
                + 3 * 4 * SparsePauliOp("IIZII")
                + SparsePauliOp("XIIII"),
            ),
        ],
    )
    def test_mp_to_pauli_for_general_operator(self, measurement_type, operator, expected):
        """Tests that a SparsePauliOp is created given any general operator that has a Pauli representation, and that it has the expected format"""
        obs = measurement_type(operator)
        register_size = 5
        pauli_op = mp_to_pauli(obs, register_size)
        assert isinstance(pauli_op, SparsePauliOp)

        pauli_op_list = list(pauli_op.paulis.to_labels()[0])
        # all qubits in register are accounted for
        assert len(pauli_op_list) == register_size
        assert pauli_op.equiv(expected.simplify())

    @pytest.mark.parametrize("measurement_type", [qml.expval, qml.var])
    @pytest.mark.parametrize(
        "operator, expected",
        [
            (X(0) @ Y(1), SparsePauliOp("IIX") @ (SparsePauliOp("IYI"))),
            (
                (X(0) + Y(1)) @ Y(1),
                (SparsePauliOp("IIX") + SparsePauliOp("IYI")) @ (SparsePauliOp("IYI")),
            ),
            (
                (X(0) + Y(1)) @ (Z(0) + Z(1)),
                (SparsePauliOp("IIX") + SparsePauliOp("IYI"))
                @ (SparsePauliOp("IIZ") + SparsePauliOp("IZI")),
            ),
            (
                2 * (X(0) + Y(1)) @ ((Z(0) + Z(1)) @ Z(2)),
                2
                * (SparsePauliOp("IIX") + SparsePauliOp("IYI"))
                @ (SparsePauliOp("IIZ") + SparsePauliOp("IZI"))
                @ SparsePauliOp("ZII"),
            ),
            (
                0.5 * (X(0) @ X(1)) + 0.7 * (X(1) @ X(2)) + 0.8 * (X(2) @ X(1)),
                0.5 * (SparsePauliOp("IIX") @ SparsePauliOp("IXI"))
                + 0.7 * (SparsePauliOp("IXI") @ SparsePauliOp("XII"))
                + 0.8 * (SparsePauliOp("XII") @ SparsePauliOp("IXI")),
            ),
        ],
    )
    def test_mp_to_pauli_tensor_products(self, measurement_type, operator, expected):
        """Tests that a SparsePauliOp is created given any general operator that has a Pauli representation, and that it is accurate"""
        obs = measurement_type(operator)
        register_size = 3

        pauli_op = mp_to_pauli(obs, register_size)
        assert isinstance(pauli_op, SparsePauliOp)

        pauli_op_list = list(pauli_op.paulis.to_labels()[0])
        # all qubits in register are accounted for
        assert len(pauli_op_list) == register_size
        assert pauli_op.equiv(expected.simplify())

    @pytest.mark.parametrize("measurement_type", [qml.expval, qml.var])
    @pytest.mark.parametrize(
        "hamiltonian, expected",
        [
            (
                qml.Hamiltonian([1, 2], [qml.X(0), qml.X(1)]),
                SparsePauliOp(["IIIIX", "IIIXI"], [1, 2]),
            ),
            (
                qml.Hamiltonian([3, -2], [qml.X(0), qml.X(0)]),
                SparsePauliOp(["IIIIX", "IIIIX"], [3, -2]),
            ),
            (
                qml.Hamiltonian([-3, 3, 0.5, 5], [qml.X(0), qml.X(0), qml.Z(1), qml.Y(2)]),
                SparsePauliOp(["IIIIX", "IIIIX", "IIIZI", "IIYII"], [-3, 3, 0.5, 5]),
            ),
            (
                qml.Hamiltonian([1], [qml.X(0)]) + 2 * qml.Z(0) @ qml.Z(1),
                SparsePauliOp("IIIIX") + 2 * SparsePauliOp("IIIIZ") @ SparsePauliOp("IIIZI"),
            ),
            (
                qml.Hamiltonian([1], [qml.X(0) @ Y(2)]) - 3 * qml.Z(4) @ qml.Z(1),
                (SparsePauliOp("IIIIX") @ SparsePauliOp("IIYII"))
                - 3 * SparsePauliOp("ZIIII") @ SparsePauliOp("IIIZI"),
            ),
        ],
    )
    def test_mp_to_pauli_for_hamiltonian(self, measurement_type, hamiltonian, expected):
        """Tests that a SparsePauliOp is created from a Hamiltonian, and that
        it has the expected format"""

        obs = measurement_type(hamiltonian)
        register_size = 5

        pauli_op = mp_to_pauli(obs, register_size)
        assert isinstance(pauli_op, SparsePauliOp)

        pauli_op_list = list(pauli_op.paulis.to_labels()[0])
        # all qubits in register are accounted for
        assert len(pauli_op_list) == register_size
        assert pauli_op.equiv(expected.simplify())

    @pytest.mark.parametrize("measurement_type", [qml.expval, qml.var])
    def test_mp_to_pauli_error_for_no_pauli_rep(self, measurement_type):
        """Tests that an error is raised when mp_to_pauli is given an operator that does not have a pauli representation"""

        obs = measurement_type(qml.X(0) @ qml.Hadamard(2))

        assert not obs.obs.pauli_rep
        with pytest.raises(ValueError, match="The operator"):
            mp_to_pauli(obs, 5)


# pylint:disable=not-context-manager
class TestControlOpIntegration:
    """Test the controlled flows integration with PennyLane"""

    @pytest.mark.parametrize("cond_type", ["clbit", "clreg", "expr1", "expr2", "expr3"])
    def test_control_flow_ops_circuit_ifelse(self, cond_type):
        """Tests mid-measurements are recognized and returned correctly."""

        qc = QuantumCircuit(3, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0, 0)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0, 1)

        condition = {
            "clbit": (0, 0),
            "clreg": (qc.cregs[0], 3),
            "expr1": expr.equal(3, qc.cregs[0]),
            "expr2": expr.bit_and(qc.cregs[0][0], True),
            "expr3": expr.bit_and(qc.cregs[0][0], qc.cregs[0][1]),
        }
        with qc.if_test(condition[cond_type]) as else_:
            qc.x(0)

        with else_:
            qc.h(0)
            qc.z(2)

        qc.rz(0.24, [0])
        qc.cx(0, 1)
        qc.measure_all()

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def loaded_qiskit_circuit():
            meas = load(qc)()
            return [qml.expval(m) for m in meas]

        @qml.qnode(dev)
        def built_pl_circuit():

            qml.Hadamard(0)
            qml.CNOT([0, 1])
            m0 = qml.measure(0)
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            m1 = qml.measure(0)

            condition = {
                "clbit": m0 == 0,
                "clreg": (m0 + 2 * m1 == 3),
                "expr1": (m0 + 2 * m1 == 3),
                "expr2": (m0 & True),
                "expr3": m0 & m1,
            }

            def ansatz_true():
                qml.PauliX(wires=0)

            def ansatz_false():
                qml.Hadamard(wires=0)
                qml.PauliZ(wires=2)

            qml.cond(condition[cond_type], ansatz_true, ansatz_false)()

            qml.RZ(0.24, wires=0)
            qml.CNOT([0, 1])
            qml.Barrier([0, 1, 2])
            return [qml.expval(m) for m in [m0, m1, qml.measure(0), qml.measure(1), qml.measure(2)]]

        assert loaded_qiskit_circuit() == built_pl_circuit()
        assert len(loaded_qiskit_circuit.tape.operations) == len(built_pl_circuit.tape.operations)
        for op1, op2 in zip(
            loaded_qiskit_circuit.tape.operations, built_pl_circuit.tape.operations
        ):
            if isinstance(op1, MidMeasureMP) or isinstance(op2, MidMeasureMP):
                assert op1.wires == op2.wires
            elif isinstance(op1, qml.ops.Conditional) or isinstance(op2, qml.ops.Conditional):
                assert qml.equal(op1.base, op2.base) and op1.meas_val.wires == op2.meas_val.wires
            else:
                assert qml.equal(op1, op2)

    # pylint: disable=too-many-statements
    @pytest.mark.parametrize("cond_type", ["clbit", "clreg", "expr1", "expr2", "expr3"])
    def test_control_flow_ops_circuit_switch(self, cond_type):
        """Tests mid-measurements are recognized and returned correctly."""

        qreg = QuantumRegister(3)
        creg = ClassicalRegister(3)
        qc = QuantumCircuit(qreg, creg)
        qc.rx(0.12, 0)
        qc.rx(0.24, 1)
        qc.rx(0.36, 2)
        qc.measure([0, 1, 2], [0, 1, 2])

        qc.add_register(QuantumRegister(2))
        qc.add_register(ClassicalRegister(2))
        qc.h(3)
        qc.cx(3, 4)
        qc.measure([3, 4], [3, 4])

        condition = {
            "clbit": creg[0],
            "clreg": qc.cregs[0],
            "expr1": expr.greater_equal(qc.cregs[0], 2),
            "expr2": expr.bit_and(qc.cregs[0][0], True),
            "expr3": expr.less_equal(qc.cregs[1], qc.cregs[0]),
        }

        with qc.switch(condition[cond_type]) as case:
            with case(0):
                qc.x(0)
            with case(1):
                qc.x(1)
            with case(case.DEFAULT):
                qc.x(2)
        qc.measure_all()

        dev = qml.device("default.qubit", wires=5, seed=24)
        qiskit_circuit = load(qc, measurements=[qml.expval(qml.PauliZ(0) @ qml.PauliY(1))])

        @qml.qnode(dev)
        def loaded_qiskit_circuit():
            return qiskit_circuit()

        @qml.qnode(dev)
        def built_pl_circuit():
            qml.RX(0.12, 0)
            qml.RX(0.24, 1)
            qml.RX(0.36, 2)
            m0 = qml.measure(0)
            m1 = qml.measure(1)
            m2 = qml.measure(2)

            qml.Hadamard(3)
            qml.CNOT([3, 4])
            m3 = qml.measure(3)
            m4 = qml.measure(4)

            mint1 = m0 + 2 * m1 + 4 * m2
            mint2 = m3 + 2 * m4
            if cond_type in ["clbit", "expr2"]:
                qml.cond(m0 == 0, qml.PauliX)([0])
                qml.cond(m0 == 1, qml.PauliX)([1])
            elif cond_type == "clreg":
                qml.cond(mint1 == 0, qml.PauliX)([0])
                qml.cond(mint1 == 1, qml.PauliX)([1])
                qml.cond((mint1 != 0) & (mint1 != 1), qml.PauliX)([2])
            elif cond_type == "expr1":
                qml.cond(mint1 >= 2, qml.PauliX)([0])
                qml.cond(mint1 < 2, qml.PauliX)([1])
            elif cond_type == "expr3":
                qml.cond(mint1 <= mint2, qml.PauliX)([0])
                qml.cond(mint1 > mint2, qml.PauliX)([1])

            qml.Barrier([0, 1, 2, 3, 4])

            return qml.expval(qml.PauliZ(0) @ qml.PauliY(1))

        assert loaded_qiskit_circuit() == built_pl_circuit()

        assert len(loaded_qiskit_circuit.tape.operations) == len(built_pl_circuit.tape.operations)
        for op1, op2 in zip(
            loaded_qiskit_circuit.tape.operations, built_pl_circuit.tape.operations
        ):
            if isinstance(op1, MidMeasureMP) or isinstance(op2, MidMeasureMP):
                assert op1.wires == op2.wires
            elif isinstance(op1, qml.ops.Conditional) or isinstance(op2, qml.ops.Conditional):
                assert qml.equal(op1.base, op2.base) and sorted(op1.meas_val.wires) == sorted(
                    op2.meas_val.wires
                )
            else:
                assert qml.equal(op1, op2)

    def test_warning_for_non_accessible_classical_info(self):
        """Tests a UserWarning is raised if we do not have access to classical info."""

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        with qc.if_test(expr.bit_and(qc.cregs[0][0], qc.cregs[0][1])):
            qc.z(0)
        qc.rz(0.543, [0])
        qc.cx(0, 1)
        qc.measure_all()

        measurements = [qml.expval(qml.PauliZ(0)), qml.vn_entropy([1])]
        quantum_circuit = load(qc, measurements=measurements)

        with pytest.warns(UserWarning):
            quantum_circuit()

    def test_direct_qnode_ui(self):
        """Test the UI where the loaded function is passed directly to qml.QNode
        along with a device"""

        dev = qml.device("default.qubit")
        angle = Parameter("angle")

        qc = QuantumCircuit(2, 2)
        qc.rx(angle, [0])
        qc.measure(0, 0)
        qc.rx(angle, [1])
        qc.cz(0, 1)

        measurements = [qml.expval(qml.PauliZ(0)), qml.vn_entropy([1])]

        @qml.qnode(dev)
        def circuit_native_pennylane(angle):
            qml.RX(angle, wires=0)
            qml.measure(0)
            qml.RX(angle, wires=1)
            qml.CZ([0, 1])
            return qml.expval(qml.PauliZ(0)), qml.vn_entropy([1])

        qnode = qml.QNode(load(qc, measurements), dev)

        assert np.allclose(qnode(0.543), circuit_native_pennylane(0.543))

    # pylint:disable=unused-variable
    def test_mid_circuit_as_terminal(self):
        """Test the control workflows where mid-circuit measurements disguise as terminal ones"""

        qc = QuantumCircuit(3, 2)

        qc.rx(0.9, 0)  # Prepare input state on qubit 0

        qc.h(1)  # Prepare Bell state on qubits 1 and 2
        qc.cx(1, 2)

        qc.cx(0, 1)  # Perform teleportation
        qc.h(0)
        qc.measure(0, 0)
        qc.measure(1, 1)

        with qc.if_test((1, 1)):
            qc.x(2)

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def qk_circuit():
            qml.from_qiskit(qc)()
            return qml.expval(qml.PauliZ(0))

        # pylint: disable=unused-variable
        @qml.qnode(dev)
        def pl_circuit():
            qml.RX(0.9, [0])
            qml.Hadamard([1])
            qml.CNOT([1, 2])
            qml.CNOT([0, 1])
            qml.Hadamard([0])
            m0 = qml.measure(0)
            m1 = qml.measure(1)
            qml.cond(m1 == 1, qml.PauliX)(wires=[2])
            return qml.expval(qml.PauliZ(0))

        assert qk_circuit() == pl_circuit()
        assert len(qk_circuit.tape.operations) == len(pl_circuit.tape.operations)
        for op1, op2 in zip(qk_circuit.tape.operations, pl_circuit.tape.operations):
            if isinstance(op1, MidMeasureMP) or isinstance(op2, MidMeasureMP):
                assert op1.wires == op2.wires
            elif isinstance(op1, qml.ops.Conditional) or isinstance(op2, qml.ops.Conditional):
                assert qml.equal(op1.base, op2.base) and sorted(op1.meas_val.wires) == sorted(
                    op2.meas_val.wires
                )
            else:
                assert qml.equal(op1, op2)

    def test_measurement_are_not_discriminated(self):
        """Test the all measurements are considered mid-circuit measurements when no terminal measurements are given"""

        qc = QuantumCircuit(3, 2)

        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0, 0)
        qc.measure(1, 1)

        with qc.if_test((1, 1)):
            qc.x(2)

        m0, m1 = load(qc)()
        w0, w1 = qml.wires.Wires([0]), qml.wires.Wires([1])
        assert isinstance(m0, qml.measurements.MeasurementValue) and m0.wires == w0
        assert isinstance(m1, qml.measurements.MeasurementValue) and m1.wires == w1

        m2 = load(qc, measurements=qml.expval(qml.PauliZ(2)))()
        w2 = qml.wires.Wires([2])
        assert isinstance(m2, qml.measurements.ExpectationMP) and m2.wires == w2


class TestPassingParameters:
    def _get_parameter_vector_test_circuit(self, qubit_device_2_wires):
        """A test circuit for testing"""
        theta = ParameterVector("v", 3)

        qc = QuantumCircuit(2)
        qc.rx(theta[0], [0])
        qc.ry(theta[1], [0])
        qc.rz(theta[2], [0])

        @qml.qnode(qubit_device_2_wires)
        def circuit_native_pennylane():
            qml.RX(0.4, wires=0)
            qml.RY(0.5, wires=0)
            qml.RZ(0.3, wires=0)
            return qml.expval(qml.PauliZ(0))

        return theta, qc, circuit_native_pennylane

    def _get_parameter_test_circuit(self, qubit_device_2_wires):

        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")

        qc = QuantumCircuit(2)
        qc.rx(c, [0])
        qc.ry(a, [0])
        qc.rz(b, [0])

        @qml.qnode(qubit_device_2_wires)
        def circuit_native_pennylane():
            qml.RX(0.4, wires=0)
            qml.RY(0.5, wires=0)
            qml.RZ(0.3, wires=0)
            return qml.expval(qml.PauliZ(0))

        return qc, circuit_native_pennylane

    def test_passing_parameters_new_interface_args(self, qubit_device_2_wires):
        """Test calling the qfunc with the new interface for setting the value
        of Qiskit Parameters by passing args in order."""

        qc, circuit_native_pennylane = self._get_parameter_test_circuit(qubit_device_2_wires)

        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit():
            load(qc)(0.5, 0.3, 0.4)  # a, b, c (alphabetical) rather than order used in qc
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit() == circuit_native_pennylane()

    def test_passing_parameters_new_interface_kwargs(self, qubit_device_2_wires):
        """Test calling the qfunc with the new interface for setting the value
        of Qiskit Parameters by passing kwargs matching the parameter names"""

        qc, circuit_native_pennylane = self._get_parameter_test_circuit(qubit_device_2_wires)

        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit():
            load(qc)(a=0.5, b=0.3, c=0.4)
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit() == circuit_native_pennylane()

    def test_passing_parameters_new_interface_mixed(self, qubit_device_2_wires):
        """Test calling the qfunc with the new interface for setting the value
        of Qiskit Parameters - by passing a combination of kwargs and args"""

        qc, circuit_native_pennylane = self._get_parameter_test_circuit(qubit_device_2_wires)

        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit():
            load(qc)(0.3, a=0.5, c=0.4)
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit() == circuit_native_pennylane()

    def test_using_parameter_vector_params_dict(self, qubit_device_2_wires):
        """Test that a parameterized QuanutmCircuit based on a ParameterVector can also be
        converted to a PennyLane template with the expected arguments passed as a params dict"""

        theta, qiskit_circuit, circuit_native_pennylane = self._get_parameter_vector_test_circuit(
            qubit_device_2_wires
        )

        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit():
            load(qiskit_circuit)({theta: [0.4, 0.5, 0.3]})
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit() == circuit_native_pennylane()

    def test_using_parameter_vector_with_positional_argument(self, qubit_device_2_wires):
        """Test that a parameterized QuanutmCircuit based on a ParameterVector can also be
        converted to a PennyLane template with the expected arguments passed as a params dict"""

        _, qiskit_circuit, circuit_native_pennylane = self._get_parameter_vector_test_circuit(
            qubit_device_2_wires
        )

        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit():
            load(qiskit_circuit)([0.4, 0.5, 0.3])
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit() == circuit_native_pennylane()

    def test_using_parameter_vector_with_keyword_argument(self, qubit_device_2_wires):
        """Test that a parameterized QuanutmCircuit based on a ParameterVector can also be
        converted to a PennyLane template with the expected arguments passed as a keyword arguement
        """

        _, qiskit_circuit, circuit_native_pennylane = self._get_parameter_vector_test_circuit(
            qubit_device_2_wires
        )

        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit():
            load(qiskit_circuit)(v=[0.4, 0.5, 0.3])
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit() == circuit_native_pennylane()


class TestLoadPauliOp:
    """Tests for the :func:`load_pauli_op()` function."""

    @pytest.mark.parametrize(
        "pauli_op, want_op",
        [
            (
                SparsePauliOp("I"),
                qml.Identity(wires=0),
            ),
            (
                SparsePauliOp("XYZ"),
                qml.prod(qml.PauliZ(wires=0), qml.PauliY(wires=1), qml.PauliX(wires=2)),
            ),
            (
                SparsePauliOp(["XY", "ZX"]),
                qml.sum(
                    qml.prod(qml.PauliX(wires=1), qml.PauliY(wires=0)),
                    qml.prod(qml.PauliZ(wires=1), qml.PauliX(wires=0)),
                ),
            ),
        ],
    )
    def test_convert_with_default_coefficients(self, pauli_op, want_op):
        """Tests that a SparsePauliOp can be converted into a PennyLane operator with the default
        coefficients.
        """
        have_op = load_pauli_op(pauli_op)
        assert qml.equal(have_op, want_op)

    @pytest.mark.parametrize(
        "pauli_op, want_op",
        [
            (
                SparsePauliOp("I", coeffs=[2]),
                qml.s_prod(2, qml.Identity(wires=0)),
            ),
            (
                SparsePauliOp(["XY", "ZX"], coeffs=[3, 7]),
                qml.sum(
                    qml.s_prod(3, qml.prod(qml.PauliX(wires=1), qml.PauliY(wires=0))),
                    qml.s_prod(7, qml.prod(qml.PauliZ(wires=1), qml.PauliX(wires=0))),
                ),
            ),
        ],
    )
    def test_convert_with_literal_coefficients(self, pauli_op, want_op):
        """Tests that a SparsePauliOp can be converted into a PennyLane operator with literal
        coefficient values.
        """
        have_op = load_pauli_op(pauli_op)
        assert qml.equal(have_op, want_op)

    def test_convert_with_parameter_coefficients(self):
        """Tests that a SparsePauliOp can be converted into a PennyLane operator by assigning values
        to each parameterized coefficient.
        """
        a, b = [Parameter(var) for var in "ab"]
        pauli_op = SparsePauliOp(["XY", "ZX"], coeffs=[a, b])

        have_op = load_pauli_op(pauli_op, params={a: 3, b: 7})
        want_op = qml.sum(
            qml.s_prod(3, qml.prod(qml.PauliX(wires=1), qml.PauliY(wires=0))),
            qml.s_prod(7, qml.prod(qml.PauliZ(wires=1), qml.PauliX(wires=0))),
        )
        assert qml.equal(have_op, want_op)

    def test_convert_too_few_coefficients(self):
        """Tests that a RuntimeError is raised if an attempt is made to convert a SparsePauliOp into
        a PennyLane operator without assigning values for all parameterized coefficients.
        """
        a, b = [Parameter(var) for var in "ab"]
        pauli_op = SparsePauliOp(["XY", "ZX"], coeffs=[a, b])

        match = (
            "Not all parameter expressions are assigned in coeffs "
            r"\[\(3\+0j\) ParameterExpression\(1\.0\*b\)\]"
        )
        with pytest.raises(RuntimeError, match=match):
            load_pauli_op(pauli_op, params={a: 3})

    def test_convert_too_many_coefficients(self):
        """Tests that a SparsePauliOp can be converted into a PennyLane operator by assigning values
        to a strict superset of the parameterized coefficients.
        """
        a, b, c = [Parameter(var) for var in "abc"]
        pauli_op = SparsePauliOp(["XY", "ZX"], coeffs=[a, b])

        have_op = load_pauli_op(pauli_op, params={a: 3, b: 7, c: 9})
        want_op = qml.sum(
            qml.s_prod(3, qml.prod(qml.PauliX(wires=1), qml.PauliY(wires=0))),
            qml.s_prod(7, qml.prod(qml.PauliZ(wires=1), qml.PauliX(wires=0))),
        )
        assert qml.equal(have_op, want_op)

    @pytest.mark.parametrize(
        "pauli_op, wires, want_op",
        [
            (
                SparsePauliOp("XYZ"),
                "ABC",
                qml.prod(qml.PauliZ(wires="A"), qml.PauliY(wires="B"), qml.PauliX(wires="C")),
            ),
            (
                SparsePauliOp(["XY", "ZX"]),
                [1, 0],
                qml.sum(
                    qml.prod(qml.PauliX(wires=0), qml.PauliY(wires=1)),
                    qml.prod(qml.PauliZ(wires=0), qml.PauliX(wires=1)),
                ),
            ),
        ],
    )
    def test_convert_with_wires(self, pauli_op, wires, want_op):
        """Tests that a SparsePauliOp can be converted into a PennyLane operator with custom wires."""
        have_op = load_pauli_op(pauli_op, wires=wires)
        assert qml.equal(have_op, want_op)

    def test_convert_with_too_few_wires(self):
        """Tests that a RuntimeError is raised if an attempt is made to convert a SparsePauliOp into
        a PennyLane operator with too few custom wires.
        """
        match = (
            "The specified number of wires - 1 - does not match "
            "the number of qubits the SparsePauliOp acts on."
        )
        with pytest.raises(RuntimeError, match=match):
            load_pauli_op(SparsePauliOp("II"), wires=[0])

    def test_convert_with_too_many_wires(self):
        """Tests that a RuntimeError is raised if an attempt is made to convert a SparsePauliOp into
        a PennyLane operator with too many custom wires.
        """
        match = (
            "The specified number of wires - 3 - does not match "
            "the number of qubits the SparsePauliOp acts on."
        )
        with pytest.raises(RuntimeError, match=match):
            load_pauli_op(SparsePauliOp("II"), wires=[0, 1, 2])

    def test_convert_with_invalid_operator(self):
        """Tests that a ValueError is raised if an attempt is made to convert an object which is not
        a SparsePauliOp into a PennyLane operator.
        """
        match = "The operator 123 is not a valid Qiskit SparsePauliOp."
        with pytest.raises(ValueError, match=match):
            load_pauli_op(123)


# pylint:disable = import-outside-toplevel, too-few-public-methods
class TestLoadNoiseModel:
    """Tests for :func:`load_noise_models()` function."""

    @staticmethod
    def _kraus_to_choi(krau_mats, optimize=False) -> np.ndarray:
        r"""Transforms Kraus representation of a channel to its Choi representation."""
        kraus_vecs = np.array([kraus.ravel(order="F") for kraus in krau_mats])
        return np.einsum("ij,ik->jk", kraus_vecs, kraus_vecs.conj(), optimize=optimize)

    def test_build_noise_model(self):
        """Tests that ``load_quantum_noise`` constructs a correct PennyLane NoiseModel from a given Qiskit noise model"""
        from qiskit_aer import noise
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        noise_model = noise.NoiseModel.from_backend(FakeOpenPulse2Q())
        loaded_noise_model = load_noise_model(noise_model)

        pl_model_map = {
            op_in("Identity")
            & wires_in(0): qml.ThermalRelaxationError(
                0.0, 26981.9403362283, 26034.6676428009, 1.0, wires=AnyWires
            ),
            op_in("Identity")
            & wires_in(1): qml.ThermalRelaxationError(
                0.0, 30732.034088541, 28335.6514829973, 1.0, wires=AnyWires
            ),
            (op_in("U1") & wires_in(0))
            | (op_in("U1") & wires_in(1)): qml.DepolarizingChannel(
                p=0.08999999999999997, wires=AnyWires
            ),
            op_in("U2")
            & wires_in(0): qml.ThermalRelaxationError(
                0.4998455776, 7.8227384666, 7.8226559459, 1.0, wires=AnyWires
            ),
            op_in("U2")
            & wires_in(1): qml.ThermalRelaxationError(
                0.4998644198, 7.8227957211, 7.8226273195, 1.0, wires=AnyWires
            ),
            op_in("U3")
            & wires_in(0): qml.ThermalRelaxationError(
                0.4996911588, 7.8227934813, 7.8226284393, 1.0, wires=AnyWires
            ),
            op_in("U3")
            & wires_in(1): qml.ThermalRelaxationError(
                0.4997288404, 7.8229079927, 7.8225711871, 1.0, wires=AnyWires
            ),
            op_in("CNOT")
            & wires_in([0, 1]): qml.QubitChannel(
                Kraus(noise_model._local_quantum_errors["cx"][(0, 1)]).data,
                wires=AnyWires,
            ),
        }

        pl_noise_model = qml.NoiseModel(
            {fcond: partial_wires(noise) for fcond, noise in pl_model_map.items()}
        )

        for (pl_k, pl_v), (qk_k, qk_v) in zip(
            pl_noise_model.model_map.items(), loaded_noise_model.model_map.items()
        ):
            pl_op, qk_op = pl_v(AnyWires), qk_v(AnyWires)
            assert repr(pl_k) == repr(qk_k)
            assert isinstance(qk_op, qml.QubitChannel)

            choi_mat1 = self._kraus_to_choi(qk_op.data)
            choi_mat2 = self._kraus_to_choi(pl_op.compute_kraus_matrices(*pl_op.data))
            assert np.allclose(choi_mat1, choi_mat2)

    @pytest.mark.parametrize(
        "verbose, decimal",
        [(True, 8), (False, None)],
    )
    def test_build_noise_model_with_args(self, verbose, decimal):
        """Tests that ``load_quantum_noise`` constructs a correct PennyLane NoiseModel with args"""
        from qiskit_aer import noise

        error_1 = noise.depolarizing_error(0.001, 1)
        error_2 = noise.depolarizing_error(0.01, 2)

        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, ["rz", "ry"])
        noise_model.add_all_qubit_quantum_error(error_2, ["cx"])
        loaded_noise_model = load_noise_model(noise_model, verbose=verbose, decimal_places=decimal)

        pauli_mats1 = list(map(qml.matrix, [qml.I(0), qml.X(0), qml.Y(0), qml.Z(0)]))
        pauli_mats2 = list(
            ft.reduce(np.kron, prod, 1.0) for prod in it.product(pauli_mats1, repeat=2)
        )
        pauli_prob1 = np.sqrt(error_1.probabilities)
        pauli_prob2 = np.sqrt(error_2.probabilities)
        kraus_ops1 = [prob * kraus_op for prob, kraus_op in zip(pauli_prob1, pauli_mats1)]
        kraus_ops2 = [prob * kraus_op for prob, kraus_op in zip(pauli_prob2, pauli_mats2)]

        c0 = qml.noise.op_in([qml.RZ, qml.RY])
        c1 = qml.noise.op_in(qml.CNOT)
        n0 = qml.noise.partial_wires(qml.QubitChannel(kraus_ops1, wires=[0]))
        n1 = qml.noise.partial_wires(qml.QubitChannel(kraus_ops2, wires=[0, 1]))
        pl_noise_model = qml.NoiseModel({c0: n0, c1: n1})

        for (pl_k, pl_v), (qk_k, qk_v) in zip(
            pl_noise_model.model_map.items(), loaded_noise_model.model_map.items()
        ):
            assert repr(pl_k) == repr(qk_k)

            pl_data = np.array(pl_v(AnyWires).data)
            if verbose:
                choi_mat1 = self._kraus_to_choi(qk_v(AnyWires).data)
                choi_mat2 = self._kraus_to_choi(pl_data)
                assert np.allclose(choi_mat1, choi_mat2)
            else:
                num_kraus, num_wires = pl_data.shape[0], int(np.log2(pl_data.shape[1]))
                assert (
                    qk_v.__name__ == f"QubitChannel(num_kraus={num_kraus}, num_wires={num_wires})"
                )
