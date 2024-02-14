import math
import sys

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import extensions as ex
from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator, SparsePauliOp

import pennylane as qml
from pennylane import numpy as np
from pennylane_qiskit.converter import (
    load,
    load_qasm,
    load_qasm_from_file,
    map_wires,
    circuit_to_qiskit,
    operation_to_qiskit,
    mp_to_pauli,
    _format_params_dict,
    _check_parameter_bound,
)
from pennylane.wires import Wires
from pennylane.tape.qscript import QuantumScript


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
        qc_1 = qc.bind_parameters({theta: 0.5})

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

    def test_parameter_was_not_bound(self, recorder):
        """Tests that an error is raised when parameters were not bound."""

        theta = Parameter("θ")
        unbound_params = {}

        with pytest.raises(
            ValueError, match="The parameter {} was not bound correctly.".format(theta)
        ):
            _check_parameter_bound(theta, unbound_params)

    def test_extra_parameters_were_passed(self, recorder):
        """Tests that loading raises an error when extra parameters were
        passed."""

        theta = Parameter("θ")
        phi = Parameter("φ")
        x = np.tensor(0.5, requires_grad=False)
        y = np.tensor(0.3, requires_grad=False)

        qc = QuantumCircuit(3, 1)

        quantum_circuit = load(qc)

        with pytest.raises(QiskitError):
            with recorder:
                quantum_circuit(params={theta: x, phi: y})

    def test_quantum_circuit_error_passing_parameters_not_required(self, recorder):
        """Tests the load method raises a QiskitError if arguments
        that are not required were passed."""

        theta = Parameter("θ")
        angle = np.tensor(0.5, requires_grad=False)

        qc = QuantumCircuit(3, 1)
        qc.z([0])

        quantum_circuit = load(qc)

        with pytest.raises(QiskitError):
            with recorder:
                quantum_circuit(params={theta: angle})

    def test_quantum_circuit_error_not_qiskit_circuit_passed(self, recorder):
        """Tests the load method raises a ValueError, if something
        that is not a QuanctumCircuit was passed."""

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
            match="The specified number of wires - {} - does not match the"
            " number of wires the loaded quantum circuit acts on.".format(len(only_two_wires)),
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
            match="The specified number of wires - {} - does not match the"
            " number of wires the loaded quantum circuit acts on.".format(
                len(more_than_three_wires)
            ),
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
    def test_controlled_rotations(self, qiskit_operation, pennylane_name, recorder):
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

    def test_operation_transformed_into_qubit_unitary(self, recorder):
        """Tests loading a circuit with operations that can be converted,
        but not natively supported by PennyLane."""

        qc = QuantumCircuit(3, 1)

        qc.ch([0], [1])

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert recorder.queue[0].name == "QubitUnitary"
        assert len(recorder.queue[0].parameters) == 1
        assert np.array_equal(recorder.queue[0].parameters[0], ex.CHGate().to_matrix())
        assert recorder.queue[0].wires == Wires([0, 1])


class TestConverterUtils:
    """Tests the utility functions used by the converter function."""

    def test_map_wires(self, recorder):
        """Tests the map_wires function for wires of a quantum circuit."""

        wires = [0]
        qc = QuantumCircuit(1)
        qc_wires = [hash(q) for q in qc.qubits]

        assert map_wires(wires, qc_wires) == {0: hash(qc.qubits[0])}

    def test_map_wires_instantiate_quantum_circuit_with_registers(self, recorder):
        """Tests the map_wires function for wires of a quantum circuit instantiated
        using quantum registers."""

        wires = [0, 1, 2]
        qr1 = QuantumRegister(1)
        qr2 = QuantumRegister(1)
        qr3 = QuantumRegister(1)
        qc = QuantumCircuit(qr1, qr2, qr3)
        qc_wires = [hash(q) for q in qc.qubits]

        mapped_wires = map_wires(wires, qc_wires)

        assert len(mapped_wires) == len(wires)
        assert list(mapped_wires.keys()) == wires
        for q in qc.qubits:
            assert hash(q) in mapped_wires.values()

    def test_map_wires_provided_non_standard_order(self, recorder):
        """Tests the map_wires function for wires of non-standard order."""

        wires = [1, 2, 0]
        qc = QuantumCircuit(3)
        qc_wires = [hash(q) for q in qc.qubits]

        mapped_wires = map_wires(wires, qc_wires)

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
            match="The specified number of wires - {} - does not match "
            "the number of wires the loaded quantum circuit acts on.".format(len(wires)),
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
        qc = EfficientSU2(3, reps=1)

        quantum_circuit = load(qc)
        params = np.arange(12)

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

        qc = QuantumCircuit(2, 2)
        qc.rx(parameter, 0)
        qc.rx(0.3, 1)
        qc.measure_all()

        # works with correct name
        load(qc)(name1=0.3)

        # raises error with incorrect name
        with pytest.raises(TypeError, match="Got unexpected parameter keyword argument 'name2'"):
            load(qc)(name2=0.3)

    def test_too_many_args(self):
        """Test that if too many positional arguments are passed to define ``Parameter`` values,
        a clear error is raised"""

        a = Parameter("a")
        b = Parameter("b")

        qc = QuantumCircuit(2, 2)
        qc.rx(a, 0)
        qc.rx(b, 1)
        qc.measure_all()

        with pytest.raises(TypeError, match="Expected 2 positional arguments but 3 were given"):
            load(qc)(0.2, 0.4, 0.5)

        with pytest.raises(TypeError, match="Expected 1 positional argument but 2 were given"):
            load(qc)(0.4, 0.5, a=0.2)

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
            TypeError, match="Missing 1 required argument to define Parameter value"
        ):
            load(qc)(0.2, 0.3)

        with pytest.raises(
            TypeError, match="Missing 1 required argument to define Parameter value"
        ):
            load(qc)(0.2, c=0.4)

        with pytest.raises(
            TypeError, match="Missing 2 required arguments to define Parameter values"
        ):
            load(qc)(b=0.3)

        with pytest.raises(
            TypeError, match="Missing 1 required argument to define Parameter value"
        ):
            load(qc)({a: 0.2, b: 0.3})

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

        with open(qft_qasm, "w") as f:
            f.write(TestConverterQasm.qft_qasm)

        quantum_circuit = load_qasm_from_file(qft_qasm)

        with recorder:
            quantum_circuit()

        assert len(recorder.queue) == 10

        assert recorder.queue[0].name == "PauliX"
        assert recorder.queue[0].parameters == []
        assert recorder.queue[0].wires == Wires([0])

        assert recorder.queue[1].name == "PauliX"
        assert recorder.queue[1].parameters == []
        assert recorder.queue[1].wires == Wires([2])

        assert recorder.queue[2].name == "Hadamard"
        assert recorder.queue[2].parameters == []
        assert recorder.queue[2].wires == Wires([0])

        assert recorder.queue[3].name == "Hadamard"
        assert recorder.queue[3].parameters == []
        assert recorder.queue[3].wires == Wires([1])

        assert recorder.queue[4].name == "Hadamard"
        assert recorder.queue[4].parameters == []
        assert recorder.queue[4].wires == Wires([2])

        assert recorder.queue[5].name == "Hadamard"
        assert recorder.queue[5].parameters == []
        assert recorder.queue[5].wires == Wires([3])

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

        assert len(recorder.queue) == 6

        assert recorder.queue[0].name == "PauliX"
        assert recorder.queue[0].parameters == []
        assert recorder.queue[0].wires == Wires([0])

        assert recorder.queue[1].name == "CNOT"
        assert recorder.queue[1].parameters == []
        assert recorder.queue[1].wires == Wires([2, 0])


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

    def test_passing_parameters_new_interface_args(self, qubit_device_2_wires):
        """Test calling the qfunc with the new interface for setting the value
        of Qiskit Parameters by passing args in order."""

        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")

        qc = QuantumCircuit(2)
        qc.rx(c, [0])
        qc.ry(a, [0])
        qc.rz(b, [0])

        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit():
            load(qc)(0.5, 0.3, 0.4)  # a, b, c (alphabetical) rather than order used in qc
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qubit_device_2_wires)
        def circuit_native_pennylane():
            qml.RX(0.4, wires=0)
            qml.RY(0.5, wires=0)
            qml.RZ(0.3, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit() == circuit_native_pennylane()

    def test_passing_parameters_new_interface_kwargs(self, qubit_device_2_wires):
        """Test calling the qfunc with the new interface for setting the value
        of Qiskit Parameters by passing kwargs matching the parameter names"""

        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")

        qc = QuantumCircuit(2)
        qc.rx(c, [0])
        qc.ry(a, [0])
        qc.rz(b, [0])

        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit():
            load(qc)(a=0.5, b=0.3, c=0.4)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qubit_device_2_wires)
        def circuit_native_pennylane():
            qml.RX(0.4, wires=0)
            qml.RY(0.5, wires=0)
            qml.RZ(0.3, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit() == circuit_native_pennylane()

    def test_passing_parameters_new_interface_mixed(self, qubit_device_2_wires):
        """Test calling the qfunc with the new interface for setting the value
        of Qiskit Parameters - by passing a combination of kwargs and args"""

        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")

        qc = QuantumCircuit(2)
        qc.rx(c, [0])
        qc.ry(a, [0])
        qc.rz(b, [0])

        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit():
            load(qc)(0.3, a=0.5, c=0.4)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qubit_device_2_wires)
        def circuit_native_pennylane():
            qml.RX(0.4, wires=0)
            qml.RY(0.5, wires=0)
            qml.RZ(0.3, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit() == circuit_native_pennylane()

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
        """Test that the gradient works correctly"""
        qc = QuantumCircuit(3)
        qiskit_params = [Parameter("param_{}".format(i)) for i in range(3)]

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
    def test_differentiable_param_is_array(self, shots, tol):
        """Test that extracting the differentiable parameters works correctly
        for arrays"""
        qc = QuantumCircuit(3)
        qiskit_params = [Parameter("param_{}".format(i)) for i in range(3)]

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

    def test_meas_circuit_in_qnode(self, qubit_device_2_wires):
        """Tests loading a converted template in a QNode with measurements."""

        angle = 0.543

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)
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
            qml.measure(0)
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
        else:
            final_instruction.operation.name == final_op_name

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
            qml.QubitStateVector(np.array([1, 0, 0, 0]), wires=[0, 1]),
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


class TestConverterUtilsPennyLaneToQiskit:

    @pytest.mark.parametrize("measurement_type", [qml.expval, qml.var])
    @pytest.mark.parametrize(
        "observable, obs_string",
        [(qml.PauliX, "X"), (qml.PauliY, "Y"), (qml.PauliZ, "Z"), (qml.Identity, "I")],
    )
    @pytest.mark.parametrize("wire", [0, 1, 2])
    @pytest.mark.parametrize("register_size", [3, 5])
    def test_mp_to_pauli(self, measurement_type, observable, obs_string, wire, register_size):
        """Tests that a SparsePauliOp is created from a Pauli observable, and that
        it has the expected format"""

        obs = measurement_type(observable(wire))

        pauli_op = mp_to_pauli(obs, register_size)
        assert isinstance(pauli_op, SparsePauliOp)

        pauli_op_list = list(pauli_op.paulis.to_labels()[0])

        # all qubits in register are accounted for
        assert len(pauli_op_list) == register_size

        # the wire the observable acts on is correctly labelled
        # wire order reversed in Qiskit, so we put it back to use PL wire as an index
        pauli_op_list.reverse()
        assert pauli_op_list.pop(wire) == obs_string

        # remaining wires are all Identity
        assert np.all([op == "I" for op in pauli_op_list])

