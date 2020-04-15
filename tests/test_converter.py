import math
import sys

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import extensions as ex
from qiskit.circuit import Parameter
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator

import pennylane as qml
from pennylane import numpy as np
from pennylane_qiskit.converter import load, load_qasm, load_qasm_from_file, map_wires


THETA = np.linspace(0.11, 3, 5)
PHI = np.linspace(0.32, 3, 5)
VARPHI = np.linspace(0.02, 3, 5)


class TestConverter:
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
        assert recorder.queue[0].name == 'RZ'
        assert recorder.queue[0].params == [angle]
        assert recorder.queue[0].wires == [0]

    def test_quantum_circuit_by_passing_parameters(self, recorder):
        """Tests the load method for a QuantumCircuit initialized by passing the number
        of registers required."""

        theta = Parameter('θ')
        angle = 0.5

        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit(params={theta: angle})

        assert len(recorder.queue) == 1
        assert recorder.queue[0].name == 'RZ'
        assert recorder.queue[0].params == [angle]
        assert recorder.queue[0].wires == [0]

    def test_loaded_quantum_circuit_and_further_pennylane_operations(self, recorder):
        """Tests that a loaded quantum circuit can be used around other PennyLane
        templates in a circuit."""

        theta = Parameter('θ')
        angle = 0.5

        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])

        quantum_circuit = load(qc)

        with recorder:
            qml.PauliZ(0)
            quantum_circuit(params={theta: angle})
            qml.Hadamard(0)

        assert len(recorder.queue) == 3
        assert recorder.queue[0].name == 'PauliZ'
        assert recorder.queue[0].params == []
        assert recorder.queue[0].wires == [0]
        assert recorder.queue[1].name == 'RZ'
        assert recorder.queue[1].params == [angle]
        assert recorder.queue[1].wires == [0]
        assert recorder.queue[2].name == 'Hadamard'
        assert recorder.queue[2].params == []
        assert recorder.queue[2].wires == [0]

    def test_quantum_circuit_with_multiple_parameters(self, recorder):
        """Tests loading a circuit with multiple parameters."""

        angle1 = 0.5
        angle2 = 0.3

        phi = Parameter('φ')
        theta = Parameter('θ')

        qc = QuantumCircuit(3, 1)
        qc.rx(phi, 1)
        qc.rz(theta, 0)

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit(params={phi: angle1, theta: angle2})

        assert len(recorder.queue) == 2
        assert recorder.queue[0].name == 'RX'
        assert recorder.queue[0].params == [angle1]
        assert recorder.queue[0].wires == [1]
        assert recorder.queue[1].name == 'RZ'
        assert recorder.queue[1].params == [angle2]
        assert recorder.queue[1].wires == [0]

    def test_quantum_circuit_with_gate_requiring_multiple_parameters(self, recorder):
        """Tests loading a circuit containing a gate that requires
        multiple parameters."""

        angle1 = 0.5
        angle2 = 0.3
        angle3 = 0.1

        phi = Parameter('φ')
        lam = Parameter('λ')
        theta = Parameter('θ')

        qc = QuantumCircuit(3, 1)
        qc.u3(phi, lam, theta, [0])

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit(params={phi: angle1, lam: angle2, theta: angle3})

        assert recorder.queue[0].name == 'U3'
        assert len(recorder.queue[0].params) == 3
        assert recorder.queue[0].params == [0.5, 0.3, 0.1]
        assert recorder.queue[0].wires == [0]

    def test_quantum_circuit_loaded_multiple_times_with_different_arguments(self, recorder):
        """Tests that a loaded quantum circuit can be called multiple times with
        different arguments."""

        theta = Parameter('θ')
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
        assert recorder.queue[0].name == 'RZ'
        assert recorder.queue[0].params == [angle1]
        assert recorder.queue[0].wires == [0]
        assert recorder.queue[1].name == 'RZ'
        assert recorder.queue[1].params == [angle2]
        assert recorder.queue[1].wires == [0]
        assert recorder.queue[2].name == 'RZ'
        assert recorder.queue[2].params == [angle3]
        assert recorder.queue[2].wires == [0]

    def test_quantum_circuit_with_bound_parameters(self, recorder):
        """Tests loading a quantum circuit that already had bound parameters."""

        theta = Parameter('θ')

        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])
        qc_1 = qc.bind_parameters({theta: 0.5})

        quantum_circuit = load(qc_1)

        with recorder:
            quantum_circuit()

        assert len(recorder.queue) == 1
        assert recorder.queue[0].name == 'RZ'
        assert recorder.queue[0].params == [0.5]
        assert recorder.queue[0].wires == [0]

    def test_pass_parameters_to_bind(self, recorder):
        """Tests parameter binding by passing parameters when loading a quantum circuit."""

        theta = Parameter('θ')

        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit(params={theta: 0.5})

        assert len(recorder.queue) == 1
        assert recorder.queue[0].name == 'RZ'
        assert recorder.queue[0].params == [0.5]
        assert recorder.queue[0].wires == [0]

    def test_parameter_was_not_bound(self, recorder):
        """Tests that loading raises an error when parameters were not bound."""

        theta = Parameter('θ')

        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])

        quantum_circuit = load(qc)

        with pytest.raises(ValueError, match='The parameter {} was not bound correctly.'.format(theta)):
            with recorder:
                quantum_circuit(params={})

    def test_invalid_parameter_expression(self, recorder):
        """Tests that an operation with multiple parameters raises an error."""

        theta = Parameter('θ')
        phi = Parameter('φ')

        qc = QuantumCircuit(3, 1)
        qc.rz(theta*phi, [0])

        quantum_circuit = load(qc)

        with pytest.raises(ValueError, match='PennyLane does not support expressions'):
            with recorder:
                quantum_circuit(params={theta: qml.variable.Variable(0), phi: qml.variable.Variable(1)})

    def test_extra_parameters_were_passed(self, recorder):
        """Tests that loading raises an error when extra parameters were
        passed."""

        theta = Parameter('θ')
        phi = Parameter('φ')

        qc = QuantumCircuit(3, 1)

        quantum_circuit = load(qc)

        with pytest.raises(QiskitError):
            with recorder:
                quantum_circuit(params={theta: 0.5, phi: 0.3})

    @pytest.mark.parametrize("qiskit_operation, pennylane_name", [(QuantumCircuit.crx, "CRX"), (QuantumCircuit.crz, "CRZ")])
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
        assert recorder.queue[0].params == [0.5]
        assert recorder.queue[0].wires == [0, 1]

    def test_cry(self, recorder):
        """Tests that the decomposition of the controlled-Y operation is being
        loaded."""
        # This test will be merged into the test_controlled_rotations test once
        # the native CRY is used in Qiskit and not a set of instructions
        # yielded from decomposition is being added

        q2 = QuantumRegister(2)
        qc = QuantumCircuit(q2)

        qc.cry(0.5, q2[0], q2[1])

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit()

        assert len(recorder.queue) == 4
        assert recorder.queue[0].name == "U3"
        assert recorder.queue[0].params == [0.25, 0, 0]
        assert recorder.queue[0].wires == [1]

        assert recorder.queue[1].name == "CNOT"
        assert recorder.queue[1].wires == [0, 1]

        assert recorder.queue[2].name == "U3"
        assert recorder.queue[2].params == [-0.25, 0, 0]
        assert recorder.queue[2].wires == [1]

        assert recorder.queue[3].name == "CNOT"
        assert recorder.queue[3].wires == [0, 1]


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

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert len(recorder.queue) == 6

        assert recorder.queue[0].name == 'PauliX'
        assert recorder.queue[0].params == []
        assert recorder.queue[0].wires == single_wire

        assert recorder.queue[1].name == 'PauliY'
        assert recorder.queue[1].params == []
        assert recorder.queue[1].wires == single_wire

        assert recorder.queue[2].name == 'PauliZ'
        assert recorder.queue[2].params == []
        assert recorder.queue[2].wires == single_wire

        assert recorder.queue[3].name == 'Hadamard'
        assert recorder.queue[3].params == []
        assert recorder.queue[3].wires == single_wire

        assert recorder.queue[4].name == 'S'
        assert recorder.queue[4].params == []
        assert recorder.queue[4].wires == single_wire

        assert recorder.queue[5].name == 'T'
        assert recorder.queue[5].params == []
        assert recorder.queue[5].wires == single_wire

    def test_one_qubit_parametrized_operations_supported_by_pennylane(self, recorder):
        """Tests loading a circuit with the one-qubit parametrized operations supported by PennyLane."""

        single_wire = [0]
        angle = 0.3333

        phi = 0.3
        lam = 0.4
        theta = 0.2

        q_reg = QuantumRegister(1)
        qc = QuantumCircuit(q_reg)

        qc.u1(angle, single_wire)
        qc.rx(angle, single_wire)
        qc.ry(angle, single_wire)
        qc.rz(angle, single_wire)
        qc.u2(phi, lam, [0])
        qc.u3(phi, lam, theta, [0])

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert recorder.queue[0].name == 'PhaseShift'
        assert recorder.queue[0].params == [angle]
        assert recorder.queue[0].wires == single_wire

        assert recorder.queue[1].name == 'RX'
        assert recorder.queue[1].params == [angle]
        assert recorder.queue[1].wires == single_wire

        assert recorder.queue[2].name == 'RY'
        assert recorder.queue[2].params == [angle]
        assert recorder.queue[2].wires == single_wire

        assert recorder.queue[3].name == 'RZ'
        assert recorder.queue[3].params == [angle]
        assert recorder.queue[3].wires == single_wire

        assert recorder.queue[4].name == 'U2'
        assert len(recorder.queue[4].params) == 2
        assert recorder.queue[4].params == [0.3, 0.4]
        assert recorder.queue[4].wires == [0]

        assert recorder.queue[5].name == 'U3'
        assert len(recorder.queue[5].params) == 3
        assert recorder.queue[5].params == [0.3, 0.4, 0.2]
        assert recorder.queue[5].wires == [0]

    def test_two_qubit_operations_supported_by_pennylane(self, recorder):
        """Tests loading a circuit with the two-qubit operations supported by PennyLane."""

        two_wires = [0, 1]

        qc = QuantumCircuit(2, 1)

        unitary_op = [[1, 0, 0, 0],
                     [0, 0, 1j, 0],
                     [0, 1j, 0, 0],
                     [0, 0, 0, 1]]
        iswap_op = Operator(unitary_op)


        qc.cx(*two_wires)
        qc.cz(*two_wires)
        qc.swap(*two_wires)
        qc.unitary(iswap_op, [0, 1], label='iswap')

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert len(recorder.queue) == 4

        assert recorder.queue[0].name == 'CNOT'
        assert recorder.queue[0].params == []
        assert recorder.queue[0].wires == two_wires

        assert recorder.queue[1].name == 'CZ'
        assert recorder.queue[1].params == []
        assert recorder.queue[1].wires == two_wires

        assert recorder.queue[2].name == 'SWAP'
        assert recorder.queue[2].params == []
        assert recorder.queue[2].wires == two_wires

        assert recorder.queue[3].name == 'QubitUnitary'
        assert len(recorder.queue[3].params) == 1
        assert np.array_equal(recorder.queue[3].params[0], np.array(unitary_op))
        assert recorder.queue[3].wires == two_wires

    def test_two_qubit_parametrized_operations_supported_by_pennylane(self, recorder):
        """Tests loading a circuit with the two-qubit parametrized operations supported by PennyLane."""

        two_wires = [0, 1]
        angle = 0.3333

        qc = QuantumCircuit(2, 1)

        qc.crz(angle, *two_wires)

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert len(recorder.queue) == 1

        assert recorder.queue[0].name == 'CRZ'
        assert recorder.queue[0].params == [angle]
        assert recorder.queue[0].wires == two_wires

    def test_three_qubit_operations_supported_by_pennylane(self, recorder):
        """Tests loading a circuit with the three-qubit operations supported by PennyLane."""

        three_wires = [0, 1, 2]

        qc = QuantumCircuit(3, 1)
        qc.cswap(*three_wires)
        qc.ccx(*three_wires)
        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit()

        assert recorder.queue[0].name == 'CSWAP'
        assert recorder.queue[0].params == []
        assert recorder.queue[0].wires == three_wires

        assert recorder.queue[1].name == 'Toffoli'
        assert len(recorder.queue[1].params) == 0
        assert recorder.queue[1].wires == three_wires

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

        assert recorder.queue[0].name == 'CSWAP'
        assert recorder.queue[0].params == []
        assert recorder.queue[0].wires == three_wires

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

        assert recorder.queue[0].name == 'CSWAP'
        assert recorder.queue[0].params == []
        assert recorder.queue[0].wires == three_wires

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

        assert recorder.queue[0].name == 'CSWAP'
        assert recorder.queue[0].params == []
        assert recorder.queue[0].wires == three_wires

    def test_operations_sdg_and_tdg(self, recorder):
        """Tests loading a circuit with the operations Sdg and Tdg gates."""

        qc = QuantumCircuit(3, 1)

        qc.sdg([0])
        qc.tdg([0])

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert recorder.queue[0].name == 'S.inv'
        assert len(recorder.queue[0].params) == 0
        assert recorder.queue[0].wires == [0]

        assert recorder.queue[1].name == 'T.inv'
        assert len(recorder.queue[1].params) == 0
        assert recorder.queue[1].wires == [0]

    def test_operation_transformed_into_qubit_unitary(self, recorder):
        """Tests loading a circuit with operations that can be converted,
         but not natively supported by PennyLane."""

        qc = QuantumCircuit(3, 1)

        qc.ch([0], [1])

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert recorder.queue[0].name == 'QubitUnitary'
        assert len(recorder.queue[0].params) == 1
        assert np.array_equal(recorder.queue[0].params[0], ex.CHGate().to_matrix())
        assert recorder.queue[0].wires == [0, 1]

    def test_quantum_circuit_error_by_passing_wrong_parameters(self, recorder):
        """Tests the load method for a QuantumCircuit raises a QiskitError,
        if the wrong type of arguments were passed."""

        theta = Parameter('θ')
        angle = 'some_string_instead_of_an_angle'

        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])

        quantum_circuit = load(qc)

        with pytest.raises(QiskitError):
            with recorder:
                quantum_circuit(params={theta: angle})

    def test_quantum_circuit_error_by_calling_wrong_parameters(self, recorder):
        """Tests that the load method for a QuantumCircuit raises a TypeError,
        if the wrong type of arguments were passed."""

        angle = 'some_string_instead_of_an_angle'

        qc = QuantumCircuit(3, 1)
        qc.rz(angle, [0])

        quantum_circuit = load(qc)

        with pytest.raises(TypeError, match="parameter expected, got <class 'str'>"):
            with recorder:
                quantum_circuit()

    def test_quantum_circuit_error_passing_parameters_not_required(self, recorder):
        """Tests the load method raises a QiskitError if arguments
        that are not required were passed."""

        theta = Parameter('θ')
        angle = 0.5

        qc = QuantumCircuit(3, 1)
        qc.z([0])

        quantum_circuit = load(qc)

        with pytest.raises(QiskitError):
            with recorder:
                quantum_circuit(params={theta: angle})

    def test_quantum_circuit_error_parameter_not_bound(self, recorder):
        """Tests the load method for a QuantumCircuit raises a ValueError,
        if one of the parameters was not bound correctly."""

        theta = Parameter('θ')

        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])

        quantum_circuit = load(qc)

        with pytest.raises(ValueError, match="The parameter {} was not bound correctly.".format(theta)):
            with recorder:
                quantum_circuit()

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

        with pytest.raises(qml.QuantumFunctionError, match='The specified number of wires - {} - does not match the'
                                                           ' number of wires the loaded quantum circuit acts on.'.
                format(len(only_two_wires))):
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

        with pytest.raises(qml.QuantumFunctionError, match="The specified number of wires - {} - does not match the"
                                                           " number of wires the loaded quantum circuit acts on.".
                format(len(more_than_three_wires))):
            with recorder:
                quantum_circuit(wires=more_than_three_wires)


class TestConverterUtils:
    """Tests the utility functions used by the converter function."""

    def test_map_wires(self, recorder):
        """Tests the map_wires function for wires of a quantum circuit."""

        wires = [0]
        qc = QuantumCircuit(1)
        qc_wires = [(q.register.name, q.index) for q in qc.qubits]

        assert map_wires(qc_wires, wires) == {0: ('q', 0)}

    def test_map_wires_instantiate_quantum_circuit_with_registers(self, recorder):
        """Tests the map_wires function for wires of a quantum circuit instantiated
        using quantum registers."""

        wires = [0, 1, 2]
        qr1 = QuantumRegister(1)
        qr2 = QuantumRegister(1)
        qr3 = QuantumRegister(1)
        qc = QuantumCircuit(qr1, qr2, qr3)
        qc_wires = [(q.register.name, q.index) for q in qc.qubits]

        mapped_wires = map_wires(qc_wires, wires)

        assert len(mapped_wires) == len(wires)
        assert list(mapped_wires.keys()) == wires
        for q in qc.qubits:
            assert (q.register.name, q.index) in mapped_wires.values()

    def test_map_wires_provided_non_standard_order(self, recorder):
        """Tests the map_wires function for wires of non-standard order."""

        wires = [1, 2, 0]
        qc = QuantumCircuit(3)
        qc_wires = [(q.register.name, q.index) for q in qc.qubits]

        mapped_wires = map_wires(qc_wires, wires)

        assert len(mapped_wires) == len(wires)
        assert set(mapped_wires.keys()) == set(wires)
        for q in qc.qubits:
            assert (q.register.name, q.index) in mapped_wires.values()
        assert mapped_wires[0][1] == 2
        assert mapped_wires[1][1] == 0
        assert mapped_wires[2][1] == 1

    def test_map_wires_exception_mismatch_in_number_of_wires(self, recorder):
        """Tests that the map_wires function raises an exception if there is a mismatch between
        wires."""

        wires = [0, 1, 2]
        qc = QuantumCircuit(1)
        qc_wires = [(q.register.name, q.index) for q in qc.qubits]

        with pytest.raises(qml.QuantumFunctionError, match='The specified number of wires - {} - does not match '
                                                           'the number of wires the loaded quantum circuit acts on.'
                .format(len(wires))):
            map_wires(wires, qc_wires)


class TestConverterWarnings:
    """Tests that the converter.load function emits warnings."""

    def test_barrier_not_supported(self, recorder):
        """Tests that a warning is raised if an unsupported instruction was reached."""
        qc = QuantumCircuit(3, 1)
        qc.barrier()

        quantum_circuit = load(qc)

        with pytest.warns(UserWarning) as record:
            with recorder:
                quantum_circuit(params={})

        # check that only one warning was raised
        assert len(record) == 1
        # check that the message matches
        assert record[0].message.args[0] == "pennylane_qiskit.converter: The Barrier instruction is not supported by" \
                                            " PennyLane, and has not been added to the template."


class TestConverterQasm:
    """Tests that the converter.load function allows conversion from qasm."""

    qft_qasm = 'OPENQASM 2.0;' \
               'include "qelib1.inc";' \
               'qreg q[4];' \
               'creg c[4];' \
               'x q[0]; ' \
               'x q[2];' \
               'barrier q;' \
               'h q[0];' \
               'cu1(pi/2) q[1],q[0];' \
               'h q[1];' \
               'cu1(pi/4) q[2],q[0];' \
               'cu1(pi/2) q[2],q[1];' \
               'h q[2];' \
               'cu1(pi/8) q[3],q[0];' \
               'cu1(pi/4) q[3],q[1];' \
               'cu1(pi/2) q[3],q[2];' \
               'h q[3];' \
               'measure q -> c;'

    @pytest.mark.skipif(sys.version_info < (3, 6), reason="tmpdir fixture requires Python >=3.6")
    def test_qasm_from_file(self, tmpdir, recorder):
        """Tests that a QuantumCircuit object is deserialized from a qasm file."""
        qft_qasm = tmpdir.join("qft.qasm")

        with open(qft_qasm, "w") as f:
            f.write(TestConverterQasm.qft_qasm)

        quantum_circuit = load_qasm_from_file(qft_qasm)

        with pytest.warns(UserWarning) as record:
            with recorder:
                quantum_circuit()

        assert len(recorder.queue) == 6

        assert recorder.queue[0].name == 'PauliX'
        assert recorder.queue[0].params == []
        assert recorder.queue[0].wires == [0]

        assert recorder.queue[1].name == 'PauliX'
        assert recorder.queue[1].params == []
        assert recorder.queue[1].wires == [2]

        assert recorder.queue[2].name == 'Hadamard'
        assert recorder.queue[2].params == []
        assert recorder.queue[2].wires == [0]

        assert recorder.queue[3].name == 'Hadamard'
        assert recorder.queue[3].params == []
        assert recorder.queue[3].wires == [1]

        assert recorder.queue[4].name == 'Hadamard'
        assert recorder.queue[4].params == []
        assert recorder.queue[4].wires == [2]

        assert recorder.queue[5].name == 'Hadamard'
        assert recorder.queue[5].params == []
        assert recorder.queue[5].wires == [3]

        assert len(record) == 11
        # check that the message matches
        assert record[0].message.args[0] == "pennylane_qiskit.converter: The {} instruction is not supported by" \
                                            " PennyLane, and has not been added to the template."\
            .format('Barrier')
        assert record[1].message.args[0] == "pennylane_qiskit.converter: The {} instruction is not supported by" \
                                            " PennyLane, and has not been added to the template."\
            .format('CU1Gate')
        assert record[7].message.args[0] == "pennylane_qiskit.converter: The {} instruction is not supported by" \
                                            " PennyLane, and has not been added to the template."\
            .format('Measure')

    def test_qasm_file_not_found_error(self):
        """Tests that an error is propagated, when a non-existing file is specified for parsing."""
        qft_qasm = 'some_qasm_file.qasm'

        with pytest.raises(FileNotFoundError):
            load_qasm_from_file(qft_qasm)

    def test_qasm_(self, recorder):
        """Tests that a QuantumCircuit object is deserialized from a qasm string."""
        qasm_string = 'include "qelib1.inc";' \
                      'qreg q[4];' \
                      'creg c[4];' \
                      'x q[0];' \
                      'cx q[2],q[0];'\
                      'measure q -> c;'

        quantum_circuit = load_qasm(qasm_string)

        with pytest.warns(UserWarning) as record:
            with recorder:
                quantum_circuit(params={})

        assert len(recorder.queue) == 2

        assert recorder.queue[0].name == 'PauliX'
        assert recorder.queue[0].params == []
        assert recorder.queue[0].wires == [0]

        assert recorder.queue[1].name == 'CNOT'
        assert recorder.queue[1].params == []
        assert recorder.queue[1].wires == [2, 0]


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

        theta = Parameter('θ')
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

        assert circuit_loaded_qiskit_circuit() == \
               circuit_native_pennylane()

    def test_passing_parameter_into_qnode(self, qubit_device_2_wires):
        """Tests passing a circuit parameter into the QNode."""

        theta = Parameter('θ')
        rotation_angle = 0.5

        qc = QuantumCircuit(2)
        qc.rz(theta, [0])

        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit(angle):
            load(qc)({theta: angle})
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qubit_device_2_wires)
        def circuit_native_pennylane(angle):
            qml.RZ(angle, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit(rotation_angle) == \
               circuit_native_pennylane(rotation_angle)

    def test_one_parameter_in_qc_one_passed_into_qnode(self, qubit_device_2_wires):
        """Tests passing a parameter by pre-defining it and then
         passing another to the QNode."""

        theta = Parameter('θ')
        phi = Parameter('φ')
        rotation_angle1 = 0.5
        rotation_angle2 = 0.3

        qc = QuantumCircuit(2)
        qc.rz(theta, [0])
        qc.rx(phi, [0])

        @qml.qnode(qubit_device_2_wires)
        def circuit_loaded_qiskit_circuit(angle):
            load(qc)({theta: angle, phi: rotation_angle2})
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qubit_device_2_wires)
        def circuit_native_pennylane(angle):
            qml.RZ(angle, wires=0)
            qml.RX(rotation_angle2, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit(rotation_angle1) == \
               circuit_native_pennylane(rotation_angle1)

    def test_initialize_with_qubit_state_vector(self, qubit_device_single_wire):
        """Tests the QuantumCircuit.initialize method in a QNode."""

        prob_amplitudes = [1/np.sqrt(2), 1/np.sqrt(2)]

        qreg = QuantumRegister(2)
        qc = QuantumCircuit(qreg)
        qc.initialize(prob_amplitudes, [qreg[0]])

        @qml.qnode(qubit_device_single_wire)
        def circuit_loaded_qiskit_circuit():
            load(qc)()
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qubit_device_single_wire)
        def circuit_native_pennylane():
            qml.QubitStateVector(np.array(prob_amplitudes), wires=[0])
            return qml.expval(qml.PauliZ(0))

        assert circuit_loaded_qiskit_circuit() == circuit_native_pennylane()

    @pytest.mark.parametrize("analytic", [True])
    @pytest.mark.parametrize("theta,phi,varphi", list(zip(THETA, PHI, VARPHI)))
    def test_gradient(self, theta, phi, varphi, analytic, tol):
        """Test that the gradient works correctly"""
        qc = QuantumCircuit(3)
        qiskit_params = [Parameter("param_{}".format(i)) for i in range(3)]

        qc.rx(qiskit_params[0], 0)
        qc.rx(qiskit_params[1], 1)
        qc.rx(qiskit_params[2], 2)
        qc.cnot(0, 1)
        qc.cnot(1, 2)

        # convert to a PennyLane circuit
        qc_pl = qml.from_qiskit(qc)

        dev = qml.device("default.qubit", wires=3, analytic=analytic)

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
            np.sin(theta) * np.sin(phi) * np.cos(varphi)
        ]

        assert np.allclose(res, expected, **tol)
