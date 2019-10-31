import math
import sys
import pytest

from pennylane_qiskit.converter import load, load_qasm_from_file, load_qasm
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info.operators import Operator
from qiskit.exceptions import QiskitError
from qiskit import extensions as ex
from pennylane import numpy as np
import pennylane as qml


class TestConverter:
    """Tests the converter function that allows converting QuantumCircuit objects
     to Pennylane templates."""

    def test_quantum_circuit_init_by_specifying_rotation_in_circuit(self, recorder):
        """Tests the load method for a QuantumCircuit initialized using separately defined
         quantum and classical registers."""

        angle = 0.5

        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)

        # quantum circuit to make an entangled bell state
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

        # quantum circuit to make an entangled bell state
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

        # quantum circuit to make an entangled bell state
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

    def test_quantum_circuit_loaded_multiple_times_with_different_arguments(self, recorder):
        """Tests that a loaded quantum circuit can be called multiple times with
        different arguments."""

        theta = Parameter('θ')
        angle1 = 0.5
        angle2 = -0.5
        angle3 = 0

        # quantum circuit to make an entangled bell state
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

    def test_extra_parameters_were_passed(self, recorder):
        """Tests that loading raises an error when extra parameters were passed."""

        theta = Parameter('θ')
        phi = Parameter('φ')

        qc = QuantumCircuit(3, 1)

        quantum_circuit = load(qc)

        with pytest.raises(QiskitError):
            with recorder:
                quantum_circuit(params={theta: 0.5, phi: 0.3})

    def test_crz(self, recorder):
        """Tests loading a circuit with the controlled-Z operation."""

        q2 = QuantumRegister(2)
        qc = QuantumCircuit(q2)
        qc.crz(0.5, q2[0], q2[1])

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit()

        assert len(recorder.queue) == 1
        assert recorder.queue[0].name == 'CRZ'
        assert recorder.queue[0].params == [0.5]
        assert recorder.queue[0].wires == [0, 1]

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
        prob_amplitudes = [1/np.sqrt(2), 1/np.sqrt(2)]

        q_reg = QuantumRegister(1)
        qc = QuantumCircuit(q_reg)

        qc.initialize(prob_amplitudes, [q_reg[0]])
        qc.rx(angle, single_wire)
        qc.ry(angle, single_wire)
        qc.rz(angle, single_wire)
        qc.u1(angle, single_wire)

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert recorder.queue[0].name == 'QubitStateVector'
        assert len(recorder.queue[0].params) == 1
        assert np.array_equal(recorder.queue[0].params[0], np.array(prob_amplitudes))
        assert recorder.queue[0].wires == list(range(int(math.log2(len(prob_amplitudes)))))

        assert recorder.queue[1].name == 'RX'
        assert recorder.queue[1].params == [angle]
        assert recorder.queue[1].wires == single_wire

        assert recorder.queue[2].name == 'RY'
        assert recorder.queue[2].params == [angle]
        assert recorder.queue[2].wires == single_wire

        assert recorder.queue[3].name == 'RZ'
        assert recorder.queue[3].params == [angle]
        assert recorder.queue[3].wires == single_wire

        assert recorder.queue[4].name == 'PhaseShift'
        assert recorder.queue[4].params == [angle]
        assert recorder.queue[4].wires == single_wire

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

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert recorder.queue[0].name == 'CSWAP'
        assert recorder.queue[0].params == []
        assert recorder.queue[0].wires == three_wires

    def test_operations_transformed_into_qubit_unitary(self, recorder):
        """Tests loading a circuit with operations that can be converted,
         but not natively supported by PennyLane."""

        qc = QuantumCircuit(3, 1)

        phi = 0.3
        lam = 0.4
        theta = 0.2

        qc.sdg([0])
        qc.tdg([0])

        qc.u2(phi, lam, [0])
        qc.u3(phi, lam, theta, [0])
        qc.ccx(0, 1, 2)

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert recorder.queue[0].name == 'QubitUnitary'
        assert len(recorder.queue[0].params) == 1
        assert np.array_equal(recorder.queue[0].params[0], ex.SdgGate().to_matrix())
        assert recorder.queue[0].wires == [0]

        assert recorder.queue[1].name == 'QubitUnitary'
        assert len(recorder.queue[1].params) == 1
        assert np.array_equal(recorder.queue[1].params[0], ex.TdgGate().to_matrix())
        assert recorder.queue[1].wires == [0]

        assert recorder.queue[2].name == 'QubitUnitary'
        assert len(recorder.queue[2].params) == 1
        assert np.array_equal(recorder.queue[2].params[0], ex.U2Gate(phi, lam).to_matrix())
        assert recorder.queue[2].wires == [0]

        assert recorder.queue[3].name == 'QubitUnitary'
        assert len(recorder.queue[3].params) == 1
        assert np.array_equal(recorder.queue[3].params[0], ex.U3Gate(phi, lam, theta).to_matrix())
        assert recorder.queue[3].wires == [0]

        assert recorder.queue[4].name == 'QubitUnitary'
        assert len(recorder.queue[4].params) == 1
        assert np.array_equal(recorder.queue[4].params[0], ex.ToffoliGate().to_matrix())
        assert recorder.queue[4].wires == [0, 1, 2]

    def test_quantum_circuit_error_by_passing_wrong_parameters(self, recorder):
        """Tests the load method for a QuantumCircuit raises a QiskitError,
        if the wrong type of arguments were passed."""

        theta = Parameter('θ')
        angle = 'some_string_instead_an_angle'

        # quantum circuit to make an entangled bell state
        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])

        quantum_circuit = load(qc)

        with pytest.raises(QiskitError):
            with recorder:
                quantum_circuit(params={theta: angle})

    def test_quantum_circuit_error_by_calling_wrong_parameters(self, recorder):
        """Tests that the load method for a QuantumCircuit raises a TypeError,
        if the wrong type of arguments were passed."""

        angle = 'some_string_instead_an_angle'

        # quantum circuit to make an entangled bell state
        qc = QuantumCircuit(3, 1)
        qc.rz(angle, [0])

        quantum_circuit = load(qc)

        with pytest.raises(TypeError, match="can't convert expression to float"):
            with recorder:
                quantum_circuit()

    def test_quantum_circuit_error_passing_parameters_not_required(self, recorder):
        """Tests the load method raises a QiskitError, if the arguments,
        that are not required were passed."""

        theta = Parameter('θ')
        angle = 0.5

        # quantum circuit to make an entangled bell state
        qc = QuantumCircuit(3, 1)
        qc.z([0])

        quantum_circuit = load(qc)

        with pytest.raises(QiskitError):
            with recorder:
                quantum_circuit(params={theta: angle})

    def test_quantum_circuit_error_not_qiskit_circuit_passed(self, recorder):
        """Tests the load method raises a ValueError, if not a QuanctumCircuit was passed."""

        qc = qml.PauliZ(0)

        quantum_circuit = load(qc)

        with pytest.raises(ValueError):
            with recorder:
                quantum_circuit()


class TestConverterWarnings:
    """Tests that the converter.load function emits warnings."""

    def test_barrier_not_supported(self, recorder):
        """Tests the load method raises a ValueError, if not a QuanctumCircuit was passed."""
        qc = QuantumCircuit(3, 1)
        qc.barrier()

        quantum_circuit = load(qc)

        with pytest.warns(UserWarning) as record:
            with recorder:
                quantum_circuit(params={})

        # check that only one warning was raised
        assert len(record) == 1
        # check that the message matches
        assert record[0].message.args[0] == "pennylane_qiskit.converter The {} instruction is not supported by" \
                                            " PennyLane, and has not been added to the template." \
            .format('Barrier')


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
        assert record[0].message.args[0] == "pennylane_qiskit.converter The {} instruction is not supported by" \
                                            " PennyLane, and has not been added to the template."\
            .format('Barrier')
        assert record[1].message.args[0] == "pennylane_qiskit.converter The {} instruction is not supported by" \
                                            " PennyLane, and has not been added to the template."\
            .format('Cu1Gate')
        assert record[7].message.args[0] == "pennylane_qiskit.converter The {} instruction is not supported by" \
                                            " PennyLane, and has not been added to the template."\
            .format('Measure')

    def test_qasm_(self, recorder):
        """Tests that a QuantumCircuit object is deserialized from a qasm string."""
        qasm_string = 'include "qelib1.inc";' \
                      'qreg q[4];' \
                      'creg c[4];' \
                      'x q[0];' \
                      'x q[2];'\
                      'measure q -> c;'

        quantum_circuit = load_qasm(qasm_string)

        with pytest.warns(UserWarning) as record:
            with recorder:
                quantum_circuit(params={})

        assert len(recorder.queue) == 2

        assert recorder.queue[0].name == 'PauliX'
        assert recorder.queue[0].params == []
        assert recorder.queue[0].wires == [0]

        assert recorder.queue[1].name == 'PauliX'
        assert recorder.queue[1].params == []
        assert recorder.queue[1].wires == [2]
