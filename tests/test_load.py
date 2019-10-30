from pennylane_qiskit.load import load
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info.operators import Operator
from qiskit.exceptions import QiskitError
from pennylane import numpy as np
import math

import pytest
import pennylane as qml


class TestLoad:
    """Test """

    def test_load_quantum_circuit_init_by_specifying_rotation_in_circuit(self, recorder):
        """Test the load method for a QuantumCircuit initialized using separately defined
         quandumt and classical registers"""

        angle = 0.5

        # quantum circuit to make an entangled bell state
        qc = QuantumCircuit(3, 1)
        qc.rz(angle, [0])

        quantum_circuit = load(qc)

        with recorder:
            quantum_circuit()

        assert len(recorder.queue) == 1
        assert recorder.queue[0].name == 'RZ'
        assert recorder.queue[0].params == [angle]
        assert recorder.queue[0].wires == [0]

    def test_load_quantum_circuit_by_passing_parameters(self, recorder):
        """Test the load method for a QuantumCircuit initialized using separately defined
         quandumt and classical registers"""

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

    def test_load_loaded_quantum_circuit_and_further_pennylane_operations(self, recorder):
        """Test the load method for a QuantumCircuit initialized using separately defined
         quandumt and classical registers"""

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

    def test_load_quantum_circuit_loaded_multiple_times_with_different_parameters(self, recorder):
        """Test the load method for a QuantumCircuit initialized using separately defined
         quandumt and classical registers"""

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

    def test_load_quantum_circuit_with_bound_parameters(self, recorder):

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

    def test_load_pass_parameters_to_bind(self, recorder):

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

    def test_load_parameter_was_not_bound(self, recorder):

        theta = Parameter('θ')
        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])

        quantum_circuit = load(qc)

        with pytest.raises(ValueError, match='The parameter {} was not bound correctly.'.format(theta)):
            with recorder:
                quantum_circuit(params={})

    def test_load_no_parameters_passed(self, recorder):

        theta = Parameter('θ')
        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])

        quantum_circuit = load(qc)

        with pytest.raises(ValueError, match='The parameter {} was not bound correctly'.format(theta)):
            with recorder:
                quantum_circuit(params={})

    def test_load_extra_parameters_were_passed(self, recorder):

        theta = Parameter('θ')
        phi = Parameter('φ')

        qc = QuantumCircuit(3, 1)

        quantum_circuit = load(qc)

        with pytest.raises(QiskitError):
            with recorder:
                quantum_circuit(params={theta: 0.5, phi: 0.3})

    def test_load_crz(self, recorder):

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

    def test_load_one_qubit_operations_supported_by_pennylane(self, recorder):

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

    def test_load_one_qubit_parametrized_operations_supported_by_pennylane(self, recorder):

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

    def test_load_two_qubit_operations_supported_by_pennylane(self, recorder):

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

    def test_load_two_qubit_parametrized_operations_supported_by_pennylane(self, recorder):

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

    def test_load_three_qubit_operations_supported_by_pennylane(self, recorder):

        three_wires = [0, 1, 2]

        qc = QuantumCircuit(3, 1)

        qc.cswap(*three_wires)

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert recorder.queue[0].name == 'CSWAP'
        assert recorder.queue[0].params == []
        assert recorder.queue[0].wires == three_wires

    def test_load_quantum_circuit_error_by_passing_wrong_parameters(self, recorder):
        """Test the load method for a QuantumCircuit raises a QiskitError,
        if the wrong type of arguments were passed"""

        theta = Parameter('θ')
        angle = 'some_string_instead_an_angle'

        # quantum circuit to make an entangled bell state
        qc = QuantumCircuit(3, 1)
        qc.rz(theta, [0])

        quantum_circuit = load(qc)

        with pytest.raises(QiskitError):
            with recorder:
                quantum_circuit(params={theta: angle})

    def test_load_quantum_circuit_error_by_calling_wrong_parameters(self, recorder):
        """Test that the load method for a QuantumCircuit raises a TypeError,
        if the wrong type of arguments were passed"""

        angle = 'some_string_instead_an_angle'

        # quantum circuit to make an entangled bell state
        qc = QuantumCircuit(3, 1)
        qc.rz(angle, [0])

        quantum_circuit = load(qc)

        with pytest.raises(TypeError, match="can't convert expression to float"):
            with recorder:
                quantum_circuit()

    def test_load_quantum_circuit_error_passing_parameters_not_required(self, recorder):
        """Test the load method raises a QiskitError, if the arguments, that are not required were passed"""

        theta = Parameter('θ')
        angle = 0.5

        # quantum circuit to make an entangled bell state
        qc = QuantumCircuit(3, 1)
        qc.z([0])

        quantum_circuit = load(qc)

        with pytest.raises(QiskitError):
            with recorder:
                quantum_circuit(params={theta: angle})

    def test_load_quantum_circuit_error_not_qiskit_circuit_passed(self, recorder):

        qc = qml.PauliZ(0)

        quantum_circuit = load(qc)

        with pytest.raises(ValueError):
            with recorder:
                quantum_circuit()



class TestLoadWarnings:

    def test_load_not_supported_gates(self, recorder):

        angle = 0.333

        qc = QuantumCircuit(3, 1)
        qc.rzz(angle, qc.qubits[0], qc.qubits[1])

        # Performing the Identity
        qc.iden(qc.qubits[0])
        qc.ch(0, 1)
        qc.ccx(0, 1, 2)
        qc.barrier()

        quantum_circuit = load(qc)

        with pytest.warns(UserWarning) as record:
            with recorder:
                quantum_circuit(params={})

        # check that only one warning was raised
        assert len(record) == 5
        # check that the message matches
        assert record[0].message.args[0] == "pennylane_qiskit.load The {} instruction is not supported by PennyLane."\
            .format('RZZGate')
        assert record[1].message.args[0] == "pennylane_qiskit.load The {} instruction is not supported by PennyLane."\
            .format('IdGate')
        assert record[2].message.args[0] == "pennylane_qiskit.load The {} instruction is not supported by PennyLane."\
            .format('CHGate')
        assert record[3].message.args[0] == "pennylane_qiskit.load The {} instruction is not supported by PennyLane." \
            .format('ToffoliGate')
        assert record[4].message.args[0] == "pennylane_qiskit.load The {} instruction is not supported by PennyLane." \
            .format('Barrier')
