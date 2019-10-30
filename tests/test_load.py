from pennylane_qiskit.load import load

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.exceptions import QiskitError
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import layers
import math

import pytest
import pennylane as qml
from pennylane import Device, DeviceError
from pennylane.operation import Sample, Variance, Expectation
from pennylane.qnode import QuantumFunctionError

import pennylane_qiskit.load

# quantum_circuit_instructions =


@pytest.fixture(scope="function")
def mock_device_with_operations(monkeypatch):
    """A mock instance of the abstract Device class with non-empty operations"""
    with monkeypatch.context() as m:
        m.setattr(Device, '__abstractmethods__', frozenset())
        m.setattr(Device, 'operations', ["PauliX", "PauliZ", "CNOT"])
        yield Device()


class TestLoad():
    """Test """

    def test_load_quantum_circuit_init_by_passing_arguments(self):
        """Test that eigs(Z) = [1, -1]"""

        theta = Parameter('θ')

        # quantum circuit to make an entangled bell state
        qc = QuantumCircuit(5, 1)

        qc.h(0)
        qc.rz(theta, [0])
        qc.cx(0, 1)
        quantum_circuit = load(qc)

        # TODO:
        # Create QuantumCircuit with Quantum and classical registers

        dev = qml.device('default.qubit', wires=2)

        x = np.array([0.53896774, 0.79503606, 0.27826503, 0.])  # n qubits to encode 2^n features

        # weights = np.array([[0.5, 0.4, 0.3], [0.5, 0.4, 0.3]])

        @qml.qnode(dev)
        def qiskit_loaded_quantum_circuit():
            quantum_circuit(start_wire_index=0, params={theta: 0.5})
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def pennylane_quantum_circuit():
            qml.Hadamard(0)
            qml.RZ(0.5, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        assert qiskit_loaded_quantum_circuit == pennylane_quantum_circuit

    def test_load_quantum_circuit_init_by_passing_arguments(self):
        """Test that eigs(Z) = [1, -1]"""

        theta = Parameter('θ')

        n = 5

        q2 = QuantumRegister(2)
        c1 = ClassicalRegister(1)
        c2 = ClassicalRegister(2)

        # quantum circuit to make an entangled bell state
        qc = QuantumCircuit(q2, c1)

        qc.h(0)
        qc.rz(theta, [0])
        qc.cx(0, 1)

        qc.barrier()
        qc.measure(0, 0)
        quantum_circuit = load(qc)

        # TODO:
        # Create QuantumCircuit with Quantum and classical registers

        dev = qml.device('default.qubit', wires=2)

        x = np.array([0.53896774, 0.79503606, 0.27826503, 0.])  # n qubits to encode 2^n features

        # weights = np.array([[0.5, 0.4, 0.3], [0.5, 0.4, 0.3]])

        @qml.qnode(dev)
        def load_quantum_circuit(x):
            qml.QubitStateVector(x, wires=[0, 1])
            quantum_circuit(start_wire_index=0, params={theta: 0.5})
            # qml.Hadamard(0)
            # qml.RZ(0.5, wires=0)
            # qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        print(load_quantum_circuit(x))


class TestLoadWarnings:

    def test_load_warns_barrier(self, recorder):
        # quantum circuit to make an entangled bell state
        qc = QuantumCircuit(5, 1)

        qc.barrier()

        # instruction = getattr(QuantumCircuit, instruction_name)

        quantum_circuit = load(qc)
        with pytest.warns(UserWarning, match=pennylane_qiskit.load.__name__ +
                                             " The {} instruction is not supported by PennyLane.".
                                                     format('Barrier')):
            with recorder:
                quantum_circuit()

    def test_load_quantum_circuit_with_binding_parameters(self, recorder):

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

    def test_load_toffoli(self, recorder):

        qc = QuantumCircuit(3, 1)
        qc.ccx(0, 1, 2)

        quantum_circuit = load(qc)
        with pytest.warns(UserWarning, match=pennylane_qiskit.load.__name__ +
                                             " The {} instruction is not supported by PennyLane.".
                                                     format('ToffoliGate')):
            with recorder:
                quantum_circuit()

    def test_load_ch(self, recorder):

        qc = QuantumCircuit(3, 1)
        qc.ch(0, 1)

        quantum_circuit = load(qc)
        with pytest.warns(UserWarning, match=pennylane_qiskit.load.__name__ +
                                             " The {} instruction is not supported by PennyLane.".
                                                     format('CHGate')):
            with recorder:
                quantum_circuit()

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
        # prob_amplitudes = [1/np.sqrt(2) * complex(1), 1/np.sqrt(2) * complex(1)]
        prob_amplitudes = [1 * complex(1), 0]

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

        assert len(recorder.queue) == 6

        assert recorder.queue[0].name == 'QubitStateVector'
        assert recorder.queue[0].params == prob_amplitudes
        assert recorder.queue[0].wires == list(range(math.log2(len(prob_amplitudes))))

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

        two_wires =  [0, 1]

        qc = QuantumCircuit(2, 1)

        qc.cx(*two_wires)
        qc.cz(*two_wires)
        qc.swap(*two_wires)

        quantum_circuit = load(qc)
        with recorder:
            quantum_circuit()

        assert len(recorder.queue) == 3

        assert recorder.queue[0].name == 'CNOT'
        assert recorder.queue[0].params == []
        assert recorder.queue[0].wires == two_wires

        assert recorder.queue[1].name == 'CZ'
        assert recorder.queue[1].params == []
        assert recorder.queue[1].wires == two_wires

        assert recorder.queue[2].name == 'SWAP'
        assert recorder.queue[2].params == []
        assert recorder.queue[2].wires == two_wires

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
