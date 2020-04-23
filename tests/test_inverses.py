import pytest

import pennylane as qml
import math
import cmath
import numpy as np

# defaults
tol = 1e-5


class TestInverses:
    """Tests that the inverse of the operations are applied."""

    # This test is ran against the state |0> with one Z expval
    @pytest.mark.parametrize("name,expected_output", [
        ("PauliX", -1),
        ("PauliY", -1),
        ("PauliZ", 1),
        ("Hadamard", 0),
        ("S", 1),
        ("T", 1),
    ])
    def test_supported_gate_inverse_single_wire_no_parameters(self, name, expected_output):
        """Tests the inverse of supported gates that act on a single wire that are not
        parameterized"""

        op = getattr(qml.ops, name)

        dev = qml.device('qiskit.aer', backend='statevector_simulator', wires=2, analytic=True)

        @qml.qnode(dev)
        def circuit():
            op(wires=0).inv()
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran against the state |Phi+> with two Z expvals
    @pytest.mark.parametrize("name,expected_output", [
        ("CNOT", [-1/2, 1]),
        ("SWAP", [-1/2, -1/2]),
        ("CZ", [-1/2, -1/2]),
    ])
    def test_supported_gate_inverse_two_wires_no_parameters(self, name, expected_output):
        """Tests the inverse of supported gates that act on two wires that are not parameterized"""

        op = getattr(qml.ops, name)

        dev = qml.device('qiskit.aer', backend='statevector_simulator', wires=2, analytic=True)

        assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(np.array([1/2, 0, 0, math.sqrt(3)/2]), wires=[0, 1])
            op(wires=[0, 1]).inv()
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,expected_output", [
        ("CSWAP", [-1, -1, 1]),
    ])
    def test_supported_gate_inverse_three_wires_no_parameters(self, name, expected_output):
        """Tests the inverse of supported gates that act on three wires that are not parameterized"""

        op = getattr(qml.ops, name)

        dev = qml.device('qiskit.aer', backend='statevector_simulator', wires=3, analytic=True)

        assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2])
            op(wires=[0, 1, 2]).inv()
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran on the state |0> with one Z expvals
    @pytest.mark.parametrize("name,par,expected_output", [
        ("PhaseShift", [math.pi/2], 1),
        ("PhaseShift", [-math.pi/4], 1),
        ("RX", [math.pi/2], 0),
        ("RX", [-math.pi/4], 1/math.sqrt(2)),
        ("RY", [math.pi/2], 0),
        ("RY", [-math.pi/4], 1/math.sqrt(2)),
        ("RZ", [math.pi/2], 1),
        ("RZ", [-math.pi/4], 1),
        ("QubitUnitary", [np.array([[1j/math.sqrt(2), 1j/math.sqrt(2)], [1j/math.sqrt(2), -1j/math.sqrt(2)]])], 0),
        ("QubitUnitary", [np.array([[-1j/math.sqrt(2), 1j/math.sqrt(2)], [1j/math.sqrt(2), 1j/math.sqrt(2)]])], 0),
    ])
    def test_supported_gate_inverse_single_wire_with_parameters(self, name, par, expected_output):
        """Test the inverse of single gates with parameters"""

        dev = qml.device('qiskit.aer', backend='statevector_simulator', wires=2, analytic=True)

        op = getattr(qml.ops, name)

        assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            op(*np.negative(par), wires=0).inv()
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran against the state 1/2|00>+sqrt(3)/2|11> with two Z expvals
    @pytest.mark.parametrize("name,par,expected_output", [
        ("CRZ", [0], [-1/2, -1/2]),
        ("CRZ", [-math.pi], [-1/2, -1/2]),
        ("CRZ", [math.pi/2], [-1/2, -1/2]),
        ("QubitUnitary", [np.array([[1, 0, 0, 0], [0, 1/math.sqrt(2), 1/math.sqrt(2), 0], [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], [0, 0, 0, 1]])], [-1/2, -1/2]),
        ("QubitUnitary", [np.array([[-1, 0, 0, 0], [0, 1/math.sqrt(2), 1/math.sqrt(2), 0], [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], [0, 0, 0, -1]])], [-1/2, -1/2]),
    ])
    def test_supported_gate_inverse_two_wires_with_parameters(self, name, par, expected_output):
        """Tests the inverse of supported gates that act on two wires that are parameterized"""

        dev = qml.device('qiskit.aer', backend='statevector_simulator', wires=2, analytic=True)

        op = getattr(qml.ops, name)

        assert dev.supports_operation(name)

        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(np.array([1/2, 0, 0, math.sqrt(3)/2]), wires=[0, 1])
            op(*np.negative(par), wires=[0, 1]).inv()
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,par,expected_output", [
        ("Rot", [math.pi/2, 0, 0], 1),
        ("Rot", [0, math.pi/2, 0], 0),
        ("Rot", [0, 0, math.pi/2], 1),
        ("Rot", [math.pi/2, -math.pi/4, -math.pi/4], 1/math.sqrt(2)),
        ("Rot", [-math.pi/4, math.pi/2, math.pi/4], 0),
        ("Rot", [-math.pi/4, math.pi/4, math.pi/2], 1/math.sqrt(2)),
    ])
    def test_unsupported_gate_inverses(self, name, par, expected_output):
        """Test the inverse of single gates with parameters"""

        dev = qml.device('qiskit.aer', backend='statevector_simulator', wires=2, analytic=True)

        op = getattr(qml.ops, name)

        @qml.qnode(dev)
        def circuit():
            op(*np.negative(par), wires=0).inv()
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("par", [np.pi/i for i in range(1, 5)])
    def test_s_gate_inverses(self, par):
        """Tests the inverse of the S gate"""

        dev = qml.device('qiskit.aer', backend='statevector_simulator', wires=2, analytic=True)

        expected_output = -0.5 * 1j * cmath.exp(-1j*par)*(-1 + cmath.exp(2j*par))

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.RZ(par, wires=[0])
            qml.S(wires=[0]).inv()
            return qml.expval(qml.PauliX(0))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("par", [np.pi/i for i in range(1, 5)])
    def test_t_gate_inverses(self, par):
        """Tests the inverse of the T gate"""

        dev = qml.device('qiskit.aer', backend='statevector_simulator', wires=2, analytic=True)

        expected_output = -math.sin(par) / math.sqrt(2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(par, wires=[0])
            qml.T(wires=[0]).inv()
            return qml.expval(qml.PauliX(0))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)
