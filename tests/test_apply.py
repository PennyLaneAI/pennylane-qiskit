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
This module contains tests for applying operations on PennyLane IBMQ devices.
"""
import pytest

import numpy as np
import pennylane as qml
from scipy.linalg import block_diag

from conftest import U, U2

# pylint: disable=protected-access, too-many-arguments, too-few-public-methods

np.random.seed(42)

# global variables and rotations
I = np.identity(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CZ = np.diag([1, 1, 1, -1])
toffoli = np.diag([1 for i in range(8)])
toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])
CSWAP = block_diag(I, I, SWAP)

# pylint: disable=unnecessary-lambda-assignment
phase_shift = lambda phi: np.array([[1, 0], [0, np.exp(1j * phi)]])
rx = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * X
ry = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Y
rz = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Z
rot = lambda a, b, c: rz(c) @ (ry(b) @ rz(a))
crz = lambda theta: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.exp(-1j * theta / 2), 0],
        [0, 0, 0, np.exp(1j * theta / 2)],
    ]
)

isingxx = lambda phi: np.array(
    [
        [np.cos(phi / 2), 0, 0, -1j * np.sin(phi / 2)],
        [0, np.cos(phi / 2), -1j * np.sin(phi / 2), 0],
        [0, -1j * np.sin(phi / 2), np.cos(phi / 2), 0],
        [-1j * np.sin(phi / 2), 0, 0, np.cos(phi / 2)],
    ]
)

isingyy = lambda phi: np.array(
    [
        [np.cos(phi / 2), 0, 0, 1j * np.sin(phi / 2)],
        [0, np.cos(phi / 2), -1j * np.sin(phi / 2), 0],
        [0, -1j * np.sin(phi / 2), np.cos(phi / 2), 0],
        [1j * np.sin(phi / 2), 0, 0, np.cos(phi / 2)],
    ]
)

isingzz = lambda phi: np.array(
    [
        [np.exp(-1.0j * phi / 2), 0, 0, 0],
        [0, np.exp(1.0j * phi / 2), 0, 0],
        [0, 0, np.exp(1.0j * phi / 2), 0],
        [0, 0, 0, np.exp(-1.0j * phi / 2)],
    ]
)


single_qubit_operations = [
    qml.Identity,
    qml.PauliX,
    qml.PauliY,
    qml.PauliZ,
    qml.Hadamard,
    qml.S,
    qml.T,
    qml.SX,
    qml.adjoint(qml.T),
    qml.adjoint(qml.S),
    qml.adjoint(qml.SX),
]

single_qubit_operations_param = [qml.PhaseShift, qml.RX, qml.RY, qml.RZ]
two_qubit = [qml.CNOT, qml.SWAP, qml.CZ, qml.ISWAP]
two_qubit_param = [qml.CRZ, qml.IsingXX, qml.IsingYY, qml.IsingZZ]
three_qubit = [qml.Toffoli, qml.CSWAP]


@pytest.mark.parametrize("shots", [None])
@pytest.mark.usefixtures("skip_unitary")
class TestAnalyticApply:
    """Test application of PennyLane operations with analytic calculation."""

    def test_qubit_state_vector(self, init_state, device, tol):
        """Test that the StatePrep operation produces the expected
        result with the apply method."""
        dev = device(1)
        state = init_state(1)

        dev.apply([qml.StatePrep(state, wires=[0])])

        res = np.abs(dev.state) ** 2
        expected = np.abs(state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("operation", single_qubit_operations)
    def test_single_qubit_operations_no_parameters(self, init_state, device, operation, tol):
        """Test that single qubit operations that take no parameters work fine
        with the apply method."""
        dev = device(1)
        state = init_state(1)
        applied_operation = operation(wires=[0])

        dev.apply([qml.StatePrep(state, wires=[0]), applied_operation])

        res = np.abs(dev.state) ** 2

        expected = np.abs(applied_operation.matrix() @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("operation", single_qubit_operations_param)
    def test_single_qubit_operations_parameters(self, init_state, device, operation, theta, tol):
        """Test that single qubit parametrized operations work fine with the
        apply method."""
        dev = device(1)
        state = init_state(1)
        applied_operation = operation(theta, wires=[0])

        dev.apply([qml.StatePrep(state, wires=[0]), applied_operation])

        res = np.abs(dev.state) ** 2
        expected = np.abs(applied_operation.matrix() @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("operation", two_qubit)
    def test_two_qubit_operations_no_parameters(self, init_state, device, operation, tol):
        """Test that two qubit operations that take no parameters work fine
        with the apply method."""
        dev = device(2)
        state = init_state(2)
        wires = [0, 1]

        applied_operation = operation(wires=wires)

        dev.apply([qml.StatePrep(state, wires=wires), applied_operation])

        res = np.abs(dev.state) ** 2
        expected = np.abs(applied_operation.matrix() @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("operation", two_qubit_param)
    def test_two_qubit_operations_parameters(self, init_state, device, operation, theta, tol):
        """Test that two qubit parametrized operations work fine with the
        apply method."""
        dev = device(2)
        state = init_state(2)
        wires = [0, 1]
        applied_operation = operation(theta, wires=wires)

        dev.apply([qml.StatePrep(state, wires=wires), applied_operation])

        res = np.abs(dev.state) ** 2
        expected = np.abs(applied_operation.matrix() @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("operation", three_qubit)
    def test_three_qubit_operations_no_parameters(self, init_state, device, operation, tol):
        """Test that three qubit operations that take no parameters work fine
        with the apply method."""
        dev = device(3)
        state = init_state(3)
        applied_operation = operation(wires=[0, 1, 2])

        dev.apply([qml.StatePrep(state, wires=[0, 1, 2]), applied_operation])

        res = np.abs(dev.state) ** 2
        expected = np.abs(applied_operation.matrix() @ state) ** 2
        assert np.allclose(res, expected, **tol)


@pytest.mark.parametrize("shots", [None])
@pytest.mark.usefixtures("run_only_for_unitary")
class TestStateApplyUnitarySimulator:
    """Test application of PennyLane operations to the unitary simulator."""

    def test_invalid_qubit(self, init_state, device):
        """Test that an exception is raised if the
        unitary matrix is applied on a unitary simulator."""
        dev = device(1)
        state = init_state(1)

        with pytest.raises(
            qml.DeviceError,
            match="The StatePrep operation is not supported on the unitary simulator backend",
        ):
            dev.apply([qml.StatePrep(state, wires=[0])])


@pytest.mark.parametrize("shots", [8192])
@pytest.mark.usefixtures("skip_unitary")
class TestNonAnalyticApply:
    """Test application of PennyLane operations with non-analytic calculation."""

    def test_qubit_state_vector(self, init_state, device, tol):
        """Test that the StatePrep operation produces the expected
        result with the apply method."""

        dev = device(1)
        state = init_state(1)
        wires = [0]

        dev.apply([qml.StatePrep(state, wires=wires)])
        dev._samples = dev.generate_samples()

        res = np.fromiter(dev.probability(), dtype=np.float64)
        expected = np.abs(state) ** 2
        assert np.allclose(res, expected, **tol)

    def test_invalid_qubit_state_vector(self, device):
        """Test that an exception is raised if the state
        vector is the wrong size"""
        dev = device(2)
        state = np.array([0, 123.432])
        wires = [0, 1]

        with pytest.raises(ValueError, match=r"State must be of length 4"):
            dev.apply([qml.StatePrep(state, wires=wires)])

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary(self, init_state, device, mat, tol):
        """Test that the QubitUnitary operation produces the expected result
        with the apply method."""
        N = int(np.log2(len(mat)))
        dev = device(N)
        state = init_state(N)
        wires = list(range(N))

        dev.apply([qml.StatePrep(state, wires=wires), qml.QubitUnitary(mat, wires=wires)])
        dev._samples = dev.generate_samples()

        res = np.fromiter(dev.probability(), dtype=np.float64)
        expected = np.abs(mat @ state) ** 2
        assert np.allclose(res, expected, **tol)

    def test_invalid_qubit_unitary(self, device):
        """Test that an exception is raised if the
        unitary matrix is the wrong size"""
        dev = device(2)
        state = np.array([[0, 123.432], [-0.432, 023.4]])

        with pytest.raises(ValueError, match=r"Input unitary must be of shape"):
            dev.apply([qml.QubitUnitary(state, wires=[0, 1])])

    @pytest.mark.parametrize("operation", single_qubit_operations)
    def test_single_qubit_operations_no_parameters(self, init_state, device, operation, tol):
        """Test that single qubit operations that take no parameters work fine
        with the apply method."""
        dev = device(1)
        state = init_state(1)
        wires = [0]
        applied_operation = operation(wires=wires)

        dev.apply([qml.StatePrep(state, wires=wires), applied_operation])
        dev._samples = dev.generate_samples()

        res = np.fromiter(dev.probability(), dtype=np.float64)
        expected = np.abs(applied_operation.matrix() @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("operation", single_qubit_operations_param)
    def test_single_qubit_operations_parameters(self, init_state, device, operation, theta, tol):
        """Test that single qubit parametrized operations work fine with the
        apply method."""
        dev = device(1)
        state = init_state(1)
        wires = [0]
        applied_operation = operation(theta, wires=wires)

        dev.apply([qml.StatePrep(state, wires=wires), applied_operation])
        dev._samples = dev.generate_samples()

        res = np.fromiter(dev.probability(), dtype=np.float64)
        expected = np.abs(applied_operation.matrix() @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("operation", two_qubit)
    def test_two_qubit_no_parameters(self, init_state, device, operation, tol):
        """Test that two qubit operations that take no parameters work fine
        with the apply method."""
        dev = device(2)
        state = init_state(2)
        wires = [0, 1]

        applied_operation = operation(wires=wires)

        dev.apply([qml.StatePrep(state, wires=wires), applied_operation])
        dev._samples = dev.generate_samples()

        res = np.fromiter(dev.probability(), dtype=np.float64)
        expected = np.abs(applied_operation.matrix() @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("operation", two_qubit_param)
    def test_two_qubit_operations_parameters(self, init_state, device, operation, theta, tol):
        """Test that two qubit parametrized operations work fine with the
        apply method."""
        dev = device(2)
        state = init_state(2)
        wires = [0, 1]

        applied_operation = operation(theta, wires=wires)

        dev.apply([qml.StatePrep(state, wires=wires), applied_operation])
        dev._samples = dev.generate_samples()

        res = np.fromiter(dev.probability(), dtype=np.float64)
        expected = np.abs(applied_operation.matrix() @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("operation", three_qubit)
    def test_three_qubit_no_parameters(self, init_state, device, operation, tol):
        """Test that three qubit operations that take no parameters work fine
        with the apply method."""
        dev = device(3)
        state = init_state(3)
        applied_operation = operation(wires=[0, 1, 2])
        wires = [0, 1, 2]

        dev.apply([qml.StatePrep(state, wires=wires), applied_operation])
        dev._samples = dev.generate_samples()

        res = np.fromiter(dev.probability(), dtype=np.float64)
        expected = np.abs(applied_operation.matrix() @ state) ** 2
        assert np.allclose(res, expected, **tol)
