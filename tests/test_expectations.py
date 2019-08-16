# Copyright 2018 Carsten Blank

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for :mod:`pennylane_qiskit` expectation values
"""
from pennylane import numpy as np

from defaults import pennylane as qml, BaseTest
from pennylane_qiskit import BasicAerDevice, IBMQDevice, AerDevice

from qiskit.providers.ibmq.exceptions import IBMQAccountError

import unittest

import pytest

import os


num_subsystems = 2
num_wires = 2
shots = 100
ibmq_shots = 100
aer_backends = ["statevector_simulator", "unitary_simulator", "qasm_simulator"]

devices = [BasicAerDevice(wires=num_wires, backend=b, shots=shots) for b in aer_backends]
devices += [AerDevice(wires=num_wires, backend=b, shots=shots) for b in aer_backends]

try:
    devices.append(IBMQDevice(wires=num_wires, shots=ibmq_shots))
except IBMQAccountError:
    pass

@pytest.mark.parametrize("device", devices)
def test_identity_expectation(device):
    """Tests that identity expectation value (i.e. the trace) is 1"""
    theta = 0.432
    phi = 0.123

    @qml.qnode(device)
    def circuit(theta, phi):
        qml.RX(theta, wires=[0])
        qml.RX(phi, wires=[1])
        qml.CNOT(wires=[0, 1])
        
        return (qml.expval(qml.Identity(wires=[0])),
                qml.expval(qml.Identity(wires=[1])))

    res = circuit(theta, phi)

    # below are the analytic expectation values for this circuit (trace should always be 1)
    assert np.allclose(
        res,
        np.array([1, 1]),
        atol=3 / np.sqrt(device.shots),
        rtol=0.,
    )


@pytest.mark.parametrize("device", devices)
def test_pauliz_expectation(device):
    """Tests that PauliZ expectation value is correct"""
    theta = 0.432
    phi = 0.123

    @qml.qnode(device)
    def circuit(theta, phi):
        qml.RX(theta, wires=[0])
        qml.RX(phi, wires=[1])
        qml.CNOT(wires=[0, 1])
        
        return (qml.expval(qml.PauliZ(wires=[0])),
                qml.expval(qml.PauliZ(wires=[1])))

    res = circuit(theta, phi)

    # below are the analytic expectation values for this circuit
    assert np.allclose(
        res,
        np.array([np.cos(theta), np.cos(theta) * np.cos(phi)]),
        atol=3 / np.sqrt(device.shots),
        rtol=0.,
    )


@pytest.mark.parametrize("device", devices)
def test_paulix_expectation(device):
    """Tests that PauliX expectation value is correct"""
    theta = 0.432
    phi = 0.123

    @qml.qnode(device)
    def circuit(theta, phi):
        qml.RX(theta, wires=[0])
        qml.RX(phi, wires=[1])
        qml.CNOT(wires=[0, 1])
        
        return (qml.expval(qml.PauliX(wires=[0])),
                qml.expval(qml.PauliX(wires=[1])))

    res = circuit(theta, phi)
    # below are the analytic expectation values for this circuit
    assert np.allclose(
        res,
        np.array([np.sin(theta) * np.sin(phi), np.sin(phi)]),
        atol=3 / np.sqrt(device.shots),
        rtol=0.,
    )


@pytest.mark.parametrize("device", devices)
def test_pauliy_expectation(device):
    """Tests that PauliY expectation value is correct"""
    theta = 0.432
    phi = 0.123

    @qml.qnode(device)
    def circuit(theta, phi):
        qml.RX(theta, wires=[0])
        qml.RX(phi, wires=[1])
        qml.CNOT(wires=[0, 1])
        
        return (qml.expval(qml.PauliY(wires=[0])),
                qml.expval(qml.PauliY(wires=[1])))

    res = circuit(theta, phi)
    
    # below are the analytic expectation values for this circuit
    assert np.allclose(
        res,
        np.array([0., -np.cos(theta) * np.sin(phi)]),
        atol=3 / np.sqrt(device.shots),
        rtol=0.,
    )


@pytest.mark.parametrize("device", devices)
def test_hadamard_expectation(device):
    """Tests that Hadamard expectation value is correct"""
    theta = 0.432
    phi = 0.123

    @qml.qnode(device)
    def circuit(theta, phi):
        qml.RX(theta, wires=[0])
        qml.RX(phi, wires=[1])
        qml.CNOT(wires=[0, 1])
        
        return (qml.expval(qml.Hadamard(wires=[0])),
                qml.expval(qml.Hadamard(wires=[1])))

    res = circuit(theta, phi)
    # below are the analytic expectation values for this circuit
    expected = np.array(
        [
            np.sin(theta) * np.sin(phi) + np.cos(theta),
            np.cos(theta) * np.cos(phi) + np.sin(phi),
        ]
    ) / np.sqrt(2.)
    assert np.allclose(
        res, expected,
        atol=5 / np.sqrt(device.shots),
        rtol=0.
    )


@pytest.mark.parametrize("device", devices)
def test_hermitian_expectation(device):
    """Tests that arbitrary Hermitian expectation values are correct"""
    theta = 0.432
    phi = 0.123
    H = np.array(
        [
            [1.02789352, 1.61296440 - 0.3498192j],
            [1.61296440 + 0.3498192j, 1.23920938 + 0j],
        ]
    )

    @qml.qnode(device)
    def circuit(theta, phi):
        qml.RX(theta, wires=[0])
        qml.RX(phi, wires=[1])
        qml.CNOT(wires=[0, 1])
        
        return (qml.expval(qml.Hermitian(H, wires=[0])),
                qml.expval(qml.Hermitian(H, wires=[1])))

    res = circuit(theta, phi)

    # below are the analytic expectation values for this circuit with arbitrary
    # Hermitian observable H
    a = H[0, 0]
    re_b = H[0, 1].real
    d = H[1, 1]
    ev1 = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
    ev2 = ((a - d) * np.cos(theta) * np.cos(phi) + 2 * re_b * np.sin(phi) + a + d) / 2
    expected = np.array([ev1, ev2])

    assert np.allclose(
        res,
        expected,
        atol=5 / np.sqrt(device.shots),
        rtol=0.
    )


@pytest.mark.parametrize("device", devices)
def test_int_wires(device):
    """Tests that passing wires as int works for expval."""
    theta = 0.432
    phi = 0.123

    @qml.qnode(device)
    def circuit(theta, phi):
        qml.RX(theta, wires=[0])
        qml.RX(phi, wires=[1])
        qml.CNOT(wires=[0, 1])
        
        return (qml.expval(qml.Identity(wires=0)),
                qml.expval(qml.Identity(wires=1)))

    res = circuit(theta, phi) 
    # below are the analytic expectation values for this circuit (trace should always be 1)
    assert np.allclose(
        res,
        np.array([1, 1]),
        atol=3 / np.sqrt(device.shots),
        rtol=3 / np.sqrt(device.shots),
    )
