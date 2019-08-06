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
Unit tests for the :mod:`pennylane_qiskit` BasisState operation.
"""
import pennylane as qml
from pennylane import numpy as np

from pennylane_qiskit import BasicAerDevice, IBMQDevice, AerDevice
import pytest
import os

num_wires = 4
bits = [
    np.array([0, 0, 0, 0]),
    np.array([0, 1, 1, 0]),
    np.array([1, 1, 1, 0]),
    np.array([1, 1, 1, 1]),
]
bits_subsystem = [
    np.array([0, 0, 0]),
    np.array([1, 0, 0]),
    np.array([0, 1, 1]),
    np.array([1, 1, 0]),
    np.array([1, 1, 1]),
]

IBMQX_TOKEN = None
ibm_options = qml.default_config["qiskit.ibmq"]

if "ibmqx_token" in ibm_options:
    IBMQX_TOKEN = ibm_options["ibmqx_token"]

elif "IBMQX_TOKEN" in os.environ and os.environ["IBMQX_TOKEN"] is not None:
    IBMQX_TOKEN = os.environ["IBMQX_TOKEN"]

devices = [BasicAerDevice(wires=num_wires), AerDevice(wires=num_wires)]

if IBMQX_TOKEN is not None:
    devices.append(
        IbmQQiskitDevice(wires=num_wires, num_runs=8 * 1024, ibmqx_token=ibmqx_token)
    )


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("bits_to_flip", bits)
def test_basis_state(device, bits_to_flip):
    """Tests BasisState with preparations on the whole system."""

    @qml.qnode(device)
    def circuit():
        qml.BasisState(bits_to_flip, wires=[0, 1, 2, 3])
        return (
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.PauliZ(2)),
            qml.expval(qml.PauliZ(3)),
        )

    assert np.allclose([1] * num_wires - 2 * bits_to_flip, np.array(circuit()))


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("bits_to_flip", bits_subsystem)
def test_basis_state_on_subsystem(device, bits_to_flip):
    """Tests BasisState with preparations on subsystems."""

    @qml.qnode(device)
    def circuit():
        qml.BasisState(bits_to_flip, wires=[i for i in range(num_wires - 1)])
        return (
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.PauliZ(2)),
            qml.expval(qml.PauliZ(3)),
        )

    assert np.allclose(
        [1] * (num_wires - 1) - 2 * bits_to_flip, np.array(circuit()[:-1])
    )


@pytest.mark.parametrize("device", devices)
def test_disallow_basis_state_after_other_operation(device):
    """Tests correct error is raised if trying to set basis after operation."""

    @qml.qnode(device)
    def circuit():
        qml.PauliX(wires=[0])
        qml.BasisState(np.array([0, 1, 0, 1]), wires=[i for i in range(num_wires - 1)])
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(qml._device.DeviceError):
        circuit()
