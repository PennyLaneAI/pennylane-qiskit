# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane_pq` devices' behavior when applying unsupported operations.
"""
from defaults import pennylane as qml
from pennylane_qiskit import BasicAerDevice, IBMQDevice, AerDevice

import os

import pytest


num_wires = 4
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
def test_unsupported_operation(device):
    """Tests if the correct error is raised for an unsupported operation"""
    @qml.qnode(device)
    def circuit():
        qml.Beamsplitter(
            0.2, 0.1, wires=[0, 1]
        )  # this expectation will never be supported
        return qml.expval(qml.QuadOperator(0.7, 0))

    with pytest.raises(qml._device.DeviceError):
        circuit()


@pytest.mark.parametrize("device", devices)
def test_unsupported_observable(device):
    """Tests if the correct error is raised for an unsupported observable"""
    @qml.qnode(device)
    def circuit():
        return qml.expval(
            qml.QuadOperator(0.7, 0)
        )  # this expectation will never be supported

    with pytest.raises(qml._device.DeviceError):
        circuit()
