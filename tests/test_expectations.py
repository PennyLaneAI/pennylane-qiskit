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
import logging as log
from pennylane import numpy as np

from defaults import pennylane as qml, BaseTest
from pennylane_qiskit import BasicAerQiskitDevice, AerQiskitDevice, IbmQQiskitDevice
import unittest
import pytest
import os

log.getLogger("defaults")

IBMQX_TOKEN = None
ibm_options = qml.default_config["qiskit.ibmq"]

if "ibmqx_token" in ibm_options:
    IBMQX_TOKEN = ibm_options["ibmqx_token"]

elif "IBMQX_TOKEN" in os.environ and os.environ["IBMQX_TOKEN"] is not None:
    IBMQX_TOKEN = os.environ["IBMQX_TOKEN"]

num_subsystems = 2
num_wires = 2
shots = 16 * 1024
ibmq_shots = 8 * 1024
devices = [BasicAerQiskitDevice(wires=num_wires), AerQiskitDevice(wires=num_wires)]

if IBMQX_TOKEN is not None:
    devices.append(
        IbmQQiskitDevice(wires=num_wires, num_runs=8 * 1024, ibmqx_token=ibmqx_token)
    )


@pytest.mark.parametrize("device", devices)
def test_identity_expectation(device):
    """Test that identity expectation value (i.e. the trace) is 1"""
    theta = 0.432
    phi = 0.123

    device.apply("RX", wires=[0], par=[theta])
    device.apply("RX", wires=[1], par=[phi])
    device.apply("CNOT", wires=[0, 1], par=[])

    O = qml.Identity
    name = "Identity"

    device._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
    res = device.pre_measure()

    res = np.array([device.expval(name, [0], []), device.expval(name, [1], [])])

    # below are the analytic expectation values for this circuit (trace should always be 1)
    assert np.allclose(res, np.array([1, 1]), atol=1 / np.sqrt(device.shots))


@pytest.mark.parametrize("device", devices)
def test_pauliz_expectation(device):
    """Tests that PauliZ expectation value is correct"""
    theta = 0.432
    phi = 0.123

    device.apply("RX", wires=[0], par=[theta])
    device.apply("RX", wires=[1], par=[phi])
    device.apply("CNOT", wires=[0, 1], par=[])

    O = qml.PauliZ
    name = "PauliZ"

    device._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
    res = device.pre_measure()

    res = np.array([device.expval(name, [0], []), device.expval(name, [1], [])])
    print(res)

    # below are the analytic expectation values for this circuit
    assert np.allclose(
        res,
        np.array([np.cos(theta), np.cos(theta) * np.cos(phi)]),
        atol=1 / np.sqrt(device.shots),
    )


@pytest.mark.parametrize("device", devices)
def test_paulix_expectation(device):
    """Tests that PauliX expectation value is correct"""
    theta = 0.432
    phi = 0.123

    device.apply("RX", wires=[0], par=[theta])
    device.apply("RX", wires=[1], par=[phi])
    device.apply("CNOT", wires=[0, 1], par=[])

    O = qml.PauliX
    name = "PauliX"

    device._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
    device.pre_measure()

    res = np.array([device.expval(name, [0], []), device.expval(name, [1], [])])
    # below are the analytic expectation values for this circuit
    assert np.allclose(
        res,
        np.array([np.sin(theta) * np.sin(phi), np.sin(phi)]),
        atol=1 / np.sqrt(device.shots),
    )


@pytest.mark.parametrize("device", devices)
def test_pauliy_expectation(device):
    """Test that PauliY expectation value is correct"""
    theta = 0.432
    phi = 0.123

    device.apply("RX", wires=[0], par=[theta])
    device.apply("RX", wires=[1], par=[phi])
    device.apply("CNOT", wires=[0, 1], par=[])

    O = qml.PauliY
    name = "PauliY"

    device._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
    device.pre_measure()

    res = np.array([device.expval(name, [0], []), device.expval(name, [1], [])])
    # below are the analytic expectation values for this circuit
    assert np.allclose(
        res, np.array([0, -np.cos(theta) * np.sin(phi)]), atol=1 / np.sqrt(device.shots)
    )


@pytest.mark.parametrize("device", devices)
def test_hadamard_expectation(device):
    """Tests that Hadamard expectation value is correct"""
    theta = 0.432
    phi = 0.123

    device.apply("RX", wires=[0], par=[theta])
    device.apply("RX", wires=[1], par=[phi])
    device.apply("CNOT", wires=[0, 1], par=[])

    O = qml.Hadamard
    name = "Hadamard"

    device._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
    device.pre_measure()

    res = np.array([device.expval(name, [0], []), device.expval(name, [1], [])])
    # below are the analytic expectation values for this circuit
    expected = np.array(
        [
            np.sin(theta) * np.sin(phi) + np.cos(theta),
            np.cos(theta) * np.cos(phi) + np.sin(phi),
        ]
    ) / np.sqrt(2)
    assert np.allclose(res, expected, atol=1 / np.sqrt(device.shots))


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

    device.apply("RX", wires=[0], par=[theta])
    device.apply("RX", wires=[1], par=[phi])
    device.apply("CNOT", wires=[0, 1], par=[])

    O = qml.Hermitian
    name = "Hermitian"

    device._obs_queue = [
        O(H, wires=[0], do_queue=False),
        O(H, wires=[1], do_queue=False),
    ]
    device.pre_measure()

    res = np.array([device.expval(name, [0], [H]), device.expval(name, [1], [H])])

    # below are the analytic expectation values for this circuit with arbitrary
    # Hermitian observable H
    a = H[0, 0]
    re_b = H[0, 1].real
    d = H[1, 1]
    ev1 = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
    ev2 = ((a - d) * np.cos(theta) * np.cos(phi) + 2 * re_b * np.sin(phi) + a + d) / 2
    expected = np.array([ev1, ev2])

    assert np.allclose(res, expected, atol=1 / np.sqrt(device.shots))


@pytest.mark.parametrize("device", devices)
def test_int_wires(device):
    """Test that passing wires as int works for expval."""
    theta = 0.432
    phi = 0.123

    device.apply("RX", wires=[0], par=[theta])
    device.apply("RX", wires=[1], par=[phi])
    device.apply("CNOT", wires=[0, 1], par=[])

    O = qml.Identity
    name = "Identity"

    device._obs_queue = [O(wires=0, do_queue=False), O(wires=1, do_queue=False)]
    res = device.pre_measure()

    res = np.array([device.expval(name, 0, []), device.expval(name, 1, [])])

    # below are the analytic expectation values for this circuit (trace should always be 1)
    assert np.allclose(res, np.array([1, 1]), atol=1 / np.sqrt(device.shots))
