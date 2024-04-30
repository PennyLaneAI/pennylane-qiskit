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
This module contains tests for testing backends and providers for PennyLane IBMQ devices.
"""
import pytest
import pennylane as qml
import qiskit

from semantic_version import Version

from pennylane_qiskit import BasicSimulatorDevice

# pylint: disable= unused-argument


@pytest.mark.skipif(
    Version(qiskit.__version__) < Version("1.0.0"),
    reason="versions below 1.0 are compatible with BasicAer",
)
def test_error_is_raised_if_initalizing_basicaer_device(monkeypatch):
    """Test that when Qiskit 1.0 is installed, an error is raised if you try
    to initialize the 'qiskit.basicaer' device."""

    # test that the correct error is actually raised in Qiskit 1.0 (rather than fx an import error)
    with pytest.raises(
        RuntimeError,
        match="Qiskit has discontinued the BasicAer device",
    ):
        qml.device("qiskit.basicaer", wires=2)


@pytest.mark.skipif(
    Version(qiskit.__version__) >= Version("1.0.0"),
    reason="versions 1.0 and above are compatible with BasicSimulator",
)
def test_error_is_raised_if_initalizing_basic_simulator_device(monkeypatch):
    """Test that when a version of Qiskit below 1.0 is installed, an error is raised if you try
    to initialize the BasicSimulatorDevice."""

    # test that the correct error is actually raised in Qiskit 1.0 (rather than fx an import error)
    with pytest.raises(
        RuntimeError,
        match="device is not compatible with version of Qiskit prior to 1.0",
    ):
        BasicSimulatorDevice(wires=2)
