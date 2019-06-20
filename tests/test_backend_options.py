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
Unit tests for :mod:`pennylane_qiskit` simple circuits.
"""
import logging as log
import unittest

import cmath
import math
from pennylane import numpy as np
from pennylane.plugins import DefaultQubit
from qiskit.providers.aer import noise
from qiskit.providers.models import BackendProperties

from defaults import pennylane as qml, BaseTest, IBMQX_TOKEN
from pennylane_qiskit import BasicAerQiskitDevice, IbmQQiskitDevice, AerQiskitDevice

log.getLogger('defaults')


class BackendOptionsTest(BaseTest):
    """test the BasisState operation.
    """

    num_subsystems = 4
    devices = None

    def setUp(self):
        super().setUp()

        self.devices = [DefaultQubit(wires=self.num_subsystems)]
        if self.args.device == 'basicaer' or self.args.device == 'all':
            self.devices.append(BasicAerQiskitDevice(wires=self.num_subsystems))
        if self.args.device == 'aer' or self.args.device == 'all':
            self.devices.append(AerQiskitDevice(wires=self.num_subsystems))
        if self.args.device == 'ibmq' or self.args.device == 'all':
            if IBMQX_TOKEN is not None:
                self.devices.append(
                    IbmQQiskitDevice(wires=self.num_subsystems, num_runs=8 * 1024, ibmqx_token=IBMQX_TOKEN))
            else:
                log.warning("Skipping test of the IbmQQiskitDevice device because IBM login credentials could not be "
                            "found in the PennyLane configuration file.")

    def test_basicaer_initial_unitary(self):
        """Test BasisState with preparations on the whole system."""
        if self.devices is None:
            return
        self.logTestName()

        if self.args.device == 'basicaer' or self.args.device == 'all':
            initial_unitary = np.array([
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0]
            ])
            dev = BasicAerQiskitDevice(wires=2, backend='unitary_simulator', initial_unitary=initial_unitary)
            # dev = BasicAerQiskitDevice(wires=self.num_subsystems, backend='unitary_simulator')

            @qml.qnode(dev)
            def circuit():
                return qml.expval.PauliZ(wires=[0]), qml.expval.PauliZ(wires=[1])

            log.info("Outcome: %s", circuit())

    def test_basicaer_chop_threshold(self):
        """Test BasisState with preparations on the whole system."""
        if self.devices is None:
            return
        self.logTestName()

        if self.args.device == 'basicaer' or self.args.device == 'all':
            dev = BasicAerQiskitDevice(wires=self.num_subsystems, chop_threshold=1e-1)

            @qml.qnode(dev)
            def circuit():
                # TODO: rotation within the tolerance should be 0
                return qml.expval.PauliZ(wires=[0]), qml.expval.PauliZ(wires=[1])

            log.info("Outcome: %s", circuit())
