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

import unittest
import logging as log
from defaults import pennylane as qml, BaseTest
import pennylane
from pennylane import numpy as np

from pennylane_qiskit import BasicAerDevice, IBMQDevice, AerDevice

log.getLogger('defaults')


class BasisStateTest(BaseTest):
    """test the BasisState operation.
    """

    num_subsystems = 4
    devices = None

    def setUp(self):
        super().setUp()

        self.devices = []
        if self.args.device == 'basicaer' or self.args.device == 'all':
            self.devices.append(BasicAerDevice(wires=self.num_subsystems))
        if self.args.device == 'aer' or self.args.device == 'all':
            self.devices.append(AerDevice(wires=self.num_subsystems))
        if self.args.device == 'ibmq' or self.args.device == 'all':
            if self.args.ibmqx_token is not None:
                self.devices.append(
                    IBMQDevice(wires=self.num_subsystems, num_runs=8 * 1024, ibmqx_token=self.args.ibmqx_token))
            else:
                log.warning("Skipping test of the IBMQDevice device because IBM login credentials could not be "
                            "found in the PennyLane configuration file.")

    def test_basis_state(self):
        """Test BasisState with preparations on the whole system."""
        if self.devices is None:
            return
        self.logTestName()

        for device in self.devices:
            for bits_to_flip in [np.array([0, 0, 0, 0]),
                                 np.array([0, 1, 1, 0]),
                                 np.array([1, 1, 1, 0]),
                                 np.array([1, 1, 1, 1])]:
                @qml.qnode(device)
                def circuit():
                    qml.BasisState(bits_to_flip, wires=list(range(self.num_subsystems)))
                    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3))

                log.info("BasisState on device %s with bitflip pattern %s.", device.name, bits_to_flip)

                self.assertAllAlmostEqual([1] * self.num_subsystems - 2 * bits_to_flip, np.array(circuit()),
                                          delta=self.tol)

    def test_basis_state_on_subsystem(self):
        """Test BasisState with preparations on subsystems."""
        if self.devices is None:
            return
        self.logTestName()

        for device in self.devices:
            for bits_to_flip in [np.array([0, 0, 0]),
                                 np.array([1, 0, 0]),
                                 np.array([0, 1, 1]),
                                 np.array([1, 1, 0]),
                                 np.array([1, 1, 1])]:
                @qml.qnode(device)
                def circuit():
                    qml.BasisState(bits_to_flip, wires=list(range(self.num_subsystems - 1)))
                    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3))

                log.info("BasisState on device %s with bitflip pattern %s (sub-system).", device.name, bits_to_flip)

                self.assertAllAlmostEqual([1] * (self.num_subsystems - 1) - 2 * bits_to_flip, np.array(circuit()[:-1]),
                                          delta=self.tol)

    def test_disallow_basis_state_after_other_operation(self):
        if self.devices is None:
            return
        self.logTestName()

        for device in self.devices:
            @qml.qnode(device)
            def circuit():
                qml.PauliX(wires=[0])
                qml.BasisState(np.array([0, 1, 0, 1]), wires=list(range(self.num_subsystems)))
                return qml.expval(qml.PauliZ(0))

            self.assertRaises(pennylane._device.DeviceError, circuit)


if __name__ == '__main__':
    print('Testing PennyLane qiskit Plugin version ' + qml.version() + ', BasisState operation.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasisStateTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
