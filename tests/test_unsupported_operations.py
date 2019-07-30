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

import logging as log
import unittest

import pennylane

from defaults import pennylane as qml, BaseTest
from pennylane_qiskit.devices import BasicAerQiskitDevice, IbmQQiskitDevice,  \
    AerQiskitDevice

log.getLogger('defaults')


class UnsupportedOperationTest(BaseTest):
    """test that unsupported operations/observables raise DeviceErrors.
    """

    num_subsystems = 4
    devices = None

    def setUp(self):
        super().setUp()

        self.devices = []
        if self.args.device == 'basicaer' or self.args.device == 'all':
            self.devices.append(BasicAerQiskitDevice(wires=self.num_subsystems))
        if self.args.device == 'aer' or self.args.device == 'all':
            self.devices.append(AerQiskitDevice(wires=self.num_subsystems))
        if self.args.device == 'ibmq' or self.args.device == 'all':
            if self.args.ibmqx_token is not None:
                self.devices.append(
                    IbmQQiskitDevice(wires=self.num_subsystems, num_runs=8 * 1024, ibmqx_token=self.args.ibmqx_token))
            else:
                log.warning(
                    "Skipping test of the IbmQQiskitDevice device because IBM login credentials could not be found in the PennyLane configuration file.")

    def test_unsupported_operation(self):
        if self.devices is None:
            return
        self.logTestName()

        for device in self.devices:
            @qml.qnode(device)
            def circuit():
                qml.Beamsplitter(0.2, 0.1, wires=[0, 1])  # this expectation will never be supported
                return qml.expval(qml.QuadOperator(0.7, 0))

            self.assertRaises(pennylane._device.DeviceError, circuit)

        for device in self.devices:
            self.assertRaises(ValueError, device.apply, 'RXY', wires=[0], par=None)

    def test_unsupported_expectation(self):
        if self.devices is None:
            return
        self.logTestName()

        for device in self.devices:
            @qml.qnode(device)
            def circuit():
                return qml.expval(qml.QuadOperator(0.7, 0))  # this expectation will never be supported

            self.assertRaises(pennylane._device.DeviceError, circuit)


if __name__ == '__main__':
    print('Testing PennyLane qiskit Plugin version ' + qml.version() + ', unsupported operations.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (UnsupportedOperationTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
