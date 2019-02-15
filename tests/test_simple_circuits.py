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
import cmath
import logging as log
import math
import unittest

from pennylane import numpy as np

from defaults import pennylane as qml, BaseTest, IBMQX_TOKEN
from pennylane_qiskit import BasicAerQiskitDevice, IbmQQiskitDevice, LegacySimulatorsQiskitDevice, AerQiskitDevice

log.getLogger('defaults')


class SimpleCircuitsTest(BaseTest):
    """test the BasisState operation.
    """

    num_subsystems = 4
    devices = None

    def setUp(self):
        super().setUp()

        self.devices = []
        if self.args.provider == 'basicaer' or self.args.provider == 'all':
            self.devices.append(BasicAerQiskitDevice(wires=self.num_subsystems))
        if self.args.provider == 'aer' or self.args.provider == 'all':
            self.devices.append(AerQiskitDevice(wires=self.num_subsystems))
        if self.args.provider == 'legacy' or self.args.provider == 'all':
            self.devices.append(LegacySimulatorsQiskitDevice(wires=self.num_subsystems))
        if self.args.provider == 'ibm' or self.args.provider == 'all':
            if IBMQX_TOKEN is not None:
                self.devices.append(IbmQQiskitDevice(wires=self.num_subsystems, num_runs=8*1024, ibmqx_token=IBMQX_TOKEN))
            else:
                log.warning("Skipping test of the IbmQQiskitDevice device because IBM login credentials could not be "
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
                    for i, p in enumerate(bits_to_flip):
                        if p == 1:
                            qml.PauliX(wires=[i])
                    return qml.expval.PauliZ(0), qml.expval.PauliZ(1), qml.expval.PauliZ(2), qml.expval.PauliZ(3)

                self.assertAllAlmostEqual([1]*self.num_subsystems-2*bits_to_flip, np.array(circuit()), delta=self.tol)

    def test_rotations_cnot(self):
        """Test BasisState with preparations on the whole system."""
        if self.devices is None:
            return
        self.logTestName()

        for device in self.devices:

            @qml.qnode(device)
            def circuit(x, y, z):
                qml.RZ(z, wires=[0])
                qml.RY(y, wires=[0])
                qml.RX(x, wires=[0])
                qml.CNOT(wires=[0, 1])
                return qml.expval.PauliZ(wires=1)

            self.assertAllAlmostEqual(0.96875, np.array(circuit(0.2, 0.1, 0.3)), delta=self.tol)

    def test_arbitrary_state(self):
        """Test BasisState with preparations on the whole system."""
        if self.devices is None:
            return
        self.logTestName()

        for device in self.devices:
            for index in range(8):
                state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                state[index] = 1.0

                @qml.qnode(device)
                def circuit():
                    qml.QubitStateVector(state, wires=[0, 1, 2])
                    return qml.expval.PauliZ(0), qml.expval.PauliZ(1), qml.expval.PauliZ(2)

                result = np.array(circuit())
                expected = np.array(list(map(lambda c: 1.0 if c == '0' else -1.0, "{:b}".format(index).zfill(3)[::-1])))
                self.assertAllAlmostEqual(expected, result, delta=self.tol)

    def test_arbitrary_unitary(self):
        """Test BasisState with preparations on the whole system."""
        if self.devices is None:
            return
        self.logTestName()

        for device in self.devices:
            test_input = [
                np.array([1, 0, 0, 1]),
                1/math.sqrt(2) * np.array([1, -cmath.exp(1.0j*cmath.pi/2), cmath.exp(1.0j*cmath.pi/4), cmath.exp(1.0j*(cmath.pi/2 + cmath.pi/4))]),
                np.array([1, 0, 0, cmath.exp(1.0j*cmath.pi/4)]),
            ]
            for input in test_input:
                @qml.qnode(device)
                def circuit():
                    qml.QubitUnitary(input, wires=[0])
                    return qml.expval.PauliZ(0)

                circuit()
                # TODO 2018-12-23 Carsten Blank: create meaningful tests


if __name__ == '__main__':
    print('Testing PennyLane qiskit Plugin version ' + qml.version() + ', BasisState operation.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (SimpleCircuitsTest, ):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
