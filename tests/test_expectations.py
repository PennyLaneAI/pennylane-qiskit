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

log.getLogger('defaults')


class TestQVMBasic(BaseTest):
    """Unit tests for the QVM simulator."""
    # pylint: disable=protected-access

    num_subsystems = 2
    shots = 16 * 1024
    ibmq_shots = 8 * 1024
    devices = None

    def setUp(self):
        super().setUp()

        self.devices = []
        if self.args.device == 'basicaer' or self.args.device == 'all':
            self.devices.append(BasicAerQiskitDevice(wires=self.num_subsystems, shots=self.shots))
        if self.args.device == 'aer' or self.args.device == 'all':
            self.devices.append(AerQiskitDevice(wires=self.num_subsystems, shots=self.shots))
        if self.args.device == 'ibmq' or self.args.device == 'all':
            if self.args.ibmqx_token is not None:
                self.devices.append(
                    IbmQQiskitDevice(wires=self.num_subsystems, shots=self.ibmq_shots, ibmqx_token=self.args.ibmqx_token))
            else:
                log.warning("Skipping test of the IbmQQiskitDevice device because IBM login credentials could not be "
                            "found in the PennyLane configuration file.")

    def test_identity_expectation(self):
        """Test that identity expectation value (i.e. the trace) is 1"""
        theta = 0.432
        phi = 0.123
        for dev in self.devices:
            dev.apply('RX', wires=[0], par=[theta])
            dev.apply('RX', wires=[1], par=[phi])
            dev.apply('CNOT', wires=[0, 1], par=[])

            O = qml.expval.qubit.Identity
            name = 'Identity'

            dev._expval_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
            res = dev.pre_expval()

            res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])

            # below are the analytic expectation values for this circuit (trace should always be 1)
            self.assertAllAlmostEqual(res, np.array([1, 1]), delta=3/np.sqrt(dev.shots))

    def test_pauliz_expectation(self):
        """Test that PauliZ expectation value is correct"""
        theta = 0.432
        phi = 0.123

        for dev in self.devices:
            dev.apply('RX', wires=[0], par=[theta])
            dev.apply('RX', wires=[1], par=[phi])
            dev.apply('CNOT', wires=[0, 1], par=[])

            O = qml.expval.PauliZ
            name = 'PauliZ'

            dev._expval_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
            res = dev.pre_expval()

            res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])

            # below are the analytic expectation values for this circuit
            self.assertAllAlmostEqual(res, np.array([np.cos(theta), np.cos(theta)*np.cos(phi)]), delta=3/np.sqrt(dev.shots))

    def test_paulix_expectation(self):
        """Test that PauliX expectation value is correct"""
        theta = 0.432
        phi = 0.123

        for dev in self.devices:
            dev.apply('RY', wires=[0], par=[theta])
            dev.apply('RY', wires=[1], par=[phi])
            dev.apply('CNOT', wires=[0, 1], par=[])

            O = qml.expval.PauliX
            name = 'PauliX'

            dev._expval_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
            dev.pre_expval()

            res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])
            # below are the analytic expectation values for this circuit
            self.assertAllAlmostEqual(res, np.array([np.sin(theta)*np.sin(phi), np.sin(phi)]), delta=3/np.sqrt(dev.shots))

    def test_pauliy_expectation(self):
        """Test that PauliY expectation value is correct"""
        theta = 0.432
        phi = 0.123

        for dev in self.devices:
            dev.apply('RX', wires=[0], par=[theta])
            dev.apply('RX', wires=[1], par=[phi])
            dev.apply('CNOT', wires=[0, 1], par=[])

            O = qml.expval.PauliY
            name = 'PauliY'

            dev._expval_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
            dev.pre_expval()

            # below are the analytic expectation values for this circuit
            res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])
            self.assertAllAlmostEqual(res, np.array([0, -np.cos(theta)*np.sin(phi)]), delta=3/np.sqrt(dev.shots))

    def test_hadamard_expectation(self):
        """Test that Hadamard expectation value is correct"""
        theta = 0.432
        phi = 0.123

        for dev in self.devices:
            dev.apply('RY', wires=[0], par=[theta])
            dev.apply('RY', wires=[1], par=[phi])
            dev.apply('CNOT', wires=[0, 1], par=[])

            O = qml.expval.Hadamard
            name = 'Hadamard'

            dev._expval_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]
            dev.pre_expval()

            res = np.array([dev.expval(name, [0], []), dev.expval(name, [1], [])])
            # below are the analytic expectation values for this circuit
            expected = np.array([np.sin(theta)*np.sin(phi)+np.cos(theta), np.cos(theta)*np.cos(phi)+np.sin(phi)])/np.sqrt(2)
            self.assertAllAlmostEqual(res, expected, delta=3/np.sqrt(dev.shots))

    def test_hermitian_expectation(self):
        """Test that arbitrary Hermitian expectation values are correct"""
        theta = 0.432
        phi = 0.123
        H = np.array([[1.02789352, 1.61296440 - 0.3498192j],
                      [1.61296440 + 0.3498192j, 1.23920938 + 0j]])

        for dev in self.devices:
            dev.apply('RY', wires=[0], par=[theta])
            dev.apply('RY', wires=[1], par=[phi])
            dev.apply('CNOT', wires=[0, 1], par=[])

            O = qml.expval.qubit.Hermitian
            name = 'Hermitian'

            dev._expval_queue = [O(H, wires=[0], do_queue=False), O(H, wires=[1], do_queue=False)]
            dev.pre_expval()

            res = np.array([dev.expval(name, [0], [H]), dev.expval(name, [1], [H])])

            # below are the analytic expectation values for this circuit with arbitrary
            # Hermitian observable H
            a = H[0, 0]
            re_b = H[0, 1].real
            d = H[1, 1]
            ev1 = ((a-d)*np.cos(theta)+2*re_b*np.sin(theta)*np.sin(phi)+a+d)/2
            ev2 = ((a-d)*np.cos(theta)*np.cos(phi)+2*re_b*np.sin(phi)+a+d)/2
            expected = np.array([ev1, ev2])

            self.assertAllAlmostEqual(res, expected, delta=5/np.sqrt(dev.shots))

    def test_int_wires(self):
        """Test that passing wires as int works for expval."""
        theta = 0.432
        phi = 0.123
        for dev in self.devices:
            dev.apply('RX', wires=[0], par=[theta])
            dev.apply('RX', wires=[1], par=[phi])
            dev.apply('CNOT', wires=[0, 1], par=[])

            O = qml.expval.qubit.Identity
            name = 'Identity'

            dev._expval_queue = [O(wires=0, do_queue=False), O(wires=1, do_queue=False)]
            res = dev.pre_expval()

            res = np.array([dev.expval(name, 0, []), dev.expval(name, 1, [])])

            # below are the analytic expectation values for this circuit (trace should always be 1)
            self.assertAllAlmostEqual(res, np.array([1, 1]), delta=3/np.sqrt(dev.shots))


if __name__ == '__main__':
    print('Testing PennyLane qiskit Plugin version ' + qml.version() + ', expectations.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (TestQVMBasic,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
