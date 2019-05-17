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
Unit tests for the :mod:`pennylane_qiskit` device initialization
"""

import logging as log
import os
import unittest

from pennylane import DeviceError
from qiskit import IBMQ
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise import NoiseModel

from defaults import pennylane as qml, BaseTest, IBMQX_TOKEN
from pennylane_qiskit import IbmQQiskitDevice, AerQiskitDevice, BasicAerQiskitDevice

log.getLogger('defaults')


class DeviceInitialization(BaseTest):
    """test aspects of the device initialization.
    """

    num_subsystems = 4
    devices = None

    def test_ibm_no_token(self):
        # if there is an IBMQX token, save it and unset it so that it doesn't interfere with this test
        token_from_environment = os.getenv('IBMQX_TOKEN')
        if token_from_environment is not None:
            del os.environ['IBMQX_TOKEN']

        if self.args.provider == 'ibm' or self.args.provider == 'all':
            self.assertRaises(ValueError, IbmQQiskitDevice, wires=self.num_subsystems,
                              msg='Expected a ValueError if no IBMQX token is present.')

        # put the IBMQX token back into place fo other tests to use
        if token_from_environment is not None:
            os.environ['IBMQX_TOKEN'] = token_from_environment
            token_from_environment_back = os.getenv('IBMQX_TOKEN')
            self.assertEqual(token_from_environment, token_from_environment_back)

    def test_shots(self):
        if self.args.provider == 'ibmq_qasm_simulator' or self.args.provider == 'all':
            shots = 5
            dev1 = IbmQQiskitDevice(wires=self.num_subsystems, shots=shots, ibmqx_token=IBMQX_TOKEN)
            self.assertEqual(shots, dev1.shots)

    def test_noise_model_for_aer(self):
        try:
            noise_model = self._get_noise_model()  # type: NoiseModel

            dev = qml.device('qiskit.aer', wires=self.num_subsystems, ibmqx_token=IBMQX_TOKEN, noise_model=noise_model)
            self.assertIsNotNone(dev._noise_model)
            self.assertEqual(noise_model.as_dict(), dev._noise_model.as_dict())

            dev2 = AerQiskitDevice(wires=self.num_subsystems, noise_model=noise_model)
            self.assertIsNotNone(dev2._noise_model)
            self.assertEqual(noise_model.as_dict(), dev2._noise_model.as_dict())

        except DeviceError:
            raise Exception("This test is expected to fail until pennylane-qiskit is installed.")

    def test_noise_model_for_basic_aer(self):
        try:
            noise_model = self._get_noise_model()

            dev = qml.device('qiskit.basicaer', wires=self.num_subsystems, ibmqx_token=IBMQX_TOKEN,
                             noise_model=noise_model)
            self.assertIsNotNone(dev._noise_model)
            self.assertEqual(noise_model.as_dict(), dev._noise_model.as_dict())

            dev2 = BasicAerQiskitDevice(wires=self.num_subsystems, noise_model=noise_model)
            self.assertIsNotNone(dev2._noise_model)
            self.assertEqual(noise_model.as_dict(), dev2._noise_model.as_dict())

        except DeviceError:
            raise Exception("This test is expected to fail until pennylane-qiskit is installed.")

    @classmethod
    def _get_noise_model(cls):
        IBMQ.enable_account(IBMQX_TOKEN)
        device = IBMQ.get_backend('ibmqx4')
        properties = device.properties()

        return noise.device.basic_device_noise_model(properties)

    def test_initiatlization_via_pennylane(self):
        for short_name in [
            'qiskit.aer',
            'qiskit.basicaer',
            'qiskit.ibm'
        ]:
            try:
                qml.device(short_name, wires=2, ibmqx_token=IBMQX_TOKEN)
            except DeviceError:
                raise Exception("This test is expected to fail until pennylane-qiskit is installed.")

    def test_ibm_device(self):
        if self.args.provider in ['ibm', 'all']:
            import qiskit
            qiskit.IBMQ.enable_account(token=IBMQX_TOKEN)
            backends = qiskit.IBMQ.backends()
            qiskit.IBMQ.disable_accounts()
            try:
                for backend in backends:
                    qml.device('qiskit.ibm', wires=1, ibmqx_token=IBMQX_TOKEN, backend=backend)
            except DeviceError:
                raise Exception("This test is expected to fail until pennylane-qiskit is installed.")

    def test_aer_device(self):
        if self.args.provider in ['aer', 'all']:
            import qiskit
            try:
                for backend in qiskit.Aer.backends():
                    qml.device('qiskit.aer', wires=1, backend=backend)
            except DeviceError:
                raise Exception("This test is expected to fail until pennylane-qiskit is installed.")

    def test_basicaer_device(self):
        if self.args.provider in ['aer', 'all']:
            import qiskit
            try:
                for backend in qiskit.BasicAer.backends():
                    qml.device('qiskit.basicaer', wires=1, backend=backend)
            except DeviceError:
                raise Exception("This test is expected to fail until pennylane-qiskit is installed.")


if __name__ == '__main__':
    print('Testing PennyLane qiskit Plugin version ' + qml.version() + ', device initialization.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (DeviceInitialization,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
