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
Unit tests for the :mod:`pennylane_qiskit` device documentation
"""

import logging as log
import re
import unittest

from defaults import pennylane as qml, BaseTest
from pennylane_qiskit.devices import BasicAerQiskitDevice, IbmQQiskitDevice, AerQiskitDevice

log.getLogger('defaults')


class DocumentationTest(BaseTest):
    """test documentation of the plugin.
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
                self.devices.append(IbmQQiskitDevice(wires=self.num_subsystems, num_runs=8 * 1024, ibmqx_token=self.args.ibmqx_token))
            else:
                log.warning("Skipping test of the IbmQQiskitDevice device because IBM login credentials could not be "
                            "found in the PennyLane configuration file.")

    def test_device_docstrings(self):
        for dev in self.devices:
            docstring = dev.__doc__
            supp_operations = dev.operations
            supp_expectations = dev.expectations
            print(docstring)
            documented_operations = (
                [re.findall(r"(?:pennylane\.|pennylane_qiskit\.ops\.)([^`> ]*)", string) for string in
                 re.findall(r"(?:(?:Extra|Supported PennyLane) Operations:\n((?:\s*:class:`[^`]+`,?\n)*))", docstring,
                            re.MULTILINE)])
            documented_operations = set([item for sublist in documented_operations for item in sublist])

            documented_expectations = (
                [re.findall(r"(?:pennylane\.expval\.|pennylane_qiskit\.expval\.)([^`> ]*)", string) for string in
                 re.findall(r"(?:(?:Extra|Supported PennyLane) Expectations:\n((?:\s*:class:`[^`]+`,?\n)*))", docstring,
                            re.MULTILINE)])
            documented_expectations = set([item for sublist in documented_expectations for item in sublist])

            supported_but_not_documented_operations = supp_operations.difference(documented_operations)

            self.assertFalse(supported_but_not_documented_operations,
                             msg='For device {} the Operations {} are supported but not documented.'.format(
                                 dev.short_name, supported_but_not_documented_operations))

            documented_but_not_supported_operations = documented_operations.difference(supp_operations)

            self.assertFalse(documented_but_not_supported_operations,
                             msg='For device {} the Operations {} are documented but not actually supported.'.format(
                                 dev.short_name, documented_but_not_supported_operations))

            supported_but_not_documented_expectations = supp_expectations.difference(documented_expectations)

            self.assertFalse(supported_but_not_documented_expectations,
                             msg='For device {} the Expectations {} are supported but not documented.'.format(
                                 dev.short_name, supported_but_not_documented_expectations))
            documented_but_not_supported_expectations = documented_expectations.difference(supp_expectations)

            self.assertFalse(documented_but_not_supported_expectations,
                             msg='For device {} the Expectations {} are documented but not actually supported.'.format(
                                 dev.short_name, documented_but_not_supported_expectations))


if __name__ == '__main__':
    print('Testing PennyLane qiskit Plugin version ' + qml.version() + ', device documentation.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (DocumentationTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
