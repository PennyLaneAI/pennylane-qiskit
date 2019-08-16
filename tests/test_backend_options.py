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

from pennylane import numpy as np
from pennylane.plugins import DefaultQubit

from defaults import pennylane as qml, BaseTest
from pennylane_qiskit import BasicAerDevice, IBMQDevice, AerDevice

log.getLogger('defaults')


class BackendOptionsTest(BaseTest):
    """test the BasisState operation.
    """

    num_subsystems = 2
    devices = None

    def setUp(self):
        super().setUp()

        self.devices = [DefaultQubit(wires=self.num_subsystems)]
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

    def test_basicaer_initial_unitary(self):
        """Test BasisState with preparations on the whole system."""
        if self.devices is None:
            return
        self.logTestName()

        if self.args.device == 'basicaer' or self.args.device == 'all':
            initial_unitaries = [
                # H \otimes Id
                np.array([
                    [np.sqrt(2), np.sqrt(2), 0, 0],
                    [np.sqrt(2), -np.sqrt(2), 0, 0],
                    [0, 0, np.sqrt(2), np.sqrt(2)],
                    [0, 0, np.sqrt(2), -np.sqrt(2)]
                ]) / 2,
                # H \otimes X
                np.array([
                    [0, 0, np.sqrt(2), np.sqrt(2)],
                    [0, 0, np.sqrt(2), -np.sqrt(2)],
                    [np.sqrt(2), np.sqrt(2), 0, 0],
                    [np.sqrt(2), -np.sqrt(2), 0, 0]
                ]) / 2
            ]
            expected_outcomes = [
                [0, 1],
                [0, -1]
            ]

            for initial_unitary, expected_outcome in zip(initial_unitaries, expected_outcomes):

                dev = BasicAerDevice(wires=self.num_subsystems, backend='unitary_simulator', initial_unitary=initial_unitary)

                @qml.qnode(dev)
                def circuit():
                    return qml.expval(qml.PauliZ(wires=[0])), qml.expval(qml.PauliZ(wires=[1]))

                measurement = circuit()

                self.assertAllAlmostEqual(measurement, expected_outcome, delta=self.tol)

                log.info("Outcome: %s", measurement)

    # def test_basicaer_chop_threshold(self):
    #     """Test BasisState with preparations on the whole system."""
    #     if self.devices is None:
    #         return
    #     self.logTestName()

    #     if self.args.device == 'basicaer' or self.args.device == 'all':
    #         dev = BasicAerDevice(wires=self.num_subsystems, chop_threshold=1e-2, backend='unitary_simulator')

    #         @qml.qnode(dev)
    #         def circuit():
    #             # An angle of 1e-1 would fail!
    #             angle = 1e-2
    #             qml.RY(angle, wires=[0])
    #             return qml.expval(qml.PauliZ(wires=[0])), qml.expval(qml.PauliZ(wires=[1]))

    #         measurement = circuit()

    #         self.assertAllAlmostEqual(measurement, [1, 1], delta=1e-15)

    #         log.info("Outcome: %s", measurement)

    def test_backend_options(self):
        if self.devices is None:
            return
        self.logTestName()

        if self.args.device == 'basicaer' or self.args.device == 'all':

            def assertOptions(dev, backend_options):
                log.info("Asserting Backend Options for {}".format(dev.backend_name))

                @qml.qnode(dev)
                def circuit():
                    return qml.expval(qml.PauliZ(wires=[0])), qml.expval(qml.PauliZ(wires=[1]))

                circuit()

                for option, value in backend_options.items():
                    self.assertAllTrue(hasattr(dev.backend, "_{}".format(option)),
                                       "For Device {}/{} the attribute _{} wasn't found!"
                                       .format(type(dev), dev.backend_name, option))
                    other_value = getattr(dev.backend, "_{}".format(option))
                    if isinstance(value, (list, np.ndarray)):
                        self.assertAllEqual(value, other_value)
                    else:
                        self.assertEqual(value, other_value)

            all_backend_options = {
                'initial_unitary': np.array([
                    [np.sqrt(2), np.sqrt(2), 0, 0],
                    [np.sqrt(2), -np.sqrt(2), 0, 0],
                    [0, 0, np.sqrt(2), np.sqrt(2)],
                    [0, 0, np.sqrt(2), -np.sqrt(2)]
                ]) / 2,
                'chop_threshold': 1e-1
            }
            dev = BasicAerDevice(wires=self.num_subsystems, backend='unitary_simulator', **all_backend_options)
            assertOptions(dev, all_backend_options)

            all_backend_options = {
                'initial_statevector': np.array([0, 0, 1, 0]),
                'chop_threshold': 1e-1
            }
            dev = BasicAerDevice(wires=self.num_subsystems, backend='statevector_simulator', **all_backend_options)
            assertOptions(dev, all_backend_options)

            all_backend_options = {
                'initial_statevector': np.array([0, 0, 1, 0])
            }
            dev = BasicAerDevice(wires=self.num_subsystems, backend='qasm_simulator',
                                       **all_backend_options)
            assertOptions(dev, all_backend_options)

        if self.args.device == 'aer' or self.args.device == 'all':

            def assertOptions(dev, backend_options):
                log.info("Asserting Backend Options for {}/{}".format(type(dev), dev.backend_name))

                @qml.qnode(dev)
                def circuit():
                    return qml.expval(qml.PauliZ(wires=[0])), qml.expval(qml.PauliZ(wires=[1]))

                circuit()

                # first is the backend options, second noise model and third is validate bool
                dict, _, _ = dev._current_job._args
                for option, value in backend_options.items():
                    self.assertAllTrue(option in dict,
                                       "For Device {}/{} the key {} wasn't found!"
                                       .format(type(dev), dev.backend_name, option))
                    other_value = dict[option]
                    if isinstance(value, (list, np.ndarray)):
                        self.assertAllEqual(value, other_value)
                    else:
                        self.assertEqual(value, other_value)

            all_backend_options = {
                'initial_unitary': np.array([
                    [np.sqrt(2), np.sqrt(2), 0, 0],
                    [np.sqrt(2), -np.sqrt(2), 0, 0],
                    [0, 0, np.sqrt(2), np.sqrt(2)],
                    [0, 0, np.sqrt(2), -np.sqrt(2)]
                ]) / 2,
                'zero_threshold': 1e-1,  # default 1e-10
                'max_parallel_threads': 1,  # default 0
                'max_parallel_experiments': 2,  # default 1
                'max_memory_mb': 8192,  # default: 0
                'statevector_parallel_threshold': 16  # default: 14
            }
            dev = AerDevice(wires=self.num_subsystems, backend='unitary_simulator', **all_backend_options)
            assertOptions(dev, all_backend_options)

            all_backend_options = {
                'zero_threshold': 1e-1,  # default 1e-10
                'max_parallel_threads': 1,  # default 0
                'max_parallel_experiments': 2,  # default 1
                'max_memory_mb': 8192,  # default: 0
                'statevector_parallel_threshold': 16  # default: 14
            }
            dev = AerDevice(wires=self.num_subsystems, backend='statevector_simulator', **all_backend_options)
            assertOptions(dev, all_backend_options)

            all_backend_options = {
                'method': 'statevector',  # default. automatic
                'zero_threshold': 1e-1,  # default 1e-10
                'max_parallel_threads': 1,  # default 0
                'max_parallel_experiments': 2,  # default 1
                'max_memory_mb': 8192,  # default: 0
                'statevector_parallel_threshold': 16,  # default: 14
                'optimize_ideal_threshold': 4,  # default: 5
                'optimize_noise_threshold': 14,  # default: 14
                'statevector_sample_measure_opt': 11,  # default: 10
                'stabilizer_max_snapshot_probabilities': 16,  # default: 32
                'extended_stabilizer_measure_sampling': True,  # default: False
                'extended_stabilizer_mixing_time': 5001,  # default: 5000
                'extended_stabilizer_approximation_error': 0.051,  # default: 0.05
                'extended_stabilizer_norm_estimation_samples': 101,  # default: 100
                'extended_stabilizer_parallel_threshold': 101  # default: 100
            }
            dev = AerDevice(wires=self.num_subsystems, backend='qasm_simulator',
                                       **all_backend_options)
            assertOptions(dev, all_backend_options)
