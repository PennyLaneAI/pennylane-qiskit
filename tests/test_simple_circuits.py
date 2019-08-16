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
from pennylane_qiskit import BasicAerDevice, IBMQDevice, AerDevice
import pennylane_qiskit

from qiskit.providers.ibmq.exceptions import IBMQAccountError

import pytest

import os

log.getLogger('defaults')

num_wires = 3

aer_backends = ["statevector_simulator", "unitary_simulator", "qasm_simulator"]

devices = [BasicAerDevice(wires=num_wires, backend=b) for b in aer_backends]
devices += [AerDevice(wires=num_wires, backend=b) for b in aer_backends]

try:
    devices.append(IBMQDevice(wires=num_wires, shots=8*1024))
except IBMQAccountError:
    pass


@pytest.mark.parametrize("device", devices)
def test_only_single_qubit_unitary(device):
    """Tests if correct error is raised when attempting multi-qubit unitaries"""
    test_input = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    @qml.qnode(device)
    def circuit():
        qml.QubitUnitary(test_input, wires=[0])
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(ValueError,
                       match="Only a single qubit unitary is supported."):
        circuit()


@pytest.mark.parametrize("device", devices)
def test_non_unitary(device):
    """Tests if correct error is raised when attempting non-unitaries"""
    test_input = np.array([[1, 1], [1, 1]])
    @qml.qnode(device)
    def circuit():
        qml.QubitUnitary(test_input, wires=[0])
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(ValueError,
                       match="Not a unitary"):
        circuit()


class CompareWithDefaultQubitTest(BaseTest):
    """Compares the behavior of the Qiskit plugin devices with the default qubit device.
    """
    num_subsystems = 3  # This should be as large as the largest gate/observable, but we cannot know that before instantiating the device. We thus check later that all gates/observables fit.
    shots = 16 * 1024
    ibmq_shots = 8 * 1024
    devices = None

    def setUp(self):
        super().setUp()

        self.devices = [DefaultQubit(wires=self.num_subsystems, shots=0)]
        if self.args.device == 'basicaer' or self.args.device == 'all':
            self.devices.append(BasicAerDevice(wires=self.num_subsystems, shots=self.shots))
        if self.args.device == 'aer' or self.args.device == 'all':
            self.devices.append(AerDevice(wires=self.num_subsystems, shots=self.shots))
        if self.args.device == 'ibmq' or self.args.device == 'all':
            if self.args.ibmqx_token is not None:
                self.devices.append(
                    IBMQDevice(wires=self.num_subsystems, shots=self.ibmq_shots, ibmqx_token=self.args.ibmqx_token))
            else:
                log.warning("Skipping test of the IBMQDevice device because IBM login credentials "
                            "could not be found in the PennyLane configuration file.")

    def test_simple_circuits(self):
        """Automatically compare the behavior on simple circuits"""
        self.logTestName()

        class IgnoreOperationException(Exception):
            pass

        outputs = {}

        rnd_int_pool = np.random.randint(0, 5, 100)
        rnd_float_pool = np.random.randn(100)
        random_ket = np.random.uniform(-1, 1, 2 ** self.num_subsystems)
        random_ket = random_ket / np.linalg.norm(random_ket)
        random_zero_one_pool = np.random.randint(2, size=100)

        for dev in self.devices:

            # run all single operation circuits
            for operation in dev.operations:
                for observable in dev.observables:
                    log.info(
                        "Running device {} with a circuit consisting of a {} Operation followed by a {} Expectation".format(
                            dev.short_name, operation, observable))

                    @qml.qnode(dev)
                    def circuit():
                        if hasattr(qml, operation):
                            operation_class = getattr(qml, operation)
                        else:
                            operation_class = getattr(pennylane_qiskit, operation)

                        if hasattr(qml.ops, observable):
                            observable_class = getattr(qml.ops, observable)

                        if operation_class.num_wires > self.num_subsystems:
                            raise IgnoreOperationException(
                                'Skipping in automatic test because the operation ' + operation + " acts on more than the default number of wires " + str(
                                    self.num_subsystems) + ". Maybe you want to increase that?")
                        if observable_class.num_wires > self.num_subsystems:
                            raise IgnoreOperationException(
                                'Skipping in automatic test because the observable ' + observable + " acts on more than the default number of wires " + str(
                                    self.num_subsystems) + ". Maybe you want to increase that?")

                        # Operations
                        operation_wires = list(
                            range(operation_class.num_wires)) if operation_class.num_wires > 0 else list(
                            range(self.num_subsystems))

                        if operation_class.par_domain == 'N':
                            operation_pars = rnd_int_pool[:operation_class.num_params]
                        elif operation_class.par_domain == 'R':
                            operation_pars = np.abs(rnd_float_pool[
                                                    :operation_class.num_params])  # todo: some operations/observables fail when parameters are negative (e.g. thermal state) but par_domain is not fine grained enough to capture this
                        elif operation_class.par_domain == 'A':
                            if str(operation) == "QubitUnitary":
                                operation_pars = [np.array([[1, 0], [0, -1]])]
                                operation_wires = [0]
                            elif str(operation) == "QubitStateVector":
                                operation_pars = [np.array(random_ket)]
                            elif str(operation) == "BasisState":
                                operation_pars = [random_zero_one_pool[:self.num_subsystems]]
                            else:
                                raise IgnoreOperationException(
                                    'Skipping in automatic test because I don\'t know how to generate parameters for the operation ' + operation)
                        else:
                            operation_pars = {}

                        # Observables
                        observable_wires = list(
                            range(observable_class.num_wires)) if observable_class.num_wires > 1 else 0
                        if observable_class.par_domain == 'N':
                            observable_pars = rnd_int_pool[:observable_class.num_params]
                        elif observable_class.par_domain == 'R':
                            observable_pars = np.abs(rnd_float_pool[
                                                     :observable_class.num_params])  # todo: some operations/observables fail when parameters are negative (e.g. thermal state) but par_domain is not fine grained enough to capture this
                        elif observable_class.par_domain == 'A':
                            if str(observable) == "Hermitian":
                                observable_pars = [np.array([[1, 1j], [-1j, 0]])]
                            else:
                                raise IgnoreOperationException(
                                    'Skipping in automatic test because I don\'t know how to generate parameters for the observable ' + observable + " with par_domain=" + str(
                                        observable_class.par_domain))
                        else:
                            observable_pars = {}

                        # Apply operator and observable
                        operation_class(*operation_pars, wires=operation_wires)
                        return qml.expval(observable_class(*observable_pars, observable_wires))

                    output = circuit()
                    if (operation, observable) not in outputs:
                        outputs[(operation, observable)] = {}

                    device_key = str(type(dev).__name__) + "(shots=" + str(dev.shots) + ")"
                    outputs[(operation, observable)][device_key] = output

        # if we could run the circuit on more than one device assert that both should have given the same output
        for (key, val) in outputs.items():
            if len(val) >= 2:
                failed_message="Outputs {} of devices [{}] do not agree for a " \
                               "circuit consisting of a {} Operation followed " \
                               "by a {} Expectation.".format(
                    str(list(val.values())), ', '.join(list(val.keys())),
                    str(key[0]), str(key[1]))

                if 'DefaultQubit(shots=0)' not in val:
                    log.info("Operation %s followed by Expectation %s has no pendant in the DefaultQubit "
                             "device. Skipping.", key[0], key[1])
                    continue

                reference_output = val['DefaultQubit(shots=0)']
                self.assertAllAlmostEqual(val.values(), len(val) * [reference_output], delta=self.tol, msg=failed_message)



class SimpleCircuitsTest(BaseTest):
    """test the BasisState operation.
    """

    num_subsystems = 4
    shots = 16 * 1024
    ibmq_shots = 8 * 1024
    devices = None

    def setUp(self):
        super().setUp()

        self.devices = [DefaultQubit(wires=self.num_subsystems)]
        if self.args.device == 'basicaer' or self.args.device == 'all':
            self.devices.append(BasicAerDevice(wires=self.num_subsystems, shots=self.shots))
        if self.args.device == 'aer' or self.args.device == 'all':
            self.devices.append(AerDevice(wires=self.num_subsystems, shots=self.shots))
        if self.args.device == 'ibmq' or self.args.device == 'all':
            if IBMQX_TOKEN is not None:
                self.devices.append(
                    IBMQDevice(wires=self.num_subsystems, shots=self.ibmq_shots, ibmqx_token=IBMQX_TOKEN))
            else:
                log.warning("Skipping test of the IBMQDevice device because IBM login credentials could not be "
                            "found in the PennyLane configuration file.")

        properties = BackendProperties.from_dict({'backend_name': 'ibmqx4',
                                                                   'backend_version': '1.0.0',
                                                                   'gates': [{'gate': 'u1',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T09:57:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.0}],
                                                                              'qubits': [0]},
                                                                             {'gate': 'u2',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T09:57:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.0009443532335046134}],
                                                                              'qubits': [0]},
                                                                             {'gate': 'u3',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T09:57:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.0018887064670092268}],
                                                                              'qubits': [0]},
                                                                             {'gate': 'u1',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T09:57:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.0}],
                                                                              'qubits': [1]},
                                                                             {'gate': 'u2',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T09:57:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.0012019552727863259}],
                                                                              'qubits': [1]},
                                                                             {'gate': 'u3',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T09:57:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.0024039105455726517}],
                                                                              'qubits': [1]},
                                                                             {'gate': 'u1',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T09:57:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.0}],
                                                                              'qubits': [2]},
                                                                             {'gate': 'u2',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T09:57:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.0012019552727863259}],
                                                                              'qubits': [2]},
                                                                             {'gate': 'u3',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T09:57:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.0024039105455726517}],
                                                                              'qubits': [2]},
                                                                             {'gate': 'u1',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T09:57:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.0}],
                                                                              'qubits': [3]},
                                                                             {'gate': 'u2',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T09:57:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.0013737021608475342}],
                                                                              'qubits': [3]},
                                                                             {'gate': 'u3',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T09:57:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.0027474043216950683}],
                                                                              'qubits': [3]},
                                                                             {'gate': 'u1',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T09:57:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.0}],
                                                                              'qubits': [4]},
                                                                             {'gate': 'u2',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T09:57:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.001803112096824766}],
                                                                              'qubits': [4]},
                                                                             {'gate': 'u3',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T09:57:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.003606224193649532}],
                                                                              'qubits': [4]},
                                                                             {'gate': 'cx',
                                                                              'name': 'CX1_0',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T01:27:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.024311890455604945}],
                                                                              'qubits': [1, 0]},
                                                                             {'gate': 'cx',
                                                                              'name': 'CX2_0',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T01:32:39+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.023484363587478657}],
                                                                              'qubits': [2, 0]},
                                                                             {'gate': 'cx',
                                                                              'name': 'CX2_1',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T01:38:20+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.04885221406150694}],
                                                                              'qubits': [2, 1]},
                                                                             {'gate': 'cx',
                                                                              'name': 'CX3_2',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T01:44:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.06682678733530181}],
                                                                              'qubits': [3, 2]},
                                                                             {'gate': 'cx',
                                                                              'name': 'CX3_4',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T01:50:07+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.05217118636435464}],
                                                                              'qubits': [3, 4]},
                                                                             {'gate': 'cx',
                                                                              'name': 'CX4_2',
                                                                              'parameters': [
                                                                                  {'date': '2019-05-08T01:56:04+00:00',
                                                                                   'name': 'gate_error',
                                                                                   'unit': '',
                                                                                   'value': 0.06446497941268642}],
                                                                              'qubits': [4, 2]}],
                                                                   'general': [],
                                                                   'last_update_date': '2019-05-08T01:56:04+00:00',
                                                                   'qconsole': False,
                                                                   'qubits': [[{'date': '2019-05-08T01:16:56+00:00',
                                                                                'name': 'T1',
                                                                                'unit': 'µs',
                                                                                'value': 43.21767480545737},
                                                                               {'date': '2019-05-08T01:17:40+00:00',
                                                                                'name': 'T2',
                                                                                'unit': 'µs',
                                                                                'value': 19.77368032971812},
                                                                               {'date': '2019-05-08T01:56:04+00:00',
                                                                                'name': 'frequency',
                                                                                'unit': 'GHz',
                                                                                'value': 5.246576101635769},
                                                                               {'date': '2019-05-08T01:16:37+00:00',
                                                                                'name': 'readout_error',
                                                                                'unit': '',
                                                                                'value': 0.08650000000000002}],
                                                                              [{'date': '2019-05-08T01:16:56+00:00',
                                                                                'name': 'T1',
                                                                                'unit': 'µs',
                                                                                'value': 43.87997000828745},
                                                                               {'date': '2019-05-08T01:18:27+00:00',
                                                                                'name': 'T2',
                                                                                'unit': 'µs',
                                                                                'value': 11.390521028550571},
                                                                               {'date': '2019-05-08T01:56:04+00:00',
                                                                                'name': 'frequency',
                                                                                'unit': 'GHz',
                                                                                'value': 5.298309751315148},
                                                                               {'date': '2019-05-08T01:16:37+00:00',
                                                                                'name': 'readout_error',
                                                                                'unit': '',
                                                                                'value': 0.07999999999999996}],
                                                                              [{'date': '2019-05-07T09:14:18+00:00',
                                                                                'name': 'T1',
                                                                                'unit': 'µs',
                                                                                'value': 48.97128225850014},
                                                                               {'date': '2019-05-08T01:19:07+00:00',
                                                                                'name': 'T2',
                                                                                'unit': 'µs',
                                                                                'value': 31.06845465651204},
                                                                               {'date': '2019-05-08T01:56:04+00:00',
                                                                                'name': 'frequency',
                                                                                'unit': 'GHz',
                                                                                'value': 5.3383288291854765},
                                                                               {'date': '2019-05-08T01:16:37+00:00',
                                                                                'name': 'readout_error',
                                                                                'unit': '',
                                                                                'value': 0.038250000000000006}],
                                                                              [{'date': '2019-05-08T01:16:56+00:00',
                                                                                'name': 'T1',
                                                                                'unit': 'µs',
                                                                                'value': 38.30486582843196},
                                                                               {'date': '2019-05-08T01:18:27+00:00',
                                                                                'name': 'T2',
                                                                                'unit': 'µs',
                                                                                'value': 32.35546811356613},
                                                                               {'date': '2019-05-08T01:56:04+00:00',
                                                                                'name': 'frequency',
                                                                                'unit': 'GHz',
                                                                                'value': 5.426109336844823},
                                                                               {'date': '2019-05-08T01:16:37+00:00',
                                                                                'name': 'readout_error',
                                                                                'unit': '',
                                                                                'value': 0.35675}],
                                                                              [{'date': '2019-05-08T01:16:56+00:00',
                                                                                'name': 'T1',
                                                                                'unit': 'µs',
                                                                                'value': 36.02606265575505},
                                                                               {'date': '2019-05-07T09:15:02+00:00',
                                                                                'name': 'T2',
                                                                                'unit': 'µs',
                                                                                'value': 4.461644223370699},
                                                                               {'date': '2019-05-08T01:56:04+00:00',
                                                                                'name': 'frequency',
                                                                                'unit': 'GHz',
                                                                                'value': 5.174501299220437},
                                                                               {'date': '2019-05-08T01:16:37+00:00',
                                                                                'name': 'readout_error',
                                                                                'unit': '',
                                                                                'value': 0.2715000000000001}]]})

        self.noise_model = noise.device.basic_device_noise_model(properties)

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
                    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3))

                self.assertAllAlmostEqual([1] * self.num_subsystems - 2 * bits_to_flip, np.array(circuit()),
                                          delta=self.tol)

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
                return qml.expval(qml.PauliZ(wires=1))

            self.assertAllAlmostEqual(0.96875, circuit(0.2, 0.1, 0.3), delta=self.tol)

    def test_arbitrary_state(self):
        """Test BasisState with preparations on the whole system."""
        if self.devices is None:
            return
        self.logTestName()

        for device in self.devices:
            for index in range(16):
                state = np.array(16 * [0.0])
                state[index] = 1.0

                @qml.qnode(device)
                def circuit():
                    qml.QubitStateVector(state, wires=[0, 1, 2, 3])
                    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3))

                result = np.array(circuit())
                expected = np.array(list(map(lambda c: 1.0 if c == '0' else -1.0, "{:b}".format(index).zfill(self.num_subsystems))))
                self.assertAllAlmostEqual(expected, result, delta=self.tol)

    def test_arbitrary_unitary(self):
        """Test BasisState with preparations on the whole system."""
        if self.devices is None:
            return
        self.logTestName()

        for device in self.devices:
            test_input = [
                np.array([[1, 0], [0, 1]]),
                1 / math.sqrt(2) * np.array([
                    [1, -cmath.exp(1.0j * cmath.pi / 2)],
                    [cmath.exp(1.0j * cmath.pi / 4), cmath.exp(1.0j * (cmath.pi / 2 + cmath.pi / 4))]
                ]),
                np.array([[1, 0], [0, cmath.exp(1.0j * cmath.pi / 4)]])
            ]
            for i in test_input:
                @qml.qnode(device)
                def circuit():
                    qml.QubitUnitary(i, wires=[0])
                    return qml.expval(qml.PauliZ(0))

                circuit()
                # TODO 2018-12-23 Carsten Blank: create meaningful tests

    def test_basis_state_noise_aer(self):

        for device in self.devices:
            if isinstance(device, AerDevice):
                device = AerDevice(wires=device.num_wires, noise_model=self.noise_model, shots=device.shots)
            if isinstance(device, BasicAerDevice):
                device = BasicAerDevice(wires=device.num_wires, noise_model=self.noise_model, shots=device.shots)

            for bits_to_flip in [np.array([0, 0, 0, 0]),
                                 np.array([0, 1, 1, 0]),
                                 np.array([1, 1, 1, 0]),
                                 np.array([1, 1, 1, 1])]:
                @qml.qnode(device)
                def circuit():
                    for i, p in enumerate(bits_to_flip):
                        if p == 1:
                            qml.PauliX(wires=[i])
                    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3))
                # The assert is almost useless. Depending on the current noise model, the expectation values might be
                # very different than what we expect. I guess it would be pretty much the best practice to assume
                # that it is at least more likely to measure a 1 than a 0, hence we need to make sure that the
                # expectation value is (strictly) greater than 0.
                self.assertAllAlmostEqual([1] * self.num_subsystems - 2 * bits_to_flip, np.array(circuit()),
                                          delta=0.99)  # change delta tolerance if test fails due to the delta error


if __name__ == '__main__':
    print('Testing PennyLane qiskit Plugin version ' + qml.version() + ', BasisState operation.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (SimpleCircuitsTest, CompareWithDefaultQubitTest):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
