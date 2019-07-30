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
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.models import BackendProperties

from defaults import pennylane as qml
from defaults import ARGS as args
from pennylane_qiskit import IbmQQiskitDevice, AerQiskitDevice, BasicAerQiskitDevice

import pytest


num_subsystems = 4
devices = None


def test_ibm_no_token():
    # if there is an IBMQX token, save it and unset it so that it doesn't interfere with this test
    token_from_environment = os.getenv("IBMQX_TOKEN")
    if token_from_environment is not None:
        del os.environ["IBMQX_TOKEN"]

    if args.device == "ibmq" or args.device == "all":
        pytest.raises(
            ValueError,
            IbmQQiskitDevice,
            wires=num_subsystems,
            msg="Expected a ValueError if no IBMQX token is present.",
        )

    # put the IBMQX token back into place for other tests to use
    if token_from_environment is not None:
        os.environ["IBMQX_TOKEN"] = token_from_environment
        token_from_environment_back = os.getenv("IBMQX_TOKEN")
        assert token_from_environment == token_from_environment_back


def test_shots():
    if args.device == "ibmq" or args.device == "all":

        if args.ibmqx_token is None:
            print(
                "Skipping test of the IbmQQiskitDevice device because IBM login credentials could not be "
                "found in the PennyLane configuration file."
            )
            return

        shots = 5
        dev1 = IbmQQiskitDevice(
            wires=num_subsystems, shots=shots, ibmqx_token=args.ibmqx_token
        )
        assert shots == dev1.shots


def test_noise_model_for_aer():
    try:
        noise_model = _get_noise_model()  # type: NoiseModel

        dev = qml.device("qiskit.aer", wires=num_subsystems, noise_model=noise_model)
        assert dev._noise_model is not None
        assert noise_model.to_dict() == dev._noise_model.to_dict()

        dev2 = AerQiskitDevice(wires=num_subsystems, noise_model=noise_model)
        assert dev2._noise_model is not None
        assert noise_model.to_dict() == dev2._noise_model.to_dict()

    except DeviceError:
        raise Exception(
            "This test is expected to fail until pennylane-qiskit is installed."
        )


def test_noise_model_for_basic_aer():
    try:
        noise_model = _get_noise_model()

        dev = qml.device(
            "qiskit.basicaer", wires=num_subsystems, noise_model=noise_model
        )
        assert dev._noise_model is not None
        assert noise_model.to_dict() == dev._noise_model.to_dict()

        dev2 = BasicAerQiskitDevice(wires=num_subsystems, noise_model=noise_model)
        assert dev2._noise_model is not None
        assert noise_model.to_dict() == dev2._noise_model.to_dict()

    except DeviceError:
        raise Exception(
            "This test is expected to fail until pennylane-qiskit is installed."
        )


def _get_noise_model():
    properties = BackendProperties.from_dict(
        {
            "backend_name": "ibmqx4",
            "backend_version": "1.0.0",
            "gates": [
                {
                    "gate": "u1",
                    "parameters": [
                        {
                            "date": "2019-05-08T09:57:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.0,
                        }
                    ],
                    "qubits": [0],
                },
                {
                    "gate": "u2",
                    "parameters": [
                        {
                            "date": "2019-05-08T09:57:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.0009443532335046134,
                        }
                    ],
                    "qubits": [0],
                },
                {
                    "gate": "u3",
                    "parameters": [
                        {
                            "date": "2019-05-08T09:57:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.0018887064670092268,
                        }
                    ],
                    "qubits": [0],
                },
                {
                    "gate": "u1",
                    "parameters": [
                        {
                            "date": "2019-05-08T09:57:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.0,
                        }
                    ],
                    "qubits": [1],
                },
                {
                    "gate": "u2",
                    "parameters": [
                        {
                            "date": "2019-05-08T09:57:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.0012019552727863259,
                        }
                    ],
                    "qubits": [1],
                },
                {
                    "gate": "u3",
                    "parameters": [
                        {
                            "date": "2019-05-08T09:57:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.0024039105455726517,
                        }
                    ],
                    "qubits": [1],
                },
                {
                    "gate": "u1",
                    "parameters": [
                        {
                            "date": "2019-05-08T09:57:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.0,
                        }
                    ],
                    "qubits": [2],
                },
                {
                    "gate": "u2",
                    "parameters": [
                        {
                            "date": "2019-05-08T09:57:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.0012019552727863259,
                        }
                    ],
                    "qubits": [2],
                },
                {
                    "gate": "u3",
                    "parameters": [
                        {
                            "date": "2019-05-08T09:57:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.0024039105455726517,
                        }
                    ],
                    "qubits": [2],
                },
                {
                    "gate": "u1",
                    "parameters": [
                        {
                            "date": "2019-05-08T09:57:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.0,
                        }
                    ],
                    "qubits": [3],
                },
                {
                    "gate": "u2",
                    "parameters": [
                        {
                            "date": "2019-05-08T09:57:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.0013737021608475342,
                        }
                    ],
                    "qubits": [3],
                },
                {
                    "gate": "u3",
                    "parameters": [
                        {
                            "date": "2019-05-08T09:57:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.0027474043216950683,
                        }
                    ],
                    "qubits": [3],
                },
                {
                    "gate": "u1",
                    "parameters": [
                        {
                            "date": "2019-05-08T09:57:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.0,
                        }
                    ],
                    "qubits": [4],
                },
                {
                    "gate": "u2",
                    "parameters": [
                        {
                            "date": "2019-05-08T09:57:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.001803112096824766,
                        }
                    ],
                    "qubits": [4],
                },
                {
                    "gate": "u3",
                    "parameters": [
                        {
                            "date": "2019-05-08T09:57:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.003606224193649532,
                        }
                    ],
                    "qubits": [4],
                },
                {
                    "gate": "cx",
                    "name": "CX1_0",
                    "parameters": [
                        {
                            "date": "2019-05-08T01:27:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.024311890455604945,
                        }
                    ],
                    "qubits": [1, 0],
                },
                {
                    "gate": "cx",
                    "name": "CX2_0",
                    "parameters": [
                        {
                            "date": "2019-05-08T01:32:39+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.023484363587478657,
                        }
                    ],
                    "qubits": [2, 0],
                },
                {
                    "gate": "cx",
                    "name": "CX2_1",
                    "parameters": [
                        {
                            "date": "2019-05-08T01:38:20+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.04885221406150694,
                        }
                    ],
                    "qubits": [2, 1],
                },
                {
                    "gate": "cx",
                    "name": "CX3_2",
                    "parameters": [
                        {
                            "date": "2019-05-08T01:44:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.06682678733530181,
                        }
                    ],
                    "qubits": [3, 2],
                },
                {
                    "gate": "cx",
                    "name": "CX3_4",
                    "parameters": [
                        {
                            "date": "2019-05-08T01:50:07+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.05217118636435464,
                        }
                    ],
                    "qubits": [3, 4],
                },
                {
                    "gate": "cx",
                    "name": "CX4_2",
                    "parameters": [
                        {
                            "date": "2019-05-08T01:56:04+00:00",
                            "name": "gate_error",
                            "unit": "",
                            "value": 0.06446497941268642,
                        }
                    ],
                    "qubits": [4, 2],
                },
            ],
            "general": [],
            "last_update_date": "2019-05-08T01:56:04+00:00",
            "qconsole": False,
            "qubits": [
                [
                    {
                        "date": "2019-05-08T01:16:56+00:00",
                        "name": "T1",
                        "unit": "µs",
                        "value": 43.21767480545737,
                    },
                    {
                        "date": "2019-05-08T01:17:40+00:00",
                        "name": "T2",
                        "unit": "µs",
                        "value": 19.77368032971812,
                    },
                    {
                        "date": "2019-05-08T01:56:04+00:00",
                        "name": "frequency",
                        "unit": "GHz",
                        "value": 5.246576101635769,
                    },
                    {
                        "date": "2019-05-08T01:16:37+00:00",
                        "name": "readout_error",
                        "unit": "",
                        "value": 0.08650000000000002,
                    },
                ],
                [
                    {
                        "date": "2019-05-08T01:16:56+00:00",
                        "name": "T1",
                        "unit": "µs",
                        "value": 43.87997000828745,
                    },
                    {
                        "date": "2019-05-08T01:18:27+00:00",
                        "name": "T2",
                        "unit": "µs",
                        "value": 11.390521028550571,
                    },
                    {
                        "date": "2019-05-08T01:56:04+00:00",
                        "name": "frequency",
                        "unit": "GHz",
                        "value": 5.298309751315148,
                    },
                    {
                        "date": "2019-05-08T01:16:37+00:00",
                        "name": "readout_error",
                        "unit": "",
                        "value": 0.07999999999999996,
                    },
                ],
                [
                    {
                        "date": "2019-05-07T09:14:18+00:00",
                        "name": "T1",
                        "unit": "µs",
                        "value": 48.97128225850014,
                    },
                    {
                        "date": "2019-05-08T01:19:07+00:00",
                        "name": "T2",
                        "unit": "µs",
                        "value": 31.06845465651204,
                    },
                    {
                        "date": "2019-05-08T01:56:04+00:00",
                        "name": "frequency",
                        "unit": "GHz",
                        "value": 5.3383288291854765,
                    },
                    {
                        "date": "2019-05-08T01:16:37+00:00",
                        "name": "readout_error",
                        "unit": "",
                        "value": 0.038250000000000006,
                    },
                ],
                [
                    {
                        "date": "2019-05-08T01:16:56+00:00",
                        "name": "T1",
                        "unit": "µs",
                        "value": 38.30486582843196,
                    },
                    {
                        "date": "2019-05-08T01:18:27+00:00",
                        "name": "T2",
                        "unit": "µs",
                        "value": 32.35546811356613,
                    },
                    {
                        "date": "2019-05-08T01:56:04+00:00",
                        "name": "frequency",
                        "unit": "GHz",
                        "value": 5.426109336844823,
                    },
                    {
                        "date": "2019-05-08T01:16:37+00:00",
                        "name": "readout_error",
                        "unit": "",
                        "value": 0.35675,
                    },
                ],
                [
                    {
                        "date": "2019-05-08T01:16:56+00:00",
                        "name": "T1",
                        "unit": "µs",
                        "value": 36.02606265575505,
                    },
                    {
                        "date": "2019-05-07T09:15:02+00:00",
                        "name": "T2",
                        "unit": "µs",
                        "value": 4.461644223370699,
                    },
                    {
                        "date": "2019-05-08T01:56:04+00:00",
                        "name": "frequency",
                        "unit": "GHz",
                        "value": 5.174501299220437,
                    },
                    {
                        "date": "2019-05-08T01:16:37+00:00",
                        "name": "readout_error",
                        "unit": "",
                        "value": 0.2715000000000001,
                    },
                ],
            ],
        }
    )

    return noise.device.basic_device_noise_model(properties)


def test_initialization_via_pennylane():
    for short_name in ["qiskit.aer", "qiskit.basicaer", "qiskit.ibmq"]:
        try:
            if args.ibmqx_token is None:
                print(
                    "Device initialization cannot be tested as there is no IBMQX token configured. "
                    "The test concerning the ibmq will be skipped."
                )
            else:
                qml.device(short_name, wires=2, ibmqx_token=args.ibmqx_token)
        except DeviceError:
            raise Exception(
                "This test is expected to fail until pennylane-qiskit is installed."
            )


def test_ibm_device():
    if args.device in ["ibmq", "all"]:

        if args.ibmqx_token is None:
            print(
                "Skipping test of the IbmQQiskitDevice device because IBM login credentials could not be "
                "found in the PennyLane configuration file."
            )
            return

        import qiskit

        qiskit.IBMQ.enable_account(token=args.ibmqx_token)
        backends = qiskit.IBMQ.backends()
        qiskit.IBMQ.disable_accounts()
        try:
            for backend in backends:
                qml.device(
                    "qiskit.ibmq",
                    wires=1,
                    ibmqx_token=args.ibmqx_token,
                    backend=backend,
                )
        except DeviceError:
            raise Exception(
                "This test is expected to fail until pennylane-qiskit is installed."
            )


def test_aer_device():
    if args.device in ["aer", "all"]:
        import qiskit

        try:
            for backend in qiskit.Aer.backends():
                qml.device("qiskit.aer", wires=1, backend=backend)
        except DeviceError:
            raise Exception(
                "This test is expected to fail until pennylane-qiskit is installed."
            )


def test_basicaer_device():
    if args.device in ["basicaer", "all"]:
        import qiskit

        try:
            for backend in qiskit.BasicAer.backends():
                qml.device("qiskit.basicaer", wires=1, backend=backend)
        except DeviceError:
            raise Exception(
                "This test is expected to fail until pennylane-qiskit is installed."
            )
