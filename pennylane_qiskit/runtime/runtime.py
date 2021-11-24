# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
This module contains classes for constructing Qiskit runtime devices for PennyLane.
"""

import numpy as np

import qiskit.result.postprocess
from qiskit.providers.ibmq import RunnerResult
from pennylane_qiskit.ibmq import IBMQDevice


class IBMQCircuitRunnerDevice(IBMQDevice):
    r"""Class for a Qiskit runtime circuit-runner program device in PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        provider (Provider): The Qiskit simulation provider
        backend (str): the desired backend
        shots (int or None): number of circuit evaluations/random samples used
            to estimate expectation values and variances of observables. For statevector backends,
            setting to ``None`` results in computing statistics like expectation values and variances analytically.

    Keyword Args:
        initial_layout:
        layout_method:
        routing_method:
        translation_method:
        seed_transpiler:
        optimization_level:
        init_qubits:
        rep_delay:
        transpiler_options:
        measurement_error_mitigation:
    """

    short_name = "qiskit.ibmq.circuitrunner"

    def __init__(self, wires, provider=None, backend="ibmq_qasm_simulator", shots=1024, **kwargs):
        self.kwargs = kwargs
        super().__init__(wires=wires, provider=provider, backend=backend, shots=shots, **kwargs)

    def generate_samples(self, circuit=None):
        counts = self._current_job.get_counts()[circuit]
        samples = []
        for key, value in counts.items():
            for i in range(0, value):
                samples.append(key)
        return np.vstack([np.array([int(i) for i in s[::-1]]) for s in samples])

    def batch_execute(self, circuits):
        compiled_circuits = []

        # Compile each circuit object
        for circuit in circuits:
            # We need to reset the device here, else it will
            # not start the next computation in the zero state
            self.reset()
            self.create_circuit_object(circuit.operations, rotations=circuit.diagonalizing_gates)

            compiled_circ = self.compile()
            compiled_circ.name = f"circ{len(compiled_circuits)}"
            compiled_circuits.append(compiled_circ)

        program_inputs = {"circuits": compiled_circuits, "shots": self.shots}

        # Initial position of virtual qubits
        # on physical qubits.
        if self.kwargs.get("initial_layout"):
            program_inputs["initial_layout"] = self.kwargs.get("initial_layout")

        # Name of layout selection pass
        # ('trivial', 'dense', 'noise_adaptive', 'sabre')
        if self.kwargs.get("layout_method"):
            program_inputs["layout_method"] = self.kwargs.get("layout_method")

        # Name of routing pass ('basic',
        # 'lookahead', 'stochastic', 'sabre').
        if self.kwargs.get("routing_method"):
            program_inputs["routing_method"] = self.kwargs.get("routing_method")  # string

        # Name of translation pass ('unroller',
        # 'translator', 'synthesis').
        if self.kwargs.get("translation_method"):
            program_inputs["translation_method"] = self.kwargs.get("translation_method")  # string

        # Sets random seed for the
        # stochastic parts of the transpiler.
        if self.kwargs.get("seed_transpiler"):
            program_inputs["seed_transpiler"] = self.kwargs.get("seed_transpiler")  # int

        # How much optimization to perform
        # on the circuits (0-3). Higher
        # levels generate more optimized circuits.
        # Default is 1.
        program_inputs["optimization_level"] = self.kwargs.get("optimization_level", 1)  # int

        # Whether to reset the qubits
        # to the ground state for
        # each shot.
        if self.kwargs.get("init_qubits"):
            program_inputs["init_qubits"] = self.kwargs.get("init_qubits")  # bool

        # Delay between programs in seconds.
        if self.kwargs.get("rep_delay"):
            program_inputs["rep_delay"] = self.kwargs.get("rep_delay")  # float

        # Additional compilation options.
        if self.kwargs.get("transpiler_options"):
            program_inputs["transpiler_options"] = self.kwargs.get("transpiler_options")  # dict

        # Whether to apply measurement error
        # mitigation. Default is False.
        program_inputs["measurement_error_mitigation"] = self.kwargs.get(
            "measurement_error_mitigation", False
        )  # bool

        # Specify the backend.
        options = {"backend_name": self.backend.name()}

        # Send circuits to the cloud for execution by the circuit-runner program.
        job = self.provider.runtime.run(
            program_id="circuit-runner", options=options, inputs=program_inputs
        )
        self._current_job = job.result(decoder=RunnerResult)

        results = []

        index = 0
        for circuit, circuit_obj in zip(circuits, compiled_circuits):

            self._samples = self.generate_samples(index)
            index += 1
            res = self.statistics(circuit.observables)
            results.append(res)

        return results


class IBMQSamplerDevice(IBMQDevice):
    r"""Class for a Qiskit runtine circuit-runner program device in PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        provider (Provider): The Qiskit simulation provider
        backend (str): the desired backend
        shots (int or None): number of circuit evaluations/random samples used
            to estimate expectation values and variances of observables. For statevector backends,
            setting to ``None`` results in computing statistics like expectation values and variances analytically.

    Keyword Args:
        initial_layout:
        layout_method:
        routing_method:
        translation_method:
        seed_transpiler:
        optimization_level:
        init_qubits:
        rep_delay:
        transpiler_options:
        measurement_error_mitigation:
    """

    short_name = "qiskit.ibmq.sampler"

    def __init__(self, wires, provider=None, backend="ibmq_qasm_simulator", shots=1024, **kwargs):
        self.kwargs = kwargs
        super().__init__(wires=wires, provider=provider, backend=backend, shots=shots, **kwargs)

    def batch_execute(self, circuits):
        compiled_circuits = []

        # Compile each circuit object
        for circuit in circuits:
            # We need to reset the device here, else it will
            # not start the next computation in the zero state
            self.reset()
            self.create_circuit_object(circuit.operations, rotations=circuit.diagonalizing_gates)

            compiled_circ = self.compile()
            compiled_circ.name = f"circ{len(compiled_circuits)}"
            compiled_circuits.append(compiled_circ)

        program_inputs = {"circuits": compiled_circuits}
        # Return mitigation overhead factor. Default
        # is False.
        if self.kwargs.get("return_mitigation_overhead"):
            program_inputs["return_mitigation_overhead"] = self.kwargs.get(
                "return_mitigation_overhead"
            )  # boolean

        # A collection of kwargs passed
        # to backend.run.
        if self.kwargs.get("run_config"):
            program_inputs["run_config"] = self.kwargs.get("run_config")  # object

        # Skip circuit transpilation. Default is
        # False.
        if self.kwargs.get("skip_transpilation"):
            program_inputs["skip_transpilation"] = False  # boolean

        # A collection of kwargs passed
        # to transpile.
        if self.kwargs.get("transpile_config"):
            program_inputs["transpile_config"] = self.kwargs.get("transpile_config")  # object

        # Use measurement mitigation to improve
        # results. Default is False.
        if self.kwargs.get("use_measurement_mitigation"):
            program_inputs["use_measurement_mitigation"] = self.kwargs.get(
                "use_measurement_mitigation"
            )  # boolean

        # Specify the backend.
        options = {"backend_name": self.backend.name()}
        # Send circuits to the cloud for execution by the circuit-runner program.
        job = self.provider.runtime.run(
            program_id="sampler", options=options, inputs=program_inputs
        )
        self._current_job = job.result()
        results = []

        index = 0
        for circuit, circuit_obj in zip(circuits, compiled_circuits):
            self._samples = self.generate_samples(index)
            index += 1
            res = self.statistics(circuit.observables)
            results.append(res)

        return results

    def generate_samples(self, circuit=None):
        counts = qiskit.result.postprocess.format_counts(
            self._current_job.get("counts")[circuit], {"memory_slots": self._circuit.num_qubits}
        )
        samples = []
        for key, value in counts.items():
            for i in range(0, value):
                samples.append(key)
        return np.vstack([np.array([int(i) for i in s[::-1]]) for s in samples])
