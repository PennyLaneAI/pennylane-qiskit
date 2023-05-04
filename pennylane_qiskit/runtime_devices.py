# Copyright 2021-2022 Xanadu Quantum Technologies Inc.

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
This module contains classes for constructing Qiskit runtime devices for PennyLane.
"""
# pylint: disable=attribute-defined-outside-init, protected-access, arguments-renamed

import numpy as np

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.constants import RunnerResult
from pennylane_qiskit.ibmq import IBMQDevice


class IBMQCircuitRunnerDevice(IBMQDevice):
    r"""Class for a Qiskit runtime circuit-runner program device in PennyLane. Circuit runner is a
    runtime program that takes one or more circuits, compiles them, executes them, and optionally
    applies measurement error mitigation.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        provider (Provider): The Qiskit simulation provider
        backend (str): the desired backend
        shots (int): Number of circuit evaluations/random samples used to estimate expectation values and variances of
         observables. Default=1024.

    Keyword Args:
        initial_layout (array[int]): Initial position of virtual qubits on physical qubits.
        layout_method (string): Name of layout selection pass ('trivial', 'dense', 'noise_adaptive', 'sabre')
        routing_method (string): Name of routing pass ('basic', 'lookahead', 'stochastic', 'sabre').
        translation_method (string): Name of translation pass ('unroller', 'translator', 'synthesis').
        seed_transpiler (int): Sets random seed for the stochastic parts of the transpiler.
        optimization_level (int): How much optimization to perform on the circuits (0-3). Higher levels generate more
         optimized circuits. Default is 1.
        init_qubits (bool): Whether to reset the qubits to the ground state for each shot.
        rep_delay (float): Delay between programs in seconds.
        transpiler_options (dict): Additional compilation options.
        measurement_error_mitigation (bool): Whether to apply measurement error mitigation. Default is False.
    """

    short_name = "qiskit.ibmq.circuit_runner"

    def __init__(self, wires, provider=None, backend="ibmq_qasm_simulator", shots=1024, **kwargs):
        self.kwargs = kwargs
        super().__init__(wires=wires, provider=provider, backend=backend, shots=shots, **kwargs)
        self.runtime_service = QiskitRuntimeService(channel="ibm_quantum")

    def batch_execute(self, circuits):
        compiled_circuits = self.compile_circuits(circuits)

        program_inputs = {"circuits": compiled_circuits, "shots": self.shots}

        for kwarg in self.kwargs:
            program_inputs[kwarg] = self.kwargs.get(kwarg)

        # Specify the backend.
        options = {"backend": self.backend.name}

        # Send circuits to the cloud for execution by the circuit-runner program.
        job = self.runtime_service.run(
            program_id="circuit-runner", options=options, inputs=program_inputs
        )
        self._current_job = job.result(decoder=RunnerResult)

        results = []

        for index, circuit in enumerate(circuits):
            self._samples = self.generate_samples(index)
            res = self.statistics(circuit)
            single_measurement = len(circuit.measurements) == 1
            res = res[0] if single_measurement else tuple(res)
            results.append(res)

        if self.tracker.active:
            job_time = {
                "total_time": self._current_job._metadata.get("time_taken"),
            }
            self.tracker.update(batches=1, batch_len=len(circuits), job_time=job_time)
            self.tracker.record()

        return results

    def generate_samples(self, circuit=None):
        r"""Returns the computational basis samples generated for all wires.

        Note that PennyLane uses the convention :math:`|q_0,q_1,\dots,q_{N-1}\rangle` where
        :math:`q_0` is the most significant bit.

        Args:
            circuit (int): position of the circuit in the batch.

        Returns:
             array[complex]: array of samples in the shape ``(dev.shots, dev.num_wires)``
        """
        counts = self._current_job.get_counts()
        # Batch of circuits
        if not isinstance(counts, dict):
            counts = self._current_job.get_counts()[circuit]

        samples = []
        for key, value in counts.items():
            for _ in range(0, value):
                samples.append(key)
        return np.vstack([np.array([int(i) for i in s[::-1]]) for s in samples])


class IBMQSamplerDevice(IBMQDevice):
    r"""Class for a Qiskit runtime sampler program device in PennyLane. Sampler is a Qiskit runtime program
    that samples distributions generated by given circuits executed on the target backend.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        provider (Provider): the Qiskit simulation provider
        backend (str): the desired backend
        shots (int or None): Number of circuit evaluations/random samples used
            to estimate expectation values and variances of observables. Default=1024.

    Keyword Args:
        circuit_indices (bool): Indices of the circuits to evaluate. Default is ``range(0, len(circuits))``.
        run_options (dict): A collection of kwargs passed to backend.run, if shots are given here it will take
            precedence over the shots arg.
        skip_transpilation (bool): Skip circuit transpilation. Default is False.
    """

    short_name = "qiskit.ibmq.sampler"

    def __init__(self, wires, provider=None, backend="ibmq_qasm_simulator", shots=1024, **kwargs):
        self.kwargs = kwargs
        super().__init__(wires=wires, provider=provider, backend=backend, shots=shots, **kwargs)
        self.runtime_service = QiskitRuntimeService(channel="ibm_quantum")

    def batch_execute(self, circuits):
        compiled_circuits = self.compile_circuits(circuits)

        program_inputs = {"circuits": compiled_circuits}

        if "circuits_indices" not in self.kwargs:
            circuit_indices = list(range(len(compiled_circuits)))
            program_inputs["circuit_indices"] = circuit_indices
        else:
            circuit_indices = self.kwargs.get("circuit_indices")

        if "run_options" in self.kwargs:
            if "shots" not in self.kwargs["run_options"]:
                self.kwargs["run_options"]["shots"] = self.shots
        else:
            self.kwargs["run_options"] = {"shots": self.shots}

        for kwarg in self.kwargs:
            program_inputs[kwarg] = self.kwargs.get(kwarg)

        # Specify the backend.
        options = {"backend": self.backend.name}
        # Send circuits to the cloud for execution by the sampler program.
        job = self.runtime_service.run(program_id="sampler", options=options, inputs=program_inputs)
        self._current_job = job.result()

        results = []

        counter = 0
        for index, circuit in enumerate(circuits):
            if index in circuit_indices:
                self._samples = self.generate_samples(counter)
                counter += 1
                res = self.statistics(circuit)
                single_measurement = len(circuit.measurements) == 1
                res = res[0] if single_measurement else tuple(res)
                results.append(res)

        if self.tracker.active:
            self.tracker.update(batches=1, batch_len=len(circuits))
            self.tracker.record()

        return results

    def generate_samples(self, circuit_id=None):
        r"""Returns the computational basis samples generated for all wires.

        Note that PennyLane uses the convention :math:`|q_0,q_1,\dots,q_{N-1}\rangle` where
        :math:`q_0` is the most significant bit.

        Args:
            circuit_id (int): position of the circuit in the batch.

        Returns:
             array[complex]: array of samples in the shape ``(dev.shots, dev.num_wires)``
        """
        counts = self._current_job.quasi_dists[circuit_id].binary_probabilities()
        keys = list(counts.keys())

        number_of_states = 2 ** len(keys[0])

        # Convert state to int
        for i, elem in enumerate(keys):
            keys[i] = int(elem, 2)

        values = list(counts.values())
        states, probs = zip(*sorted(zip(keys, values)))

        states = list(states)
        probs = list(probs)

        # If prob for a state is 0, it does not appear in counts.
        if len(states) != number_of_states:
            for i in range(0, number_of_states):
                if states[i] != i:
                    states.insert(i, i)
                    probs.insert(i, 0.0)

        return self.states_to_binary(
            self.sample_basis_states(number_of_states, probs), self.num_wires
        )
