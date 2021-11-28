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
# pylint: disable=attribute-defined-outside-init
# pylint: disable=protected-access

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
        shots (int): number of circuit evaluations/random samples used to estimate expectation values and variances of
         observables.

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

    short_name = "qiskit.ibmq.circuitrunner"

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

        program_inputs = {"circuits": compiled_circuits, "shots": self.shots}

        if self.kwargs.get("initial_layout"):
            program_inputs["initial_layout"] = self.kwargs.get("initial_layout")

        if self.kwargs.get("layout_method"):
            program_inputs["layout_method"] = self.kwargs.get("layout_method")

        if self.kwargs.get("routing_method"):
            program_inputs["routing_method"] = self.kwargs.get("routing_method")

        if self.kwargs.get("translation_method"):
            program_inputs["translation_method"] = self.kwargs.get("translation_method")

        if self.kwargs.get("seed_transpiler"):
            program_inputs["seed_transpiler"] = self.kwargs.get("seed_transpiler")

        program_inputs["optimization_level"] = self.kwargs.get("optimization_level", 1)

        if self.kwargs.get("init_qubits"):
            program_inputs["init_qubits"] = self.kwargs.get("init_qubits")

        if self.kwargs.get("rep_delay"):
            program_inputs["rep_delay"] = self.kwargs.get("rep_delay")

        if self.kwargs.get("transpiler_options"):
            program_inputs["transpiler_options"] = self.kwargs.get("transpiler_options")

        program_inputs["measurement_error_mitigation"] = self.kwargs.get(
            "measurement_error_mitigation", False
        )

        # Specify the backend.
        options = {"backend_name": self.backend.name()}

        # Send circuits to the cloud for execution by the circuit-runner program.
        job = self.provider.runtime.run(
            program_id="circuit-runner", options=options, inputs=program_inputs
        )
        self._current_job = job.result(decoder=RunnerResult)

        results = []

        for index, circuit in enumerate(circuits):
            self._samples = self.generate_samples(index)
            res = self.statistics(circuit.observables)
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
    r"""Class for a Qiskit runtime sampler program device in PennyLane.

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
        return_mitigation_overhead (bool): Return mitigation overhead factor. Default is False.
        run_config (dict): A collection of kwargs passed to backend.run.
        skip_transpilation (bool): Skip circuit transpilation. Default is False.
        transpile_config (dict): A collection of kwargs passed to transpile.
        use_measurement_mitigation (bool): Use measurement mitigation to improve results. Default is False.
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

        if self.kwargs.get("return_mitigation_overhead"):
            program_inputs["return_mitigation_overhead"] = self.kwargs.get(
                "return_mitigation_overhead"
            )

        if self.kwargs.get("run_config"):
            program_inputs["run_config"] = self.kwargs.get("run_config")

        program_inputs["skip_transpilation"] = self.kwargs.get("skip_transpilation", False)

        if self.kwargs.get("transpile_config"):
            program_inputs["transpile_config"] = self.kwargs.get("transpile_config")

        if self.kwargs.get("use_measurement_mitigation"):
            program_inputs["use_measurement_mitigation"] = self.kwargs.get(
                "use_measurement_mitigation"
            )

        # Specify the backend.
        options = {"backend_name": self.backend.name()}
        # Send circuits to the cloud for execution by the sampler program.
        job = self.provider.runtime.run(
            program_id="sampler", options=options, inputs=program_inputs
        )
        self._current_job = job.result()
        results = []

        for index, circuit in enumerate(circuits):
            self._samples = self.generate_samples(index)
            res = self.statistics(circuit.observables)
            results.append(res)

        if self.tracker.active:
            self.tracker.update(batches=1, batch_len=len(circuits))
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

        counts = qiskit.result.postprocess.format_counts(
            self._current_job.get("counts")[circuit], {"memory_slots": self._circuit.num_qubits}
        )

        samples = []
        for key, value in counts.items():
            for _ in range(0, value):
                samples.append(key)
        return np.vstack([np.array([int(i) for i in s[::-1]]) for s in samples])
