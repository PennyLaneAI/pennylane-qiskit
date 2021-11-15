import qiskit.result.postprocess
from qiskit.ignis.mitigation.expval import expectation_value
from .ibmq import IBMQDevice
from qiskit.providers.ibmq import RunnerResult
import numpy as np

class IBMQCircuitRunnerDevice(IBMQDevice):

    short_name = "qiskit.ibmq.circuitrunner"

    def __init__(self, wires, provider=None, backend="ibmq_qasm_simulator", shots=1024, **kwargs):
        self.kwargs = kwargs
        super().__init__(wires=wires, provider=provider, backend=backend, shots=shots, **kwargs)

    def run(self, qcirc):

        program_inputs = {'circuits': qcirc, 'shots': self.shots}

        # Initial position of virtual qubits
        # on physical qubits.
        if self.kwargs.get("initial_layout"):
            program_inputs["initial_layout"] = self.kwargs.get("initial_layout")

        # Name of layout selection pass
        # ('trivial', 'dense', 'noise_adaptive', 'sabre')
        if self.kwargs.get('layout_method'):
            program_inputs["layout_method"] = self.kwargs.get('layout_method')

        # Name of routing pass ('basic',
        # 'lookahead', 'stochastic', 'sabre').
        if self.kwargs.get('routing_method'):
            program_inputs["routing_method"] = self.kwargs.get('routing_method')  # string

        # Name of translation pass ('unroller',
        # 'translator', 'synthesis').
        if self.kwargs.get('translation_method'):
            program_inputs["translation_method"] = self.kwargs.get('translation_method')  # string

        # Sets random seed for the
        # stochastic parts of the transpiler.
        if self.kwargs.get('seed_transpiler'):
            program_inputs["seed_transpiler"] = self.kwargs.get('seed_transpiler')  # int

        # How much optimization to perform
        # on the circuits (0-3). Higher
        # levels generate more optimized circuits.
        # Default is 1.
        program_inputs["optimization_level"] = self.kwargs.get('optimization_level', 1) # int

        # Whether to reset the qubits
        # to the ground state for
        # each shot.
        if self.kwargs.get('init_qubits'):
            program_inputs["init_qubits"] = self.kwargs.get('init_qubits')  # bool

        # Delay between programs in seconds.
        if self.kwargs.get('rep_delay'):
            program_inputs["rep_delay"] = self.kwargs.get('rep_delay')  # float

        # Additional compilation options.
        if self.kwargs.get('transpiler_options'):
            program_inputs["transpiler_options"] = self.kwargs.get('transpiler_options')  # dict

        # Whether to apply measurement error
        # mitigation. Default is False.
        program_inputs["measurement_error_mitigation"] = self.kwargs.get('measurement_error_mitigation', False)  # bool

        # Specify the backend.
        options = {'backend_name': self.backend.name()}

        # Send circuits to the cloud for execution by the circuit-runner program.
        job = self.provider.runtime.run(program_id="circuit-runner",
                                                      options=options,
                                                      inputs=program_inputs)
        self._current_job = job.result(decoder=RunnerResult)

    def generate_samples(self):
        counts = self._current_job.get_counts()
        samples = []
        for key, value in counts.items():
            for i in range(0, value):
                samples.append(key)
        return np.vstack([np.array([int(i) for i in s[::-1]]) for s in samples])

class IBMQSamplerDevice(IBMQDevice):

    short_name = "qiskit.ibmq.sampler"

    def __init__(self, wires, provider=None, backend="ibmq_qasm_simulator", shots=1024, **kwargs):
        self.kwargs = kwargs
        super().__init__(wires=wires, provider=provider, backend=backend, shots=shots, **kwargs)

    def run(self, qcirc):
        print(self.kwargs)
        program_inputs = {'circuits': qcirc}
        # Return mitigation overhead factor. Default
        # is False.
        if self.kwargs.get('return_mitigation_overhead'):
            program_inputs['return_mitigation_overhead'] = self.kwargs.get('return_mitigation_overhead') # boolean

        # A collection of kwargs passed
        # to backend.run.
        if self.kwargs.get('run_config'):
            program_inputs['run_config'] = self.kwargs.get('run_config') # object

        # Skip circuit transpilation. Default is
        # False.
        if self.kwargs.get('skip_transpilation'):
            program_inputs['skip_transpilation'] = False # boolean

        # A collection of kwargs passed
        # to transpile.
        if self.kwargs.get('transpiler_config'):
            program_inputs['transpiler_config'] = self.kwargs.get('transpiler_config') # object
        else:
            program_inputs['transpiler_config'] = {'optimization_level': 1}

        # Use measurement mitigation to improve
        # results. Default is False.
        if self.kwargs.get('use_measurement_mitigation'):
            program_inputs['use_measurement_mitigation'] = self.kwargs.get('use_measurement_mitigation') # boolean

        # Specify the backend.
        options = {'backend_name': self.backend.name()}
        # Send circuits to the cloud for execution by the circuit-runner program.
        print(program_inputs)
        job = self.provider.runtime.run(program_id="sampler",
                                        options=options,
                                        inputs=program_inputs)
        self._current_job = job.result()

    def generate_samples(self):
        counts = qiskit.result.postprocess.format_counts(self._current_job[0][0], {'memory_slots': self._circuit.num_qubits})
        samples = []
        for key, value in counts.items():
            for i in range(0, value):
                samples.append(key)
        return np.vstack([np.array([int(i) for i in s[::-1]]) for s in samples])