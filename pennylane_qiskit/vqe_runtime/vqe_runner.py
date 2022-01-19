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
"""
This module contains a function to run aa custom PennyLane VQE problem on qiskit runtime.
"""
# pylint: disable=too-few-public-methods,protected-access,too-many-arguments,too-many-branches,too-many-statements

import warnings
import inspect
from collections import OrderedDict

import numpy as np
import pennylane as qml

from pennylane_qiskit.qiskit_device import QiskitDevice
from pennylane_qiskit.ibmq import connect
import qiskit.circuit.library.n_local as lib_local
from qiskit.providers.ibmq.runtime import ResultDecoder
from qiskit.circuit import ParameterVector, QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import IBMQ
from qiskit.providers.ibmq.exceptions import IBMQAccountError

from scipy.optimize import OptimizeResult


class VQEResultDecoder(ResultDecoder):
    """The class is used to decode the result from the runtime problem and return it as a
    Scipy Optimizer result.
    """

    @classmethod
    def decode(cls, data):
        """Decode the data from the VQE program."""
        data = super().decode(data)
        return OptimizeResult(data)


class RuntimeJobWrapper:
    """A simple Job wrapper that attaches intermediate results directly to the job object itself
    in the `intermediate_results attribute` via the `_callback` function.
    """

    def __init__(self):
        self._job = None
        self._decoder = VQEResultDecoder
        self.intermediate_results = {
            "nfev": [],
            "parameters": [],
            "function": [],
            "step": [],
            "accepted": [],
        }

    def _callback(self, *args):
        """The callback function that attaches intermediate results to the wrapper:

        Args:
            nfev (int): Number of evaluation.
            xk (array_like): A list or NumPy array to attach.
            fk (float): Value of the function.
            step (float): Value of the step.
            accepted (bool): True if the loss function value has improved, False otherwise.
        """
        _, (nfev, xk, fk, step, accepted) = args
        self.intermediate_results["nfev"].append(nfev)
        self.intermediate_results["parameters"].append(xk)
        self.intermediate_results["function"].append(fk)
        self.intermediate_results["step"].append(step)
        self.intermediate_results["accepted"].append(accepted)

    def _scipy_callback(self, *args):
        """The callback function that attaches intermediate results to the wrapper:

        Args:
            xk (array_like): A list or NumPy array to attach.
        """
        _, xk = args
        self.intermediate_results["parameters"].append(xk)

    def __getattr__(self, attr):
        if attr == "result":
            return self.result

        if attr in dir(self._job):
            return getattr(self._job, attr)
        raise AttributeError(f"Class does not have {attr}.")

    def result(self):
        """Get the result of the job as a SciPy OptimizerResult object.

        This method blocks until the job is done, cancelled, or raises an error.

        Returns:
            OptimizerResult: A optimizer result object.
        """
        return self._job.result(decoder=self._decoder)


def upload_vqe_runner(hub="ibm-q", group="open", project="main", **kwargs):
    """Upload the custom VQE runner to the IBMQ cloud.

    Args:
        hub (str): Ibmq provider hub.
        group (str): Ibmq provider group.
        project (str): Ibmq provider project.

    Returns:
        str: Program id that can be used to run the program.
    """

    connect(kwargs)

    # Specify a single hub, group and project
    hub = kwargs.get("hub", "ibm-q")
    group = kwargs.get("group", "open")
    project = kwargs.get("project", "main")

    meta = {
        "name": "vqe-runtime",
        "description": "A sample VQE program.",
        "max_execution_time": 100000,
        "spec": {},
    }

    meta["spec"]["parameters"] = {
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "properties": {
            "hamiltonian": {
                "description": "Hamiltonian whose ground state we want to find.",
                "type": "array",
            },
            "x0": {
                "description": "Initial vector of parameters for the quantum circuit.",
                "type": "array",
            },
            "ansatz": {
                "description": "Qiskit circuit or name of ansatz quantum circuit to use, default='EfficientSU2'",
                "type": "[QuantumCircuit,string]",
                "default": "EfficientSU2",
            },
            "ansatz_config": {
                "description": "Configuration parameters for the ansatz circuit.",
                "type": "dict",
            },
            "optimizer": {
                "description": "Classical optimizer to use, default='SPSA'.",
                "type": "string",
                "default": "SPSA",
            },
            "optimizer_config": {
                "description": "Configuration parameters for the optimizer.",
                "type": "dict",
            },
            "shots": {
                "description": "The number of shots used for each circuit evaluation.",
                "type": "integer",
            },
            "use_measurement_mitigation": {
                "description": "Use measurement mitigation, default=False.",
                "type": "boolean",
                "default": False,
            },
        },
        "required": ["hamiltonian", "x0"],
    }

    meta["spec"]["return_values"] = {
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "description": "Final result in SciPy optimizer format",
        "type": "object",
    }

    meta["spec"]["intermadiate_results"] = {
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "description": "Dictionnary containing: "
                       "The number of evaluation at current optimization step."
                       "Parameter vector at current optimization step."
                       "Function value at the current optimization step."
                       "The size of the step.",
        "type": "dict",
    }

    provider = IBMQ.get_provider(hub=hub, group=group, project=project)

    program_id = provider.runtime.upload_program(
        data="pennylane_runtime/vqe_runtime.py", metadata=meta
    )
    return program_id


def delete_vqe_runner(provider, program_id):
    """Delete the desired program on IBMQ platform.
    Args:
        provider (object): IBMQ provider.
        program_id (str): Id of the qiskit runtime to be deleted.
    """
    provider.runtime.delete_program(program_id)


def vqe_runner(
        backend,
        hamiltonian,
        x0,
        program_id,
        ansatz="EfficientSU2",
        ansatz_config=None,
        optimizer="SPSA",
        optimizer_config=None,
        shots=8192,
        use_measurement_mitigation=False,
        **kwargs,
):
    """Routine that executes a given VQE problem via the sample-vqe program on the target backend.

    Args:
        backend (ProgramBackend): Qiskit backend instance.
        hamiltonian (qml.Hamiltonian): Hamiltonian whose ground state we want to find.
        x0 (array_like): Initial vector of parameters.
        program_id(str): Id of the program, it has to be generated by using the upload_vqe_runner function.
        Once the program is uploaded, you can find the id in your program list online.
        ansatz (Quantum function or str): Optional, a PennyLane quantum function or the name of the Qiskit
            ansatz quantum circuit to use, default='EfficientSU2'
        ansatz_config (dict): Optional, configuration parameters for the ansatz circuit if from Qiskit library.
        optimizer (str): Optional, string specifying classical optimizer, default='SPSA'.
        optimizer_config (dict): Optional, configuration parameters for the optimizer.
        shots (int): Optional, number of shots to take per circuit, default=1024.
        use_measurement_mitigation (bool): Optional, use measurement mitigation, default=False.

    Returns:
        OptimizeResult: The result in SciPy optimization format.
    """
    # Init the dictionnaries
    if ansatz_config is None:
        ansatz_config = {}

    if optimizer_config is None:
        optimizer_config = {"maxiter": 100}

    if not isinstance(hamiltonian, qml.Hamiltonian):
        raise qml.QuantumFunctionError("A PennyLane Hamiltonian object is required.")

    connect(kwargs)

    options = {"backend_name": backend}

    inputs = {}

    # Extract the number of qubit from the hamiltonian
    _, observables = hamiltonian.terms

    qubit_set = set()
    for obs in observables:
        for qubit in obs.wires.tolist():
            qubit_set.add(qubit)

    num_qubits_h = list(qubit_set)[-1] + 1

    # Validate circuit ansatz and number of qubits
    if not isinstance(ansatz, str):
        inputs["x0"], inputs["ansatz"] = _pennylane_to_qiskit_ansatz(ansatz, x0, num_qubits_h)

    # The circuit will be taken from the Qiskit library as a str was passed.
    else:

        num_qubits = num_qubits_h

        ansatz_circ = getattr(lib_local, ansatz, None)
        if ansatz_circ is None:
            raise ValueError(f"Ansatz {ansatz} not in n_local circuit library.")

        inputs["ansatz"] = ansatz
        inputs["ansatz_config"] = ansatz_config

        # If given x0, validate its length against num_params in ansatz
        x0 = np.asarray(x0)
        ansatz_circ = ansatz_circ(num_qubits, **ansatz_config)
        num_params = ansatz_circ.num_parameters

        if x0.shape[0] != num_params:
            warnings.warn("The shape of parameters array is not correct, a random initialization has been applied.")
            x0 = 2 * np.pi * np.random.rand(num_params)

        inputs["x0"] = x0

    # Transform the PennyLane hamilonian to a suitable form
    hamiltonian = hamiltonian_to_list_string(hamiltonian, num_qubits)

    inputs["hamiltonian"] = hamiltonian

    # Set the rest of the inputs
    inputs["optimizer"] = optimizer
    inputs["optimizer_config"] = optimizer_config
    inputs["shots"] = shots
    inputs["use_measurement_mitigation"] = use_measurement_mitigation

    # Specify a single hub, group and project
    hub = kwargs.get("hub", "ibm-q")
    group = kwargs.get("group", "open")
    project = kwargs.get("project", "main")

    provider = IBMQ.get_provider(hub=hub, group=group, project=project)

    rt_job = RuntimeJobWrapper()

    # Callbacks functions are different between optimizers.
    if optimizer in ["SPSA", "QNSPSA"]:
        job = provider.runtime.run(
            program_id, options=options, inputs=inputs, callback=rt_job._callback
        )
    else:
        job = provider.runtime.run(
            program_id, options=options, inputs=inputs, callback=rt_job._scipy_callback
        )
    rt_job._job = job

    return rt_job


def _pennylane_to_qiskit_ansatz(ansatz, x0, num_qubits_h):
    r"""Convert a ansatz from PennyLane to a circuit in Qiskit.

        Args:
            ansatz (Quantum Function): A PennyLane quantum function that represents the circuit.
            x0 (
            num_qubits_h (int): Number of qubits evaluated from the hamiltonian.

        Returns:
            list[tuple[float,str]]: Hamiltonian in a format for the runtime program.
        """
    if isinstance(ansatz, (qml.QNode, qml.tape.QuantumTape)):
        raise qml.QuantumFunctionError("The ansatz must be a callable quantum function.")

    if callable(ansatz):
        if len(inspect.getfullargspec(ansatz).args) != 1:
            raise qml.QuantumFunctionError("Param should be a single vector.")
        try:
            tape_param = x0[0] if len(x0) == 1 else x0
            tape = qml.transforms.make_tape(ansatz)(tape_param).expand(
                depth=5, stop_at=lambda obj: obj.name in QiskitDevice._operation_map
            )
        except IndexError as e:
            raise qml.QuantumFunctionError("Not enough parameters in X0.") from e

        num_params = tape.num_params

        if len(x0) != num_params:
            warnings.warn("Due to the tape expansion, the number of parameters has increased.")
            x0 = 2 * np.pi * np.random.rand(num_params)

        # Raise exception if there are no operations
        if len(tape.operations) == 0:
            raise qml.QuantumFunctionError("Function contains no quantum operations.")

        # if no wire ordering is specified, take wire list from tape
        wires = tape.wires

        # Compare the number of qubits from the circuit and from the hamiltonian
        num_qubits_c = len(wires)

        # Set the number of qubits
        num_qubits = num_qubits_c if num_qubits_c > num_qubits_h else num_qubits_h

        consecutive_wires = qml.wires.Wires(range(num_qubits))
        wires_map = OrderedDict(zip(wires, consecutive_wires))

        # From here: Create the Qiskit ansatz circuit
        params_vector = ParameterVector("p", num_params)

        reg = QuantumRegister(num_qubits, "q")
        circuit_ansatz = QuantumCircuit(reg, name="vqe")

        circuits = []

        j = 0
        for operation in tape.operations:
            wires = operation.wires.map(wires_map)
            par = operation.parameters
            operation = operation.name
            mapped_operation = QiskitDevice._operation_map[operation]

            qregs = [reg[i] for i in wires.labels]

            if operation.split(".inv")[0] in ("QubitUnitary", "QubitStateVector"):
                # Need to revert the order of the quantum registers used in
                # Qiskit such that it matches the PennyLane ordering
                qregs = list(reversed(qregs))

            dag = circuit_to_dag(QuantumCircuit(reg, name=""))

            if operation in ("QubitUnitary", "QubitStateVector"):
                # Parameters are matrices
                gate = mapped_operation(par[0])
            else:
                # Parameters for the operation
                if par:
                    op_num_params = len(par)
                    par = []
                    for num in range(op_num_params):
                        par.append(params_vector[j + num])
                    j += op_num_params
                    gate = mapped_operation(*par)
                # No parameters are needed
                else:
                    gate = mapped_operation

            if operation.endswith(".inv"):
                gate = gate.inverse()

            dag.apply_operation_back(gate, qargs=qregs)
            circuit = dag_to_circuit(dag)
            circuits.append(circuit)

        for circuit in circuits:
            circuit_ansatz &= circuit

        return x0, circuit_ansatz

    else:
        raise ValueError("Input ansatz is not a quantum function or a string.")


def hamiltonian_to_list_string(hamiltonian, num_qubits):
    r"""Convert a hamiltonian from PennyLane to a list of coefficient and strings.

    Args:
        hamiltonian (qml.Hamiltonian): A Hamiltonian from PennyLane.
        num_qubits (int): Number of qubits.

    Returns:
        list[tuple[float,str]]: Hamiltonian in a format for the runtime program.
    """

    coeff, observables = hamiltonian.terms

    authorized_obs = {"PauliX", "PauliY", "PauliZ", "Hadamard", "Identity"}

    for obs in observables:
        # Tensors
        if isinstance(obs.name, list):
            for ob in obs.name:
                if ob not in authorized_obs:
                    raise qml.QuantumFunctionError("Observable is not accepted.")
        else:
            if obs.name not in authorized_obs:
                raise qml.QuantumFunctionError("Observable is not accepted.")

    # Create string Hamiltonian
    obs_str = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Hadamard": "H", "Identity": "I"}

    obs_org = []
    # Map the PennyLane hamiltonian to a list PauliY(1) @ PauliY(0) -> [[[0,'Y'], [1,'Y']]]
    for obs in observables:
        # Tensors
        if isinstance(obs.name, list):
            internal = []
            for i, j in zip(obs.wires.tolist(), obs.name):
                internal.append([i, obs_str[j]])
            internal.sort()
            obs_org.append(internal)
        else:
            obs_org.append([[obs.wires.tolist()[0], obs_str[obs.name]]])

    # Create the hamiltonian terms as lists of strings [[[0,'Y'], [1,'Y']]] -> [['YI'], ['IY']]
    obs_list = []
    for elem in obs_org:
        empty_obs = ["I"] * num_qubits
        for el in elem:
            wire = el[0]
            observable = el[1]
            empty_obs[wire] = observable
        obs_list.append(empty_obs)

    # Create the list of tuple with coeff and hammiltonians as strings [['YI'], ['IY']] -> [(1, 'YI'), (1, 'IY')]
    hamiltonian = []
    for i, elem in enumerate(obs_list):
        result = "".join(elem)
        hamiltonian.append((coeff[i], result))

    return hamiltonian
