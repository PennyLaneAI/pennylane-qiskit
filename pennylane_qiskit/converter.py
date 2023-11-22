# Copyright 2019 Xanadu Quantum Technologies Inc.

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
This module contains functions for converting between Qiskit QuantumCircuit objects
and PennyLane circuits.
"""
from typing import Dict, Any
import warnings

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import extensions as ex
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.exceptions import QiskitError
from sympy import lambdify

import pennylane as qml
import pennylane.ops as pennylane_ops

# pylint: disable=too-many-instance-attributes

QISKIT_OPERATION_MAP_SELF_ADJOINT = {
    # native PennyLane operations also native to qiskit
    "PauliX": ex.XGate,
    "PauliY": ex.YGate,
    "PauliZ": ex.ZGate,
    "Hadamard": ex.HGate,
    "CNOT": ex.CXGate,
    "CZ": ex.CZGate,
    "SWAP": ex.SwapGate,
    "ISWAP": ex.iSwapGate,
    "RX": ex.RXGate,
    "RY": ex.RYGate,
    "RZ": ex.RZGate,
    "Identity": ex.IGate,
    "CSWAP": ex.CSwapGate,
    "CRX": ex.CRXGate,
    "CRY": ex.CRYGate,
    "CRZ": ex.CRZGate,
    "PhaseShift": ex.PhaseGate,
    "QubitStateVector": ex.Initialize,
    "StatePrep": ex.Initialize,
    "Toffoli": ex.CCXGate,
    "QubitUnitary": ex.UnitaryGate,
    "U1": ex.U1Gate,
    "U2": ex.U2Gate,
    "U3": ex.U3Gate,
    "IsingZZ": ex.RZZGate,
    "IsingYY": ex.RYYGate,
    "IsingXX": ex.RXXGate,
}

QISKIT_OPERATION_INVERSES_MAP_SELF_ADJOINT = {
    "Adjoint(" + k + ")": v for k, v in QISKIT_OPERATION_MAP_SELF_ADJOINT.items()
}

# Separate dictionary for the inverses as the operations dictionary needs
# to be invertible for the conversion functionality to work
QISKIT_OPERATION_MAP_NON_SELF_ADJOINT = {"S": ex.SGate, "T": ex.TGate, "SX": ex.SXGate}
QISKIT_OPERATION_INVERSES_MAP_NON_SELF_ADJOINT = {
    "Adjoint(S)": ex.SdgGate,
    "Adjoint(T)": ex.TdgGate,
    "Adjoint(SX)": ex.SXdgGate,
}

QISKIT_OPERATION_MAP = {
    **QISKIT_OPERATION_MAP_SELF_ADJOINT,
    **QISKIT_OPERATION_MAP_NON_SELF_ADJOINT,
}
QISKIT_OPERATION_INVERSES_MAP = {
    **QISKIT_OPERATION_INVERSES_MAP_SELF_ADJOINT,
    **QISKIT_OPERATION_INVERSES_MAP_NON_SELF_ADJOINT,
}

FULL_OPERATION_MAP = {**QISKIT_OPERATION_MAP, **QISKIT_OPERATION_INVERSES_MAP}

inv_map = {v.__name__: k for k, v in QISKIT_OPERATION_MAP.items()}

dagger_map = {"SdgGate": qml.S, "TdgGate": qml.T, "SXdgGate": qml.SX}



def _check_parameter_bound(param: Parameter, var_ref_map: Dict[Parameter, Any]):
    """Utility function determining if a certain parameter in a QuantumCircuit has
    been bound.

    Args:
        param (qiskit.circuit.Parameter): the parameter to be checked
        var_ref_map (dict[qiskit.circuit.Parameter, Any]):
            a dictionary mapping qiskit parameters to trainable parameter values
    """
    if isinstance(param, Parameter) and param not in var_ref_map:
        raise ValueError("The parameter {} was not bound correctly.".format(param))


def _extract_variable_refs(params: Dict[Parameter, Any]) -> Dict[Parameter, Any]:
    """Iterate through the parameter mapping to be bound to the circuit,
    and return a dictionary containing the trainable parameters.

    Args:
        params (dict): dictionary of the parameters in the circuit to their corresponding values

    Returns:
        dict[qiskit.circuit.Parameter, Any]: a dictionary mapping
            qiskit parameters to trainable parameter values
    """
    variable_refs = {}
    # map qiskit parameters to PennyLane trainable parameter values
    if params is not None:
        for k, v in params.items():
            if getattr(v, "requires_grad", True):
                # Values can be arrays of size 1, need to extract the Python scalar
                # (this can happen e.g. when indexing into a PennyLane numpy array)
                if isinstance(v, np.ndarray):
                    v = v.item()
                variable_refs[k] = v

    return variable_refs  # map qiskit parameters to trainable parameter values


def _check_circuit_and_bind_parameters(
    quantum_circuit: QuantumCircuit, params: dict, diff_params: dict
) -> QuantumCircuit:
    """Utility function for checking for a valid quantum circuit and then binding parameters.

    Args:
        quantum_circuit (QuantumCircuit): the quantum circuit to check and bind the parameters for
        params (dict): dictionary of the parameters in the circuit to their corresponding values
        diff_params (dict): dictionary mapping the differentiable parameters to trainable parameter
            values
    Returns:
        QuantumCircuit: quantum circuit with bound parameters
    """
    if not isinstance(quantum_circuit, QuantumCircuit):
        raise ValueError(
            "The circuit {} is not a valid Qiskit QuantumCircuit.".format(quantum_circuit)
        )

    if params is None:
        return quantum_circuit

    for k in diff_params:
        # Since we don't bind trainable values to Qiskit circuits,
        # we must remove them from the binding dictionary before binding.
        del params[k]

    return quantum_circuit.bind_parameters(params)


def map_wires(qc_wires: list, wires: list) -> dict:
    """Utility function mapping the wires specified in a quantum circuit with the wires
    specified by the user for the template.

    Args:
        qc_wires (list): wires from the converted quantum circuit
        wires (list): wires specified for the template

    Returns:
        dict[int, int]: map from quantum circuit wires to the user defined wires
    """
    if wires is None:
        return dict(zip(qc_wires, range(len(qc_wires))))

    if len(qc_wires) == len(wires):
        return dict(zip(qc_wires, wires))

    raise qml.QuantumFunctionError(
        "The specified number of wires - {} - does not match "
        "the number of wires the loaded quantum circuit acts on.".format(len(wires))
    )


def execute_supported_operation(operation_name: str, parameters: list, wires: list):
    """Utility function that executes an operation that is natively supported by PennyLane.

    Args:
        operation_name (str): Name of the PL operator to be executed
        parameters (str): parameters of the operation that will be executed
        wires (list): wires of the operation
    """
    operation = getattr(pennylane_ops, operation_name)

    if not parameters:
        operation(wires=wires)
    elif operation_name in ["QubitStateVector", "StatePrep"]:
        operation(np.array(parameters), wires=wires)
    else:
        operation(*parameters, wires=wires)


def load(quantum_circuit: QuantumCircuit):
    """Loads a PennyLane template from a Qiskit QuantumCircuit.
    Warnings are created for each of the QuantumCircuit instructions that were
    not incorporated in the PennyLane template.

    Args:
        quantum_circuit (qiskit.QuantumCircuit): the QuantumCircuit to be converted

    Returns:
        function: the resulting PennyLane template
    """

    def _function(params: dict = None, wires: list = None):
        """Returns a PennyLane template created based on the input QuantumCircuit.
        Warnings are created for each of the QuantumCircuit instructions that were
        not incorporated in the PennyLane template.

        Args:
            params (dict): specifies the parameters that need to be bound in the QuantumCircuit
            wires (Sequence[int] or int): The wires the converted template acts on.
                Note that if the original QuantumCircuit acted on :math:`N` qubits,
                then this must be a list of length :math:`N`.

        Returns:
            function: the new PennyLane template
        """
        var_ref_map = _extract_variable_refs(params)
        qc = _check_circuit_and_bind_parameters(quantum_circuit, params, var_ref_map)

        # Wires from a qiskit circuit have unique IDs, so their hashes are unique too
        qc_wires = [hash(q) for q in qc.qubits]

        wire_map = map_wires(qc_wires, wires)

        # Processing the dictionary of parameters passed
        for op, qargs, cargs in qc.data:
            # the new Singleton classes have different names than the objects they represent, but base_class.__name__ still matches
            instruction_name = getattr(op, "base_class", op.__class__).__name__

            operation_wires = [wire_map[hash(qubit)] for qubit in qargs]

            # New Qiskit gates that are not natively supported by PL (identical
            # gates exist with a different name)
            # TODO: remove the following when gates have been renamed in PennyLane
            instruction_name = "U3Gate" if instruction_name == "UGate" else instruction_name

            # pylint:disable=protected-access
            if (
                instruction_name in inv_map
                and inv_map[instruction_name] in pennylane_ops._qubit__ops__
            ):
                # Extract the bound parameters from the operation. If the bound parameters are a
                # Qiskit ParameterExpression, then replace it with the corresponding PennyLane
                # variable from the var_ref_map dictionary.

                pl_parameters = []
                for p in op.params:
                    _check_parameter_bound(p, var_ref_map)

                    if isinstance(p, ParameterExpression):
                        if p.parameters:  # non-empty set = has unbound parameters
                            ordered_params = tuple(p.parameters)

                            f = lambdify(ordered_params, p._symbol_expr, modules=qml.numpy)
                            f_args = []
                            for i_ordered_params in ordered_params:
                                f_args.append(var_ref_map.get(i_ordered_params))
                            pl_parameters.append(f(*f_args))
                        else:  # needed for qiskit<0.43.1
                            pl_parameters.append(float(p))  # pragma: no cover
                    else:
                        pl_parameters.append(p)

                execute_supported_operation(
                    inv_map[instruction_name], pl_parameters, operation_wires
                )

            elif instruction_name in dagger_map:
                gate = dagger_map[instruction_name]
                qml.adjoint(gate)(wires=operation_wires)

            else:
                try:
                    operation_matrix = op.to_matrix()
                    pennylane_ops.QubitUnitary(operation_matrix, wires=operation_wires)
                except (AttributeError, QiskitError):
                    warnings.warn(
                        __name__ + ": The {} instruction is not supported by PennyLane,"
                        " and has not been added to the template.".format(instruction_name),
                        UserWarning,
                    )

    return _function


def load_qasm(qasm_string: str):
    """Loads a PennyLane template from a QASM string.
    Args:
        qasm_string (str): the name of the QASM string
    Returns:
        function: the new PennyLane template
    """
    return load(QuantumCircuit.from_qasm_str(qasm_string))


def load_qasm_from_file(file: str):
    """Loads a PennyLane template from a QASM file.
    Args:
        file (str): the name of the QASM file
    Returns:
        function: the new PennyLane template
    """
    return load(QuantumCircuit.from_qasm_file(file))

def circuit_to_qiskit(circuit, register_size, diagonalize=True, measure=True):
    """Builds the circuit objects based on the operations and measurements
    specified to apply.

    Args:
        operations (list[~.Operation]): operations to apply to the device

    Keyword args:
        rotations (list[~.Operation]): Operations that rotate the circuit
            pre-measurement into the eigenbasis of the observables.
    """

    reg = QuantumRegister(register_size, "q")
    creg = ClassicalRegister(register_size, "c")
    qc = QuantumCircuit(register_size, register_size, name="temp")

    for op in circuit.operations:
        qc &= operation_to_qiskit(op, register_size)

    # rotate the state for measurement in the computational basis
    if diagonalize:
        rotations = circuit.diagonalizing_gates
        for rot in rotations:
            qc &= operation_to_qiskit(rot, register_size)

    if measure:
        # barrier ensures we first do all operations, then do all measurements
        qc.barrier(reg)
        # we always measure the full register
        qc.measure(reg, creg)

    return qc

def operation_to_qiskit(operation, register_size=None):
    """Take a Pennylane operator and convert to a Qiskit circuit

    Args:
        operation (List[pennylane.Operation]): operation to be converted
        num_qubits (int): the total number of qubits on the device

    Returns:
        QuantumCircuit: a quantum circuit objects containing the translated operation
    """
    op_wires = operation.wires
    par = operation.parameters

    # make quantum and classical registers for the full register
    if register_size is None:
        register_size = len(op_wires)
    reg = QuantumRegister(register_size, "q")
    creg = ClassicalRegister(register_size, "c")

    for idx, p in enumerate(par):
        if isinstance(p, np.ndarray):
            # Convert arrays so that Qiskit accepts the parameter
            par[idx] = p.tolist()

    operation = operation.name

    mapped_operation = FULL_OPERATION_MAP[operation]

    qregs = [reg[i] for i in op_wires.labels]

    adjoint = operation.startswith("Adjoint(")
    split_op = operation.split("Adjoint(")

    # Need to revert the order of the quantum registers used in
    # Qiskit such that it matches the PennyLane ordering
    if adjoint:
        if split_op[1] in ("QubitUnitary)", "QubitStateVector)", "StatePrep)"):
            qregs = list(reversed(qregs))
    else:
        if split_op[0] in ("QubitUnitary", "QubitStateVector", "StatePrep"):
            qregs = list(reversed(qregs))

    dag = circuit_to_dag(QuantumCircuit(reg, creg, name=""))
    gate = mapped_operation(*par)

    dag.apply_operation_back(gate, qargs=qregs)
    circuit = dag_to_circuit(dag)

    return circuit

def mp_to_pauli(mp, register_size):
    """Convert a Pauli observable to a SparsePauliOp for measurement via Estimator

        Args:
            mp(Union[ExpectationMP, VarianceMP]): MeasurementProcess to be converted to a SparsePauliOp
            register_size(int): total size of the qubit register being measured
    """

    # ToDo: I believe this could be extended to cover expectation values of Hamiltonians

    observables = {"PauliX": "X",
                   "PauliY": "Y",
                   "PauliZ": "Z",
                   "Identity": "I"}

    pauli_string = ["I"] * register_size
    pauli_string[mp.wires[0]] = observables[mp.name]

    pauli_string = ('').join(pauli_string)

    return SparsePauliOp(pauli_string)
