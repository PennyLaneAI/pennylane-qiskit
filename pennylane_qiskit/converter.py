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
This module contains functions for converting Qiskit QuantumCircuit objects
into PennyLane circuit templates.
"""
from typing import Dict, Any
import warnings
from functools import partial

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression, Measure, Barrier, IfElseOp
from qiskit.circuit.library import GlobalPhaseGate
from qiskit.exceptions import QiskitError
from sympy import lambdify

import pennylane as qml
import pennylane.ops as pennylane_ops
from pennylane_qiskit.qiskit_device import QISKIT_OPERATION_MAP

# pylint: disable=too-many-instance-attributes

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
        raise ValueError(f"The parameter {param} was not bound correctly.")


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
        raise ValueError(f"The circuit {quantum_circuit} is not a valid Qiskit QuantumCircuit.")

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
        f"The specified number of wires - {len(wires)} - does not match "
        "the number of wires the loaded quantum circuit acts on."
    )


# pylint:disable=too-many-statements, too-many-branches
def load(quantum_circuit: QuantumCircuit, measurements=None):
    """Loads a PennyLane template from a Qiskit QuantumCircuit.
    Warnings are created for each of the QuantumCircuit instructions that were
    not incorporated in the PennyLane template.

    Args:
        quantum_circuit (qiskit.QuantumCircuit): the QuantumCircuit to be converted
        measurements (list[pennylane.measurements.MeasurementProcess]): the list of PennyLane
            `measurements <https://docs.pennylane.ai/en/stable/introduction/measurements.html>`_
            that overrides the terminal measurements that may be present in the input circuit.

    Returns:
        function: the resulting PennyLane template
    """

    # pylint:disable=fixme, protected-access, unnecessary-lambda-assignment
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

        # Stores the measurements encountered in the circuit
        terminal_meas = []
        mid_circ_meas, mid_circ_regs = [], {}

        # Processing the dictionary of parameters passed
        for idx, (ops, qargs, cargs) in enumerate(qc.data):
            # the new Singleton classes have different names than the objects they represent,
            # but base_class.__name__ still matches
            instruction_name = getattr(ops, "base_class", ops.__class__).__name__
            # New Qiskit gates that are not natively supported by PL (identical
            # gates exist with a different name)
            # TODO: remove the following when gates have been renamed in PennyLane
            instruction_name = "U3Gate" if instruction_name == "UGate" else instruction_name

            # Define operator builders and helpers
            operation_func = None
            operation_overlapper = lambda op: op
            operation_wires = [wire_map[hash(qubit)] for qubit in qargs]
            operation_kwargs = {"wires": operation_wires}
            operation_args = []
            operation_cond = False

            # Extract the bound parameters from the operation. If the bound parameters are a
            # Qiskit ParameterExpression, then replace it with the corresponding PennyLane
            # variable from the var_ref_map dictionary.
            operation_params = []
            for p in ops.params:
                _check_parameter_bound(p, var_ref_map)

                if isinstance(p, ParameterExpression):
                    if p.parameters:  # non-empty set = has unbound parameters
                        ordered_params = tuple(p.parameters)
                        f = lambdify(ordered_params, p._symbol_expr, modules=qml.numpy)
                        f_args = []
                        for i_ordered_params in ordered_params:
                            f_args.append(var_ref_map.get(i_ordered_params))
                        operation_params.append(f(*f_args))
                    else:  # needed for qiskit<0.43.1
                        operation_params.append(float(p))  # pragma: no cover
                else:
                    operation_params.append(p)

            if instruction_name in dagger_map:
                operation_func = dagger_map[instruction_name]
                operation_overlapper = qml.adjoint

            elif instruction_name in inv_map:
                operation_name = inv_map[instruction_name]
                operation_func = getattr(pennylane_ops, operation_name)
                operation_args.extend(operation_params)
                if operation_name in ["QubitStateVector", "StatePrep"]:
                    operation_args = [np.array(operation_params)]

            elif isinstance(ops, Measure):
                # Store the current operation wires
                op_wires = set(operation_wires)
                # Look-ahead for more gate(s) on its wire(s)
                meas_terminal = True
                for next_op, next_qargs, __ in qc.data[idx + 1 :]:
                    # Check if the subsequent whether next_op is measurement interfering
                    if not isinstance(next_op, (Barrier, GlobalPhaseGate)):
                        next_op_wires = set(wire_map[hash(qubit)] for qubit in next_qargs)
                        # Check if there's any overlapping wires
                        if next_op_wires.intersection(op_wires):
                            meas_terminal = False
                            break

                # Allows for adding terminal measurements
                if meas_terminal:
                    terminal_meas.extend(operation_wires)

                # Allows for queing the mid-circuit measurements
                else:
                    operation_func = qml.measure
                    mid_circ_meas.append(qml.measure(wires=operation_wires))

                    # Allows for tracking conditional operations
                    for carg in cargs:
                        mid_circ_regs[carg] = mid_circ_meas[-1]

            # TODO: this can contain logic for the bigger ControlFlowOps
            elif isinstance(ops, IfElseOp):
                operation_cond = True

            else:

                try:
                    operation_args = [ops.to_matrix()]
                    operation_func = qml.QubitUnitary

                except (AttributeError, QiskitError):
                    warnings.warn(
                        f"{__name__}: The {instruction_name} instruction is not supported by PennyLane,"
                        " and has not been added to the template.",
                        UserWarning,
                    )

            # Check if it is a conditional operation
            if ops.condition and ops.condition[0] in mid_circ_regs:
                # Used for branch inversion to match PL formalism
                # True --> Keep | False --> Invert
                res_bit = ops.condition[1]

                if operation_cond:
                    with qml.QueuingManager.stop_recording():
                        branch_funcs = [
                            partial(
                                load(branch_inst, measurements=None), params=params, wires=wires
                            )
                            for branch_inst in operation_params
                            if isinstance(branch_inst, QuantumCircuit)
                        ]

                        if len(branch_funcs) == 1:
                            true_fn = [branch_funcs[0], qml.Identity][~res_bit]
                            false_fn = [branch_funcs[1], None][res_bit]

                        elif len(branch_funcs) == 2:
                            true_fn = branch_funcs[~res_bit]
                            false_fn = branch_funcs[res_bit]

                else:
                    true_fn = [operation_func, qml.Identity][~res_bit]
                    false_fn = [operation_func, None][res_bit]

                qml.cond(mid_circ_regs[ops.condition[0]], true_fn, false_fn)(
                    *operation_args, **operation_kwargs
                )

            # Check if it is not a mid-circuit measurement
            elif operation_func and not isinstance(ops, Measure):
                operation_overlapper(operation_func)(*operation_args, **operation_kwargs)

        # Use the user-provided measurements
        if measurements:
            if qml.queuing.QueuingManager.active_context():
                return [qml.apply(meas) for meas in measurements]
            return measurements

        return tuple(mid_circ_meas + list(map(qml.measure, terminal_meas))) or None

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
