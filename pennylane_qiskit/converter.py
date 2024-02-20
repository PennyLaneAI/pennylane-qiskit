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
from functools import partial, reduce

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector
from qiskit.circuit import Measure, Barrier, ControlFlowOp
from qiskit.circuit.controlflow.switch_case import _DefaultCaseType
from qiskit.circuit.library import GlobalPhaseGate
from qiskit.exceptions import QiskitError
from sympy import lambdify

import pennylane as qml
import pennylane.ops as pennylane_ops
from pennylane_qiskit.qiskit_device import QISKIT_OPERATION_MAP

# pylint: disable=too-many-instance-attributes

inv_map = {v.__name__: k for k, v in QISKIT_OPERATION_MAP.items()}

dagger_map = {"SdgGate": qml.S, "TdgGate": qml.T, "SXdgGate": qml.SX}

referral_to_forum = (
    "\n \nIf you are experiencing any difficulties with converting circuits from Qiskit, you can reach out "
    "\non the PennyLane forum at https://discuss.pennylane.ai/c/pennylane-plugins/pennylane-qiskit/"
)


def _check_parameter_bound(param: Parameter, unbound_params: Dict[Parameter, Any]):
    """Utility function determining if a certain parameter in a QuantumCircuit has
    been bound.

    Args:
        param (qiskit.circuit.Parameter): the parameter to be checked
        unbound_params (dict[qiskit.circuit.Parameter, Any]):
            a dictionary mapping qiskit parameters to trainable parameter values
    """
    if isinstance(param, Parameter) and param not in unbound_params:
        raise ValueError(f"The parameter {param} was not bound correctly.".format(param))


def _process_basic_param_args(params, *args, **kwargs):
    """Process the basic conditions for parameter dictionary computation.

    Returns:
        params (dict): A dictionary mapping ``quantum_circuit.parameters`` to values
        flag (bool): Indicating whether the returned ``params`` can be used.
    """

    # if no kwargs are passed, and a dictionary has been passed as a single argument, then assume it is params
    if params is None and not kwargs and (len(args) == 1 and isinstance(args[0], dict)):
        return (args[0], True)

    if not args and not kwargs:
        return (params, True)

    # make params dict if using args and/or kwargs
    if params is not None:
        raise RuntimeError(
            "Cannot define parameters via the params kwarg when passing Parameter values "
            "as individual args or kwargs."
        )

    return ({}, False)


def _expected_parameters(quantum_circuit):
    """Gets the expected parameters and a string of their names from the QuantumCircuit.
    Primarily serves to change a list of Parameters and ParameterVectorElements into a list
    of Parameters and ParameterVectors. I.e.:

    [Parameter('a'), ParameterVectorElement('v[0]'), ParameterVectorElement('v[1]'), ParameterVectorElement('v[2]')]

    becomes [Parameter('a'), ParameterVector(name='v', length=3)].

    Returns:
        expected_params: The reorganized list of Parameters, containing Parameter and ParameterVector
        param_name_string: a string listing the parameter names, i.e. in the example above, 'a, v'

    """

    expected_params = {}
    for p in quantum_circuit.parameters:
        # we want the p.vector if p is a ParameterVectorElement, otherwise p
        param = getattr(p, "vector", p)
        expected_params.update({param.name: param})

    param_name_string = ", ".join(expected_params.keys())

    return expected_params, param_name_string


def _format_params_dict(quantum_circuit, params, *args, **kwargs):
    """Processes the inputs for calling the quantum function and returns
    a dictionary of the format ``{Parameter("name"): value}`` for all the parameters.

    For a ``quantum_circuit`` with parameters ``[Parameter("phi"), Parameter("psi"), Parameter("theta")]``,
    inputs can be one of the following:
        1. the kwargs passed to when calling the qfunc, other than ``params`` and ``wires``.
            The keys in kwargs are expected to correspond to the names of the Parameters on
            the QuantumCircuit, i.e. ``(qc, None, phi=0.35, theta=0.2, psi=1.7)``
        2. the args passed when calling the qfunc, i.e. ``(qc, None, 0.35, 1.7, 0.2)``. These
           are assigned to the Parameters as the are ordered on the Qiskit QuantumCircuit:
           alphabetized by Parameter name
        3. Some combination of args and kwargs, i.e. ``(qc, None, 0.35, 0.2, psi=1.7)``
        4. (legacy) ``params`` from the kwarg ``params`` of the qfunc call, which is expected
            to already be a dictionary of the format ``{Parameter("name"): value}``, i.e.
            ``(qc, {Parameter("phi"): 0.35, Parameter("psi"): 1.7, Parameter("theta"): 0.2})``
        5. (legacy) ``params`` passed as a single arg, which is expected
            to already be a dictionary of the format ``{Parameter("name"): value}``, i.e.
            ``(qc, None, {Parameter("phi"): 0.35, Parameter("psi"): 1.7, Parameter("theta"): 0.2})``

    Returns:
        params (dict): A dictionary mapping ``quantum_circuit.parameters`` to values
    """

    params, flag = _process_basic_param_args(params, *args, **kwargs)

    if flag:
        return params

    expected_params, param_name_string = _expected_parameters(quantum_circuit)

    # populate it with any parameters defined as kwargs
    for k, v in kwargs.items():
        # the key needs to be the actual Parameter, whereas kwargs keys are parameter names
        if not k in expected_params:
            raise TypeError(
                f"Got unexpected parameter keyword argument '{k}'. Circuit contains parameters: {param_name_string} {referral_to_forum}"
            )
        params[expected_params[k]] = v

    # get any parameters not defined in kwargs (may be all of them) and match to args in order
    expected_arg_params = [param for name, param in expected_params.items() if name not in kwargs]
    has_param_vectors = np.any([isinstance(p, ParameterVector) for p in expected_arg_params])

    # if too many args were passed to the function call, raise an error
    # all other checks regarding correct arguments will be processed in _check_circuit_and_assign_parameters
    # (based on the full params dict generated by this function), but this information can only be captured here
    if len(args) > len(expected_arg_params):
        param_vector_info = (
            "Note that PennyLane expects to recieve a ParameterVector as a single argument "
            "containing all ParameterVectorElements."
            if has_param_vectors
            else ""
        )
        raise TypeError(
            f"Expected {len(expected_arg_params)} positional argument{'s' if len(expected_arg_params) > 1 else ''} but {len(args)} were given. {param_vector_info} {referral_to_forum}"
        )
    params.update(dict(zip(expected_arg_params, args)))

    return params


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
            if qml.math.requires_grad(v):
                # Values can be arrays of size 1, need to extract the Python scalar
                # (this can happen e.g. when indexing into a PennyLane numpy array)
                if isinstance(v, np.ndarray) and v.size == 1:
                    v = v.item()
                variable_refs[k] = v

    return variable_refs  # map qiskit parameters to trainable parameter values


def _check_circuit_and_assign_parameters(
    quantum_circuit: QuantumCircuit, params: dict, diff_params: dict
) -> QuantumCircuit:
    """Utility function for checking for a valid quantum circuit and then assigning the parameters.

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

    # confirm parameter names are valid for conversion to PennyLane
    for name in ["wires", "params"]:
        if name in [p.name for p in quantum_circuit.parameters]:
            raise RuntimeError(
                f"Cannot interpret QuantumCircuit with parameter '{name}' as a PennyLane "
                f"quantum function, as this argument is reserved. {referral_to_forum}"
            )

    expected_params, param_name_string = _expected_parameters(quantum_circuit)

    if params is None:
        if quantum_circuit.parameters:
            s = "s" if len(quantum_circuit.parameters) > 1 else ""
            raise TypeError(
                f"Missing required argument{s} to define Parameter value{s} for: {param_name_string} {referral_to_forum}"
            )
        return quantum_circuit

    # if any parameters are missing a value, raise an error
    undefined_params = [name for name, param in expected_params.items() if param not in params]
    if undefined_params:
        s = "s" if len(undefined_params) > 1 else ""
        param_names = ", ".join(undefined_params)
        raise TypeError(
            f"Missing {len(undefined_params)} required argument{s} to define Parameter value{s} for: {param_names}. {referral_to_forum}"
        )

    for k in diff_params:
        # Since we don't bind trainable values to Qiskit circuits,
        # we must remove them from the binding dictionary before binding.
        del params[k]

    return quantum_circuit.assign_parameters(params)


def _get_operation_params(instruction, unbound_params) -> list:
    """Extract the bound parameters from the operation.

    If the bound parameters are a Qiskit ParameterExpression, then replace it with
    the corresponding PennyLane variable from the unbound_params dictionary.

    Args:
        instruction (qiskit.circuit.Instruction): a qiskit's quantum circuit instruction
        unbound_params dict[qiskit.circuit.Parameter, Any]: a dictionary mapping
            qiskit parameters to trainable parameter values

    Returns:
        list: bound parameters of the given instruction
    """
    operation_params = []
    for p in instruction.params:
        _check_parameter_bound(p, unbound_params)

        if isinstance(p, ParameterExpression):
            if p.parameters:  # non-empty set = has unbound parameters
                ordered_params = tuple(p.parameters)
                f = lambdify(ordered_params, getattr(p, "_symbol_expr"), modules=qml.numpy)
                f_args = []
                for i_ordered_params in ordered_params:
                    f_args.append(unbound_params.get(i_ordered_params))
                operation_params.append(f(*f_args))
            else:  # needed for qiskit<0.43.1
                operation_params.append(float(p))  # pragma: no cover
        else:
            operation_params.append(p)

    return operation_params


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

    # pylint:disable=too-many-branches, fixme, protected-access
    def _function(*args, params: dict = None, wires: list = None, **kwargs):
        """Returns a PennyLane quantum function created based on the input QuantumCircuit.
        Warnings are created for each of the QuantumCircuit instructions that were
        not incorporated in the PennyLane template.

        If the QuantumCircuit contains unbound parameters, this function must be passed
        parameter values. These can be passed as positional arguments (provided in the same
        order that they are stored on the QuantumCircuit used to generate this function),
        keyword arguments with keywords matching the parameter names, or as a single dictionary
        with the parameters (not parameter names) as keys.

        Args:
            *args: positional arguments defining the value of the parameters that need to be assigned
            in the QuantumCircuit

        Kwargs:
            **kwargs: keyword arguments matching the names of the parameters beind defined
            params (dict): a dictionary with Parameter keys, specifying the parameters that need
                to be bound in the QuantumCircuit
            wires (Sequence[int] or int): The wires the converted template acts on.
                Note that if the original QuantumCircuit acted on :math:`N` qubits,
                then this must be a list of length :math:`N`.

        Returns:
            function: the new PennyLane template

        .. warning::

            Because this function takes ``params`` and ``wires`` as arguments in addition to taking
            any args and kwargs to set parameter values, the ``params`` and ``wires`` arguments are
            reserved, and cannot be parameter names in the QuantumCircuit.

        **Example**

        This function was created by doing something like:

        .. code-block:: python

            from qiskit.circuit import Parameter, QuantumCircuit
            import pennylane as qml

            a = Parameter("alpha")
            b = Parameter("beta")
            c = Parameter("gamma")

            qc = QuantumCircuit(2, 2)
            qc.rx(a, 0)
            qc.rx(b * c, 1)

            this_function = qml.from_qiskit(qc, measurements=[qml.PauliZ(0), qml.PauliZ(1)])

        For the circuit above, based on ``Parameters`` with names 'alpha', 'beta' and 'gamma', all of the following are
        valid inputs to set the Parameter values for this template:

        .. code-block:: python

            # positional arguments passed in alphabetical order (alpha, beta, gamma)
            circuit = this_function(np.pi, 0.5, np.pi)

            # kwargs are parameter names
            circuit = this_function(alpha=np.pi, beta=0.5, gamma=np.pi)

            # kwargs and args can be combined
            circuit = this_function(np.pi, np.pi, beta=0.5)

            # a dictionary - note that keys are not the names, but the Parameters themselves
            circuit = this_function(params={a: np.pi, b: 0.5, c: np.pi})

        """

        # organize parameters, format trainable parameter values correctly,
        # and then bind the parameters to the circuit
        params = _format_params_dict(quantum_circuit, params, *args, **kwargs)
        unbound_params = _extract_variable_refs(params)
        qc = _check_circuit_and_assign_parameters(quantum_circuit, params, unbound_params)

        # Wires from a qiskit circuit have unique IDs, so their hashes are unique too
        qc_wires = [hash(q) for q in qc.qubits]

        wire_map = map_wires(qc_wires, wires)

        # Stores the measurements encountered in the circuit
        # terminal_meas / mid_circ_meas -> terminal / mid-circuit measurements
        # mid_circ_regs -> maps the classical registers to the measurements done
        terminal_meas, mid_circ_meas = [], []
        mid_circ_regs = {}

        # Processing the dictionary of parameters passed
        for idx, circuit_instruction in enumerate(qc.data):
            (instruction, qargs, cargs) = circuit_instruction
            # the new Singleton classes have different names than the objects they represent,
            # but base_class.__name__ still matches
            instruction_name = getattr(instruction, "base_class", instruction.__class__).__name__
            # New Qiskit gates that are not natively supported by PL (identical
            # gates exist with a different name)
            # TODO: remove the following when gates have been renamed in PennyLane
            instruction_name = "U3Gate" if instruction_name == "UGate" else instruction_name

            # Define operator builders and helpers
            # _class -> PennyLane operation class object mapped from the Qiskit operation
            # _args and _kwargs -> Parameters required for instantiation of `_class`
            operation_class = None
            operation_wires = [wire_map[hash(qubit)] for qubit in qargs]
            operation_kwargs = {"wires": operation_wires}
            operation_args = []

            # Extract the bound parameters from the operation. If the bound parameters are a
            # Qiskit ParameterExpression, then replace it with the corresponding PennyLane
            # variable from the unbound_params dictionary.
            operation_params = _get_operation_params(instruction, unbound_params)

            if instruction_name in dagger_map:
                operation_class = qml.adjoint(dagger_map[instruction_name])

            elif instruction_name in inv_map:
                operation_class = getattr(pennylane_ops, inv_map[instruction_name])
                operation_args.extend(operation_params)
                if operation_class in (qml.QubitStateVector, qml.StatePrep):
                    operation_args = [np.array(operation_params)]

            elif isinstance(instruction, Measure):
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
                    operation_class = qml.measure
                    mid_circ_meas.append(qml.measure(wires=operation_wires))

                    # Allows for tracking conditional operations
                    for carg in cargs:
                        mid_circ_regs[carg] = mid_circ_meas[-1]

            else:

                try:
                    if not isinstance(instruction, (ControlFlowOp,)):
                        operation_args = [instruction.to_matrix()]
                        operation_class = qml.QubitUnitary

                except (AttributeError, QiskitError):
                    warnings.warn(
                        f"{__name__}: The {instruction_name} instruction is not supported by PennyLane,"
                        " and has not been added to the template.",
                        UserWarning,
                    )

            # Check if it is a conditional operation or conditional instruction
            instruction_cond = instruction.condition and instruction.condition[0] in mid_circ_regs
            if instruction_cond or isinstance(instruction, ControlFlowOp):
                # Iteratively recurse over to build different branches
                with qml.QueuingManager.stop_recording():
                    branch_funcs = [
                        partial(load(branch_inst, measurements=None), params=params, wires=wires)
                        for branch_inst in operation_params
                        if isinstance(branch_inst, QuantumCircuit)
                    ]

                # Get the functions for handling condition
                true_fn, false_fn, elif_fns, cond_op = _conditional_funcs(
                    instruction, cargs, operation_class, branch_funcs, instruction_name
                )
                res_reg, res_bit = cond_op

                # Check for elif branches (doesn't require qjit)
                if elif_fns:
                    m_val = sum(2**idx * mid_circ_regs[clbit] for idx, clbit in enumerate(res_reg))
                    for elif_bit, elif_branch in elif_fns:
                        qml.cond(m_val == elif_bit, elif_branch)(
                            *operation_args, **operation_kwargs
                        )

                # Check if just conditional requires some extra work
                if isinstance(res_bit, str):
                    # Handles the default case in the SwitchCaseOp
                    if res_bit == "SwitchDefault":
                        elif_bits = [elif_bit for (elif_bit, _) in elif_fns]
                        qml.cond(
                            reduce(
                                lambda m0, m1: m0 & m1,
                                [(m_val != elif_bit) for elif_bit in elif_bits],
                            ),
                            true_fn,
                        )(*operation_args, **operation_kwargs)
                # Just do the routine conditional
                else:
                    qml.cond(
                        mid_circ_regs[res_reg] == res_bit,
                        true_fn,
                        false_fn,
                    )(*operation_args, **operation_kwargs)

            # Check if it is not a mid-circuit measurement
            elif operation_class and not isinstance(instruction, Measure):
                operation_class(*operation_args, **operation_kwargs)

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


# pylint:disable=fixme, protected-access
def _conditional_funcs(ops, cargs, operation_class, branch_funcs, ctrl_flow_type):
    """Builds the conditional functions for Controlled flows
    
    This method returns the arguments to be used by the `qml.cond`
    for creating a classically controlled flow. 
    These are the branches (`true_fn`, `false_fn`, `elif_fns`) and
    the qiskit's classical condition, which has to be converted to
    the corresponding PennyLane mid-circuit measurement.
    """
    true_fn, false_fn, elif_fns = operation_class, None, ()
    # Logic for using legacy c_if
    if not isinstance(ops, ControlFlowOp):
        return true_fn, false_fn, elif_fns, ops.condition

    # Logic for handling IfElseOp
    if ctrl_flow_type == "IfElseOp":
        true_fn = branch_funcs[0]
        if len(branch_funcs) == 2:
            false_fn = branch_funcs[1]

    # Logic for handling SwitchCaseOp
    elif ctrl_flow_type == "SwitchCaseOp":
        elif_fns = []
        for res_bit, case in ops._case_map.items():
            if not isinstance(case, _DefaultCaseType):
                elif_fns.append((res_bit, branch_funcs[case]))
        ops.condition = [tuple(cargs), "SwitchCase"]
        if any((isinstance(case, _DefaultCaseType) for case in ops._case_map)):
            true_fn = branch_funcs[-1]
            ops.condition = [tuple(cargs), "SwitchDefault"]

    return true_fn, false_fn, elif_fns, ops.condition
