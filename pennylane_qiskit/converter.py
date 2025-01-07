# Copyright 2021-2024 Xanadu Quantum Technologies Inc.

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
from typing import Dict, Any, Iterable, Sequence, Union
import warnings
from functools import partial, reduce

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector
from qiskit.circuit import Measure, Barrier, ControlFlowOp, Clbit
from qiskit.circuit import library as lib
from qiskit.circuit.classical import expr
from qiskit.circuit.controlflow.switch_case import _DefaultCaseType
from qiskit.circuit.library import GlobalPhaseGate
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import SparsePauliOp
from sympy import lambdify

import pennylane as qml
from pennylane.noise.conditionals import WiresIn, _rename
from pennylane.operation import AnyWires
import pennylane.ops as pennylane_ops
from pennylane.tape.tape import rotations_and_diagonal_measurements

from .noise_models import _build_noise_model_map

# pylint: disable=too-many-instance-attributes
QISKIT_OPERATION_MAP = {
    # native PennyLane operations also native to qiskit
    "PauliX": lib.XGate,
    "PauliY": lib.YGate,
    "PauliZ": lib.ZGate,
    "Hadamard": lib.HGate,
    "CNOT": lib.CXGate,
    "CZ": lib.CZGate,
    "SWAP": lib.SwapGate,
    "ISWAP": lib.iSwapGate,
    "RX": lib.RXGate,
    "RY": lib.RYGate,
    "RZ": lib.RZGate,
    "Identity": lib.IGate,
    "CSWAP": lib.CSwapGate,
    "CRX": lib.CRXGate,
    "CRY": lib.CRYGate,
    "CRZ": lib.CRZGate,
    "PhaseShift": lib.PhaseGate,
    "StatePrep": lib.Initialize,
    "Toffoli": lib.CCXGate,
    "QubitUnitary": lib.UnitaryGate,
    "U1": lib.U1Gate,
    "U2": lib.U2Gate,
    "U3": lib.U3Gate,
    "IsingZZ": lib.RZZGate,
    "IsingYY": lib.RYYGate,
    "IsingXX": lib.RXXGate,
    "S": lib.SGate,
    "T": lib.TGate,
    "SX": lib.SXGate,
    "Adjoint(S)": lib.SdgGate,
    "Adjoint(T)": lib.TdgGate,
    "Adjoint(SX)": lib.SXdgGate,
    "CY": lib.CYGate,
    "CH": lib.CHGate,
    "CPhase": lib.CPhaseGate,
    "CCZ": lib.CCZGate,
    "ECR": lib.ECRGate,
    "Barrier": lib.Barrier,
    "Adjoint(GlobalPhase)": lib.GlobalPhaseGate,
}

inv_map = {v.__name__: k for k, v in QISKIT_OPERATION_MAP.items()}

dagger_map = {
    "SdgGate": qml.S,
    "TdgGate": qml.T,
    "SXdgGate": qml.SX,
    "GlobalPhaseGate": qml.GlobalPhase,
}

referral_to_forum = (
    "\n \nIf you are experiencing any difficulties with converting circuits from Qiskit, you can reach out "
    "\non the PennyLane forum at https://discuss.pennylane.ai/c/pennylane-plugins/pennylane-qiskit/"
)


def _check_parameter_bound(
    param: Parameter,
    unbound_params: Dict[Union[Parameter, ParameterVector], Any],
):
    """Utility function determining if a certain parameter in a QuantumCircuit has
    been bound.

    Args:
        param (qiskit.circuit.Parameter): the parameter to be checked
        unbound_params (dict[qiskit.circuit.Parameter | qiskit.circuit.ParameterVector, Any]):
            a dictionary mapping qiskit parameters (or vectors) to trainable parameter values
    """
    if isinstance(param, ParameterVectorElement):
        if param.vector not in unbound_params:
            raise ValueError(f"The vector of parameter {param} was not bound correctly.")

    elif isinstance(param, Parameter):
        if param not in unbound_params:
            raise ValueError(f"The parameter {param} was not bound correctly.")


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

    # Disabling "strict" assignment allows extra parameters to be ignored.
    return quantum_circuit.assign_parameters(params, strict=False)


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
                f_args = []
                f_params = []

                # Ensure duplicate subparameters are only appended once.
                f_param_names = set()

                for subparam in p.parameters:
                    if isinstance(subparam, ParameterVectorElement):
                        # Unfortunately, parameter vector elements are named using square brackets.
                        # As a result, element names are not a valid Python identifier which causes
                        # issues with SymPy. To get around this, we create a temporary parameter
                        # representing the entire vector and pass that into the SymPy function.
                        parameter = Parameter(subparam.vector.name)
                        argument = unbound_params.get(subparam.vector)
                    else:
                        parameter = subparam
                        argument = unbound_params.get(subparam)

                    if parameter.name not in f_param_names:
                        f_param_names.add(parameter.name)
                        f_params.append(parameter)
                        f_args.append(argument)

                f_expr = getattr(p, "_symbol_expr")
                f = lambdify(f_params, f_expr, modules=qml.numpy)

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
        measurements (None | pennylane.measurements.MeasurementProcess | list[pennylane.measurements.MeasurementProcess]):
            the PennyLane `measurements <https://docs.pennylane.ai/en/stable/introduction/measurements.html>`_
            that override the terminal measurements that may be present in the input circuit

    Returns:
        function: The resulting PennyLane template.
    """

    # pylint:disable=too-many-branches, fixme, protected-access, too-many-nested-blocks
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
        # pylint: disable=too-many-nested-blocks
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
            # operation_class -> PennyLane operation class object mapped from the Qiskit operation
            # operation_args and operation_kwargs -> Parameters required for the
            # instantiation of `operation_class`
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
                operation_args.extend(operation_params)

            elif instruction_name in inv_map:
                operation_class = getattr(pennylane_ops, inv_map[instruction_name])
                operation_args.extend(operation_params)
                if operation_class is qml.StatePrep:
                    operation_args = [np.array(operation_params)]

            elif isinstance(instruction, Measure):
                # Store the current operation wires and registers
                op_wires, op_cregs = set(operation_wires), set(cargs)
                # If final measurements are given then discriminate
                # between the types of measurement encountered.
                meas_terminal = False
                if measurements is not None:
                    # Look-ahead for more gate(s) on its wire(s)
                    meas_terminal = True
                    for next_op, next_qargs, next_cargs in qc.data[idx + 1 :]:
                        # Check if the subsequent conditional is measurement needing
                        if isinstance(next_op, ControlFlowOp):
                            if set(next_cargs) & op_cregs:
                                meas_terminal = False
                                break
                        elif next_op.condition:  # For legacy c_if
                            next_op_reg = next_op.condition[0]
                            if isinstance(next_op_reg, Clbit):
                                next_op_reg = [next_op_reg]
                            if set(next_op_reg) & op_cregs:
                                meas_terminal = False
                                break
                        # Check if the subsequent next_op is measurement interfering
                        if not isinstance(next_op, (Barrier, GlobalPhaseGate)):
                            next_op_wires = set(wire_map[hash(qubit)] for qubit in next_qargs)
                            # Check if there's any overlapping wires
                            if next_op_wires & op_wires:
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
            if instruction.condition or isinstance(instruction, ControlFlowOp):
                # Iteratively recurse over to build different branches for the condition
                with qml.QueuingManager.stop_recording():
                    branch_funcs = [
                        partial(load(branch_inst, measurements=None), params=params, wires=wires)
                        for branch_inst in operation_params
                        if isinstance(branch_inst, QuantumCircuit)
                    ]

                # Get the functions for handling condition
                # true fns | false fns -> true | false branches
                # inst_cond -> qiskit's conditional expression
                true_fns, false_fns, inst_cond = _conditional_funcs(
                    instruction, operation_class, branch_funcs, instruction_name
                )

                # Process qiskit condition to PL mid-circ meas conditions
                # pl_meas_conds -> PL's conditional expression with mid-circuit meas.
                # length(pl_meas_conds) == len(true_fns) ==> True
                pl_meas_conds = _process_condition(inst_cond, mid_circ_regs, instruction_name)

                # Iterate over each of the conditional triplet and apply the condition via qml.cond
                for pl_meas_cond, true_fn, false_fn in zip(pl_meas_conds, true_fns, false_fns):
                    qml.cond(pl_meas_cond, true_fn, false_fn)(*operation_args, **operation_kwargs)

            # Check if it is not a mid-circuit measurement
            elif operation_class and not isinstance(instruction, Measure):
                operation_class(*operation_args, **operation_kwargs)

        # Use the user-provided measurements
        # an empty iterable is treated as a user-provided measurement with no measurements to queue
        if measurements is not None:
            if not qml.queuing.QueuingManager.active_context():
                return measurements

            if isinstance(measurements, Iterable):
                return [qml.apply(meas) for meas in measurements]

            return qml.apply(measurements)

        return tuple(mid_circ_meas + list(map(qml.measure, terminal_meas))) or None

    return _function


def load_qasm(qasm_string: str, measurements=None):
    """Loads a PennyLane template from a QASM string.

    Args:
        qasm_string (str): the name of the QASM string
        measurements (None | pennylane.measurements.MeasurementProcess | list[pennylane.measurements.MeasurementProcess]):
            the PennyLane `measurements <https://docs.pennylane.ai/en/stable/introduction/measurements.html>`_
            that override the terminal measurements that may be present in the input circuit

    Returns:
        function: the new PennyLane template
    """
    return load(QuantumCircuit.from_qasm_str(qasm_string), measurements=measurements)


def load_qasm_from_file(file: str):
    """Loads a PennyLane template from a QASM file.
    Args:
        file (str): the name of the QASM file
    Returns:
        function: the new PennyLane template
    """

    return load(QuantumCircuit.from_qasm_file(file), measurements=[])


# diagonalize is currently only used if measuring
# maybe always diagonalize when measuring, and never when not?
# will this be used for a user-facing function to convert from PL to Qiskit as well?
def circuit_to_qiskit(circuit, register_size, diagonalize=True, measure=True):
    """Builds the circuit objects based on the operations and measurements
    specified to apply.

    Args:
        circuit (QuantumTape): the circuit applied
            to the device
        register_size (int): the total number of qubits on the device the circuit is
            executed on; this must include any qubits not used in the given
            circuit to ensure correct indexing of the returned samples

    Keyword args:
        diagonalize (bool): whether or not to apply diagonalizing gates before the
            measurements
        measure (bool): whether or not to apply measurements at the end of the circuit;
            a full circuit is represented either as a Qiskit circuit with operations
            and measurements (measure=True), or a Qiskit circuit with only operations,
            paired with a Qiskit Estimator defining the measurement process.

    Returns:
        QuantumCircuit: the qiskit equivalent of the given circuit
    """

    reg = QuantumRegister(register_size)

    if not measure:
        qc = QuantumCircuit(reg, name="temp")

        for op in circuit.operations:
            qc &= operation_to_qiskit(op, reg)

        return qc

    creg = ClassicalRegister(register_size)
    qc = QuantumCircuit(reg, creg, name="temp")

    for op in circuit.operations:
        qc &= operation_to_qiskit(op, reg, creg)

    # rotate the state for measurement in the computational basis
    # ToDo: check this in cases with multiple different bases
    if diagonalize:
        rotations, measurements = rotations_and_diagonal_measurements(circuit)
        for _, m in enumerate(measurements):
            if m.obs is not None:
                rotations.extend(m.obs.diagonalizing_gates())

        for rot in rotations:
            qc &= operation_to_qiskit(rot, reg, creg)

    # barrier ensures we first do all operations, then do all measurements
    qc.barrier(reg)
    # we always measure the full register
    qc.measure(reg, creg)

    return qc


def operation_to_qiskit(operation, reg, creg=None):
    """Take a Pennylane operator and convert to a Qiskit circuit

    Args:
        operation (List[pennylane.Operation]): operation to be converted
        reg (Quantum Register): the total number of qubits on the device
        creg (Classical Register): classical register

    Returns:
        QuantumCircuit: a quantum circuit objects containing the translated operation
    """
    op_wires = operation.wires
    par = operation.parameters

    for idx, p in enumerate(par):
        if isinstance(p, np.ndarray):
            # Convert arrays so that Qiskit accepts the parameter
            par[idx] = p.tolist()

    operation = operation.name

    mapped_operation = QISKIT_OPERATION_MAP[operation]

    qregs = [reg[i] for i in op_wires.labels]

    # Need to revert the order of the quantum registers used in
    # Qiskit such that it matches the PennyLane ordering
    if operation in ("QubitUnitary", "StatePrep"):
        qregs = list(reversed(qregs))

    if creg:
        dag = circuit_to_dag(QuantumCircuit(reg, creg, name=""))
    else:
        dag = circuit_to_dag(QuantumCircuit(reg, name=""))
    gate = mapped_operation(*par)

    dag.apply_operation_back(gate, qargs=qregs)
    circuit = dag_to_circuit(dag)

    return circuit


def mp_to_pauli(mp, register_size):
    """Convert a Pauli observable to a SparsePauliOp for measurement via Estimator

    Args:
        mp(Union[ExpectationMP, VarianceMP]): MeasurementProcess to be converted to a SparsePauliOp
        register_size(int): total size of the qubit register being measured

    Returns:
        SparsePauliOp: the ``SparsePauliOp`` of the given Pauli observable
    """
    op = mp.obs

    if op.pauli_rep:
        pauli_strings = [
            "".join(
                ["I" if i not in pauli_term.wires else pauli_term[i] for i in range(register_size)][
                    ::-1
                ]  ## Qiskit follows opposite wire order convention
            )
            for pauli_term in op.pauli_rep.keys()
        ]
        coeffs = list(op.pauli_rep.values())
    else:
        raise ValueError(f"The operator {op} does not have a representation for SparsePauliOp")

    return SparsePauliOp(data=pauli_strings, coeffs=coeffs).simplify()


def load_pauli_op(
    pauli_op: SparsePauliOp,
    params: Any = None,
    wires: Union[Sequence, None] = None,
) -> qml.operation.Operator:
    """Loads a PennyLane ``Operator`` from a Qiskit `SparsePauliOp
    <https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp>`_.

    Args:
        pauli_op (qiskit.quantum_info.SparsePauliOp): the ``SparsePauliOp`` to be converted
        params (Any): optional assignment of coefficient values for the ``SparsePauliOp``; see the
            `Qiskit documentation <https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp#assign_parameters>`__
            to learn more about the expected format of these parameters
        wires (Sequence | None): optional assignment of wires for the converted ``SparsePauliOp``;
            if the original ``SparsePauliOp`` acted on :math:`N` qubits, then this must be a
            sequence of length :math:`N`

    Returns:
        pennylane.operation.Operator: The equivalent PennyLane operator.

    .. note::

        The wire ordering convention differs between PennyLane and Qiskit: PennyLane wires are
        enumerated from left to right, while the Qiskit convention is to enumerate from right to
        left. This means a ``SparsePauliOp`` term defined by the string ``"XYZ"`` applies ``Z`` on
        wire 0, ``Y`` on wire 1, and ``X`` on wire 2. For more details, see the
        `String representation <https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Pauli>`_
        section of the Qiskit documentation for the ``Pauli`` class.

    **Example**

    Consider the following script which creates a Qiskit ``SparsePauliOp``:

    .. code-block:: python

        from qiskit.quantum_info import SparsePauliOp

        qiskit_op = SparsePauliOp(["II", "XY"])

    The ``SparsePauliOp`` contains two terms and acts over two qubits:

    >>> qiskit_op
    SparsePauliOp(['II', 'XY'],
                  coeffs=[1.+0.j, 1.+0.j])

    To convert the ``SparsePauliOp`` into a PennyLane ``Operator``, use:

    >>> from pennylane_qiskit import load_pauli_op
    >>> load_pauli_op(qiskit_op)
    I(0) + X(1) @ Y(0)

    .. details::
        :title: Usage Details

        You can convert a parameterized ``SparsePauliOp`` into a PennyLane operator by assigning
        literal values to each coefficient parameter. For example, the script

        .. code-block:: python

            import numpy as np
            from qiskit.circuit import Parameter

            a, b, c = [Parameter(var) for var in "abc"]
            param_qiskit_op = SparsePauliOp(["II", "XZ", "YX"], coeffs=np.array([a, b, c]))

        defines a ``SparsePauliOp`` with three coefficients (parameters):

        >>> param_qiskit_op
        SparsePauliOp(['II', 'XZ', 'YX'],
              coeffs=[ParameterExpression(1.0*a), ParameterExpression(1.0*b),
         ParameterExpression(1.0*c)])

        The ``SparsePauliOp`` can be converted into a PennyLane operator by calling the conversion
        function and specifying the value of each parameter using the ``params`` argument:

        >>> load_pauli_op(param_qiskit_op, params={a: 2, b: 3, c: 4})
        (
            (2+0j) * I(0)
          + (3+0j) * (X(1) @ Z(0))
          + (4+0j) * (Y(1) @ X(0))
        )

        Similarly, a custom wire mapping can be applied to a ``SparsePauliOp`` as follows:

        >>> wired_qiskit_op = SparsePauliOp("XYZ")
        >>> wired_qiskit_op
        SparsePauliOp(['XYZ'],
              coeffs=[1.+0.j])
        >>> load_pauli_op(wired_qiskit_op, wires=[3, 5, 7])
        Y(5) @ Z(3) @ X(7)
    """
    if not isinstance(pauli_op, SparsePauliOp):
        raise ValueError(f"The operator {pauli_op} is not a valid Qiskit SparsePauliOp.")

    if wires is not None and len(wires) != pauli_op.num_qubits:
        raise RuntimeError(
            f"The specified number of wires - {len(wires)} - does not match the "
            f"number of qubits the SparsePauliOp acts on."
        )

    wire_map = map_wires(range(pauli_op.num_qubits), wires)

    if params:
        pauli_op = pauli_op.assign_parameters(params)

    op_map = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ, "I": qml.Identity}

    coeffs = pauli_op.coeffs
    if ParameterExpression in [type(c) for c in coeffs]:
        raise RuntimeError(f"Not all parameter expressions are assigned in coeffs {coeffs}")

    qiskit_terms = pauli_op.paulis
    pl_terms = []

    for term in qiskit_terms:
        # term is a special Qiskit type. Iterating over the term goes right to left
        # in accordance with Qiskit wire order convention, i.e. `enumerate("XZ")` will be
        # [(0, "Z"), (1, "X")], so we don't need to reverse to match the PL convention.
        operators = [op_map[str(op)](wire_map[wire]) for wire, op in enumerate(term)]
        pl_terms.append(qml.prod(*operators).simplify())

    return qml.dot(coeffs, pl_terms)


# pylint:disable=protected-access
def _conditional_funcs(inst, operation_class, branch_funcs, ctrl_flow_type):
    """Builds the conditional functions for Controlled flows.

    This method returns the arguments to be used by the ``qml.cond``
    for creating the classically controlled flow. These are the
    branches - ``true_fns`` and ``false_fns``, that contains the quantum
    functions to be applied based on the results of the condition.

    Additionally, we also return qiskit's classical condition,
    which we convert to the corresponding PennyLane mid-circuit
    measurement in the ``_process_condition`` method. These conditions
    are stored in the ``condition`` attribute for all the controlled ops,
    except for ``SwitchCaseOp`` for which it is stored in the ``target``
    attribute. For the latter operation, we set the ``condition`` ourselves
    with information from ``target`` and the information required
    by us for the processing of the condition.

    Args:
        inst (Instruction): Qiskit's ``Instruction`` object
        operation_class (Operation): PennyLane ``Operation`` for legacy controlled functionality
        branch_funcs (List[partial]): Iterable of possible branching circuits for the condition.
        ctrl_flow_type (str): represents the type of ``ControlledFlowOp``

    Returns:
        Tuple[true_fns, false_fns, condition]: the condition and the corresponding branches
    """
    true_fns, false_fns = [operation_class], [None]

    # Logic for using legacy c_if
    if not isinstance(inst, ControlFlowOp):
        return true_fns, false_fns, inst.condition

    # Logic for handling IfElseOp
    if ctrl_flow_type == "IfElseOp":
        true_fns[0] = branch_funcs[0]
        if len(branch_funcs) == 2:
            false_fns[0] = branch_funcs[1]

    # Logic for handling SwitchCaseOp
    elif ctrl_flow_type == "SwitchCaseOp":
        true_fns, res_bits = [], []
        for case, b_idx in inst._case_map.items():
            if not isinstance(case, _DefaultCaseType):
                true_fns.append(branch_funcs[b_idx])
                res_bits.append(case)

        # Switch ops condition is None by default
        # So we make up a custom one for it ourselves
        inst.condition = [inst.target, res_bits, "SwitchCase"]
        if any((isinstance(case, _DefaultCaseType) for case in inst._case_map)):
            true_fns.append(branch_funcs[-1])
            # Marker for we have a default case scenario
            inst.condition[-1] = "SwitchDefault"
        false_fns = [None] * len(true_fns)

    return true_fns, false_fns, inst.condition


def _process_condition(cond_op, mid_circ_regs, instruction_name):
    """Process the condition to corresponding measurement value.

    In Qiskit, the generic form of condition is of two types:
    1. ``tuple[ClassicalRegister, int] or tuple[Clbit, int]``
    2. ``expr.Expr``
    In addition for this, we have another custom type:
    3. ``List(Target: Condition, Vals: List[Int], Type: str)``

    Args:
        cond_op (condition): condition as described above
        mid_circ_regs (dict): dictionary that maps the utilized qiskit's classical bits
            to the performed PennyLane's mid-circuit measurements
        instruction_name (str): represents the name of the instruction. Used in raising
            an informative warning in case processing of the condition fails.

    Returns:
        pl_meas: list of corresponding mid-circuit measurements to be used in ``qml.cond``
    """
    # container for PL measurements operators
    pl_meas = []
    condition = cond_op

    # Check if the condition is as a tuple -> (Clbit/Clreg, Val)
    if isinstance(condition, tuple):
        clbits = [condition[0]] if isinstance(condition[0], Clbit) else list(condition[0])

        # Proceed only if we have access to all conditioned classical bits
        if all(clbit in mid_circ_regs for clbit in clbits):
            pl_meas.append(
                sum(2**idx * mid_circ_regs[clbit] for idx, clbit in enumerate(clbits))
                == condition[1]
            )

    # Check if the condition is coming form a SwitchCase -> (Target, Vals, Type)
    if isinstance(condition, list):
        meas_pl_ops = _process_switch_condition(condition, mid_circ_regs)
        pl_meas.extend(meas_pl_ops)

    # Check if the condition is an Expr
    if isinstance(condition, expr.Expr):
        meas_pl_op = _expr_evaluation(condition, mid_circ_regs)
        if meas_pl_op is not None:
            pl_meas.append(meas_pl_op)

    # Check if were able to add the meas values
    # Else raise a warning before returning
    if pl_meas:
        return pl_meas

    warnings.warn(
        f"The provided {condition} for {instruction_name} uses additional classical information that cannot be returned or processed.",
        UserWarning,
    )
    return pl_meas


def _process_switch_condition(condition, mid_circ_regs):
    """Helper method for processing condition for SwtichCaseOp.

    Args:
        condition (condition): condition as described in ``_process_condition`` of the
            third type - ``List(Target: Condition, Vals: List[Int], Type: str)``
        mid_circ_regs (dict): dictionary that maps the utilized qiskit's classical bits
            to the performed PennyLane's mid-circuit measurements

    Returns:
        meas_pl_ops: list of corresponding mid-circuit measurements to be used in ``qml.cond``
    """
    use_switch_default = True
    # if the target is not an Expr
    if not isinstance(condition[0], expr.Expr):
        # Prepare the classical bits used for the condition
        clbits = [condition[0]] if isinstance(condition[0], Clbit) else list(condition[0])

        # Proceed only if we have access to all conditioned classical bits
        meas_pl_op = None
        if all(clbit in mid_circ_regs for clbit in clbits):
            # Build an integer representation for each switch case
            meas_pl_op = sum(2**idx * mid_circ_regs[clbit] for idx, clbit in enumerate(clbits))
            # Non Expr-based condition can have 2**#clbits outputs: 0, ..., 2**#clbits - 1
            # If all of them are already covered in the given cases, skip the default case.
            use_switch_default = bool(set(condition[1]) ^ set(range(2 ** len(clbits))))

    # if the target is an Expr
    else:
        meas_pl_op = _expr_evaluation(condition[0], mid_circ_regs)
        # Expr-based condition can have two Boolean outputs: 0 and 1
        # If both of them are already covered in the given cases, skip the default case.
        use_switch_default = bool(set(condition[1]) ^ {0, 1})

    meas_pl_ops = []
    if meas_pl_op is not None:
        # Add a measurement for each of the switch cases
        meas_pl_ops.extend([meas_pl_op == clval for clval in condition[1]])
        # If we have default case, add an additional measurement for it
        if condition[2] == "SwitchDefault" and use_switch_default:
            meas_pl_ops.append(
                reduce(
                    lambda m0, m1: m0 & m1,
                    [(meas_pl_op != clval) for clval in condition[1]],
                )
            )
    return meas_pl_ops


# pylint:disable = unbalanced-tuple-unpacking
def _expr_evaluation(condition, mid_circ_regs):
    """Evaluates the ``Expr`` condition

    Args:
        condition (condition): condition as described in ``_process_condition``
            of the second type - ``Expr``
        mid_circ_regs (dict): dictionary that maps the utilized qiskit's classical bits
            to the performed PennyLane's mid-circuit measurements

    Returns:
        condition_res: corresponding mid-circuit measurements to be used in ``qml.cond``
    """

    # Maps qiskit `expr` names to their mathematical logic
    _expr_mapping = {
        "BIT_AND": lambda meas1, meas2: meas1 & meas2,
        "BIT_OR": lambda meas1, meas2: meas1 | meas2,
        "BIT_XOR": lambda meas1, meas2: meas1 ^ meas2,
        "LOGIC_AND": lambda meas1, meas2: meas1 and meas2,
        "LOGIC_OR": lambda meas1, meas2: meas1 or meas2,
        "EQUAL": lambda meas1, meas2: meas1 == meas2,
        "NOT_EQUAL": lambda meas1, meas2: meas1 != meas2,
        "LESS": lambda meas1, meas2: meas1 < meas2,
        "LESS_EQUAL": lambda meas1, meas2: meas1 <= meas2,
        "GREATER": lambda meas1, meas2: meas1 > meas2,
        "GREATER_EQUAL": lambda meas1, meas2: meas1 >= meas2,
    }

    # Get the left and right classical controls
    cond1, cond2 = condition.left, condition.right

    # Iterate over each and extract classical bits
    clbits, clvals = [], []
    for _, carg in enumerate([cond1, cond2]):
        # We don't need to work with Qiskit's Cast expr op,
        # as we'll be casting stuff manually by ourselves.
        if isinstance(carg, expr.Cast):
            carg = carg.operand

        if isinstance(carg, expr.Value):
            clvals.append([carg.value])
        elif isinstance(carg, expr.Var):
            clbits.append([carg.var] if isinstance(carg.var, Clbit) else list(carg.var))

    # Proceed only if we have access to all conditioned classical bits
    for idx, clreg in enumerate(clbits):
        if not all(clbit in mid_circ_regs for clbit in clreg):
            return None
        clbits[idx] = [mid_circ_regs[clbit] for clbit in clreg]

    # Flag for tracking if it is a bitwise operation.
    # bitwise = true -> apply on each bit of the binary forms
    # bitwise = false -> apply on the integer forms as whole
    bitwise_flag = False
    condition_name = condition.op.name
    if condition_name[:3] == "BIT":
        bitwise_flag = True

    # divide the bits among left and right cbit registers
    if len(clbits) == 2:
        condition_res = _expr_eval_clregs(clbits, _expr_mapping[condition_name], bitwise_flag)

    # divide the bits into a cbit register and integer
    else:
        condition_res = _expr_eval_clvals(
            clbits, clvals, _expr_mapping[condition_name], bitwise_flag
        )

    return condition_res


def _expr_eval_clregs(clbits, expr_func, bitwise=False):
    """Helper method for ``Expr`` evaluation when two registers are present.

    Args:
        clbits (List[List[int], List[int]]): list of two registers represented by the
            corresponding mid-circuit measurements mapped from their classical bits.
        expr_func (lambda): mapped lambda func from ``_expr_mapping`` in the ``_expr_evaluation``
            method that performs the corresponding mathematical logic.
        bitwise (bool): flag that specifies if the ``expr_func`` is performed on individual bits.

    Returns:
        condition_res: corresponding mid-circuit measurements to be used in ``qml.cond``
    """
    clreg1, clreg2 = clbits
    # Make both the bits of the same width with padding
    # We swap the registers so that we only have to pad right one.
    if len(clreg1) < len(clreg2):
        clreg1, clreg2 = clreg2, clreg1
    clreg2 = [0] * (len(clreg2) - len(clreg1)) + clreg2

    # For bitwise operations we need to work with individual bits
    # So we build an integer form 'after' performing the operation.
    if bitwise:
        condition_res = sum(
            2**idx * expr_func(meas1, meas2)
            for idx, (meas1, meas2) in enumerate(zip(clreg1, clreg2))
        )

    # For other operations we need to work with all bits at once,
    # So we build an integer form 'before' performing the operation.
    else:
        meas1, meas2 = [
            sum(2**idx * meas for idx, meas in enumerate(clreg)) for clreg in [clreg1, clreg2]
        ]
        condition_res = expr_func(meas1, meas2)

    return condition_res


def _expr_eval_clvals(clbits, clvals, expr_func, bitwise=False):
    """Helper method for ``Expr`` evaluation when one register and one integer value is present.

    Args:
        clbits (List[List[int]]): list of two registers represented by the
            corresponding mid-circuit measurements mapped from their classical bits.
        clvals (List[List[int]])
        expr_func (lambda): mapped lambda func from ``_expr_mapping`` in the ``_expr_evaluation``
            method that performs the corresponding mathematical logic.
        bitwise (bool): flag that specifies if the ``expr_func`` is performed on individual bits.

    Returns:
        condition_res: corresponding mid-circuit measurements to be used in ``qml.cond``
    """
    [clreg1], [[clreg2]] = clbits, clvals
    # For bitwise operations, we first need a binary form for clreg2
    if bitwise:
        # Number of bits should be max of the binary-rep of the clvals or clreg.
        num_bits = max(len(clreg1), int(np.ceil(np.log2(clreg2))))
        clreg2 = map(int, np.binary_repr(clreg2, width=num_bits))
        clreg1 = [0] * (num_bits - len(clreg1)) + clreg1
        condition_res = sum(
            2**idx * expr_func(meas1, meas2)
            for idx, (meas1, meas2) in enumerate(zip(clreg1, clreg2))
        )

    # For other operations, we just need the integer form of clreg1
    else:
        meas1 = sum(2**idx * meas for idx, meas in enumerate(clreg1))
        condition_res = expr_func(meas1, clreg2)

    return condition_res


def load_noise_model(
    noise_model, verbose: bool = False, decimal_places: Union[int, None] = None
) -> qml.NoiseModel:
    """Loads a PennyLane `NoiseModel <https://docs.pennylane.ai/en/stable/code/api/pennylane.NoiseModel.html>`_
    from a Qiskit `noise model <https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html>`_.

    Args:
        noise_model (qiskit_aer.noise.NoiseModel): a Qiskit noise model object
        verbose (bool): when printing a ``NoiseModel``, a complete list of Kraus matrices for each ``qml.QubitChannel``
            is displayed with ``verbose=True``. By default, ``verbose=False`` and only the number of Kraus matrices and
            the number of qubits they act on is displayed for brevity.
        decimal_places (int | None): number of decimal places to round the elements of Kraus matrices when they are being
            displayed for each ``qml.QubitChannel`` when ``verbose=True``.

    Returns:
        pennylane.NoiseModel: An equivalent noise model constructed in PennyLane

    Raises:
        ValueError: When an encountered quantum error cannot be converted.

    .. note::

        Currently, PennyLane noise models do not support readout errors, so those will be skipped during conversion.

    **Example**

    Consider the following noise model constructed in Qiskit:

    >>> import qiskit_aer.noise as noise
    >>> error_1 = noise.depolarizing_error(0.001, 1) # 1-qubit noise
    >>> error_2 = noise.depolarizing_error(0.01, 2) # 2-qubit noise
    >>> noise_model = noise.NoiseModel()
    >>> noise_model.add_all_qubit_quantum_error(error_1, ['rz', 'ry'])
    >>> noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

    This noise model can be converted into PennyLane using:

    >>> load_noise_model(noise_model)
    NoiseModel({
        OpIn(['RZ', 'RY']): QubitChannel(num_kraus=4, num_wires=1)
        OpIn(['CNOT']): QubitChannel(num_kraus=16, num_wires=2)
    })
    """
    # Build model maps for quantum error and readout errors in the noise model
    qerror_dmap, _ = _build_noise_model_map(noise_model)
    model_map = {}
    for error, wires_map in qerror_dmap.items():
        conditions = []
        for wires, operations in wires_map.items():
            cond = qml.noise.op_in(operations)
            if wires != AnyWires:
                cond &= WiresIn(wires)
            conditions.append(cond)
        fcond = reduce(lambda cond1, cond2: cond1 | cond2, conditions)

        noise = qml.noise.partial_wires(error)
        if isinstance(error, qml.QubitChannel):
            if not verbose:
                kraus_shape = qml.math.shape(error.data)
                n_kraus, n_wires = kraus_shape[0], int(np.log2(kraus_shape[1]))
                noise = _rename(f"QubitChannel(num_kraus={n_kraus}, num_wires={n_wires})")(noise)
            elif verbose and decimal_places is not None:
                kraus_matrices = list(np.round(error.data, decimals=decimal_places))
                noise = _rename(f"QubitChannel(Klist={kraus_matrices})")(noise)

        model_map[fcond] = noise

    return qml.NoiseModel(model_map)
