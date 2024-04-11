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
from qiskit.circuit import Parameter, ParameterExpression, Measure, Barrier
from qiskit.circuit.library import GlobalPhaseGate
from qiskit.exceptions import QiskitError
from sympy import lambdify

import pennylane as qml
import pennylane.ops as pennylane_ops

# pylint: disable=too-many-instance-attributes

QISKIT_OPERATION_MAP = {
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
    "S": ex.SGate,
    "T": ex.TGate,
    "SX": ex.SXGate,
    "Adjoint(S)": ex.SdgGate,
    "Adjoint(T)": ex.TdgGate,
    "Adjoint(SX)": ex.SXdgGate,
}


inv_map = {v.__name__: k for k, v in QISKIT_OPERATION_MAP.items()}

dagger_map = {"SdgGate": qml.S, "TdgGate": qml.T, "SXdgGate": qml.SX}


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

    # if no kwargs are passed, and a dictionary has been passed as a single argument, then assume it is params
    if params is None and not kwargs and (len(args) == 1 and isinstance(args[0], dict)):
        return args[0]

    if not args and not kwargs:
        return params

    # make params dict if using args and/or kwargs
    if params is not None:
        raise RuntimeError(
            "Cannot define parameters via the params kwarg when passing Parameter values "
            "as individual args or kwargs."
        )

    # create en empty params dict
    params = {}

    # populate it with any parameters defined as kwargs
    for k, v in kwargs.items():
        # the key needs to be the actual Parameter, whereas kwargs keys are parameter names
        qc_param = [p for p in quantum_circuit.parameters if p.name == k]
        if not qc_param:
            param_names = ", ".join([p.name for p in quantum_circuit.parameters])
            raise TypeError(
                f"Got unexpected parameter keyword argument '{k}'. Circuit contains parameters: {param_names}"
            )
        params[qc_param[0]] = v

    # get any parameters not defined in kwargs (may be all of them) and match to args in order
    arg_parameters = [p for p in quantum_circuit.parameters if p.name not in kwargs]
    # if too many args were passed to the function call, raise an error
    # all other checks regarding correct arguments will be processed in _check_circuit_and_assign_parameters
    # (based on the full params dict generated by this function), but this information can only be captured here
    if len(args) > len(arg_parameters):
        raise TypeError(
            f"Expected {len(arg_parameters)} positional argument{'s' if len(arg_parameters) > 1 else ''} but {len(args)} were given"
        )
    params.update(dict(zip(arg_parameters, args)))

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
            if getattr(v, "requires_grad", True):
                # Values can be arrays of size 1, need to extract the Python scalar
                # (this can happen e.g. when indexing into a PennyLane numpy array)
                if isinstance(v, np.ndarray):
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
                f"quantum function, as this argument is reserved"
            )

    if params is None:
        if quantum_circuit.parameters:
            s = "s" if len(quantum_circuit.parameters) > 1 else ""
            param_names = ", ".join([p.name for p in quantum_circuit.parameters])
            raise TypeError(
                f"Missing required argument{s} to define Parameter value{s} for: {param_names}"
            )
        return quantum_circuit

    # if any parameters are missing a value, raise an error
    undefined_params = set(quantum_circuit.parameters) - set(params)
    if undefined_params:
        s = "s" if len(undefined_params) > 1 else ""
        param_names = ", ".join([p.name for p in undefined_params])
        raise TypeError(
            f"Missing {len(undefined_params)} required argument{s} to define Parameter value{s} for: {param_names}"
        )

    for k in diff_params:
        # Since we don't bind trainable values to Qiskit circuits,
        # we must remove them from the binding dictionary before binding.
        del params[k]

    return quantum_circuit.assign_parameters(params)


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

    # pylint:disable=too-many-branches
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

        # organize parameters, format trainable parameter values correctly, and then bind the parameters to the circuit
        params = _format_params_dict(quantum_circuit, params, *args, **kwargs)
        unbound_params = _extract_variable_refs(params)
        qc = _check_circuit_and_assign_parameters(quantum_circuit, params, unbound_params)

        # Wires from a qiskit circuit have unique IDs, so their hashes are unique too
        qc_wires = [hash(q) for q in qc.qubits]

        wire_map = map_wires(qc_wires, wires)

        # Stores the measurements encountered in the circuit
        mid_circ_meas, terminal_meas = [], []

        # Processing the dictionary of parameters passed
        for idx, (op, qargs, _) in enumerate(qc.data):
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
                # variable from the unbound_params dictionary.

                pl_parameters = []
                for p in op.params:
                    _check_parameter_bound(p, unbound_params)

                    if isinstance(p, ParameterExpression):
                        if p.parameters:  # non-empty set = has unbound parameters
                            ordered_params = tuple(p.parameters)

                            f = lambdify(ordered_params, p._symbol_expr, modules=qml.numpy)
                            f_args = []
                            for i_ordered_params in ordered_params:
                                f_args.append(unbound_params.get(i_ordered_params))
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

            elif isinstance(op, Measure):
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

                # Allows for queing the mid-circuit measurements
                if not meas_terminal:
                    mid_circ_meas.append(qml.measure(wires=operation_wires))
                else:
                    terminal_meas.extend(operation_wires)

            else:
                try:
                    operation_matrix = op.to_matrix()
                    pennylane_ops.QubitUnitary(operation_matrix, wires=operation_wires)
                except (AttributeError, QiskitError):
                    warnings.warn(
                        f"{__name__}: The {instruction_name} instruction is not supported by PennyLane,"
                        " and has not been added to the template.",
                        UserWarning,
                    )
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
        rotations = circuit.diagonalizing_gates
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
        reg (int): the total number of qubits on the device

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
    if operation in ("QubitUnitary", "QubitStateVector", "StatePrep"):
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
    """

    # ToDo: I believe this could be extended to cover expectation values of Hamiltonians

    observables = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Identity": "I"}

    pauli_string = ["I"] * register_size
    pauli_string[mp.wires[0]] = observables[mp.obs.name]

    # Qiskit orders wires in the opposite direction compared to PL
    pauli_string.reverse()

    pauli_string = ("").join(pauli_string)

    return SparsePauliOp(pauli_string)
