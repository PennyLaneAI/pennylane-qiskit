# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This module contains the base class for all PennyLane Lightning simulator devices,
and interfaces with C++ for improved performance.
"""
from itertools import islice, product
from typing import List

import numpy as np
import pennylane as qml
from pennylane import BasisState, QubitDevice, StatePrep
from pennylane.devices import DefaultQubitLegacy
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operation
from pennylane.wires import Wires

from ._serialize import QuantumScriptSerializer
from ._version import __version__


def _chunk_iterable(iteration, num_chunks):
    "Lazy-evaluated chunking of given iterable from https://stackoverflow.com/a/22045226"
    iteration = iter(iteration)
    return iter(lambda: tuple(islice(iteration, num_chunks)), ())


class LightningBase(QubitDevice):
    """PennyLane Lightning Base device.

    This intermediate base class provides device-agnostic functionalities.

    Args:
        wires (int): the number of wires to initialize the device with
        c_dtype: Datatypes for statevector representation. Must be one of
            ``np.complex64`` or ``np.complex128``.
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            stochastic return values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically.
        batch_obs (bool): Determine whether we process observables in parallel when computing
            the jacobian. This value is only relevant when the lightning qubit is built with
            OpenMP.
    """

    pennylane_requires = ">=0.34"
    version = __version__
    author = "Xanadu Inc."
    short_name = "lightning.base"
    _CPP_BINARY_AVAILABLE = True

    def __init__(
        self,
        wires,
        *,
        c_dtype=np.complex128,
        shots=None,
        batch_obs=False,
    ):
        if c_dtype is np.complex64:
            r_dtype = np.float32
            self.use_csingle = True
        elif c_dtype is np.complex128:
            r_dtype = np.float64
            self.use_csingle = False
        else:
            raise TypeError(f"Unsupported complex type: {c_dtype}")
        super().__init__(wires, shots=shots, r_dtype=r_dtype, c_dtype=c_dtype)
        self._batch_obs = batch_obs

    @property
    def stopping_condition(self):
        """.BooleanFn: Returns the stopping condition for the device. The returned
        function accepts a queueable object (including a PennyLane operation
        and observable) and returns ``True`` if supported by the device."""

        def accepts_obj(obj):
            if obj.name == "QFT":
                return len(obj.wires) < 10
            if obj.name == "GroverOperator":
                return len(obj.wires) < 13
            is_not_tape = not isinstance(obj, qml.tape.QuantumTape)
            is_supported = getattr(self, "supports_operation", lambda name: False)(obj.name)
            return is_not_tape and is_supported

        return qml.BooleanFn(accepts_obj)

    # pylint: disable=missing-function-docstring
    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qubit",
            supports_analytic_computation=True,
            supports_broadcasting=False,
            returns_state=True,
        )
        return capabilities

    # To be able to validate the adjoint method [_validate_adjoint_method(device)],
    #  the qnode requires the definition of:
    # ["_apply_operation", "_apply_unitary", "adjoint_jacobian"]
    # pylint: disable=missing-function-docstring
    def _apply_operation(self):
        pass

    # pylint: disable=missing-function-docstring
    def _apply_unitary(self):
        pass

    def _init_process_jacobian_tape(self, tape, starting_state, use_device_state):
        """Generate an initial state vector for ``_process_jacobian_tape``."""

    @property
    def create_ops_list(self):
        """Returns create_ops_list function of the matching precision."""

    def probability_lightning(self, wires):
        """Return the probability of each computational basis state."""

    def vjp(self, measurements, grad_vec, starting_state=None, use_device_state=False):
        """Generate the processing function required to compute the vector-Jacobian
        products of a tape.
        """

    def probability(self, wires=None, shot_range=None, bin_size=None):
        """Return the probability of each computational basis state.

        Devices that require a finite number of shots always return the
        estimated probability.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided are traced out of the system.
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.

        Returns:
            array[float]: list of the probabilities
        """
        if self.shots is not None:
            return self.estimate_probability(wires=wires, shot_range=shot_range, bin_size=bin_size)

        wires = wires or self.wires
        wires = Wires(wires)

        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        if (
            device_wires
            and len(device_wires) > 1
            and (not np.all(np.array(device_wires)[:-1] <= np.array(device_wires)[1:]))
        ):
            raise RuntimeError(
                "Lightning does not currently support out-of-order indices for probabilities"
            )
        return self.probability_lightning(device_wires)

    def _get_diagonalizing_gates(self, circuit: qml.tape.QuantumTape) -> List[Operation]:
        # pylint: disable=no-member, protected-access
        def skip_diagonalizing(obs):
            return isinstance(obs, qml.Hamiltonian) or (
                isinstance(obs, qml.ops.Sum) and obs._pauli_rep is not None
            )

        meas_filtered = list(
            filter(lambda m: m.obs is None or not skip_diagonalizing(m.obs), circuit.measurements)
        )
        return super()._get_diagonalizing_gates(qml.tape.QuantumScript(measurements=meas_filtered))

    def _preprocess_state_vector(self, state, device_wires):
        """Initialize the internal state vector in a specified state.

        Args:
            state (array[complex]): normalized input state of length ``2**len(wires)``
                or broadcasted state of shape ``(batch_size, 2**len(wires))``
            device_wires (Wires): wires that get initialized in the state

        Returns:
            array[complex]: normalized input state of length ``2**len(wires)``
                or broadcasted state of shape ``(batch_size, 2**len(wires))``
            array[int]: indices for which the state is changed to input state vector elements
        """

        # translate to wire labels used by device
        device_wires = self.map_wires(device_wires)

        # special case for integral types
        if state.dtype.kind == "i":
            state = qml.numpy.array(state, dtype=self.C_DTYPE)
        state = self._asarray(state, dtype=self.C_DTYPE)

        if len(device_wires) == self.num_wires and Wires(sorted(device_wires)) == device_wires:
            return None, state

        # generate basis states on subset of qubits via the cartesian product
        basis_states = np.array(list(product([0, 1], repeat=len(device_wires))))

        # get basis states to alter on full set of qubits
        unravelled_indices = np.zeros((2 ** len(device_wires), self.num_wires), dtype=int)
        unravelled_indices[:, device_wires] = basis_states

        # get indices for which the state is changed to input state vector elements
        ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)
        return ravelled_indices, state

    def _get_basis_state_index(self, state, wires):
        """Returns the basis state index of a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s and 1s
            wires (Wires): wires that the provided computational state should be initialized on

        Returns:
            int: basis state index
        """
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        # length of basis state parameter
        n_basis_state = len(state)

        if not set(state.tolist()).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

        if n_basis_state != len(device_wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        # get computational basis state number
        basis_states = 2 ** (self.num_wires - 1 - np.array(device_wires))
        basis_states = qml.math.convert_like(basis_states, state)
        return int(qml.math.dot(state, basis_states))

    # pylint: disable=too-many-function-args, assignment-from-no-return, too-many-arguments
    def _process_jacobian_tape(
        self, tape, starting_state, use_device_state, use_mpi: bool = False, split_obs: bool = False
    ):
        state_vector = self._init_process_jacobian_tape(tape, starting_state, use_device_state)

        obs_serialized, obs_idx_offsets = QuantumScriptSerializer(
            self.short_name, self.use_csingle, use_mpi, split_obs
        ).serialize_observables(tape, self.wire_map)

        ops_serialized, use_sp = QuantumScriptSerializer(
            self.short_name, self.use_csingle, use_mpi, split_obs
        ).serialize_ops(tape, self.wire_map)

        ops_serialized = self.create_ops_list(*ops_serialized)

        # We need to filter out indices in trainable_params which do not
        # correspond to operators.
        trainable_params = sorted(tape.trainable_params)
        if len(trainable_params) == 0:
            return None

        tp_shift = []
        record_tp_rows = []
        all_params = 0

        for op_idx, trainable_param in enumerate(trainable_params):
            # get op_idx-th operator among differentiable operators
            operation, _, _ = tape.get_operation(op_idx)
            if isinstance(operation, Operation) and not isinstance(
                operation, (BasisState, StatePrep)
            ):
                # We now just ignore non-op or state preps
                tp_shift.append(trainable_param)
                record_tp_rows.append(all_params)
            all_params += 1

        if use_sp:
            # When the first element of the tape is state preparation. Still, I am not sure
            # whether there must be only one state preparation...
            tp_shift = [i - 1 for i in tp_shift]

        return {
            "state_vector": state_vector,
            "obs_serialized": obs_serialized,
            "ops_serialized": ops_serialized,
            "tp_shift": tp_shift,
            "record_tp_rows": record_tp_rows,
            "all_params": all_params,
            "obs_idx_offsets": obs_idx_offsets,
        }

    # pylint: disable=unnecessary-pass
    @staticmethod
    def _check_adjdiff_supported_measurements(measurements: List[MeasurementProcess]):
        """Check whether given list of measurement is supported by adjoint_differentiation.

        Args:
            measurements (List[MeasurementProcess]): a list of measurement processes to check.

        Returns:
            Expectation or State: a common return type of measurements.
        """
        pass

    @staticmethod
    def _adjoint_jacobian_processing(jac):
        """
        Post-process the Jacobian matrix returned by ``adjoint_jacobian`` for
        the new return type system.
        """
        jac = np.squeeze(jac)

        if jac.ndim == 0:
            return np.array(jac)

        if jac.ndim == 1:
            return tuple(np.array(j) for j in jac)

        # must be 2-dimensional
        return tuple(tuple(np.array(j_) for j_ in j) for j in jac)

    # pylint: disable=too-many-arguments
    def batch_vjp(
        self, tapes, grad_vecs, reduction="append", starting_state=None, use_device_state=False
    ):
        """Generate the processing function required to compute the vector-Jacobian products
        of a batch of tapes.

        Args:
            tapes (Sequence[.QuantumTape]): sequence of quantum tapes to differentiate
            grad_vecs (Sequence[tensor_like]): Sequence of gradient-output vectors ``grad_vec``.
                Must be the same length as ``tapes``. Each ``grad_vec`` tensor should have
                shape matching the output shape of the corresponding tape.
            reduction (str): Determines how the vector-Jacobian products are returned.
                If ``append``, then the output of the function will be of the form
                ``List[tensor_like]``, with each element corresponding to the VJP of each
                input tape. If ``extend``, then the output VJPs will be concatenated.
            starting_state (tensor_like): post-forward pass state to start execution with.
                It should be complex-valued. Takes precedence over ``use_device_state``.
            use_device_state (bool): use current device state to initialize. A forward pass of
                the same circuit should be the last thing the device has executed.
                If a ``starting_state`` is provided, that takes precedence.

        Returns:
            The processing function required to compute the vector-Jacobian products
            of a batch of tapes.
        """
        fns = []

        # Loop through the tapes and grad_vecs vector
        for tape, grad_vec in zip(tapes, grad_vecs):
            fun = self.vjp(
                tape.measurements,
                grad_vec,
                starting_state=starting_state,
                use_device_state=use_device_state,
            )
            fns.append(fun)

        def processing_fns(tapes):
            vjps = []
            for tape, fun in zip(tapes, fns):
                vjp = fun(tape)

                # make sure vjp is iterable if using extend reduction
                if (
                    not isinstance(vjp, tuple)
                    and getattr(reduction, "__name__", reduction) == "extend"
                ):
                    vjp = (vjp,)

                if isinstance(reduction, str):
                    getattr(vjps, reduction)(vjp)
                elif callable(reduction):
                    reduction(vjps, vjp)

            return vjps

        return processing_fns


class LightningBaseFallBack(DefaultQubitLegacy):  # pragma: no cover
    # pylint: disable=missing-class-docstring, too-few-public-methods
    pennylane_requires = ">=0.34"
    version = __version__
    author = "Xanadu Inc."
    _CPP_BINARY_AVAILABLE = False

    def __init__(self, wires, *, c_dtype=np.complex128, **kwargs):
        if c_dtype is np.complex64:
            r_dtype = np.float32
        elif c_dtype is np.complex128:
            r_dtype = np.float64
        else:
            raise TypeError(f"Unsupported complex type: {c_dtype}")
        super().__init__(wires, r_dtype=r_dtype, c_dtype=c_dtype, **kwargs)

    @property
    def state_vector(self):
        """Returns a handle to the statevector."""
        return self._state
