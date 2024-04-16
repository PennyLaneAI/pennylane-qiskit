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
This module contains the :class:`~.LightningQubit` class, a PennyLane simulator device that
interfaces with C++ for fast linear algebra calculations.
"""

from pathlib import Path
from typing import List, Sequence
from warnings import warn

import numpy as np

from pennylane_lightning.core.lightning_base import (
    LightningBase,
    LightningBaseFallBack,
    _chunk_iterable,
)

try:
    # pylint: disable=import-error, no-name-in-module
    from pennylane_lightning.lightning_qubit_ops import (
        MeasurementsC64,
        MeasurementsC128,
        StateVectorC64,
        StateVectorC128,
        allocate_aligned_array,
        backend_info,
        best_alignment,
        get_alignment,
    )

    LQ_CPP_BINARY_AVAILABLE = True
except ImportError:
    LQ_CPP_BINARY_AVAILABLE = False

if LQ_CPP_BINARY_AVAILABLE:
    from os import getenv

    import pennylane as qml
    from pennylane import (
        BasisState,
        DeviceError,
        Projector,
        QuantumFunctionError,
        Rot,
        StatePrep,
        math,
    )
    from pennylane.measurements import Expectation, MeasurementProcess, State
    from pennylane.operation import Tensor
    from pennylane.ops.op_math import Adjoint
    from pennylane.wires import Wires

    # pylint: disable=import-error, no-name-in-module, ungrouped-imports
    from pennylane_lightning.core._serialize import QuantumScriptSerializer
    from pennylane_lightning.core._version import __version__
    from pennylane_lightning.lightning_qubit_ops.algorithms import (
        AdjointJacobianC64,
        AdjointJacobianC128,
        VectorJacobianProductC64,
        VectorJacobianProductC128,
        create_ops_listC64,
        create_ops_listC128,
    )

    def _state_dtype(dtype):
        if dtype not in [np.complex128, np.complex64]:  # pragma: no cover
            raise ValueError(f"Data type is not supported for state-vector computation: {dtype}")
        return StateVectorC128 if dtype == np.complex128 else StateVectorC64

    allowed_operations = {
        "Identity",
        "BasisState",
        "QubitStateVector",
        "StatePrep",
        "QubitUnitary",
        "ControlledQubitUnitary",
        "MultiControlledX",
        "DiagonalQubitUnitary",
        "PauliX",
        "PauliY",
        "PauliZ",
        "MultiRZ",
        "GlobalPhase",
        "Hadamard",
        "S",
        "Adjoint(S)",
        "T",
        "Adjoint(T)",
        "SX",
        "Adjoint(SX)",
        "CNOT",
        "SWAP",
        "ISWAP",
        "PSWAP",
        "Adjoint(ISWAP)",
        "SISWAP",
        "Adjoint(SISWAP)",
        "SQISW",
        "CSWAP",
        "Toffoli",
        "CY",
        "CZ",
        "PhaseShift",
        "ControlledPhaseShift",
        "CPhase",
        "RX",
        "RY",
        "RZ",
        "Rot",
        "CRX",
        "CRY",
        "CRZ",
        "C(PauliX)",
        "C(PauliY)",
        "C(PauliZ)",
        "C(Hadamard)",
        "C(S)",
        "C(T)",
        "C(PhaseShift)",
        "C(RX)",
        "C(RY)",
        "C(RZ)",
        "C(Rot)",
        "C(SWAP)",
        "C(IsingXX)",
        "C(IsingXY)",
        "C(IsingYY)",
        "C(IsingZZ)",
        "C(SingleExcitation)",
        "C(SingleExcitationMinus)",
        "C(SingleExcitationPlus)",
        "C(DoubleExcitation)",
        "C(DoubleExcitationMinus)",
        "C(DoubleExcitationPlus)",
        "C(MultiRZ)",
        "C(GlobalPhase)",
        "CRot",
        "IsingXX",
        "IsingYY",
        "IsingZZ",
        "IsingXY",
        "SingleExcitation",
        "SingleExcitationPlus",
        "SingleExcitationMinus",
        "DoubleExcitation",
        "DoubleExcitationPlus",
        "DoubleExcitationMinus",
        "QubitCarry",
        "QubitSum",
        "OrbitalRotation",
        "QFT",
        "ECR",
        "BlockEncode",
    }

    allowed_observables = {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "Hermitian",
        "Identity",
        "Projector",
        "SparseHamiltonian",
        "Hamiltonian",
        "Sum",
        "SProd",
        "Prod",
        "Exp",
    }

    class LightningQubit(LightningBase):
        """PennyLane Lightning Qubit device.

        A device that interfaces with C++ to perform fast linear algebra calculations.

        Use of this device requires pre-built binaries or compilation from source. Check out the
        :doc:`/lightning_qubit/installation` guide for more details.

        Args:
            wires (int): the number of wires to initialize the device with
            c_dtype: Datatypes for statevector representation. Must be one of
                ``np.complex64`` or ``np.complex128``.
            shots (int): How many times the circuit should be evaluated (or sampled) to estimate
                the expectation values. Defaults to ``None`` if not specified. Setting
                to ``None`` results in computing statistics like expectation values and
                variances analytically.
            mcmc (bool): Determine whether to use the approximate Markov Chain Monte Carlo
                sampling method when generating samples.
            kernel_name (str): name of transition kernel. The current version supports
                two kernels: ``"Local"`` and ``"NonZeroRandom"``.
                The local kernel conducts a bit-flip local transition between states.
                The local kernel generates a random qubit site and then generates a random
                number to determine the new bit at that qubit site. The ``"NonZeroRandom"`` kernel
                randomly transits between states that have nonzero probability.
            num_burnin (int): number of steps that will be dropped. Increasing this value will
                result in a closer approximation but increased runtime.
            batch_obs (bool): Determine whether we process observables in parallel when
                computing the jacobian. This value is only relevant when the lightning
                qubit is built with OpenMP.
        """

        name = "Lightning Qubit PennyLane plugin"
        short_name = "lightning.qubit"
        operations = allowed_operations
        observables = allowed_observables
        _backend_info = backend_info
        config = Path(__file__).parent / "lightning_qubit.toml"

        def __init__(  # pylint: disable=too-many-arguments
            self,
            wires,
            *,
            c_dtype=np.complex128,
            shots=None,
            mcmc=False,
            kernel_name="Local",
            num_burnin=100,
            batch_obs=False,
        ):
            super().__init__(wires, shots=shots, c_dtype=c_dtype)

            # Create the initial state. Internally, we store the
            # state as an array of dimension [2]*wires.
            self._qubit_state = _state_dtype(c_dtype)(self.num_wires)
            self._batch_obs = batch_obs
            self._mcmc = mcmc
            if self._mcmc:
                if kernel_name not in [
                    "Local",
                    "NonZeroRandom",
                ]:
                    raise NotImplementedError(
                        f"The {kernel_name} is not supported and currently "
                        "only 'Local' and 'NonZeroRandom' kernels are supported."
                    )
                shots = shots if isinstance(shots, Sequence) else [shots]
                if any(num_burnin >= s for s in shots):
                    raise ValueError("Shots should be greater than num_burnin.")
                self._kernel_name = kernel_name
                self._num_burnin = num_burnin

        @staticmethod
        def _asarray(arr, dtype=None):
            arr = np.asarray(arr)  # arr is not copied

            if arr.dtype.kind not in ["f", "c"]:
                return arr

            if not dtype:
                dtype = arr.dtype

            # We allocate a new aligned memory and copy data to there if alignment or dtype
            # mismatches
            # Note that get_alignment does not necessarily return CPUMemoryModel(Unaligned)
            # numpy allocated memory as the memory location happens to be aligned.
            if int(get_alignment(arr)) < int(best_alignment()) or arr.dtype != dtype:
                new_arr = allocate_aligned_array(arr.size, np.dtype(dtype), False).reshape(
                    arr.shape
                )
                if len(arr.shape):
                    new_arr[:] = arr
                else:
                    np.copyto(new_arr, arr)
                arr = new_arr
            return arr

        def _create_basis_state(self, index):
            """Return a computational basis state over all wires.
            Args:
                index (int): integer representing the computational basis state.
            """
            self._qubit_state.setBasisState(index)

        def reset(self):
            """Reset the device"""
            super().reset()

            # init the state vector to |00..0>
            self._qubit_state.resetStateVector()

        @property
        def create_ops_list(self):
            """Returns create_ops_list function matching ``use_csingle`` precision."""
            return create_ops_listC64 if self.use_csingle else create_ops_listC128

        @property
        def measurements(self):
            """Returns a Measurements object matching ``use_csingle`` precision."""
            return (
                MeasurementsC64(self.state_vector)
                if self.use_csingle
                else MeasurementsC128(self.state_vector)
            )

        @property
        def state(self):
            """Copy the state vector data to a numpy array.

            **Example**

            >>> dev = qml.device('lightning.kokkos', wires=1)
            >>> dev.apply([qml.PauliX(wires=[0])])
            >>> print(dev.state)
            [0.+0.j 1.+0.j]
            """
            state = np.zeros(2**self.num_wires, dtype=self.C_DTYPE)
            state = self._asarray(state, dtype=self.C_DTYPE)
            self._qubit_state.getState(state)
            return state

        @property
        def state_vector(self):
            """Returns a handle to the statevector."""
            return self._qubit_state

        def _apply_state_vector(self, state, device_wires: Wires):
            """Initialize the internal state vector in a specified state.
            Args:
                state (array[complex]): normalized input state of length ``2**len(wires)``
                    or broadcasted state of shape ``(batch_size, 2**len(wires))``
                device_wires (Wires): wires that get initialized in the state
            """

            if isinstance(state, self._qubit_state.__class__):
                state_data = allocate_aligned_array(state.size, np.dtype(self.C_DTYPE), True)
                state.getState(state_data)
                state = state_data

            ravelled_indices, state = self._preprocess_state_vector(state, device_wires)

            # translate to wire labels used by device
            device_wires = self.map_wires(device_wires)
            output_shape = [2] * self.num_wires

            if len(device_wires) == self.num_wires and Wires(sorted(device_wires)) == device_wires:
                # Initialize the entire device state with the input state
                state = self._reshape(state, output_shape).ravel(order="C")
                self._qubit_state.UpdateData(state)
                return

            self._qubit_state.setStateVector(ravelled_indices, state)  # this operation on device

        def _apply_basis_state(self, state, wires):
            """Initialize the state vector in a specified computational basis state.

            Args:
                state (array[int]): computational basis state of shape ``(wires,)``
                    consisting of 0s and 1s.
                wires (Wires): wires that the provided computational state should be
                    initialized on

            Note: This function does not support broadcasted inputs yet.
            """
            num = self._get_basis_state_index(state, wires)
            self._create_basis_state(num)

        def _apply_lightning_controlled(self, operation):
            """Apply an arbitrary controlled operation to the state tensor.

            Args:
                operation (~pennylane.operation.Operation): operation to apply

            Returns:
                array[complex]: the output state tensor
            """
            state = self.state_vector

            basename = "PauliX" if operation.name == "MultiControlledX" else operation.base.name
            if basename == "Identity":
                return
            method = getattr(state, f"{basename}", None)
            control_wires = self.wires.indices(operation.control_wires)
            control_values = (
                [bool(int(i)) for i in operation.hyperparameters["control_values"]]
                if operation.name == "MultiControlledX"
                else operation.control_values
            )
            if operation.name == "MultiControlledX":
                target_wires = list(set(self.wires.indices(operation.wires)) - set(control_wires))
            else:
                target_wires = self.wires.indices(operation.target_wires)
            if method is not None:  # apply n-controlled specialized gate
                inv = False
                param = operation.parameters
                method(control_wires, control_values, target_wires, inv, param)
            else:  # apply gate as an n-controlled matrix
                method = getattr(state, "applyControlledMatrix")
                target_wires = self.wires.indices(operation.target_wires)
                try:
                    method(
                        qml.matrix(operation.base),
                        control_wires,
                        control_values,
                        target_wires,
                        False,
                    )
                except AttributeError:  # pragma: no cover
                    # To support older versions of PL
                    method(
                        operation.base.matrix, control_wires, control_values, target_wires, False
                    )

        def apply_lightning(self, operations):
            """Apply a list of operations to the state tensor.

            Args:
                operations (list[~pennylane.operation.Operation]): operations to apply

            Returns:
                array[complex]: the output state tensor
            """
            state = self.state_vector

            # Skip over identity operations instead of performing
            # matrix multiplication with it.
            for operation in operations:
                if isinstance(operation, Adjoint):
                    name = operation.base.name
                    invert_param = True
                else:
                    name = operation.name
                    invert_param = False
                if name == "Identity":
                    continue
                method = getattr(state, name, None)
                wires = self.wires.indices(operation.wires)

                if method is not None:  # apply specialized gate
                    param = operation.parameters
                    method(wires, invert_param, param)
                elif (
                    name[0:2] == "C("
                    or name == "ControlledQubitUnitary"
                    or name == "MultiControlledX"
                ):  # apply n-controlled gate
                    self._apply_lightning_controlled(operation)
                else:  # apply gate as a matrix
                    # Inverse can be set to False since qml.matrix(operation) is already in
                    # inverted form
                    method = getattr(state, "applyMatrix")
                    try:
                        method(qml.matrix(operation), wires, False)
                    except AttributeError:  # pragma: no cover
                        # To support older versions of PL
                        method(operation.matrix, wires, False)

        # pylint: disable=unused-argument
        def apply(self, operations, rotations=None, **kwargs):
            """Applies operations to the state vector."""
            # State preparation is currently done in Python
            if operations:  # make sure operations[0] exists
                if isinstance(operations[0], StatePrep):
                    self._apply_state_vector(
                        operations[0].parameters[0].copy(), operations[0].wires
                    )
                    operations = operations[1:]
                elif isinstance(operations[0], BasisState):
                    self._apply_basis_state(operations[0].parameters[0], operations[0].wires)
                    operations = operations[1:]

            for operation in operations:
                if isinstance(operation, (StatePrep, BasisState)):
                    raise DeviceError(
                        f"Operation {operation.name} cannot be used after other "
                        f"Operations have already been applied on a {self.short_name} device."
                    )

            self.apply_lightning(operations)

        # pylint: disable=protected-access
        def expval(self, observable, shot_range=None, bin_size=None):
            """Expectation value of the supplied observable.

            Args:
                observable: A PennyLane observable.
                shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                    to use. If not specified, all samples are used.
                bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                    returns the measurement statistic separately over each bin. If not
                    provided, the entire shot range is treated as a single bin.

            Returns:
                Expectation value of the observable
            """
            if observable.name in [
                "Projector",
            ]:
                diagonalizing_gates = observable.diagonalizing_gates()
                if self.shots is None and diagonalizing_gates:
                    self.apply(diagonalizing_gates)
                results = super().expval(observable, shot_range=shot_range, bin_size=bin_size)
                if self.shots is None and diagonalizing_gates:
                    self.apply([qml.adjoint(g, lazy=False) for g in reversed(diagonalizing_gates)])
                return results

            if self.shots is not None:
                # estimate the expectation value
                # LightningQubit doesn't support sampling yet
                samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
                return np.squeeze(np.mean(samples, axis=0))

            measurements = (
                MeasurementsC64(self.state_vector)
                if self.use_csingle
                else MeasurementsC128(self.state_vector)
            )
            if observable.name == "SparseHamiltonian":
                csr_hamiltonian = observable.sparse_matrix(wire_order=self.wires).tocsr(copy=False)
                return measurements.expval(
                    csr_hamiltonian.indptr,
                    csr_hamiltonian.indices,
                    csr_hamiltonian.data,
                )

            if (
                observable.name in ["Hamiltonian", "Hermitian"]
                or (observable.arithmetic_depth > 0)
                or isinstance(observable.name, List)
            ):
                ob_serialized = QuantumScriptSerializer(self.short_name, self.use_csingle)._ob(
                    observable, self.wire_map
                )
                return measurements.expval(ob_serialized)

            # translate to wire labels used by device
            observable_wires = self.map_wires(observable.wires)

            return measurements.expval(observable.name, observable_wires)

        def var(self, observable, shot_range=None, bin_size=None):
            """Variance of the supplied observable.

            Args:
                observable: A PennyLane observable.
                shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                    to use. If not specified, all samples are used.
                bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                    returns the measurement statistic separately over each bin. If not
                    provided, the entire shot range is treated as a single bin.

            Returns:
                Variance of the observable
            """
            if observable.name in [
                "Projector",
            ]:
                diagonalizing_gates = observable.diagonalizing_gates()
                if self.shots is None and diagonalizing_gates:
                    self.apply(diagonalizing_gates)
                results = super().var(observable, shot_range=shot_range, bin_size=bin_size)
                if self.shots is None and diagonalizing_gates:
                    self.apply([qml.adjoint(g, lazy=False) for g in reversed(diagonalizing_gates)])
                return results

            if self.shots is not None:
                # estimate the var
                # LightningQubit doesn't support sampling yet
                samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
                return np.squeeze(np.var(samples, axis=0))

            measurements = (
                MeasurementsC64(self.state_vector)
                if self.use_csingle
                else MeasurementsC128(self.state_vector)
            )

            if observable.name == "SparseHamiltonian":
                csr_hamiltonian = observable.sparse_matrix(wire_order=self.wires).tocsr(copy=False)
                return measurements.var(
                    csr_hamiltonian.indptr,
                    csr_hamiltonian.indices,
                    csr_hamiltonian.data,
                )

            if (
                observable.name in ["Hamiltonian", "Hermitian"]
                or (observable.arithmetic_depth > 0)
                or isinstance(observable.name, List)
            ):
                ob_serialized = QuantumScriptSerializer(self.short_name, self.use_csingle)._ob(
                    observable, self.wire_map
                )
                return measurements.var(ob_serialized)

            # translate to wire labels used by device
            observable_wires = self.map_wires(observable.wires)

            return measurements.var(observable.name, observable_wires)

        def generate_samples(self):
            """Generate samples

            Returns:
                array[int]: array of samples in binary representation with shape
                    ``(dev.shots, dev.num_wires)``
            """
            measurements = (
                MeasurementsC64(self.state_vector)
                if self.use_csingle
                else MeasurementsC128(self.state_vector)
            )
            if self._mcmc:
                return measurements.generate_mcmc_samples(
                    len(self.wires), self._kernel_name, self._num_burnin, self.shots
                ).astype(int, copy=False)
            return measurements.generate_samples(len(self.wires), self.shots).astype(
                int, copy=False
            )

        def probability_lightning(self, wires):
            """Return the probability of each computational basis state.

            Args:
                wires (Iterable[Number, str], Number, str, Wires): wires to return
                    marginal probabilities for. Wires not provided are traced out of the system.

            Returns:
                array[float]: list of the probabilities
            """
            return (
                MeasurementsC64(self.state_vector)
                if self.use_csingle
                else MeasurementsC128(self.state_vector)
            ).probs(wires)

        # pylint: disable=attribute-defined-outside-init
        def sample(self, observable, shot_range=None, bin_size=None, counts=False):
            """Return samples of an observable."""
            diagonalizing_gates = observable.diagonalizing_gates()
            if diagonalizing_gates:
                self.apply(diagonalizing_gates)
            if not isinstance(observable, qml.PauliZ):
                self._samples = self.generate_samples()
            results = super().sample(
                observable, shot_range=shot_range, bin_size=bin_size, counts=counts
            )
            if diagonalizing_gates:
                self.apply([qml.adjoint(g, lazy=False) for g in reversed(diagonalizing_gates)])
            return results

        @staticmethod
        def _check_adjdiff_supported_measurements(
            measurements: List[MeasurementProcess],
        ):
            """Check whether given list of measurement is supported by adjoint_differentiation.

            Args:
                measurements (List[MeasurementProcess]): a list of measurement processes to check.

            Returns:
                Expectation or State: a common return type of measurements.
            """
            if not measurements:
                return None

            if len(measurements) == 1 and measurements[0].return_type is State:
                return State

            # Now the return_type of measurement processes must be expectation
            if any(measurement.return_type is not Expectation for measurement in measurements):
                raise QuantumFunctionError(
                    "Adjoint differentiation method does not support expectation return type "
                    "mixed with other return types"
                )

            for measurement in measurements:
                if isinstance(measurement.obs, Tensor):
                    if any(isinstance(obs, Projector) for obs in measurement.obs.non_identity_obs):
                        raise QuantumFunctionError(
                            "Adjoint differentiation method does "
                            "not support the Projector observable"
                        )
                elif isinstance(measurement.obs, Projector):
                    raise QuantumFunctionError(
                        "Adjoint differentiation method does not support the Projector observable"
                    )
            return Expectation

        @staticmethod
        def _check_adjdiff_supported_operations(operations):
            """Check Lightning adjoint differentiation method support for a tape.

            Raise ``QuantumFunctionError`` if ``tape`` contains not supported measurements,
            observables, or operations by the Lightning adjoint differentiation method.

            Args:
                tape (.QuantumTape): quantum tape to differentiate.
            """
            for operation in operations:
                if operation.num_params > 1 and not isinstance(operation, Rot):
                    raise QuantumFunctionError(
                        f"The {operation.name} operation is not supported using "
                        'the "adjoint" differentiation method'
                    )

        def _init_process_jacobian_tape(self, tape, starting_state, use_device_state):
            """Generate an initial state vector for ``_process_jacobian_tape``."""
            if starting_state is not None:
                if starting_state.size != 2 ** len(self.wires):
                    raise QuantumFunctionError(
                        "The number of qubits of starting_state must be the same as "
                        "that of the device."
                    )
                self._apply_state_vector(starting_state, self.wires)
            elif not use_device_state:
                self.reset()
                self.apply(tape.operations)
            return self.state_vector

        def adjoint_jacobian(self, tape, starting_state=None, use_device_state=False):
            """Computes and returns the Jacobian with the adjoint method."""
            if self.shots is not None:
                warn(
                    "Requested adjoint differentiation to be computed with finite shots. "
                    "The derivative is always exact when using the adjoint "
                    "differentiation method.",
                    UserWarning,
                )

            tape_return_type = self._check_adjdiff_supported_measurements(tape.measurements)

            if not tape_return_type:  # the tape does not have measurements
                return np.array([], dtype=self.state.dtype)

            if tape_return_type is State:
                raise QuantumFunctionError(
                    "This method does not support statevector return type. "
                    "Use vjp method instead for this purpose."
                )

            self._check_adjdiff_supported_operations(tape.operations)

            processed_data = self._process_jacobian_tape(tape, starting_state, use_device_state)

            if not processed_data:  # training_params is empty
                return np.array([], dtype=self.state.dtype)

            trainable_params = processed_data["tp_shift"]

            # If requested batching over observables, chunk into OMP_NUM_THREADS sized chunks.
            # This will allow use of Lightning with adjoint for large-qubit numbers AND large
            # numbers of observables, enabling choice between compute time and memory use.
            requested_threads = int(getenv("OMP_NUM_THREADS", "1"))

            adjoint_jacobian = AdjointJacobianC64() if self.use_csingle else AdjointJacobianC128()

            if self._batch_obs and requested_threads > 1:
                obs_partitions = _chunk_iterable(
                    processed_data["obs_serialized"], requested_threads
                )
                jac = []
                for obs_chunk in obs_partitions:
                    jac_local = adjoint_jacobian(
                        processed_data["state_vector"],
                        obs_chunk,
                        processed_data["ops_serialized"],
                        trainable_params,
                    )
                    jac.extend(jac_local)
            else:
                jac = adjoint_jacobian(
                    processed_data["state_vector"],
                    processed_data["obs_serialized"],
                    processed_data["ops_serialized"],
                    trainable_params,
                )
            jac = np.array(jac)
            jac = jac.reshape(-1, len(trainable_params))
            jac_r = np.zeros((jac.shape[0], processed_data["all_params"]))
            jac_r[:, processed_data["record_tp_rows"]] = jac
            if hasattr(qml, "active_return"):  # pragma: no cover
                return self._adjoint_jacobian_processing(jac_r) if qml.active_return() else jac_r
            return self._adjoint_jacobian_processing(jac_r)

        # pylint: disable=line-too-long, inconsistent-return-statements
        def vjp(self, measurements, grad_vec, starting_state=None, use_device_state=False):
            """Generate the processing function required to compute the vector-Jacobian products
            of a tape.

            This function can be used with multiple expectation values or a quantum state.
            When a quantum state is given,

            .. code-block:: python

                vjp_f = dev.vjp([qml.state()], grad_vec)
                vjp = vjp_f(tape)

            computes :math:`w = (w_1,\\cdots,w_m)` where

            .. math::

                w_k = \\langle v| \\frac{\\partial}{\\partial \\theta_k} | \\psi_{\\pmb{\\theta}} \\rangle.

            Here, :math:`m` is the total number of trainable parameters, :math:`\\pmb{\\theta}`
            is the vector of trainable parameters and :math:`\\psi_{\\pmb{\\theta}}`
            is the output quantum state.

            Args:
                measurements (list): List of measurement processes for vector-Jacobian product.
                    Now it must be expectation values or a quantum state.
                grad_vec (tensor_like): Gradient-output vector. Must have shape matching the output
                    shape of the corresponding tape, i.e. number of measurements if
                    the return type is expectation or :math:`2^N` if the return type is statevector
                starting_state (tensor_like): post-forward pass state to start execution with.
                    It should be complex-valued. Takes precedence over ``use_device_state``.
                use_device_state (bool): use current device state to initialize.
                    A forward pass of the same circuit should be the last thing
                    the device has executed. If a ``starting_state`` is provided,
                    that takes precedence.

            Returns:
                The processing function required to compute the vector-Jacobian products of a tape.
            """
            if self.shots is not None:
                warn(
                    "Requested adjoint differentiation to be computed with finite shots. "
                    "The derivative is always exact when using the adjoint differentiation "
                    "method.",
                    UserWarning,
                )

            tape_return_type = self._check_adjdiff_supported_measurements(measurements)

            if math.allclose(grad_vec, 0) or tape_return_type is None:
                return lambda tape: math.convert_like(
                    np.zeros(len(tape.trainable_params)), grad_vec
                )

            if tape_return_type is Expectation:
                if len(grad_vec) != len(measurements):
                    raise ValueError(
                        "Number of observables in the tape must be the same as the "
                        "length of grad_vec in the vjp method"
                    )

                if np.iscomplexobj(grad_vec):
                    raise ValueError(
                        "The vjp method only works with a real-valued grad_vec when the "
                        "tape is returning an expectation value"
                    )

                ham = qml.Hamiltonian(grad_vec, [m.obs for m in measurements])

                def processing_fn_expval(tape):
                    nonlocal ham
                    num_params = len(tape.trainable_params)

                    if num_params == 0:
                        return np.array([], dtype=self.state.dtype)

                    new_tape = tape.copy()
                    new_tape._measurements = [qml.expval(ham)]

                    return self.adjoint_jacobian(new_tape, starting_state, use_device_state)

                return processing_fn_expval

            if tape_return_type is State:
                if len(grad_vec) != 2 ** len(self.wires):
                    raise ValueError(
                        "Size of the provided vector grad_vec must be the same as "
                        "the size of the statevector"
                    )
                if np.isrealobj(grad_vec):
                    warn(
                        "The vjp method only works with complex-valued grad_vec when "
                        "the tape is returning a statevector. Upcasting grad_vec."
                    )

                grad_vec = grad_vec.astype(self.C_DTYPE)

                def processing_fn_state(tape):
                    nonlocal grad_vec
                    processed_data = self._process_jacobian_tape(
                        tape, starting_state, use_device_state
                    )
                    calculate_vjp = (
                        VectorJacobianProductC64()
                        if self.use_csingle
                        else VectorJacobianProductC128()
                    )

                    return calculate_vjp(
                        processed_data["state_vector"],
                        processed_data["ops_serialized"],
                        grad_vec,
                        processed_data["tp_shift"],
                    )

                return processing_fn_state

else:

    class LightningQubit(LightningBaseFallBack):  # pragma: no cover
        # pylint: disable=missing-class-docstring, too-few-public-methods
        name = "Lightning qubit PennyLane plugin [No binaries found - Fallback: default.qubit]"
        short_name = "lightning.qubit"

        def __init__(self, wires, *, c_dtype=np.complex128, **kwargs):
            warn(
                "Pre-compiled binaries for lightning.qubit are not available. Falling back to "
                "using the Python-based default.qubit implementation. To manually compile from "
                "source, follow the instructions at "
                "https://pennylane-lightning.readthedocs.io/en/latest/installation.html.",
                UserWarning,
            )
            super().__init__(wires, c_dtype=c_dtype, **kwargs)
