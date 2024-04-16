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
Helper functions for serializing quantum tapes.
"""
from typing import List, Tuple
import numpy as np
from pennylane import (
    BasisState,
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    Identity,
    StatePrep,
    Rot,
    Hamiltonian,
    SparseHamiltonian,
    QubitUnitary,
)
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap

from pennylane import matrix, DeviceError

pauli_name_map = {
    "I": "Identity",
    "X": "PauliX",
    "Y": "PauliY",
    "Z": "PauliZ",
}


class QuantumScriptSerializer:
    """Serializer class for `pennylane.tape.QuantumScript` data.

    Args:
    device_name: device shortname.
    use_csingle (bool): whether to use np.complex64 instead of np.complex128

    """

    # pylint: disable=import-outside-toplevel, too-many-instance-attributes, c-extension-no-member
    def __init__(
        self, device_name, use_csingle: bool = False, use_mpi: bool = False, split_obs: bool = False
    ):
        self.use_csingle = use_csingle
        self.device_name = device_name
        self.split_obs = split_obs
        if device_name == "lightning.qubit":
            try:
                import pennylane_lightning.lightning_qubit_ops as lightning_ops
            except ImportError as exception:
                raise ImportError(
                    f"Pre-compiled binaries for {device_name} are not available."
                ) from exception
        elif device_name == "lightning.kokkos":
            try:
                import pennylane_lightning.lightning_kokkos_ops as lightning_ops
            except ImportError as exception:
                raise ImportError(
                    f"Pre-compiled binaries for {device_name} are not available."
                ) from exception
        elif device_name == "lightning.gpu":
            try:
                import pennylane_lightning.lightning_gpu_ops as lightning_ops
            except ImportError as exception:
                raise ImportError(
                    f"Pre-compiled binaries for {device_name} are not available."
                ) from exception
        else:
            raise DeviceError(f'The device name "{device_name}" is not a valid option.')
        self.statevector_c64 = lightning_ops.StateVectorC64
        self.statevector_c128 = lightning_ops.StateVectorC128
        self.named_obs_c64 = lightning_ops.observables.NamedObsC64
        self.named_obs_c128 = lightning_ops.observables.NamedObsC128
        self.hermitian_obs_c64 = lightning_ops.observables.HermitianObsC64
        self.hermitian_obs_c128 = lightning_ops.observables.HermitianObsC128
        self.tensor_prod_obs_c64 = lightning_ops.observables.TensorProdObsC64
        self.tensor_prod_obs_c128 = lightning_ops.observables.TensorProdObsC128
        self.hamiltonian_c64 = lightning_ops.observables.HamiltonianC64
        self.hamiltonian_c128 = lightning_ops.observables.HamiltonianC128
        self.sparse_hamiltonian_c64 = lightning_ops.observables.SparseHamiltonianC64
        self.sparse_hamiltonian_c128 = lightning_ops.observables.SparseHamiltonianC128

        self._use_mpi = use_mpi

        if self._use_mpi:
            self.statevector_mpi_c64 = lightning_ops.StateVectorMPIC64
            self.statevector_mpi_c128 = lightning_ops.StateVectorMPIC128
            self.named_obs_mpi_c64 = lightning_ops.observablesMPI.NamedObsMPIC64
            self.named_obs_mpi_c128 = lightning_ops.observablesMPI.NamedObsMPIC128
            self.hermitian_obs_mpi_c64 = lightning_ops.observablesMPI.HermitianObsMPIC64
            self.hermitian_obs_mpi_c128 = lightning_ops.observablesMPI.HermitianObsMPIC128
            self.tensor_prod_obs_mpi_c64 = lightning_ops.observablesMPI.TensorProdObsMPIC64
            self.tensor_prod_obs_mpi_c128 = lightning_ops.observablesMPI.TensorProdObsMPIC128
            self.hamiltonian_mpi_c64 = lightning_ops.observablesMPI.HamiltonianMPIC64
            self.hamiltonian_mpi_c128 = lightning_ops.observablesMPI.HamiltonianMPIC128
            self.sparse_hamiltonian_mpi_c64 = lightning_ops.observablesMPI.SparseHamiltonianMPIC64
            self.sparse_hamiltonian_mpi_c128 = lightning_ops.observablesMPI.SparseHamiltonianMPIC128

            self._mpi_manager = lightning_ops.MPIManager

    @property
    def ctype(self):
        """Complex type."""
        return np.complex64 if self.use_csingle else np.complex128

    @property
    def rtype(self):
        """Real type."""
        return np.float32 if self.use_csingle else np.float64

    @property
    def sv_type(self):
        """State vector matching ``use_csingle`` precision (and MPI if it is supported)."""
        if self._use_mpi:
            return self.statevector_mpi_c64 if self.use_csingle else self.statevector_mpi_c128
        return self.statevector_c64 if self.use_csingle else self.statevector_c128

    @property
    def named_obs(self):
        """Named observable matching ``use_csingle`` precision."""
        if self._use_mpi:
            return self.named_obs_mpi_c64 if self.use_csingle else self.named_obs_mpi_c128
        return self.named_obs_c64 if self.use_csingle else self.named_obs_c128

    @property
    def hermitian_obs(self):
        """Hermitian observable matching ``use_csingle`` precision."""
        if self._use_mpi:
            return self.hermitian_obs_mpi_c64 if self.use_csingle else self.hermitian_obs_mpi_c128
        return self.hermitian_obs_c64 if self.use_csingle else self.hermitian_obs_c128

    @property
    def tensor_obs(self):
        """Tensor product observable matching ``use_csingle`` precision."""
        if self._use_mpi:
            return (
                self.tensor_prod_obs_mpi_c64 if self.use_csingle else self.tensor_prod_obs_mpi_c128
            )
        return self.tensor_prod_obs_c64 if self.use_csingle else self.tensor_prod_obs_c128

    @property
    def hamiltonian_obs(self):
        """Hamiltonian observable matching ``use_csingle`` precision."""
        if self._use_mpi:
            return self.hamiltonian_mpi_c64 if self.use_csingle else self.hamiltonian_mpi_c128
        return self.hamiltonian_c64 if self.use_csingle else self.hamiltonian_c128

    @property
    def sparse_hamiltonian_obs(self):
        """SparseHamiltonian observable matching ``use_csingle`` precision."""
        if self._use_mpi:
            return (
                self.sparse_hamiltonian_mpi_c64
                if self.use_csingle
                else self.sparse_hamiltonian_mpi_c128
            )
        return self.sparse_hamiltonian_c64 if self.use_csingle else self.sparse_hamiltonian_c128

    def _named_obs(self, observable, wires_map: dict):
        """Serializes a Named observable"""
        wires = [wires_map[w] for w in observable.wires]
        if observable.name == "Identity":
            wires = wires[:1]
        return self.named_obs(observable.name, wires)

    def _hermitian_ob(self, observable, wires_map: dict):
        """Serializes a Hermitian observable"""
        assert not isinstance(observable, Tensor)

        wires = [wires_map[w] for w in observable.wires]
        return self.hermitian_obs(matrix(observable).ravel().astype(self.ctype), wires)

    def _tensor_ob(self, observable, wires_map: dict):
        """Serialize a tensor observable"""
        assert isinstance(observable, Tensor)
        return self.tensor_obs([self._ob(obs, wires_map) for obs in observable.obs])

    def _hamiltonian(self, observable, wires_map: dict):
        coeffs = np.array(unwrap(observable.coeffs)).astype(self.rtype)
        terms = [self._ob(t, wires_map) for t in observable.ops]

        if self.split_obs:
            return [self.hamiltonian_obs([c], [t]) for (c, t) in zip(coeffs, terms)]

        return self.hamiltonian_obs(coeffs, terms)

    def _sparse_hamiltonian(self, observable, wires_map: dict):
        """Serialize an observable (Sparse Hamiltonian)

        Args:
            observable (Observable): the input observable (Sparse Hamiltonian)
            wire_map (dict): a dictionary mapping input wires to the device's backend wires

        Returns:
            sparse_hamiltonian_obs (SparseHamiltonianC64 or SparseHamiltonianC128): A Sparse Hamiltonian observable object compatible with the C++ backend
        """

        if self._use_mpi:
            Hmat = Hamiltonian([1.0], [Identity(0)]).sparse_matrix()
            H_sparse = SparseHamiltonian(Hmat, wires=range(1))
            spm = H_sparse.sparse_matrix()
            # Only root 0 needs the overall sparsematrix data
            if self._mpi_manager().getRank() == 0:
                spm = observable.sparse_matrix()
            self._mpi_manager().Barrier()
        else:
            spm = observable.sparse_matrix()
        data = np.array(spm.data).astype(self.ctype)
        indices = np.array(spm.indices).astype(np.int64)
        offsets = np.array(spm.indptr).astype(np.int64)

        wires = []
        wires_list = observable.wires.tolist()
        wires.extend([wires_map[w] for w in wires_list])

        return self.sparse_hamiltonian_obs(data, indices, offsets, wires)

    def _pauli_word(self, observable, wires_map: dict):
        """Serialize a :class:`pennylane.pauli.PauliWord` into a Named or Tensor observable."""
        if len(observable) == 1:
            wire, pauli = list(observable.items())[0]
            return self.named_obs(pauli_name_map[pauli], [wires_map[wire]])

        return self.tensor_obs(
            [
                self.named_obs(pauli_name_map[pauli], [wires_map[wire]])
                for wire, pauli in observable.items()
            ]
        )

    def _pauli_sentence(self, observable, wires_map: dict):
        """Serialize a :class:`pennylane.pauli.PauliSentence` into a Hamiltonian."""
        pwords, coeffs = zip(*observable.items())
        terms = [self._pauli_word(pw, wires_map) for pw in pwords]
        coeffs = np.array(coeffs).astype(self.rtype)

        if self.split_obs:
            return [self.hamiltonian_obs([c], [t]) for (c, t) in zip(coeffs, terms)]
        return self.hamiltonian_obs(coeffs, terms)

    # pylint: disable=protected-access
    def _ob(self, observable, wires_map):
        """Serialize a :class:`pennylane.operation.Observable` into an Observable."""
        if isinstance(observable, Tensor):
            return self._tensor_ob(observable, wires_map)
        if observable.name == "Hamiltonian":
            return self._hamiltonian(observable, wires_map)
        if observable.name == "SparseHamiltonian":
            return self._sparse_hamiltonian(observable, wires_map)
        if isinstance(observable, (PauliX, PauliY, PauliZ, Identity, Hadamard)):
            return self._named_obs(observable, wires_map)
        if observable._pauli_rep is not None:
            return self._pauli_sentence(observable._pauli_rep, wires_map)
        return self._hermitian_ob(observable, wires_map)

    def serialize_observables(self, tape: QuantumTape, wires_map: dict) -> List:
        """Serializes the observables of an input tape.

        Args:
            tape (QuantumTape): the input quantum tape
            wires_map (dict): a dictionary mapping input wires to the device's backend wires

        Returns:
            list(ObsStructC128 or ObsStructC64): A list of observable objects compatible with
                the C++ backend
        """

        serialized_obs = []
        offset_indices = [0]

        for observable in tape.observables:
            ser_ob = self._ob(observable, wires_map)
            if isinstance(ser_ob, list):
                serialized_obs.extend(ser_ob)
                offset_indices.append(offset_indices[-1] + len(ser_ob))
            else:
                serialized_obs.append(ser_ob)
                offset_indices.append(offset_indices[-1] + 1)
        return serialized_obs, offset_indices

    def serialize_ops(
        self, tape: QuantumTape, wires_map: dict
    ) -> Tuple[
        List[List[str]],
        List[np.ndarray],
        List[List[int]],
        List[bool],
        List[np.ndarray],
        List[List[int]],
        List[List[bool]],
    ]:
        """Serializes the operations of an input tape.

        The state preparation operations are not included.

        Args:
            tape (QuantumTape): the input quantum tape
            wires_map (dict): a dictionary mapping input wires to the device's backend wires

        Returns:
            Tuple[list, list, list, list, list]: A serialization of the operations, containing a
            list of operation names, a list of operation parameters, a list of observable wires,
            a list of inverses, and a list of matrices for the operations that do not have a
            dedicated kernel.
        """
        names = []
        params = []
        controlled_wires = []
        controlled_values = []
        wires = []
        mats = []

        uses_stateprep = False

        def get_wires(operation, single_op):
            if operation.name[0:2] == "C(" or operation.name == "MultiControlledX":
                name = "PauliX" if operation.name == "MultiControlledX" else operation.base.name
                controlled_wires_list = operation.control_wires
                if operation.name == "MultiControlledX":
                    wires_list = list(set(operation.wires) - set(controlled_wires_list))
                else:
                    wires_list = operation.target_wires
                control_values_list = (
                    [bool(int(i)) for i in operation.hyperparameters["control_values"]]
                    if operation.name == "MultiControlledX"
                    else operation.control_values
                )
                if not hasattr(self.sv_type, name):
                    single_op = QubitUnitary(matrix(single_op.base), single_op.base.wires)
                    name = single_op.name
            else:
                name = single_op.name
                wires_list = single_op.wires.tolist()
                controlled_wires_list = []
                control_values_list = []
            return single_op, name, wires_list, controlled_wires_list, control_values_list

        for operation in tape.operations:
            if isinstance(operation, (BasisState, StatePrep)):
                uses_stateprep = True
                continue
            if isinstance(operation, Rot):
                op_list = operation.expand().operations
            else:
                op_list = [operation]

            for single_op in op_list:
                (
                    single_op,
                    name,
                    wires_list,
                    controlled_wires_list,
                    controlled_values_list,
                ) = get_wires(operation, single_op)
                names.append(name)
                # QubitUnitary is a special case, it has a parameter which is not differentiable.
                # We thus pass a dummy 0.0 parameter which will not be referenced
                if name == "QubitUnitary":
                    params.append([0.0])
                    mats.append(matrix(single_op))
                elif not hasattr(self.sv_type, name):
                    params.append([])
                    mats.append(matrix(single_op))
                else:
                    params.append(single_op.parameters)
                    mats.append([])

                controlled_values.append(controlled_values_list)
                controlled_wires.append([wires_map[w] for w in controlled_wires_list])
                wires.append([wires_map[w] for w in wires_list])

        inverses = [False] * len(names)
        return (
            names,
            params,
            wires,
            inverses,
            mats,
            controlled_wires,
            controlled_values,
        ), uses_stateprep


def global_phase_diagonal(par, wires, controls, control_values):
    """Returns the diagonal of a C(GlobalPhase) operator."""
    diag = np.ones(2 ** len(wires), dtype=np.complex128)
    controls = np.array(controls)
    control_values = np.array(control_values)
    ind = np.argsort(controls)
    controls = controls[ind[-1::-1]]
    control_values = control_values[ind[-1::-1]]
    idx = np.arange(2 ** len(wires), dtype=np.int64).reshape([2 for _ in wires])
    for c, w in zip(control_values, controls):
        idx = np.take(idx, np.array(int(c)), w)
    diag[idx.ravel()] = np.exp(-1j * par)
    return diag
