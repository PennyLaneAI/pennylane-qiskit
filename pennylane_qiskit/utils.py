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
Utility functions
=================

.. currentmodule:: pennylane_qiskit.utils

This module contains various utility functions.

.. autosummary::
   spectral_decomposition
   mat_vec_product

Code details
~~~~~~~~~~~~
"""
import numpy as np
from numpy.linalg import eigh


def spectral_decomposition(A):
    r"""Spectral decomposition of a Hermitian matrix.

    Args:
        A (array): Hermitian matrix

    Returns:
        (vector[float], list[array[complex]]): (a, P): eigenvalues and hermitian projectors
            such that :math:`A = \sum_k a_k P_k`.
    """
    d, v = eigh(A)
    P = []
    for k in range(d.shape[0]):
        temp = v[:, k]
        P.append(np.outer(temp, temp.conj()))
    return d, P


def mat_vec_product(mat, vec, wires, total_wires):
    r"""Apply multiplication of a matrix to subsystems of the quantum state.

    Args:
        mat (array): matrix to multiply
        vec (array): state vector to multiply
        wires (Sequence[int]): target subsystems
        total_wires (int): total number of subsystems

    Returns:
        array: output vector after applying ``mat`` to input ``vec`` on specified subsystems
    """
    num_wires = len(wires)

    if mat.shape != (2 ** num_wires, 2 ** num_wires):
        raise ValueError(
            "Please specify a {N} x {N} matrix for {w} wires.".format(N=2 ** num_wires, w=num_wires)
        )

    # first, we need to reshape both the matrix and vector
    # into blocks of 2x2 matrices, in order to do the higher
    # order matrix multiplication

    # Reshape the matrix to ``size=[2, 2, 2, ..., 2]``,
    # where ``len(size) == 2*len(wires)``
    mat = np.reshape(mat, [2] * len(wires) * 2)

    # Reshape the state vector to ``size=[2, 2, ..., 2]``,
    # where ``len(size) == num_wires``.
    # Each wire corresponds to a subsystem.
    vec = np.reshape(vec, [2] * total_wires)

    # Calculate the axes on which the matrix multiplication
    # takes place. For the state vector, this simply
    # corresponds to the requested wires. For the matrix,
    # it is the latter half of the dimensions (the 'bra' dimensions).
    axes = (np.arange(len(wires), 2 * len(wires)), wires)

    # After the tensor dot operation, the resulting array
    # will have shape ``size=[2, 2, ..., 2]``,
    # where ``len(size) == num_wires``, corresponding
    # to a valid state of the system.
    tdot = np.tensordot(mat, vec, axes=axes)

    # Tensordot causes the axes given in `wires` to end up in the first positions
    # of the resulting tensor. This corresponds to a (partial) transpose of
    # the correct output state
    # We'll need to invert this permutation to put the indices in the correct place
    unused_idxs = [idx for idx in range(total_wires) if idx not in wires]
    perm = wires + unused_idxs

    # argsort gives the inverse permutation
    inv_perm = np.argsort(perm)
    state_multi_index = np.transpose(tdot, inv_perm)

    return np.reshape(state_multi_index, 2 ** total_wires)
