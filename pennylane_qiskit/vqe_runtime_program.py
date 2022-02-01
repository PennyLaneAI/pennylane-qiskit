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
This module contains a custom VQE runtime program that can be uploaded to IBMQ.
"""
# pylint: disable=too-many-arguments,too-many-branches,too-many-statements
import numpy as np
import scipy.optimize as opt
from scipy.optimize import OptimizeResult
import mthree

import qiskit.circuit.library.n_local as lib_local
from qiskit.algorithms.optimizers import SPSA, QNSPSA
from qiskit import QuantumCircuit, transpile


def opstr_to_meas_circ(op_str):
    """Takes a list of operator strings and creates a Qiskit circuit with the correct pre-measurement rotations.

    Args:
        op_str (list): List of strings representing the operators needed for measurements.

    Returns:
        list: List of circuits for rotations before measurement.
    """
    num_qubits = len(op_str[0])
    circs = []
    for op in op_str:
        qc = QuantumCircuit(num_qubits)
        for idx, item in enumerate(op):
            if item == "X":
                qc.h(idx)
            elif item == "Y":
                qc.sdg(idx)
                qc.h(idx)
            elif item == "H":
                qc.ry(-np.pi / 4, idx)
        circs.append(qc)
    return circs


def main(
    backend,
    user_messenger,
    hamiltonian,
    x0,
    ansatz="EfficientSU2",
    ansatz_config=None,
    optimizer="SPSA",
    optimizer_config=None,
    shots=8192,
    use_measurement_mitigation=False,
):
    """
    The main sample VQE program.

    Args:
        backend (qiskit.providers.ibmq.runtime.ProgramBackend): Qiskit backend instance.
        user_messenger (qiskit.providers.ibmq.runtime.UserMessenger): Used to communicate with the program user.
        hamiltonian (list): Hamiltonian whose ground state we want to find. e.g. [(1, XY),(1, IH)].
        x0 (array_like): Initial vector of parameters.
        ansatz (str): Optional, QuantumCircuit or the name of ansatz quantum circuit to use, default='EfficientSU2'.
        ansatz_config (dict): Optional, configuration parameters for the ansatz circuit.
        optimizer (str): Optional, string specifying classical optimizer, default='SPSA'.
        optimizer_config (dict): Optional, configuration parameters for the optimizer.
        shots (int): Optional, number of shots to take per circuit.
        use_measurement_mitigation (bool): Optional, use measurement mitigation, default=False.

    Returns:
        OptimizeResult: The result in SciPy optimization format.
    """

    if ansatz_config is None:
        ansatz_config = {}

    if optimizer_config is None:
        optimizer_config = {"maxiter": 100}

    coeffs = np.array([item[0] for item in hamiltonian], dtype=complex)
    op_strings = [item[1] for item in hamiltonian]

    num_qubits = len(op_strings[0])

    # Get the Qiskit circuit from the library if a str was given
    if isinstance(ansatz, str):
        ansatz_instance = getattr(lib_local, ansatz)
        ansatz_circuit = ansatz_instance(num_qubits, **ansatz_config)
    else:
        ansatz_circuit = ansatz

    meas_circs = opstr_to_meas_circ(op_strings)

    meas_strings = [
        string.replace("X", "Z").replace("Y", "Z").replace("H", "Z") for string in op_strings
    ]

    # Take the ansatz circuits and add measurements
    full_circs = [ansatz_circuit.compose(mcirc).measure_all(inplace=False) for mcirc in meas_circs]

    num_params = ansatz_circuit.num_parameters

    # Check initial state
    if x0 is not None:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape[0] != num_params:
            shape = x0.shape[0]
            raise ValueError(
                f"Number of params in x0 ({shape}) does not match number \
                              of ansatz parameters ({num_params})."
            )
    else:
        x0 = 2 * np.pi * np.random.rand(num_params)

    # Transpile the circuits
    trans_dict = {}
    if not backend.configuration().simulator:
        trans_dict = {"layout_method": "sabre", "routing_method": "sabre"}
    trans_circs = transpile(full_circs, backend, optimization_level=3, **trans_dict)

    # Measurement mitigation
    if use_measurement_mitigation:
        maps = mthree.utils.final_measurement_mapping(trans_circs)
        mit = mthree.M3Mitigation(backend)
        mit.cals_from_system(maps)

    def callback(*args):
        user_messenger.publish(args)

    def vqe_func(params):
        # Binds parameters to the transpiled circuits.
        bound_circs = [circ.bind_parameters(params) for circ in trans_circs]

        # Submit the job and get the counts
        counts = backend.run(bound_circs, shots=shots).result().get_counts()

        if use_measurement_mitigation:
            quasi_collection = mit.apply_correction(counts, maps)
            expvals = quasi_collection.expval(meas_strings)
        else:
            expvals = mthree.utils.expval(counts, meas_strings)

        energy = np.sum(coeffs * expvals).real
        return energy

    # SPSA and QNSPSA are taken from Qiskit and not SciPy
    if optimizer == "SPSA":
        spsa = SPSA(**optimizer_config, callback=callback)
        x, loss, nfev = spsa.optimize(num_params, vqe_func, initial_point=x0)
        res = OptimizeResult(
            fun=loss,
            x=x,
            nit=optimizer_config["maxiter"],
            nfev=nfev,
            message="Optimization terminated successfully.",
            success=True,
        )
    elif optimizer == "QNSPSA":
        fidelity = QNSPSA.get_fidelity(ansatz_circuit)
        spsa = QNSPSA(fidelity, **optimizer_config, callback=callback)
        x, loss, nfev = spsa.optimize(num_params, vqe_func, initial_point=x0)
        res = OptimizeResult(
            fun=loss,
            x=x,
            nit=optimizer_config["maxiter"],
            nfev=nfev,
            message="Optimization terminated successfully.",
            success=True,
        )
    # SciPy optimizers
    else:
        res = opt.minimize(
            vqe_func, x0, method=optimizer, options=optimizer_config, callback=callback
        )

    return res
