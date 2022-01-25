Qiskit Runtime Programs
=======================

PennyLane-Qiskit supports running PennyLane on IBM Q hardware via the Qiskit runtime programs ``circuit-runner``
and ``sampler``. You can choose between those two runtime programs and also have the possibility to choose the
backend on which the circuits will be run. Those two devices inherit directly from the ``IBMQ`` device and work the
the same way, you can refer to the corresponding documentation for details about token and providers.

You can use the ``circuit_runner`` and ``sampler`` devices by using their short names, for example:

.. code-block:: python

    dev = qml.device('qiskit.ibmq.circuit_runner', wires=2, backend='ibmq_qasm_simulator', shots=8000, **kwargs)


.. code-block:: python

    dev = qml.device('qiskit.ibmq.sampler', wires=2, backend='ibmq_qasm_simulator', shots=8000, **kwargs)


Custom Runtime Programs
~~~~~~~~~~~~~~~~~~~~~~~

Not all Qiskit runtime programs correspond to complete devices but rather solve specific problems (VQE, QAOA, etc...).
We created a custom Qiskit runtime program for solving VQE problems in PennyLane ``runtime_programs\vqe_runtime_program.py``.
In order to use this program you need to upload on IBMQ (only once), get the program ID and use the VQE runner.

.. code-block:: python

    from pennylane_qiskit import upload_vqe_runner, vqe_runner

    IBMQ.enable_account(token)

    program_id = upload_vqe_runner(hub="ibm-q", group="open", project="main")

    def vqe_circuit(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=0)

    coeffs = [1, 1]
    obs = [qml.PauliX(0), qml.PauliZ(0)]
    hamiltonian = qml.Hamiltonian(coeffs, obs)

    job = vqe_runner(
        program_id=program_id,
        backend="ibmq_qasm_simulator",
        hamiltonian=hamiltonian,
        ansatz=vqe_circuit,
        x0=[3.97507603, 3.00854038],
        shots=shots,
        optimizer="SPSA",
        optimizer_config={"maxiter": 40},
        kwargs={"hub": "ibm-q", "group": "open", "project": "main"},
    )

More details on Qiskit runtime programs in the `IBMQ runtime documentation <https://qiskit.org/documentation/apidoc/ibmq_runtime.html>`_.
