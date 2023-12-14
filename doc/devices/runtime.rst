Qiskit Runtime Programs
=======================

PennyLane-Qiskit supports running PennyLane on IBM Q hardware via the Qiskit runtime programs ``circuit-runner``
and ``sampler``. You can choose between those two runtime programs and also have the possibility to choose the
backend on which the circuits will be run. Those two devices inherit directly from the ``IBMQ`` device and work the
the same way, you can refer to the corresponding documentation for details about token and providers
`IBMQ documentation for PennyLane <https://pennylaneqiskit.readthedocs.io/en/latest/devices/ibmq.html>`_.

You can use the ``circuit_runner`` and ``sampler`` devices by using their short names, for example:

.. code-block:: python

    dev = qml.device('qiskit.ibmq.circuit_runner', wires=2, backend='ibmq_qasm_simulator', shots=8000, **kwargs)


.. code-block:: python

    dev = qml.device('qiskit.ibmq.sampler', wires=2, backend='ibmq_qasm_simulator', shots=8000, **kwargs)

More details on Qiskit runtime programs in the `IBMQ runtime documentation <https://qiskit.org/documentation/partners/qiskit_ibm_runtime/index.html>`_.
