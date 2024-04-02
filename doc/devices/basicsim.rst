The BasicSim device
===================

While the ``'qiskit.aer'`` device is the standard go-to simulator that is provided along
the Qiskit main package installation, there exists a natively included python simulator
that is slower but will work usually without the need to install other dependencies
(C++, BLAS, and so on). This simulator can be used through the device ``'qiskit.basicsim'``:

.. code-block:: python

    import pennylane as qml
    dev = qml.device('qiskit.basicsim', wires=2)

This device uses the Qiskit ``BasicSimulator`` backend from the
`basic_provider <https://docs.quantum.ibm.com/api/qiskit/providers_basic_provider>`_ module in Qiskit.