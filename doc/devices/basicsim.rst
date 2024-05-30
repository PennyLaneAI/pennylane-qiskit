.. _basicsim device page:

The BasicSim device
===================

Qiskit comes packed with a
`basic pure-Python simulator <https://docs.quantum.ibm.com/api/qiskit/qiskit.providers.basic_provider.BasicSimulator>`_
that can be accessed in this plugin through:

.. code-block:: python

    import pennylane as qml
    dev = qml.device('qiskit.basicsim', wires=2)

This device uses the Qiskit ``BasicSimulator`` backend from the
`basic_provider <https://docs.quantum.ibm.com/api/qiskit/providers_basic_provider>`_ module in Qiskit.
In Qiskit, ``BasicSimulator`` uses ``shots=1024`` by default. As such, the same convention has been applied
to `'qiskit.basicsim'` in PennyLane. 

.. code-block:: python

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(0)
        qml.Hadamard(1)
        return qml.count(wires=1)

.. code-block:: pycon
    
    >>> circuit()
    {'0': tensor(520, requires_grad=True), '1': tensor(504, requires_grad=True)} 

.. note::

    The `Qiskit Aer <https://qiskit.github.io/qiskit-aer/>`_ device
    provides a fast simulator that is also capable of simulating
    noise. It is available as :ref:`"qiskit.aer" <aer device page>`, but the backend must be
    installed separately with ``pip install qiskit-aer``.
    
