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

.. note::

    The `Qiskit Aer <https://qiskit.github.io/qiskit-aer/>`_ device
    provides a fast simulator that is also capable of simulating
    noise. It is available as :ref:`"qiskit.aer" <aer device page>`.
    