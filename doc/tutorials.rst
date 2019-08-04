Tutorials
=========

To see the PennyLane Qiskit plugin in action, you can use any of the `qubit based tutorials <https://pennylane.readthedocs.io/en/latest/tutorials/notebooks.html>`_
from the PennyLane documentation, and simply replace ``'default.qubit'`` with any of the available Qiskit devices, such as ``'qiskit.aer'``:

.. code-block:: python

    dev = qml.device('qiskit.aer', wires=XXX)

You can also try to run tutorials, such as the qubit rotation tutorial, on actual quantum hardware by using the ``'qiskit.ibmq'`` device.
