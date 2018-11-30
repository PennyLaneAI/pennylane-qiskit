Tutorials
=========

To see the PennyLane qiskit plugin in action you can use any of the `qubit based tutorials <https://pennylane.readthedocs.io/en/latest/tutorials/notebooks.html>`_
from the documentation of PennyLane and run them on a ``'qiskit.aer'`` device by replacing the ``'default.qubit'`` device used there with

.. code-block:: python

    dev = qml.device('qiskit.aer', wires=XXX)

for an appropriate number of wires.

You can also try to run, e.g., the qubit rotation code on actual quantum hardware by using the ``'qiskit.ibm'`` device.
