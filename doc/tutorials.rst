Tutorials
=========

To see the PennyLane-Qiskit plugin in action, you can use any of the qubit based `tutorials
from the PennyLane documentation <https://pennylane.ai/qml/beginner.html>`_, for example
the tutorial on `qubit rotation <https://pennylane.ai/qml/tutorial/tutorial_qubit_rotation.html>`_,
and simply replace ``'default.qubit'`` with any of the available Qiskit devices, such as ``'qiskit.aer'``:

.. code-block:: python

    dev = qml.device('qiskit.aer', wires=XXX)

You can also try to run tutorials, such as the qubit rotation tutorial, on actual quantum hardware by using the ``'qiskit.ibmq'`` device.
