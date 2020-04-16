The BasicAer device
===================

While the ``'qiskit.aer'`` device is the standard go-to simulator that is provided along
the Qiskit main package installation, there exists a natively included python simulator
that is slower but will work usually without the need to install other dependencies
(C++, BLAS, and so on). This simulator can be used through the device ``'qiskit.basicaer'``:

.. code-block:: python

    import pennylane as qml
    dev = qml.device('qiskit.basicaer', wires=2)

As with the ``'qiskit.aer'`` device, there are different backends available, which you can find
by calling

.. code-block:: python

    dev.capabilities()['backend']

.. note::

    Currently, PennyLane does not support the ``'pulse_simulator'`` backend.

The backends are used in the same manner as specified for the ``'qiskit.aer'`` device.
The ``'qiskit.basicaer'`` device, however, does not support the simulation of noise models.
