Overview
========

Devices
~~~~~~~

Once the PennyLane-Qiskit plugin is installed, the following Qiskit devices
can be accessed straightaway in PennyLane.

.. devicegalleryitem::
    :name: 'qiskit.aer'
    :description: Qiskit's staple simulator and has great features like noise models.
    :link: devices/aer

.. devicegalleryitem::
    :name: 'qiskit.basicaer'
    :description: A simplified version of the Aer device, which requires fewer dependencies.
    :link: /devices/basicaer

.. devicegalleryitem::
    :name: 'qiskit.ibmq'
    :description: Allows integration with qiskit's hardware backends, and hardware-specific simulators.
    :link: devices/ibmq

.. raw:: html

        <div style='clear:both'></div>

A device is chosen by calling (here for 2 wires):

.. code-block:: python

    import pennylane as qml
    dev = qml.device('qiskit.aer', wires=2)

Qiskit devices have different **backends**, which define which actual simulator or hardware is used by the
device. Different simulator backends are optimized for different types of circuits. A backend can be defined as
follows:

.. code-block:: python

    dev = qml.device('qiskit.aer', wires=2, backend='unitary_simulator')


Tutorials
~~~~~~~~~

To see the PennyLane-Qiskit plugin in action, you can use any of the qubit based `tutorials
from the PennyLane documentation <https://pennylane.ai/qml/beginner.html>`_, for example
the tutorial on `qubit rotation <https://pennylane.ai/qml/tutorial/tutorial_qubit_rotation.html>`_,
and simply replace ``'default.qubit'`` with any of the available Qiskit devices, such as ``'qiskit.aer'``:

.. code-block:: python

    dev = qml.device('qiskit.aer', wires=XXX)

You can also try to run tutorials, such as the qubit rotation tutorial, on actual quantum hardware by
using the ``'qiskit.ibmq'`` device.
