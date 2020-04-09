PennyLane-Qiskit Plugin
#######################

:Release: |release|

.. include:: ../README.rst
  :start-after:	header-start-inclusion-marker-do-not-remove
  :end-before: header-end-inclusion-marker-do-not-remove


Once the PennyLane-Qiskit plugin is installed, the the Qiskit devices
can be accessed straightaway in PennyLane, without the need to import new packages.

Devices
~~~~~~~

Currently, there are three different devices available:

.. devicegalleryitem::
    :name: 'qiskit.aer'
    :description: Qiskit's staple simulator with great features such as noise models.
    :link: devices/aer.html

.. devicegalleryitem::
    :name: 'qiskit.basicaer'
    :description: A simplified version of the Aer device, which requires fewer dependencies.
    :link: devices/basicaer.html

.. devicegalleryitem::
    :name: 'qiskit.ibmq'
    :description: Allows integration with qiskit's hardware backends, and hardware-specific simulators.
    :link: devices/ibmq.html

.. raw:: html

        <div style='clear:both'></div>
        </br>

For example, the ``'qiskit.aer'`` device with two wires is called like this:

.. code-block:: python

    import pennylane as qml
    dev = qml.device('qiskit.aer', wires=2)


Backends
~~~~~~~~

Qiskit devices have different **backends**, which define which actual simulator or hardware is used by the
device. Different simulator backends are optimized for different types of circuits. A backend can be defined as
follows:

.. code-block:: python

    dev = qml.device('qiskit.aer', wires=2, backend='unitary_simulator')

PennyLane chooses the ``qasm_simulator`` as the default backend if no backend is specified.
For more details on the ``qasm_simulator``, including available backend options, see
`Qiskit Qasm Simulator documentation <https://qiskit.org/documentation/stubs/qiskit.providers.aer.QasmSimulator.html>`_.

Tutorials
~~~~~~~~~

To see the PennyLane-Qiskit plugin in action, you can use any of the qubit based `demos
from the PennyLane documentation <https://pennylane.ai/qml/demonstrations.html>`_, for example
the tutorial on `qubit rotation <https://pennylane.ai/qml/demos/tutorial_qubit_rotation.html>`_,
and simply replace ``'default.qubit'`` with any of the available Qiskit devices, such as ``'qiskit.aer'``:

.. code-block:: python

    dev = qml.device('qiskit.aer', wires=XXX)

You can also try to run tutorials, such as the qubit rotation tutorial, on actual quantum hardware by
using the ``'qiskit.ibmq'`` device.

To filter tutorials that explicitly use a qiskit device, use the "Qiskit" filter on the right panel of the
`demos <https://pennylane.ai/qml/demonstrations.html>`_.



.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   installation
   support

.. toctree::
   :maxdepth: 2
   :caption: Usage
   :hidden:

   devices/aer
   devices/basicaer
   devices/ibmq

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   code
