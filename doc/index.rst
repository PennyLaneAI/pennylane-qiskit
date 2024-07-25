PennyLane-Qiskit Plugin
#######################

:Release: |release|

.. include:: ../README.rst
  :start-after:   header-start-inclusion-marker-do-not-remove
  :end-before: header-end-inclusion-marker-do-not-remove


Once the PennyLane-Qiskit plugin is installed, the Qiskit devices
can be accessed straightaway in PennyLane, without the need to import new packages.

Devices
~~~~~~~

The following devices are available:

.. title-card::
    :name: 'qiskit.aer'
    :description: Qiskit's staple simulator with great features such as noise models.
    :link: devices/aer.html


.. title-card::
    :name: 'qiskit.basicsim'
    :description: A simple local Python simulator running the Qiskit ``BasicSimulator``.
    :link: devices/basicsim.html

.. title-card::
    :name: 'qiskit.remote'
    :description: Allows integration with any Qiskit backend.
    :link: devices/remote.html

.. raw:: html

    <div style='clear:both'></div>
    </br>

For example, the ``'qiskit.aer'`` device with two wires is called like this:

.. code-block:: python

    import pennylane as qml
    dev = qml.device('qiskit.aer', wires=2)


Backends
~~~~~~~~

Qiskit devices have different **backends**, which define the actual simulator or hardware 
used by the device. A backend instance should be initalized and passed to the device.

Different simulator backends are optimized for different purposes. To change what backend is used, 
a simulator backend can be defined as follows:

.. code-block:: python

    from qiskit_aer import UnitarySimulator

    dev = qml.device('qiskit.aer', wires=<num_qubits>, backend=UnitarySimulator())

.. note::

    For ``'qiskit.aer'``, PennyLane chooses the ``aer_simulator`` as the default backend if no 
    backend is specified. For more details on the ``aer_simulator``, including available backend 
    options, see `Qiskit Aer Simulator documentation <https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html#qiskit_aer.AerSimulator.run>`_.

To access a real device, we can use the ``'qiskit.remote'`` device. A real hardware backend can 
be defined as follows:

.. code-block:: python

    from qiskit_ibm_runtime import QiskitRuntimeService

    QiskitRuntimeService.save_account(channel="ibm_quantum", token="<IQP_TOKEN>")

    # To access saved credentials for the IBM quantum channel and select an instance
    service = QiskitRuntimeService(channel="ibm_quantum", instance="my_hub/my_group/my_project")
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=<num_qubits>)

    # passing a string in backend would result in an error
    dev = qml.device('qiskit.remote', wires=<num_qubits>, backend=backend)

Tutorials
~~~~~~~~~

Check out these demos to see the PennyLane-Qiskit plugin in action:

.. raw:: html

    <div class="row">

.. demogalleryitem::
    :name: Ensemble classification with Forest and Qiskit devices
    :figure: https://pennylane.ai/_static/demonstration_assets/ensemble_multi_qpu/ensemble_diagram.png
    :link:  https://pennylane.ai/qml/demos/ensemble_multi_qpu.html
    :tooltip: Use multiple QPUs to improve classification

.. demogalleryitem::
    :name: Quantum volume
    :figure: https://pennylane.ai/_static/demonstration_assets/quantum_volume/quantum_volume_thumbnail.png
    :link:  https://pennylane.ai/qml/demos/quantum_volume.html
    :tooltip: Learn how to compute the quantum volume of a quantum processor

.. demogalleryitem::
    :name: Using PennyLane with IBM's quantum devices and Qiskit
    :figure: https://pennylane.ai/_static/demonstration_assets/ibm_pennylane/thumbnail_tutorial_ibm_pennylane.png
    :link: https://pennylane.ai/qml/demos/ibm_pennylane
    :tooltip: Use IBM devices with PennyLane through the pennylane-qiskit plugin

.. raw:: html

    </div></div><div style='clear:both'> <br/>


You can also try it out using any of the qubit based `demos from the PennyLane documentation
<https://pennylane.ai/qml/demonstrations.html>`_, for example the tutorial on
`qubit rotation <https://pennylane.ai/qml/demos/tutorial_qubit_rotation.html>`_.
Simply replace ``'default.qubit'`` with any of the available Qiskit devices,
such as ``'qiskit.aer'``, or ``'qiskit.remote'`` if you have an API key for
hardware access.

.. raw:: html

    <br/>


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
   devices/basicsim
   devices/remote

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   code
