The Aer device
==============

You can instantiate a ``'qiskit.aer'`` device for PennyLane with:

.. code-block:: python

    import pennylane as qml
    dev = qml.device('qiskit.aer', wires=2)

This device can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.
A simple quantum function that returns the expectation value of a measurement and depends on three classical input
parameters would look like:

.. code-block:: python

    @qml.qnode(dev)
    def circuit(x, y, z):
        qml.RZ(z, wires=[0])
        qml.RY(y, wires=[0])
        qml.RX(x, wires=[0])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(wires=1))

You can then execute the circuit like any other function to get the quantum mechanical expectation value.

.. code-block:: python

    circuit(0.2, 0.1, 0.3)

Backends
~~~~~~~~

Qiskit's Aer layer has several backends, for example ``'qasm_simulator'``,
``'statevector_simulator'``, ``'unitary_simulator'``. For more information on backends, please visit the
`qiskit documentation <https://qiskit.org/documentation/the_elements.html#aer>`_ and
`qiskit tutorials <https://qiskit.org/documentation/tutorials/advanced/aer/1_aer_provider.html>`_.

You can change the default device's backend with

.. code-block:: python

    dev = qml.device('qiskit.aer', wires=2, backend='unitary_simulator')

To get a current overview what backends are available you can query this by

.. code-block:: python

    dev.capabilities()['backend']

.. note::

    Currently, PennyLane does not support the ``'pulse_simulator'`` backend.

Noise models
~~~~~~~~~~~~

One great feature of the ``'qiskit.aer'`` device is the ability to simulate noise. There are different noise models,
which you can instantiate and apply to the device by calling

.. code-block:: python

    import pennylane as qml

    import qiskit
    from qiskit.providers.aer.noise.device import basic_device_noise_model

    qiskit.IBMQ.load_account()
    provider = qiskit.IBMQ.get_provider(group='open')
    ibmq_16_melbourne = provider.get_backend('ibmq_16_melbourne')
    device_properties = ibmq_16_melbourne.properties()

    noise_model = basic_device_noise_model(device_properties)

    dev = qml.device('qiskit.aer', wires=2, noise_model=noise_model)

Please refer to the Qiskit documentation for more information on
`noise models <https://qiskit.org/documentation/tutorials/advanced/aer/3_building_noise_models.html>`_.
