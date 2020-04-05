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

Qiskit's Aer has several backends, for example ``'qasm_simulator'``,
``'statevector_simulator'``, ``'unitary_simulator'``.
If no backend is specified, PennyLane uses the ``'qasm_simulator'`.
For more information on available backends, please visit the
`qiskit documentation <https://qiskit.org/documentation/the_elements.html#aer>`_ and the
`qiskit tutorials <https://qiskit.org/documentation/tutorials/advanced/aer/1_aer_provider.html>`_.

You can change an ``'qiskit.aer'`` device's backend with

.. code-block:: python

    dev = qml.device('qiskit.aer', wires=2, backend='unitary_simulator')

To get a current overview what backends are available you can query

.. code-block:: python

    dev.capabilities()['backend']

.. note::

    Currently, PennyLane currently does not support the ``'pulse_simulator'`` backend.

Noise models
~~~~~~~~~~~~

One great feature of the ``'qiskit.aer'`` device is the ability to simulate noise. There are different noise models,
which you can instantiate and apply to the device as follows
(adapting `this <https://qiskit.org/documentation/apidoc/aer_noise.html>`_ qiskit tutorial):

.. code-block:: python

    import pennylane as qml

    import qiskit
    import qiskit.providers.aer.noise as noise

    # Error probabilities
    prob_1 = 0.001  # 1-qubit gate
    prob_2 = 0.01   # 2-qubit gate

    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(prob_1, 1)
    error_2 = noise.depolarizing_error(prob_2, 2)

    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

    # Create a PennyLane device
    dev = qml.device('qiskit.aer', wires=2, noise_model=noise_model)

    # Create a PennyLane quantum node run on the device
    @qml.qnode(dev)
    def circuit(x, y, z):
        qml.RZ(z, wires=[0])
        qml.RY(y, wires=[0])
        qml.RX(x, wires=[0])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(wires=1))

    # Result of noisy simulator
    print(circuit(0.2, 0.1, 0.3))

Please refer to the Qiskit documentation for more information on
`noise models <https://qiskit.org/documentation/tutorials/advanced/aer/3_building_noise_models.html>`_.
