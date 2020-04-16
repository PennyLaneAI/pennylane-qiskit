The Aer device
==============
The ``qiskit.aer`` device provided by the PennyLane-Qiskit plugin allows you to use PennyLane
to deploy and run your quantum machine learning models on the backends and simulators provided
by `Qiskit Aer <https://qiskit.org/aer/>`_.

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
`Aer provider documentation <https://qiskit.org/documentation/apidoc/aer_provider.html>`_ and the
`Qiskit Aer tutorials <https://qiskit.org/documentation/tutorials/advanced/aer/1_aer_provider.html>`_.


To get a current overview what backends are available you can query

.. code-block:: python

    dev.capabilities()['backend']

or, alternatively,

.. code-block:: python

    from qiskit import Aer
    Aer.backends()

.. note::

    Currently, PennyLane does not support the ``'pulse_simulator'`` backend.

You can change a ``'qiskit.aer'`` device's backend with the ``backend`` argument when creating the ``device``:

.. code-block:: python

    dev = qml.device('qiskit.aer', wires=2, backend='unitary_simulator')

Backend options
~~~~~~~~~~~~~~~

Qiskit's backends can take different *backend options*, for example to specify the numerical
precision of the simulation.
You can find a list of backend options in the backends' respective API documentations:

* `'qasm_simulator' <https://qiskit.org/documentation/stubs/qiskit.providers.aer.QasmSimulator.html>`_
* `'statevector_simulator' <https://qiskit.org/documentation/stubs/qiskit.providers.aer.StatevectorSimulator.html>`_
* `'unitary_simulator' <https://qiskit.org/documentation/stubs/qiskit.providers.aer.UnitarySimulator.html>`_

The options are set via

.. code-block:: python

    dev = qml.device('qiskit.aer', wires=2, backend='unitary_simulator',
                     backend_options={"validation_threshold": 1e-6})

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
