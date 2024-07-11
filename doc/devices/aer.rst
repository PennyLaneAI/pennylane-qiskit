.. _aer device page:

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

You can then execute the circuit like any other function to get the expectation value of a Pauli 
operator.

.. code-block:: python

    circuit(0.2, 0.1, 0.3)

Backend Methods and Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default backend is the ``AerSimulator``. However, multiple other backends are also available.
To get a current overview what backends are available you can query

.. code-block:: python

    from qiskit_aer import Aer
    Aer.backends()

.. note::

    Currently, PennyLane does not support the ``'pulse_simulator'`` backend.

You can change a ``'qiskit.aer'`` device's backend with the ``backend`` argument when creating the ``device``:

.. code-block:: python

    from qiskit_aer import UnitarySimulator
    dev = qml.device('qiskit.aer', wires=2, backend=UnitarySimulator())

.. note::

    Occassionally, you may see others pass in a string as a backend. For example:

    .. code-block:: python

        dev = qml.device('qiskit.aer', wires=2, backend='unitary_simulator')

    At the time of writing, this is still functional for the Aer devices. However, this will soon be 
    deprecated and may not function as intended. To ensure accurate results, we recommend passing in 
    a backend instance.

The ``AerSimulator`` backend has several available methods, which
can be passed via the ``method`` keyword argument. For example
``'automatic'``, ``'statevector'``, and ``'unitary'``.

.. code-block:: python

    dev = qml.device("qiskit.aer", wires=2, backend=AerSimulator(), method="automatic")

Each of these methods can take different *run options*, for example to specify the numerical
precision of the simulation.

The options are set via additional keyword arguments:

.. code-block:: python

    dev = qml.device(
        'qiskit.aer',
        wires=2,
        backend=AerSimulator(),
        validation_threshold=1e-6
    )

For more information on available methods and their options, please visit the `AerSimulator
documentation <https://qiskit.org/ecosystem/aer/stubs/qiskit_aer.AerSimulator.html>`_.

.. warning::

    The ``AerSimulator`` methods ``"stabilizer"``, ``"extended_stabilizer"``, ``"matrix_product_state"``,
    and ``"superop"`` are currently not supported.

Noise models
~~~~~~~~~~~~

One great feature of the ``'qiskit.aer'`` device is the ability to simulate noise. There are 
different noise models, which you can instantiate and apply to the device as follows (adapted 
from a `Qiskit tutorial <https://qiskit.github.io/qiskit-aer/tutorials/4_custom_gate_noise.html>`_.):

.. code-block:: python

    import pennylane as qml

    import qiskit
    from qiskit_aer import noise

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
`noise models <https://qiskit.github.io/qiskit-aer/tutorials/3_building_noise_models.html>`_.
