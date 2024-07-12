The Remote device
===================

The ``'qiskit.remote'`` device is a generic adapter to use any Qiskit backend as interface
for a PennyLane device.

To access IBM backends, we recommend using `Qiskit Runtime <https://docs.quantum.ibm.com/api/migration-guides/qiskit-runtime-from-ibm-provider>`_.

.. code-block:: python

    from qiskit_ibm_runtime import QiskitRuntimeService

    QiskitRuntimeService.save_account(channel="ibm_quantum", token="<YOUR_IBMQ_TOKEN>")

    # To access saved credentials for the IBM quantum channel and select an instance
    service = QiskitRuntimeService(channel="ibm_quantum", instance="my_hub/my_group/my_project")
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=<num_qubits>)

    dev = qml.device('qiskit.remote', wires=<num_qubits>, backend=backend)


.. note:: 

    Certain third-party backends may be using the deprecated ``Provider`` interface, in which case
    you can get the backend instance from providers with complex search options using their 
    ``get_backend()`` method. For example:

    .. code-block:: python

        import pennylane as qml

        def configured_backend():
            backend = SomeProvider.get_backend(...)
        backend.options.update_options(...) # Set backend options this way
        return backend

        dev = qml.device('qiskit.remote', wires=2, backend=configured_backend())

After installing the plugin, this device can be used just like any other PennyLane device for defining and evaluating QNodes.
For example, a simple quantum function that returns the expectation value of a measurement and depends on
three classical input parameters can be decorated with ``qml.qnode`` as usual to construct a ``QNode``:

.. code-block:: python

    @qml.qnode(dev)
    def circuit(x, y, z):
        qml.RZ(z, wires=[0])
        qml.RY(y, wires=[0])
        qml.RX(x, wires=[0])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(wires=1))

You can then execute the above quantum circuit to get the expectation value of a Pauli operator.

.. code-block:: python

    circuit(0.2, 0.1, 0.3)

The ``'qiskit.remote'`` device also supports the use of `local simulators <https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/fake_provider>`_ such as ``FakeManila``.

.. code-block:: python
    
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    backend = FakeManilaV2()

    # You could use an Aer simulator instead by using the following code:
    # from qiskit_aer import AerSimulator
    # backend = AerSimulator()

    dev = qml.device('qiskit.remote', wires=5, backend=backend)

Device options
~~~~~~~~~~~~~~

The ``'qiskit.remote'`` device uses the `EstimatorV2 <https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.EstimatorV2/>`_
and the `SamplerV2 <https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.SamplerV2>`_  runtime primitives to execute
the measurements. To set options for `transpilation <https://docs.quantum.ibm.com/run/configure-runtime-compilation>`_
or `runtime <https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/options>`_, simply pass the keyword arguments into the device.

.. code-block:: python

    dev = qml.device(
        "qiskit.remote", 
        wires=5, 
        backend=backend, 
        resilience_level=1, 
        optimization_level=1, 
        seed_transpiler=42
    )
    # to change options, re-initialize the device
    dev = qml.device(
        "qiskit.remote", 
        wires=5, 
        backend=backend, 
        resilience_level=1, 
        optimization_level=2, 
        seed_transpiler=24
    )

This device is not compatible with analytic mode, so an error will be raised if ``shots=0`` or ``shots=None``.
The default value of the shots argument is ``1024``. You can set the number of shots on device initialization using the 
``shots`` keyword, or you can choose the number of shots on circuit execution.

.. code-block:: python

    dev = qml.device("qiskit.remote", wires=5, backend=backend, shots=4096)

    @qml.qnode(dev)
    def circuit(x, y, z):
        qml.RZ(z, wires=[0])
        qml.RY(y, wires=[0])
        qml.RX(x, wires=[0])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(wires=1))
    
    # Runs with 4096 shots
    circuit(0.2, 0.1, 0.3)

    # Runs with 10000 shots
    circuit(0.2, 0.1, 0.3, shots=10000)
