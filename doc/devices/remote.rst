The Remote device
===================

The ``'qiskit.remote'`` device is a generic adapter to use any Qiskit backend as interface
for a PennyLane device.

This device is useful when retrieving backends from providers with complex search options in
their ``get_backend()`` method, or for setting options on a backend prior to wrapping it as
PennyLane device.

.. code-block:: python

    import pennylane as qml

    def configured_backend():
        backend = SomeProvider.get_backend(...)
	backend.options.update_options(...)
	return backend

    dev = qml.device('qiskit.remote', wires=2, backend=configured_backend())

.. warning::

    Retrieving a backend from a provider has been deprecated and may not be supported 
    in the future. To access Qiskit backends, we recommend migrating to 
    `Qiskit Runtime <https://docs.quantum.ibm.com/api/migration-guides/qiskit-runtime-from-ibm-provider>`_.

.. code-block:: python

    from qiskit_ibm_runtime import QiskitRuntimeService

    QiskitRuntimeService.save_account(channel="ibm_quantum", token="<IQP_TOKEN>", overwrite=True, default=true)

    # To access saved credentials for the IBM quantum channel and select an instance
    service = QiskitRuntimeService(channel="ibm_quantum", instance="my_hub/my_group/my_project")
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=<num_qubits>)

    dev = qml.device('qiskit.remote', wires=<num_qubits>, backend=backend)

The ``'qiskit.remote'`` device also supports the use of local simulators such as FakeManila.

.. code-block:: python
    
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    backend = FakeManilaV2()

    # You could use an Aer simulator instead by using the following code:
    # from qiskit_aer import AerSimulator
    # backend = AerSimulator()

    dev = qml.device('qiskit.remote', wires=2, backend=backend)

The ``'qiskit.remote'`` device uses the `EstimatorV2 <https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.EstimatorV2/>`_
and the `SamplerV2 <https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.SamplerV2>`_  runtime primitives to execute
the measurements. To set options for `transpilation <https://docs.quantum.ibm.com/run/configure-runtime-compilation>`_
or `runtime <https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/options>`_, simply pass the keyword arguments into the device.

.. code-block:: python

    dev = qml.device("qiskit.remote", wires=5, backend=backend, resilience_level=1, optimization_level=1, seed_transpiler=42)
    # to change options, re-initialize the device
    dev = qml.device("qiskit.remote", wires=5, backend=backend, resilience_level=1, optimization_level=2, seed_transpiler=24)
