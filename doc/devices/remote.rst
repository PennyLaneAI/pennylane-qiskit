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
