IBM Q Experience
================

PennyLane-Qiskit supports running PennyLane on IBM Q hardware via the ``qistkit.ibmq`` device.
You can choose between different backends - either simulators tailor-made to emulate the real hardware,
or the real hardware itself.

Find out which backends are available by calling

.. code-block:: python

    import pennylane as qml

    dev = qml.device('qiskit.ibmq', wires=2)
    dev.capabilities()['backend']

Accounts and Tokens
~~~~~~~~~~~~~~~~~~~

By default, the ``qiskit.ibmq`` device will attempt to use an already active or stored
IBM Q account. If none are available, you may also directly pass your IBM Q API token,
as well as an optional URL:

.. code-block:: python

    dev = qml.device('qiskit.ibmq', wires=2, backend='ibmq_qasm_simulator', ibmqx_token="XXX")


In order to avoid accidentally publishing your token, it is best to store it using the
``qiskit.IBMQ.save_account()`` function. Alternatively, you can specify the token or URL via the
`PennyLane configuration file <https://pennylane.readthedocs.io/en/latest/introduction/configuration.html>`__ by
adding a section such as

.. code::

  [qiskit.global]

    [qiskit.ibmq]
    ibmqx_token = "XXX"
    ibmqx_url = "XXX"

Note that, by default, the ``qiskit.ibmq`` device uses the simulator backend
``ibmq_qasm_simulator``, but this may be changed to any of the real backends as given by

.. code-block:: python

    dev.capabilities()['backend']

When getting a ``qiskit.ibmq`` device a Qiskit provider is used to connect to the IBM Q systems.

Custom providers can be passed as arguments when a ``qiskit.ibmq`` device is created:

.. code-block:: python

    from qiskit import IBMQ
    provider = IBMQ.enable_account('XYZ')

    import pennylane as qml
    dev = qml.device('qiskit.ibmq', wires=2, backend='ibmq_qasm_simulator', provider=provider)

If no provider is passed explicitly, then the official provider is used, with options of ``hub='ibm-q'``, ``group='open'`` and ``project='main'``.

Custom provider options can be passed as keyword arguments when creating a device:

.. code-block:: python

    import pennylane as qml
    dev = qml.device('qiskit.ibmq', wires=2, backend='ibmq_qasm_simulator', ibmqx_token='XXX', hub='MYHUB', group='MYGROUP', project='MYPROJECT')

More details on Qiskit providers can be found at the `IBMQ provider documentation <https://qiskit.org/documentation/apidoc/ibmq-provider.html>`_.
