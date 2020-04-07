IBM Q Experience
================

PennyLane-Qiskit supports running PennyLane on IBM Q hardware via the ``qistkit.ibmq`` device.
You can choose between different backends - either simulators tailor-made to emulate the real hardware,
or the real hardware itself.

Accounts and Tokens
~~~~~~~~~~~~~~~~~~~

By default, the ``qiskit.ibmq`` device will attempt to use an already active or stored
IBM Q account. If the device finds no account it will raise an error:

.. code::

    'No active IBM Q account, and no IBM Q token provided.

You can use the ``qiskit.IBMQ.save_account("<my_token>")`` function to permanently store an account,
and the ``qiskit.IBMQ.load_account()`` function to load the stored account in a given session.
Alternatively, you can specify the token with PennyLane via the
`PennyLane configuration file <https://pennylane.readthedocs.io/en/latest/introduction/configuration.html>`__ by
adding the section

.. code::

  [qiskit.global]

    [qiskit.ibmq]
    ibmqx_token = "XXX"

You may also directly pass your IBM Q API token to the device:

.. code-block:: python

    dev = qml.device('qiskit.ibmq', wires=2, backend='ibmq_qasm_simulator', ibmqx_token="XXX")


.. warning:: Never publish code containing your token online.

Backends
~~~~~~~~

By default, the ``qiskit.ibmq`` device uses the simulator backend
``ibmq_qasm_simulator``, but this may be changed to any of the real backends as returned by

.. code-block:: python

    dev.capabilities()['backend']

Most of the backends of the ``qiskit.ibmq`` device, such as ``ibmq_london`` or ``ibmq_16_melbourne``,
are *hardware backends*. Running PennyLane with these backends means to send the circuit as a job to the actual quantum
computer and retrieve the results via the cloud.

Specifying providers
~~~~~~~~~~~~~~~~~~~~

Custom providers can be passed as arguments when a ``qiskit.ibmq`` device is created:

.. code-block:: python

    from qiskit import IBMQ
    provider = IBMQ.enable_account('XYZ')

    import pennylane as qml
    dev = qml.device('qiskit.ibmq', wires=2, backend='ibmq_qasm_simulator', provider=provider)

If no provider is passed explicitly, then the official provider options are used,
``hub='ibm-q'``, ``group='open'`` and ``project='main'``.

Custom provider options can also be passed as keyword arguments when creating a device:

.. code-block:: python

    import pennylane as qml
    dev = qml.device('qiskit.ibmq', wires=2, backend='ibmq_qasm_simulator',
                     ibmqx_token='XXX', hub='MYHUB', group='MYGROUP', project='MYPROJECT')

More details on Qiskit providers can be found
in the `IBMQ provider documentation <https://qiskit.org/documentation/apidoc/ibmq-provider.html>`_.
