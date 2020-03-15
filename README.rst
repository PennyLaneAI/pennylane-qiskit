PennyLane Qiskit Plugin
#######################

.. image:: https://img.shields.io/travis/com/XanaduAI/pennylane-qiskit/master.svg?style=popout-square
    :alt: Travis
    :target: https://travis-ci.com/XanaduAI/pennylane-qiskit

.. image:: https://img.shields.io/codecov/c/github/XanaduAI/pennylane-qiskit/master.svg?style=popout-square
    :alt: Codecov coverage
    :target: https://codecov.io/gh/XanaduAI/pennylane-qiskit

.. image:: https://img.shields.io/codacy/grade/f4132f03ce224f82bd3e8ba436b52af3.svg?style=popout-square
    :alt: Codacy grade
    :target: https://www.codacy.com/app/XanaduAI/pennylane-qiskit

.. image:: https://img.shields.io/readthedocs/pennylane-qiskit.svg?style=popout-square
    :alt: Read the Docs
    :target: https://pennylane-qiskit.readthedocs.io

.. image:: https://img.shields.io/pypi/v/PennyLane-qiskit.svg?style=popout-square
    :alt: PyPI
    :target: https://pypi.org/project/PennyLane-qiskit

.. image:: https://img.shields.io/pypi/pyversions/PennyLane-qiskit.svg?style=popout-square
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/PennyLane-qiskit

.. header-start-inclusion-marker-do-not-remove

`PennyLane <https://pennylane.readthedocs.io>`_ is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

`Qiskit <https://qiskit.org/documentation/>`_ is an open-source compilation framework capable of targeting various
types of hardware and a high-performance quantum computer simulator with emulation capabilities, and various
compiler plug-ins.

This PennyLane plugin allows to use both the software and hardware backends of Qiskit as devices for PennyLane.


Features
========

* Provides three devices to be used with PennyLane: ``qiskit.basicaer``, ``qiskit.aer`` and ``qiskit.ibmq``.
  These devices provide access to the various backends.

* Supports a wide range of PennyLane operations and expectation values across the providers.

* Combine Qiskit's high performance simulator and hardware backend support with PennyLane's automatic
  differentiation and optimization.

.. header-end-inclusion-marker-do-not-remove
.. installation-start-inclusion-marker-do-not-remove

Installation
============

This plugin requires Python version 3.5 and above, as well as PennyLane and Qiskit.
Installation of this plugin, as well as all dependencies, can be done using ``pip``:

.. code-block:: bash

    pip install pennylane-qiskit

To test that the PennyLane Qiskit plugin is working correctly you can run

.. code-block:: bash

    $ make test

in the source folder. Tests restricted to a specific provider can be run by executing
``make test-basicaer``, ``make test-aer`` ``make test-ibmq``.

.. note::

    Tests on the `IBMQ device <https://pennylane-qiskit.readthedocs.io/en/latest/code/ibmq.html>`_ can
    only be run if a ``ibmqx_token`` for the
    `IBM Q experience <https://quantum-computing.ibm.com/>`_ is
    configured in the `PennyLane configuration file
    <https://pennylane.readthedocs.io/en/latest/introduction/configuration.html>`_.

    If this is the case, running ``make test`` also executes tests on the ``ibmq`` device.
    By default tests on the ``ibmq`` device run with ``ibmq_qasm_simulator`` backend
    and those done by the ``basicaer`` and ``aer`` device are run with the ``qasm_simulator``
    backend. At the time of writing this means that the test are "free".
    Please verify that this is also the case for your account.

.. installation-end-inclusion-marker-do-not-remove
.. gettingstarted-start-inclusion-marker-do-not-remove

Getting started
===============

Once the PennyLane-Qiskit plugin is installed, the three provided Qiskit devices
can be accessed straightaway in PennyLane.

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

You can also change the default device's backend with

.. code-block:: python

    dev = qml.device('qiskit.aer', wires=2, backend='unitary_simulator')

To get a current overview what backends are available you can query this by

.. code-block:: python

    dev.capabilities()['backend']

While the device ``'qiskit.aer'`` is the standard go-to simulator that is provided along
the Qiskit main package installation, there exists a natively included python simulator
that is slower but will work usually without the need to check out other dependencies
(gcc, blas and so on) which can be used by ``'qiskit.basicaer'``.

Another important difference between the two is that while ``'qiskit.aer'``
supports a simulation with noise, ``'qiskit.basicaer'`` does not.

Noise models
============

You can instantiate a noise model and apply it to the device by calling

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
`noise models <https://qiskit.org/aer>`_.

IBM Q Experience
================

PennyLane-Qiskit supports running PennyLane on IBM Q hardware via the ``qistkit.ibmq`` device.
You can choose between different backends - either simulators or real hardware.

.. code-block:: python

    import pennylane as qml
    dev = qml.device('qiskit.ibmq', wires=2, backend='ibmq_16_melbourne')

By default, the ``qiskit.ibmq`` device will attempt to use an already active or stored
IBM Q account. If none are available, you may also directly pass your IBM Q API token,
as well as an optional URL:

.. code-block:: python

    import pennylane as qml
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

.. gettingstarted-end-inclusion-marker-do-not-remove

Please refer to the `plugin documentation <https://pennylane-qiskit.readthedocs.io/>`_ as
well as to the `PennyLane documentation <https://pennylane.readthedocs.io/>`_ for further reference.

.. howtocite-start-inclusion-marker-do-not-remove

How to cite
===========

If you are doing research using PennyLane, please cite `our whitepaper <https://arxiv.org/abs/1811.04968>`_:

  Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, Carsten Blank, Keri McKiernan and Nathan Killoran.
  PennyLane. *arXiv*, 2018. arXiv:1811.04968

.. howtocite-end-inclusion-marker-do-not-remove

Contributing
============

We welcome contributions - simply fork the repository of this plugin, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.
All contributers to this plugin will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects
or applications built on PennyLane.


Authors
=======

Shahnawaz Ahmed, Carsten Blank, Sebastian Boerakker, Christian Gogolin, Josh Izaac.

.. support-start-inclusion-marker-do-not-remove

Support
=======

- **Source Code:** https://github.com/XanaduAI/pennylane-qiskit
- **Issue Tracker:** https://github.com/XanaduAI/pennylane-qiskit/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.

.. support-end-inclusion-marker-do-not-remove
.. license-start-inclusion-marker-do-not-remove

License
=======

The PennyLane qiskit plugin is **free** and **open source**, released under
the `Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.

.. license-end-inclusion-marker-do-not-remove
