PennyLane qiskit Plugin
#########################

.. image:: https://img.shields.io/travis/com/carstenblank/pennylane-qiskit/master.svg?style=for-the-badge
    :alt: Travis
    :target: https://travis-ci.com/carstenblank/pennylane-qiskit

.. image:: https://img.shields.io/codecov/c/github/carstenblank/pennylane-qiskit/master.svg?style=for-the-badge
    :alt: Codecov coverage
    :target: https://codecov.io/gh/carstenblank/pennylane-qiskit

.. image:: https://img.shields.io/codacy/grade/f4132f03ce224f82bd3e8ba436b52af3.svg?style=for-the-badge
    :alt: Codacy grade
    :target: https://www.codacy.com/app/carstenblank/pennylane-qiskit?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=carstenblank/pennylane-qiskit&amp;utm_campaign=Badge_Grade

.. image:: https://img.shields.io/readthedocs/pennylane-qiskit.svg?style=for-the-badge
    :alt: Read the Docs
    :target: https://pennylane-qiskit.readthedocs.io

.. image:: https://img.shields.io/pypi/v/PennyLane-qiskit.svg?style=for-the-badge
    :alt: PyPI
    :target: https://pypi.org/project/PennyLane-qiskit

.. image:: https://img.shields.io/pypi/pyversions/PennyLane-qiskit.svg?style=for-the-badge
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/PennyLane-qiskit

.. header-start-inclusion-marker-do-not-remove

`PennyLane <https://pennylane.readthedocs.io>`_ is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

`qiskit <https://qiskit.org/documentation/>`_ is an open-source compilation framework capable of targeting various
types of hardware and a high-performance quantum computer simulator with emulation capabilities, and various
compiler plug-ins.

This PennyLane plugin allows to use both the software and hardware backends of qiskit as devices for PennyLane.


Features
========

* Provides two providers to be used with PennyLane: ``qiskit.basicaer`` and ``qiskit.ibm``. These provide access to the respective qiskit backends.

* Supports a wide range of PennyLane operations and expectation values across the providers.

* Combine qiskit high performance simulator and hardware backend support with PennyLane's automatic differentiation and optimization.

.. header-end-inclusion-marker-do-not-remove
.. installation-start-inclusion-marker-do-not-remove

Installation
============

This plugin requires Python version 3.5 and above, as well as PennyLane and qiskit.
Installation of this plugin, as well as all dependencies, can be done using pip:

.. code-block:: bash

    $ python -m pip install pennylane_qiskit

To test that the PennyLane qiskit plugin is working correctly you can run

.. code-block:: bash

    $ make test

in the source folder. Tests restricted to a specific provider can be run by executing :code:`make test-aer` or :code:`make test-ibm`.

.. note::
    Tests on the `ibm provider <https://pennylane-qiskit.readthedocs.io/en/latest/devices.html>`_ can
    only be run if a :code:`ibmqx_token` for the `IBM Q experience <https://quantumexperience.ng.bluemix.net/qx/experience>`_ is
    configured in the `PennyLane configuration file <https://pennylane.readthedocs.io/configuration.html>`_.
    If this is the case, running :code:`make test` also executes tests on the :code:`ibm` provider. By default tests on
    the :code:`ibm` provider run with :code:`ibmq_qasm_simulator` backend and those done by the :code:`basicaer` provider are
    run with the :code:`qasm_simulator` backend. At the time of writing this means that the test are "free".
    Please verify that this is also the case for your account.
.. installation-end-inclusion-marker-do-not-remove
.. gettingstarted-start-inclusion-marker-do-not-remove

Getting started
===============

You can instantiate a :code:`'qiskit.basicaer'` device for PennyLane with:

.. code-block:: python

    import pennylane as qml
    dev = qml.device('qiskit.basicaer', wires=2)

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
        return qml.expval.PauliZ(wires=1)

You can then execute the circuit like any other function to get the quantum mechanical expectation value.

.. code-block:: python

    circuit(0.2, 0.1, 0.3)

You can also change the default device's backend with

.. code-block:: python

    dev = qml.device('qiskit.basicaer', wires=2, backend='unitary_simulator')

To get a current overview what backends are available you can query this by

.. code-block:: python

    dev.capabilities()['backend']

Running your code on an IBM Quantum Experience simulator or even a real hardware chip is just as easy. Instead of the
device above, you would instantiate a :code:`'qiskit.ibm'` device by giving your IBM Quantum Experience token:

.. code-block:: python

    import pennylane as qml
    dev = qml.device('qiskit.ibm', wires=2, ibmqx_token="XXX")

In order to avoid accidentally publishing your token, you should better specify it via the `PennyLane configuration file <https://pennylane.readthedocs.io/en/latest/code/configuration.html>`__ by adding a section such as

.. code::

  [qiskit.global]

    [qiskit.ibm]
    ibmqx_token = "XXX"

It is also possible to define an environment variable :code:`IBMQX_TOKEN`, from which the token will be taken if not provided in another way.

Per default the backend :code:`ibm` uses the simulator backend :code:`ibmq_qasm_simulator`, but you can change that
to be any of the real backends as given by

.. code-block:: python

    dev.capabilities()['backend']

.. gettingstarted-end-inclusion-marker-do-not-remove

Please refer to the `documentation of the PennyLane qiskit Plugin <https://pennylane-qiskit.readthedocs.io/>`_ as
well as well as to the `documentation of PennyLane <https://pennylane.readthedocs.io/>`_ for further reference.

.. howtocite-start-inclusion-marker-do-not-remove

How to cite
===========

If you are doing research using PennyLane, please cite `our whitepaper <https://arxiv.org/abs/1811.04968>`_:

  Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, and Nathan Killoran. PennyLane. *arXiv*, 2018. arXiv:1811.04968

.. howtocite-end-inclusion-marker-do-not-remove

Contributing
============

We welcome contributions - simply fork the repository of this plugin, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.
All contributers to this plugin will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects or applications built on PennyLane.


Authors
=======

Carsten Blank, Sebastian Boerakker, Christian Gogolin, Josh Izaac

.. support-start-inclusion-marker-do-not-remove

Support
=======

- **Source Code:** https://github.com/carstenblank/pennylane-qiskit
- **Issue Tracker:** https://github.com/carstenblank/pennylane-qiskit/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.

.. support-end-inclusion-marker-do-not-remove
.. license-start-inclusion-marker-do-not-remove

License
=======

The PennyLane qiskit plugin is **free** and **open source**, released under
the `Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.

.. license-end-inclusion-marker-do-not-remove
