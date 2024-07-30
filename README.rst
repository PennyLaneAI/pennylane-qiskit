PennyLane-Qiskit Plugin
#######################

.. image:: https://img.shields.io/github/actions/workflow/status/PennyLaneAI/pennylane-qiskit/tests.yml?branch=master&logo=github&style=flat-square
    :alt: GitHub Workflow Status (branch)
    :target: https://github.com/PennyLaneAI/pennylane-qiskit/actions?query=workflow%3ATests

.. image:: https://img.shields.io/codecov/c/github/PennyLaneAI/pennylane-qiskit/master.svg?logo=codecov&style=flat-square
    :alt: Codecov coverage
    :target: https://codecov.io/gh/PennyLaneAI/pennylane-qiskit

.. image:: https://img.shields.io/codefactor/grade/github/PennyLaneAI/pennylane-qiskit/master?logo=codefactor&style=flat-square
    :alt: CodeFactor Grade
    :target: https://www.codefactor.io/repository/github/pennylaneai/pennylane-qiskit

.. image:: https://readthedocs.com/projects/xanaduai-pennylane-qiskit/badge/?version=latest&style=flat-square
    :alt: Read the Docs
    :target: https://docs.pennylane.ai/projects/qiskit

.. image:: https://img.shields.io/pypi/v/PennyLane-qiskit.svg?style=flat-square
    :alt: PyPI
    :target: https://pypi.org/project/PennyLane-qiskit

.. image:: https://img.shields.io/pypi/pyversions/PennyLane-qiskit.svg?style=flat-square
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/PennyLane-qiskit

.. header-start-inclusion-marker-do-not-remove

The PennyLane-Qiskit plugin integrates the Qiskit quantum computing framework with PennyLane's
quantum machine learning capabilities.

`PennyLane <https://pennylane.readthedocs.io>`_ is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

`Qiskit <https://qiskit.org/documentation/>`_ is an open-source framework for quantum computing.

.. header-end-inclusion-marker-do-not-remove

Features
========

* Provides three devices to be used with PennyLane: ``qiskit.aer``, ``qiskit.basicaer`` and ``qiskit.ibmq``.
  These devices provide access to the various backends, including the IBM hardware accessible through the cloud.

* Supports a wide range of PennyLane operations and expectation values across the providers.

* Combine Qiskit's high performance simulator and hardware backend support with PennyLane's automatic
  differentiation and optimization.

.. installation-start-inclusion-marker-do-not-remove

Installation
============

This plugin requires Python version 3.9 and above, as well as PennyLane and Qiskit.
Installation of this plugin, as well as all dependencies, can be done using ``pip``:

.. code-block:: bash

    pip install pennylane-qiskit

To test that the PennyLane-Qiskit plugin is working correctly you can run

.. code-block:: bash

    make test

in the source folder.

.. warning::
    
    When installing the Pennylane-Qiskit plugin, we recommend starting with a clean environment.
    This is especially pertinent when upgrading from a pre-1.0 version of Qiskit, as described
    in `Qiskit's migration guide <https://docs.quantum.ibm.com/api/migration-guides/qiskit-1.0-installation>`_. 

.. installation-end-inclusion-marker-do-not-remove

Please refer to the `plugin documentation <https://pennylaneqiskit.readthedocs.io/>`_ as
well as to the `PennyLane documentation <https://pennylane.readthedocs.io/>`_ for further reference.

Contributing
============

We welcome contributions - simply fork the repository of this plugin, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.
All contributers to this plugin will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects
or applications built on PennyLane.

Authors
=======

PennyLane-Qiskit is the work of `many contributors <https://github.com/PennyLaneAI/pennylane-qiskit/graphs/contributors>`_.

If you are doing research using PennyLane and PennyLane-Qiskit, please cite `our paper <https://arxiv.org/abs/1811.04968>`_:

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, M. Sohaib Alam, Shahnawaz Ahmed,
    Juan Miguel Arrazola, Carsten Blank, Alain Delgado, Soran Jahangiri, Keri McKiernan, Johannes Jakob Meyer,
    Zeyue Niu, Antal Száva, and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018. arXiv:1811.04968

.. support-start-inclusion-marker-do-not-remove

Support
=======

- **Source Code:** https://github.com/PennyLaneAI/pennylane-qiskit
- **Issue Tracker:** https://github.com/PennyLaneAI/pennylane-qiskit/issues
- **PennyLane Forum:** https://discuss.pennylane.ai

If you are having issues, please let us know by posting the issue on our Github issue tracker, or
by asking a question in the forum.

.. support-end-inclusion-marker-do-not-remove
.. license-start-inclusion-marker-do-not-remove

License
=======

The PennyLane qiskit plugin is **free** and **open source**, released under
the `Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.

.. license-end-inclusion-marker-do-not-remove
