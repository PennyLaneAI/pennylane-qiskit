
*********
Changelog
*********

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_.

  **Types of changes:**

  - **Added**: for new features.
  - **Changed**: for changes in existing functionality.
  - **Deprecated**: for soon-to-be removed features.
  - **Removed**: for now removed features.
  - **Fixed**: for any bug fixes.
  - **Security**: in case of vulnerabilities.

`UNRELEASED`_
=============


`0.0.5`_ - `0.0.6`
===================

Added
------

- Added the noise model to ``BasicAerQiskitDevice`` and ``AerQiskitDevice`` with unit tests (#13)
- Added unit tests for the docs (#7/#20)

Fixed
------

- Fixed the ''multiple'' ``backends`` keyword bug (#13).
- Fixed the ``par_domain`` of the operations (real valued parameters) in ``pos.py`` (#21)
- Fixed documentation problems
- Started to 'fix' ''low'' code quality due to type hints with codacy (#15)
- Small typos and code cleanups

`0.0.2`_ - 2018-12-23
======================

Added
------

- The device ``IbmQQiskitDevice`` now can load the ``ibmqx_token`` from the environment variabe ``IBMQX_TOKEN``

Changed
--------

- Renamed ``psops`` to ``qisktops``
- The setup/build requirements are now read from ``requirements.txt``. The file is included now in the distribution file.
- Due to changes in qiskit (0.7.0) the Device ``AerDevice`` must be correctly termed to ``BasicAerDevice``
- Due to the removal of ``qiskit.unroll.CircuitBackend`` the complete plugin logic needed an overhaul. Now the
    ``QuantumCircuit`` is used in conjunction with the new converters ``dag_to_circuit`` and ``circuit_to_dag``
    as well as the usage of the base class ``Instruction``. We hope that this change will work for most cases.


Fixed
------

- The Readme: links and the usage of _device_ instead of _provider_, to stay in PennyLane-lingo
- Update to qiskit 0.7.0 made changes necessary: import location have changes


.. _UNRELEASED: https://github.com/carstenblank/pennylane-qiskit/compare/0.0.2...HEAD
.. _0.0.2: https://github.com/carstenblank/pennylane-qiskit/compare/0.0.1...0.0.2
.. _Keep a Changelog: http://keepachangelog.com/en/1.0.0/
