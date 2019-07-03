
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
==============


`0.0.7`_ - `0.0.8`_
====================

Added
------

- Added noise model and backend options to the devices as well as new observables.
- Added support for all PennyLane observables for calculating expectation values.
- Added (copied & adjusted from pennylane-forest) test for the expectation values.
- Added the necessary DefaultQubit device for comparison.
- Added logging.

Changed
--------

- Changed expval_queue to obs_queue and expectations to observables as per latest pennylane.
- Reversed qregs to match the default qubit device behavior.
- Renamed devices correctly.
- Made wires explicit. If num_wires of operation is 0 then use the whole system as wires!
- Renamed the IBMQ device from :code:`qiskit.ibm` to :code:`qiskit.ibmq`.

Fixed
------

- Fixed the Unitary gate.
- Fixed the token loading and the shots.
- Fixed and updated to qiskit v0.10.1
- Fixed the valid expectation values of all devices. Along with it tests where fixed.


Removed
--------

- Removed the IBMQX_TOKEN import and replace with the correct args.
- Removed the unconditional make coverage.
- Removed default qubit device, this is not tested in this package!


`0.0.6`_ - `0.0.7`_
====================

Removed
--------

- Removed the `LegacySimulatorsQiskitDevice` as this isn't supported in `qiskit` anymore.

Changed
--------

- updated to `qiskit` version `0.10.*`.


`0.0.5`_ - `0.0.6`_
====================

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

`2018-12-23`_ - `0.0.5`_
=========================

Fixed
------

- Readme & documentation
- :code:`setup.py`
- Code style clean-ups & code cleaning.
- Build setup with travis CI
- Removed unsied `num_runs` kwargs
- Fixed the overlapping kwargs with
- Better tests taken from the _ProjectQ_ plugin
- remove the sleep after computation.

Added
------

- `AerQiskitDevice` and `LegacySimulatorQiskitDevice` with tests


`0.0.2`_ - `2018-12-23`_
=========================

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


.. _UNRELEASED: https://github.com/carstenblank/pennylane-qiskit/compare/0.0.7...HEAD
.. _0.0.2: https://github.com/carstenblank/pennylane-qiskit/compare/0.0.1...0.0.2
.. _2018-12-23: https://github.com/carstenblank/pennylane-qiskit/compare/0.0.2...3b4ef02b5f3518a983350866048562b4a1f51832
.. _0.0.5: https://github.com/carstenblank/pennylane-qiskit/compare/3b4ef02b5f3518a983350866048562b4a1f51832...0.0.5
.. _0.0.6: https://github.com/carstenblank/pennylane-qiskit/compare/0.0.5...0.0.6
.. _0.0.7: https://github.com/carstenblank/pennylane-qiskit/compare/0.0.6...0.0.7
.. _Keep a Changelog: http://keepachangelog.com/en/1.0.0/
