# Release 0.10.0-dev

### New features since last release

### Breaking changes

### Improvements

### Documentation

### Bug fixes

### Contributors

This release contains contributions from (in alphabetical order):

---

# Release 0.9.0

### New features since last release

### Breaking changes

* Now supports Qiskit version 0.19.1. As a result of breaking changes
  within Qiskit, version 0.18 and below are no longer supported.
  [(#81)](https://github.com/XanaduAI/pennylane-qiskit/pull/81)
  [(#85)](https://github.com/XanaduAI/pennylane-qiskit/pull/85)
  [(#86)](https://github.com/XanaduAI/pennylane-qiskit/pull/86)

### Improvements

* Ported the ``QiskitDevice`` class to inherit from the ``QubitDevice`` class
  defined in PennyLane to use unified qubit operations and ease development.
  [(#83)](https://github.com/XanaduAI/pennylane-qiskit/pull/83)

* Added a test for returning probabilities when using the `IBMQDevice`.
  [(#82)](https://github.com/XanaduAI/pennylane-qiskit/pull/82)

### Documentation

* Major redesign of the documentation, making it easier to navigate.
  [(#78)](https://github.com/XanaduAI/pennylane-qiskit/pull/78)
  [(#79)](https://github.com/XanaduAI/pennylane-qiskit/pull/79)

### Bug fixes

* Added a type conversion of parameters for parametrized gates, and renamed
  various gates for Qiskit version 0.18.0 support.
  [(#81)](https://github.com/XanaduAI/pennylane-qiskit/pull/81)

* Renamed `QiskitDevice.probabilities` to `QiskitDevice.probability` to overload
  `pennylane.Device.probability`. This fixes a bug that raises `NotImplementedError`
  when a QNode is used to compute probabilities on a IBMQ device.
  [(#80)](https://github.com/XanaduAI/pennylane-qiskit/pull/80)


### Contributors

This release contains contributions from (in alphabetical order):

Rafael Haenel, Josh Izaac, Maria Schuld, Antal Száva

---

# Release 0.8.2

### Bug fixes

* Fixed a bug where users with `IBMQ` tokens linked to multiple
  providers would experience an error.
  [(#74)](https://github.com/XanaduAI/pennylane-qiskit/pull/74)

### Contributors

This release contains contributions from (in alphabetical order):

Antal Száva

---

# Release 0.8.1

### Bug fixes

* Fixed a bug where gradient computations always returned 0 when
  loading a parametrized Qiskit circuit as a PennyLane template.
  [(#71)](https://github.com/XanaduAI/pennylane-qiskit/pull/71)

### Contributors

This release contains contributions from (in alphabetical order):

Josh Izaac

---

# Release 0.8.0

### Bug fixes

* Removed v1 `IBMQ` credentials, disallowed `unitary_simulator` backend to
  have `memory=True` and discontinuing support for `QubitStateVector` on
  the `unitary_simulator` backend due to Qiskit's 0.14.0 version.
  [#65](https://github.com/XanaduAI/pennylane-qiskit/pull/65)

### Contributors

This release contains contributions from (in alphabetical order):

Antal Száva

---

# Release 0.7.1

### Bug fixes

* Set `analytic=False` as default, since by default warnings are raised
  for hardware simulators.
  [#64](https://github.com/XanaduAI/pennylane-qiskit/pull/64)

### Contributors

This release contains contributions from (in alphabetical order):

Antal Száva

---

# Release 0.7.0

### New features since last release

* Added the ability to automatically convert Qiskit `QuantumCircuits`
  or QASM circuits directly into PennyLane templates. The loaded
  operations can be used directly inside PennyLane circuits.
  [#55](https://github.com/XanaduAI/pennylane-qiskit/pull/55)

* Updated the list of operations such that each operation is now
  used from PennyLane. Added capability to specify the inverses
  of operations.
  [#58](https://github.com/XanaduAI/pennylane-qiskit/pull/58)

### Improvements

* Added integration tests for converting objects from Qiskit
  using PennyLane.
  [#57](https://github.com/XanaduAI/pennylane-qiskit/pull/57)

* Added warnings for hardware simulators using `analytic==True`
  when calculating expectations and variances.
  [#59](https://github.com/XanaduAI/pennylane-qiskit/pull/59)

### Bug fixes

* Removed `gates.py` including operations `Rot` and `BasisState`
  such that these operations are decomposed by PennyLane and no
  errors arise for the `BasisState` initialized with all-zero states.
  [#60](https://github.com/XanaduAI/pennylane-qiskit/pull/60)

### Contributors

This release contains contributions from (in alphabetical order):

Antal Száva

---

# Release 0.6.0

### New features since last release

* All Qiskit devices now support tensor observables using the
  `return expval(qml.PauliZ(0) @ qml.Hermitian(A, [1, 2]))`
  syntax introduced in PennyLane v0.6.

### Contributors

This release contains contributions from (in alphabetical order):

Josh Izaac

---

# Release 0.5.1

### Bug fixes

* Fixed a bug where backend keyword arguments, such as `backend_options`
  and `noise_model`, were being passed to backends that did not support it.
  [#51](https://github.com/XanaduAI/pennylane-qiskit/pull/51)

### Contributors

This release contains contributions from (in alphabetical order):

Josh Izaac

---

# Release 0.5.0

This is a significant release, bringing the plugin up to date with the latest
PennyLane and Qiskit features.

### New features since last release

* The plugin and the tests have been completely re-written from scratch, to ensure
  high quality and remove technical debt.
  [#44](https://github.com/XanaduAI/pennylane-qiskit/pull/44)
  [#43](https://github.com/XanaduAI/pennylane-qiskit/pull/43)

* Samples and variance support have been added to all devices.
  [#46](https://github.com/XanaduAI/pennylane-qiskit/pull/46)

* Multi-qubit hermitian observables are now supported, due to
  support being added in Qiskit version 0.12.

* Support has been added for IBM Q Experience 2.0.
  [#44](https://github.com/XanaduAI/pennylane-qiskit/pull/44)

### Improvements

* Hardware and software devices are now treated identically, with expectations,
  variance, and samples computed via the _probability_, not the amplitudes.
  This has several consequences:

  - It makes the code cleaner and simpler, as there is now one defined
    way of computing statistics.

  - It is faster in most cases, as this does not require computing
    large matrix-vector products, or Kronecker products. Instead,
    eigenvalues of single- and multi-qubit observables are computed using
    dyanamic programming.

  - It reduces the number of tests required.

* Test suite now includes tests against the IBM Q Experience, in addition
  to the local simulators.

### Bug fixes

* Due to the move to IBM Q 2.0 credentials, users remaining with IBM Q v1.0
  now must pass an additional URL argument to the service. The plugin
  has been modified to allow the user to pass this argument if required.
  [#44](https://github.com/XanaduAI/pennylane-qiskit/pull/44)

### Contributors
This release contains contributions from:

Shahnawaz Ahmed, Josh Izaac

---

# Release 0.0.8

### New features since last release

- Added noise model and backend options to the devices as well as new observables.
- Added support for all PennyLane observables for calculating expectation values.
- Added (copied & adjusted from pennylane-forest) test for the expectation values.
- Added the necessary DefaultQubit device for comparison.
- Added logging.

### Improvements

- Changed expval_queue to obs_queue and expectations to observables as per latest pennylane.
- Reversed qregs to match the default qubit device behavior.
- Renamed devices correctly.
- Made wires explicit. If num_wires of operation is 0 then use the whole system as wires!
- Renamed the IBMQ device from `qiskit.ibm` to `qiskit.ibmq`.

### Fixed

- Fixed the Unitary gate.
- Fixed the token loading and the shots.
- Fixed and updated to qiskit v0.10.1
- Fixed the valid expectation values of all devices. Along with it tests where fixed.

### Removed

- Removed the IBMQX_TOKEN import and replace with the correct args.
- Removed the unconditional make coverage.
- Removed default qubit device, this is not tested in this package!

---

# Release 0.0.7

### Removed

- Removed the `LegacySimulatorsQiskitDevice` as this isn't supported in `qiskit` anymore.

### Improvements

- updated to `qiskit` version `0.10.*`.

---

# Release 0.0.6

### Added

- Added the noise model to `BasicAerQiskitDevice` and `AerQiskitDevice` with unit tests (#13)
- Added unit tests for the docs (#7/#20)

### Fixed

- Fixed the ''multiple'' `backends` keyword bug (#13).
- Fixed the `par_domain` of the operations (real valued parameters) in `pos.py` (#21)
- Fixed documentation problems
- Started to 'fix' ''low'' code quality due to type hints with codacy (#15)
- Small typos and code cleanups

---

# Release 0.0.5

### New features

- `AerQiskitDevice` and `LegacySimulatorQiskitDevice` with tests

### Fixed

- Readme & documentation
- `setup.py`
- Code style clean-ups & code cleaning.
- Build setup with travis CI
- Removed unsied `num_runs` kwargs
- Fixed the overlapping kwargs with
- Better tests taken from the _ProjectQ_ plugin
- remove the sleep after computation.

---

# Release 2018-12-23

### New features

- The device `IbmQQiskitDevice` now can load the `ibmqx_token` from the environment variabe `IBMQX_TOKEN`

### Improvements

- Renamed `psops` to `qisktops`
- The setup/build requirements are now read from `requirements.txt`. The file is included now in the distribution file.
- Due to changes in qiskit (0.7.0) the Device `AerDevice` must be correctly termed to `BasicAerDevice`
- Due to the removal of `qiskit.unroll.CircuitBackend` the complete plugin logic needed an overhaul. Now the
  `QuantumCircuit` is used in conjunction with the new converters `dag_to_circuit` and `circuit_to_dag`
  as well as the usage of the base class `Instruction`. We hope that this change will work for most cases.

### Fixed

- The Readme: links and the usage of _device_ instead of _provider_, to stay in PennyLane-lingo
- Update to qiskit 0.7.0 made changes necessary: import location have changes
