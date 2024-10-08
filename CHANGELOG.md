# Release 0.39.0-dev

### New features since last release

### Improvements 🛠

### Breaking changes 💔

### Deprecations 👋

### Documentation 📝

### Bug fixes 🐛

### Contributors ✍️

This release contains contributions from (in alphabetical order):

---
# Release 0.38.1

### Bug fixes 🐛

* Due to the removal of the `Session` and `Backend` keywords in the 0.30 release of `qiskit-ibm-runtime`, the PennyLane-Qiskit
  plugin now pins to `qiskit-ibm-runtime<=0.29`.
  [(#587)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/587)

### Contributors ✍️

This release contains contributions from (in alphabetical order):

Austin Huang
Mudit Pandey

---
# Release 0.38.0

### New features since last release

* Added support for converting Qiskit noise models to
  PennyLane ``NoiseModels`` using ``load_noise_model``.
  [(#577)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/577)
  [(#578)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/578)

* Qiskit Sessions can now be used for the ``qiskit.remote`` device with the ``qiskit_session`` context
  manager.
  [(#551)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/551)

### Improvements 🛠

* Qiskit Runtime Primitives are supported by the ``qiskit.remote`` device. Circuits ran using the ``qiskit.remote``
  device will automatically call the SamplerV2 and EstimatorV2 primitives appropriately. Additionally, runtime options can be passed as keyword arguments directly to the ``qiskit.remote`` device.
  [(#513)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/513)

### Breaking changes 💔

* Support has been removed for Qiskit versions below 0.46. The minimum required version for Qiskit is now 1.0. 
  If you want to continue to use older versions of Qiskit with the plugin, please use version 0.36 of 
  the Pennylane-Qiskit plugin. 
  [(#536)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/536)

* The test suite no longer runs for Qiskit versions below 0.46.
  [(#536)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/536)

* The ``qiskit.basicaer`` device has been removed because it is not supported for versions of Qiskit above 0.46.
  [(#546)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/546)

* The IBM quantum devices, ``qiskit.ibmq``, ``qiskit.ibmq.circuit_runner`` and ``qiskit.ibmq.sampler``, have been removed due to deprecations of the IBMProvider and the cloud simulator "ibmq_qasm_simulator".
  [(#550)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/550)

### Documentation 📝

* The Pennylane-Qiskit plugin page has been updated to reflect the changes in both the plugin's 
capabilities and Qiskit.
  [#563](https://github.com/PennyLaneAI/pennylane-qiskit/pull/563)

### Contributors ✍️

This release contains contributions from (in alphabetical order):

Utkarsh Azad
Lillian M. A. Frederiksen
Austin Huang

---
# Release 0.37.0

### Improvements 🛠

* Updated `load_qasm` to take the optional kwarg `measurements` which get performed at the end of the loaded circuit and `load_qasm` can now detect mid-circuit measurements from `qasm`.
[(#555)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/555)

* Improvements have been made to load circuits with `SwitchCaseOp` gates with default case.
  [(#514)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/514)

### Contributors ✍️

This release contains contributions from (in alphabetical order):
Utkarsh Azad
Lillian M. A. Frederiksen
Austin Huang
Mashhood Khan

---
# Release 0.36.0

### New features since last release

* Support is added for using the plugin devices with Qiskit 1.0. As the backend provider ``qiskit.BasicAer`` 
  is no longer supported by Qiskit in 1.0, this added support does not extend to the ``"qiskit.aer"`` device. 
  Instead, a ``"qiskit.basicsim"`` device is added, with the new Qiskit implementation of a Python simulator 
  device, ``BasicSimulator``, as the backend.
  [(#493)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/493)

* Backwards compatibility with Qiskit BackendV2 has now been implemented. Previously, only backends of type
  BackendV1 were supported but now users can choose to use BackendV2 as well.
  [(#514)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/514)

### Improvements 🛠

* Following updates to allow device compatibility with Qiskit 1.0, the version of `qiskit-ibm-runtime` is 
  no longer capped.
  [(#508)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/508)

* The test suite now runs with the most recent `qiskit` and `qiskit-ibm-runtime`, and well as with 
  `'qiskit==0.45'` and `qiskit-ibm-runtime<0.21` to monitor backward-compatibility.
  [(#508)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/508)

### Contributors ✍️

This release contains contributions from (in alphabetical order):
Lillian M. A. Frederiksen
Austin Huang

---
# Release 0.35.1

### Bug fixes 🐛

* Following the 0.21 release of `qiskit-ibm-runtime`, which requires Qiskit 1.0, the PennyLane-Qiskit plugin pins to 
  `qiskit-ibm-runtime<0.21`. This prevents `pip install pennylane-qiskit` from installing Qiskit 1.0 (via the requirements 
  of `qiskit-ibm-runtime`), which will break any environments that already have a 0.X.X version of Qiskit installed.
  [(#486)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/486)

### Contributors ✍️
Lillian Frederiksen

---
# Release 0.35.0

### Improvements 🛠

* The UI for passing parameters to a `qfunc` generated when loading a Qiskit `QuantumCircuit`
  into PennyLane is updated to allow passing parameters as args or kwargs, rather than as
  a dictionary. The old dictionary UI continues to be supported.
  [(#406)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/406)
  [(#428)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/428)

* Measurement operations are now added to the PennyLane template when a `QuantumCircuit`
  is converted using `load`. Additionally, one can override any existing terminal
  measurements by providing a list of PennyLane
  `measurements <https://docs.pennylane.ai/en/stable/introduction/measurements.html>`_ themselves.
  [(#405)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/405)
  [(#466)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/466)
  [(#467)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/467)

* Added the support for converting conditional operations based on mid-circuit measurements and
  two of the `ControlFlowOp` operations - `IfElseOp` and `SwitchCaseOp` when converting
  a `QuantumCircuit` using `load`.
  [(#417)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/417)
  [(#465)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/465)

* Qiskit's classical `Expr` conditionals can also be used with the supported
  `ControlFlowOp` operations.
  [(#432)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/432)

* Added conversion support for more Qiskit gates to native PennyLane operations -
  `Barrier`, `CYGate`, `CHGate`, `CPhase`, `CCZGate`, `ECRGate`, and `GlobalPhaseGate`.
  [(#449)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/449)

* Added the ability to convert a Qiskit `SparsePauliOp` instance into a PennyLane `Operator`.
  [(#401)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/401)
  [(#453)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/453)

* Added a `pennylane.io` entry point for converting Qiskit operators.
  [(#453)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/453)

* Unused parameters are now ignored when a `QuantumCircuit` is converted using `load`.
  [(#454)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/454)

### Bug fixes 🐛

* `QiskitDevice.batch_execute()` now gracefully handles empty lists of circuits.
  [(#459)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/459)

* It is now possible to compute the gradient of a circuit with `ParameterVector` elements.
  [(#458)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/458)

### Contributors ✍️

This release contains contributions from (in alphabetical order):

Mikhail Andrenkov
Utkarsh Azad
Lillian Frederiksen

---
# Release 0.34.0

### Bug fixes 🐛

* The kwargs `job_tags` and `session_id` are passed to the correct arguments in the
  `circuit_runner` device so that they will be used in the Qiskit backend; these
  were previously ignored.
  [(#358)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/358)

* The `generate_samples` method for the `IBMQSamplerDevice` is updated to get counts
  from the nearest probability distribution rather than the quasi-distribution (which
  may contain negative probabilities and therefore raise errors).
  [(#357)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/357)

* The `generate_samples` method for the `IBMQSamplerDevice` now avoids raising an
  indexing error when some states are not populated, and labels states according to
  the Pennylane convention instead of Qiskit convention.
  [(#357)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/357)


### Contributors ✍️

This release contains contributions from (in alphabetical order):

Lillian Frederiksen
Francesco Scala


---
# Release 0.33.1

### Improvements 🛠

* Stop using the now-deprecated `tape.is_sampled` property.
  [(#348)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/348)

### Bug fixes 🐛

* Update conversion of PennyLane to Qiskit operators to accommodate
  the addition of Singleton classes in the newest version of Qiskit.
  [(#347)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/347)

### Contributors ✍️

This release contains contributions from (in alphabetical order):

Lillian Frederiksen,
Matthew Silverman

---
# Release 0.33.0

### Improvements 🛠

* Logic updated to support Aer V2 device naming conventions.
  [(#343)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/343)

### Breaking changes 💔

* The old return type system has been removed.
  [(#331)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/331)

### Contributors ✍️

This release contains contributions from (in alphabetical order):

Mudit Pandey,
Matthew Silverman

---
# Release 0.32.0

### Improvements 🛠

* Added support for `qml.StatePrep` as a state preparation operation.
  [(#326)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/326)

### Breaking changes 💔

* Support for Python 3.8 has been removed.
  [(#328)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/328)

### Contributors ✍️

This release contains contributions from (in alphabetical order):

Mudit Pandey,
Jay Soni,

---
# Release 0.31.0

### New features since last release

* Added a `RemoteDevice` (PennyLane device name: `qiskit.remote`) that accepts a backend
  instance directly. [(#304)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/304)

### Breaking changes

* The `vqe_runner` has been removed, as the Qiskit Runtime VQE program has been retired.
  [(#313)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/313)

### Bug fixes

* The list of supported gates is replaced with `pennylane.ops._qubit__ops__` so that the `CZ` gate is included.
  [(#305)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/305)

### Contributors

This release contains contributions from (in alphabetical order):

Matthew Silverman,
Frederik Wilde,
Etienne Wodey (Alpine Quantum Technologies GmbH)

---
# Release 0.30.1

### Breaking changes

* `vqe_runner` has been updated to use IBMQ's VQE program. The first argument, `program_id`, has
  now been removed. The `upload_vqe_runner` and `delete_vqe_runner` functions have also been removed.
  [(#298)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/298)

### Improvements

* Updated many small things across the plugin to match re-works and deprecations in `qiskit`. The plugin
  can still be used in the same way as before. However, we suggest you authenticate with
  `qiskit_ibm_provider.IBMProvider` instead of `qiskit.IBMQ` from now on, as the latter is deprecated.
  [(#301)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/301)

### Contributors

This release contains contributions from (in alphabetical order):

Matthew Silverman

# Release 0.30.0

### Breaking changes

* The new return system from PennyLane is adopted in the plugin as well.
  [(#281)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/281)

### Contributors

This release contains contributions from (in alphabetical order):

Romain Moyard.

---
# Release 0.29.0

### Breaking changes

* `.inv` is replaced by `qml.adjoint` in PennyLane `0.30.0` and therefore the plugin is adapted as well.
  [(#260)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/260)

* The minimum required version of PennyLane is bumped to `0.28`. The current plugin
  does not work with PennyLane v0.27.

### Bug fixes

* The number of executions of the device is now correct.
  [(#259)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/259)

### Contributors

This release contains contributions from (in alphabetical order):

Christina Lee
Romain Moyard

---
# Release 0.28.0

### Breaking changes

* Changed the signature of the `QubitDevice.statistics` method from

  ```python
  def statistics(self, observables, shot_range=None, bin_size=None, circuit=None):
  ```

  to

  ```python
  def statistics(self, circuit: QuantumScript, shot_range=None, bin_size=None):
  ```

  [#3421](https://github.com/PennyLaneAI/pennylane/pull/3421)

### Improvements

* Adds testing for Python 3.11.
  [(#237)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/237)

### Bug fixes

* Do not try to connect with an IBMQX token if it is falsy.
  [(#234)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/234)

### Contributors

This release contains contributions from (in alphabetical order):

Christina Lee
Albert Mitjans-Coma
Matthew Silverman

---
# Release 0.27.0

### New features since last release

* Add support for the `ISWAP` operation.
  [(#229)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/229)

### Bug fixes

* Fix Cobyla success bool for VQE runtime.
  [(#231)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/231)

### Contributors

This release contains contributions from (in alphabetical order):

Romain Moyard, Matthew Silverman

---
# Release 0.24.0

### Improvements

* Improvement of the different `requirements.txt` and `requirements-ci.txt` files.
  [(#212)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/212)

* The plugin now natively supports the adjoint of the `S`, `T`, and `SX` gates.
  [(#216)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/216)

### Documentation

* Use the centralized [Xanadu Sphinx Theme](https://github.com/XanaduAI/xanadu-sphinx-theme)
  to style the Sphinx documentation.
  [(#215)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/215)

### Bug fixes

* Defines the missing `returns_state` entry of the
  `capabilities` dictionary for devices.
  [(#220)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/220)

### Contributors

This release contains contributions from (in alphabetical order):

Mikhail Andrenkov, Christina Lee, Romain Moyard, Antal Száva

---
# Release 0.23.0

### New features since last release

* Add support for the operations`IsingXX`, `IsingYY`, `IsingZZ`
  [(#209)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/209)

### Bug fixes

* Fix runtime sampler due to changes on Qiskit side.
  [(#201)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/201)

* Pin `jinja2` to 3.0.3 because of sphinx incompatibility.
  [(#207)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/207)

### Contributors

This release contains contributions from (in alphabetical order):

Samuel Banning, Romain Moyard

---

# Release 0.22.0

### Improvements

* Changed a validation check such that it handles qubit numbers represented as
  strings.
  [(#184)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/184)

* Changed the VQE callback function for SciPy optimizers.
  [(#187)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/187)

* Switched from using the deprecated `qiskit.circuit.measure.measure` function
  to using a method.
  [(#191)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/191)

### Bug fixes

* Changed the access to Hamiltonian terms `hamiltonian.terms()` as a method.
  [(#190)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/190)

### Contributors

This release contains contributions from (in alphabetical order):

Thomas Bromley, Andrea Mari, Romain Moyard, Antal Száva

---

# Release 0.21.0

### New features since last release

* Add two devices for runtime programs and one VQE runtime program solver.
  [(#157)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/157)

### Improvements

* Improved the login flow when IBMQ tokens are specified as environment variables.
  [(#169)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/169)

### Documentation

* Improved the quality of docstrings in the library.
  [(#174)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/174)

### Contributors

This release contains contributions from (in alphabetical order):

Guillermo Alonso-Linaje, Romain Moyard, Tanner Rogalsky, Jay Soni, Antal Száva

---

# Release 0.20.0

### New features since last release

* Defined the `QiskitDevice.batch_execute` method, to allow
  Qiskit backends to run multiple quantum circuits at the same time. This
  addition allows submitting batches of circuits to IBMQ e.g., when computing
  gradients internally.
  [(#156)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/156)
  [(#163)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/163)
  [(#167)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/167)

### Improvements

* Added native support for the `qml.Identity` operation to the Qiskit devices and converters.
  [(#162)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/162)

* Added support for the `qml.SX` operation to the Qiskit devices.
  [(#158)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/158)

* Added support for returning job execution times.
  [(#160)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/160)

* Added support for Python 3.10.
  [(#170)](https://github.com/PennyLaneAI/pennylane-forest/pull/170)

### Contributors

This release contains contributions from (in alphabetical order):

Guillermo Alonso-Linaje, David Ittah, Romain Moyard, Antal Száva

---

# Release 0.18.0

### Improvements

* Removed adding the `verbose` option to the arguments passed to the backend
  such that no warnings are raised.
  [(#151)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/151)

### Contributors

This release contains contributions from (in alphabetical order):

Christina Lee, Jay Soni, Antal Száva

---

# Release 0.17.0

### Improvements

* Removed a validation check for `qml.QubitUnitary` that existed in the device and
  adjusted a related test case.
  [(#144)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/144)

### Contributors

This release contains contributions from (in alphabetical order):

Christina Lee, Antal Száva

---

# Release 0.16.0

### New features since last release

* Added support for the new `qml.Projector` observable in
  PennyLane v0.16 to the Qiskit devices.
  [(#137)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/137)

### Breaking changes

* Deprecated Python 3.6.
  [(#140)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/140)

### Improvements

* The plugin can now load Qiskit circuits with more complicated `ParameterExpression` variables.
  [(#139)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/139)

### Contributors

This release contains contributions from (in alphabetical order):

Christina Lee, Vincent Wong

# Release 0.15.0

### Breaking changes

* For compatibility with PennyLane v0.15, the `analytic` keyword argument
  has been removed from all devices. Statistics can still be computed analytically
  by setting `shots=None`.
  [(#130)](https://github.com/XanaduAI/pennylane-qiskit/pull/130)

### Improvements

* PennyLane-Qiskit has been upgraded to work with Qiskit version 0.25.
  [(#132)](https://github.com/XanaduAI/pennylane-qiskit/pull/132)

### Contributors

This release contains contributions from (in alphabetical order):

Christina Lee, Olivia Di Matteo, Josh Izaac, Antal Száva

---

# Release 0.14.0

### Bug fixes

* With the release of Qiskit 0.23.3 gate parameters cannot be arrays.  The device now converts arrays to lists.
  [(#126)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/126)

* When parsing the IBMQ token and the IBMQ URL, the values passed as keywords take precedence over environment variables.
  [(#121)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/121)

* When loading Qiskit circuits to PennyLane templates using `load` in `converter.py`, parameters with `requires_grad=False` are bound to the circuit.  The old version bound objects that were not PennyLane `Variable`'s, but that object class is now deprecated.
  [(#127)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/127)

### Contributors

This release contains contributions from (in alphabetical order):

Christina Lee, Antal Szava

---

# Release 0.13.0

### Improvements

* The provided devices are now compatible with Qiskit 0.23.1.
  [(#116)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/116)

### Bug fixes

* The Aer devices store the noise models correctly.
  [(#114)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/114)

### Contributors

This release contains contributions from (in alphabetical order):

Olivia Di Matteo, Josh Izaac, Antal Száva

---

# Release 0.12.0

### Improvements

* Qiskit devices are now allowed to pass transpilation options.
  [(#108)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/108)

* The provided devices are now compatible with Qiskit 0.23.
  [(#112)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/112)

### Bug fixes

* Removes PySCF from the plugin `setup.py` and `requirements.txt`.
  [(#103)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/103)
  [(#104)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/104)

* Fixed a bug related to extracting differentiable parameters for the Qiskit
  converter and PennyLane array indexing.
  [(#106)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/106)

### Contributors

This release contains contributions from (in alphabetical order):

Josh Izaac, Nathan Killoran, Sagar Pahwa, Antal Száva

---

# Release 0.11.0

### New features since last release

* Qiskit devices now support custom wire labels.
  [(#99)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/99)
  [(#100)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/100)

  One can now specify any string or number as a custom wire label,
  and use these labels to address subsystems on the device:

  ```python
  dev = qml.device('qiskit.ibmq', wires=['q1', 'ancilla', 0, 1])

  def circuit():
    qml.Hadamard(wires='q1')
    qml.CNOT(wires=[1, 'ancilla'])
  ```

### Improvements

* Adds support for Qiskit v0.20.
  [(#101)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/101)

### Bug fixes

* When converting QASM or Qiskit circuit to PennyLane templates, the `CU1` gate
  is now natively supported and converted to a `QubitUnitary`.
  [(#101)](https://github.com/PennyLaneAI/pennylane-qiskit/pull/101)

### Contributors

This release contains contributions from (in alphabetical order):

Maria Schuld, Antal Száva

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

* Ported the `QiskitDevice` class to inherit from the `QubitDevice` class
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
