# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
====================================================================
Scheduling (:mod:`qiskit_ibm_runtime.transpiler.passes.scheduling`)
====================================================================

.. currentmodule:: qiskit_ibm_runtime.transpiler.passes.scheduling

A collection of scheduling passes for working with IBM Quantum's next-generation
backends that support advanced "dynamic circuit" capabilities. Ie.,
circuits with support for classical control-flow/feedback based off
of measurement results.

.. warning::
    You should not mix these scheduling passes with Qiskit's builtin scheduling
    passes as they will negatively interact with the scheduling routines for
    dynamic circuits. This includes setting ``scheduling_method`` in
    :func:`~qiskit.compiler.transpile` or
    :func:`~qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager`.

Below we demonstrate how to schedule and pad a teleportation circuit with delays
for a dynamic circuit backend's execution model:

.. jupyter-execute::

    from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.transpiler.passmanager import PassManager

    from qiskit_ibm_runtime.transpiler.passes.scheduling import DynamicCircuitInstructionDurations
    from qiskit_ibm_runtime.transpiler.passes.scheduling import ALAPScheduleAnalysis
    from qiskit_ibm_runtime.transpiler.passes.scheduling import PadDelay
    from qiskit_ibm_runtime.fake_provider import FakeJakarta

    backend = FakeJakarta()

    # Temporary workaround for mock backends. For real backends this is not required.
    backend.configuration().basis_gates.append("if_else")


    # Use this duration class to get appropriate durations for dynamic
    # circuit backend scheduling
    durations = DynamicCircuitInstructionDurations.from_backend(backend)
    # Generate the main Qiskit transpile passes.
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    # Configure the as-late-as-possible scheduling pass
    pm.scheduling = PassManager([ALAPScheduleAnalysis(durations), PadDelay(durations)])

    qr = QuantumRegister(3)
    crz = ClassicalRegister(1, name="crz")
    crx = ClassicalRegister(1, name="crx")
    result = ClassicalRegister(1, name="result")

    teleport = QuantumCircuit(qr, crz, crx, result, name="Teleport")

    teleport.h(qr[1])
    teleport.cx(qr[1], qr[2])
    teleport.cx(qr[0], qr[1])
    teleport.h(qr[0])
    teleport.measure(qr[0], crz)
    teleport.measure(qr[1], crx)
    with teleport.if_test((crz, 1)):
        teleport.z(qr[2])
    with teleport.if_test((crx, 1)):
        teleport.x(qr[2])
    teleport.measure(qr[2], result)

    # Transpile.
    scheduled_teleport = pm.run(teleport)

    scheduled_teleport.draw(output="mpl", style="iqp")


Instead of padding with delays we may also insert a dynamical decoupling sequence
using the :class:`PadDynamicalDecoupling` pass as shown below:

.. jupyter-execute::

    from qiskit.circuit.library import XGate

    from qiskit_ibm_runtime.transpiler.passes.scheduling import PadDynamicalDecoupling


    dd_sequence = [XGate(), XGate()]

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    pm.scheduling = PassManager(
        [
            ALAPScheduleAnalysis(durations),
            PadDynamicalDecoupling(durations, dd_sequence),
        ]
    )

    dd_teleport = pm.run(teleport)

    dd_teleport.draw(output="mpl", style="iqp")

When compiling a circuit with Qiskit, it is more efficient and more robust to perform all the
transformations in a single transpilation.  This has been done above by extending Qiskit's preset
pass managers.  For example, Qiskit's :func:`~qiskit.compiler.transpile` function internally builds
its pass set by using :func:`~qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager`.
This returns instances of :class:`~qiskit.transpiler.StagedPassManager`, which can be extended.


Scheduling old format ``c_if`` conditioned gates
------------------------------------------------

Scheduling with old format ``c_if`` conditioned gates is not supported.

.. jupyter-execute::

    qc_c_if = QuantumCircuit(1, 1)
    qc_c_if.x(0).c_if(0, 1)
    qc_c_if.draw(output="mpl", style="iqp")

The :class:`.IBMBackend` configures a translation plugin
:class:`.IBMTranslationPlugin` to automatically
apply transformations and optimizations for IBM hardware backends when invoking
:func:`~qiskit.compiler.transpile`. This will automatically convert all old style ``c_if``
conditioned gates to new-style control-flow.
We may then schedule the transpiled circuit without further modification.

.. jupyter-execute::

    # Temporary workaround for mock backends. For real backends this is not required.
    backend.get_translation_stage_plugin = lambda: "ibm_dynamic_circuits"

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    pm.scheduling = PassManager(
        [
            ALAPScheduleAnalysis(durations),
            PadDynamicalDecoupling(durations, dd_sequence),
        ]
    )

    qc_if_dd = pm.run(qc_c_if, backend)
    qc_if_dd.draw(output="mpl", style="iqp")


If you are not using the transpiler plugin stages to
work around this please manually run the pass
:class:`qiskit.transpiler.passes.ConvertConditionsToIfOps`
prior to your scheduling pass.

.. jupyter-execute::

    from qiskit.transpiler.passes import ConvertConditionsToIfOps

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    pm.scheduling = PassManager(
          [
              ConvertConditionsToIfOps(),
              ALAPScheduleAnalysis(durations),
              PadDelay(durations),
          ]
    )

    qc_if_dd = pm.run(qc_c_if)
    qc_if_dd.draw(output="mpl", style="iqp")


Exploiting IBM backend's local parallel "fast-path"
---------------------------------------------------

IBM quantum hardware supports a localized "fast-path" which enables a block of gates
applied to a *single qubit* that are conditional on an immediately predecessor measurement
*of the same qubit* to be completed with lower latency. The hardware is also
able to do this in *parallel* on disjoint qubits that satisfy this condition.

For example, the conditional gates below are performed in parallel with lower latency
as the measurements flow directly into the conditional blocks which in turn only apply
gates to the same measurement qubit.

.. jupyter-execute::

        qc = QuantumCircuit(2, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        # Conditional blocks will be performed in parallel in the hardware
        with qc.if_test((0, 1)):
            qc.x(0)
        with qc.if_test((1, 1)):
            qc.x(1)

        qc.draw(output="mpl", style="iqp")


The circuit below will not use the fast-path as the conditional gate is
on a different qubit than the measurement qubit.

.. jupyter-execute::

        qc = QuantumCircuit(2, 2)
        qc.measure(0, 0)
        with qc.if_test((0, 1)):
            qc.x(1)

        qc.draw(output="mpl", style="iqp")

Similarly, the circuit below contains gates on multiple qubits
and will not be performed using the fast-path.

.. jupyter-execute::

        qc = QuantumCircuit(2, 2)
        qc.measure(0, 0)
        with qc.if_test((0, 1)):
            qc.x(0)
            qc.x(1)

        qc.draw(output="mpl", style="iqp")

A fast-path block may contain multiple gates as long as they are on the fast-path qubit.
If there are multiple fast-path blocks being performed in parallel each block will be
padded out to the duration of the longest block.

.. jupyter-execute::

        qc = QuantumCircuit(2, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        # Conditional blocks will be performed in parallel in the hardware
        with qc.if_test((0, 1)):
            qc.x(0)
            # Will be padded out to a duration of 1600 on the backend.
        with qc.if_test((1, 1)):
            qc.delay(1600, 1)

        qc.draw(output="mpl", style="iqp")

This behavior is also applied to the else condition of a fast-path eligible branch.

.. jupyter-execute::

        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        # Conditional blocks will be performed in parallel in the hardware
        with qc.if_test((0, 1)) as else_:
            qc.x(0)
            # Will be padded out to a duration of 1600 on the backend.
        with else_:
            qc.delay(1600, 0)

        qc.draw(output="mpl", style="iqp")


If a single measurement result is used with several conditional blocks, if there is a fast-path
eligible block it will be applied followed by the non-fast-path blocks which will execute with
the standard higher latency conditional branch.

.. jupyter-execute::

        qc = QuantumCircuit(2, 2)
        qc.measure(0, 0)
        # Conditional blocks will be performed in parallel in the hardware
        with qc.if_test((0, 1)):
            # Uses fast-path
            qc.x(0)
        with qc.if_test((0, 1)):
            # Does not use fast-path
            qc.x(1)

        qc.draw(output="mpl", style="iqp")

If you wish to prevent the usage of the fast-path you may insert a barrier between the measurement and
the conditional branch.

.. jupyter-execute::

        qc = QuantumCircuit(1, 2)
        qc.measure(0, 0)
        # Barrier prevents the fast-path.
        qc.barrier()
        with qc.if_test((0, 1)):
            qc.x(0)

        qc.draw(output="mpl", style="iqp")

Conditional measurements are not eligible for the fast-path.

.. jupyter-execute::

        qc = QuantumCircuit(1, 2)
        qc.measure(0, 0)
        with qc.if_test((0, 1)):
            # Does not use the fast-path
            qc.measure(0, 1)

        qc.draw(output="mpl", style="iqp")

Similarly nested control-flow is not eligible.

.. jupyter-execute::

        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        with qc.if_test((0, 1)):
            # Does not use the fast-path
            qc.x(0)
            with qc.if_test((0, 1)):
                qc.x(0)

        qc.draw(output="mpl", style="iqp")


The scheduler is aware of the fast-path behavior and will not insert delays on idle qubits
in blocks that satisfy the fast-path conditions so as to avoid preventing the backend
compiler from performing the necessary optimizations to utilize the fast-path. If
there are fast-path blocks that will be performed in parallel they currently *will not*
be padded out by the scheduler to ensure they are of the same duration in Qiskit

.. jupyter-execute::

    dd_sequence = [XGate(), XGate()]

    pm = PassManager(
        [
            ALAPScheduleAnalysis(durations),
            PadDynamicalDecoupling(durations, dd_sequence),
        ]
    )

    qc = QuantumCircuit(2, 2)
    qc.measure(0, 0)
    qc.measure(1, 1)
    with qc.if_test((0, 1)):
        qc.x(0)
        # Is currently not padded to ensure
        # a duration of 1000. If you desire
        # this you would need to manually add
        # qc.delay(840, 0)
    with qc.if_test((1, 1)):
        qc.delay(1000, 0)


    qc.draw(output="mpl", style="iqp")

    qc_dd = pm.run(qc)

    qc_dd.draw(output="mpl", style="iqp")

.. note::
    If there are qubits that are *not* involved in a fast-path decision it is not
    currently possible to use them in a fast-path branch in parallel with the fast-path
    qubits resulting from a measurement. This will be revised in the future as we
    further improve these capabilities.

    For example:

    .. jupyter-execute::

        qc = QuantumCircuit(3, 2)
        qc.x(1)
        qc.measure(0, 0)
        with qc.if_test((0, 1)):
            qc.x(0)
        # Qubit 1 sits idle throughout the fast-path decision
        with qc.if_test((1, 0)):
            # Qubit 2 is idle but there is no measurement
            # to make it fast-path eligible. This will
            # however avoid a communication event in the hardware
            # since the condition is compile time evaluated.
            qc.x(2)

        qc.draw(output="mpl", style="iqp")


Scheduling & Dynamical Decoupling
=================================
.. autosummary::
   :toctree: ../stubs/

    BlockBasePadder
    ALAPScheduleAnalysis
    ASAPScheduleAnalysis
    DynamicCircuitInstructionDurations
    PadDelay
    PadDynamicalDecoupling
"""

from .block_base_padder import BlockBasePadder
from .dynamical_decoupling import PadDynamicalDecoupling
from .pad_delay import PadDelay
from .scheduler import ALAPScheduleAnalysis, ASAPScheduleAnalysis
from .utils import DynamicCircuitInstructionDurations
