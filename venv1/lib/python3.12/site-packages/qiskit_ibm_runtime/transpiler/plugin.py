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

"""Plugin for IBM provider backend transpiler stages."""

from typing import Optional

from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.passes import ConvertConditionsToIfOps

from qiskit_ibm_runtime.transpiler.passes.basis.convert_id_to_delay import (
    ConvertIdToDelay,
)


class IBMTranslationPlugin(PassManagerStagePlugin):
    """A translation stage plugin for targeting Qiskit circuits
    to IBM Quantum systems."""

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig,
        optimization_level: Optional[int] = None,
    ) -> PassManager:
        """Build IBMTranslationPlugin PassManager."""

        translator_pm = common.generate_translation_passmanager(
            target=pass_manager_config.target,
            basis_gates=pass_manager_config.basis_gates,
            approximation_degree=pass_manager_config.approximation_degree,
            coupling_map=pass_manager_config.coupling_map,
            backend_props=pass_manager_config.backend_properties,
            unitary_synthesis_method=pass_manager_config.unitary_synthesis_method,
            unitary_synthesis_plugin_config=pass_manager_config.unitary_synthesis_plugin_config,
            hls_config=pass_manager_config.hls_config,
        )

        plugin_passes = []
        instruction_durations = pass_manager_config.instruction_durations
        if instruction_durations:
            plugin_passes.append(ConvertIdToDelay(instruction_durations))

        return PassManager(plugin_passes) + translator_pm


class IBMDynamicTranslationPlugin(PassManagerStagePlugin):
    """A translation stage plugin for targeting Qiskit circuits
    to IBM Quantum systems."""

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig,
        optimization_level: Optional[int] = None,
    ) -> PassManager:
        """Build IBMTranslationPlugin PassManager."""

        translator_pm = common.generate_translation_passmanager(
            target=pass_manager_config.target,
            basis_gates=pass_manager_config.basis_gates,
            approximation_degree=pass_manager_config.approximation_degree,
            coupling_map=pass_manager_config.coupling_map,
            backend_props=pass_manager_config.backend_properties,
            unitary_synthesis_method=pass_manager_config.unitary_synthesis_method,
            unitary_synthesis_plugin_config=pass_manager_config.unitary_synthesis_plugin_config,
            hls_config=pass_manager_config.hls_config,
        )

        instruction_durations = pass_manager_config.instruction_durations
        plugin_passes = []
        if pass_manager_config.target is not None:
            id_supported = "id" in pass_manager_config.target
        else:
            id_supported = "id" in pass_manager_config.basis_gates

        if instruction_durations and not id_supported:
            plugin_passes.append(ConvertIdToDelay(instruction_durations))

        # Only inject control-flow conversion pass at level 0 and level 1. As of
        # qiskit 0.22.x transpile() with level 2 and 3 does not support
        # control flow instructions (including if_else). This can be
        # removed when higher optimization levels support control flow
        # instructions.
        if optimization_level in {0, 1}:
            plugin_passes += [ConvertConditionsToIfOps()]

        return PassManager(plugin_passes) + translator_pm
