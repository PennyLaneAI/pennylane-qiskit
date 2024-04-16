# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=wildcard-import,unused-argument

"""
Fake provider class that provides access to fake backends.
"""

from qiskit.providers.provider import ProviderV1
from qiskit.providers.exceptions import QiskitBackendNotFoundError

from .backends import *


class FakeProviderFactory:
    """Fake provider factory class."""

    def __init__(self) -> None:
        self.fake_provider = FakeProvider()

    def load_account(self) -> None:
        """Fake load_account method to mirror the IBMQ provider."""
        pass

    def enable_account(self, *args, **kwargs) -> None:  # type: ignore
        """Fake enable_account method to mirror the IBMQ provider factory."""
        pass

    def disable_account(self) -> None:
        """Fake disable_account method to mirror the IBMQ provider factory."""
        pass

    def save_account(self, *args, **kwargs) -> None:  # type: ignore
        """Fake save_account method to mirror the IBMQ provider factory."""
        pass

    @staticmethod
    def delete_account() -> None:
        """Fake delete_account method to mirror the IBMQ provider factory."""
        pass

    def update_account(self, force: bool = False) -> None:
        """Fake update_account method to mirror the IBMQ provider factory."""
        pass

    def providers(self) -> list:
        """Fake providers method to mirror the IBMQ provider."""
        return [self.fake_provider]

    def get_provider(self, hub: str = None, group: str = None, projec: str = None):  # type: ignore
        """Fake get_provider method to mirror the IBMQ provider."""
        return self.fake_provider


class FakeProviderForBackendV2(ProviderV1):
    """Fake provider containing fake V2 backends.

    Only filtering backends by name is implemented. This class contains all fake V2 backends
    available in the :mod:`qiskit_ibm_runtime.fake_provider`.
    """

    def backend(self, name=None, **kwargs):  # type: ignore
        """
        Filter backends in provider by name.
        """
        backend = self._backends[0]
        if name:
            filtered_backends = [backend for backend in self._backends if backend.name == name]
            if not filtered_backends:
                raise QiskitBackendNotFoundError()

            backend = filtered_backends[0]

        return backend

    def backends(self, name=None, **kwargs):  # type: ignore
        return self._backends

    def __init__(self) -> None:
        self._backends = [
            FakeAlgiers(),  # type: ignore
            FakeAlmadenV2(),  # type: ignore
            FakeArmonkV2(),  # type: ignore
            FakeAthensV2(),  # type: ignore
            FakeAuckland(),  # type: ignore
            FakeBelemV2(),  # type: ignore
            FakeBoeblingenV2(),  # type: ignore
            FakeBogotaV2(),  # type: ignore
            FakeBrisbane(),  # type: ignore
            FakeBrooklynV2(),  # type: ignore
            FakeBurlingtonV2(),  # type: ignore
            FakeCairoV2(),  # type: ignore
            FakeCambridgeV2(),  # type: ignore
            FakeCasablancaV2(),  # type: ignore
            FakeCusco(),  # type: ignore
            FakeEssexV2(),  # type: ignore
            FakeGeneva(),  # type: ignore
            FakeGuadalupeV2(),  # type: ignore
            FakeHanoiV2(),  # type: ignore
            FakeJakartaV2(),  # type: ignore
            FakeJohannesburgV2(),  # type: ignore
            FakeKawasaki(),  # type: ignore
            FakeKolkataV2(),  # type: ignore
            FakeKyiv(),  # type: ignore
            FakeKyoto(),  # type: ignore
            FakeLagosV2(),  # type: ignore
            FakeLimaV2(),  # type: ignore
            FakeLondonV2(),  # type: ignore
            FakeManhattanV2(),  # type: ignore
            FakeManilaV2(),  # type: ignore
            FakeMelbourneV2(),  # type: ignore
            FakeMontrealV2(),  # type: ignore
            FakeMumbaiV2(),  # type: ignore
            FakeNairobiV2(),  # type: ignore
            FakeOsaka(),  # type: ignore
            FakeOslo(),  # type: ignore
            FakeOurenseV2(),  # type: ignore
            FakeParisV2(),  # type: ignore
            FakePeekskill(),  # type: ignore
            FakePerth(),  # type: ignore
            FakePrague(),  # type: ignore
            FakePoughkeepsieV2(),  # type: ignore
            FakeQuebec(),  # type: ignore
            FakeQuitoV2(),  # type: ignore
            FakeRochesterV2(),  # type: ignore
            FakeRomeV2(),  # type: ignore
            FakeSantiagoV2(),  # type: ignore
            FakeSherbrooke(),  # type: ignore
            FakeSingaporeV2(),  # type: ignore
            FakeSydneyV2(),  # type: ignore
            FakeTorino(),  # type: ignore
            FakeTorontoV2(),  # type: ignore
            FakeValenciaV2(),  # type: ignore
            FakeVigoV2(),  # type: ignore
            FakeWashingtonV2(),  # type: ignore
            FakeYorktownV2(),  # type: ignore
        ]

        super().__init__()


class FakeProvider(ProviderV1):
    """Fake provider containing fake V1 backends.

    Only filtering backends by name is implemented. This class contains all fake V1 backends
    available in the :mod:`qiskit_ibm_runtime.fake_provider`.
    """

    def get_backend(self, name=None, **kwargs):  # type: ignore
        backend = self._backends[0]
        if name:
            filtered_backends = [backend for backend in self._backends if backend.name() == name]
            if not filtered_backends:
                raise QiskitBackendNotFoundError()

            backend = filtered_backends[0]

        return backend

    def backends(self, name=None, **kwargs):  # type: ignore
        return self._backends

    def __init__(self) -> None:
        self._backends = [
            FakeAlmaden(),  # type: ignore
            FakeArmonk(),  # type: ignore
            FakeAthens(),  # type: ignore
            FakeBelem(),  # type: ignore
            FakeBoeblingen(),  # type: ignore
            FakeBogota(),  # type: ignore
            FakeBrooklyn(),  # type: ignore
            FakeBurlington(),  # type: ignore
            FakeCairo(),  # type: ignore
            FakeCambridge(),  # type: ignore
            FakeCambridgeAlternativeBasis(),  # type: ignore
            FakeCasablanca(),  # type: ignore
            FakeEssex(),  # type: ignore
            FakeGuadalupe(),  # type: ignore
            FakeHanoi(),  # type: ignore
            FakeJakarta(),  # type: ignore
            FakeJohannesburg(),  # type: ignore
            FakeKolkata(),  # type: ignore
            FakeLagos(),  # type: ignore
            FakeLima(),  # type: ignore
            FakeLondon(),  # type: ignore
            FakeManila(),  # type: ignore
            FakeManhattan(),  # type: ignore
            FakeMelbourne(),  # type: ignore
            FakeMontreal(),  # type: ignore
            FakeMumbai(),  # type: ignore
            FakeNairobi(),  # type: ignore
            FakeOurense(),  # type: ignore
            FakeParis(),  # type: ignore
            FakePoughkeepsie(),  # type: ignore
            FakeQuito(),  # type: ignore
            FakeRochester(),  # type: ignore
            FakeRome(),  # type: ignore
            FakeRueschlikon(),  # type: ignore
            FakeSantiago(),  # type: ignore
            FakeSingapore(),  # type: ignore
            FakeSydney(),  # type: ignore
            FakeTenerife(),  # type: ignore
            FakeTokyo(),  # type: ignore
            FakeToronto(),  # type: ignore
            FakeValencia(),  # type: ignore
            FakeVigo(),  # type: ignore
            FakeWashington(),  # type: ignore
            FakeYorktown(),  # type: ignore
        ]

        super().__init__()
