# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the :class:`~.IBMQDevice` class, a PennyLane device that allows
evaluation and differentiation of IBM Q's Quantum Processing Units (QPUs)
using PennyLane.
"""
import os

from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_provider.exceptions import IBMAccountError
from qiskit_ibm_provider.accounts.exceptions import AccountsError

from .qiskit_device import QiskitDevice


class IBMQDevice(QiskitDevice):
    """A PennyLane device for the IBMQ API (remote) backend.

    For more details, see the `Qiskit documentation <https://qiskit.org/documentation/>`_

    You need to register at `IBMQ <https://quantum-computing.ibm.com/>`_ in order to
    recieve a token that is used for authentication using the API.

    As of the writing of this documentation, the API is free of charge, although
    there is a credit system to limit access to the quantum devices.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Note that for some backends, the number
            of wires has to match the number of qubits accessible.
        provider (Provider): The IBM Q provider you wish to use. If not provided,
            then the default provider returned by ``IBMProvider()`` is used.
        backend (str): the desired provider backend
        shots (int): number of circuit evaluations/random samples used
            to estimate expectation values and variances of observables

    Keyword Args:
        ibmqx_token (str): The IBM Q API token. If not provided, the environment
            variable ``IBMQX_TOKEN`` is used.
        ibmqx_url (str): The IBM Q URL. If not provided, the environment
            variable ``IBMQX_URL`` is used, followed by the default URL.
        noise_model (NoiseModel): NoiseModel Object from ``qiskit.providers.aer.noise``.
            Only applicable for simulator backends.
        hub (str): Name of the provider hub.
        group (str): Name of the provider group.
        project (str): Name of the provider project.
    """

    short_name = "qiskit.ibmq"

    def __init__(self, wires, provider=None, backend="ibmq_qasm_simulator", shots=1024, **kwargs):
        # Connection to IBMQ
        connect(kwargs)

        hub = kwargs.get("hub", "ibm-q")
        group = kwargs.get("group", "open")
        project = kwargs.get("project", "main")
        instance = "/".join([hub, group, project])

        # get a provider
        p = provider or IBMProvider(instance=instance)

        super().__init__(wires=wires, provider=p, backend=backend, shots=shots, **kwargs)

    def batch_execute(self, circuits):  # pragma: no cover
        res = super().batch_execute(circuits)
        if self.tracker.active:
            self._track_run()
        return res

    def _track_run(self):  # pragma: no cover
        """Provide runtime information."""

        time_per_step = self._current_job.time_per_step()
        job_time = {
            "creating": (time_per_step["CREATED"] - time_per_step["CREATING"]).total_seconds(),
            "validating": (
                time_per_step["VALIDATED"] - time_per_step["VALIDATING"]
            ).total_seconds(),
            "queued": (time_per_step["RUNNING"] - time_per_step["QUEUED"]).total_seconds(),
            "running": (time_per_step["COMPLETED"] - time_per_step["RUNNING"]).total_seconds(),
        }
        self.tracker.update(job_time=job_time)
        self.tracker.record()


def connect(kwargs):
    """Function that allows connection to IBMQ.

    Args:
        kwargs(dict): dictionary that contains the token and the url"""

    hub = kwargs.get("hub", "ibm-q")
    group = kwargs.get("group", "open")
    project = kwargs.get("project", "main")
    instance = "/".join([hub, group, project])

    token = kwargs.get("ibmqx_token", None) or os.getenv("IBMQX_TOKEN")
    url = kwargs.get("ibmqx_url", None) or os.getenv("IBMQX_URL")

    saved_accounts = IBMProvider.saved_accounts()
    if not token:
        if not saved_accounts:
            raise IBMAccountError("No active IBM Q account, and no IBM Q token provided.")
        try:
            IBMProvider(url=url, instance=instance)
        except AccountsError as e:
            raise AccountsError(
                f"Accounts were found ({set(saved_accounts)}), but all failed to load."
            ) from e
        return
    for account in saved_accounts.values():
        if account["token"] == token:
            return
    IBMProvider.save_account(token=token, url=url, instance=instance)
