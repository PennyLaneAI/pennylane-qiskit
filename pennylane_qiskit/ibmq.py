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
from qiskit_ibm_provider.job import IBMJobError

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
        timeout_secs (int): A timeout value in seconds to wait for job results from an IBMQ backend.
            The default value of ``None`` means no timeout

    Keyword Args:
        ibmqx_token (str): The IBM Q API token. If not provided, the environment
            variable ``IBMQX_TOKEN`` is used.
        ibmqx_url (str): The IBM Q URL. If not provided, the environment
            variable ``IBMQX_URL`` is used, followed by the default URL.
        noise_model (NoiseModel): NoiseModel Object from ``qiskit_aer.noise``.
            Only applicable for simulator backends.
        hub (str): Name of the provider hub.
        group (str): Name of the provider group.
        project (str): Name of the provider project.
    """

    short_name = "qiskit.ibmq"

    def __init__(
        self,
        wires,
        provider=None,
        backend="ibmq_qasm_simulator",
        shots=1024,
        timeout_secs=None,
        **kwargs,
    ):  # pylint:disable=too-many-arguments
        # Connection to IBMQ
        connect(kwargs)

        hub = kwargs.get("hub", "ibm-q")
        group = kwargs.get("group", "open")
        project = kwargs.get("project", "main")
        instance = "/".join([hub, group, project])

        # get a provider
        p = provider or IBMProvider(instance=instance)

        super().__init__(wires=wires, provider=p, backend=backend, shots=shots, **kwargs)
        self.timeout_secs = timeout_secs

    def batch_execute(self, circuits):  # pragma: no cover, pylint:disable=arguments-differ
        res = super().batch_execute(circuits, timeout=self.timeout_secs)
        if self.tracker.active:
            self._track_run()
        return res

    def _track_run(self):  # pragma: no cover
        """Provide runtime information."""

        expected_keys = {"created", "running", "finished"}
        time_per_step = self._current_job.time_per_step()
        if not set(time_per_step).issuperset(expected_keys):
            # self._current_job.result() should have already run by now
            # tests see a race condition, so this is ample time for that case
            timeout_secs = self.timeout_secs or 60
            self._current_job.wait_for_final_state(timeout=timeout_secs)
            self._current_job.refresh()
            time_per_step = self._current_job.time_per_step()
            if not set(time_per_step).issuperset(expected_keys):
                raise IBMJobError(
                    f"time_per_step had keys {set(time_per_step)}, needs {expected_keys}. If your program takes a long time, you may want to configure the device with a higher `timeout_secs`"
                )

        job_time = {
            "queued": (time_per_step["running"] - time_per_step["created"]).total_seconds(),
            "running": (time_per_step["finished"] - time_per_step["running"]).total_seconds(),
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
