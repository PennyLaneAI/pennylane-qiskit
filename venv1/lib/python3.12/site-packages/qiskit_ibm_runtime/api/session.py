# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Session customized for IBM Quantum access."""

import inspect
import os
import re
import logging
import sys
from typing import Dict, Optional, Any, Tuple, Union
from pathlib import PurePath
import importlib.metadata

from requests import Session, RequestException, Response
from requests.adapters import HTTPAdapter
from requests.auth import AuthBase
from urllib3.util.retry import Retry

from qiskit_ibm_runtime.utils.utils import filter_data

from .exceptions import RequestsApiError
from ..exceptions import IBMNotAuthorizedError
from ..version import __version__ as ibm_runtime_version

STATUS_FORCELIST = (
    500,  # General server error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
    520,  # Cloudflare general error
    521,  # Cloudflare web server is down
    522,  # Cloudflare connection timeout
    524,  # Cloudflare Timeout
)
CUSTOM_HEADER_ENV_VAR = "QISKIT_IBM_RUNTIME_CUSTOM_CLIENT_APP_HEADER"
QE_PROVIDER_HEADER_ENV_VAR = "QE_CUSTOM_CLIENT_APP_HEADER"
USAGE_DATA_OPT_OUT_ENV_VAR = "USAGE_DATA_OPT_OUT"

logger = logging.getLogger(__name__)
# Regex used to match the `/backends` endpoint, capturing the device name as group(2).
# The number of letters for group(2) must be greater than 1, so it does not match
# the `/devices/v/1` endpoint.
# Capture groups: (/backends/)(<device_name>)(</optional rest of the url>)
RE_BACKENDS_ENDPOINT = re.compile(r"^(.*/backends/)([^/}]{2,})(.*)$", re.IGNORECASE)


def _get_client_header() -> str:
    """Return the client version."""

    if os.getenv(USAGE_DATA_OPT_OUT_ENV_VAR, "False") == "True":
        return ""

    qiskit_pkgs = [
        "qiskit",
        "qiskit_terra",
        "qiskit_aer",
        "qiskit_experiments",
        "qiskit_nature",
        "qiskit_machine_learning",
        "qiskit_optimization",
        "qiskit_finance",
        "circuit_knitting_toolbox",
    ]

    pkg_versions = {"qiskit_ibm_runtime": f"qiskit_ibm_runtime-{ibm_runtime_version}"}
    for pkg_name in qiskit_pkgs:
        try:
            version_info = f"{pkg_name}-{importlib.metadata.version(pkg_name)}"

            if pkg_name in sys.modules:
                version_info += "*"

            pkg_versions[pkg_name] = version_info
        except Exception:  # pylint: disable=broad-except
            pass
    return f"qiskit-version-2/{','.join(pkg_versions.values())}"


CLIENT_APPLICATION = _get_client_header()


class PostForcelistRetry(Retry):
    """Custom ``urllib3.Retry`` class that performs retry on ``POST`` errors in the force list.

    Retrying of ``POST`` requests are allowed *only* when the status code
    returned is on the ``STATUS_FORCELIST``. While ``POST``
    requests are recommended not to be retried due to not being idempotent,
    the IBM Quantum API guarantees that retrying on specific 5xx errors is safe.
    """

    def increment(  # type: ignore[no-untyped-def]
        self,
        method=None,
        url=None,
        response=None,
        error=None,
        _pool=None,
        _stacktrace=None,
    ):
        """Overwrites parent class increment method for logging."""
        if logger.getEffectiveLevel() is logging.DEBUG:
            status = data = headers = None
            if response:
                status = response.status
                data = response.data
                headers = response.headers
            logger.debug(
                "Retrying method=%s, url=%s, status=%s, error=%s, data=%s, headers=%s",
                method,
                url,
                status,
                error,
                data,
                headers,
            )
        return super().increment(
            method=method,
            url=url,
            response=response,
            error=error,
            _pool=_pool,
            _stacktrace=_stacktrace,
        )

    def is_retry(self, method: str, status_code: int, has_retry_after: bool = False) -> bool:
        """Indicate whether the request should be retried.

        Args:
            method: Request method.
            status_code: Status code.
            has_retry_after: Whether retry has been done before.

        Returns:
            ``True`` if the request should be retried, ``False`` otherwise.
        """
        if method.upper() == "POST" and status_code in self.status_forcelist:
            return True

        return super().is_retry(method, status_code, has_retry_after)


class RetrySession(Session):
    """Custom session with retry and handling of specific parameters.

    This is a child class of ``requests.Session``. It has its own retry
    policy and handles IBM Quantum specific parameters.
    """

    def __init__(
        self,
        base_url: str,
        retries_total: int = 5,
        retries_connect: int = 3,
        backoff_factor: float = 0.5,
        verify: bool = True,
        proxies: Optional[Dict[str, str]] = None,
        auth: Optional[AuthBase] = None,
        timeout: Tuple[float, Union[float, None]] = (5.0, None),
    ) -> None:
        """RetrySession constructor.

        Args:
            base_url: Base URL for the session's requests.
            retries_total: Number of total retries for the requests.
            retries_connect: Number of connect retries for the requests.
            backoff_factor: Backoff factor between retry attempts.
            verify: Whether to enable SSL verification.
            proxies: Proxy URLs mapped by protocol.
            auth: Authentication handler.
            timeout: Timeout for the requests, in the form of (connection_timeout,
                total_timeout).
        """
        super().__init__()

        self.base_url = base_url
        self.custom_header: Optional[str] = None
        self._initialize_retry(retries_total, retries_connect, backoff_factor)
        self._initialize_session_parameters(verify, proxies or {}, auth)
        self._timeout = timeout

    def __del__(self) -> None:
        """RetrySession destructor. Closes the session."""
        try:
            self.close()
        except Exception:  # pylint: disable=broad-except
            # ignore errors that may happen during cleanup
            pass

    def _initialize_retry(
        self, retries_total: int, retries_connect: int, backoff_factor: float
    ) -> None:
        """Set the session retry policy.

        Args:
            retries_total: Number of total retries for the requests.
            retries_connect: Number of connect retries for the requests.
            backoff_factor: Backoff factor between retry attempts.
        """
        retry = PostForcelistRetry(
            total=retries_total,
            connect=retries_connect,
            backoff_factor=backoff_factor,
            status_forcelist=STATUS_FORCELIST,
        )

        retry_adapter = HTTPAdapter(max_retries=retry)
        self.mount("http://", retry_adapter)
        self.mount("https://", retry_adapter)

    def _initialize_session_parameters(
        self, verify: bool, proxies: Dict[str, str], auth: Optional[AuthBase] = None
    ) -> None:
        """Set the session parameters and attributes.

        Args:
            verify: Whether to enable SSL verification.
            proxies: Proxy URLs mapped by protocol.
            auth: Authentication handler.
        """
        self.custom_header = os.getenv(CUSTOM_HEADER_ENV_VAR) or os.getenv(
            QE_PROVIDER_HEADER_ENV_VAR
        )
        self.auth = auth
        self.proxies = proxies or {}
        self.verify = verify

    def request(  # type: ignore[override]
        self, method: str, url: str, bare: bool = False, **kwargs: Any
    ) -> Response:
        """Construct, prepare, and send a ``Request``.

        If `bare` is not specified, prepend the base URL to the input `url`.
        Timeout value is passed if proxies are not used.

        Args:
            method: Method for the new request (e.g. ``POST``).
            url: URL for the new request.
            bare: If ``True``, do not send IBM Quantum specific information
                (such as access token) in the request or modify the input `url`.
            **kwargs: Additional arguments for the request.

        Returns:
            Response object.

        Raises:
            RequestsApiError: If the request failed.
            IBMNotAuthorizedError: If the auth token is invalid.
        """
        # pylint: disable=arguments-differ
        if bare:
            final_url = url
            # Explicitly pass `None` as the `access_token` param, disabling it.
            params = kwargs.get("params", {})
            params.update({"access_token": None})
            kwargs.update({"params": params})
        else:
            final_url = self.base_url + url

        # Add a timeout to the connection for non-proxy connections.
        if not self.proxies and "timeout" not in kwargs:
            kwargs.update({"timeout": self._timeout})

        headers = self.headers.copy()  # type: ignore
        headers.update(kwargs.pop("headers", {}))

        # Set default caller
        headers.update({"X-Qx-Client-Application": f"{CLIENT_APPLICATION}/qiskit"})

        if not os.getenv(USAGE_DATA_OPT_OUT_ENV_VAR, "False") == "True":
            # Use PurePath in order to support arbitrary path formats
            callers = {
                PurePath("qiskit/algorithms"),
                PurePath("qiskit_ibm_runtime/sampler.py"),
                PurePath("qiskit_ibm_runtime/estimator.py"),
                "qiskit_machine_learning",
                "qiskit_nature",
                "qiskit_optimization",
                "qiskit_experiments",
                "qiskit_finance",
                "circuit_knitting_toolbox",
            }
            stack = inspect.stack()
            stack.reverse()

            found_caller = False
            for frame in stack:
                frame_path = str(PurePath(frame.filename))
                for caller in callers:
                    if str(caller) in frame_path:
                        caller_str = str(caller) + frame_path.split(str(caller), 1)[-1]
                        if os.name == "nt":
                            sanitized_caller_str = caller_str.replace("\\", "~")
                        else:
                            sanitized_caller_str = caller_str.replace("/", "~")
                        if self.custom_header:
                            headers.update(
                                {
                                    "X-Qx-Client-Application": f"{CLIENT_APPLICATION}/"
                                    f"{sanitized_caller_str}/{self.custom_header}"
                                }
                            )
                        else:
                            headers.update(
                                {
                                    "X-Qx-Client-Application": f"{CLIENT_APPLICATION}"
                                    f"/{sanitized_caller_str}"
                                }
                            )
                        found_caller = True
                        break  # break out of the inner loop
                if found_caller:
                    break  # break out of the outer loop
        self.headers = headers
        self._set_custom_header()

        try:
            self._log_request_info(final_url, method, kwargs)
            response = super().request(method, final_url, headers=headers, **kwargs)
            response.raise_for_status()
        except RequestException as ex:
            # Wrap the requests exceptions into a IBM Q custom one, for
            # compatibility.
            message = str(ex)
            status_code = -1
            if ex.response is not None:
                status_code = ex.response.status_code
                try:
                    error_json = ex.response.json()["error"]
                    message += ". {}, Error code: {}.".format(
                        error_json["message"], error_json["code"]
                    )
                    logger.debug(
                        "Response uber-trace-id: %s",
                        ex.response.headers["uber-trace-id"],
                    )
                except Exception:  # pylint: disable=broad-except
                    # the response did not contain the expected json.
                    message += f". {ex.response.text}"
            if status_code == 401:
                raise IBMNotAuthorizedError(message) from ex
            raise RequestsApiError(message, status_code) from ex

        return response

    def _log_request_info(self, url: str, method: str, request_data: Dict[str, Any]) -> None:
        """Log the request data, filtering out specific information.

        Note:
            The string ``...`` is used to denote information that has been filtered out
            from the request, within the url and request data. Currently, the backend name
            is filtered out from endpoint URLs, using a regex to capture the name, and from
            the data sent to the server when submitting a job.

            The request data is only logged for the following URLs, since they contain useful
            information: ``/Jobs`` (POST), ``/Jobs/status`` (GET),
            and ``/backends/<device_name>/properties`` (GET).

        Args:
            url: URL for the new request.
            method: Method for the new request (e.g. ``POST``)
            request_data:Additional arguments for the request.

        Raises:
            Exception: If there was an error logging the request information.
        """
        # Replace the device name in the URL with `...` if it matches, otherwise leave it as is.
        filtered_url = re.sub(RE_BACKENDS_ENDPOINT, "\\1...\\3", url)

        if self._is_worth_logging(filtered_url):
            try:
                if logger.getEffectiveLevel() is logging.DEBUG:
                    request_data_to_log = ""
                    if filtered_url in ("/devices/.../properties", "/Jobs"):
                        # Log filtered request data for these endpoints.
                        request_data_to_log = "Request Data: {}.".format(filter_data(request_data))
                    logger.debug(
                        "Endpoint: %s. Method: %s. %s",
                        filtered_url,
                        method.upper(),
                        request_data_to_log,
                    )
            except Exception as ex:  # pylint: disable=broad-except
                # Catch general exception so as not to disturb the program if filtering fails.
                logger.info("Filtering failed when logging request information: %s", str(ex))

    def _is_worth_logging(self, endpoint_url: str) -> bool:
        """Returns whether the endpoint URL should be logged.

        The checks in place help filter out endpoint URL logs that would add noise
        and no helpful information.

        Args:
            endpoint_url: The endpoint URL that will be logged.

        Returns:
            Whether the endpoint URL should be logged.
        """
        if endpoint_url.endswith(
            (
                "/queue/status",
                "/devices/v/1",
                "/Jobs/status",
                "/.../properties",
                "/.../defaults",
            )
        ):
            return False
        if endpoint_url.startswith(("/users", "/version")):
            return False
        if endpoint_url == "/Network":
            return False
        if "objectstorage" in endpoint_url:
            return False
        if "bookings" in endpoint_url:
            return False

        return True

    def _set_custom_header(self) -> None:
        """Set custom header."""
        headers = self.headers.copy()  # type: ignore
        if self.custom_header:
            current = headers["X-Qx-Client-Application"]
            if self.custom_header not in current:
                headers.update({"X-Qx-Client-Application": f"{current}/{self.custom_header}"})
                self.headers = headers

    def __getstate__(self) -> Dict:
        """Overwrite Session's getstate to include all attributes."""
        state = super().__getstate__()  # type: ignore
        state.update(self.__dict__)
        return state
