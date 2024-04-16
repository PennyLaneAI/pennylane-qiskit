# pylint: disable=missing-docstring
import json
import time

import jwt
import pytest
import responses

from ibm_cloud_sdk_core import MCSPTokenManager, ApiException

OPERATION_PATH = '/siusermgr/api/1.0/apikeys/token'
MOCK_URL = 'https://mcsp.ibm.com'


# utility function to construct a mock token server response containing an access token.
def get_mock_token_response(issued_at, time_to_live) -> str:
    access_token_layout = {
        "username": "dummy",
        "role": "Admin",
        "permissions": ["administrator", "manage_catalog"],
        "sub": "admin",
        "iss": "sss",
        "aud": "sss",
        "uid": "sss",
        "iat": issued_at,
        "exp": issued_at + time_to_live,
    }

    access_token = jwt.encode(
        access_token_layout, 'secret', algorithm='HS256', headers={'kid': '230498151c214b788dd97f22b85410a5'}
    )

    token_server_response = {"token": access_token, "token_type": "jwt", "expires_in": time_to_live}
    # For convenience, return both the server response and the access_token.
    return (json.dumps(token_server_response), access_token)


@responses.activate
def test_request_token():
    (response, access_token) = get_mock_token_response(time.time(), 30)
    responses.add(responses.POST, MOCK_URL + OPERATION_PATH, body=response, status=200)

    token_manager = MCSPTokenManager(apikey="my-api-key", url=MOCK_URL, disable_ssl_verification=True)
    token = token_manager.get_token()

    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == MOCK_URL + OPERATION_PATH
    assert token == access_token


@responses.activate
def test_request_token_unsuccessful():
    response = """{
        "errorCode": "BXNIM0415E",
        "errorMessage": "Provided API key could not be found"
    }
    """
    responses.add(responses.POST, url=MOCK_URL + OPERATION_PATH, body=response, status=400)

    token_manager = MCSPTokenManager(apikey="bad-api-key", url=MOCK_URL)
    with pytest.raises(ApiException):
        token_manager.request_token()

    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == MOCK_URL + OPERATION_PATH
    assert responses.calls[0].response.text == response
