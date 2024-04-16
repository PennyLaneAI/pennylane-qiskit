# pylint: disable=missing-docstring
import json
import time
import jwt
import pytest
import responses

from ibm_cloud_sdk_core.authenticators import MCSPAuthenticator, Authenticator


OPERATION_PATH = '/siusermgr/api/1.0/apikeys/token'
MOCK_URL = 'https://mcsp.ibm.com'


def test_mcsp_authenticator():
    authenticator = MCSPAuthenticator('my-api-key', MOCK_URL)
    assert authenticator is not None
    assert authenticator.authentication_type() == Authenticator.AUTHTYPE_MCSP
    assert authenticator.token_manager.url == MOCK_URL
    assert authenticator.token_manager.disable_ssl_verification is False
    assert authenticator.token_manager.headers == {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
    }
    assert authenticator.token_manager.proxies is None

    authenticator.set_disable_ssl_verification(True)
    assert authenticator.token_manager.disable_ssl_verification is True

    with pytest.raises(TypeError) as err:
        authenticator.set_headers('dummy')
    assert str(err.value) == 'headers must be a dictionary'

    authenticator.set_headers({'dummy': 'headers'})
    assert authenticator.token_manager.headers == {'dummy': 'headers'}

    with pytest.raises(TypeError) as err:
        authenticator.set_proxies('dummy')
    assert str(err.value) == 'proxies must be a dictionary'

    authenticator.set_proxies({'dummy': 'proxies'})
    assert authenticator.token_manager.proxies == {'dummy': 'proxies'}


def test_disable_ssl_verification():
    authenticator = MCSPAuthenticator('my-api-key', MOCK_URL, disable_ssl_verification=True)
    assert authenticator.token_manager.disable_ssl_verification is True

    authenticator.set_disable_ssl_verification(False)
    assert authenticator.token_manager.disable_ssl_verification is False


def test_invalid_disable_ssl_verification_type():
    with pytest.raises(TypeError) as err:
        authenticator = MCSPAuthenticator('my-api-key', MOCK_URL, disable_ssl_verification='True')
    assert str(err.value) == 'disable_ssl_verification must be a bool'

    authenticator = MCSPAuthenticator('my-api-key', MOCK_URL)
    assert authenticator.token_manager.disable_ssl_verification is False

    with pytest.raises(TypeError) as err:
        authenticator.set_disable_ssl_verification('True')
    assert str(err.value) == 'status must be a bool'


def test_mcsp_authenticator_validate_failed():
    with pytest.raises(ValueError) as err:
        MCSPAuthenticator(apikey=None, url=MOCK_URL)
    assert str(err.value) == 'The apikey shouldn\'t be None.'

    with pytest.raises(ValueError) as err:
        MCSPAuthenticator(apikey='my-api-key', url=None)
    assert str(err.value) == 'The url shouldn\'t be None.'


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
def test_get_token():
    (response, access_token) = get_mock_token_response(time.time(), 7200)
    responses.add(responses.POST, MOCK_URL + OPERATION_PATH, body=response, status=200)

    auth_headers = {'Host': 'mcsp.cloud.ibm.com:443'}
    authenticator = MCSPAuthenticator(apikey='my-api-key', url=MOCK_URL, headers=auth_headers)

    # Authenticate the request and verify the Authorization header.
    request = {'headers': {}}
    authenticator.authenticate(request)
    assert request['headers']['Authorization'] == 'Bearer ' + access_token

    # Verify that the "get token" request contained the Host header.
    assert responses.calls[0].request.headers.get('Host') == 'mcsp.cloud.ibm.com:443'


@responses.activate
def test_get_token_cached():
    (response, access_token) = get_mock_token_response(time.time(), 7200)
    responses.add(responses.POST, MOCK_URL + OPERATION_PATH, body=response, status=200)

    authenticator = MCSPAuthenticator(apikey='my-api-key', url=MOCK_URL)

    # Authenticate the request and verify the Authorization header.
    request = {'headers': {}}
    authenticator.authenticate(request)
    assert request['headers']['Authorization'] == 'Bearer ' + access_token

    # Authenticate a second request and verify that we used the same access token.
    request = {'headers': {}}
    authenticator.authenticate(request)
    assert request['headers']['Authorization'] == 'Bearer ' + access_token


@responses.activate
def test_get_token_background_refresh():
    t1 = time.time()
    t2 = t1 + 7200

    # Setup the first token response.
    (response1, access_token1) = get_mock_token_response(t1, 7200)
    responses.add(responses.POST, MOCK_URL + OPERATION_PATH, body=response1, status=200)

    # Setup the second token response.
    (response2, access_token2) = get_mock_token_response(t2, 7200)
    responses.add(responses.POST, MOCK_URL + OPERATION_PATH, body=response2, status=200)

    authenticator = MCSPAuthenticator(apikey="my-api-key", url=MOCK_URL)

    # Authenticate the request and verify that the first access_token is used.
    request = {'headers': {}}
    authenticator.authenticate(request)
    assert request['headers']['Authorization'] == 'Bearer ' + access_token1

    # Now put the token manager in the refresh window to trigger a background refresh scenario.
    authenticator.token_manager.refresh_time = t1 - 1

    # Authenticate a second request and verify that the correct access token is used.
    # Note: Ideally, the token manager would trigger the refresh in a separate thread
    # and it "should" return the first access token for this second authentication request
    # while the token manager is obtaining a new access token.
    # Unfortunately, the TokenManager class  method does the refresh request synchronously,
    # so we get back the second access token here instead.
    # If we "fix" the TokenManager class to refresh asynchronously, we'll need to
    # change this test case to expect the first access token here.
    request = {'headers': {}}
    authenticator.authenticate(request)
    assert request['headers']['Authorization'] == 'Bearer ' + access_token2

    # Wait for the background refresh to finish.
    # No need to wait due to the synchronous logic in the TokenManager class mentioned above.
    # time.sleep(2)

    # Authenticate another request and verify that the second access token is used again.
    request = {'headers': {}}
    authenticator.authenticate(request)
    assert request['headers']['Authorization'] == 'Bearer ' + access_token2
