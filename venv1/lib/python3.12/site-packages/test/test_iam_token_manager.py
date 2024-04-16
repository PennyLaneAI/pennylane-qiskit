# coding: utf-8

# Copyright 2021, 2024 IBM All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=missing-docstring
import os
import time

import jwt
import pytest
import responses

from ibm_cloud_sdk_core import IAMTokenManager, ApiException, get_authenticator_from_environment

# pylint: disable=line-too-long
TEST_ACCESS_TOKEN_1 = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6ImhlbGxvIiwicm9sZSI6InVzZXIiLCJwZXJtaXNzaW9ucyI6WyJhZG1pbmlzdHJhdG9yIiwiZGVwbG95bWVudF9hZG1pbiJdLCJzdWIiOiJoZWxsbyIsImlzcyI6IkpvaG4iLCJhdWQiOiJEU1giLCJ1aWQiOiI5OTkiLCJpYXQiOjE1NjAyNzcwNTEsImV4cCI6MTU2MDI4MTgxOSwianRpIjoiMDRkMjBiMjUtZWUyZC00MDBmLTg2MjMtOGNkODA3MGI1NDY4In0.cIodB4I6CCcX8vfIImz7Cytux3GpWyObt9Gkur5g1QI'
TEST_ACCESS_TOKEN_2 = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiIsImtpZCI6IjIzMDQ5ODE1MWMyMTRiNzg4ZGQ5N2YyMmI4NTQxMGE1In0.eyJ1c2VybmFtZSI6ImR1bW15Iiwicm9sZSI6IkFkbWluIiwicGVybWlzc2lvbnMiOlsiYWRtaW5pc3RyYXRvciIsIm1hbmFnZV9jYXRhbG9nIl0sInN1YiI6ImFkbWluIiwiaXNzIjoic3NzIiwiYXVkIjoic3NzIiwidWlkIjoic3NzIiwiaWF0IjozNjAwLCJleHAiOjE2MjgwMDcwODF9.zvUDpgqWIWs7S1CuKv40ERw1IZ5FqSFqQXsrwZJyfRM'
TEST_REFRESH_TOKEN = 'Xj7Gle500MachEOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6ImhlbGxvIiwicm9sZSI6InVzZXIiLCJwZXJtaXNzaW9ucyI6WyJhZG1pbmlzdHJhdG9yIiwiZGVwbG95bWVudF9hZG1pbiJdLCJzdWIiOiJoZWxsbyIsImlzcyI6IkpvaG4iLCJhdWQiOiJEU1giLCJ1aWQiOiI5OTkiLCJpYXQiOjE1NjAyNzcwNTEsImV4cCI6MTU2MDI4MTgxOSwianRpIjoiMDRkMjBiMjUtZWUyZC00MDBmLTg2MjMtOGNkODA3MGI1NDY4In0.cIodB4I6CCcX8vfIImz7Cytux3GpWyObt9Gkur5g1QI'
EXPIRATION_WINDOW = 10


def _get_current_time() -> int:
    return int(time.time())


def get_access_token() -> str:
    access_token_layout = {
        "username": "dummy",
        "role": "Admin",
        "permissions": ["administrator", "manage_catalog"],
        "sub": "admin",
        "iss": "sss",
        "aud": "sss",
        "uid": "sss",
        "iat": 3600,
        "exp": int(time.time()),
    }

    access_token = jwt.encode(
        access_token_layout, 'secret', algorithm='HS256', headers={'kid': '230498151c214b788dd97f22b85410a5'}
    )
    return access_token


@responses.activate
def test_request_token_auth_default():
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    response = """{
        "access_token": "oAeisG8yqPY7sFR_x66Z15",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expiration": 1524167011,
        "refresh_token": "jy4gl91BQ"
    }"""
    responses.add(responses.POST, url=iam_url, body=response, status=200)

    token_manager = IAMTokenManager("apikey")
    token_manager.request_token()

    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == iam_url
    assert responses.calls[0].request.headers.get('Authorization') is None
    assert responses.calls[0].response.text == response


@responses.activate
def test_request_token_auth_in_ctor():
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    response = """{
        "access_token": "oAeisG8yqPY7sFR_x66Z15",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expiration": 1524167011,
        "refresh_token": "jy4gl91BQ"
    }"""
    default_auth_header = 'Basic Yng6Yng='
    responses.add(responses.POST, url=iam_url, body=response, status=200)

    token_manager = IAMTokenManager("apikey", url=iam_url, client_id='foo', client_secret='bar')
    token_manager.request_token()

    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == iam_url
    assert responses.calls[0].request.headers['Authorization'] != default_auth_header
    assert responses.calls[0].response.text == response
    assert 'scope' not in responses.calls[0].response.request.body


@responses.activate
def test_request_token_auth_in_ctor_with_scope():
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    response = """{
        "access_token": "oAeisG8yqPY7sFR_x66Z15",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expiration": 1524167011,
        "refresh_token": "jy4gl91BQ"
    }"""
    default_auth_header = 'Basic Yng6Yng='
    responses.add(responses.POST, url=iam_url, body=response, status=200)

    token_manager = IAMTokenManager("apikey", url=iam_url, client_id='foo', client_secret='bar', scope='john snow')
    token_manager.request_token()

    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == iam_url
    assert responses.calls[0].request.headers['Authorization'] != default_auth_header
    assert responses.calls[0].response.text == response
    assert 'scope=john+snow' in responses.calls[0].response.request.body


@responses.activate
def test_request_token_unsuccessful():
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    response = """{
        "context": {
            "requestId": "38a0e9c226d94764820d92aa623eb0f6",
            "requestType": "incoming.Identity_Token",
            "userAgent": "ibm-python-sdk-core-1.0.0",
            "url": "https://iam.cloud.ibm.com",
            "instanceId": "iamid-4.5-6788-90b137c-75f48695b5-kl4wx",
            "threadId": "169de5",
            "host": "iamid-4.5-6788-90b137c-75f48695b5-kl4wx",
            "startTime": "29.10.2019 12:31:00:300 GMT",
            "endTime": "29.10.2019 12:31:00:381 GMT",
            "elapsedTime": "81",
            "locale": "en_US",
            "clusterName": "iam-id-prdal12-8brn"
        },
        "errorCode": "BXNIM0415E",
        "errorMessage": "Provided API key could not be found"
    }
    """
    responses.add(responses.POST, url=iam_url, body=response, status=400)

    token_manager = IAMTokenManager("apikey")
    with pytest.raises(ApiException):
        token_manager.request_token()

    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == iam_url
    assert responses.calls[0].response.text == response


@responses.activate
def test_request_token_auth_in_ctor_client_id_only():
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    response = """{
        "access_token": "oAeisG8yqPY7sFR_x66Z15",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expiration": 1524167011,
        "refresh_token": "jy4gl91BQ"
    }"""
    responses.add(responses.POST, url=iam_url, body=response, status=200)

    token_manager = IAMTokenManager("iam_apikey", url=iam_url, client_id='foo')
    token_manager.request_token()

    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == iam_url
    assert responses.calls[0].request.headers.get('Authorization') is None
    assert responses.calls[0].response.text == response
    assert 'scope' not in responses.calls[0].response.request.body


@responses.activate
def test_request_token_auth_in_ctor_secret_only():
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    response = """{
        "access_token": "oAeisG8yqPY7sFR_x66Z15",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expiration": 1524167011,
        "refresh_token": "jy4gl91BQ"
    }"""
    responses.add(responses.POST, url=iam_url, body=response, status=200)

    token_manager = IAMTokenManager("iam_apikey", url=iam_url, client_id=None, client_secret='bar')
    token_manager.request_token()

    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == iam_url
    assert responses.calls[0].request.headers.get('Authorization') is None
    assert responses.calls[0].response.text == response
    assert 'scope' not in responses.calls[0].response.request.body


@responses.activate
def test_request_token_auth_in_setter():
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    response = """{
        "access_token": "oAeisG8yqPY7sFR_x66Z15",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expiration": 1524167011,
        "refresh_token": "jy4gl91BQ"
    }"""
    default_auth_header = 'Basic Yng6Yng='
    responses.add(responses.POST, url=iam_url, body=response, status=200)

    token_manager = IAMTokenManager("iam_apikey")
    token_manager.set_client_id_and_secret('foo', 'bar')
    token_manager.request_token()

    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == iam_url
    assert responses.calls[0].request.headers['Authorization'] != default_auth_header
    assert responses.calls[0].response.text == response
    assert 'scope' not in responses.calls[0].response.request.body


@responses.activate
def test_request_token_auth_in_setter_client_id_only():
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    response = """{
        "access_token": "oAeisG8yqPY7sFR_x66Z15",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expiration": 1524167011,
        "refresh_token": "jy4gl91BQ"
    }"""
    responses.add(responses.POST, url=iam_url, body=response, status=200)

    token_manager = IAMTokenManager("iam_apikey")
    token_manager.set_client_id_and_secret('foo', None)
    token_manager.request_token()

    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == iam_url
    assert responses.calls[0].request.headers.get('Authorization') is None
    assert responses.calls[0].response.text == response
    assert 'scope' not in responses.calls[0].response.request.body


@responses.activate
def test_request_token_auth_in_setter_secret_only():
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    response = """{
        "access_token": "oAeisG8yqPY7sFR_x66Z15",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expiration": 1524167011,
        "refresh_token": "jy4gl91BQ"
    }"""
    responses.add(responses.POST, url=iam_url, body=response, status=200)

    token_manager = IAMTokenManager("iam_apikey")
    token_manager.set_client_id_and_secret(None, 'bar')
    token_manager.set_headers({'user': 'header'})
    token_manager.request_token()

    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == iam_url
    assert responses.calls[0].request.headers.get('Authorization') is None
    assert responses.calls[0].response.text == response
    assert 'scope' not in responses.calls[0].response.request.body


@responses.activate
def test_request_token_auth_in_setter_scope():
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    response = """{
        "access_token": "oAeisG8yqPY7sFR_x66Z15",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expiration": 1524167011,
        "refresh_token": "jy4gl91BQ"
    }"""
    responses.add(responses.POST, url=iam_url, body=response, status=200)

    token_manager = IAMTokenManager("iam_apikey")
    token_manager.set_client_id_and_secret(None, 'bar')
    token_manager.set_headers({'user': 'header'})
    token_manager.set_scope('john snow')
    token_manager.request_token()

    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == iam_url
    assert responses.calls[0].request.headers.get('Authorization') is None
    assert responses.calls[0].response.text == response
    assert 'scope=john+snow' in responses.calls[0].response.request.body


@responses.activate
def test_get_token_success():
    iam_url = "https://iam.cloud.ibm.com/identity/token"

    # Create two mock responses with different access tokens.
    response1 = """{
        "access_token": "%s",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expiration": 1600003600,
        "refresh_token": "jy4gl91BQ"
    }""" % (
        TEST_ACCESS_TOKEN_1
    )
    response2 = """{
        "access_token": "%s",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expiration": 1600007200,
        "refresh_token": "jy4gl91BQ"
    }""" % (
        TEST_ACCESS_TOKEN_2
    )

    token_manager = IAMTokenManager("iam_apikey")

    access_token = token_manager.access_token
    assert access_token is None

    responses.add(responses.POST, url=iam_url, body=response1, status=200)
    access_token = token_manager.get_token()
    assert access_token == TEST_ACCESS_TOKEN_1
    assert token_manager.access_token == TEST_ACCESS_TOKEN_1

    # Verify that the token manager returns the cached value.
    # Before we call `get_token` again, set the expiration and refresh time
    # so that we do not fetch a new access token.
    # This is necessary because we are using a fixed JWT response.
    token_manager.expire_time = _get_current_time() + 1000
    token_manager.refresh_time = _get_current_time() + 1000
    access_token = token_manager.get_token()
    assert access_token == TEST_ACCESS_TOKEN_1
    assert token_manager.access_token == TEST_ACCESS_TOKEN_1

    # Force expiration to get the second token.
    # We'll set the expiration time to be current-time + EXPIRATION_WINDOW (10 secs)
    # because we want the access token to be considered as "expired"
    # when we reach the IAM-server reported expiration time minus 10 secs.
    responses.add(responses.POST, url=iam_url, body=response2, status=200)
    token_manager.expire_time = _get_current_time() + EXPIRATION_WINDOW
    token_manager.refresh_time = _get_current_time() + 1000
    access_token = token_manager.get_token()
    assert access_token == TEST_ACCESS_TOKEN_2
    assert token_manager.access_token == TEST_ACCESS_TOKEN_2


@responses.activate
def test_get_refresh_token():
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    access_token_str = get_access_token()
    response = """{
        "access_token": "%s",
        "token_type": "Bearer",
        "expires_in": 3600,
        "expiration": 1524167011,
        "refresh_token": "jy4gl91BQ"
    }""" % (
        access_token_str
    )
    responses.add(responses.POST, url=iam_url, body=response, status=200)

    token_manager = IAMTokenManager("iam_apikey")
    token_manager.get_token()

    assert len(responses.calls) == 2
    assert token_manager.refresh_token == "jy4gl91BQ"


#
# In order to run the following integration test with a live IAM server:
#
# 1. Create file "iamtest.env" in the project root.
# It should look like this:
# IAMTEST1_AUTH_URL=<url> e.g. https://iam.cloud.ibm.com
# IAMTEST1_AUTH_TYPE=iam
# IAMTEST1_APIKEY=<apikey>
# IAMTEST2_AUTH_URL=<url> e.g. https://iam.test.cloud.ibm.com
# IAMTEST2_AUTH_TYPE=iam
# IAMTEST2_APIKEY=<apikey>
# IAMTEST2_CLIENT_ID=<client id>
# IAMTEST2_CLIENT_SECRET=<client secret>
#
# 2. Comment out the "@pytest.mark.skip" decorator below.
#
# 3. Run this command:
# python3 -m pytest -s test -k "test_iam_live_token_server"
# (or just run tests like normal and this test function will be invoked)
#


@pytest.mark.skip(reason="avoid integration test in automated builds")
def test_iam_live_token_server():
    # Get two iam authenticators from the environment.
    # "iamtest1" uses the production IAM token server
    # "iamtest2" uses the staging IAM token server
    os.environ['IBM_CREDENTIALS_FILE'] = "iamtest.env"

    # Test "iamtest1" service
    auth1 = get_authenticator_from_environment("iamtest1")
    assert auth1 is not None
    assert auth1.token_manager is not None
    assert auth1.token_manager.url is not None

    request = {'method': "GET"}
    request["url"] = ""
    request["headers"] = {}

    assert auth1.token_manager.refresh_token is None

    auth1.authenticate(request)

    assert request.get("headers") is not None
    assert request["headers"].get("Authorization") is not None
    assert "Bearer " in request["headers"].get("Authorization")

    # Test "iamtest2" service
    auth2 = get_authenticator_from_environment("iamtest2")
    assert auth2 is not None
    assert auth2.token_manager is not None
    assert auth2.token_manager.url is not None

    request = {'method': "GET"}
    request["url"] = ""
    request["headers"] = {}

    assert auth2.token_manager.refresh_token is None

    auth2.authenticate(request)

    assert auth2.token_manager.refresh_token is not None

    assert request.get("headers") is not None
    assert request["headers"].get("Authorization") is not None
    assert "Bearer " in request["headers"].get("Authorization")

    # print('Refresh token: ', auth2.token_manager.refresh_token)
