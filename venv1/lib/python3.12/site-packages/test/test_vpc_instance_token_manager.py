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
import json
import logging
import time

import pytest
import responses

from ibm_cloud_sdk_core import ApiException, VPCInstanceTokenManager


# pylint: disable=line-too-long
TEST_ACCESS_TOKEN_1 = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6ImhlbGxvIiwicm9sZSI6InVzZXIiLCJwZXJtaXNzaW9ucyI6WyJhZG1pbmlzdHJhdG9yIiwiZGVwbG95bWVudF9hZG1pbiJdLCJzdWIiOiJoZWxsbyIsImlzcyI6IkpvaG4iLCJhdWQiOiJEU1giLCJ1aWQiOiI5OTkiLCJpYXQiOjE1NjAyNzcwNTEsImV4cCI6MTU2MDI4MTgxOSwianRpIjoiMDRkMjBiMjUtZWUyZC00MDBmLTg2MjMtOGNkODA3MGI1NDY4In0.cIodB4I6CCcX8vfIImz7Cytux3GpWyObt9Gkur5g1QI'
TEST_ACCESS_TOKEN_2 = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiIsImtpZCI6IjIzMDQ5ODE1MWMyMTRiNzg4ZGQ5N2YyMmI4NTQxMGE1In0.eyJ1c2VybmFtZSI6ImR1bW15Iiwicm9sZSI6IkFkbWluIiwicGVybWlzc2lvbnMiOlsiYWRtaW5pc3RyYXRvciIsIm1hbmFnZV9jYXRhbG9nIl0sInN1YiI6ImFkbWluIiwiaXNzIjoic3NzIiwiYXVkIjoic3NzIiwidWlkIjoic3NzIiwiaWF0IjozNjAwLCJleHAiOjE2MjgwMDcwODF9.zvUDpgqWIWs7S1CuKv40ERw1IZ5FqSFqQXsrwZJyfRM'
TEST_TOKEN = 'abc123'
TEST_IAM_TOKEN = 'iam-abc123'
TEST_IAM_PROFILE_CRN = 'crn:iam-profile:123'
TEST_IAM_PROFILE_ID = 'iam-id-123'
EXPIRATION_WINDOW = 10


def _get_current_time() -> int:
    return int(time.time())


def test_constructor():
    token_manager = VPCInstanceTokenManager(
        iam_profile_crn=TEST_IAM_PROFILE_CRN,
    )

    assert token_manager.iam_profile_crn is TEST_IAM_PROFILE_CRN
    assert token_manager.iam_profile_id is None
    assert token_manager.access_token is None


def test_setters():
    token_manager = VPCInstanceTokenManager(
        iam_profile_crn=TEST_IAM_PROFILE_CRN,
    )

    assert token_manager.iam_profile_crn is TEST_IAM_PROFILE_CRN
    assert token_manager.iam_profile_id is None
    assert token_manager.access_token is None

    token_manager.set_iam_profile_crn(None)
    assert token_manager.iam_profile_crn is None

    token_manager.set_iam_profile_id(TEST_IAM_PROFILE_ID)
    assert token_manager.iam_profile_id == TEST_IAM_PROFILE_ID


@responses.activate
def test_retrieve_instance_identity_token(caplog):
    caplog.set_level(logging.DEBUG)

    token_manager = VPCInstanceTokenManager(
        iam_profile_crn=TEST_IAM_PROFILE_CRN,
        url='http://someurl.com',
    )

    response = {
        'access_token': TEST_TOKEN,
    }

    responses.add(responses.PUT, 'http://someurl.com/instance_identity/v1/token', body=json.dumps(response), status=200)

    ii_token = token_manager.retrieve_instance_identity_token()
    assert len(responses.calls) == 1
    assert responses.calls[0].request.headers['Content-Type'] == 'application/json'
    assert responses.calls[0].request.headers['Accept'] == 'application/json'
    assert responses.calls[0].request.headers['Metadata-Flavor'] == 'ibm'
    assert responses.calls[0].request.params['version'] == '2022-03-01'
    assert responses.calls[0].request.body == '{"expires_in": 300}'
    assert ii_token == TEST_TOKEN
    # Check the logs.
    # pylint: disable=line-too-long
    assert (
        caplog.record_tuples[0][2]
        == 'Invoking VPC \'create_access_token\' operation: http://someurl.com/instance_identity/v1/token'
    )
    assert caplog.record_tuples[1][2] == 'Returned from VPC \'create_access_token\' operation."'


@responses.activate
def test_retrieve_instance_identity_token_failed(caplog):
    caplog.set_level(logging.DEBUG)

    token_manager = VPCInstanceTokenManager(
        iam_profile_crn=TEST_IAM_PROFILE_CRN,
        url='http://someurl.com',
    )

    response = {
        'errors': ['Ooops'],
    }

    responses.add(responses.PUT, 'http://someurl.com/instance_identity/v1/token', body=json.dumps(response), status=400)

    with pytest.raises(ApiException):
        token_manager.retrieve_instance_identity_token()

    assert len(responses.calls) == 1
    # Check the logs.
    # pylint: disable=line-too-long
    assert (
        caplog.record_tuples[0][2]
        == 'Invoking VPC \'create_access_token\' operation: http://someurl.com/instance_identity/v1/token'
    )


@responses.activate
def test_request_token_with_crn(caplog):
    caplog.set_level(logging.DEBUG)

    token_manager = VPCInstanceTokenManager(
        iam_profile_crn=TEST_IAM_PROFILE_CRN,
    )

    # Mock the retrieve instance identity token method.
    def mock_retrieve_instance_identity_token():
        return TEST_TOKEN

    token_manager.retrieve_instance_identity_token = mock_retrieve_instance_identity_token

    response = {
        'access_token': TEST_IAM_TOKEN,
    }

    responses.add(
        responses.POST, 'http://169.254.169.254/instance_identity/v1/iam_token', body=json.dumps(response), status=200
    )

    response = token_manager.request_token()
    assert len(responses.calls) == 1
    assert responses.calls[0].request.headers['Content-Type'] == 'application/json'
    assert responses.calls[0].request.headers['Accept'] == 'application/json'
    assert responses.calls[0].request.headers['Authorization'] == 'Bearer ' + TEST_TOKEN
    assert responses.calls[0].request.body == '{"trusted_profile": {"crn": "crn:iam-profile:123"}}'
    assert responses.calls[0].request.params['version'] == '2022-03-01'
    # Check the logs.
    # pylint: disable=line-too-long
    assert (
        caplog.record_tuples[0][2]
        == 'Invoking VPC \'create_iam_token\' operation: http://169.254.169.254/instance_identity/v1/iam_token'
    )
    assert caplog.record_tuples[1][2] == 'Returned from VPC \'create_iam_token\' operation."'


@responses.activate
def test_request_token_with_id(caplog):
    caplog.set_level(logging.DEBUG)

    token_manager = VPCInstanceTokenManager(
        iam_profile_id=TEST_IAM_PROFILE_ID,
    )

    # Mock the retrieve instance identity token method.
    def mock_retrieve_instance_identity_token():
        return TEST_TOKEN

    token_manager.retrieve_instance_identity_token = mock_retrieve_instance_identity_token

    response = {
        'access_token': TEST_IAM_TOKEN,
    }

    responses.add(
        responses.POST, 'http://169.254.169.254/instance_identity/v1/iam_token', body=json.dumps(response), status=200
    )

    response = token_manager.request_token()
    assert len(responses.calls) == 1
    assert responses.calls[0].request.headers['Content-Type'] == 'application/json'
    assert responses.calls[0].request.headers['Accept'] == 'application/json'
    assert responses.calls[0].request.headers['Authorization'] == 'Bearer ' + TEST_TOKEN
    assert responses.calls[0].request.body == '{"trusted_profile": {"id": "iam-id-123"}}'
    assert responses.calls[0].request.params['version'] == '2022-03-01'
    # Check the logs.
    # pylint: disable=line-too-long
    assert (
        caplog.record_tuples[0][2]
        == 'Invoking VPC \'create_iam_token\' operation: http://169.254.169.254/instance_identity/v1/iam_token'
    )
    assert caplog.record_tuples[1][2] == 'Returned from VPC \'create_iam_token\' operation."'


@responses.activate
def test_request_token(caplog):
    caplog.set_level(logging.DEBUG)

    token_manager = VPCInstanceTokenManager()

    # Mock the retrieve instance identity token method.
    def mock_retrieve_instance_identity_token():
        return TEST_TOKEN

    token_manager.retrieve_instance_identity_token = mock_retrieve_instance_identity_token

    response = {
        'access_token': TEST_IAM_TOKEN,
    }

    responses.add(
        responses.POST, 'http://169.254.169.254/instance_identity/v1/iam_token', body=json.dumps(response), status=200
    )

    response = token_manager.request_token()
    assert len(responses.calls) == 1
    assert responses.calls[0].request.headers['Content-Type'] == 'application/json'
    assert responses.calls[0].request.headers['Accept'] == 'application/json'
    assert responses.calls[0].request.headers['Authorization'] == 'Bearer ' + TEST_TOKEN
    assert responses.calls[0].request.body is None
    assert responses.calls[0].request.params['version'] == '2022-03-01'
    # Check the logs.
    # pylint: disable=line-too-long
    assert (
        caplog.record_tuples[0][2]
        == 'Invoking VPC \'create_iam_token\' operation: http://169.254.169.254/instance_identity/v1/iam_token'
    )
    assert caplog.record_tuples[1][2] == 'Returned from VPC \'create_iam_token\' operation."'


@responses.activate
def test_request_token_failed(caplog):
    caplog.set_level(logging.DEBUG)

    token_manager = VPCInstanceTokenManager(
        iam_profile_id=TEST_IAM_PROFILE_ID,
    )

    # Mock the retrieve instance identity token method.
    def mock_retrieve_instance_identity_token():
        return TEST_TOKEN

    token_manager.retrieve_instance_identity_token = mock_retrieve_instance_identity_token

    response = {
        'errors': ['Ooops'],
    }

    responses.add(
        responses.POST, 'http://169.254.169.254/instance_identity/v1/iam_token', body=json.dumps(response), status=400
    )

    with pytest.raises(ApiException):
        token_manager.request_token()
    assert len(responses.calls) == 1
    # Check the logs.
    # pylint: disable=line-too-long
    assert (
        caplog.record_tuples[0][2]
        == 'Invoking VPC \'create_iam_token\' operation: http://169.254.169.254/instance_identity/v1/iam_token'
    )


@responses.activate
def test_access_token():
    token_manager = VPCInstanceTokenManager(
        iam_profile_id=TEST_IAM_PROFILE_ID,
    )

    response_ii = {
        'access_token': TEST_TOKEN,
    }
    response_iam = {
        'access_token': TEST_ACCESS_TOKEN_1,
    }

    responses.add(
        responses.PUT, 'http://169.254.169.254/instance_identity/v1/token', body=json.dumps(response_ii), status=200
    )
    responses.add(
        responses.POST,
        'http://169.254.169.254/instance_identity/v1/iam_token',
        body=json.dumps(response_iam),
        status=200,
    )

    assert token_manager.access_token is None
    assert token_manager.expire_time == 0
    assert token_manager.refresh_time == 0

    token_manager.get_token()
    assert token_manager.access_token == TEST_ACCESS_TOKEN_1
    assert token_manager.expire_time > 0
    assert token_manager.refresh_time > 0


@responses.activate
def test_get_token_success():
    token_manager = VPCInstanceTokenManager()

    # Mock the retrieve instance identity token method.
    def mock_retrieve_instance_identity_token():
        return TEST_TOKEN

    token_manager.retrieve_instance_identity_token = mock_retrieve_instance_identity_token

    response1 = {
        'access_token': TEST_ACCESS_TOKEN_1,
    }
    response2 = {
        'access_token': TEST_ACCESS_TOKEN_2,
    }

    responses.add(
        responses.POST, 'http://169.254.169.254/instance_identity/v1/iam_token', body=json.dumps(response1), status=200
    )

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
    responses.add(
        responses.POST, 'http://169.254.169.254/instance_identity/v1/iam_token', body=json.dumps(response2), status=200
    )
    token_manager.expire_time = _get_current_time() + EXPIRATION_WINDOW
    access_token = token_manager.get_token()
    assert access_token == TEST_ACCESS_TOKEN_2
    assert token_manager.access_token == TEST_ACCESS_TOKEN_2
