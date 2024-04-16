# pylint: disable=missing-docstring
import os

from ibm_cloud_sdk_core import get_authenticator_from_environment

# Note: Only the unit tests are run by default.
#
# In order to test with a live MCSP token server, create file "mcsptest.env" in the project root.
# It should look like this:
#
# 	MCSPTEST_AUTH_URL=<url>   e.g. https://iam.cloud.ibm.com
# 	MCSPTEST_AUTH_TYPE=mcsp
# 	MCSPTEST_APIKEY=<apikey>
#
# Then run this command:
# pytest test_integration/test_mcsp_authenticator_integration.py


def test_mcsp_authenticator():
    os.environ['IBM_CREDENTIALS_FILE'] = 'mcsptest.env'

    authenticator = get_authenticator_from_environment('mcsptest1')
    assert authenticator is not None

    request = {'headers': {}}
    authenticator.authenticate(request)
    assert request['headers']['Authorization'] is not None
    assert 'Bearer' in request['headers']['Authorization']
