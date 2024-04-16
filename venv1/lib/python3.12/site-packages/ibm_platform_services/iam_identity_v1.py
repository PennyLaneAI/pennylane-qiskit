# coding: utf-8

# (C) Copyright IBM Corp. 2024.
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

# IBM OpenAPI SDK Code Generator Version: 3.86.2-8b8592a4-20240313-204553

"""
The IAM Identity Service API allows for the management of Account Settings and Identities
(Service IDs, ApiKeys).

API Version: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import json

from ibm_cloud_sdk_core import BaseService, DetailedResponse
from ibm_cloud_sdk_core.authenticators.authenticator import Authenticator
from ibm_cloud_sdk_core.get_authenticator import get_authenticator_from_environment
from ibm_cloud_sdk_core.utils import convert_model, datetime_to_string, string_to_datetime

from .common import get_sdk_headers

##############################################################################
# Service
##############################################################################


class IamIdentityV1(BaseService):
    """The iam_identity V1 service."""

    DEFAULT_SERVICE_URL = 'https://iam.cloud.ibm.com'
    DEFAULT_SERVICE_NAME = 'iam_identity'

    @classmethod
    def new_instance(
        cls,
        service_name: str = DEFAULT_SERVICE_NAME,
    ) -> 'IamIdentityV1':
        """
        Return a new client for the iam_identity service using the specified
               parameters and external configuration.
        """
        authenticator = get_authenticator_from_environment(service_name)
        service = cls(authenticator)
        service.configure_service(service_name)
        return service

    def __init__(
        self,
        authenticator: Authenticator = None,
    ) -> None:
        """
        Construct a new client for the iam_identity service.

        :param Authenticator authenticator: The authenticator specifies the authentication mechanism.
               Get up to date information from https://github.com/IBM/python-sdk-core/blob/main/README.md
               about initializing the authenticator of your choice.
        """
        BaseService.__init__(self, service_url=self.DEFAULT_SERVICE_URL, authenticator=authenticator)

    #########################
    # API key operations
    #########################

    def list_api_keys(
        self,
        *,
        account_id: Optional[str] = None,
        iam_id: Optional[str] = None,
        pagesize: Optional[int] = None,
        pagetoken: Optional[str] = None,
        scope: Optional[str] = None,
        type: Optional[str] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        include_history: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get API keys for a given service or user IAM ID and account ID.

        Returns the list of API key details for a given service or user IAM ID and account
        ID. Users can manage user API keys for themself, or service ID API keys for
        service IDs that are bound to an entity they have access to. In case of service
        IDs and their API keys, a user must be either an account owner, a IBM Cloud org
        manager or IBM Cloud space developer in order to manage service IDs of the entity.

        :param str account_id: (optional) Account ID of the API keys to query. If a
               service IAM ID is specified in iam_id then account_id must match the
               account of the IAM ID. If a user IAM ID is specified in iam_id then then
               account_id must match the account of the Authorization token.
        :param str iam_id: (optional) IAM ID of the API keys to be queried. The IAM
               ID may be that of a user or a service. For a user IAM ID iam_id must match
               the Authorization token.
        :param int pagesize: (optional) Optional size of a single page. Default is
               20 items per page. Valid range is 1 to 100.
        :param str pagetoken: (optional) Optional Prev or Next page token returned
               from a previous query execution. Default is start with first page.
        :param str scope: (optional) Optional parameter to define the scope of the
               queried API keys. Can be 'entity' (default) or 'account'.
        :param str type: (optional) Optional parameter to filter the type of the
               queried API keys. Can be 'user' or 'serviceid'.
        :param str sort: (optional) Optional sort property, valid values are name,
               description, created_at and created_by. If specified, the items are sorted
               by the value of this property.
        :param str order: (optional) Optional sort order, valid values are asc and
               desc. Default: asc.
        :param bool include_history: (optional) Defines if the entity history is
               included in the response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ApiKeyList` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='list_api_keys',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
            'iam_id': iam_id,
            'pagesize': pagesize,
            'pagetoken': pagetoken,
            'scope': scope,
            'type': type,
            'sort': sort,
            'order': order,
            'include_history': include_history,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/apikeys'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def create_api_key(
        self,
        name: str,
        iam_id: str,
        *,
        description: Optional[str] = None,
        account_id: Optional[str] = None,
        apikey: Optional[str] = None,
        store_value: Optional[bool] = None,
        support_sessions: Optional[bool] = None,
        action_when_leaked: Optional[str] = None,
        entity_lock: Optional[str] = None,
        entity_disable: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create an API key.

        Creates an API key for a UserID or service ID. Users can manage user API keys for
        themself, or service ID API keys for service IDs that are bound to an entity they
        have access to.

        :param str name: Name of the API key. The name is not checked for
               uniqueness. Therefore multiple names with the same value can exist. Access
               is done via the UUID of the API key.
        :param str iam_id: The iam_id that this API key authenticates.
        :param str description: (optional) The optional description of the API key.
               The 'description' property is only available if a description was provided
               during a create of an API key.
        :param str account_id: (optional) The account ID of the API key.
        :param str apikey: (optional) You can optionally passthrough the API key
               value for this API key. If passed, a minimum length validation of 32
               characters for that apiKey value is done, i.e. the value can contain any
               characters and can even be non-URL safe, but the minimum length requirement
               must be met. If omitted, the API key management will create an URL safe
               opaque API key value. The value of the API key is checked for uniqueness.
               Ensure enough variations when passing in this value.
        :param bool store_value: (optional) Send true or false to set whether the
               API key value is retrievable in the future by using the Get details of an
               API key request. If you create an API key for a user, you must specify
               `false` or omit the value. We don't allow storing of API keys for users.
        :param bool support_sessions: (optional) Defines if the API key supports
               sessions. Sessions are only supported for user apikeys.
        :param str action_when_leaked: (optional) Defines the action to take when
               API key is leaked, valid values are 'none', 'disable' and 'delete'.
        :param str entity_lock: (optional) Indicates if the API key is locked for
               further write operations. False by default.
        :param str entity_disable: (optional) Indicates if the API key is disabled.
               False by default.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ApiKey` object
        """

        if name is None:
            raise ValueError('name must be provided')
        if iam_id is None:
            raise ValueError('iam_id must be provided')
        headers = {
            'Entity-Lock': entity_lock,
            'Entity-Disable': entity_disable,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='create_api_key',
        )
        headers.update(sdk_headers)

        data = {
            'name': name,
            'iam_id': iam_id,
            'description': description,
            'account_id': account_id,
            'apikey': apikey,
            'store_value': store_value,
            'support_sessions': support_sessions,
            'action_when_leaked': action_when_leaked,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/apikeys'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def get_api_keys_details(
        self,
        *,
        iam_api_key: Optional[str] = None,
        include_history: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get details of an API key by its value.

        Returns the details of an API key by its value. Users can manage user API keys for
        themself, or service ID API keys for service IDs that are bound to an entity they
        have access to.

        :param str iam_api_key: (optional) API key value.
        :param bool include_history: (optional) Defines if the entity history is
               included in the response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ApiKey` object
        """

        headers = {
            'IAM-ApiKey': iam_api_key,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_api_keys_details',
        )
        headers.update(sdk_headers)

        params = {
            'include_history': include_history,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/apikeys/details'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def get_api_key(
        self,
        id: str,
        *,
        include_history: Optional[bool] = None,
        include_activity: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get details of an API key.

        Returns the details of an API key. Users can manage user API keys for themself, or
        service ID API keys for service IDs that are bound to an entity they have access
        to. In case of service IDs and their API keys, a user must be either an account
        owner, a IBM Cloud org manager or IBM Cloud space developer in order to manage
        service IDs of the entity.

        :param str id: Unique ID of the API key.
        :param bool include_history: (optional) Defines if the entity history is
               included in the response.
        :param bool include_activity: (optional) Defines if the entity's activity
               is included in the response. Retrieving activity data is an expensive
               operation, so only request this when needed.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ApiKey` object
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_api_key',
        )
        headers.update(sdk_headers)

        params = {
            'include_history': include_history,
            'include_activity': include_activity,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/apikeys/{id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def update_api_key(
        self,
        id: str,
        if_match: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        support_sessions: Optional[bool] = None,
        action_when_leaked: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Updates an API key.

        Updates properties of an API key. This does NOT affect existing access tokens.
        Their token content will stay unchanged until the access token is refreshed. To
        update an API key, pass the property to be modified. To delete one property's
        value, pass the property with an empty value "".Users can manage user API keys for
        themself, or service ID API keys for service IDs that are bound to an entity they
        have access to.

        :param str id: Unique ID of the API key to be updated.
        :param str if_match: Version of the API key to be updated. Specify the
               version that you retrieved when reading the API key. This value helps
               identifying parallel usage of this API. Pass * to indicate to update any
               version available. This might result in stale updates.
        :param str name: (optional) The name of the API key to update. If specified
               in the request the parameter must not be empty. The name is not checked for
               uniqueness. Failure to this will result in an Error condition.
        :param str description: (optional) The description of the API key to
               update. If specified an empty description will clear the description of the
               API key. If a non empty value is provided the API key will be updated.
        :param bool support_sessions: (optional) Defines if the API key supports
               sessions. Sessions are only supported for user apikeys.
        :param str action_when_leaked: (optional) Defines the action to take when
               API key is leaked, valid values are 'none', 'disable' and 'delete'.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ApiKey` object
        """

        if not id:
            raise ValueError('id must be provided')
        if not if_match:
            raise ValueError('if_match must be provided')
        headers = {
            'If-Match': if_match,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='update_api_key',
        )
        headers.update(sdk_headers)

        data = {
            'name': name,
            'description': description,
            'support_sessions': support_sessions,
            'action_when_leaked': action_when_leaked,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/apikeys/{id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PUT',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_api_key(
        self,
        id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Deletes an API key.

        Deletes an API key. Existing tokens will remain valid until expired. Users can
        manage user API keys for themself, or service ID API keys for service IDs that are
        bound to an entity they have access to.

        :param str id: Unique ID of the API key.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='delete_api_key',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/apikeys/{id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def lock_api_key(
        self,
        id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Lock the API key.

        Locks an API key by ID. Users can manage user API keys for themself, or service ID
        API keys for service IDs that are bound to an entity they have access to. In case
        of service IDs and their API keys, a user must be either an account owner, a IBM
        Cloud org manager or IBM Cloud space developer in order to manage service IDs of
        the entity.

        :param str id: Unique ID of the API key.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='lock_api_key',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/apikeys/{id}/lock'.format(**path_param_dict)
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def unlock_api_key(
        self,
        id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Unlock the API key.

        Unlocks an API key by ID. Users can manage user API keys for themself, or service
        ID API keys for service IDs that are bound to an entity they have access to. In
        case of service IDs and their API keys, a user must be either an account owner, a
        IBM Cloud org manager or IBM Cloud space developer in order to manage service IDs
        of the entity.

        :param str id: Unique ID of the API key.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='unlock_api_key',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/apikeys/{id}/lock'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def disable_api_key(
        self,
        id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        disable the API key.

        Disable an API key. Users can manage user API keys for themself, or service ID API
        keys for service IDs that are bound to an entity they have access to.

        :param str id: Unique ID of the API key.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='disable_api_key',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/apikeys/{id}/disable'.format(**path_param_dict)
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def enable_api_key(
        self,
        id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Enable the API key.

        Enable an API key. Users can manage user API keys for themself, or service ID API
        keys for service IDs that are bound to an entity they have access to.

        :param str id: Unique ID of the API key.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='enable_api_key',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/apikeys/{id}/disable'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # Service ID operations
    #########################

    def list_service_ids(
        self,
        *,
        account_id: Optional[str] = None,
        name: Optional[str] = None,
        pagesize: Optional[int] = None,
        pagetoken: Optional[str] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        include_history: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List service IDs.

        Returns a list of service IDs. Users can manage user API keys for themself, or
        service ID API keys for service IDs that are bound to an entity they have access
        to. Note: apikey details are only included in the response when creating a Service
        ID with an api key.

        :param str account_id: (optional) Account ID of the service ID(s) to query.
               This parameter is required (unless using a pagetoken).
        :param str name: (optional) Name of the service ID(s) to query. Optional.20
               items per page. Valid range is 1 to 100.
        :param int pagesize: (optional) Optional size of a single page. Default is
               20 items per page. Valid range is 1 to 100.
        :param str pagetoken: (optional) Optional Prev or Next page token returned
               from a previous query execution. Default is start with first page.
        :param str sort: (optional) Optional sort property, valid values are name,
               description, created_at and modified_at. If specified, the items are sorted
               by the value of this property.
        :param str order: (optional) Optional sort order, valid values are asc and
               desc. Default: asc.
        :param bool include_history: (optional) Defines if the entity history is
               included in the response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ServiceIdList` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='list_service_ids',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
            'name': name,
            'pagesize': pagesize,
            'pagetoken': pagetoken,
            'sort': sort,
            'order': order,
            'include_history': include_history,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/serviceids/'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def create_service_id(
        self,
        account_id: str,
        name: str,
        *,
        description: Optional[str] = None,
        unique_instance_crns: Optional[List[str]] = None,
        apikey: Optional['ApiKeyInsideCreateServiceIdRequest'] = None,
        entity_lock: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create a service ID.

        Creates a service ID for an IBM Cloud account. Users can manage user API keys for
        themself, or service ID API keys for service IDs that are bound to an entity they
        have access to.

        :param str account_id: ID of the account the service ID belongs to.
        :param str name: Name of the Service Id. The name is not checked for
               uniqueness. Therefore multiple names with the same value can exist. Access
               is done via the UUID of the Service Id.
        :param str description: (optional) The optional description of the Service
               Id. The 'description' property is only available if a description was
               provided during a create of a Service Id.
        :param List[str] unique_instance_crns: (optional) Optional list of CRNs
               (string array) which point to the services connected to the service ID.
        :param ApiKeyInsideCreateServiceIdRequest apikey: (optional) Parameters for
               the API key in the Create service Id V1 REST request.
        :param str entity_lock: (optional) Indicates if the service ID is locked
               for further write operations. False by default.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ServiceId` object
        """

        if account_id is None:
            raise ValueError('account_id must be provided')
        if name is None:
            raise ValueError('name must be provided')
        if apikey is not None:
            apikey = convert_model(apikey)
        headers = {
            'Entity-Lock': entity_lock,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='create_service_id',
        )
        headers.update(sdk_headers)

        data = {
            'account_id': account_id,
            'name': name,
            'description': description,
            'unique_instance_crns': unique_instance_crns,
            'apikey': apikey,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/serviceids/'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def get_service_id(
        self,
        id: str,
        *,
        include_history: Optional[bool] = None,
        include_activity: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get details of a service ID.

        Returns the details of a service ID. Users can manage user API keys for themself,
        or service ID API keys for service IDs that are bound to an entity they have
        access to. Note: apikey details are only included in the response when creating a
        Service ID with an api key.

        :param str id: Unique ID of the service ID.
        :param bool include_history: (optional) Defines if the entity history is
               included in the response.
        :param bool include_activity: (optional) Defines if the entity's activity
               is included in the response. Retrieving activity data is an expensive
               operation, so only request this when needed.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ServiceId` object
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_service_id',
        )
        headers.update(sdk_headers)

        params = {
            'include_history': include_history,
            'include_activity': include_activity,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/serviceids/{id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def update_service_id(
        self,
        id: str,
        if_match: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        unique_instance_crns: Optional[List[str]] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Update service ID.

        Updates properties of a service ID. This does NOT affect existing access tokens.
        Their token content will stay unchanged until the access token is refreshed. To
        update a service ID, pass the property to be modified. To delete one property's
        value, pass the property with an empty value "".Users can manage user API keys for
        themself, or service ID API keys for service IDs that are bound to an entity they
        have access to. Note: apikey details are only included in the response when
        creating a Service ID with an apikey.

        :param str id: Unique ID of the service ID to be updated.
        :param str if_match: Version of the service ID to be updated. Specify the
               version that you retrieved as entity_tag (ETag header) when reading the
               service ID. This value helps identifying parallel usage of this API. Pass *
               to indicate to update any version available. This might result in stale
               updates.
        :param str name: (optional) The name of the service ID to update. If
               specified in the request the parameter must not be empty. The name is not
               checked for uniqueness. Failure to this will result in an Error condition.
        :param str description: (optional) The description of the service ID to
               update. If specified an empty description will clear the description of the
               service ID. If an non empty value is provided the service ID will be
               updated.
        :param List[str] unique_instance_crns: (optional) List of CRNs which point
               to the services connected to this service ID. If specified an empty list
               will clear all existing unique instance crns of the service ID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ServiceId` object
        """

        if not id:
            raise ValueError('id must be provided')
        if not if_match:
            raise ValueError('if_match must be provided')
        headers = {
            'If-Match': if_match,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='update_service_id',
        )
        headers.update(sdk_headers)

        data = {
            'name': name,
            'description': description,
            'unique_instance_crns': unique_instance_crns,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/serviceids/{id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PUT',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_service_id(
        self,
        id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Deletes a service ID and associated API keys.

        Deletes a service ID and all API keys associated to it. Before deleting the
        service ID, all associated API keys are deleted. In case a Delete Conflict (status
        code 409) a retry of the request may help as the service ID is only deleted if the
        associated API keys were successfully deleted before. Users can manage user API
        keys for themself, or service ID API keys for service IDs that are bound to an
        entity they have access to.

        :param str id: Unique ID of the service ID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='delete_service_id',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/serviceids/{id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def lock_service_id(
        self,
        id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Lock the service ID.

        Locks a service ID by ID. Users can manage user API keys for themself, or service
        ID API keys for service IDs that are bound to an entity they have access to. In
        case of service IDs and their API keys, a user must be either an account owner, a
        IBM Cloud org manager or IBM Cloud space developer in order to manage service IDs
        of the entity.

        :param str id: Unique ID of the service ID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='lock_service_id',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/serviceids/{id}/lock'.format(**path_param_dict)
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def unlock_service_id(
        self,
        id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Unlock the service ID.

        Unlocks a service ID by ID. Users can manage user API keys for themself, or
        service ID API keys for service IDs that are bound to an entity they have access
        to. In case of service IDs and their API keys, a user must be either an account
        owner, a IBM Cloud org manager or IBM Cloud space developer in order to manage
        service IDs of the entity.

        :param str id: Unique ID of the service ID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='unlock_service_id',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/serviceids/{id}/lock'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # Trusted profiles operations
    #########################

    def create_profile(
        self,
        name: str,
        account_id: str,
        *,
        description: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create a trusted profile.

        Create a trusted profile for a given account ID.

        :param str name: Name of the trusted profile. The name is checked for
               uniqueness. Therefore trusted profiles with the same names can not exist in
               the same account.
        :param str account_id: The account ID of the trusted profile.
        :param str description: (optional) The optional description of the trusted
               profile. The 'description' property is only available if a description was
               provided during creation of trusted profile.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TrustedProfile` object
        """

        if name is None:
            raise ValueError('name must be provided')
        if account_id is None:
            raise ValueError('account_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='create_profile',
        )
        headers.update(sdk_headers)

        data = {
            'name': name,
            'account_id': account_id,
            'description': description,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/profiles'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def list_profiles(
        self,
        account_id: str,
        *,
        name: Optional[str] = None,
        pagesize: Optional[int] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        include_history: Optional[bool] = None,
        pagetoken: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List trusted profiles.

        List the trusted profiles in an account. The `account_id` query parameter
        determines the account from which to retrieve the list of trusted profiles.

        :param str account_id: Account ID to query for trusted profiles.
        :param str name: (optional) Name of the trusted profile to query.
        :param int pagesize: (optional) Optional size of a single page. Default is
               20 items per page. Valid range is 1 to 100.
        :param str sort: (optional) Optional sort property, valid values are name,
               description, created_at and modified_at. If specified, the items are sorted
               by the value of this property.
        :param str order: (optional) Optional sort order, valid values are asc and
               desc. Default: asc.
        :param bool include_history: (optional) Defines if the entity history is
               included in the response.
        :param str pagetoken: (optional) Optional Prev or Next page token returned
               from a previous query execution. Default is start with first page.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TrustedProfilesList` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='list_profiles',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
            'name': name,
            'pagesize': pagesize,
            'sort': sort,
            'order': order,
            'include_history': include_history,
            'pagetoken': pagetoken,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/profiles'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def get_profile(
        self,
        profile_id: str,
        *,
        include_activity: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get a trusted profile.

        Retrieve a trusted profile by its `profile-id`. Only the trusted profile's data is
        returned (`name`, `description`, `iam_id`, etc.), not the federated users or
        compute resources that qualify to apply the trusted profile.

        :param str profile_id: ID of the trusted profile to get.
        :param bool include_activity: (optional) Defines if the entity's activity
               is included in the response. Retrieving activity data is an expensive
               operation, so only request this when needed.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TrustedProfile` object
        """

        if not profile_id:
            raise ValueError('profile_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_profile',
        )
        headers.update(sdk_headers)

        params = {
            'include_activity': include_activity,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['profile-id']
        path_param_values = self.encode_path_vars(profile_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profiles/{profile-id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def update_profile(
        self,
        profile_id: str,
        if_match: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Update a trusted profile.

        Update the name or description of an existing trusted profile.

        :param str profile_id: ID of the trusted profile to be updated.
        :param str if_match: Version of the trusted profile to be updated. Specify
               the version that you retrived when reading list of trusted profiles. This
               value helps to identify any parallel usage of trusted profile. Pass * to
               indicate to update any version available. This might result in stale
               updates.
        :param str name: (optional) The name of the trusted profile to update. If
               specified in the request the parameter must not be empty. The name is
               checked for uniqueness. Failure to this will result in an Error condition.
        :param str description: (optional) The description of the trusted profile
               to update. If specified an empty description will clear the description of
               the trusted profile. If a non empty value is provided the trusted profile
               will be updated.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TrustedProfile` object
        """

        if not profile_id:
            raise ValueError('profile_id must be provided')
        if not if_match:
            raise ValueError('if_match must be provided')
        headers = {
            'If-Match': if_match,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='update_profile',
        )
        headers.update(sdk_headers)

        data = {
            'name': name,
            'description': description,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['profile-id']
        path_param_values = self.encode_path_vars(profile_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profiles/{profile-id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PUT',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_profile(
        self,
        profile_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete a trusted profile.

        Delete a trusted profile. When you delete trusted profile, compute resources and
        federated users are unlinked from the profile and can no longer apply the trusted
        profile identity.

        :param str profile_id: ID of the trusted profile.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not profile_id:
            raise ValueError('profile_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='delete_profile',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['profile-id']
        path_param_values = self.encode_path_vars(profile_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profiles/{profile-id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def create_claim_rule(
        self,
        profile_id: str,
        type: str,
        conditions: List['ProfileClaimRuleConditions'],
        *,
        context: Optional['ResponseContext'] = None,
        name: Optional[str] = None,
        realm_name: Optional[str] = None,
        cr_type: Optional[str] = None,
        expiration: Optional[int] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create claim rule for a trusted profile.

        Create a claim rule for a trusted profile. There is a limit of 20 rules per
        trusted profile.

        :param str profile_id: ID of the trusted profile to create a claim rule.
        :param str type: Type of the claim rule, either 'Profile-SAML' or
               'Profile-CR'.
        :param List[ProfileClaimRuleConditions] conditions: Conditions of this
               claim rule.
        :param ResponseContext context: (optional) Context with key properties for
               problem determination.
        :param str name: (optional) Name of the claim rule to be created or
               updated.
        :param str realm_name: (optional) The realm name of the Idp this claim rule
               applies to. This field is required only if the type is specified as
               'Profile-SAML'.
        :param str cr_type: (optional) The compute resource type the rule applies
               to, required only if type is specified as 'Profile-CR'. Valid values are
               VSI, IKS_SA, ROKS_SA.
        :param int expiration: (optional) Session expiration in seconds, only
               required if type is 'Profile-SAML'.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ProfileClaimRule` object
        """

        if not profile_id:
            raise ValueError('profile_id must be provided')
        if type is None:
            raise ValueError('type must be provided')
        if conditions is None:
            raise ValueError('conditions must be provided')
        conditions = [convert_model(x) for x in conditions]
        if context is not None:
            context = convert_model(context)
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='create_claim_rule',
        )
        headers.update(sdk_headers)

        data = {
            'type': type,
            'conditions': conditions,
            'context': context,
            'name': name,
            'realm_name': realm_name,
            'cr_type': cr_type,
            'expiration': expiration,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['profile-id']
        path_param_values = self.encode_path_vars(profile_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profiles/{profile-id}/rules'.format(**path_param_dict)
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def list_claim_rules(
        self,
        profile_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        List claim rules for a trusted profile.

        Get a list of all claim rules for a trusted profile. The `profile-id` query
        parameter determines the profile from which to retrieve the list of claim rules.

        :param str profile_id: ID of the trusted profile.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ProfileClaimRuleList` object
        """

        if not profile_id:
            raise ValueError('profile_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='list_claim_rules',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['profile-id']
        path_param_values = self.encode_path_vars(profile_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profiles/{profile-id}/rules'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def get_claim_rule(
        self,
        profile_id: str,
        rule_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get a claim rule for a trusted profile.

        A specific claim rule can be fetched for a given trusted profile ID and rule ID.

        :param str profile_id: ID of the trusted profile.
        :param str rule_id: ID of the claim rule to get.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ProfileClaimRule` object
        """

        if not profile_id:
            raise ValueError('profile_id must be provided')
        if not rule_id:
            raise ValueError('rule_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_claim_rule',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['profile-id', 'rule-id']
        path_param_values = self.encode_path_vars(profile_id, rule_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profiles/{profile-id}/rules/{rule-id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def update_claim_rule(
        self,
        profile_id: str,
        rule_id: str,
        if_match: str,
        type: str,
        conditions: List['ProfileClaimRuleConditions'],
        *,
        context: Optional['ResponseContext'] = None,
        name: Optional[str] = None,
        realm_name: Optional[str] = None,
        cr_type: Optional[str] = None,
        expiration: Optional[int] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Update claim rule for a trusted profile.

        Update a specific claim rule for a given trusted profile ID and rule ID.

        :param str profile_id: ID of the trusted profile.
        :param str rule_id: ID of the claim rule to update.
        :param str if_match: Version of the claim rule to be updated. Specify the
               version that you retrived when reading list of claim rules. This value
               helps to identify any parallel usage of claim rule. Pass * to indicate to
               update any version available. This might result in stale updates.
        :param str type: Type of the claim rule, either 'Profile-SAML' or
               'Profile-CR'.
        :param List[ProfileClaimRuleConditions] conditions: Conditions of this
               claim rule.
        :param ResponseContext context: (optional) Context with key properties for
               problem determination.
        :param str name: (optional) Name of the claim rule to be created or
               updated.
        :param str realm_name: (optional) The realm name of the Idp this claim rule
               applies to. This field is required only if the type is specified as
               'Profile-SAML'.
        :param str cr_type: (optional) The compute resource type the rule applies
               to, required only if type is specified as 'Profile-CR'. Valid values are
               VSI, IKS_SA, ROKS_SA.
        :param int expiration: (optional) Session expiration in seconds, only
               required if type is 'Profile-SAML'.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ProfileClaimRule` object
        """

        if not profile_id:
            raise ValueError('profile_id must be provided')
        if not rule_id:
            raise ValueError('rule_id must be provided')
        if not if_match:
            raise ValueError('if_match must be provided')
        if type is None:
            raise ValueError('type must be provided')
        if conditions is None:
            raise ValueError('conditions must be provided')
        conditions = [convert_model(x) for x in conditions]
        if context is not None:
            context = convert_model(context)
        headers = {
            'If-Match': if_match,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='update_claim_rule',
        )
        headers.update(sdk_headers)

        data = {
            'type': type,
            'conditions': conditions,
            'context': context,
            'name': name,
            'realm_name': realm_name,
            'cr_type': cr_type,
            'expiration': expiration,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['profile-id', 'rule-id']
        path_param_values = self.encode_path_vars(profile_id, rule_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profiles/{profile-id}/rules/{rule-id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PUT',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_claim_rule(
        self,
        profile_id: str,
        rule_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete a claim rule.

        Delete a claim rule. When you delete a claim rule, federated user or compute
        resources are no longer required to meet the conditions of the claim rule in order
        to apply the trusted profile.

        :param str profile_id: ID of the trusted profile.
        :param str rule_id: ID of the claim rule to delete.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not profile_id:
            raise ValueError('profile_id must be provided')
        if not rule_id:
            raise ValueError('rule_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='delete_claim_rule',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['profile-id', 'rule-id']
        path_param_values = self.encode_path_vars(profile_id, rule_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profiles/{profile-id}/rules/{rule-id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def create_link(
        self,
        profile_id: str,
        cr_type: str,
        link: 'CreateProfileLinkRequestLink',
        *,
        name: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create link to a trusted profile.

        Create a direct link between a specific compute resource and a trusted profile,
        rather than creating conditions that a compute resource must fulfill to apply a
        trusted profile.

        :param str profile_id: ID of the trusted profile.
        :param str cr_type: The compute resource type. Valid values are VSI,
               IKS_SA, ROKS_SA.
        :param CreateProfileLinkRequestLink link: Link details.
        :param str name: (optional) Optional name of the Link.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ProfileLink` object
        """

        if not profile_id:
            raise ValueError('profile_id must be provided')
        if cr_type is None:
            raise ValueError('cr_type must be provided')
        if link is None:
            raise ValueError('link must be provided')
        link = convert_model(link)
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='create_link',
        )
        headers.update(sdk_headers)

        data = {
            'cr_type': cr_type,
            'link': link,
            'name': name,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['profile-id']
        path_param_values = self.encode_path_vars(profile_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profiles/{profile-id}/links'.format(**path_param_dict)
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def list_links(
        self,
        profile_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        List links to a trusted profile.

        Get a list of links to a trusted profile.

        :param str profile_id: ID of the trusted profile.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ProfileLinkList` object
        """

        if not profile_id:
            raise ValueError('profile_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='list_links',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['profile-id']
        path_param_values = self.encode_path_vars(profile_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profiles/{profile-id}/links'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def get_link(
        self,
        profile_id: str,
        link_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get link to a trusted profile.

        Get a specific link to a trusted profile by `link_id`.

        :param str profile_id: ID of the trusted profile.
        :param str link_id: ID of the link.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ProfileLink` object
        """

        if not profile_id:
            raise ValueError('profile_id must be provided')
        if not link_id:
            raise ValueError('link_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_link',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['profile-id', 'link-id']
        path_param_values = self.encode_path_vars(profile_id, link_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profiles/{profile-id}/links/{link-id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_link(
        self,
        profile_id: str,
        link_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete link to a trusted profile.

        Delete a link between a compute resource and a trusted profile.

        :param str profile_id: ID of the trusted profile.
        :param str link_id: ID of the link.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not profile_id:
            raise ValueError('profile_id must be provided')
        if not link_id:
            raise ValueError('link_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='delete_link',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['profile-id', 'link-id']
        path_param_values = self.encode_path_vars(profile_id, link_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profiles/{profile-id}/links/{link-id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def get_profile_identities(
        self,
        profile_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get a list of identities that can assume the trusted profile.

        Get a list of identities that can assume the trusted profile.

        :param str profile_id: ID of the trusted profile.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ProfileIdentitiesResponse` object
        """

        if not profile_id:
            raise ValueError('profile_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_profile_identities',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['profile-id']
        path_param_values = self.encode_path_vars(profile_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profiles/{profile-id}/identities'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def set_profile_identities(
        self,
        profile_id: str,
        if_match: str,
        *,
        identities: Optional[List['ProfileIdentityRequest']] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Update the list of identities that can assume the trusted profile.

        Update the list of identities that can assume the trusted profile.

        :param str profile_id: ID of the trusted profile.
        :param str if_match: Entity tag of the Identities to be updated. Specify
               the tag that you retrieved when reading the Profile Identities. This value
               helps identify parallel usage of this API. Pass * to indicate updating any
               available version, which may result in stale updates.
        :param List[ProfileIdentityRequest] identities: (optional) List of
               identities that can assume the trusted profile.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ProfileIdentitiesResponse` object
        """

        if not profile_id:
            raise ValueError('profile_id must be provided')
        if not if_match:
            raise ValueError('if_match must be provided')
        if identities is not None:
            identities = [convert_model(x) for x in identities]
        headers = {
            'If-Match': if_match,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='set_profile_identities',
        )
        headers.update(sdk_headers)

        data = {
            'identities': identities,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['profile-id']
        path_param_values = self.encode_path_vars(profile_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profiles/{profile-id}/identities'.format(**path_param_dict)
        request = self.prepare_request(
            method='PUT',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def set_profile_identity(
        self,
        profile_id: str,
        identity_type: str,
        identifier: str,
        type: str,
        *,
        accounts: Optional[List[str]] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Add a specific identity that can assume the trusted profile.

        Add a specific identity that can assume the trusted profile.

        :param str profile_id: ID of the trusted profile.
        :param str identity_type: Type of the identity.
        :param str identifier: Identifier of the identity that can assume the
               trusted profiles. This can be a user identifier (IAM id), serviceid or crn.
               Internally it uses account id of the service id for the identifier
               'serviceid' and for the identifier 'crn' it uses account id contained in
               the CRN.
        :param str type: Type of the identity.
        :param List[str] accounts: (optional) Only valid for the type user.
               Accounts from which a user can assume the trusted profile.
        :param str description: (optional) Description of the identity that can
               assume the trusted profile. This is optional field for all the types of
               identities. When this field is not set for the identity type 'serviceid'
               then the description of the service id is used. Description is recommended
               for the identity type 'crn' E.g. 'Instance 1234 of IBM Cloud Service
               project'.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ProfileIdentityResponse` object
        """

        if not profile_id:
            raise ValueError('profile_id must be provided')
        if not identity_type:
            raise ValueError('identity_type must be provided')
        if identifier is None:
            raise ValueError('identifier must be provided')
        if type is None:
            raise ValueError('type must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='set_profile_identity',
        )
        headers.update(sdk_headers)

        data = {
            'identifier': identifier,
            'type': type,
            'accounts': accounts,
            'description': description,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['profile-id', 'identity-type']
        path_param_values = self.encode_path_vars(profile_id, identity_type)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profiles/{profile-id}/identities/{identity-type}'.format(**path_param_dict)
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def get_profile_identity(
        self,
        profile_id: str,
        identity_type: str,
        identifier_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get the identity that can assume the trusted profile.

        Get the identity that can assume the trusted profile.

        :param str profile_id: ID of the trusted profile.
        :param str identity_type: Type of the identity.
        :param str identifier_id: Identifier of the identity that can assume the
               trusted profiles.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ProfileIdentityResponse` object
        """

        if not profile_id:
            raise ValueError('profile_id must be provided')
        if not identity_type:
            raise ValueError('identity_type must be provided')
        if not identifier_id:
            raise ValueError('identifier_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_profile_identity',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['profile-id', 'identity-type', 'identifier-id']
        path_param_values = self.encode_path_vars(profile_id, identity_type, identifier_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profiles/{profile-id}/identities/{identity-type}/{identifier-id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_profile_identity(
        self,
        profile_id: str,
        identity_type: str,
        identifier_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete the identity that can assume the trusted profile.

        Delete the identity that can assume the trusted profile.

        :param str profile_id: ID of the trusted profile.
        :param str identity_type: Type of the identity.
        :param str identifier_id: Identifier of the identity that can assume the
               trusted profiles.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not profile_id:
            raise ValueError('profile_id must be provided')
        if not identity_type:
            raise ValueError('identity_type must be provided')
        if not identifier_id:
            raise ValueError('identifier_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='delete_profile_identity',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['profile-id', 'identity-type', 'identifier-id']
        path_param_values = self.encode_path_vars(profile_id, identity_type, identifier_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profiles/{profile-id}/identities/{identity-type}/{identifier-id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # Account settings
    #########################

    def get_account_settings(
        self,
        account_id: str,
        *,
        include_history: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get account configurations.

        Returns the details of an account's configuration.

        :param str account_id: Unique ID of the account.
        :param bool include_history: (optional) Defines if the entity history is
               included in the response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccountSettingsResponse` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_account_settings',
        )
        headers.update(sdk_headers)

        params = {
            'include_history': include_history,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id']
        path_param_values = self.encode_path_vars(account_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/accounts/{account_id}/settings/identity'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def update_account_settings(
        self,
        if_match: str,
        account_id: str,
        *,
        restrict_create_service_id: Optional[str] = None,
        restrict_create_platform_apikey: Optional[str] = None,
        allowed_ip_addresses: Optional[str] = None,
        mfa: Optional[str] = None,
        user_mfa: Optional[List['AccountSettingsUserMFA']] = None,
        session_expiration_in_seconds: Optional[str] = None,
        session_invalidation_in_seconds: Optional[str] = None,
        max_sessions_per_identity: Optional[str] = None,
        system_access_token_expiration_in_seconds: Optional[str] = None,
        system_refresh_token_expiration_in_seconds: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Update account configurations.

        Allows a user to configure settings on their account with regards to MFA, MFA
        excemption list, session lifetimes, access control for creating new identities,
        and enforcing IP restrictions on token creation.

        :param str if_match: Version of the account settings to be updated. Specify
               the version that you retrieved as entity_tag (ETag header) when reading the
               account. This value helps identifying parallel usage of this API. Pass * to
               indicate to update any version available. This might result in stale
               updates.
        :param str account_id: The id of the account to update the settings for.
        :param str restrict_create_service_id: (optional) Defines whether or not
               creating a service ID is access controlled. Valid values:
                 * RESTRICTED - only users assigned the 'Service ID creator' role on the
               IAM Identity Service can create service IDs, including the account owner
                 * NOT_RESTRICTED - all members of an account can create service IDs
                 * NOT_SET - to 'unset' a previous set value.
        :param str restrict_create_platform_apikey: (optional) Defines whether or
               not creating platform API keys is access controlled. Valid values:
                 * RESTRICTED - only users assigned the 'User API key creator' role on the
               IAM Identity Service can create API keys, including the account owner
                 * NOT_RESTRICTED - all members of an account can create platform API keys
                 * NOT_SET - to 'unset' a previous set value.
        :param str allowed_ip_addresses: (optional) Defines the IP addresses and
               subnets from which IAM tokens can be created for the account.
        :param str mfa: (optional) Defines the MFA trait for the account. Valid
               values:
                 * NONE - No MFA trait set
                 * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
                 * TOTP - For all non-federated IBMId users
                 * TOTP4ALL - For all users
                 * LEVEL1 - Email-based MFA for all users
                 * LEVEL2 - TOTP-based MFA for all users
                 * LEVEL3 - U2F MFA for all users.
        :param List[AccountSettingsUserMFA] user_mfa: (optional) List of users that
               are exempted from the MFA requirement of the account.
        :param str session_expiration_in_seconds: (optional) Defines the session
               expiration in seconds for the account. Valid values:
                 * Any whole number between between '900' and '86400'
                 * NOT_SET - To unset account setting and use service default.
        :param str session_invalidation_in_seconds: (optional) Defines the period
               of time in seconds in which a session will be invalidated due to
               inactivity. Valid values:
                 * Any whole number between '900' and '7200'
                 * NOT_SET - To unset account setting and use service default.
        :param str max_sessions_per_identity: (optional) Defines the max allowed
               sessions per identity required by the account. Value values:
                 * Any whole number greater than 0
                 * NOT_SET - To unset account setting and use service default.
        :param str system_access_token_expiration_in_seconds: (optional) Defines
               the access token expiration in seconds. Valid values:
                 * Any whole number between '900' and '3600'
                 * NOT_SET - To unset account setting and use service default.
        :param str system_refresh_token_expiration_in_seconds: (optional) Defines
               the refresh token expiration in seconds. Valid values:
                 * Any whole number between '900' and '259200'
                 * NOT_SET - To unset account setting and use service default.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccountSettingsResponse` object
        """

        if not if_match:
            raise ValueError('if_match must be provided')
        if not account_id:
            raise ValueError('account_id must be provided')
        if user_mfa is not None:
            user_mfa = [convert_model(x) for x in user_mfa]
        headers = {
            'If-Match': if_match,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='update_account_settings',
        )
        headers.update(sdk_headers)

        data = {
            'restrict_create_service_id': restrict_create_service_id,
            'restrict_create_platform_apikey': restrict_create_platform_apikey,
            'allowed_ip_addresses': allowed_ip_addresses,
            'mfa': mfa,
            'user_mfa': user_mfa,
            'session_expiration_in_seconds': session_expiration_in_seconds,
            'session_invalidation_in_seconds': session_invalidation_in_seconds,
            'max_sessions_per_identity': max_sessions_per_identity,
            'system_access_token_expiration_in_seconds': system_access_token_expiration_in_seconds,
            'system_refresh_token_expiration_in_seconds': system_refresh_token_expiration_in_seconds,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id']
        path_param_values = self.encode_path_vars(account_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/accounts/{account_id}/settings/identity'.format(**path_param_dict)
        request = self.prepare_request(
            method='PUT',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # MFA enrollment status
    #########################

    def get_mfa_status(
        self,
        account_id: str,
        iam_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get MFA enrollment status for a single user in the account.

        Get MFA enrollment status for a single user in the account.

        :param str account_id: ID of the account.
        :param str iam_id: iam_id of the user. This user must be the member of the
               account.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `UserMfaEnrollments` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not iam_id:
            raise ValueError('iam_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_mfa_status',
        )
        headers.update(sdk_headers)

        params = {
            'iam_id': iam_id,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id']
        path_param_values = self.encode_path_vars(account_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/mfa/accounts/{account_id}/status'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def create_mfa_report(
        self,
        account_id: str,
        *,
        type: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Trigger MFA enrollment status report for the account.

        Trigger MFA enrollment status report for the account by specifying the account ID.
        It can take a few minutes to generate the report for retrieval.

        :param str account_id: ID of the account.
        :param str type: (optional) Optional report type. The supported value is
               'mfa_status'. List MFA enrollment status for all the identities.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ReportReference` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='create_mfa_report',
        )
        headers.update(sdk_headers)

        params = {
            'type': type,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id']
        path_param_values = self.encode_path_vars(account_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/mfa/accounts/{account_id}/report'.format(**path_param_dict)
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def get_mfa_report(
        self,
        account_id: str,
        reference: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get MFA enrollment status report for the account.

        Get MFA enrollment status report for the account by specifying the account ID and
        the reference that is generated by triggering the report. Reports older than a day
        are deleted when generating a new report.

        :param str account_id: ID of the account.
        :param str reference: Reference for the report to be generated, You can use
               'latest' to get the latest report for the given account.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ReportMfaEnrollmentStatus` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not reference:
            raise ValueError('reference must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_mfa_report',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id', 'reference']
        path_param_values = self.encode_path_vars(account_id, reference)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/mfa/accounts/{account_id}/report/{reference}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # accountSettingsAssignments
    #########################

    def list_account_settings_assignments(
        self,
        *,
        account_id: Optional[str] = None,
        template_id: Optional[str] = None,
        template_version: Optional[str] = None,
        target: Optional[str] = None,
        target_type: Optional[str] = None,
        limit: Optional[int] = None,
        pagetoken: Optional[str] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        include_history: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List assignments.

        List account settings assignments.

        :param str account_id: (optional) Account ID of the Assignments to query.
               This parameter is required unless using a pagetoken.
        :param str template_id: (optional) Filter results by Template Id.
        :param str template_version: (optional) Filter results Template Version.
        :param str target: (optional) Filter results by the assignment target.
        :param str target_type: (optional) Filter results by the assignment's
               target type.
        :param int limit: (optional) Optional size of a single page. Default is 20
               items per page. Valid range is 1 to 100.
        :param str pagetoken: (optional) Optional Prev or Next page token returned
               from a previous query execution. Default is start with first page.
        :param str sort: (optional) If specified, the items are sorted by the value
               of this property.
        :param str order: (optional) Sort order.
        :param bool include_history: (optional) Defines if the entity history is
               included in the response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TemplateAssignmentListResponse` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='list_account_settings_assignments',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
            'template_id': template_id,
            'template_version': template_version,
            'target': target,
            'target_type': target_type,
            'limit': limit,
            'pagetoken': pagetoken,
            'sort': sort,
            'order': order,
            'include_history': include_history,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/account_settings_assignments/'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def create_account_settings_assignment(
        self,
        template_id: str,
        template_version: int,
        target_type: str,
        target: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create assignment.

        Create an assigment for an account settings template.

        :param str template_id: ID of the template to assign.
        :param int template_version: Version of the template to assign.
        :param str target_type: Type of target to deploy to.
        :param str target: Identifier of target to deploy to.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TemplateAssignmentResponse` object
        """

        if template_id is None:
            raise ValueError('template_id must be provided')
        if template_version is None:
            raise ValueError('template_version must be provided')
        if target_type is None:
            raise ValueError('target_type must be provided')
        if target is None:
            raise ValueError('target must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='create_account_settings_assignment',
        )
        headers.update(sdk_headers)

        data = {
            'template_id': template_id,
            'template_version': template_version,
            'target_type': target_type,
            'target': target,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/account_settings_assignments/'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def get_account_settings_assignment(
        self,
        assignment_id: str,
        *,
        include_history: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get assignment.

        Get an assigment for an account settings template.

        :param str assignment_id: ID of the Assignment Record.
        :param bool include_history: (optional) Defines if the entity history is
               included in the response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TemplateAssignmentResponse` object
        """

        if not assignment_id:
            raise ValueError('assignment_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_account_settings_assignment',
        )
        headers.update(sdk_headers)

        params = {
            'include_history': include_history,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['assignment_id']
        path_param_values = self.encode_path_vars(assignment_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/account_settings_assignments/{assignment_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_account_settings_assignment(
        self,
        assignment_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete assignment.

        Delete an account settings template assignment. This removes any IAM resources
        created by this assignment in child accounts.

        :param str assignment_id: ID of the Assignment Record.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ExceptionResponse` object
        """

        if not assignment_id:
            raise ValueError('assignment_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='delete_account_settings_assignment',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['assignment_id']
        path_param_values = self.encode_path_vars(assignment_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/account_settings_assignments/{assignment_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def update_account_settings_assignment(
        self,
        assignment_id: str,
        if_match: str,
        template_version: int,
        **kwargs,
    ) -> DetailedResponse:
        """
        Update assignment.

        Update an account settings assignment. Call this method to retry failed
        assignments or migrate the settings in child accounts to a new version.

        :param str assignment_id: ID of the Assignment Record.
        :param str if_match: Version of the assignment to be updated. Specify the
               version that you retrieved when reading the assignment. This value  helps
               identifying parallel usage of this API. Pass * to indicate to update any
               version available. This might result in stale updates.
        :param int template_version: Template version to be applied to the
               assignment. To retry all failed assignments, provide the existing version.
               To migrate to a different version, provide the new version number.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TemplateAssignmentResponse` object
        """

        if not assignment_id:
            raise ValueError('assignment_id must be provided')
        if not if_match:
            raise ValueError('if_match must be provided')
        if template_version is None:
            raise ValueError('template_version must be provided')
        headers = {
            'If-Match': if_match,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='update_account_settings_assignment',
        )
        headers.update(sdk_headers)

        data = {
            'template_version': template_version,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['assignment_id']
        path_param_values = self.encode_path_vars(assignment_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/account_settings_assignments/{assignment_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PATCH',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # accountSettingsTemplate
    #########################

    def list_account_settings_templates(
        self,
        *,
        account_id: Optional[str] = None,
        limit: Optional[str] = None,
        pagetoken: Optional[str] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        include_history: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List account settings templates.

        List account settings templates in an enterprise account.

        :param str account_id: (optional) Account ID of the account settings
               templates to query. This parameter is required unless using a pagetoken.
        :param str limit: (optional) Optional size of a single page.
        :param str pagetoken: (optional) Optional Prev or Next page token returned
               from a previous query execution. Default is start with first page.
        :param str sort: (optional) Optional sort property. If specified, the
               returned templated are sorted according to this property.
        :param str order: (optional) Optional sort order.
        :param str include_history: (optional) Defines if the entity history is
               included in the response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccountSettingsTemplateList` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='list_account_settings_templates',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
            'limit': limit,
            'pagetoken': pagetoken,
            'sort': sort,
            'order': order,
            'include_history': include_history,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/account_settings_templates'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def create_account_settings_template(
        self,
        *,
        account_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        account_settings: Optional['AccountSettingsComponent'] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create an account settings template.

        Create a new account settings template in an enterprise account.

        :param str account_id: (optional) ID of the account where the template
               resides.
        :param str name: (optional) The name of the trusted profile template. This
               is visible only in the enterprise account.
        :param str description: (optional) The description of the trusted profile
               template. Describe the template for enterprise account users.
        :param AccountSettingsComponent account_settings: (optional)
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccountSettingsTemplateResponse` object
        """

        if account_settings is not None:
            account_settings = convert_model(account_settings)
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='create_account_settings_template',
        )
        headers.update(sdk_headers)

        data = {
            'account_id': account_id,
            'name': name,
            'description': description,
            'account_settings': account_settings,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/account_settings_templates'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def get_latest_account_settings_template_version(
        self,
        template_id: str,
        *,
        include_history: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get latest version of an account settings template.

        Get the latest version of a specific account settings template in an enterprise
        account.

        :param str template_id: ID of the account settings template.
        :param bool include_history: (optional) Defines if the entity history is
               included in the response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccountSettingsTemplateResponse` object
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_latest_account_settings_template_version',
        )
        headers.update(sdk_headers)

        params = {
            'include_history': include_history,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['template_id']
        path_param_values = self.encode_path_vars(template_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/account_settings_templates/{template_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_all_versions_of_account_settings_template(
        self,
        template_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete all versions of an account settings template.

        Delete all versions of an account settings template in an enterprise account. If
        any version is assigned to child accounts, you must first delete the assignment.

        :param str template_id: ID of the account settings template.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='delete_all_versions_of_account_settings_template',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['template_id']
        path_param_values = self.encode_path_vars(template_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/account_settings_templates/{template_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def list_versions_of_account_settings_template(
        self,
        template_id: str,
        *,
        limit: Optional[str] = None,
        pagetoken: Optional[str] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        include_history: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List account settings template versions.

        List the versions of a specific account settings template in an enterprise
        account.

        :param str template_id: ID of the account settings template.
        :param str limit: (optional) Optional size of a single page.
        :param str pagetoken: (optional) Optional Prev or Next page token returned
               from a previous query execution. Default is start with first page.
        :param str sort: (optional) Optional sort property. If specified, the
               returned templated are sorted according to this property.
        :param str order: (optional) Optional sort order.
        :param str include_history: (optional) Defines if the entity history is
               included in the response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccountSettingsTemplateList` object
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='list_versions_of_account_settings_template',
        )
        headers.update(sdk_headers)

        params = {
            'limit': limit,
            'pagetoken': pagetoken,
            'sort': sort,
            'order': order,
            'include_history': include_history,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['template_id']
        path_param_values = self.encode_path_vars(template_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/account_settings_templates/{template_id}/versions'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def create_account_settings_template_version(
        self,
        template_id: str,
        *,
        account_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        account_settings: Optional['AccountSettingsComponent'] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create a new version of an account settings template.

        Create a new version of an account settings template in an Enterprise Account.

        :param str template_id: ID of the account settings template.
        :param str account_id: (optional) ID of the account where the template
               resides.
        :param str name: (optional) The name of the trusted profile template. This
               is visible only in the enterprise account.
        :param str description: (optional) The description of the trusted profile
               template. Describe the template for enterprise account users.
        :param AccountSettingsComponent account_settings: (optional)
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccountSettingsTemplateResponse` object
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        if account_settings is not None:
            account_settings = convert_model(account_settings)
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='create_account_settings_template_version',
        )
        headers.update(sdk_headers)

        data = {
            'account_id': account_id,
            'name': name,
            'description': description,
            'account_settings': account_settings,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['template_id']
        path_param_values = self.encode_path_vars(template_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/account_settings_templates/{template_id}/versions'.format(**path_param_dict)
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def get_account_settings_template_version(
        self,
        template_id: str,
        version: str,
        *,
        include_history: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get version of an account settings template.

        Get a specific version of an account settings template in an Enterprise Account.

        :param str template_id: ID of the account settings template.
        :param str version: Version of the account settings template.
        :param bool include_history: (optional) Defines if the entity history is
               included in the response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccountSettingsTemplateResponse` object
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        if not version:
            raise ValueError('version must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_account_settings_template_version',
        )
        headers.update(sdk_headers)

        params = {
            'include_history': include_history,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['template_id', 'version']
        path_param_values = self.encode_path_vars(template_id, version)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/account_settings_templates/{template_id}/versions/{version}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def update_account_settings_template_version(
        self,
        if_match: str,
        template_id: str,
        version: str,
        *,
        account_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        account_settings: Optional['AccountSettingsComponent'] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Update version of an account settings template.

        Update a specific version of an account settings template in an Enterprise
        Account.

        :param str if_match: Entity tag of the Template to be updated. Specify the
               tag that you retrieved when reading the account settings template. This
               value helps identifying parallel usage of this API. Pass * to indicate to
               update any version available. This might result in stale updates.
        :param str template_id: ID of the account settings template.
        :param str version: Version of the account settings template.
        :param str account_id: (optional) ID of the account where the template
               resides.
        :param str name: (optional) The name of the trusted profile template. This
               is visible only in the enterprise account.
        :param str description: (optional) The description of the trusted profile
               template. Describe the template for enterprise account users.
        :param AccountSettingsComponent account_settings: (optional)
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccountSettingsTemplateResponse` object
        """

        if not if_match:
            raise ValueError('if_match must be provided')
        if not template_id:
            raise ValueError('template_id must be provided')
        if not version:
            raise ValueError('version must be provided')
        if account_settings is not None:
            account_settings = convert_model(account_settings)
        headers = {
            'If-Match': if_match,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='update_account_settings_template_version',
        )
        headers.update(sdk_headers)

        data = {
            'account_id': account_id,
            'name': name,
            'description': description,
            'account_settings': account_settings,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['template_id', 'version']
        path_param_values = self.encode_path_vars(template_id, version)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/account_settings_templates/{template_id}/versions/{version}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PUT',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_account_settings_template_version(
        self,
        template_id: str,
        version: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete version of an account settings template.

        Delete a specific version of an account settings template in an Enterprise
        Account.

        :param str template_id: ID of the account settings template.
        :param str version: Version of the account settings template.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        if not version:
            raise ValueError('version must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='delete_account_settings_template_version',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['template_id', 'version']
        path_param_values = self.encode_path_vars(template_id, version)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/account_settings_templates/{template_id}/versions/{version}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def commit_account_settings_template(
        self,
        template_id: str,
        version: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Commit a template version.

        Commit a specific version of an account settings template in an Enterprise
        Account. A Template must be committed before being assigned, and once committed,
        can no longer be modified.

        :param str template_id: ID of the account settings template.
        :param str version: Version of the account settings template.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        if not version:
            raise ValueError('version must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='commit_account_settings_template',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['template_id', 'version']
        path_param_values = self.encode_path_vars(template_id, version)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/account_settings_templates/{template_id}/versions/{version}/commit'.format(**path_param_dict)
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # activityOperations
    #########################

    def create_report(
        self,
        account_id: str,
        *,
        type: Optional[str] = None,
        duration: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Trigger activity report for the account.

        Trigger activity report for the account by specifying the account ID. It can take
        a few minutes to generate the report for retrieval.

        :param str account_id: ID of the account.
        :param str type: (optional) Optional report type. The supported value is
               'inactive'. List all identities that have not authenticated within the time
               indicated by duration.
        :param str duration: (optional) Optional duration of the report. The
               supported unit of duration is hours.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ReportReference` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='create_report',
        )
        headers.update(sdk_headers)

        params = {
            'type': type,
            'duration': duration,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id']
        path_param_values = self.encode_path_vars(account_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/activity/accounts/{account_id}/report'.format(**path_param_dict)
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def get_report(
        self,
        account_id: str,
        reference: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get activity report for the account.

        Get activity report for the account by specifying the account ID and the reference
        that is generated by triggering the report. Reports older than a day are deleted
        when generating a new report.

        :param str account_id: ID of the account.
        :param str reference: Reference for the report to be generated, You can use
               'latest' to get the latest report for the given account.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Report` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not reference:
            raise ValueError('reference must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_report',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id', 'reference']
        path_param_values = self.encode_path_vars(account_id, reference)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/activity/accounts/{account_id}/report/{reference}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # trustedProfileAssignments
    #########################

    def list_trusted_profile_assignments(
        self,
        *,
        account_id: Optional[str] = None,
        template_id: Optional[str] = None,
        template_version: Optional[str] = None,
        target: Optional[str] = None,
        target_type: Optional[str] = None,
        limit: Optional[int] = None,
        pagetoken: Optional[str] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        include_history: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List assignments.

        List trusted profile template assignments.

        :param str account_id: (optional) Account ID of the Assignments to query.
               This parameter is required unless using a pagetoken.
        :param str template_id: (optional) Filter results by Template Id.
        :param str template_version: (optional) Filter results Template Version.
        :param str target: (optional) Filter results by the assignment target.
        :param str target_type: (optional) Filter results by the assignment's
               target type.
        :param int limit: (optional) Optional size of a single page. Default is 20
               items per page. Valid range is 1 to 100.
        :param str pagetoken: (optional) Optional Prev or Next page token returned
               from a previous query execution. Default is start with first page.
        :param str sort: (optional) If specified, the items are sorted by the value
               of this property.
        :param str order: (optional) Sort order.
        :param bool include_history: (optional) Defines if the entity history is
               included in the response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TemplateAssignmentListResponse` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='list_trusted_profile_assignments',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
            'template_id': template_id,
            'template_version': template_version,
            'target': target,
            'target_type': target_type,
            'limit': limit,
            'pagetoken': pagetoken,
            'sort': sort,
            'order': order,
            'include_history': include_history,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/profile_assignments/'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def create_trusted_profile_assignment(
        self,
        template_id: str,
        template_version: int,
        target_type: str,
        target: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create assignment.

        Create an assigment for a trusted profile template.

        :param str template_id: ID of the template to assign.
        :param int template_version: Version of the template to assign.
        :param str target_type: Type of target to deploy to.
        :param str target: Identifier of target to deploy to.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TemplateAssignmentResponse` object
        """

        if template_id is None:
            raise ValueError('template_id must be provided')
        if template_version is None:
            raise ValueError('template_version must be provided')
        if target_type is None:
            raise ValueError('target_type must be provided')
        if target is None:
            raise ValueError('target must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='create_trusted_profile_assignment',
        )
        headers.update(sdk_headers)

        data = {
            'template_id': template_id,
            'template_version': template_version,
            'target_type': target_type,
            'target': target,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/profile_assignments/'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def get_trusted_profile_assignment(
        self,
        assignment_id: str,
        *,
        include_history: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get assignment.

        Get an assigment for a trusted profile template.

        :param str assignment_id: ID of the Assignment Record.
        :param bool include_history: (optional) Defines if the entity history is
               included in the response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TemplateAssignmentResponse` object
        """

        if not assignment_id:
            raise ValueError('assignment_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_trusted_profile_assignment',
        )
        headers.update(sdk_headers)

        params = {
            'include_history': include_history,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['assignment_id']
        path_param_values = self.encode_path_vars(assignment_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profile_assignments/{assignment_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_trusted_profile_assignment(
        self,
        assignment_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete assignment.

        Delete a trusted profile assignment. This removes any IAM resources created by
        this assignment in child accounts.

        :param str assignment_id: ID of the Assignment Record.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ExceptionResponse` object
        """

        if not assignment_id:
            raise ValueError('assignment_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='delete_trusted_profile_assignment',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['assignment_id']
        path_param_values = self.encode_path_vars(assignment_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profile_assignments/{assignment_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def update_trusted_profile_assignment(
        self,
        assignment_id: str,
        if_match: str,
        template_version: int,
        **kwargs,
    ) -> DetailedResponse:
        """
        Update assignment.

        Update a trusted profile assignment. Call this method to retry failed assignments
        or migrate the trusted profile in child accounts to a new version.

        :param str assignment_id: ID of the Assignment Record.
        :param str if_match: Version of the Assignment to be updated. Specify the
               version that you retrieved when reading the Assignment. This value  helps
               identifying parallel usage of this API. Pass * to indicate to update any
               version available. This might result in stale updates.
        :param int template_version: Template version to be applied to the
               assignment. To retry all failed assignments, provide the existing version.
               To migrate to a different version, provide the new version number.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TemplateAssignmentResponse` object
        """

        if not assignment_id:
            raise ValueError('assignment_id must be provided')
        if not if_match:
            raise ValueError('if_match must be provided')
        if template_version is None:
            raise ValueError('template_version must be provided')
        headers = {
            'If-Match': if_match,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='update_trusted_profile_assignment',
        )
        headers.update(sdk_headers)

        data = {
            'template_version': template_version,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['assignment_id']
        path_param_values = self.encode_path_vars(assignment_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profile_assignments/{assignment_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PATCH',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # trustedProfileTemplate
    #########################

    def list_profile_templates(
        self,
        *,
        account_id: Optional[str] = None,
        limit: Optional[str] = None,
        pagetoken: Optional[str] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        include_history: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List trusted profile templates.

        List the trusted profile templates in an enterprise account.

        :param str account_id: (optional) Account ID of the trusted profile
               templates to query. This parameter is required unless using a pagetoken.
        :param str limit: (optional) Optional size of a single page.
        :param str pagetoken: (optional) Optional Prev or Next page token returned
               from a previous query execution. Default is start with first page.
        :param str sort: (optional) Optional sort property. If specified, the
               returned templates are sorted according to this property.
        :param str order: (optional) Optional sort order.
        :param str include_history: (optional) Defines if the entity history is
               included in the response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TrustedProfileTemplateList` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='list_profile_templates',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
            'limit': limit,
            'pagetoken': pagetoken,
            'sort': sort,
            'order': order,
            'include_history': include_history,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/profile_templates'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def create_profile_template(
        self,
        *,
        account_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        profile: Optional['TemplateProfileComponentRequest'] = None,
        policy_template_references: Optional[List['PolicyTemplateReference']] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create a trusted profile template.

        Create a new trusted profile template in an enterprise account.

        :param str account_id: (optional) ID of the account where the template
               resides.
        :param str name: (optional) The name of the trusted profile template. This
               is visible only in the enterprise account. Required field when creating a
               new template. Otherwise this field is optional. If the field is included it
               will change the name value for all existing versions of the template.
        :param str description: (optional) The description of the trusted profile
               template. Describe the template for enterprise account users.
        :param TemplateProfileComponentRequest profile: (optional) Input body
               parameters for the TemplateProfileComponent.
        :param List[PolicyTemplateReference] policy_template_references: (optional)
               Existing policy templates that you can reference to assign access in the
               trusted profile component.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TrustedProfileTemplateResponse` object
        """

        if profile is not None:
            profile = convert_model(profile)
        if policy_template_references is not None:
            policy_template_references = [convert_model(x) for x in policy_template_references]
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='create_profile_template',
        )
        headers.update(sdk_headers)

        data = {
            'account_id': account_id,
            'name': name,
            'description': description,
            'profile': profile,
            'policy_template_references': policy_template_references,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/profile_templates'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def get_latest_profile_template_version(
        self,
        template_id: str,
        *,
        include_history: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get latest version of a trusted profile template.

        Get the latest version of a trusted profile template in an enterprise account.

        :param str template_id: ID of the trusted profile template.
        :param bool include_history: (optional) Defines if the entity history is
               included in the response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TrustedProfileTemplateResponse` object
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_latest_profile_template_version',
        )
        headers.update(sdk_headers)

        params = {
            'include_history': include_history,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['template_id']
        path_param_values = self.encode_path_vars(template_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profile_templates/{template_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_all_versions_of_profile_template(
        self,
        template_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete all versions of a trusted profile template.

        Delete all versions of a trusted profile template in an enterprise account. If any
        version is assigned to child accounts, you must first delete the assignment.

        :param str template_id: ID of the trusted profile template.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='delete_all_versions_of_profile_template',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['template_id']
        path_param_values = self.encode_path_vars(template_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profile_templates/{template_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def list_versions_of_profile_template(
        self,
        template_id: str,
        *,
        limit: Optional[str] = None,
        pagetoken: Optional[str] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        include_history: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List trusted profile template versions.

        List the versions of a trusted profile template in an enterprise account.

        :param str template_id: ID of the trusted profile template.
        :param str limit: (optional) Optional size of a single page.
        :param str pagetoken: (optional) Optional Prev or Next page token returned
               from a previous query execution. Default is start with first page.
        :param str sort: (optional) Optional sort property. If specified, the
               returned templated are sorted according to this property.
        :param str order: (optional) Optional sort order.
        :param str include_history: (optional) Defines if the entity history is
               included in the response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TrustedProfileTemplateList` object
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='list_versions_of_profile_template',
        )
        headers.update(sdk_headers)

        params = {
            'limit': limit,
            'pagetoken': pagetoken,
            'sort': sort,
            'order': order,
            'include_history': include_history,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['template_id']
        path_param_values = self.encode_path_vars(template_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profile_templates/{template_id}/versions'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def create_profile_template_version(
        self,
        template_id: str,
        *,
        account_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        profile: Optional['TemplateProfileComponentRequest'] = None,
        policy_template_references: Optional[List['PolicyTemplateReference']] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create new version of a trusted profile template.

        Create a new version of a trusted profile template in an enterprise account.

        :param str template_id: ID of the trusted profile template.
        :param str account_id: (optional) ID of the account where the template
               resides.
        :param str name: (optional) The name of the trusted profile template. This
               is visible only in the enterprise account. Required field when creating a
               new template. Otherwise this field is optional. If the field is included it
               will change the name value for all existing versions of the template.
        :param str description: (optional) The description of the trusted profile
               template. Describe the template for enterprise account users.
        :param TemplateProfileComponentRequest profile: (optional) Input body
               parameters for the TemplateProfileComponent.
        :param List[PolicyTemplateReference] policy_template_references: (optional)
               Existing policy templates that you can reference to assign access in the
               trusted profile component.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TrustedProfileTemplateResponse` object
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        if profile is not None:
            profile = convert_model(profile)
        if policy_template_references is not None:
            policy_template_references = [convert_model(x) for x in policy_template_references]
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='create_profile_template_version',
        )
        headers.update(sdk_headers)

        data = {
            'account_id': account_id,
            'name': name,
            'description': description,
            'profile': profile,
            'policy_template_references': policy_template_references,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['template_id']
        path_param_values = self.encode_path_vars(template_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profile_templates/{template_id}/versions'.format(**path_param_dict)
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def get_profile_template_version(
        self,
        template_id: str,
        version: str,
        *,
        include_history: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get version of trusted profile template.

        Get a specific version of a trusted profile template in an enterprise account.

        :param str template_id: ID of the trusted profile template.
        :param str version: Version of the Profile Template.
        :param bool include_history: (optional) Defines if the entity history is
               included in the response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TrustedProfileTemplateResponse` object
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        if not version:
            raise ValueError('version must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_profile_template_version',
        )
        headers.update(sdk_headers)

        params = {
            'include_history': include_history,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['template_id', 'version']
        path_param_values = self.encode_path_vars(template_id, version)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profile_templates/{template_id}/versions/{version}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def update_profile_template_version(
        self,
        if_match: str,
        template_id: str,
        version: str,
        *,
        account_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        profile: Optional['TemplateProfileComponentRequest'] = None,
        policy_template_references: Optional[List['PolicyTemplateReference']] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Update version of trusted profile template.

        Update a specific version of a trusted profile template in an enterprise account.

        :param str if_match: Entity tag of the Template to be updated. Specify the
               tag that you retrieved when reading the Profile Template. This value helps
               identifying parallel usage of this API. Pass * to indicate to update any
               version available. This might result in stale updates.
        :param str template_id: ID of the trusted profile template.
        :param str version: Version of the Profile Template.
        :param str account_id: (optional) ID of the account where the template
               resides.
        :param str name: (optional) The name of the trusted profile template. This
               is visible only in the enterprise account. Required field when creating a
               new template. Otherwise this field is optional. If the field is included it
               will change the name value for all existing versions of the template.
        :param str description: (optional) The description of the trusted profile
               template. Describe the template for enterprise account users.
        :param TemplateProfileComponentRequest profile: (optional) Input body
               parameters for the TemplateProfileComponent.
        :param List[PolicyTemplateReference] policy_template_references: (optional)
               Existing policy templates that you can reference to assign access in the
               trusted profile component.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TrustedProfileTemplateResponse` object
        """

        if not if_match:
            raise ValueError('if_match must be provided')
        if not template_id:
            raise ValueError('template_id must be provided')
        if not version:
            raise ValueError('version must be provided')
        if profile is not None:
            profile = convert_model(profile)
        if policy_template_references is not None:
            policy_template_references = [convert_model(x) for x in policy_template_references]
        headers = {
            'If-Match': if_match,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='update_profile_template_version',
        )
        headers.update(sdk_headers)

        data = {
            'account_id': account_id,
            'name': name,
            'description': description,
            'profile': profile,
            'policy_template_references': policy_template_references,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['template_id', 'version']
        path_param_values = self.encode_path_vars(template_id, version)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profile_templates/{template_id}/versions/{version}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PUT',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_profile_template_version(
        self,
        template_id: str,
        version: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete version of trusted profile template.

        Delete a specific version of a trusted profile template in an enterprise account.
        If the version is assigned to child accounts, you must first delete the
        assignment.

        :param str template_id: ID of the trusted profile template.
        :param str version: Version of the Profile Template.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        if not version:
            raise ValueError('version must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='delete_profile_template_version',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['template_id', 'version']
        path_param_values = self.encode_path_vars(template_id, version)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profile_templates/{template_id}/versions/{version}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def commit_profile_template(
        self,
        template_id: str,
        version: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Commit a template version.

        Commit a specific version of a trusted profile template in an enterprise account.
        You must commit a template before you can assign it to child accounts. Once a
        template is committed, you can no longer modify the template.

        :param str template_id: ID of the trusted profile template.
        :param str version: Version of the Profile Template.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        if not version:
            raise ValueError('version must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='commit_profile_template',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['template_id', 'version']
        path_param_values = self.encode_path_vars(template_id, version)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/profile_templates/{template_id}/versions/{version}/commit'.format(**path_param_dict)
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response


class ListApiKeysEnums:
    """
    Enums for list_api_keys parameters.
    """

    class Scope(str, Enum):
        """
        Optional parameter to define the scope of the queried API keys. Can be 'entity'
        (default) or 'account'.
        """

        ENTITY = 'entity'
        ACCOUNT = 'account'

    class Type(str, Enum):
        """
        Optional parameter to filter the type of the queried API keys. Can be 'user' or
        'serviceid'.
        """

        USER = 'user'
        SERVICEID = 'serviceid'

    class Order(str, Enum):
        """
        Optional sort order, valid values are asc and desc. Default: asc.
        """

        ASC = 'asc'
        DESC = 'desc'


class ListServiceIdsEnums:
    """
    Enums for list_service_ids parameters.
    """

    class Order(str, Enum):
        """
        Optional sort order, valid values are asc and desc. Default: asc.
        """

        ASC = 'asc'
        DESC = 'desc'


class ListProfilesEnums:
    """
    Enums for list_profiles parameters.
    """

    class Order(str, Enum):
        """
        Optional sort order, valid values are asc and desc. Default: asc.
        """

        ASC = 'asc'
        DESC = 'desc'


class SetProfileIdentityEnums:
    """
    Enums for set_profile_identity parameters.
    """

    class IdentityType(str, Enum):
        """
        Type of the identity.
        """

        USER = 'user'
        SERVICEID = 'serviceid'
        CRN = 'crn'


class GetProfileIdentityEnums:
    """
    Enums for get_profile_identity parameters.
    """

    class IdentityType(str, Enum):
        """
        Type of the identity.
        """

        USER = 'user'
        SERVICEID = 'serviceid'
        CRN = 'crn'


class DeleteProfileIdentityEnums:
    """
    Enums for delete_profile_identity parameters.
    """

    class IdentityType(str, Enum):
        """
        Type of the identity.
        """

        USER = 'user'
        SERVICEID = 'serviceid'
        CRN = 'crn'


class ListAccountSettingsAssignmentsEnums:
    """
    Enums for list_account_settings_assignments parameters.
    """

    class TargetType(str, Enum):
        """
        Filter results by the assignment's target type.
        """

        ACCOUNT = 'Account'
        ACCOUNTGROUP = 'AccountGroup'

    class Sort(str, Enum):
        """
        If specified, the items are sorted by the value of this property.
        """

        TEMPLATE_ID = 'template_id'
        CREATED_AT = 'created_at'
        LAST_MODIFIED_AT = 'last_modified_at'

    class Order(str, Enum):
        """
        Sort order.
        """

        ASC = 'asc'
        DESC = 'desc'


class ListAccountSettingsTemplatesEnums:
    """
    Enums for list_account_settings_templates parameters.
    """

    class Sort(str, Enum):
        """
        Optional sort property. If specified, the returned templated are sorted according
        to this property.
        """

        CREATED_AT = 'created_at'
        LAST_MODIFIED_AT = 'last_modified_at'
        NAME = 'name'

    class Order(str, Enum):
        """
        Optional sort order.
        """

        ASC = 'asc'
        DESC = 'desc'


class ListVersionsOfAccountSettingsTemplateEnums:
    """
    Enums for list_versions_of_account_settings_template parameters.
    """

    class Sort(str, Enum):
        """
        Optional sort property. If specified, the returned templated are sorted according
        to this property.
        """

        CREATED_AT = 'created_at'
        LAST_MODIFIED_AT = 'last_modified_at'
        NAME = 'name'

    class Order(str, Enum):
        """
        Optional sort order.
        """

        ASC = 'asc'
        DESC = 'desc'


class ListTrustedProfileAssignmentsEnums:
    """
    Enums for list_trusted_profile_assignments parameters.
    """

    class TargetType(str, Enum):
        """
        Filter results by the assignment's target type.
        """

        ACCOUNT = 'Account'
        ACCOUNTGROUP = 'AccountGroup'

    class Sort(str, Enum):
        """
        If specified, the items are sorted by the value of this property.
        """

        TEMPLATE_ID = 'template_id'
        CREATED_AT = 'created_at'
        LAST_MODIFIED_AT = 'last_modified_at'

    class Order(str, Enum):
        """
        Sort order.
        """

        ASC = 'asc'
        DESC = 'desc'


class ListProfileTemplatesEnums:
    """
    Enums for list_profile_templates parameters.
    """

    class Sort(str, Enum):
        """
        Optional sort property. If specified, the returned templates are sorted according
        to this property.
        """

        CREATED_AT = 'created_at'
        LAST_MODIFIED_AT = 'last_modified_at'
        NAME = 'name'

    class Order(str, Enum):
        """
        Optional sort order.
        """

        ASC = 'asc'
        DESC = 'desc'


class ListVersionsOfProfileTemplateEnums:
    """
    Enums for list_versions_of_profile_template parameters.
    """

    class Sort(str, Enum):
        """
        Optional sort property. If specified, the returned templated are sorted according
        to this property.
        """

        CREATED_AT = 'created_at'
        LAST_MODIFIED_AT = 'last_modified_at'
        NAME = 'name'

    class Order(str, Enum):
        """
        Optional sort order.
        """

        ASC = 'asc'
        DESC = 'desc'


##############################################################################
# Models
##############################################################################


class AccountBasedMfaEnrollment:
    """
    AccountBasedMfaEnrollment.

    :param MfaEnrollmentTypeStatus security_questions:
    :param MfaEnrollmentTypeStatus totp:
    :param MfaEnrollmentTypeStatus verisign:
    :param bool complies: The enrollment complies to the effective requirement.
    """

    def __init__(
        self,
        security_questions: 'MfaEnrollmentTypeStatus',
        totp: 'MfaEnrollmentTypeStatus',
        verisign: 'MfaEnrollmentTypeStatus',
        complies: bool,
    ) -> None:
        """
        Initialize a AccountBasedMfaEnrollment object.

        :param MfaEnrollmentTypeStatus security_questions:
        :param MfaEnrollmentTypeStatus totp:
        :param MfaEnrollmentTypeStatus verisign:
        :param bool complies: The enrollment complies to the effective requirement.
        """
        self.security_questions = security_questions
        self.totp = totp
        self.verisign = verisign
        self.complies = complies

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccountBasedMfaEnrollment':
        """Initialize a AccountBasedMfaEnrollment object from a json dictionary."""
        args = {}
        if (security_questions := _dict.get('security_questions')) is not None:
            args['security_questions'] = MfaEnrollmentTypeStatus.from_dict(security_questions)
        else:
            raise ValueError('Required property \'security_questions\' not present in AccountBasedMfaEnrollment JSON')
        if (totp := _dict.get('totp')) is not None:
            args['totp'] = MfaEnrollmentTypeStatus.from_dict(totp)
        else:
            raise ValueError('Required property \'totp\' not present in AccountBasedMfaEnrollment JSON')
        if (verisign := _dict.get('verisign')) is not None:
            args['verisign'] = MfaEnrollmentTypeStatus.from_dict(verisign)
        else:
            raise ValueError('Required property \'verisign\' not present in AccountBasedMfaEnrollment JSON')
        if (complies := _dict.get('complies')) is not None:
            args['complies'] = complies
        else:
            raise ValueError('Required property \'complies\' not present in AccountBasedMfaEnrollment JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccountBasedMfaEnrollment object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'security_questions') and self.security_questions is not None:
            if isinstance(self.security_questions, dict):
                _dict['security_questions'] = self.security_questions
            else:
                _dict['security_questions'] = self.security_questions.to_dict()
        if hasattr(self, 'totp') and self.totp is not None:
            if isinstance(self.totp, dict):
                _dict['totp'] = self.totp
            else:
                _dict['totp'] = self.totp.to_dict()
        if hasattr(self, 'verisign') and self.verisign is not None:
            if isinstance(self.verisign, dict):
                _dict['verisign'] = self.verisign
            else:
                _dict['verisign'] = self.verisign.to_dict()
        if hasattr(self, 'complies') and self.complies is not None:
            _dict['complies'] = self.complies
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AccountBasedMfaEnrollment object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccountBasedMfaEnrollment') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccountBasedMfaEnrollment') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AccountSettingsComponent:
    """
    AccountSettingsComponent.

    :param str restrict_create_service_id: (optional) Defines whether or not
          creating a service ID is access controlled. Valid values:
            * RESTRICTED - only users assigned the 'Service ID creator' role on the IAM
          Identity Service can create service IDs, including the account owner
            * NOT_RESTRICTED - all members of an account can create service IDs
            * NOT_SET - to 'unset' a previous set value.
    :param str restrict_create_platform_apikey: (optional) Defines whether or not
          creating platform API keys is access controlled. Valid values:
            * RESTRICTED - to apply access control
            * NOT_RESTRICTED - to remove access control
            * NOT_SET - to 'unset' a previous set value.
    :param str allowed_ip_addresses: (optional) Defines the IP addresses and subnets
          from which IAM tokens can be created for the account.
    :param str mfa: (optional) Defines the MFA trait for the account. Valid values:
            * NONE - No MFA trait set
            * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
            * TOTP - For all non-federated IBMId users
            * TOTP4ALL - For all users
            * LEVEL1 - Email-based MFA for all users
            * LEVEL2 - TOTP-based MFA for all users
            * LEVEL3 - U2F MFA for all users.
    :param List[AccountSettingsUserMFA] user_mfa: (optional) List of users that are
          exempted from the MFA requirement of the account.
    :param str session_expiration_in_seconds: (optional) Defines the session
          expiration in seconds for the account. Valid values:
            * Any whole number between between '900' and '86400'
            * NOT_SET - To unset account setting and use service default.
    :param str session_invalidation_in_seconds: (optional) Defines the period of
          time in seconds in which a session will be invalidated due to inactivity. Valid
          values:
            * Any whole number between '900' and '7200'
            * NOT_SET - To unset account setting and use service default.
    :param str max_sessions_per_identity: (optional) Defines the max allowed
          sessions per identity required by the account. Valid values:
            * Any whole number greater than 0
            * NOT_SET - To unset account setting and use service default.
    :param str system_access_token_expiration_in_seconds: (optional) Defines the
          access token expiration in seconds. Valid values:
            * Any whole number between '900' and '3600'
            * NOT_SET - To unset account setting and use service default.
    :param str system_refresh_token_expiration_in_seconds: (optional) Defines the
          refresh token expiration in seconds. Valid values:
            * Any whole number between '900' and '259200'
            * NOT_SET - To unset account setting and use service default.
    """

    def __init__(
        self,
        *,
        restrict_create_service_id: Optional[str] = None,
        restrict_create_platform_apikey: Optional[str] = None,
        allowed_ip_addresses: Optional[str] = None,
        mfa: Optional[str] = None,
        user_mfa: Optional[List['AccountSettingsUserMFA']] = None,
        session_expiration_in_seconds: Optional[str] = None,
        session_invalidation_in_seconds: Optional[str] = None,
        max_sessions_per_identity: Optional[str] = None,
        system_access_token_expiration_in_seconds: Optional[str] = None,
        system_refresh_token_expiration_in_seconds: Optional[str] = None,
    ) -> None:
        """
        Initialize a AccountSettingsComponent object.

        :param str restrict_create_service_id: (optional) Defines whether or not
               creating a service ID is access controlled. Valid values:
                 * RESTRICTED - only users assigned the 'Service ID creator' role on the
               IAM Identity Service can create service IDs, including the account owner
                 * NOT_RESTRICTED - all members of an account can create service IDs
                 * NOT_SET - to 'unset' a previous set value.
        :param str restrict_create_platform_apikey: (optional) Defines whether or
               not creating platform API keys is access controlled. Valid values:
                 * RESTRICTED - to apply access control
                 * NOT_RESTRICTED - to remove access control
                 * NOT_SET - to 'unset' a previous set value.
        :param str allowed_ip_addresses: (optional) Defines the IP addresses and
               subnets from which IAM tokens can be created for the account.
        :param str mfa: (optional) Defines the MFA trait for the account. Valid
               values:
                 * NONE - No MFA trait set
                 * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
                 * TOTP - For all non-federated IBMId users
                 * TOTP4ALL - For all users
                 * LEVEL1 - Email-based MFA for all users
                 * LEVEL2 - TOTP-based MFA for all users
                 * LEVEL3 - U2F MFA for all users.
        :param List[AccountSettingsUserMFA] user_mfa: (optional) List of users that
               are exempted from the MFA requirement of the account.
        :param str session_expiration_in_seconds: (optional) Defines the session
               expiration in seconds for the account. Valid values:
                 * Any whole number between between '900' and '86400'
                 * NOT_SET - To unset account setting and use service default.
        :param str session_invalidation_in_seconds: (optional) Defines the period
               of time in seconds in which a session will be invalidated due to
               inactivity. Valid values:
                 * Any whole number between '900' and '7200'
                 * NOT_SET - To unset account setting and use service default.
        :param str max_sessions_per_identity: (optional) Defines the max allowed
               sessions per identity required by the account. Valid values:
                 * Any whole number greater than 0
                 * NOT_SET - To unset account setting and use service default.
        :param str system_access_token_expiration_in_seconds: (optional) Defines
               the access token expiration in seconds. Valid values:
                 * Any whole number between '900' and '3600'
                 * NOT_SET - To unset account setting and use service default.
        :param str system_refresh_token_expiration_in_seconds: (optional) Defines
               the refresh token expiration in seconds. Valid values:
                 * Any whole number between '900' and '259200'
                 * NOT_SET - To unset account setting and use service default.
        """
        self.restrict_create_service_id = restrict_create_service_id
        self.restrict_create_platform_apikey = restrict_create_platform_apikey
        self.allowed_ip_addresses = allowed_ip_addresses
        self.mfa = mfa
        self.user_mfa = user_mfa
        self.session_expiration_in_seconds = session_expiration_in_seconds
        self.session_invalidation_in_seconds = session_invalidation_in_seconds
        self.max_sessions_per_identity = max_sessions_per_identity
        self.system_access_token_expiration_in_seconds = system_access_token_expiration_in_seconds
        self.system_refresh_token_expiration_in_seconds = system_refresh_token_expiration_in_seconds

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccountSettingsComponent':
        """Initialize a AccountSettingsComponent object from a json dictionary."""
        args = {}
        if (restrict_create_service_id := _dict.get('restrict_create_service_id')) is not None:
            args['restrict_create_service_id'] = restrict_create_service_id
        if (restrict_create_platform_apikey := _dict.get('restrict_create_platform_apikey')) is not None:
            args['restrict_create_platform_apikey'] = restrict_create_platform_apikey
        if (allowed_ip_addresses := _dict.get('allowed_ip_addresses')) is not None:
            args['allowed_ip_addresses'] = allowed_ip_addresses
        if (mfa := _dict.get('mfa')) is not None:
            args['mfa'] = mfa
        if (user_mfa := _dict.get('user_mfa')) is not None:
            args['user_mfa'] = [AccountSettingsUserMFA.from_dict(v) for v in user_mfa]
        if (session_expiration_in_seconds := _dict.get('session_expiration_in_seconds')) is not None:
            args['session_expiration_in_seconds'] = session_expiration_in_seconds
        if (session_invalidation_in_seconds := _dict.get('session_invalidation_in_seconds')) is not None:
            args['session_invalidation_in_seconds'] = session_invalidation_in_seconds
        if (max_sessions_per_identity := _dict.get('max_sessions_per_identity')) is not None:
            args['max_sessions_per_identity'] = max_sessions_per_identity
        if (
            system_access_token_expiration_in_seconds := _dict.get('system_access_token_expiration_in_seconds')
        ) is not None:
            args['system_access_token_expiration_in_seconds'] = system_access_token_expiration_in_seconds
        if (
            system_refresh_token_expiration_in_seconds := _dict.get('system_refresh_token_expiration_in_seconds')
        ) is not None:
            args['system_refresh_token_expiration_in_seconds'] = system_refresh_token_expiration_in_seconds
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccountSettingsComponent object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'restrict_create_service_id') and self.restrict_create_service_id is not None:
            _dict['restrict_create_service_id'] = self.restrict_create_service_id
        if hasattr(self, 'restrict_create_platform_apikey') and self.restrict_create_platform_apikey is not None:
            _dict['restrict_create_platform_apikey'] = self.restrict_create_platform_apikey
        if hasattr(self, 'allowed_ip_addresses') and self.allowed_ip_addresses is not None:
            _dict['allowed_ip_addresses'] = self.allowed_ip_addresses
        if hasattr(self, 'mfa') and self.mfa is not None:
            _dict['mfa'] = self.mfa
        if hasattr(self, 'user_mfa') and self.user_mfa is not None:
            user_mfa_list = []
            for v in self.user_mfa:
                if isinstance(v, dict):
                    user_mfa_list.append(v)
                else:
                    user_mfa_list.append(v.to_dict())
            _dict['user_mfa'] = user_mfa_list
        if hasattr(self, 'session_expiration_in_seconds') and self.session_expiration_in_seconds is not None:
            _dict['session_expiration_in_seconds'] = self.session_expiration_in_seconds
        if hasattr(self, 'session_invalidation_in_seconds') and self.session_invalidation_in_seconds is not None:
            _dict['session_invalidation_in_seconds'] = self.session_invalidation_in_seconds
        if hasattr(self, 'max_sessions_per_identity') and self.max_sessions_per_identity is not None:
            _dict['max_sessions_per_identity'] = self.max_sessions_per_identity
        if (
            hasattr(self, 'system_access_token_expiration_in_seconds')
            and self.system_access_token_expiration_in_seconds is not None
        ):
            _dict['system_access_token_expiration_in_seconds'] = self.system_access_token_expiration_in_seconds
        if (
            hasattr(self, 'system_refresh_token_expiration_in_seconds')
            and self.system_refresh_token_expiration_in_seconds is not None
        ):
            _dict['system_refresh_token_expiration_in_seconds'] = self.system_refresh_token_expiration_in_seconds
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AccountSettingsComponent object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccountSettingsComponent') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccountSettingsComponent') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class RestrictCreateServiceIdEnum(str, Enum):
        """
        Defines whether or not creating a service ID is access controlled. Valid values:
          * RESTRICTED - only users assigned the 'Service ID creator' role on the IAM
        Identity Service can create service IDs, including the account owner
          * NOT_RESTRICTED - all members of an account can create service IDs
          * NOT_SET - to 'unset' a previous set value.
        """

        RESTRICTED = 'RESTRICTED'
        NOT_RESTRICTED = 'NOT_RESTRICTED'
        NOT_SET = 'NOT_SET'

    class RestrictCreatePlatformApikeyEnum(str, Enum):
        """
        Defines whether or not creating platform API keys is access controlled. Valid
        values:
          * RESTRICTED - to apply access control
          * NOT_RESTRICTED - to remove access control
          * NOT_SET - to 'unset' a previous set value.
        """

        RESTRICTED = 'RESTRICTED'
        NOT_RESTRICTED = 'NOT_RESTRICTED'
        NOT_SET = 'NOT_SET'

    class MfaEnum(str, Enum):
        """
        Defines the MFA trait for the account. Valid values:
          * NONE - No MFA trait set
          * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
          * TOTP - For all non-federated IBMId users
          * TOTP4ALL - For all users
          * LEVEL1 - Email-based MFA for all users
          * LEVEL2 - TOTP-based MFA for all users
          * LEVEL3 - U2F MFA for all users.
        """

        NONE = 'NONE'
        NONE_NO_ROPC = 'NONE_NO_ROPC'
        TOTP = 'TOTP'
        TOTP4ALL = 'TOTP4ALL'
        LEVEL1 = 'LEVEL1'
        LEVEL2 = 'LEVEL2'
        LEVEL3 = 'LEVEL3'


class AccountSettingsResponse:
    """
    Response body format for Account Settings REST requests.

    :param ResponseContext context: (optional) Context with key properties for
          problem determination.
    :param str account_id: Unique ID of the account.
    :param str restrict_create_service_id: Defines whether or not creating a service
          ID is access controlled. Valid values:
            * RESTRICTED - only users assigned the 'Service ID creator' role on the IAM
          Identity Service can create service IDs, including the account owner
            * NOT_RESTRICTED - all members of an account can create service IDs
            * NOT_SET - to 'unset' a previous set value.
    :param str restrict_create_platform_apikey: Defines whether or not creating
          platform API keys is access controlled. Valid values:
            * RESTRICTED - to apply access control
            * NOT_RESTRICTED - to remove access control
            * NOT_SET - to 'unset' a previous set value.
    :param str allowed_ip_addresses: Defines the IP addresses and subnets from which
          IAM tokens can be created for the account.
    :param str entity_tag: Version of the account settings.
    :param str mfa: Defines the MFA trait for the account. Valid values:
            * NONE - No MFA trait set
            * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
            * TOTP - For all non-federated IBMId users
            * TOTP4ALL - For all users
            * LEVEL1 - Email-based MFA for all users
            * LEVEL2 - TOTP-based MFA for all users
            * LEVEL3 - U2F MFA for all users.
    :param List[AccountSettingsUserMFA] user_mfa: List of users that are exempted
          from the MFA requirement of the account.
    :param List[EnityHistoryRecord] history: (optional) History of the Account
          Settings.
    :param str session_expiration_in_seconds: Defines the session expiration in
          seconds for the account. Valid values:
            * Any whole number between between '900' and '86400'
            * NOT_SET - To unset account setting and use service default.
    :param str session_invalidation_in_seconds: Defines the period of time in
          seconds in which a session will be invalidated due to inactivity. Valid values:
            * Any whole number between '900' and '7200'
            * NOT_SET - To unset account setting and use service default.
    :param str max_sessions_per_identity: Defines the max allowed sessions per
          identity required by the account. Valid values:
            * Any whole number greater than 0
            * NOT_SET - To unset account setting and use service default.
    :param str system_access_token_expiration_in_seconds: Defines the access token
          expiration in seconds. Valid values:
            * Any whole number between '900' and '3600'
            * NOT_SET - To unset account setting and use service default.
    :param str system_refresh_token_expiration_in_seconds: Defines the refresh token
          expiration in seconds. Valid values:
            * Any whole number between '900' and '259200'
            * NOT_SET - To unset account setting and use service default.
    """

    def __init__(
        self,
        account_id: str,
        restrict_create_service_id: str,
        restrict_create_platform_apikey: str,
        allowed_ip_addresses: str,
        entity_tag: str,
        mfa: str,
        user_mfa: List['AccountSettingsUserMFA'],
        session_expiration_in_seconds: str,
        session_invalidation_in_seconds: str,
        max_sessions_per_identity: str,
        system_access_token_expiration_in_seconds: str,
        system_refresh_token_expiration_in_seconds: str,
        *,
        context: Optional['ResponseContext'] = None,
        history: Optional[List['EnityHistoryRecord']] = None,
    ) -> None:
        """
        Initialize a AccountSettingsResponse object.

        :param str account_id: Unique ID of the account.
        :param str restrict_create_service_id: Defines whether or not creating a
               service ID is access controlled. Valid values:
                 * RESTRICTED - only users assigned the 'Service ID creator' role on the
               IAM Identity Service can create service IDs, including the account owner
                 * NOT_RESTRICTED - all members of an account can create service IDs
                 * NOT_SET - to 'unset' a previous set value.
        :param str restrict_create_platform_apikey: Defines whether or not creating
               platform API keys is access controlled. Valid values:
                 * RESTRICTED - to apply access control
                 * NOT_RESTRICTED - to remove access control
                 * NOT_SET - to 'unset' a previous set value.
        :param str allowed_ip_addresses: Defines the IP addresses and subnets from
               which IAM tokens can be created for the account.
        :param str entity_tag: Version of the account settings.
        :param str mfa: Defines the MFA trait for the account. Valid values:
                 * NONE - No MFA trait set
                 * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
                 * TOTP - For all non-federated IBMId users
                 * TOTP4ALL - For all users
                 * LEVEL1 - Email-based MFA for all users
                 * LEVEL2 - TOTP-based MFA for all users
                 * LEVEL3 - U2F MFA for all users.
        :param List[AccountSettingsUserMFA] user_mfa: List of users that are
               exempted from the MFA requirement of the account.
        :param str session_expiration_in_seconds: Defines the session expiration in
               seconds for the account. Valid values:
                 * Any whole number between between '900' and '86400'
                 * NOT_SET - To unset account setting and use service default.
        :param str session_invalidation_in_seconds: Defines the period of time in
               seconds in which a session will be invalidated due to inactivity. Valid
               values:
                 * Any whole number between '900' and '7200'
                 * NOT_SET - To unset account setting and use service default.
        :param str max_sessions_per_identity: Defines the max allowed sessions per
               identity required by the account. Valid values:
                 * Any whole number greater than 0
                 * NOT_SET - To unset account setting and use service default.
        :param str system_access_token_expiration_in_seconds: Defines the access
               token expiration in seconds. Valid values:
                 * Any whole number between '900' and '3600'
                 * NOT_SET - To unset account setting and use service default.
        :param str system_refresh_token_expiration_in_seconds: Defines the refresh
               token expiration in seconds. Valid values:
                 * Any whole number between '900' and '259200'
                 * NOT_SET - To unset account setting and use service default.
        :param ResponseContext context: (optional) Context with key properties for
               problem determination.
        :param List[EnityHistoryRecord] history: (optional) History of the Account
               Settings.
        """
        self.context = context
        self.account_id = account_id
        self.restrict_create_service_id = restrict_create_service_id
        self.restrict_create_platform_apikey = restrict_create_platform_apikey
        self.allowed_ip_addresses = allowed_ip_addresses
        self.entity_tag = entity_tag
        self.mfa = mfa
        self.user_mfa = user_mfa
        self.history = history
        self.session_expiration_in_seconds = session_expiration_in_seconds
        self.session_invalidation_in_seconds = session_invalidation_in_seconds
        self.max_sessions_per_identity = max_sessions_per_identity
        self.system_access_token_expiration_in_seconds = system_access_token_expiration_in_seconds
        self.system_refresh_token_expiration_in_seconds = system_refresh_token_expiration_in_seconds

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccountSettingsResponse':
        """Initialize a AccountSettingsResponse object from a json dictionary."""
        args = {}
        if (context := _dict.get('context')) is not None:
            args['context'] = ResponseContext.from_dict(context)
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        else:
            raise ValueError('Required property \'account_id\' not present in AccountSettingsResponse JSON')
        if (restrict_create_service_id := _dict.get('restrict_create_service_id')) is not None:
            args['restrict_create_service_id'] = restrict_create_service_id
        else:
            raise ValueError(
                'Required property \'restrict_create_service_id\' not present in AccountSettingsResponse JSON'
            )
        if (restrict_create_platform_apikey := _dict.get('restrict_create_platform_apikey')) is not None:
            args['restrict_create_platform_apikey'] = restrict_create_platform_apikey
        else:
            raise ValueError(
                'Required property \'restrict_create_platform_apikey\' not present in AccountSettingsResponse JSON'
            )
        if (allowed_ip_addresses := _dict.get('allowed_ip_addresses')) is not None:
            args['allowed_ip_addresses'] = allowed_ip_addresses
        else:
            raise ValueError('Required property \'allowed_ip_addresses\' not present in AccountSettingsResponse JSON')
        if (entity_tag := _dict.get('entity_tag')) is not None:
            args['entity_tag'] = entity_tag
        else:
            raise ValueError('Required property \'entity_tag\' not present in AccountSettingsResponse JSON')
        if (mfa := _dict.get('mfa')) is not None:
            args['mfa'] = mfa
        else:
            raise ValueError('Required property \'mfa\' not present in AccountSettingsResponse JSON')
        if (user_mfa := _dict.get('user_mfa')) is not None:
            args['user_mfa'] = [AccountSettingsUserMFA.from_dict(v) for v in user_mfa]
        else:
            raise ValueError('Required property \'user_mfa\' not present in AccountSettingsResponse JSON')
        if (history := _dict.get('history')) is not None:
            args['history'] = [EnityHistoryRecord.from_dict(v) for v in history]
        if (session_expiration_in_seconds := _dict.get('session_expiration_in_seconds')) is not None:
            args['session_expiration_in_seconds'] = session_expiration_in_seconds
        else:
            raise ValueError(
                'Required property \'session_expiration_in_seconds\' not present in AccountSettingsResponse JSON'
            )
        if (session_invalidation_in_seconds := _dict.get('session_invalidation_in_seconds')) is not None:
            args['session_invalidation_in_seconds'] = session_invalidation_in_seconds
        else:
            raise ValueError(
                'Required property \'session_invalidation_in_seconds\' not present in AccountSettingsResponse JSON'
            )
        if (max_sessions_per_identity := _dict.get('max_sessions_per_identity')) is not None:
            args['max_sessions_per_identity'] = max_sessions_per_identity
        else:
            raise ValueError(
                'Required property \'max_sessions_per_identity\' not present in AccountSettingsResponse JSON'
            )
        if (
            system_access_token_expiration_in_seconds := _dict.get('system_access_token_expiration_in_seconds')
        ) is not None:
            args['system_access_token_expiration_in_seconds'] = system_access_token_expiration_in_seconds
        else:
            raise ValueError(
                'Required property \'system_access_token_expiration_in_seconds\' not present in AccountSettingsResponse JSON'
            )
        if (
            system_refresh_token_expiration_in_seconds := _dict.get('system_refresh_token_expiration_in_seconds')
        ) is not None:
            args['system_refresh_token_expiration_in_seconds'] = system_refresh_token_expiration_in_seconds
        else:
            raise ValueError(
                'Required property \'system_refresh_token_expiration_in_seconds\' not present in AccountSettingsResponse JSON'
            )
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccountSettingsResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'context') and self.context is not None:
            if isinstance(self.context, dict):
                _dict['context'] = self.context
            else:
                _dict['context'] = self.context.to_dict()
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'restrict_create_service_id') and self.restrict_create_service_id is not None:
            _dict['restrict_create_service_id'] = self.restrict_create_service_id
        if hasattr(self, 'restrict_create_platform_apikey') and self.restrict_create_platform_apikey is not None:
            _dict['restrict_create_platform_apikey'] = self.restrict_create_platform_apikey
        if hasattr(self, 'allowed_ip_addresses') and self.allowed_ip_addresses is not None:
            _dict['allowed_ip_addresses'] = self.allowed_ip_addresses
        if hasattr(self, 'entity_tag') and self.entity_tag is not None:
            _dict['entity_tag'] = self.entity_tag
        if hasattr(self, 'mfa') and self.mfa is not None:
            _dict['mfa'] = self.mfa
        if hasattr(self, 'user_mfa') and self.user_mfa is not None:
            user_mfa_list = []
            for v in self.user_mfa:
                if isinstance(v, dict):
                    user_mfa_list.append(v)
                else:
                    user_mfa_list.append(v.to_dict())
            _dict['user_mfa'] = user_mfa_list
        if hasattr(self, 'history') and self.history is not None:
            history_list = []
            for v in self.history:
                if isinstance(v, dict):
                    history_list.append(v)
                else:
                    history_list.append(v.to_dict())
            _dict['history'] = history_list
        if hasattr(self, 'session_expiration_in_seconds') and self.session_expiration_in_seconds is not None:
            _dict['session_expiration_in_seconds'] = self.session_expiration_in_seconds
        if hasattr(self, 'session_invalidation_in_seconds') and self.session_invalidation_in_seconds is not None:
            _dict['session_invalidation_in_seconds'] = self.session_invalidation_in_seconds
        if hasattr(self, 'max_sessions_per_identity') and self.max_sessions_per_identity is not None:
            _dict['max_sessions_per_identity'] = self.max_sessions_per_identity
        if (
            hasattr(self, 'system_access_token_expiration_in_seconds')
            and self.system_access_token_expiration_in_seconds is not None
        ):
            _dict['system_access_token_expiration_in_seconds'] = self.system_access_token_expiration_in_seconds
        if (
            hasattr(self, 'system_refresh_token_expiration_in_seconds')
            and self.system_refresh_token_expiration_in_seconds is not None
        ):
            _dict['system_refresh_token_expiration_in_seconds'] = self.system_refresh_token_expiration_in_seconds
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AccountSettingsResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccountSettingsResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccountSettingsResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class RestrictCreateServiceIdEnum(str, Enum):
        """
        Defines whether or not creating a service ID is access controlled. Valid values:
          * RESTRICTED - only users assigned the 'Service ID creator' role on the IAM
        Identity Service can create service IDs, including the account owner
          * NOT_RESTRICTED - all members of an account can create service IDs
          * NOT_SET - to 'unset' a previous set value.
        """

        RESTRICTED = 'RESTRICTED'
        NOT_RESTRICTED = 'NOT_RESTRICTED'
        NOT_SET = 'NOT_SET'

    class RestrictCreatePlatformApikeyEnum(str, Enum):
        """
        Defines whether or not creating platform API keys is access controlled. Valid
        values:
          * RESTRICTED - to apply access control
          * NOT_RESTRICTED - to remove access control
          * NOT_SET - to 'unset' a previous set value.
        """

        RESTRICTED = 'RESTRICTED'
        NOT_RESTRICTED = 'NOT_RESTRICTED'
        NOT_SET = 'NOT_SET'

    class MfaEnum(str, Enum):
        """
        Defines the MFA trait for the account. Valid values:
          * NONE - No MFA trait set
          * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
          * TOTP - For all non-federated IBMId users
          * TOTP4ALL - For all users
          * LEVEL1 - Email-based MFA for all users
          * LEVEL2 - TOTP-based MFA for all users
          * LEVEL3 - U2F MFA for all users.
        """

        NONE = 'NONE'
        NONE_NO_ROPC = 'NONE_NO_ROPC'
        TOTP = 'TOTP'
        TOTP4ALL = 'TOTP4ALL'
        LEVEL1 = 'LEVEL1'
        LEVEL2 = 'LEVEL2'
        LEVEL3 = 'LEVEL3'


class AccountSettingsTemplateList:
    """
    AccountSettingsTemplateList.

    :param ResponseContext context: (optional) Context with key properties for
          problem determination.
    :param int offset: (optional) The offset of the current page.
    :param int limit: (optional) Optional size of a single page.
    :param str first: (optional) Link to the first page.
    :param str previous: (optional) Link to the previous available page. If
          'previous' property is not part of the response no previous page is available.
    :param str next: (optional) Link to the next available page. If 'next' property
          is not part of the response no next page is available.
    :param List[AccountSettingsTemplateResponse] account_settings_templates: List of
          account settings templates based on the query paramters and the page size. The
          account_settings_templates array is always part of the response but might be
          empty depending on the query parameter values provided.
    """

    def __init__(
        self,
        account_settings_templates: List['AccountSettingsTemplateResponse'],
        *,
        context: Optional['ResponseContext'] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        first: Optional[str] = None,
        previous: Optional[str] = None,
        next: Optional[str] = None,
    ) -> None:
        """
        Initialize a AccountSettingsTemplateList object.

        :param List[AccountSettingsTemplateResponse] account_settings_templates:
               List of account settings templates based on the query paramters and the
               page size. The account_settings_templates array is always part of the
               response but might be empty depending on the query parameter values
               provided.
        :param ResponseContext context: (optional) Context with key properties for
               problem determination.
        :param int offset: (optional) The offset of the current page.
        :param int limit: (optional) Optional size of a single page.
        :param str first: (optional) Link to the first page.
        :param str previous: (optional) Link to the previous available page. If
               'previous' property is not part of the response no previous page is
               available.
        :param str next: (optional) Link to the next available page. If 'next'
               property is not part of the response no next page is available.
        """
        self.context = context
        self.offset = offset
        self.limit = limit
        self.first = first
        self.previous = previous
        self.next = next
        self.account_settings_templates = account_settings_templates

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccountSettingsTemplateList':
        """Initialize a AccountSettingsTemplateList object from a json dictionary."""
        args = {}
        if (context := _dict.get('context')) is not None:
            args['context'] = ResponseContext.from_dict(context)
        if (offset := _dict.get('offset')) is not None:
            args['offset'] = offset
        if (limit := _dict.get('limit')) is not None:
            args['limit'] = limit
        if (first := _dict.get('first')) is not None:
            args['first'] = first
        if (previous := _dict.get('previous')) is not None:
            args['previous'] = previous
        if (next := _dict.get('next')) is not None:
            args['next'] = next
        if (account_settings_templates := _dict.get('account_settings_templates')) is not None:
            args['account_settings_templates'] = [
                AccountSettingsTemplateResponse.from_dict(v) for v in account_settings_templates
            ]
        else:
            raise ValueError(
                'Required property \'account_settings_templates\' not present in AccountSettingsTemplateList JSON'
            )
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccountSettingsTemplateList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'context') and self.context is not None:
            if isinstance(self.context, dict):
                _dict['context'] = self.context
            else:
                _dict['context'] = self.context.to_dict()
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'first') and self.first is not None:
            _dict['first'] = self.first
        if hasattr(self, 'previous') and self.previous is not None:
            _dict['previous'] = self.previous
        if hasattr(self, 'next') and self.next is not None:
            _dict['next'] = self.next
        if hasattr(self, 'account_settings_templates') and self.account_settings_templates is not None:
            account_settings_templates_list = []
            for v in self.account_settings_templates:
                if isinstance(v, dict):
                    account_settings_templates_list.append(v)
                else:
                    account_settings_templates_list.append(v.to_dict())
            _dict['account_settings_templates'] = account_settings_templates_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AccountSettingsTemplateList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccountSettingsTemplateList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccountSettingsTemplateList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AccountSettingsTemplateResponse:
    """
    Response body format for account settings template REST requests.

    :param str id: ID of the the template.
    :param int version: Version of the the template.
    :param str account_id: ID of the account where the template resides.
    :param str name: The name of the trusted profile template. This is visible only
          in the enterprise account.
    :param str description: (optional) The description of the trusted profile
          template. Describe the template for enterprise account users.
    :param bool committed: Committed flag determines if the template is ready for
          assignment.
    :param AccountSettingsComponent account_settings:
    :param List[EnityHistoryRecord] history: (optional) History of the Template.
    :param str entity_tag: Entity tag for this templateId-version combination.
    :param str crn: Cloud resource name.
    :param str created_at: (optional) Template Created At.
    :param str created_by_id: (optional) IAMid of the creator.
    :param str last_modified_at: (optional) Template last modified at.
    :param str last_modified_by_id: (optional) IAMid of the identity that made the
          latest modification.
    """

    def __init__(
        self,
        id: str,
        version: int,
        account_id: str,
        name: str,
        committed: bool,
        account_settings: 'AccountSettingsComponent',
        entity_tag: str,
        crn: str,
        *,
        description: Optional[str] = None,
        history: Optional[List['EnityHistoryRecord']] = None,
        created_at: Optional[str] = None,
        created_by_id: Optional[str] = None,
        last_modified_at: Optional[str] = None,
        last_modified_by_id: Optional[str] = None,
    ) -> None:
        """
        Initialize a AccountSettingsTemplateResponse object.

        :param str id: ID of the the template.
        :param int version: Version of the the template.
        :param str account_id: ID of the account where the template resides.
        :param str name: The name of the trusted profile template. This is visible
               only in the enterprise account.
        :param bool committed: Committed flag determines if the template is ready
               for assignment.
        :param AccountSettingsComponent account_settings:
        :param str entity_tag: Entity tag for this templateId-version combination.
        :param str crn: Cloud resource name.
        :param str description: (optional) The description of the trusted profile
               template. Describe the template for enterprise account users.
        :param List[EnityHistoryRecord] history: (optional) History of the
               Template.
        :param str created_at: (optional) Template Created At.
        :param str created_by_id: (optional) IAMid of the creator.
        :param str last_modified_at: (optional) Template last modified at.
        :param str last_modified_by_id: (optional) IAMid of the identity that made
               the latest modification.
        """
        self.id = id
        self.version = version
        self.account_id = account_id
        self.name = name
        self.description = description
        self.committed = committed
        self.account_settings = account_settings
        self.history = history
        self.entity_tag = entity_tag
        self.crn = crn
        self.created_at = created_at
        self.created_by_id = created_by_id
        self.last_modified_at = last_modified_at
        self.last_modified_by_id = last_modified_by_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccountSettingsTemplateResponse':
        """Initialize a AccountSettingsTemplateResponse object from a json dictionary."""
        args = {}
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        else:
            raise ValueError('Required property \'id\' not present in AccountSettingsTemplateResponse JSON')
        if (version := _dict.get('version')) is not None:
            args['version'] = version
        else:
            raise ValueError('Required property \'version\' not present in AccountSettingsTemplateResponse JSON')
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        else:
            raise ValueError('Required property \'account_id\' not present in AccountSettingsTemplateResponse JSON')
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        else:
            raise ValueError('Required property \'name\' not present in AccountSettingsTemplateResponse JSON')
        if (description := _dict.get('description')) is not None:
            args['description'] = description
        if (committed := _dict.get('committed')) is not None:
            args['committed'] = committed
        else:
            raise ValueError('Required property \'committed\' not present in AccountSettingsTemplateResponse JSON')
        if (account_settings := _dict.get('account_settings')) is not None:
            args['account_settings'] = AccountSettingsComponent.from_dict(account_settings)
        else:
            raise ValueError(
                'Required property \'account_settings\' not present in AccountSettingsTemplateResponse JSON'
            )
        if (history := _dict.get('history')) is not None:
            args['history'] = [EnityHistoryRecord.from_dict(v) for v in history]
        if (entity_tag := _dict.get('entity_tag')) is not None:
            args['entity_tag'] = entity_tag
        else:
            raise ValueError('Required property \'entity_tag\' not present in AccountSettingsTemplateResponse JSON')
        if (crn := _dict.get('crn')) is not None:
            args['crn'] = crn
        else:
            raise ValueError('Required property \'crn\' not present in AccountSettingsTemplateResponse JSON')
        if (created_at := _dict.get('created_at')) is not None:
            args['created_at'] = created_at
        if (created_by_id := _dict.get('created_by_id')) is not None:
            args['created_by_id'] = created_by_id
        if (last_modified_at := _dict.get('last_modified_at')) is not None:
            args['last_modified_at'] = last_modified_at
        if (last_modified_by_id := _dict.get('last_modified_by_id')) is not None:
            args['last_modified_by_id'] = last_modified_by_id
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccountSettingsTemplateResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'version') and self.version is not None:
            _dict['version'] = self.version
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'committed') and self.committed is not None:
            _dict['committed'] = self.committed
        if hasattr(self, 'account_settings') and self.account_settings is not None:
            if isinstance(self.account_settings, dict):
                _dict['account_settings'] = self.account_settings
            else:
                _dict['account_settings'] = self.account_settings.to_dict()
        if hasattr(self, 'history') and self.history is not None:
            history_list = []
            for v in self.history:
                if isinstance(v, dict):
                    history_list.append(v)
                else:
                    history_list.append(v.to_dict())
            _dict['history'] = history_list
        if hasattr(self, 'entity_tag') and self.entity_tag is not None:
            _dict['entity_tag'] = self.entity_tag
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = self.created_at
        if hasattr(self, 'created_by_id') and self.created_by_id is not None:
            _dict['created_by_id'] = self.created_by_id
        if hasattr(self, 'last_modified_at') and self.last_modified_at is not None:
            _dict['last_modified_at'] = self.last_modified_at
        if hasattr(self, 'last_modified_by_id') and self.last_modified_by_id is not None:
            _dict['last_modified_by_id'] = self.last_modified_by_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AccountSettingsTemplateResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccountSettingsTemplateResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccountSettingsTemplateResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AccountSettingsUserMFA:
    """
    AccountSettingsUserMFA.

    :param str iam_id: The iam_id of the user.
    :param str mfa: Defines the MFA requirement for the user. Valid values:
            * NONE - No MFA trait set
            * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
            * TOTP - For all non-federated IBMId users
            * TOTP4ALL - For all users
            * LEVEL1 - Email-based MFA for all users
            * LEVEL2 - TOTP-based MFA for all users
            * LEVEL3 - U2F MFA for all users.
    """

    def __init__(
        self,
        iam_id: str,
        mfa: str,
    ) -> None:
        """
        Initialize a AccountSettingsUserMFA object.

        :param str iam_id: The iam_id of the user.
        :param str mfa: Defines the MFA requirement for the user. Valid values:
                 * NONE - No MFA trait set
                 * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
                 * TOTP - For all non-federated IBMId users
                 * TOTP4ALL - For all users
                 * LEVEL1 - Email-based MFA for all users
                 * LEVEL2 - TOTP-based MFA for all users
                 * LEVEL3 - U2F MFA for all users.
        """
        self.iam_id = iam_id
        self.mfa = mfa

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccountSettingsUserMFA':
        """Initialize a AccountSettingsUserMFA object from a json dictionary."""
        args = {}
        if (iam_id := _dict.get('iam_id')) is not None:
            args['iam_id'] = iam_id
        else:
            raise ValueError('Required property \'iam_id\' not present in AccountSettingsUserMFA JSON')
        if (mfa := _dict.get('mfa')) is not None:
            args['mfa'] = mfa
        else:
            raise ValueError('Required property \'mfa\' not present in AccountSettingsUserMFA JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccountSettingsUserMFA object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'iam_id') and self.iam_id is not None:
            _dict['iam_id'] = self.iam_id
        if hasattr(self, 'mfa') and self.mfa is not None:
            _dict['mfa'] = self.mfa
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AccountSettingsUserMFA object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccountSettingsUserMFA') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccountSettingsUserMFA') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class MfaEnum(str, Enum):
        """
        Defines the MFA requirement for the user. Valid values:
          * NONE - No MFA trait set
          * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
          * TOTP - For all non-federated IBMId users
          * TOTP4ALL - For all users
          * LEVEL1 - Email-based MFA for all users
          * LEVEL2 - TOTP-based MFA for all users
          * LEVEL3 - U2F MFA for all users.
        """

        NONE = 'NONE'
        NONE_NO_ROPC = 'NONE_NO_ROPC'
        TOTP = 'TOTP'
        TOTP4ALL = 'TOTP4ALL'
        LEVEL1 = 'LEVEL1'
        LEVEL2 = 'LEVEL2'
        LEVEL3 = 'LEVEL3'


class Activity:
    """
    Activity.

    :param str last_authn: (optional) Time when the entity was last authenticated.
    :param int authn_count: Authentication count, number of times the entity was
          authenticated.
    """

    def __init__(
        self,
        authn_count: int,
        *,
        last_authn: Optional[str] = None,
    ) -> None:
        """
        Initialize a Activity object.

        :param int authn_count: Authentication count, number of times the entity
               was authenticated.
        :param str last_authn: (optional) Time when the entity was last
               authenticated.
        """
        self.last_authn = last_authn
        self.authn_count = authn_count

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Activity':
        """Initialize a Activity object from a json dictionary."""
        args = {}
        if (last_authn := _dict.get('last_authn')) is not None:
            args['last_authn'] = last_authn
        if (authn_count := _dict.get('authn_count')) is not None:
            args['authn_count'] = authn_count
        else:
            raise ValueError('Required property \'authn_count\' not present in Activity JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Activity object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'last_authn') and self.last_authn is not None:
            _dict['last_authn'] = self.last_authn
        if hasattr(self, 'authn_count') and self.authn_count is not None:
            _dict['authn_count'] = self.authn_count
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Activity object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Activity') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Activity') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ApiKey:
    """
    Response body format for API key V1 REST requests.

    :param ResponseContext context: (optional) Context with key properties for
          problem determination.
    :param str id: Unique identifier of this API Key.
    :param str entity_tag: (optional) Version of the API Key details object. You
          need to specify this value when updating the API key to avoid stale updates.
    :param str crn: Cloud Resource Name of the item. Example Cloud Resource Name:
          'crn:v1:bluemix:public:iam-identity:us-south:a/myaccount::apikey:1234-9012-5678'.
    :param bool locked: The API key cannot be changed if set to true.
    :param bool disabled: (optional) Defines if API key is disabled, API key cannot
          be used if 'disabled' is set to true.
    :param datetime created_at: (optional) If set contains a date time string of the
          creation date in ISO format.
    :param str created_by: IAM ID of the user or service which created the API key.
    :param datetime modified_at: (optional) If set contains a date time string of
          the last modification date in ISO format.
    :param str name: Name of the API key. The name is not checked for uniqueness.
          Therefore multiple names with the same value can exist. Access is done via the
          UUID of the API key.
    :param bool support_sessions: (optional) Defines if the API key supports
          sessions. Sessions are only supported for user apikeys.
    :param str action_when_leaked: (optional) Defines the action to take when API
          key is leaked, valid values are 'none', 'disable' and 'delete'.
    :param str description: (optional) The optional description of the API key. The
          'description' property is only available if a description was provided during a
          create of an API key.
    :param str iam_id: The iam_id that this API key authenticates.
    :param str account_id: ID of the account that this API key authenticates for.
    :param str apikey: The API key value. This property only contains the API key
          value for the following cases: create an API key, update a service ID API key
          that stores the API key value as retrievable, or get a service ID API key that
          stores the API key value as retrievable. All other operations don't return the
          API key value, for example all user API key related operations, except for
          create, don't contain the API key value.
    :param List[EnityHistoryRecord] history: (optional) History of the API key.
    :param Activity activity: (optional)
    """

    def __init__(
        self,
        id: str,
        crn: str,
        locked: bool,
        created_by: str,
        name: str,
        iam_id: str,
        account_id: str,
        apikey: str,
        *,
        context: Optional['ResponseContext'] = None,
        entity_tag: Optional[str] = None,
        disabled: Optional[bool] = None,
        created_at: Optional[datetime] = None,
        modified_at: Optional[datetime] = None,
        support_sessions: Optional[bool] = None,
        action_when_leaked: Optional[str] = None,
        description: Optional[str] = None,
        history: Optional[List['EnityHistoryRecord']] = None,
        activity: Optional['Activity'] = None,
    ) -> None:
        """
        Initialize a ApiKey object.

        :param str id: Unique identifier of this API Key.
        :param str crn: Cloud Resource Name of the item. Example Cloud Resource
               Name:
               'crn:v1:bluemix:public:iam-identity:us-south:a/myaccount::apikey:1234-9012-5678'.
        :param bool locked: The API key cannot be changed if set to true.
        :param str created_by: IAM ID of the user or service which created the API
               key.
        :param str name: Name of the API key. The name is not checked for
               uniqueness. Therefore multiple names with the same value can exist. Access
               is done via the UUID of the API key.
        :param str iam_id: The iam_id that this API key authenticates.
        :param str account_id: ID of the account that this API key authenticates
               for.
        :param str apikey: The API key value. This property only contains the API
               key value for the following cases: create an API key, update a service ID
               API key that stores the API key value as retrievable, or get a service ID
               API key that stores the API key value as retrievable. All other operations
               don't return the API key value, for example all user API key related
               operations, except for create, don't contain the API key value.
        :param ResponseContext context: (optional) Context with key properties for
               problem determination.
        :param str entity_tag: (optional) Version of the API Key details object.
               You need to specify this value when updating the API key to avoid stale
               updates.
        :param bool disabled: (optional) Defines if API key is disabled, API key
               cannot be used if 'disabled' is set to true.
        :param datetime created_at: (optional) If set contains a date time string
               of the creation date in ISO format.
        :param datetime modified_at: (optional) If set contains a date time string
               of the last modification date in ISO format.
        :param bool support_sessions: (optional) Defines if the API key supports
               sessions. Sessions are only supported for user apikeys.
        :param str action_when_leaked: (optional) Defines the action to take when
               API key is leaked, valid values are 'none', 'disable' and 'delete'.
        :param str description: (optional) The optional description of the API key.
               The 'description' property is only available if a description was provided
               during a create of an API key.
        :param List[EnityHistoryRecord] history: (optional) History of the API key.
        :param Activity activity: (optional)
        """
        self.context = context
        self.id = id
        self.entity_tag = entity_tag
        self.crn = crn
        self.locked = locked
        self.disabled = disabled
        self.created_at = created_at
        self.created_by = created_by
        self.modified_at = modified_at
        self.name = name
        self.support_sessions = support_sessions
        self.action_when_leaked = action_when_leaked
        self.description = description
        self.iam_id = iam_id
        self.account_id = account_id
        self.apikey = apikey
        self.history = history
        self.activity = activity

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ApiKey':
        """Initialize a ApiKey object from a json dictionary."""
        args = {}
        if (context := _dict.get('context')) is not None:
            args['context'] = ResponseContext.from_dict(context)
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        else:
            raise ValueError('Required property \'id\' not present in ApiKey JSON')
        if (entity_tag := _dict.get('entity_tag')) is not None:
            args['entity_tag'] = entity_tag
        if (crn := _dict.get('crn')) is not None:
            args['crn'] = crn
        else:
            raise ValueError('Required property \'crn\' not present in ApiKey JSON')
        if (locked := _dict.get('locked')) is not None:
            args['locked'] = locked
        else:
            raise ValueError('Required property \'locked\' not present in ApiKey JSON')
        if (disabled := _dict.get('disabled')) is not None:
            args['disabled'] = disabled
        if (created_at := _dict.get('created_at')) is not None:
            args['created_at'] = string_to_datetime(created_at)
        if (created_by := _dict.get('created_by')) is not None:
            args['created_by'] = created_by
        else:
            raise ValueError('Required property \'created_by\' not present in ApiKey JSON')
        if (modified_at := _dict.get('modified_at')) is not None:
            args['modified_at'] = string_to_datetime(modified_at)
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        else:
            raise ValueError('Required property \'name\' not present in ApiKey JSON')
        if (support_sessions := _dict.get('support_sessions')) is not None:
            args['support_sessions'] = support_sessions
        if (action_when_leaked := _dict.get('action_when_leaked')) is not None:
            args['action_when_leaked'] = action_when_leaked
        if (description := _dict.get('description')) is not None:
            args['description'] = description
        if (iam_id := _dict.get('iam_id')) is not None:
            args['iam_id'] = iam_id
        else:
            raise ValueError('Required property \'iam_id\' not present in ApiKey JSON')
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        else:
            raise ValueError('Required property \'account_id\' not present in ApiKey JSON')
        if (apikey := _dict.get('apikey')) is not None:
            args['apikey'] = apikey
        else:
            raise ValueError('Required property \'apikey\' not present in ApiKey JSON')
        if (history := _dict.get('history')) is not None:
            args['history'] = [EnityHistoryRecord.from_dict(v) for v in history]
        if (activity := _dict.get('activity')) is not None:
            args['activity'] = Activity.from_dict(activity)
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ApiKey object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'context') and self.context is not None:
            if isinstance(self.context, dict):
                _dict['context'] = self.context
            else:
                _dict['context'] = self.context.to_dict()
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'entity_tag') and self.entity_tag is not None:
            _dict['entity_tag'] = self.entity_tag
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'locked') and self.locked is not None:
            _dict['locked'] = self.locked
        if hasattr(self, 'disabled') and self.disabled is not None:
            _dict['disabled'] = self.disabled
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'created_by') and self.created_by is not None:
            _dict['created_by'] = self.created_by
        if hasattr(self, 'modified_at') and self.modified_at is not None:
            _dict['modified_at'] = datetime_to_string(self.modified_at)
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'support_sessions') and self.support_sessions is not None:
            _dict['support_sessions'] = self.support_sessions
        if hasattr(self, 'action_when_leaked') and self.action_when_leaked is not None:
            _dict['action_when_leaked'] = self.action_when_leaked
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'iam_id') and self.iam_id is not None:
            _dict['iam_id'] = self.iam_id
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'apikey') and self.apikey is not None:
            _dict['apikey'] = self.apikey
        if hasattr(self, 'history') and self.history is not None:
            history_list = []
            for v in self.history:
                if isinstance(v, dict):
                    history_list.append(v)
                else:
                    history_list.append(v.to_dict())
            _dict['history'] = history_list
        if hasattr(self, 'activity') and self.activity is not None:
            if isinstance(self.activity, dict):
                _dict['activity'] = self.activity
            else:
                _dict['activity'] = self.activity.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ApiKey object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ApiKey') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ApiKey') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ApiKeyInsideCreateServiceIdRequest:
    """
    Parameters for the API key in the Create service Id V1 REST request.

    :param str name: Name of the API key. The name is not checked for uniqueness.
          Therefore multiple names with the same value can exist. Access is done via the
          UUID of the API key.
    :param str description: (optional) The optional description of the API key. The
          'description' property is only available if a description was provided during a
          create of an API key.
    :param str apikey: (optional) You can optionally passthrough the API key value
          for this API key. If passed, a minimum length validation of 32 characters for
          that apiKey value is done, i.e. the value can contain any characters and can
          even be non-URL safe, but the minimum length requirement must be met. If
          omitted, the API key management will create an URL safe opaque API key value.
          The value of the API key is checked for uniqueness. Ensure enough variations
          when passing in this value.
    :param bool store_value: (optional) Send true or false to set whether the API
          key value is retrievable in the future by using the Get details of an API key
          request. If you create an API key for a user, you must specify `false` or omit
          the value. We don't allow storing of API keys for users.
    """

    def __init__(
        self,
        name: str,
        *,
        description: Optional[str] = None,
        apikey: Optional[str] = None,
        store_value: Optional[bool] = None,
    ) -> None:
        """
        Initialize a ApiKeyInsideCreateServiceIdRequest object.

        :param str name: Name of the API key. The name is not checked for
               uniqueness. Therefore multiple names with the same value can exist. Access
               is done via the UUID of the API key.
        :param str description: (optional) The optional description of the API key.
               The 'description' property is only available if a description was provided
               during a create of an API key.
        :param str apikey: (optional) You can optionally passthrough the API key
               value for this API key. If passed, a minimum length validation of 32
               characters for that apiKey value is done, i.e. the value can contain any
               characters and can even be non-URL safe, but the minimum length requirement
               must be met. If omitted, the API key management will create an URL safe
               opaque API key value. The value of the API key is checked for uniqueness.
               Ensure enough variations when passing in this value.
        :param bool store_value: (optional) Send true or false to set whether the
               API key value is retrievable in the future by using the Get details of an
               API key request. If you create an API key for a user, you must specify
               `false` or omit the value. We don't allow storing of API keys for users.
        """
        self.name = name
        self.description = description
        self.apikey = apikey
        self.store_value = store_value

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ApiKeyInsideCreateServiceIdRequest':
        """Initialize a ApiKeyInsideCreateServiceIdRequest object from a json dictionary."""
        args = {}
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        else:
            raise ValueError('Required property \'name\' not present in ApiKeyInsideCreateServiceIdRequest JSON')
        if (description := _dict.get('description')) is not None:
            args['description'] = description
        if (apikey := _dict.get('apikey')) is not None:
            args['apikey'] = apikey
        if (store_value := _dict.get('store_value')) is not None:
            args['store_value'] = store_value
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ApiKeyInsideCreateServiceIdRequest object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'apikey') and self.apikey is not None:
            _dict['apikey'] = self.apikey
        if hasattr(self, 'store_value') and self.store_value is not None:
            _dict['store_value'] = self.store_value
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ApiKeyInsideCreateServiceIdRequest object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ApiKeyInsideCreateServiceIdRequest') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ApiKeyInsideCreateServiceIdRequest') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ApiKeyList:
    """
    Response body format for the List API keys V1 REST request.

    :param ResponseContext context: (optional) Context with key properties for
          problem determination.
    :param int offset: (optional) The offset of the current page.
    :param int limit: (optional) Optional size of a single page. Default is 20 items
          per page. Valid range is 1 to 100.
    :param str first: (optional) Link to the first page.
    :param str previous: (optional) Link to the previous available page. If
          'previous' property is not part of the response no previous page is available.
    :param str next: (optional) Link to the next available page. If 'next' property
          is not part of the response no next page is available.
    :param List[ApiKey] apikeys: List of API keys based on the query paramters and
          the page size. The apikeys array is always part of the response but might be
          empty depending on the query parameters values provided.
    """

    def __init__(
        self,
        apikeys: List['ApiKey'],
        *,
        context: Optional['ResponseContext'] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        first: Optional[str] = None,
        previous: Optional[str] = None,
        next: Optional[str] = None,
    ) -> None:
        """
        Initialize a ApiKeyList object.

        :param List[ApiKey] apikeys: List of API keys based on the query paramters
               and the page size. The apikeys array is always part of the response but
               might be empty depending on the query parameters values provided.
        :param ResponseContext context: (optional) Context with key properties for
               problem determination.
        :param int offset: (optional) The offset of the current page.
        :param int limit: (optional) Optional size of a single page. Default is 20
               items per page. Valid range is 1 to 100.
        :param str first: (optional) Link to the first page.
        :param str previous: (optional) Link to the previous available page. If
               'previous' property is not part of the response no previous page is
               available.
        :param str next: (optional) Link to the next available page. If 'next'
               property is not part of the response no next page is available.
        """
        self.context = context
        self.offset = offset
        self.limit = limit
        self.first = first
        self.previous = previous
        self.next = next
        self.apikeys = apikeys

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ApiKeyList':
        """Initialize a ApiKeyList object from a json dictionary."""
        args = {}
        if (context := _dict.get('context')) is not None:
            args['context'] = ResponseContext.from_dict(context)
        if (offset := _dict.get('offset')) is not None:
            args['offset'] = offset
        if (limit := _dict.get('limit')) is not None:
            args['limit'] = limit
        if (first := _dict.get('first')) is not None:
            args['first'] = first
        if (previous := _dict.get('previous')) is not None:
            args['previous'] = previous
        if (next := _dict.get('next')) is not None:
            args['next'] = next
        if (apikeys := _dict.get('apikeys')) is not None:
            args['apikeys'] = [ApiKey.from_dict(v) for v in apikeys]
        else:
            raise ValueError('Required property \'apikeys\' not present in ApiKeyList JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ApiKeyList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'context') and self.context is not None:
            if isinstance(self.context, dict):
                _dict['context'] = self.context
            else:
                _dict['context'] = self.context.to_dict()
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'first') and self.first is not None:
            _dict['first'] = self.first
        if hasattr(self, 'previous') and self.previous is not None:
            _dict['previous'] = self.previous
        if hasattr(self, 'next') and self.next is not None:
            _dict['next'] = self.next
        if hasattr(self, 'apikeys') and self.apikeys is not None:
            apikeys_list = []
            for v in self.apikeys:
                if isinstance(v, dict):
                    apikeys_list.append(v)
                else:
                    apikeys_list.append(v.to_dict())
            _dict['apikeys'] = apikeys_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ApiKeyList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ApiKeyList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ApiKeyList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ApikeyActivity:
    """
    Apikeys activity details.

    :param str id: Unique id of the apikey.
    :param str name: (optional) Name provided during creation of the apikey.
    :param str type: Type of the apikey. Supported values are `serviceid` and
          `user`.
    :param ApikeyActivityServiceid serviceid: (optional) serviceid details will be
          present if type is `serviceid`.
    :param ApikeyActivityUser user: (optional) user details will be present if type
          is `user`.
    :param str last_authn: (optional) Time when the apikey was last authenticated.
    """

    def __init__(
        self,
        id: str,
        type: str,
        *,
        name: Optional[str] = None,
        serviceid: Optional['ApikeyActivityServiceid'] = None,
        user: Optional['ApikeyActivityUser'] = None,
        last_authn: Optional[str] = None,
    ) -> None:
        """
        Initialize a ApikeyActivity object.

        :param str id: Unique id of the apikey.
        :param str type: Type of the apikey. Supported values are `serviceid` and
               `user`.
        :param str name: (optional) Name provided during creation of the apikey.
        :param ApikeyActivityServiceid serviceid: (optional) serviceid details will
               be present if type is `serviceid`.
        :param ApikeyActivityUser user: (optional) user details will be present if
               type is `user`.
        :param str last_authn: (optional) Time when the apikey was last
               authenticated.
        """
        self.id = id
        self.name = name
        self.type = type
        self.serviceid = serviceid
        self.user = user
        self.last_authn = last_authn

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ApikeyActivity':
        """Initialize a ApikeyActivity object from a json dictionary."""
        args = {}
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        else:
            raise ValueError('Required property \'id\' not present in ApikeyActivity JSON')
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        if (type := _dict.get('type')) is not None:
            args['type'] = type
        else:
            raise ValueError('Required property \'type\' not present in ApikeyActivity JSON')
        if (serviceid := _dict.get('serviceid')) is not None:
            args['serviceid'] = ApikeyActivityServiceid.from_dict(serviceid)
        if (user := _dict.get('user')) is not None:
            args['user'] = ApikeyActivityUser.from_dict(user)
        if (last_authn := _dict.get('last_authn')) is not None:
            args['last_authn'] = last_authn
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ApikeyActivity object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'serviceid') and self.serviceid is not None:
            if isinstance(self.serviceid, dict):
                _dict['serviceid'] = self.serviceid
            else:
                _dict['serviceid'] = self.serviceid.to_dict()
        if hasattr(self, 'user') and self.user is not None:
            if isinstance(self.user, dict):
                _dict['user'] = self.user
            else:
                _dict['user'] = self.user.to_dict()
        if hasattr(self, 'last_authn') and self.last_authn is not None:
            _dict['last_authn'] = self.last_authn
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ApikeyActivity object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ApikeyActivity') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ApikeyActivity') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ApikeyActivityServiceid:
    """
    serviceid details will be present if type is `serviceid`.

    :param str id: (optional) Unique identifier of this Service Id.
    :param str name: (optional) Name provided during creation of the serviceid.
    """

    def __init__(
        self,
        *,
        id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize a ApikeyActivityServiceid object.

        :param str id: (optional) Unique identifier of this Service Id.
        :param str name: (optional) Name provided during creation of the serviceid.
        """
        self.id = id
        self.name = name

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ApikeyActivityServiceid':
        """Initialize a ApikeyActivityServiceid object from a json dictionary."""
        args = {}
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ApikeyActivityServiceid object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ApikeyActivityServiceid object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ApikeyActivityServiceid') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ApikeyActivityServiceid') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ApikeyActivityUser:
    """
    user details will be present if type is `user`.

    :param str iam_id: (optional) IAMid of the user.
    :param str name: (optional) Name of the user.
    :param str username: (optional) Username of the user.
    :param str email: (optional) Email of the user.
    """

    def __init__(
        self,
        *,
        iam_id: Optional[str] = None,
        name: Optional[str] = None,
        username: Optional[str] = None,
        email: Optional[str] = None,
    ) -> None:
        """
        Initialize a ApikeyActivityUser object.

        :param str iam_id: (optional) IAMid of the user.
        :param str name: (optional) Name of the user.
        :param str username: (optional) Username of the user.
        :param str email: (optional) Email of the user.
        """
        self.iam_id = iam_id
        self.name = name
        self.username = username
        self.email = email

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ApikeyActivityUser':
        """Initialize a ApikeyActivityUser object from a json dictionary."""
        args = {}
        if (iam_id := _dict.get('iam_id')) is not None:
            args['iam_id'] = iam_id
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        if (username := _dict.get('username')) is not None:
            args['username'] = username
        if (email := _dict.get('email')) is not None:
            args['email'] = email
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ApikeyActivityUser object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'iam_id') and self.iam_id is not None:
            _dict['iam_id'] = self.iam_id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'username') and self.username is not None:
            _dict['username'] = self.username
        if hasattr(self, 'email') and self.email is not None:
            _dict['email'] = self.email
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ApikeyActivityUser object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ApikeyActivityUser') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ApikeyActivityUser') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class CreateProfileLinkRequestLink:
    """
    Link details.

    :param str crn: The CRN of the compute resource.
    :param str namespace: The compute resource namespace, only required if cr_type
          is IKS_SA or ROKS_SA.
    :param str name: (optional) Name of the compute resource, only required if
          cr_type is IKS_SA or ROKS_SA.
    """

    def __init__(
        self,
        crn: str,
        namespace: str,
        *,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize a CreateProfileLinkRequestLink object.

        :param str crn: The CRN of the compute resource.
        :param str namespace: The compute resource namespace, only required if
               cr_type is IKS_SA or ROKS_SA.
        :param str name: (optional) Name of the compute resource, only required if
               cr_type is IKS_SA or ROKS_SA.
        """
        self.crn = crn
        self.namespace = namespace
        self.name = name

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CreateProfileLinkRequestLink':
        """Initialize a CreateProfileLinkRequestLink object from a json dictionary."""
        args = {}
        if (crn := _dict.get('crn')) is not None:
            args['crn'] = crn
        else:
            raise ValueError('Required property \'crn\' not present in CreateProfileLinkRequestLink JSON')
        if (namespace := _dict.get('namespace')) is not None:
            args['namespace'] = namespace
        else:
            raise ValueError('Required property \'namespace\' not present in CreateProfileLinkRequestLink JSON')
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CreateProfileLinkRequestLink object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'namespace') and self.namespace is not None:
            _dict['namespace'] = self.namespace
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this CreateProfileLinkRequestLink object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CreateProfileLinkRequestLink') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CreateProfileLinkRequestLink') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class EnityHistoryRecord:
    """
    Response body format for an entity history record.

    :param str timestamp: Timestamp when the action was triggered.
    :param str iam_id: IAM ID of the identity which triggered the action.
    :param str iam_id_account: Account of the identity which triggered the action.
    :param str action: Action of the history entry.
    :param List[str] params: Params of the history entry.
    :param str message: Message which summarizes the executed action.
    """

    def __init__(
        self,
        timestamp: str,
        iam_id: str,
        iam_id_account: str,
        action: str,
        params: List[str],
        message: str,
    ) -> None:
        """
        Initialize a EnityHistoryRecord object.

        :param str timestamp: Timestamp when the action was triggered.
        :param str iam_id: IAM ID of the identity which triggered the action.
        :param str iam_id_account: Account of the identity which triggered the
               action.
        :param str action: Action of the history entry.
        :param List[str] params: Params of the history entry.
        :param str message: Message which summarizes the executed action.
        """
        self.timestamp = timestamp
        self.iam_id = iam_id
        self.iam_id_account = iam_id_account
        self.action = action
        self.params = params
        self.message = message

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'EnityHistoryRecord':
        """Initialize a EnityHistoryRecord object from a json dictionary."""
        args = {}
        if (timestamp := _dict.get('timestamp')) is not None:
            args['timestamp'] = timestamp
        else:
            raise ValueError('Required property \'timestamp\' not present in EnityHistoryRecord JSON')
        if (iam_id := _dict.get('iam_id')) is not None:
            args['iam_id'] = iam_id
        else:
            raise ValueError('Required property \'iam_id\' not present in EnityHistoryRecord JSON')
        if (iam_id_account := _dict.get('iam_id_account')) is not None:
            args['iam_id_account'] = iam_id_account
        else:
            raise ValueError('Required property \'iam_id_account\' not present in EnityHistoryRecord JSON')
        if (action := _dict.get('action')) is not None:
            args['action'] = action
        else:
            raise ValueError('Required property \'action\' not present in EnityHistoryRecord JSON')
        if (params := _dict.get('params')) is not None:
            args['params'] = params
        else:
            raise ValueError('Required property \'params\' not present in EnityHistoryRecord JSON')
        if (message := _dict.get('message')) is not None:
            args['message'] = message
        else:
            raise ValueError('Required property \'message\' not present in EnityHistoryRecord JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a EnityHistoryRecord object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'timestamp') and self.timestamp is not None:
            _dict['timestamp'] = self.timestamp
        if hasattr(self, 'iam_id') and self.iam_id is not None:
            _dict['iam_id'] = self.iam_id
        if hasattr(self, 'iam_id_account') and self.iam_id_account is not None:
            _dict['iam_id_account'] = self.iam_id_account
        if hasattr(self, 'action') and self.action is not None:
            _dict['action'] = self.action
        if hasattr(self, 'params') and self.params is not None:
            _dict['params'] = self.params
        if hasattr(self, 'message') and self.message is not None:
            _dict['message'] = self.message
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this EnityHistoryRecord object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'EnityHistoryRecord') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'EnityHistoryRecord') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class EntityActivity:
    """
    EntityActivity.

    :param str id: Unique id of the entity.
    :param str name: (optional) Name provided during creation of the entity.
    :param str last_authn: (optional) Time when the entity was last authenticated.
    """

    def __init__(
        self,
        id: str,
        *,
        name: Optional[str] = None,
        last_authn: Optional[str] = None,
    ) -> None:
        """
        Initialize a EntityActivity object.

        :param str id: Unique id of the entity.
        :param str name: (optional) Name provided during creation of the entity.
        :param str last_authn: (optional) Time when the entity was last
               authenticated.
        """
        self.id = id
        self.name = name
        self.last_authn = last_authn

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'EntityActivity':
        """Initialize a EntityActivity object from a json dictionary."""
        args = {}
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        else:
            raise ValueError('Required property \'id\' not present in EntityActivity JSON')
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        if (last_authn := _dict.get('last_authn')) is not None:
            args['last_authn'] = last_authn
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a EntityActivity object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'last_authn') and self.last_authn is not None:
            _dict['last_authn'] = self.last_authn
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this EntityActivity object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'EntityActivity') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'EntityActivity') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Error:
    """
    Error information.

    :param str code: Error code of the REST Exception.
    :param str message_code: Error message code of the REST Exception.
    :param str message: Error message of the REST Exception. Error messages are
          derived base on the input locale of the REST request and the available Message
          catalogs. Dynamic fallback to 'us-english' is happening if no message catalog is
          available for the provided input locale.
    :param str details: (optional) Error details of the REST Exception.
    """

    def __init__(
        self,
        code: str,
        message_code: str,
        message: str,
        *,
        details: Optional[str] = None,
    ) -> None:
        """
        Initialize a Error object.

        :param str code: Error code of the REST Exception.
        :param str message_code: Error message code of the REST Exception.
        :param str message: Error message of the REST Exception. Error messages are
               derived base on the input locale of the REST request and the available
               Message catalogs. Dynamic fallback to 'us-english' is happening if no
               message catalog is available for the provided input locale.
        :param str details: (optional) Error details of the REST Exception.
        """
        self.code = code
        self.message_code = message_code
        self.message = message
        self.details = details

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Error':
        """Initialize a Error object from a json dictionary."""
        args = {}
        if (code := _dict.get('code')) is not None:
            args['code'] = code
        else:
            raise ValueError('Required property \'code\' not present in Error JSON')
        if (message_code := _dict.get('message_code')) is not None:
            args['message_code'] = message_code
        else:
            raise ValueError('Required property \'message_code\' not present in Error JSON')
        if (message := _dict.get('message')) is not None:
            args['message'] = message
        else:
            raise ValueError('Required property \'message\' not present in Error JSON')
        if (details := _dict.get('details')) is not None:
            args['details'] = details
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Error object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'code') and self.code is not None:
            _dict['code'] = self.code
        if hasattr(self, 'message_code') and self.message_code is not None:
            _dict['message_code'] = self.message_code
        if hasattr(self, 'message') and self.message is not None:
            _dict['message'] = self.message
        if hasattr(self, 'details') and self.details is not None:
            _dict['details'] = self.details
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Error object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Error') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Error') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ExceptionResponse:
    """
    Response body parameters in case of error situations.

    :param ResponseContext context: (optional) Context with key properties for
          problem determination.
    :param str status_code: Error message code of the REST Exception.
    :param List[Error] errors: List of errors that occured.
    :param str trace: (optional) Unique ID of the requst.
    """

    def __init__(
        self,
        status_code: str,
        errors: List['Error'],
        *,
        context: Optional['ResponseContext'] = None,
        trace: Optional[str] = None,
    ) -> None:
        """
        Initialize a ExceptionResponse object.

        :param str status_code: Error message code of the REST Exception.
        :param List[Error] errors: List of errors that occured.
        :param ResponseContext context: (optional) Context with key properties for
               problem determination.
        :param str trace: (optional) Unique ID of the requst.
        """
        self.context = context
        self.status_code = status_code
        self.errors = errors
        self.trace = trace

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ExceptionResponse':
        """Initialize a ExceptionResponse object from a json dictionary."""
        args = {}
        if (context := _dict.get('context')) is not None:
            args['context'] = ResponseContext.from_dict(context)
        if (status_code := _dict.get('status_code')) is not None:
            args['status_code'] = status_code
        else:
            raise ValueError('Required property \'status_code\' not present in ExceptionResponse JSON')
        if (errors := _dict.get('errors')) is not None:
            args['errors'] = [Error.from_dict(v) for v in errors]
        else:
            raise ValueError('Required property \'errors\' not present in ExceptionResponse JSON')
        if (trace := _dict.get('trace')) is not None:
            args['trace'] = trace
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ExceptionResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'context') and self.context is not None:
            if isinstance(self.context, dict):
                _dict['context'] = self.context
            else:
                _dict['context'] = self.context.to_dict()
        if hasattr(self, 'status_code') and self.status_code is not None:
            _dict['status_code'] = self.status_code
        if hasattr(self, 'errors') and self.errors is not None:
            errors_list = []
            for v in self.errors:
                if isinstance(v, dict):
                    errors_list.append(v)
                else:
                    errors_list.append(v.to_dict())
            _dict['errors'] = errors_list
        if hasattr(self, 'trace') and self.trace is not None:
            _dict['trace'] = self.trace
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ExceptionResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ExceptionResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ExceptionResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class IdBasedMfaEnrollment:
    """
    IdBasedMfaEnrollment.

    :param str trait_account_default: Defines the MFA trait for the account. Valid
          values:
            * NONE - No MFA trait set
            * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
            * TOTP - For all non-federated IBMId users
            * TOTP4ALL - For all users
            * LEVEL1 - Email-based MFA for all users
            * LEVEL2 - TOTP-based MFA for all users
            * LEVEL3 - U2F MFA for all users.
    :param str trait_user_specific: (optional) Defines the MFA trait for the
          account. Valid values:
            * NONE - No MFA trait set
            * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
            * TOTP - For all non-federated IBMId users
            * TOTP4ALL - For all users
            * LEVEL1 - Email-based MFA for all users
            * LEVEL2 - TOTP-based MFA for all users
            * LEVEL3 - U2F MFA for all users.
    :param str trait_effective: Defines the MFA trait for the account. Valid values:
            * NONE - No MFA trait set
            * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
            * TOTP - For all non-federated IBMId users
            * TOTP4ALL - For all users
            * LEVEL1 - Email-based MFA for all users
            * LEVEL2 - TOTP-based MFA for all users
            * LEVEL3 - U2F MFA for all users.
    :param bool complies: The enrollment complies to the effective requirement.
    :param str comply_state: (optional) Defines comply state for the account. Valid
          values:
            * NO - User does not comply in the given account.
            * ACCOUNT- User complies in the given account, but does not comply in at least
          one of the other account memberships.
            * CROSS_ACCOUNT - User complies in the given account and across all other
          account memberships.
    """

    def __init__(
        self,
        trait_account_default: str,
        trait_effective: str,
        complies: bool,
        *,
        trait_user_specific: Optional[str] = None,
        comply_state: Optional[str] = None,
    ) -> None:
        """
        Initialize a IdBasedMfaEnrollment object.

        :param str trait_account_default: Defines the MFA trait for the account.
               Valid values:
                 * NONE - No MFA trait set
                 * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
                 * TOTP - For all non-federated IBMId users
                 * TOTP4ALL - For all users
                 * LEVEL1 - Email-based MFA for all users
                 * LEVEL2 - TOTP-based MFA for all users
                 * LEVEL3 - U2F MFA for all users.
        :param str trait_effective: Defines the MFA trait for the account. Valid
               values:
                 * NONE - No MFA trait set
                 * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
                 * TOTP - For all non-federated IBMId users
                 * TOTP4ALL - For all users
                 * LEVEL1 - Email-based MFA for all users
                 * LEVEL2 - TOTP-based MFA for all users
                 * LEVEL3 - U2F MFA for all users.
        :param bool complies: The enrollment complies to the effective requirement.
        :param str trait_user_specific: (optional) Defines the MFA trait for the
               account. Valid values:
                 * NONE - No MFA trait set
                 * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
                 * TOTP - For all non-federated IBMId users
                 * TOTP4ALL - For all users
                 * LEVEL1 - Email-based MFA for all users
                 * LEVEL2 - TOTP-based MFA for all users
                 * LEVEL3 - U2F MFA for all users.
        :param str comply_state: (optional) Defines comply state for the account.
               Valid values:
                 * NO - User does not comply in the given account.
                 * ACCOUNT- User complies in the given account, but does not comply in at
               least one of the other account memberships.
                 * CROSS_ACCOUNT - User complies in the given account and across all other
               account memberships.
        """
        self.trait_account_default = trait_account_default
        self.trait_user_specific = trait_user_specific
        self.trait_effective = trait_effective
        self.complies = complies
        self.comply_state = comply_state

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'IdBasedMfaEnrollment':
        """Initialize a IdBasedMfaEnrollment object from a json dictionary."""
        args = {}
        if (trait_account_default := _dict.get('trait_account_default')) is not None:
            args['trait_account_default'] = trait_account_default
        else:
            raise ValueError('Required property \'trait_account_default\' not present in IdBasedMfaEnrollment JSON')
        if (trait_user_specific := _dict.get('trait_user_specific')) is not None:
            args['trait_user_specific'] = trait_user_specific
        if (trait_effective := _dict.get('trait_effective')) is not None:
            args['trait_effective'] = trait_effective
        else:
            raise ValueError('Required property \'trait_effective\' not present in IdBasedMfaEnrollment JSON')
        if (complies := _dict.get('complies')) is not None:
            args['complies'] = complies
        else:
            raise ValueError('Required property \'complies\' not present in IdBasedMfaEnrollment JSON')
        if (comply_state := _dict.get('comply_state')) is not None:
            args['comply_state'] = comply_state
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a IdBasedMfaEnrollment object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'trait_account_default') and self.trait_account_default is not None:
            _dict['trait_account_default'] = self.trait_account_default
        if hasattr(self, 'trait_user_specific') and self.trait_user_specific is not None:
            _dict['trait_user_specific'] = self.trait_user_specific
        if hasattr(self, 'trait_effective') and self.trait_effective is not None:
            _dict['trait_effective'] = self.trait_effective
        if hasattr(self, 'complies') and self.complies is not None:
            _dict['complies'] = self.complies
        if hasattr(self, 'comply_state') and self.comply_state is not None:
            _dict['comply_state'] = self.comply_state
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this IdBasedMfaEnrollment object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'IdBasedMfaEnrollment') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'IdBasedMfaEnrollment') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class TraitAccountDefaultEnum(str, Enum):
        """
        Defines the MFA trait for the account. Valid values:
          * NONE - No MFA trait set
          * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
          * TOTP - For all non-federated IBMId users
          * TOTP4ALL - For all users
          * LEVEL1 - Email-based MFA for all users
          * LEVEL2 - TOTP-based MFA for all users
          * LEVEL3 - U2F MFA for all users.
        """

        NONE = 'NONE'
        NONE_NO_ROPC = 'NONE_NO_ROPC'
        TOTP = 'TOTP'
        TOTP4ALL = 'TOTP4ALL'
        LEVEL1 = 'LEVEL1'
        LEVEL2 = 'LEVEL2'
        LEVEL3 = 'LEVEL3'

    class TraitUserSpecificEnum(str, Enum):
        """
        Defines the MFA trait for the account. Valid values:
          * NONE - No MFA trait set
          * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
          * TOTP - For all non-federated IBMId users
          * TOTP4ALL - For all users
          * LEVEL1 - Email-based MFA for all users
          * LEVEL2 - TOTP-based MFA for all users
          * LEVEL3 - U2F MFA for all users.
        """

        NONE = 'NONE'
        NONE_NO_ROPC = 'NONE_NO_ROPC'
        TOTP = 'TOTP'
        TOTP4ALL = 'TOTP4ALL'
        LEVEL1 = 'LEVEL1'
        LEVEL2 = 'LEVEL2'
        LEVEL3 = 'LEVEL3'

    class TraitEffectiveEnum(str, Enum):
        """
        Defines the MFA trait for the account. Valid values:
          * NONE - No MFA trait set
          * NONE_NO_ROPC- No MFA, disable CLI logins with only a password
          * TOTP - For all non-federated IBMId users
          * TOTP4ALL - For all users
          * LEVEL1 - Email-based MFA for all users
          * LEVEL2 - TOTP-based MFA for all users
          * LEVEL3 - U2F MFA for all users.
        """

        NONE = 'NONE'
        NONE_NO_ROPC = 'NONE_NO_ROPC'
        TOTP = 'TOTP'
        TOTP4ALL = 'TOTP4ALL'
        LEVEL1 = 'LEVEL1'
        LEVEL2 = 'LEVEL2'
        LEVEL3 = 'LEVEL3'

    class ComplyStateEnum(str, Enum):
        """
        Defines comply state for the account. Valid values:
          * NO - User does not comply in the given account.
          * ACCOUNT- User complies in the given account, but does not comply in at least
        one of the other account memberships.
          * CROSS_ACCOUNT - User complies in the given account and across all other
        account memberships.
        """

        NO = 'NO'
        ACCOUNT = 'ACCOUNT'
        CROSS_ACCOUNT = 'CROSS_ACCOUNT'


class MfaEnrollmentTypeStatus:
    """
    MfaEnrollmentTypeStatus.

    :param bool required: Describes whether the enrollment type is required.
    :param bool enrolled: Describes whether the enrollment type is enrolled.
    """

    def __init__(
        self,
        required: bool,
        enrolled: bool,
    ) -> None:
        """
        Initialize a MfaEnrollmentTypeStatus object.

        :param bool required: Describes whether the enrollment type is required.
        :param bool enrolled: Describes whether the enrollment type is enrolled.
        """
        self.required = required
        self.enrolled = enrolled

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'MfaEnrollmentTypeStatus':
        """Initialize a MfaEnrollmentTypeStatus object from a json dictionary."""
        args = {}
        if (required := _dict.get('required')) is not None:
            args['required'] = required
        else:
            raise ValueError('Required property \'required\' not present in MfaEnrollmentTypeStatus JSON')
        if (enrolled := _dict.get('enrolled')) is not None:
            args['enrolled'] = enrolled
        else:
            raise ValueError('Required property \'enrolled\' not present in MfaEnrollmentTypeStatus JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a MfaEnrollmentTypeStatus object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'required') and self.required is not None:
            _dict['required'] = self.required
        if hasattr(self, 'enrolled') and self.enrolled is not None:
            _dict['enrolled'] = self.enrolled
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this MfaEnrollmentTypeStatus object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'MfaEnrollmentTypeStatus') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'MfaEnrollmentTypeStatus') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class MfaEnrollments:
    """
    MfaEnrollments.

    :param str effective_mfa_type: currently effective mfa type i.e. id_based_mfa or
          account_based_mfa.
    :param IdBasedMfaEnrollment id_based_mfa: (optional)
    :param AccountBasedMfaEnrollment account_based_mfa: (optional)
    """

    def __init__(
        self,
        effective_mfa_type: str,
        *,
        id_based_mfa: Optional['IdBasedMfaEnrollment'] = None,
        account_based_mfa: Optional['AccountBasedMfaEnrollment'] = None,
    ) -> None:
        """
        Initialize a MfaEnrollments object.

        :param str effective_mfa_type: currently effective mfa type i.e.
               id_based_mfa or account_based_mfa.
        :param IdBasedMfaEnrollment id_based_mfa: (optional)
        :param AccountBasedMfaEnrollment account_based_mfa: (optional)
        """
        self.effective_mfa_type = effective_mfa_type
        self.id_based_mfa = id_based_mfa
        self.account_based_mfa = account_based_mfa

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'MfaEnrollments':
        """Initialize a MfaEnrollments object from a json dictionary."""
        args = {}
        if (effective_mfa_type := _dict.get('effective_mfa_type')) is not None:
            args['effective_mfa_type'] = effective_mfa_type
        else:
            raise ValueError('Required property \'effective_mfa_type\' not present in MfaEnrollments JSON')
        if (id_based_mfa := _dict.get('id_based_mfa')) is not None:
            args['id_based_mfa'] = IdBasedMfaEnrollment.from_dict(id_based_mfa)
        if (account_based_mfa := _dict.get('account_based_mfa')) is not None:
            args['account_based_mfa'] = AccountBasedMfaEnrollment.from_dict(account_based_mfa)
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a MfaEnrollments object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'effective_mfa_type') and self.effective_mfa_type is not None:
            _dict['effective_mfa_type'] = self.effective_mfa_type
        if hasattr(self, 'id_based_mfa') and self.id_based_mfa is not None:
            if isinstance(self.id_based_mfa, dict):
                _dict['id_based_mfa'] = self.id_based_mfa
            else:
                _dict['id_based_mfa'] = self.id_based_mfa.to_dict()
        if hasattr(self, 'account_based_mfa') and self.account_based_mfa is not None:
            if isinstance(self.account_based_mfa, dict):
                _dict['account_based_mfa'] = self.account_based_mfa
            else:
                _dict['account_based_mfa'] = self.account_based_mfa.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this MfaEnrollments object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'MfaEnrollments') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'MfaEnrollments') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class PolicyTemplateReference:
    """
    Metadata for external access policy.

    :param str id: ID of Access Policy Template.
    :param str version: Version of Access Policy Template.
    """

    def __init__(
        self,
        id: str,
        version: str,
    ) -> None:
        """
        Initialize a PolicyTemplateReference object.

        :param str id: ID of Access Policy Template.
        :param str version: Version of Access Policy Template.
        """
        self.id = id
        self.version = version

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'PolicyTemplateReference':
        """Initialize a PolicyTemplateReference object from a json dictionary."""
        args = {}
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        else:
            raise ValueError('Required property \'id\' not present in PolicyTemplateReference JSON')
        if (version := _dict.get('version')) is not None:
            args['version'] = version
        else:
            raise ValueError('Required property \'version\' not present in PolicyTemplateReference JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a PolicyTemplateReference object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'version') and self.version is not None:
            _dict['version'] = self.version
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this PolicyTemplateReference object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'PolicyTemplateReference') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'PolicyTemplateReference') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ProfileClaimRule:
    """
    ProfileClaimRule.

    :param str id: the unique identifier of the claim rule.
    :param str entity_tag: version of the claim rule.
    :param datetime created_at: If set contains a date time string of the creation
          date in ISO format.
    :param datetime modified_at: (optional) If set contains a date time string of
          the last modification date in ISO format.
    :param str name: (optional) The optional claim rule name.
    :param str type: Type of the claim rule, either 'Profile-SAML' or 'Profile-CR'.
    :param str realm_name: (optional) The realm name of the Idp this claim rule
          applies to.
    :param int expiration: Session expiration in seconds.
    :param str cr_type: (optional) The compute resource type. Not required if type
          is Profile-SAML. Valid values are VSI, IKS_SA, ROKS_SA.
    :param List[ProfileClaimRuleConditions] conditions: Conditions of this claim
          rule.
    """

    def __init__(
        self,
        id: str,
        entity_tag: str,
        created_at: datetime,
        type: str,
        expiration: int,
        conditions: List['ProfileClaimRuleConditions'],
        *,
        modified_at: Optional[datetime] = None,
        name: Optional[str] = None,
        realm_name: Optional[str] = None,
        cr_type: Optional[str] = None,
    ) -> None:
        """
        Initialize a ProfileClaimRule object.

        :param str id: the unique identifier of the claim rule.
        :param str entity_tag: version of the claim rule.
        :param datetime created_at: If set contains a date time string of the
               creation date in ISO format.
        :param str type: Type of the claim rule, either 'Profile-SAML' or
               'Profile-CR'.
        :param int expiration: Session expiration in seconds.
        :param List[ProfileClaimRuleConditions] conditions: Conditions of this
               claim rule.
        :param datetime modified_at: (optional) If set contains a date time string
               of the last modification date in ISO format.
        :param str name: (optional) The optional claim rule name.
        :param str realm_name: (optional) The realm name of the Idp this claim rule
               applies to.
        :param str cr_type: (optional) The compute resource type. Not required if
               type is Profile-SAML. Valid values are VSI, IKS_SA, ROKS_SA.
        """
        self.id = id
        self.entity_tag = entity_tag
        self.created_at = created_at
        self.modified_at = modified_at
        self.name = name
        self.type = type
        self.realm_name = realm_name
        self.expiration = expiration
        self.cr_type = cr_type
        self.conditions = conditions

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ProfileClaimRule':
        """Initialize a ProfileClaimRule object from a json dictionary."""
        args = {}
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        else:
            raise ValueError('Required property \'id\' not present in ProfileClaimRule JSON')
        if (entity_tag := _dict.get('entity_tag')) is not None:
            args['entity_tag'] = entity_tag
        else:
            raise ValueError('Required property \'entity_tag\' not present in ProfileClaimRule JSON')
        if (created_at := _dict.get('created_at')) is not None:
            args['created_at'] = string_to_datetime(created_at)
        else:
            raise ValueError('Required property \'created_at\' not present in ProfileClaimRule JSON')
        if (modified_at := _dict.get('modified_at')) is not None:
            args['modified_at'] = string_to_datetime(modified_at)
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        if (type := _dict.get('type')) is not None:
            args['type'] = type
        else:
            raise ValueError('Required property \'type\' not present in ProfileClaimRule JSON')
        if (realm_name := _dict.get('realm_name')) is not None:
            args['realm_name'] = realm_name
        if (expiration := _dict.get('expiration')) is not None:
            args['expiration'] = expiration
        else:
            raise ValueError('Required property \'expiration\' not present in ProfileClaimRule JSON')
        if (cr_type := _dict.get('cr_type')) is not None:
            args['cr_type'] = cr_type
        if (conditions := _dict.get('conditions')) is not None:
            args['conditions'] = [ProfileClaimRuleConditions.from_dict(v) for v in conditions]
        else:
            raise ValueError('Required property \'conditions\' not present in ProfileClaimRule JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ProfileClaimRule object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'entity_tag') and self.entity_tag is not None:
            _dict['entity_tag'] = self.entity_tag
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'modified_at') and self.modified_at is not None:
            _dict['modified_at'] = datetime_to_string(self.modified_at)
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'realm_name') and self.realm_name is not None:
            _dict['realm_name'] = self.realm_name
        if hasattr(self, 'expiration') and self.expiration is not None:
            _dict['expiration'] = self.expiration
        if hasattr(self, 'cr_type') and self.cr_type is not None:
            _dict['cr_type'] = self.cr_type
        if hasattr(self, 'conditions') and self.conditions is not None:
            conditions_list = []
            for v in self.conditions:
                if isinstance(v, dict):
                    conditions_list.append(v)
                else:
                    conditions_list.append(v.to_dict())
            _dict['conditions'] = conditions_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ProfileClaimRule object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ProfileClaimRule') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ProfileClaimRule') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ProfileClaimRuleConditions:
    """
    ProfileClaimRuleConditions.

    :param str claim: The claim to evaluate against. [Learn
          more](/docs/account?topic=account-iam-condition-properties&interface=ui#cr-attribute-names).
    :param str operator: The operation to perform on the claim. valid values are
          EQUALS, NOT_EQUALS, EQUALS_IGNORE_CASE, NOT_EQUALS_IGNORE_CASE, CONTAINS, IN.
    :param str value: The stringified JSON value that the claim is compared to using
          the operator.
    """

    def __init__(
        self,
        claim: str,
        operator: str,
        value: str,
    ) -> None:
        """
        Initialize a ProfileClaimRuleConditions object.

        :param str claim: The claim to evaluate against. [Learn
               more](/docs/account?topic=account-iam-condition-properties&interface=ui#cr-attribute-names).
        :param str operator: The operation to perform on the claim. valid values
               are EQUALS, NOT_EQUALS, EQUALS_IGNORE_CASE, NOT_EQUALS_IGNORE_CASE,
               CONTAINS, IN.
        :param str value: The stringified JSON value that the claim is compared to
               using the operator.
        """
        self.claim = claim
        self.operator = operator
        self.value = value

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ProfileClaimRuleConditions':
        """Initialize a ProfileClaimRuleConditions object from a json dictionary."""
        args = {}
        if (claim := _dict.get('claim')) is not None:
            args['claim'] = claim
        else:
            raise ValueError('Required property \'claim\' not present in ProfileClaimRuleConditions JSON')
        if (operator := _dict.get('operator')) is not None:
            args['operator'] = operator
        else:
            raise ValueError('Required property \'operator\' not present in ProfileClaimRuleConditions JSON')
        if (value := _dict.get('value')) is not None:
            args['value'] = value
        else:
            raise ValueError('Required property \'value\' not present in ProfileClaimRuleConditions JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ProfileClaimRuleConditions object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'claim') and self.claim is not None:
            _dict['claim'] = self.claim
        if hasattr(self, 'operator') and self.operator is not None:
            _dict['operator'] = self.operator
        if hasattr(self, 'value') and self.value is not None:
            _dict['value'] = self.value
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ProfileClaimRuleConditions object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ProfileClaimRuleConditions') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ProfileClaimRuleConditions') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ProfileClaimRuleList:
    """
    ProfileClaimRuleList.

    :param ResponseContext context: (optional) Context with key properties for
          problem determination.
    :param List[ProfileClaimRule] rules: List of claim rules.
    """

    def __init__(
        self,
        rules: List['ProfileClaimRule'],
        *,
        context: Optional['ResponseContext'] = None,
    ) -> None:
        """
        Initialize a ProfileClaimRuleList object.

        :param List[ProfileClaimRule] rules: List of claim rules.
        :param ResponseContext context: (optional) Context with key properties for
               problem determination.
        """
        self.context = context
        self.rules = rules

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ProfileClaimRuleList':
        """Initialize a ProfileClaimRuleList object from a json dictionary."""
        args = {}
        if (context := _dict.get('context')) is not None:
            args['context'] = ResponseContext.from_dict(context)
        if (rules := _dict.get('rules')) is not None:
            args['rules'] = [ProfileClaimRule.from_dict(v) for v in rules]
        else:
            raise ValueError('Required property \'rules\' not present in ProfileClaimRuleList JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ProfileClaimRuleList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'context') and self.context is not None:
            if isinstance(self.context, dict):
                _dict['context'] = self.context
            else:
                _dict['context'] = self.context.to_dict()
        if hasattr(self, 'rules') and self.rules is not None:
            rules_list = []
            for v in self.rules:
                if isinstance(v, dict):
                    rules_list.append(v)
                else:
                    rules_list.append(v.to_dict())
            _dict['rules'] = rules_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ProfileClaimRuleList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ProfileClaimRuleList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ProfileClaimRuleList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ProfileIdentitiesResponse:
    """
    ProfileIdentitiesResponse.

    :param str entity_tag: (optional) Entity tag of the profile identities response.
    :param List[ProfileIdentityResponse] identities: (optional) List of identities.
    """

    def __init__(
        self,
        *,
        entity_tag: Optional[str] = None,
        identities: Optional[List['ProfileIdentityResponse']] = None,
    ) -> None:
        """
        Initialize a ProfileIdentitiesResponse object.

        :param str entity_tag: (optional) Entity tag of the profile identities
               response.
        :param List[ProfileIdentityResponse] identities: (optional) List of
               identities.
        """
        self.entity_tag = entity_tag
        self.identities = identities

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ProfileIdentitiesResponse':
        """Initialize a ProfileIdentitiesResponse object from a json dictionary."""
        args = {}
        if (entity_tag := _dict.get('entity_tag')) is not None:
            args['entity_tag'] = entity_tag
        if (identities := _dict.get('identities')) is not None:
            args['identities'] = [ProfileIdentityResponse.from_dict(v) for v in identities]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ProfileIdentitiesResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'entity_tag') and self.entity_tag is not None:
            _dict['entity_tag'] = self.entity_tag
        if hasattr(self, 'identities') and self.identities is not None:
            identities_list = []
            for v in self.identities:
                if isinstance(v, dict):
                    identities_list.append(v)
                else:
                    identities_list.append(v.to_dict())
            _dict['identities'] = identities_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ProfileIdentitiesResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ProfileIdentitiesResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ProfileIdentitiesResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ProfileIdentityRequest:
    """
    ProfileIdentityRequest.

    :param str identifier: Identifier of the identity that can assume the trusted
          profiles. This can be a user identifier (IAM id), serviceid or crn. Internally
          it uses account id of the service id for the identifier 'serviceid' and for the
          identifier 'crn' it uses account id contained in the CRN.
    :param str type: Type of the identity.
    :param List[str] accounts: (optional) Only valid for the type user. Accounts
          from which a user can assume the trusted profile.
    :param str description: (optional) Description of the identity that can assume
          the trusted profile. This is optional field for all the types of identities.
          When this field is not set for the identity type 'serviceid' then the
          description of the service id is used. Description is recommended for the
          identity type 'crn' E.g. 'Instance 1234 of IBM Cloud Service project'.
    """

    def __init__(
        self,
        identifier: str,
        type: str,
        *,
        accounts: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Initialize a ProfileIdentityRequest object.

        :param str identifier: Identifier of the identity that can assume the
               trusted profiles. This can be a user identifier (IAM id), serviceid or crn.
               Internally it uses account id of the service id for the identifier
               'serviceid' and for the identifier 'crn' it uses account id contained in
               the CRN.
        :param str type: Type of the identity.
        :param List[str] accounts: (optional) Only valid for the type user.
               Accounts from which a user can assume the trusted profile.
        :param str description: (optional) Description of the identity that can
               assume the trusted profile. This is optional field for all the types of
               identities. When this field is not set for the identity type 'serviceid'
               then the description of the service id is used. Description is recommended
               for the identity type 'crn' E.g. 'Instance 1234 of IBM Cloud Service
               project'.
        """
        self.identifier = identifier
        self.type = type
        self.accounts = accounts
        self.description = description

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ProfileIdentityRequest':
        """Initialize a ProfileIdentityRequest object from a json dictionary."""
        args = {}
        if (identifier := _dict.get('identifier')) is not None:
            args['identifier'] = identifier
        else:
            raise ValueError('Required property \'identifier\' not present in ProfileIdentityRequest JSON')
        if (type := _dict.get('type')) is not None:
            args['type'] = type
        else:
            raise ValueError('Required property \'type\' not present in ProfileIdentityRequest JSON')
        if (accounts := _dict.get('accounts')) is not None:
            args['accounts'] = accounts
        if (description := _dict.get('description')) is not None:
            args['description'] = description
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ProfileIdentityRequest object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'identifier') and self.identifier is not None:
            _dict['identifier'] = self.identifier
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'accounts') and self.accounts is not None:
            _dict['accounts'] = self.accounts
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ProfileIdentityRequest object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ProfileIdentityRequest') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ProfileIdentityRequest') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class TypeEnum(str, Enum):
        """
        Type of the identity.
        """

        USER = 'user'
        SERVICEID = 'serviceid'
        CRN = 'crn'


class ProfileIdentityResponse:
    """
    ProfileIdentityResponse.

    :param str iam_id: IAM ID of the identity.
    :param str identifier: Identifier of the identity that can assume the trusted
          profiles. This can be a user identifier (IAM id), serviceid or crn. Internally
          it uses account id of the service id for the identifier 'serviceid' and for the
          identifier 'crn' it uses account id contained in the CRN.
    :param str type: Type of the identity.
    :param List[str] accounts: (optional) Only valid for the type user. Accounts
          from which a user can assume the trusted profile.
    :param str description: (optional) Description of the identity that can assume
          the trusted profile. This is optional field for all the types of identities.
          When this field is not set for the identity type 'serviceid' then the
          description of the service id is used. Description is recommended for the
          identity type 'crn' E.g. 'Instance 1234 of IBM Cloud Service project'.
    """

    def __init__(
        self,
        iam_id: str,
        identifier: str,
        type: str,
        *,
        accounts: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Initialize a ProfileIdentityResponse object.

        :param str iam_id: IAM ID of the identity.
        :param str identifier: Identifier of the identity that can assume the
               trusted profiles. This can be a user identifier (IAM id), serviceid or crn.
               Internally it uses account id of the service id for the identifier
               'serviceid' and for the identifier 'crn' it uses account id contained in
               the CRN.
        :param str type: Type of the identity.
        :param List[str] accounts: (optional) Only valid for the type user.
               Accounts from which a user can assume the trusted profile.
        :param str description: (optional) Description of the identity that can
               assume the trusted profile. This is optional field for all the types of
               identities. When this field is not set for the identity type 'serviceid'
               then the description of the service id is used. Description is recommended
               for the identity type 'crn' E.g. 'Instance 1234 of IBM Cloud Service
               project'.
        """
        self.iam_id = iam_id
        self.identifier = identifier
        self.type = type
        self.accounts = accounts
        self.description = description

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ProfileIdentityResponse':
        """Initialize a ProfileIdentityResponse object from a json dictionary."""
        args = {}
        if (iam_id := _dict.get('iam_id')) is not None:
            args['iam_id'] = iam_id
        else:
            raise ValueError('Required property \'iam_id\' not present in ProfileIdentityResponse JSON')
        if (identifier := _dict.get('identifier')) is not None:
            args['identifier'] = identifier
        else:
            raise ValueError('Required property \'identifier\' not present in ProfileIdentityResponse JSON')
        if (type := _dict.get('type')) is not None:
            args['type'] = type
        else:
            raise ValueError('Required property \'type\' not present in ProfileIdentityResponse JSON')
        if (accounts := _dict.get('accounts')) is not None:
            args['accounts'] = accounts
        if (description := _dict.get('description')) is not None:
            args['description'] = description
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ProfileIdentityResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'iam_id') and self.iam_id is not None:
            _dict['iam_id'] = self.iam_id
        if hasattr(self, 'identifier') and self.identifier is not None:
            _dict['identifier'] = self.identifier
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'accounts') and self.accounts is not None:
            _dict['accounts'] = self.accounts
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ProfileIdentityResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ProfileIdentityResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ProfileIdentityResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class TypeEnum(str, Enum):
        """
        Type of the identity.
        """

        USER = 'user'
        SERVICEID = 'serviceid'
        CRN = 'crn'


class ProfileLink:
    """
    Link details.

    :param str id: the unique identifier of the link.
    :param str entity_tag: version of the link.
    :param datetime created_at: If set contains a date time string of the creation
          date in ISO format.
    :param datetime modified_at: If set contains a date time string of the last
          modification date in ISO format.
    :param str name: (optional) Optional name of the Link.
    :param str cr_type: The compute resource type. Valid values are VSI, IKS_SA,
          ROKS_SA.
    :param ProfileLinkLink link:
    """

    def __init__(
        self,
        id: str,
        entity_tag: str,
        created_at: datetime,
        modified_at: datetime,
        cr_type: str,
        link: 'ProfileLinkLink',
        *,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize a ProfileLink object.

        :param str id: the unique identifier of the link.
        :param str entity_tag: version of the link.
        :param datetime created_at: If set contains a date time string of the
               creation date in ISO format.
        :param datetime modified_at: If set contains a date time string of the last
               modification date in ISO format.
        :param str cr_type: The compute resource type. Valid values are VSI,
               IKS_SA, ROKS_SA.
        :param ProfileLinkLink link:
        :param str name: (optional) Optional name of the Link.
        """
        self.id = id
        self.entity_tag = entity_tag
        self.created_at = created_at
        self.modified_at = modified_at
        self.name = name
        self.cr_type = cr_type
        self.link = link

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ProfileLink':
        """Initialize a ProfileLink object from a json dictionary."""
        args = {}
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        else:
            raise ValueError('Required property \'id\' not present in ProfileLink JSON')
        if (entity_tag := _dict.get('entity_tag')) is not None:
            args['entity_tag'] = entity_tag
        else:
            raise ValueError('Required property \'entity_tag\' not present in ProfileLink JSON')
        if (created_at := _dict.get('created_at')) is not None:
            args['created_at'] = string_to_datetime(created_at)
        else:
            raise ValueError('Required property \'created_at\' not present in ProfileLink JSON')
        if (modified_at := _dict.get('modified_at')) is not None:
            args['modified_at'] = string_to_datetime(modified_at)
        else:
            raise ValueError('Required property \'modified_at\' not present in ProfileLink JSON')
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        if (cr_type := _dict.get('cr_type')) is not None:
            args['cr_type'] = cr_type
        else:
            raise ValueError('Required property \'cr_type\' not present in ProfileLink JSON')
        if (link := _dict.get('link')) is not None:
            args['link'] = ProfileLinkLink.from_dict(link)
        else:
            raise ValueError('Required property \'link\' not present in ProfileLink JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ProfileLink object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'entity_tag') and self.entity_tag is not None:
            _dict['entity_tag'] = self.entity_tag
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'modified_at') and self.modified_at is not None:
            _dict['modified_at'] = datetime_to_string(self.modified_at)
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'cr_type') and self.cr_type is not None:
            _dict['cr_type'] = self.cr_type
        if hasattr(self, 'link') and self.link is not None:
            if isinstance(self.link, dict):
                _dict['link'] = self.link
            else:
                _dict['link'] = self.link.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ProfileLink object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ProfileLink') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ProfileLink') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ProfileLinkLink:
    """
    ProfileLinkLink.

    :param str crn: (optional) The CRN of the compute resource.
    :param str namespace: (optional) The compute resource namespace, only required
          if cr_type is IKS_SA or ROKS_SA.
    :param str name: (optional) Name of the compute resource, only required if
          cr_type is IKS_SA or ROKS_SA.
    """

    def __init__(
        self,
        *,
        crn: Optional[str] = None,
        namespace: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize a ProfileLinkLink object.

        :param str crn: (optional) The CRN of the compute resource.
        :param str namespace: (optional) The compute resource namespace, only
               required if cr_type is IKS_SA or ROKS_SA.
        :param str name: (optional) Name of the compute resource, only required if
               cr_type is IKS_SA or ROKS_SA.
        """
        self.crn = crn
        self.namespace = namespace
        self.name = name

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ProfileLinkLink':
        """Initialize a ProfileLinkLink object from a json dictionary."""
        args = {}
        if (crn := _dict.get('crn')) is not None:
            args['crn'] = crn
        if (namespace := _dict.get('namespace')) is not None:
            args['namespace'] = namespace
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ProfileLinkLink object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'namespace') and self.namespace is not None:
            _dict['namespace'] = self.namespace
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ProfileLinkLink object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ProfileLinkLink') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ProfileLinkLink') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ProfileLinkList:
    """
    ProfileLinkList.

    :param List[ProfileLink] links: List of links to a trusted profile.
    """

    def __init__(
        self,
        links: List['ProfileLink'],
    ) -> None:
        """
        Initialize a ProfileLinkList object.

        :param List[ProfileLink] links: List of links to a trusted profile.
        """
        self.links = links

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ProfileLinkList':
        """Initialize a ProfileLinkList object from a json dictionary."""
        args = {}
        if (links := _dict.get('links')) is not None:
            args['links'] = [ProfileLink.from_dict(v) for v in links]
        else:
            raise ValueError('Required property \'links\' not present in ProfileLinkList JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ProfileLinkList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'links') and self.links is not None:
            links_list = []
            for v in self.links:
                if isinstance(v, dict):
                    links_list.append(v)
                else:
                    links_list.append(v.to_dict())
            _dict['links'] = links_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ProfileLinkList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ProfileLinkList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ProfileLinkList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Report:
    """
    Report.

    :param str created_by: IAMid of the user who triggered the report.
    :param str reference: Unique reference used to generate the report.
    :param str report_duration: Duration in hours for which the report is generated.
    :param str report_start_time: Start time of the report.
    :param str report_end_time: End time of the report.
    :param List[UserActivity] users: (optional) List of users.
    :param List[ApikeyActivity] apikeys: (optional) List of apikeys.
    :param List[EntityActivity] serviceids: (optional) List of serviceids.
    :param List[EntityActivity] profiles: (optional) List of profiles.
    """

    def __init__(
        self,
        created_by: str,
        reference: str,
        report_duration: str,
        report_start_time: str,
        report_end_time: str,
        *,
        users: Optional[List['UserActivity']] = None,
        apikeys: Optional[List['ApikeyActivity']] = None,
        serviceids: Optional[List['EntityActivity']] = None,
        profiles: Optional[List['EntityActivity']] = None,
    ) -> None:
        """
        Initialize a Report object.

        :param str created_by: IAMid of the user who triggered the report.
        :param str reference: Unique reference used to generate the report.
        :param str report_duration: Duration in hours for which the report is
               generated.
        :param str report_start_time: Start time of the report.
        :param str report_end_time: End time of the report.
        :param List[UserActivity] users: (optional) List of users.
        :param List[ApikeyActivity] apikeys: (optional) List of apikeys.
        :param List[EntityActivity] serviceids: (optional) List of serviceids.
        :param List[EntityActivity] profiles: (optional) List of profiles.
        """
        self.created_by = created_by
        self.reference = reference
        self.report_duration = report_duration
        self.report_start_time = report_start_time
        self.report_end_time = report_end_time
        self.users = users
        self.apikeys = apikeys
        self.serviceids = serviceids
        self.profiles = profiles

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Report':
        """Initialize a Report object from a json dictionary."""
        args = {}
        if (created_by := _dict.get('created_by')) is not None:
            args['created_by'] = created_by
        else:
            raise ValueError('Required property \'created_by\' not present in Report JSON')
        if (reference := _dict.get('reference')) is not None:
            args['reference'] = reference
        else:
            raise ValueError('Required property \'reference\' not present in Report JSON')
        if (report_duration := _dict.get('report_duration')) is not None:
            args['report_duration'] = report_duration
        else:
            raise ValueError('Required property \'report_duration\' not present in Report JSON')
        if (report_start_time := _dict.get('report_start_time')) is not None:
            args['report_start_time'] = report_start_time
        else:
            raise ValueError('Required property \'report_start_time\' not present in Report JSON')
        if (report_end_time := _dict.get('report_end_time')) is not None:
            args['report_end_time'] = report_end_time
        else:
            raise ValueError('Required property \'report_end_time\' not present in Report JSON')
        if (users := _dict.get('users')) is not None:
            args['users'] = [UserActivity.from_dict(v) for v in users]
        if (apikeys := _dict.get('apikeys')) is not None:
            args['apikeys'] = [ApikeyActivity.from_dict(v) for v in apikeys]
        if (serviceids := _dict.get('serviceids')) is not None:
            args['serviceids'] = [EntityActivity.from_dict(v) for v in serviceids]
        if (profiles := _dict.get('profiles')) is not None:
            args['profiles'] = [EntityActivity.from_dict(v) for v in profiles]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Report object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'created_by') and self.created_by is not None:
            _dict['created_by'] = self.created_by
        if hasattr(self, 'reference') and self.reference is not None:
            _dict['reference'] = self.reference
        if hasattr(self, 'report_duration') and self.report_duration is not None:
            _dict['report_duration'] = self.report_duration
        if hasattr(self, 'report_start_time') and self.report_start_time is not None:
            _dict['report_start_time'] = self.report_start_time
        if hasattr(self, 'report_end_time') and self.report_end_time is not None:
            _dict['report_end_time'] = self.report_end_time
        if hasattr(self, 'users') and self.users is not None:
            users_list = []
            for v in self.users:
                if isinstance(v, dict):
                    users_list.append(v)
                else:
                    users_list.append(v.to_dict())
            _dict['users'] = users_list
        if hasattr(self, 'apikeys') and self.apikeys is not None:
            apikeys_list = []
            for v in self.apikeys:
                if isinstance(v, dict):
                    apikeys_list.append(v)
                else:
                    apikeys_list.append(v.to_dict())
            _dict['apikeys'] = apikeys_list
        if hasattr(self, 'serviceids') and self.serviceids is not None:
            serviceids_list = []
            for v in self.serviceids:
                if isinstance(v, dict):
                    serviceids_list.append(v)
                else:
                    serviceids_list.append(v.to_dict())
            _dict['serviceids'] = serviceids_list
        if hasattr(self, 'profiles') and self.profiles is not None:
            profiles_list = []
            for v in self.profiles:
                if isinstance(v, dict):
                    profiles_list.append(v)
                else:
                    profiles_list.append(v.to_dict())
            _dict['profiles'] = profiles_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Report object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Report') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Report') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ReportMfaEnrollmentStatus:
    """
    ReportMfaEnrollmentStatus.

    :param str created_by: IAMid of the user who triggered the report.
    :param str reference: Unique reference used to generate the report.
    :param str report_time: Date time at which report is generated. Date is in ISO
          format.
    :param str account_id: BSS account id of the user who triggered the report.
    :param str ims_account_id: (optional) IMS account id of the user who triggered
          the report.
    :param List[UserReportMfaEnrollmentStatus] users: (optional) List of users.
    """

    def __init__(
        self,
        created_by: str,
        reference: str,
        report_time: str,
        account_id: str,
        *,
        ims_account_id: Optional[str] = None,
        users: Optional[List['UserReportMfaEnrollmentStatus']] = None,
    ) -> None:
        """
        Initialize a ReportMfaEnrollmentStatus object.

        :param str created_by: IAMid of the user who triggered the report.
        :param str reference: Unique reference used to generate the report.
        :param str report_time: Date time at which report is generated. Date is in
               ISO format.
        :param str account_id: BSS account id of the user who triggered the report.
        :param str ims_account_id: (optional) IMS account id of the user who
               triggered the report.
        :param List[UserReportMfaEnrollmentStatus] users: (optional) List of users.
        """
        self.created_by = created_by
        self.reference = reference
        self.report_time = report_time
        self.account_id = account_id
        self.ims_account_id = ims_account_id
        self.users = users

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ReportMfaEnrollmentStatus':
        """Initialize a ReportMfaEnrollmentStatus object from a json dictionary."""
        args = {}
        if (created_by := _dict.get('created_by')) is not None:
            args['created_by'] = created_by
        else:
            raise ValueError('Required property \'created_by\' not present in ReportMfaEnrollmentStatus JSON')
        if (reference := _dict.get('reference')) is not None:
            args['reference'] = reference
        else:
            raise ValueError('Required property \'reference\' not present in ReportMfaEnrollmentStatus JSON')
        if (report_time := _dict.get('report_time')) is not None:
            args['report_time'] = report_time
        else:
            raise ValueError('Required property \'report_time\' not present in ReportMfaEnrollmentStatus JSON')
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        else:
            raise ValueError('Required property \'account_id\' not present in ReportMfaEnrollmentStatus JSON')
        if (ims_account_id := _dict.get('ims_account_id')) is not None:
            args['ims_account_id'] = ims_account_id
        if (users := _dict.get('users')) is not None:
            args['users'] = [UserReportMfaEnrollmentStatus.from_dict(v) for v in users]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ReportMfaEnrollmentStatus object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'created_by') and self.created_by is not None:
            _dict['created_by'] = self.created_by
        if hasattr(self, 'reference') and self.reference is not None:
            _dict['reference'] = self.reference
        if hasattr(self, 'report_time') and self.report_time is not None:
            _dict['report_time'] = self.report_time
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'ims_account_id') and self.ims_account_id is not None:
            _dict['ims_account_id'] = self.ims_account_id
        if hasattr(self, 'users') and self.users is not None:
            users_list = []
            for v in self.users:
                if isinstance(v, dict):
                    users_list.append(v)
                else:
                    users_list.append(v.to_dict())
            _dict['users'] = users_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ReportMfaEnrollmentStatus object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ReportMfaEnrollmentStatus') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ReportMfaEnrollmentStatus') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ReportReference:
    """
    ReportReference.

    :param str reference: Reference for the report to be generated.
    """

    def __init__(
        self,
        reference: str,
    ) -> None:
        """
        Initialize a ReportReference object.

        :param str reference: Reference for the report to be generated.
        """
        self.reference = reference

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ReportReference':
        """Initialize a ReportReference object from a json dictionary."""
        args = {}
        if (reference := _dict.get('reference')) is not None:
            args['reference'] = reference
        else:
            raise ValueError('Required property \'reference\' not present in ReportReference JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ReportReference object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'reference') and self.reference is not None:
            _dict['reference'] = self.reference
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ReportReference object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ReportReference') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ReportReference') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ResponseContext:
    """
    Context with key properties for problem determination.

    :param str transaction_id: (optional) The transaction ID of the inbound REST
          request.
    :param str operation: (optional) The operation of the inbound REST request.
    :param str user_agent: (optional) The user agent of the inbound REST request.
    :param str url: (optional) The URL of that cluster.
    :param str instance_id: (optional) The instance ID of the server instance
          processing the request.
    :param str thread_id: (optional) The thread ID of the server instance processing
          the request.
    :param str host: (optional) The host of the server instance processing the
          request.
    :param str start_time: (optional) The start time of the request.
    :param str end_time: (optional) The finish time of the request.
    :param str elapsed_time: (optional) The elapsed time in msec.
    :param str cluster_name: (optional) The cluster name.
    """

    def __init__(
        self,
        *,
        transaction_id: Optional[str] = None,
        operation: Optional[str] = None,
        user_agent: Optional[str] = None,
        url: Optional[str] = None,
        instance_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        host: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        elapsed_time: Optional[str] = None,
        cluster_name: Optional[str] = None,
    ) -> None:
        """
        Initialize a ResponseContext object.

        :param str transaction_id: (optional) The transaction ID of the inbound
               REST request.
        :param str operation: (optional) The operation of the inbound REST request.
        :param str user_agent: (optional) The user agent of the inbound REST
               request.
        :param str url: (optional) The URL of that cluster.
        :param str instance_id: (optional) The instance ID of the server instance
               processing the request.
        :param str thread_id: (optional) The thread ID of the server instance
               processing the request.
        :param str host: (optional) The host of the server instance processing the
               request.
        :param str start_time: (optional) The start time of the request.
        :param str end_time: (optional) The finish time of the request.
        :param str elapsed_time: (optional) The elapsed time in msec.
        :param str cluster_name: (optional) The cluster name.
        """
        self.transaction_id = transaction_id
        self.operation = operation
        self.user_agent = user_agent
        self.url = url
        self.instance_id = instance_id
        self.thread_id = thread_id
        self.host = host
        self.start_time = start_time
        self.end_time = end_time
        self.elapsed_time = elapsed_time
        self.cluster_name = cluster_name

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResponseContext':
        """Initialize a ResponseContext object from a json dictionary."""
        args = {}
        if (transaction_id := _dict.get('transaction_id')) is not None:
            args['transaction_id'] = transaction_id
        if (operation := _dict.get('operation')) is not None:
            args['operation'] = operation
        if (user_agent := _dict.get('user_agent')) is not None:
            args['user_agent'] = user_agent
        if (url := _dict.get('url')) is not None:
            args['url'] = url
        if (instance_id := _dict.get('instance_id')) is not None:
            args['instance_id'] = instance_id
        if (thread_id := _dict.get('thread_id')) is not None:
            args['thread_id'] = thread_id
        if (host := _dict.get('host')) is not None:
            args['host'] = host
        if (start_time := _dict.get('start_time')) is not None:
            args['start_time'] = start_time
        if (end_time := _dict.get('end_time')) is not None:
            args['end_time'] = end_time
        if (elapsed_time := _dict.get('elapsed_time')) is not None:
            args['elapsed_time'] = elapsed_time
        if (cluster_name := _dict.get('cluster_name')) is not None:
            args['cluster_name'] = cluster_name
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResponseContext object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'transaction_id') and self.transaction_id is not None:
            _dict['transaction_id'] = self.transaction_id
        if hasattr(self, 'operation') and self.operation is not None:
            _dict['operation'] = self.operation
        if hasattr(self, 'user_agent') and self.user_agent is not None:
            _dict['user_agent'] = self.user_agent
        if hasattr(self, 'url') and self.url is not None:
            _dict['url'] = self.url
        if hasattr(self, 'instance_id') and self.instance_id is not None:
            _dict['instance_id'] = self.instance_id
        if hasattr(self, 'thread_id') and self.thread_id is not None:
            _dict['thread_id'] = self.thread_id
        if hasattr(self, 'host') and self.host is not None:
            _dict['host'] = self.host
        if hasattr(self, 'start_time') and self.start_time is not None:
            _dict['start_time'] = self.start_time
        if hasattr(self, 'end_time') and self.end_time is not None:
            _dict['end_time'] = self.end_time
        if hasattr(self, 'elapsed_time') and self.elapsed_time is not None:
            _dict['elapsed_time'] = self.elapsed_time
        if hasattr(self, 'cluster_name') and self.cluster_name is not None:
            _dict['cluster_name'] = self.cluster_name
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResponseContext object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResponseContext') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResponseContext') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ServiceId:
    """
    Response body format for service ID V1 REST requests.

    :param ResponseContext context: (optional) Context with key properties for
          problem determination.
    :param str id: Unique identifier of this Service Id.
    :param str iam_id: Cloud wide identifier for identities of this service ID.
    :param str entity_tag: Version of the service ID details object. You need to
          specify this value when updating the service ID to avoid stale updates.
    :param str crn: Cloud Resource Name of the item. Example Cloud Resource Name:
          'crn:v1:bluemix:public:iam-identity:us-south:a/myaccount::serviceid:1234-5678-9012'.
    :param bool locked: The service ID cannot be changed if set to true.
    :param datetime created_at: If set contains a date time string of the creation
          date in ISO format.
    :param datetime modified_at: If set contains a date time string of the last
          modification date in ISO format.
    :param str account_id: ID of the account the service ID belongs to.
    :param str name: Name of the Service Id. The name is not checked for uniqueness.
          Therefore multiple names with the same value can exist. Access is done via the
          UUID of the Service Id.
    :param str description: (optional) The optional description of the Service Id.
          The 'description' property is only available if a description was provided
          during a create of a Service Id.
    :param List[str] unique_instance_crns: (optional) Optional list of CRNs (string
          array) which point to the services connected to the service ID.
    :param List[EnityHistoryRecord] history: (optional) History of the Service ID.
    :param ApiKey apikey: (optional) Response body format for API key V1 REST
          requests.
    :param Activity activity: (optional)
    """

    def __init__(
        self,
        id: str,
        iam_id: str,
        entity_tag: str,
        crn: str,
        locked: bool,
        created_at: datetime,
        modified_at: datetime,
        account_id: str,
        name: str,
        *,
        context: Optional['ResponseContext'] = None,
        description: Optional[str] = None,
        unique_instance_crns: Optional[List[str]] = None,
        history: Optional[List['EnityHistoryRecord']] = None,
        apikey: Optional['ApiKey'] = None,
        activity: Optional['Activity'] = None,
    ) -> None:
        """
        Initialize a ServiceId object.

        :param str id: Unique identifier of this Service Id.
        :param str iam_id: Cloud wide identifier for identities of this service ID.
        :param str entity_tag: Version of the service ID details object. You need
               to specify this value when updating the service ID to avoid stale updates.
        :param str crn: Cloud Resource Name of the item. Example Cloud Resource
               Name:
               'crn:v1:bluemix:public:iam-identity:us-south:a/myaccount::serviceid:1234-5678-9012'.
        :param bool locked: The service ID cannot be changed if set to true.
        :param datetime created_at: If set contains a date time string of the
               creation date in ISO format.
        :param datetime modified_at: If set contains a date time string of the last
               modification date in ISO format.
        :param str account_id: ID of the account the service ID belongs to.
        :param str name: Name of the Service Id. The name is not checked for
               uniqueness. Therefore multiple names with the same value can exist. Access
               is done via the UUID of the Service Id.
        :param ResponseContext context: (optional) Context with key properties for
               problem determination.
        :param str description: (optional) The optional description of the Service
               Id. The 'description' property is only available if a description was
               provided during a create of a Service Id.
        :param List[str] unique_instance_crns: (optional) Optional list of CRNs
               (string array) which point to the services connected to the service ID.
        :param List[EnityHistoryRecord] history: (optional) History of the Service
               ID.
        :param ApiKey apikey: (optional) Response body format for API key V1 REST
               requests.
        :param Activity activity: (optional)
        """
        self.context = context
        self.id = id
        self.iam_id = iam_id
        self.entity_tag = entity_tag
        self.crn = crn
        self.locked = locked
        self.created_at = created_at
        self.modified_at = modified_at
        self.account_id = account_id
        self.name = name
        self.description = description
        self.unique_instance_crns = unique_instance_crns
        self.history = history
        self.apikey = apikey
        self.activity = activity

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ServiceId':
        """Initialize a ServiceId object from a json dictionary."""
        args = {}
        if (context := _dict.get('context')) is not None:
            args['context'] = ResponseContext.from_dict(context)
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        else:
            raise ValueError('Required property \'id\' not present in ServiceId JSON')
        if (iam_id := _dict.get('iam_id')) is not None:
            args['iam_id'] = iam_id
        else:
            raise ValueError('Required property \'iam_id\' not present in ServiceId JSON')
        if (entity_tag := _dict.get('entity_tag')) is not None:
            args['entity_tag'] = entity_tag
        else:
            raise ValueError('Required property \'entity_tag\' not present in ServiceId JSON')
        if (crn := _dict.get('crn')) is not None:
            args['crn'] = crn
        else:
            raise ValueError('Required property \'crn\' not present in ServiceId JSON')
        if (locked := _dict.get('locked')) is not None:
            args['locked'] = locked
        else:
            raise ValueError('Required property \'locked\' not present in ServiceId JSON')
        if (created_at := _dict.get('created_at')) is not None:
            args['created_at'] = string_to_datetime(created_at)
        else:
            raise ValueError('Required property \'created_at\' not present in ServiceId JSON')
        if (modified_at := _dict.get('modified_at')) is not None:
            args['modified_at'] = string_to_datetime(modified_at)
        else:
            raise ValueError('Required property \'modified_at\' not present in ServiceId JSON')
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        else:
            raise ValueError('Required property \'account_id\' not present in ServiceId JSON')
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        else:
            raise ValueError('Required property \'name\' not present in ServiceId JSON')
        if (description := _dict.get('description')) is not None:
            args['description'] = description
        if (unique_instance_crns := _dict.get('unique_instance_crns')) is not None:
            args['unique_instance_crns'] = unique_instance_crns
        if (history := _dict.get('history')) is not None:
            args['history'] = [EnityHistoryRecord.from_dict(v) for v in history]
        if (apikey := _dict.get('apikey')) is not None:
            args['apikey'] = ApiKey.from_dict(apikey)
        if (activity := _dict.get('activity')) is not None:
            args['activity'] = Activity.from_dict(activity)
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ServiceId object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'context') and self.context is not None:
            if isinstance(self.context, dict):
                _dict['context'] = self.context
            else:
                _dict['context'] = self.context.to_dict()
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'iam_id') and self.iam_id is not None:
            _dict['iam_id'] = self.iam_id
        if hasattr(self, 'entity_tag') and self.entity_tag is not None:
            _dict['entity_tag'] = self.entity_tag
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'locked') and self.locked is not None:
            _dict['locked'] = self.locked
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'modified_at') and self.modified_at is not None:
            _dict['modified_at'] = datetime_to_string(self.modified_at)
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'unique_instance_crns') and self.unique_instance_crns is not None:
            _dict['unique_instance_crns'] = self.unique_instance_crns
        if hasattr(self, 'history') and self.history is not None:
            history_list = []
            for v in self.history:
                if isinstance(v, dict):
                    history_list.append(v)
                else:
                    history_list.append(v.to_dict())
            _dict['history'] = history_list
        if hasattr(self, 'apikey') and self.apikey is not None:
            if isinstance(self.apikey, dict):
                _dict['apikey'] = self.apikey
            else:
                _dict['apikey'] = self.apikey.to_dict()
        if hasattr(self, 'activity') and self.activity is not None:
            if isinstance(self.activity, dict):
                _dict['activity'] = self.activity
            else:
                _dict['activity'] = self.activity.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ServiceId object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ServiceId') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ServiceId') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ServiceIdList:
    """
    Response body format for the list service ID V1 REST request.

    :param ResponseContext context: (optional) Context with key properties for
          problem determination.
    :param int offset: (optional) The offset of the current page.
    :param int limit: (optional) Optional size of a single page. Default is 20 items
          per page. Valid range is 1 to 100.
    :param str first: (optional) Link to the first page.
    :param str previous: (optional) Link to the previous available page. If
          'previous' property is not part of the response no previous page is available.
    :param str next: (optional) Link to the next available page. If 'next' property
          is not part of the response no next page is available.
    :param List[ServiceId] serviceids: List of service IDs based on the query
          paramters and the page size. The service IDs array is always part of the
          response but might be empty depending on the query parameter values provided.
    """

    def __init__(
        self,
        serviceids: List['ServiceId'],
        *,
        context: Optional['ResponseContext'] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        first: Optional[str] = None,
        previous: Optional[str] = None,
        next: Optional[str] = None,
    ) -> None:
        """
        Initialize a ServiceIdList object.

        :param List[ServiceId] serviceids: List of service IDs based on the query
               paramters and the page size. The service IDs array is always part of the
               response but might be empty depending on the query parameter values
               provided.
        :param ResponseContext context: (optional) Context with key properties for
               problem determination.
        :param int offset: (optional) The offset of the current page.
        :param int limit: (optional) Optional size of a single page. Default is 20
               items per page. Valid range is 1 to 100.
        :param str first: (optional) Link to the first page.
        :param str previous: (optional) Link to the previous available page. If
               'previous' property is not part of the response no previous page is
               available.
        :param str next: (optional) Link to the next available page. If 'next'
               property is not part of the response no next page is available.
        """
        self.context = context
        self.offset = offset
        self.limit = limit
        self.first = first
        self.previous = previous
        self.next = next
        self.serviceids = serviceids

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ServiceIdList':
        """Initialize a ServiceIdList object from a json dictionary."""
        args = {}
        if (context := _dict.get('context')) is not None:
            args['context'] = ResponseContext.from_dict(context)
        if (offset := _dict.get('offset')) is not None:
            args['offset'] = offset
        if (limit := _dict.get('limit')) is not None:
            args['limit'] = limit
        if (first := _dict.get('first')) is not None:
            args['first'] = first
        if (previous := _dict.get('previous')) is not None:
            args['previous'] = previous
        if (next := _dict.get('next')) is not None:
            args['next'] = next
        if (serviceids := _dict.get('serviceids')) is not None:
            args['serviceids'] = [ServiceId.from_dict(v) for v in serviceids]
        else:
            raise ValueError('Required property \'serviceids\' not present in ServiceIdList JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ServiceIdList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'context') and self.context is not None:
            if isinstance(self.context, dict):
                _dict['context'] = self.context
            else:
                _dict['context'] = self.context.to_dict()
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'first') and self.first is not None:
            _dict['first'] = self.first
        if hasattr(self, 'previous') and self.previous is not None:
            _dict['previous'] = self.previous
        if hasattr(self, 'next') and self.next is not None:
            _dict['next'] = self.next
        if hasattr(self, 'serviceids') and self.serviceids is not None:
            serviceids_list = []
            for v in self.serviceids:
                if isinstance(v, dict):
                    serviceids_list.append(v)
                else:
                    serviceids_list.append(v.to_dict())
            _dict['serviceids'] = serviceids_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ServiceIdList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ServiceIdList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ServiceIdList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TemplateAssignmentListResponse:
    """
    List Response body format for Template Assignments Records.

    :param ResponseContext context: (optional) Context with key properties for
          problem determination.
    :param int offset: (optional) The offset of the current page.
    :param int limit: (optional) Optional size of a single page. Default is 20 items
          per page. Valid range is 1 to 100.
    :param str first: (optional) Link to the first page.
    :param str previous: (optional) Link to the previous available page. If
          'previous' property is not part of the response no previous page is available.
    :param str next: (optional) Link to the next available page. If 'next' property
          is not part of the response no next page is available.
    :param List[TemplateAssignmentResponse] assignments: List of Assignments based
          on the query paramters and the page size. The assignments array is always part
          of the response but might be empty depending on the query parameter values
          provided.
    """

    def __init__(
        self,
        assignments: List['TemplateAssignmentResponse'],
        *,
        context: Optional['ResponseContext'] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        first: Optional[str] = None,
        previous: Optional[str] = None,
        next: Optional[str] = None,
    ) -> None:
        """
        Initialize a TemplateAssignmentListResponse object.

        :param List[TemplateAssignmentResponse] assignments: List of Assignments
               based on the query paramters and the page size. The assignments array is
               always part of the response but might be empty depending on the query
               parameter values provided.
        :param ResponseContext context: (optional) Context with key properties for
               problem determination.
        :param int offset: (optional) The offset of the current page.
        :param int limit: (optional) Optional size of a single page. Default is 20
               items per page. Valid range is 1 to 100.
        :param str first: (optional) Link to the first page.
        :param str previous: (optional) Link to the previous available page. If
               'previous' property is not part of the response no previous page is
               available.
        :param str next: (optional) Link to the next available page. If 'next'
               property is not part of the response no next page is available.
        """
        self.context = context
        self.offset = offset
        self.limit = limit
        self.first = first
        self.previous = previous
        self.next = next
        self.assignments = assignments

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TemplateAssignmentListResponse':
        """Initialize a TemplateAssignmentListResponse object from a json dictionary."""
        args = {}
        if (context := _dict.get('context')) is not None:
            args['context'] = ResponseContext.from_dict(context)
        if (offset := _dict.get('offset')) is not None:
            args['offset'] = offset
        if (limit := _dict.get('limit')) is not None:
            args['limit'] = limit
        if (first := _dict.get('first')) is not None:
            args['first'] = first
        if (previous := _dict.get('previous')) is not None:
            args['previous'] = previous
        if (next := _dict.get('next')) is not None:
            args['next'] = next
        if (assignments := _dict.get('assignments')) is not None:
            args['assignments'] = [TemplateAssignmentResponse.from_dict(v) for v in assignments]
        else:
            raise ValueError('Required property \'assignments\' not present in TemplateAssignmentListResponse JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TemplateAssignmentListResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'context') and self.context is not None:
            if isinstance(self.context, dict):
                _dict['context'] = self.context
            else:
                _dict['context'] = self.context.to_dict()
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'first') and self.first is not None:
            _dict['first'] = self.first
        if hasattr(self, 'previous') and self.previous is not None:
            _dict['previous'] = self.previous
        if hasattr(self, 'next') and self.next is not None:
            _dict['next'] = self.next
        if hasattr(self, 'assignments') and self.assignments is not None:
            assignments_list = []
            for v in self.assignments:
                if isinstance(v, dict):
                    assignments_list.append(v)
                else:
                    assignments_list.append(v.to_dict())
            _dict['assignments'] = assignments_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TemplateAssignmentListResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TemplateAssignmentListResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TemplateAssignmentListResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TemplateAssignmentResource:
    """
    Body parameters for created resource.

    :param str id: (optional) Id of the created resource.
    """

    def __init__(
        self,
        *,
        id: Optional[str] = None,
    ) -> None:
        """
        Initialize a TemplateAssignmentResource object.

        :param str id: (optional) Id of the created resource.
        """
        self.id = id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TemplateAssignmentResource':
        """Initialize a TemplateAssignmentResource object from a json dictionary."""
        args = {}
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TemplateAssignmentResource object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TemplateAssignmentResource object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TemplateAssignmentResource') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TemplateAssignmentResource') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TemplateAssignmentResourceError:
    """
    Body parameters for assignment error.

    :param str name: (optional) Name of the error.
    :param str error_code: (optional) Internal error code.
    :param str message: (optional) Error message detailing the nature of the error.
    :param str status_code: (optional) Internal status code for the error.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        error_code: Optional[str] = None,
        message: Optional[str] = None,
        status_code: Optional[str] = None,
    ) -> None:
        """
        Initialize a TemplateAssignmentResourceError object.

        :param str name: (optional) Name of the error.
        :param str error_code: (optional) Internal error code.
        :param str message: (optional) Error message detailing the nature of the
               error.
        :param str status_code: (optional) Internal status code for the error.
        """
        self.name = name
        self.error_code = error_code
        self.message = message
        self.status_code = status_code

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TemplateAssignmentResourceError':
        """Initialize a TemplateAssignmentResourceError object from a json dictionary."""
        args = {}
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        if (error_code := _dict.get('errorCode')) is not None:
            args['error_code'] = error_code
        if (message := _dict.get('message')) is not None:
            args['message'] = message
        if (status_code := _dict.get('statusCode')) is not None:
            args['status_code'] = status_code
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TemplateAssignmentResourceError object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'error_code') and self.error_code is not None:
            _dict['errorCode'] = self.error_code
        if hasattr(self, 'message') and self.message is not None:
            _dict['message'] = self.message
        if hasattr(self, 'status_code') and self.status_code is not None:
            _dict['statusCode'] = self.status_code
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TemplateAssignmentResourceError object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TemplateAssignmentResourceError') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TemplateAssignmentResourceError') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TemplateAssignmentResponse:
    """
    Response body format for Template Assignment Record.

    :param ResponseContext context: (optional) Context with key properties for
          problem determination.
    :param str id: Assignment record Id.
    :param str account_id: Enterprise account Id.
    :param str template_id: Template Id.
    :param int template_version: Template version.
    :param str target_type: Assignment target type.
    :param str target: Assignment target.
    :param str status: Assignment status.
    :param List[TemplateAssignmentResponseResource] resources: (optional) Status
          breakdown per target account of IAM resources created or errors encountered in
          attempting to create those IAM resources. IAM resources are only included in the
          response providing the assignment is not in progress. IAM resources are also
          only included when getting a single assignment, and excluded by list APIs.
    :param List[EnityHistoryRecord] history: (optional) Assignment history.
    :param str href: (optional) Href.
    :param str created_at: Assignment created at.
    :param str created_by_id: IAMid of the identity that created the assignment.
    :param str last_modified_at: Assignment modified at.
    :param str last_modified_by_id: IAMid of the identity that last modified the
          assignment.
    :param str entity_tag: Entity tag for this assignment record.
    """

    def __init__(
        self,
        id: str,
        account_id: str,
        template_id: str,
        template_version: int,
        target_type: str,
        target: str,
        status: str,
        created_at: str,
        created_by_id: str,
        last_modified_at: str,
        last_modified_by_id: str,
        entity_tag: str,
        *,
        context: Optional['ResponseContext'] = None,
        resources: Optional[List['TemplateAssignmentResponseResource']] = None,
        history: Optional[List['EnityHistoryRecord']] = None,
        href: Optional[str] = None,
    ) -> None:
        """
        Initialize a TemplateAssignmentResponse object.

        :param str id: Assignment record Id.
        :param str account_id: Enterprise account Id.
        :param str template_id: Template Id.
        :param int template_version: Template version.
        :param str target_type: Assignment target type.
        :param str target: Assignment target.
        :param str status: Assignment status.
        :param str created_at: Assignment created at.
        :param str created_by_id: IAMid of the identity that created the
               assignment.
        :param str last_modified_at: Assignment modified at.
        :param str last_modified_by_id: IAMid of the identity that last modified
               the assignment.
        :param str entity_tag: Entity tag for this assignment record.
        :param ResponseContext context: (optional) Context with key properties for
               problem determination.
        :param List[TemplateAssignmentResponseResource] resources: (optional)
               Status breakdown per target account of IAM resources created or errors
               encountered in attempting to create those IAM resources. IAM resources are
               only included in the response providing the assignment is not in progress.
               IAM resources are also only included when getting a single assignment, and
               excluded by list APIs.
        :param List[EnityHistoryRecord] history: (optional) Assignment history.
        :param str href: (optional) Href.
        """
        self.context = context
        self.id = id
        self.account_id = account_id
        self.template_id = template_id
        self.template_version = template_version
        self.target_type = target_type
        self.target = target
        self.status = status
        self.resources = resources
        self.history = history
        self.href = href
        self.created_at = created_at
        self.created_by_id = created_by_id
        self.last_modified_at = last_modified_at
        self.last_modified_by_id = last_modified_by_id
        self.entity_tag = entity_tag

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TemplateAssignmentResponse':
        """Initialize a TemplateAssignmentResponse object from a json dictionary."""
        args = {}
        if (context := _dict.get('context')) is not None:
            args['context'] = ResponseContext.from_dict(context)
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        else:
            raise ValueError('Required property \'id\' not present in TemplateAssignmentResponse JSON')
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        else:
            raise ValueError('Required property \'account_id\' not present in TemplateAssignmentResponse JSON')
        if (template_id := _dict.get('template_id')) is not None:
            args['template_id'] = template_id
        else:
            raise ValueError('Required property \'template_id\' not present in TemplateAssignmentResponse JSON')
        if (template_version := _dict.get('template_version')) is not None:
            args['template_version'] = template_version
        else:
            raise ValueError('Required property \'template_version\' not present in TemplateAssignmentResponse JSON')
        if (target_type := _dict.get('target_type')) is not None:
            args['target_type'] = target_type
        else:
            raise ValueError('Required property \'target_type\' not present in TemplateAssignmentResponse JSON')
        if (target := _dict.get('target')) is not None:
            args['target'] = target
        else:
            raise ValueError('Required property \'target\' not present in TemplateAssignmentResponse JSON')
        if (status := _dict.get('status')) is not None:
            args['status'] = status
        else:
            raise ValueError('Required property \'status\' not present in TemplateAssignmentResponse JSON')
        if (resources := _dict.get('resources')) is not None:
            args['resources'] = [TemplateAssignmentResponseResource.from_dict(v) for v in resources]
        if (history := _dict.get('history')) is not None:
            args['history'] = [EnityHistoryRecord.from_dict(v) for v in history]
        if (href := _dict.get('href')) is not None:
            args['href'] = href
        if (created_at := _dict.get('created_at')) is not None:
            args['created_at'] = created_at
        else:
            raise ValueError('Required property \'created_at\' not present in TemplateAssignmentResponse JSON')
        if (created_by_id := _dict.get('created_by_id')) is not None:
            args['created_by_id'] = created_by_id
        else:
            raise ValueError('Required property \'created_by_id\' not present in TemplateAssignmentResponse JSON')
        if (last_modified_at := _dict.get('last_modified_at')) is not None:
            args['last_modified_at'] = last_modified_at
        else:
            raise ValueError('Required property \'last_modified_at\' not present in TemplateAssignmentResponse JSON')
        if (last_modified_by_id := _dict.get('last_modified_by_id')) is not None:
            args['last_modified_by_id'] = last_modified_by_id
        else:
            raise ValueError('Required property \'last_modified_by_id\' not present in TemplateAssignmentResponse JSON')
        if (entity_tag := _dict.get('entity_tag')) is not None:
            args['entity_tag'] = entity_tag
        else:
            raise ValueError('Required property \'entity_tag\' not present in TemplateAssignmentResponse JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TemplateAssignmentResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'context') and self.context is not None:
            if isinstance(self.context, dict):
                _dict['context'] = self.context
            else:
                _dict['context'] = self.context.to_dict()
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'template_id') and self.template_id is not None:
            _dict['template_id'] = self.template_id
        if hasattr(self, 'template_version') and self.template_version is not None:
            _dict['template_version'] = self.template_version
        if hasattr(self, 'target_type') and self.target_type is not None:
            _dict['target_type'] = self.target_type
        if hasattr(self, 'target') and self.target is not None:
            _dict['target'] = self.target
        if hasattr(self, 'status') and self.status is not None:
            _dict['status'] = self.status
        if hasattr(self, 'resources') and self.resources is not None:
            resources_list = []
            for v in self.resources:
                if isinstance(v, dict):
                    resources_list.append(v)
                else:
                    resources_list.append(v.to_dict())
            _dict['resources'] = resources_list
        if hasattr(self, 'history') and self.history is not None:
            history_list = []
            for v in self.history:
                if isinstance(v, dict):
                    history_list.append(v)
                else:
                    history_list.append(v.to_dict())
            _dict['history'] = history_list
        if hasattr(self, 'href') and self.href is not None:
            _dict['href'] = self.href
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = self.created_at
        if hasattr(self, 'created_by_id') and self.created_by_id is not None:
            _dict['created_by_id'] = self.created_by_id
        if hasattr(self, 'last_modified_at') and self.last_modified_at is not None:
            _dict['last_modified_at'] = self.last_modified_at
        if hasattr(self, 'last_modified_by_id') and self.last_modified_by_id is not None:
            _dict['last_modified_by_id'] = self.last_modified_by_id
        if hasattr(self, 'entity_tag') and self.entity_tag is not None:
            _dict['entity_tag'] = self.entity_tag
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TemplateAssignmentResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TemplateAssignmentResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TemplateAssignmentResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TemplateAssignmentResponseResource:
    """
    Overview of resources assignment per target account.

    :param str target: Target account where the IAM resource is created.
    :param TemplateAssignmentResponseResourceDetail profile: (optional)
    :param TemplateAssignmentResponseResourceDetail account_settings: (optional)
    :param List[TemplateAssignmentResponseResourceDetail] policy_template_refs:
          (optional) Policy resource(s) included only for trusted profile assignments with
          policy references.
    """

    def __init__(
        self,
        target: str,
        *,
        profile: Optional['TemplateAssignmentResponseResourceDetail'] = None,
        account_settings: Optional['TemplateAssignmentResponseResourceDetail'] = None,
        policy_template_refs: Optional[List['TemplateAssignmentResponseResourceDetail']] = None,
    ) -> None:
        """
        Initialize a TemplateAssignmentResponseResource object.

        :param str target: Target account where the IAM resource is created.
        :param TemplateAssignmentResponseResourceDetail profile: (optional)
        :param TemplateAssignmentResponseResourceDetail account_settings:
               (optional)
        :param List[TemplateAssignmentResponseResourceDetail] policy_template_refs:
               (optional) Policy resource(s) included only for trusted profile assignments
               with policy references.
        """
        self.target = target
        self.profile = profile
        self.account_settings = account_settings
        self.policy_template_refs = policy_template_refs

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TemplateAssignmentResponseResource':
        """Initialize a TemplateAssignmentResponseResource object from a json dictionary."""
        args = {}
        if (target := _dict.get('target')) is not None:
            args['target'] = target
        else:
            raise ValueError('Required property \'target\' not present in TemplateAssignmentResponseResource JSON')
        if (profile := _dict.get('profile')) is not None:
            args['profile'] = TemplateAssignmentResponseResourceDetail.from_dict(profile)
        if (account_settings := _dict.get('account_settings')) is not None:
            args['account_settings'] = TemplateAssignmentResponseResourceDetail.from_dict(account_settings)
        if (policy_template_refs := _dict.get('policy_template_refs')) is not None:
            args['policy_template_refs'] = [
                TemplateAssignmentResponseResourceDetail.from_dict(v) for v in policy_template_refs
            ]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TemplateAssignmentResponseResource object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'target') and self.target is not None:
            _dict['target'] = self.target
        if hasattr(self, 'profile') and self.profile is not None:
            if isinstance(self.profile, dict):
                _dict['profile'] = self.profile
            else:
                _dict['profile'] = self.profile.to_dict()
        if hasattr(self, 'account_settings') and self.account_settings is not None:
            if isinstance(self.account_settings, dict):
                _dict['account_settings'] = self.account_settings
            else:
                _dict['account_settings'] = self.account_settings.to_dict()
        if hasattr(self, 'policy_template_refs') and self.policy_template_refs is not None:
            policy_template_refs_list = []
            for v in self.policy_template_refs:
                if isinstance(v, dict):
                    policy_template_refs_list.append(v)
                else:
                    policy_template_refs_list.append(v.to_dict())
            _dict['policy_template_refs'] = policy_template_refs_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TemplateAssignmentResponseResource object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TemplateAssignmentResponseResource') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TemplateAssignmentResponseResource') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TemplateAssignmentResponseResourceDetail:
    """
    TemplateAssignmentResponseResourceDetail.

    :param str id: (optional) Policy Template Id, only returned for a profile
          assignment with policy references.
    :param str version: (optional) Policy version, only returned for a profile
          assignment with policy references.
    :param TemplateAssignmentResource resource_created: (optional) Body parameters
          for created resource.
    :param TemplateAssignmentResourceError error_message: (optional) Body parameters
          for assignment error.
    :param str status: Status for the target account's assignment.
    """

    def __init__(
        self,
        status: str,
        *,
        id: Optional[str] = None,
        version: Optional[str] = None,
        resource_created: Optional['TemplateAssignmentResource'] = None,
        error_message: Optional['TemplateAssignmentResourceError'] = None,
    ) -> None:
        """
        Initialize a TemplateAssignmentResponseResourceDetail object.

        :param str status: Status for the target account's assignment.
        :param str id: (optional) Policy Template Id, only returned for a profile
               assignment with policy references.
        :param str version: (optional) Policy version, only returned for a profile
               assignment with policy references.
        :param TemplateAssignmentResource resource_created: (optional) Body
               parameters for created resource.
        :param TemplateAssignmentResourceError error_message: (optional) Body
               parameters for assignment error.
        """
        self.id = id
        self.version = version
        self.resource_created = resource_created
        self.error_message = error_message
        self.status = status

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TemplateAssignmentResponseResourceDetail':
        """Initialize a TemplateAssignmentResponseResourceDetail object from a json dictionary."""
        args = {}
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        if (version := _dict.get('version')) is not None:
            args['version'] = version
        if (resource_created := _dict.get('resource_created')) is not None:
            args['resource_created'] = TemplateAssignmentResource.from_dict(resource_created)
        if (error_message := _dict.get('error_message')) is not None:
            args['error_message'] = TemplateAssignmentResourceError.from_dict(error_message)
        if (status := _dict.get('status')) is not None:
            args['status'] = status
        else:
            raise ValueError(
                'Required property \'status\' not present in TemplateAssignmentResponseResourceDetail JSON'
            )
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TemplateAssignmentResponseResourceDetail object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'version') and self.version is not None:
            _dict['version'] = self.version
        if hasattr(self, 'resource_created') and self.resource_created is not None:
            if isinstance(self.resource_created, dict):
                _dict['resource_created'] = self.resource_created
            else:
                _dict['resource_created'] = self.resource_created.to_dict()
        if hasattr(self, 'error_message') and self.error_message is not None:
            if isinstance(self.error_message, dict):
                _dict['error_message'] = self.error_message
            else:
                _dict['error_message'] = self.error_message.to_dict()
        if hasattr(self, 'status') and self.status is not None:
            _dict['status'] = self.status
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TemplateAssignmentResponseResourceDetail object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TemplateAssignmentResponseResourceDetail') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TemplateAssignmentResponseResourceDetail') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TemplateProfileComponentRequest:
    """
    Input body parameters for the TemplateProfileComponent.

    :param str name: Name of the Profile.
    :param str description: (optional) Description of the Profile.
    :param List[TrustedProfileTemplateClaimRule] rules: (optional) Rules for the
          Profile.
    :param List[ProfileIdentityRequest] identities: (optional) Identities for the
          Profile.
    """

    def __init__(
        self,
        name: str,
        *,
        description: Optional[str] = None,
        rules: Optional[List['TrustedProfileTemplateClaimRule']] = None,
        identities: Optional[List['ProfileIdentityRequest']] = None,
    ) -> None:
        """
        Initialize a TemplateProfileComponentRequest object.

        :param str name: Name of the Profile.
        :param str description: (optional) Description of the Profile.
        :param List[TrustedProfileTemplateClaimRule] rules: (optional) Rules for
               the Profile.
        :param List[ProfileIdentityRequest] identities: (optional) Identities for
               the Profile.
        """
        self.name = name
        self.description = description
        self.rules = rules
        self.identities = identities

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TemplateProfileComponentRequest':
        """Initialize a TemplateProfileComponentRequest object from a json dictionary."""
        args = {}
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        else:
            raise ValueError('Required property \'name\' not present in TemplateProfileComponentRequest JSON')
        if (description := _dict.get('description')) is not None:
            args['description'] = description
        if (rules := _dict.get('rules')) is not None:
            args['rules'] = [TrustedProfileTemplateClaimRule.from_dict(v) for v in rules]
        if (identities := _dict.get('identities')) is not None:
            args['identities'] = [ProfileIdentityRequest.from_dict(v) for v in identities]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TemplateProfileComponentRequest object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'rules') and self.rules is not None:
            rules_list = []
            for v in self.rules:
                if isinstance(v, dict):
                    rules_list.append(v)
                else:
                    rules_list.append(v.to_dict())
            _dict['rules'] = rules_list
        if hasattr(self, 'identities') and self.identities is not None:
            identities_list = []
            for v in self.identities:
                if isinstance(v, dict):
                    identities_list.append(v)
                else:
                    identities_list.append(v.to_dict())
            _dict['identities'] = identities_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TemplateProfileComponentRequest object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TemplateProfileComponentRequest') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TemplateProfileComponentRequest') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TemplateProfileComponentResponse:
    """
    Input body parameters for the TemplateProfileComponent.

    :param str name: Name of the Profile.
    :param str description: (optional) Description of the Profile.
    :param List[TrustedProfileTemplateClaimRule] rules: (optional) Rules for the
          Profile.
    :param List[ProfileIdentityResponse] identities: (optional) Identities for the
          Profile.
    """

    def __init__(
        self,
        name: str,
        *,
        description: Optional[str] = None,
        rules: Optional[List['TrustedProfileTemplateClaimRule']] = None,
        identities: Optional[List['ProfileIdentityResponse']] = None,
    ) -> None:
        """
        Initialize a TemplateProfileComponentResponse object.

        :param str name: Name of the Profile.
        :param str description: (optional) Description of the Profile.
        :param List[TrustedProfileTemplateClaimRule] rules: (optional) Rules for
               the Profile.
        :param List[ProfileIdentityResponse] identities: (optional) Identities for
               the Profile.
        """
        self.name = name
        self.description = description
        self.rules = rules
        self.identities = identities

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TemplateProfileComponentResponse':
        """Initialize a TemplateProfileComponentResponse object from a json dictionary."""
        args = {}
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        else:
            raise ValueError('Required property \'name\' not present in TemplateProfileComponentResponse JSON')
        if (description := _dict.get('description')) is not None:
            args['description'] = description
        if (rules := _dict.get('rules')) is not None:
            args['rules'] = [TrustedProfileTemplateClaimRule.from_dict(v) for v in rules]
        if (identities := _dict.get('identities')) is not None:
            args['identities'] = [ProfileIdentityResponse.from_dict(v) for v in identities]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TemplateProfileComponentResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'rules') and self.rules is not None:
            rules_list = []
            for v in self.rules:
                if isinstance(v, dict):
                    rules_list.append(v)
                else:
                    rules_list.append(v.to_dict())
            _dict['rules'] = rules_list
        if hasattr(self, 'identities') and self.identities is not None:
            identities_list = []
            for v in self.identities:
                if isinstance(v, dict):
                    identities_list.append(v)
                else:
                    identities_list.append(v.to_dict())
            _dict['identities'] = identities_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TemplateProfileComponentResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TemplateProfileComponentResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TemplateProfileComponentResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TrustedProfile:
    """
    Response body format for trusted profile V1 REST requests.

    :param ResponseContext context: (optional) Context with key properties for
          problem determination.
    :param str id: the unique identifier of the trusted profile.
          Example:'Profile-94497d0d-2ac3-41bf-a993-a49d1b14627c'.
    :param str entity_tag: Version of the trusted profile details object. You need
          to specify this value when updating the trusted profile to avoid stale updates.
    :param str crn: Cloud Resource Name of the item. Example Cloud Resource Name:
          'crn:v1:bluemix:public:iam-identity:us-south:a/myaccount::profile:Profile-94497d0d-2ac3-41bf-a993-a49d1b14627c'.
    :param str name: Name of the trusted profile. The name is checked for
          uniqueness. Therefore trusted profiles with the same names can not exist in the
          same account.
    :param str description: (optional) The optional description of the trusted
          profile. The 'description' property is only available if a description was
          provided during a create of a trusted profile.
    :param datetime created_at: (optional) If set contains a date time string of the
          creation date in ISO format.
    :param datetime modified_at: (optional) If set contains a date time string of
          the last modification date in ISO format.
    :param str iam_id: The iam_id of this trusted profile.
    :param str account_id: ID of the account that this trusted profile belong to.
    :param str template_id: (optional) ID of the IAM template that was used to
          create an enterprise-managed trusted profile in your account. When returned,
          this indicates that the trusted profile is created from and managed by a
          template in the root enterprise account.
    :param str assignment_id: (optional) ID of the assignment that was used to
          create an enterprise-managed trusted profile in your account. When returned,
          this indicates that the trusted profile is created from and managed by a
          template in the root enterprise account.
    :param int ims_account_id: (optional) IMS acount ID of the trusted profile.
    :param int ims_user_id: (optional) IMS user ID of the trusted profile.
    :param List[EnityHistoryRecord] history: (optional) History of the trusted
          profile.
    :param Activity activity: (optional)
    """

    def __init__(
        self,
        id: str,
        entity_tag: str,
        crn: str,
        name: str,
        iam_id: str,
        account_id: str,
        *,
        context: Optional['ResponseContext'] = None,
        description: Optional[str] = None,
        created_at: Optional[datetime] = None,
        modified_at: Optional[datetime] = None,
        template_id: Optional[str] = None,
        assignment_id: Optional[str] = None,
        ims_account_id: Optional[int] = None,
        ims_user_id: Optional[int] = None,
        history: Optional[List['EnityHistoryRecord']] = None,
        activity: Optional['Activity'] = None,
    ) -> None:
        """
        Initialize a TrustedProfile object.

        :param str id: the unique identifier of the trusted profile.
               Example:'Profile-94497d0d-2ac3-41bf-a993-a49d1b14627c'.
        :param str entity_tag: Version of the trusted profile details object. You
               need to specify this value when updating the trusted profile to avoid stale
               updates.
        :param str crn: Cloud Resource Name of the item. Example Cloud Resource
               Name:
               'crn:v1:bluemix:public:iam-identity:us-south:a/myaccount::profile:Profile-94497d0d-2ac3-41bf-a993-a49d1b14627c'.
        :param str name: Name of the trusted profile. The name is checked for
               uniqueness. Therefore trusted profiles with the same names can not exist in
               the same account.
        :param str iam_id: The iam_id of this trusted profile.
        :param str account_id: ID of the account that this trusted profile belong
               to.
        :param ResponseContext context: (optional) Context with key properties for
               problem determination.
        :param str description: (optional) The optional description of the trusted
               profile. The 'description' property is only available if a description was
               provided during a create of a trusted profile.
        :param datetime created_at: (optional) If set contains a date time string
               of the creation date in ISO format.
        :param datetime modified_at: (optional) If set contains a date time string
               of the last modification date in ISO format.
        :param str template_id: (optional) ID of the IAM template that was used to
               create an enterprise-managed trusted profile in your account. When
               returned, this indicates that the trusted profile is created from and
               managed by a template in the root enterprise account.
        :param str assignment_id: (optional) ID of the assignment that was used to
               create an enterprise-managed trusted profile in your account. When
               returned, this indicates that the trusted profile is created from and
               managed by a template in the root enterprise account.
        :param int ims_account_id: (optional) IMS acount ID of the trusted profile.
        :param int ims_user_id: (optional) IMS user ID of the trusted profile.
        :param List[EnityHistoryRecord] history: (optional) History of the trusted
               profile.
        :param Activity activity: (optional)
        """
        self.context = context
        self.id = id
        self.entity_tag = entity_tag
        self.crn = crn
        self.name = name
        self.description = description
        self.created_at = created_at
        self.modified_at = modified_at
        self.iam_id = iam_id
        self.account_id = account_id
        self.template_id = template_id
        self.assignment_id = assignment_id
        self.ims_account_id = ims_account_id
        self.ims_user_id = ims_user_id
        self.history = history
        self.activity = activity

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TrustedProfile':
        """Initialize a TrustedProfile object from a json dictionary."""
        args = {}
        if (context := _dict.get('context')) is not None:
            args['context'] = ResponseContext.from_dict(context)
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        else:
            raise ValueError('Required property \'id\' not present in TrustedProfile JSON')
        if (entity_tag := _dict.get('entity_tag')) is not None:
            args['entity_tag'] = entity_tag
        else:
            raise ValueError('Required property \'entity_tag\' not present in TrustedProfile JSON')
        if (crn := _dict.get('crn')) is not None:
            args['crn'] = crn
        else:
            raise ValueError('Required property \'crn\' not present in TrustedProfile JSON')
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        else:
            raise ValueError('Required property \'name\' not present in TrustedProfile JSON')
        if (description := _dict.get('description')) is not None:
            args['description'] = description
        if (created_at := _dict.get('created_at')) is not None:
            args['created_at'] = string_to_datetime(created_at)
        if (modified_at := _dict.get('modified_at')) is not None:
            args['modified_at'] = string_to_datetime(modified_at)
        if (iam_id := _dict.get('iam_id')) is not None:
            args['iam_id'] = iam_id
        else:
            raise ValueError('Required property \'iam_id\' not present in TrustedProfile JSON')
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        else:
            raise ValueError('Required property \'account_id\' not present in TrustedProfile JSON')
        if (template_id := _dict.get('template_id')) is not None:
            args['template_id'] = template_id
        if (assignment_id := _dict.get('assignment_id')) is not None:
            args['assignment_id'] = assignment_id
        if (ims_account_id := _dict.get('ims_account_id')) is not None:
            args['ims_account_id'] = ims_account_id
        if (ims_user_id := _dict.get('ims_user_id')) is not None:
            args['ims_user_id'] = ims_user_id
        if (history := _dict.get('history')) is not None:
            args['history'] = [EnityHistoryRecord.from_dict(v) for v in history]
        if (activity := _dict.get('activity')) is not None:
            args['activity'] = Activity.from_dict(activity)
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TrustedProfile object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'context') and self.context is not None:
            if isinstance(self.context, dict):
                _dict['context'] = self.context
            else:
                _dict['context'] = self.context.to_dict()
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'entity_tag') and self.entity_tag is not None:
            _dict['entity_tag'] = self.entity_tag
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'modified_at') and self.modified_at is not None:
            _dict['modified_at'] = datetime_to_string(self.modified_at)
        if hasattr(self, 'iam_id') and self.iam_id is not None:
            _dict['iam_id'] = self.iam_id
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'template_id') and self.template_id is not None:
            _dict['template_id'] = self.template_id
        if hasattr(self, 'assignment_id') and self.assignment_id is not None:
            _dict['assignment_id'] = self.assignment_id
        if hasattr(self, 'ims_account_id') and self.ims_account_id is not None:
            _dict['ims_account_id'] = self.ims_account_id
        if hasattr(self, 'ims_user_id') and self.ims_user_id is not None:
            _dict['ims_user_id'] = self.ims_user_id
        if hasattr(self, 'history') and self.history is not None:
            history_list = []
            for v in self.history:
                if isinstance(v, dict):
                    history_list.append(v)
                else:
                    history_list.append(v.to_dict())
            _dict['history'] = history_list
        if hasattr(self, 'activity') and self.activity is not None:
            if isinstance(self.activity, dict):
                _dict['activity'] = self.activity
            else:
                _dict['activity'] = self.activity.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TrustedProfile object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TrustedProfile') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TrustedProfile') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TrustedProfileTemplateClaimRule:
    """
    TrustedProfileTemplateClaimRule.

    :param str name: (optional) Name of the claim rule to be created or updated.
    :param str type: Type of the claim rule.
    :param str realm_name: (optional) The realm name of the Idp this claim rule
          applies to. This field is required only if the type is specified as
          'Profile-SAML'.
    :param int expiration: (optional) Session expiration in seconds, only required
          if type is 'Profile-SAML'.
    :param List[ProfileClaimRuleConditions] conditions: Conditions of this claim
          rule.
    """

    def __init__(
        self,
        type: str,
        conditions: List['ProfileClaimRuleConditions'],
        *,
        name: Optional[str] = None,
        realm_name: Optional[str] = None,
        expiration: Optional[int] = None,
    ) -> None:
        """
        Initialize a TrustedProfileTemplateClaimRule object.

        :param str type: Type of the claim rule.
        :param List[ProfileClaimRuleConditions] conditions: Conditions of this
               claim rule.
        :param str name: (optional) Name of the claim rule to be created or
               updated.
        :param str realm_name: (optional) The realm name of the Idp this claim rule
               applies to. This field is required only if the type is specified as
               'Profile-SAML'.
        :param int expiration: (optional) Session expiration in seconds, only
               required if type is 'Profile-SAML'.
        """
        self.name = name
        self.type = type
        self.realm_name = realm_name
        self.expiration = expiration
        self.conditions = conditions

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TrustedProfileTemplateClaimRule':
        """Initialize a TrustedProfileTemplateClaimRule object from a json dictionary."""
        args = {}
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        if (type := _dict.get('type')) is not None:
            args['type'] = type
        else:
            raise ValueError('Required property \'type\' not present in TrustedProfileTemplateClaimRule JSON')
        if (realm_name := _dict.get('realm_name')) is not None:
            args['realm_name'] = realm_name
        if (expiration := _dict.get('expiration')) is not None:
            args['expiration'] = expiration
        if (conditions := _dict.get('conditions')) is not None:
            args['conditions'] = [ProfileClaimRuleConditions.from_dict(v) for v in conditions]
        else:
            raise ValueError('Required property \'conditions\' not present in TrustedProfileTemplateClaimRule JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TrustedProfileTemplateClaimRule object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'realm_name') and self.realm_name is not None:
            _dict['realm_name'] = self.realm_name
        if hasattr(self, 'expiration') and self.expiration is not None:
            _dict['expiration'] = self.expiration
        if hasattr(self, 'conditions') and self.conditions is not None:
            conditions_list = []
            for v in self.conditions:
                if isinstance(v, dict):
                    conditions_list.append(v)
                else:
                    conditions_list.append(v.to_dict())
            _dict['conditions'] = conditions_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TrustedProfileTemplateClaimRule object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TrustedProfileTemplateClaimRule') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TrustedProfileTemplateClaimRule') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class TypeEnum(str, Enum):
        """
        Type of the claim rule.
        """

        PROFILE_SAML = 'Profile-SAML'


class TrustedProfileTemplateList:
    """
    TrustedProfileTemplateList.

    :param ResponseContext context: (optional) Context with key properties for
          problem determination.
    :param int offset: (optional) The offset of the current page.
    :param int limit: (optional) Optional size of a single page.
    :param str first: (optional) Link to the first page.
    :param str previous: (optional) Link to the previous available page. If
          'previous' property is not part of the response no previous page is available.
    :param str next: (optional) Link to the next available page. If 'next' property
          is not part of the response no next page is available.
    :param List[TrustedProfileTemplateResponse] profile_templates: List of Profile
          Templates based on the query paramters and the page size. The profile_templates
          array is always part of the response but might be empty depending on the query
          parameter values provided.
    """

    def __init__(
        self,
        profile_templates: List['TrustedProfileTemplateResponse'],
        *,
        context: Optional['ResponseContext'] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        first: Optional[str] = None,
        previous: Optional[str] = None,
        next: Optional[str] = None,
    ) -> None:
        """
        Initialize a TrustedProfileTemplateList object.

        :param List[TrustedProfileTemplateResponse] profile_templates: List of
               Profile Templates based on the query paramters and the page size. The
               profile_templates array is always part of the response but might be empty
               depending on the query parameter values provided.
        :param ResponseContext context: (optional) Context with key properties for
               problem determination.
        :param int offset: (optional) The offset of the current page.
        :param int limit: (optional) Optional size of a single page.
        :param str first: (optional) Link to the first page.
        :param str previous: (optional) Link to the previous available page. If
               'previous' property is not part of the response no previous page is
               available.
        :param str next: (optional) Link to the next available page. If 'next'
               property is not part of the response no next page is available.
        """
        self.context = context
        self.offset = offset
        self.limit = limit
        self.first = first
        self.previous = previous
        self.next = next
        self.profile_templates = profile_templates

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TrustedProfileTemplateList':
        """Initialize a TrustedProfileTemplateList object from a json dictionary."""
        args = {}
        if (context := _dict.get('context')) is not None:
            args['context'] = ResponseContext.from_dict(context)
        if (offset := _dict.get('offset')) is not None:
            args['offset'] = offset
        if (limit := _dict.get('limit')) is not None:
            args['limit'] = limit
        if (first := _dict.get('first')) is not None:
            args['first'] = first
        if (previous := _dict.get('previous')) is not None:
            args['previous'] = previous
        if (next := _dict.get('next')) is not None:
            args['next'] = next
        if (profile_templates := _dict.get('profile_templates')) is not None:
            args['profile_templates'] = [TrustedProfileTemplateResponse.from_dict(v) for v in profile_templates]
        else:
            raise ValueError('Required property \'profile_templates\' not present in TrustedProfileTemplateList JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TrustedProfileTemplateList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'context') and self.context is not None:
            if isinstance(self.context, dict):
                _dict['context'] = self.context
            else:
                _dict['context'] = self.context.to_dict()
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'first') and self.first is not None:
            _dict['first'] = self.first
        if hasattr(self, 'previous') and self.previous is not None:
            _dict['previous'] = self.previous
        if hasattr(self, 'next') and self.next is not None:
            _dict['next'] = self.next
        if hasattr(self, 'profile_templates') and self.profile_templates is not None:
            profile_templates_list = []
            for v in self.profile_templates:
                if isinstance(v, dict):
                    profile_templates_list.append(v)
                else:
                    profile_templates_list.append(v.to_dict())
            _dict['profile_templates'] = profile_templates_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TrustedProfileTemplateList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TrustedProfileTemplateList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TrustedProfileTemplateList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TrustedProfileTemplateResponse:
    """
    Response body format for Trusted Profile Template REST requests.

    :param str id: ID of the the template.
    :param int version: Version of the the template.
    :param str account_id: ID of the account where the template resides.
    :param str name: The name of the trusted profile template. This is visible only
          in the enterprise account.
    :param str description: (optional) The description of the trusted profile
          template. Describe the template for enterprise account users.
    :param bool committed: (optional) Committed flag determines if the template is
          ready for assignment.
    :param TemplateProfileComponentResponse profile: (optional) Input body
          parameters for the TemplateProfileComponent.
    :param List[PolicyTemplateReference] policy_template_references: (optional)
          Existing policy templates that you can reference to assign access in the trusted
          profile component.
    :param List[EnityHistoryRecord] history: (optional) History of the trusted
          profile template.
    :param str entity_tag: (optional) Entity tag for this templateId-version
          combination.
    :param str crn: (optional) Cloud resource name.
    :param str created_at: (optional) Timestamp of when the template was created.
    :param str created_by_id: (optional) IAMid of the creator.
    :param str last_modified_at: (optional) Timestamp of when the template was last
          modified.
    :param str last_modified_by_id: (optional) IAMid of the identity that made the
          latest modification.
    """

    def __init__(
        self,
        id: str,
        version: int,
        account_id: str,
        name: str,
        *,
        description: Optional[str] = None,
        committed: Optional[bool] = None,
        profile: Optional['TemplateProfileComponentResponse'] = None,
        policy_template_references: Optional[List['PolicyTemplateReference']] = None,
        history: Optional[List['EnityHistoryRecord']] = None,
        entity_tag: Optional[str] = None,
        crn: Optional[str] = None,
        created_at: Optional[str] = None,
        created_by_id: Optional[str] = None,
        last_modified_at: Optional[str] = None,
        last_modified_by_id: Optional[str] = None,
    ) -> None:
        """
        Initialize a TrustedProfileTemplateResponse object.

        :param str id: ID of the the template.
        :param int version: Version of the the template.
        :param str account_id: ID of the account where the template resides.
        :param str name: The name of the trusted profile template. This is visible
               only in the enterprise account.
        :param str description: (optional) The description of the trusted profile
               template. Describe the template for enterprise account users.
        :param bool committed: (optional) Committed flag determines if the template
               is ready for assignment.
        :param TemplateProfileComponentResponse profile: (optional) Input body
               parameters for the TemplateProfileComponent.
        :param List[PolicyTemplateReference] policy_template_references: (optional)
               Existing policy templates that you can reference to assign access in the
               trusted profile component.
        :param List[EnityHistoryRecord] history: (optional) History of the trusted
               profile template.
        :param str entity_tag: (optional) Entity tag for this templateId-version
               combination.
        :param str crn: (optional) Cloud resource name.
        :param str created_at: (optional) Timestamp of when the template was
               created.
        :param str created_by_id: (optional) IAMid of the creator.
        :param str last_modified_at: (optional) Timestamp of when the template was
               last modified.
        :param str last_modified_by_id: (optional) IAMid of the identity that made
               the latest modification.
        """
        self.id = id
        self.version = version
        self.account_id = account_id
        self.name = name
        self.description = description
        self.committed = committed
        self.profile = profile
        self.policy_template_references = policy_template_references
        self.history = history
        self.entity_tag = entity_tag
        self.crn = crn
        self.created_at = created_at
        self.created_by_id = created_by_id
        self.last_modified_at = last_modified_at
        self.last_modified_by_id = last_modified_by_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TrustedProfileTemplateResponse':
        """Initialize a TrustedProfileTemplateResponse object from a json dictionary."""
        args = {}
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        else:
            raise ValueError('Required property \'id\' not present in TrustedProfileTemplateResponse JSON')
        if (version := _dict.get('version')) is not None:
            args['version'] = version
        else:
            raise ValueError('Required property \'version\' not present in TrustedProfileTemplateResponse JSON')
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        else:
            raise ValueError('Required property \'account_id\' not present in TrustedProfileTemplateResponse JSON')
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        else:
            raise ValueError('Required property \'name\' not present in TrustedProfileTemplateResponse JSON')
        if (description := _dict.get('description')) is not None:
            args['description'] = description
        if (committed := _dict.get('committed')) is not None:
            args['committed'] = committed
        if (profile := _dict.get('profile')) is not None:
            args['profile'] = TemplateProfileComponentResponse.from_dict(profile)
        if (policy_template_references := _dict.get('policy_template_references')) is not None:
            args['policy_template_references'] = [
                PolicyTemplateReference.from_dict(v) for v in policy_template_references
            ]
        if (history := _dict.get('history')) is not None:
            args['history'] = [EnityHistoryRecord.from_dict(v) for v in history]
        if (entity_tag := _dict.get('entity_tag')) is not None:
            args['entity_tag'] = entity_tag
        if (crn := _dict.get('crn')) is not None:
            args['crn'] = crn
        if (created_at := _dict.get('created_at')) is not None:
            args['created_at'] = created_at
        if (created_by_id := _dict.get('created_by_id')) is not None:
            args['created_by_id'] = created_by_id
        if (last_modified_at := _dict.get('last_modified_at')) is not None:
            args['last_modified_at'] = last_modified_at
        if (last_modified_by_id := _dict.get('last_modified_by_id')) is not None:
            args['last_modified_by_id'] = last_modified_by_id
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TrustedProfileTemplateResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'version') and self.version is not None:
            _dict['version'] = self.version
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'committed') and self.committed is not None:
            _dict['committed'] = self.committed
        if hasattr(self, 'profile') and self.profile is not None:
            if isinstance(self.profile, dict):
                _dict['profile'] = self.profile
            else:
                _dict['profile'] = self.profile.to_dict()
        if hasattr(self, 'policy_template_references') and self.policy_template_references is not None:
            policy_template_references_list = []
            for v in self.policy_template_references:
                if isinstance(v, dict):
                    policy_template_references_list.append(v)
                else:
                    policy_template_references_list.append(v.to_dict())
            _dict['policy_template_references'] = policy_template_references_list
        if hasattr(self, 'history') and self.history is not None:
            history_list = []
            for v in self.history:
                if isinstance(v, dict):
                    history_list.append(v)
                else:
                    history_list.append(v.to_dict())
            _dict['history'] = history_list
        if hasattr(self, 'entity_tag') and self.entity_tag is not None:
            _dict['entity_tag'] = self.entity_tag
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = self.created_at
        if hasattr(self, 'created_by_id') and self.created_by_id is not None:
            _dict['created_by_id'] = self.created_by_id
        if hasattr(self, 'last_modified_at') and self.last_modified_at is not None:
            _dict['last_modified_at'] = self.last_modified_at
        if hasattr(self, 'last_modified_by_id') and self.last_modified_by_id is not None:
            _dict['last_modified_by_id'] = self.last_modified_by_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TrustedProfileTemplateResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TrustedProfileTemplateResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TrustedProfileTemplateResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TrustedProfilesList:
    """
    Response body format for the List trusted profiles V1 REST request.

    :param ResponseContext context: (optional) Context with key properties for
          problem determination.
    :param int offset: (optional) The offset of the current page.
    :param int limit: (optional) Optional size of a single page. Default is 20 items
          per page. Valid range is 1 to 100.
    :param str first: (optional) Link to the first page.
    :param str previous: (optional) Link to the previous available page. If
          'previous' property is not part of the response no previous page is available.
    :param str next: (optional) Link to the next available page. If 'next' property
          is not part of the response no next page is available.
    :param List[TrustedProfile] profiles: List of trusted profiles.
    """

    def __init__(
        self,
        profiles: List['TrustedProfile'],
        *,
        context: Optional['ResponseContext'] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        first: Optional[str] = None,
        previous: Optional[str] = None,
        next: Optional[str] = None,
    ) -> None:
        """
        Initialize a TrustedProfilesList object.

        :param List[TrustedProfile] profiles: List of trusted profiles.
        :param ResponseContext context: (optional) Context with key properties for
               problem determination.
        :param int offset: (optional) The offset of the current page.
        :param int limit: (optional) Optional size of a single page. Default is 20
               items per page. Valid range is 1 to 100.
        :param str first: (optional) Link to the first page.
        :param str previous: (optional) Link to the previous available page. If
               'previous' property is not part of the response no previous page is
               available.
        :param str next: (optional) Link to the next available page. If 'next'
               property is not part of the response no next page is available.
        """
        self.context = context
        self.offset = offset
        self.limit = limit
        self.first = first
        self.previous = previous
        self.next = next
        self.profiles = profiles

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TrustedProfilesList':
        """Initialize a TrustedProfilesList object from a json dictionary."""
        args = {}
        if (context := _dict.get('context')) is not None:
            args['context'] = ResponseContext.from_dict(context)
        if (offset := _dict.get('offset')) is not None:
            args['offset'] = offset
        if (limit := _dict.get('limit')) is not None:
            args['limit'] = limit
        if (first := _dict.get('first')) is not None:
            args['first'] = first
        if (previous := _dict.get('previous')) is not None:
            args['previous'] = previous
        if (next := _dict.get('next')) is not None:
            args['next'] = next
        if (profiles := _dict.get('profiles')) is not None:
            args['profiles'] = [TrustedProfile.from_dict(v) for v in profiles]
        else:
            raise ValueError('Required property \'profiles\' not present in TrustedProfilesList JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TrustedProfilesList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'context') and self.context is not None:
            if isinstance(self.context, dict):
                _dict['context'] = self.context
            else:
                _dict['context'] = self.context.to_dict()
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'first') and self.first is not None:
            _dict['first'] = self.first
        if hasattr(self, 'previous') and self.previous is not None:
            _dict['previous'] = self.previous
        if hasattr(self, 'next') and self.next is not None:
            _dict['next'] = self.next
        if hasattr(self, 'profiles') and self.profiles is not None:
            profiles_list = []
            for v in self.profiles:
                if isinstance(v, dict):
                    profiles_list.append(v)
                else:
                    profiles_list.append(v.to_dict())
            _dict['profiles'] = profiles_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TrustedProfilesList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TrustedProfilesList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TrustedProfilesList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class UserActivity:
    """
    UserActivity.

    :param str iam_id: IAMid of the user.
    :param str name: (optional) Name of the user.
    :param str username: Username of the user.
    :param str email: (optional) Email of the user.
    :param str last_authn: (optional) Time when the user was last authenticated.
    """

    def __init__(
        self,
        iam_id: str,
        username: str,
        *,
        name: Optional[str] = None,
        email: Optional[str] = None,
        last_authn: Optional[str] = None,
    ) -> None:
        """
        Initialize a UserActivity object.

        :param str iam_id: IAMid of the user.
        :param str username: Username of the user.
        :param str name: (optional) Name of the user.
        :param str email: (optional) Email of the user.
        :param str last_authn: (optional) Time when the user was last
               authenticated.
        """
        self.iam_id = iam_id
        self.name = name
        self.username = username
        self.email = email
        self.last_authn = last_authn

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'UserActivity':
        """Initialize a UserActivity object from a json dictionary."""
        args = {}
        if (iam_id := _dict.get('iam_id')) is not None:
            args['iam_id'] = iam_id
        else:
            raise ValueError('Required property \'iam_id\' not present in UserActivity JSON')
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        if (username := _dict.get('username')) is not None:
            args['username'] = username
        else:
            raise ValueError('Required property \'username\' not present in UserActivity JSON')
        if (email := _dict.get('email')) is not None:
            args['email'] = email
        if (last_authn := _dict.get('last_authn')) is not None:
            args['last_authn'] = last_authn
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a UserActivity object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'iam_id') and self.iam_id is not None:
            _dict['iam_id'] = self.iam_id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'username') and self.username is not None:
            _dict['username'] = self.username
        if hasattr(self, 'email') and self.email is not None:
            _dict['email'] = self.email
        if hasattr(self, 'last_authn') and self.last_authn is not None:
            _dict['last_authn'] = self.last_authn
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this UserActivity object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'UserActivity') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'UserActivity') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class UserMfaEnrollments:
    """
    UserMfaEnrollments.

    :param str iam_id: IAMid of the user.
    :param str effective_mfa_type: (optional) currently effective mfa type i.e.
          id_based_mfa or account_based_mfa.
    :param IdBasedMfaEnrollment id_based_mfa: (optional)
    :param AccountBasedMfaEnrollment account_based_mfa: (optional)
    """

    def __init__(
        self,
        iam_id: str,
        *,
        effective_mfa_type: Optional[str] = None,
        id_based_mfa: Optional['IdBasedMfaEnrollment'] = None,
        account_based_mfa: Optional['AccountBasedMfaEnrollment'] = None,
    ) -> None:
        """
        Initialize a UserMfaEnrollments object.

        :param str iam_id: IAMid of the user.
        :param str effective_mfa_type: (optional) currently effective mfa type i.e.
               id_based_mfa or account_based_mfa.
        :param IdBasedMfaEnrollment id_based_mfa: (optional)
        :param AccountBasedMfaEnrollment account_based_mfa: (optional)
        """
        self.iam_id = iam_id
        self.effective_mfa_type = effective_mfa_type
        self.id_based_mfa = id_based_mfa
        self.account_based_mfa = account_based_mfa

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'UserMfaEnrollments':
        """Initialize a UserMfaEnrollments object from a json dictionary."""
        args = {}
        if (iam_id := _dict.get('iam_id')) is not None:
            args['iam_id'] = iam_id
        else:
            raise ValueError('Required property \'iam_id\' not present in UserMfaEnrollments JSON')
        if (effective_mfa_type := _dict.get('effective_mfa_type')) is not None:
            args['effective_mfa_type'] = effective_mfa_type
        if (id_based_mfa := _dict.get('id_based_mfa')) is not None:
            args['id_based_mfa'] = IdBasedMfaEnrollment.from_dict(id_based_mfa)
        if (account_based_mfa := _dict.get('account_based_mfa')) is not None:
            args['account_based_mfa'] = AccountBasedMfaEnrollment.from_dict(account_based_mfa)
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a UserMfaEnrollments object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'iam_id') and self.iam_id is not None:
            _dict['iam_id'] = self.iam_id
        if hasattr(self, 'effective_mfa_type') and self.effective_mfa_type is not None:
            _dict['effective_mfa_type'] = self.effective_mfa_type
        if hasattr(self, 'id_based_mfa') and self.id_based_mfa is not None:
            if isinstance(self.id_based_mfa, dict):
                _dict['id_based_mfa'] = self.id_based_mfa
            else:
                _dict['id_based_mfa'] = self.id_based_mfa.to_dict()
        if hasattr(self, 'account_based_mfa') and self.account_based_mfa is not None:
            if isinstance(self.account_based_mfa, dict):
                _dict['account_based_mfa'] = self.account_based_mfa
            else:
                _dict['account_based_mfa'] = self.account_based_mfa.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this UserMfaEnrollments object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'UserMfaEnrollments') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'UserMfaEnrollments') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class UserReportMfaEnrollmentStatus:
    """
    UserReportMfaEnrollmentStatus.

    :param str iam_id: IAMid of the user.
    :param str name: (optional) Name of the user.
    :param str username: Username of the user.
    :param str email: (optional) Email of the user.
    :param MfaEnrollments enrollments:
    """

    def __init__(
        self,
        iam_id: str,
        username: str,
        enrollments: 'MfaEnrollments',
        *,
        name: Optional[str] = None,
        email: Optional[str] = None,
    ) -> None:
        """
        Initialize a UserReportMfaEnrollmentStatus object.

        :param str iam_id: IAMid of the user.
        :param str username: Username of the user.
        :param MfaEnrollments enrollments:
        :param str name: (optional) Name of the user.
        :param str email: (optional) Email of the user.
        """
        self.iam_id = iam_id
        self.name = name
        self.username = username
        self.email = email
        self.enrollments = enrollments

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'UserReportMfaEnrollmentStatus':
        """Initialize a UserReportMfaEnrollmentStatus object from a json dictionary."""
        args = {}
        if (iam_id := _dict.get('iam_id')) is not None:
            args['iam_id'] = iam_id
        else:
            raise ValueError('Required property \'iam_id\' not present in UserReportMfaEnrollmentStatus JSON')
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        if (username := _dict.get('username')) is not None:
            args['username'] = username
        else:
            raise ValueError('Required property \'username\' not present in UserReportMfaEnrollmentStatus JSON')
        if (email := _dict.get('email')) is not None:
            args['email'] = email
        if (enrollments := _dict.get('enrollments')) is not None:
            args['enrollments'] = MfaEnrollments.from_dict(enrollments)
        else:
            raise ValueError('Required property \'enrollments\' not present in UserReportMfaEnrollmentStatus JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a UserReportMfaEnrollmentStatus object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'iam_id') and self.iam_id is not None:
            _dict['iam_id'] = self.iam_id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'username') and self.username is not None:
            _dict['username'] = self.username
        if hasattr(self, 'email') and self.email is not None:
            _dict['email'] = self.email
        if hasattr(self, 'enrollments') and self.enrollments is not None:
            if isinstance(self.enrollments, dict):
                _dict['enrollments'] = self.enrollments
            else:
                _dict['enrollments'] = self.enrollments.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this UserReportMfaEnrollmentStatus object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'UserReportMfaEnrollmentStatus') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'UserReportMfaEnrollmentStatus') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other
