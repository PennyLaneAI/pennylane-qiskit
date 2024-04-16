# coding: utf-8

# (C) Copyright IBM Corp. 2022.
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

# IBM OpenAPI SDK Code Generator Version: 99-SNAPSHOT-f381b8c9-20221101-115055

"""
Manage lifecycle of your Cloud resources using Resource Controller APIs. Resources are
provisioned globally in an account scope. Supports asynchronous provisioning of resources.
Enables consumption of a global resource through a Cloud Foundry space in any region.

API Version: 2.0
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List
import json

from ibm_cloud_sdk_core import BaseService, DetailedResponse, get_query_param
from ibm_cloud_sdk_core.authenticators.authenticator import Authenticator
from ibm_cloud_sdk_core.get_authenticator import get_authenticator_from_environment
from ibm_cloud_sdk_core.utils import convert_model, datetime_to_string, string_to_datetime

from .common import get_sdk_headers

##############################################################################
# Service
##############################################################################


class ResourceControllerV2(BaseService):
    """The resource_controller V2 service."""

    DEFAULT_SERVICE_URL = 'https://resource-controller.cloud.ibm.com'
    DEFAULT_SERVICE_NAME = 'resource_controller'

    @classmethod
    def new_instance(
        cls,
        service_name: str = DEFAULT_SERVICE_NAME,
    ) -> 'ResourceControllerV2':
        """
        Return a new client for the resource_controller service using the specified
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
        Construct a new client for the resource_controller service.

        :param Authenticator authenticator: The authenticator specifies the authentication mechanism.
               Get up to date information from https://github.com/IBM/python-sdk-core/blob/main/README.md
               about initializing the authenticator of your choice.
        """
        BaseService.__init__(self, service_url=self.DEFAULT_SERVICE_URL, authenticator=authenticator)

    #########################
    # Resource Instances
    #########################

    def list_resource_instances(
        self,
        *,
        guid: str = None,
        name: str = None,
        resource_group_id: str = None,
        resource_id: str = None,
        resource_plan_id: str = None,
        type: str = None,
        sub_type: str = None,
        limit: int = None,
        start: str = None,
        state: str = None,
        updated_from: str = None,
        updated_to: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get a list of all resource instances.

        View a list of all available resource instances. Resources is a broad term that
        could mean anything from a service instance to a virtual machine associated with
        the customer account.

        :param str guid: (optional) The GUID of the instance.
        :param str name: (optional) The human-readable name of the instance.
        :param str resource_group_id: (optional) The ID of the resource group.
        :param str resource_id: (optional) The unique ID of the offering. This
               value is provided by and stored in the global catalog.
        :param str resource_plan_id: (optional) The unique ID of the plan
               associated with the offering. This value is provided by and stored in the
               global catalog.
        :param str type: (optional) The type of the instance, for example,
               `service_instance`.
        :param str sub_type: (optional) The sub-type of instance, for example,
               `kms`.
        :param int limit: (optional) Limit on how many items should be returned.
        :param str start: (optional) An optional token that indicates the beginning
               of the page of results to be returned. Any additional query parameters are
               ignored if a page token is present. If omitted, the first page of results
               is returned. This value is obtained from the 'start' query parameter in the
               'next_url' field of the operation response.
        :param str state: (optional) The state of the instance. If not specified,
               instances in state `active` and `provisioning` are returned.
        :param str updated_from: (optional) Start date inclusive filter.
        :param str updated_to: (optional) End date inclusive filter.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceInstancesList` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='list_resource_instances'
        )
        headers.update(sdk_headers)

        params = {
            'guid': guid,
            'name': name,
            'resource_group_id': resource_group_id,
            'resource_id': resource_id,
            'resource_plan_id': resource_plan_id,
            'type': type,
            'sub_type': sub_type,
            'limit': limit,
            'start': start,
            'state': state,
            'updated_from': updated_from,
            'updated_to': updated_to,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v2/resource_instances'
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def create_resource_instance(
        self,
        name: str,
        target: str,
        resource_group: str,
        resource_plan_id: str,
        *,
        tags: List[str] = None,
        allow_cleanup: bool = None,
        parameters: dict = None,
        entity_lock: bool = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create (provision) a new resource instance.

        When you provision a service you get an instance of that service. An instance
        represents the resource with which you create, and additionally, represents a
        chargeable record of which billing can occur.

        :param str name: The name of the instance. Must be 180 characters or less
               and cannot include any special characters other than `(space) - . _ :`.
        :param str target: The deployment location where the instance should be
               hosted.
        :param str resource_group: The ID of the resource group.
        :param str resource_plan_id: The unique ID of the plan associated with the
               offering. This value is provided by and stored in the global catalog.
        :param List[str] tags: (optional) Tags that are attached to the instance
               after provisioning. These tags can be searched and managed through the
               Tagging API in IBM Cloud.
        :param bool allow_cleanup: (optional) A boolean that dictates if the
               resource instance should be deleted (cleaned up) during the processing of a
               region instance delete call.
        :param dict parameters: (optional) Configuration options represented as
               key-value pairs that are passed through to the target resource brokers.
        :param bool entity_lock: (optional) Indicates if the resource instance is
               locked for further update or delete operations. It does not affect actions
               performed on child resources like aliases, bindings or keys. False by
               default.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceInstance` object
        """

        if name is None:
            raise ValueError('name must be provided')
        if target is None:
            raise ValueError('target must be provided')
        if resource_group is None:
            raise ValueError('resource_group must be provided')
        if resource_plan_id is None:
            raise ValueError('resource_plan_id must be provided')
        headers = {'Entity-Lock': entity_lock}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='create_resource_instance'
        )
        headers.update(sdk_headers)

        data = {
            'name': name,
            'target': target,
            'resource_group': resource_group,
            'resource_plan_id': resource_plan_id,
            'tags': tags,
            'allow_cleanup': allow_cleanup,
            'parameters': parameters,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v2/resource_instances'
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def get_resource_instance(self, id: str, **kwargs) -> DetailedResponse:
        """
        Get a resource instance.

        Retrieve a resource instance by URL-encoded CRN or GUID. Find more details on a
        particular instance, like when it was provisioned and who provisioned it.

        :param str id: The resource instance URL-encoded CRN or GUID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceInstance` object
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='get_resource_instance'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/resource_instances/{id}'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def delete_resource_instance(self, id: str, *, recursive: bool = None, **kwargs) -> DetailedResponse:
        """
        Delete a resource instance.

        Delete a resource instance by URL-encoded CRN or GUID. If the resource instance
        has any resource keys or aliases associated with it, use the `recursive=true`
        parameter to delete it.

        :param str id: The resource instance URL-encoded CRN or GUID.
        :param bool recursive: (optional) Will delete resource bindings, keys and
               aliases associated with the instance.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='delete_resource_instance'
        )
        headers.update(sdk_headers)

        params = {'recursive': recursive}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/resource_instances/{id}'.format(**path_param_dict)
        request = self.prepare_request(method='DELETE', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def update_resource_instance(
        self,
        id: str,
        *,
        name: str = None,
        parameters: dict = None,
        resource_plan_id: str = None,
        allow_cleanup: bool = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Update a resource instance.

        Use the resource instance URL-encoded CRN or GUID to make updates to the resource
        instance, like changing the name or plan.

        :param str id: The resource instance URL-encoded CRN or GUID.
        :param str name: (optional) The new name of the instance. Must be 180
               characters or less and cannot include any special characters other than
               `(space) - . _ :`.
        :param dict parameters: (optional) The new configuration options for the
               instance.
        :param str resource_plan_id: (optional) The unique ID of the plan
               associated with the offering. This value is provided by and stored in the
               global catalog.
        :param bool allow_cleanup: (optional) A boolean that dictates if the
               resource instance should be deleted (cleaned up) during the processing of a
               region instance delete call.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceInstance` object
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='update_resource_instance'
        )
        headers.update(sdk_headers)

        data = {
            'name': name,
            'parameters': parameters,
            'resource_plan_id': resource_plan_id,
            'allow_cleanup': allow_cleanup,
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
        url = '/v2/resource_instances/{id}'.format(**path_param_dict)
        request = self.prepare_request(method='PATCH', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def list_resource_aliases_for_instance(
        self, id: str, *, limit: int = None, start: str = None, **kwargs
    ) -> DetailedResponse:
        """
        Get a list of all resource aliases for the instance.

        Retrieving a list of all resource aliases can help you find out who's using the
        resource instance.

        :param str id: The resource instance URL-encoded CRN or GUID.
        :param int limit: (optional) Limit on how many items should be returned.
        :param str start: (optional) An optional token that indicates the beginning
               of the page of results to be returned. Any additional query parameters are
               ignored if a page token is present. If omitted, the first page of results
               is returned. This value is obtained from the 'start' query parameter in the
               'next_url' field of the operation response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceAliasesList` object
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='list_resource_aliases_for_instance',
        )
        headers.update(sdk_headers)

        params = {'limit': limit, 'start': start}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/resource_instances/{id}/resource_aliases'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def list_resource_keys_for_instance(
        self, id: str, *, limit: int = None, start: str = None, **kwargs
    ) -> DetailedResponse:
        """
        Get a list of all the resource keys for the instance.

        You may have many resource keys for one resource instance. For example, you may
        have a different resource key for each user or each role.

        :param str id: The resource instance URL-encoded CRN or GUID.
        :param int limit: (optional) Limit on how many items should be returned.
        :param str start: (optional) An optional token that indicates the beginning
               of the page of results to be returned. Any additional query parameters are
               ignored if a page token is present. If omitted, the first page of results
               is returned. This value is obtained from the 'start' query parameter in the
               'next_url' field of the operation response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceKeysList` object
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='list_resource_keys_for_instance'
        )
        headers.update(sdk_headers)

        params = {'limit': limit, 'start': start}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/resource_instances/{id}/resource_keys'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def lock_resource_instance(self, id: str, **kwargs) -> DetailedResponse:
        """
        Lock a resource instance.

        Locks a resource instance. A locked instance can not be updated or deleted. It
        does not affect actions performed on child resources like aliases, bindings, or
        keys.

        :param str id: The resource instance URL-encoded CRN or GUID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceInstance` object
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='lock_resource_instance'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/resource_instances/{id}/lock'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def unlock_resource_instance(self, id: str, **kwargs) -> DetailedResponse:
        """
        Unlock a resource instance.

        Unlock a resource instance to update or delete it. Unlocking a resource instance
        does not affect child resources like aliases, bindings or keys.

        :param str id: The resource instance URL-encoded CRN or GUID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceInstance` object
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='unlock_resource_instance'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/resource_instances/{id}/lock'.format(**path_param_dict)
        request = self.prepare_request(method='DELETE', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def cancel_lastop_resource_instance(self, id: str, **kwargs) -> DetailedResponse:
        """
        Cancel the in progress last operation of the resource instance.

        Cancel the in progress last operation of the resource instance. After successful
        cancellation, the resource instance is removed.

        :param str id: The resource instance URL-encoded CRN or GUID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceInstance` object
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='cancel_lastop_resource_instance'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/resource_instances/{id}/last_operation'.format(**path_param_dict)
        request = self.prepare_request(method='DELETE', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    #########################
    # Resource Keys
    #########################

    def list_resource_keys(
        self,
        *,
        guid: str = None,
        name: str = None,
        resource_group_id: str = None,
        resource_id: str = None,
        limit: int = None,
        start: str = None,
        updated_from: str = None,
        updated_to: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get a list of all of the resource keys.

        View all of the resource keys that exist for all of your resource instances.

        :param str guid: (optional) The GUID of the key.
        :param str name: (optional) The human-readable name of the key.
        :param str resource_group_id: (optional) The ID of the resource group.
        :param str resource_id: (optional) The unique ID of the offering. This
               value is provided by and stored in the global catalog.
        :param int limit: (optional) Limit on how many items should be returned.
        :param str start: (optional) An optional token that indicates the beginning
               of the page of results to be returned. Any additional query parameters are
               ignored if a page token is present. If omitted, the first page of results
               is returned. This value is obtained from the 'start' query parameter in the
               'next_url' field of the operation response.
        :param str updated_from: (optional) Start date inclusive filter.
        :param str updated_to: (optional) End date inclusive filter.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceKeysList` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='list_resource_keys'
        )
        headers.update(sdk_headers)

        params = {
            'guid': guid,
            'name': name,
            'resource_group_id': resource_group_id,
            'resource_id': resource_id,
            'limit': limit,
            'start': start,
            'updated_from': updated_from,
            'updated_to': updated_to,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v2/resource_keys'
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def create_resource_key(
        self, name: str, source: str, *, parameters: 'ResourceKeyPostParameters' = None, role: str = None, **kwargs
    ) -> DetailedResponse:
        """
        Create a new resource key.

        A resource key is a saved credential you can use to authenticate with a resource
        instance.

        :param str name: The name of the key.
        :param str source: The ID of resource instance or alias.
        :param ResourceKeyPostParameters parameters: (optional) Configuration
               options represented as key-value pairs. Service defined options are passed
               through to the target resource brokers, whereas platform defined options
               are not.
        :param str role: (optional) The base IAM service role name (Reader, Writer,
               or Manager), or the service or custom role CRN. Refer to service’s
               documentation for supported roles.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceKey` object
        """

        if name is None:
            raise ValueError('name must be provided')
        if source is None:
            raise ValueError('source must be provided')
        if parameters is not None:
            parameters = convert_model(parameters)
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='create_resource_key'
        )
        headers.update(sdk_headers)

        data = {'name': name, 'source': source, 'parameters': parameters, 'role': role}
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v2/resource_keys'
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def get_resource_key(self, id: str, **kwargs) -> DetailedResponse:
        """
        Get resource key.

        View the details of a resource key by URL-encoded CRN or GUID, like the
        credentials for the key and who created it.

        :param str id: The resource key URL-encoded CRN or GUID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceKey` object
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='get_resource_key'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/resource_keys/{id}'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def delete_resource_key(self, id: str, **kwargs) -> DetailedResponse:
        """
        Delete a resource key.

        Deleting a resource key does not affect any resource instance or resource alias
        associated with the key.

        :param str id: The resource key URL-encoded CRN or GUID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='delete_resource_key'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/resource_keys/{id}'.format(**path_param_dict)
        request = self.prepare_request(method='DELETE', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def update_resource_key(self, id: str, name: str, **kwargs) -> DetailedResponse:
        """
        Update a resource key.

        Use the resource key URL-encoded CRN or GUID to update the resource key.

        :param str id: The resource key URL-encoded CRN or GUID.
        :param str name: The new name of the key. Must be 180 characters or less
               and cannot include any special characters other than `(space) - . _ :`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceKey` object
        """

        if not id:
            raise ValueError('id must be provided')
        if name is None:
            raise ValueError('name must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='update_resource_key'
        )
        headers.update(sdk_headers)

        data = {'name': name}
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
        url = '/v2/resource_keys/{id}'.format(**path_param_dict)
        request = self.prepare_request(method='PATCH', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    #########################
    # Resource Bindings
    #########################

    def list_resource_bindings(
        self,
        *,
        guid: str = None,
        name: str = None,
        resource_group_id: str = None,
        resource_id: str = None,
        region_binding_id: str = None,
        limit: int = None,
        start: str = None,
        updated_from: str = None,
        updated_to: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get a list of all resource bindings.

        View all of the resource bindings that exist for all of your resource aliases.

        :param str guid: (optional) The GUID of the binding.
        :param str name: (optional) The human-readable name of the binding.
        :param str resource_group_id: (optional) The ID of the resource group.
        :param str resource_id: (optional) The unique ID of the offering (service
               name). This value is provided by and stored in the global catalog.
        :param str region_binding_id: (optional) The ID of the binding in the
               target environment. For example, `service_binding_id` in a given IBM Cloud
               environment.
        :param int limit: (optional) Limit on how many items should be returned.
        :param str start: (optional) An optional token that indicates the beginning
               of the page of results to be returned. Any additional query parameters are
               ignored if a page token is present. If omitted, the first page of results
               is returned. This value is obtained from the 'start' query parameter in the
               'next_url' field of the operation response.
        :param str updated_from: (optional) Start date inclusive filter.
        :param str updated_to: (optional) End date inclusive filter.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceBindingsList` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='list_resource_bindings'
        )
        headers.update(sdk_headers)

        params = {
            'guid': guid,
            'name': name,
            'resource_group_id': resource_group_id,
            'resource_id': resource_id,
            'region_binding_id': region_binding_id,
            'limit': limit,
            'start': start,
            'updated_from': updated_from,
            'updated_to': updated_to,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v2/resource_bindings'
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def create_resource_binding(
        self,
        source: str,
        target: str,
        *,
        name: str = None,
        parameters: 'ResourceBindingPostParameters' = None,
        role: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create a new resource binding.

        A resource binding connects credentials to a resource alias. The credentials are
        in the form of a resource key.

        :param str source: The ID of resource alias.
        :param str target: The CRN of application to bind to in a specific
               environment, for example, Dallas YP, CFEE instance.
        :param str name: (optional) The name of the binding. Must be 180 characters
               or less and cannot include any special characters other than `(space) - . _
               :`.
        :param ResourceBindingPostParameters parameters: (optional) Configuration
               options represented as key-value pairs. Service defined options are passed
               through to the target resource brokers, whereas platform defined options
               are not.
        :param str role: (optional) The base IAM service role name (Reader, Writer,
               or Manager), or the service or custom role CRN. Refer to service’s
               documentation for supported roles.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceBinding` object
        """

        if source is None:
            raise ValueError('source must be provided')
        if target is None:
            raise ValueError('target must be provided')
        if parameters is not None:
            parameters = convert_model(parameters)
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='create_resource_binding'
        )
        headers.update(sdk_headers)

        data = {'source': source, 'target': target, 'name': name, 'parameters': parameters, 'role': role}
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v2/resource_bindings'
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def get_resource_binding(self, id: str, **kwargs) -> DetailedResponse:
        """
        Get a resource binding.

        View a resource binding and all of its details, like who created it, the
        credential, and the resource alias that the binding is associated with.

        :param str id: The resource binding URL-encoded CRN or GUID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceBinding` object
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='get_resource_binding'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/resource_bindings/{id}'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def delete_resource_binding(self, id: str, **kwargs) -> DetailedResponse:
        """
        Delete a resource binding.

        Deleting a resource binding does not affect the resource alias that the binding is
        associated with.

        :param str id: The resource binding URL-encoded CRN or GUID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='delete_resource_binding'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/resource_bindings/{id}'.format(**path_param_dict)
        request = self.prepare_request(method='DELETE', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def update_resource_binding(self, id: str, name: str, **kwargs) -> DetailedResponse:
        """
        Update a resource binding.

        Use the resource binding URL-encoded CRN or GUID to update the resource binding.

        :param str id: The resource binding URL-encoded CRN or GUID.
        :param str name: The new name of the binding. Must be 180 characters or
               less and cannot include any special characters other than `(space) - . _
               :`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceBinding` object
        """

        if not id:
            raise ValueError('id must be provided')
        if name is None:
            raise ValueError('name must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='update_resource_binding'
        )
        headers.update(sdk_headers)

        data = {'name': name}
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
        url = '/v2/resource_bindings/{id}'.format(**path_param_dict)
        request = self.prepare_request(method='PATCH', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    #########################
    # Resource Aliases
    #########################

    def list_resource_aliases(
        self,
        *,
        guid: str = None,
        name: str = None,
        resource_instance_id: str = None,
        region_instance_id: str = None,
        resource_id: str = None,
        resource_group_id: str = None,
        limit: int = None,
        start: str = None,
        updated_from: str = None,
        updated_to: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get a list of all resource aliases.

        View all of the resource aliases that exist for every resource instance.

        :param str guid: (optional) The GUID of the alias.
        :param str name: (optional) The human-readable name of the alias.
        :param str resource_instance_id: (optional) The ID of the resource
               instance.
        :param str region_instance_id: (optional) The ID of the instance in the
               target environment. For example, `service_instance_id` in a given IBM Cloud
               environment.
        :param str resource_id: (optional) The unique ID of the offering (service
               name). This value is provided by and stored in the global catalog.
        :param str resource_group_id: (optional) The ID of the resource group.
        :param int limit: (optional) Limit on how many items should be returned.
        :param str start: (optional) An optional token that indicates the beginning
               of the page of results to be returned. Any additional query parameters are
               ignored if a page token is present. If omitted, the first page of results
               is returned. This value is obtained from the 'start' query parameter in the
               'next_url' field of the operation response.
        :param str updated_from: (optional) Start date inclusive filter.
        :param str updated_to: (optional) End date inclusive filter.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceAliasesList` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='list_resource_aliases'
        )
        headers.update(sdk_headers)

        params = {
            'guid': guid,
            'name': name,
            'resource_instance_id': resource_instance_id,
            'region_instance_id': region_instance_id,
            'resource_id': resource_id,
            'resource_group_id': resource_group_id,
            'limit': limit,
            'start': start,
            'updated_from': updated_from,
            'updated_to': updated_to,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v2/resource_aliases'
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def create_resource_alias(self, name: str, source: str, target: str, **kwargs) -> DetailedResponse:
        """
        Create a new resource alias.

        Alias a resource instance into a targeted environment's (name)space.

        :param str name: The name of the alias. Must be 180 characters or less and
               cannot include any special characters other than `(space) - . _ :`.
        :param str source: The ID of resource instance.
        :param str target: The CRN of target name(space) in a specific environment,
               for example, space in Dallas YP, CFEE instance etc.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceAlias` object
        """

        if name is None:
            raise ValueError('name must be provided')
        if source is None:
            raise ValueError('source must be provided')
        if target is None:
            raise ValueError('target must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='create_resource_alias'
        )
        headers.update(sdk_headers)

        data = {'name': name, 'source': source, 'target': target}
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v2/resource_aliases'
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def get_resource_alias(self, id: str, **kwargs) -> DetailedResponse:
        """
        Get a resource alias.

        View a resource alias and all of its details, like who created it and the resource
        instance that it's associated with.

        :param str id: The resource alias URL-encoded CRN or GUID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceAlias` object
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='get_resource_alias'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/resource_aliases/{id}'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def delete_resource_alias(self, id: str, *, recursive: bool = None, **kwargs) -> DetailedResponse:
        """
        Delete a resource alias.

        Delete a resource alias by URL-encoded CRN or GUID. If the resource alias has any
        resource keys or bindings associated with it, use the `recursive=true` parameter
        to delete it.

        :param str id: The resource alias URL-encoded CRN or GUID.
        :param bool recursive: (optional) Deletes the resource bindings and keys
               associated with the alias.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='delete_resource_alias'
        )
        headers.update(sdk_headers)

        params = {'recursive': recursive}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/resource_aliases/{id}'.format(**path_param_dict)
        request = self.prepare_request(method='DELETE', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def update_resource_alias(self, id: str, name: str, **kwargs) -> DetailedResponse:
        """
        Update a resource alias.

        Use the resource alias URL-encoded CRN or GUID to update the resource alias.

        :param str id: The resource alias URL-encoded CRN or GUID.
        :param str name: The new name of the alias. Must be 180 characters or less
               and cannot include any special characters other than `(space) - . _ :`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceAlias` object
        """

        if not id:
            raise ValueError('id must be provided')
        if name is None:
            raise ValueError('name must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='update_resource_alias'
        )
        headers.update(sdk_headers)

        data = {'name': name}
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
        url = '/v2/resource_aliases/{id}'.format(**path_param_dict)
        request = self.prepare_request(method='PATCH', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def list_resource_bindings_for_alias(
        self, id: str, *, limit: int = None, start: str = None, **kwargs
    ) -> DetailedResponse:
        """
        Get a list of all resource bindings for the alias.

        View all of the resource bindings associated with a specific resource alias.

        :param str id: The resource alias URL-encoded CRN or GUID.
        :param int limit: (optional) Limit on how many items should be returned.
        :param str start: (optional) An optional token that indicates the beginning
               of the page of results to be returned. Any additional query parameters are
               ignored if a page token is present. If omitted, the first page of results
               is returned. This value is obtained from the 'start' query parameter in the
               'next_url' field of the operation response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceBindingsList` object
        """

        if not id:
            raise ValueError('id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='list_resource_bindings_for_alias',
        )
        headers.update(sdk_headers)

        params = {'limit': limit, 'start': start}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['id']
        path_param_values = self.encode_path_vars(id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/resource_aliases/{id}/resource_bindings'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    #########################
    # Resource Reclamations
    #########################

    def list_reclamations(
        self, *, account_id: str = None, resource_instance_id: str = None, **kwargs
    ) -> DetailedResponse:
        """
        Get a list of all reclamations.

        View all of the resource reclamations that exist for every resource instance.

        :param str account_id: (optional) An alpha-numeric value identifying the
               account ID.
        :param str resource_instance_id: (optional) The GUID of the resource
               instance.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ReclamationsList` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='list_reclamations'
        )
        headers.update(sdk_headers)

        params = {'account_id': account_id, 'resource_instance_id': resource_instance_id}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/reclamations'
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def run_reclamation_action(
        self, id: str, action_name: str, *, request_by: str = None, comment: str = None, **kwargs
    ) -> DetailedResponse:
        """
        Perform a reclamation action.

        Reclaim a resource instance so that it can no longer be used, or restore the
        resource instance so that it's usable again.

        :param str id: The ID associated with the reclamation.
        :param str action_name: The reclamation action name. Specify `reclaim` to
               delete a resource, or `restore` to restore a resource.
        :param str request_by: (optional) The request initiator, if different from
               the request token.
        :param str comment: (optional) A comment to describe the action.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Reclamation` object
        """

        if not id:
            raise ValueError('id must be provided')
        if not action_name:
            raise ValueError('action_name must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V2', operation_id='run_reclamation_action'
        )
        headers.update(sdk_headers)

        data = {'request_by': request_by, 'comment': comment}
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['id', 'action_name']
        path_param_values = self.encode_path_vars(id, action_name)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/reclamations/{id}/actions/{action_name}'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response


class ListResourceInstancesEnums:
    """
    Enums for list_resource_instances parameters.
    """

    class State(str, Enum):
        """
        The state of the instance. If not specified, instances in state `active` and
        `provisioning` are returned.
        """

        ACTIVE = 'active'
        INACTIVE = 'inactive'
        FAILED = 'failed'
        PENDING_RECLAMATION = 'pending_reclamation'
        PROVISIONING = 'provisioning'
        PRE_PROVISIONING = 'pre_provisioning'
        REMOVED = 'removed'


##############################################################################
# Models
##############################################################################


class Credentials:
    """
    The credentials for a resource.

    :attr str redacted: (optional) If present, the user doesn't have the correct
          access to view the credentials and the details are redacted.  The string value
          identifies the level of access that's required to view the credential. For
          additional information, see [viewing a
          credential](https://cloud.ibm.com/docs/account?topic=account-service_credentials&interface=ui#viewing-credentials-ui).
    :attr str apikey: (optional) The API key for the credentials.
    :attr str iam_apikey_description: (optional) The optional description of the API
          key.
    :attr str iam_apikey_name: (optional) The name of the API key.
    :attr str iam_role_crn: (optional) The Cloud Resource Name for the role of the
          credentials.
    :attr str iam_serviceid_crn: (optional) The Cloud Resource Name for the service
          ID of the credentials.
    """

    # The set of defined properties for the class
    _properties = frozenset(
        [
            'redacted',
            'REDACTED',
            'apikey',
            'iam_apikey_description',
            'iam_apikey_name',
            'iam_role_crn',
            'iam_serviceid_crn',
        ]
    )

    def __init__(
        self,
        *,
        redacted: str = None,
        apikey: str = None,
        iam_apikey_description: str = None,
        iam_apikey_name: str = None,
        iam_role_crn: str = None,
        iam_serviceid_crn: str = None,
        **kwargs,
    ) -> None:
        """
        Initialize a Credentials object.

        :param str redacted: (optional) If present, the user doesn't have the
               correct access to view the credentials and the details are redacted.  The
               string value identifies the level of access that's required to view the
               credential. For additional information, see [viewing a
               credential](https://cloud.ibm.com/docs/account?topic=account-service_credentials&interface=ui#viewing-credentials-ui).
        :param str apikey: (optional) The API key for the credentials.
        :param str iam_apikey_description: (optional) The optional description of
               the API key.
        :param str iam_apikey_name: (optional) The name of the API key.
        :param str iam_role_crn: (optional) The Cloud Resource Name for the role of
               the credentials.
        :param str iam_serviceid_crn: (optional) The Cloud Resource Name for the
               service ID of the credentials.
        :param **kwargs: (optional) Any additional properties.
        """
        self.redacted = redacted
        self.apikey = apikey
        self.iam_apikey_description = iam_apikey_description
        self.iam_apikey_name = iam_apikey_name
        self.iam_role_crn = iam_role_crn
        self.iam_serviceid_crn = iam_serviceid_crn
        for _key, _value in kwargs.items():
            setattr(self, _key, _value)

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Credentials':
        """Initialize a Credentials object from a json dictionary."""
        args = {}
        if 'REDACTED' in _dict:
            args['redacted'] = _dict.get('REDACTED')
        if 'apikey' in _dict:
            args['apikey'] = _dict.get('apikey')
        if 'iam_apikey_description' in _dict:
            args['iam_apikey_description'] = _dict.get('iam_apikey_description')
        if 'iam_apikey_name' in _dict:
            args['iam_apikey_name'] = _dict.get('iam_apikey_name')
        if 'iam_role_crn' in _dict:
            args['iam_role_crn'] = _dict.get('iam_role_crn')
        if 'iam_serviceid_crn' in _dict:
            args['iam_serviceid_crn'] = _dict.get('iam_serviceid_crn')
        args.update({k: v for (k, v) in _dict.items() if k not in cls._properties})
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Credentials object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'redacted') and self.redacted is not None:
            _dict['REDACTED'] = self.redacted
        if hasattr(self, 'apikey') and self.apikey is not None:
            _dict['apikey'] = self.apikey
        if hasattr(self, 'iam_apikey_description') and self.iam_apikey_description is not None:
            _dict['iam_apikey_description'] = self.iam_apikey_description
        if hasattr(self, 'iam_apikey_name') and self.iam_apikey_name is not None:
            _dict['iam_apikey_name'] = self.iam_apikey_name
        if hasattr(self, 'iam_role_crn') and self.iam_role_crn is not None:
            _dict['iam_role_crn'] = self.iam_role_crn
        if hasattr(self, 'iam_serviceid_crn') and self.iam_serviceid_crn is not None:
            _dict['iam_serviceid_crn'] = self.iam_serviceid_crn
        for _key in [k for k in vars(self).keys() if k not in Credentials._properties]:
            _dict[_key] = getattr(self, _key)
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def get_properties(self) -> Dict:
        """Return a dictionary of arbitrary properties from this instance of Credentials"""
        _dict = {}

        for _key in [k for k in vars(self).keys() if k not in Credentials._properties]:
            _dict[_key] = getattr(self, _key)
        return _dict

    def set_properties(self, _dict: dict):
        """Set a dictionary of arbitrary properties to this instance of Credentials"""
        for _key in [k for k in vars(self).keys() if k not in Credentials._properties]:
            delattr(self, _key)

        for _key, _value in _dict.items():
            if _key not in Credentials._properties:
                setattr(self, _key, _value)

    def __str__(self) -> str:
        """Return a `str` version of this Credentials object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Credentials') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Credentials') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class RedactedEnum(str, Enum):
        """
        If present, the user doesn't have the correct access to view the credentials and
        the details are redacted.  The string value identifies the level of access that's
        required to view the credential. For additional information, see [viewing a
        credential](https://cloud.ibm.com/docs/account?topic=account-service_credentials&interface=ui#viewing-credentials-ui).
        """

        REDACTED = 'REDACTED'
        REDACTED_EXPLICIT = 'REDACTED_EXPLICIT'


class PlanHistoryItem:
    """
    An element of the plan history of the instance.

    :attr str resource_plan_id: The unique ID of the plan associated with the
          offering. This value is provided by and stored in the global catalog.
    :attr datetime start_date: The date on which the plan was changed.
    :attr str requestor_id: (optional) The subject who made the plan change.
    """

    def __init__(self, resource_plan_id: str, start_date: datetime, *, requestor_id: str = None) -> None:
        """
        Initialize a PlanHistoryItem object.

        :param str resource_plan_id: The unique ID of the plan associated with the
               offering. This value is provided by and stored in the global catalog.
        :param datetime start_date: The date on which the plan was changed.
        :param str requestor_id: (optional) The subject who made the plan change.
        """
        self.resource_plan_id = resource_plan_id
        self.start_date = start_date
        self.requestor_id = requestor_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'PlanHistoryItem':
        """Initialize a PlanHistoryItem object from a json dictionary."""
        args = {}
        if 'resource_plan_id' in _dict:
            args['resource_plan_id'] = _dict.get('resource_plan_id')
        else:
            raise ValueError('Required property \'resource_plan_id\' not present in PlanHistoryItem JSON')
        if 'start_date' in _dict:
            args['start_date'] = string_to_datetime(_dict.get('start_date'))
        else:
            raise ValueError('Required property \'start_date\' not present in PlanHistoryItem JSON')
        if 'requestor_id' in _dict:
            args['requestor_id'] = _dict.get('requestor_id')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a PlanHistoryItem object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'resource_plan_id') and self.resource_plan_id is not None:
            _dict['resource_plan_id'] = self.resource_plan_id
        if hasattr(self, 'start_date') and self.start_date is not None:
            _dict['start_date'] = datetime_to_string(self.start_date)
        if hasattr(self, 'requestor_id') and self.requestor_id is not None:
            _dict['requestor_id'] = self.requestor_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this PlanHistoryItem object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'PlanHistoryItem') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'PlanHistoryItem') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Reclamation:
    """
    A reclamation.

    :attr str id: (optional) The ID associated with the reclamation.
    :attr str entity_id: (optional) The ID of the entity for the reclamation.
    :attr str entity_type_id: (optional) The ID of the entity type for the
          reclamation.
    :attr str entity_crn: (optional) The full Cloud Resource Name (CRN) associated
          with the binding. For more information about this format, see [Cloud Resource
          Names](https://cloud.ibm.com/docs/overview?topic=overview-crn).
    :attr str resource_instance_id: (optional) The ID of the resource instance.
    :attr str resource_group_id: (optional) The ID of the resource group.
    :attr str account_id: (optional) An alpha-numeric value identifying the account
          ID.
    :attr str policy_id: (optional) The ID of policy for the reclamation.
    :attr str state: (optional) The state of the reclamation.
    :attr str target_time: (optional) The target time that the reclamation retention
          period end.
    :attr dict custom_properties: (optional) The custom properties of the
          reclamation.
    :attr datetime created_at: (optional) The date when the reclamation was created.
    :attr str created_by: (optional) The subject who created the reclamation.
    :attr datetime updated_at: (optional) The date when the reclamation was last
          updated.
    :attr str updated_by: (optional) The subject who updated the reclamation.
    """

    def __init__(
        self,
        *,
        id: str = None,
        entity_id: str = None,
        entity_type_id: str = None,
        entity_crn: str = None,
        resource_instance_id: str = None,
        resource_group_id: str = None,
        account_id: str = None,
        policy_id: str = None,
        state: str = None,
        target_time: str = None,
        custom_properties: dict = None,
        created_at: datetime = None,
        created_by: str = None,
        updated_at: datetime = None,
        updated_by: str = None,
    ) -> None:
        """
        Initialize a Reclamation object.

        :param str id: (optional) The ID associated with the reclamation.
        :param str entity_id: (optional) The ID of the entity for the reclamation.
        :param str entity_type_id: (optional) The ID of the entity type for the
               reclamation.
        :param str entity_crn: (optional) The full Cloud Resource Name (CRN)
               associated with the binding. For more information about this format, see
               [Cloud Resource
               Names](https://cloud.ibm.com/docs/overview?topic=overview-crn).
        :param str resource_instance_id: (optional) The ID of the resource
               instance.
        :param str resource_group_id: (optional) The ID of the resource group.
        :param str account_id: (optional) An alpha-numeric value identifying the
               account ID.
        :param str policy_id: (optional) The ID of policy for the reclamation.
        :param str state: (optional) The state of the reclamation.
        :param str target_time: (optional) The target time that the reclamation
               retention period end.
        :param dict custom_properties: (optional) The custom properties of the
               reclamation.
        :param datetime created_at: (optional) The date when the reclamation was
               created.
        :param str created_by: (optional) The subject who created the reclamation.
        :param datetime updated_at: (optional) The date when the reclamation was
               last updated.
        :param str updated_by: (optional) The subject who updated the reclamation.
        """
        self.id = id
        self.entity_id = entity_id
        self.entity_type_id = entity_type_id
        self.entity_crn = entity_crn
        self.resource_instance_id = resource_instance_id
        self.resource_group_id = resource_group_id
        self.account_id = account_id
        self.policy_id = policy_id
        self.state = state
        self.target_time = target_time
        self.custom_properties = custom_properties
        self.created_at = created_at
        self.created_by = created_by
        self.updated_at = updated_at
        self.updated_by = updated_by

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Reclamation':
        """Initialize a Reclamation object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'entity_id' in _dict:
            args['entity_id'] = _dict.get('entity_id')
        if 'entity_type_id' in _dict:
            args['entity_type_id'] = _dict.get('entity_type_id')
        if 'entity_crn' in _dict:
            args['entity_crn'] = _dict.get('entity_crn')
        if 'resource_instance_id' in _dict:
            args['resource_instance_id'] = _dict.get('resource_instance_id')
        if 'resource_group_id' in _dict:
            args['resource_group_id'] = _dict.get('resource_group_id')
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        if 'policy_id' in _dict:
            args['policy_id'] = _dict.get('policy_id')
        if 'state' in _dict:
            args['state'] = _dict.get('state')
        if 'target_time' in _dict:
            args['target_time'] = _dict.get('target_time')
        if 'custom_properties' in _dict:
            args['custom_properties'] = _dict.get('custom_properties')
        if 'created_at' in _dict:
            args['created_at'] = string_to_datetime(_dict.get('created_at'))
        if 'created_by' in _dict:
            args['created_by'] = _dict.get('created_by')
        if 'updated_at' in _dict:
            args['updated_at'] = string_to_datetime(_dict.get('updated_at'))
        if 'updated_by' in _dict:
            args['updated_by'] = _dict.get('updated_by')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Reclamation object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'entity_id') and self.entity_id is not None:
            _dict['entity_id'] = self.entity_id
        if hasattr(self, 'entity_type_id') and self.entity_type_id is not None:
            _dict['entity_type_id'] = self.entity_type_id
        if hasattr(self, 'entity_crn') and self.entity_crn is not None:
            _dict['entity_crn'] = self.entity_crn
        if hasattr(self, 'resource_instance_id') and self.resource_instance_id is not None:
            _dict['resource_instance_id'] = self.resource_instance_id
        if hasattr(self, 'resource_group_id') and self.resource_group_id is not None:
            _dict['resource_group_id'] = self.resource_group_id
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'policy_id') and self.policy_id is not None:
            _dict['policy_id'] = self.policy_id
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        if hasattr(self, 'target_time') and self.target_time is not None:
            _dict['target_time'] = self.target_time
        if hasattr(self, 'custom_properties') and self.custom_properties is not None:
            _dict['custom_properties'] = self.custom_properties
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'created_by') and self.created_by is not None:
            _dict['created_by'] = self.created_by
        if hasattr(self, 'updated_at') and self.updated_at is not None:
            _dict['updated_at'] = datetime_to_string(self.updated_at)
        if hasattr(self, 'updated_by') and self.updated_by is not None:
            _dict['updated_by'] = self.updated_by
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Reclamation object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Reclamation') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Reclamation') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ReclamationsList:
    """
    A list of reclamations.

    :attr List[Reclamation] resources: (optional) A list of reclamations.
    """

    def __init__(self, *, resources: List['Reclamation'] = None) -> None:
        """
        Initialize a ReclamationsList object.

        :param List[Reclamation] resources: (optional) A list of reclamations.
        """
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ReclamationsList':
        """Initialize a ReclamationsList object from a json dictionary."""
        args = {}
        if 'resources' in _dict:
            args['resources'] = [Reclamation.from_dict(v) for v in _dict.get('resources')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ReclamationsList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'resources') and self.resources is not None:
            resources_list = []
            for v in self.resources:
                if isinstance(v, dict):
                    resources_list.append(v)
                else:
                    resources_list.append(v.to_dict())
            _dict['resources'] = resources_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ReclamationsList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ReclamationsList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ReclamationsList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ResourceAlias:
    """
    A resource alias.

    :attr str id: (optional) The ID associated with the alias.
    :attr str guid: (optional) The GUID of the alias.
    :attr str url: (optional) When you created a new alias, a relative URL path is
          created identifying the location of the alias.
    :attr datetime created_at: (optional) The date when the alias was created.
    :attr datetime updated_at: (optional) The date when the alias was last updated.
    :attr datetime deleted_at: (optional) The date when the alias was deleted.
    :attr str created_by: (optional) The subject who created the alias.
    :attr str updated_by: (optional) The subject who updated the alias.
    :attr str deleted_by: (optional) The subject who deleted the alias.
    :attr str name: (optional) The human-readable name of the alias.
    :attr str resource_instance_id: (optional) The ID of the resource instance that
          is being aliased.
    :attr str target_crn: (optional) The CRN of the target namespace in the specific
          environment.
    :attr str account_id: (optional) An alpha-numeric value identifying the account
          ID.
    :attr str resource_id: (optional) The unique ID of the offering. This value is
          provided by and stored in the global catalog.
    :attr str resource_group_id: (optional) The ID of the resource group.
    :attr str crn: (optional) The CRN of the alias. For more information about this
          format, see [Cloud Resource
          Names](https://cloud.ibm.com/docs/overview?topic=overview-crn).
    :attr str region_instance_id: (optional) The ID of the instance in the target
          environment. For example, `service_instance_id` in a given IBM Cloud
          environment.
    :attr str region_instance_crn: (optional) The CRN of the instance in the target
          environment.
    :attr str state: (optional) The state of the alias.
    :attr bool migrated: (optional) A boolean that dictates if the alias was
          migrated from a previous CF instance.
    :attr str resource_instance_url: (optional) The relative path to the resource
          instance.
    :attr str resource_bindings_url: (optional) The relative path to the resource
          bindings for the alias.
    :attr str resource_keys_url: (optional) The relative path to the resource keys
          for the alias.
    """

    def __init__(
        self,
        *,
        id: str = None,
        guid: str = None,
        url: str = None,
        created_at: datetime = None,
        updated_at: datetime = None,
        deleted_at: datetime = None,
        created_by: str = None,
        updated_by: str = None,
        deleted_by: str = None,
        name: str = None,
        resource_instance_id: str = None,
        target_crn: str = None,
        account_id: str = None,
        resource_id: str = None,
        resource_group_id: str = None,
        crn: str = None,
        region_instance_id: str = None,
        region_instance_crn: str = None,
        state: str = None,
        migrated: bool = None,
        resource_instance_url: str = None,
        resource_bindings_url: str = None,
        resource_keys_url: str = None,
    ) -> None:
        """
        Initialize a ResourceAlias object.

        :param str id: (optional) The ID associated with the alias.
        :param str guid: (optional) The GUID of the alias.
        :param str url: (optional) When you created a new alias, a relative URL
               path is created identifying the location of the alias.
        :param datetime created_at: (optional) The date when the alias was created.
        :param datetime updated_at: (optional) The date when the alias was last
               updated.
        :param datetime deleted_at: (optional) The date when the alias was deleted.
        :param str created_by: (optional) The subject who created the alias.
        :param str updated_by: (optional) The subject who updated the alias.
        :param str deleted_by: (optional) The subject who deleted the alias.
        :param str name: (optional) The human-readable name of the alias.
        :param str resource_instance_id: (optional) The ID of the resource instance
               that is being aliased.
        :param str target_crn: (optional) The CRN of the target namespace in the
               specific environment.
        :param str account_id: (optional) An alpha-numeric value identifying the
               account ID.
        :param str resource_id: (optional) The unique ID of the offering. This
               value is provided by and stored in the global catalog.
        :param str resource_group_id: (optional) The ID of the resource group.
        :param str crn: (optional) The CRN of the alias. For more information about
               this format, see [Cloud Resource
               Names](https://cloud.ibm.com/docs/overview?topic=overview-crn).
        :param str region_instance_id: (optional) The ID of the instance in the
               target environment. For example, `service_instance_id` in a given IBM Cloud
               environment.
        :param str region_instance_crn: (optional) The CRN of the instance in the
               target environment.
        :param str state: (optional) The state of the alias.
        :param bool migrated: (optional) A boolean that dictates if the alias was
               migrated from a previous CF instance.
        :param str resource_instance_url: (optional) The relative path to the
               resource instance.
        :param str resource_bindings_url: (optional) The relative path to the
               resource bindings for the alias.
        :param str resource_keys_url: (optional) The relative path to the resource
               keys for the alias.
        """
        self.id = id
        self.guid = guid
        self.url = url
        self.created_at = created_at
        self.updated_at = updated_at
        self.deleted_at = deleted_at
        self.created_by = created_by
        self.updated_by = updated_by
        self.deleted_by = deleted_by
        self.name = name
        self.resource_instance_id = resource_instance_id
        self.target_crn = target_crn
        self.account_id = account_id
        self.resource_id = resource_id
        self.resource_group_id = resource_group_id
        self.crn = crn
        self.region_instance_id = region_instance_id
        self.region_instance_crn = region_instance_crn
        self.state = state
        self.migrated = migrated
        self.resource_instance_url = resource_instance_url
        self.resource_bindings_url = resource_bindings_url
        self.resource_keys_url = resource_keys_url

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourceAlias':
        """Initialize a ResourceAlias object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'guid' in _dict:
            args['guid'] = _dict.get('guid')
        if 'url' in _dict:
            args['url'] = _dict.get('url')
        if 'created_at' in _dict:
            args['created_at'] = string_to_datetime(_dict.get('created_at'))
        if 'updated_at' in _dict:
            args['updated_at'] = string_to_datetime(_dict.get('updated_at'))
        if 'deleted_at' in _dict:
            args['deleted_at'] = string_to_datetime(_dict.get('deleted_at'))
        if 'created_by' in _dict:
            args['created_by'] = _dict.get('created_by')
        if 'updated_by' in _dict:
            args['updated_by'] = _dict.get('updated_by')
        if 'deleted_by' in _dict:
            args['deleted_by'] = _dict.get('deleted_by')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'resource_instance_id' in _dict:
            args['resource_instance_id'] = _dict.get('resource_instance_id')
        if 'target_crn' in _dict:
            args['target_crn'] = _dict.get('target_crn')
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        if 'resource_id' in _dict:
            args['resource_id'] = _dict.get('resource_id')
        if 'resource_group_id' in _dict:
            args['resource_group_id'] = _dict.get('resource_group_id')
        if 'crn' in _dict:
            args['crn'] = _dict.get('crn')
        if 'region_instance_id' in _dict:
            args['region_instance_id'] = _dict.get('region_instance_id')
        if 'region_instance_crn' in _dict:
            args['region_instance_crn'] = _dict.get('region_instance_crn')
        if 'state' in _dict:
            args['state'] = _dict.get('state')
        if 'migrated' in _dict:
            args['migrated'] = _dict.get('migrated')
        if 'resource_instance_url' in _dict:
            args['resource_instance_url'] = _dict.get('resource_instance_url')
        if 'resource_bindings_url' in _dict:
            args['resource_bindings_url'] = _dict.get('resource_bindings_url')
        if 'resource_keys_url' in _dict:
            args['resource_keys_url'] = _dict.get('resource_keys_url')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourceAlias object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'guid') and self.guid is not None:
            _dict['guid'] = self.guid
        if hasattr(self, 'url') and self.url is not None:
            _dict['url'] = self.url
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'updated_at') and self.updated_at is not None:
            _dict['updated_at'] = datetime_to_string(self.updated_at)
        if hasattr(self, 'deleted_at') and self.deleted_at is not None:
            _dict['deleted_at'] = datetime_to_string(self.deleted_at)
        if hasattr(self, 'created_by') and self.created_by is not None:
            _dict['created_by'] = self.created_by
        if hasattr(self, 'updated_by') and self.updated_by is not None:
            _dict['updated_by'] = self.updated_by
        if hasattr(self, 'deleted_by') and self.deleted_by is not None:
            _dict['deleted_by'] = self.deleted_by
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'resource_instance_id') and self.resource_instance_id is not None:
            _dict['resource_instance_id'] = self.resource_instance_id
        if hasattr(self, 'target_crn') and self.target_crn is not None:
            _dict['target_crn'] = self.target_crn
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'resource_id') and self.resource_id is not None:
            _dict['resource_id'] = self.resource_id
        if hasattr(self, 'resource_group_id') and self.resource_group_id is not None:
            _dict['resource_group_id'] = self.resource_group_id
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'region_instance_id') and self.region_instance_id is not None:
            _dict['region_instance_id'] = self.region_instance_id
        if hasattr(self, 'region_instance_crn') and self.region_instance_crn is not None:
            _dict['region_instance_crn'] = self.region_instance_crn
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        if hasattr(self, 'migrated') and self.migrated is not None:
            _dict['migrated'] = self.migrated
        if hasattr(self, 'resource_instance_url') and self.resource_instance_url is not None:
            _dict['resource_instance_url'] = self.resource_instance_url
        if hasattr(self, 'resource_bindings_url') and self.resource_bindings_url is not None:
            _dict['resource_bindings_url'] = self.resource_bindings_url
        if hasattr(self, 'resource_keys_url') and self.resource_keys_url is not None:
            _dict['resource_keys_url'] = self.resource_keys_url
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResourceAlias object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourceAlias') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourceAlias') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ResourceAliasesList:
    """
    A list of resource aliases.

    :attr int rows_count: The number of resource aliases in `resources`.
    :attr str next_url: The URL for requesting the next page of results.
    :attr List[ResourceAlias] resources: A list of resource aliases.
    """

    def __init__(self, rows_count: int, next_url: str, resources: List['ResourceAlias']) -> None:
        """
        Initialize a ResourceAliasesList object.

        :param int rows_count: The number of resource aliases in `resources`.
        :param str next_url: The URL for requesting the next page of results.
        :param List[ResourceAlias] resources: A list of resource aliases.
        """
        self.rows_count = rows_count
        self.next_url = next_url
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourceAliasesList':
        """Initialize a ResourceAliasesList object from a json dictionary."""
        args = {}
        if 'rows_count' in _dict:
            args['rows_count'] = _dict.get('rows_count')
        else:
            raise ValueError('Required property \'rows_count\' not present in ResourceAliasesList JSON')
        if 'next_url' in _dict:
            args['next_url'] = _dict.get('next_url')
        else:
            raise ValueError('Required property \'next_url\' not present in ResourceAliasesList JSON')
        if 'resources' in _dict:
            args['resources'] = [ResourceAlias.from_dict(v) for v in _dict.get('resources')]
        else:
            raise ValueError('Required property \'resources\' not present in ResourceAliasesList JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourceAliasesList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'rows_count') and self.rows_count is not None:
            _dict['rows_count'] = self.rows_count
        if hasattr(self, 'next_url') and self.next_url is not None:
            _dict['next_url'] = self.next_url
        if hasattr(self, 'resources') and self.resources is not None:
            resources_list = []
            for v in self.resources:
                if isinstance(v, dict):
                    resources_list.append(v)
                else:
                    resources_list.append(v.to_dict())
            _dict['resources'] = resources_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResourceAliasesList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourceAliasesList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourceAliasesList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ResourceBinding:
    """
    A resource binding.

    :attr str id: (optional) The ID associated with the binding.
    :attr str guid: (optional) The GUID of the binding.
    :attr str url: (optional) When you provision a new binding, a relative URL path
          is created identifying the location of the binding.
    :attr datetime created_at: (optional) The date when the binding was created.
    :attr datetime updated_at: (optional) The date when the binding was last
          updated.
    :attr datetime deleted_at: (optional) The date when the binding was deleted.
    :attr str created_by: (optional) The subject who created the binding.
    :attr str updated_by: (optional) The subject who updated the binding.
    :attr str deleted_by: (optional) The subject who deleted the binding.
    :attr str source_crn: (optional) The CRN of resource alias associated to the
          binding.
    :attr str target_crn: (optional) The CRN of target resource, for example,
          application, in a specific environment.
    :attr str crn: (optional) The full Cloud Resource Name (CRN) associated with the
          binding. For more information about this format, see [Cloud Resource
          Names](https://cloud.ibm.com/docs/overview?topic=overview-crn).
    :attr str region_binding_id: (optional) The ID of the binding in the target
          environment. For example, `service_binding_id` in a given IBM Cloud environment.
    :attr str region_binding_crn: (optional) The CRN of the binding in the target
          environment.
    :attr str name: (optional) The human-readable name of the binding.
    :attr str account_id: (optional) An alpha-numeric value identifying the account
          ID.
    :attr str resource_group_id: (optional) The ID of the resource group.
    :attr str state: (optional) The state of the binding.
    :attr Credentials credentials: (optional) The credentials for the binding.
          Additional key-value pairs are passed through from the resource brokers. After a
          credential is created for a service, it can be viewed at any time for users that
          need the API key value. However, all users must have the correct level of access
          to see the details of a credential that includes the API key value. For
          additional details, see [viewing a
          credential](https://cloud.ibm.com/docs/account?topic=account-service_credentials&interface=ui#viewing-credentials-ui)
          or the service’s documentation.
    :attr bool iam_compatible: (optional) Specifies whether the binding’s
          credentials support IAM.
    :attr str resource_id: (optional) The unique ID of the offering. This value is
          provided by and stored in the global catalog.
    :attr bool migrated: (optional) A boolean that dictates if the alias was
          migrated from a previous CF instance.
    :attr str resource_alias_url: (optional) The relative path to the resource alias
          that this binding is associated with.
    """

    def __init__(
        self,
        *,
        id: str = None,
        guid: str = None,
        url: str = None,
        created_at: datetime = None,
        updated_at: datetime = None,
        deleted_at: datetime = None,
        created_by: str = None,
        updated_by: str = None,
        deleted_by: str = None,
        source_crn: str = None,
        target_crn: str = None,
        crn: str = None,
        region_binding_id: str = None,
        region_binding_crn: str = None,
        name: str = None,
        account_id: str = None,
        resource_group_id: str = None,
        state: str = None,
        credentials: 'Credentials' = None,
        iam_compatible: bool = None,
        resource_id: str = None,
        migrated: bool = None,
        resource_alias_url: str = None,
    ) -> None:
        """
        Initialize a ResourceBinding object.

        :param str id: (optional) The ID associated with the binding.
        :param str guid: (optional) The GUID of the binding.
        :param str url: (optional) When you provision a new binding, a relative URL
               path is created identifying the location of the binding.
        :param datetime created_at: (optional) The date when the binding was
               created.
        :param datetime updated_at: (optional) The date when the binding was last
               updated.
        :param datetime deleted_at: (optional) The date when the binding was
               deleted.
        :param str created_by: (optional) The subject who created the binding.
        :param str updated_by: (optional) The subject who updated the binding.
        :param str deleted_by: (optional) The subject who deleted the binding.
        :param str source_crn: (optional) The CRN of resource alias associated to
               the binding.
        :param str target_crn: (optional) The CRN of target resource, for example,
               application, in a specific environment.
        :param str crn: (optional) The full Cloud Resource Name (CRN) associated
               with the binding. For more information about this format, see [Cloud
               Resource Names](https://cloud.ibm.com/docs/overview?topic=overview-crn).
        :param str region_binding_id: (optional) The ID of the binding in the
               target environment. For example, `service_binding_id` in a given IBM Cloud
               environment.
        :param str region_binding_crn: (optional) The CRN of the binding in the
               target environment.
        :param str name: (optional) The human-readable name of the binding.
        :param str account_id: (optional) An alpha-numeric value identifying the
               account ID.
        :param str resource_group_id: (optional) The ID of the resource group.
        :param str state: (optional) The state of the binding.
        :param Credentials credentials: (optional) The credentials for the binding.
               Additional key-value pairs are passed through from the resource brokers.
               After a credential is created for a service, it can be viewed at any time
               for users that need the API key value. However, all users must have the
               correct level of access to see the details of a credential that includes
               the API key value. For additional details, see [viewing a
               credential](https://cloud.ibm.com/docs/account?topic=account-service_credentials&interface=ui#viewing-credentials-ui)
               or the service’s documentation.
        :param bool iam_compatible: (optional) Specifies whether the binding’s
               credentials support IAM.
        :param str resource_id: (optional) The unique ID of the offering. This
               value is provided by and stored in the global catalog.
        :param bool migrated: (optional) A boolean that dictates if the alias was
               migrated from a previous CF instance.
        :param str resource_alias_url: (optional) The relative path to the resource
               alias that this binding is associated with.
        """
        self.id = id
        self.guid = guid
        self.url = url
        self.created_at = created_at
        self.updated_at = updated_at
        self.deleted_at = deleted_at
        self.created_by = created_by
        self.updated_by = updated_by
        self.deleted_by = deleted_by
        self.source_crn = source_crn
        self.target_crn = target_crn
        self.crn = crn
        self.region_binding_id = region_binding_id
        self.region_binding_crn = region_binding_crn
        self.name = name
        self.account_id = account_id
        self.resource_group_id = resource_group_id
        self.state = state
        self.credentials = credentials
        self.iam_compatible = iam_compatible
        self.resource_id = resource_id
        self.migrated = migrated
        self.resource_alias_url = resource_alias_url

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourceBinding':
        """Initialize a ResourceBinding object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'guid' in _dict:
            args['guid'] = _dict.get('guid')
        if 'url' in _dict:
            args['url'] = _dict.get('url')
        if 'created_at' in _dict:
            args['created_at'] = string_to_datetime(_dict.get('created_at'))
        if 'updated_at' in _dict:
            args['updated_at'] = string_to_datetime(_dict.get('updated_at'))
        if 'deleted_at' in _dict:
            args['deleted_at'] = string_to_datetime(_dict.get('deleted_at'))
        if 'created_by' in _dict:
            args['created_by'] = _dict.get('created_by')
        if 'updated_by' in _dict:
            args['updated_by'] = _dict.get('updated_by')
        if 'deleted_by' in _dict:
            args['deleted_by'] = _dict.get('deleted_by')
        if 'source_crn' in _dict:
            args['source_crn'] = _dict.get('source_crn')
        if 'target_crn' in _dict:
            args['target_crn'] = _dict.get('target_crn')
        if 'crn' in _dict:
            args['crn'] = _dict.get('crn')
        if 'region_binding_id' in _dict:
            args['region_binding_id'] = _dict.get('region_binding_id')
        if 'region_binding_crn' in _dict:
            args['region_binding_crn'] = _dict.get('region_binding_crn')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        if 'resource_group_id' in _dict:
            args['resource_group_id'] = _dict.get('resource_group_id')
        if 'state' in _dict:
            args['state'] = _dict.get('state')
        if 'credentials' in _dict:
            args['credentials'] = Credentials.from_dict(_dict.get('credentials'))
        if 'iam_compatible' in _dict:
            args['iam_compatible'] = _dict.get('iam_compatible')
        if 'resource_id' in _dict:
            args['resource_id'] = _dict.get('resource_id')
        if 'migrated' in _dict:
            args['migrated'] = _dict.get('migrated')
        if 'resource_alias_url' in _dict:
            args['resource_alias_url'] = _dict.get('resource_alias_url')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourceBinding object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'guid') and self.guid is not None:
            _dict['guid'] = self.guid
        if hasattr(self, 'url') and self.url is not None:
            _dict['url'] = self.url
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'updated_at') and self.updated_at is not None:
            _dict['updated_at'] = datetime_to_string(self.updated_at)
        if hasattr(self, 'deleted_at') and self.deleted_at is not None:
            _dict['deleted_at'] = datetime_to_string(self.deleted_at)
        if hasattr(self, 'created_by') and self.created_by is not None:
            _dict['created_by'] = self.created_by
        if hasattr(self, 'updated_by') and self.updated_by is not None:
            _dict['updated_by'] = self.updated_by
        if hasattr(self, 'deleted_by') and self.deleted_by is not None:
            _dict['deleted_by'] = self.deleted_by
        if hasattr(self, 'source_crn') and self.source_crn is not None:
            _dict['source_crn'] = self.source_crn
        if hasattr(self, 'target_crn') and self.target_crn is not None:
            _dict['target_crn'] = self.target_crn
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'region_binding_id') and self.region_binding_id is not None:
            _dict['region_binding_id'] = self.region_binding_id
        if hasattr(self, 'region_binding_crn') and self.region_binding_crn is not None:
            _dict['region_binding_crn'] = self.region_binding_crn
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'resource_group_id') and self.resource_group_id is not None:
            _dict['resource_group_id'] = self.resource_group_id
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        if hasattr(self, 'credentials') and self.credentials is not None:
            if isinstance(self.credentials, dict):
                _dict['credentials'] = self.credentials
            else:
                _dict['credentials'] = self.credentials.to_dict()
        if hasattr(self, 'iam_compatible') and self.iam_compatible is not None:
            _dict['iam_compatible'] = self.iam_compatible
        if hasattr(self, 'resource_id') and self.resource_id is not None:
            _dict['resource_id'] = self.resource_id
        if hasattr(self, 'migrated') and self.migrated is not None:
            _dict['migrated'] = self.migrated
        if hasattr(self, 'resource_alias_url') and self.resource_alias_url is not None:
            _dict['resource_alias_url'] = self.resource_alias_url
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResourceBinding object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourceBinding') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourceBinding') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ResourceBindingPostParameters:
    """
    Configuration options represented as key-value pairs. Service defined options are
    passed through to the target resource brokers, whereas platform defined options are
    not.

    :attr str serviceid_crn: (optional) An optional platform defined option to reuse
          an existing IAM serviceId for the role assignment.
    """

    # The set of defined properties for the class
    _properties = frozenset(['serviceid_crn'])

    def __init__(self, *, serviceid_crn: str = None, **kwargs) -> None:
        """
        Initialize a ResourceBindingPostParameters object.

        :param str serviceid_crn: (optional) An optional platform defined option to
               reuse an existing IAM serviceId for the role assignment.
        :param **kwargs: (optional) Any additional properties.
        """
        self.serviceid_crn = serviceid_crn
        for _key, _value in kwargs.items():
            setattr(self, _key, _value)

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourceBindingPostParameters':
        """Initialize a ResourceBindingPostParameters object from a json dictionary."""
        args = {}
        if 'serviceid_crn' in _dict:
            args['serviceid_crn'] = _dict.get('serviceid_crn')
        args.update({k: v for (k, v) in _dict.items() if k not in cls._properties})
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourceBindingPostParameters object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'serviceid_crn') and self.serviceid_crn is not None:
            _dict['serviceid_crn'] = self.serviceid_crn
        for _key in [k for k in vars(self).keys() if k not in ResourceBindingPostParameters._properties]:
            _dict[_key] = getattr(self, _key)
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def get_properties(self) -> Dict:
        """Return a dictionary of arbitrary properties from this instance of ResourceBindingPostParameters"""
        _dict = {}

        for _key in [k for k in vars(self).keys() if k not in ResourceBindingPostParameters._properties]:
            _dict[_key] = getattr(self, _key)
        return _dict

    def set_properties(self, _dict: dict):
        """Set a dictionary of arbitrary properties to this instance of ResourceBindingPostParameters"""
        for _key in [k for k in vars(self).keys() if k not in ResourceBindingPostParameters._properties]:
            delattr(self, _key)

        for _key, _value in _dict.items():
            if _key not in ResourceBindingPostParameters._properties:
                setattr(self, _key, _value)

    def __str__(self) -> str:
        """Return a `str` version of this ResourceBindingPostParameters object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourceBindingPostParameters') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourceBindingPostParameters') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ResourceBindingsList:
    """
    A list of resource bindings.

    :attr int rows_count: The number of resource bindings in `resources`.
    :attr str next_url: The URL for requesting the next page of results.
    :attr List[ResourceBinding] resources: A list of resource bindings.
    """

    def __init__(self, rows_count: int, next_url: str, resources: List['ResourceBinding']) -> None:
        """
        Initialize a ResourceBindingsList object.

        :param int rows_count: The number of resource bindings in `resources`.
        :param str next_url: The URL for requesting the next page of results.
        :param List[ResourceBinding] resources: A list of resource bindings.
        """
        self.rows_count = rows_count
        self.next_url = next_url
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourceBindingsList':
        """Initialize a ResourceBindingsList object from a json dictionary."""
        args = {}
        if 'rows_count' in _dict:
            args['rows_count'] = _dict.get('rows_count')
        else:
            raise ValueError('Required property \'rows_count\' not present in ResourceBindingsList JSON')
        if 'next_url' in _dict:
            args['next_url'] = _dict.get('next_url')
        else:
            raise ValueError('Required property \'next_url\' not present in ResourceBindingsList JSON')
        if 'resources' in _dict:
            args['resources'] = [ResourceBinding.from_dict(v) for v in _dict.get('resources')]
        else:
            raise ValueError('Required property \'resources\' not present in ResourceBindingsList JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourceBindingsList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'rows_count') and self.rows_count is not None:
            _dict['rows_count'] = self.rows_count
        if hasattr(self, 'next_url') and self.next_url is not None:
            _dict['next_url'] = self.next_url
        if hasattr(self, 'resources') and self.resources is not None:
            resources_list = []
            for v in self.resources:
                if isinstance(v, dict):
                    resources_list.append(v)
                else:
                    resources_list.append(v.to_dict())
            _dict['resources'] = resources_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResourceBindingsList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourceBindingsList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourceBindingsList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ResourceInstance:
    """
    A resource instance.

    :attr str id: (optional) The ID associated with the instance.
    :attr str guid: (optional) The GUID of the instance.
    :attr str url: (optional) When you provision a new resource, a relative URL path
          is created identifying the location of the instance.
    :attr datetime created_at: (optional) The date when the instance was created.
    :attr datetime updated_at: (optional) The date when the instance was last
          updated.
    :attr datetime deleted_at: (optional) The date when the instance was deleted.
    :attr str created_by: (optional) The subject who created the instance.
    :attr str updated_by: (optional) The subject who updated the instance.
    :attr str deleted_by: (optional) The subject who deleted the instance.
    :attr datetime scheduled_reclaim_at: (optional) The date when the instance was
          scheduled for reclamation.
    :attr datetime restored_at: (optional) The date when the instance under
          reclamation was restored.
    :attr str restored_by: (optional) The subject who restored the instance back
          from reclamation.
    :attr str scheduled_reclaim_by: (optional) The subject who initiated the
          instance reclamation.
    :attr str name: (optional) The human-readable name of the instance.
    :attr str region_id: (optional) The deployment location where the instance was
          provisioned.
    :attr str account_id: (optional) An alpha-numeric value identifying the account
          ID.
    :attr str reseller_channel_id: (optional) The unique ID of the reseller channel
          where the instance was provisioned from.
    :attr str resource_plan_id: (optional) The unique ID of the plan associated with
          the offering. This value is provided by and stored in the global catalog.
    :attr str resource_group_id: (optional) The ID of the resource group.
    :attr str resource_group_crn: (optional) The CRN of the resource group.
    :attr str target_crn: (optional) The deployment CRN as defined in the global
          catalog. The Cloud Resource Name (CRN) of the deployment location where the
          instance is provisioned.
    :attr dict parameters: (optional) The current configuration parameters of the
          instance.
    :attr bool allow_cleanup: (optional) A boolean that dictates if the resource
          instance should be deleted (cleaned up) during the processing of a region
          instance delete call.
    :attr str crn: (optional) The full Cloud Resource Name (CRN) associated with the
          instance. For more information about this format, see [Cloud Resource
          Names](https://cloud.ibm.com/docs/overview?topic=overview-crn).
    :attr str state: (optional) The current state of the instance. For example, if
          the instance is deleted, it will return removed.
    :attr str type: (optional) The type of the instance, for example,
          `service_instance`.
    :attr str sub_type: (optional) The sub-type of instance, for example, `cfaas`.
    :attr str resource_id: (optional) The unique ID of the offering. This value is
          provided by and stored in the global catalog.
    :attr str dashboard_url: (optional) The resource-broker-provided URL to access
          administrative features of the instance.
    :attr ResourceInstanceLastOperation last_operation: (optional) The status of the
          last operation requested on the instance.
    :attr str resource_aliases_url: (optional) The relative path to the resource
          aliases for the instance.
    :attr str resource_bindings_url: (optional) The relative path to the resource
          bindings for the instance.
    :attr str resource_keys_url: (optional) The relative path to the resource keys
          for the instance.
    :attr List[PlanHistoryItem] plan_history: (optional) The plan history of the
          instance.
    :attr bool migrated: (optional) A boolean that dictates if the resource instance
          was migrated from a previous CF instance.
    :attr dict extensions: (optional) Additional instance properties, contributed by
          the service and/or platform, are represented as key-value pairs.
    :attr str controlled_by: (optional) The CRN of the resource that has control of
          the instance.
    :attr bool locked: (optional) A boolean that dictates if the resource instance
          is locked or not.
    """

    def __init__(
        self,
        *,
        id: str = None,
        guid: str = None,
        url: str = None,
        created_at: datetime = None,
        updated_at: datetime = None,
        deleted_at: datetime = None,
        created_by: str = None,
        updated_by: str = None,
        deleted_by: str = None,
        scheduled_reclaim_at: datetime = None,
        restored_at: datetime = None,
        restored_by: str = None,
        scheduled_reclaim_by: str = None,
        name: str = None,
        region_id: str = None,
        account_id: str = None,
        reseller_channel_id: str = None,
        resource_plan_id: str = None,
        resource_group_id: str = None,
        resource_group_crn: str = None,
        target_crn: str = None,
        parameters: dict = None,
        allow_cleanup: bool = None,
        crn: str = None,
        state: str = None,
        type: str = None,
        sub_type: str = None,
        resource_id: str = None,
        dashboard_url: str = None,
        last_operation: 'ResourceInstanceLastOperation' = None,
        resource_aliases_url: str = None,
        resource_bindings_url: str = None,
        resource_keys_url: str = None,
        plan_history: List['PlanHistoryItem'] = None,
        migrated: bool = None,
        extensions: dict = None,
        controlled_by: str = None,
        locked: bool = None,
    ) -> None:
        """
        Initialize a ResourceInstance object.

        :param str id: (optional) The ID associated with the instance.
        :param str guid: (optional) The GUID of the instance.
        :param str url: (optional) When you provision a new resource, a relative
               URL path is created identifying the location of the instance.
        :param datetime created_at: (optional) The date when the instance was
               created.
        :param datetime updated_at: (optional) The date when the instance was last
               updated.
        :param datetime deleted_at: (optional) The date when the instance was
               deleted.
        :param str created_by: (optional) The subject who created the instance.
        :param str updated_by: (optional) The subject who updated the instance.
        :param str deleted_by: (optional) The subject who deleted the instance.
        :param datetime scheduled_reclaim_at: (optional) The date when the instance
               was scheduled for reclamation.
        :param datetime restored_at: (optional) The date when the instance under
               reclamation was restored.
        :param str restored_by: (optional) The subject who restored the instance
               back from reclamation.
        :param str scheduled_reclaim_by: (optional) The subject who initiated the
               instance reclamation.
        :param str name: (optional) The human-readable name of the instance.
        :param str region_id: (optional) The deployment location where the instance
               was provisioned.
        :param str account_id: (optional) An alpha-numeric value identifying the
               account ID.
        :param str reseller_channel_id: (optional) The unique ID of the reseller
               channel where the instance was provisioned from.
        :param str resource_plan_id: (optional) The unique ID of the plan
               associated with the offering. This value is provided by and stored in the
               global catalog.
        :param str resource_group_id: (optional) The ID of the resource group.
        :param str resource_group_crn: (optional) The CRN of the resource group.
        :param str target_crn: (optional) The deployment CRN as defined in the
               global catalog. The Cloud Resource Name (CRN) of the deployment location
               where the instance is provisioned.
        :param dict parameters: (optional) The current configuration parameters of
               the instance.
        :param bool allow_cleanup: (optional) A boolean that dictates if the
               resource instance should be deleted (cleaned up) during the processing of a
               region instance delete call.
        :param str crn: (optional) The full Cloud Resource Name (CRN) associated
               with the instance. For more information about this format, see [Cloud
               Resource Names](https://cloud.ibm.com/docs/overview?topic=overview-crn).
        :param str state: (optional) The current state of the instance. For
               example, if the instance is deleted, it will return removed.
        :param str type: (optional) The type of the instance, for example,
               `service_instance`.
        :param str sub_type: (optional) The sub-type of instance, for example,
               `cfaas`.
        :param str resource_id: (optional) The unique ID of the offering. This
               value is provided by and stored in the global catalog.
        :param str dashboard_url: (optional) The resource-broker-provided URL to
               access administrative features of the instance.
        :param ResourceInstanceLastOperation last_operation: (optional) The status
               of the last operation requested on the instance.
        :param str resource_aliases_url: (optional) The relative path to the
               resource aliases for the instance.
        :param str resource_bindings_url: (optional) The relative path to the
               resource bindings for the instance.
        :param str resource_keys_url: (optional) The relative path to the resource
               keys for the instance.
        :param List[PlanHistoryItem] plan_history: (optional) The plan history of
               the instance.
        :param bool migrated: (optional) A boolean that dictates if the resource
               instance was migrated from a previous CF instance.
        :param dict extensions: (optional) Additional instance properties,
               contributed by the service and/or platform, are represented as key-value
               pairs.
        :param str controlled_by: (optional) The CRN of the resource that has
               control of the instance.
        :param bool locked: (optional) A boolean that dictates if the resource
               instance is locked or not.
        """
        self.id = id
        self.guid = guid
        self.url = url
        self.created_at = created_at
        self.updated_at = updated_at
        self.deleted_at = deleted_at
        self.created_by = created_by
        self.updated_by = updated_by
        self.deleted_by = deleted_by
        self.scheduled_reclaim_at = scheduled_reclaim_at
        self.restored_at = restored_at
        self.restored_by = restored_by
        self.scheduled_reclaim_by = scheduled_reclaim_by
        self.name = name
        self.region_id = region_id
        self.account_id = account_id
        self.reseller_channel_id = reseller_channel_id
        self.resource_plan_id = resource_plan_id
        self.resource_group_id = resource_group_id
        self.resource_group_crn = resource_group_crn
        self.target_crn = target_crn
        self.parameters = parameters
        self.allow_cleanup = allow_cleanup
        self.crn = crn
        self.state = state
        self.type = type
        self.sub_type = sub_type
        self.resource_id = resource_id
        self.dashboard_url = dashboard_url
        self.last_operation = last_operation
        self.resource_aliases_url = resource_aliases_url
        self.resource_bindings_url = resource_bindings_url
        self.resource_keys_url = resource_keys_url
        self.plan_history = plan_history
        self.migrated = migrated
        self.extensions = extensions
        self.controlled_by = controlled_by
        self.locked = locked

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourceInstance':
        """Initialize a ResourceInstance object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'guid' in _dict:
            args['guid'] = _dict.get('guid')
        if 'url' in _dict:
            args['url'] = _dict.get('url')
        if 'created_at' in _dict:
            args['created_at'] = string_to_datetime(_dict.get('created_at'))
        if 'updated_at' in _dict:
            args['updated_at'] = string_to_datetime(_dict.get('updated_at'))
        if 'deleted_at' in _dict:
            args['deleted_at'] = string_to_datetime(_dict.get('deleted_at'))
        if 'created_by' in _dict:
            args['created_by'] = _dict.get('created_by')
        if 'updated_by' in _dict:
            args['updated_by'] = _dict.get('updated_by')
        if 'deleted_by' in _dict:
            args['deleted_by'] = _dict.get('deleted_by')
        if 'scheduled_reclaim_at' in _dict:
            args['scheduled_reclaim_at'] = string_to_datetime(_dict.get('scheduled_reclaim_at'))
        if 'restored_at' in _dict:
            args['restored_at'] = string_to_datetime(_dict.get('restored_at'))
        if 'restored_by' in _dict:
            args['restored_by'] = _dict.get('restored_by')
        if 'scheduled_reclaim_by' in _dict:
            args['scheduled_reclaim_by'] = _dict.get('scheduled_reclaim_by')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'region_id' in _dict:
            args['region_id'] = _dict.get('region_id')
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        if 'reseller_channel_id' in _dict:
            args['reseller_channel_id'] = _dict.get('reseller_channel_id')
        if 'resource_plan_id' in _dict:
            args['resource_plan_id'] = _dict.get('resource_plan_id')
        if 'resource_group_id' in _dict:
            args['resource_group_id'] = _dict.get('resource_group_id')
        if 'resource_group_crn' in _dict:
            args['resource_group_crn'] = _dict.get('resource_group_crn')
        if 'target_crn' in _dict:
            args['target_crn'] = _dict.get('target_crn')
        if 'parameters' in _dict:
            args['parameters'] = _dict.get('parameters')
        if 'allow_cleanup' in _dict:
            args['allow_cleanup'] = _dict.get('allow_cleanup')
        if 'crn' in _dict:
            args['crn'] = _dict.get('crn')
        if 'state' in _dict:
            args['state'] = _dict.get('state')
        if 'type' in _dict:
            args['type'] = _dict.get('type')
        if 'sub_type' in _dict:
            args['sub_type'] = _dict.get('sub_type')
        if 'resource_id' in _dict:
            args['resource_id'] = _dict.get('resource_id')
        if 'dashboard_url' in _dict:
            args['dashboard_url'] = _dict.get('dashboard_url')
        if 'last_operation' in _dict:
            args['last_operation'] = ResourceInstanceLastOperation.from_dict(_dict.get('last_operation'))
        if 'resource_aliases_url' in _dict:
            args['resource_aliases_url'] = _dict.get('resource_aliases_url')
        if 'resource_bindings_url' in _dict:
            args['resource_bindings_url'] = _dict.get('resource_bindings_url')
        if 'resource_keys_url' in _dict:
            args['resource_keys_url'] = _dict.get('resource_keys_url')
        if 'plan_history' in _dict:
            args['plan_history'] = [PlanHistoryItem.from_dict(v) for v in _dict.get('plan_history')]
        if 'migrated' in _dict:
            args['migrated'] = _dict.get('migrated')
        if 'extensions' in _dict:
            args['extensions'] = _dict.get('extensions')
        if 'controlled_by' in _dict:
            args['controlled_by'] = _dict.get('controlled_by')
        if 'locked' in _dict:
            args['locked'] = _dict.get('locked')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourceInstance object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'guid') and self.guid is not None:
            _dict['guid'] = self.guid
        if hasattr(self, 'url') and self.url is not None:
            _dict['url'] = self.url
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'updated_at') and self.updated_at is not None:
            _dict['updated_at'] = datetime_to_string(self.updated_at)
        if hasattr(self, 'deleted_at') and self.deleted_at is not None:
            _dict['deleted_at'] = datetime_to_string(self.deleted_at)
        if hasattr(self, 'created_by') and self.created_by is not None:
            _dict['created_by'] = self.created_by
        if hasattr(self, 'updated_by') and self.updated_by is not None:
            _dict['updated_by'] = self.updated_by
        if hasattr(self, 'deleted_by') and self.deleted_by is not None:
            _dict['deleted_by'] = self.deleted_by
        if hasattr(self, 'scheduled_reclaim_at') and self.scheduled_reclaim_at is not None:
            _dict['scheduled_reclaim_at'] = datetime_to_string(self.scheduled_reclaim_at)
        if hasattr(self, 'restored_at') and self.restored_at is not None:
            _dict['restored_at'] = datetime_to_string(self.restored_at)
        if hasattr(self, 'restored_by') and self.restored_by is not None:
            _dict['restored_by'] = self.restored_by
        if hasattr(self, 'scheduled_reclaim_by') and self.scheduled_reclaim_by is not None:
            _dict['scheduled_reclaim_by'] = self.scheduled_reclaim_by
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'region_id') and self.region_id is not None:
            _dict['region_id'] = self.region_id
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'reseller_channel_id') and self.reseller_channel_id is not None:
            _dict['reseller_channel_id'] = self.reseller_channel_id
        if hasattr(self, 'resource_plan_id') and self.resource_plan_id is not None:
            _dict['resource_plan_id'] = self.resource_plan_id
        if hasattr(self, 'resource_group_id') and self.resource_group_id is not None:
            _dict['resource_group_id'] = self.resource_group_id
        if hasattr(self, 'resource_group_crn') and self.resource_group_crn is not None:
            _dict['resource_group_crn'] = self.resource_group_crn
        if hasattr(self, 'target_crn') and self.target_crn is not None:
            _dict['target_crn'] = self.target_crn
        if hasattr(self, 'parameters') and self.parameters is not None:
            _dict['parameters'] = self.parameters
        if hasattr(self, 'allow_cleanup') and self.allow_cleanup is not None:
            _dict['allow_cleanup'] = self.allow_cleanup
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'sub_type') and self.sub_type is not None:
            _dict['sub_type'] = self.sub_type
        if hasattr(self, 'resource_id') and self.resource_id is not None:
            _dict['resource_id'] = self.resource_id
        if hasattr(self, 'dashboard_url') and self.dashboard_url is not None:
            _dict['dashboard_url'] = self.dashboard_url
        if hasattr(self, 'last_operation') and self.last_operation is not None:
            if isinstance(self.last_operation, dict):
                _dict['last_operation'] = self.last_operation
            else:
                _dict['last_operation'] = self.last_operation.to_dict()
        if hasattr(self, 'resource_aliases_url') and self.resource_aliases_url is not None:
            _dict['resource_aliases_url'] = self.resource_aliases_url
        if hasattr(self, 'resource_bindings_url') and self.resource_bindings_url is not None:
            _dict['resource_bindings_url'] = self.resource_bindings_url
        if hasattr(self, 'resource_keys_url') and self.resource_keys_url is not None:
            _dict['resource_keys_url'] = self.resource_keys_url
        if hasattr(self, 'plan_history') and self.plan_history is not None:
            plan_history_list = []
            for v in self.plan_history:
                if isinstance(v, dict):
                    plan_history_list.append(v)
                else:
                    plan_history_list.append(v.to_dict())
            _dict['plan_history'] = plan_history_list
        if hasattr(self, 'migrated') and self.migrated is not None:
            _dict['migrated'] = self.migrated
        if hasattr(self, 'extensions') and self.extensions is not None:
            _dict['extensions'] = self.extensions
        if hasattr(self, 'controlled_by') and self.controlled_by is not None:
            _dict['controlled_by'] = self.controlled_by
        if hasattr(self, 'locked') and self.locked is not None:
            _dict['locked'] = self.locked
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResourceInstance object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourceInstance') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourceInstance') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class StateEnum(str, Enum):
        """
        The current state of the instance. For example, if the instance is deleted, it
        will return removed.
        """

        ACTIVE = 'active'
        INACTIVE = 'inactive'
        REMOVED = 'removed'
        PENDING_REMOVAL = 'pending_removal'
        PENDING_RECLAMATION = 'pending_reclamation'
        FAILED = 'failed'
        PROVISIONING = 'provisioning'
        PRE_PROVISIONING = 'pre_provisioning'


class ResourceInstanceLastOperation:
    """
    The status of the last operation requested on the instance.

    :attr str type: The last operation type of the resource instance.
    :attr str state: The last operation state of the resoure instance. This
          indicates if the resource's last operation is in progress, succeeded or failed.
    :attr str sub_type: (optional) The last operation sub type of the resoure
          instance.
    :attr bool async_: A boolean that indicates if the resource is provisioned
          asynchronously or not.
    :attr str description: The description of the status of last operation.
    :attr str reason_code: (optional) Optional string that states the reason code
          for the last operation state change.
    :attr float poll_after: (optional) A field which indicates the time after which
          the instance's last operation is to be polled.
    :attr bool cancelable: A boolean that indicates if the resource's last operation
          is cancelable or not.
    :attr bool poll: A boolean that indicates if the resource broker's last
          operation can be polled or not.
    """

    # The set of defined properties for the class
    _properties = frozenset(
        [
            'type',
            'state',
            'sub_type',
            'async_',
            'async',
            'description',
            'reason_code',
            'poll_after',
            'cancelable',
            'poll',
        ]
    )

    def __init__(
        self,
        type: str,
        state: str,
        async_: bool,
        description: str,
        cancelable: bool,
        poll: bool,
        *,
        sub_type: str = None,
        reason_code: str = None,
        poll_after: float = None,
        **kwargs,
    ) -> None:
        """
        Initialize a ResourceInstanceLastOperation object.

        :param str type: The last operation type of the resource instance.
        :param str state: The last operation state of the resoure instance. This
               indicates if the resource's last operation is in progress, succeeded or
               failed.
        :param bool async_: A boolean that indicates if the resource is provisioned
               asynchronously or not.
        :param str description: The description of the status of last operation.
        :param bool cancelable: A boolean that indicates if the resource's last
               operation is cancelable or not.
        :param bool poll: A boolean that indicates if the resource broker's last
               operation can be polled or not.
        :param str sub_type: (optional) The last operation sub type of the resoure
               instance.
        :param str reason_code: (optional) Optional string that states the reason
               code for the last operation state change.
        :param float poll_after: (optional) A field which indicates the time after
               which the instance's last operation is to be polled.
        :param **kwargs: (optional) Any additional properties.
        """
        self.type = type
        self.state = state
        self.sub_type = sub_type
        self.async_ = async_
        self.description = description
        self.reason_code = reason_code
        self.poll_after = poll_after
        self.cancelable = cancelable
        self.poll = poll
        for _key, _value in kwargs.items():
            setattr(self, _key, _value)

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourceInstanceLastOperation':
        """Initialize a ResourceInstanceLastOperation object from a json dictionary."""
        args = {}
        if 'type' in _dict:
            args['type'] = _dict.get('type')
        else:
            raise ValueError('Required property \'type\' not present in ResourceInstanceLastOperation JSON')
        if 'state' in _dict:
            args['state'] = _dict.get('state')
        else:
            raise ValueError('Required property \'state\' not present in ResourceInstanceLastOperation JSON')
        if 'sub_type' in _dict:
            args['sub_type'] = _dict.get('sub_type')
        if 'async' in _dict:
            args['async_'] = _dict.get('async')
        else:
            raise ValueError('Required property \'async\' not present in ResourceInstanceLastOperation JSON')
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        else:
            raise ValueError('Required property \'description\' not present in ResourceInstanceLastOperation JSON')
        if 'reason_code' in _dict:
            args['reason_code'] = _dict.get('reason_code')
        if 'poll_after' in _dict:
            args['poll_after'] = _dict.get('poll_after')
        if 'cancelable' in _dict:
            args['cancelable'] = _dict.get('cancelable')
        else:
            raise ValueError('Required property \'cancelable\' not present in ResourceInstanceLastOperation JSON')
        if 'poll' in _dict:
            args['poll'] = _dict.get('poll')
        else:
            raise ValueError('Required property \'poll\' not present in ResourceInstanceLastOperation JSON')
        args.update({k: v for (k, v) in _dict.items() if k not in cls._properties})
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourceInstanceLastOperation object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        if hasattr(self, 'sub_type') and self.sub_type is not None:
            _dict['sub_type'] = self.sub_type
        if hasattr(self, 'async_') and self.async_ is not None:
            _dict['async'] = self.async_
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'reason_code') and self.reason_code is not None:
            _dict['reason_code'] = self.reason_code
        if hasattr(self, 'poll_after') and self.poll_after is not None:
            _dict['poll_after'] = self.poll_after
        if hasattr(self, 'cancelable') and self.cancelable is not None:
            _dict['cancelable'] = self.cancelable
        if hasattr(self, 'poll') and self.poll is not None:
            _dict['poll'] = self.poll
        for _key in [k for k in vars(self).keys() if k not in ResourceInstanceLastOperation._properties]:
            _dict[_key] = getattr(self, _key)
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def get_properties(self) -> Dict:
        """Return a dictionary of arbitrary properties from this instance of ResourceInstanceLastOperation"""
        _dict = {}

        for _key in [k for k in vars(self).keys() if k not in ResourceInstanceLastOperation._properties]:
            _dict[_key] = getattr(self, _key)
        return _dict

    def set_properties(self, _dict: dict):
        """Set a dictionary of arbitrary properties to this instance of ResourceInstanceLastOperation"""
        for _key in [k for k in vars(self).keys() if k not in ResourceInstanceLastOperation._properties]:
            delattr(self, _key)

        for _key, _value in _dict.items():
            if _key not in ResourceInstanceLastOperation._properties:
                setattr(self, _key, _value)

    def __str__(self) -> str:
        """Return a `str` version of this ResourceInstanceLastOperation object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourceInstanceLastOperation') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourceInstanceLastOperation') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class StateEnum(str, Enum):
        """
        The last operation state of the resoure instance. This indicates if the resource's
        last operation is in progress, succeeded or failed.
        """

        IN_PROGRESS = 'in progress'
        SUCCEEDED = 'succeeded'
        FAILED = 'failed'


class ResourceInstancesList:
    """
    A list of resource instances.

    :attr int rows_count: The number of resource instances in `resources`.
    :attr str next_url: The URL for requesting the next page of results.
    :attr List[ResourceInstance] resources: A list of resource instances.
    """

    def __init__(self, rows_count: int, next_url: str, resources: List['ResourceInstance']) -> None:
        """
        Initialize a ResourceInstancesList object.

        :param int rows_count: The number of resource instances in `resources`.
        :param str next_url: The URL for requesting the next page of results.
        :param List[ResourceInstance] resources: A list of resource instances.
        """
        self.rows_count = rows_count
        self.next_url = next_url
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourceInstancesList':
        """Initialize a ResourceInstancesList object from a json dictionary."""
        args = {}
        if 'rows_count' in _dict:
            args['rows_count'] = _dict.get('rows_count')
        else:
            raise ValueError('Required property \'rows_count\' not present in ResourceInstancesList JSON')
        if 'next_url' in _dict:
            args['next_url'] = _dict.get('next_url')
        else:
            raise ValueError('Required property \'next_url\' not present in ResourceInstancesList JSON')
        if 'resources' in _dict:
            args['resources'] = [ResourceInstance.from_dict(v) for v in _dict.get('resources')]
        else:
            raise ValueError('Required property \'resources\' not present in ResourceInstancesList JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourceInstancesList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'rows_count') and self.rows_count is not None:
            _dict['rows_count'] = self.rows_count
        if hasattr(self, 'next_url') and self.next_url is not None:
            _dict['next_url'] = self.next_url
        if hasattr(self, 'resources') and self.resources is not None:
            resources_list = []
            for v in self.resources:
                if isinstance(v, dict):
                    resources_list.append(v)
                else:
                    resources_list.append(v.to_dict())
            _dict['resources'] = resources_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResourceInstancesList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourceInstancesList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourceInstancesList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ResourceKey:
    """
    A resource key.

    :attr str id: (optional) The ID associated with the key.
    :attr str guid: (optional) The GUID of the key.
    :attr str url: (optional) When you created a new key, a relative URL path is
          created identifying the location of the key.
    :attr datetime created_at: (optional) The date when the key was created.
    :attr datetime updated_at: (optional) The date when the key was last updated.
    :attr datetime deleted_at: (optional) The date when the key was deleted.
    :attr str created_by: (optional) The subject who created the key.
    :attr str updated_by: (optional) The subject who updated the key.
    :attr str deleted_by: (optional) The subject who deleted the key.
    :attr str source_crn: (optional) The CRN of resource instance or alias
          associated to the key.
    :attr str name: (optional) The human-readable name of the key.
    :attr str crn: (optional) The full Cloud Resource Name (CRN) associated with the
          key. For more information about this format, see [Cloud Resource
          Names](https://cloud.ibm.com/docs/overview?topic=overview-crn).
    :attr str state: (optional) The state of the key.
    :attr str account_id: (optional) An alpha-numeric value identifying the account
          ID.
    :attr str resource_group_id: (optional) The ID of the resource group.
    :attr str resource_id: (optional) The unique ID of the offering. This value is
          provided by and stored in the global catalog.
    :attr Credentials credentials: (optional) The credentials for the key.
          Additional key-value pairs are passed through from the resource brokers. After a
          credential is created for a service, it can be viewed at any time for users that
          need the API key value. However, all users must have the correct level of access
          to see the details of a credential that includes the API key value. For
          additional details, see [viewing a
          credential](https://cloud.ibm.com/docs/account?topic=account-service_credentials&interface=ui#viewing-credentials-ui)
          or the service’s documentation.
    :attr bool iam_compatible: (optional) Specifies whether the key’s credentials
          support IAM.
    :attr bool migrated: (optional) A boolean that dictates if the alias was
          migrated from a previous CF instance.
    :attr str resource_instance_url: (optional) The relative path to the resource.
    :attr str resource_alias_url: (optional) The relative path to the resource alias
          that this binding is associated with.
    """

    def __init__(
        self,
        *,
        id: str = None,
        guid: str = None,
        url: str = None,
        created_at: datetime = None,
        updated_at: datetime = None,
        deleted_at: datetime = None,
        created_by: str = None,
        updated_by: str = None,
        deleted_by: str = None,
        source_crn: str = None,
        name: str = None,
        crn: str = None,
        state: str = None,
        account_id: str = None,
        resource_group_id: str = None,
        resource_id: str = None,
        credentials: 'Credentials' = None,
        iam_compatible: bool = None,
        migrated: bool = None,
        resource_instance_url: str = None,
        resource_alias_url: str = None,
    ) -> None:
        """
        Initialize a ResourceKey object.

        :param str id: (optional) The ID associated with the key.
        :param str guid: (optional) The GUID of the key.
        :param str url: (optional) When you created a new key, a relative URL path
               is created identifying the location of the key.
        :param datetime created_at: (optional) The date when the key was created.
        :param datetime updated_at: (optional) The date when the key was last
               updated.
        :param datetime deleted_at: (optional) The date when the key was deleted.
        :param str created_by: (optional) The subject who created the key.
        :param str updated_by: (optional) The subject who updated the key.
        :param str deleted_by: (optional) The subject who deleted the key.
        :param str source_crn: (optional) The CRN of resource instance or alias
               associated to the key.
        :param str name: (optional) The human-readable name of the key.
        :param str crn: (optional) The full Cloud Resource Name (CRN) associated
               with the key. For more information about this format, see [Cloud Resource
               Names](https://cloud.ibm.com/docs/overview?topic=overview-crn).
        :param str state: (optional) The state of the key.
        :param str account_id: (optional) An alpha-numeric value identifying the
               account ID.
        :param str resource_group_id: (optional) The ID of the resource group.
        :param str resource_id: (optional) The unique ID of the offering. This
               value is provided by and stored in the global catalog.
        :param Credentials credentials: (optional) The credentials for the key.
               Additional key-value pairs are passed through from the resource brokers.
               After a credential is created for a service, it can be viewed at any time
               for users that need the API key value. However, all users must have the
               correct level of access to see the details of a credential that includes
               the API key value. For additional details, see [viewing a
               credential](https://cloud.ibm.com/docs/account?topic=account-service_credentials&interface=ui#viewing-credentials-ui)
               or the service’s documentation.
        :param bool iam_compatible: (optional) Specifies whether the key’s
               credentials support IAM.
        :param bool migrated: (optional) A boolean that dictates if the alias was
               migrated from a previous CF instance.
        :param str resource_instance_url: (optional) The relative path to the
               resource.
        :param str resource_alias_url: (optional) The relative path to the resource
               alias that this binding is associated with.
        """
        self.id = id
        self.guid = guid
        self.url = url
        self.created_at = created_at
        self.updated_at = updated_at
        self.deleted_at = deleted_at
        self.created_by = created_by
        self.updated_by = updated_by
        self.deleted_by = deleted_by
        self.source_crn = source_crn
        self.name = name
        self.crn = crn
        self.state = state
        self.account_id = account_id
        self.resource_group_id = resource_group_id
        self.resource_id = resource_id
        self.credentials = credentials
        self.iam_compatible = iam_compatible
        self.migrated = migrated
        self.resource_instance_url = resource_instance_url
        self.resource_alias_url = resource_alias_url

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourceKey':
        """Initialize a ResourceKey object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'guid' in _dict:
            args['guid'] = _dict.get('guid')
        if 'url' in _dict:
            args['url'] = _dict.get('url')
        if 'created_at' in _dict:
            args['created_at'] = string_to_datetime(_dict.get('created_at'))
        if 'updated_at' in _dict:
            args['updated_at'] = string_to_datetime(_dict.get('updated_at'))
        if 'deleted_at' in _dict:
            args['deleted_at'] = string_to_datetime(_dict.get('deleted_at'))
        if 'created_by' in _dict:
            args['created_by'] = _dict.get('created_by')
        if 'updated_by' in _dict:
            args['updated_by'] = _dict.get('updated_by')
        if 'deleted_by' in _dict:
            args['deleted_by'] = _dict.get('deleted_by')
        if 'source_crn' in _dict:
            args['source_crn'] = _dict.get('source_crn')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'crn' in _dict:
            args['crn'] = _dict.get('crn')
        if 'state' in _dict:
            args['state'] = _dict.get('state')
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        if 'resource_group_id' in _dict:
            args['resource_group_id'] = _dict.get('resource_group_id')
        if 'resource_id' in _dict:
            args['resource_id'] = _dict.get('resource_id')
        if 'credentials' in _dict:
            args['credentials'] = Credentials.from_dict(_dict.get('credentials'))
        if 'iam_compatible' in _dict:
            args['iam_compatible'] = _dict.get('iam_compatible')
        if 'migrated' in _dict:
            args['migrated'] = _dict.get('migrated')
        if 'resource_instance_url' in _dict:
            args['resource_instance_url'] = _dict.get('resource_instance_url')
        if 'resource_alias_url' in _dict:
            args['resource_alias_url'] = _dict.get('resource_alias_url')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourceKey object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'guid') and self.guid is not None:
            _dict['guid'] = self.guid
        if hasattr(self, 'url') and self.url is not None:
            _dict['url'] = self.url
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'updated_at') and self.updated_at is not None:
            _dict['updated_at'] = datetime_to_string(self.updated_at)
        if hasattr(self, 'deleted_at') and self.deleted_at is not None:
            _dict['deleted_at'] = datetime_to_string(self.deleted_at)
        if hasattr(self, 'created_by') and self.created_by is not None:
            _dict['created_by'] = self.created_by
        if hasattr(self, 'updated_by') and self.updated_by is not None:
            _dict['updated_by'] = self.updated_by
        if hasattr(self, 'deleted_by') and self.deleted_by is not None:
            _dict['deleted_by'] = self.deleted_by
        if hasattr(self, 'source_crn') and self.source_crn is not None:
            _dict['source_crn'] = self.source_crn
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'resource_group_id') and self.resource_group_id is not None:
            _dict['resource_group_id'] = self.resource_group_id
        if hasattr(self, 'resource_id') and self.resource_id is not None:
            _dict['resource_id'] = self.resource_id
        if hasattr(self, 'credentials') and self.credentials is not None:
            if isinstance(self.credentials, dict):
                _dict['credentials'] = self.credentials
            else:
                _dict['credentials'] = self.credentials.to_dict()
        if hasattr(self, 'iam_compatible') and self.iam_compatible is not None:
            _dict['iam_compatible'] = self.iam_compatible
        if hasattr(self, 'migrated') and self.migrated is not None:
            _dict['migrated'] = self.migrated
        if hasattr(self, 'resource_instance_url') and self.resource_instance_url is not None:
            _dict['resource_instance_url'] = self.resource_instance_url
        if hasattr(self, 'resource_alias_url') and self.resource_alias_url is not None:
            _dict['resource_alias_url'] = self.resource_alias_url
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResourceKey object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourceKey') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourceKey') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ResourceKeyPostParameters:
    """
    Configuration options represented as key-value pairs. Service defined options are
    passed through to the target resource brokers, whereas platform defined options are
    not.

    :attr str serviceid_crn: (optional) An optional platform defined option to reuse
          an existing IAM serviceId for the role assignment.
    """

    # The set of defined properties for the class
    _properties = frozenset(['serviceid_crn'])

    def __init__(self, *, serviceid_crn: str = None, **kwargs) -> None:
        """
        Initialize a ResourceKeyPostParameters object.

        :param str serviceid_crn: (optional) An optional platform defined option to
               reuse an existing IAM serviceId for the role assignment.
        :param **kwargs: (optional) Any additional properties.
        """
        self.serviceid_crn = serviceid_crn
        for _key, _value in kwargs.items():
            setattr(self, _key, _value)

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourceKeyPostParameters':
        """Initialize a ResourceKeyPostParameters object from a json dictionary."""
        args = {}
        if 'serviceid_crn' in _dict:
            args['serviceid_crn'] = _dict.get('serviceid_crn')
        args.update({k: v for (k, v) in _dict.items() if k not in cls._properties})
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourceKeyPostParameters object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'serviceid_crn') and self.serviceid_crn is not None:
            _dict['serviceid_crn'] = self.serviceid_crn
        for _key in [k for k in vars(self).keys() if k not in ResourceKeyPostParameters._properties]:
            _dict[_key] = getattr(self, _key)
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def get_properties(self) -> Dict:
        """Return a dictionary of arbitrary properties from this instance of ResourceKeyPostParameters"""
        _dict = {}

        for _key in [k for k in vars(self).keys() if k not in ResourceKeyPostParameters._properties]:
            _dict[_key] = getattr(self, _key)
        return _dict

    def set_properties(self, _dict: dict):
        """Set a dictionary of arbitrary properties to this instance of ResourceKeyPostParameters"""
        for _key in [k for k in vars(self).keys() if k not in ResourceKeyPostParameters._properties]:
            delattr(self, _key)

        for _key, _value in _dict.items():
            if _key not in ResourceKeyPostParameters._properties:
                setattr(self, _key, _value)

    def __str__(self) -> str:
        """Return a `str` version of this ResourceKeyPostParameters object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourceKeyPostParameters') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourceKeyPostParameters') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ResourceKeysList:
    """
    A list of resource keys.

    :attr int rows_count: The number of resource keys in `resources`.
    :attr str next_url: The URL for requesting the next page of results.
    :attr List[ResourceKey] resources: A list of resource keys.
    """

    def __init__(self, rows_count: int, next_url: str, resources: List['ResourceKey']) -> None:
        """
        Initialize a ResourceKeysList object.

        :param int rows_count: The number of resource keys in `resources`.
        :param str next_url: The URL for requesting the next page of results.
        :param List[ResourceKey] resources: A list of resource keys.
        """
        self.rows_count = rows_count
        self.next_url = next_url
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourceKeysList':
        """Initialize a ResourceKeysList object from a json dictionary."""
        args = {}
        if 'rows_count' in _dict:
            args['rows_count'] = _dict.get('rows_count')
        else:
            raise ValueError('Required property \'rows_count\' not present in ResourceKeysList JSON')
        if 'next_url' in _dict:
            args['next_url'] = _dict.get('next_url')
        else:
            raise ValueError('Required property \'next_url\' not present in ResourceKeysList JSON')
        if 'resources' in _dict:
            args['resources'] = [ResourceKey.from_dict(v) for v in _dict.get('resources')]
        else:
            raise ValueError('Required property \'resources\' not present in ResourceKeysList JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourceKeysList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'rows_count') and self.rows_count is not None:
            _dict['rows_count'] = self.rows_count
        if hasattr(self, 'next_url') and self.next_url is not None:
            _dict['next_url'] = self.next_url
        if hasattr(self, 'resources') and self.resources is not None:
            resources_list = []
            for v in self.resources:
                if isinstance(v, dict):
                    resources_list.append(v)
                else:
                    resources_list.append(v.to_dict())
            _dict['resources'] = resources_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResourceKeysList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourceKeysList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourceKeysList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


##############################################################################
# Pagers
##############################################################################


class ResourceInstancesPager:
    """
    ResourceInstancesPager can be used to simplify the use of the "list_resource_instances" method.
    """

    def __init__(
        self,
        *,
        client: ResourceControllerV2,
        guid: str = None,
        name: str = None,
        resource_group_id: str = None,
        resource_id: str = None,
        resource_plan_id: str = None,
        type: str = None,
        sub_type: str = None,
        limit: int = None,
        state: str = None,
        updated_from: str = None,
        updated_to: str = None,
    ) -> None:
        """
        Initialize a ResourceInstancesPager object.
        :param str guid: (optional) The GUID of the instance.
        :param str name: (optional) The human-readable name of the instance.
        :param str resource_group_id: (optional) The ID of the resource group.
        :param str resource_id: (optional) The unique ID of the offering. This
               value is provided by and stored in the global catalog.
        :param str resource_plan_id: (optional) The unique ID of the plan
               associated with the offering. This value is provided by and stored in the
               global catalog.
        :param str type: (optional) The type of the instance, for example,
               `service_instance`.
        :param str sub_type: (optional) The sub-type of instance, for example,
               `kms`.
        :param int limit: (optional) Limit on how many items should be returned.
        :param str state: (optional) The state of the instance. If not specified,
               instances in state `active` and `provisioning` are returned.
        :param str updated_from: (optional) Start date inclusive filter.
        :param str updated_to: (optional) End date inclusive filter.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._guid = guid
        self._name = name
        self._resource_group_id = resource_group_id
        self._resource_id = resource_id
        self._resource_plan_id = resource_plan_id
        self._type = type
        self._sub_type = sub_type
        self._limit = limit
        self._state = state
        self._updated_from = updated_from
        self._updated_to = updated_to

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of ResourceInstance.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.list_resource_instances(
            guid=self._guid,
            name=self._name,
            resource_group_id=self._resource_group_id,
            resource_id=self._resource_id,
            resource_plan_id=self._resource_plan_id,
            type=self._type,
            sub_type=self._sub_type,
            limit=self._limit,
            state=self._state,
            updated_from=self._updated_from,
            updated_to=self._updated_to,
            start=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next_url')
        if next_page_link is not None:
            next = get_query_param(next_page_link, 'start')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('resources')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of ResourceInstance.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results


class ResourceAliasesForInstancePager:
    """
    ResourceAliasesForInstancePager can be used to simplify the use of the "list_resource_aliases_for_instance" method.
    """

    def __init__(
        self,
        *,
        client: ResourceControllerV2,
        id: str,
        limit: int = None,
    ) -> None:
        """
        Initialize a ResourceAliasesForInstancePager object.
        :param str id: The resource instance URL-encoded CRN or GUID.
        :param int limit: (optional) Limit on how many items should be returned.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._id = id
        self._limit = limit

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of ResourceAlias.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.list_resource_aliases_for_instance(
            id=self._id,
            limit=self._limit,
            start=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next_url')
        if next_page_link is not None:
            next = get_query_param(next_page_link, 'start')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('resources')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of ResourceAlias.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results


class ResourceKeysForInstancePager:
    """
    ResourceKeysForInstancePager can be used to simplify the use of the "list_resource_keys_for_instance" method.
    """

    def __init__(
        self,
        *,
        client: ResourceControllerV2,
        id: str,
        limit: int = None,
    ) -> None:
        """
        Initialize a ResourceKeysForInstancePager object.
        :param str id: The resource instance URL-encoded CRN or GUID.
        :param int limit: (optional) Limit on how many items should be returned.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._id = id
        self._limit = limit

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of ResourceKey.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.list_resource_keys_for_instance(
            id=self._id,
            limit=self._limit,
            start=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next_url')
        if next_page_link is not None:
            next = get_query_param(next_page_link, 'start')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('resources')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of ResourceKey.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results


class ResourceKeysPager:
    """
    ResourceKeysPager can be used to simplify the use of the "list_resource_keys" method.
    """

    def __init__(
        self,
        *,
        client: ResourceControllerV2,
        guid: str = None,
        name: str = None,
        resource_group_id: str = None,
        resource_id: str = None,
        limit: int = None,
        updated_from: str = None,
        updated_to: str = None,
    ) -> None:
        """
        Initialize a ResourceKeysPager object.
        :param str guid: (optional) The GUID of the key.
        :param str name: (optional) The human-readable name of the key.
        :param str resource_group_id: (optional) The ID of the resource group.
        :param str resource_id: (optional) The unique ID of the offering. This
               value is provided by and stored in the global catalog.
        :param int limit: (optional) Limit on how many items should be returned.
        :param str updated_from: (optional) Start date inclusive filter.
        :param str updated_to: (optional) End date inclusive filter.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._guid = guid
        self._name = name
        self._resource_group_id = resource_group_id
        self._resource_id = resource_id
        self._limit = limit
        self._updated_from = updated_from
        self._updated_to = updated_to

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of ResourceKey.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.list_resource_keys(
            guid=self._guid,
            name=self._name,
            resource_group_id=self._resource_group_id,
            resource_id=self._resource_id,
            limit=self._limit,
            updated_from=self._updated_from,
            updated_to=self._updated_to,
            start=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next_url')
        if next_page_link is not None:
            next = get_query_param(next_page_link, 'start')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('resources')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of ResourceKey.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results


class ResourceBindingsPager:
    """
    ResourceBindingsPager can be used to simplify the use of the "list_resource_bindings" method.
    """

    def __init__(
        self,
        *,
        client: ResourceControllerV2,
        guid: str = None,
        name: str = None,
        resource_group_id: str = None,
        resource_id: str = None,
        region_binding_id: str = None,
        limit: int = None,
        updated_from: str = None,
        updated_to: str = None,
    ) -> None:
        """
        Initialize a ResourceBindingsPager object.
        :param str guid: (optional) The GUID of the binding.
        :param str name: (optional) The human-readable name of the binding.
        :param str resource_group_id: (optional) The ID of the resource group.
        :param str resource_id: (optional) The unique ID of the offering (service
               name). This value is provided by and stored in the global catalog.
        :param str region_binding_id: (optional) The ID of the binding in the
               target environment. For example, `service_binding_id` in a given IBM Cloud
               environment.
        :param int limit: (optional) Limit on how many items should be returned.
        :param str updated_from: (optional) Start date inclusive filter.
        :param str updated_to: (optional) End date inclusive filter.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._guid = guid
        self._name = name
        self._resource_group_id = resource_group_id
        self._resource_id = resource_id
        self._region_binding_id = region_binding_id
        self._limit = limit
        self._updated_from = updated_from
        self._updated_to = updated_to

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of ResourceBinding.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.list_resource_bindings(
            guid=self._guid,
            name=self._name,
            resource_group_id=self._resource_group_id,
            resource_id=self._resource_id,
            region_binding_id=self._region_binding_id,
            limit=self._limit,
            updated_from=self._updated_from,
            updated_to=self._updated_to,
            start=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next_url')
        if next_page_link is not None:
            next = get_query_param(next_page_link, 'start')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('resources')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of ResourceBinding.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results


class ResourceAliasesPager:
    """
    ResourceAliasesPager can be used to simplify the use of the "list_resource_aliases" method.
    """

    def __init__(
        self,
        *,
        client: ResourceControllerV2,
        guid: str = None,
        name: str = None,
        resource_instance_id: str = None,
        region_instance_id: str = None,
        resource_id: str = None,
        resource_group_id: str = None,
        limit: int = None,
        updated_from: str = None,
        updated_to: str = None,
    ) -> None:
        """
        Initialize a ResourceAliasesPager object.
        :param str guid: (optional) The GUID of the alias.
        :param str name: (optional) The human-readable name of the alias.
        :param str resource_instance_id: (optional) The ID of the resource
               instance.
        :param str region_instance_id: (optional) The ID of the instance in the
               target environment. For example, `service_instance_id` in a given IBM Cloud
               environment.
        :param str resource_id: (optional) The unique ID of the offering (service
               name). This value is provided by and stored in the global catalog.
        :param str resource_group_id: (optional) The ID of the resource group.
        :param int limit: (optional) Limit on how many items should be returned.
        :param str updated_from: (optional) Start date inclusive filter.
        :param str updated_to: (optional) End date inclusive filter.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._guid = guid
        self._name = name
        self._resource_instance_id = resource_instance_id
        self._region_instance_id = region_instance_id
        self._resource_id = resource_id
        self._resource_group_id = resource_group_id
        self._limit = limit
        self._updated_from = updated_from
        self._updated_to = updated_to

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of ResourceAlias.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.list_resource_aliases(
            guid=self._guid,
            name=self._name,
            resource_instance_id=self._resource_instance_id,
            region_instance_id=self._region_instance_id,
            resource_id=self._resource_id,
            resource_group_id=self._resource_group_id,
            limit=self._limit,
            updated_from=self._updated_from,
            updated_to=self._updated_to,
            start=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next_url')
        if next_page_link is not None:
            next = get_query_param(next_page_link, 'start')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('resources')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of ResourceAlias.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results


class ResourceBindingsForAliasPager:
    """
    ResourceBindingsForAliasPager can be used to simplify the use of the "list_resource_bindings_for_alias" method.
    """

    def __init__(
        self,
        *,
        client: ResourceControllerV2,
        id: str,
        limit: int = None,
    ) -> None:
        """
        Initialize a ResourceBindingsForAliasPager object.
        :param str id: The resource alias URL-encoded CRN or GUID.
        :param int limit: (optional) Limit on how many items should be returned.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._id = id
        self._limit = limit

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of ResourceBinding.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.list_resource_bindings_for_alias(
            id=self._id,
            limit=self._limit,
            start=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next_url')
        if next_page_link is not None:
            next = get_query_param(next_page_link, 'start')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('resources')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of ResourceBinding.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results
