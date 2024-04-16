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

# IBM OpenAPI SDK Code Generator Version: 3.87.0-91c7c775-20240320-213027

"""
Manage your tags with the Tagging API in IBM Cloud. You can attach, detach, delete, or
list all of the tags in your billing account with the Tagging API. The tag name must be
unique within a billing account. You can create tags in two formats: `key:value` or
`label`. The tagging API supports three types of tag: `user` `service`, and `access` tags.
`service` tags cannot be attached to IMS resources. `service` tags must be in the form
`service_prefix:tag_label` where `service_prefix` identifies the Service owning the tag.
`access` tags cannot be attached to IMS and Cloud Foundry resources. They must be in the
form `key:value`.

API Version: 1.2.0
"""

from enum import Enum
from typing import Dict, List, Optional
import json

from ibm_cloud_sdk_core import BaseService, DetailedResponse
from ibm_cloud_sdk_core.authenticators.authenticator import Authenticator
from ibm_cloud_sdk_core.get_authenticator import get_authenticator_from_environment
from ibm_cloud_sdk_core.utils import convert_list, convert_model

from .common import get_sdk_headers

##############################################################################
# Service
##############################################################################


class GlobalTaggingV1(BaseService):
    """The global_tagging V1 service."""

    DEFAULT_SERVICE_URL = 'https://tags.global-search-tagging.cloud.ibm.com'
    DEFAULT_SERVICE_NAME = 'global_tagging'

    @classmethod
    def new_instance(
        cls,
        service_name: str = DEFAULT_SERVICE_NAME,
    ) -> 'GlobalTaggingV1':
        """
        Return a new client for the global_tagging service using the specified
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
        Construct a new client for the global_tagging service.

        :param Authenticator authenticator: The authenticator specifies the authentication mechanism.
               Get up to date information from https://github.com/IBM/python-sdk-core/blob/main/README.md
               about initializing the authenticator of your choice.
        """
        BaseService.__init__(self, service_url=self.DEFAULT_SERVICE_URL, authenticator=authenticator)

    #########################
    # tags
    #########################

    def list_tags(
        self,
        *,
        x_request_id: Optional[str] = None,
        x_correlation_id: Optional[str] = None,
        account_id: Optional[str] = None,
        tag_type: Optional[str] = None,
        full_data: Optional[bool] = None,
        providers: Optional[List[str]] = None,
        attached_to: Optional[str] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        timeout: Optional[int] = None,
        order_by_name: Optional[str] = None,
        attached_only: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get all tags.

        Lists all tags that are in a billing account. Use the `attached_to` parameter to
        return the list of tags that are attached to the specified resource.

        :param str x_request_id: (optional) An alphanumeric string that is used to
               trace the request. The value  may include ASCII alphanumerics and any of
               following segment separators: space ( ), comma (,), hyphen, (-), and
               underscore (_) and may have a length up to 1024 bytes. The value is
               considered invalid and must be ignored if that value includes any other
               character or is longer than 1024 bytes or is fewer than 8 characters. If
               not specified or invalid, it is automatically replaced by a random (version
               4) UUID.
        :param str x_correlation_id: (optional) An alphanumeric string that is used
               to trace the request as a part of a larger context: the same value is used
               for downstream requests and retries of those requests. The value may
               include ASCII alphanumerics and any of following segment separators: space
               ( ), comma (,), hyphen, (-), and underscore (_) and may have a length up to
               1024 bytes. The value is considered invalid and must be ignored if that
               value includes any other character or is longer than 1024 bytes or is fewer
               than 8 characters. If not specified or invalid, it is automatically
               replaced by a random (version 4) UUID.
        :param str account_id: (optional) The ID of the billing account to list the
               tags for. If it is not set, then it is taken from the authorization token.
               This parameter is required if `tag_type` is set to `service`.
        :param str tag_type: (optional) The type of the tag you want to list.
               Supported values are `user`, `service` and `access`.
        :param bool full_data: (optional) If set to `true`, this query returns the
               provider, `ghost`, `ims` or `ghost,ims`, where the tag exists and the
               number of attached resources.
        :param List[str] providers: (optional) Select a provider. Supported values
               are `ghost` and `ims`. To list both Global Search and Tagging tags and
               infrastructure tags, use `ghost,ims`. `service` and `access` tags can only
               be attached to resources that are onboarded to Global Search and Tagging,
               so you should not set this parameter to list them.
        :param str attached_to: (optional) If you want to return only the list of
               tags that are attached to a specified resource, pass the ID of the resource
               on this parameter. For resources that are onboarded to Global Search and
               Tagging, the resource ID is the CRN; for IMS resources, it is the IMS ID.
               When using this parameter, you must specify the appropriate provider (`ims`
               or `ghost`).
        :param int offset: (optional) The offset is the index of the item from
               which you want to start returning data from.
        :param int limit: (optional) The number of tags to return.
        :param int timeout: (optional) The timeout in milliseconds, bounds the
               request to run within the specified time value. It returns the accumulated
               results until time runs out.
        :param str order_by_name: (optional) Order the output by tag name.
        :param bool attached_only: (optional) Filter on attached tags. If `true`,
               it returns only tags that are attached to one or more resources. If
               `false`, it returns all tags.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TagList` object
        """

        headers = {
            'x-request-id': x_request_id,
            'x-correlation-id': x_correlation_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='list_tags',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
            'tag_type': tag_type,
            'full_data': full_data,
            'providers': convert_list(providers),
            'attached_to': attached_to,
            'offset': offset,
            'limit': limit,
            'timeout': timeout,
            'order_by_name': order_by_name,
            'attached_only': attached_only,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v3/tags'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def create_tag(
        self,
        tag_names: List[str],
        *,
        x_request_id: Optional[str] = None,
        x_correlation_id: Optional[str] = None,
        account_id: Optional[str] = None,
        tag_type: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create an access management tag.

        Create an access management tag. To create an `access` tag, you must have the
        access listed in the [Granting users access to tag
        resources](https://cloud.ibm.com/docs/account?topic=account-access) documentation.
        `service` and `user` tags cannot be created upfront. They are created when they
        are attached for the first time to a resource.

        :param List[str] tag_names: An array of tag names to create.
        :param str x_request_id: (optional) An alphanumeric string that is used to
               trace the request. The value  may include ASCII alphanumerics and any of
               following segment separators: space ( ), comma (,), hyphen, (-), and
               underscore (_) and may have a length up to 1024 bytes. The value is
               considered invalid and must be ignored if that value includes any other
               character or is longer than 1024 bytes or is fewer than 8 characters. If
               not specified or invalid, it is automatically replaced by a random (version
               4) UUID.
        :param str x_correlation_id: (optional) An alphanumeric string that is used
               to trace the request as a part of a larger context: the same value is used
               for downstream requests and retries of those requests. The value may
               include ASCII alphanumerics and any of following segment separators: space
               ( ), comma (,), hyphen, (-), and underscore (_) and may have a length up to
               1024 bytes. The value is considered invalid and must be ignored if that
               value includes any other character or is longer than 1024 bytes or is fewer
               than 8 characters. If not specified or invalid, it is automatically
               replaced by a random (version 4) UUID.
        :param str account_id: (optional) The ID of the billing account where the
               tag must be created.
        :param str tag_type: (optional) The type of the tags you want to create.
               The only allowed value is `access`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `CreateTagResults` object
        """

        if tag_names is None:
            raise ValueError('tag_names must be provided')
        headers = {
            'x-request-id': x_request_id,
            'x-correlation-id': x_correlation_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='create_tag',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
            'tag_type': tag_type,
        }

        data = {
            'tag_names': tag_names,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v3/tags'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            params=params,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_tag_all(
        self,
        *,
        x_request_id: Optional[str] = None,
        x_correlation_id: Optional[str] = None,
        providers: Optional[str] = None,
        account_id: Optional[str] = None,
        tag_type: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete all unused tags.

        Delete the tags that are not attached to any resource.

        :param str x_request_id: (optional) An alphanumeric string that is used to
               trace the request. The value  may include ASCII alphanumerics and any of
               following segment separators: space ( ), comma (,), hyphen, (-), and
               underscore (_) and may have a length up to 1024 bytes. The value is
               considered invalid and must be ignored if that value includes any other
               character or is longer than 1024 bytes or is fewer than 8 characters. If
               not specified or invalid, it is automatically replaced by a random (version
               4) UUID.
        :param str x_correlation_id: (optional) An alphanumeric string that is used
               to trace the request as a part of a larger context: the same value is used
               for downstream requests and retries of those requests. The value may
               include ASCII alphanumerics and any of following segment separators: space
               ( ), comma (,), hyphen, (-), and underscore (_) and may have a length up to
               1024 bytes. The value is considered invalid and must be ignored if that
               value includes any other character or is longer than 1024 bytes or is fewer
               than 8 characters. If not specified or invalid, it is automatically
               replaced by a random (version 4) UUID.
        :param str providers: (optional) Select a provider. Supported values are
               `ghost` and `ims`.
        :param str account_id: (optional) The ID of the billing account to delete
               the tags for. If it is not set, then it is taken from the authorization
               token. It is a required parameter if `tag_type` is set to `service`.
        :param str tag_type: (optional) The type of the tag. Supported values are
               `user`, `service` and `access`. `service` and `access` are not supported
               for IMS resources (`providers` parameter set to `ims`).
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `DeleteTagsResult` object
        """

        headers = {
            'x-request-id': x_request_id,
            'x-correlation-id': x_correlation_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='delete_tag_all',
        )
        headers.update(sdk_headers)

        params = {
            'providers': providers,
            'account_id': account_id,
            'tag_type': tag_type,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v3/tags'
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_tag(
        self,
        tag_name: str,
        *,
        x_request_id: Optional[str] = None,
        x_correlation_id: Optional[str] = None,
        providers: Optional[List[str]] = None,
        account_id: Optional[str] = None,
        tag_type: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete an unused tag.

        Delete an existing tag. A tag can be deleted only if it is not attached to any
        resource.

        :param str tag_name: The name of tag to be deleted.
        :param str x_request_id: (optional) An alphanumeric string that is used to
               trace the request. The value  may include ASCII alphanumerics and any of
               following segment separators: space ( ), comma (,), hyphen, (-), and
               underscore (_) and may have a length up to 1024 bytes. The value is
               considered invalid and must be ignored if that value includes any other
               character or is longer than 1024 bytes or is fewer than 8 characters. If
               not specified or invalid, it is automatically replaced by a random (version
               4) UUID.
        :param str x_correlation_id: (optional) An alphanumeric string that is used
               to trace the request as a part of a larger context: the same value is used
               for downstream requests and retries of those requests. The value may
               include ASCII alphanumerics and any of following segment separators: space
               ( ), comma (,), hyphen, (-), and underscore (_) and may have a length up to
               1024 bytes. The value is considered invalid and must be ignored if that
               value includes any other character or is longer than 1024 bytes or is fewer
               than 8 characters. If not specified or invalid, it is automatically
               replaced by a random (version 4) UUID.
        :param List[str] providers: (optional) Select a provider. Supported values
               are `ghost` and `ims`. To delete tags both in Global Search and Tagging and
               in IMS, use `ghost,ims`.
        :param str account_id: (optional) The ID of the billing account to delete
               the tag for. It is a required parameter if `tag_type` is set to `service`,
               otherwise it is inferred from the authorization IAM token.
        :param str tag_type: (optional) The type of the tag. Supported values are
               `user`, `service` and `access`. `service` and `access` are not supported
               for IMS resources (`providers` parameter set to `ims`).
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `DeleteTagResults` object
        """

        if not tag_name:
            raise ValueError('tag_name must be provided')
        headers = {
            'x-request-id': x_request_id,
            'x-correlation-id': x_correlation_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='delete_tag',
        )
        headers.update(sdk_headers)

        params = {
            'providers': convert_list(providers),
            'account_id': account_id,
            'tag_type': tag_type,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['tag_name']
        path_param_values = self.encode_path_vars(tag_name)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v3/tags/{tag_name}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def attach_tag(
        self,
        resources: List['Resource'],
        *,
        tag_name: Optional[str] = None,
        tag_names: Optional[List[str]] = None,
        x_request_id: Optional[str] = None,
        x_correlation_id: Optional[str] = None,
        account_id: Optional[str] = None,
        tag_type: Optional[str] = None,
        replace: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Attach tags.

        Attaches one or more tags to one or more resources. Each resource can have no more
        than 1000 tags per each 'user' and 'service' type, and no more than 250 'access'
        tags (which is the account limit).

        :param List[Resource] resources: List of resources on which the tag or tags
               are attached.
        :param str tag_name: (optional) The name of the tag to attach.
        :param List[str] tag_names: (optional) An array of tag names to attach.
        :param str x_request_id: (optional) An alphanumeric string that is used to
               trace the request. The value  may include ASCII alphanumerics and any of
               following segment separators: space ( ), comma (,), hyphen, (-), and
               underscore (_) and may have a length up to 1024 bytes. The value is
               considered invalid and must be ignored if that value includes any other
               character or is longer than 1024 bytes or is fewer than 8 characters. If
               not specified or invalid, it is automatically replaced by a random (version
               4) UUID.
        :param str x_correlation_id: (optional) An alphanumeric string that is used
               to trace the request as a part of a larger context: the same value is used
               for downstream requests and retries of those requests. The value may
               include ASCII alphanumerics and any of following segment separators: space
               ( ), comma (,), hyphen, (-), and underscore (_) and may have a length up to
               1024 bytes. The value is considered invalid and must be ignored if that
               value includes any other character or is longer than 1024 bytes or is fewer
               than 8 characters. If not specified or invalid, it is automatically
               replaced by a random (version 4) UUID.
        :param str account_id: (optional) The ID of the billing account of the
               tagged resource. It is a required parameter if `tag_type` is set to
               `service`. Otherwise, it is inferred from the authorization IAM token.
        :param str tag_type: (optional) The type of the tag. Supported values are
               `user`, `service` and `access`. `service` and `access` are not supported
               for IMS resources.
        :param bool replace: (optional) Flag to request replacement of all attached
               tags. Set 'true' if you want to replace all the list of tags attached to
               the resource. Default value is false.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TagResults` object
        """

        if resources is None:
            raise ValueError('resources must be provided')
        resources = [convert_model(x) for x in resources]
        headers = {
            'x-request-id': x_request_id,
            'x-correlation-id': x_correlation_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='attach_tag',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
            'tag_type': tag_type,
            'replace': replace,
        }

        data = {
            'resources': resources,
            'tag_name': tag_name,
            'tag_names': tag_names,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v3/tags/attach'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            params=params,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def detach_tag(
        self,
        resources: List['Resource'],
        *,
        tag_name: Optional[str] = None,
        tag_names: Optional[List[str]] = None,
        x_request_id: Optional[str] = None,
        x_correlation_id: Optional[str] = None,
        account_id: Optional[str] = None,
        tag_type: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Detach tags.

        Detaches one or more tags from one or more resources.

        :param List[Resource] resources: List of resources on which the tag or tags
               are detached.
        :param str tag_name: (optional) The name of the tag to detach.
        :param List[str] tag_names: (optional) An array of tag names to detach.
        :param str x_request_id: (optional) An alphanumeric string that is used to
               trace the request. The value  may include ASCII alphanumerics and any of
               following segment separators: space ( ), comma (,), hyphen, (-), and
               underscore (_) and may have a length up to 1024 bytes. The value is
               considered invalid and must be ignored if that value includes any other
               character or is longer than 1024 bytes or is fewer than 8 characters. If
               not specified or invalid, it is automatically replaced by a random (version
               4) UUID.
        :param str x_correlation_id: (optional) An alphanumeric string that is used
               to trace the request as a part of a larger context: the same value is used
               for downstream requests and retries of those requests. The value may
               include ASCII alphanumerics and any of following segment separators: space
               ( ), comma (,), hyphen, (-), and underscore (_) and may have a length up to
               1024 bytes. The value is considered invalid and must be ignored if that
               value includes any other character or is longer than 1024 bytes or is fewer
               than 8 characters. If not specified or invalid, it is automatically
               replaced by a random (version 4) UUID.
        :param str account_id: (optional) The ID of the billing account of the
               untagged resource. It is a required parameter if `tag_type` is set to
               `service`, otherwise it is inferred from the authorization IAM token.
        :param str tag_type: (optional) The type of the tag. Supported values are
               `user`, `service` and `access`. `service` and `access` are not supported
               for IMS resources.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TagResults` object
        """

        if resources is None:
            raise ValueError('resources must be provided')
        resources = [convert_model(x) for x in resources]
        headers = {
            'x-request-id': x_request_id,
            'x-correlation-id': x_correlation_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='detach_tag',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
            'tag_type': tag_type,
        }

        data = {
            'resources': resources,
            'tag_name': tag_name,
            'tag_names': tag_names,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v3/tags/detach'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            params=params,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response


class ListTagsEnums:
    """
    Enums for list_tags parameters.
    """

    class TagType(str, Enum):
        """
        The type of the tag you want to list. Supported values are `user`, `service` and
        `access`.
        """

        USER = 'user'
        SERVICE = 'service'
        ACCESS = 'access'

    class Providers(str, Enum):
        """
        Select a provider. Supported values are `ghost` and `ims`. To list both Global
        Search and Tagging tags and infrastructure tags, use `ghost,ims`. `service` and
        `access` tags can only be attached to resources that are onboarded to Global
        Search and Tagging, so you should not set this parameter to list them.
        """

        GHOST = 'ghost'
        IMS = 'ims'

    class OrderByName(str, Enum):
        """
        Order the output by tag name.
        """

        ASC = 'asc'
        DESC = 'desc'


class CreateTagEnums:
    """
    Enums for create_tag parameters.
    """

    class TagType(str, Enum):
        """
        The type of the tags you want to create. The only allowed value is `access`.
        """

        ACCESS = 'access'


class DeleteTagAllEnums:
    """
    Enums for delete_tag_all parameters.
    """

    class Providers(str, Enum):
        """
        Select a provider. Supported values are `ghost` and `ims`.
        """

        GHOST = 'ghost'
        IMS = 'ims'

    class TagType(str, Enum):
        """
        The type of the tag. Supported values are `user`, `service` and `access`.
        `service` and `access` are not supported for IMS resources (`providers` parameter
        set to `ims`).
        """

        USER = 'user'
        SERVICE = 'service'
        ACCESS = 'access'


class DeleteTagEnums:
    """
    Enums for delete_tag parameters.
    """

    class Providers(str, Enum):
        """
        Select a provider. Supported values are `ghost` and `ims`. To delete tags both in
        Global Search and Tagging and in IMS, use `ghost,ims`.
        """

        GHOST = 'ghost'
        IMS = 'ims'

    class TagType(str, Enum):
        """
        The type of the tag. Supported values are `user`, `service` and `access`.
        `service` and `access` are not supported for IMS resources (`providers` parameter
        set to `ims`).
        """

        USER = 'user'
        SERVICE = 'service'
        ACCESS = 'access'


class AttachTagEnums:
    """
    Enums for attach_tag parameters.
    """

    class TagType(str, Enum):
        """
        The type of the tag. Supported values are `user`, `service` and `access`.
        `service` and `access` are not supported for IMS resources.
        """

        USER = 'user'
        SERVICE = 'service'
        ACCESS = 'access'


class DetachTagEnums:
    """
    Enums for detach_tag parameters.
    """

    class TagType(str, Enum):
        """
        The type of the tag. Supported values are `user`, `service` and `access`.
        `service` and `access` are not supported for IMS resources.
        """

        USER = 'user'
        SERVICE = 'service'
        ACCESS = 'access'


##############################################################################
# Models
##############################################################################


class CreateTagResults:
    """
    Results of a create tag(s) request.

    :param List[CreateTagResultsResultsItem] results: (optional) Array of results of
          a create_tag request.
    """

    def __init__(
        self,
        *,
        results: Optional[List['CreateTagResultsResultsItem']] = None,
    ) -> None:
        """
        Initialize a CreateTagResults object.

        :param List[CreateTagResultsResultsItem] results: (optional) Array of
               results of a create_tag request.
        """
        self.results = results

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CreateTagResults':
        """Initialize a CreateTagResults object from a json dictionary."""
        args = {}
        if (results := _dict.get('results')) is not None:
            args['results'] = [CreateTagResultsResultsItem.from_dict(v) for v in results]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CreateTagResults object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'results') and self.results is not None:
            results_list = []
            for v in self.results:
                if isinstance(v, dict):
                    results_list.append(v)
                else:
                    results_list.append(v.to_dict())
            _dict['results'] = results_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this CreateTagResults object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CreateTagResults') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CreateTagResults') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class CreateTagResultsResultsItem:
    """
    CreateTagResultsResultsItem.

    :param str tag_name: (optional) The name of the tag created.
    :param bool is_error: (optional) true if the tag was not created (for example,
          the tag already exists).
    """

    def __init__(
        self,
        *,
        tag_name: Optional[str] = None,
        is_error: Optional[bool] = None,
    ) -> None:
        """
        Initialize a CreateTagResultsResultsItem object.

        :param str tag_name: (optional) The name of the tag created.
        :param bool is_error: (optional) true if the tag was not created (for
               example, the tag already exists).
        """
        self.tag_name = tag_name
        self.is_error = is_error

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CreateTagResultsResultsItem':
        """Initialize a CreateTagResultsResultsItem object from a json dictionary."""
        args = {}
        if (tag_name := _dict.get('tag_name')) is not None:
            args['tag_name'] = tag_name
        if (is_error := _dict.get('is_error')) is not None:
            args['is_error'] = is_error
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CreateTagResultsResultsItem object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'tag_name') and self.tag_name is not None:
            _dict['tag_name'] = self.tag_name
        if hasattr(self, 'is_error') and self.is_error is not None:
            _dict['is_error'] = self.is_error
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this CreateTagResultsResultsItem object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CreateTagResultsResultsItem') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CreateTagResultsResultsItem') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class DeleteTagResults:
    """
    Results of a delete_tag request.

    :param List[DeleteTagResultsItem] results: (optional) Array of results of a
          delete_tag request.
    """

    def __init__(
        self,
        *,
        results: Optional[List['DeleteTagResultsItem']] = None,
    ) -> None:
        """
        Initialize a DeleteTagResults object.

        :param List[DeleteTagResultsItem] results: (optional) Array of results of a
               delete_tag request.
        """
        self.results = results

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'DeleteTagResults':
        """Initialize a DeleteTagResults object from a json dictionary."""
        args = {}
        if (results := _dict.get('results')) is not None:
            args['results'] = [DeleteTagResultsItem.from_dict(v) for v in results]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a DeleteTagResults object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'results') and self.results is not None:
            results_list = []
            for v in self.results:
                if isinstance(v, dict):
                    results_list.append(v)
                else:
                    results_list.append(v.to_dict())
            _dict['results'] = results_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this DeleteTagResults object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'DeleteTagResults') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'DeleteTagResults') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class DeleteTagResultsItem:
    """
    Result of a delete_tag request.

    :param str provider: (optional) The provider of the tag.
    :param bool is_error: (optional) It is `true` if the operation exits with an
          error (for example, the tag does not exist).
    """

    # The set of defined properties for the class
    _properties = frozenset(['provider', 'is_error'])

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        is_error: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a DeleteTagResultsItem object.

        :param str provider: (optional) The provider of the tag.
        :param bool is_error: (optional) It is `true` if the operation exits with
               an error (for example, the tag does not exist).
        :param **kwargs: (optional) Any additional properties.
        """
        self.provider = provider
        self.is_error = is_error
        for _key, _value in kwargs.items():
            setattr(self, _key, _value)

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'DeleteTagResultsItem':
        """Initialize a DeleteTagResultsItem object from a json dictionary."""
        args = {}
        if (provider := _dict.get('provider')) is not None:
            args['provider'] = provider
        if (is_error := _dict.get('is_error')) is not None:
            args['is_error'] = is_error
        args.update({k: v for (k, v) in _dict.items() if k not in cls._properties})
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a DeleteTagResultsItem object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'provider') and self.provider is not None:
            _dict['provider'] = self.provider
        if hasattr(self, 'is_error') and self.is_error is not None:
            _dict['is_error'] = self.is_error
        for _key in [k for k in vars(self).keys() if k not in DeleteTagResultsItem._properties]:
            _dict[_key] = getattr(self, _key)
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def get_properties(self) -> Dict:
        """Return a dictionary of arbitrary properties from this instance of DeleteTagResultsItem"""
        _dict = {}

        for _key in [k for k in vars(self).keys() if k not in DeleteTagResultsItem._properties]:
            _dict[_key] = getattr(self, _key)
        return _dict

    def set_properties(self, _dict: dict):
        """Set a dictionary of arbitrary properties to this instance of DeleteTagResultsItem"""
        for _key in [k for k in vars(self).keys() if k not in DeleteTagResultsItem._properties]:
            delattr(self, _key)

        for _key, _value in _dict.items():
            if _key not in DeleteTagResultsItem._properties:
                setattr(self, _key, _value)

    def __str__(self) -> str:
        """Return a `str` version of this DeleteTagResultsItem object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'DeleteTagResultsItem') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'DeleteTagResultsItem') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class ProviderEnum(str, Enum):
        """
        The provider of the tag.
        """

        GHOST = 'ghost'
        IMS = 'ims'


class DeleteTagsResult:
    """
    Results of deleting unattatched tags.

    :param int total_count: (optional) The number of tags that have been deleted.
    :param bool errors: (optional) It is set to true if there is at least one tag
          operation in error.
    :param List[DeleteTagsResultItem] items: (optional) The list of tag operation
          results.
    """

    def __init__(
        self,
        *,
        total_count: Optional[int] = None,
        errors: Optional[bool] = None,
        items: Optional[List['DeleteTagsResultItem']] = None,
    ) -> None:
        """
        Initialize a DeleteTagsResult object.

        :param int total_count: (optional) The number of tags that have been
               deleted.
        :param bool errors: (optional) It is set to true if there is at least one
               tag operation in error.
        :param List[DeleteTagsResultItem] items: (optional) The list of tag
               operation results.
        """
        self.total_count = total_count
        self.errors = errors
        self.items = items

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'DeleteTagsResult':
        """Initialize a DeleteTagsResult object from a json dictionary."""
        args = {}
        if (total_count := _dict.get('total_count')) is not None:
            args['total_count'] = total_count
        if (errors := _dict.get('errors')) is not None:
            args['errors'] = errors
        if (items := _dict.get('items')) is not None:
            args['items'] = [DeleteTagsResultItem.from_dict(v) for v in items]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a DeleteTagsResult object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'total_count') and self.total_count is not None:
            _dict['total_count'] = self.total_count
        if hasattr(self, 'errors') and self.errors is not None:
            _dict['errors'] = self.errors
        if hasattr(self, 'items') and self.items is not None:
            items_list = []
            for v in self.items:
                if isinstance(v, dict):
                    items_list.append(v)
                else:
                    items_list.append(v.to_dict())
            _dict['items'] = items_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this DeleteTagsResult object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'DeleteTagsResult') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'DeleteTagsResult') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class DeleteTagsResultItem:
    """
    Result of a delete_tags request.

    :param str tag_name: (optional) The name of the deleted tag.
    :param bool is_error: (optional) true if the tag was not deleted.
    """

    def __init__(
        self,
        *,
        tag_name: Optional[str] = None,
        is_error: Optional[bool] = None,
    ) -> None:
        """
        Initialize a DeleteTagsResultItem object.

        :param str tag_name: (optional) The name of the deleted tag.
        :param bool is_error: (optional) true if the tag was not deleted.
        """
        self.tag_name = tag_name
        self.is_error = is_error

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'DeleteTagsResultItem':
        """Initialize a DeleteTagsResultItem object from a json dictionary."""
        args = {}
        if (tag_name := _dict.get('tag_name')) is not None:
            args['tag_name'] = tag_name
        if (is_error := _dict.get('is_error')) is not None:
            args['is_error'] = is_error
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a DeleteTagsResultItem object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'tag_name') and self.tag_name is not None:
            _dict['tag_name'] = self.tag_name
        if hasattr(self, 'is_error') and self.is_error is not None:
            _dict['is_error'] = self.is_error
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this DeleteTagsResultItem object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'DeleteTagsResultItem') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'DeleteTagsResultItem') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Resource:
    """
    A resource that might have tags that are attached.

    :param str resource_id: The CRN or IMS ID of the resource.
    :param str resource_type: (optional) The IMS resource type of the resource.
    """

    def __init__(
        self,
        resource_id: str,
        *,
        resource_type: Optional[str] = None,
    ) -> None:
        """
        Initialize a Resource object.

        :param str resource_id: The CRN or IMS ID of the resource.
        :param str resource_type: (optional) The IMS resource type of the resource.
        """
        self.resource_id = resource_id
        self.resource_type = resource_type

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Resource':
        """Initialize a Resource object from a json dictionary."""
        args = {}
        if (resource_id := _dict.get('resource_id')) is not None:
            args['resource_id'] = resource_id
        else:
            raise ValueError('Required property \'resource_id\' not present in Resource JSON')
        if (resource_type := _dict.get('resource_type')) is not None:
            args['resource_type'] = resource_type
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Resource object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'resource_id') and self.resource_id is not None:
            _dict['resource_id'] = self.resource_id
        if hasattr(self, 'resource_type') and self.resource_type is not None:
            _dict['resource_type'] = self.resource_type
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Resource object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Resource') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Resource') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Tag:
    """
    A tag.

    :param str name: The name of the tag.
    """

    def __init__(
        self,
        name: str,
    ) -> None:
        """
        Initialize a Tag object.

        :param str name: The name of the tag.
        """
        self.name = name

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Tag':
        """Initialize a Tag object from a json dictionary."""
        args = {}
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        else:
            raise ValueError('Required property \'name\' not present in Tag JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Tag object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Tag object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Tag') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Tag') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TagList:
    """
    A list of tags.

    :param int total_count: (optional) Set the occurrences of the total tags that
          are associated with this account.
    :param int offset: (optional) The offset at which tags are returned.
    :param int limit: (optional) The number of tags requested to be returned.
    :param List[Tag] items: (optional) Array of output results.
    """

    def __init__(
        self,
        *,
        total_count: Optional[int] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        items: Optional[List['Tag']] = None,
    ) -> None:
        """
        Initialize a TagList object.

        :param int total_count: (optional) Set the occurrences of the total tags
               that are associated with this account.
        :param int offset: (optional) The offset at which tags are returned.
        :param int limit: (optional) The number of tags requested to be returned.
        :param List[Tag] items: (optional) Array of output results.
        """
        self.total_count = total_count
        self.offset = offset
        self.limit = limit
        self.items = items

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TagList':
        """Initialize a TagList object from a json dictionary."""
        args = {}
        if (total_count := _dict.get('total_count')) is not None:
            args['total_count'] = total_count
        if (offset := _dict.get('offset')) is not None:
            args['offset'] = offset
        if (limit := _dict.get('limit')) is not None:
            args['limit'] = limit
        if (items := _dict.get('items')) is not None:
            args['items'] = [Tag.from_dict(v) for v in items]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TagList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'total_count') and self.total_count is not None:
            _dict['total_count'] = self.total_count
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'items') and self.items is not None:
            items_list = []
            for v in self.items:
                if isinstance(v, dict):
                    items_list.append(v)
                else:
                    items_list.append(v.to_dict())
            _dict['items'] = items_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TagList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TagList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TagList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TagResults:
    """
    Results of an attach_tag or detach_tag request.

    :param List[TagResultsItem] results: (optional) Array of results of an
          attach_tag or detach_tag request.
    """

    def __init__(
        self,
        *,
        results: Optional[List['TagResultsItem']] = None,
    ) -> None:
        """
        Initialize a TagResults object.

        :param List[TagResultsItem] results: (optional) Array of results of an
               attach_tag or detach_tag request.
        """
        self.results = results

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TagResults':
        """Initialize a TagResults object from a json dictionary."""
        args = {}
        if (results := _dict.get('results')) is not None:
            args['results'] = [TagResultsItem.from_dict(v) for v in results]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TagResults object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'results') and self.results is not None:
            results_list = []
            for v in self.results:
                if isinstance(v, dict):
                    results_list.append(v)
                else:
                    results_list.append(v.to_dict())
            _dict['results'] = results_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TagResults object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TagResults') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TagResults') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TagResultsItem:
    """
    Result of an attach_tag or detach_tag request for a tagged resource.

    :param str resource_id: The CRN or IMS ID of the resource.
    :param bool is_error: (optional) It is `true` if the operation exits with an
          error.
    """

    def __init__(
        self,
        resource_id: str,
        *,
        is_error: Optional[bool] = None,
    ) -> None:
        """
        Initialize a TagResultsItem object.

        :param str resource_id: The CRN or IMS ID of the resource.
        :param bool is_error: (optional) It is `true` if the operation exits with
               an error.
        """
        self.resource_id = resource_id
        self.is_error = is_error

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TagResultsItem':
        """Initialize a TagResultsItem object from a json dictionary."""
        args = {}
        if (resource_id := _dict.get('resource_id')) is not None:
            args['resource_id'] = resource_id
        else:
            raise ValueError('Required property \'resource_id\' not present in TagResultsItem JSON')
        if (is_error := _dict.get('is_error')) is not None:
            args['is_error'] = is_error
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TagResultsItem object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'resource_id') and self.resource_id is not None:
            _dict['resource_id'] = self.resource_id
        if hasattr(self, 'is_error') and self.is_error is not None:
            _dict['is_error'] = self.is_error
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TagResultsItem object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TagResultsItem') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TagResultsItem') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other
