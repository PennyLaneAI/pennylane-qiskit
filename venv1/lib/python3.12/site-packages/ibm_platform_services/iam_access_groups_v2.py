# coding: utf-8

# (C) Copyright IBM Corp. 2023.
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

# IBM OpenAPI SDK Code Generator Version: 3.78.0-67aec9b7-20230818-174940

"""
The IAM Access Groups API allows for the management of access groups (Create, Read,
Update, Delete) as well as the management of memberships and rules within the group
container.

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


class IamAccessGroupsV2(BaseService):
    """The iam-access-groups V2 service."""

    DEFAULT_SERVICE_URL = 'https://iam.cloud.ibm.com'
    DEFAULT_SERVICE_NAME = 'iam_access_groups'

    @classmethod
    def new_instance(
        cls,
        service_name: str = DEFAULT_SERVICE_NAME,
    ) -> 'IamAccessGroupsV2':
        """
        Return a new client for the iam-access-groups service using the specified
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
        Construct a new client for the iam-access-groups service.

        :param Authenticator authenticator: The authenticator specifies the authentication mechanism.
               Get up to date information from https://github.com/IBM/python-sdk-core/blob/main/README.md
               about initializing the authenticator of your choice.
        """
        BaseService.__init__(self, service_url=self.DEFAULT_SERVICE_URL, authenticator=authenticator)

    #########################
    # Access group operations
    #########################

    def create_access_group(
        self,
        account_id: str,
        name: str,
        *,
        description: str = None,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create an access group.

        Create a new access group to assign multiple users and service ids to multiple
        policies. The group will be created in the account specified by the `account_id`
        parameter. The group name is a required field, but a description is optional.
        Because the group's name does not have to be unique, it is possible to create
        multiple groups with the same name.

        :param str account_id: Account ID of the API keys(s) to query. If a service
               IAM ID is specified in iam_id then account_id must match the account of the
               IAM ID. If a user IAM ID is specified in iam_id then then account_id must
               match the account of the Authorization token.
        :param str name: Give the access group a unique name that doesn't conflict
               with an existing access group in the account. This field is
               case-insensitive and has a limit of 100 characters.
        :param str description: (optional) Assign an optional description for the
               access group. This field has a limit of 250 characters.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Group` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if name is None:
            raise ValueError('name must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='create_access_group',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
        }

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

        url = '/v2/groups'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            params=params,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def list_access_groups(
        self,
        account_id: str,
        *,
        transaction_id: str = None,
        iam_id: str = None,
        search: str = None,
        membership_type: str = None,
        limit: int = None,
        offset: int = None,
        sort: str = None,
        show_federated: bool = None,
        hide_public_access: bool = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List access groups.

        This API lists access groups within an account. Parameters for pagination and
        sorting can be used to filter the results. The `account_id` query parameter
        determines which account to retrieve groups from. Only the groups you have access
        to are returned (either because of a policy on a specific group or account level
        access (admin, editor, or viewer)). There may be more groups in the account that
        aren't shown if you lack the aforementioned permissions.

        :param str account_id: Account ID of the API keys(s) to query. If a service
               IAM ID is specified in iam_id then account_id must match the account of the
               IAM ID. If a user IAM ID is specified in iam_id then then account_id must
               match the account of the Authorization token.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param str iam_id: (optional) Return groups for member ID (IBMid, service
               ID or trusted profile ID).
        :param str search: (optional) Use search to filter access groups list by
               id, name or description.
               * `search=id:<ACCESS_GROUP_ID>` - To list access groups by id
               * `search=name:<ACCESS_GROUP_NAME>` - To list access groups by name
               * `search=description:<ACCESS_GROUP_DESC>` - To list access groups by
               description.
        :param str membership_type: (optional) Membership type need to be specified
               along with iam_id and must be either `static`, `dynamic` or `all`. If
               membership type is `static`, members explicitly added to the group will be
               shown. If membership type is `dynamic`, members accessing the access group
               at the moment via dynamic rules will be shown. If membership type is `all`,
               both static and dynamic members will be shown.
        :param int limit: (optional) Return up to this limit of results where limit
               is between 0 and 100.
        :param int offset: (optional) The offset of the first result item to be
               returned.
        :param str sort: (optional) Sort the results by id, name, description, or
               is_federated flag.
        :param bool show_federated: (optional) If show_federated is true, each
               group listed will return an is_federated value that is set to true if rules
               exist for the group.
        :param bool hide_public_access: (optional) If hide_public_access is true,
               do not include the Public Access Group in the results.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `GroupsList` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='list_access_groups',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
            'iam_id': iam_id,
            'search': search,
            'membership_type': membership_type,
            'limit': limit,
            'offset': offset,
            'sort': sort,
            'show_federated': show_federated,
            'hide_public_access': hide_public_access,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v2/groups'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def get_access_group(
        self,
        access_group_id: str,
        *,
        transaction_id: str = None,
        show_federated: bool = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get an access group.

        Retrieve an access group by its `access_group_id`. Only the groups data is
        returned (group name, description, account_id, ...), not membership or rule
        information. A revision number is returned in the `ETag` header, which is needed
        when updating the access group.

        :param str access_group_id: The access group identifier.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param bool show_federated: (optional) If show_federated is true, the group
               will return an is_federated value that is set to true if rules exist for
               the group.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Group` object
        """

        if not access_group_id:
            raise ValueError('access_group_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='get_access_group',
        )
        headers.update(sdk_headers)

        params = {
            'show_federated': show_federated,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['access_group_id']
        path_param_values = self.encode_path_vars(access_group_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/groups/{access_group_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def update_access_group(
        self,
        access_group_id: str,
        if_match: str,
        *,
        name: str = None,
        description: str = None,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Update an access group.

        Update the group name or description of an existing access group using this API.
        An `If-Match` header must be populated with the group's most recent revision
        number (which can be acquired in the `Get an access group` API).

        :param str access_group_id: The access group identifier.
        :param str if_match: The current revision number of the group being
               updated. This can be found in the Create/Get access group response ETag
               header.
        :param str name: (optional) Give the access group a unique name that
               doesn't conflict with an existing access group in the account. This field
               is case-insensitive and has a limit of 100 characters.
        :param str description: (optional) Assign an optional description for the
               access group. This field has a limit of 250 characters.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Group` object
        """

        if not access_group_id:
            raise ValueError('access_group_id must be provided')
        if not if_match:
            raise ValueError('if_match must be provided')
        headers = {
            'If-Match': if_match,
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='update_access_group',
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

        path_param_keys = ['access_group_id']
        path_param_values = self.encode_path_vars(access_group_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/groups/{access_group_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PATCH',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_access_group(
        self,
        access_group_id: str,
        *,
        transaction_id: str = None,
        force: bool = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete an access group.

        This API is used for deleting an access group. If the access group has no members
        or rules associated with it, the group and its policies will be deleted. However,
        if rules or members do exist, set the `force` parameter to true to delete the
        group as well as its associated members, rules, and policies.

        :param str access_group_id: The access group identifier.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param bool force: (optional) If force is true, delete the group as well as
               its associated members and rules.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not access_group_id:
            raise ValueError('access_group_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='delete_access_group',
        )
        headers.update(sdk_headers)

        params = {
            'force': force,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['access_group_id']
        path_param_values = self.encode_path_vars(access_group_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/groups/{access_group_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # Membership operations
    #########################

    def is_member_of_access_group(
        self,
        access_group_id: str,
        iam_id: str,
        *,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Check membership in an access group.

        This HEAD operation determines if a given `iam_id` is present in a group either
        explicitly or via dynamic rules. No response body is returned with this request.
        If the membership exists, a `204 - No Content` status code is returned. If the
        membership or the group does not exist, a `404 - Not Found` status code is
        returned.

        :param str access_group_id: The access group identifier.
        :param str iam_id: The IAM identifier.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not access_group_id:
            raise ValueError('access_group_id must be provided')
        if not iam_id:
            raise ValueError('iam_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='is_member_of_access_group',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['access_group_id', 'iam_id']
        path_param_values = self.encode_path_vars(access_group_id, iam_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/groups/{access_group_id}/members/{iam_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='HEAD',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def add_members_to_access_group(
        self,
        access_group_id: str,
        *,
        members: List['AddGroupMembersRequestMembersItem'] = None,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Add members to an access group.

        Use this API to add users (`IBMid-...`), service IDs (`iam-ServiceId-...`) or
        trusted profiles (`iam-Profile-...`) to an access group. Any member added gains
        access to resources defined in the group's policies. To revoke a given members's
        access, simply remove them from the group. There is no limit to the number of
        members one group can have, but each `iam_id` can only be added to 50 groups.
        Additionally, this API request payload can add up to 50 members per call.

        :param str access_group_id: The access group identifier.
        :param List[AddGroupMembersRequestMembersItem] members: (optional) An array
               of member objects to add to an access group.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AddGroupMembersResponse` object
        """

        if not access_group_id:
            raise ValueError('access_group_id must be provided')
        if members is not None:
            members = [convert_model(x) for x in members]
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='add_members_to_access_group',
        )
        headers.update(sdk_headers)

        data = {
            'members': members,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['access_group_id']
        path_param_values = self.encode_path_vars(access_group_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/groups/{access_group_id}/members'.format(**path_param_dict)
        request = self.prepare_request(
            method='PUT',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def list_access_group_members(
        self,
        access_group_id: str,
        *,
        transaction_id: str = None,
        membership_type: str = None,
        limit: int = None,
        offset: int = None,
        type: str = None,
        verbose: bool = None,
        sort: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List access group members.

        List all members of a given group using this API. Parameters for pagination and
        sorting can be used to filter the results. The most useful query parameter may be
        the `verbose` flag. If `verbose=true`, user, service ID and trusted profile names
        will be retrieved for each `iam_id`. If performance is a concern, leave the
        `verbose` parameter off so that name information does not get retrieved.

        :param str access_group_id: The access group identifier.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param str membership_type: (optional) Filters members by membership type.
               Filter by `static`, `dynamic` or `all`. `static` lists the members
               explicitly added to the access group, and `dynamic` lists the members that
               are part of the access group at that time via dynamic rules. `all` lists
               both static and dynamic members.
        :param int limit: (optional) Return up to this limit of results where limit
               is between 0 and 100.
        :param int offset: (optional) The offset of the first result item to be
               returned.
        :param str type: (optional) Filter the results by member type.
        :param bool verbose: (optional) Return user's email and name for each user
               ID or the name for each service ID or trusted profile.
        :param str sort: (optional) If verbose is true, sort the results by id,
               name, or email.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `GroupMembersList` object
        """

        if not access_group_id:
            raise ValueError('access_group_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='list_access_group_members',
        )
        headers.update(sdk_headers)

        params = {
            'membership_type': membership_type,
            'limit': limit,
            'offset': offset,
            'type': type,
            'verbose': verbose,
            'sort': sort,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['access_group_id']
        path_param_values = self.encode_path_vars(access_group_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/groups/{access_group_id}/members'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def remove_member_from_access_group(
        self,
        access_group_id: str,
        iam_id: str,
        *,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete member from an access group.

        Remove one member from a group using this API. If the operation is successful,
        only a `204 - No Content` response with no body is returned. However, if any error
        occurs, the standard error format will be returned. Dynamic member cannot be
        deleted using this API. Dynamic rules needs to be adjusted to delete dynamic
        members.

        :param str access_group_id: The access group identifier.
        :param str iam_id: The IAM identifier.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not access_group_id:
            raise ValueError('access_group_id must be provided')
        if not iam_id:
            raise ValueError('iam_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='remove_member_from_access_group',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['access_group_id', 'iam_id']
        path_param_values = self.encode_path_vars(access_group_id, iam_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/groups/{access_group_id}/members/{iam_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def remove_members_from_access_group(
        self,
        access_group_id: str,
        *,
        members: List[str] = None,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete members from an access group.

        Remove multiple members from a group using this API. On a successful call, this
        API will always return 207. It is the caller's responsibility to iterate across
        the body to determine successful deletion of each member. This API request payload
        can delete up to 50 members per call. This API doesnt delete dynamic members
        accessing the access group via dynamic rules.

        :param str access_group_id: The access group identifier.
        :param List[str] members: (optional) The `iam_id`s to remove from the
               access group. This field has a limit of 50 `iam_id`s.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `DeleteGroupBulkMembersResponse` object
        """

        if not access_group_id:
            raise ValueError('access_group_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='remove_members_from_access_group',
        )
        headers.update(sdk_headers)

        data = {
            'members': members,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['access_group_id']
        path_param_values = self.encode_path_vars(access_group_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/groups/{access_group_id}/members/delete'.format(**path_param_dict)
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def remove_member_from_all_access_groups(
        self,
        account_id: str,
        iam_id: str,
        *,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete member from all access groups.

        This API removes a given member from every group they are a member of within the
        specified account. By using one operation, you can revoke one member's access to
        all access groups in the account. If a partial failure occurs on deletion, the
        response will be shown in the body.

        :param str account_id: Account ID of the API keys(s) to query. If a service
               IAM ID is specified in iam_id then account_id must match the account of the
               IAM ID. If a user IAM ID is specified in iam_id then then account_id must
               match the account of the Authorization token.
        :param str iam_id: The IAM identifier.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `DeleteFromAllGroupsResponse` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not iam_id:
            raise ValueError('iam_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='remove_member_from_all_access_groups',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['iam_id']
        path_param_values = self.encode_path_vars(iam_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/groups/_allgroups/members/{iam_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def add_member_to_multiple_access_groups(
        self,
        account_id: str,
        iam_id: str,
        *,
        type: str = None,
        groups: List[str] = None,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Add member to multiple access groups.

        This API will add a member to multiple access groups in an account. The limit of
        how many groups that can be in the request is 50. The response is a list of
        results that show if adding the member to each group was successful or not.

        :param str account_id: Account ID of the API keys(s) to query. If a service
               IAM ID is specified in iam_id then account_id must match the account of the
               IAM ID. If a user IAM ID is specified in iam_id then then account_id must
               match the account of the Authorization token.
        :param str iam_id: The IAM identifier.
        :param str type: (optional) The type of the member, must be either "user",
               "service" or "profile".
        :param List[str] groups: (optional) The ids of the access groups a given
               member is to be added to.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AddMembershipMultipleGroupsResponse` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not iam_id:
            raise ValueError('iam_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='add_member_to_multiple_access_groups',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
        }

        data = {
            'type': type,
            'groups': groups,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['iam_id']
        path_param_values = self.encode_path_vars(iam_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/groups/_allgroups/members/{iam_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PUT',
            url=url,
            headers=headers,
            params=params,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # Rule operations
    #########################

    def add_access_group_rule(
        self,
        access_group_id: str,
        expiration: int,
        realm_name: str,
        conditions: List['RuleConditions'],
        *,
        name: str = None,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create rule for an access group.

        Rules can be used to dynamically add users to an access group. If a user's SAML
        assertions match the rule's conditions during login, the user will be dynamically
        added to the group. The duration of the user's access to the group is determined
        by the `expiration` field. After access expires, the user will need to log in
        again to regain access. Note that the condition's value field must be a
        stringified JSON value. [Consult this documentation for further explanation of
        dynamic rules.](/docs/account?topic=account-rules).

        :param str access_group_id: The access group identifier.
        :param int expiration: Session duration in hours. Access group membership
               is revoked after this time period expires. Users must log back in to
               refresh their access group membership.
        :param str realm_name: The URL of the identity provider (IdP).
        :param List[RuleConditions] conditions: A list of conditions that
               identities must satisfy to gain access group membership.
        :param str name: (optional) The name of the dynaimic rule.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Rule` object
        """

        if not access_group_id:
            raise ValueError('access_group_id must be provided')
        if expiration is None:
            raise ValueError('expiration must be provided')
        if realm_name is None:
            raise ValueError('realm_name must be provided')
        if conditions is None:
            raise ValueError('conditions must be provided')
        conditions = [convert_model(x) for x in conditions]
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='add_access_group_rule',
        )
        headers.update(sdk_headers)

        data = {
            'expiration': expiration,
            'realm_name': realm_name,
            'conditions': conditions,
            'name': name,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['access_group_id']
        path_param_values = self.encode_path_vars(access_group_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/groups/{access_group_id}/rules'.format(**path_param_dict)
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def list_access_group_rules(
        self,
        access_group_id: str,
        *,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List access group rules.

        This API lists all rules in a given access group. Because only a few rules are
        created on each group, there is no pagination or sorting support on this API.

        :param str access_group_id: The access group identifier.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `RulesList` object
        """

        if not access_group_id:
            raise ValueError('access_group_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='list_access_group_rules',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['access_group_id']
        path_param_values = self.encode_path_vars(access_group_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/groups/{access_group_id}/rules'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def get_access_group_rule(
        self,
        access_group_id: str,
        rule_id: str,
        *,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get an access group rule.

        Retrieve a rule from an access group. A revision number is returned in the `ETag`
        header, which is needed when updating the rule.

        :param str access_group_id: The access group identifier.
        :param str rule_id: The rule to get.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Rule` object
        """

        if not access_group_id:
            raise ValueError('access_group_id must be provided')
        if not rule_id:
            raise ValueError('rule_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='get_access_group_rule',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['access_group_id', 'rule_id']
        path_param_values = self.encode_path_vars(access_group_id, rule_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/groups/{access_group_id}/rules/{rule_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def replace_access_group_rule(
        self,
        access_group_id: str,
        rule_id: str,
        if_match: str,
        expiration: int,
        realm_name: str,
        conditions: List['RuleConditions'],
        *,
        name: str = None,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Replace an access group rule.

        Update the body of an existing rule using this API. An `If-Match` header must be
        populated with the rule's most recent revision number (which can be acquired in
        the `Get an access group rule` API).

        :param str access_group_id: The access group identifier.
        :param str rule_id: The rule to get.
        :param str if_match: The current revision number of the rule being updated.
               This can be found in the Get Rule response ETag header.
        :param int expiration: Session duration in hours. Access group membership
               is revoked after this time period expires. Users must log back in to
               refresh their access group membership.
        :param str realm_name: The URL of the identity provider (IdP).
        :param List[RuleConditions] conditions: A list of conditions that
               identities must satisfy to gain access group membership.
        :param str name: (optional) The name of the dynaimic rule.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Rule` object
        """

        if not access_group_id:
            raise ValueError('access_group_id must be provided')
        if not rule_id:
            raise ValueError('rule_id must be provided')
        if not if_match:
            raise ValueError('if_match must be provided')
        if expiration is None:
            raise ValueError('expiration must be provided')
        if realm_name is None:
            raise ValueError('realm_name must be provided')
        if conditions is None:
            raise ValueError('conditions must be provided')
        conditions = [convert_model(x) for x in conditions]
        headers = {
            'If-Match': if_match,
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='replace_access_group_rule',
        )
        headers.update(sdk_headers)

        data = {
            'expiration': expiration,
            'realm_name': realm_name,
            'conditions': conditions,
            'name': name,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['access_group_id', 'rule_id']
        path_param_values = self.encode_path_vars(access_group_id, rule_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/groups/{access_group_id}/rules/{rule_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PUT',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def remove_access_group_rule(
        self,
        access_group_id: str,
        rule_id: str,
        *,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete an access group rule.

        Remove one rule from a group using this API. If the operation is successful, only
        a `204 - No Content` response with no body is returned. However, if any error
        occurs, the standard error format will be returned.

        :param str access_group_id: The access group identifier.
        :param str rule_id: The rule to get.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not access_group_id:
            raise ValueError('access_group_id must be provided')
        if not rule_id:
            raise ValueError('rule_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='remove_access_group_rule',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['access_group_id', 'rule_id']
        path_param_values = self.encode_path_vars(access_group_id, rule_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/groups/{access_group_id}/rules/{rule_id}'.format(**path_param_dict)
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
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get account settings.

        Retrieve the access groups settings for a specific account.

        :param str account_id: Account ID of the API keys(s) to query. If a service
               IAM ID is specified in iam_id then account_id must match the account of the
               IAM ID. If a user IAM ID is specified in iam_id then then account_id must
               match the account of the Authorization token.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccountSettings` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='get_account_settings',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v2/groups/settings'
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
        account_id: str,
        *,
        public_access_enabled: bool = None,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Update account settings.

        Update the access groups settings for a specific account. Note: When the
        `public_access_enabled` setting is set to false, all policies within the account
        attached to the Public Access group will be deleted. Only set
        `public_access_enabled` to false if you are sure that you want those policies to
        be removed.

        :param str account_id: Account ID of the API keys(s) to query. If a service
               IAM ID is specified in iam_id then account_id must match the account of the
               IAM ID. If a user IAM ID is specified in iam_id then then account_id must
               match the account of the Authorization token.
        :param bool public_access_enabled: (optional) This flag controls the public
               access feature within the account. It is set to true by default. Note: When
               this flag is set to false, all policies within the account attached to the
               Public Access group will be deleted.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccountSettings` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='update_account_settings',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
        }

        data = {
            'public_access_enabled': public_access_enabled,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v2/groups/settings'
        request = self.prepare_request(
            method='PATCH',
            url=url,
            headers=headers,
            params=params,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # Template operations
    #########################

    def create_template(
        self,
        name: str,
        account_id: str,
        *,
        description: str = None,
        group: 'AccessGroupRequest' = None,
        policy_template_references: List['PolicyTemplates'] = None,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create template.

        Create an access group template. Make sure that the template is generic enough to
        apply to multiple different child accounts. Before you can assign an access group
        template to child accounts, you must commit it so that no further changes can be
        made to the version.

        :param str name: Give the access group template a unique name that doesn't
               conflict with an existing access group templates in the account.
        :param str account_id: Enterprise account id in which the template will be
               created.
        :param str description: (optional) Assign an optional description for the
               access group template.
        :param AccessGroupRequest group: (optional) Access Group Component.
        :param List[PolicyTemplates] policy_template_references: (optional)
               Existing policy templates that you can reference to assign access in the
               Access group input component.
        :param str transaction_id: (optional) An optional transaction id for the
               request.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TemplateResponse` object
        """

        if name is None:
            raise ValueError('name must be provided')
        if account_id is None:
            raise ValueError('account_id must be provided')
        if group is not None:
            group = convert_model(group)
        if policy_template_references is not None:
            policy_template_references = [convert_model(x) for x in policy_template_references]
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='create_template',
        )
        headers.update(sdk_headers)

        data = {
            'name': name,
            'account_id': account_id,
            'description': description,
            'group': group,
            'policy_template_references': policy_template_references,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/group_templates'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def list_templates(
        self,
        account_id: str,
        *,
        transaction_id: str = None,
        limit: int = None,
        offset: int = None,
        verbose: bool = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List templates.

        List the access group templates in an enterprise account.

        :param str account_id: Enterprise account ID.
        :param str transaction_id: (optional) An optional transaction id for the
               request.
        :param int limit: (optional) Return up to this limit of results where limit
               is between 0 and 100.
        :param int offset: (optional) The offset of the first result item to be
               returned.
        :param bool verbose: (optional) If `verbose=true`, IAM resource details are
               returned. If performance is a concern, leave the `verbose` parameter off so
               that details are not retrieved.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ListTemplatesResponse` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='list_templates',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
            'limit': limit,
            'offset': offset,
            'verbose': verbose,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/group_templates'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def create_template_version(
        self,
        template_id: str,
        *,
        name: str = None,
        description: str = None,
        group: 'AccessGroupRequest' = None,
        policy_template_references: List['PolicyTemplates'] = None,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create template version.

        Create a new version of an access group template.

        :param str template_id: ID of the template that you want to create a new
               version of.
        :param str name: (optional) This is an optional field. If the field is
               included it will change the name value for all existing versions of the
               template..
        :param str description: (optional) Assign an optional description for the
               access group template version.
        :param AccessGroupRequest group: (optional) Access Group Component.
        :param List[PolicyTemplates] policy_template_references: (optional) The
               policy templates associated with the template version.
        :param str transaction_id: (optional) An optional transaction id for the
               request.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TemplateVersionResponse` object
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        if group is not None:
            group = convert_model(group)
        if policy_template_references is not None:
            policy_template_references = [convert_model(x) for x in policy_template_references]
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='create_template_version',
        )
        headers.update(sdk_headers)

        data = {
            'name': name,
            'description': description,
            'group': group,
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
        url = '/v1/group_templates/{template_id}/versions'.format(**path_param_dict)
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def list_template_versions(
        self,
        template_id: str,
        *,
        limit: int = None,
        offset: int = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List template versions.

        List all the versions of an access group template.

        :param str template_id: ID of the template that you want to list all
               versions of.
        :param int limit: (optional) Return up to this limit of results where limit
               is between 0 and 100.
        :param int offset: (optional) The offset of the first result item to be
               returned.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ListTemplateVersionsResponse` object
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='list_template_versions',
        )
        headers.update(sdk_headers)

        params = {
            'limit': limit,
            'offset': offset,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['template_id']
        path_param_values = self.encode_path_vars(template_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/group_templates/{template_id}/versions'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def get_template_version(
        self,
        template_id: str,
        version_num: str,
        *,
        verbose: bool = None,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get template version.

        Get a specific version of a template.

        :param str template_id: ID of the template to get a specific version of.
        :param str version_num: Version number.
        :param bool verbose: (optional) If `verbose=true`, IAM resource details are
               returned. If performance is a concern, leave the `verbose` parameter off so
               that details are not retrieved.
        :param str transaction_id: (optional) An optional transaction id for the
               request.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TemplateVersionResponse` object
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        if not version_num:
            raise ValueError('version_num must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='get_template_version',
        )
        headers.update(sdk_headers)

        params = {
            'verbose': verbose,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['template_id', 'version_num']
        path_param_values = self.encode_path_vars(template_id, version_num)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/group_templates/{template_id}/versions/{version_num}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def update_template_version(
        self,
        template_id: str,
        version_num: str,
        if_match: str,
        *,
        name: str = None,
        description: str = None,
        group: 'AccessGroupRequest' = None,
        policy_template_references: List['PolicyTemplates'] = None,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Update template version.

        Update a template version. You can only update a version that isn't committed.
        Create a new version if you need to update a committed version.

        :param str template_id: ID of the template.
        :param str version_num: Version number of the template.
        :param str if_match: ETag value of the template version document.
        :param str name: (optional) This is an optional field. If the field is
               included it will change the name value for all existing versions of the
               template..
        :param str description: (optional) Assign an optional description for the
               access group template version.
        :param AccessGroupRequest group: (optional) Access Group Component.
        :param List[PolicyTemplates] policy_template_references: (optional) The
               policy templates associated with the template version.
        :param str transaction_id: (optional) transaction id in header.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TemplateVersionResponse` object
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        if not version_num:
            raise ValueError('version_num must be provided')
        if not if_match:
            raise ValueError('if_match must be provided')
        if group is not None:
            group = convert_model(group)
        if policy_template_references is not None:
            policy_template_references = [convert_model(x) for x in policy_template_references]
        headers = {
            'If-Match': if_match,
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='update_template_version',
        )
        headers.update(sdk_headers)

        data = {
            'name': name,
            'description': description,
            'group': group,
            'policy_template_references': policy_template_references,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['template_id', 'version_num']
        path_param_values = self.encode_path_vars(template_id, version_num)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/group_templates/{template_id}/versions/{version_num}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PUT',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_template_version(
        self,
        template_id: str,
        version_num: str,
        *,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete template version.

        Delete a template version. You must remove all assignments for a template version
        before you can delete it.

        :param str template_id: ID of the template to delete.
        :param str version_num: version number in path.
        :param str transaction_id: (optional) An optional transaction id for the
               request.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        if not version_num:
            raise ValueError('version_num must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='delete_template_version',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['template_id', 'version_num']
        path_param_values = self.encode_path_vars(template_id, version_num)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/group_templates/{template_id}/versions/{version_num}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def commit_template(
        self,
        template_id: str,
        version_num: str,
        if_match: str,
        *,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Commit a template.

        Commit a template version. You must do this before you can assign a template
        version to child accounts. After you commit the template version, you can't make
        any further changes.

        :param str template_id: ID of the template to commit.
        :param str version_num: version number in path.
        :param str if_match: ETag value of the template version document.
        :param str transaction_id: (optional) An optional transaction id for the
               request.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        if not version_num:
            raise ValueError('version_num must be provided')
        if not if_match:
            raise ValueError('if_match must be provided')
        headers = {
            'If-Match': if_match,
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='commit_template',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['template_id', 'version_num']
        path_param_values = self.encode_path_vars(template_id, version_num)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/group_templates/{template_id}/versions/{version_num}/commit'.format(**path_param_dict)
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def get_latest_template_version(
        self,
        template_id: str,
        *,
        verbose: bool = None,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get latest template version.

        Get the latest version of a template.

        :param str template_id: ID of the template to get a specific version of.
        :param bool verbose: (optional) If `verbose=true`, IAM resource details are
               returned. If performance is a concern, leave the `verbose` parameter off so
               that details are not retrieved.
        :param str transaction_id: (optional) An optional transaction id for the
               request.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TemplateVersionResponse` object
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='get_latest_template_version',
        )
        headers.update(sdk_headers)

        params = {
            'verbose': verbose,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['template_id']
        path_param_values = self.encode_path_vars(template_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/group_templates/{template_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_template(
        self,
        template_id: str,
        *,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete template.

        Endpoint to delete a template. All access assigned by that template is deleted
        from all of the accounts where the template was assigned.

        :param str template_id: template id parameter.
        :param str transaction_id: (optional) An optional transaction id for the
               request.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not template_id:
            raise ValueError('template_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='delete_template',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['template_id']
        path_param_values = self.encode_path_vars(template_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/group_templates/{template_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # Template assignment operations
    #########################

    def create_assignment(
        self,
        template_id: str,
        template_version: str,
        target_type: str,
        target: str,
        *,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create assignment.

        Assign a template version to accounts that have enabled enterprise-managed IAM.
        You can specify individual accounts, or an entire account group to assign the
        template to all current and future child accounts of that account group.

        :param str template_id: The unique identifier of the template to be
               assigned.
        :param str template_version: The version number of the template to be
               assigned.
        :param str target_type: The type of the entity to which the template should
               be assigned, e.g. 'Account', 'AccountGroup', etc.
        :param str target: The unique identifier of the entity to which the
               template should be assigned.
        :param str transaction_id: (optional) An optional transaction id for the
               request.
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
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='create_assignment',
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

        url = '/v1/group_assignments'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def list_assignments(
        self,
        account_id: str,
        *,
        template_id: str = None,
        template_version: str = None,
        target: str = None,
        status: str = None,
        transaction_id: str = None,
        limit: int = None,
        offset: int = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List assignments.

        List template assignments from an enterprise account.

        :param str account_id: Enterprise account ID.
        :param str template_id: (optional) Filter results by Template Id.
        :param str template_version: (optional) Filter results by Template Version.
        :param str target: (optional) Filter results by the assignment target.
        :param str status: (optional) Filter results by the assignment status.
        :param str transaction_id: (optional) An optional transaction id for the
               request.
        :param int limit: (optional) Return up to this limit of results where limit
               is between 0 and 100.
        :param int offset: (optional) The offset of the first result item to be
               returned.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ListTemplateAssignmentResponse` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='list_assignments',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
            'template_id': template_id,
            'template_version': template_version,
            'target': target,
            'status': status,
            'limit': limit,
            'offset': offset,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/group_assignments'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def get_assignment(
        self,
        assignment_id: str,
        *,
        transaction_id: str = None,
        verbose: bool = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get assignment.

        Get a specific template assignment.

        :param str assignment_id: Assignment ID.
        :param str transaction_id: (optional) An optional transaction id for the
               request.
        :param bool verbose: (optional) Returns resources access group template
               assigned, possible values `true` or `false`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TemplateAssignmentVerboseResponse` object
        """

        if not assignment_id:
            raise ValueError('assignment_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='get_assignment',
        )
        headers.update(sdk_headers)

        params = {
            'verbose': verbose,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['assignment_id']
        path_param_values = self.encode_path_vars(assignment_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/group_assignments/{assignment_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def update_assignment(
        self,
        assignment_id: str,
        if_match: str,
        template_version: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Update Assignment.

        Endpoint to update template assignment.

        :param str assignment_id: ID of the Assignment Record.
        :param str if_match: Version of the Assignment to be updated. Specify the
               version that you retrieved when reading the Assignment. This value helps
               identifying parallel usage of this API. Pass * to indicate to update any
               version available. This might result in stale updates.
        :param str template_version: Template version which shall be applied to the
               assignment.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `TemplateAssignmentVerboseResponse` object
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
            service_version='V2',
            operation_id='update_assignment',
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
        url = '/v1/group_assignments/{assignment_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PATCH',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_assignment(
        self,
        assignment_id: str,
        *,
        transaction_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete assignment.

        Delete an access group template assignment.

        :param str assignment_id: assignment id path parameter.
        :param str transaction_id: (optional) An optional transaction id for the
               request.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not assignment_id:
            raise ValueError('assignment_id must be provided')
        headers = {
            'Transaction-Id': transaction_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='delete_assignment',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['assignment_id']
        path_param_values = self.encode_path_vars(assignment_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/group_assignments/{assignment_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response


class ListAssignmentsEnums:
    """
    Enums for list_assignments parameters.
    """

    class Status(str, Enum):
        """
        Filter results by the assignment status.
        """

        ACCEPTED = 'accepted'
        IN_PROGRESS = 'in_progress'
        SUCCEEDED = 'succeeded'
        FAILED = 'failed'


##############################################################################
# Models
##############################################################################


class AccessActionControls:
    """
    Control whether or not access group administrators in child accounts can add access
    policies to the enterprise-managed access group in their account.

    :attr bool add: (optional) Action control for adding access policies to an
          enterprise-managed access group in a child account. If an access group
          administrator in a child account adds a policy, they can always update or remove
          it. Note that if conflicts arise between an update to this control in a new
          version and polices added to the access group by an administrator in a child
          account, you must resolve those conflicts in the child account. This prevents
          breaking access in the child account. For more information, see [Working with
          versions](https://test.cloud.ibm.com/docs/secure-enterprise?topic=secure-enterprise-working-with-versions#new-version-scenarios).
    """

    def __init__(
        self,
        *,
        add: bool = None,
    ) -> None:
        """
        Initialize a AccessActionControls object.

        :param bool add: (optional) Action control for adding access policies to an
               enterprise-managed access group in a child account. If an access group
               administrator in a child account adds a policy, they can always update or
               remove it. Note that if conflicts arise between an update to this control
               in a new version and polices added to the access group by an administrator
               in a child account, you must resolve those conflicts in the child account.
               This prevents breaking access in the child account. For more information,
               see [Working with
               versions](https://test.cloud.ibm.com/docs/secure-enterprise?topic=secure-enterprise-working-with-versions#new-version-scenarios).
        """
        self.add = add

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccessActionControls':
        """Initialize a AccessActionControls object from a json dictionary."""
        args = {}
        if 'add' in _dict:
            args['add'] = _dict.get('add')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccessActionControls object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'add') and self.add is not None:
            _dict['add'] = self.add
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AccessActionControls object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccessActionControls') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccessActionControls') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AccessGroupRequest:
    """
    Access Group Component.

    :attr str name: Give the access group a unique name that doesn't conflict with
          other templates access group name in the given account. This is shown in child
          accounts.
    :attr str description: (optional) Access group description. This is shown in
          child accounts.
    :attr Members members: (optional) Array of enterprise users to add to the
          template. All enterprise users that you add to the template must be invited to
          the child accounts where the template is assigned.
    :attr Assertions assertions: (optional) Assertions Input Component.
    :attr GroupActionControls action_controls: (optional) Access group action
          controls component.
    """

    def __init__(
        self,
        name: str,
        *,
        description: str = None,
        members: 'Members' = None,
        assertions: 'Assertions' = None,
        action_controls: 'GroupActionControls' = None,
    ) -> None:
        """
        Initialize a AccessGroupRequest object.

        :param str name: Give the access group a unique name that doesn't conflict
               with other templates access group name in the given account. This is shown
               in child accounts.
        :param str description: (optional) Access group description. This is shown
               in child accounts.
        :param Members members: (optional) Array of enterprise users to add to the
               template. All enterprise users that you add to the template must be invited
               to the child accounts where the template is assigned.
        :param Assertions assertions: (optional) Assertions Input Component.
        :param GroupActionControls action_controls: (optional) Access group action
               controls component.
        """
        self.name = name
        self.description = description
        self.members = members
        self.assertions = assertions
        self.action_controls = action_controls

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccessGroupRequest':
        """Initialize a AccessGroupRequest object from a json dictionary."""
        args = {}
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        else:
            raise ValueError('Required property \'name\' not present in AccessGroupRequest JSON')
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        if 'members' in _dict:
            args['members'] = Members.from_dict(_dict.get('members'))
        if 'assertions' in _dict:
            args['assertions'] = Assertions.from_dict(_dict.get('assertions'))
        if 'action_controls' in _dict:
            args['action_controls'] = GroupActionControls.from_dict(_dict.get('action_controls'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccessGroupRequest object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'members') and self.members is not None:
            if isinstance(self.members, dict):
                _dict['members'] = self.members
            else:
                _dict['members'] = self.members.to_dict()
        if hasattr(self, 'assertions') and self.assertions is not None:
            if isinstance(self.assertions, dict):
                _dict['assertions'] = self.assertions
            else:
                _dict['assertions'] = self.assertions.to_dict()
        if hasattr(self, 'action_controls') and self.action_controls is not None:
            if isinstance(self.action_controls, dict):
                _dict['action_controls'] = self.action_controls
            else:
                _dict['action_controls'] = self.action_controls.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AccessGroupRequest object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccessGroupRequest') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccessGroupRequest') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AccessGroupResponse:
    """
    Access Group Component.

    :attr str name: Give the access group a unique name that doesn't conflict with
          other templates access group name in the given account. This is shown in child
          accounts.
    :attr str description: (optional) Access group description. This is shown in
          child accounts.
    :attr Members members: (optional) Array of enterprise users to add to the
          template. All enterprise users that you add to the template must be invited to
          the child accounts where the template is assigned.
    :attr Assertions assertions: (optional) Assertions Input Component.
    :attr GroupActionControls action_controls: (optional) Access group action
          controls component.
    """

    def __init__(
        self,
        name: str,
        *,
        description: str = None,
        members: 'Members' = None,
        assertions: 'Assertions' = None,
        action_controls: 'GroupActionControls' = None,
    ) -> None:
        """
        Initialize a AccessGroupResponse object.

        :param str name: Give the access group a unique name that doesn't conflict
               with other templates access group name in the given account. This is shown
               in child accounts.
        :param str description: (optional) Access group description. This is shown
               in child accounts.
        :param Members members: (optional) Array of enterprise users to add to the
               template. All enterprise users that you add to the template must be invited
               to the child accounts where the template is assigned.
        :param Assertions assertions: (optional) Assertions Input Component.
        :param GroupActionControls action_controls: (optional) Access group action
               controls component.
        """
        self.name = name
        self.description = description
        self.members = members
        self.assertions = assertions
        self.action_controls = action_controls

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccessGroupResponse':
        """Initialize a AccessGroupResponse object from a json dictionary."""
        args = {}
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        else:
            raise ValueError('Required property \'name\' not present in AccessGroupResponse JSON')
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        if 'members' in _dict:
            args['members'] = Members.from_dict(_dict.get('members'))
        if 'assertions' in _dict:
            args['assertions'] = Assertions.from_dict(_dict.get('assertions'))
        if 'action_controls' in _dict:
            args['action_controls'] = GroupActionControls.from_dict(_dict.get('action_controls'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccessGroupResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'members') and self.members is not None:
            if isinstance(self.members, dict):
                _dict['members'] = self.members
            else:
                _dict['members'] = self.members.to_dict()
        if hasattr(self, 'assertions') and self.assertions is not None:
            if isinstance(self.assertions, dict):
                _dict['assertions'] = self.assertions
            else:
                _dict['assertions'] = self.assertions.to_dict()
        if hasattr(self, 'action_controls') and self.action_controls is not None:
            if isinstance(self.action_controls, dict):
                _dict['action_controls'] = self.action_controls
            else:
                _dict['action_controls'] = self.action_controls.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AccessGroupResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccessGroupResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccessGroupResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AccountSettings:
    """
    The access groups settings for a specific account.

    :attr str account_id: (optional) The account id of the settings being shown.
    :attr datetime last_modified_at: (optional) The timestamp the settings were last
          edited at.
    :attr str last_modified_by_id: (optional) The `iam_id` of the entity that last
          modified the settings.
    :attr bool public_access_enabled: (optional) This flag controls the public
          access feature within the account. It is set to true by default. Note: When this
          flag is set to false, all policies within the account attached to the Public
          Access group will be deleted.
    """

    def __init__(
        self,
        *,
        account_id: str = None,
        last_modified_at: datetime = None,
        last_modified_by_id: str = None,
        public_access_enabled: bool = None,
    ) -> None:
        """
        Initialize a AccountSettings object.

        :param str account_id: (optional) The account id of the settings being
               shown.
        :param datetime last_modified_at: (optional) The timestamp the settings
               were last edited at.
        :param str last_modified_by_id: (optional) The `iam_id` of the entity that
               last modified the settings.
        :param bool public_access_enabled: (optional) This flag controls the public
               access feature within the account. It is set to true by default. Note: When
               this flag is set to false, all policies within the account attached to the
               Public Access group will be deleted.
        """
        self.account_id = account_id
        self.last_modified_at = last_modified_at
        self.last_modified_by_id = last_modified_by_id
        self.public_access_enabled = public_access_enabled

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccountSettings':
        """Initialize a AccountSettings object from a json dictionary."""
        args = {}
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        if 'last_modified_at' in _dict:
            args['last_modified_at'] = string_to_datetime(_dict.get('last_modified_at'))
        if 'last_modified_by_id' in _dict:
            args['last_modified_by_id'] = _dict.get('last_modified_by_id')
        if 'public_access_enabled' in _dict:
            args['public_access_enabled'] = _dict.get('public_access_enabled')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccountSettings object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'last_modified_at') and self.last_modified_at is not None:
            _dict['last_modified_at'] = datetime_to_string(self.last_modified_at)
        if hasattr(self, 'last_modified_by_id') and self.last_modified_by_id is not None:
            _dict['last_modified_by_id'] = self.last_modified_by_id
        if hasattr(self, 'public_access_enabled') and self.public_access_enabled is not None:
            _dict['public_access_enabled'] = self.public_access_enabled
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AccountSettings object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccountSettings') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccountSettings') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AddGroupMembersRequestMembersItem:
    """
    AddGroupMembersRequestMembersItem.

    :attr str iam_id: The IBMid, service ID or trusted profile ID of the member.
    :attr str type: The type of the member, must be either "user", "service" or
          "profile".
    """

    def __init__(
        self,
        iam_id: str,
        type: str,
    ) -> None:
        """
        Initialize a AddGroupMembersRequestMembersItem object.

        :param str iam_id: The IBMid, service ID or trusted profile ID of the
               member.
        :param str type: The type of the member, must be either "user", "service"
               or "profile".
        """
        self.iam_id = iam_id
        self.type = type

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AddGroupMembersRequestMembersItem':
        """Initialize a AddGroupMembersRequestMembersItem object from a json dictionary."""
        args = {}
        if 'iam_id' in _dict:
            args['iam_id'] = _dict.get('iam_id')
        else:
            raise ValueError('Required property \'iam_id\' not present in AddGroupMembersRequestMembersItem JSON')
        if 'type' in _dict:
            args['type'] = _dict.get('type')
        else:
            raise ValueError('Required property \'type\' not present in AddGroupMembersRequestMembersItem JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AddGroupMembersRequestMembersItem object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'iam_id') and self.iam_id is not None:
            _dict['iam_id'] = self.iam_id
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AddGroupMembersRequestMembersItem object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AddGroupMembersRequestMembersItem') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AddGroupMembersRequestMembersItem') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AddGroupMembersResponse:
    """
    The members added to an access group.

    :attr List[AddGroupMembersResponseMembersItem] members: (optional) The members
          added to an access group.
    """

    def __init__(
        self,
        *,
        members: List['AddGroupMembersResponseMembersItem'] = None,
    ) -> None:
        """
        Initialize a AddGroupMembersResponse object.

        :param List[AddGroupMembersResponseMembersItem] members: (optional) The
               members added to an access group.
        """
        self.members = members

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AddGroupMembersResponse':
        """Initialize a AddGroupMembersResponse object from a json dictionary."""
        args = {}
        if 'members' in _dict:
            args['members'] = [AddGroupMembersResponseMembersItem.from_dict(v) for v in _dict.get('members')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AddGroupMembersResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'members') and self.members is not None:
            members_list = []
            for v in self.members:
                if isinstance(v, dict):
                    members_list.append(v)
                else:
                    members_list.append(v.to_dict())
            _dict['members'] = members_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AddGroupMembersResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AddGroupMembersResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AddGroupMembersResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AddGroupMembersResponseMembersItem:
    """
    AddGroupMembersResponseMembersItem.

    :attr str iam_id: (optional) The IBMid or Service Id of the member.
    :attr str type: (optional) The member type - either `user`, `service` or
          `profile`.
    :attr datetime created_at: (optional) The timestamp of when the membership was
          created.
    :attr str created_by_id: (optional) The `iam_id` of the entity that created the
          membership.
    :attr int status_code: (optional) The outcome of the operation on this `iam_id`.
    :attr str trace: (optional) A transaction-id that can be used for debugging
          purposes.
    :attr List[Error] errors: (optional) A list of errors that occurred when trying
          to add members to a group.
    """

    def __init__(
        self,
        *,
        iam_id: str = None,
        type: str = None,
        created_at: datetime = None,
        created_by_id: str = None,
        status_code: int = None,
        trace: str = None,
        errors: List['Error'] = None,
    ) -> None:
        """
        Initialize a AddGroupMembersResponseMembersItem object.

        :param str iam_id: (optional) The IBMid or Service Id of the member.
        :param str type: (optional) The member type - either `user`, `service` or
               `profile`.
        :param datetime created_at: (optional) The timestamp of when the membership
               was created.
        :param str created_by_id: (optional) The `iam_id` of the entity that
               created the membership.
        :param int status_code: (optional) The outcome of the operation on this
               `iam_id`.
        :param str trace: (optional) A transaction-id that can be used for
               debugging purposes.
        :param List[Error] errors: (optional) A list of errors that occurred when
               trying to add members to a group.
        """
        self.iam_id = iam_id
        self.type = type
        self.created_at = created_at
        self.created_by_id = created_by_id
        self.status_code = status_code
        self.trace = trace
        self.errors = errors

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AddGroupMembersResponseMembersItem':
        """Initialize a AddGroupMembersResponseMembersItem object from a json dictionary."""
        args = {}
        if 'iam_id' in _dict:
            args['iam_id'] = _dict.get('iam_id')
        if 'type' in _dict:
            args['type'] = _dict.get('type')
        if 'created_at' in _dict:
            args['created_at'] = string_to_datetime(_dict.get('created_at'))
        if 'created_by_id' in _dict:
            args['created_by_id'] = _dict.get('created_by_id')
        if 'status_code' in _dict:
            args['status_code'] = _dict.get('status_code')
        if 'trace' in _dict:
            args['trace'] = _dict.get('trace')
        if 'errors' in _dict:
            args['errors'] = [Error.from_dict(v) for v in _dict.get('errors')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AddGroupMembersResponseMembersItem object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'iam_id') and self.iam_id is not None:
            _dict['iam_id'] = self.iam_id
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'created_by_id') and self.created_by_id is not None:
            _dict['created_by_id'] = self.created_by_id
        if hasattr(self, 'status_code') and self.status_code is not None:
            _dict['status_code'] = self.status_code
        if hasattr(self, 'trace') and self.trace is not None:
            _dict['trace'] = self.trace
        if hasattr(self, 'errors') and self.errors is not None:
            errors_list = []
            for v in self.errors:
                if isinstance(v, dict):
                    errors_list.append(v)
                else:
                    errors_list.append(v.to_dict())
            _dict['errors'] = errors_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AddGroupMembersResponseMembersItem object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AddGroupMembersResponseMembersItem') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AddGroupMembersResponseMembersItem') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AddMembershipMultipleGroupsResponse:
    """
    The response from the add member to multiple access groups request.

    :attr str iam_id: (optional) The iam_id of a member.
    :attr List[AddMembershipMultipleGroupsResponseGroupsItem] groups: (optional) The
          list of access groups a member was added to.
    """

    def __init__(
        self,
        *,
        iam_id: str = None,
        groups: List['AddMembershipMultipleGroupsResponseGroupsItem'] = None,
    ) -> None:
        """
        Initialize a AddMembershipMultipleGroupsResponse object.

        :param str iam_id: (optional) The iam_id of a member.
        :param List[AddMembershipMultipleGroupsResponseGroupsItem] groups:
               (optional) The list of access groups a member was added to.
        """
        self.iam_id = iam_id
        self.groups = groups

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AddMembershipMultipleGroupsResponse':
        """Initialize a AddMembershipMultipleGroupsResponse object from a json dictionary."""
        args = {}
        if 'iam_id' in _dict:
            args['iam_id'] = _dict.get('iam_id')
        if 'groups' in _dict:
            args['groups'] = [AddMembershipMultipleGroupsResponseGroupsItem.from_dict(v) for v in _dict.get('groups')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AddMembershipMultipleGroupsResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'iam_id') and self.iam_id is not None:
            _dict['iam_id'] = self.iam_id
        if hasattr(self, 'groups') and self.groups is not None:
            groups_list = []
            for v in self.groups:
                if isinstance(v, dict):
                    groups_list.append(v)
                else:
                    groups_list.append(v.to_dict())
            _dict['groups'] = groups_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AddMembershipMultipleGroupsResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AddMembershipMultipleGroupsResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AddMembershipMultipleGroupsResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AddMembershipMultipleGroupsResponseGroupsItem:
    """
    AddMembershipMultipleGroupsResponseGroupsItem.

    :attr str access_group_id: (optional) The access group that the member is to be
          added to.
    :attr int status_code: (optional) The outcome of the add membership operation on
          this `access_group_id`.
    :attr str trace: (optional) A transaction-id that can be used for debugging
          purposes.
    :attr List[Error] errors: (optional) List of errors encountered when adding
          member to access group.
    """

    def __init__(
        self,
        *,
        access_group_id: str = None,
        status_code: int = None,
        trace: str = None,
        errors: List['Error'] = None,
    ) -> None:
        """
        Initialize a AddMembershipMultipleGroupsResponseGroupsItem object.

        :param str access_group_id: (optional) The access group that the member is
               to be added to.
        :param int status_code: (optional) The outcome of the add membership
               operation on this `access_group_id`.
        :param str trace: (optional) A transaction-id that can be used for
               debugging purposes.
        :param List[Error] errors: (optional) List of errors encountered when
               adding member to access group.
        """
        self.access_group_id = access_group_id
        self.status_code = status_code
        self.trace = trace
        self.errors = errors

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AddMembershipMultipleGroupsResponseGroupsItem':
        """Initialize a AddMembershipMultipleGroupsResponseGroupsItem object from a json dictionary."""
        args = {}
        if 'access_group_id' in _dict:
            args['access_group_id'] = _dict.get('access_group_id')
        if 'status_code' in _dict:
            args['status_code'] = _dict.get('status_code')
        if 'trace' in _dict:
            args['trace'] = _dict.get('trace')
        if 'errors' in _dict:
            args['errors'] = [Error.from_dict(v) for v in _dict.get('errors')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AddMembershipMultipleGroupsResponseGroupsItem object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'access_group_id') and self.access_group_id is not None:
            _dict['access_group_id'] = self.access_group_id
        if hasattr(self, 'status_code') and self.status_code is not None:
            _dict['status_code'] = self.status_code
        if hasattr(self, 'trace') and self.trace is not None:
            _dict['trace'] = self.trace
        if hasattr(self, 'errors') and self.errors is not None:
            errors_list = []
            for v in self.errors:
                if isinstance(v, dict):
                    errors_list.append(v)
                else:
                    errors_list.append(v.to_dict())
            _dict['errors'] = errors_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AddMembershipMultipleGroupsResponseGroupsItem object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AddMembershipMultipleGroupsResponseGroupsItem') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AddMembershipMultipleGroupsResponseGroupsItem') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Assertions:
    """
    Assertions Input Component.

    :attr List[AssertionsRule] rules: (optional) Dynamic rules to automatically add
          federated users to access groups based on specific identity attributes.
    :attr AssertionsActionControls action_controls: (optional) Control whether or
          not access group administrators in child accounts can add, remove, and update
          dynamic rules for the enterprise-managed access group in their account. The
          inner level RuleActionControls override these `remove` and `update` action
          controls.
    """

    def __init__(
        self,
        *,
        rules: List['AssertionsRule'] = None,
        action_controls: 'AssertionsActionControls' = None,
    ) -> None:
        """
        Initialize a Assertions object.

        :param List[AssertionsRule] rules: (optional) Dynamic rules to
               automatically add federated users to access groups based on specific
               identity attributes.
        :param AssertionsActionControls action_controls: (optional) Control whether
               or not access group administrators in child accounts can add, remove, and
               update dynamic rules for the enterprise-managed access group in their
               account. The inner level RuleActionControls override these `remove` and
               `update` action controls.
        """
        self.rules = rules
        self.action_controls = action_controls

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Assertions':
        """Initialize a Assertions object from a json dictionary."""
        args = {}
        if 'rules' in _dict:
            args['rules'] = [AssertionsRule.from_dict(v) for v in _dict.get('rules')]
        if 'action_controls' in _dict:
            args['action_controls'] = AssertionsActionControls.from_dict(_dict.get('action_controls'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Assertions object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'rules') and self.rules is not None:
            rules_list = []
            for v in self.rules:
                if isinstance(v, dict):
                    rules_list.append(v)
                else:
                    rules_list.append(v.to_dict())
            _dict['rules'] = rules_list
        if hasattr(self, 'action_controls') and self.action_controls is not None:
            if isinstance(self.action_controls, dict):
                _dict['action_controls'] = self.action_controls
            else:
                _dict['action_controls'] = self.action_controls.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Assertions object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Assertions') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Assertions') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AssertionsActionControls:
    """
    Control whether or not access group administrators in child accounts can add, remove,
    and update dynamic rules for the enterprise-managed access group in their account. The
    inner level RuleActionControls override these `remove` and `update` action controls.

    :attr bool add: (optional) Action control for adding dynamic rules to an
          enterprise-managed access group. If an access group administrator in a child
          account adds a dynamic rule, they can always update or remove it. Note that if
          conflicts arise between an update to this control and rules added or updated by
          an administrator in the child account, you must resolve those conflicts in the
          child account. This prevents breaking access that the rules might grant in the
          child account. For more information, see [Working with versions].
    :attr bool remove: (optional) Action control for removing enterprise-managed
          dynamic rules in an enterprise-managed access group. Note that if a rule is
          removed from an enterprise-managed access group by an administrator in a child
          account and and you reassign the template, the rule is reinstated.
    """

    def __init__(
        self,
        *,
        add: bool = None,
        remove: bool = None,
    ) -> None:
        """
        Initialize a AssertionsActionControls object.

        :param bool add: (optional) Action control for adding dynamic rules to an
               enterprise-managed access group. If an access group administrator in a
               child account adds a dynamic rule, they can always update or remove it.
               Note that if conflicts arise between an update to this control and rules
               added or updated by an administrator in the child account, you must resolve
               those conflicts in the child account. This prevents breaking access that
               the rules might grant in the child account. For more information, see
               [Working with versions].
        :param bool remove: (optional) Action control for removing
               enterprise-managed dynamic rules in an enterprise-managed access group.
               Note that if a rule is removed from an enterprise-managed access group by
               an administrator in a child account and and you reassign the template, the
               rule is reinstated.
        """
        self.add = add
        self.remove = remove

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AssertionsActionControls':
        """Initialize a AssertionsActionControls object from a json dictionary."""
        args = {}
        if 'add' in _dict:
            args['add'] = _dict.get('add')
        if 'remove' in _dict:
            args['remove'] = _dict.get('remove')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AssertionsActionControls object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'add') and self.add is not None:
            _dict['add'] = self.add
        if hasattr(self, 'remove') and self.remove is not None:
            _dict['remove'] = self.remove
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AssertionsActionControls object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AssertionsActionControls') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AssertionsActionControls') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AssertionsRule:
    """
    Rule Input component.

    :attr str name: (optional) Dynamic rule name.
    :attr int expiration: (optional) Session duration in hours. Access group
          membership is revoked after this time period expires. Users must log back in to
          refresh their access group membership.
    :attr str realm_name: (optional) The identity provider (IdP) URL.
    :attr List[Conditions] conditions: (optional) Conditions of membership. You can
          think of this as a key:value pair.
    :attr RuleActionControls action_controls: (optional) Control whether or not
          access group administrators in child accounts can update and remove this dynamic
          rule in the enterprise-managed access group in their account.This overrides
          outer level AssertionsActionControls.
    """

    def __init__(
        self,
        *,
        name: str = None,
        expiration: int = None,
        realm_name: str = None,
        conditions: List['Conditions'] = None,
        action_controls: 'RuleActionControls' = None,
    ) -> None:
        """
        Initialize a AssertionsRule object.

        :param str name: (optional) Dynamic rule name.
        :param int expiration: (optional) Session duration in hours. Access group
               membership is revoked after this time period expires. Users must log back
               in to refresh their access group membership.
        :param str realm_name: (optional) The identity provider (IdP) URL.
        :param List[Conditions] conditions: (optional) Conditions of membership.
               You can think of this as a key:value pair.
        :param RuleActionControls action_controls: (optional) Control whether or
               not access group administrators in child accounts can update and remove
               this dynamic rule in the enterprise-managed access group in their
               account.This overrides outer level AssertionsActionControls.
        """
        self.name = name
        self.expiration = expiration
        self.realm_name = realm_name
        self.conditions = conditions
        self.action_controls = action_controls

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AssertionsRule':
        """Initialize a AssertionsRule object from a json dictionary."""
        args = {}
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'expiration' in _dict:
            args['expiration'] = _dict.get('expiration')
        if 'realm_name' in _dict:
            args['realm_name'] = _dict.get('realm_name')
        if 'conditions' in _dict:
            args['conditions'] = [Conditions.from_dict(v) for v in _dict.get('conditions')]
        if 'action_controls' in _dict:
            args['action_controls'] = RuleActionControls.from_dict(_dict.get('action_controls'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AssertionsRule object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'expiration') and self.expiration is not None:
            _dict['expiration'] = self.expiration
        if hasattr(self, 'realm_name') and self.realm_name is not None:
            _dict['realm_name'] = self.realm_name
        if hasattr(self, 'conditions') and self.conditions is not None:
            conditions_list = []
            for v in self.conditions:
                if isinstance(v, dict):
                    conditions_list.append(v)
                else:
                    conditions_list.append(v.to_dict())
            _dict['conditions'] = conditions_list
        if hasattr(self, 'action_controls') and self.action_controls is not None:
            if isinstance(self.action_controls, dict):
                _dict['action_controls'] = self.action_controls
            else:
                _dict['action_controls'] = self.action_controls.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AssertionsRule object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AssertionsRule') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AssertionsRule') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AssignmentResourceAccessGroup:
    """
    Assignment Resource Access Group.

    :attr AssignmentResourceEntry group: Assignment resource entry.
    :attr List[AssignmentResourceEntry] members: List of member resources of the
          group.
    :attr List[AssignmentResourceEntry] rules: List of rules associated with the
          group.
    """

    def __init__(
        self,
        group: 'AssignmentResourceEntry',
        members: List['AssignmentResourceEntry'],
        rules: List['AssignmentResourceEntry'],
    ) -> None:
        """
        Initialize a AssignmentResourceAccessGroup object.

        :param AssignmentResourceEntry group: Assignment resource entry.
        :param List[AssignmentResourceEntry] members: List of member resources of
               the group.
        :param List[AssignmentResourceEntry] rules: List of rules associated with
               the group.
        """
        self.group = group
        self.members = members
        self.rules = rules

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AssignmentResourceAccessGroup':
        """Initialize a AssignmentResourceAccessGroup object from a json dictionary."""
        args = {}
        if 'group' in _dict:
            args['group'] = AssignmentResourceEntry.from_dict(_dict.get('group'))
        else:
            raise ValueError('Required property \'group\' not present in AssignmentResourceAccessGroup JSON')
        if 'members' in _dict:
            args['members'] = [AssignmentResourceEntry.from_dict(v) for v in _dict.get('members')]
        else:
            raise ValueError('Required property \'members\' not present in AssignmentResourceAccessGroup JSON')
        if 'rules' in _dict:
            args['rules'] = [AssignmentResourceEntry.from_dict(v) for v in _dict.get('rules')]
        else:
            raise ValueError('Required property \'rules\' not present in AssignmentResourceAccessGroup JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AssignmentResourceAccessGroup object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'group') and self.group is not None:
            if isinstance(self.group, dict):
                _dict['group'] = self.group
            else:
                _dict['group'] = self.group.to_dict()
        if hasattr(self, 'members') and self.members is not None:
            members_list = []
            for v in self.members:
                if isinstance(v, dict):
                    members_list.append(v)
                else:
                    members_list.append(v.to_dict())
            _dict['members'] = members_list
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
        """Return a `str` version of this AssignmentResourceAccessGroup object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AssignmentResourceAccessGroup') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AssignmentResourceAccessGroup') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AssignmentResourceEntry:
    """
    Assignment resource entry.

    :attr str id: Assignment Resource Entry Id.
    :attr str name: (optional) Optional name of the resource.
    :attr str version: (optional) Optional version of the resource.
    :attr str resource: Resource in assignment resource entry.
    :attr str error: Error in assignment resource entry.
    :attr str operation: (optional) Optional operation on the resource.
    :attr str status: Status of assignment resource entry.
    """

    def __init__(
        self,
        id: str,
        resource: str,
        error: str,
        status: str,
        *,
        name: str = None,
        version: str = None,
        operation: str = None,
    ) -> None:
        """
        Initialize a AssignmentResourceEntry object.

        :param str id: Assignment Resource Entry Id.
        :param str resource: Resource in assignment resource entry.
        :param str error: Error in assignment resource entry.
        :param str status: Status of assignment resource entry.
        :param str name: (optional) Optional name of the resource.
        :param str version: (optional) Optional version of the resource.
        :param str operation: (optional) Optional operation on the resource.
        """
        self.id = id
        self.name = name
        self.version = version
        self.resource = resource
        self.error = error
        self.operation = operation
        self.status = status

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AssignmentResourceEntry':
        """Initialize a AssignmentResourceEntry object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        else:
            raise ValueError('Required property \'id\' not present in AssignmentResourceEntry JSON')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'version' in _dict:
            args['version'] = _dict.get('version')
        if 'resource' in _dict:
            args['resource'] = _dict.get('resource')
        else:
            raise ValueError('Required property \'resource\' not present in AssignmentResourceEntry JSON')
        if 'error' in _dict:
            args['error'] = _dict.get('error')
        else:
            raise ValueError('Required property \'error\' not present in AssignmentResourceEntry JSON')
        if 'operation' in _dict:
            args['operation'] = _dict.get('operation')
        if 'status' in _dict:
            args['status'] = _dict.get('status')
        else:
            raise ValueError('Required property \'status\' not present in AssignmentResourceEntry JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AssignmentResourceEntry object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'version') and self.version is not None:
            _dict['version'] = self.version
        if hasattr(self, 'resource') and self.resource is not None:
            _dict['resource'] = self.resource
        if hasattr(self, 'error') and self.error is not None:
            _dict['error'] = self.error
        if hasattr(self, 'operation') and self.operation is not None:
            _dict['operation'] = self.operation
        if hasattr(self, 'status') and self.status is not None:
            _dict['status'] = self.status
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AssignmentResourceEntry object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AssignmentResourceEntry') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AssignmentResourceEntry') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Conditions:
    """
    Condition Input component.

    :attr str claim: (optional) The key in the key:value pair.
    :attr str operator: (optional) Compares the claim and the value.
    :attr str value: (optional) The value in the key:value pair.
    """

    def __init__(
        self,
        *,
        claim: str = None,
        operator: str = None,
        value: str = None,
    ) -> None:
        """
        Initialize a Conditions object.

        :param str claim: (optional) The key in the key:value pair.
        :param str operator: (optional) Compares the claim and the value.
        :param str value: (optional) The value in the key:value pair.
        """
        self.claim = claim
        self.operator = operator
        self.value = value

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Conditions':
        """Initialize a Conditions object from a json dictionary."""
        args = {}
        if 'claim' in _dict:
            args['claim'] = _dict.get('claim')
        if 'operator' in _dict:
            args['operator'] = _dict.get('operator')
        if 'value' in _dict:
            args['value'] = _dict.get('value')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Conditions object from a json dictionary."""
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
        """Return a `str` version of this Conditions object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Conditions') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Conditions') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class DeleteFromAllGroupsResponse:
    """
    The response from the delete member from access groups request.

    :attr str iam_id: (optional) The `iam_id` of the member to removed from groups.
    :attr List[DeleteFromAllGroupsResponseGroupsItem] groups: (optional) The groups
          the member was removed from.
    """

    def __init__(
        self,
        *,
        iam_id: str = None,
        groups: List['DeleteFromAllGroupsResponseGroupsItem'] = None,
    ) -> None:
        """
        Initialize a DeleteFromAllGroupsResponse object.

        :param str iam_id: (optional) The `iam_id` of the member to removed from
               groups.
        :param List[DeleteFromAllGroupsResponseGroupsItem] groups: (optional) The
               groups the member was removed from.
        """
        self.iam_id = iam_id
        self.groups = groups

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'DeleteFromAllGroupsResponse':
        """Initialize a DeleteFromAllGroupsResponse object from a json dictionary."""
        args = {}
        if 'iam_id' in _dict:
            args['iam_id'] = _dict.get('iam_id')
        if 'groups' in _dict:
            args['groups'] = [DeleteFromAllGroupsResponseGroupsItem.from_dict(v) for v in _dict.get('groups')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a DeleteFromAllGroupsResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'iam_id') and self.iam_id is not None:
            _dict['iam_id'] = self.iam_id
        if hasattr(self, 'groups') and self.groups is not None:
            groups_list = []
            for v in self.groups:
                if isinstance(v, dict):
                    groups_list.append(v)
                else:
                    groups_list.append(v.to_dict())
            _dict['groups'] = groups_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this DeleteFromAllGroupsResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'DeleteFromAllGroupsResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'DeleteFromAllGroupsResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class DeleteFromAllGroupsResponseGroupsItem:
    """
    DeleteFromAllGroupsResponseGroupsItem.

    :attr str access_group_id: (optional) The access group that the member is to be
          deleted from.
    :attr int status_code: (optional) The outcome of the delete operation on this
          `access_group_id`.
    :attr str trace: (optional) A transaction-id that can be used for debugging
          purposes.
    :attr List[Error] errors: (optional) A list of errors that occurred when trying
          to remove a member from groups.
    """

    def __init__(
        self,
        *,
        access_group_id: str = None,
        status_code: int = None,
        trace: str = None,
        errors: List['Error'] = None,
    ) -> None:
        """
        Initialize a DeleteFromAllGroupsResponseGroupsItem object.

        :param str access_group_id: (optional) The access group that the member is
               to be deleted from.
        :param int status_code: (optional) The outcome of the delete operation on
               this `access_group_id`.
        :param str trace: (optional) A transaction-id that can be used for
               debugging purposes.
        :param List[Error] errors: (optional) A list of errors that occurred when
               trying to remove a member from groups.
        """
        self.access_group_id = access_group_id
        self.status_code = status_code
        self.trace = trace
        self.errors = errors

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'DeleteFromAllGroupsResponseGroupsItem':
        """Initialize a DeleteFromAllGroupsResponseGroupsItem object from a json dictionary."""
        args = {}
        if 'access_group_id' in _dict:
            args['access_group_id'] = _dict.get('access_group_id')
        if 'status_code' in _dict:
            args['status_code'] = _dict.get('status_code')
        if 'trace' in _dict:
            args['trace'] = _dict.get('trace')
        if 'errors' in _dict:
            args['errors'] = [Error.from_dict(v) for v in _dict.get('errors')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a DeleteFromAllGroupsResponseGroupsItem object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'access_group_id') and self.access_group_id is not None:
            _dict['access_group_id'] = self.access_group_id
        if hasattr(self, 'status_code') and self.status_code is not None:
            _dict['status_code'] = self.status_code
        if hasattr(self, 'trace') and self.trace is not None:
            _dict['trace'] = self.trace
        if hasattr(self, 'errors') and self.errors is not None:
            errors_list = []
            for v in self.errors:
                if isinstance(v, dict):
                    errors_list.append(v)
                else:
                    errors_list.append(v.to_dict())
            _dict['errors'] = errors_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this DeleteFromAllGroupsResponseGroupsItem object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'DeleteFromAllGroupsResponseGroupsItem') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'DeleteFromAllGroupsResponseGroupsItem') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class DeleteGroupBulkMembersResponse:
    """
    The access group id and the members removed from it.

    :attr str access_group_id: (optional) The access group id.
    :attr List[DeleteGroupBulkMembersResponseMembersItem] members: (optional) The
          `iam_id`s removed from the access group.
    """

    def __init__(
        self,
        *,
        access_group_id: str = None,
        members: List['DeleteGroupBulkMembersResponseMembersItem'] = None,
    ) -> None:
        """
        Initialize a DeleteGroupBulkMembersResponse object.

        :param str access_group_id: (optional) The access group id.
        :param List[DeleteGroupBulkMembersResponseMembersItem] members: (optional)
               The `iam_id`s removed from the access group.
        """
        self.access_group_id = access_group_id
        self.members = members

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'DeleteGroupBulkMembersResponse':
        """Initialize a DeleteGroupBulkMembersResponse object from a json dictionary."""
        args = {}
        if 'access_group_id' in _dict:
            args['access_group_id'] = _dict.get('access_group_id')
        if 'members' in _dict:
            args['members'] = [DeleteGroupBulkMembersResponseMembersItem.from_dict(v) for v in _dict.get('members')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a DeleteGroupBulkMembersResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'access_group_id') and self.access_group_id is not None:
            _dict['access_group_id'] = self.access_group_id
        if hasattr(self, 'members') and self.members is not None:
            members_list = []
            for v in self.members:
                if isinstance(v, dict):
                    members_list.append(v)
                else:
                    members_list.append(v.to_dict())
            _dict['members'] = members_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this DeleteGroupBulkMembersResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'DeleteGroupBulkMembersResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'DeleteGroupBulkMembersResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class DeleteGroupBulkMembersResponseMembersItem:
    """
    DeleteGroupBulkMembersResponseMembersItem.

    :attr str iam_id: (optional) The `iam_id` to be deleted.
    :attr str trace: (optional) A transaction-id that can be used for debugging
          purposes.
    :attr int status_code: (optional) The outcome of the delete membership operation
          on this `access_group_id`.
    :attr List[Error] errors: (optional) A list of errors that occurred when trying
          to remove a member from groups.
    """

    def __init__(
        self,
        *,
        iam_id: str = None,
        trace: str = None,
        status_code: int = None,
        errors: List['Error'] = None,
    ) -> None:
        """
        Initialize a DeleteGroupBulkMembersResponseMembersItem object.

        :param str iam_id: (optional) The `iam_id` to be deleted.
        :param str trace: (optional) A transaction-id that can be used for
               debugging purposes.
        :param int status_code: (optional) The outcome of the delete membership
               operation on this `access_group_id`.
        :param List[Error] errors: (optional) A list of errors that occurred when
               trying to remove a member from groups.
        """
        self.iam_id = iam_id
        self.trace = trace
        self.status_code = status_code
        self.errors = errors

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'DeleteGroupBulkMembersResponseMembersItem':
        """Initialize a DeleteGroupBulkMembersResponseMembersItem object from a json dictionary."""
        args = {}
        if 'iam_id' in _dict:
            args['iam_id'] = _dict.get('iam_id')
        if 'trace' in _dict:
            args['trace'] = _dict.get('trace')
        if 'status_code' in _dict:
            args['status_code'] = _dict.get('status_code')
        if 'errors' in _dict:
            args['errors'] = [Error.from_dict(v) for v in _dict.get('errors')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a DeleteGroupBulkMembersResponseMembersItem object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'iam_id') and self.iam_id is not None:
            _dict['iam_id'] = self.iam_id
        if hasattr(self, 'trace') and self.trace is not None:
            _dict['trace'] = self.trace
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
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this DeleteGroupBulkMembersResponseMembersItem object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'DeleteGroupBulkMembersResponseMembersItem') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'DeleteGroupBulkMembersResponseMembersItem') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Error:
    """
    Error contains the code and message for an error returned to the user code is a string
    identifying the problem, examples "missing_field", "reserved_value" message is a
    string explaining the solution to the problem that was encountered.

    :attr str code: (optional) A human-readable error code represented by a snake
          case string.
    :attr str message: (optional) A specific error message that details the issue or
          an action to take.
    """

    def __init__(
        self,
        *,
        code: str = None,
        message: str = None,
    ) -> None:
        """
        Initialize a Error object.

        :param str code: (optional) A human-readable error code represented by a
               snake case string.
        :param str message: (optional) A specific error message that details the
               issue or an action to take.
        """
        self.code = code
        self.message = message

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Error':
        """Initialize a Error object from a json dictionary."""
        args = {}
        if 'code' in _dict:
            args['code'] = _dict.get('code')
        if 'message' in _dict:
            args['message'] = _dict.get('message')
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
        if hasattr(self, 'message') and self.message is not None:
            _dict['message'] = self.message
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


class Group:
    """
    An IAM access group.

    :attr str id: (optional) The group's access group ID.
    :attr str name: (optional) The group's name.
    :attr str description: (optional) The group's description - if defined.
    :attr str account_id: (optional) The account id where the group was created.
    :attr datetime created_at: (optional) The timestamp of when the group was
          created.
    :attr str created_by_id: (optional) The `iam_id` of the entity that created the
          group.
    :attr datetime last_modified_at: (optional) The timestamp of when the group was
          last edited.
    :attr str last_modified_by_id: (optional) The `iam_id` of the entity that last
          modified the group name or description.
    :attr str href: (optional) A url to the given group resource.
    :attr bool is_federated: (optional) This is set to true if rules exist for the
          group.
    """

    def __init__(
        self,
        *,
        id: str = None,
        name: str = None,
        description: str = None,
        account_id: str = None,
        created_at: datetime = None,
        created_by_id: str = None,
        last_modified_at: datetime = None,
        last_modified_by_id: str = None,
        href: str = None,
        is_federated: bool = None,
    ) -> None:
        """
        Initialize a Group object.

        :param str id: (optional) The group's access group ID.
        :param str name: (optional) The group's name.
        :param str description: (optional) The group's description - if defined.
        :param str account_id: (optional) The account id where the group was
               created.
        :param str href: (optional) A url to the given group resource.
        :param bool is_federated: (optional) This is set to true if rules exist for
               the group.
        """
        self.id = id
        self.name = name
        self.description = description
        self.account_id = account_id
        self.created_at = created_at
        self.created_by_id = created_by_id
        self.last_modified_at = last_modified_at
        self.last_modified_by_id = last_modified_by_id
        self.href = href
        self.is_federated = is_federated

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Group':
        """Initialize a Group object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        if 'created_at' in _dict:
            args['created_at'] = string_to_datetime(_dict.get('created_at'))
        if 'created_by_id' in _dict:
            args['created_by_id'] = _dict.get('created_by_id')
        if 'last_modified_at' in _dict:
            args['last_modified_at'] = string_to_datetime(_dict.get('last_modified_at'))
        if 'last_modified_by_id' in _dict:
            args['last_modified_by_id'] = _dict.get('last_modified_by_id')
        if 'href' in _dict:
            args['href'] = _dict.get('href')
        if 'is_federated' in _dict:
            args['is_federated'] = _dict.get('is_federated')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Group object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'created_at') and getattr(self, 'created_at') is not None:
            _dict['created_at'] = datetime_to_string(getattr(self, 'created_at'))
        if hasattr(self, 'created_by_id') and getattr(self, 'created_by_id') is not None:
            _dict['created_by_id'] = getattr(self, 'created_by_id')
        if hasattr(self, 'last_modified_at') and getattr(self, 'last_modified_at') is not None:
            _dict['last_modified_at'] = datetime_to_string(getattr(self, 'last_modified_at'))
        if hasattr(self, 'last_modified_by_id') and getattr(self, 'last_modified_by_id') is not None:
            _dict['last_modified_by_id'] = getattr(self, 'last_modified_by_id')
        if hasattr(self, 'href') and self.href is not None:
            _dict['href'] = self.href
        if hasattr(self, 'is_federated') and self.is_federated is not None:
            _dict['is_federated'] = self.is_federated
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Group object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Group') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Group') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class GroupActionControls:
    """
    Access group action controls component.

    :attr AccessActionControls access: (optional) Control whether or not access
          group administrators in child accounts can add access policies to the
          enterprise-managed access group in their account.
    """

    def __init__(
        self,
        *,
        access: 'AccessActionControls' = None,
    ) -> None:
        """
        Initialize a GroupActionControls object.

        :param AccessActionControls access: (optional) Control whether or not
               access group administrators in child accounts can add access policies to
               the enterprise-managed access group in their account.
        """
        self.access = access

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'GroupActionControls':
        """Initialize a GroupActionControls object from a json dictionary."""
        args = {}
        if 'access' in _dict:
            args['access'] = AccessActionControls.from_dict(_dict.get('access'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a GroupActionControls object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'access') and self.access is not None:
            if isinstance(self.access, dict):
                _dict['access'] = self.access
            else:
                _dict['access'] = self.access.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this GroupActionControls object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'GroupActionControls') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'GroupActionControls') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class GroupMembersList:
    """
    The members of a group.

    :attr int limit: Limit on how many items can be returned.
    :attr int offset: The offset of the first item returned in the result set.
    :attr int total_count: The total number of items that match the query.
    :attr HrefStruct first: (optional) A link object.
    :attr HrefStruct previous: (optional) A link object.
    :attr HrefStruct next: (optional) A link object.
    :attr HrefStruct last: (optional) A link object.
    :attr List[ListGroupMembersResponseMember] members: (optional) The members of an
          access group.
    """

    def __init__(
        self,
        limit: int,
        offset: int,
        total_count: int,
        *,
        first: 'HrefStruct' = None,
        previous: 'HrefStruct' = None,
        next: 'HrefStruct' = None,
        last: 'HrefStruct' = None,
        members: List['ListGroupMembersResponseMember'] = None,
    ) -> None:
        """
        Initialize a GroupMembersList object.

        :param int limit: Limit on how many items can be returned.
        :param int offset: The offset of the first item returned in the result set.
        :param int total_count: The total number of items that match the query.
        :param HrefStruct first: (optional) A link object.
        :param HrefStruct previous: (optional) A link object.
        :param HrefStruct next: (optional) A link object.
        :param HrefStruct last: (optional) A link object.
        :param List[ListGroupMembersResponseMember] members: (optional) The members
               of an access group.
        """
        self.limit = limit
        self.offset = offset
        self.total_count = total_count
        self.first = first
        self.previous = previous
        self.next = next
        self.last = last
        self.members = members

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'GroupMembersList':
        """Initialize a GroupMembersList object from a json dictionary."""
        args = {}
        if 'limit' in _dict:
            args['limit'] = _dict.get('limit')
        else:
            raise ValueError('Required property \'limit\' not present in GroupMembersList JSON')
        if 'offset' in _dict:
            args['offset'] = _dict.get('offset')
        else:
            raise ValueError('Required property \'offset\' not present in GroupMembersList JSON')
        if 'total_count' in _dict:
            args['total_count'] = _dict.get('total_count')
        else:
            raise ValueError('Required property \'total_count\' not present in GroupMembersList JSON')
        if 'first' in _dict:
            args['first'] = HrefStruct.from_dict(_dict.get('first'))
        if 'previous' in _dict:
            args['previous'] = HrefStruct.from_dict(_dict.get('previous'))
        if 'next' in _dict:
            args['next'] = HrefStruct.from_dict(_dict.get('next'))
        if 'last' in _dict:
            args['last'] = HrefStruct.from_dict(_dict.get('last'))
        if 'members' in _dict:
            args['members'] = [ListGroupMembersResponseMember.from_dict(v) for v in _dict.get('members')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a GroupMembersList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        if hasattr(self, 'total_count') and self.total_count is not None:
            _dict['total_count'] = self.total_count
        if hasattr(self, 'first') and self.first is not None:
            if isinstance(self.first, dict):
                _dict['first'] = self.first
            else:
                _dict['first'] = self.first.to_dict()
        if hasattr(self, 'previous') and self.previous is not None:
            if isinstance(self.previous, dict):
                _dict['previous'] = self.previous
            else:
                _dict['previous'] = self.previous.to_dict()
        if hasattr(self, 'next') and self.next is not None:
            if isinstance(self.next, dict):
                _dict['next'] = self.next
            else:
                _dict['next'] = self.next.to_dict()
        if hasattr(self, 'last') and self.last is not None:
            if isinstance(self.last, dict):
                _dict['last'] = self.last
            else:
                _dict['last'] = self.last.to_dict()
        if hasattr(self, 'members') and self.members is not None:
            members_list = []
            for v in self.members:
                if isinstance(v, dict):
                    members_list.append(v)
                else:
                    members_list.append(v.to_dict())
            _dict['members'] = members_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this GroupMembersList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'GroupMembersList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'GroupMembersList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class GroupTemplate:
    """
    Response output for template.

    :attr str id: The ID of the access group template.
    :attr str name: The name of the access group template.
    :attr str description: The description of the access group template.
    :attr str version: The version of the access group template.
    :attr bool committed: A boolean indicating whether the access group template is
          committed. You must commit a template before you can assign it to child
          accounts.
    :attr AccessGroupResponse group: Access Group Component.
    :attr List[PolicyTemplates] policy_template_references: References to policy
          templates assigned to the access group template.
    :attr str href: The URL of the access group template resource.
    :attr datetime created_at: The date and time when the access group template was
          created.
    :attr str created_by_id: The ID of the user who created the access group
          template.
    :attr datetime last_modified_at: The date and time when the access group
          template was last modified.
    :attr str last_modified_by_id: The ID of the user who last modified the access
          group template.
    """

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        version: str,
        committed: bool,
        group: 'AccessGroupResponse',
        policy_template_references: List['PolicyTemplates'],
        href: str,
        created_at: datetime,
        created_by_id: str,
        last_modified_at: datetime,
        last_modified_by_id: str,
    ) -> None:
        """
        Initialize a GroupTemplate object.

        :param str id: The ID of the access group template.
        :param str name: The name of the access group template.
        :param str description: The description of the access group template.
        :param str version: The version of the access group template.
        :param bool committed: A boolean indicating whether the access group
               template is committed. You must commit a template before you can assign it
               to child accounts.
        :param AccessGroupResponse group: Access Group Component.
        :param List[PolicyTemplates] policy_template_references: References to
               policy templates assigned to the access group template.
        :param str href: The URL of the access group template resource.
        :param datetime created_at: The date and time when the access group
               template was created.
        :param str created_by_id: The ID of the user who created the access group
               template.
        :param datetime last_modified_at: The date and time when the access group
               template was last modified.
        :param str last_modified_by_id: The ID of the user who last modified the
               access group template.
        """
        self.id = id
        self.name = name
        self.description = description
        self.version = version
        self.committed = committed
        self.group = group
        self.policy_template_references = policy_template_references
        self.href = href
        self.created_at = created_at
        self.created_by_id = created_by_id
        self.last_modified_at = last_modified_at
        self.last_modified_by_id = last_modified_by_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'GroupTemplate':
        """Initialize a GroupTemplate object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        else:
            raise ValueError('Required property \'id\' not present in GroupTemplate JSON')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        else:
            raise ValueError('Required property \'name\' not present in GroupTemplate JSON')
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        else:
            raise ValueError('Required property \'description\' not present in GroupTemplate JSON')
        if 'version' in _dict:
            args['version'] = _dict.get('version')
        else:
            raise ValueError('Required property \'version\' not present in GroupTemplate JSON')
        if 'committed' in _dict:
            args['committed'] = _dict.get('committed')
        else:
            raise ValueError('Required property \'committed\' not present in GroupTemplate JSON')
        if 'group' in _dict:
            args['group'] = AccessGroupResponse.from_dict(_dict.get('group'))
        else:
            raise ValueError('Required property \'group\' not present in GroupTemplate JSON')
        if 'policy_template_references' in _dict:
            args['policy_template_references'] = [
                PolicyTemplates.from_dict(v) for v in _dict.get('policy_template_references')
            ]
        else:
            raise ValueError('Required property \'policy_template_references\' not present in GroupTemplate JSON')
        if 'href' in _dict:
            args['href'] = _dict.get('href')
        else:
            raise ValueError('Required property \'href\' not present in GroupTemplate JSON')
        if 'created_at' in _dict:
            args['created_at'] = string_to_datetime(_dict.get('created_at'))
        else:
            raise ValueError('Required property \'created_at\' not present in GroupTemplate JSON')
        if 'created_by_id' in _dict:
            args['created_by_id'] = _dict.get('created_by_id')
        else:
            raise ValueError('Required property \'created_by_id\' not present in GroupTemplate JSON')
        if 'last_modified_at' in _dict:
            args['last_modified_at'] = string_to_datetime(_dict.get('last_modified_at'))
        else:
            raise ValueError('Required property \'last_modified_at\' not present in GroupTemplate JSON')
        if 'last_modified_by_id' in _dict:
            args['last_modified_by_id'] = _dict.get('last_modified_by_id')
        else:
            raise ValueError('Required property \'last_modified_by_id\' not present in GroupTemplate JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a GroupTemplate object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'version') and self.version is not None:
            _dict['version'] = self.version
        if hasattr(self, 'committed') and self.committed is not None:
            _dict['committed'] = self.committed
        if hasattr(self, 'group') and self.group is not None:
            if isinstance(self.group, dict):
                _dict['group'] = self.group
            else:
                _dict['group'] = self.group.to_dict()
        if hasattr(self, 'policy_template_references') and self.policy_template_references is not None:
            policy_template_references_list = []
            for v in self.policy_template_references:
                if isinstance(v, dict):
                    policy_template_references_list.append(v)
                else:
                    policy_template_references_list.append(v.to_dict())
            _dict['policy_template_references'] = policy_template_references_list
        if hasattr(self, 'href') and self.href is not None:
            _dict['href'] = self.href
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'created_by_id') and self.created_by_id is not None:
            _dict['created_by_id'] = self.created_by_id
        if hasattr(self, 'last_modified_at') and self.last_modified_at is not None:
            _dict['last_modified_at'] = datetime_to_string(self.last_modified_at)
        if hasattr(self, 'last_modified_by_id') and self.last_modified_by_id is not None:
            _dict['last_modified_by_id'] = self.last_modified_by_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this GroupTemplate object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'GroupTemplate') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'GroupTemplate') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class GroupsList:
    """
    The list of access groups returned as part of a response.

    :attr int limit: Limit on how many items can be returned.
    :attr int offset: The offset of the first item returned in the result set.
    :attr int total_count: The total number of items that match the query.
    :attr HrefStruct first: (optional) A link object.
    :attr HrefStruct previous: (optional) A link object.
    :attr HrefStruct next: (optional) A link object.
    :attr HrefStruct last: (optional) A link object.
    :attr List[Group] groups: (optional) An array of access groups.
    """

    def __init__(
        self,
        limit: int,
        offset: int,
        total_count: int,
        *,
        first: 'HrefStruct' = None,
        previous: 'HrefStruct' = None,
        next: 'HrefStruct' = None,
        last: 'HrefStruct' = None,
        groups: List['Group'] = None,
    ) -> None:
        """
        Initialize a GroupsList object.

        :param int limit: Limit on how many items can be returned.
        :param int offset: The offset of the first item returned in the result set.
        :param int total_count: The total number of items that match the query.
        :param HrefStruct first: (optional) A link object.
        :param HrefStruct previous: (optional) A link object.
        :param HrefStruct next: (optional) A link object.
        :param HrefStruct last: (optional) A link object.
        :param List[Group] groups: (optional) An array of access groups.
        """
        self.limit = limit
        self.offset = offset
        self.total_count = total_count
        self.first = first
        self.previous = previous
        self.next = next
        self.last = last
        self.groups = groups

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'GroupsList':
        """Initialize a GroupsList object from a json dictionary."""
        args = {}
        if 'limit' in _dict:
            args['limit'] = _dict.get('limit')
        else:
            raise ValueError('Required property \'limit\' not present in GroupsList JSON')
        if 'offset' in _dict:
            args['offset'] = _dict.get('offset')
        else:
            raise ValueError('Required property \'offset\' not present in GroupsList JSON')
        if 'total_count' in _dict:
            args['total_count'] = _dict.get('total_count')
        else:
            raise ValueError('Required property \'total_count\' not present in GroupsList JSON')
        if 'first' in _dict:
            args['first'] = HrefStruct.from_dict(_dict.get('first'))
        if 'previous' in _dict:
            args['previous'] = HrefStruct.from_dict(_dict.get('previous'))
        if 'next' in _dict:
            args['next'] = HrefStruct.from_dict(_dict.get('next'))
        if 'last' in _dict:
            args['last'] = HrefStruct.from_dict(_dict.get('last'))
        if 'groups' in _dict:
            args['groups'] = [Group.from_dict(v) for v in _dict.get('groups')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a GroupsList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        if hasattr(self, 'total_count') and self.total_count is not None:
            _dict['total_count'] = self.total_count
        if hasattr(self, 'first') and self.first is not None:
            if isinstance(self.first, dict):
                _dict['first'] = self.first
            else:
                _dict['first'] = self.first.to_dict()
        if hasattr(self, 'previous') and self.previous is not None:
            if isinstance(self.previous, dict):
                _dict['previous'] = self.previous
            else:
                _dict['previous'] = self.previous.to_dict()
        if hasattr(self, 'next') and self.next is not None:
            if isinstance(self.next, dict):
                _dict['next'] = self.next
            else:
                _dict['next'] = self.next.to_dict()
        if hasattr(self, 'last') and self.last is not None:
            if isinstance(self.last, dict):
                _dict['last'] = self.last
            else:
                _dict['last'] = self.last.to_dict()
        if hasattr(self, 'groups') and self.groups is not None:
            groups_list = []
            for v in self.groups:
                if isinstance(v, dict):
                    groups_list.append(v)
                else:
                    groups_list.append(v.to_dict())
            _dict['groups'] = groups_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this GroupsList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'GroupsList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'GroupsList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class HrefStruct:
    """
    A link object.

    :attr str href: (optional) A string containing the link’s URL.
    """

    def __init__(
        self,
        *,
        href: str = None,
    ) -> None:
        """
        Initialize a HrefStruct object.

        :param str href: (optional) A string containing the link’s URL.
        """
        self.href = href

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'HrefStruct':
        """Initialize a HrefStruct object from a json dictionary."""
        args = {}
        if 'href' in _dict:
            args['href'] = _dict.get('href')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a HrefStruct object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'href') and self.href is not None:
            _dict['href'] = self.href
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this HrefStruct object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'HrefStruct') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'HrefStruct') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ListGroupMembersResponseMember:
    """
    A single member of an access group in a list.

    :attr str iam_id: (optional) The IBMid or Service Id of the member.
    :attr str type: (optional) The member type - either `user`, `service` or
          `profile`.
    :attr str membership_type: (optional) The membership type - either `static` or
          `dynamic`.
    :attr str name: (optional) The user's or service id's name.
    :attr str email: (optional) If the member type is user, this is the user's
          email.
    :attr str description: (optional) If the member type is service, this is the
          service id's description.
    :attr str href: (optional) A url to the given member resource.
    :attr datetime created_at: (optional) The timestamp the membership was created
          at.
    :attr str created_by_id: (optional) The `iam_id` of the entity that created the
          membership.
    """

    def __init__(
        self,
        *,
        iam_id: str = None,
        type: str = None,
        membership_type: str = None,
        name: str = None,
        email: str = None,
        description: str = None,
        href: str = None,
        created_at: datetime = None,
        created_by_id: str = None,
    ) -> None:
        """
        Initialize a ListGroupMembersResponseMember object.

        :param str iam_id: (optional) The IBMid or Service Id of the member.
        :param str type: (optional) The member type - either `user`, `service` or
               `profile`.
        :param str membership_type: (optional) The membership type - either
               `static` or `dynamic`.
        :param str name: (optional) The user's or service id's name.
        :param str email: (optional) If the member type is user, this is the user's
               email.
        :param str description: (optional) If the member type is service, this is
               the service id's description.
        :param str href: (optional) A url to the given member resource.
        :param datetime created_at: (optional) The timestamp the membership was
               created at.
        :param str created_by_id: (optional) The `iam_id` of the entity that
               created the membership.
        """
        self.iam_id = iam_id
        self.type = type
        self.membership_type = membership_type
        self.name = name
        self.email = email
        self.description = description
        self.href = href
        self.created_at = created_at
        self.created_by_id = created_by_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ListGroupMembersResponseMember':
        """Initialize a ListGroupMembersResponseMember object from a json dictionary."""
        args = {}
        if 'iam_id' in _dict:
            args['iam_id'] = _dict.get('iam_id')
        if 'type' in _dict:
            args['type'] = _dict.get('type')
        if 'membership_type' in _dict:
            args['membership_type'] = _dict.get('membership_type')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'email' in _dict:
            args['email'] = _dict.get('email')
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        if 'href' in _dict:
            args['href'] = _dict.get('href')
        if 'created_at' in _dict:
            args['created_at'] = string_to_datetime(_dict.get('created_at'))
        if 'created_by_id' in _dict:
            args['created_by_id'] = _dict.get('created_by_id')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ListGroupMembersResponseMember object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'iam_id') and self.iam_id is not None:
            _dict['iam_id'] = self.iam_id
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'membership_type') and self.membership_type is not None:
            _dict['membership_type'] = self.membership_type
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'email') and self.email is not None:
            _dict['email'] = self.email
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'href') and self.href is not None:
            _dict['href'] = self.href
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'created_by_id') and self.created_by_id is not None:
            _dict['created_by_id'] = self.created_by_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ListGroupMembersResponseMember object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ListGroupMembersResponseMember') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ListGroupMembersResponseMember') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ListTemplateAssignmentResponse:
    """
    Response object containing a list of template assignments.

    :attr int limit: Maximum number of items returned in the response.
    :attr int offset: Index of the first item returned in the response.
    :attr int total_count: Total number of items matching the query.
    :attr HrefStruct first: A link object.
    :attr HrefStruct last: A link object.
    :attr List[TemplateAssignmentResponse] assignments: List of template
          assignments.
    """

    def __init__(
        self,
        limit: int,
        offset: int,
        total_count: int,
        first: 'HrefStruct',
        last: 'HrefStruct',
        assignments: List['TemplateAssignmentResponse'],
    ) -> None:
        """
        Initialize a ListTemplateAssignmentResponse object.

        :param int limit: Maximum number of items returned in the response.
        :param int offset: Index of the first item returned in the response.
        :param int total_count: Total number of items matching the query.
        :param HrefStruct first: A link object.
        :param HrefStruct last: A link object.
        :param List[TemplateAssignmentResponse] assignments: List of template
               assignments.
        """
        self.limit = limit
        self.offset = offset
        self.total_count = total_count
        self.first = first
        self.last = last
        self.assignments = assignments

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ListTemplateAssignmentResponse':
        """Initialize a ListTemplateAssignmentResponse object from a json dictionary."""
        args = {}
        if 'limit' in _dict:
            args['limit'] = _dict.get('limit')
        else:
            raise ValueError('Required property \'limit\' not present in ListTemplateAssignmentResponse JSON')
        if 'offset' in _dict:
            args['offset'] = _dict.get('offset')
        else:
            raise ValueError('Required property \'offset\' not present in ListTemplateAssignmentResponse JSON')
        if 'total_count' in _dict:
            args['total_count'] = _dict.get('total_count')
        else:
            raise ValueError('Required property \'total_count\' not present in ListTemplateAssignmentResponse JSON')
        if 'first' in _dict:
            args['first'] = HrefStruct.from_dict(_dict.get('first'))
        else:
            raise ValueError('Required property \'first\' not present in ListTemplateAssignmentResponse JSON')
        if 'last' in _dict:
            args['last'] = HrefStruct.from_dict(_dict.get('last'))
        else:
            raise ValueError('Required property \'last\' not present in ListTemplateAssignmentResponse JSON')
        if 'assignments' in _dict:
            args['assignments'] = [TemplateAssignmentResponse.from_dict(v) for v in _dict.get('assignments')]
        else:
            raise ValueError('Required property \'assignments\' not present in ListTemplateAssignmentResponse JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ListTemplateAssignmentResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        if hasattr(self, 'total_count') and self.total_count is not None:
            _dict['total_count'] = self.total_count
        if hasattr(self, 'first') and self.first is not None:
            if isinstance(self.first, dict):
                _dict['first'] = self.first
            else:
                _dict['first'] = self.first.to_dict()
        if hasattr(self, 'last') and self.last is not None:
            if isinstance(self.last, dict):
                _dict['last'] = self.last
            else:
                _dict['last'] = self.last.to_dict()
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
        """Return a `str` version of this ListTemplateAssignmentResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ListTemplateAssignmentResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ListTemplateAssignmentResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ListTemplateVersionResponse:
    """
    Response object for a single access group template version.

    :attr str name: The name of the template.
    :attr str description: The description of the template.
    :attr str account_id: The ID of the account associated with the template.
    :attr str version: The version number of the template.
    :attr bool committed: A boolean indicating whether the template is committed or
          not.
    :attr AccessGroupResponse group: Access Group Component.
    :attr List[PolicyTemplates] policy_template_references: A list of policy
          templates associated with the template.
    :attr str href: The URL to the template resource.
    :attr str created_at: The date and time the template was created.
    :attr str created_by_id: The ID of the user who created the template.
    :attr str last_modified_at: The date and time the template was last modified.
    :attr str last_modified_by_id: The ID of the user who last modified the
          template.
    """

    def __init__(
        self,
        name: str,
        description: str,
        account_id: str,
        version: str,
        committed: bool,
        group: 'AccessGroupResponse',
        policy_template_references: List['PolicyTemplates'],
        href: str,
        created_at: str,
        created_by_id: str,
        last_modified_at: str,
        last_modified_by_id: str,
    ) -> None:
        """
        Initialize a ListTemplateVersionResponse object.

        :param str name: The name of the template.
        :param str description: The description of the template.
        :param str account_id: The ID of the account associated with the template.
        :param str version: The version number of the template.
        :param bool committed: A boolean indicating whether the template is
               committed or not.
        :param AccessGroupResponse group: Access Group Component.
        :param List[PolicyTemplates] policy_template_references: A list of policy
               templates associated with the template.
        :param str href: The URL to the template resource.
        :param str created_at: The date and time the template was created.
        :param str created_by_id: The ID of the user who created the template.
        :param str last_modified_at: The date and time the template was last
               modified.
        :param str last_modified_by_id: The ID of the user who last modified the
               template.
        """
        self.name = name
        self.description = description
        self.account_id = account_id
        self.version = version
        self.committed = committed
        self.group = group
        self.policy_template_references = policy_template_references
        self.href = href
        self.created_at = created_at
        self.created_by_id = created_by_id
        self.last_modified_at = last_modified_at
        self.last_modified_by_id = last_modified_by_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ListTemplateVersionResponse':
        """Initialize a ListTemplateVersionResponse object from a json dictionary."""
        args = {}
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        else:
            raise ValueError('Required property \'name\' not present in ListTemplateVersionResponse JSON')
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        else:
            raise ValueError('Required property \'description\' not present in ListTemplateVersionResponse JSON')
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        else:
            raise ValueError('Required property \'account_id\' not present in ListTemplateVersionResponse JSON')
        if 'version' in _dict:
            args['version'] = _dict.get('version')
        else:
            raise ValueError('Required property \'version\' not present in ListTemplateVersionResponse JSON')
        if 'committed' in _dict:
            args['committed'] = _dict.get('committed')
        else:
            raise ValueError('Required property \'committed\' not present in ListTemplateVersionResponse JSON')
        if 'group' in _dict:
            args['group'] = AccessGroupResponse.from_dict(_dict.get('group'))
        else:
            raise ValueError('Required property \'group\' not present in ListTemplateVersionResponse JSON')
        if 'policy_template_references' in _dict:
            args['policy_template_references'] = [
                PolicyTemplates.from_dict(v) for v in _dict.get('policy_template_references')
            ]
        else:
            raise ValueError(
                'Required property \'policy_template_references\' not present in ListTemplateVersionResponse JSON'
            )
        if 'href' in _dict:
            args['href'] = _dict.get('href')
        else:
            raise ValueError('Required property \'href\' not present in ListTemplateVersionResponse JSON')
        if 'created_at' in _dict:
            args['created_at'] = _dict.get('created_at')
        else:
            raise ValueError('Required property \'created_at\' not present in ListTemplateVersionResponse JSON')
        if 'created_by_id' in _dict:
            args['created_by_id'] = _dict.get('created_by_id')
        else:
            raise ValueError('Required property \'created_by_id\' not present in ListTemplateVersionResponse JSON')
        if 'last_modified_at' in _dict:
            args['last_modified_at'] = _dict.get('last_modified_at')
        else:
            raise ValueError('Required property \'last_modified_at\' not present in ListTemplateVersionResponse JSON')
        if 'last_modified_by_id' in _dict:
            args['last_modified_by_id'] = _dict.get('last_modified_by_id')
        else:
            raise ValueError(
                'Required property \'last_modified_by_id\' not present in ListTemplateVersionResponse JSON'
            )
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ListTemplateVersionResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'version') and self.version is not None:
            _dict['version'] = self.version
        if hasattr(self, 'committed') and self.committed is not None:
            _dict['committed'] = self.committed
        if hasattr(self, 'group') and self.group is not None:
            if isinstance(self.group, dict):
                _dict['group'] = self.group
            else:
                _dict['group'] = self.group.to_dict()
        if hasattr(self, 'policy_template_references') and self.policy_template_references is not None:
            policy_template_references_list = []
            for v in self.policy_template_references:
                if isinstance(v, dict):
                    policy_template_references_list.append(v)
                else:
                    policy_template_references_list.append(v.to_dict())
            _dict['policy_template_references'] = policy_template_references_list
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
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ListTemplateVersionResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ListTemplateVersionResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ListTemplateVersionResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ListTemplateVersionsResponse:
    """
    Response object for listing template versions.

    :attr int limit: The maximum number of IAM resources to return.
    :attr int offset: The offset of the first IAM resource in the list.
    :attr int total_count: The total number of IAM resources in the list.
    :attr HrefStruct first: A link object.
    :attr HrefStruct previous: (optional) A link object.
    :attr HrefStruct next: (optional) A link object.
    :attr HrefStruct last: A link object.
    :attr List[ListTemplateVersionResponse] group_template_versions: A list of
          access group template versions.
    """

    def __init__(
        self,
        limit: int,
        offset: int,
        total_count: int,
        first: 'HrefStruct',
        last: 'HrefStruct',
        group_template_versions: List['ListTemplateVersionResponse'],
        *,
        previous: 'HrefStruct' = None,
        next: 'HrefStruct' = None,
    ) -> None:
        """
        Initialize a ListTemplateVersionsResponse object.

        :param int limit: The maximum number of IAM resources to return.
        :param int offset: The offset of the first IAM resource in the list.
        :param int total_count: The total number of IAM resources in the list.
        :param HrefStruct first: A link object.
        :param HrefStruct last: A link object.
        :param List[ListTemplateVersionResponse] group_template_versions: A list of
               access group template versions.
        :param HrefStruct previous: (optional) A link object.
        :param HrefStruct next: (optional) A link object.
        """
        self.limit = limit
        self.offset = offset
        self.total_count = total_count
        self.first = first
        self.previous = previous
        self.next = next
        self.last = last
        self.group_template_versions = group_template_versions

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ListTemplateVersionsResponse':
        """Initialize a ListTemplateVersionsResponse object from a json dictionary."""
        args = {}
        if 'limit' in _dict:
            args['limit'] = _dict.get('limit')
        else:
            raise ValueError('Required property \'limit\' not present in ListTemplateVersionsResponse JSON')
        if 'offset' in _dict:
            args['offset'] = _dict.get('offset')
        else:
            raise ValueError('Required property \'offset\' not present in ListTemplateVersionsResponse JSON')
        if 'total_count' in _dict:
            args['total_count'] = _dict.get('total_count')
        else:
            raise ValueError('Required property \'total_count\' not present in ListTemplateVersionsResponse JSON')
        if 'first' in _dict:
            args['first'] = HrefStruct.from_dict(_dict.get('first'))
        else:
            raise ValueError('Required property \'first\' not present in ListTemplateVersionsResponse JSON')
        if 'previous' in _dict:
            args['previous'] = HrefStruct.from_dict(_dict.get('previous'))
        if 'next' in _dict:
            args['next'] = HrefStruct.from_dict(_dict.get('next'))
        if 'last' in _dict:
            args['last'] = HrefStruct.from_dict(_dict.get('last'))
        else:
            raise ValueError('Required property \'last\' not present in ListTemplateVersionsResponse JSON')
        if 'group_template_versions' in _dict:
            args['group_template_versions'] = [
                ListTemplateVersionResponse.from_dict(v) for v in _dict.get('group_template_versions')
            ]
        else:
            raise ValueError(
                'Required property \'group_template_versions\' not present in ListTemplateVersionsResponse JSON'
            )
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ListTemplateVersionsResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        if hasattr(self, 'total_count') and self.total_count is not None:
            _dict['total_count'] = self.total_count
        if hasattr(self, 'first') and self.first is not None:
            if isinstance(self.first, dict):
                _dict['first'] = self.first
            else:
                _dict['first'] = self.first.to_dict()
        if hasattr(self, 'previous') and self.previous is not None:
            if isinstance(self.previous, dict):
                _dict['previous'] = self.previous
            else:
                _dict['previous'] = self.previous.to_dict()
        if hasattr(self, 'next') and self.next is not None:
            if isinstance(self.next, dict):
                _dict['next'] = self.next
            else:
                _dict['next'] = self.next.to_dict()
        if hasattr(self, 'last') and self.last is not None:
            if isinstance(self.last, dict):
                _dict['last'] = self.last
            else:
                _dict['last'] = self.last.to_dict()
        if hasattr(self, 'group_template_versions') and self.group_template_versions is not None:
            group_template_versions_list = []
            for v in self.group_template_versions:
                if isinstance(v, dict):
                    group_template_versions_list.append(v)
                else:
                    group_template_versions_list.append(v.to_dict())
            _dict['group_template_versions'] = group_template_versions_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ListTemplateVersionsResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ListTemplateVersionsResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ListTemplateVersionsResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ListTemplatesResponse:
    """
    Response object for listing templates.

    :attr int limit: The maximum number of IAM resources to return.
    :attr int offset: The offset of the first IAM resource in the list.
    :attr int total_count: The total number of IAM resources in the list.
    :attr HrefStruct first: A link object.
    :attr HrefStruct previous: (optional) A link object.
    :attr HrefStruct next: (optional) A link object.
    :attr HrefStruct last: A link object.
    :attr List[GroupTemplate] group_templates: A list of access group templates.
    """

    def __init__(
        self,
        limit: int,
        offset: int,
        total_count: int,
        first: 'HrefStruct',
        last: 'HrefStruct',
        group_templates: List['GroupTemplate'],
        *,
        previous: 'HrefStruct' = None,
        next: 'HrefStruct' = None,
    ) -> None:
        """
        Initialize a ListTemplatesResponse object.

        :param int limit: The maximum number of IAM resources to return.
        :param int offset: The offset of the first IAM resource in the list.
        :param int total_count: The total number of IAM resources in the list.
        :param HrefStruct first: A link object.
        :param HrefStruct last: A link object.
        :param List[GroupTemplate] group_templates: A list of access group
               templates.
        :param HrefStruct previous: (optional) A link object.
        :param HrefStruct next: (optional) A link object.
        """
        self.limit = limit
        self.offset = offset
        self.total_count = total_count
        self.first = first
        self.previous = previous
        self.next = next
        self.last = last
        self.group_templates = group_templates

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ListTemplatesResponse':
        """Initialize a ListTemplatesResponse object from a json dictionary."""
        args = {}
        if 'limit' in _dict:
            args['limit'] = _dict.get('limit')
        else:
            raise ValueError('Required property \'limit\' not present in ListTemplatesResponse JSON')
        if 'offset' in _dict:
            args['offset'] = _dict.get('offset')
        else:
            raise ValueError('Required property \'offset\' not present in ListTemplatesResponse JSON')
        if 'total_count' in _dict:
            args['total_count'] = _dict.get('total_count')
        else:
            raise ValueError('Required property \'total_count\' not present in ListTemplatesResponse JSON')
        if 'first' in _dict:
            args['first'] = HrefStruct.from_dict(_dict.get('first'))
        else:
            raise ValueError('Required property \'first\' not present in ListTemplatesResponse JSON')
        if 'previous' in _dict:
            args['previous'] = HrefStruct.from_dict(_dict.get('previous'))
        if 'next' in _dict:
            args['next'] = HrefStruct.from_dict(_dict.get('next'))
        if 'last' in _dict:
            args['last'] = HrefStruct.from_dict(_dict.get('last'))
        else:
            raise ValueError('Required property \'last\' not present in ListTemplatesResponse JSON')
        if 'group_templates' in _dict:
            args['group_templates'] = [GroupTemplate.from_dict(v) for v in _dict.get('group_templates')]
        else:
            raise ValueError('Required property \'group_templates\' not present in ListTemplatesResponse JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ListTemplatesResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        if hasattr(self, 'total_count') and self.total_count is not None:
            _dict['total_count'] = self.total_count
        if hasattr(self, 'first') and self.first is not None:
            if isinstance(self.first, dict):
                _dict['first'] = self.first
            else:
                _dict['first'] = self.first.to_dict()
        if hasattr(self, 'previous') and self.previous is not None:
            if isinstance(self.previous, dict):
                _dict['previous'] = self.previous
            else:
                _dict['previous'] = self.previous.to_dict()
        if hasattr(self, 'next') and self.next is not None:
            if isinstance(self.next, dict):
                _dict['next'] = self.next
            else:
                _dict['next'] = self.next.to_dict()
        if hasattr(self, 'last') and self.last is not None:
            if isinstance(self.last, dict):
                _dict['last'] = self.last
            else:
                _dict['last'] = self.last.to_dict()
        if hasattr(self, 'group_templates') and self.group_templates is not None:
            group_templates_list = []
            for v in self.group_templates:
                if isinstance(v, dict):
                    group_templates_list.append(v)
                else:
                    group_templates_list.append(v.to_dict())
            _dict['group_templates'] = group_templates_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ListTemplatesResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ListTemplatesResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ListTemplatesResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Members:
    """
    Array of enterprise users to add to the template. All enterprise users that you add to
    the template must be invited to the child accounts where the template is assigned.

    :attr List[str] users: (optional) Array of enterprise users to add to the
          template. All enterprise users that you add to the template must be invited to
          the child accounts where the template is assigned.
    :attr List[str] services: (optional) Array of service IDs to add to the
          template.
    :attr MembersActionControls action_controls: (optional) Control whether or not
          access group administrators in child accounts can add and remove members from
          the enterprise-managed access group in their account.
    """

    def __init__(
        self,
        *,
        users: List[str] = None,
        services: List[str] = None,
        action_controls: 'MembersActionControls' = None,
    ) -> None:
        """
        Initialize a Members object.

        :param List[str] users: (optional) Array of enterprise users to add to the
               template. All enterprise users that you add to the template must be invited
               to the child accounts where the template is assigned.
        :param List[str] services: (optional) Array of service IDs to add to the
               template.
        :param MembersActionControls action_controls: (optional) Control whether or
               not access group administrators in child accounts can add and remove
               members from the enterprise-managed access group in their account.
        """
        self.users = users
        self.services = services
        self.action_controls = action_controls

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Members':
        """Initialize a Members object from a json dictionary."""
        args = {}
        if 'users' in _dict:
            args['users'] = _dict.get('users')
        if 'services' in _dict:
            args['services'] = _dict.get('services')
        if 'action_controls' in _dict:
            args['action_controls'] = MembersActionControls.from_dict(_dict.get('action_controls'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Members object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'users') and self.users is not None:
            _dict['users'] = self.users
        if hasattr(self, 'services') and self.services is not None:
            _dict['services'] = self.services
        if hasattr(self, 'action_controls') and self.action_controls is not None:
            if isinstance(self.action_controls, dict):
                _dict['action_controls'] = self.action_controls
            else:
                _dict['action_controls'] = self.action_controls.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Members object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Members') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Members') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class MembersActionControls:
    """
    Control whether or not access group administrators in child accounts can add and
    remove members from the enterprise-managed access group in their account.

    :attr bool add: (optional) Action control for adding child account members to an
          enterprise-managed access group. If an access group administrator in a child
          account adds a member, they can always remove them. Note that if conflicts arise
          between an update to this control in a new version and members added by an
          administrator in the child account, you must resolve those conflicts in the
          child account. This prevents breaking access in the child account. For more
          information, see [Working with versions]
          (https://test.cloud.ibm.com/docs/secure-enterprise?topic=secure-enterprise-working-with-versions#new-version-scenarios).
    :attr bool remove: (optional) Action control for removing enterprise-managed
          members from an enterprise-managed access group. Note that if an enterprise
          member is removed from an enterprise-managed access group in a child account and
          you reassign the template, the membership is reinstated.
    """

    def __init__(
        self,
        *,
        add: bool = None,
        remove: bool = None,
    ) -> None:
        """
        Initialize a MembersActionControls object.

        :param bool add: (optional) Action control for adding child account members
               to an enterprise-managed access group. If an access group administrator in
               a child account adds a member, they can always remove them. Note that if
               conflicts arise between an update to this control in a new version and
               members added by an administrator in the child account, you must resolve
               those conflicts in the child account. This prevents breaking access in the
               child account. For more information, see [Working with versions]
               (https://test.cloud.ibm.com/docs/secure-enterprise?topic=secure-enterprise-working-with-versions#new-version-scenarios).
        :param bool remove: (optional) Action control for removing
               enterprise-managed members from an enterprise-managed access group. Note
               that if an enterprise member is removed from an enterprise-managed access
               group in a child account and you reassign the template, the membership is
               reinstated.
        """
        self.add = add
        self.remove = remove

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'MembersActionControls':
        """Initialize a MembersActionControls object from a json dictionary."""
        args = {}
        if 'add' in _dict:
            args['add'] = _dict.get('add')
        if 'remove' in _dict:
            args['remove'] = _dict.get('remove')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a MembersActionControls object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'add') and self.add is not None:
            _dict['add'] = self.add
        if hasattr(self, 'remove') and self.remove is not None:
            _dict['remove'] = self.remove
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this MembersActionControls object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'MembersActionControls') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'MembersActionControls') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class PolicyTemplates:
    """
    Policy Templates Input component.

    :attr str id: (optional) Policy template ID.
    :attr str version: (optional) Policy template version.
    """

    def __init__(
        self,
        *,
        id: str = None,
        version: str = None,
    ) -> None:
        """
        Initialize a PolicyTemplates object.

        :param str id: (optional) Policy template ID.
        :param str version: (optional) Policy template version.
        """
        self.id = id
        self.version = version

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'PolicyTemplates':
        """Initialize a PolicyTemplates object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'version' in _dict:
            args['version'] = _dict.get('version')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a PolicyTemplates object from a json dictionary."""
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
        """Return a `str` version of this PolicyTemplates object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'PolicyTemplates') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'PolicyTemplates') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ResourceListWithTargetAccountID:
    """
    Object containing details of a resource list with target account ID.

    :attr str target: (optional) The ID of the entity that the resource list applies
          to.
    :attr AssignmentResourceAccessGroup group: (optional) Assignment Resource Access
          Group.
    :attr List[AssignmentResourceEntry] policy_template_references: (optional) List
          of policy template references for the resource list.
    """

    def __init__(
        self,
        *,
        target: str = None,
        group: 'AssignmentResourceAccessGroup' = None,
        policy_template_references: List['AssignmentResourceEntry'] = None,
    ) -> None:
        """
        Initialize a ResourceListWithTargetAccountID object.

        :param str target: (optional) The ID of the entity that the resource list
               applies to.
        :param AssignmentResourceAccessGroup group: (optional) Assignment Resource
               Access Group.
        :param List[AssignmentResourceEntry] policy_template_references: (optional)
               List of policy template references for the resource list.
        """
        self.target = target
        self.group = group
        self.policy_template_references = policy_template_references

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourceListWithTargetAccountID':
        """Initialize a ResourceListWithTargetAccountID object from a json dictionary."""
        args = {}
        if 'target' in _dict:
            args['target'] = _dict.get('target')
        if 'group' in _dict:
            args['group'] = AssignmentResourceAccessGroup.from_dict(_dict.get('group'))
        if 'policy_template_references' in _dict:
            args['policy_template_references'] = [
                AssignmentResourceEntry.from_dict(v) for v in _dict.get('policy_template_references')
            ]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourceListWithTargetAccountID object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'target') and self.target is not None:
            _dict['target'] = self.target
        if hasattr(self, 'group') and self.group is not None:
            if isinstance(self.group, dict):
                _dict['group'] = self.group
            else:
                _dict['group'] = self.group.to_dict()
        if hasattr(self, 'policy_template_references') and self.policy_template_references is not None:
            policy_template_references_list = []
            for v in self.policy_template_references:
                if isinstance(v, dict):
                    policy_template_references_list.append(v)
                else:
                    policy_template_references_list.append(v.to_dict())
            _dict['policy_template_references'] = policy_template_references_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResourceListWithTargetAccountID object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourceListWithTargetAccountID') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourceListWithTargetAccountID') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Rule:
    """
    A dynamic rule of an access group.

    :attr str id: (optional) The rule id.
    :attr str name: (optional) The name of the rule.
    :attr int expiration: (optional) Session duration in hours. Access group
          membership is revoked after this time period expires. Users must log back in to
          refresh their access group membership. Must be between 1 and 24.
    :attr str realm_name: (optional) The URL of the identity provider.
    :attr str access_group_id: (optional) The group id that the dynamic rule is
          assigned to.
    :attr str account_id: (optional) The account id that the group is in.
    :attr List[RuleConditions] conditions: (optional) A list of conditions that
          identities must satisfy to gain access group membership.
    :attr datetime created_at: (optional) The timestamp for when the rule was
          created.
    :attr str created_by_id: (optional) The `iam_id` of the entity that created the
          dynamic rule.
    :attr datetime last_modified_at: (optional) The timestamp for when the dynamic
          rule was last edited.
    :attr str last_modified_by_id: (optional) The IAM id that last modified the
          rule.
    """

    def __init__(
        self,
        *,
        id: str = None,
        name: str = None,
        expiration: int = None,
        realm_name: str = None,
        access_group_id: str = None,
        account_id: str = None,
        conditions: List['RuleConditions'] = None,
        created_at: datetime = None,
        created_by_id: str = None,
        last_modified_at: datetime = None,
        last_modified_by_id: str = None,
    ) -> None:
        """
        Initialize a Rule object.

        :param str id: (optional) The rule id.
        :param str name: (optional) The name of the rule.
        :param int expiration: (optional) Session duration in hours. Access group
               membership is revoked after this time period expires. Users must log back
               in to refresh their access group membership. Must be between 1 and 24.
        :param str realm_name: (optional) The URL of the identity provider.
        :param str access_group_id: (optional) The group id that the dynamic rule
               is assigned to.
        :param str account_id: (optional) The account id that the group is in.
        :param List[RuleConditions] conditions: (optional) A list of conditions
               that identities must satisfy to gain access group membership.
        :param datetime created_at: (optional) The timestamp for when the rule was
               created.
        :param str created_by_id: (optional) The `iam_id` of the entity that
               created the dynamic rule.
        :param datetime last_modified_at: (optional) The timestamp for when the
               dynamic rule was last edited.
        :param str last_modified_by_id: (optional) The IAM id that last modified
               the rule.
        """
        self.id = id
        self.name = name
        self.expiration = expiration
        self.realm_name = realm_name
        self.access_group_id = access_group_id
        self.account_id = account_id
        self.conditions = conditions
        self.created_at = created_at
        self.created_by_id = created_by_id
        self.last_modified_at = last_modified_at
        self.last_modified_by_id = last_modified_by_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Rule':
        """Initialize a Rule object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'expiration' in _dict:
            args['expiration'] = _dict.get('expiration')
        if 'realm_name' in _dict:
            args['realm_name'] = _dict.get('realm_name')
        if 'access_group_id' in _dict:
            args['access_group_id'] = _dict.get('access_group_id')
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        if 'conditions' in _dict:
            args['conditions'] = [RuleConditions.from_dict(v) for v in _dict.get('conditions')]
        if 'created_at' in _dict:
            args['created_at'] = string_to_datetime(_dict.get('created_at'))
        if 'created_by_id' in _dict:
            args['created_by_id'] = _dict.get('created_by_id')
        if 'last_modified_at' in _dict:
            args['last_modified_at'] = string_to_datetime(_dict.get('last_modified_at'))
        if 'last_modified_by_id' in _dict:
            args['last_modified_by_id'] = _dict.get('last_modified_by_id')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Rule object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'expiration') and self.expiration is not None:
            _dict['expiration'] = self.expiration
        if hasattr(self, 'realm_name') and self.realm_name is not None:
            _dict['realm_name'] = self.realm_name
        if hasattr(self, 'access_group_id') and self.access_group_id is not None:
            _dict['access_group_id'] = self.access_group_id
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'conditions') and self.conditions is not None:
            conditions_list = []
            for v in self.conditions:
                if isinstance(v, dict):
                    conditions_list.append(v)
                else:
                    conditions_list.append(v.to_dict())
            _dict['conditions'] = conditions_list
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'created_by_id') and self.created_by_id is not None:
            _dict['created_by_id'] = self.created_by_id
        if hasattr(self, 'last_modified_at') and self.last_modified_at is not None:
            _dict['last_modified_at'] = datetime_to_string(self.last_modified_at)
        if hasattr(self, 'last_modified_by_id') and self.last_modified_by_id is not None:
            _dict['last_modified_by_id'] = self.last_modified_by_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Rule object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Rule') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Rule') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class RuleActionControls:
    """
    Control whether or not access group administrators in child accounts can update and
    remove this dynamic rule in the enterprise-managed access group in their account.This
    overrides outer level AssertionsActionControls.

    :attr bool remove: (optional) Action control for removing this
          enterprise-managed dynamic rule.
    """

    def __init__(
        self,
        *,
        remove: bool = None,
    ) -> None:
        """
        Initialize a RuleActionControls object.

        :param bool remove: (optional) Action control for removing this
               enterprise-managed dynamic rule.
        """
        self.remove = remove

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'RuleActionControls':
        """Initialize a RuleActionControls object from a json dictionary."""
        args = {}
        if 'remove' in _dict:
            args['remove'] = _dict.get('remove')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a RuleActionControls object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'remove') and self.remove is not None:
            _dict['remove'] = self.remove
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this RuleActionControls object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'RuleActionControls') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'RuleActionControls') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class RuleConditions:
    """
    The conditions of a dynamic rule.

    :attr str claim: The claim to evaluate against. This will be found in the `ext`
          claims of a user's login request.
    :attr str operator: The operation to perform on the claim.
    :attr str value: The stringified JSON value that the claim is compared to using
          the operator.
    """

    def __init__(
        self,
        claim: str,
        operator: str,
        value: str,
    ) -> None:
        """
        Initialize a RuleConditions object.

        :param str claim: The claim to evaluate against. This will be found in the
               `ext` claims of a user's login request.
        :param str operator: The operation to perform on the claim.
        :param str value: The stringified JSON value that the claim is compared to
               using the operator.
        """
        self.claim = claim
        self.operator = operator
        self.value = value

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'RuleConditions':
        """Initialize a RuleConditions object from a json dictionary."""
        args = {}
        if 'claim' in _dict:
            args['claim'] = _dict.get('claim')
        else:
            raise ValueError('Required property \'claim\' not present in RuleConditions JSON')
        if 'operator' in _dict:
            args['operator'] = _dict.get('operator')
        else:
            raise ValueError('Required property \'operator\' not present in RuleConditions JSON')
        if 'value' in _dict:
            args['value'] = _dict.get('value')
        else:
            raise ValueError('Required property \'value\' not present in RuleConditions JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a RuleConditions object from a json dictionary."""
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
        """Return a `str` version of this RuleConditions object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'RuleConditions') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'RuleConditions') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class OperatorEnum(str, Enum):
        """
        The operation to perform on the claim.
        """

        EQUALS = 'EQUALS'
        EQUALS_IGNORE_CASE = 'EQUALS_IGNORE_CASE'
        IN = 'IN'
        NOT_EQUALS_IGNORE_CASE = 'NOT_EQUALS_IGNORE_CASE'
        NOT_EQUALS = 'NOT_EQUALS'
        CONTAINS = 'CONTAINS'


class RulesList:
    """
    A list of dynamic rules attached to the access group.

    :attr List[Rule] rules: (optional) A list of dynamic rules.
    """

    def __init__(
        self,
        *,
        rules: List['Rule'] = None,
    ) -> None:
        """
        Initialize a RulesList object.

        :param List[Rule] rules: (optional) A list of dynamic rules.
        """
        self.rules = rules

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'RulesList':
        """Initialize a RulesList object from a json dictionary."""
        args = {}
        if 'rules' in _dict:
            args['rules'] = [Rule.from_dict(v) for v in _dict.get('rules')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a RulesList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
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
        """Return a `str` version of this RulesList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'RulesList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'RulesList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TemplateAssignmentResponse:
    """
    Response object containing the details of a template assignment.

    :attr str id: The ID of the assignment.
    :attr str account_id: The ID of the account that the assignment belongs to.
    :attr str template_id: The ID of the template that the assignment is based on.
    :attr str template_version: The version of the template that the assignment is
          based on.
    :attr str target_type: The type of the entity that the assignment applies to.
    :attr str target: The ID of the entity that the assignment applies to.
    :attr str operation: The operation that the assignment applies to (e.g.
          'assign', 'update', 'remove').
    :attr str status: The status of the assignment (e.g. 'accepted', 'in_progress',
          'succeeded', 'failed', 'superseded').
    :attr str href: The URL of the assignment resource.
    :attr datetime created_at: The date and time when the assignment was created.
    :attr str created_by_id: The user or system that created the assignment.
    :attr datetime last_modified_at: The date and time when the assignment was last
          updated.
    :attr str last_modified_by_id: The user or system that last updated the
          assignment.
    """

    def __init__(
        self,
        id: str,
        account_id: str,
        template_id: str,
        template_version: str,
        target_type: str,
        target: str,
        operation: str,
        status: str,
        href: str,
        created_at: datetime,
        created_by_id: str,
        last_modified_at: datetime,
        last_modified_by_id: str,
    ) -> None:
        """
        Initialize a TemplateAssignmentResponse object.

        :param str id: The ID of the assignment.
        :param str account_id: The ID of the account that the assignment belongs
               to.
        :param str template_id: The ID of the template that the assignment is based
               on.
        :param str template_version: The version of the template that the
               assignment is based on.
        :param str target_type: The type of the entity that the assignment applies
               to.
        :param str target: The ID of the entity that the assignment applies to.
        :param str operation: The operation that the assignment applies to (e.g.
               'assign', 'update', 'remove').
        :param str status: The status of the assignment (e.g. 'accepted',
               'in_progress', 'succeeded', 'failed', 'superseded').
        :param str href: The URL of the assignment resource.
        :param datetime created_at: The date and time when the assignment was
               created.
        :param str created_by_id: The user or system that created the assignment.
        :param datetime last_modified_at: The date and time when the assignment was
               last updated.
        :param str last_modified_by_id: The user or system that last updated the
               assignment.
        """
        self.id = id
        self.account_id = account_id
        self.template_id = template_id
        self.template_version = template_version
        self.target_type = target_type
        self.target = target
        self.operation = operation
        self.status = status
        self.href = href
        self.created_at = created_at
        self.created_by_id = created_by_id
        self.last_modified_at = last_modified_at
        self.last_modified_by_id = last_modified_by_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TemplateAssignmentResponse':
        """Initialize a TemplateAssignmentResponse object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        else:
            raise ValueError('Required property \'id\' not present in TemplateAssignmentResponse JSON')
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        else:
            raise ValueError('Required property \'account_id\' not present in TemplateAssignmentResponse JSON')
        if 'template_id' in _dict:
            args['template_id'] = _dict.get('template_id')
        else:
            raise ValueError('Required property \'template_id\' not present in TemplateAssignmentResponse JSON')
        if 'template_version' in _dict:
            args['template_version'] = _dict.get('template_version')
        else:
            raise ValueError('Required property \'template_version\' not present in TemplateAssignmentResponse JSON')
        if 'target_type' in _dict:
            args['target_type'] = _dict.get('target_type')
        else:
            raise ValueError('Required property \'target_type\' not present in TemplateAssignmentResponse JSON')
        if 'target' in _dict:
            args['target'] = _dict.get('target')
        else:
            raise ValueError('Required property \'target\' not present in TemplateAssignmentResponse JSON')
        if 'operation' in _dict:
            args['operation'] = _dict.get('operation')
        else:
            raise ValueError('Required property \'operation\' not present in TemplateAssignmentResponse JSON')
        if 'status' in _dict:
            args['status'] = _dict.get('status')
        else:
            raise ValueError('Required property \'status\' not present in TemplateAssignmentResponse JSON')
        if 'href' in _dict:
            args['href'] = _dict.get('href')
        else:
            raise ValueError('Required property \'href\' not present in TemplateAssignmentResponse JSON')
        if 'created_at' in _dict:
            args['created_at'] = string_to_datetime(_dict.get('created_at'))
        else:
            raise ValueError('Required property \'created_at\' not present in TemplateAssignmentResponse JSON')
        if 'created_by_id' in _dict:
            args['created_by_id'] = _dict.get('created_by_id')
        else:
            raise ValueError('Required property \'created_by_id\' not present in TemplateAssignmentResponse JSON')
        if 'last_modified_at' in _dict:
            args['last_modified_at'] = string_to_datetime(_dict.get('last_modified_at'))
        else:
            raise ValueError('Required property \'last_modified_at\' not present in TemplateAssignmentResponse JSON')
        if 'last_modified_by_id' in _dict:
            args['last_modified_by_id'] = _dict.get('last_modified_by_id')
        else:
            raise ValueError('Required property \'last_modified_by_id\' not present in TemplateAssignmentResponse JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TemplateAssignmentResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
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
        if hasattr(self, 'operation') and self.operation is not None:
            _dict['operation'] = self.operation
        if hasattr(self, 'status') and self.status is not None:
            _dict['status'] = self.status
        if hasattr(self, 'href') and self.href is not None:
            _dict['href'] = self.href
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'created_by_id') and self.created_by_id is not None:
            _dict['created_by_id'] = self.created_by_id
        if hasattr(self, 'last_modified_at') and self.last_modified_at is not None:
            _dict['last_modified_at'] = datetime_to_string(self.last_modified_at)
        if hasattr(self, 'last_modified_by_id') and self.last_modified_by_id is not None:
            _dict['last_modified_by_id'] = self.last_modified_by_id
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

    class TargetTypeEnum(str, Enum):
        """
        The type of the entity that the assignment applies to.
        """

        ACCOUNT = 'Account'
        ACCOUNTGROUP = 'AccountGroup'

    class OperationEnum(str, Enum):
        """
        The operation that the assignment applies to (e.g. 'assign', 'update', 'remove').
        """

        ASSIGN = 'assign'
        UPDATE = 'update'
        REMOVE = 'remove'

    class StatusEnum(str, Enum):
        """
        The status of the assignment (e.g. 'accepted', 'in_progress', 'succeeded',
        'failed', 'superseded').
        """

        ACCEPTED = 'accepted'
        IN_PROGRESS = 'in_progress'
        SUCCEEDED = 'succeeded'
        FAILED = 'failed'
        SUPERSEDED = 'superseded'


class TemplateAssignmentVerboseResponse:
    """
    Response object containing the details of a template assignment.

    :attr str id: The ID of the assignment.
    :attr str account_id: The ID of the account that the assignment belongs to.
    :attr str template_id: The ID of the template that the assignment is based on.
    :attr str template_version: The version of the template that the assignment is
          based on.
    :attr str target_type: The type of the entity that the assignment applies to.
    :attr str target: The ID of the entity that the assignment applies to.
    :attr str operation: The operation that the assignment applies to (e.g.
          'create', 'update', 'delete').
    :attr str status: The status of the assignment (e.g. 'pending', 'success',
          'failure').
    :attr List[ResourceListWithTargetAccountID] resources: (optional) List of
          resources for the assignment.
    :attr str href: The URL of the assignment resource.
    :attr datetime created_at: The date and time when the assignment was created.
    :attr str created_by_id: The user or system that created the assignment.
    :attr datetime last_modified_at: The date and time when the assignment was last
          updated.
    :attr str last_modified_by_id: The user or system that last updated the
          assignment.
    """

    def __init__(
        self,
        id: str,
        account_id: str,
        template_id: str,
        template_version: str,
        target_type: str,
        target: str,
        operation: str,
        status: str,
        href: str,
        created_at: datetime,
        created_by_id: str,
        last_modified_at: datetime,
        last_modified_by_id: str,
        *,
        resources: List['ResourceListWithTargetAccountID'] = None,
    ) -> None:
        """
        Initialize a TemplateAssignmentVerboseResponse object.

        :param str id: The ID of the assignment.
        :param str account_id: The ID of the account that the assignment belongs
               to.
        :param str template_id: The ID of the template that the assignment is based
               on.
        :param str template_version: The version of the template that the
               assignment is based on.
        :param str target_type: The type of the entity that the assignment applies
               to.
        :param str target: The ID of the entity that the assignment applies to.
        :param str operation: The operation that the assignment applies to (e.g.
               'create', 'update', 'delete').
        :param str status: The status of the assignment (e.g. 'pending', 'success',
               'failure').
        :param str href: The URL of the assignment resource.
        :param datetime created_at: The date and time when the assignment was
               created.
        :param str created_by_id: The user or system that created the assignment.
        :param datetime last_modified_at: The date and time when the assignment was
               last updated.
        :param str last_modified_by_id: The user or system that last updated the
               assignment.
        :param List[ResourceListWithTargetAccountID] resources: (optional) List of
               resources for the assignment.
        """
        self.id = id
        self.account_id = account_id
        self.template_id = template_id
        self.template_version = template_version
        self.target_type = target_type
        self.target = target
        self.operation = operation
        self.status = status
        self.resources = resources
        self.href = href
        self.created_at = created_at
        self.created_by_id = created_by_id
        self.last_modified_at = last_modified_at
        self.last_modified_by_id = last_modified_by_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TemplateAssignmentVerboseResponse':
        """Initialize a TemplateAssignmentVerboseResponse object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        else:
            raise ValueError('Required property \'id\' not present in TemplateAssignmentVerboseResponse JSON')
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        else:
            raise ValueError('Required property \'account_id\' not present in TemplateAssignmentVerboseResponse JSON')
        if 'template_id' in _dict:
            args['template_id'] = _dict.get('template_id')
        else:
            raise ValueError('Required property \'template_id\' not present in TemplateAssignmentVerboseResponse JSON')
        if 'template_version' in _dict:
            args['template_version'] = _dict.get('template_version')
        else:
            raise ValueError(
                'Required property \'template_version\' not present in TemplateAssignmentVerboseResponse JSON'
            )
        if 'target_type' in _dict:
            args['target_type'] = _dict.get('target_type')
        else:
            raise ValueError('Required property \'target_type\' not present in TemplateAssignmentVerboseResponse JSON')
        if 'target' in _dict:
            args['target'] = _dict.get('target')
        else:
            raise ValueError('Required property \'target\' not present in TemplateAssignmentVerboseResponse JSON')
        if 'operation' in _dict:
            args['operation'] = _dict.get('operation')
        else:
            raise ValueError('Required property \'operation\' not present in TemplateAssignmentVerboseResponse JSON')
        if 'status' in _dict:
            args['status'] = _dict.get('status')
        else:
            raise ValueError('Required property \'status\' not present in TemplateAssignmentVerboseResponse JSON')
        if 'resources' in _dict:
            args['resources'] = [ResourceListWithTargetAccountID.from_dict(v) for v in _dict.get('resources')]
        if 'href' in _dict:
            args['href'] = _dict.get('href')
        else:
            raise ValueError('Required property \'href\' not present in TemplateAssignmentVerboseResponse JSON')
        if 'created_at' in _dict:
            args['created_at'] = string_to_datetime(_dict.get('created_at'))
        else:
            raise ValueError('Required property \'created_at\' not present in TemplateAssignmentVerboseResponse JSON')
        if 'created_by_id' in _dict:
            args['created_by_id'] = _dict.get('created_by_id')
        else:
            raise ValueError(
                'Required property \'created_by_id\' not present in TemplateAssignmentVerboseResponse JSON'
            )
        if 'last_modified_at' in _dict:
            args['last_modified_at'] = string_to_datetime(_dict.get('last_modified_at'))
        else:
            raise ValueError(
                'Required property \'last_modified_at\' not present in TemplateAssignmentVerboseResponse JSON'
            )
        if 'last_modified_by_id' in _dict:
            args['last_modified_by_id'] = _dict.get('last_modified_by_id')
        else:
            raise ValueError(
                'Required property \'last_modified_by_id\' not present in TemplateAssignmentVerboseResponse JSON'
            )
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TemplateAssignmentVerboseResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
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
        if hasattr(self, 'operation') and self.operation is not None:
            _dict['operation'] = self.operation
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
        if hasattr(self, 'href') and self.href is not None:
            _dict['href'] = self.href
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'created_by_id') and self.created_by_id is not None:
            _dict['created_by_id'] = self.created_by_id
        if hasattr(self, 'last_modified_at') and self.last_modified_at is not None:
            _dict['last_modified_at'] = datetime_to_string(self.last_modified_at)
        if hasattr(self, 'last_modified_by_id') and self.last_modified_by_id is not None:
            _dict['last_modified_by_id'] = self.last_modified_by_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TemplateAssignmentVerboseResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TemplateAssignmentVerboseResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TemplateAssignmentVerboseResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TemplateResponse:
    """
    Response output for template.

    :attr str id: The ID of the access group template.
    :attr str name: The name of the access group template.
    :attr str description: The description of the access group template.
    :attr str account_id: The ID of the account to which the access group template
          is assigned.
    :attr str version: The version of the access group template.
    :attr bool committed: A boolean indicating whether the access group template is
          committed. You must commit a template before you can assign it to child
          accounts.
    :attr AccessGroupResponse group: Access Group Component.
    :attr List[PolicyTemplates] policy_template_references: References to policy
          templates assigned to the access group template.
    :attr str href: The URL of the access group template resource.
    :attr datetime created_at: The date and time when the access group template was
          created.
    :attr str created_by_id: The ID of the user who created the access group
          template.
    :attr datetime last_modified_at: The date and time when the access group
          template was last modified.
    :attr str last_modified_by_id: The ID of the user who last modified the access
          group template.
    """

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        account_id: str,
        version: str,
        committed: bool,
        group: 'AccessGroupResponse',
        policy_template_references: List['PolicyTemplates'],
        href: str,
        created_at: datetime,
        created_by_id: str,
        last_modified_at: datetime,
        last_modified_by_id: str,
    ) -> None:
        """
        Initialize a TemplateResponse object.

        :param str id: The ID of the access group template.
        :param str name: The name of the access group template.
        :param str description: The description of the access group template.
        :param str account_id: The ID of the account to which the access group
               template is assigned.
        :param str version: The version of the access group template.
        :param bool committed: A boolean indicating whether the access group
               template is committed. You must commit a template before you can assign it
               to child accounts.
        :param AccessGroupResponse group: Access Group Component.
        :param List[PolicyTemplates] policy_template_references: References to
               policy templates assigned to the access group template.
        :param str href: The URL of the access group template resource.
        :param datetime created_at: The date and time when the access group
               template was created.
        :param str created_by_id: The ID of the user who created the access group
               template.
        :param datetime last_modified_at: The date and time when the access group
               template was last modified.
        :param str last_modified_by_id: The ID of the user who last modified the
               access group template.
        """
        self.id = id
        self.name = name
        self.description = description
        self.account_id = account_id
        self.version = version
        self.committed = committed
        self.group = group
        self.policy_template_references = policy_template_references
        self.href = href
        self.created_at = created_at
        self.created_by_id = created_by_id
        self.last_modified_at = last_modified_at
        self.last_modified_by_id = last_modified_by_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TemplateResponse':
        """Initialize a TemplateResponse object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        else:
            raise ValueError('Required property \'id\' not present in TemplateResponse JSON')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        else:
            raise ValueError('Required property \'name\' not present in TemplateResponse JSON')
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        else:
            raise ValueError('Required property \'description\' not present in TemplateResponse JSON')
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        else:
            raise ValueError('Required property \'account_id\' not present in TemplateResponse JSON')
        if 'version' in _dict:
            args['version'] = _dict.get('version')
        else:
            raise ValueError('Required property \'version\' not present in TemplateResponse JSON')
        if 'committed' in _dict:
            args['committed'] = _dict.get('committed')
        else:
            raise ValueError('Required property \'committed\' not present in TemplateResponse JSON')
        if 'group' in _dict:
            args['group'] = AccessGroupResponse.from_dict(_dict.get('group'))
        else:
            raise ValueError('Required property \'group\' not present in TemplateResponse JSON')
        if 'policy_template_references' in _dict:
            args['policy_template_references'] = [
                PolicyTemplates.from_dict(v) for v in _dict.get('policy_template_references')
            ]
        else:
            raise ValueError('Required property \'policy_template_references\' not present in TemplateResponse JSON')
        if 'href' in _dict:
            args['href'] = _dict.get('href')
        else:
            raise ValueError('Required property \'href\' not present in TemplateResponse JSON')
        if 'created_at' in _dict:
            args['created_at'] = string_to_datetime(_dict.get('created_at'))
        else:
            raise ValueError('Required property \'created_at\' not present in TemplateResponse JSON')
        if 'created_by_id' in _dict:
            args['created_by_id'] = _dict.get('created_by_id')
        else:
            raise ValueError('Required property \'created_by_id\' not present in TemplateResponse JSON')
        if 'last_modified_at' in _dict:
            args['last_modified_at'] = string_to_datetime(_dict.get('last_modified_at'))
        else:
            raise ValueError('Required property \'last_modified_at\' not present in TemplateResponse JSON')
        if 'last_modified_by_id' in _dict:
            args['last_modified_by_id'] = _dict.get('last_modified_by_id')
        else:
            raise ValueError('Required property \'last_modified_by_id\' not present in TemplateResponse JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TemplateResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'version') and self.version is not None:
            _dict['version'] = self.version
        if hasattr(self, 'committed') and self.committed is not None:
            _dict['committed'] = self.committed
        if hasattr(self, 'group') and self.group is not None:
            if isinstance(self.group, dict):
                _dict['group'] = self.group
            else:
                _dict['group'] = self.group.to_dict()
        if hasattr(self, 'policy_template_references') and self.policy_template_references is not None:
            policy_template_references_list = []
            for v in self.policy_template_references:
                if isinstance(v, dict):
                    policy_template_references_list.append(v)
                else:
                    policy_template_references_list.append(v.to_dict())
            _dict['policy_template_references'] = policy_template_references_list
        if hasattr(self, 'href') and self.href is not None:
            _dict['href'] = self.href
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'created_by_id') and self.created_by_id is not None:
            _dict['created_by_id'] = self.created_by_id
        if hasattr(self, 'last_modified_at') and self.last_modified_at is not None:
            _dict['last_modified_at'] = datetime_to_string(self.last_modified_at)
        if hasattr(self, 'last_modified_by_id') and self.last_modified_by_id is not None:
            _dict['last_modified_by_id'] = self.last_modified_by_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TemplateResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TemplateResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TemplateResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TemplateVersionResponse:
    """
    Response output for template.

    :attr str id: The ID of the access group template.
    :attr str name: The name of the access group template.
    :attr str description: The description of the access group template.
    :attr str account_id: The ID of the account to which the access group template
          is assigned.
    :attr str version: The version of the access group template.
    :attr bool committed: A boolean indicating whether the access group template is
          committed. You must commit a template before you can assign it to child
          accounts.
    :attr AccessGroupResponse group: Access Group Component.
    :attr List[PolicyTemplates] policy_template_references: References to policy
          templates assigned to the access group template.
    :attr str href: The URL of the access group template resource.
    :attr datetime created_at: The date and time when the access group template was
          created.
    :attr str created_by_id: The ID of the user who created the access group
          template.
    :attr datetime last_modified_at: The date and time when the access group
          template was last modified.
    :attr str last_modified_by_id: The ID of the user who last modified the access
          group template.
    """

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        account_id: str,
        version: str,
        committed: bool,
        group: 'AccessGroupResponse',
        policy_template_references: List['PolicyTemplates'],
        href: str,
        created_at: datetime,
        created_by_id: str,
        last_modified_at: datetime,
        last_modified_by_id: str,
    ) -> None:
        """
        Initialize a TemplateVersionResponse object.

        :param str id: The ID of the access group template.
        :param str name: The name of the access group template.
        :param str description: The description of the access group template.
        :param str account_id: The ID of the account to which the access group
               template is assigned.
        :param str version: The version of the access group template.
        :param bool committed: A boolean indicating whether the access group
               template is committed. You must commit a template before you can assign it
               to child accounts.
        :param AccessGroupResponse group: Access Group Component.
        :param List[PolicyTemplates] policy_template_references: References to
               policy templates assigned to the access group template.
        :param str href: The URL of the access group template resource.
        :param datetime created_at: The date and time when the access group
               template was created.
        :param str created_by_id: The ID of the user who created the access group
               template.
        :param datetime last_modified_at: The date and time when the access group
               template was last modified.
        :param str last_modified_by_id: The ID of the user who last modified the
               access group template.
        """
        self.id = id
        self.name = name
        self.description = description
        self.account_id = account_id
        self.version = version
        self.committed = committed
        self.group = group
        self.policy_template_references = policy_template_references
        self.href = href
        self.created_at = created_at
        self.created_by_id = created_by_id
        self.last_modified_at = last_modified_at
        self.last_modified_by_id = last_modified_by_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TemplateVersionResponse':
        """Initialize a TemplateVersionResponse object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        else:
            raise ValueError('Required property \'id\' not present in TemplateVersionResponse JSON')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        else:
            raise ValueError('Required property \'name\' not present in TemplateVersionResponse JSON')
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        else:
            raise ValueError('Required property \'description\' not present in TemplateVersionResponse JSON')
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        else:
            raise ValueError('Required property \'account_id\' not present in TemplateVersionResponse JSON')
        if 'version' in _dict:
            args['version'] = _dict.get('version')
        else:
            raise ValueError('Required property \'version\' not present in TemplateVersionResponse JSON')
        if 'committed' in _dict:
            args['committed'] = _dict.get('committed')
        else:
            raise ValueError('Required property \'committed\' not present in TemplateVersionResponse JSON')
        if 'group' in _dict:
            args['group'] = AccessGroupResponse.from_dict(_dict.get('group'))
        else:
            raise ValueError('Required property \'group\' not present in TemplateVersionResponse JSON')
        if 'policy_template_references' in _dict:
            args['policy_template_references'] = [
                PolicyTemplates.from_dict(v) for v in _dict.get('policy_template_references')
            ]
        else:
            raise ValueError(
                'Required property \'policy_template_references\' not present in TemplateVersionResponse JSON'
            )
        if 'href' in _dict:
            args['href'] = _dict.get('href')
        else:
            raise ValueError('Required property \'href\' not present in TemplateVersionResponse JSON')
        if 'created_at' in _dict:
            args['created_at'] = string_to_datetime(_dict.get('created_at'))
        else:
            raise ValueError('Required property \'created_at\' not present in TemplateVersionResponse JSON')
        if 'created_by_id' in _dict:
            args['created_by_id'] = _dict.get('created_by_id')
        else:
            raise ValueError('Required property \'created_by_id\' not present in TemplateVersionResponse JSON')
        if 'last_modified_at' in _dict:
            args['last_modified_at'] = string_to_datetime(_dict.get('last_modified_at'))
        else:
            raise ValueError('Required property \'last_modified_at\' not present in TemplateVersionResponse JSON')
        if 'last_modified_by_id' in _dict:
            args['last_modified_by_id'] = _dict.get('last_modified_by_id')
        else:
            raise ValueError('Required property \'last_modified_by_id\' not present in TemplateVersionResponse JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TemplateVersionResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'version') and self.version is not None:
            _dict['version'] = self.version
        if hasattr(self, 'committed') and self.committed is not None:
            _dict['committed'] = self.committed
        if hasattr(self, 'group') and self.group is not None:
            if isinstance(self.group, dict):
                _dict['group'] = self.group
            else:
                _dict['group'] = self.group.to_dict()
        if hasattr(self, 'policy_template_references') and self.policy_template_references is not None:
            policy_template_references_list = []
            for v in self.policy_template_references:
                if isinstance(v, dict):
                    policy_template_references_list.append(v)
                else:
                    policy_template_references_list.append(v.to_dict())
            _dict['policy_template_references'] = policy_template_references_list
        if hasattr(self, 'href') and self.href is not None:
            _dict['href'] = self.href
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        if hasattr(self, 'created_by_id') and self.created_by_id is not None:
            _dict['created_by_id'] = self.created_by_id
        if hasattr(self, 'last_modified_at') and self.last_modified_at is not None:
            _dict['last_modified_at'] = datetime_to_string(self.last_modified_at)
        if hasattr(self, 'last_modified_by_id') and self.last_modified_by_id is not None:
            _dict['last_modified_by_id'] = self.last_modified_by_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TemplateVersionResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TemplateVersionResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TemplateVersionResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


##############################################################################
# Pagers
##############################################################################


class AccessGroupsPager:
    """
    AccessGroupsPager can be used to simplify the use of the "list_access_groups" method.
    """

    def __init__(
        self,
        *,
        client: IamAccessGroupsV2,
        account_id: str,
        transaction_id: str = None,
        iam_id: str = None,
        search: str = None,
        membership_type: str = None,
        limit: int = None,
        sort: str = None,
        show_federated: bool = None,
        hide_public_access: bool = None,
    ) -> None:
        """
        Initialize a AccessGroupsPager object.
        :param str account_id: Account ID of the API keys(s) to query. If a service
               IAM ID is specified in iam_id then account_id must match the account of the
               IAM ID. If a user IAM ID is specified in iam_id then then account_id must
               match the account of the Authorization token.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param str iam_id: (optional) Return groups for member ID (IBMid, service
               ID or trusted profile ID).
        :param str search: (optional) Use search to filter access groups list by
               id, name or description.
               * `search=id:<ACCESS_GROUP_ID>` - To list access groups by id
               * `search=name:<ACCESS_GROUP_NAME>` - To list access groups by name
               * `search=description:<ACCESS_GROUP_DESC>` - To list access groups by
               description.
        :param str membership_type: (optional) Membership type need to be specified
               along with iam_id and must be either `static`, `dynamic` or `all`. If
               membership type is `static`, members explicitly added to the group will be
               shown. If membership type is `dynamic`, members accessing the access group
               at the moment via dynamic rules will be shown. If membership type is `all`,
               both static and dynamic members will be shown.
        :param int limit: (optional) Return up to this limit of results where limit
               is between 0 and 100.
        :param str sort: (optional) Sort the results by id, name, description, or
               is_federated flag.
        :param bool show_federated: (optional) If show_federated is true, each
               group listed will return an is_federated value that is set to true if rules
               exist for the group.
        :param bool hide_public_access: (optional) If hide_public_access is true,
               do not include the Public Access Group in the results.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._account_id = account_id
        self._transaction_id = transaction_id
        self._iam_id = iam_id
        self._search = search
        self._membership_type = membership_type
        self._limit = limit
        self._sort = sort
        self._show_federated = show_federated
        self._hide_public_access = hide_public_access

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of Group.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.list_access_groups(
            account_id=self._account_id,
            transaction_id=self._transaction_id,
            iam_id=self._iam_id,
            search=self._search,
            membership_type=self._membership_type,
            limit=self._limit,
            sort=self._sort,
            show_federated=self._show_federated,
            hide_public_access=self._hide_public_access,
            offset=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next')
        if next_page_link is not None:
            next = get_query_param(next_page_link.get('href'), 'offset')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('groups')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of Group.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results


class AccessGroupMembersPager:
    """
    AccessGroupMembersPager can be used to simplify the use of the "list_access_group_members" method.
    """

    def __init__(
        self,
        *,
        client: IamAccessGroupsV2,
        access_group_id: str,
        transaction_id: str = None,
        membership_type: str = None,
        limit: int = None,
        type: str = None,
        verbose: bool = None,
        sort: str = None,
    ) -> None:
        """
        Initialize a AccessGroupMembersPager object.
        :param str access_group_id: The access group identifier.
        :param str transaction_id: (optional) An optional transaction ID can be
               passed to your request, which can be useful for tracking calls through
               multiple services by using one identifier. The header key must be set to
               Transaction-Id and the value is anything that you choose. If no transaction
               ID is passed in, then a random ID is generated.
        :param str membership_type: (optional) Filters members by membership type.
               Filter by `static`, `dynamic` or `all`. `static` lists the members
               explicitly added to the access group, and `dynamic` lists the members that
               are part of the access group at that time via dynamic rules. `all` lists
               both static and dynamic members.
        :param int limit: (optional) Return up to this limit of results where limit
               is between 0 and 100.
        :param str type: (optional) Filter the results by member type.
        :param bool verbose: (optional) Return user's email and name for each user
               ID or the name for each service ID or trusted profile.
        :param str sort: (optional) If verbose is true, sort the results by id,
               name, or email.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._access_group_id = access_group_id
        self._transaction_id = transaction_id
        self._membership_type = membership_type
        self._limit = limit
        self._type = type
        self._verbose = verbose
        self._sort = sort

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of ListGroupMembersResponseMember.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.list_access_group_members(
            access_group_id=self._access_group_id,
            transaction_id=self._transaction_id,
            membership_type=self._membership_type,
            limit=self._limit,
            type=self._type,
            verbose=self._verbose,
            sort=self._sort,
            offset=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next')
        if next_page_link is not None:
            next = get_query_param(next_page_link.get('href'), 'offset')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('members')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of ListGroupMembersResponseMember.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results


class TemplatesPager:
    """
    TemplatesPager can be used to simplify the use of the "list_templates" method.
    """

    def __init__(
        self,
        *,
        client: IamAccessGroupsV2,
        account_id: str,
        transaction_id: str = None,
        limit: int = None,
        verbose: bool = None,
    ) -> None:
        """
        Initialize a TemplatesPager object.
        :param str account_id: Enterprise account ID.
        :param str transaction_id: (optional) An optional transaction id for the
               request.
        :param int limit: (optional) Return up to this limit of results where limit
               is between 0 and 100.
        :param bool verbose: (optional) If `verbose=true`, IAM resource details are
               returned. If performance is a concern, leave the `verbose` parameter off so
               that details are not retrieved.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._account_id = account_id
        self._transaction_id = transaction_id
        self._limit = limit
        self._verbose = verbose

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of GroupTemplate.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.list_templates(
            account_id=self._account_id,
            transaction_id=self._transaction_id,
            limit=self._limit,
            verbose=self._verbose,
            offset=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next')
        if next_page_link is not None:
            next = get_query_param(next_page_link.get('href'), 'offset')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('group_templates')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of GroupTemplate.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results


class TemplateVersionsPager:
    """
    TemplateVersionsPager can be used to simplify the use of the "list_template_versions" method.
    """

    def __init__(
        self,
        *,
        client: IamAccessGroupsV2,
        template_id: str,
        limit: int = None,
    ) -> None:
        """
        Initialize a TemplateVersionsPager object.
        :param str template_id: ID of the template that you want to list all
               versions of.
        :param int limit: (optional) Return up to this limit of results where limit
               is between 0 and 100.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._template_id = template_id
        self._limit = limit

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of ListTemplateVersionResponse.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.list_template_versions(
            template_id=self._template_id,
            limit=self._limit,
            offset=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next')
        if next_page_link is not None:
            next = get_query_param(next_page_link.get('href'), 'offset')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('group_template_versions')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of ListTemplateVersionResponse.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results
