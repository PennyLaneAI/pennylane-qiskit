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

# IBM OpenAPI SDK Code Generator Version: 3.70.0-7df966bf-20230419-195904

"""
Manage the lifecycle of your users using User Management APIs.

API Version: 1.0
"""

from typing import Dict, List
import json

from ibm_cloud_sdk_core import BaseService, DetailedResponse, get_query_param
from ibm_cloud_sdk_core.authenticators.authenticator import Authenticator
from ibm_cloud_sdk_core.get_authenticator import get_authenticator_from_environment
from ibm_cloud_sdk_core.utils import convert_model

from .common import get_sdk_headers

##############################################################################
# Service
##############################################################################


class UserManagementV1(BaseService):
    """The User Management V1 service."""

    DEFAULT_SERVICE_URL = 'https://user-management.cloud.ibm.com'
    DEFAULT_SERVICE_NAME = 'user_management'

    @classmethod
    def new_instance(
        cls,
        service_name: str = DEFAULT_SERVICE_NAME,
    ) -> 'UserManagementV1':
        """
        Return a new client for the User Management service using the specified
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
        Construct a new client for the User Management service.

        :param Authenticator authenticator: The authenticator specifies the authentication mechanism.
               Get up to date information from https://github.com/IBM/python-sdk-core/blob/main/README.md
               about initializing the authenticator of your choice.
        """
        BaseService.__init__(self, service_url=self.DEFAULT_SERVICE_URL, authenticator=authenticator)

    #########################
    # Users
    #########################

    def list_users(
        self,
        account_id: str,
        *,
        limit: int = None,
        include_settings: bool = None,
        search: str = None,
        start: str = None,
        user_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List users.

        Retrieve users in the account. You can use the IAM service token or a user token
        for authorization. To use this method, the requesting user or service ID must have
        at least the viewer, editor, or administrator role on the User Management service.
        If unrestricted view is enabled, the user can see all users in the same account
        without an IAM role. If restricted view is enabled and user has the viewer,
        editor, or administrator role on the user management service, the API returns all
        users in the account. If unrestricted view is enabled and the user does not have
        these roles, the API returns only the current user. Users are returned in a
        paginated list with a default limit of 100 users. You can iterate through all
        users by following the `next_url` field. Additional substring search fields are
        supported to filter the users.

        :param str account_id: The account ID of the specified user.
        :param int limit: (optional) The number of results to be returned.
        :param bool include_settings: (optional) The user settings to be returned.
               Set to true to view language, allowed IP address, and authentication
               settings.
        :param str search: (optional) The desired search results to be returned. To
               view the list of users with the additional search filter, use the following
               query options: `firstname`, `lastname`, `email`, `state`, `substate`,
               `iam_id`, `realm`, and `userId`. HTML URL encoding for the search query and
               `:` must be used. For example, search=state%3AINVALID returns a list of
               invalid users. Multiple search queries can be combined to obtain `OR`
               results using `,` operator (not URL encoded). For example,
               search=state%3AINVALID,email%3Amail.test.ibm.com.
        :param str start: (optional) An optional token that indicates the beginning
               of the page of results to be returned. If omitted, the first page of
               results is returned. This value is obtained from the 'next_url' field of
               the operation response.
        :param str user_id: (optional) Filter users based on their user ID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `UserList` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='list_users',
        )
        headers.update(sdk_headers)

        params = {
            'limit': limit,
            'include_settings': include_settings,
            'search': search,
            '_start': start,
            'user_id': user_id,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id']
        path_param_values = self.encode_path_vars(account_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/accounts/{account_id}/users'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def invite_users(
        self,
        account_id: str,
        *,
        users: List['InviteUser'] = None,
        iam_policy: List['InviteUserIamPolicy'] = None,
        access_groups: List[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Invite users to an account.

        Invite users to the account. You must use a user token for authorization. Service
        IDs can't invite users to the account. To use this method, the requesting user
        must have the editor or administrator role on the User Management service. For
        more information, see the [Inviting
        users](https://cloud.ibm.com/docs/account?topic=account-iamuserinv) documentation.
        You can specify the user account role and the corresponding IAM policy information
        in the request body. <br/><br/>When you invite a user to an account, the user is
        initially created in the `PROCESSING` state. After the user is successfully
        created, all specified permissions are configured, and the activation email is
        sent, the invited user is transitioned to the `PENDING` state. When the invited
        user clicks the activation email and creates and confirms their IBM Cloud account,
        the user is transitioned to `ACTIVE` state. If the user email is already verified,
        no email is generated.

        :param str account_id: The account ID of the specified user.
        :param List[InviteUser] users: (optional) A list of users to be invited.
        :param List[InviteUserIamPolicy] iam_policy: (optional) A list of IAM
               policies.
        :param List[str] access_groups: (optional) A list of access groups.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `InvitedUserList` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if users is not None:
            users = [convert_model(x) for x in users]
        if iam_policy is not None:
            iam_policy = [convert_model(x) for x in iam_policy]
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='invite_users',
        )
        headers.update(sdk_headers)

        data = {
            'users': users,
            'iam_policy': iam_policy,
            'access_groups': access_groups,
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
        url = '/v2/accounts/{account_id}/users'.format(**path_param_dict)
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def get_user_profile(
        self,
        account_id: str,
        iam_id: str,
        *,
        include_activity: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get user profile.

        Retrieve a user's profile by the user's IAM ID in your account. You can use the
        IAM service token or a user token for authorization. To use this method, the
        requesting user or service ID must have at least the viewer, editor, or
        administrator role on the User Management service.

        :param str account_id: The account ID of the specified user.
        :param str iam_id: The user's IAM ID.
        :param str include_activity: (optional) Include activity information of the
               user, such as the last authentication timestamp.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `UserProfile` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not iam_id:
            raise ValueError('iam_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_user_profile',
        )
        headers.update(sdk_headers)

        params = {
            'include_activity': include_activity,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id', 'iam_id']
        path_param_values = self.encode_path_vars(account_id, iam_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/accounts/{account_id}/users/{iam_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def update_user_profile(
        self,
        account_id: str,
        iam_id: str,
        *,
        firstname: str = None,
        lastname: str = None,
        state: str = None,
        email: str = None,
        phonenumber: str = None,
        altphonenumber: str = None,
        photo: str = None,
        include_activity: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Partially update user profile.

        Partially update a user's profile by user's IAM ID. You can use the IAM service
        token or a user token for authorization. To use this method, the requesting user
        or service ID must have at least the editor or administrator role on the User
        Management service. A user or service ID with these roles can change a user's
        state between `ACTIVE`, `VPN_ONLY`, or `DISABLED_CLASSIC_INFRASTRUCTURE`, but they
        can't change the state to `PROCESSING` or `PENDING` because these are system
        states. For other request body fields, a user can update their own profile without
        having User Management service permissions.

        :param str account_id: The account ID of the specified user.
        :param str iam_id: The user's IAM ID.
        :param str firstname: (optional) The first name of the user.
        :param str lastname: (optional) The last name of the user.
        :param str state: (optional) The state of the user. Possible values are
               `PROCESSING`, `PENDING`, `ACTIVE`, `DISABLED_CLASSIC_INFRASTRUCTURE`, and
               `VPN_ONLY`.
        :param str email: (optional) The email address of the user.
        :param str phonenumber: (optional) The phone number of the user.
        :param str altphonenumber: (optional) The alternative phone number of the
               user.
        :param str photo: (optional) A link to a photo of the user.
        :param str include_activity: (optional) Include activity information of the
               user, such as the last authentication timestamp.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not iam_id:
            raise ValueError('iam_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='update_user_profile',
        )
        headers.update(sdk_headers)

        params = {
            'include_activity': include_activity,
        }

        data = {
            'firstname': firstname,
            'lastname': lastname,
            'state': state,
            'email': email,
            'phonenumber': phonenumber,
            'altphonenumber': altphonenumber,
            'photo': photo,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['account_id', 'iam_id']
        path_param_values = self.encode_path_vars(account_id, iam_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/accounts/{account_id}/users/{iam_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PATCH',
            url=url,
            headers=headers,
            params=params,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def remove_user(
        self,
        account_id: str,
        iam_id: str,
        *,
        include_activity: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Remove user from account.

        Remove users from an account by user's IAM ID. You must use a user token for
        authorization. Service IDs can't remove users from an account. To use this method,
        the requesting user must have the editor or administrator role on the User
        Management service. For more information, see the [Removing
        users](https://cloud.ibm.com/docs/account?topic=account-remove) documentation.

        :param str account_id: The account ID of the specified user.
        :param str iam_id: The user's IAM ID.
        :param str include_activity: (optional) Include activity information of the
               user, such as the last authentication timestamp.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not iam_id:
            raise ValueError('iam_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='remove_user',
        )
        headers.update(sdk_headers)

        params = {
            'include_activity': include_activity,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['account_id', 'iam_id']
        path_param_values = self.encode_path_vars(account_id, iam_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/accounts/{account_id}/users/{iam_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def accept(
        self,
        *,
        account_id: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Accept an invitation.

        Accept a user invitation to an account. You can use the user's token for
        authorization. To use this method, the requesting user must provide the account ID
        for the account that they are accepting an invitation for. If the user already
        accepted the invitation request, it returns 204 with no response body.

        :param str account_id: (optional) The account ID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='accept',
        )
        headers.update(sdk_headers)

        data = {
            'account_id': account_id,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        url = '/v2/users/accept'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def v3_remove_user(
        self,
        account_id: str,
        iam_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Remove user from account (Asynchronous).

        Remove users from an account by using the user's IAM ID. You must use a user token
        for authorization. Service IDs can't remove users from an account. If removing the
        user fails it will set the user's state to ERROR_WHILE_DELETING. To use this
        method, the requesting user must have the editor or administrator role on the User
        Management service. For more information, see the [Removing
        users](https://cloud.ibm.com/docs/account?topic=account-remove) documentation.

        :param str account_id: The account ID of the specified user.
        :param str iam_id: The user's IAM ID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not iam_id:
            raise ValueError('iam_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='v3_remove_user',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['account_id', 'iam_id']
        path_param_values = self.encode_path_vars(account_id, iam_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v3/accounts/{account_id}/users/{iam_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # User Settings
    #########################

    def get_user_settings(
        self,
        account_id: str,
        iam_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get user settings.

        Retrieve a user's settings by the user's IAM ID. You can use the IAM service token
        or a user token for authorization. To use this method, the requesting user or
        service ID must have the viewer, editor, or administrator role on the User
        Management service. <br/><br/>The user settings have several fields. The
        `language` field is the language setting for the user interface display language.
        The `notification_language` field is the language setting for phone and email
        notifications. The `allowed_ip_addresses` field specifies a list of IP addresses
        that the user can log in and perform operations from as described in [Allowing
        specific IP addresses for a
        user](https://cloud.ibm.com/docs/account?topic=account-ips). For information about
        the `self_manage` field, review information about the [user-managed login
        setting](https://cloud.ibm.com/docs/account?topic=account-types).

        :param str account_id: The account ID of the specified user.
        :param str iam_id: The user's IAM ID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `UserSettings` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not iam_id:
            raise ValueError('iam_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_user_settings',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id', 'iam_id']
        path_param_values = self.encode_path_vars(account_id, iam_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/accounts/{account_id}/users/{iam_id}/settings'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def update_user_settings(
        self,
        account_id: str,
        iam_id: str,
        *,
        language: str = None,
        notification_language: str = None,
        allowed_ip_addresses: str = None,
        self_manage: bool = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Partially update user settings.

        Update a user's settings by the user's IAM ID. You can use the IAM service token
        or a user token for authorization. To fully use this method, the user or service
        ID must have the editor or administrator role on the User Management service.
        Without these roles, a user can update only their own `language` or
        `notification_language` fields. If `self_manage` is `true`, the user can also
        update the `allowed_ip_addresses` field.

        :param str account_id: The account ID of the specified user.
        :param str iam_id: The user's IAM ID.
        :param str language: (optional) The console UI language. By default, this
               field is empty.
        :param str notification_language: (optional) The language for email and
               phone notifications. By default, this field is empty.
        :param str allowed_ip_addresses: (optional) A comma-separated list of IP
               addresses.
        :param bool self_manage: (optional) Whether user managed login is enabled.
               The default value is `false`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not iam_id:
            raise ValueError('iam_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='update_user_settings',
        )
        headers.update(sdk_headers)

        data = {
            'language': language,
            'notification_language': notification_language,
            'allowed_ip_addresses': allowed_ip_addresses,
            'self_manage': self_manage,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['account_id', 'iam_id']
        path_param_values = self.encode_path_vars(account_id, iam_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/accounts/{account_id}/users/{iam_id}/settings'.format(**path_param_dict)
        request = self.prepare_request(
            method='PATCH',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response


##############################################################################
# Models
##############################################################################


class InvitedUser:
    """
    Information about a user that has been invited to join an account.

    :attr str email: (optional) The email address associated with the invited user.
    :attr str id: (optional) The id associated with the invited user.
    :attr str state: (optional) The state of the invitation for the user.
    """

    def __init__(
        self,
        *,
        email: str = None,
        id: str = None,
        state: str = None,
    ) -> None:
        """
        Initialize a InvitedUser object.

        :param str email: (optional) The email address associated with the invited
               user.
        :param str id: (optional) The id associated with the invited user.
        :param str state: (optional) The state of the invitation for the user.
        """
        self.email = email
        self.id = id
        self.state = state

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'InvitedUser':
        """Initialize a InvitedUser object from a json dictionary."""
        args = {}
        if 'email' in _dict:
            args['email'] = _dict.get('email')
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'state' in _dict:
            args['state'] = _dict.get('state')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a InvitedUser object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'email') and self.email is not None:
            _dict['email'] = self.email
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this InvitedUser object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'InvitedUser') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'InvitedUser') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class InvitedUserList:
    """
    A collection of invited users.  This is the response returned by the invite_users
    operation.

    :attr List[InvitedUser] resources: (optional) The list of users that have been
          invited to join the account.
    """

    def __init__(
        self,
        *,
        resources: List['InvitedUser'] = None,
    ) -> None:
        """
        Initialize a InvitedUserList object.

        :param List[InvitedUser] resources: (optional) The list of users that have
               been invited to join the account.
        """
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'InvitedUserList':
        """Initialize a InvitedUserList object from a json dictionary."""
        args = {}
        if 'resources' in _dict:
            args['resources'] = [InvitedUser.from_dict(v) for v in _dict.get('resources')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a InvitedUserList object from a json dictionary."""
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
        """Return a `str` version of this InvitedUserList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'InvitedUserList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'InvitedUserList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class UserList:
    """
    The users returned.

    :attr int total_results: The number of users returned.
    :attr int limit: A limit to the number of users returned in a page.
    :attr str first_url: (optional) The first URL of the get users API.
    :attr str next_url: (optional) The next URL of the get users API.
    :attr List[UserProfile] resources: (optional) A list of users in the account.
    """

    def __init__(
        self,
        total_results: int,
        limit: int,
        *,
        first_url: str = None,
        next_url: str = None,
        resources: List['UserProfile'] = None,
    ) -> None:
        """
        Initialize a UserList object.

        :param int total_results: The number of users returned.
        :param int limit: A limit to the number of users returned in a page.
        :param str first_url: (optional) The first URL of the get users API.
        :param str next_url: (optional) The next URL of the get users API.
        :param List[UserProfile] resources: (optional) A list of users in the
               account.
        """
        self.total_results = total_results
        self.limit = limit
        self.first_url = first_url
        self.next_url = next_url
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'UserList':
        """Initialize a UserList object from a json dictionary."""
        args = {}
        if 'total_results' in _dict:
            args['total_results'] = _dict.get('total_results')
        else:
            raise ValueError('Required property \'total_results\' not present in UserList JSON')
        if 'limit' in _dict:
            args['limit'] = _dict.get('limit')
        else:
            raise ValueError('Required property \'limit\' not present in UserList JSON')
        if 'first_url' in _dict:
            args['first_url'] = _dict.get('first_url')
        if 'next_url' in _dict:
            args['next_url'] = _dict.get('next_url')
        if 'resources' in _dict:
            args['resources'] = [UserProfile.from_dict(v) for v in _dict.get('resources')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a UserList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'total_results') and self.total_results is not None:
            _dict['total_results'] = self.total_results
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'first_url') and self.first_url is not None:
            _dict['first_url'] = self.first_url
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
        """Return a `str` version of this UserList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'UserList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'UserList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class UserProfile:
    """
    Returned the user profile.

    :attr str id: (optional) An alphanumeric value identifying the user profile.
    :attr str iam_id: (optional) An alphanumeric value identifying the user's IAM
          ID.
    :attr str realm: (optional) The realm of the user. The value is either `IBMid`
          or `SL`.
    :attr str user_id: (optional) The user ID used for login.
    :attr str firstname: (optional) The first name of the user.
    :attr str lastname: (optional) The last name of the user.
    :attr str state: (optional) The state of the user. Possible values are
          `PROCESSING`, `PENDING`, `ACTIVE`, `DISABLED_CLASSIC_INFRASTRUCTURE`, and
          `VPN_ONLY`.
    :attr str email: (optional) The email address of the user.
    :attr str phonenumber: (optional) The phone number of the user.
    :attr str altphonenumber: (optional) The alternative phone number of the user.
    :attr str photo: (optional) A link to a photo of the user.
    :attr str account_id: (optional) An alphanumeric value identifying the account
          ID.
    :attr str added_on: (optional) The timestamp for when the user was added to the
          account.
    """

    def __init__(
        self,
        *,
        id: str = None,
        iam_id: str = None,
        realm: str = None,
        user_id: str = None,
        firstname: str = None,
        lastname: str = None,
        state: str = None,
        email: str = None,
        phonenumber: str = None,
        altphonenumber: str = None,
        photo: str = None,
        account_id: str = None,
        added_on: str = None,
    ) -> None:
        """
        Initialize a UserProfile object.

        :param str id: (optional) An alphanumeric value identifying the user
               profile.
        :param str iam_id: (optional) An alphanumeric value identifying the user's
               IAM ID.
        :param str realm: (optional) The realm of the user. The value is either
               `IBMid` or `SL`.
        :param str user_id: (optional) The user ID used for login.
        :param str firstname: (optional) The first name of the user.
        :param str lastname: (optional) The last name of the user.
        :param str state: (optional) The state of the user. Possible values are
               `PROCESSING`, `PENDING`, `ACTIVE`, `DISABLED_CLASSIC_INFRASTRUCTURE`, and
               `VPN_ONLY`.
        :param str email: (optional) The email address of the user.
        :param str phonenumber: (optional) The phone number of the user.
        :param str altphonenumber: (optional) The alternative phone number of the
               user.
        :param str photo: (optional) A link to a photo of the user.
        :param str account_id: (optional) An alphanumeric value identifying the
               account ID.
        :param str added_on: (optional) The timestamp for when the user was added
               to the account.
        """
        self.id = id
        self.iam_id = iam_id
        self.realm = realm
        self.user_id = user_id
        self.firstname = firstname
        self.lastname = lastname
        self.state = state
        self.email = email
        self.phonenumber = phonenumber
        self.altphonenumber = altphonenumber
        self.photo = photo
        self.account_id = account_id
        self.added_on = added_on

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'UserProfile':
        """Initialize a UserProfile object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'iam_id' in _dict:
            args['iam_id'] = _dict.get('iam_id')
        if 'realm' in _dict:
            args['realm'] = _dict.get('realm')
        if 'user_id' in _dict:
            args['user_id'] = _dict.get('user_id')
        if 'firstname' in _dict:
            args['firstname'] = _dict.get('firstname')
        if 'lastname' in _dict:
            args['lastname'] = _dict.get('lastname')
        if 'state' in _dict:
            args['state'] = _dict.get('state')
        if 'email' in _dict:
            args['email'] = _dict.get('email')
        if 'phonenumber' in _dict:
            args['phonenumber'] = _dict.get('phonenumber')
        if 'altphonenumber' in _dict:
            args['altphonenumber'] = _dict.get('altphonenumber')
        if 'photo' in _dict:
            args['photo'] = _dict.get('photo')
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        if 'added_on' in _dict:
            args['added_on'] = _dict.get('added_on')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a UserProfile object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'iam_id') and self.iam_id is not None:
            _dict['iam_id'] = self.iam_id
        if hasattr(self, 'realm') and self.realm is not None:
            _dict['realm'] = self.realm
        if hasattr(self, 'user_id') and self.user_id is not None:
            _dict['user_id'] = self.user_id
        if hasattr(self, 'firstname') and self.firstname is not None:
            _dict['firstname'] = self.firstname
        if hasattr(self, 'lastname') and self.lastname is not None:
            _dict['lastname'] = self.lastname
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        if hasattr(self, 'email') and self.email is not None:
            _dict['email'] = self.email
        if hasattr(self, 'phonenumber') and self.phonenumber is not None:
            _dict['phonenumber'] = self.phonenumber
        if hasattr(self, 'altphonenumber') and self.altphonenumber is not None:
            _dict['altphonenumber'] = self.altphonenumber
        if hasattr(self, 'photo') and self.photo is not None:
            _dict['photo'] = self.photo
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'added_on') and self.added_on is not None:
            _dict['added_on'] = self.added_on
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this UserProfile object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'UserProfile') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'UserProfile') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class UserSettings:
    """
    The user settings returned.

    :attr str language: (optional) The console UI language. By default, this field
          is empty.
    :attr str notification_language: (optional) The language for email and phone
          notifications. By default, this field is empty.
    :attr str allowed_ip_addresses: (optional) A comma-separated list of IP
          addresses.
    :attr bool self_manage: (optional) Whether user managed login is enabled. The
          default value is `false`.
    """

    def __init__(
        self,
        *,
        language: str = None,
        notification_language: str = None,
        allowed_ip_addresses: str = None,
        self_manage: bool = None,
    ) -> None:
        """
        Initialize a UserSettings object.

        :param str language: (optional) The console UI language. By default, this
               field is empty.
        :param str notification_language: (optional) The language for email and
               phone notifications. By default, this field is empty.
        :param str allowed_ip_addresses: (optional) A comma-separated list of IP
               addresses.
        :param bool self_manage: (optional) Whether user managed login is enabled.
               The default value is `false`.
        """
        self.language = language
        self.notification_language = notification_language
        self.allowed_ip_addresses = allowed_ip_addresses
        self.self_manage = self_manage

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'UserSettings':
        """Initialize a UserSettings object from a json dictionary."""
        args = {}
        if 'language' in _dict:
            args['language'] = _dict.get('language')
        if 'notification_language' in _dict:
            args['notification_language'] = _dict.get('notification_language')
        if 'allowed_ip_addresses' in _dict:
            args['allowed_ip_addresses'] = _dict.get('allowed_ip_addresses')
        if 'self_manage' in _dict:
            args['self_manage'] = _dict.get('self_manage')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a UserSettings object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'language') and self.language is not None:
            _dict['language'] = self.language
        if hasattr(self, 'notification_language') and self.notification_language is not None:
            _dict['notification_language'] = self.notification_language
        if hasattr(self, 'allowed_ip_addresses') and self.allowed_ip_addresses is not None:
            _dict['allowed_ip_addresses'] = self.allowed_ip_addresses
        if hasattr(self, 'self_manage') and self.self_manage is not None:
            _dict['self_manage'] = self.self_manage
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this UserSettings object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'UserSettings') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'UserSettings') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Attribute:
    """
    An attribute/value pair.

    :attr str name: (optional) The name of the attribute.
    :attr str value: (optional) The value of the attribute.
    """

    def __init__(
        self,
        *,
        name: str = None,
        value: str = None,
    ) -> None:
        """
        Initialize a Attribute object.

        :param str name: (optional) The name of the attribute.
        :param str value: (optional) The value of the attribute.
        """
        self.name = name
        self.value = value

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Attribute':
        """Initialize a Attribute object from a json dictionary."""
        args = {}
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'value' in _dict:
            args['value'] = _dict.get('value')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Attribute object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'value') and self.value is not None:
            _dict['value'] = self.value
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Attribute object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Attribute') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Attribute') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class InviteUser:
    """
    Invite a user.

    :attr str email: (optional) The email of the user to be invited.
    :attr str account_role: (optional) The account role of the user to be invited.
    """

    def __init__(
        self,
        *,
        email: str = None,
        account_role: str = None,
    ) -> None:
        """
        Initialize a InviteUser object.

        :param str email: (optional) The email of the user to be invited.
        :param str account_role: (optional) The account role of the user to be
               invited.
        """
        self.email = email
        self.account_role = account_role

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'InviteUser':
        """Initialize a InviteUser object from a json dictionary."""
        args = {}
        if 'email' in _dict:
            args['email'] = _dict.get('email')
        if 'account_role' in _dict:
            args['account_role'] = _dict.get('account_role')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a InviteUser object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'email') and self.email is not None:
            _dict['email'] = self.email
        if hasattr(self, 'account_role') and self.account_role is not None:
            _dict['account_role'] = self.account_role
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this InviteUser object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'InviteUser') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'InviteUser') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class InviteUserIamPolicy:
    """
    Invite a user to an IAM policy.

    :attr str type: The policy type. This can be either "access" or "authorization".
    :attr List[Role] roles: (optional) A list of IAM roles.
    :attr List[Resource] resources: (optional) A list of resources.
    """

    def __init__(
        self,
        type: str,
        *,
        roles: List['Role'] = None,
        resources: List['Resource'] = None,
    ) -> None:
        """
        Initialize a InviteUserIamPolicy object.

        :param str type: The policy type. This can be either "access" or
               "authorization".
        :param List[Role] roles: (optional) A list of IAM roles.
        :param List[Resource] resources: (optional) A list of resources.
        """
        self.type = type
        self.roles = roles
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'InviteUserIamPolicy':
        """Initialize a InviteUserIamPolicy object from a json dictionary."""
        args = {}
        if 'type' in _dict:
            args['type'] = _dict.get('type')
        else:
            raise ValueError('Required property \'type\' not present in InviteUserIamPolicy JSON')
        if 'roles' in _dict:
            args['roles'] = [Role.from_dict(v) for v in _dict.get('roles')]
        if 'resources' in _dict:
            args['resources'] = [Resource.from_dict(v) for v in _dict.get('resources')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a InviteUserIamPolicy object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'roles') and self.roles is not None:
            roles_list = []
            for v in self.roles:
                if isinstance(v, dict):
                    roles_list.append(v)
                else:
                    roles_list.append(v.to_dict())
            _dict['roles'] = roles_list
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
        """Return a `str` version of this InviteUserIamPolicy object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'InviteUserIamPolicy') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'InviteUserIamPolicy') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Resource:
    """
    A collection of attribute value pairs.

    :attr List[Attribute] attributes: (optional) A list of IAM attributes.
    """

    def __init__(
        self,
        *,
        attributes: List['Attribute'] = None,
    ) -> None:
        """
        Initialize a Resource object.

        :param List[Attribute] attributes: (optional) A list of IAM attributes.
        """
        self.attributes = attributes

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Resource':
        """Initialize a Resource object from a json dictionary."""
        args = {}
        if 'attributes' in _dict:
            args['attributes'] = [Attribute.from_dict(v) for v in _dict.get('attributes')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Resource object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'attributes') and self.attributes is not None:
            attributes_list = []
            for v in self.attributes:
                if isinstance(v, dict):
                    attributes_list.append(v)
                else:
                    attributes_list.append(v.to_dict())
            _dict['attributes'] = attributes_list
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


class Role:
    """
    The role of an IAM policy.

    :attr str role_id: (optional) An alphanumeric value identifying the origin.
    """

    def __init__(
        self,
        *,
        role_id: str = None,
    ) -> None:
        """
        Initialize a Role object.

        :param str role_id: (optional) An alphanumeric value identifying the
               origin.
        """
        self.role_id = role_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Role':
        """Initialize a Role object from a json dictionary."""
        args = {}
        if 'role_id' in _dict:
            args['role_id'] = _dict.get('role_id')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Role object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'role_id') and self.role_id is not None:
            _dict['role_id'] = self.role_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Role object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Role') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Role') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


##############################################################################
# Pagers
##############################################################################


class UsersPager:
    """
    UsersPager can be used to simplify the use of the "list_users" method.
    """

    def __init__(
        self,
        *,
        client: UserManagementV1,
        account_id: str,
        limit: int = None,
        include_settings: bool = None,
        search: str = None,
        user_id: str = None,
    ) -> None:
        """
        Initialize a UsersPager object.
        :param str account_id: The account ID of the specified user.
        :param int limit: (optional) The number of results to be returned.
        :param bool include_settings: (optional) The user settings to be returned.
               Set to true to view language, allowed IP address, and authentication
               settings.
        :param str search: (optional) The desired search results to be returned. To
               view the list of users with the additional search filter, use the following
               query options: `firstname`, `lastname`, `email`, `state`, `substate`,
               `iam_id`, `realm`, and `userId`. HTML URL encoding for the search query and
               `:` must be used. For example, search=state%3AINVALID returns a list of
               invalid users. Multiple search queries can be combined to obtain `OR`
               results using `,` operator (not URL encoded). For example,
               search=state%3AINVALID,email%3Amail.test.ibm.com.
        :param str user_id: (optional) Filter users based on their user ID.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._account_id = account_id
        self._limit = limit
        self._include_settings = include_settings
        self._search = search
        self._user_id = user_id

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of UserProfile.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.list_users(
            account_id=self._account_id,
            limit=self._limit,
            include_settings=self._include_settings,
            search=self._search,
            user_id=self._user_id,
            start=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next_url')
        if next_page_link is not None:
            next = get_query_param(next_page_link, '_start')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('resources')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of UserProfile.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results
