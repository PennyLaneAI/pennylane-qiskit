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
The Enterprise Management API enables you to create and manage an enterprise, account
groups, and accounts within the enterprise.

API Version: 1.0
"""

from datetime import datetime
from typing import Dict, List, Optional
import json

from ibm_cloud_sdk_core import BaseService, DetailedResponse, get_query_param
from ibm_cloud_sdk_core.authenticators.authenticator import Authenticator
from ibm_cloud_sdk_core.get_authenticator import get_authenticator_from_environment
from ibm_cloud_sdk_core.utils import convert_model, datetime_to_string, string_to_datetime

from .common import get_sdk_headers

##############################################################################
# Service
##############################################################################


class EnterpriseManagementV1(BaseService):
    """The Enterprise Management V1 service."""

    DEFAULT_SERVICE_URL = 'https://enterprise.cloud.ibm.com/v1'
    DEFAULT_SERVICE_NAME = 'enterprise_management'

    @classmethod
    def new_instance(
        cls,
        service_name: str = DEFAULT_SERVICE_NAME,
    ) -> 'EnterpriseManagementV1':
        """
        Return a new client for the Enterprise Management service using the
               specified parameters and external configuration.
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
        Construct a new client for the Enterprise Management service.

        :param Authenticator authenticator: The authenticator specifies the authentication mechanism.
               Get up to date information from https://github.com/IBM/python-sdk-core/blob/main/README.md
               about initializing the authenticator of your choice.
        """
        BaseService.__init__(self, service_url=self.DEFAULT_SERVICE_URL, authenticator=authenticator)

    #########################
    # Enterprise Operations
    #########################

    def create_enterprise(
        self,
        source_account_id: str,
        name: str,
        primary_contact_iam_id: str,
        *,
        domain: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create an enterprise.

        Create a new enterprise, which you can use to centrally manage multiple accounts.
        To create an enterprise, you must have an active Subscription account.
        <br/><br/>The API creates an enterprise entity, which is the root of the
        enterprise hierarchy. It also creates a new enterprise account that is used to
        manage the enterprise. All subscriptions, support entitlements, credits, and
        discounts from the source subscription account are migrated to the enterprise
        account, and the source account becomes a child account in the hierarchy. The user
        that you assign as the enterprise primary contact is also assigned as the owner of
        the enterprise account.

        :param str source_account_id: The ID of the account that is used to create
               the enterprise.
        :param str name: The name of the enterprise. This field must have 3 - 60
               characters.
        :param str primary_contact_iam_id: The IAM ID of the enterprise primary
               contact, such as `IBMid-0123ABC`. The IAM ID must already exist.
        :param str domain: (optional) A domain or subdomain for the enterprise,
               such as `example.com` or `my.example.com`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `CreateEnterpriseResponse` object
        """

        if source_account_id is None:
            raise ValueError('source_account_id must be provided')
        if name is None:
            raise ValueError('name must be provided')
        if primary_contact_iam_id is None:
            raise ValueError('primary_contact_iam_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='create_enterprise',
        )
        headers.update(sdk_headers)

        data = {
            'source_account_id': source_account_id,
            'name': name,
            'primary_contact_iam_id': primary_contact_iam_id,
            'domain': domain,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/enterprises'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def list_enterprises(
        self,
        *,
        enterprise_account_id: Optional[str] = None,
        account_group_id: Optional[str] = None,
        account_id: Optional[str] = None,
        next_docid: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List enterprises.

        Retrieve all enterprises for a given ID by passing the IDs on query parameters. If
        no ID is passed, the enterprises for which the calling identity is the primary
        contact are returned. You can use pagination parameters to filter the results.
        <br/><br/>This method ensures that only the enterprises that the user has access
        to are returned. Access can be controlled either through a policy on a specific
        enterprise, or account-level platform services access roles, such as
        Administrator, Editor, Operator, or Viewer. When you call the method with the
        `enterprise_account_id` or `account_id` query parameter, the account ID in the
        token is compared with that in the query parameter. If these account IDs match,
        authentication isn't performed and the enterprise information is returned. If the
        account IDs don't match, authentication is performed and only then is the
        enterprise information returned in the response.

        :param str enterprise_account_id: (optional) Get enterprises for a given
               enterprise account ID.
        :param str account_group_id: (optional) Get enterprises for a given account
               group ID.
        :param str account_id: (optional) Get enterprises for a given account ID.
        :param str next_docid: (optional) The first item to be returned in the page
               of results. This value can be obtained from the next_url property from the
               previous call of the operation. If not specified, then the first page of
               results is returned.
        :param int limit: (optional) Return results up to this limit. Valid values
               are between `0` and `100`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ListEnterprisesResponse` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='list_enterprises',
        )
        headers.update(sdk_headers)

        params = {
            'enterprise_account_id': enterprise_account_id,
            'account_group_id': account_group_id,
            'account_id': account_id,
            'next_docid': next_docid,
            'limit': limit,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/enterprises'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def get_enterprise(
        self,
        enterprise_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get enterprise by ID.

        Retrieve an enterprise by the `enterprise_id` parameter. All data related to the
        enterprise is returned only if the caller has access to retrieve the enterprise.

        :param str enterprise_id: The ID of the enterprise to retrieve.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Enterprise` object
        """

        if not enterprise_id:
            raise ValueError('enterprise_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_enterprise',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['enterprise_id']
        path_param_values = self.encode_path_vars(enterprise_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/enterprises/{enterprise_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def update_enterprise(
        self,
        enterprise_id: str,
        *,
        name: Optional[str] = None,
        domain: Optional[str] = None,
        primary_contact_iam_id: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Update an enterprise.

        Update the name, domain, or IAM ID of the primary contact for an existing
        enterprise. The new primary contact must already be a user in the enterprise
        account.

        :param str enterprise_id: The ID of the enterprise to retrieve.
        :param str name: (optional) The new name of the enterprise. This field must
               have 3 - 60 characters.
        :param str domain: (optional) The new domain of the enterprise. This field
               has a limit of 60 characters.
        :param str primary_contact_iam_id: (optional) The IAM ID of the user to be
               the new primary contact for the enterprise.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not enterprise_id:
            raise ValueError('enterprise_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='update_enterprise',
        )
        headers.update(sdk_headers)

        data = {
            'name': name,
            'domain': domain,
            'primary_contact_iam_id': primary_contact_iam_id,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['enterprise_id']
        path_param_values = self.encode_path_vars(enterprise_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/enterprises/{enterprise_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PATCH',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # Account Operations
    #########################

    def import_account_to_enterprise(
        self,
        enterprise_id: str,
        account_id: str,
        *,
        parent: Optional[str] = None,
        billing_unit_id: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Import an account into an enterprise.

        Import an existing stand-alone account into an enterprise. The existing account
        can be any type: trial (`TRIAL`), Lite (`STANDARD`), Pay-As-You-Go (`PAYG`), or
        Subscription (`SUBSCRIPTION`). In the case of a `SUBSCRIPTION` account, the
        credits, promotional offers, and discounts are migrated to the billing unit of the
        enterprise. For a billable account (`PAYG` or `SUBSCRIPTION`), the country and
        currency code of the existing account and the billing unit of the enterprise must
        match. The API returns a `202` response and performs asynchronous operations to
        import the account into the enterprise. <br/></br>For more information about
        impacts to the account, see [Adding accounts to an
        enterprise](https://{DomainName}/docs/account?topic=account-enterprise-add).

        :param str enterprise_id: The ID of the enterprise to import the
               stand-alone account into.
        :param str account_id: The ID of the existing stand-alone account to be
               imported.
        :param str parent: (optional) The CRN of the expected parent of the
               imported account. The parent is the enterprise or account group that the
               account is added to.
        :param str billing_unit_id: (optional) The ID of the [billing
               unit](/apidocs/enterprise-apis/billing-unit) to use for billing this
               account in the enterprise.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not enterprise_id:
            raise ValueError('enterprise_id must be provided')
        if not account_id:
            raise ValueError('account_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='import_account_to_enterprise',
        )
        headers.update(sdk_headers)

        data = {
            'parent': parent,
            'billing_unit_id': billing_unit_id,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['enterprise_id', 'account_id']
        path_param_values = self.encode_path_vars(enterprise_id, account_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/enterprises/{enterprise_id}/import/accounts/{account_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PUT',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def create_account(
        self,
        parent: str,
        name: str,
        owner_iam_id: str,
        *,
        traits: Optional['CreateAccountRequestTraits'] = None,
        options: Optional['CreateAccountRequestOptions'] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create a new account in an enterprise.

        Create a new account as a part of an existing enterprise. The API creates an
        account entity under the parent that is specified in the payload of the request.
        The request also takes in the name and the owner of this new account. The owner
        must have a valid IBMid that's registered with IBM Cloud, but they don't need to
        be a user in the enterprise account.

        :param str parent: The CRN of the parent under which the account will be
               created. The parent can be an existing account group or the enterprise
               itself.
        :param str name: The name of the account. This field must have 3 - 60
               characters.
        :param str owner_iam_id: The IAM ID of the account owner, such as
               `IBMid-0123ABC`. The IAM ID must already exist.
        :param CreateAccountRequestTraits traits: (optional) The traits object can
               be used to set properties on child accounts of an enterprise. You can pass
               a field to opt-out of the default multi-factor authentication setting or
               enable enterprise-managed IAM when creating a child account in the
               enterprise. This is an optional field.
        :param CreateAccountRequestOptions options: (optional) The options object
               can be used to set properties on child accounts of an enterprise. You can
               pass a field to to create IAM service id with IAM api keyg when creating a
               child account in the enterprise. This is an optional field.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `CreateAccountResponse` object
        """

        if parent is None:
            raise ValueError('parent must be provided')
        if name is None:
            raise ValueError('name must be provided')
        if owner_iam_id is None:
            raise ValueError('owner_iam_id must be provided')
        if traits is not None:
            traits = convert_model(traits)
        if options is not None:
            options = convert_model(options)
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='create_account',
        )
        headers.update(sdk_headers)

        data = {
            'parent': parent,
            'name': name,
            'owner_iam_id': owner_iam_id,
            'traits': traits,
            'options': options,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/accounts'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def list_accounts(
        self,
        *,
        enterprise_id: Optional[str] = None,
        account_group_id: Optional[str] = None,
        next_docid: Optional[str] = None,
        parent: Optional[str] = None,
        limit: Optional[int] = None,
        include_deleted: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List accounts.

        Retrieve all accounts based on the values that are passed in the query parameters.
        If no query parameter is passed, all of the accounts in the enterprise for which
        the calling identity has access are returned. <br/><br/>You can use pagination
        parameters to filter the results. The `limit` field can be used to limit the
        number of results that are displayed for this method.<br/><br/>This method ensures
        that only the accounts that the user has access to are returned. Access can be
        controlled either through a policy on a specific account, or account-level
        platform services access roles, such as Administrator, Editor, Operator, or
        Viewer. When you call the method with the `enterprise_id`, `account_group_id` or
        `parent` query parameter, all of the accounts that are immediate children of this
        entity are returned. Authentication is performed on all the accounts before they
        are returned to the user to ensure that only those accounts are returned to which
        the calling identity has access to.

        :param str enterprise_id: (optional) Get accounts that are either immediate
               children or are a part of the hierarchy for a given enterprise ID.
        :param str account_group_id: (optional) Get accounts that are either
               immediate children or are a part of the hierarchy for a given account group
               ID.
        :param str next_docid: (optional) The first item to be returned in the page
               of results. This value can be obtained from the next_url property from the
               previous call of the operation. If not specified, then the first page of
               results is returned.
        :param str parent: (optional) Get accounts that are either immediate
               children or are a part of the hierarchy for a given parent CRN.
        :param int limit: (optional) Return results up to this limit. Valid values
               are between `0` and `100`.
        :param bool include_deleted: (optional) Include the deleted accounts from
               an enterprise when used in conjunction with enterprise_id.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ListAccountsResponse` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='list_accounts',
        )
        headers.update(sdk_headers)

        params = {
            'enterprise_id': enterprise_id,
            'account_group_id': account_group_id,
            'next_docid': next_docid,
            'parent': parent,
            'limit': limit,
            'include_deleted': include_deleted,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/accounts'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def get_account(
        self,
        account_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get account by ID.

        Retrieve an account by the `account_id` parameter. All data related to the account
        is returned only if the caller has access to retrieve the account.

        :param str account_id: The ID of the target account.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Account` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_account',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id']
        path_param_values = self.encode_path_vars(account_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/accounts/{account_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def update_account(
        self,
        account_id: str,
        parent: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Move an account within the enterprise.

        Move an account to a different parent within the same enterprise.

        :param str account_id: The ID of the target account.
        :param str parent: The CRN of the new parent within the enterprise.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if parent is None:
            raise ValueError('parent must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='update_account',
        )
        headers.update(sdk_headers)

        data = {
            'parent': parent,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['account_id']
        path_param_values = self.encode_path_vars(account_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/accounts/{account_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PATCH',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_account(
        self,
        account_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Remove an account from its enterprise.

        Remove an account from the enterprise its currently in. After an account is
        removed, it will be canceled and cannot be reactivated.

        :param str account_id: The ID of the target account.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='delete_account',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['account_id']
        path_param_values = self.encode_path_vars(account_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/accounts/{account_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # Account Group Operations
    #########################

    def create_account_group(
        self,
        parent: str,
        name: str,
        primary_contact_iam_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create an account group.

        Create a new account group, which can be used to group together multiple accounts.
        To create an account group, you must have an existing enterprise. The API creates
        an account group entity under the parent that is specified in the payload of the
        request. The request also takes in the name and the primary contact of this new
        account group.

        :param str parent: The CRN of the parent under which the account group will
               be created. The parent can be an existing account group or the enterprise
               itself.
        :param str name: The name of the account group. This field must have 3 - 60
               characters.
        :param str primary_contact_iam_id: The IAM ID of the primary contact for
               this account group, such as `IBMid-0123ABC`. The IAM ID must already exist.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `CreateAccountGroupResponse` object
        """

        if parent is None:
            raise ValueError('parent must be provided')
        if name is None:
            raise ValueError('name must be provided')
        if primary_contact_iam_id is None:
            raise ValueError('primary_contact_iam_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='create_account_group',
        )
        headers.update(sdk_headers)

        data = {
            'parent': parent,
            'name': name,
            'primary_contact_iam_id': primary_contact_iam_id,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/account-groups'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def list_account_groups(
        self,
        *,
        enterprise_id: Optional[str] = None,
        parent_account_group_id: Optional[str] = None,
        next_docid: Optional[str] = None,
        parent: Optional[str] = None,
        limit: Optional[int] = None,
        include_deleted: Optional[bool] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List account groups.

        Retrieve all account groups based on the values that are passed in the query
        parameters. If no query parameter is passed, all of the account groups in the
        enterprise for which the calling identity has access are returned. <br/><br/>You
        can use pagination parameters to filter the results. The `limit` field can be used
        to limit the number of results that are displayed for this method.<br/><br/>This
        method ensures that only the account groups that the user has access to are
        returned. Access can be controlled either through a policy on a specific account
        group, or account-level platform services access roles, such as Administrator,
        Editor, Operator, or Viewer. When you call the method with the `enterprise_id`,
        `parent_account_group_id` or `parent` query parameter, all of the account groups
        that are immediate children of this entity are returned. Authentication is
        performed on all account groups before they are returned to the user to ensure
        that only those account groups are returned to which the calling identity has
        access.

        :param str enterprise_id: (optional) Get account groups that are either
               immediate children or are a part of the hierarchy for a given enterprise
               ID.
        :param str parent_account_group_id: (optional) Get account groups that are
               either immediate children or are a part of the hierarchy for a given
               account group ID.
        :param str next_docid: (optional) The first item to be returned in the page
               of results. This value can be obtained from the next_url property from the
               previous call of the operation. If not specified, then the first page of
               results is returned.
        :param str parent: (optional) Get account groups that are either immediate
               children or are a part of the hierarchy for a given parent CRN.
        :param int limit: (optional) Return results up to this limit. Valid values
               are between `0` and `100`.
        :param bool include_deleted: (optional) Include the deleted account groups
               from an enterprise when used in conjunction with other query parameters.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ListAccountGroupsResponse` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='list_account_groups',
        )
        headers.update(sdk_headers)

        params = {
            'enterprise_id': enterprise_id,
            'parent_account_group_id': parent_account_group_id,
            'next_docid': next_docid,
            'parent': parent,
            'limit': limit,
            'include_deleted': include_deleted,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/account-groups'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def get_account_group(
        self,
        account_group_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get account group by ID.

        Retrieve an account by the `account_group_id` parameter. All data related to the
        account group is returned only if the caller has access to retrieve the account
        group.

        :param str account_group_id: The ID of the account group to retrieve.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccountGroup` object
        """

        if not account_group_id:
            raise ValueError('account_group_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_account_group',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_group_id']
        path_param_values = self.encode_path_vars(account_group_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/account-groups/{account_group_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def update_account_group(
        self,
        account_group_id: str,
        *,
        name: Optional[str] = None,
        primary_contact_iam_id: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Update an account group.

        Update the name or IAM ID of the primary contact for an existing account group.
        The new primary contact must already be a user in the enterprise account.

        :param str account_group_id: The ID of the account group to retrieve.
        :param str name: (optional) The new name of the account group. This field
               must have 3 - 60 characters.
        :param str primary_contact_iam_id: (optional) The IAM ID of the user to be
               the new primary contact for the account group.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not account_group_id:
            raise ValueError('account_group_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='update_account_group',
        )
        headers.update(sdk_headers)

        data = {
            'name': name,
            'primary_contact_iam_id': primary_contact_iam_id,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['account_group_id']
        path_param_values = self.encode_path_vars(account_group_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/account-groups/{account_group_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='PATCH',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_account_group(
        self,
        account_group_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete an account group from the enterprise.

        Delete an existing account group from the enterprise. You can't delete an account
        group that has child account groups, the delete request will fail. This API
        doesn't perform a recursive delete on the child account groups, it only deletes
        the current account group.

        :param str account_group_id: The ID of the account group to retrieve.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not account_group_id:
            raise ValueError('account_group_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='delete_account_group',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        path_param_keys = ['account_group_id']
        path_param_values = self.encode_path_vars(account_group_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/account-groups/{account_group_id}'.format(**path_param_dict)
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response


##############################################################################
# Models
##############################################################################


class Account:
    """
    An account resource.

    :param str url: (optional) The URL of the account.
    :param str id: (optional) The account ID.
    :param str crn: (optional) The Cloud Resource Name (CRN) of the account.
    :param str parent: (optional) The CRN of the parent of the account.
    :param str enterprise_account_id: (optional) The enterprise account ID.
    :param str enterprise_id: (optional) The enterprise ID that the account is a
          part of.
    :param str enterprise_path: (optional) The path from the enterprise to this
          particular account.
    :param str name: (optional) The name of the account.
    :param str state: (optional) The state of the account.
    :param str owner_iam_id: (optional) The IAM ID of the owner of the account.
    :param bool paid: (optional) The type of account - whether it is free or paid.
    :param str owner_email: (optional) The email address of the owner of the
          account.
    :param bool is_enterprise_account: (optional) The flag to indicate whether the
          account is an enterprise account or not.
    :param datetime created_at: (optional) The time stamp at which the account was
          created.
    :param str created_by: (optional) The IAM ID of the user or service that created
          the account.
    :param datetime updated_at: (optional) The time stamp at which the account was
          last updated.
    :param str updated_by: (optional) The IAM ID of the user or service that updated
          the account.
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        id: Optional[str] = None,
        crn: Optional[str] = None,
        parent: Optional[str] = None,
        enterprise_account_id: Optional[str] = None,
        enterprise_id: Optional[str] = None,
        enterprise_path: Optional[str] = None,
        name: Optional[str] = None,
        state: Optional[str] = None,
        owner_iam_id: Optional[str] = None,
        paid: Optional[bool] = None,
        owner_email: Optional[str] = None,
        is_enterprise_account: Optional[bool] = None,
        created_at: Optional[datetime] = None,
        created_by: Optional[str] = None,
        updated_at: Optional[datetime] = None,
        updated_by: Optional[str] = None,
    ) -> None:
        """
        Initialize a Account object.

        :param str url: (optional) The URL of the account.
        :param str id: (optional) The account ID.
        :param str crn: (optional) The Cloud Resource Name (CRN) of the account.
        :param str parent: (optional) The CRN of the parent of the account.
        :param str enterprise_account_id: (optional) The enterprise account ID.
        :param str enterprise_id: (optional) The enterprise ID that the account is
               a part of.
        :param str enterprise_path: (optional) The path from the enterprise to this
               particular account.
        :param str name: (optional) The name of the account.
        :param str state: (optional) The state of the account.
        :param str owner_iam_id: (optional) The IAM ID of the owner of the account.
        :param bool paid: (optional) The type of account - whether it is free or
               paid.
        :param str owner_email: (optional) The email address of the owner of the
               account.
        :param bool is_enterprise_account: (optional) The flag to indicate whether
               the account is an enterprise account or not.
        :param datetime created_at: (optional) The time stamp at which the account
               was created.
        :param str created_by: (optional) The IAM ID of the user or service that
               created the account.
        :param datetime updated_at: (optional) The time stamp at which the account
               was last updated.
        :param str updated_by: (optional) The IAM ID of the user or service that
               updated the account.
        """
        self.url = url
        self.id = id
        self.crn = crn
        self.parent = parent
        self.enterprise_account_id = enterprise_account_id
        self.enterprise_id = enterprise_id
        self.enterprise_path = enterprise_path
        self.name = name
        self.state = state
        self.owner_iam_id = owner_iam_id
        self.paid = paid
        self.owner_email = owner_email
        self.is_enterprise_account = is_enterprise_account
        self.created_at = created_at
        self.created_by = created_by
        self.updated_at = updated_at
        self.updated_by = updated_by

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Account':
        """Initialize a Account object from a json dictionary."""
        args = {}
        if (url := _dict.get('url')) is not None:
            args['url'] = url
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        if (crn := _dict.get('crn')) is not None:
            args['crn'] = crn
        if (parent := _dict.get('parent')) is not None:
            args['parent'] = parent
        if (enterprise_account_id := _dict.get('enterprise_account_id')) is not None:
            args['enterprise_account_id'] = enterprise_account_id
        if (enterprise_id := _dict.get('enterprise_id')) is not None:
            args['enterprise_id'] = enterprise_id
        if (enterprise_path := _dict.get('enterprise_path')) is not None:
            args['enterprise_path'] = enterprise_path
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        if (state := _dict.get('state')) is not None:
            args['state'] = state
        if (owner_iam_id := _dict.get('owner_iam_id')) is not None:
            args['owner_iam_id'] = owner_iam_id
        if (paid := _dict.get('paid')) is not None:
            args['paid'] = paid
        if (owner_email := _dict.get('owner_email')) is not None:
            args['owner_email'] = owner_email
        if (is_enterprise_account := _dict.get('is_enterprise_account')) is not None:
            args['is_enterprise_account'] = is_enterprise_account
        if (created_at := _dict.get('created_at')) is not None:
            args['created_at'] = string_to_datetime(created_at)
        if (created_by := _dict.get('created_by')) is not None:
            args['created_by'] = created_by
        if (updated_at := _dict.get('updated_at')) is not None:
            args['updated_at'] = string_to_datetime(updated_at)
        if (updated_by := _dict.get('updated_by')) is not None:
            args['updated_by'] = updated_by
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Account object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'url') and self.url is not None:
            _dict['url'] = self.url
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'parent') and self.parent is not None:
            _dict['parent'] = self.parent
        if hasattr(self, 'enterprise_account_id') and self.enterprise_account_id is not None:
            _dict['enterprise_account_id'] = self.enterprise_account_id
        if hasattr(self, 'enterprise_id') and self.enterprise_id is not None:
            _dict['enterprise_id'] = self.enterprise_id
        if hasattr(self, 'enterprise_path') and self.enterprise_path is not None:
            _dict['enterprise_path'] = self.enterprise_path
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        if hasattr(self, 'owner_iam_id') and self.owner_iam_id is not None:
            _dict['owner_iam_id'] = self.owner_iam_id
        if hasattr(self, 'paid') and self.paid is not None:
            _dict['paid'] = self.paid
        if hasattr(self, 'owner_email') and self.owner_email is not None:
            _dict['owner_email'] = self.owner_email
        if hasattr(self, 'is_enterprise_account') and self.is_enterprise_account is not None:
            _dict['is_enterprise_account'] = self.is_enterprise_account
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
        """Return a `str` version of this Account object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Account') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Account') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AccountGroup:
    """
    An account group resource.

    :param str url: (optional) The URL of the account group.
    :param str id: (optional) The account group ID.
    :param str crn: (optional) The Cloud Resource Name (CRN) of the account group.
    :param str parent: (optional) The CRN of the parent of the account group.
    :param str enterprise_account_id: (optional) The enterprise account ID.
    :param str enterprise_id: (optional) The enterprise ID that the account group is
          a part of.
    :param str enterprise_path: (optional) The path from the enterprise to this
          particular account group.
    :param str name: (optional) The name of the account group.
    :param str state: (optional) The state of the account group.
    :param str primary_contact_iam_id: (optional) The IAM ID of the primary contact
          of the account group.
    :param str primary_contact_email: (optional) The email address of the primary
          contact of the account group.
    :param datetime created_at: (optional) The time stamp at which the account group
          was created.
    :param str created_by: (optional) The IAM ID of the user or service that created
          the account group.
    :param datetime updated_at: (optional) The time stamp at which the account group
          was last updated.
    :param str updated_by: (optional) The IAM ID of the user or service that updated
          the account group.
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        id: Optional[str] = None,
        crn: Optional[str] = None,
        parent: Optional[str] = None,
        enterprise_account_id: Optional[str] = None,
        enterprise_id: Optional[str] = None,
        enterprise_path: Optional[str] = None,
        name: Optional[str] = None,
        state: Optional[str] = None,
        primary_contact_iam_id: Optional[str] = None,
        primary_contact_email: Optional[str] = None,
        created_at: Optional[datetime] = None,
        created_by: Optional[str] = None,
        updated_at: Optional[datetime] = None,
        updated_by: Optional[str] = None,
    ) -> None:
        """
        Initialize a AccountGroup object.

        :param str url: (optional) The URL of the account group.
        :param str id: (optional) The account group ID.
        :param str crn: (optional) The Cloud Resource Name (CRN) of the account
               group.
        :param str parent: (optional) The CRN of the parent of the account group.
        :param str enterprise_account_id: (optional) The enterprise account ID.
        :param str enterprise_id: (optional) The enterprise ID that the account
               group is a part of.
        :param str enterprise_path: (optional) The path from the enterprise to this
               particular account group.
        :param str name: (optional) The name of the account group.
        :param str state: (optional) The state of the account group.
        :param str primary_contact_iam_id: (optional) The IAM ID of the primary
               contact of the account group.
        :param str primary_contact_email: (optional) The email address of the
               primary contact of the account group.
        :param datetime created_at: (optional) The time stamp at which the account
               group was created.
        :param str created_by: (optional) The IAM ID of the user or service that
               created the account group.
        :param datetime updated_at: (optional) The time stamp at which the account
               group was last updated.
        :param str updated_by: (optional) The IAM ID of the user or service that
               updated the account group.
        """
        self.url = url
        self.id = id
        self.crn = crn
        self.parent = parent
        self.enterprise_account_id = enterprise_account_id
        self.enterprise_id = enterprise_id
        self.enterprise_path = enterprise_path
        self.name = name
        self.state = state
        self.primary_contact_iam_id = primary_contact_iam_id
        self.primary_contact_email = primary_contact_email
        self.created_at = created_at
        self.created_by = created_by
        self.updated_at = updated_at
        self.updated_by = updated_by

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccountGroup':
        """Initialize a AccountGroup object from a json dictionary."""
        args = {}
        if (url := _dict.get('url')) is not None:
            args['url'] = url
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        if (crn := _dict.get('crn')) is not None:
            args['crn'] = crn
        if (parent := _dict.get('parent')) is not None:
            args['parent'] = parent
        if (enterprise_account_id := _dict.get('enterprise_account_id')) is not None:
            args['enterprise_account_id'] = enterprise_account_id
        if (enterprise_id := _dict.get('enterprise_id')) is not None:
            args['enterprise_id'] = enterprise_id
        if (enterprise_path := _dict.get('enterprise_path')) is not None:
            args['enterprise_path'] = enterprise_path
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        if (state := _dict.get('state')) is not None:
            args['state'] = state
        if (primary_contact_iam_id := _dict.get('primary_contact_iam_id')) is not None:
            args['primary_contact_iam_id'] = primary_contact_iam_id
        if (primary_contact_email := _dict.get('primary_contact_email')) is not None:
            args['primary_contact_email'] = primary_contact_email
        if (created_at := _dict.get('created_at')) is not None:
            args['created_at'] = string_to_datetime(created_at)
        if (created_by := _dict.get('created_by')) is not None:
            args['created_by'] = created_by
        if (updated_at := _dict.get('updated_at')) is not None:
            args['updated_at'] = string_to_datetime(updated_at)
        if (updated_by := _dict.get('updated_by')) is not None:
            args['updated_by'] = updated_by
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccountGroup object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'url') and self.url is not None:
            _dict['url'] = self.url
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'parent') and self.parent is not None:
            _dict['parent'] = self.parent
        if hasattr(self, 'enterprise_account_id') and self.enterprise_account_id is not None:
            _dict['enterprise_account_id'] = self.enterprise_account_id
        if hasattr(self, 'enterprise_id') and self.enterprise_id is not None:
            _dict['enterprise_id'] = self.enterprise_id
        if hasattr(self, 'enterprise_path') and self.enterprise_path is not None:
            _dict['enterprise_path'] = self.enterprise_path
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        if hasattr(self, 'primary_contact_iam_id') and self.primary_contact_iam_id is not None:
            _dict['primary_contact_iam_id'] = self.primary_contact_iam_id
        if hasattr(self, 'primary_contact_email') and self.primary_contact_email is not None:
            _dict['primary_contact_email'] = self.primary_contact_email
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
        """Return a `str` version of this AccountGroup object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccountGroup') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccountGroup') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class CreateAccountGroupResponse:
    """
    A newly-created account group.

    :param str account_group_id: (optional) The ID of the account group entity that
          was created.
    """

    def __init__(
        self,
        *,
        account_group_id: Optional[str] = None,
    ) -> None:
        """
        Initialize a CreateAccountGroupResponse object.

        :param str account_group_id: (optional) The ID of the account group entity
               that was created.
        """
        self.account_group_id = account_group_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CreateAccountGroupResponse':
        """Initialize a CreateAccountGroupResponse object from a json dictionary."""
        args = {}
        if (account_group_id := _dict.get('account_group_id')) is not None:
            args['account_group_id'] = account_group_id
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CreateAccountGroupResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'account_group_id') and self.account_group_id is not None:
            _dict['account_group_id'] = self.account_group_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this CreateAccountGroupResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CreateAccountGroupResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CreateAccountGroupResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class CreateAccountRequestOptions:
    """
    The options object can be used to set properties on child accounts of an enterprise.
    You can pass a field to to create IAM service id with IAM api keyg when creating a
    child account in the enterprise. This is an optional field.

    :param bool create_iam_service_id_with_apikey_and_owner_policies: (optional) By
          default create_iam_service_id_with_apikey_and_owner_policies is turned off for a
          newly created child account. You can enable this property by passing 'true' in
          this boolean field. IAM service id has account owner IAM policies and the API
          key associated with it can generate a token and setup resources in the account.
          This is an optional field.
    """

    def __init__(
        self,
        *,
        create_iam_service_id_with_apikey_and_owner_policies: Optional[bool] = None,
    ) -> None:
        """
        Initialize a CreateAccountRequestOptions object.

        :param bool create_iam_service_id_with_apikey_and_owner_policies:
               (optional) By default create_iam_service_id_with_apikey_and_owner_policies
               is turned off for a newly created child account. You can enable this
               property by passing 'true' in this boolean field. IAM service id has
               account owner IAM policies and the API key associated with it can generate
               a token and setup resources in the account. This is an optional field.
        """
        self.create_iam_service_id_with_apikey_and_owner_policies = create_iam_service_id_with_apikey_and_owner_policies

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CreateAccountRequestOptions':
        """Initialize a CreateAccountRequestOptions object from a json dictionary."""
        args = {}
        if (
            create_iam_service_id_with_apikey_and_owner_policies := _dict.get(
                'create_iam_service_id_with_apikey_and_owner_policies'
            )
        ) is not None:
            args['create_iam_service_id_with_apikey_and_owner_policies'] = (
                create_iam_service_id_with_apikey_and_owner_policies
            )
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CreateAccountRequestOptions object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if (
            hasattr(self, 'create_iam_service_id_with_apikey_and_owner_policies')
            and self.create_iam_service_id_with_apikey_and_owner_policies is not None
        ):
            _dict['create_iam_service_id_with_apikey_and_owner_policies'] = (
                self.create_iam_service_id_with_apikey_and_owner_policies
            )
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this CreateAccountRequestOptions object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CreateAccountRequestOptions') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CreateAccountRequestOptions') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class CreateAccountRequestTraits:
    """
    The traits object can be used to set properties on child accounts of an enterprise.
    You can pass a field to opt-out of the default multi-factor authentication setting or
    enable enterprise-managed IAM when creating a child account in the enterprise. This is
    an optional field.

    :param str mfa: (optional) By default MFA is set to `NONE_NO_ROPC` on a child
          account, which disables CLI logins with only a password. To opt out, pass the
          traits object with the mfa field set to empty string. This is an optional field.
    :param bool enterprise_iam_managed: (optional) By default enterprise-managed IAM
          is turned off for a newly created child account. You can enable this property by
          passing 'true' in this boolean field. Enabling enterprise-managed IAM allows the
          enterprise account to assign IAM resources, like access groups, trusted
          profiles, and account settings, to the child account. This is an optional field.
    """

    def __init__(
        self,
        *,
        mfa: Optional[str] = None,
        enterprise_iam_managed: Optional[bool] = None,
    ) -> None:
        """
        Initialize a CreateAccountRequestTraits object.

        :param str mfa: (optional) By default MFA is set to `NONE_NO_ROPC` on a
               child account, which disables CLI logins with only a password. To opt out,
               pass the traits object with the mfa field set to empty string. This is an
               optional field.
        :param bool enterprise_iam_managed: (optional) By default
               enterprise-managed IAM is turned off for a newly created child account. You
               can enable this property by passing 'true' in this boolean field. Enabling
               enterprise-managed IAM allows the enterprise account to assign IAM
               resources, like access groups, trusted profiles, and account settings, to
               the child account. This is an optional field.
        """
        self.mfa = mfa
        self.enterprise_iam_managed = enterprise_iam_managed

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CreateAccountRequestTraits':
        """Initialize a CreateAccountRequestTraits object from a json dictionary."""
        args = {}
        if (mfa := _dict.get('mfa')) is not None:
            args['mfa'] = mfa
        if (enterprise_iam_managed := _dict.get('enterprise_iam_managed')) is not None:
            args['enterprise_iam_managed'] = enterprise_iam_managed
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CreateAccountRequestTraits object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'mfa') and self.mfa is not None:
            _dict['mfa'] = self.mfa
        if hasattr(self, 'enterprise_iam_managed') and self.enterprise_iam_managed is not None:
            _dict['enterprise_iam_managed'] = self.enterprise_iam_managed
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this CreateAccountRequestTraits object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CreateAccountRequestTraits') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CreateAccountRequestTraits') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class CreateAccountResponse:
    """
    A newly-created account.

    :param str account_id: (optional) The ID of the account entity that was created.
    """

    def __init__(
        self,
        *,
        account_id: Optional[str] = None,
    ) -> None:
        """
        Initialize a CreateAccountResponse object.

        :param str account_id: (optional) The ID of the account entity that was
               created.
        """
        self.account_id = account_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CreateAccountResponse':
        """Initialize a CreateAccountResponse object from a json dictionary."""
        args = {}
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CreateAccountResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this CreateAccountResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CreateAccountResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CreateAccountResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class CreateEnterpriseResponse:
    """
    The response from calling create enterprise.

    :param str enterprise_id: (optional) The ID of the enterprise entity that was
          created. This entity is the root of the hierarchy.
    :param str enterprise_account_id: (optional) The ID of the enterprise account
          that was created. The enterprise account is used to manage billing and access to
          the enterprise management.
    """

    def __init__(
        self,
        *,
        enterprise_id: Optional[str] = None,
        enterprise_account_id: Optional[str] = None,
    ) -> None:
        """
        Initialize a CreateEnterpriseResponse object.

        :param str enterprise_id: (optional) The ID of the enterprise entity that
               was created. This entity is the root of the hierarchy.
        :param str enterprise_account_id: (optional) The ID of the enterprise
               account that was created. The enterprise account is used to manage billing
               and access to the enterprise management.
        """
        self.enterprise_id = enterprise_id
        self.enterprise_account_id = enterprise_account_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CreateEnterpriseResponse':
        """Initialize a CreateEnterpriseResponse object from a json dictionary."""
        args = {}
        if (enterprise_id := _dict.get('enterprise_id')) is not None:
            args['enterprise_id'] = enterprise_id
        if (enterprise_account_id := _dict.get('enterprise_account_id')) is not None:
            args['enterprise_account_id'] = enterprise_account_id
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CreateEnterpriseResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'enterprise_id') and self.enterprise_id is not None:
            _dict['enterprise_id'] = self.enterprise_id
        if hasattr(self, 'enterprise_account_id') and self.enterprise_account_id is not None:
            _dict['enterprise_account_id'] = self.enterprise_account_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this CreateEnterpriseResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CreateEnterpriseResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CreateEnterpriseResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Enterprise:
    """
    An enterprise resource.

    :param str url: (optional) The URL of the enterprise.
    :param str id: (optional) The enterprise ID.
    :param str enterprise_account_id: (optional) The enterprise account ID.
    :param str crn: (optional) The Cloud Resource Name (CRN) of the enterprise.
    :param str name: (optional) The name of the enterprise.
    :param str domain: (optional) The domain of the enterprise.
    :param str state: (optional) The state of the enterprise.
    :param str primary_contact_iam_id: (optional) The IAM ID of the primary contact
          of the enterprise, such as `IBMid-0123ABC`.
    :param str primary_contact_email: (optional) The email of the primary contact of
          the enterprise.
    :param str source_account_id: (optional) The ID of the account that is used to
          create the enterprise.
    :param datetime created_at: (optional) The time stamp at which the enterprise
          was created.
    :param str created_by: (optional) The IAM ID of the user or service that created
          the enterprise.
    :param datetime updated_at: (optional) The time stamp at which the enterprise
          was last updated.
    :param str updated_by: (optional) The IAM ID of the user or service that updated
          the enterprise.
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        id: Optional[str] = None,
        enterprise_account_id: Optional[str] = None,
        crn: Optional[str] = None,
        name: Optional[str] = None,
        domain: Optional[str] = None,
        state: Optional[str] = None,
        primary_contact_iam_id: Optional[str] = None,
        primary_contact_email: Optional[str] = None,
        source_account_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        created_by: Optional[str] = None,
        updated_at: Optional[datetime] = None,
        updated_by: Optional[str] = None,
    ) -> None:
        """
        Initialize a Enterprise object.

        :param str url: (optional) The URL of the enterprise.
        :param str id: (optional) The enterprise ID.
        :param str enterprise_account_id: (optional) The enterprise account ID.
        :param str crn: (optional) The Cloud Resource Name (CRN) of the enterprise.
        :param str name: (optional) The name of the enterprise.
        :param str domain: (optional) The domain of the enterprise.
        :param str state: (optional) The state of the enterprise.
        :param str primary_contact_iam_id: (optional) The IAM ID of the primary
               contact of the enterprise, such as `IBMid-0123ABC`.
        :param str primary_contact_email: (optional) The email of the primary
               contact of the enterprise.
        :param str source_account_id: (optional) The ID of the account that is used
               to create the enterprise.
        :param datetime created_at: (optional) The time stamp at which the
               enterprise was created.
        :param str created_by: (optional) The IAM ID of the user or service that
               created the enterprise.
        :param datetime updated_at: (optional) The time stamp at which the
               enterprise was last updated.
        :param str updated_by: (optional) The IAM ID of the user or service that
               updated the enterprise.
        """
        self.url = url
        self.id = id
        self.enterprise_account_id = enterprise_account_id
        self.crn = crn
        self.name = name
        self.domain = domain
        self.state = state
        self.primary_contact_iam_id = primary_contact_iam_id
        self.primary_contact_email = primary_contact_email
        self.source_account_id = source_account_id
        self.created_at = created_at
        self.created_by = created_by
        self.updated_at = updated_at
        self.updated_by = updated_by

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Enterprise':
        """Initialize a Enterprise object from a json dictionary."""
        args = {}
        if (url := _dict.get('url')) is not None:
            args['url'] = url
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        if (enterprise_account_id := _dict.get('enterprise_account_id')) is not None:
            args['enterprise_account_id'] = enterprise_account_id
        if (crn := _dict.get('crn')) is not None:
            args['crn'] = crn
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        if (domain := _dict.get('domain')) is not None:
            args['domain'] = domain
        if (state := _dict.get('state')) is not None:
            args['state'] = state
        if (primary_contact_iam_id := _dict.get('primary_contact_iam_id')) is not None:
            args['primary_contact_iam_id'] = primary_contact_iam_id
        if (primary_contact_email := _dict.get('primary_contact_email')) is not None:
            args['primary_contact_email'] = primary_contact_email
        if (source_account_id := _dict.get('source_account_id')) is not None:
            args['source_account_id'] = source_account_id
        if (created_at := _dict.get('created_at')) is not None:
            args['created_at'] = string_to_datetime(created_at)
        if (created_by := _dict.get('created_by')) is not None:
            args['created_by'] = created_by
        if (updated_at := _dict.get('updated_at')) is not None:
            args['updated_at'] = string_to_datetime(updated_at)
        if (updated_by := _dict.get('updated_by')) is not None:
            args['updated_by'] = updated_by
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Enterprise object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'url') and self.url is not None:
            _dict['url'] = self.url
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'enterprise_account_id') and self.enterprise_account_id is not None:
            _dict['enterprise_account_id'] = self.enterprise_account_id
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'domain') and self.domain is not None:
            _dict['domain'] = self.domain
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        if hasattr(self, 'primary_contact_iam_id') and self.primary_contact_iam_id is not None:
            _dict['primary_contact_iam_id'] = self.primary_contact_iam_id
        if hasattr(self, 'primary_contact_email') and self.primary_contact_email is not None:
            _dict['primary_contact_email'] = self.primary_contact_email
        if hasattr(self, 'source_account_id') and self.source_account_id is not None:
            _dict['source_account_id'] = self.source_account_id
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
        """Return a `str` version of this Enterprise object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Enterprise') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Enterprise') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ListAccountGroupsResponse:
    """
    The list_account_groups operation response.

    :param int rows_count: (optional) The number of enterprises returned from
          calling list account groups.
    :param str next_url: (optional) A string that represents the link to the next
          page of results.
    :param List[AccountGroup] resources: (optional) A list of account groups.
    """

    def __init__(
        self,
        *,
        rows_count: Optional[int] = None,
        next_url: Optional[str] = None,
        resources: Optional[List['AccountGroup']] = None,
    ) -> None:
        """
        Initialize a ListAccountGroupsResponse object.

        :param int rows_count: (optional) The number of enterprises returned from
               calling list account groups.
        :param str next_url: (optional) A string that represents the link to the
               next page of results.
        :param List[AccountGroup] resources: (optional) A list of account groups.
        """
        self.rows_count = rows_count
        self.next_url = next_url
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ListAccountGroupsResponse':
        """Initialize a ListAccountGroupsResponse object from a json dictionary."""
        args = {}
        if (rows_count := _dict.get('rows_count')) is not None:
            args['rows_count'] = rows_count
        if (next_url := _dict.get('next_url')) is not None:
            args['next_url'] = next_url
        if (resources := _dict.get('resources')) is not None:
            args['resources'] = [AccountGroup.from_dict(v) for v in resources]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ListAccountGroupsResponse object from a json dictionary."""
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
        """Return a `str` version of this ListAccountGroupsResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ListAccountGroupsResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ListAccountGroupsResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ListAccountsResponse:
    """
    The list_accounts operation response.

    :param int rows_count: (optional) The number of enterprises returned from
          calling list accounts.
    :param str next_url: (optional) A string that represents the link to the next
          page of results.
    :param List[Account] resources: (optional) A list of accounts.
    """

    def __init__(
        self,
        *,
        rows_count: Optional[int] = None,
        next_url: Optional[str] = None,
        resources: Optional[List['Account']] = None,
    ) -> None:
        """
        Initialize a ListAccountsResponse object.

        :param int rows_count: (optional) The number of enterprises returned from
               calling list accounts.
        :param str next_url: (optional) A string that represents the link to the
               next page of results.
        :param List[Account] resources: (optional) A list of accounts.
        """
        self.rows_count = rows_count
        self.next_url = next_url
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ListAccountsResponse':
        """Initialize a ListAccountsResponse object from a json dictionary."""
        args = {}
        if (rows_count := _dict.get('rows_count')) is not None:
            args['rows_count'] = rows_count
        if (next_url := _dict.get('next_url')) is not None:
            args['next_url'] = next_url
        if (resources := _dict.get('resources')) is not None:
            args['resources'] = [Account.from_dict(v) for v in resources]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ListAccountsResponse object from a json dictionary."""
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
        """Return a `str` version of this ListAccountsResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ListAccountsResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ListAccountsResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ListEnterprisesResponse:
    """
    The response from calling list enterprises.

    :param int rows_count: (optional) The number of enterprises returned from
          calling list enterprise.
    :param str next_url: (optional) A string that represents the link to the next
          page of results.
    :param List[Enterprise] resources: (optional) A list of enterprise objects.
    """

    def __init__(
        self,
        *,
        rows_count: Optional[int] = None,
        next_url: Optional[str] = None,
        resources: Optional[List['Enterprise']] = None,
    ) -> None:
        """
        Initialize a ListEnterprisesResponse object.

        :param int rows_count: (optional) The number of enterprises returned from
               calling list enterprise.
        :param str next_url: (optional) A string that represents the link to the
               next page of results.
        :param List[Enterprise] resources: (optional) A list of enterprise objects.
        """
        self.rows_count = rows_count
        self.next_url = next_url
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ListEnterprisesResponse':
        """Initialize a ListEnterprisesResponse object from a json dictionary."""
        args = {}
        if (rows_count := _dict.get('rows_count')) is not None:
            args['rows_count'] = rows_count
        if (next_url := _dict.get('next_url')) is not None:
            args['next_url'] = next_url
        if (resources := _dict.get('resources')) is not None:
            args['resources'] = [Enterprise.from_dict(v) for v in resources]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ListEnterprisesResponse object from a json dictionary."""
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
        """Return a `str` version of this ListEnterprisesResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ListEnterprisesResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ListEnterprisesResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


##############################################################################
# Pagers
##############################################################################


class EnterprisesPager:
    """
    EnterprisesPager can be used to simplify the use of the "list_enterprises" method.
    """

    def __init__(
        self,
        *,
        client: EnterpriseManagementV1,
        enterprise_account_id: str = None,
        account_group_id: str = None,
        account_id: str = None,
        limit: int = None,
    ) -> None:
        """
        Initialize a EnterprisesPager object.
        :param str enterprise_account_id: (optional) Get enterprises for a given
               enterprise account ID.
        :param str account_group_id: (optional) Get enterprises for a given account
               group ID.
        :param str account_id: (optional) Get enterprises for a given account ID.
        :param int limit: (optional) Return results up to this limit. Valid values
               are between `0` and `100`.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._enterprise_account_id = enterprise_account_id
        self._account_group_id = account_group_id
        self._account_id = account_id
        self._limit = limit

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of Enterprise.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.list_enterprises(
            enterprise_account_id=self._enterprise_account_id,
            account_group_id=self._account_group_id,
            account_id=self._account_id,
            limit=self._limit,
            next_docid=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next_url')
        if next_page_link is not None:
            next = get_query_param(next_page_link, 'next_docid')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('resources')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of Enterprise.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results


class AccountsPager:
    """
    AccountsPager can be used to simplify the use of the "list_accounts" method.
    """

    def __init__(
        self,
        *,
        client: EnterpriseManagementV1,
        enterprise_id: str = None,
        account_group_id: str = None,
        parent: str = None,
        limit: int = None,
        include_deleted: bool = None,
    ) -> None:
        """
        Initialize a AccountsPager object.
        :param str enterprise_id: (optional) Get accounts that are either immediate
               children or are a part of the hierarchy for a given enterprise ID.
        :param str account_group_id: (optional) Get accounts that are either
               immediate children or are a part of the hierarchy for a given account group
               ID.
        :param str parent: (optional) Get accounts that are either immediate
               children or are a part of the hierarchy for a given parent CRN.
        :param int limit: (optional) Return results up to this limit. Valid values
               are between `0` and `100`.
        :param bool include_deleted: (optional) Include the deleted accounts from
               an enterprise when used in conjunction with enterprise_id.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._enterprise_id = enterprise_id
        self._account_group_id = account_group_id
        self._parent = parent
        self._limit = limit
        self._include_deleted = include_deleted

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of Account.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.list_accounts(
            enterprise_id=self._enterprise_id,
            account_group_id=self._account_group_id,
            parent=self._parent,
            limit=self._limit,
            include_deleted=self._include_deleted,
            next_docid=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next_url')
        if next_page_link is not None:
            next = get_query_param(next_page_link, 'next_docid')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('resources')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of Account.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results


class AccountGroupsPager:
    """
    AccountGroupsPager can be used to simplify the use of the "list_account_groups" method.
    """

    def __init__(
        self,
        *,
        client: EnterpriseManagementV1,
        enterprise_id: str = None,
        parent_account_group_id: str = None,
        parent: str = None,
        limit: int = None,
        include_deleted: bool = None,
    ) -> None:
        """
        Initialize a AccountGroupsPager object.
        :param str enterprise_id: (optional) Get account groups that are either
               immediate children or are a part of the hierarchy for a given enterprise
               ID.
        :param str parent_account_group_id: (optional) Get account groups that are
               either immediate children or are a part of the hierarchy for a given
               account group ID.
        :param str parent: (optional) Get account groups that are either immediate
               children or are a part of the hierarchy for a given parent CRN.
        :param int limit: (optional) Return results up to this limit. Valid values
               are between `0` and `100`.
        :param bool include_deleted: (optional) Include the deleted account groups
               from an enterprise when used in conjunction with other query parameters.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._enterprise_id = enterprise_id
        self._parent_account_group_id = parent_account_group_id
        self._parent = parent
        self._limit = limit
        self._include_deleted = include_deleted

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of AccountGroup.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.list_account_groups(
            enterprise_id=self._enterprise_id,
            parent_account_group_id=self._parent_account_group_id,
            parent=self._parent,
            limit=self._limit,
            include_deleted=self._include_deleted,
            next_docid=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next_url')
        if next_page_link is not None:
            next = get_query_param(next_page_link, 'next_docid')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('resources')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of AccountGroup.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results
