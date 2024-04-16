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
Usage reports for IBM Cloud accounts

API Version: 4.0.6
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import json

from ibm_cloud_sdk_core import BaseService, DetailedResponse, get_query_param
from ibm_cloud_sdk_core.authenticators.authenticator import Authenticator
from ibm_cloud_sdk_core.get_authenticator import get_authenticator_from_environment
from ibm_cloud_sdk_core.utils import datetime_to_string, string_to_datetime

from .common import get_sdk_headers

##############################################################################
# Service
##############################################################################


class UsageReportsV4(BaseService):
    """The Usage Reports V4 service."""

    DEFAULT_SERVICE_URL = 'https://billing.cloud.ibm.com'
    DEFAULT_SERVICE_NAME = 'usage_reports'

    @classmethod
    def new_instance(
        cls,
        service_name: str = DEFAULT_SERVICE_NAME,
    ) -> 'UsageReportsV4':
        """
        Return a new client for the Usage Reports service using the specified
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
        Construct a new client for the Usage Reports service.

        :param Authenticator authenticator: The authenticator specifies the authentication mechanism.
               Get up to date information from https://github.com/IBM/python-sdk-core/blob/main/README.md
               about initializing the authenticator of your choice.
        """
        BaseService.__init__(self, service_url=self.DEFAULT_SERVICE_URL, authenticator=authenticator)

    #########################
    # Account operations
    #########################

    def get_account_summary(
        self,
        account_id: str,
        billingmonth: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get account summary.

        Returns the summary for the account for a given month. Account billing managers
        are authorized to access this report.

        :param str account_id: Account ID for which the usage report is requested.
        :param str billingmonth: The billing month for which the usage report is
               requested.  Format is yyyy-mm.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccountSummary` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not billingmonth:
            raise ValueError('billingmonth must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V4',
            operation_id='get_account_summary',
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id', 'billingmonth']
        path_param_values = self.encode_path_vars(account_id, billingmonth)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v4/accounts/{account_id}/summary/{billingmonth}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
        )

        response = self.send(request, **kwargs)
        return response

    def get_account_usage(
        self,
        account_id: str,
        billingmonth: str,
        *,
        names: Optional[bool] = None,
        accept_language: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get account usage.

        Usage for all the resources and plans in an account for a given month. Account
        billing managers are authorized to access this report.

        :param str account_id: Account ID for which the usage report is requested.
        :param str billingmonth: The billing month for which the usage report is
               requested.  Format is yyyy-mm.
        :param bool names: (optional) Include the name of every resource, plan,
               resource instance, organization, and resource group.
        :param str accept_language: (optional) Prioritize the names returned in the
               order of the specified languages. Language will default to English.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccountUsage` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not billingmonth:
            raise ValueError('billingmonth must be provided')
        headers = {
            'Accept-Language': accept_language,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V4',
            operation_id='get_account_usage',
        )
        headers.update(sdk_headers)

        params = {
            '_names': names,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id', 'billingmonth']
        path_param_values = self.encode_path_vars(account_id, billingmonth)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v4/accounts/{account_id}/usage/{billingmonth}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # Resource operations
    #########################

    def get_resource_group_usage(
        self,
        account_id: str,
        resource_group_id: str,
        billingmonth: str,
        *,
        names: Optional[bool] = None,
        accept_language: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get resource group usage.

        Usage for all the resources and plans in a resource group in a given month.
        Account billing managers or resource group billing managers are authorized to
        access this report.

        :param str account_id: Account ID for which the usage report is requested.
        :param str resource_group_id: Resource group for which the usage report is
               requested.
        :param str billingmonth: The billing month for which the usage report is
               requested.  Format is yyyy-mm.
        :param bool names: (optional) Include the name of every resource, plan,
               resource instance, organization, and resource group.
        :param str accept_language: (optional) Prioritize the names returned in the
               order of the specified languages. Language will default to English.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResourceGroupUsage` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not resource_group_id:
            raise ValueError('resource_group_id must be provided')
        if not billingmonth:
            raise ValueError('billingmonth must be provided')
        headers = {
            'Accept-Language': accept_language,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V4',
            operation_id='get_resource_group_usage',
        )
        headers.update(sdk_headers)

        params = {
            '_names': names,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id', 'resource_group_id', 'billingmonth']
        path_param_values = self.encode_path_vars(account_id, resource_group_id, billingmonth)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v4/accounts/{account_id}/resource_groups/{resource_group_id}/usage/{billingmonth}'.format(
            **path_param_dict
        )
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def get_resource_usage_account(
        self,
        account_id: str,
        billingmonth: str,
        *,
        names: Optional[bool] = None,
        tags: Optional[bool] = None,
        accept_language: Optional[str] = None,
        limit: Optional[int] = None,
        start: Optional[str] = None,
        resource_group_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        resource_instance_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        plan_id: Optional[str] = None,
        region: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get resource instance usage in an account.

        Query for resource instance usage in an account. Filter the results with query
        parameters. Account billing administrator is authorized to access this report.

        :param str account_id: Account ID for which the usage report is requested.
        :param str billingmonth: The billing month for which the usage report is
               requested.  Format is yyyy-mm.
        :param bool names: (optional) Include the name of every resource, plan,
               resource instance, organization, and resource group.
        :param bool tags: (optional) Include the tags associated with every
               resource instance. By default it is always `true`.
        :param str accept_language: (optional) Prioritize the names returned in the
               order of the specified languages. Language will default to English.
        :param int limit: (optional) Number of usage records returned. The default
               value is 30. Maximum value is 200.
        :param str start: (optional) The offset from which the records must be
               fetched. Offset information is included in the response.
        :param str resource_group_id: (optional) Filter by resource group.
        :param str organization_id: (optional) Filter by organization_id.
        :param str resource_instance_id: (optional) Filter by resource instance_id.
        :param str resource_id: (optional) Filter by resource_id.
        :param str plan_id: (optional) Filter by plan_id.
        :param str region: (optional) Region in which the resource instance is
               provisioned.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `InstancesUsage` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not billingmonth:
            raise ValueError('billingmonth must be provided')
        headers = {
            'Accept-Language': accept_language,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V4',
            operation_id='get_resource_usage_account',
        )
        headers.update(sdk_headers)

        params = {
            '_names': names,
            '_tags': tags,
            '_limit': limit,
            '_start': start,
            'resource_group_id': resource_group_id,
            'organization_id': organization_id,
            'resource_instance_id': resource_instance_id,
            'resource_id': resource_id,
            'plan_id': plan_id,
            'region': region,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id', 'billingmonth']
        path_param_values = self.encode_path_vars(account_id, billingmonth)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v4/accounts/{account_id}/resource_instances/usage/{billingmonth}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def get_resource_usage_resource_group(
        self,
        account_id: str,
        resource_group_id: str,
        billingmonth: str,
        *,
        names: Optional[bool] = None,
        tags: Optional[bool] = None,
        accept_language: Optional[str] = None,
        limit: Optional[int] = None,
        start: Optional[str] = None,
        resource_instance_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        plan_id: Optional[str] = None,
        region: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get resource instance usage in a resource group.

        Query for resource instance usage in a resource group. Filter the results with
        query parameters. Account billing administrator and resource group billing
        administrators are authorized to access this report.

        :param str account_id: Account ID for which the usage report is requested.
        :param str resource_group_id: Resource group for which the usage report is
               requested.
        :param str billingmonth: The billing month for which the usage report is
               requested.  Format is yyyy-mm.
        :param bool names: (optional) Include the name of every resource, plan,
               resource instance, organization, and resource group.
        :param bool tags: (optional) Include the tags associated with every
               resource instance. By default it is always `true`.
        :param str accept_language: (optional) Prioritize the names returned in the
               order of the specified languages. Language will default to English.
        :param int limit: (optional) Number of usage records returned. The default
               value is 30. Maximum value is 200.
        :param str start: (optional) The offset from which the records must be
               fetched. Offset information is included in the response.
        :param str resource_instance_id: (optional) Filter by resource instance id.
        :param str resource_id: (optional) Filter by resource_id.
        :param str plan_id: (optional) Filter by plan_id.
        :param str region: (optional) Region in which the resource instance is
               provisioned.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `InstancesUsage` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not resource_group_id:
            raise ValueError('resource_group_id must be provided')
        if not billingmonth:
            raise ValueError('billingmonth must be provided')
        headers = {
            'Accept-Language': accept_language,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V4',
            operation_id='get_resource_usage_resource_group',
        )
        headers.update(sdk_headers)

        params = {
            '_names': names,
            '_tags': tags,
            '_limit': limit,
            '_start': start,
            'resource_instance_id': resource_instance_id,
            'resource_id': resource_id,
            'plan_id': plan_id,
            'region': region,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id', 'resource_group_id', 'billingmonth']
        path_param_values = self.encode_path_vars(account_id, resource_group_id, billingmonth)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v4/accounts/{account_id}/resource_groups/{resource_group_id}/resource_instances/usage/{billingmonth}'.format(
            **path_param_dict
        )
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def get_resource_usage_org(
        self,
        account_id: str,
        organization_id: str,
        billingmonth: str,
        *,
        names: Optional[bool] = None,
        tags: Optional[bool] = None,
        accept_language: Optional[str] = None,
        limit: Optional[int] = None,
        start: Optional[str] = None,
        resource_instance_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        plan_id: Optional[str] = None,
        region: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get resource instance usage in an organization.

        Query for resource instance usage in an organization. Filter the results with
        query parameters. Account billing administrator and organization billing
        administrators are authorized to access this report.

        :param str account_id: Account ID for which the usage report is requested.
        :param str organization_id: ID of the organization.
        :param str billingmonth: The billing month for which the usage report is
               requested.  Format is yyyy-mm.
        :param bool names: (optional) Include the name of every resource, plan,
               resource instance, organization, and resource group.
        :param bool tags: (optional) Include the tags associated with every
               resource instance. By default it is always `true`.
        :param str accept_language: (optional) Prioritize the names returned in the
               order of the specified languages. Language will default to English.
        :param int limit: (optional) Number of usage records returned. The default
               value is 30. Maximum value is 200.
        :param str start: (optional) The offset from which the records must be
               fetched. Offset information is included in the response.
        :param str resource_instance_id: (optional) Filter by resource instance id.
        :param str resource_id: (optional) Filter by resource_id.
        :param str plan_id: (optional) Filter by plan_id.
        :param str region: (optional) Region in which the resource instance is
               provisioned.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `InstancesUsage` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not organization_id:
            raise ValueError('organization_id must be provided')
        if not billingmonth:
            raise ValueError('billingmonth must be provided')
        headers = {
            'Accept-Language': accept_language,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V4',
            operation_id='get_resource_usage_org',
        )
        headers.update(sdk_headers)

        params = {
            '_names': names,
            '_tags': tags,
            '_limit': limit,
            '_start': start,
            'resource_instance_id': resource_instance_id,
            'resource_id': resource_id,
            'plan_id': plan_id,
            'region': region,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id', 'organization_id', 'billingmonth']
        path_param_values = self.encode_path_vars(account_id, organization_id, billingmonth)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = (
            '/v4/accounts/{account_id}/organizations/{organization_id}/resource_instances/usage/{billingmonth}'.format(
                **path_param_dict
            )
        )
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # Organization operations
    #########################

    def get_org_usage(
        self,
        account_id: str,
        organization_id: str,
        billingmonth: str,
        *,
        names: Optional[bool] = None,
        accept_language: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get organization usage.

        Usage for all the resources and plans in an organization in a given month. Account
        billing managers or organization billing managers are authorized to access this
        report.

        :param str account_id: Account ID for which the usage report is requested.
        :param str organization_id: ID of the organization.
        :param str billingmonth: The billing month for which the usage report is
               requested.  Format is yyyy-mm.
        :param bool names: (optional) Include the name of every resource, plan,
               resource instance, organization, and resource group.
        :param str accept_language: (optional) Prioritize the names returned in the
               order of the specified languages. Language will default to English.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `OrgUsage` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not organization_id:
            raise ValueError('organization_id must be provided')
        if not billingmonth:
            raise ValueError('billingmonth must be provided')
        headers = {
            'Accept-Language': accept_language,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V4',
            operation_id='get_org_usage',
        )
        headers.update(sdk_headers)

        params = {
            '_names': names,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id', 'organization_id', 'billingmonth']
        path_param_values = self.encode_path_vars(account_id, organization_id, billingmonth)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v4/accounts/{account_id}/organizations/{organization_id}/usage/{billingmonth}'.format(**path_param_dict)
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # Billing reports snapshot
    #########################

    def create_reports_snapshot_config(
        self,
        account_id: str,
        interval: str,
        cos_bucket: str,
        cos_location: str,
        *,
        cos_reports_folder: Optional[str] = None,
        report_types: Optional[List[str]] = None,
        versioning: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Setup the snapshot configuration.

        Snapshots of the billing reports would be taken on a periodic interval and stored
        based on the configuration setup by the customer for the given Account Id.

        :param str account_id: Account ID for which billing report snapshot is
               configured.
        :param str interval: Frequency of taking the snapshot of the billing
               reports.
        :param str cos_bucket: The name of the COS bucket to store the snapshot of
               the billing reports.
        :param str cos_location: Region of the COS instance.
        :param str cos_reports_folder: (optional) The billing reports root folder
               to store the billing reports snapshots. Defaults to
               "IBMCloud-Billing-Reports".
        :param List[str] report_types: (optional) The type of billing reports to
               take snapshot of. Possible values are [account_summary, enterprise_summary,
               account_resource_instance_usage].
        :param str versioning: (optional) A new version of report is created or the
               existing report version is overwritten with every update. Defaults to
               "new".
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `SnapshotConfig` object
        """

        if account_id is None:
            raise ValueError('account_id must be provided')
        if interval is None:
            raise ValueError('interval must be provided')
        if cos_bucket is None:
            raise ValueError('cos_bucket must be provided')
        if cos_location is None:
            raise ValueError('cos_location must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V4',
            operation_id='create_reports_snapshot_config',
        )
        headers.update(sdk_headers)

        data = {
            'account_id': account_id,
            'interval': interval,
            'cos_bucket': cos_bucket,
            'cos_location': cos_location,
            'cos_reports_folder': cos_reports_folder,
            'report_types': report_types,
            'versioning': versioning,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/billing-reports-snapshot-config'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def get_reports_snapshot_config(
        self,
        account_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Fetch the snapshot configuration.

        Returns the configuration of snapshot of the billing reports setup by the customer
        for the given Account Id.

        :param str account_id: Account ID for which the billing report snapshot is
               configured.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `SnapshotConfig` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V4',
            operation_id='get_reports_snapshot_config',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/billing-reports-snapshot-config'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def update_reports_snapshot_config(
        self,
        account_id: str,
        *,
        interval: Optional[str] = None,
        cos_bucket: Optional[str] = None,
        cos_location: Optional[str] = None,
        cos_reports_folder: Optional[str] = None,
        report_types: Optional[List[str]] = None,
        versioning: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Update the snapshot configuration.

        Updates the configuration of snapshot of the billing reports setup by the customer
        for the given Account Id.

        :param str account_id: Account ID for which billing report snapshot is
               configured.
        :param str interval: (optional) Frequency of taking the snapshot of the
               billing reports.
        :param str cos_bucket: (optional) The name of the COS bucket to store the
               snapshot of the billing reports.
        :param str cos_location: (optional) Region of the COS instance.
        :param str cos_reports_folder: (optional) The billing reports root folder
               to store the billing reports snapshots.
        :param List[str] report_types: (optional) The type of billing reports to
               take snapshot of. Possible values are [account_summary, enterprise_summary,
               account_resource_instance_usage].
        :param str versioning: (optional) A new version of report is created or the
               existing report version is overwritten with every update.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `SnapshotConfig` object
        """

        if account_id is None:
            raise ValueError('account_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V4',
            operation_id='update_reports_snapshot_config',
        )
        headers.update(sdk_headers)

        data = {
            'account_id': account_id,
            'interval': interval,
            'cos_bucket': cos_bucket,
            'cos_location': cos_location,
            'cos_reports_folder': cos_reports_folder,
            'report_types': report_types,
            'versioning': versioning,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/billing-reports-snapshot-config'
        request = self.prepare_request(
            method='PATCH',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def delete_reports_snapshot_config(
        self,
        account_id: str,
        **kwargs,
    ) -> DetailedResponse:
        """
        Delete the snapshot configuration.

        Delete the configuration of snapshot of the billing reports setup by the customer
        for the given Account Id.

        :param str account_id: Account ID for which the billing report snapshot is
               configured.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V4',
            operation_id='delete_reports_snapshot_config',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']

        url = '/v1/billing-reports-snapshot-config'
        request = self.prepare_request(
            method='DELETE',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    def validate_reports_snapshot_config(
        self,
        account_id: str,
        *,
        interval: str = None,
        cos_bucket: str = None,
        cos_location: str = None,
        cos_reports_folder: str = None,
        report_types: List[str] = None,
        versioning: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Verify billing to COS authorization.

        Verify billing service to COS bucket authorization for the given account_id. If
        COS bucket information is not provided, COS bucket information is retrieved from
        the configuration file.

        :param str account_id: Account ID for which billing report snapshot is
               configured.
        :param str interval: (optional) Frequency of taking the snapshot of the
               billing reports.
        :param str cos_bucket: (optional) The name of the COS bucket to store the
               snapshot of the billing reports.
        :param str cos_location: (optional) Region of the COS instance.
        :param str cos_reports_folder: (optional) The billing reports root folder
               to store the billing reports snapshots. Defaults to
               "IBMCloud-Billing-Reports".
        :param List[str] report_types: (optional) The type of billing reports to
               take snapshot of. Possible values are [account_summary, enterprise_summary,
               account_resource_instance_usage].
        :param str versioning: (optional) A new version of report is created or the
               existing report version is overwritten with every update. Defaults to
               "new".
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `SnapshotConfigValidateResponse` object
        """

        if account_id is None:
            raise ValueError('account_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V4',
            operation_id='validate_reports_snapshot_config',
        )
        headers.update(sdk_headers)

        data = {
            'account_id': account_id,
            'interval': interval,
            'cos_bucket': cos_bucket,
            'cos_location': cos_location,
            'cos_reports_folder': cos_reports_folder,
            'report_types': report_types,
            'versioning': versioning,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/billing-reports-snapshot-config/validate'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response

    def get_reports_snapshot(
        self,
        account_id: str,
        month: str,
        *,
        date_from: int = None,
        date_to: int = None,
        limit: int = None,
        start: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Fetch the current or past snapshots.

        Returns the billing reports snapshots captured for the given Account Id in the
        specific time period.

        :param str account_id: Account ID for which the billing report snapshot is
               requested.
        :param str month: The month for which billing report snapshot is requested.
                Format is yyyy-mm.
        :param int date_from: (optional) Timestamp in milliseconds for which
               billing report snapshot is requested.
        :param int date_to: (optional) Timestamp in milliseconds for which billing
               report snapshot is requested.
        :param int limit: (optional) Number of usage records returned. The default
               value is 30. Maximum value is 200.
        :param str start: (optional) The offset from which the records must be
               fetched. Offset information is included in the response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `SnapshotList` object
        """

        if not account_id:
            raise ValueError('account_id must be provided')
        if not month:
            raise ValueError('month must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V4',
            operation_id='get_reports_snapshot',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
            'month': month,
            'date_from': date_from,
            'date_to': date_to,
            '_limit': limit,
            '_start': start,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/billing-reports-snapshots'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response


##############################################################################
# Models
##############################################################################


class AccountSummary:
    """
    A summary of charges and credits for an account.

    :param str account_id: The ID of the account.
    :param List[Resource] account_resources: (optional) The list of account
          resources for the month.
    :param str month: The month in which usages were incurred. Represented in
          yyyy-mm format.
    :param str billing_country_code: Country.
    :param str billing_currency_code: The currency in which the account is billed.
    :param ResourcesSummary resources: Charges related to cloud resources.
    :param List[Offer] offers: The list of offers applicable for the account for the
          month.
    :param List[SupportSummary] support: Support-related charges.
    :param List[object] support_resources: (optional) The list of support resources
          for the month.
    :param SubscriptionSummary subscription: A summary of charges and credits
          related to a subscription.
    """

    def __init__(
        self,
        account_id: str,
        month: str,
        billing_country_code: str,
        billing_currency_code: str,
        resources: 'ResourcesSummary',
        offers: List['Offer'],
        support: List['SupportSummary'],
        subscription: 'SubscriptionSummary',
        *,
        account_resources: Optional[List['Resource']] = None,
        support_resources: Optional[List[object]] = None,
    ) -> None:
        """
        Initialize a AccountSummary object.

        :param str account_id: The ID of the account.
        :param str month: The month in which usages were incurred. Represented in
               yyyy-mm format.
        :param str billing_country_code: Country.
        :param str billing_currency_code: The currency in which the account is
               billed.
        :param ResourcesSummary resources: Charges related to cloud resources.
        :param List[Offer] offers: The list of offers applicable for the account
               for the month.
        :param List[SupportSummary] support: Support-related charges.
        :param SubscriptionSummary subscription: A summary of charges and credits
               related to a subscription.
        :param List[Resource] account_resources: (optional) The list of account
               resources for the month.
        :param List[object] support_resources: (optional) The list of support
               resources for the month.
        """
        self.account_id = account_id
        self.account_resources = account_resources
        self.month = month
        self.billing_country_code = billing_country_code
        self.billing_currency_code = billing_currency_code
        self.resources = resources
        self.offers = offers
        self.support = support
        self.support_resources = support_resources
        self.subscription = subscription

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccountSummary':
        """Initialize a AccountSummary object from a json dictionary."""
        args = {}
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        else:
            raise ValueError('Required property \'account_id\' not present in AccountSummary JSON')
        if (account_resources := _dict.get('account_resources')) is not None:
            args['account_resources'] = [Resource.from_dict(v) for v in account_resources]
        if (month := _dict.get('month')) is not None:
            args['month'] = month
        else:
            raise ValueError('Required property \'month\' not present in AccountSummary JSON')
        if (billing_country_code := _dict.get('billing_country_code')) is not None:
            args['billing_country_code'] = billing_country_code
        else:
            raise ValueError('Required property \'billing_country_code\' not present in AccountSummary JSON')
        if (billing_currency_code := _dict.get('billing_currency_code')) is not None:
            args['billing_currency_code'] = billing_currency_code
        else:
            raise ValueError('Required property \'billing_currency_code\' not present in AccountSummary JSON')
        if (resources := _dict.get('resources')) is not None:
            args['resources'] = ResourcesSummary.from_dict(resources)
        else:
            raise ValueError('Required property \'resources\' not present in AccountSummary JSON')
        if (offers := _dict.get('offers')) is not None:
            args['offers'] = [Offer.from_dict(v) for v in offers]
        else:
            raise ValueError('Required property \'offers\' not present in AccountSummary JSON')
        if (support := _dict.get('support')) is not None:
            args['support'] = [SupportSummary.from_dict(v) for v in support]
        else:
            raise ValueError('Required property \'support\' not present in AccountSummary JSON')
        if (support_resources := _dict.get('support_resources')) is not None:
            args['support_resources'] = support_resources
        if (subscription := _dict.get('subscription')) is not None:
            args['subscription'] = SubscriptionSummary.from_dict(subscription)
        else:
            raise ValueError('Required property \'subscription\' not present in AccountSummary JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccountSummary object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'account_resources') and self.account_resources is not None:
            account_resources_list = []
            for v in self.account_resources:
                if isinstance(v, dict):
                    account_resources_list.append(v)
                else:
                    account_resources_list.append(v.to_dict())
            _dict['account_resources'] = account_resources_list
        if hasattr(self, 'month') and self.month is not None:
            _dict['month'] = self.month
        if hasattr(self, 'billing_country_code') and self.billing_country_code is not None:
            _dict['billing_country_code'] = self.billing_country_code
        if hasattr(self, 'billing_currency_code') and self.billing_currency_code is not None:
            _dict['billing_currency_code'] = self.billing_currency_code
        if hasattr(self, 'resources') and self.resources is not None:
            if isinstance(self.resources, dict):
                _dict['resources'] = self.resources
            else:
                _dict['resources'] = self.resources.to_dict()
        if hasattr(self, 'offers') and self.offers is not None:
            offers_list = []
            for v in self.offers:
                if isinstance(v, dict):
                    offers_list.append(v)
                else:
                    offers_list.append(v.to_dict())
            _dict['offers'] = offers_list
        if hasattr(self, 'support') and self.support is not None:
            support_list = []
            for v in self.support:
                if isinstance(v, dict):
                    support_list.append(v)
                else:
                    support_list.append(v.to_dict())
            _dict['support'] = support_list
        if hasattr(self, 'support_resources') and self.support_resources is not None:
            _dict['support_resources'] = self.support_resources
        if hasattr(self, 'subscription') and self.subscription is not None:
            if isinstance(self.subscription, dict):
                _dict['subscription'] = self.subscription
            else:
                _dict['subscription'] = self.subscription.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AccountSummary object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccountSummary') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccountSummary') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AccountUsage:
    """
    The aggregated usage and charges for all the plans in the account.

    :param str account_id: The ID of the account.
    :param str pricing_country: The target country pricing that should be used.
    :param str currency_code: The currency for the cost fields in the resources,
          plans and metrics.
    :param str month: The month.
    :param List[Resource] resources: All the resource used in the account.
    :param float currency_rate: (optional) The value of the account's currency in
          USD.
    """

    def __init__(
        self,
        account_id: str,
        pricing_country: str,
        currency_code: str,
        month: str,
        resources: List['Resource'],
        *,
        currency_rate: Optional[float] = None,
    ) -> None:
        """
        Initialize a AccountUsage object.

        :param str account_id: The ID of the account.
        :param str pricing_country: The target country pricing that should be used.
        :param str currency_code: The currency for the cost fields in the
               resources, plans and metrics.
        :param str month: The month.
        :param List[Resource] resources: All the resource used in the account.
        :param float currency_rate: (optional) The value of the account's currency
               in USD.
        """
        self.account_id = account_id
        self.pricing_country = pricing_country
        self.currency_code = currency_code
        self.month = month
        self.resources = resources
        self.currency_rate = currency_rate

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccountUsage':
        """Initialize a AccountUsage object from a json dictionary."""
        args = {}
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        else:
            raise ValueError('Required property \'account_id\' not present in AccountUsage JSON')
        if (pricing_country := _dict.get('pricing_country')) is not None:
            args['pricing_country'] = pricing_country
        else:
            raise ValueError('Required property \'pricing_country\' not present in AccountUsage JSON')
        if (currency_code := _dict.get('currency_code')) is not None:
            args['currency_code'] = currency_code
        else:
            raise ValueError('Required property \'currency_code\' not present in AccountUsage JSON')
        if (month := _dict.get('month')) is not None:
            args['month'] = month
        else:
            raise ValueError('Required property \'month\' not present in AccountUsage JSON')
        if (resources := _dict.get('resources')) is not None:
            args['resources'] = [Resource.from_dict(v) for v in resources]
        else:
            raise ValueError('Required property \'resources\' not present in AccountUsage JSON')
        if (currency_rate := _dict.get('currency_rate')) is not None:
            args['currency_rate'] = currency_rate
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccountUsage object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'pricing_country') and self.pricing_country is not None:
            _dict['pricing_country'] = self.pricing_country
        if hasattr(self, 'currency_code') and self.currency_code is not None:
            _dict['currency_code'] = self.currency_code
        if hasattr(self, 'month') and self.month is not None:
            _dict['month'] = self.month
        if hasattr(self, 'resources') and self.resources is not None:
            resources_list = []
            for v in self.resources:
                if isinstance(v, dict):
                    resources_list.append(v)
                else:
                    resources_list.append(v.to_dict())
            _dict['resources'] = resources_list
        if hasattr(self, 'currency_rate') and self.currency_rate is not None:
            _dict['currency_rate'] = self.currency_rate
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AccountUsage object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccountUsage') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccountUsage') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Discount:
    """
    Information about a discount that is associated with a metric.

    :param str ref: The reference ID of the discount.
    :param str name: (optional) The name of the discount indicating category.
    :param str display_name: (optional) The name of the discount.
    :param float discount: The discount percentage.
    """

    def __init__(
        self,
        ref: str,
        discount: float,
        *,
        name: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> None:
        """
        Initialize a Discount object.

        :param str ref: The reference ID of the discount.
        :param float discount: The discount percentage.
        :param str name: (optional) The name of the discount indicating category.
        :param str display_name: (optional) The name of the discount.
        """
        self.ref = ref
        self.name = name
        self.display_name = display_name
        self.discount = discount

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Discount':
        """Initialize a Discount object from a json dictionary."""
        args = {}
        if (ref := _dict.get('ref')) is not None:
            args['ref'] = ref
        else:
            raise ValueError('Required property \'ref\' not present in Discount JSON')
        if (name := _dict.get('name')) is not None:
            args['name'] = name
        if (display_name := _dict.get('display_name')) is not None:
            args['display_name'] = display_name
        if (discount := _dict.get('discount')) is not None:
            args['discount'] = discount
        else:
            raise ValueError('Required property \'discount\' not present in Discount JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Discount object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'ref') and self.ref is not None:
            _dict['ref'] = self.ref
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'display_name') and self.display_name is not None:
            _dict['display_name'] = self.display_name
        if hasattr(self, 'discount') and self.discount is not None:
            _dict['discount'] = self.discount
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Discount object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Discount') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Discount') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class InstanceUsage:
    """
    The aggregated usage and charges for an instance.

    :param str account_id: The ID of the account.
    :param str resource_instance_id: The ID of the resource instance.
    :param str resource_instance_name: (optional) The name of the resource instance.
    :param str resource_id: The ID of the resource.
    :param str catalog_id: (optional) The catalog ID of the resource.
    :param str resource_name: (optional) The name of the resource.
    :param str resource_group_id: (optional) The ID of the resource group.
    :param str resource_group_name: (optional) The name of the resource group.
    :param str organization_id: (optional) The ID of the organization.
    :param str organization_name: (optional) The name of the organization.
    :param str space_id: (optional) The ID of the space.
    :param str space_name: (optional) The name of the space.
    :param str consumer_id: (optional) The ID of the consumer.
    :param str region: (optional) The region where instance was provisioned.
    :param str pricing_region: (optional) The pricing region where the usage that
          was submitted was rated.
    :param str pricing_country: The target country pricing that should be used.
    :param str currency_code: The currency for the cost fields in the resources,
          plans and metrics.
    :param bool billable: Is the cost charged to the account.
    :param str parent_resource_instance_id: (optional) The resource instance id of
          the parent resource associated with this instance.
    :param str plan_id: The ID of the plan where the instance was provisioned and
          rated.
    :param str plan_name: (optional) The name of the plan where the instance was
          provisioned and rated.
    :param str pricing_plan_id: (optional) The ID of the pricing plan used to rate
          the usage.
    :param str month: The month.
    :param List[Metric] usage: All the resource used in the account.
    :param bool pending: (optional) Pending charge from classic infrastructure.
    :param float currency_rate: (optional) The value of the account's currency in
          USD.
    :param List[object] tags: (optional) The user tags associated with a resource
          instance.
    :param List[object] service_tags: (optional) The service tags associated with a
          resource instance.
    """

    def __init__(
        self,
        account_id: str,
        resource_instance_id: str,
        resource_id: str,
        pricing_country: str,
        currency_code: str,
        billable: bool,
        plan_id: str,
        month: str,
        usage: List['Metric'],
        *,
        resource_instance_name: Optional[str] = None,
        catalog_id: Optional[str] = None,
        resource_name: Optional[str] = None,
        resource_group_id: Optional[str] = None,
        resource_group_name: Optional[str] = None,
        organization_id: Optional[str] = None,
        organization_name: Optional[str] = None,
        space_id: Optional[str] = None,
        space_name: Optional[str] = None,
        consumer_id: Optional[str] = None,
        region: Optional[str] = None,
        pricing_region: Optional[str] = None,
        parent_resource_instance_id: Optional[str] = None,
        plan_name: Optional[str] = None,
        pricing_plan_id: Optional[str] = None,
        pending: Optional[bool] = None,
        currency_rate: Optional[float] = None,
        tags: Optional[List[object]] = None,
        service_tags: Optional[List[object]] = None,
    ) -> None:
        """
        Initialize a InstanceUsage object.

        :param str account_id: The ID of the account.
        :param str resource_instance_id: The ID of the resource instance.
        :param str resource_id: The ID of the resource.
        :param str pricing_country: The target country pricing that should be used.
        :param str currency_code: The currency for the cost fields in the
               resources, plans and metrics.
        :param bool billable: Is the cost charged to the account.
        :param str plan_id: The ID of the plan where the instance was provisioned
               and rated.
        :param str month: The month.
        :param List[Metric] usage: All the resource used in the account.
        :param str resource_instance_name: (optional) The name of the resource
               instance.
        :param str catalog_id: (optional) The catalog ID of the resource.
        :param str resource_name: (optional) The name of the resource.
        :param str resource_group_id: (optional) The ID of the resource group.
        :param str resource_group_name: (optional) The name of the resource group.
        :param str organization_id: (optional) The ID of the organization.
        :param str organization_name: (optional) The name of the organization.
        :param str space_id: (optional) The ID of the space.
        :param str space_name: (optional) The name of the space.
        :param str consumer_id: (optional) The ID of the consumer.
        :param str region: (optional) The region where instance was provisioned.
        :param str pricing_region: (optional) The pricing region where the usage
               that was submitted was rated.
        :param str parent_resource_instance_id: (optional) The resource instance id
               of the parent resource associated with this instance.
        :param str plan_name: (optional) The name of the plan where the instance
               was provisioned and rated.
        :param str pricing_plan_id: (optional) The ID of the pricing plan used to
               rate the usage.
        :param bool pending: (optional) Pending charge from classic infrastructure.
        :param float currency_rate: (optional) The value of the account's currency
               in USD.
        :param List[object] tags: (optional) The user tags associated with a
               resource instance.
        :param List[object] service_tags: (optional) The service tags associated
               with a resource instance.
        """
        self.account_id = account_id
        self.resource_instance_id = resource_instance_id
        self.resource_instance_name = resource_instance_name
        self.resource_id = resource_id
        self.catalog_id = catalog_id
        self.resource_name = resource_name
        self.resource_group_id = resource_group_id
        self.resource_group_name = resource_group_name
        self.organization_id = organization_id
        self.organization_name = organization_name
        self.space_id = space_id
        self.space_name = space_name
        self.consumer_id = consumer_id
        self.region = region
        self.pricing_region = pricing_region
        self.pricing_country = pricing_country
        self.currency_code = currency_code
        self.billable = billable
        self.parent_resource_instance_id = parent_resource_instance_id
        self.plan_id = plan_id
        self.plan_name = plan_name
        self.pricing_plan_id = pricing_plan_id
        self.month = month
        self.usage = usage
        self.pending = pending
        self.currency_rate = currency_rate
        self.tags = tags
        self.service_tags = service_tags

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'InstanceUsage':
        """Initialize a InstanceUsage object from a json dictionary."""
        args = {}
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        else:
            raise ValueError('Required property \'account_id\' not present in InstanceUsage JSON')
        if (resource_instance_id := _dict.get('resource_instance_id')) is not None:
            args['resource_instance_id'] = resource_instance_id
        else:
            raise ValueError('Required property \'resource_instance_id\' not present in InstanceUsage JSON')
        if (resource_instance_name := _dict.get('resource_instance_name')) is not None:
            args['resource_instance_name'] = resource_instance_name
        if (resource_id := _dict.get('resource_id')) is not None:
            args['resource_id'] = resource_id
        else:
            raise ValueError('Required property \'resource_id\' not present in InstanceUsage JSON')
        if (catalog_id := _dict.get('catalog_id')) is not None:
            args['catalog_id'] = catalog_id
        if (resource_name := _dict.get('resource_name')) is not None:
            args['resource_name'] = resource_name
        if (resource_group_id := _dict.get('resource_group_id')) is not None:
            args['resource_group_id'] = resource_group_id
        if (resource_group_name := _dict.get('resource_group_name')) is not None:
            args['resource_group_name'] = resource_group_name
        if (organization_id := _dict.get('organization_id')) is not None:
            args['organization_id'] = organization_id
        if (organization_name := _dict.get('organization_name')) is not None:
            args['organization_name'] = organization_name
        if (space_id := _dict.get('space_id')) is not None:
            args['space_id'] = space_id
        if (space_name := _dict.get('space_name')) is not None:
            args['space_name'] = space_name
        if (consumer_id := _dict.get('consumer_id')) is not None:
            args['consumer_id'] = consumer_id
        if (region := _dict.get('region')) is not None:
            args['region'] = region
        if (pricing_region := _dict.get('pricing_region')) is not None:
            args['pricing_region'] = pricing_region
        if (pricing_country := _dict.get('pricing_country')) is not None:
            args['pricing_country'] = pricing_country
        else:
            raise ValueError('Required property \'pricing_country\' not present in InstanceUsage JSON')
        if (currency_code := _dict.get('currency_code')) is not None:
            args['currency_code'] = currency_code
        else:
            raise ValueError('Required property \'currency_code\' not present in InstanceUsage JSON')
        if (billable := _dict.get('billable')) is not None:
            args['billable'] = billable
        else:
            raise ValueError('Required property \'billable\' not present in InstanceUsage JSON')
        if (parent_resource_instance_id := _dict.get('parent_resource_instance_id')) is not None:
            args['parent_resource_instance_id'] = parent_resource_instance_id
        if (plan_id := _dict.get('plan_id')) is not None:
            args['plan_id'] = plan_id
        else:
            raise ValueError('Required property \'plan_id\' not present in InstanceUsage JSON')
        if (plan_name := _dict.get('plan_name')) is not None:
            args['plan_name'] = plan_name
        if (pricing_plan_id := _dict.get('pricing_plan_id')) is not None:
            args['pricing_plan_id'] = pricing_plan_id
        if (month := _dict.get('month')) is not None:
            args['month'] = month
        else:
            raise ValueError('Required property \'month\' not present in InstanceUsage JSON')
        if (usage := _dict.get('usage')) is not None:
            args['usage'] = [Metric.from_dict(v) for v in usage]
        else:
            raise ValueError('Required property \'usage\' not present in InstanceUsage JSON')
        if (pending := _dict.get('pending')) is not None:
            args['pending'] = pending
        if (currency_rate := _dict.get('currency_rate')) is not None:
            args['currency_rate'] = currency_rate
        if (tags := _dict.get('tags')) is not None:
            args['tags'] = tags
        if (service_tags := _dict.get('service_tags')) is not None:
            args['service_tags'] = service_tags
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a InstanceUsage object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'resource_instance_id') and self.resource_instance_id is not None:
            _dict['resource_instance_id'] = self.resource_instance_id
        if hasattr(self, 'resource_instance_name') and self.resource_instance_name is not None:
            _dict['resource_instance_name'] = self.resource_instance_name
        if hasattr(self, 'resource_id') and self.resource_id is not None:
            _dict['resource_id'] = self.resource_id
        if hasattr(self, 'catalog_id') and self.catalog_id is not None:
            _dict['catalog_id'] = self.catalog_id
        if hasattr(self, 'resource_name') and self.resource_name is not None:
            _dict['resource_name'] = self.resource_name
        if hasattr(self, 'resource_group_id') and self.resource_group_id is not None:
            _dict['resource_group_id'] = self.resource_group_id
        if hasattr(self, 'resource_group_name') and self.resource_group_name is not None:
            _dict['resource_group_name'] = self.resource_group_name
        if hasattr(self, 'organization_id') and self.organization_id is not None:
            _dict['organization_id'] = self.organization_id
        if hasattr(self, 'organization_name') and self.organization_name is not None:
            _dict['organization_name'] = self.organization_name
        if hasattr(self, 'space_id') and self.space_id is not None:
            _dict['space_id'] = self.space_id
        if hasattr(self, 'space_name') and self.space_name is not None:
            _dict['space_name'] = self.space_name
        if hasattr(self, 'consumer_id') and self.consumer_id is not None:
            _dict['consumer_id'] = self.consumer_id
        if hasattr(self, 'region') and self.region is not None:
            _dict['region'] = self.region
        if hasattr(self, 'pricing_region') and self.pricing_region is not None:
            _dict['pricing_region'] = self.pricing_region
        if hasattr(self, 'pricing_country') and self.pricing_country is not None:
            _dict['pricing_country'] = self.pricing_country
        if hasattr(self, 'currency_code') and self.currency_code is not None:
            _dict['currency_code'] = self.currency_code
        if hasattr(self, 'billable') and self.billable is not None:
            _dict['billable'] = self.billable
        if hasattr(self, 'parent_resource_instance_id') and self.parent_resource_instance_id is not None:
            _dict['parent_resource_instance_id'] = self.parent_resource_instance_id
        if hasattr(self, 'plan_id') and self.plan_id is not None:
            _dict['plan_id'] = self.plan_id
        if hasattr(self, 'plan_name') and self.plan_name is not None:
            _dict['plan_name'] = self.plan_name
        if hasattr(self, 'pricing_plan_id') and self.pricing_plan_id is not None:
            _dict['pricing_plan_id'] = self.pricing_plan_id
        if hasattr(self, 'month') and self.month is not None:
            _dict['month'] = self.month
        if hasattr(self, 'usage') and self.usage is not None:
            usage_list = []
            for v in self.usage:
                if isinstance(v, dict):
                    usage_list.append(v)
                else:
                    usage_list.append(v.to_dict())
            _dict['usage'] = usage_list
        if hasattr(self, 'pending') and self.pending is not None:
            _dict['pending'] = self.pending
        if hasattr(self, 'currency_rate') and self.currency_rate is not None:
            _dict['currency_rate'] = self.currency_rate
        if hasattr(self, 'tags') and self.tags is not None:
            _dict['tags'] = self.tags
        if hasattr(self, 'service_tags') and self.service_tags is not None:
            _dict['service_tags'] = self.service_tags
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this InstanceUsage object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'InstanceUsage') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'InstanceUsage') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class InstancesUsageFirst:
    """
    The link to the first page of the search query.

    :param str href: (optional) A link to a page of query results.
    """

    def __init__(
        self,
        *,
        href: Optional[str] = None,
    ) -> None:
        """
        Initialize a InstancesUsageFirst object.

        :param str href: (optional) A link to a page of query results.
        """
        self.href = href

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'InstancesUsageFirst':
        """Initialize a InstancesUsageFirst object from a json dictionary."""
        args = {}
        if (href := _dict.get('href')) is not None:
            args['href'] = href
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a InstancesUsageFirst object from a json dictionary."""
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
        """Return a `str` version of this InstancesUsageFirst object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'InstancesUsageFirst') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'InstancesUsageFirst') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class InstancesUsageNext:
    """
    The link to the next page of the search query.

    :param str href: (optional) A link to a page of query results.
    :param str offset: (optional) The value of the `_start` query parameter to fetch
          the next page.
    """

    def __init__(
        self,
        *,
        href: Optional[str] = None,
        offset: Optional[str] = None,
    ) -> None:
        """
        Initialize a InstancesUsageNext object.

        :param str href: (optional) A link to a page of query results.
        :param str offset: (optional) The value of the `_start` query parameter to
               fetch the next page.
        """
        self.href = href
        self.offset = offset

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'InstancesUsageNext':
        """Initialize a InstancesUsageNext object from a json dictionary."""
        args = {}
        if (href := _dict.get('href')) is not None:
            args['href'] = href
        if (offset := _dict.get('offset')) is not None:
            args['offset'] = offset
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a InstancesUsageNext object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'href') and self.href is not None:
            _dict['href'] = self.href
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this InstancesUsageNext object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'InstancesUsageNext') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'InstancesUsageNext') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class InstancesUsage:
    """
    The list of instance usage reports.

    :param int limit: (optional) The max number of reports in the response.
    :param int count: (optional) The number of reports in the response.
    :param InstancesUsageFirst first: (optional) The link to the first page of the
          search query.
    :param InstancesUsageNext next: (optional) The link to the next page of the
          search query.
    :param List[InstanceUsage] resources: (optional) The list of instance usage
          reports.
    """

    def __init__(
        self,
        *,
        limit: Optional[int] = None,
        count: Optional[int] = None,
        first: Optional['InstancesUsageFirst'] = None,
        next: Optional['InstancesUsageNext'] = None,
        resources: Optional[List['InstanceUsage']] = None,
    ) -> None:
        """
        Initialize a InstancesUsage object.

        :param int limit: (optional) The max number of reports in the response.
        :param int count: (optional) The number of reports in the response.
        :param InstancesUsageFirst first: (optional) The link to the first page of
               the search query.
        :param InstancesUsageNext next: (optional) The link to the next page of the
               search query.
        :param List[InstanceUsage] resources: (optional) The list of instance usage
               reports.
        """
        self.limit = limit
        self.count = count
        self.first = first
        self.next = next
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'InstancesUsage':
        """Initialize a InstancesUsage object from a json dictionary."""
        args = {}
        if (limit := _dict.get('limit')) is not None:
            args['limit'] = limit
        if (count := _dict.get('count')) is not None:
            args['count'] = count
        if (first := _dict.get('first')) is not None:
            args['first'] = InstancesUsageFirst.from_dict(first)
        if (next := _dict.get('next')) is not None:
            args['next'] = InstancesUsageNext.from_dict(next)
        if (resources := _dict.get('resources')) is not None:
            args['resources'] = [InstanceUsage.from_dict(v) for v in resources]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a InstancesUsage object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'count') and self.count is not None:
            _dict['count'] = self.count
        if hasattr(self, 'first') and self.first is not None:
            if isinstance(self.first, dict):
                _dict['first'] = self.first
            else:
                _dict['first'] = self.first.to_dict()
        if hasattr(self, 'next') and self.next is not None:
            if isinstance(self.next, dict):
                _dict['next'] = self.next
            else:
                _dict['next'] = self.next.to_dict()
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
        """Return a `str` version of this InstancesUsage object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'InstancesUsage') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'InstancesUsage') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Metric:
    """
    Information about a metric.

    :param str metric: The ID of the metric.
    :param str metric_name: (optional) The name of the metric.
    :param float quantity: The aggregated value for the metric.
    :param float rateable_quantity: (optional) The quantity that is used for
          calculating charges.
    :param float cost: The cost incurred by the metric.
    :param float rated_cost: Pre-discounted cost incurred by the metric.
    :param List[object] price: (optional) The price with which the cost was
          calculated.
    :param str unit: (optional) The unit that qualifies the quantity.
    :param str unit_name: (optional) The name of the unit.
    :param bool non_chargeable: (optional) When set to `true`, the cost is for
          informational purpose and is not included while calculating the plan charges.
    :param List[Discount] discounts: All the discounts applicable to the metric.
    """

    def __init__(
        self,
        metric: str,
        quantity: float,
        cost: float,
        rated_cost: float,
        discounts: List['Discount'],
        *,
        metric_name: Optional[str] = None,
        rateable_quantity: Optional[float] = None,
        price: Optional[List[object]] = None,
        unit: Optional[str] = None,
        unit_name: Optional[str] = None,
        non_chargeable: Optional[bool] = None,
    ) -> None:
        """
        Initialize a Metric object.

        :param str metric: The ID of the metric.
        :param float quantity: The aggregated value for the metric.
        :param float cost: The cost incurred by the metric.
        :param float rated_cost: Pre-discounted cost incurred by the metric.
        :param List[Discount] discounts: All the discounts applicable to the
               metric.
        :param str metric_name: (optional) The name of the metric.
        :param float rateable_quantity: (optional) The quantity that is used for
               calculating charges.
        :param List[object] price: (optional) The price with which the cost was
               calculated.
        :param str unit: (optional) The unit that qualifies the quantity.
        :param str unit_name: (optional) The name of the unit.
        :param bool non_chargeable: (optional) When set to `true`, the cost is for
               informational purpose and is not included while calculating the plan
               charges.
        """
        self.metric = metric
        self.metric_name = metric_name
        self.quantity = quantity
        self.rateable_quantity = rateable_quantity
        self.cost = cost
        self.rated_cost = rated_cost
        self.price = price
        self.unit = unit
        self.unit_name = unit_name
        self.non_chargeable = non_chargeable
        self.discounts = discounts

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Metric':
        """Initialize a Metric object from a json dictionary."""
        args = {}
        if (metric := _dict.get('metric')) is not None:
            args['metric'] = metric
        else:
            raise ValueError('Required property \'metric\' not present in Metric JSON')
        if (metric_name := _dict.get('metric_name')) is not None:
            args['metric_name'] = metric_name
        if (quantity := _dict.get('quantity')) is not None:
            args['quantity'] = quantity
        else:
            raise ValueError('Required property \'quantity\' not present in Metric JSON')
        if (rateable_quantity := _dict.get('rateable_quantity')) is not None:
            args['rateable_quantity'] = rateable_quantity
        if (cost := _dict.get('cost')) is not None:
            args['cost'] = cost
        else:
            raise ValueError('Required property \'cost\' not present in Metric JSON')
        if (rated_cost := _dict.get('rated_cost')) is not None:
            args['rated_cost'] = rated_cost
        else:
            raise ValueError('Required property \'rated_cost\' not present in Metric JSON')
        if (price := _dict.get('price')) is not None:
            args['price'] = price
        if (unit := _dict.get('unit')) is not None:
            args['unit'] = unit
        if (unit_name := _dict.get('unit_name')) is not None:
            args['unit_name'] = unit_name
        if (non_chargeable := _dict.get('non_chargeable')) is not None:
            args['non_chargeable'] = non_chargeable
        if (discounts := _dict.get('discounts')) is not None:
            args['discounts'] = [Discount.from_dict(v) for v in discounts]
        else:
            raise ValueError('Required property \'discounts\' not present in Metric JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Metric object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'metric') and self.metric is not None:
            _dict['metric'] = self.metric
        if hasattr(self, 'metric_name') and self.metric_name is not None:
            _dict['metric_name'] = self.metric_name
        if hasattr(self, 'quantity') and self.quantity is not None:
            _dict['quantity'] = self.quantity
        if hasattr(self, 'rateable_quantity') and self.rateable_quantity is not None:
            _dict['rateable_quantity'] = self.rateable_quantity
        if hasattr(self, 'cost') and self.cost is not None:
            _dict['cost'] = self.cost
        if hasattr(self, 'rated_cost') and self.rated_cost is not None:
            _dict['rated_cost'] = self.rated_cost
        if hasattr(self, 'price') and self.price is not None:
            _dict['price'] = self.price
        if hasattr(self, 'unit') and self.unit is not None:
            _dict['unit'] = self.unit
        if hasattr(self, 'unit_name') and self.unit_name is not None:
            _dict['unit_name'] = self.unit_name
        if hasattr(self, 'non_chargeable') and self.non_chargeable is not None:
            _dict['non_chargeable'] = self.non_chargeable
        if hasattr(self, 'discounts') and self.discounts is not None:
            discounts_list = []
            for v in self.discounts:
                if isinstance(v, dict):
                    discounts_list.append(v)
                else:
                    discounts_list.append(v.to_dict())
            _dict['discounts'] = discounts_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Metric object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Metric') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Metric') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Offer:
    """
    Information about an individual offer.

    :param str offer_id: The ID of the offer.
    :param float credits_total: The total credits before applying the offer.
    :param str offer_template: The template with which the offer was generated.
    :param datetime valid_from: The date from which the offer is valid.
    :param datetime expires_on: The date until the offer is valid.
    :param OfferCredits credits: Credit information related to an offer.
    """

    def __init__(
        self,
        offer_id: str,
        credits_total: float,
        offer_template: str,
        valid_from: datetime,
        expires_on: datetime,
        credits: 'OfferCredits',
    ) -> None:
        """
        Initialize a Offer object.

        :param str offer_id: The ID of the offer.
        :param float credits_total: The total credits before applying the offer.
        :param str offer_template: The template with which the offer was generated.
        :param datetime valid_from: The date from which the offer is valid.
        :param datetime expires_on: The date until the offer is valid.
        :param OfferCredits credits: Credit information related to an offer.
        """
        self.offer_id = offer_id
        self.credits_total = credits_total
        self.offer_template = offer_template
        self.valid_from = valid_from
        self.expires_on = expires_on
        self.credits = credits

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Offer':
        """Initialize a Offer object from a json dictionary."""
        args = {}
        if (offer_id := _dict.get('offer_id')) is not None:
            args['offer_id'] = offer_id
        else:
            raise ValueError('Required property \'offer_id\' not present in Offer JSON')
        if (credits_total := _dict.get('credits_total')) is not None:
            args['credits_total'] = credits_total
        else:
            raise ValueError('Required property \'credits_total\' not present in Offer JSON')
        if (offer_template := _dict.get('offer_template')) is not None:
            args['offer_template'] = offer_template
        else:
            raise ValueError('Required property \'offer_template\' not present in Offer JSON')
        if (valid_from := _dict.get('valid_from')) is not None:
            args['valid_from'] = string_to_datetime(valid_from)
        else:
            raise ValueError('Required property \'valid_from\' not present in Offer JSON')
        if (expires_on := _dict.get('expires_on')) is not None:
            args['expires_on'] = string_to_datetime(expires_on)
        else:
            raise ValueError('Required property \'expires_on\' not present in Offer JSON')
        if (credits := _dict.get('credits')) is not None:
            args['credits'] = OfferCredits.from_dict(credits)
        else:
            raise ValueError('Required property \'credits\' not present in Offer JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Offer object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'offer_id') and self.offer_id is not None:
            _dict['offer_id'] = self.offer_id
        if hasattr(self, 'credits_total') and self.credits_total is not None:
            _dict['credits_total'] = self.credits_total
        if hasattr(self, 'offer_template') and self.offer_template is not None:
            _dict['offer_template'] = self.offer_template
        if hasattr(self, 'valid_from') and self.valid_from is not None:
            _dict['valid_from'] = datetime_to_string(self.valid_from)
        if hasattr(self, 'expires_on') and self.expires_on is not None:
            _dict['expires_on'] = datetime_to_string(self.expires_on)
        if hasattr(self, 'credits') and self.credits is not None:
            if isinstance(self.credits, dict):
                _dict['credits'] = self.credits
            else:
                _dict['credits'] = self.credits.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Offer object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Offer') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Offer') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class OfferCredits:
    """
    Credit information related to an offer.

    :param float starting_balance: The available credits in the offer at the
          beginning of the month.
    :param float used: The credits used in this month.
    :param float balance: The remaining credits in the offer.
    """

    def __init__(
        self,
        starting_balance: float,
        used: float,
        balance: float,
    ) -> None:
        """
        Initialize a OfferCredits object.

        :param float starting_balance: The available credits in the offer at the
               beginning of the month.
        :param float used: The credits used in this month.
        :param float balance: The remaining credits in the offer.
        """
        self.starting_balance = starting_balance
        self.used = used
        self.balance = balance

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'OfferCredits':
        """Initialize a OfferCredits object from a json dictionary."""
        args = {}
        if (starting_balance := _dict.get('starting_balance')) is not None:
            args['starting_balance'] = starting_balance
        else:
            raise ValueError('Required property \'starting_balance\' not present in OfferCredits JSON')
        if (used := _dict.get('used')) is not None:
            args['used'] = used
        else:
            raise ValueError('Required property \'used\' not present in OfferCredits JSON')
        if (balance := _dict.get('balance')) is not None:
            args['balance'] = balance
        else:
            raise ValueError('Required property \'balance\' not present in OfferCredits JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a OfferCredits object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'starting_balance') and self.starting_balance is not None:
            _dict['starting_balance'] = self.starting_balance
        if hasattr(self, 'used') and self.used is not None:
            _dict['used'] = self.used
        if hasattr(self, 'balance') and self.balance is not None:
            _dict['balance'] = self.balance
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this OfferCredits object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'OfferCredits') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'OfferCredits') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class OrgUsage:
    """
    The aggregated usage and charges for all the plans in the org.

    :param str account_id: The ID of the account.
    :param str organization_id: The ID of the organization.
    :param str organization_name: (optional) The name of the organization.
    :param str pricing_country: The target country pricing that should be used.
    :param str currency_code: The currency for the cost fields in the resources,
          plans and metrics.
    :param str month: The month.
    :param List[Resource] resources: All the resource used in the account.
    :param float currency_rate: (optional) The value of the account's currency in
          USD.
    """

    def __init__(
        self,
        account_id: str,
        organization_id: str,
        pricing_country: str,
        currency_code: str,
        month: str,
        resources: List['Resource'],
        *,
        organization_name: Optional[str] = None,
        currency_rate: Optional[float] = None,
    ) -> None:
        """
        Initialize a OrgUsage object.

        :param str account_id: The ID of the account.
        :param str organization_id: The ID of the organization.
        :param str pricing_country: The target country pricing that should be used.
        :param str currency_code: The currency for the cost fields in the
               resources, plans and metrics.
        :param str month: The month.
        :param List[Resource] resources: All the resource used in the account.
        :param str organization_name: (optional) The name of the organization.
        :param float currency_rate: (optional) The value of the account's currency
               in USD.
        """
        self.account_id = account_id
        self.organization_id = organization_id
        self.organization_name = organization_name
        self.pricing_country = pricing_country
        self.currency_code = currency_code
        self.month = month
        self.resources = resources
        self.currency_rate = currency_rate

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'OrgUsage':
        """Initialize a OrgUsage object from a json dictionary."""
        args = {}
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        else:
            raise ValueError('Required property \'account_id\' not present in OrgUsage JSON')
        if (organization_id := _dict.get('organization_id')) is not None:
            args['organization_id'] = organization_id
        else:
            raise ValueError('Required property \'organization_id\' not present in OrgUsage JSON')
        if (organization_name := _dict.get('organization_name')) is not None:
            args['organization_name'] = organization_name
        if (pricing_country := _dict.get('pricing_country')) is not None:
            args['pricing_country'] = pricing_country
        else:
            raise ValueError('Required property \'pricing_country\' not present in OrgUsage JSON')
        if (currency_code := _dict.get('currency_code')) is not None:
            args['currency_code'] = currency_code
        else:
            raise ValueError('Required property \'currency_code\' not present in OrgUsage JSON')
        if (month := _dict.get('month')) is not None:
            args['month'] = month
        else:
            raise ValueError('Required property \'month\' not present in OrgUsage JSON')
        if (resources := _dict.get('resources')) is not None:
            args['resources'] = [Resource.from_dict(v) for v in resources]
        else:
            raise ValueError('Required property \'resources\' not present in OrgUsage JSON')
        if (currency_rate := _dict.get('currency_rate')) is not None:
            args['currency_rate'] = currency_rate
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a OrgUsage object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'organization_id') and self.organization_id is not None:
            _dict['organization_id'] = self.organization_id
        if hasattr(self, 'organization_name') and self.organization_name is not None:
            _dict['organization_name'] = self.organization_name
        if hasattr(self, 'pricing_country') and self.pricing_country is not None:
            _dict['pricing_country'] = self.pricing_country
        if hasattr(self, 'currency_code') and self.currency_code is not None:
            _dict['currency_code'] = self.currency_code
        if hasattr(self, 'month') and self.month is not None:
            _dict['month'] = self.month
        if hasattr(self, 'resources') and self.resources is not None:
            resources_list = []
            for v in self.resources:
                if isinstance(v, dict):
                    resources_list.append(v)
                else:
                    resources_list.append(v.to_dict())
            _dict['resources'] = resources_list
        if hasattr(self, 'currency_rate') and self.currency_rate is not None:
            _dict['currency_rate'] = self.currency_rate
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this OrgUsage object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'OrgUsage') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'OrgUsage') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Plan:
    """
    The aggregated values for the plan.

    :param str plan_id: The ID of the plan.
    :param str plan_name: (optional) The name of the plan.
    :param str pricing_region: (optional) The pricing region for the plan.
    :param str pricing_plan_id: (optional) The ID of the pricing plan used to rate
          the usage.
    :param bool billable: Indicates if the plan charges are billed to the customer.
    :param float cost: The total cost incurred by the plan.
    :param float rated_cost: Total pre-discounted cost incurred by the plan.
    :param List[Metric] usage: All the metrics in the plan.
    :param List[Discount] discounts: All the discounts applicable to the plan.
    :param bool pending: (optional) Pending charge from classic infrastructure.
    """

    def __init__(
        self,
        plan_id: str,
        billable: bool,
        cost: float,
        rated_cost: float,
        usage: List['Metric'],
        discounts: List['Discount'],
        *,
        plan_name: Optional[str] = None,
        pricing_region: Optional[str] = None,
        pricing_plan_id: Optional[str] = None,
        pending: Optional[bool] = None,
    ) -> None:
        """
        Initialize a Plan object.

        :param str plan_id: The ID of the plan.
        :param bool billable: Indicates if the plan charges are billed to the
               customer.
        :param float cost: The total cost incurred by the plan.
        :param float rated_cost: Total pre-discounted cost incurred by the plan.
        :param List[Metric] usage: All the metrics in the plan.
        :param List[Discount] discounts: All the discounts applicable to the plan.
        :param str plan_name: (optional) The name of the plan.
        :param str pricing_region: (optional) The pricing region for the plan.
        :param str pricing_plan_id: (optional) The ID of the pricing plan used to
               rate the usage.
        :param bool pending: (optional) Pending charge from classic infrastructure.
        """
        self.plan_id = plan_id
        self.plan_name = plan_name
        self.pricing_region = pricing_region
        self.pricing_plan_id = pricing_plan_id
        self.billable = billable
        self.cost = cost
        self.rated_cost = rated_cost
        self.usage = usage
        self.discounts = discounts
        self.pending = pending

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Plan':
        """Initialize a Plan object from a json dictionary."""
        args = {}
        if (plan_id := _dict.get('plan_id')) is not None:
            args['plan_id'] = plan_id
        else:
            raise ValueError('Required property \'plan_id\' not present in Plan JSON')
        if (plan_name := _dict.get('plan_name')) is not None:
            args['plan_name'] = plan_name
        if (pricing_region := _dict.get('pricing_region')) is not None:
            args['pricing_region'] = pricing_region
        if (pricing_plan_id := _dict.get('pricing_plan_id')) is not None:
            args['pricing_plan_id'] = pricing_plan_id
        if (billable := _dict.get('billable')) is not None:
            args['billable'] = billable
        else:
            raise ValueError('Required property \'billable\' not present in Plan JSON')
        if (cost := _dict.get('cost')) is not None:
            args['cost'] = cost
        else:
            raise ValueError('Required property \'cost\' not present in Plan JSON')
        if (rated_cost := _dict.get('rated_cost')) is not None:
            args['rated_cost'] = rated_cost
        else:
            raise ValueError('Required property \'rated_cost\' not present in Plan JSON')
        if (usage := _dict.get('usage')) is not None:
            args['usage'] = [Metric.from_dict(v) for v in usage]
        else:
            raise ValueError('Required property \'usage\' not present in Plan JSON')
        if (discounts := _dict.get('discounts')) is not None:
            args['discounts'] = [Discount.from_dict(v) for v in discounts]
        else:
            raise ValueError('Required property \'discounts\' not present in Plan JSON')
        if (pending := _dict.get('pending')) is not None:
            args['pending'] = pending
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Plan object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'plan_id') and self.plan_id is not None:
            _dict['plan_id'] = self.plan_id
        if hasattr(self, 'plan_name') and self.plan_name is not None:
            _dict['plan_name'] = self.plan_name
        if hasattr(self, 'pricing_region') and self.pricing_region is not None:
            _dict['pricing_region'] = self.pricing_region
        if hasattr(self, 'pricing_plan_id') and self.pricing_plan_id is not None:
            _dict['pricing_plan_id'] = self.pricing_plan_id
        if hasattr(self, 'billable') and self.billable is not None:
            _dict['billable'] = self.billable
        if hasattr(self, 'cost') and self.cost is not None:
            _dict['cost'] = self.cost
        if hasattr(self, 'rated_cost') and self.rated_cost is not None:
            _dict['rated_cost'] = self.rated_cost
        if hasattr(self, 'usage') and self.usage is not None:
            usage_list = []
            for v in self.usage:
                if isinstance(v, dict):
                    usage_list.append(v)
                else:
                    usage_list.append(v.to_dict())
            _dict['usage'] = usage_list
        if hasattr(self, 'discounts') and self.discounts is not None:
            discounts_list = []
            for v in self.discounts:
                if isinstance(v, dict):
                    discounts_list.append(v)
                else:
                    discounts_list.append(v.to_dict())
            _dict['discounts'] = discounts_list
        if hasattr(self, 'pending') and self.pending is not None:
            _dict['pending'] = self.pending
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Plan object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Plan') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Plan') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Resource:
    """
    The container for all the plans in the resource.

    :param str resource_id: The ID of the resource.
    :param str catalog_id: (optional) The catalog ID of the resource.
    :param str resource_name: (optional) The name of the resource.
    :param float billable_cost: The billable charges for the account.
    :param float billable_rated_cost: The pre-discounted billable charges for the
          account.
    :param float non_billable_cost: The non-billable charges for the account.
    :param float non_billable_rated_cost: The pre-discounted non-billable charges
          for the account.
    :param List[Plan] plans: All the plans in the resource.
    :param List[Discount] discounts: All the discounts applicable to the resource.
    """

    def __init__(
        self,
        resource_id: str,
        billable_cost: float,
        billable_rated_cost: float,
        non_billable_cost: float,
        non_billable_rated_cost: float,
        plans: List['Plan'],
        discounts: List['Discount'],
        *,
        catalog_id: Optional[str] = None,
        resource_name: Optional[str] = None,
    ) -> None:
        """
        Initialize a Resource object.

        :param str resource_id: The ID of the resource.
        :param float billable_cost: The billable charges for the account.
        :param float billable_rated_cost: The pre-discounted billable charges for
               the account.
        :param float non_billable_cost: The non-billable charges for the account.
        :param float non_billable_rated_cost: The pre-discounted non-billable
               charges for the account.
        :param List[Plan] plans: All the plans in the resource.
        :param List[Discount] discounts: All the discounts applicable to the
               resource.
        :param str catalog_id: (optional) The catalog ID of the resource.
        :param str resource_name: (optional) The name of the resource.
        """
        self.resource_id = resource_id
        self.catalog_id = catalog_id
        self.resource_name = resource_name
        self.billable_cost = billable_cost
        self.billable_rated_cost = billable_rated_cost
        self.non_billable_cost = non_billable_cost
        self.non_billable_rated_cost = non_billable_rated_cost
        self.plans = plans
        self.discounts = discounts

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Resource':
        """Initialize a Resource object from a json dictionary."""
        args = {}
        if (resource_id := _dict.get('resource_id')) is not None:
            args['resource_id'] = resource_id
        else:
            raise ValueError('Required property \'resource_id\' not present in Resource JSON')
        if (catalog_id := _dict.get('catalog_id')) is not None:
            args['catalog_id'] = catalog_id
        if (resource_name := _dict.get('resource_name')) is not None:
            args['resource_name'] = resource_name
        if (billable_cost := _dict.get('billable_cost')) is not None:
            args['billable_cost'] = billable_cost
        else:
            raise ValueError('Required property \'billable_cost\' not present in Resource JSON')
        if (billable_rated_cost := _dict.get('billable_rated_cost')) is not None:
            args['billable_rated_cost'] = billable_rated_cost
        else:
            raise ValueError('Required property \'billable_rated_cost\' not present in Resource JSON')
        if (non_billable_cost := _dict.get('non_billable_cost')) is not None:
            args['non_billable_cost'] = non_billable_cost
        else:
            raise ValueError('Required property \'non_billable_cost\' not present in Resource JSON')
        if (non_billable_rated_cost := _dict.get('non_billable_rated_cost')) is not None:
            args['non_billable_rated_cost'] = non_billable_rated_cost
        else:
            raise ValueError('Required property \'non_billable_rated_cost\' not present in Resource JSON')
        if (plans := _dict.get('plans')) is not None:
            args['plans'] = [Plan.from_dict(v) for v in plans]
        else:
            raise ValueError('Required property \'plans\' not present in Resource JSON')
        if (discounts := _dict.get('discounts')) is not None:
            args['discounts'] = [Discount.from_dict(v) for v in discounts]
        else:
            raise ValueError('Required property \'discounts\' not present in Resource JSON')
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
        if hasattr(self, 'catalog_id') and self.catalog_id is not None:
            _dict['catalog_id'] = self.catalog_id
        if hasattr(self, 'resource_name') and self.resource_name is not None:
            _dict['resource_name'] = self.resource_name
        if hasattr(self, 'billable_cost') and self.billable_cost is not None:
            _dict['billable_cost'] = self.billable_cost
        if hasattr(self, 'billable_rated_cost') and self.billable_rated_cost is not None:
            _dict['billable_rated_cost'] = self.billable_rated_cost
        if hasattr(self, 'non_billable_cost') and self.non_billable_cost is not None:
            _dict['non_billable_cost'] = self.non_billable_cost
        if hasattr(self, 'non_billable_rated_cost') and self.non_billable_rated_cost is not None:
            _dict['non_billable_rated_cost'] = self.non_billable_rated_cost
        if hasattr(self, 'plans') and self.plans is not None:
            plans_list = []
            for v in self.plans:
                if isinstance(v, dict):
                    plans_list.append(v)
                else:
                    plans_list.append(v.to_dict())
            _dict['plans'] = plans_list
        if hasattr(self, 'discounts') and self.discounts is not None:
            discounts_list = []
            for v in self.discounts:
                if isinstance(v, dict):
                    discounts_list.append(v)
                else:
                    discounts_list.append(v.to_dict())
            _dict['discounts'] = discounts_list
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


class ResourceGroupUsage:
    """
    The aggregated usage and charges for all the plans in the resource group.

    :param str account_id: The ID of the account.
    :param str resource_group_id: The ID of the resource group.
    :param str resource_group_name: (optional) The name of the resource group.
    :param str pricing_country: The target country pricing that should be used.
    :param str currency_code: The currency for the cost fields in the resources,
          plans and metrics.
    :param str month: The month.
    :param List[Resource] resources: All the resource used in the account.
    :param float currency_rate: (optional) The value of the account's currency in
          USD.
    """

    def __init__(
        self,
        account_id: str,
        resource_group_id: str,
        pricing_country: str,
        currency_code: str,
        month: str,
        resources: List['Resource'],
        *,
        resource_group_name: Optional[str] = None,
        currency_rate: Optional[float] = None,
    ) -> None:
        """
        Initialize a ResourceGroupUsage object.

        :param str account_id: The ID of the account.
        :param str resource_group_id: The ID of the resource group.
        :param str pricing_country: The target country pricing that should be used.
        :param str currency_code: The currency for the cost fields in the
               resources, plans and metrics.
        :param str month: The month.
        :param List[Resource] resources: All the resource used in the account.
        :param str resource_group_name: (optional) The name of the resource group.
        :param float currency_rate: (optional) The value of the account's currency
               in USD.
        """
        self.account_id = account_id
        self.resource_group_id = resource_group_id
        self.resource_group_name = resource_group_name
        self.pricing_country = pricing_country
        self.currency_code = currency_code
        self.month = month
        self.resources = resources
        self.currency_rate = currency_rate

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourceGroupUsage':
        """Initialize a ResourceGroupUsage object from a json dictionary."""
        args = {}
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        else:
            raise ValueError('Required property \'account_id\' not present in ResourceGroupUsage JSON')
        if (resource_group_id := _dict.get('resource_group_id')) is not None:
            args['resource_group_id'] = resource_group_id
        else:
            raise ValueError('Required property \'resource_group_id\' not present in ResourceGroupUsage JSON')
        if (resource_group_name := _dict.get('resource_group_name')) is not None:
            args['resource_group_name'] = resource_group_name
        if (pricing_country := _dict.get('pricing_country')) is not None:
            args['pricing_country'] = pricing_country
        else:
            raise ValueError('Required property \'pricing_country\' not present in ResourceGroupUsage JSON')
        if (currency_code := _dict.get('currency_code')) is not None:
            args['currency_code'] = currency_code
        else:
            raise ValueError('Required property \'currency_code\' not present in ResourceGroupUsage JSON')
        if (month := _dict.get('month')) is not None:
            args['month'] = month
        else:
            raise ValueError('Required property \'month\' not present in ResourceGroupUsage JSON')
        if (resources := _dict.get('resources')) is not None:
            args['resources'] = [Resource.from_dict(v) for v in resources]
        else:
            raise ValueError('Required property \'resources\' not present in ResourceGroupUsage JSON')
        if (currency_rate := _dict.get('currency_rate')) is not None:
            args['currency_rate'] = currency_rate
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourceGroupUsage object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'resource_group_id') and self.resource_group_id is not None:
            _dict['resource_group_id'] = self.resource_group_id
        if hasattr(self, 'resource_group_name') and self.resource_group_name is not None:
            _dict['resource_group_name'] = self.resource_group_name
        if hasattr(self, 'pricing_country') and self.pricing_country is not None:
            _dict['pricing_country'] = self.pricing_country
        if hasattr(self, 'currency_code') and self.currency_code is not None:
            _dict['currency_code'] = self.currency_code
        if hasattr(self, 'month') and self.month is not None:
            _dict['month'] = self.month
        if hasattr(self, 'resources') and self.resources is not None:
            resources_list = []
            for v in self.resources:
                if isinstance(v, dict):
                    resources_list.append(v)
                else:
                    resources_list.append(v.to_dict())
            _dict['resources'] = resources_list
        if hasattr(self, 'currency_rate') and self.currency_rate is not None:
            _dict['currency_rate'] = self.currency_rate
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResourceGroupUsage object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourceGroupUsage') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourceGroupUsage') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ResourcesSummary:
    """
    Charges related to cloud resources.

    :param float billable_cost: The billable charges for all cloud resources used in
          the account.
    :param float non_billable_cost: Non-billable charges for all cloud resources
          used in the account.
    """

    def __init__(
        self,
        billable_cost: float,
        non_billable_cost: float,
    ) -> None:
        """
        Initialize a ResourcesSummary object.

        :param float billable_cost: The billable charges for all cloud resources
               used in the account.
        :param float non_billable_cost: Non-billable charges for all cloud
               resources used in the account.
        """
        self.billable_cost = billable_cost
        self.non_billable_cost = non_billable_cost

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourcesSummary':
        """Initialize a ResourcesSummary object from a json dictionary."""
        args = {}
        if (billable_cost := _dict.get('billable_cost')) is not None:
            args['billable_cost'] = billable_cost
        else:
            raise ValueError('Required property \'billable_cost\' not present in ResourcesSummary JSON')
        if (non_billable_cost := _dict.get('non_billable_cost')) is not None:
            args['non_billable_cost'] = non_billable_cost
        else:
            raise ValueError('Required property \'non_billable_cost\' not present in ResourcesSummary JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourcesSummary object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'billable_cost') and self.billable_cost is not None:
            _dict['billable_cost'] = self.billable_cost
        if hasattr(self, 'non_billable_cost') and self.non_billable_cost is not None:
            _dict['non_billable_cost'] = self.non_billable_cost
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResourcesSummary object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourcesSummary') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourcesSummary') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class SnapshotConfigHistoryItem:
    """
    SnapshotConfigHistoryItem.

    :param float start_time: (optional) Timestamp in milliseconds when the snapshot
          configuration was created.
    :param float end_time: (optional) Timestamp in milliseconds when the snapshot
          configuration ends.
    :param str updated_by: (optional) Account that updated the billing snapshot
          configuration.
    :param str account_id: (optional) Account ID for which billing report snapshot
          is configured.
    :param str state: (optional) Status of the billing snapshot configuration.
          Possible values are [enabled, disabled].
    :param str account_type: (optional) Type of account. Possible values
          [enterprise, account].
    :param str interval: (optional) Frequency of taking the snapshot of the billing
          reports.
    :param str versioning: (optional) A new version of report is created or the
          existing report version is overwritten with every update.
    :param List[str] report_types: (optional) The type of billing reports to take
          snapshot of. Possible values are [account_summary, enterprise_summary,
          account_resource_instance_usage].
    :param str compression: (optional) Compression format of the snapshot report.
    :param str content_type: (optional) Type of content stored in snapshot report.
    :param str cos_reports_folder: (optional) The billing reports root folder to
          store the billing reports snapshots. Defaults to "IBMCloud-Billing-Reports".
    :param str cos_bucket: (optional) The name of the COS bucket to store the
          snapshot of the billing reports.
    :param str cos_location: (optional) Region of the COS instance.
    :param str cos_endpoint: (optional) The endpoint of the COS instance.
    """

    def __init__(
        self,
        *,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        updated_by: Optional[str] = None,
        account_id: Optional[str] = None,
        state: Optional[str] = None,
        account_type: Optional[str] = None,
        interval: Optional[str] = None,
        versioning: Optional[str] = None,
        report_types: Optional[List[str]] = None,
        compression: Optional[str] = None,
        content_type: Optional[str] = None,
        cos_reports_folder: Optional[str] = None,
        cos_bucket: Optional[str] = None,
        cos_location: Optional[str] = None,
        cos_endpoint: Optional[str] = None,
    ) -> None:
        """
        Initialize a SnapshotConfigHistoryItem object.

        :param float start_time: (optional) Timestamp in milliseconds when the
               snapshot configuration was created.
        :param float end_time: (optional) Timestamp in milliseconds when the
               snapshot configuration ends.
        :param str updated_by: (optional) Account that updated the billing snapshot
               configuration.
        :param str account_id: (optional) Account ID for which billing report
               snapshot is configured.
        :param str state: (optional) Status of the billing snapshot configuration.
               Possible values are [enabled, disabled].
        :param str account_type: (optional) Type of account. Possible values
               [enterprise, account].
        :param str interval: (optional) Frequency of taking the snapshot of the
               billing reports.
        :param str versioning: (optional) A new version of report is created or the
               existing report version is overwritten with every update.
        :param List[str] report_types: (optional) The type of billing reports to
               take snapshot of. Possible values are [account_summary, enterprise_summary,
               account_resource_instance_usage].
        :param str compression: (optional) Compression format of the snapshot
               report.
        :param str content_type: (optional) Type of content stored in snapshot
               report.
        :param str cos_reports_folder: (optional) The billing reports root folder
               to store the billing reports snapshots. Defaults to
               "IBMCloud-Billing-Reports".
        :param str cos_bucket: (optional) The name of the COS bucket to store the
               snapshot of the billing reports.
        :param str cos_location: (optional) Region of the COS instance.
        :param str cos_endpoint: (optional) The endpoint of the COS instance.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.updated_by = updated_by
        self.account_id = account_id
        self.state = state
        self.account_type = account_type
        self.interval = interval
        self.versioning = versioning
        self.report_types = report_types
        self.compression = compression
        self.content_type = content_type
        self.cos_reports_folder = cos_reports_folder
        self.cos_bucket = cos_bucket
        self.cos_location = cos_location
        self.cos_endpoint = cos_endpoint

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SnapshotConfigHistoryItem':
        """Initialize a SnapshotConfigHistoryItem object from a json dictionary."""
        args = {}
        if (start_time := _dict.get('start_time')) is not None:
            args['start_time'] = start_time
        if (end_time := _dict.get('end_time')) is not None:
            args['end_time'] = end_time
        if (updated_by := _dict.get('updated_by')) is not None:
            args['updated_by'] = updated_by
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        if (state := _dict.get('state')) is not None:
            args['state'] = state
        if (account_type := _dict.get('account_type')) is not None:
            args['account_type'] = account_type
        if (interval := _dict.get('interval')) is not None:
            args['interval'] = interval
        if (versioning := _dict.get('versioning')) is not None:
            args['versioning'] = versioning
        if (report_types := _dict.get('report_types')) is not None:
            args['report_types'] = report_types
        if (compression := _dict.get('compression')) is not None:
            args['compression'] = compression
        if (content_type := _dict.get('content_type')) is not None:
            args['content_type'] = content_type
        if (cos_reports_folder := _dict.get('cos_reports_folder')) is not None:
            args['cos_reports_folder'] = cos_reports_folder
        if (cos_bucket := _dict.get('cos_bucket')) is not None:
            args['cos_bucket'] = cos_bucket
        if (cos_location := _dict.get('cos_location')) is not None:
            args['cos_location'] = cos_location
        if (cos_endpoint := _dict.get('cos_endpoint')) is not None:
            args['cos_endpoint'] = cos_endpoint
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SnapshotConfigHistoryItem object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'start_time') and self.start_time is not None:
            _dict['start_time'] = self.start_time
        if hasattr(self, 'end_time') and self.end_time is not None:
            _dict['end_time'] = self.end_time
        if hasattr(self, 'updated_by') and self.updated_by is not None:
            _dict['updated_by'] = self.updated_by
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        if hasattr(self, 'account_type') and self.account_type is not None:
            _dict['account_type'] = self.account_type
        if hasattr(self, 'interval') and self.interval is not None:
            _dict['interval'] = self.interval
        if hasattr(self, 'versioning') and self.versioning is not None:
            _dict['versioning'] = self.versioning
        if hasattr(self, 'report_types') and self.report_types is not None:
            _dict['report_types'] = self.report_types
        if hasattr(self, 'compression') and self.compression is not None:
            _dict['compression'] = self.compression
        if hasattr(self, 'content_type') and self.content_type is not None:
            _dict['content_type'] = self.content_type
        if hasattr(self, 'cos_reports_folder') and self.cos_reports_folder is not None:
            _dict['cos_reports_folder'] = self.cos_reports_folder
        if hasattr(self, 'cos_bucket') and self.cos_bucket is not None:
            _dict['cos_bucket'] = self.cos_bucket
        if hasattr(self, 'cos_location') and self.cos_location is not None:
            _dict['cos_location'] = self.cos_location
        if hasattr(self, 'cos_endpoint') and self.cos_endpoint is not None:
            _dict['cos_endpoint'] = self.cos_endpoint
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this SnapshotConfigHistoryItem object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SnapshotConfigHistoryItem') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SnapshotConfigHistoryItem') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class StateEnum(str, Enum):
        """
        Status of the billing snapshot configuration. Possible values are [enabled,
        disabled].
        """

        ENABLED = 'enabled'
        DISABLED = 'disabled'

    class AccountTypeEnum(str, Enum):
        """
        Type of account. Possible values [enterprise, account].
        """

        ACCOUNT = 'account'
        ENTERPRISE = 'enterprise'

    class IntervalEnum(str, Enum):
        """
        Frequency of taking the snapshot of the billing reports.
        """

        DAILY = 'daily'

    class VersioningEnum(str, Enum):
        """
        A new version of report is created or the existing report version is overwritten
        with every update.
        """

        NEW = 'new'
        OVERWRITE = 'overwrite'

    class ReportTypesEnum(str, Enum):
        """
        report_types.
        """

        ACCOUNT_SUMMARY = 'account_summary'
        ENTERPRISE_SUMMARY = 'enterprise_summary'
        ACCOUNT_RESOURCE_INSTANCE_USAGE = 'account_resource_instance_usage'


class SnapshotList:
    """
    List of billing reports snapshots.

    :param float count: (optional) Number of total snapshots.
    :param SnapshotListFirst first: (optional) Reference to the first page of the
          search query.
    :param SnapshotListNext next: (optional) Reference to the next page of the
          search query if any.
    :param List[SnapshotListSnapshotsItem] snapshots: (optional)
    """

    def __init__(
        self,
        *,
        count: Optional[float] = None,
        first: Optional['SnapshotListFirst'] = None,
        next: Optional['SnapshotListNext'] = None,
        snapshots: Optional[List['SnapshotListSnapshotsItem']] = None,
    ) -> None:
        """
        Initialize a SnapshotList object.

        :param float count: (optional) Number of total snapshots.
        :param SnapshotListFirst first: (optional) Reference to the first page of
               the search query.
        :param SnapshotListNext next: (optional) Reference to the next page of the
               search query if any.
        :param List[SnapshotListSnapshotsItem] snapshots: (optional)
        """
        self.count = count
        self.first = first
        self.next = next
        self.snapshots = snapshots

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SnapshotList':
        """Initialize a SnapshotList object from a json dictionary."""
        args = {}
        if (count := _dict.get('count')) is not None:
            args['count'] = count
        if (first := _dict.get('first')) is not None:
            args['first'] = SnapshotListFirst.from_dict(first)
        if (next := _dict.get('next')) is not None:
            args['next'] = SnapshotListNext.from_dict(next)
        if (snapshots := _dict.get('snapshots')) is not None:
            args['snapshots'] = [SnapshotListSnapshotsItem.from_dict(v) for v in snapshots]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SnapshotList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'count') and self.count is not None:
            _dict['count'] = self.count
        if hasattr(self, 'first') and self.first is not None:
            if isinstance(self.first, dict):
                _dict['first'] = self.first
            else:
                _dict['first'] = self.first.to_dict()
        if hasattr(self, 'next') and self.next is not None:
            if isinstance(self.next, dict):
                _dict['next'] = self.next
            else:
                _dict['next'] = self.next.to_dict()
        if hasattr(self, 'snapshots') and self.snapshots is not None:
            snapshots_list = []
            for v in self.snapshots:
                if isinstance(v, dict):
                    snapshots_list.append(v)
                else:
                    snapshots_list.append(v.to_dict())
            _dict['snapshots'] = snapshots_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this SnapshotList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SnapshotList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SnapshotList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class SnapshotListFirst:
    """
    Reference to the first page of the search query.

    :param str href: (optional)
    """

    def __init__(
        self,
        *,
        href: Optional[str] = None,
    ) -> None:
        """
        Initialize a SnapshotListFirst object.

        :param str href: (optional)
        """
        self.href = href

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SnapshotListFirst':
        """Initialize a SnapshotListFirst object from a json dictionary."""
        args = {}
        if (href := _dict.get('href')) is not None:
            args['href'] = href
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SnapshotListFirst object from a json dictionary."""
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
        """Return a `str` version of this SnapshotListFirst object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SnapshotListFirst') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SnapshotListFirst') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class SnapshotListNext:
    """
    Reference to the next page of the search query if any.

    :attr str href: (optional)
    :attr str offset: (optional)
    """

    def __init__(
        self,
        *,
        href: str = None,
        offset: str = None,
    ) -> None:
        """
        Initialize a SnapshotListNext object.

        :param str href: (optional)
        :param str offset: (optional)
        """
        self.href = href
        self.offset = offset

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SnapshotListNext':
        """Initialize a SnapshotListNext object from a json dictionary."""
        args = {}
        if (href := _dict.get('href')) is not None:
            args['href'] = href
        if (offset := _dict.get('offset')) is not None:
            args['offset'] = offset
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SnapshotListNext object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'href') and self.href is not None:
            _dict['href'] = self.href
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this SnapshotListNext object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SnapshotListNext') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SnapshotListNext') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class SnapshotListSnapshotsItem:
    """
    Snapshot Schema.

    :param str account_id: (optional) Account ID for which billing report snapshot
          is configured.
    :param str month: (optional) Month of captured snapshot.
    :param str account_type: (optional) Type of account. Possible values are
          [enterprise, account].
    :param float expected_processed_at: (optional) Timestamp of snapshot processed.
    :param str state: (optional) Status of the billing snapshot configuration.
          Possible values are [enabled, disabled].
    :param SnapshotListSnapshotsItemBillingPeriod billing_period: (optional) Period
          of billing in snapshot.
    :param str snapshot_id: (optional) Id of the snapshot captured.
    :param str charset: (optional) Character encoding used.
    :param str compression: (optional) Compression format of the snapshot report.
    :param str content_type: (optional) Type of content stored in snapshot report.
    :param str bucket: (optional) The name of the COS bucket to store the snapshot
          of the billing reports.
    :param str version: (optional) Version of the snapshot.
    :param str created_on: (optional) Date and time of creation of snapshot.
    :param List[SnapshotListSnapshotsItemReportTypesItem] report_types: (optional)
          List of report types configured for the snapshot.
    :param List[SnapshotListSnapshotsItemFilesItem] files: (optional) List of
          location of reports.
    :param float processed_at: (optional) Timestamp at which snapshot is captured.
    """

    def __init__(
        self,
        *,
        account_id: Optional[str] = None,
        month: Optional[str] = None,
        account_type: Optional[str] = None,
        expected_processed_at: Optional[int] = None,
        state: Optional[str] = None,
        billing_period: Optional['SnapshotListSnapshotsItemBillingPeriod'] = None,
        snapshot_id: Optional[str] = None,
        charset: Optional[str] = None,
        compression: Optional[str] = None,
        content_type: Optional[str] = None,
        bucket: Optional[str] = None,
        version: Optional[str] = None,
        created_on: Optional[str] = None,
        report_types: Optional[List['SnapshotListSnapshotsItemReportTypesItem']] = None,
        files: Optional[List['SnapshotListSnapshotsItemFilesItem']] = None,
        processed_at: Optional[int] = None,
    ) -> None:
        """
        Initialize a SnapshotListSnapshotsItem object.

        :param str account_id: (optional) Account ID for which billing report
               snapshot is configured.
        :param str month: (optional) Month of captured snapshot.
        :param str account_type: (optional) Type of account. Possible values are
               [enterprise, account].
        :param int expected_processed_at: (optional) Timestamp of snapshot
               processed.
        :param str state: (optional) Status of the billing snapshot configuration.
               Possible values are [enabled, disabled].
        :param SnapshotListSnapshotsItemBillingPeriod billing_period: (optional)
               Period of billing in snapshot.
        :param str snapshot_id: (optional) Id of the snapshot captured.
        :param str charset: (optional) Character encoding used.
        :param str compression: (optional) Compression format of the snapshot
               report.
        :param str content_type: (optional) Type of content stored in snapshot
               report.
        :param str bucket: (optional) The name of the COS bucket to store the
               snapshot of the billing reports.
        :param str version: (optional) Version of the snapshot.
        :param str created_on: (optional) Date and time of creation of snapshot.
        :param List[SnapshotListSnapshotsItemReportTypesItem] report_types:
               (optional) List of report types configured for the snapshot.
        :param List[SnapshotListSnapshotsItemFilesItem] files: (optional) List of
               location of reports.
        :param int processed_at: (optional) Timestamp at which snapshot is
               captured.
        """
        self.account_id = account_id
        self.month = month
        self.account_type = account_type
        self.expected_processed_at = expected_processed_at
        self.state = state
        self.billing_period = billing_period
        self.snapshot_id = snapshot_id
        self.charset = charset
        self.compression = compression
        self.content_type = content_type
        self.bucket = bucket
        self.version = version
        self.created_on = created_on
        self.report_types = report_types
        self.files = files
        self.processed_at = processed_at

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SnapshotListSnapshotsItem':
        """Initialize a SnapshotListSnapshotsItem object from a json dictionary."""
        args = {}
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        if (month := _dict.get('month')) is not None:
            args['month'] = month
        if (account_type := _dict.get('account_type')) is not None:
            args['account_type'] = account_type
        if (expected_processed_at := _dict.get('expected_processed_at')) is not None:
            args['expected_processed_at'] = expected_processed_at
        if (state := _dict.get('state')) is not None:
            args['state'] = state
        if (billing_period := _dict.get('billing_period')) is not None:
            args['billing_period'] = SnapshotListSnapshotsItemBillingPeriod.from_dict(billing_period)
        if (snapshot_id := _dict.get('snapshot_id')) is not None:
            args['snapshot_id'] = snapshot_id
        if (charset := _dict.get('charset')) is not None:
            args['charset'] = charset
        if (compression := _dict.get('compression')) is not None:
            args['compression'] = compression
        if (content_type := _dict.get('content_type')) is not None:
            args['content_type'] = content_type
        if (bucket := _dict.get('bucket')) is not None:
            args['bucket'] = bucket
        if (version := _dict.get('version')) is not None:
            args['version'] = version
        if (created_on := _dict.get('created_on')) is not None:
            args['created_on'] = created_on
        if (report_types := _dict.get('report_types')) is not None:
            args['report_types'] = [SnapshotListSnapshotsItemReportTypesItem.from_dict(v) for v in report_types]
        if (files := _dict.get('files')) is not None:
            args['files'] = [SnapshotListSnapshotsItemFilesItem.from_dict(v) for v in files]
        if (processed_at := _dict.get('processed_at')) is not None:
            args['processed_at'] = processed_at
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SnapshotListSnapshotsItem object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'month') and self.month is not None:
            _dict['month'] = self.month
        if hasattr(self, 'account_type') and self.account_type is not None:
            _dict['account_type'] = self.account_type
        if hasattr(self, 'expected_processed_at') and self.expected_processed_at is not None:
            _dict['expected_processed_at'] = self.expected_processed_at
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        if hasattr(self, 'billing_period') and self.billing_period is not None:
            if isinstance(self.billing_period, dict):
                _dict['billing_period'] = self.billing_period
            else:
                _dict['billing_period'] = self.billing_period.to_dict()
        if hasattr(self, 'snapshot_id') and self.snapshot_id is not None:
            _dict['snapshot_id'] = self.snapshot_id
        if hasattr(self, 'charset') and self.charset is not None:
            _dict['charset'] = self.charset
        if hasattr(self, 'compression') and self.compression is not None:
            _dict['compression'] = self.compression
        if hasattr(self, 'content_type') and self.content_type is not None:
            _dict['content_type'] = self.content_type
        if hasattr(self, 'bucket') and self.bucket is not None:
            _dict['bucket'] = self.bucket
        if hasattr(self, 'version') and self.version is not None:
            _dict['version'] = self.version
        if hasattr(self, 'created_on') and self.created_on is not None:
            _dict['created_on'] = self.created_on
        if hasattr(self, 'report_types') and self.report_types is not None:
            report_types_list = []
            for v in self.report_types:
                if isinstance(v, dict):
                    report_types_list.append(v)
                else:
                    report_types_list.append(v.to_dict())
            _dict['report_types'] = report_types_list
        if hasattr(self, 'files') and self.files is not None:
            files_list = []
            for v in self.files:
                if isinstance(v, dict):
                    files_list.append(v)
                else:
                    files_list.append(v.to_dict())
            _dict['files'] = files_list
        if hasattr(self, 'processed_at') and self.processed_at is not None:
            _dict['processed_at'] = self.processed_at
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this SnapshotListSnapshotsItem object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SnapshotListSnapshotsItem') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SnapshotListSnapshotsItem') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class AccountTypeEnum(str, Enum):
        """
        Type of account. Possible values are [enterprise, account].
        """

        ACCOUNT = 'account'
        ENTERPRISE = 'enterprise'

    class StateEnum(str, Enum):
        """
        Status of the billing snapshot configuration. Possible values are [enabled,
        disabled].
        """

        ENABLED = 'enabled'
        DISABLED = 'disabled'


class SnapshotListSnapshotsItemBillingPeriod:
    """
    Period of billing in snapshot.

    :param str start: (optional) Date and time of start of billing in the respective
          snapshot.
    :param str end: (optional) Date and time of end of billing in the respective
          snapshot.
    """

    def __init__(
        self,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> None:
        """
        Initialize a SnapshotListSnapshotsItemBillingPeriod object.

        :param str start: (optional) Date and time of start of billing in the
               respective snapshot.
        :param str end: (optional) Date and time of end of billing in the
               respective snapshot.
        """
        self.start = start
        self.end = end

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SnapshotListSnapshotsItemBillingPeriod':
        """Initialize a SnapshotListSnapshotsItemBillingPeriod object from a json dictionary."""
        args = {}
        if (start := _dict.get('start')) is not None:
            args['start'] = start
        if (end := _dict.get('end')) is not None:
            args['end'] = end
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SnapshotListSnapshotsItemBillingPeriod object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'start') and self.start is not None:
            _dict['start'] = self.start
        if hasattr(self, 'end') and self.end is not None:
            _dict['end'] = self.end
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this SnapshotListSnapshotsItemBillingPeriod object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SnapshotListSnapshotsItemBillingPeriod') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SnapshotListSnapshotsItemBillingPeriod') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class SnapshotListSnapshotsItemFilesItem:
    """
    SnapshotListSnapshotsItemFilesItem.

    :param str report_types: (optional) The type of billing report stored. Possible
          values are [account_summary, enterprise_summary,
          account_resource_instance_usage].
    :param str location: (optional) Absolute path of the billing report in the COS
          instance.
    :param str account_id: (optional) Account ID for which billing report is
          captured.
    """

    def __init__(
        self,
        *,
        report_types: Optional[str] = None,
        location: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> None:
        """
        Initialize a SnapshotListSnapshotsItemFilesItem object.

        :param str report_types: (optional) The type of billing report stored.
               Possible values are [account_summary, enterprise_summary,
               account_resource_instance_usage].
        :param str location: (optional) Absolute path of the billing report in the
               COS instance.
        :param str account_id: (optional) Account ID for which billing report is
               captured.
        """
        self.report_types = report_types
        self.location = location
        self.account_id = account_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SnapshotListSnapshotsItemFilesItem':
        """Initialize a SnapshotListSnapshotsItemFilesItem object from a json dictionary."""
        args = {}
        if (report_types := _dict.get('report_types')) is not None:
            args['report_types'] = report_types
        if (location := _dict.get('location')) is not None:
            args['location'] = location
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SnapshotListSnapshotsItemFilesItem object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'report_types') and self.report_types is not None:
            _dict['report_types'] = self.report_types
        if hasattr(self, 'location') and self.location is not None:
            _dict['location'] = self.location
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this SnapshotListSnapshotsItemFilesItem object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SnapshotListSnapshotsItemFilesItem') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SnapshotListSnapshotsItemFilesItem') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class ReportTypesEnum(str, Enum):
        """
        The type of billing report stored. Possible values are [account_summary,
        enterprise_summary, account_resource_instance_usage].
        """

        ACCOUNT_SUMMARY = 'account_summary'
        ENTERPRISE_SUMMARY = 'enterprise_summary'
        ACCOUNT_RESOURCE_INSTANCE_USAGE = 'account_resource_instance_usage'


class SnapshotListSnapshotsItemReportTypesItem:
    """
    SnapshotListSnapshotsItemReportTypesItem.

    :param str type: (optional) The type of billing report of the snapshot. Possible
          values are [account_summary, enterprise_summary,
          account_resource_instance_usage].
    :param str version: (optional) Version of the snapshot.
    """

    def __init__(
        self,
        *,
        type: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None:
        """
        Initialize a SnapshotListSnapshotsItemReportTypesItem object.

        :param str type: (optional) The type of billing report of the snapshot.
               Possible values are [account_summary, enterprise_summary,
               account_resource_instance_usage].
        :param str version: (optional) Version of the snapshot.
        """
        self.type = type
        self.version = version

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SnapshotListSnapshotsItemReportTypesItem':
        """Initialize a SnapshotListSnapshotsItemReportTypesItem object from a json dictionary."""
        args = {}
        if (type := _dict.get('type')) is not None:
            args['type'] = type
        if (version := _dict.get('version')) is not None:
            args['version'] = version
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SnapshotListSnapshotsItemReportTypesItem object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'version') and self.version is not None:
            _dict['version'] = self.version
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this SnapshotListSnapshotsItemReportTypesItem object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SnapshotListSnapshotsItemReportTypesItem') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SnapshotListSnapshotsItemReportTypesItem') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class TypeEnum(str, Enum):
        """
        The type of billing report of the snapshot. Possible values are [account_summary,
        enterprise_summary, account_resource_instance_usage].
        """

        ACCOUNT_SUMMARY = 'account_summary'
        ENTERPRISE_SUMMARY = 'enterprise_summary'
        ACCOUNT_RESOURCE_INSTANCE_USAGE = 'account_resource_instance_usage'


class SnapshotConfig:
    """
    Billing reports snapshot configuration.

    :param str account_id: (optional) Account ID for which billing report snapshot
          is configured.
    :param str state: (optional) Status of the billing snapshot configuration.
          Possible values are [enabled, disabled].
    :param str account_type: (optional) Type of account. Possible values are
          [enterprise, account].
    :param str interval: (optional) Frequency of taking the snapshot of the billing
          reports.
    :param str versioning: (optional) A new version of report is created or the
          existing report version is overwritten with every update.
    :param List[str] report_types: (optional) The type of billing reports to take
          snapshot of. Possible values are [account_summary, enterprise_summary,
          account_resource_instance_usage].
    :param str compression: (optional) Compression format of the snapshot report.
    :param str content_type: (optional) Type of content stored in snapshot report.
    :param str cos_reports_folder: (optional) The billing reports root folder to
          store the billing reports snapshots. Defaults to "IBMCloud-Billing-Reports".
    :param str cos_bucket: (optional) The name of the COS bucket to store the
          snapshot of the billing reports.
    :param str cos_location: (optional) Region of the COS instance.
    :param str cos_endpoint: (optional) The endpoint of the COS instance.
    :param float created_at: (optional) Timestamp in milliseconds when the snapshot
          configuration was created.
    :param float last_updated_at: (optional) Timestamp in milliseconds when the
          snapshot configuration was last updated.
    :param List[SnapshotConfigHistoryItem] history: (optional) List of previous
          versions of the snapshot configurations.
    """

    def __init__(
        self,
        *,
        account_id: Optional[str] = None,
        state: Optional[str] = None,
        account_type: Optional[str] = None,
        interval: Optional[str] = None,
        versioning: Optional[str] = None,
        report_types: Optional[List[str]] = None,
        compression: Optional[str] = None,
        content_type: Optional[str] = None,
        cos_reports_folder: Optional[str] = None,
        cos_bucket: Optional[str] = None,
        cos_location: Optional[str] = None,
        cos_endpoint: Optional[str] = None,
        created_at: Optional[int] = None,
        last_updated_at: Optional[int] = None,
        history: Optional[List['SnapshotConfigHistoryItem']] = None,
    ) -> None:
        """
        Initialize a SnapshotConfig object.

        :param str account_id: (optional) Account ID for which billing report
               snapshot is configured.
        :param str state: (optional) Status of the billing snapshot configuration.
               Possible values are [enabled, disabled].
        :param str account_type: (optional) Type of account. Possible values are
               [enterprise, account].
        :param str interval: (optional) Frequency of taking the snapshot of the
               billing reports.
        :param str versioning: (optional) A new version of report is created or the
               existing report version is overwritten with every update.
        :param List[str] report_types: (optional) The type of billing reports to
               take snapshot of. Possible values are [account_summary, enterprise_summary,
               account_resource_instance_usage].
        :param str compression: (optional) Compression format of the snapshot
               report.
        :param str content_type: (optional) Type of content stored in snapshot
               report.
        :param str cos_reports_folder: (optional) The billing reports root folder
               to store the billing reports snapshots. Defaults to
               "IBMCloud-Billing-Reports".
        :param str cos_bucket: (optional) The name of the COS bucket to store the
               snapshot of the billing reports.
        :param str cos_location: (optional) Region of the COS instance.
        :param str cos_endpoint: (optional) The endpoint of the COS instance.
        :param int created_at: (optional) Timestamp in milliseconds when the
               snapshot configuration was created.
        :param int last_updated_at: (optional) Timestamp in milliseconds when the
               snapshot configuration was last updated.
        :param List[SnapshotConfigHistoryItem] history: (optional) List of previous
               versions of the snapshot configurations.
        """
        self.account_id = account_id
        self.state = state
        self.account_type = account_type
        self.interval = interval
        self.versioning = versioning
        self.report_types = report_types
        self.compression = compression
        self.content_type = content_type
        self.cos_reports_folder = cos_reports_folder
        self.cos_bucket = cos_bucket
        self.cos_location = cos_location
        self.cos_endpoint = cos_endpoint
        self.created_at = created_at
        self.last_updated_at = last_updated_at
        self.history = history

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SnapshotConfig':
        """Initialize a SnapshotConfig object from a json dictionary."""
        args = {}
        if (account_id := _dict.get('account_id')) is not None:
            args['account_id'] = account_id
        if (state := _dict.get('state')) is not None:
            args['state'] = state
        if (account_type := _dict.get('account_type')) is not None:
            args['account_type'] = account_type
        if (interval := _dict.get('interval')) is not None:
            args['interval'] = interval
        if (versioning := _dict.get('versioning')) is not None:
            args['versioning'] = versioning
        if (report_types := _dict.get('report_types')) is not None:
            args['report_types'] = report_types
        if (compression := _dict.get('compression')) is not None:
            args['compression'] = compression
        if (content_type := _dict.get('content_type')) is not None:
            args['content_type'] = content_type
        if (cos_reports_folder := _dict.get('cos_reports_folder')) is not None:
            args['cos_reports_folder'] = cos_reports_folder
        if (cos_bucket := _dict.get('cos_bucket')) is not None:
            args['cos_bucket'] = cos_bucket
        if (cos_location := _dict.get('cos_location')) is not None:
            args['cos_location'] = cos_location
        if (cos_endpoint := _dict.get('cos_endpoint')) is not None:
            args['cos_endpoint'] = cos_endpoint
        if (created_at := _dict.get('created_at')) is not None:
            args['created_at'] = created_at
        if (last_updated_at := _dict.get('last_updated_at')) is not None:
            args['last_updated_at'] = last_updated_at
        if (history := _dict.get('history')) is not None:
            args['history'] = [SnapshotConfigHistoryItem.from_dict(v) for v in history]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SnapshotConfig object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        if hasattr(self, 'account_type') and self.account_type is not None:
            _dict['account_type'] = self.account_type
        if hasattr(self, 'interval') and self.interval is not None:
            _dict['interval'] = self.interval
        if hasattr(self, 'versioning') and self.versioning is not None:
            _dict['versioning'] = self.versioning
        if hasattr(self, 'report_types') and self.report_types is not None:
            _dict['report_types'] = self.report_types
        if hasattr(self, 'compression') and self.compression is not None:
            _dict['compression'] = self.compression
        if hasattr(self, 'content_type') and self.content_type is not None:
            _dict['content_type'] = self.content_type
        if hasattr(self, 'cos_reports_folder') and self.cos_reports_folder is not None:
            _dict['cos_reports_folder'] = self.cos_reports_folder
        if hasattr(self, 'cos_bucket') and self.cos_bucket is not None:
            _dict['cos_bucket'] = self.cos_bucket
        if hasattr(self, 'cos_location') and self.cos_location is not None:
            _dict['cos_location'] = self.cos_location
        if hasattr(self, 'cos_endpoint') and self.cos_endpoint is not None:
            _dict['cos_endpoint'] = self.cos_endpoint
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = self.created_at
        if hasattr(self, 'last_updated_at') and self.last_updated_at is not None:
            _dict['last_updated_at'] = self.last_updated_at
        if hasattr(self, 'history') and self.history is not None:
            history_list = []
            for v in self.history:
                if isinstance(v, dict):
                    history_list.append(v)
                else:
                    history_list.append(v.to_dict())
            _dict['history'] = history_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this SnapshotConfig object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SnapshotConfig') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SnapshotConfig') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class StateEnum(str, Enum):
        """
        Status of the billing snapshot configuration. Possible values are [enabled,
        disabled].
        """

        ENABLED = 'enabled'
        DISABLED = 'disabled'

    class AccountTypeEnum(str, Enum):
        """
        Type of account. Possible values are [enterprise, account].
        """

        ACCOUNT = 'account'
        ENTERPRISE = 'enterprise'

    class IntervalEnum(str, Enum):
        """
        Frequency of taking the snapshot of the billing reports.
        """

        DAILY = 'daily'

    class VersioningEnum(str, Enum):
        """
        A new version of report is created or the existing report version is overwritten
        with every update.
        """

        NEW = 'new'
        OVERWRITE = 'overwrite'

    class ReportTypesEnum(str, Enum):
        """
        report_types.
        """

        ACCOUNT_SUMMARY = 'account_summary'
        ENTERPRISE_SUMMARY = 'enterprise_summary'
        ACCOUNT_RESOURCE_INSTANCE_USAGE = 'account_resource_instance_usage'


class SnapshotConfigValidateResponse:
    """
    Validated billing service to COS bucket authorization.

    :attr str account_id: (optional) Account ID for which billing report snapshot is
          configured.
    :attr str cos_bucket: (optional) The name of the COS bucket to store the
          snapshot of the billing reports.
    :attr str cos_location: (optional) Region of the COS instance.
    """

    def __init__(
        self,
        *,
        account_id: str = None,
        cos_bucket: str = None,
        cos_location: str = None,
    ) -> None:
        """
        Initialize a SnapshotConfigValidateResponse object.

        :param str account_id: (optional) Account ID for which billing report
               snapshot is configured.
        :param str cos_bucket: (optional) The name of the COS bucket to store the
               snapshot of the billing reports.
        :param str cos_location: (optional) Region of the COS instance.
        """
        self.account_id = account_id
        self.cos_bucket = cos_bucket
        self.cos_location = cos_location

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SnapshotConfigValidateResponse':
        """Initialize a SnapshotConfigValidateResponse object from a json dictionary."""
        args = {}
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        if 'cos_bucket' in _dict:
            args['cos_bucket'] = _dict.get('cos_bucket')
        if 'cos_location' in _dict:
            args['cos_location'] = _dict.get('cos_location')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SnapshotConfigValidateResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'cos_bucket') and self.cos_bucket is not None:
            _dict['cos_bucket'] = self.cos_bucket
        if hasattr(self, 'cos_location') and self.cos_location is not None:
            _dict['cos_location'] = self.cos_location
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this SnapshotConfigValidateResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SnapshotConfigValidateResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SnapshotConfigValidateResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Subscription:
    """
    Subscription.

    :param str subscription_id: The ID of the subscription.
    :param str charge_agreement_number: The charge agreement number of the
          subsciption.
    :param str type: Type of the subscription.
    :param float subscription_amount: The credits available in the subscription for
          the month.
    :param datetime start: The date from which the subscription was active.
    :param datetime end: (optional) The date until which the subscription is active.
          End time is unavailable for PayGO accounts.
    :param float credits_total: The total credits available in the subscription.
    :param List[SubscriptionTerm] terms: The terms through which the subscription is
          split into.
    """

    def __init__(
        self,
        subscription_id: str,
        charge_agreement_number: str,
        type: str,
        subscription_amount: float,
        start: datetime,
        credits_total: float,
        terms: List['SubscriptionTerm'],
        *,
        end: Optional[datetime] = None,
    ) -> None:
        """
        Initialize a Subscription object.

        :param str subscription_id: The ID of the subscription.
        :param str charge_agreement_number: The charge agreement number of the
               subsciption.
        :param str type: Type of the subscription.
        :param float subscription_amount: The credits available in the subscription
               for the month.
        :param datetime start: The date from which the subscription was active.
        :param float credits_total: The total credits available in the
               subscription.
        :param List[SubscriptionTerm] terms: The terms through which the
               subscription is split into.
        :param datetime end: (optional) The date until which the subscription is
               active. End time is unavailable for PayGO accounts.
        """
        self.subscription_id = subscription_id
        self.charge_agreement_number = charge_agreement_number
        self.type = type
        self.subscription_amount = subscription_amount
        self.start = start
        self.end = end
        self.credits_total = credits_total
        self.terms = terms

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Subscription':
        """Initialize a Subscription object from a json dictionary."""
        args = {}
        if (subscription_id := _dict.get('subscription_id')) is not None:
            args['subscription_id'] = subscription_id
        else:
            raise ValueError('Required property \'subscription_id\' not present in Subscription JSON')
        if (charge_agreement_number := _dict.get('charge_agreement_number')) is not None:
            args['charge_agreement_number'] = charge_agreement_number
        else:
            raise ValueError('Required property \'charge_agreement_number\' not present in Subscription JSON')
        if (type := _dict.get('type')) is not None:
            args['type'] = type
        else:
            raise ValueError('Required property \'type\' not present in Subscription JSON')
        if (subscription_amount := _dict.get('subscription_amount')) is not None:
            args['subscription_amount'] = subscription_amount
        else:
            raise ValueError('Required property \'subscription_amount\' not present in Subscription JSON')
        if (start := _dict.get('start')) is not None:
            args['start'] = string_to_datetime(start)
        else:
            raise ValueError('Required property \'start\' not present in Subscription JSON')
        if (end := _dict.get('end')) is not None:
            args['end'] = string_to_datetime(end)
        if (credits_total := _dict.get('credits_total')) is not None:
            args['credits_total'] = credits_total
        else:
            raise ValueError('Required property \'credits_total\' not present in Subscription JSON')
        if (terms := _dict.get('terms')) is not None:
            args['terms'] = [SubscriptionTerm.from_dict(v) for v in terms]
        else:
            raise ValueError('Required property \'terms\' not present in Subscription JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Subscription object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'subscription_id') and self.subscription_id is not None:
            _dict['subscription_id'] = self.subscription_id
        if hasattr(self, 'charge_agreement_number') and self.charge_agreement_number is not None:
            _dict['charge_agreement_number'] = self.charge_agreement_number
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'subscription_amount') and self.subscription_amount is not None:
            _dict['subscription_amount'] = self.subscription_amount
        if hasattr(self, 'start') and self.start is not None:
            _dict['start'] = datetime_to_string(self.start)
        if hasattr(self, 'end') and self.end is not None:
            _dict['end'] = datetime_to_string(self.end)
        if hasattr(self, 'credits_total') and self.credits_total is not None:
            _dict['credits_total'] = self.credits_total
        if hasattr(self, 'terms') and self.terms is not None:
            terms_list = []
            for v in self.terms:
                if isinstance(v, dict):
                    terms_list.append(v)
                else:
                    terms_list.append(v.to_dict())
            _dict['terms'] = terms_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Subscription object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Subscription') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Subscription') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class SubscriptionSummary:
    """
    A summary of charges and credits related to a subscription.

    :param float overage: (optional) The charges after exhausting subscription
          credits and offers credits.
    :param List[Subscription] subscriptions: (optional) The list of subscriptions
          applicable for the month.
    """

    def __init__(
        self,
        *,
        overage: Optional[float] = None,
        subscriptions: Optional[List['Subscription']] = None,
    ) -> None:
        """
        Initialize a SubscriptionSummary object.

        :param float overage: (optional) The charges after exhausting subscription
               credits and offers credits.
        :param List[Subscription] subscriptions: (optional) The list of
               subscriptions applicable for the month.
        """
        self.overage = overage
        self.subscriptions = subscriptions

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SubscriptionSummary':
        """Initialize a SubscriptionSummary object from a json dictionary."""
        args = {}
        if (overage := _dict.get('overage')) is not None:
            args['overage'] = overage
        if (subscriptions := _dict.get('subscriptions')) is not None:
            args['subscriptions'] = [Subscription.from_dict(v) for v in subscriptions]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SubscriptionSummary object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'overage') and self.overage is not None:
            _dict['overage'] = self.overage
        if hasattr(self, 'subscriptions') and self.subscriptions is not None:
            subscriptions_list = []
            for v in self.subscriptions:
                if isinstance(v, dict):
                    subscriptions_list.append(v)
                else:
                    subscriptions_list.append(v.to_dict())
            _dict['subscriptions'] = subscriptions_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this SubscriptionSummary object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SubscriptionSummary') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SubscriptionSummary') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class SubscriptionTerm:
    """
    SubscriptionTerm.

    :param datetime start: The start date of the term.
    :param datetime end: The end date of the term.
    :param SubscriptionTermCredits credits: Information about credits related to a
          subscription.
    """

    def __init__(
        self,
        start: datetime,
        end: datetime,
        credits: 'SubscriptionTermCredits',
    ) -> None:
        """
        Initialize a SubscriptionTerm object.

        :param datetime start: The start date of the term.
        :param datetime end: The end date of the term.
        :param SubscriptionTermCredits credits: Information about credits related
               to a subscription.
        """
        self.start = start
        self.end = end
        self.credits = credits

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SubscriptionTerm':
        """Initialize a SubscriptionTerm object from a json dictionary."""
        args = {}
        if (start := _dict.get('start')) is not None:
            args['start'] = string_to_datetime(start)
        else:
            raise ValueError('Required property \'start\' not present in SubscriptionTerm JSON')
        if (end := _dict.get('end')) is not None:
            args['end'] = string_to_datetime(end)
        else:
            raise ValueError('Required property \'end\' not present in SubscriptionTerm JSON')
        if (credits := _dict.get('credits')) is not None:
            args['credits'] = SubscriptionTermCredits.from_dict(credits)
        else:
            raise ValueError('Required property \'credits\' not present in SubscriptionTerm JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SubscriptionTerm object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'start') and self.start is not None:
            _dict['start'] = datetime_to_string(self.start)
        if hasattr(self, 'end') and self.end is not None:
            _dict['end'] = datetime_to_string(self.end)
        if hasattr(self, 'credits') and self.credits is not None:
            if isinstance(self.credits, dict):
                _dict['credits'] = self.credits
            else:
                _dict['credits'] = self.credits.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this SubscriptionTerm object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SubscriptionTerm') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SubscriptionTerm') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class SubscriptionTermCredits:
    """
    Information about credits related to a subscription.

    :param float total: The total credits available for the term.
    :param float starting_balance: The unused credits in the term at the beginning
          of the month.
    :param float used: The credits used in this month.
    :param float balance: The remaining credits in this term.
    """

    def __init__(
        self,
        total: float,
        starting_balance: float,
        used: float,
        balance: float,
    ) -> None:
        """
        Initialize a SubscriptionTermCredits object.

        :param float total: The total credits available for the term.
        :param float starting_balance: The unused credits in the term at the
               beginning of the month.
        :param float used: The credits used in this month.
        :param float balance: The remaining credits in this term.
        """
        self.total = total
        self.starting_balance = starting_balance
        self.used = used
        self.balance = balance

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SubscriptionTermCredits':
        """Initialize a SubscriptionTermCredits object from a json dictionary."""
        args = {}
        if (total := _dict.get('total')) is not None:
            args['total'] = total
        else:
            raise ValueError('Required property \'total\' not present in SubscriptionTermCredits JSON')
        if (starting_balance := _dict.get('starting_balance')) is not None:
            args['starting_balance'] = starting_balance
        else:
            raise ValueError('Required property \'starting_balance\' not present in SubscriptionTermCredits JSON')
        if (used := _dict.get('used')) is not None:
            args['used'] = used
        else:
            raise ValueError('Required property \'used\' not present in SubscriptionTermCredits JSON')
        if (balance := _dict.get('balance')) is not None:
            args['balance'] = balance
        else:
            raise ValueError('Required property \'balance\' not present in SubscriptionTermCredits JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SubscriptionTermCredits object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'total') and self.total is not None:
            _dict['total'] = self.total
        if hasattr(self, 'starting_balance') and self.starting_balance is not None:
            _dict['starting_balance'] = self.starting_balance
        if hasattr(self, 'used') and self.used is not None:
            _dict['used'] = self.used
        if hasattr(self, 'balance') and self.balance is not None:
            _dict['balance'] = self.balance
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this SubscriptionTermCredits object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SubscriptionTermCredits') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SubscriptionTermCredits') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class SupportSummary:
    """
    SupportSummary.

    :param float cost: The monthly support cost.
    :param str type: The type of support.
    :param float overage: Additional support cost for the month.
    """

    def __init__(
        self,
        cost: float,
        type: str,
        overage: float,
    ) -> None:
        """
        Initialize a SupportSummary object.

        :param float cost: The monthly support cost.
        :param str type: The type of support.
        :param float overage: Additional support cost for the month.
        """
        self.cost = cost
        self.type = type
        self.overage = overage

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SupportSummary':
        """Initialize a SupportSummary object from a json dictionary."""
        args = {}
        if (cost := _dict.get('cost')) is not None:
            args['cost'] = cost
        else:
            raise ValueError('Required property \'cost\' not present in SupportSummary JSON')
        if (type := _dict.get('type')) is not None:
            args['type'] = type
        else:
            raise ValueError('Required property \'type\' not present in SupportSummary JSON')
        if (overage := _dict.get('overage')) is not None:
            args['overage'] = overage
        else:
            raise ValueError('Required property \'overage\' not present in SupportSummary JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SupportSummary object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'cost') and self.cost is not None:
            _dict['cost'] = self.cost
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'overage') and self.overage is not None:
            _dict['overage'] = self.overage
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this SupportSummary object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SupportSummary') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SupportSummary') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


##############################################################################
# Pagers
##############################################################################


class GetResourceUsageAccountPager:
    """
    GetResourceUsageAccountPager can be used to simplify the use of the "get_resource_usage_account" method.
    """

    def __init__(
        self,
        *,
        client: UsageReportsV4,
        account_id: str,
        billingmonth: str,
        names: bool = None,
        tags: bool = None,
        accept_language: str = None,
        limit: int = None,
        resource_group_id: str = None,
        organization_id: str = None,
        resource_instance_id: str = None,
        resource_id: str = None,
        plan_id: str = None,
        region: str = None,
    ) -> None:
        """
        Initialize a GetResourceUsageAccountPager object.
        :param str account_id: Account ID for which the usage report is requested.
        :param str billingmonth: The billing month for which the usage report is
               requested.  Format is yyyy-mm.
        :param bool names: (optional) Include the name of every resource, plan,
               resource instance, organization, and resource group.
        :param bool tags: (optional) Include the tags associated with every
               resource instance. By default it is always `true`.
        :param str accept_language: (optional) Prioritize the names returned in the
               order of the specified languages. Language will default to English.
        :param int limit: (optional) Number of usage records returned. The default
               value is 30. Maximum value is 200.
        :param str resource_group_id: (optional) Filter by resource group.
        :param str organization_id: (optional) Filter by organization_id.
        :param str resource_instance_id: (optional) Filter by resource instance_id.
        :param str resource_id: (optional) Filter by resource_id.
        :param str plan_id: (optional) Filter by plan_id.
        :param str region: (optional) Region in which the resource instance is
               provisioned.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._account_id = account_id
        self._billingmonth = billingmonth
        self._names = names
        self._tags = tags
        self._accept_language = accept_language
        self._limit = limit
        self._resource_group_id = resource_group_id
        self._organization_id = organization_id
        self._resource_instance_id = resource_instance_id
        self._resource_id = resource_id
        self._plan_id = plan_id
        self._region = region

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of InstanceUsage.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.get_resource_usage_account(
            account_id=self._account_id,
            billingmonth=self._billingmonth,
            names=self._names,
            tags=self._tags,
            accept_language=self._accept_language,
            limit=self._limit,
            resource_group_id=self._resource_group_id,
            organization_id=self._organization_id,
            resource_instance_id=self._resource_instance_id,
            resource_id=self._resource_id,
            plan_id=self._plan_id,
            region=self._region,
            start=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next')
        if next_page_link is not None:
            next = get_query_param(next_page_link.get('href'), '_start')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('resources')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of InstanceUsage.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results


class GetResourceUsageResourceGroupPager:
    """
    GetResourceUsageResourceGroupPager can be used to simplify the use of the "get_resource_usage_resource_group" method.
    """

    def __init__(
        self,
        *,
        client: UsageReportsV4,
        account_id: str,
        resource_group_id: str,
        billingmonth: str,
        names: bool = None,
        tags: bool = None,
        accept_language: str = None,
        limit: int = None,
        resource_instance_id: str = None,
        resource_id: str = None,
        plan_id: str = None,
        region: str = None,
    ) -> None:
        """
        Initialize a GetResourceUsageResourceGroupPager object.
        :param str account_id: Account ID for which the usage report is requested.
        :param str resource_group_id: Resource group for which the usage report is
               requested.
        :param str billingmonth: The billing month for which the usage report is
               requested.  Format is yyyy-mm.
        :param bool names: (optional) Include the name of every resource, plan,
               resource instance, organization, and resource group.
        :param bool tags: (optional) Include the tags associated with every
               resource instance. By default it is always `true`.
        :param str accept_language: (optional) Prioritize the names returned in the
               order of the specified languages. Language will default to English.
        :param int limit: (optional) Number of usage records returned. The default
               value is 30. Maximum value is 200.
        :param str resource_instance_id: (optional) Filter by resource instance id.
        :param str resource_id: (optional) Filter by resource_id.
        :param str plan_id: (optional) Filter by plan_id.
        :param str region: (optional) Region in which the resource instance is
               provisioned.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._account_id = account_id
        self._resource_group_id = resource_group_id
        self._billingmonth = billingmonth
        self._names = names
        self._tags = tags
        self._accept_language = accept_language
        self._limit = limit
        self._resource_instance_id = resource_instance_id
        self._resource_id = resource_id
        self._plan_id = plan_id
        self._region = region

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of InstanceUsage.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.get_resource_usage_resource_group(
            account_id=self._account_id,
            resource_group_id=self._resource_group_id,
            billingmonth=self._billingmonth,
            names=self._names,
            tags=self._tags,
            accept_language=self._accept_language,
            limit=self._limit,
            resource_instance_id=self._resource_instance_id,
            resource_id=self._resource_id,
            plan_id=self._plan_id,
            region=self._region,
            start=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next')
        if next_page_link is not None:
            next = get_query_param(next_page_link.get('href'), '_start')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('resources')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of InstanceUsage.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results


class GetResourceUsageOrgPager:
    """
    GetResourceUsageOrgPager can be used to simplify the use of the "get_resource_usage_org" method.
    """

    def __init__(
        self,
        *,
        client: UsageReportsV4,
        account_id: str,
        organization_id: str,
        billingmonth: str,
        names: bool = None,
        tags: bool = None,
        accept_language: str = None,
        limit: int = None,
        resource_instance_id: str = None,
        resource_id: str = None,
        plan_id: str = None,
        region: str = None,
    ) -> None:
        """
        Initialize a GetResourceUsageOrgPager object.
        :param str account_id: Account ID for which the usage report is requested.
        :param str organization_id: ID of the organization.
        :param str billingmonth: The billing month for which the usage report is
               requested.  Format is yyyy-mm.
        :param bool names: (optional) Include the name of every resource, plan,
               resource instance, organization, and resource group.
        :param bool tags: (optional) Include the tags associated with every
               resource instance. By default it is always `true`.
        :param str accept_language: (optional) Prioritize the names returned in the
               order of the specified languages. Language will default to English.
        :param int limit: (optional) Number of usage records returned. The default
               value is 30. Maximum value is 200.
        :param str resource_instance_id: (optional) Filter by resource instance id.
        :param str resource_id: (optional) Filter by resource_id.
        :param str plan_id: (optional) Filter by plan_id.
        :param str region: (optional) Region in which the resource instance is
               provisioned.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._account_id = account_id
        self._organization_id = organization_id
        self._billingmonth = billingmonth
        self._names = names
        self._tags = tags
        self._accept_language = accept_language
        self._limit = limit
        self._resource_instance_id = resource_instance_id
        self._resource_id = resource_id
        self._plan_id = plan_id
        self._region = region

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of InstanceUsage.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.get_resource_usage_org(
            account_id=self._account_id,
            organization_id=self._organization_id,
            billingmonth=self._billingmonth,
            names=self._names,
            tags=self._tags,
            accept_language=self._accept_language,
            limit=self._limit,
            resource_instance_id=self._resource_instance_id,
            resource_id=self._resource_id,
            plan_id=self._plan_id,
            region=self._region,
            start=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next')
        if next_page_link is not None:
            next = get_query_param(next_page_link.get('href'), '_start')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('resources')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of InstanceUsage.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results


class GetReportsSnapshotPager:
    """
    GetReportsSnapshotPager can be used to simplify the use of the "get_reports_snapshot" method.
    """

    def __init__(
        self,
        *,
        client: UsageReportsV4,
        account_id: str,
        month: str,
        date_from: int = None,
        date_to: int = None,
        limit: int = None,
    ) -> None:
        """
        Initialize a GetReportsSnapshotPager object.
        :param str account_id: Account ID for which the billing report snapshot is
               requested.
        :param str month: The month for which billing report snapshot is requested.
                Format is yyyy-mm.
        :param int date_from: (optional) Timestamp in milliseconds for which
               billing report snapshot is requested.
        :param int date_to: (optional) Timestamp in milliseconds for which billing
               report snapshot is requested.
        :param int limit: (optional) Number of usage records returned. The default
               value is 30. Maximum value is 200.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._account_id = account_id
        self._month = month
        self._date_from = date_from
        self._date_to = date_to
        self._limit = limit

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of SnapshotListSnapshotsItem.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.get_reports_snapshot(
            account_id=self._account_id,
            month=self._month,
            date_from=self._date_from,
            date_to=self._date_to,
            limit=self._limit,
            start=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next')
        if next_page_link is not None:
            next = get_query_param(next_page_link.get('href'), '_start')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('snapshots')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of SnapshotListSnapshotsItem.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results
