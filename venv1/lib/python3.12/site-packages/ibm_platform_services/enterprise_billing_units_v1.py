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

# IBM OpenAPI SDK Code Generator Version: 3.64.1-cee95189-20230124-211647

"""
Billing units for IBM Cloud Enterprise

API Version: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List
import json

from ibm_cloud_sdk_core import BaseService, DetailedResponse, get_query_param
from ibm_cloud_sdk_core.authenticators.authenticator import Authenticator
from ibm_cloud_sdk_core.get_authenticator import get_authenticator_from_environment
from ibm_cloud_sdk_core.utils import datetime_to_string, string_to_datetime

from .common import get_sdk_headers

##############################################################################
# Service
##############################################################################


class EnterpriseBillingUnitsV1(BaseService):
    """The Enterprise Billing Units V1 service."""

    DEFAULT_SERVICE_URL = 'https://billing.cloud.ibm.com'
    DEFAULT_SERVICE_NAME = 'enterprise_billing_units'

    @classmethod
    def new_instance(
        cls,
        service_name: str = DEFAULT_SERVICE_NAME,
    ) -> 'EnterpriseBillingUnitsV1':
        """
        Return a new client for the Enterprise Billing Units service using the
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
        Construct a new client for the Enterprise Billing Units service.

        :param Authenticator authenticator: The authenticator specifies the authentication mechanism.
               Get up to date information from https://github.com/IBM/python-sdk-core/blob/main/README.md
               about initializing the authenticator of your choice.
        """
        BaseService.__init__(self, service_url=self.DEFAULT_SERVICE_URL, authenticator=authenticator)

    #########################
    # Billing Units
    #########################

    def get_billing_unit(self, billing_unit_id: str, **kwargs) -> DetailedResponse:
        """
        Get billing unit by ID.

        Return the billing unit information if it exists.

        :param str billing_unit_id: The ID of the requested billing unit.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `BillingUnit` object
        """

        if not billing_unit_id:
            raise ValueError('billing_unit_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_billing_unit'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['billing_unit_id']
        path_param_values = self.encode_path_vars(billing_unit_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v1/billing-units/{billing_unit_id}'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def list_billing_units(
        self,
        *,
        account_id: str = None,
        enterprise_id: str = None,
        account_group_id: str = None,
        limit: int = None,
        start: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        List billing units.

        Return matching billing unit information if any exists. Omits internal properties
        and enterprise account ID from the billing unit.

        :param str account_id: (optional) The enterprise account ID.
        :param str enterprise_id: (optional) The enterprise ID.
        :param str account_group_id: (optional) The account group ID.
        :param int limit: (optional) Return results up to this limit. Valid values
               are between 0 and 100.
        :param str start: (optional) The pagination offset. This represents the
               index of the first returned result.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `BillingUnitsList` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='list_billing_units'
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
            'enterprise_id': enterprise_id,
            'account_group_id': account_group_id,
            'limit': limit,
            'start': start,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/billing-units'
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    #########################
    # Billing Options
    #########################

    def list_billing_options(
        self, billing_unit_id: str, *, limit: int = None, start: str = None, **kwargs
    ) -> DetailedResponse:
        """
        List billing options.

        Return matching billing options if any exist. Show subscriptions and promotional
        offers that are available to a billing unit.

        :param str billing_unit_id: The billing unit ID.
        :param int limit: (optional) Return results up to this limit. Valid values
               are between 0 and 100.
        :param str start: (optional) The pagination offset. This represents the
               index of the first returned result.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `BillingOptionsList` object
        """

        if not billing_unit_id:
            raise ValueError('billing_unit_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='list_billing_options'
        )
        headers.update(sdk_headers)

        params = {
            'billing_unit_id': billing_unit_id,
            'limit': limit,
            'start': start,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/billing-options'
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    #########################
    # Credit Pools
    #########################

    def get_credit_pools(
        self,
        billing_unit_id: str,
        *,
        date: str = None,
        type: str = None,
        limit: int = None,
        start: str = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get credit pools.

        Get credit pools for a billing unit. Credit pools can be either platform or
        support credit pools. The platform credit pool contains credit from platform
        subscriptions and promotional offers. The support credit pool contains credit from
        support subscriptions.

        :param str billing_unit_id: The ID of the billing unit.
        :param str date: (optional) The date in the format of YYYY-MM.
        :param str type: (optional) Filters the credit pool by type, either
               `PLATFORM` or `SUPPORT`.
        :param int limit: (optional) Return results up to this limit. Valid values
               are between 0 and 100.
        :param str start: (optional) The pagination offset. This represents the
               index of the first returned result.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `CreditPoolsList` object
        """

        if not billing_unit_id:
            raise ValueError('billing_unit_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_credit_pools'
        )
        headers.update(sdk_headers)

        params = {
            'billing_unit_id': billing_unit_id,
            'date': date,
            'type': type,
            'limit': limit,
            'start': start,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/credit-pools'
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response


##############################################################################
# Models
##############################################################################


class BillingOption:
    """
    Information about a billing option.

    :attr str id: (optional) The ID of the billing option.
    :attr str billing_unit_id: (optional) The ID of the billing unit that's
          associated with the billing option.
    :attr datetime start_date: (optional) The start date of billing option.
    :attr datetime end_date: (optional) The end date of billing option.
    :attr str state: (optional) The state of the billing option. The valid values
          include `ACTIVE, `SUSPENDED`, and `CANCELED`.
    :attr str type: (optional) The type of billing option. The valid values are
          `SUBSCRIPTION` and `OFFER`.
    :attr str category: (optional) The category of the billing option. The valid
          values are `PLATFORM`, `SERVICE`, and `SUPPORT`.
    :attr dict payment_instrument: (optional) The payment method for support.
    :attr int duration_in_months: (optional) The duration of the billing options in
          months.
    :attr int line_item_id: (optional) The line item ID for support.
    :attr dict billing_system: (optional) The support billing system.
    :attr str renewal_mode_code: (optional) The renewal code for support. This code
          denotes whether the subscription automatically renews, is assessed monthly, and
          so on.
    :attr datetime updated_at: (optional) The date when the billing option was
          updated.
    """

    def __init__(
        self,
        *,
        id: str = None,
        billing_unit_id: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        state: str = None,
        type: str = None,
        category: str = None,
        payment_instrument: dict = None,
        duration_in_months: int = None,
        line_item_id: int = None,
        billing_system: dict = None,
        renewal_mode_code: str = None,
        updated_at: datetime = None,
    ) -> None:
        """
        Initialize a BillingOption object.

        :param str id: (optional) The ID of the billing option.
        :param str billing_unit_id: (optional) The ID of the billing unit that's
               associated with the billing option.
        :param datetime start_date: (optional) The start date of billing option.
        :param datetime end_date: (optional) The end date of billing option.
        :param str state: (optional) The state of the billing option. The valid
               values include `ACTIVE, `SUSPENDED`, and `CANCELED`.
        :param str type: (optional) The type of billing option. The valid values
               are `SUBSCRIPTION` and `OFFER`.
        :param str category: (optional) The category of the billing option. The
               valid values are `PLATFORM`, `SERVICE`, and `SUPPORT`.
        :param dict payment_instrument: (optional) The payment method for support.
        :param int duration_in_months: (optional) The duration of the billing
               options in months.
        :param int line_item_id: (optional) The line item ID for support.
        :param dict billing_system: (optional) The support billing system.
        :param str renewal_mode_code: (optional) The renewal code for support. This
               code denotes whether the subscription automatically renews, is assessed
               monthly, and so on.
        :param datetime updated_at: (optional) The date when the billing option was
               updated.
        """
        self.id = id
        self.billing_unit_id = billing_unit_id
        self.start_date = start_date
        self.end_date = end_date
        self.state = state
        self.type = type
        self.category = category
        self.payment_instrument = payment_instrument
        self.duration_in_months = duration_in_months
        self.line_item_id = line_item_id
        self.billing_system = billing_system
        self.renewal_mode_code = renewal_mode_code
        self.updated_at = updated_at

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'BillingOption':
        """Initialize a BillingOption object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'billing_unit_id' in _dict:
            args['billing_unit_id'] = _dict.get('billing_unit_id')
        if 'start_date' in _dict:
            args['start_date'] = string_to_datetime(_dict.get('start_date'))
        if 'end_date' in _dict:
            args['end_date'] = string_to_datetime(_dict.get('end_date'))
        if 'state' in _dict:
            args['state'] = _dict.get('state')
        if 'type' in _dict:
            args['type'] = _dict.get('type')
        if 'category' in _dict:
            args['category'] = _dict.get('category')
        if 'payment_instrument' in _dict:
            args['payment_instrument'] = _dict.get('payment_instrument')
        if 'duration_in_months' in _dict:
            args['duration_in_months'] = _dict.get('duration_in_months')
        if 'line_item_id' in _dict:
            args['line_item_id'] = _dict.get('line_item_id')
        if 'billing_system' in _dict:
            args['billing_system'] = _dict.get('billing_system')
        if 'renewal_mode_code' in _dict:
            args['renewal_mode_code'] = _dict.get('renewal_mode_code')
        if 'updated_at' in _dict:
            args['updated_at'] = string_to_datetime(_dict.get('updated_at'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a BillingOption object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'billing_unit_id') and self.billing_unit_id is not None:
            _dict['billing_unit_id'] = self.billing_unit_id
        if hasattr(self, 'start_date') and self.start_date is not None:
            _dict['start_date'] = datetime_to_string(self.start_date)
        if hasattr(self, 'end_date') and self.end_date is not None:
            _dict['end_date'] = datetime_to_string(self.end_date)
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'category') and self.category is not None:
            _dict['category'] = self.category
        if hasattr(self, 'payment_instrument') and self.payment_instrument is not None:
            _dict['payment_instrument'] = self.payment_instrument
        if hasattr(self, 'duration_in_months') and self.duration_in_months is not None:
            _dict['duration_in_months'] = self.duration_in_months
        if hasattr(self, 'line_item_id') and self.line_item_id is not None:
            _dict['line_item_id'] = self.line_item_id
        if hasattr(self, 'billing_system') and self.billing_system is not None:
            _dict['billing_system'] = self.billing_system
        if hasattr(self, 'renewal_mode_code') and self.renewal_mode_code is not None:
            _dict['renewal_mode_code'] = self.renewal_mode_code
        if hasattr(self, 'updated_at') and self.updated_at is not None:
            _dict['updated_at'] = datetime_to_string(self.updated_at)
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this BillingOption object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'BillingOption') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'BillingOption') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class StateEnum(str, Enum):
        """
        The state of the billing option. The valid values include `ACTIVE, `SUSPENDED`,
        and `CANCELED`.
        """

        ACTIVE = 'ACTIVE'
        SUSPENDED = 'SUSPENDED'
        CANCELED = 'CANCELED'

    class TypeEnum(str, Enum):
        """
        The type of billing option. The valid values are `SUBSCRIPTION` and `OFFER`.
        """

        SUBSCRIPTION = 'SUBSCRIPTION'
        OFFER = 'OFFER'

    class CategoryEnum(str, Enum):
        """
        The category of the billing option. The valid values are `PLATFORM`, `SERVICE`,
        and `SUPPORT`.
        """

        PLATFORM = 'PLATFORM'
        SERVICE = 'SERVICE'
        SUPPORT = 'SUPPORT'


class BillingOptionsList:
    """
    A search result containing zero or more billing options.

    :attr int rows_count: (optional) A count of the billing units that were found by
          the query.
    :attr str next_url: (optional) Bookmark URL to query for next batch of billing
          units. This returns `null` if no additional pages are required.
    :attr List[BillingOption] resources: (optional) A list of billing units found.
    """

    def __init__(
        self, *, rows_count: int = None, next_url: str = None, resources: List['BillingOption'] = None
    ) -> None:
        """
        Initialize a BillingOptionsList object.

        :param int rows_count: (optional) A count of the billing units that were
               found by the query.
        :param str next_url: (optional) Bookmark URL to query for next batch of
               billing units. This returns `null` if no additional pages are required.
        :param List[BillingOption] resources: (optional) A list of billing units
               found.
        """
        self.rows_count = rows_count
        self.next_url = next_url
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'BillingOptionsList':
        """Initialize a BillingOptionsList object from a json dictionary."""
        args = {}
        if 'rows_count' in _dict:
            args['rows_count'] = _dict.get('rows_count')
        if 'next_url' in _dict:
            args['next_url'] = _dict.get('next_url')
        if 'resources' in _dict:
            args['resources'] = [BillingOption.from_dict(v) for v in _dict.get('resources')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a BillingOptionsList object from a json dictionary."""
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
        """Return a `str` version of this BillingOptionsList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'BillingOptionsList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'BillingOptionsList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class BillingUnit:
    """
    Information about a billing unit.

    :attr str id: (optional) The ID of the billing unit, which is a globally unique
          identifier (GUID).
    :attr str crn: (optional) The Cloud Resource Name (CRN) of the billing unit,
          scoped to the enterprise account ID.
    :attr str name: (optional) The name of the billing unit.
    :attr str enterprise_id: (optional) The ID of the enterprise to which the
          billing unit is associated.
    :attr str currency_code: (optional) The currency code for the billing unit.
    :attr str country_code: (optional) The country code for the billing unit.
    :attr bool master: (optional) A flag that indicates whether this billing unit is
          the primary billing mechanism for the enterprise.
    :attr datetime created_at: (optional) The creation date of the billing unit.
    """

    def __init__(
        self,
        *,
        id: str = None,
        crn: str = None,
        name: str = None,
        enterprise_id: str = None,
        currency_code: str = None,
        country_code: str = None,
        master: bool = None,
        created_at: datetime = None,
    ) -> None:
        """
        Initialize a BillingUnit object.

        :param str id: (optional) The ID of the billing unit, which is a globally
               unique identifier (GUID).
        :param str crn: (optional) The Cloud Resource Name (CRN) of the billing
               unit, scoped to the enterprise account ID.
        :param str name: (optional) The name of the billing unit.
        :param str enterprise_id: (optional) The ID of the enterprise to which the
               billing unit is associated.
        :param str currency_code: (optional) The currency code for the billing
               unit.
        :param str country_code: (optional) The country code for the billing unit.
        :param bool master: (optional) A flag that indicates whether this billing
               unit is the primary billing mechanism for the enterprise.
        :param datetime created_at: (optional) The creation date of the billing
               unit.
        """
        self.id = id
        self.crn = crn
        self.name = name
        self.enterprise_id = enterprise_id
        self.currency_code = currency_code
        self.country_code = country_code
        self.master = master
        self.created_at = created_at

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'BillingUnit':
        """Initialize a BillingUnit object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'crn' in _dict:
            args['crn'] = _dict.get('crn')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'enterprise_id' in _dict:
            args['enterprise_id'] = _dict.get('enterprise_id')
        if 'currency_code' in _dict:
            args['currency_code'] = _dict.get('currency_code')
        if 'country_code' in _dict:
            args['country_code'] = _dict.get('country_code')
        if 'master' in _dict:
            args['master'] = _dict.get('master')
        if 'created_at' in _dict:
            args['created_at'] = string_to_datetime(_dict.get('created_at'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a BillingUnit object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'enterprise_id') and self.enterprise_id is not None:
            _dict['enterprise_id'] = self.enterprise_id
        if hasattr(self, 'currency_code') and self.currency_code is not None:
            _dict['currency_code'] = self.currency_code
        if hasattr(self, 'country_code') and self.country_code is not None:
            _dict['country_code'] = self.country_code
        if hasattr(self, 'master') and self.master is not None:
            _dict['master'] = self.master
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = datetime_to_string(self.created_at)
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this BillingUnit object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'BillingUnit') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'BillingUnit') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class BillingUnitsList:
    """
    A search result contining zero or more billing units.

    :attr int rows_count: (optional) A count of the billing units that were found by
          the query.
    :attr str next_url: (optional) Bookmark URL to query for next batch of billing
          units. This returns `null` if no additional pages are required.
    :attr List[BillingUnit] resources: (optional) A list of billing units found.
    """

    def __init__(self, *, rows_count: int = None, next_url: str = None, resources: List['BillingUnit'] = None) -> None:
        """
        Initialize a BillingUnitsList object.

        :param int rows_count: (optional) A count of the billing units that were
               found by the query.
        :param str next_url: (optional) Bookmark URL to query for next batch of
               billing units. This returns `null` if no additional pages are required.
        :param List[BillingUnit] resources: (optional) A list of billing units
               found.
        """
        self.rows_count = rows_count
        self.next_url = next_url
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'BillingUnitsList':
        """Initialize a BillingUnitsList object from a json dictionary."""
        args = {}
        if 'rows_count' in _dict:
            args['rows_count'] = _dict.get('rows_count')
        if 'next_url' in _dict:
            args['next_url'] = _dict.get('next_url')
        if 'resources' in _dict:
            args['resources'] = [BillingUnit.from_dict(v) for v in _dict.get('resources')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a BillingUnitsList object from a json dictionary."""
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
        """Return a `str` version of this BillingUnitsList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'BillingUnitsList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'BillingUnitsList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class CreditPool:
    """
    The credit pool for a billing unit.

    :attr str type: (optional) The type of credit, either `PLATFORM` or `SUPPORT`.
    :attr str currency_code: (optional) The currency code of the associated billing
          unit.
    :attr str billing_unit_id: (optional) The ID of the billing unit that's
          associated with the credit pool. This value is a globally unique identifier
          (GUID).
    :attr List[TermCredits] term_credits: (optional) A list of active subscription
          terms available within a credit pool.
    :attr CreditPoolOverage overage: (optional) Overage that was generated on the
          credit pool.
    """

    def __init__(
        self,
        *,
        type: str = None,
        currency_code: str = None,
        billing_unit_id: str = None,
        term_credits: List['TermCredits'] = None,
        overage: 'CreditPoolOverage' = None,
    ) -> None:
        """
        Initialize a CreditPool object.

        :param str type: (optional) The type of credit, either `PLATFORM` or
               `SUPPORT`.
        :param str currency_code: (optional) The currency code of the associated
               billing unit.
        :param str billing_unit_id: (optional) The ID of the billing unit that's
               associated with the credit pool. This value is a globally unique identifier
               (GUID).
        :param List[TermCredits] term_credits: (optional) A list of active
               subscription terms available within a credit pool.
        :param CreditPoolOverage overage: (optional) Overage that was generated on
               the credit pool.
        """
        self.type = type
        self.currency_code = currency_code
        self.billing_unit_id = billing_unit_id
        self.term_credits = term_credits
        self.overage = overage

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CreditPool':
        """Initialize a CreditPool object from a json dictionary."""
        args = {}
        if 'type' in _dict:
            args['type'] = _dict.get('type')
        if 'currency_code' in _dict:
            args['currency_code'] = _dict.get('currency_code')
        if 'billing_unit_id' in _dict:
            args['billing_unit_id'] = _dict.get('billing_unit_id')
        if 'term_credits' in _dict:
            args['term_credits'] = [TermCredits.from_dict(v) for v in _dict.get('term_credits')]
        if 'overage' in _dict:
            args['overage'] = CreditPoolOverage.from_dict(_dict.get('overage'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CreditPool object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'currency_code') and self.currency_code is not None:
            _dict['currency_code'] = self.currency_code
        if hasattr(self, 'billing_unit_id') and self.billing_unit_id is not None:
            _dict['billing_unit_id'] = self.billing_unit_id
        if hasattr(self, 'term_credits') and self.term_credits is not None:
            term_credits_list = []
            for v in self.term_credits:
                if isinstance(v, dict):
                    term_credits_list.append(v)
                else:
                    term_credits_list.append(v.to_dict())
            _dict['term_credits'] = term_credits_list
        if hasattr(self, 'overage') and self.overage is not None:
            if isinstance(self.overage, dict):
                _dict['overage'] = self.overage
            else:
                _dict['overage'] = self.overage.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this CreditPool object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CreditPool') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CreditPool') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class TypeEnum(str, Enum):
        """
        The type of credit, either `PLATFORM` or `SUPPORT`.
        """

        PLATFORM = 'PLATFORM'
        SUPPORT = 'SUPPORT'


class CreditPoolOverage:
    """
    Overage that was generated on the credit pool.

    :attr float cost: (optional) The number of credits used as overage.
    :attr List[dict] resources: (optional) A list of resources that generated
          overage.
    """

    def __init__(self, *, cost: float = None, resources: List[dict] = None) -> None:
        """
        Initialize a CreditPoolOverage object.

        :param float cost: (optional) The number of credits used as overage.
        :param List[dict] resources: (optional) A list of resources that generated
               overage.
        """
        self.cost = cost
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CreditPoolOverage':
        """Initialize a CreditPoolOverage object from a json dictionary."""
        args = {}
        if 'cost' in _dict:
            args['cost'] = _dict.get('cost')
        if 'resources' in _dict:
            args['resources'] = _dict.get('resources')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CreditPoolOverage object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'cost') and self.cost is not None:
            _dict['cost'] = self.cost
        if hasattr(self, 'resources') and self.resources is not None:
            _dict['resources'] = self.resources
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this CreditPoolOverage object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CreditPoolOverage') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CreditPoolOverage') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class CreditPoolsList:
    """
    A search result containing zero or more credit pools.

    :attr int rows_count: (optional) The number of credit pools that were found by
          the query.
    :attr str next_url: (optional) A bookmark URL to the query for the next batch of
          billing units. Use a value of `null` if no additional pages are required.
    :attr List[CreditPool] resources: (optional) A list of credit pools found by the
          query.
    """

    def __init__(self, *, rows_count: int = None, next_url: str = None, resources: List['CreditPool'] = None) -> None:
        """
        Initialize a CreditPoolsList object.

        :param int rows_count: (optional) The number of credit pools that were
               found by the query.
        :param str next_url: (optional) A bookmark URL to the query for the next
               batch of billing units. Use a value of `null` if no additional pages are
               required.
        :param List[CreditPool] resources: (optional) A list of credit pools found
               by the query.
        """
        self.rows_count = rows_count
        self.next_url = next_url
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CreditPoolsList':
        """Initialize a CreditPoolsList object from a json dictionary."""
        args = {}
        if 'rows_count' in _dict:
            args['rows_count'] = _dict.get('rows_count')
        if 'next_url' in _dict:
            args['next_url'] = _dict.get('next_url')
        if 'resources' in _dict:
            args['resources'] = [CreditPool.from_dict(v) for v in _dict.get('resources')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CreditPoolsList object from a json dictionary."""
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
        """Return a `str` version of this CreditPoolsList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CreditPoolsList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CreditPoolsList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TermCredits:
    """
    The subscription term that is active in the current month.

    :attr str billing_option_id: (optional) The ID of the billing option from which
          the subscription term is derived.
    :attr str category: (optional) The category of the credit pool. The valid values
          are `PLATFORM`, `OFFER`, or `SERVICE` for platform credit and `SUPPORT` for
          support credit.
    :attr datetime start_date: (optional) The start date of the term in ISO format.
    :attr datetime end_date: (optional) The end date of the term in ISO format.
    :attr float total_credits: (optional) The total credit available in this term.
    :attr float starting_balance: (optional) The balance of available credit at the
          start of the current month.
    :attr float used_credits: (optional) The amount of credit used during the
          current month.
    :attr float current_balance: (optional) The balance of remaining credit in the
          subscription term.
    :attr List[dict] resources: (optional) A list of resources that used credit
          during the month.
    """

    def __init__(
        self,
        *,
        billing_option_id: str = None,
        category: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        total_credits: float = None,
        starting_balance: float = None,
        used_credits: float = None,
        current_balance: float = None,
        resources: List[dict] = None,
    ) -> None:
        """
        Initialize a TermCredits object.

        :param str billing_option_id: (optional) The ID of the billing option from
               which the subscription term is derived.
        :param str category: (optional) The category of the credit pool. The valid
               values are `PLATFORM`, `OFFER`, or `SERVICE` for platform credit and
               `SUPPORT` for support credit.
        :param datetime start_date: (optional) The start date of the term in ISO
               format.
        :param datetime end_date: (optional) The end date of the term in ISO
               format.
        :param float total_credits: (optional) The total credit available in this
               term.
        :param float starting_balance: (optional) The balance of available credit
               at the start of the current month.
        :param float used_credits: (optional) The amount of credit used during the
               current month.
        :param float current_balance: (optional) The balance of remaining credit in
               the subscription term.
        :param List[dict] resources: (optional) A list of resources that used
               credit during the month.
        """
        self.billing_option_id = billing_option_id
        self.category = category
        self.start_date = start_date
        self.end_date = end_date
        self.total_credits = total_credits
        self.starting_balance = starting_balance
        self.used_credits = used_credits
        self.current_balance = current_balance
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'TermCredits':
        """Initialize a TermCredits object from a json dictionary."""
        args = {}
        if 'billing_option_id' in _dict:
            args['billing_option_id'] = _dict.get('billing_option_id')
        if 'category' in _dict:
            args['category'] = _dict.get('category')
        if 'start_date' in _dict:
            args['start_date'] = string_to_datetime(_dict.get('start_date'))
        if 'end_date' in _dict:
            args['end_date'] = string_to_datetime(_dict.get('end_date'))
        if 'total_credits' in _dict:
            args['total_credits'] = _dict.get('total_credits')
        if 'starting_balance' in _dict:
            args['starting_balance'] = _dict.get('starting_balance')
        if 'used_credits' in _dict:
            args['used_credits'] = _dict.get('used_credits')
        if 'current_balance' in _dict:
            args['current_balance'] = _dict.get('current_balance')
        if 'resources' in _dict:
            args['resources'] = _dict.get('resources')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a TermCredits object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'billing_option_id') and self.billing_option_id is not None:
            _dict['billing_option_id'] = self.billing_option_id
        if hasattr(self, 'category') and self.category is not None:
            _dict['category'] = self.category
        if hasattr(self, 'start_date') and self.start_date is not None:
            _dict['start_date'] = datetime_to_string(self.start_date)
        if hasattr(self, 'end_date') and self.end_date is not None:
            _dict['end_date'] = datetime_to_string(self.end_date)
        if hasattr(self, 'total_credits') and self.total_credits is not None:
            _dict['total_credits'] = self.total_credits
        if hasattr(self, 'starting_balance') and self.starting_balance is not None:
            _dict['starting_balance'] = self.starting_balance
        if hasattr(self, 'used_credits') and self.used_credits is not None:
            _dict['used_credits'] = self.used_credits
        if hasattr(self, 'current_balance') and self.current_balance is not None:
            _dict['current_balance'] = self.current_balance
        if hasattr(self, 'resources') and self.resources is not None:
            _dict['resources'] = self.resources
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this TermCredits object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'TermCredits') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'TermCredits') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class CategoryEnum(str, Enum):
        """
        The category of the credit pool. The valid values are `PLATFORM`, `OFFER`, or
        `SERVICE` for platform credit and `SUPPORT` for support credit.
        """

        PLATFORM = 'PLATFORM'
        OFFER = 'OFFER'
        SERVICE = 'SERVICE'
        SUPPORT = 'SUPPORT'


##############################################################################
# Pagers
##############################################################################


class BillingUnitsPager:
    """
    BillingUnitsPager can be used to simplify the use of the "list_billing_units" method.
    """

    def __init__(
        self,
        *,
        client: EnterpriseBillingUnitsV1,
        account_id: str = None,
        enterprise_id: str = None,
        account_group_id: str = None,
        limit: int = None,
    ) -> None:
        """
        Initialize a BillingUnitsPager object.
        :param str account_id: (optional) The enterprise account ID.
        :param str enterprise_id: (optional) The enterprise ID.
        :param str account_group_id: (optional) The account group ID.
        :param int limit: (optional) Return results up to this limit. Valid values
               are between 0 and 100.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._account_id = account_id
        self._enterprise_id = enterprise_id
        self._account_group_id = account_group_id
        self._limit = limit

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of BillingUnit.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.list_billing_units(
            account_id=self._account_id,
            enterprise_id=self._enterprise_id,
            account_group_id=self._account_group_id,
            limit=self._limit,
            start=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next_url')
        if next_page_link is not None:
            next = get_query_param(next_page_link.rstrip('&'), 'start')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('resources')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of BillingUnit.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results


class BillingOptionsPager:
    """
    BillingOptionsPager can be used to simplify the use of the "list_billing_options" method.
    """

    def __init__(
        self,
        *,
        client: EnterpriseBillingUnitsV1,
        billing_unit_id: str,
        limit: int = None,
    ) -> None:
        """
        Initialize a BillingOptionsPager object.
        :param str billing_unit_id: The billing unit ID.
        :param int limit: (optional) Return results up to this limit. Valid values
               are between 0 and 100.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._billing_unit_id = billing_unit_id
        self._limit = limit

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of BillingOption.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.list_billing_options(
            billing_unit_id=self._billing_unit_id,
            limit=self._limit,
            start=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next_url')
        if next_page_link is not None:
            next = get_query_param(next_page_link.rstrip('&'), 'start')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('resources')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of BillingOption.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results
