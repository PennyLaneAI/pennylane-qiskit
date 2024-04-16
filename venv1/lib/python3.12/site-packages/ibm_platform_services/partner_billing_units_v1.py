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

# IBM OpenAPI SDK Code Generator Version: 3.85.0-75c38f8f-20240206-210220

"""
Billing units for IBM Cloud partners

API Version: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import json

from ibm_cloud_sdk_core import BaseService, DetailedResponse
from ibm_cloud_sdk_core.authenticators.authenticator import Authenticator
from ibm_cloud_sdk_core.get_authenticator import get_authenticator_from_environment
from ibm_cloud_sdk_core.utils import datetime_to_string, string_to_datetime

from .common import get_sdk_headers

##############################################################################
# Service
##############################################################################


class PartnerBillingUnitsV1(BaseService):
    """The Partner Billing Units V1 service."""

    DEFAULT_SERVICE_URL = 'https://partner.cloud.ibm.com'
    DEFAULT_SERVICE_NAME = 'partner_billing_units'

    @classmethod
    def new_instance(
        cls,
        service_name: str = DEFAULT_SERVICE_NAME,
    ) -> 'PartnerBillingUnitsV1':
        """
        Return a new client for the Partner Billing Units service using the
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
        Construct a new client for the Partner Billing Units service.

        :param Authenticator authenticator: The authenticator specifies the authentication mechanism.
               Get up to date information from https://github.com/IBM/python-sdk-core/blob/main/README.md
               about initializing the authenticator of your choice.
        """
        BaseService.__init__(self, service_url=self.DEFAULT_SERVICE_URL, authenticator=authenticator)

    #########################
    # Billing Options
    #########################

    def get_billing_options(
        self,
        partner_id: str,
        *,
        customer_id: Optional[str] = None,
        reseller_id: Optional[str] = None,
        date: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get customers billing options.

        Returns the billing options for the requested customer for a given month.

        :param str partner_id: Enterprise ID of the distributor or reseller for
               which the report is requested.
        :param str customer_id: (optional) Enterprise ID of the customer for which
               the report is requested.
        :param str reseller_id: (optional) Enterprise ID of the reseller for which
               the report is requested.
        :param str date: (optional) The billing month for which the usage report is
               requested. Format is yyyy-mm. Defaults to current month.
        :param int limit: (optional) Number of usage records returned. The default
               value is 30. Maximum value is 200.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `BillingOptionsSummary` object
        """

        if not partner_id:
            raise ValueError('partner_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_billing_options',
        )
        headers.update(sdk_headers)

        params = {
            'partner_id': partner_id,
            'customer_id': customer_id,
            'reseller_id': reseller_id,
            'date': date,
            '_limit': limit,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/billing-options'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response

    #########################
    # Credit Pools
    #########################

    def get_credit_pools_report(
        self,
        partner_id: str,
        *,
        customer_id: Optional[str] = None,
        reseller_id: Optional[str] = None,
        date: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get subscription burn-down report.

        Returns the subscription or commitment burn-down reports for the end customers for
        a given month.

        :param str partner_id: Enterprise ID of the distributor or reseller for
               which the report is requested.
        :param str customer_id: (optional) Enterprise ID of the customer for which
               the report is requested.
        :param str reseller_id: (optional) Enterprise ID of the reseller for which
               the report is requested.
        :param str date: (optional) The billing month for which the usage report is
               requested. Format is yyyy-mm. Defaults to current month.
        :param int limit: (optional) Number of usage records returned. The default
               value is 30. Maximum value is 200.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `CreditPoolsReportSummary` object
        """

        if not partner_id:
            raise ValueError('partner_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_credit_pools_report',
        )
        headers.update(sdk_headers)

        params = {
            'partner_id': partner_id,
            'customer_id': customer_id,
            'reseller_id': reseller_id,
            'date': date,
            '_limit': limit,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/credit-pools'
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


class BillingOptionsSummaryFirst:
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
        Initialize a BillingOptionsSummaryFirst object.

        :param str href: (optional) A link to a page of query results.
        """
        self.href = href

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'BillingOptionsSummaryFirst':
        """Initialize a BillingOptionsSummaryFirst object from a json dictionary."""
        args = {}
        if (href := _dict.get('href')) is not None:
            args['href'] = href
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a BillingOptionsSummaryFirst object from a json dictionary."""
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
        """Return a `str` version of this BillingOptionsSummaryFirst object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'BillingOptionsSummaryFirst') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'BillingOptionsSummaryFirst') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class BillingOptionsSummaryNext:
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
        Initialize a BillingOptionsSummaryNext object.

        :param str href: (optional) A link to a page of query results.
        :param str offset: (optional) The value of the `_start` query parameter to
               fetch the next page.
        """
        self.href = href
        self.offset = offset

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'BillingOptionsSummaryNext':
        """Initialize a BillingOptionsSummaryNext object from a json dictionary."""
        args = {}
        if (href := _dict.get('href')) is not None:
            args['href'] = href
        if (offset := _dict.get('offset')) is not None:
            args['offset'] = offset
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a BillingOptionsSummaryNext object from a json dictionary."""
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
        """Return a `str` version of this BillingOptionsSummaryNext object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'BillingOptionsSummaryNext') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'BillingOptionsSummaryNext') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class BillingOption:
    """
    Billing options report for the end customers.

    :param str id: (optional) The ID of the billing option.
    :param str billing_unit_id: (optional) The ID of the billing unit that's
          associated with the billing option.
    :param str customer_id: (optional) Account ID of the customer.
    :param str customer_type: (optional) The customer type. The valid values are
          `ENTERPRISE`, `ACCOUNT`, and `ACCOUNT_GROUP`.
    :param str customer_name: (optional) A user-defined name for the customer.
    :param str reseller_id: (optional) ID of the reseller in the heirarchy of the
          requested customer.
    :param str reseller_name: (optional) Name of the reseller in the heirarchy of
          the requested customer.
    :param str month: (optional) The billing month for which the burn-down report is
          requested. Format is yyyy-mm. Defaults to current month.
    :param List[dict] errors: (optional) Errors in the billing.
    :param str type: (optional) The type of billing option. The valid values are
          `SUBSCRIPTION` and `OFFER`.
    :param datetime start_date: (optional) The start date of billing option.
    :param datetime end_date: (optional) The end date of billing option.
    :param str state: (optional) The state of the billing option. The valid values
          include `ACTIVE, `SUSPENDED`, and `CANCELED`.
    :param str category: (optional) The category of the billing option. The valid
          values are `PLATFORM`, `SERVICE`, and `SUPPORT`.
    :param dict payment_instrument: (optional) The payment method for support.
    :param str part_number: (optional) Part number of the offering.
    :param str catalog_id: (optional) ID of the catalog containing this offering.
    :param str order_id: (optional) ID of the order containing this offering.
    :param str po_number: (optional) PO Number of the offering.
    :param str subscription_model: (optional) Subscription model.
    :param int duration_in_months: (optional) The duration of the billing options in
          months.
    :param float monthly_amount: (optional) Amount billed monthly for this offering.
    :param dict billing_system: (optional) The support billing system.
    :param str country_code: (optional) The country code for the billing unit.
    :param str currency_code: (optional) The currency code of the billing unit.
    """

    def __init__(
        self,
        *,
        id: Optional[str] = None,
        billing_unit_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        customer_type: Optional[str] = None,
        customer_name: Optional[str] = None,
        reseller_id: Optional[str] = None,
        reseller_name: Optional[str] = None,
        month: Optional[str] = None,
        errors: Optional[List[dict]] = None,
        type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        state: Optional[str] = None,
        category: Optional[str] = None,
        payment_instrument: Optional[dict] = None,
        part_number: Optional[str] = None,
        catalog_id: Optional[str] = None,
        order_id: Optional[str] = None,
        po_number: Optional[str] = None,
        subscription_model: Optional[str] = None,
        duration_in_months: Optional[int] = None,
        monthly_amount: Optional[float] = None,
        billing_system: Optional[dict] = None,
        country_code: Optional[str] = None,
        currency_code: Optional[str] = None,
    ) -> None:
        """
        Initialize a BillingOption object.

        :param str id: (optional) The ID of the billing option.
        :param str billing_unit_id: (optional) The ID of the billing unit that's
               associated with the billing option.
        :param str customer_id: (optional) Account ID of the customer.
        :param str customer_type: (optional) The customer type. The valid values
               are `ENTERPRISE`, `ACCOUNT`, and `ACCOUNT_GROUP`.
        :param str customer_name: (optional) A user-defined name for the customer.
        :param str reseller_id: (optional) ID of the reseller in the heirarchy of
               the requested customer.
        :param str reseller_name: (optional) Name of the reseller in the heirarchy
               of the requested customer.
        :param str month: (optional) The billing month for which the burn-down
               report is requested. Format is yyyy-mm. Defaults to current month.
        :param List[dict] errors: (optional) Errors in the billing.
        :param str type: (optional) The type of billing option. The valid values
               are `SUBSCRIPTION` and `OFFER`.
        :param datetime start_date: (optional) The start date of billing option.
        :param datetime end_date: (optional) The end date of billing option.
        :param str state: (optional) The state of the billing option. The valid
               values include `ACTIVE, `SUSPENDED`, and `CANCELED`.
        :param str category: (optional) The category of the billing option. The
               valid values are `PLATFORM`, `SERVICE`, and `SUPPORT`.
        :param dict payment_instrument: (optional) The payment method for support.
        :param str part_number: (optional) Part number of the offering.
        :param str catalog_id: (optional) ID of the catalog containing this
               offering.
        :param str order_id: (optional) ID of the order containing this offering.
        :param str po_number: (optional) PO Number of the offering.
        :param str subscription_model: (optional) Subscription model.
        :param int duration_in_months: (optional) The duration of the billing
               options in months.
        :param float monthly_amount: (optional) Amount billed monthly for this
               offering.
        :param dict billing_system: (optional) The support billing system.
        :param str country_code: (optional) The country code for the billing unit.
        :param str currency_code: (optional) The currency code of the billing unit.
        """
        self.id = id
        self.billing_unit_id = billing_unit_id
        self.customer_id = customer_id
        self.customer_type = customer_type
        self.customer_name = customer_name
        self.reseller_id = reseller_id
        self.reseller_name = reseller_name
        self.month = month
        self.errors = errors
        self.type = type
        self.start_date = start_date
        self.end_date = end_date
        self.state = state
        self.category = category
        self.payment_instrument = payment_instrument
        self.part_number = part_number
        self.catalog_id = catalog_id
        self.order_id = order_id
        self.po_number = po_number
        self.subscription_model = subscription_model
        self.duration_in_months = duration_in_months
        self.monthly_amount = monthly_amount
        self.billing_system = billing_system
        self.country_code = country_code
        self.currency_code = currency_code

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'BillingOption':
        """Initialize a BillingOption object from a json dictionary."""
        args = {}
        if (id := _dict.get('id')) is not None:
            args['id'] = id
        if (billing_unit_id := _dict.get('billing_unit_id')) is not None:
            args['billing_unit_id'] = billing_unit_id
        if (customer_id := _dict.get('customer_id')) is not None:
            args['customer_id'] = customer_id
        if (customer_type := _dict.get('customer_type')) is not None:
            args['customer_type'] = customer_type
        if (customer_name := _dict.get('customer_name')) is not None:
            args['customer_name'] = customer_name
        if (reseller_id := _dict.get('reseller_id')) is not None:
            args['reseller_id'] = reseller_id
        if (reseller_name := _dict.get('reseller_name')) is not None:
            args['reseller_name'] = reseller_name
        if (month := _dict.get('month')) is not None:
            args['month'] = month
        if (errors := _dict.get('errors')) is not None:
            args['errors'] = errors
        if (type := _dict.get('type')) is not None:
            args['type'] = type
        if (start_date := _dict.get('start_date')) is not None:
            args['start_date'] = string_to_datetime(start_date)
        if (end_date := _dict.get('end_date')) is not None:
            args['end_date'] = string_to_datetime(end_date)
        if (state := _dict.get('state')) is not None:
            args['state'] = state
        if (category := _dict.get('category')) is not None:
            args['category'] = category
        if (payment_instrument := _dict.get('payment_instrument')) is not None:
            args['payment_instrument'] = payment_instrument
        if (part_number := _dict.get('part_number')) is not None:
            args['part_number'] = part_number
        if (catalog_id := _dict.get('catalog_id')) is not None:
            args['catalog_id'] = catalog_id
        if (order_id := _dict.get('order_id')) is not None:
            args['order_id'] = order_id
        if (po_number := _dict.get('po_number')) is not None:
            args['po_number'] = po_number
        if (subscription_model := _dict.get('subscription_model')) is not None:
            args['subscription_model'] = subscription_model
        if (duration_in_months := _dict.get('duration_in_months')) is not None:
            args['duration_in_months'] = duration_in_months
        if (monthly_amount := _dict.get('monthly_amount')) is not None:
            args['monthly_amount'] = monthly_amount
        if (billing_system := _dict.get('billing_system')) is not None:
            args['billing_system'] = billing_system
        if (country_code := _dict.get('country_code')) is not None:
            args['country_code'] = country_code
        if (currency_code := _dict.get('currency_code')) is not None:
            args['currency_code'] = currency_code
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
        if hasattr(self, 'customer_id') and self.customer_id is not None:
            _dict['customer_id'] = self.customer_id
        if hasattr(self, 'customer_type') and self.customer_type is not None:
            _dict['customer_type'] = self.customer_type
        if hasattr(self, 'customer_name') and self.customer_name is not None:
            _dict['customer_name'] = self.customer_name
        if hasattr(self, 'reseller_id') and self.reseller_id is not None:
            _dict['reseller_id'] = self.reseller_id
        if hasattr(self, 'reseller_name') and self.reseller_name is not None:
            _dict['reseller_name'] = self.reseller_name
        if hasattr(self, 'month') and self.month is not None:
            _dict['month'] = self.month
        if hasattr(self, 'errors') and self.errors is not None:
            _dict['errors'] = self.errors
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'start_date') and self.start_date is not None:
            _dict['start_date'] = datetime_to_string(self.start_date)
        if hasattr(self, 'end_date') and self.end_date is not None:
            _dict['end_date'] = datetime_to_string(self.end_date)
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        if hasattr(self, 'category') and self.category is not None:
            _dict['category'] = self.category
        if hasattr(self, 'payment_instrument') and self.payment_instrument is not None:
            _dict['payment_instrument'] = self.payment_instrument
        if hasattr(self, 'part_number') and self.part_number is not None:
            _dict['part_number'] = self.part_number
        if hasattr(self, 'catalog_id') and self.catalog_id is not None:
            _dict['catalog_id'] = self.catalog_id
        if hasattr(self, 'order_id') and self.order_id is not None:
            _dict['order_id'] = self.order_id
        if hasattr(self, 'po_number') and self.po_number is not None:
            _dict['po_number'] = self.po_number
        if hasattr(self, 'subscription_model') and self.subscription_model is not None:
            _dict['subscription_model'] = self.subscription_model
        if hasattr(self, 'duration_in_months') and self.duration_in_months is not None:
            _dict['duration_in_months'] = self.duration_in_months
        if hasattr(self, 'monthly_amount') and self.monthly_amount is not None:
            _dict['monthly_amount'] = self.monthly_amount
        if hasattr(self, 'billing_system') and self.billing_system is not None:
            _dict['billing_system'] = self.billing_system
        if hasattr(self, 'country_code') and self.country_code is not None:
            _dict['country_code'] = self.country_code
        if hasattr(self, 'currency_code') and self.currency_code is not None:
            _dict['currency_code'] = self.currency_code
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

    class CustomerTypeEnum(str, Enum):
        """
        The customer type. The valid values are `ENTERPRISE`, `ACCOUNT`, and
        `ACCOUNT_GROUP`.
        """

        ENTERPRISE = 'ENTERPRISE'
        ACCOUNT = 'ACCOUNT'
        ACCOUNT_GROUP = 'ACCOUNT_GROUP'

    class TypeEnum(str, Enum):
        """
        The type of billing option. The valid values are `SUBSCRIPTION` and `OFFER`.
        """

        SUBSCRIPTION = 'SUBSCRIPTION'
        OFFER = 'OFFER'

    class StateEnum(str, Enum):
        """
        The state of the billing option. The valid values include `ACTIVE, `SUSPENDED`,
        and `CANCELED`.
        """

        ACTIVE = 'ACTIVE'
        SUSPENDED = 'SUSPENDED'
        CANCELED = 'CANCELED'

    class CategoryEnum(str, Enum):
        """
        The category of the billing option. The valid values are `PLATFORM`, `SERVICE`,
        and `SUPPORT`.
        """

        PLATFORM = 'PLATFORM'
        SERVICE = 'SERVICE'
        SUPPORT = 'SUPPORT'


class BillingOptionsSummary:
    """
    The billing options report for the customer.

    :param int limit: (optional) The max number of reports in the response.
    :param BillingOptionsSummaryFirst first: (optional) The link to the first page
          of the search query.
    :param BillingOptionsSummaryNext next: (optional) The link to the next page of
          the search query.
    :param List[BillingOption] resources: (optional) Aggregated usage report of all
          requested partners.
    """

    def __init__(
        self,
        *,
        limit: Optional[int] = None,
        first: Optional['BillingOptionsSummaryFirst'] = None,
        next: Optional['BillingOptionsSummaryNext'] = None,
        resources: Optional[List['BillingOption']] = None,
    ) -> None:
        """
        Initialize a BillingOptionsSummary object.

        :param int limit: (optional) The max number of reports in the response.
        :param BillingOptionsSummaryFirst first: (optional) The link to the first
               page of the search query.
        :param BillingOptionsSummaryNext next: (optional) The link to the next page
               of the search query.
        :param List[BillingOption] resources: (optional) Aggregated usage report of
               all requested partners.
        """
        self.limit = limit
        self.first = first
        self.next = next
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'BillingOptionsSummary':
        """Initialize a BillingOptionsSummary object from a json dictionary."""
        args = {}
        if (limit := _dict.get('limit')) is not None:
            args['limit'] = limit
        if (first := _dict.get('first')) is not None:
            args['first'] = BillingOptionsSummaryFirst.from_dict(first)
        if (next := _dict.get('next')) is not None:
            args['next'] = BillingOptionsSummaryNext.from_dict(next)
        if (resources := _dict.get('resources')) is not None:
            args['resources'] = [BillingOption.from_dict(v) for v in resources]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a BillingOptionsSummary object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
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
        """Return a `str` version of this BillingOptionsSummary object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'BillingOptionsSummary') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'BillingOptionsSummary') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class CreditPoolsReportSummaryFirst:
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
        Initialize a CreditPoolsReportSummaryFirst object.

        :param str href: (optional) A link to a page of query results.
        """
        self.href = href

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CreditPoolsReportSummaryFirst':
        """Initialize a CreditPoolsReportSummaryFirst object from a json dictionary."""
        args = {}
        if (href := _dict.get('href')) is not None:
            args['href'] = href
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CreditPoolsReportSummaryFirst object from a json dictionary."""
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
        """Return a `str` version of this CreditPoolsReportSummaryFirst object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CreditPoolsReportSummaryFirst') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CreditPoolsReportSummaryFirst') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class CreditPoolsReportSummaryNext:
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
        Initialize a CreditPoolsReportSummaryNext object.

        :param str href: (optional) A link to a page of query results.
        :param str offset: (optional) The value of the `_start` query parameter to
               fetch the next page.
        """
        self.href = href
        self.offset = offset

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CreditPoolsReportSummaryNext':
        """Initialize a CreditPoolsReportSummaryNext object from a json dictionary."""
        args = {}
        if (href := _dict.get('href')) is not None:
            args['href'] = href
        if (offset := _dict.get('offset')) is not None:
            args['offset'] = offset
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CreditPoolsReportSummaryNext object from a json dictionary."""
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
        """Return a `str` version of this CreditPoolsReportSummaryNext object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CreditPoolsReportSummaryNext') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CreditPoolsReportSummaryNext') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class CreditPoolsReport:
    """
    Aggregated subscription burn-down report for the end customers.

    :param str type: (optional) The category of the billing option. The valid values
          are `PLATFORM`, `SERVICE` and `SUPPORT`.
    :param str billing_unit_id: (optional) The ID of the billing unit that's
          associated with the billing option.
    :param str customer_id: (optional) Account ID of the customer.
    :param str customer_type: (optional) The customer type. The valid values are
          `ENTERPRISE`, `ACCOUNT`, and `ACCOUNT_GROUP`.
    :param str customer_name: (optional) A user-defined name for the customer.
    :param str reseller_id: (optional) ID of the reseller in the heirarchy of the
          requested customer.
    :param str reseller_name: (optional) Name of the reseller in the heirarchy of
          the requested customer.
    :param str month: (optional) The billing month for which the burn-down report is
          requested. Format is yyyy-mm. Defaults to current month.
    :param str currency_code: (optional) The currency code of the billing unit.
    :param List[TermCredits] term_credits: (optional) A list of active subscription
          terms available within a credit.
    :param Overage overage: (optional) Overage that was generated on the credit
          pool.
    """

    def __init__(
        self,
        *,
        type: Optional[str] = None,
        billing_unit_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        customer_type: Optional[str] = None,
        customer_name: Optional[str] = None,
        reseller_id: Optional[str] = None,
        reseller_name: Optional[str] = None,
        month: Optional[str] = None,
        currency_code: Optional[str] = None,
        term_credits: Optional[List['TermCredits']] = None,
        overage: Optional['Overage'] = None,
    ) -> None:
        """
        Initialize a CreditPoolsReport object.

        :param str type: (optional) The category of the billing option. The valid
               values are `PLATFORM`, `SERVICE` and `SUPPORT`.
        :param str billing_unit_id: (optional) The ID of the billing unit that's
               associated with the billing option.
        :param str customer_id: (optional) Account ID of the customer.
        :param str customer_type: (optional) The customer type. The valid values
               are `ENTERPRISE`, `ACCOUNT`, and `ACCOUNT_GROUP`.
        :param str customer_name: (optional) A user-defined name for the customer.
        :param str reseller_id: (optional) ID of the reseller in the heirarchy of
               the requested customer.
        :param str reseller_name: (optional) Name of the reseller in the heirarchy
               of the requested customer.
        :param str month: (optional) The billing month for which the burn-down
               report is requested. Format is yyyy-mm. Defaults to current month.
        :param str currency_code: (optional) The currency code of the billing unit.
        :param List[TermCredits] term_credits: (optional) A list of active
               subscription terms available within a credit.
        :param Overage overage: (optional) Overage that was generated on the credit
               pool.
        """
        self.type = type
        self.billing_unit_id = billing_unit_id
        self.customer_id = customer_id
        self.customer_type = customer_type
        self.customer_name = customer_name
        self.reseller_id = reseller_id
        self.reseller_name = reseller_name
        self.month = month
        self.currency_code = currency_code
        self.term_credits = term_credits
        self.overage = overage

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CreditPoolsReport':
        """Initialize a CreditPoolsReport object from a json dictionary."""
        args = {}
        if (type := _dict.get('type')) is not None:
            args['type'] = type
        if (billing_unit_id := _dict.get('billing_unit_id')) is not None:
            args['billing_unit_id'] = billing_unit_id
        if (customer_id := _dict.get('customer_id')) is not None:
            args['customer_id'] = customer_id
        if (customer_type := _dict.get('customer_type')) is not None:
            args['customer_type'] = customer_type
        if (customer_name := _dict.get('customer_name')) is not None:
            args['customer_name'] = customer_name
        if (reseller_id := _dict.get('reseller_id')) is not None:
            args['reseller_id'] = reseller_id
        if (reseller_name := _dict.get('reseller_name')) is not None:
            args['reseller_name'] = reseller_name
        if (month := _dict.get('month')) is not None:
            args['month'] = month
        if (currency_code := _dict.get('currency_code')) is not None:
            args['currency_code'] = currency_code
        if (term_credits := _dict.get('term_credits')) is not None:
            args['term_credits'] = [TermCredits.from_dict(v) for v in term_credits]
        if (overage := _dict.get('overage')) is not None:
            args['overage'] = Overage.from_dict(overage)
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CreditPoolsReport object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'billing_unit_id') and self.billing_unit_id is not None:
            _dict['billing_unit_id'] = self.billing_unit_id
        if hasattr(self, 'customer_id') and self.customer_id is not None:
            _dict['customer_id'] = self.customer_id
        if hasattr(self, 'customer_type') and self.customer_type is not None:
            _dict['customer_type'] = self.customer_type
        if hasattr(self, 'customer_name') and self.customer_name is not None:
            _dict['customer_name'] = self.customer_name
        if hasattr(self, 'reseller_id') and self.reseller_id is not None:
            _dict['reseller_id'] = self.reseller_id
        if hasattr(self, 'reseller_name') and self.reseller_name is not None:
            _dict['reseller_name'] = self.reseller_name
        if hasattr(self, 'month') and self.month is not None:
            _dict['month'] = self.month
        if hasattr(self, 'currency_code') and self.currency_code is not None:
            _dict['currency_code'] = self.currency_code
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
        """Return a `str` version of this CreditPoolsReport object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CreditPoolsReport') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CreditPoolsReport') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class TypeEnum(str, Enum):
        """
        The category of the billing option. The valid values are `PLATFORM`, `SERVICE` and
        `SUPPORT`.
        """

        PLATFORM = 'PLATFORM'
        SERVICE = 'SERVICE'
        SUPPORT = 'SUPPORT'

    class CustomerTypeEnum(str, Enum):
        """
        The customer type. The valid values are `ENTERPRISE`, `ACCOUNT`, and
        `ACCOUNT_GROUP`.
        """

        ENTERPRISE = 'ENTERPRISE'
        ACCOUNT = 'ACCOUNT'
        ACCOUNT_GROUP = 'ACCOUNT_GROUP'


class CreditPoolsReportSummary:
    """
    The aggregated credit pools report.

    :param int limit: (optional) The max number of reports in the response.
    :param CreditPoolsReportSummaryFirst first: (optional) The link to the first
          page of the search query.
    :param CreditPoolsReportSummaryNext next: (optional) The link to the next page
          of the search query.
    :param List[CreditPoolsReport] resources: (optional) Aggregated usage report of
          all requested partners.
    """

    def __init__(
        self,
        *,
        limit: Optional[int] = None,
        first: Optional['CreditPoolsReportSummaryFirst'] = None,
        next: Optional['CreditPoolsReportSummaryNext'] = None,
        resources: Optional[List['CreditPoolsReport']] = None,
    ) -> None:
        """
        Initialize a CreditPoolsReportSummary object.

        :param int limit: (optional) The max number of reports in the response.
        :param CreditPoolsReportSummaryFirst first: (optional) The link to the
               first page of the search query.
        :param CreditPoolsReportSummaryNext next: (optional) The link to the next
               page of the search query.
        :param List[CreditPoolsReport] resources: (optional) Aggregated usage
               report of all requested partners.
        """
        self.limit = limit
        self.first = first
        self.next = next
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CreditPoolsReportSummary':
        """Initialize a CreditPoolsReportSummary object from a json dictionary."""
        args = {}
        if (limit := _dict.get('limit')) is not None:
            args['limit'] = limit
        if (first := _dict.get('first')) is not None:
            args['first'] = CreditPoolsReportSummaryFirst.from_dict(first)
        if (next := _dict.get('next')) is not None:
            args['next'] = CreditPoolsReportSummaryNext.from_dict(next)
        if (resources := _dict.get('resources')) is not None:
            args['resources'] = [CreditPoolsReport.from_dict(v) for v in resources]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CreditPoolsReportSummary object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
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
        """Return a `str` version of this CreditPoolsReportSummary object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CreditPoolsReportSummary') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CreditPoolsReportSummary') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Overage:
    """
    Overage that was generated on the credit pool.

    :param float cost: (optional) The number of credits used as overage.
    :param List[dict] resources: (optional) A list of resources that generated
          overage.
    """

    def __init__(
        self,
        *,
        cost: Optional[float] = None,
        resources: Optional[List[dict]] = None,
    ) -> None:
        """
        Initialize a Overage object.

        :param float cost: (optional) The number of credits used as overage.
        :param List[dict] resources: (optional) A list of resources that generated
               overage.
        """
        self.cost = cost
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Overage':
        """Initialize a Overage object from a json dictionary."""
        args = {}
        if (cost := _dict.get('cost')) is not None:
            args['cost'] = cost
        if (resources := _dict.get('resources')) is not None:
            args['resources'] = resources
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Overage object from a json dictionary."""
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
        """Return a `str` version of this Overage object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Overage') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Overage') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class TermCredits:
    """
    The subscription term that is active in the requested month.

    :param str billing_option_id: (optional) The ID of the billing option from which
          the subscription term is derived.
    :param str billing_option_model: (optional) Billing option model.
    :param str category: (optional) The category of the billing option. The valid
          values are `PLATFORM`, `SERVICE`, and `SUPPORT`.
    :param datetime start_date: (optional) The start date of the term in ISO format.
    :param datetime end_date: (optional) The end date of the term in ISO format.
    :param float total_credits: (optional) The total credit available in this term.
    :param float starting_balance: (optional) The balance of available credit at the
          start of the current month.
    :param float used_credits: (optional) The amount of credit used during the
          current month.
    :param float current_balance: (optional) The balance of remaining credit in the
          subscription term.
    :param List[dict] resources: (optional) A list of resources that used credit
          during the month.
    """

    def __init__(
        self,
        *,
        billing_option_id: Optional[str] = None,
        billing_option_model: Optional[str] = None,
        category: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        total_credits: Optional[float] = None,
        starting_balance: Optional[float] = None,
        used_credits: Optional[float] = None,
        current_balance: Optional[float] = None,
        resources: Optional[List[dict]] = None,
    ) -> None:
        """
        Initialize a TermCredits object.

        :param str billing_option_id: (optional) The ID of the billing option from
               which the subscription term is derived.
        :param str billing_option_model: (optional) Billing option model.
        :param str category: (optional) The category of the billing option. The
               valid values are `PLATFORM`, `SERVICE`, and `SUPPORT`.
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
        self.billing_option_model = billing_option_model
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
        if (billing_option_id := _dict.get('billing_option_id')) is not None:
            args['billing_option_id'] = billing_option_id
        if (billing_option_model := _dict.get('billing_option_model')) is not None:
            args['billing_option_model'] = billing_option_model
        if (category := _dict.get('category')) is not None:
            args['category'] = category
        if (start_date := _dict.get('start_date')) is not None:
            args['start_date'] = string_to_datetime(start_date)
        if (end_date := _dict.get('end_date')) is not None:
            args['end_date'] = string_to_datetime(end_date)
        if (total_credits := _dict.get('total_credits')) is not None:
            args['total_credits'] = total_credits
        if (starting_balance := _dict.get('starting_balance')) is not None:
            args['starting_balance'] = starting_balance
        if (used_credits := _dict.get('used_credits')) is not None:
            args['used_credits'] = used_credits
        if (current_balance := _dict.get('current_balance')) is not None:
            args['current_balance'] = current_balance
        if (resources := _dict.get('resources')) is not None:
            args['resources'] = resources
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
        if hasattr(self, 'billing_option_model') and self.billing_option_model is not None:
            _dict['billing_option_model'] = self.billing_option_model
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
        The category of the billing option. The valid values are `PLATFORM`, `SERVICE`,
        and `SUPPORT`.
        """

        PLATFORM = 'PLATFORM'
        SERVICE = 'SERVICE'
        SUPPORT = 'SUPPORT'
