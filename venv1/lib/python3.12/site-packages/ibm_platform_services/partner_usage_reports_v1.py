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
Usage reports for IBM Cloud partner entities

API Version: 1.0.0
"""

from enum import Enum
from typing import Dict, List, Optional
import json

from ibm_cloud_sdk_core import BaseService, DetailedResponse
from ibm_cloud_sdk_core.authenticators.authenticator import Authenticator
from ibm_cloud_sdk_core.get_authenticator import get_authenticator_from_environment

from .common import get_sdk_headers

##############################################################################
# Service
##############################################################################


class PartnerUsageReportsV1(BaseService):
    """The Partner Usage Reports V1 service."""

    DEFAULT_SERVICE_URL = 'https://partner.cloud.ibm.com'
    DEFAULT_SERVICE_NAME = 'partner_usage_reports'

    @classmethod
    def new_instance(
        cls,
        service_name: str = DEFAULT_SERVICE_NAME,
    ) -> 'PartnerUsageReportsV1':
        """
        Return a new client for the Partner Usage Reports service using the
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
        Construct a new client for the Partner Usage Reports service.

        :param Authenticator authenticator: The authenticator specifies the authentication mechanism.
               Get up to date information from https://github.com/IBM/python-sdk-core/blob/main/README.md
               about initializing the authenticator of your choice.
        """
        BaseService.__init__(self, service_url=self.DEFAULT_SERVICE_URL, authenticator=authenticator)

    #########################
    # Partner Usage Reports
    #########################

    def get_resource_usage_report(
        self,
        partner_id: str,
        *,
        reseller_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        children: Optional[bool] = None,
        month: Optional[str] = None,
        viewpoint: Optional[str] = None,
        recurse: Optional[bool] = None,
        limit: Optional[int] = None,
        offset: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get partner resource usage report.

        Returns the summary for the partner for a given month. Partner billing managers
        are authorized to access this report.

        :param str partner_id: Enterprise ID of the distributor or reseller for
               which the report is requested.
        :param str reseller_id: (optional) Enterprise ID of the reseller for which
               the report is requested. This parameter cannot be used along with
               `customer_id` query parameter.
        :param str customer_id: (optional) Enterprise ID of the child customer for
               which the report is requested. This parameter cannot be used along with
               `reseller_id` query parameter.
        :param bool children: (optional) Get report rolled-up to the direct
               children of the requested entity. Defaults to false. This parameter cannot
               be used along with `customer_id` query parameter.
        :param str month: (optional) The billing month for which the usage report
               is requested. Format is `yyyy-mm`. Defaults to current month.
        :param str viewpoint: (optional) Enables partner to view the cost of
               provisioned services as applicable at each level of the hierarchy. Defaults
               to the type of the calling partner. The valid values are `DISTRIBUTOR`,
               `RESELLER` and `END_CUSTOMER`.
        :param bool recurse: (optional) Get usage report rolled-up to the end
               customers of the requested entity. Defaults to false. This parameter cannot
               be used along with `reseller_id` query parameter or `customer_id` query
               parameter.
        :param int limit: (optional) Number of usage records to be returned. The
               default value is 30. Maximum value is 200.
        :param str offset: (optional) An opaque value representing the offset of
               the first item to be returned by a search query. If not specified, then the
               first page of results is returned. To retrieve the next page of search
               results, use the 'offset' query parameter value within the 'next.href' URL
               found within a prior search query response.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `PartnerUsageReportSummary` object
        """

        if not partner_id:
            raise ValueError('partner_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V1',
            operation_id='get_resource_usage_report',
        )
        headers.update(sdk_headers)

        params = {
            'partner_id': partner_id,
            'reseller_id': reseller_id,
            'customer_id': customer_id,
            'children': children,
            'month': month,
            'viewpoint': viewpoint,
            'recurse': recurse,
            'limit': limit,
            'offset': offset,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v1/resource-usage-reports'
        request = self.prepare_request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
        )

        response = self.send(request, **kwargs)
        return response


class GetResourceUsageReportEnums:
    """
    Enums for get_resource_usage_report parameters.
    """

    class Viewpoint(str, Enum):
        """
        Enables partner to view the cost of provisioned services as applicable at each
        level of the hierarchy. Defaults to the type of the calling partner. The valid
        values are `DISTRIBUTOR`, `RESELLER` and `END_CUSTOMER`.
        """

        DISTRIBUTOR = 'DISTRIBUTOR'
        RESELLER = 'RESELLER'
        END_CUSTOMER = 'END_CUSTOMER'


##############################################################################
# Models
##############################################################################


class MetricUsage:
    """
    An object that represents a metric.

    :param str metric: The name of the metric.
    :param str unit: A unit to qualify the quantity.
    :param float quantity: The aggregated value for the metric.
    :param float rateable_quantity: The quantity that is used for calculating
          charges.
    :param float cost: The cost that was incurred by the metric.
    :param float rated_cost: The pre-discounted cost that was incurred by the
          metric.
    :param List[dict] price: (optional) The price with which cost was calculated.
    """

    def __init__(
        self,
        metric: str,
        unit: str,
        quantity: float,
        rateable_quantity: float,
        cost: float,
        rated_cost: float,
        *,
        price: Optional[List[dict]] = None,
    ) -> None:
        """
        Initialize a MetricUsage object.

        :param str metric: The name of the metric.
        :param str unit: A unit to qualify the quantity.
        :param float quantity: The aggregated value for the metric.
        :param float rateable_quantity: The quantity that is used for calculating
               charges.
        :param float cost: The cost that was incurred by the metric.
        :param float rated_cost: The pre-discounted cost that was incurred by the
               metric.
        :param List[dict] price: (optional) The price with which cost was
               calculated.
        """
        self.metric = metric
        self.unit = unit
        self.quantity = quantity
        self.rateable_quantity = rateable_quantity
        self.cost = cost
        self.rated_cost = rated_cost
        self.price = price

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'MetricUsage':
        """Initialize a MetricUsage object from a json dictionary."""
        args = {}
        if (metric := _dict.get('metric')) is not None:
            args['metric'] = metric
        else:
            raise ValueError('Required property \'metric\' not present in MetricUsage JSON')
        if (unit := _dict.get('unit')) is not None:
            args['unit'] = unit
        else:
            raise ValueError('Required property \'unit\' not present in MetricUsage JSON')
        if (quantity := _dict.get('quantity')) is not None:
            args['quantity'] = quantity
        else:
            raise ValueError('Required property \'quantity\' not present in MetricUsage JSON')
        if (rateable_quantity := _dict.get('rateable_quantity')) is not None:
            args['rateable_quantity'] = rateable_quantity
        else:
            raise ValueError('Required property \'rateable_quantity\' not present in MetricUsage JSON')
        if (cost := _dict.get('cost')) is not None:
            args['cost'] = cost
        else:
            raise ValueError('Required property \'cost\' not present in MetricUsage JSON')
        if (rated_cost := _dict.get('rated_cost')) is not None:
            args['rated_cost'] = rated_cost
        else:
            raise ValueError('Required property \'rated_cost\' not present in MetricUsage JSON')
        if (price := _dict.get('price')) is not None:
            args['price'] = price
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a MetricUsage object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'metric') and self.metric is not None:
            _dict['metric'] = self.metric
        if hasattr(self, 'unit') and self.unit is not None:
            _dict['unit'] = self.unit
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
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this MetricUsage object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'MetricUsage') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'MetricUsage') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class PartnerUsageReportSummaryFirst:
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
        Initialize a PartnerUsageReportSummaryFirst object.

        :param str href: (optional) A link to a page of query results.
        """
        self.href = href

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'PartnerUsageReportSummaryFirst':
        """Initialize a PartnerUsageReportSummaryFirst object from a json dictionary."""
        args = {}
        if (href := _dict.get('href')) is not None:
            args['href'] = href
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a PartnerUsageReportSummaryFirst object from a json dictionary."""
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
        """Return a `str` version of this PartnerUsageReportSummaryFirst object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'PartnerUsageReportSummaryFirst') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'PartnerUsageReportSummaryFirst') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class PartnerUsageReportSummaryNext:
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
        Initialize a PartnerUsageReportSummaryNext object.

        :param str href: (optional) A link to a page of query results.
        :param str offset: (optional) The value of the `_start` query parameter to
               fetch the next page.
        """
        self.href = href
        self.offset = offset

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'PartnerUsageReportSummaryNext':
        """Initialize a PartnerUsageReportSummaryNext object from a json dictionary."""
        args = {}
        if (href := _dict.get('href')) is not None:
            args['href'] = href
        if (offset := _dict.get('offset')) is not None:
            args['offset'] = offset
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a PartnerUsageReportSummaryNext object from a json dictionary."""
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
        """Return a `str` version of this PartnerUsageReportSummaryNext object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'PartnerUsageReportSummaryNext') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'PartnerUsageReportSummaryNext') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class PartnerUsageReport:
    """
    Aggregated usage report of a partner.

    :param str entity_id: (optional) The ID of the entity.
    :param str entity_type: (optional) The entity type.
    :param str entity_crn: (optional) The Cloud Resource Name (CRN) of the entity
          towards which the resource usages were rolled up.
    :param str entity_name: (optional) A user-defined name for the entity, such as
          the enterprise name or account group name.
    :param str entity_partner_type: (optional) Role of the `entity_id` for which the
          usage report is fetched.
    :param str viewpoint: (optional) Enables partner to view the cost of provisioned
          services as applicable at each level of the hierarchy.
    :param str month: (optional) The billing month for which the usage report is
          requested. Format is yyyy-mm.
    :param str currency_code: (optional) The currency code of the billing unit.
    :param str country_code: (optional) The country code of the billing unit.
    :param float billable_cost: (optional) Billable charges that are aggregated from
          all entities in the report.
    :param float billable_rated_cost: (optional) Aggregated billable charges before
          discounts.
    :param float non_billable_cost: (optional) Non-billable charges that are
          aggregated from all entities in the report.
    :param float non_billable_rated_cost: (optional) Aggregated non-billable charges
          before discounts.
    :param List[ResourceUsage] resources: (optional)
    """

    def __init__(
        self,
        *,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_crn: Optional[str] = None,
        entity_name: Optional[str] = None,
        entity_partner_type: Optional[str] = None,
        viewpoint: Optional[str] = None,
        month: Optional[str] = None,
        currency_code: Optional[str] = None,
        country_code: Optional[str] = None,
        billable_cost: Optional[float] = None,
        billable_rated_cost: Optional[float] = None,
        non_billable_cost: Optional[float] = None,
        non_billable_rated_cost: Optional[float] = None,
        resources: Optional[List['ResourceUsage']] = None,
    ) -> None:
        """
        Initialize a PartnerUsageReport object.

        :param str entity_id: (optional) The ID of the entity.
        :param str entity_type: (optional) The entity type.
        :param str entity_crn: (optional) The Cloud Resource Name (CRN) of the
               entity towards which the resource usages were rolled up.
        :param str entity_name: (optional) A user-defined name for the entity, such
               as the enterprise name or account group name.
        :param str entity_partner_type: (optional) Role of the `entity_id` for
               which the usage report is fetched.
        :param str viewpoint: (optional) Enables partner to view the cost of
               provisioned services as applicable at each level of the hierarchy.
        :param str month: (optional) The billing month for which the usage report
               is requested. Format is yyyy-mm.
        :param str currency_code: (optional) The currency code of the billing unit.
        :param str country_code: (optional) The country code of the billing unit.
        :param float billable_cost: (optional) Billable charges that are aggregated
               from all entities in the report.
        :param float billable_rated_cost: (optional) Aggregated billable charges
               before discounts.
        :param float non_billable_cost: (optional) Non-billable charges that are
               aggregated from all entities in the report.
        :param float non_billable_rated_cost: (optional) Aggregated non-billable
               charges before discounts.
        :param List[ResourceUsage] resources: (optional)
        """
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.entity_crn = entity_crn
        self.entity_name = entity_name
        self.entity_partner_type = entity_partner_type
        self.viewpoint = viewpoint
        self.month = month
        self.currency_code = currency_code
        self.country_code = country_code
        self.billable_cost = billable_cost
        self.billable_rated_cost = billable_rated_cost
        self.non_billable_cost = non_billable_cost
        self.non_billable_rated_cost = non_billable_rated_cost
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'PartnerUsageReport':
        """Initialize a PartnerUsageReport object from a json dictionary."""
        args = {}
        if (entity_id := _dict.get('entity_id')) is not None:
            args['entity_id'] = entity_id
        if (entity_type := _dict.get('entity_type')) is not None:
            args['entity_type'] = entity_type
        if (entity_crn := _dict.get('entity_crn')) is not None:
            args['entity_crn'] = entity_crn
        if (entity_name := _dict.get('entity_name')) is not None:
            args['entity_name'] = entity_name
        if (entity_partner_type := _dict.get('entity_partner_type')) is not None:
            args['entity_partner_type'] = entity_partner_type
        if (viewpoint := _dict.get('viewpoint')) is not None:
            args['viewpoint'] = viewpoint
        if (month := _dict.get('month')) is not None:
            args['month'] = month
        if (currency_code := _dict.get('currency_code')) is not None:
            args['currency_code'] = currency_code
        if (country_code := _dict.get('country_code')) is not None:
            args['country_code'] = country_code
        if (billable_cost := _dict.get('billable_cost')) is not None:
            args['billable_cost'] = billable_cost
        if (billable_rated_cost := _dict.get('billable_rated_cost')) is not None:
            args['billable_rated_cost'] = billable_rated_cost
        if (non_billable_cost := _dict.get('non_billable_cost')) is not None:
            args['non_billable_cost'] = non_billable_cost
        if (non_billable_rated_cost := _dict.get('non_billable_rated_cost')) is not None:
            args['non_billable_rated_cost'] = non_billable_rated_cost
        if (resources := _dict.get('resources')) is not None:
            args['resources'] = [ResourceUsage.from_dict(v) for v in resources]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a PartnerUsageReport object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'entity_id') and self.entity_id is not None:
            _dict['entity_id'] = self.entity_id
        if hasattr(self, 'entity_type') and self.entity_type is not None:
            _dict['entity_type'] = self.entity_type
        if hasattr(self, 'entity_crn') and self.entity_crn is not None:
            _dict['entity_crn'] = self.entity_crn
        if hasattr(self, 'entity_name') and self.entity_name is not None:
            _dict['entity_name'] = self.entity_name
        if hasattr(self, 'entity_partner_type') and self.entity_partner_type is not None:
            _dict['entity_partner_type'] = self.entity_partner_type
        if hasattr(self, 'viewpoint') and self.viewpoint is not None:
            _dict['viewpoint'] = self.viewpoint
        if hasattr(self, 'month') and self.month is not None:
            _dict['month'] = self.month
        if hasattr(self, 'currency_code') and self.currency_code is not None:
            _dict['currency_code'] = self.currency_code
        if hasattr(self, 'country_code') and self.country_code is not None:
            _dict['country_code'] = self.country_code
        if hasattr(self, 'billable_cost') and self.billable_cost is not None:
            _dict['billable_cost'] = self.billable_cost
        if hasattr(self, 'billable_rated_cost') and self.billable_rated_cost is not None:
            _dict['billable_rated_cost'] = self.billable_rated_cost
        if hasattr(self, 'non_billable_cost') and self.non_billable_cost is not None:
            _dict['non_billable_cost'] = self.non_billable_cost
        if hasattr(self, 'non_billable_rated_cost') and self.non_billable_rated_cost is not None:
            _dict['non_billable_rated_cost'] = self.non_billable_rated_cost
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
        """Return a `str` version of this PartnerUsageReport object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'PartnerUsageReport') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'PartnerUsageReport') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class PartnerUsageReportSummary:
    """
    The aggregated partner usage report.

    :param int limit: (optional) The maximum number of usage records in the
          response.
    :param PartnerUsageReportSummaryFirst first: (optional) The link to the first
          page of the search query.
    :param PartnerUsageReportSummaryNext next: (optional) The link to the next page
          of the search query.
    :param List[PartnerUsageReport] reports: (optional) Aggregated usage report of
          all requested partners.
    """

    def __init__(
        self,
        *,
        limit: Optional[int] = None,
        first: Optional['PartnerUsageReportSummaryFirst'] = None,
        next: Optional['PartnerUsageReportSummaryNext'] = None,
        reports: Optional[List['PartnerUsageReport']] = None,
    ) -> None:
        """
        Initialize a PartnerUsageReportSummary object.

        :param int limit: (optional) The maximum number of usage records in the
               response.
        :param PartnerUsageReportSummaryFirst first: (optional) The link to the
               first page of the search query.
        :param PartnerUsageReportSummaryNext next: (optional) The link to the next
               page of the search query.
        :param List[PartnerUsageReport] reports: (optional) Aggregated usage report
               of all requested partners.
        """
        self.limit = limit
        self.first = first
        self.next = next
        self.reports = reports

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'PartnerUsageReportSummary':
        """Initialize a PartnerUsageReportSummary object from a json dictionary."""
        args = {}
        if (limit := _dict.get('limit')) is not None:
            args['limit'] = limit
        if (first := _dict.get('first')) is not None:
            args['first'] = PartnerUsageReportSummaryFirst.from_dict(first)
        if (next := _dict.get('next')) is not None:
            args['next'] = PartnerUsageReportSummaryNext.from_dict(next)
        if (reports := _dict.get('reports')) is not None:
            args['reports'] = [PartnerUsageReport.from_dict(v) for v in reports]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a PartnerUsageReportSummary object from a json dictionary."""
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
        if hasattr(self, 'reports') and self.reports is not None:
            reports_list = []
            for v in self.reports:
                if isinstance(v, dict):
                    reports_list.append(v)
                else:
                    reports_list.append(v.to_dict())
            _dict['reports'] = reports_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this PartnerUsageReportSummary object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'PartnerUsageReportSummary') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'PartnerUsageReportSummary') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class PlanUsage:
    """
    Aggregated values for the plan.

    :param str plan_id: The ID of the plan.
    :param str pricing_region: (optional) The pricing region for the plan.
    :param str pricing_plan_id: (optional) The pricing plan with which the usage was
          rated.
    :param bool billable: Whether the plan charges are billed to the customer.
    :param float cost: The total cost that was incurred by the plan.
    :param float rated_cost: The total pre-discounted cost that was incurred by the
          plan.
    :param List[MetricUsage] usage: All of the metrics in the plan.
    """

    def __init__(
        self,
        plan_id: str,
        billable: bool,
        cost: float,
        rated_cost: float,
        usage: List['MetricUsage'],
        *,
        pricing_region: Optional[str] = None,
        pricing_plan_id: Optional[str] = None,
    ) -> None:
        """
        Initialize a PlanUsage object.

        :param str plan_id: The ID of the plan.
        :param bool billable: Whether the plan charges are billed to the customer.
        :param float cost: The total cost that was incurred by the plan.
        :param float rated_cost: The total pre-discounted cost that was incurred by
               the plan.
        :param List[MetricUsage] usage: All of the metrics in the plan.
        :param str pricing_region: (optional) The pricing region for the plan.
        :param str pricing_plan_id: (optional) The pricing plan with which the
               usage was rated.
        """
        self.plan_id = plan_id
        self.pricing_region = pricing_region
        self.pricing_plan_id = pricing_plan_id
        self.billable = billable
        self.cost = cost
        self.rated_cost = rated_cost
        self.usage = usage

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'PlanUsage':
        """Initialize a PlanUsage object from a json dictionary."""
        args = {}
        if (plan_id := _dict.get('plan_id')) is not None:
            args['plan_id'] = plan_id
        else:
            raise ValueError('Required property \'plan_id\' not present in PlanUsage JSON')
        if (pricing_region := _dict.get('pricing_region')) is not None:
            args['pricing_region'] = pricing_region
        if (pricing_plan_id := _dict.get('pricing_plan_id')) is not None:
            args['pricing_plan_id'] = pricing_plan_id
        if (billable := _dict.get('billable')) is not None:
            args['billable'] = billable
        else:
            raise ValueError('Required property \'billable\' not present in PlanUsage JSON')
        if (cost := _dict.get('cost')) is not None:
            args['cost'] = cost
        else:
            raise ValueError('Required property \'cost\' not present in PlanUsage JSON')
        if (rated_cost := _dict.get('rated_cost')) is not None:
            args['rated_cost'] = rated_cost
        else:
            raise ValueError('Required property \'rated_cost\' not present in PlanUsage JSON')
        if (usage := _dict.get('usage')) is not None:
            args['usage'] = [MetricUsage.from_dict(v) for v in usage]
        else:
            raise ValueError('Required property \'usage\' not present in PlanUsage JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a PlanUsage object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'plan_id') and self.plan_id is not None:
            _dict['plan_id'] = self.plan_id
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
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this PlanUsage object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'PlanUsage') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'PlanUsage') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ResourceUsage:
    """
    A container for all the plans in the resource.

    :param str resource_id: The ID of the resource.
    :param str resource_name: (optional) The name of the resource.
    :param float billable_cost: The billable charges for the partner.
    :param float billable_rated_cost: The pre-discounted billable charges for the
          partner.
    :param float non_billable_cost: The non-billable charges for the partner.
    :param float non_billable_rated_cost: The pre-discounted, non-billable charges
          for the partner.
    :param List[PlanUsage] plans: All of the plans in the resource.
    """

    def __init__(
        self,
        resource_id: str,
        billable_cost: float,
        billable_rated_cost: float,
        non_billable_cost: float,
        non_billable_rated_cost: float,
        plans: List['PlanUsage'],
        *,
        resource_name: Optional[str] = None,
    ) -> None:
        """
        Initialize a ResourceUsage object.

        :param str resource_id: The ID of the resource.
        :param float billable_cost: The billable charges for the partner.
        :param float billable_rated_cost: The pre-discounted billable charges for
               the partner.
        :param float non_billable_cost: The non-billable charges for the partner.
        :param float non_billable_rated_cost: The pre-discounted, non-billable
               charges for the partner.
        :param List[PlanUsage] plans: All of the plans in the resource.
        :param str resource_name: (optional) The name of the resource.
        """
        self.resource_id = resource_id
        self.resource_name = resource_name
        self.billable_cost = billable_cost
        self.billable_rated_cost = billable_rated_cost
        self.non_billable_cost = non_billable_cost
        self.non_billable_rated_cost = non_billable_rated_cost
        self.plans = plans

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourceUsage':
        """Initialize a ResourceUsage object from a json dictionary."""
        args = {}
        if (resource_id := _dict.get('resource_id')) is not None:
            args['resource_id'] = resource_id
        else:
            raise ValueError('Required property \'resource_id\' not present in ResourceUsage JSON')
        if (resource_name := _dict.get('resource_name')) is not None:
            args['resource_name'] = resource_name
        if (billable_cost := _dict.get('billable_cost')) is not None:
            args['billable_cost'] = billable_cost
        else:
            raise ValueError('Required property \'billable_cost\' not present in ResourceUsage JSON')
        if (billable_rated_cost := _dict.get('billable_rated_cost')) is not None:
            args['billable_rated_cost'] = billable_rated_cost
        else:
            raise ValueError('Required property \'billable_rated_cost\' not present in ResourceUsage JSON')
        if (non_billable_cost := _dict.get('non_billable_cost')) is not None:
            args['non_billable_cost'] = non_billable_cost
        else:
            raise ValueError('Required property \'non_billable_cost\' not present in ResourceUsage JSON')
        if (non_billable_rated_cost := _dict.get('non_billable_rated_cost')) is not None:
            args['non_billable_rated_cost'] = non_billable_rated_cost
        else:
            raise ValueError('Required property \'non_billable_rated_cost\' not present in ResourceUsage JSON')
        if (plans := _dict.get('plans')) is not None:
            args['plans'] = [PlanUsage.from_dict(v) for v in plans]
        else:
            raise ValueError('Required property \'plans\' not present in ResourceUsage JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourceUsage object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'resource_id') and self.resource_id is not None:
            _dict['resource_id'] = self.resource_id
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
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResourceUsage object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourceUsage') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourceUsage') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


##############################################################################
# Pagers
##############################################################################


class GetResourceUsageReportPager:
    """
    GetResourceUsageReportPager can be used to simplify the use of the "get_resource_usage_report" method.
    """

    def __init__(
        self,
        *,
        client: PartnerUsageReportsV1,
        partner_id: str,
        reseller_id: str = None,
        customer_id: str = None,
        children: bool = None,
        month: str = None,
        viewpoint: str = None,
        recurse: bool = None,
        limit: int = None,
    ) -> None:
        """
        Initialize a GetResourceUsageReportPager object.
        :param str partner_id: Enterprise ID of the distributor or reseller for
               which the report is requested.
        :param str reseller_id: (optional) Enterprise ID of the reseller for which
               the report is requested. This parameter cannot be used along with
               `customer_id` query parameter.
        :param str customer_id: (optional) Enterprise ID of the child customer for
               which the report is requested. This parameter cannot be used along with
               `reseller_id` query parameter.
        :param bool children: (optional) Get report rolled-up to the direct
               children of the requested entity. Defaults to false. This parameter cannot
               be used along with `customer_id` query parameter.
        :param str month: (optional) The billing month for which the usage report
               is requested. Format is `yyyy-mm`. Defaults to current month.
        :param str viewpoint: (optional) Enables partner to view the cost of
               provisioned services as applicable at each level of the hierarchy. Defaults
               to the type of the calling partner. The valid values are `DISTRIBUTOR`,
               `RESELLER` and `END_CUSTOMER`.
        :param bool recurse: (optional) Get usage report rolled-up to the end
               customers of the requested entity. Defaults to false. This parameter cannot
               be used along with `reseller_id` query parameter or `customer_id` query
               parameter.
        :param int limit: (optional) Number of usage records to be returned. The
               default value is 30. Maximum value is 200.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._partner_id = partner_id
        self._reseller_id = reseller_id
        self._customer_id = customer_id
        self._children = children
        self._month = month
        self._viewpoint = viewpoint
        self._recurse = recurse
        self._limit = limit

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of PartnerUsageReport.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.get_resource_usage_report(
            partner_id=self._partner_id,
            reseller_id=self._reseller_id,
            customer_id=self._customer_id,
            children=self._children,
            month=self._month,
            viewpoint=self._viewpoint,
            recurse=self._recurse,
            limit=self._limit,
            offset=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next')
        if next_page_link is not None:
            next = next_page_link.get('offset')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('reports')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of PartnerUsageReport.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results
