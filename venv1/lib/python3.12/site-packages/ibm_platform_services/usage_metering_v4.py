# coding: utf-8

# (C) Copyright IBM Corp. 2020.
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

# IBM OpenAPI SDK Code Generator Version: 99-SNAPSHOT-d753183b-20201209-163011

"""
IBM Cloud Usage Metering is a platform service that enables service providers to submit
metrics collected for  resource instances provisioned by IBM Cloud users. IBM and
third-party service providers that are delivering  an integrated billing service in IBM
Cloud are required to submit usage for all active service instances every hour.  This is
important because inability to report usage can lead to loss of revenue collection for
IBM,  in turn causing loss of revenue share for the service providers.
"""

from typing import Dict, List
import json

from ibm_cloud_sdk_core import BaseService, DetailedResponse
from ibm_cloud_sdk_core.authenticators.authenticator import Authenticator
from ibm_cloud_sdk_core.get_authenticator import get_authenticator_from_environment
from ibm_cloud_sdk_core.utils import convert_model

from .common import get_sdk_headers

##############################################################################
# Service
##############################################################################


class UsageMeteringV4(BaseService):
    """The usage_metering V4 service."""

    DEFAULT_SERVICE_URL = 'https://billing.cloud.ibm.com'
    DEFAULT_SERVICE_NAME = 'usage_metering'

    @classmethod
    def new_instance(
        cls,
        service_name: str = DEFAULT_SERVICE_NAME,
    ) -> 'UsageMeteringV4':
        """
        Return a new client for the usage_metering service using the specified
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
        Construct a new client for the usage_metering service.

        :param Authenticator authenticator: The authenticator specifies the authentication mechanism.
               Get up to date information from https://github.com/IBM/python-sdk-core/blob/master/README.md
               about initializing the authenticator of your choice.
        """
        BaseService.__init__(self, service_url=self.DEFAULT_SERVICE_URL, authenticator=authenticator)

    #########################
    # Resource Usage
    #########################

    def report_resource_usage(
        self, resource_id: str, resource_usage: List['ResourceInstanceUsage'], **kwargs
    ) -> DetailedResponse:
        """
        Report Resource Controller resource usage.

        Report usage for resource instances that were provisioned through the resource
        controller.

        :param str resource_id: The resource for which the usage is submitted.
        :param List[ResourceInstanceUsage] resource_usage:
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ResponseAccepted` object
        """

        if resource_id is None:
            raise ValueError('resource_id must be provided')
        if resource_usage is None:
            raise ValueError('resource_usage must be provided')
        resource_usage = [convert_model(x) for x in resource_usage]
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V4', operation_id='report_resource_usage'
        )
        headers.update(sdk_headers)

        data = json.dumps(resource_usage)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['resource_id']
        path_param_values = self.encode_path_vars(resource_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v4/metering/resources/{resource_id}/usage'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request)
        return response


##############################################################################
# Models
##############################################################################


class MeasureAndQuantity:
    """
    A usage measurement.

    :attr str measure: The name of the measure.
    :attr object quantity: For consumption-based submissions, `quantity` can be a
          double or integer value. For event-based submissions that do not have binary
          states, previous and current values are required, such as `{ "previous": 1,
          "current": 2 }`.
    """

    def __init__(self, measure: str, quantity: object) -> None:
        """
        Initialize a MeasureAndQuantity object.

        :param str measure: The name of the measure.
        :param object quantity: For consumption-based submissions, `quantity` can
               be a double or integer value. For event-based submissions that do not have
               binary states, previous and current values are required, such as `{
               "previous": 1, "current": 2 }`.
        """
        self.measure = measure
        self.quantity = quantity

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'MeasureAndQuantity':
        """Initialize a MeasureAndQuantity object from a json dictionary."""
        args = {}
        if 'measure' in _dict:
            args['measure'] = _dict.get('measure')
        else:
            raise ValueError('Required property \'measure\' not present in MeasureAndQuantity JSON')
        if 'quantity' in _dict:
            args['quantity'] = _dict.get('quantity')
        else:
            raise ValueError('Required property \'quantity\' not present in MeasureAndQuantity JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a MeasureAndQuantity object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'measure') and self.measure is not None:
            _dict['measure'] = self.measure
        if hasattr(self, 'quantity') and self.quantity is not None:
            _dict['quantity'] = self.quantity
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this MeasureAndQuantity object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'MeasureAndQuantity') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'MeasureAndQuantity') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ResourceInstanceUsage:
    """
    Usage information for a resource instance.

    :attr str resource_instance_id: The ID of the instance that incurred the usage.
          The ID is a CRN for instances that are provisioned with the resource controller.
    :attr str plan_id: The plan with which the instance's usage should be metered.
    :attr str region: (optional) The pricing region to which the usage must be
          aggregated. This field is required if the ID is not a CRN or if the CRN does not
          have a region.
    :attr int start: The time from which the resource instance was metered in the
          format milliseconds since epoch.
    :attr int end: The time until which the resource instance was metered in the
          format milliseconds since epoch. This value is the same as start value for
          event-based submissions.
    :attr List[MeasureAndQuantity] measured_usage: Usage measurements for the
          resource instance.
    :attr str consumer_id: (optional) If an instance's usage should be aggregated at
          the consumer level, specify the ID of the consumer. Usage is accumulated to the
          instance-consumer combination.
    """

    def __init__(
        self,
        resource_instance_id: str,
        plan_id: str,
        start: int,
        end: int,
        measured_usage: List['MeasureAndQuantity'],
        *,
        region: str = None,
        consumer_id: str = None
    ) -> None:
        """
        Initialize a ResourceInstanceUsage object.

        :param str resource_instance_id: The ID of the instance that incurred the
               usage. The ID is a CRN for instances that are provisioned with the resource
               controller.
        :param str plan_id: The plan with which the instance's usage should be
               metered.
        :param int start: The time from which the resource instance was metered in
               the format milliseconds since epoch.
        :param int end: The time until which the resource instance was metered in
               the format milliseconds since epoch. This value is the same as start value
               for event-based submissions.
        :param List[MeasureAndQuantity] measured_usage: Usage measurements for the
               resource instance.
        :param str region: (optional) The pricing region to which the usage must be
               aggregated. This field is required if the ID is not a CRN or if the CRN
               does not have a region.
        :param str consumer_id: (optional) If an instance's usage should be
               aggregated at the consumer level, specify the ID of the consumer. Usage is
               accumulated to the instance-consumer combination.
        """
        self.resource_instance_id = resource_instance_id
        self.plan_id = plan_id
        self.region = region
        self.start = start
        self.end = end
        self.measured_usage = measured_usage
        self.consumer_id = consumer_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourceInstanceUsage':
        """Initialize a ResourceInstanceUsage object from a json dictionary."""
        args = {}
        if 'resource_instance_id' in _dict:
            args['resource_instance_id'] = _dict.get('resource_instance_id')
        else:
            raise ValueError('Required property \'resource_instance_id\' not present in ResourceInstanceUsage JSON')
        if 'plan_id' in _dict:
            args['plan_id'] = _dict.get('plan_id')
        else:
            raise ValueError('Required property \'plan_id\' not present in ResourceInstanceUsage JSON')
        if 'region' in _dict:
            args['region'] = _dict.get('region')
        if 'start' in _dict:
            args['start'] = _dict.get('start')
        else:
            raise ValueError('Required property \'start\' not present in ResourceInstanceUsage JSON')
        if 'end' in _dict:
            args['end'] = _dict.get('end')
        else:
            raise ValueError('Required property \'end\' not present in ResourceInstanceUsage JSON')
        if 'measured_usage' in _dict:
            args['measured_usage'] = [MeasureAndQuantity.from_dict(x) for x in _dict.get('measured_usage')]
        else:
            raise ValueError('Required property \'measured_usage\' not present in ResourceInstanceUsage JSON')
        if 'consumer_id' in _dict:
            args['consumer_id'] = _dict.get('consumer_id')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourceInstanceUsage object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'resource_instance_id') and self.resource_instance_id is not None:
            _dict['resource_instance_id'] = self.resource_instance_id
        if hasattr(self, 'plan_id') and self.plan_id is not None:
            _dict['plan_id'] = self.plan_id
        if hasattr(self, 'region') and self.region is not None:
            _dict['region'] = self.region
        if hasattr(self, 'start') and self.start is not None:
            _dict['start'] = self.start
        if hasattr(self, 'end') and self.end is not None:
            _dict['end'] = self.end
        if hasattr(self, 'measured_usage') and self.measured_usage is not None:
            _dict['measured_usage'] = [x.to_dict() for x in self.measured_usage]
        if hasattr(self, 'consumer_id') and self.consumer_id is not None:
            _dict['consumer_id'] = self.consumer_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResourceInstanceUsage object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourceInstanceUsage') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourceInstanceUsage') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ResourceUsageDetails:
    """
    Resource usage details.

    :attr int status: A response code similar to HTTP status codes.
    :attr str location: The location of the usage.
    :attr str code: (optional) The error code that was encountered.
    :attr str message: (optional) A description of the error.
    """

    def __init__(self, status: int, location: str, *, code: str = None, message: str = None) -> None:
        """
        Initialize a ResourceUsageDetails object.

        :param int status: A response code similar to HTTP status codes.
        :param str location: The location of the usage.
        :param str code: (optional) The error code that was encountered.
        :param str message: (optional) A description of the error.
        """
        self.status = status
        self.location = location
        self.code = code
        self.message = message

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourceUsageDetails':
        """Initialize a ResourceUsageDetails object from a json dictionary."""
        args = {}
        if 'status' in _dict:
            args['status'] = _dict.get('status')
        else:
            raise ValueError('Required property \'status\' not present in ResourceUsageDetails JSON')
        if 'location' in _dict:
            args['location'] = _dict.get('location')
        else:
            raise ValueError('Required property \'location\' not present in ResourceUsageDetails JSON')
        if 'code' in _dict:
            args['code'] = _dict.get('code')
        if 'message' in _dict:
            args['message'] = _dict.get('message')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourceUsageDetails object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'status') and self.status is not None:
            _dict['status'] = self.status
        if hasattr(self, 'location') and self.location is not None:
            _dict['location'] = self.location
        if hasattr(self, 'code') and self.code is not None:
            _dict['code'] = self.code
        if hasattr(self, 'message') and self.message is not None:
            _dict['message'] = self.message
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResourceUsageDetails object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourceUsageDetails') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourceUsageDetails') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ResponseAccepted:
    """
    Response when usage submitted is accepted.

    :attr List[ResourceUsageDetails] resources: Response body that contains the
          status of each submitted usage record.
    """

    def __init__(self, resources: List['ResourceUsageDetails']) -> None:
        """
        Initialize a ResponseAccepted object.

        :param List[ResourceUsageDetails] resources: Response body that contains
               the status of each submitted usage record.
        """
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResponseAccepted':
        """Initialize a ResponseAccepted object from a json dictionary."""
        args = {}
        if 'resources' in _dict:
            args['resources'] = [ResourceUsageDetails.from_dict(x) for x in _dict.get('resources')]
        else:
            raise ValueError('Required property \'resources\' not present in ResponseAccepted JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResponseAccepted object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'resources') and self.resources is not None:
            _dict['resources'] = [x.to_dict() for x in self.resources]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResponseAccepted object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResponseAccepted') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResponseAccepted') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other
