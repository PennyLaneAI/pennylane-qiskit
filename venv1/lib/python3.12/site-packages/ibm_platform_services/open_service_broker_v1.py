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
Contribute resources to the IBM Cloud catalog by implementing a `service broker` that
conforms to the [Open Service Broker
API](https://github.com/openservicebrokerapi/servicebroker/blob/master/spec.md) version
2.12  specification and provides enablement extensions for integration with IBM Cloud and
the Resource Controller provisioning model.
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


class OpenServiceBrokerV1(BaseService):
    """The Open Service Broker V1 service."""

    DEFAULT_SERVICE_URL = None
    DEFAULT_SERVICE_NAME = 'open_service_broker'

    @classmethod
    def new_instance(
        cls,
        service_name: str = DEFAULT_SERVICE_NAME,
    ) -> 'OpenServiceBrokerV1':
        """
        Return a new client for the Open Service Broker service using the specified
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
        Construct a new client for the Open Service Broker service.

        :param Authenticator authenticator: The authenticator specifies the authentication mechanism.
               Get up to date information from https://github.com/IBM/python-sdk-core/blob/master/README.md
               about initializing the authenticator of your choice.
        """
        BaseService.__init__(self, service_url=self.DEFAULT_SERVICE_URL, authenticator=authenticator)

    #########################
    # Enable and Disable Instances
    #########################

    def get_service_instance_state(self, instance_id: str, **kwargs) -> DetailedResponse:
        """
        Get the current state of the service instance.

        Get the current state information associated with the service instance.
        As a service provider you need a way to manage provisioned service instances.  If
        an account comes past due, you may need a to disable the service (without deleting
        it), and when the account is settled re-enable the service.
        This endpoint allows both the provider and IBM Cloud to query for the state of a
        provisioned service instance.  For example, IBM Cloud may query the provider to
        figure out if a given service is disabled or not and present that state to the
        user.

        :param str instance_id: The `instance_id` of a service instance is provided
               by the IBM Cloud platform. This ID will be used for future requests to bind
               and deprovision, so the broker can use it to correlate the resource it
               creates.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Resp1874644Root` object
        """

        if instance_id is None:
            raise ValueError('instance_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_service_instance_state'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['instance_id']
        path_param_values = self.encode_path_vars(instance_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/bluemix_v1/service_instances/{instance_id}'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request)
        return response

    def replace_service_instance_state(
        self, instance_id: str, *, enabled: bool = None, initiator_id: str = None, reason_code: str = None, **kwargs
    ) -> DetailedResponse:
        """
        Update the state of a provisioned service instance.

        Update (disable or enable) the state of a provisioned service instance. As a
        service provider you need a way to manage provisioned service instances. If an
        account comes past due, you may need a to disable the service (without deleting
        it), and when the account is settled re-enable the service. This endpoint allows
        the provider to enable or disable the state of a provisioned service instance. It
        is the service provider's responsibility to disable access to the service instance
        when the disable endpoint is invoked and to re-enable that access when the enable
        endpoint is invoked. When your service broker receives an enable / disable
        request, it should take whatever action is necessary to enable / disable
        (respectively) the service.  Additionally, If a bind request comes in for a
        disabled service, the broker should reject that request with any code other than
        `204`, and provide a user-facing message in the description.

        :param str instance_id: The `instance_id` of a service instance is provided
               by the IBM Cloud platform. This ID will be used for future requests to bind
               and deprovision, so the broker can use it to correlate the resource it
               creates.
        :param bool enabled: (optional) Indicates the current state of the service
               instance.
        :param str initiator_id: (optional) Optional string that shows the user ID
               that is initiating the call.
        :param str reason_code: (optional) Optional string that states the reason
               code for the service instance state change. Valid values are
               `IBMCLOUD_ACCT_ACTIVATE`, `IBMCLOUD_RECLAMATION_RESTORE`, or
               `IBMCLOUD_SERVICE_INSTANCE_BELOW_CAP` for enable calls;
               `IBMCLOUD_ACCT_SUSPEND`, `IBMCLOUD_RECLAMATION_SCHEDULE`, or
               `IBMCLOUD_SERVICE_INSTANCE_ABOVE_CAP` for disable calls; and
               `IBMCLOUD_ADMIN_REQUEST` for enable and disable calls.<br/><br/>Previously
               accepted values had a `BMX_` prefix, such as `BMX_ACCT_ACTIVATE`. These
               values are deprecated.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Resp2448145Root` object
        """

        if instance_id is None:
            raise ValueError('instance_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='replace_service_instance_state'
        )
        headers.update(sdk_headers)

        data = {'enabled': enabled, 'initiator_id': initiator_id, 'reason_code': reason_code}
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['instance_id']
        path_param_values = self.encode_path_vars(instance_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/bluemix_v1/service_instances/{instance_id}'.format(**path_param_dict)
        request = self.prepare_request(method='PUT', url=url, headers=headers, data=data)

        response = self.send(request)
        return response

    #########################
    # Resource Instances
    #########################

    def replace_service_instance(
        self,
        instance_id: str,
        *,
        organization_guid: str = None,
        plan_id: str = None,
        service_id: str = None,
        space_guid: str = None,
        context: 'Context' = None,
        parameters: dict = None,
        accepts_incomplete: bool = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Create (provision) a service instance.

        Create a service instance with GUID. When your service broker receives a provision
        request from the IBM Cloud platform, it MUST take whatever action is necessary to
        create a new resource.
        When a user creates a service instance from the IBM Cloud console or the IBM Cloud
        CLI, the IBM Cloud platform validates that the user has permission to create the
        service instance using IBM Cloud IAM. After this validation occurs, your service
        broker's provision endpoint (PUT /v2/resource_instances/:instance_id) will be
        invoked. When provisioning occurs, the IBM Cloud platform provides the following
        values:
        - The IBM Cloud context is included in the context variable
        - The X-Broker-API-Originating-Identity will have the IBM IAM ID of the user that
        initiated the request
        - The parameters section will include the requested location (and additional
        parameters required by your service).

        :param str instance_id: The `instance_id` of a service instance is provided
               by the IBM Cloud platform. This ID will be used for future requests to bind
               and deprovision, so the broker can use it to correlate the resource it
               creates.
        :param str organization_guid: (optional) Deprecated in favor of `context`.
               The IBM Cloud platform GUID for the organization under which the service
               instance is to be provisioned. Although most brokers will not use this
               field, it might be helpful for executing operations on a user's behalf. It
               MUST be a non-empty string.
        :param str plan_id: (optional) The ID of the plan for which the service
               instance has been requested, which is stored in the catalog.json of your
               broker. This value should be a GUID and it MUST be unique to a service.
        :param str service_id: (optional) The ID of the service stored in the
               catalog.json of your broker. This value should be a GUID and it MUST be a
               non-empty string.
        :param str space_guid: (optional) Deprecated in favor of `context`. The
               identifier for the project space within the IBM Cloud platform
               organization. Although most brokers will not use this field, it might be
               helpful for executing operations on a user's behalf. It MUST be a non-empty
               string.
        :param Context context: (optional) Platform specific contextual information
               under which the service instance is to be provisioned.
        :param dict parameters: (optional) Configuration options for the service
               instance. An opaque object, controller treats this as a blob. Brokers
               should ensure that the client has provided valid configuration parameters
               and values for the operation. If this field is not present in the request
               message, then the broker MUST NOT change the parameters of the instance as
               a result of this request.
        :param bool accepts_incomplete: (optional) A value of true indicates that
               both the IBM Cloud platform and the requesting client support asynchronous
               deprovisioning. If this parameter is not included in the request, and the
               broker can only deprovision a service instance of the requested plan
               asynchronously, the broker MUST reject the request with a `422`
               Unprocessable Entity.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Resp2079872Root` object
        """

        if instance_id is None:
            raise ValueError('instance_id must be provided')
        if context is not None:
            context = convert_model(context)
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='replace_service_instance'
        )
        headers.update(sdk_headers)

        params = {'accepts_incomplete': accepts_incomplete}

        data = {
            'organization_guid': organization_guid,
            'plan_id': plan_id,
            'service_id': service_id,
            'space_guid': space_guid,
            'context': context,
            'parameters': parameters,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['instance_id']
        path_param_values = self.encode_path_vars(instance_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/service_instances/{instance_id}'.format(**path_param_dict)
        request = self.prepare_request(method='PUT', url=url, headers=headers, params=params, data=data)

        response = self.send(request)
        return response

    def update_service_instance(
        self,
        instance_id: str,
        *,
        service_id: str = None,
        context: 'Context' = None,
        parameters: dict = None,
        plan_id: str = None,
        previous_values: dict = None,
        accepts_incomplete: bool = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Update a service instance.

        Patch an instance by GUID. Enabling this endpoint allows your user to change plans
        and service parameters in a provisioned service instance. If your offering
        supports multiple plans, and you want users to be able to change plans for a
        provisioned instance, you will need to enable the ability for users to update
        their service instance.
        To enable support for the update of the plan, a broker MUST declare support per
        service by specifying  `"plan_updateable": true` in your brokers' catalog.json.

        :param str instance_id: The ID of a previously provisioned service
               instance.
        :param str service_id: (optional) The ID of the service stored in the
               catalog.json of your broker. This value should be a GUID. It MUST be a
               non-empty string.
        :param Context context: (optional) Platform specific contextual information
               under which the service instance is to be provisioned.
        :param dict parameters: (optional) Configuration options for the service
               instance. An opaque object, controller treats this as a blob. Brokers
               should ensure that the client has provided valid configuration parameters
               and values for the operation. If this field is not present in the request
               message, then the broker MUST NOT change the parameters of the instance as
               a result of this request.
        :param str plan_id: (optional) The ID of the plan for which the service
               instance has been requested, which is stored in the catalog.json of your
               broker. This value should be a GUID. MUST be unique to a service. If
               present, MUST be a non-empty string. If this field is not present in the
               request message, then the broker MUST NOT change the plan of the instance
               as a result of this request.
        :param dict previous_values: (optional) Information about the service
               instance prior to the update.
        :param bool accepts_incomplete: (optional) A value of true indicates that
               both the IBM Cloud platform and the requesting client support asynchronous
               deprovisioning. If this parameter is not included in the request, and the
               broker can only deprovision a service instance of the requested plan
               asynchronously, the broker MUST reject the request with a `422`
               Unprocessable Entity.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Resp2079874Root` object
        """

        if instance_id is None:
            raise ValueError('instance_id must be provided')
        if context is not None:
            context = convert_model(context)
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='update_service_instance'
        )
        headers.update(sdk_headers)

        params = {'accepts_incomplete': accepts_incomplete}

        data = {
            'service_id': service_id,
            'context': context,
            'parameters': parameters,
            'plan_id': plan_id,
            'previous_values': previous_values,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['instance_id']
        path_param_values = self.encode_path_vars(instance_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/service_instances/{instance_id}'.format(**path_param_dict)
        request = self.prepare_request(method='PATCH', url=url, headers=headers, params=params, data=data)

        response = self.send(request)
        return response

    def delete_service_instance(
        self, service_id: str, plan_id: str, instance_id: str, *, accepts_incomplete: bool = None, **kwargs
    ) -> DetailedResponse:
        """
        Delete (deprovision) a service instance.

        Delete (deprovision) a service instance by GUID. When a service broker receives a
        deprovision request from the IBM Cloud platform, it MUST delete any resources it
        created during the provision. Usually this means that all resources are
        immediately reclaimed for future provisions.

        :param str service_id: The ID of the service stored in the catalog.json of
               your broker. This value should be a GUID. MUST be a non-empty string.
        :param str plan_id: The ID of the plan for which the service instance has
               been requested, which is stored in the catalog.json of your broker. This
               value should be a GUID. MUST be a non-empty string.
        :param str instance_id: The ID of a previously provisioned service
               instance.
        :param bool accepts_incomplete: (optional) A value of true indicates that
               both the IBM Cloud platform and the requesting client support asynchronous
               deprovisioning. If this parameter is not included in the request, and the
               broker can only deprovision a service instance of the requested plan
               asynchronously, the broker MUST reject the request with a `422`
               Unprocessable Entity.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Resp2079874Root` object
        """

        if service_id is None:
            raise ValueError('service_id must be provided')
        if plan_id is None:
            raise ValueError('plan_id must be provided')
        if instance_id is None:
            raise ValueError('instance_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='delete_service_instance'
        )
        headers.update(sdk_headers)

        params = {'service_id': service_id, 'plan_id': plan_id, 'accepts_incomplete': accepts_incomplete}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['instance_id']
        path_param_values = self.encode_path_vars(instance_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/service_instances/{instance_id}'.format(**path_param_dict)
        request = self.prepare_request(method='DELETE', url=url, headers=headers, params=params)

        response = self.send(request)
        return response

    #########################
    # Catalog
    #########################

    def list_catalog(self, **kwargs) -> DetailedResponse:
        """
        Get the catalog metadata stored within the broker.

        This endpoints defines the contract between the broker and the IBM Cloud platform
        for the services and plans that the broker supports. This endpoint returns the
        catalog metadata stored within your broker. These values define the minimal
        provisioning contract between your service and the IBM Cloud platform. All
        additional catalog metadata that is not required for provisioning is stored within
        the IBM Cloud catalog, and any updates to catalog display values that are used to
        render your dashboard like links, icons, and i18n translated metadata should be
        updated in the Resource Management Console (RMC), and not housed in your broker.
        None of metadata stored in your broker is displayed in the IBM Cloud console or
        the IBM Cloud CLI; the console and CLI will return what was set withn RMC and
        stored in the IBM Cloud catalog.

        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Resp1874650Root` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='list_catalog'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        url = '/v2/catalog'
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request)
        return response

    #########################
    # Last Operation (Async)
    #########################

    def get_last_operation(
        self, instance_id: str, *, operation: str = None, plan_id: str = None, service_id: str = None, **kwargs
    ) -> DetailedResponse:
        """
        Get the current status of a provision in-progress for a service instance.

        Get `last_operation` for instance by GUID (for asynchronous provision calls). When
        a broker returns status code `202 Accepted` during a provision, update, or
        deprovision call, the IBM Cloud platform will begin polling the `last_operation`
        endpoint to obtain the state of the last requested operation. The broker response
        MUST contain the field `state` and MAY contain the field `description`.
        Valid values for `state` are `in progress`, `succeeded`, and `failed`. The
        platform will poll the `last_operation `endpoint as long as the broker returns
        "state": "in progress". Returning "state": "succeeded" or "state": "failed" will
        cause the platform to cease polling. The value provided for description will be
        passed through to the platform API client and can be used to provide additional
        detail for users about the progress of the operation.

        :param str instance_id: The unique instance ID generated during
               provisioning by the IBM Cloud platform.
        :param str operation: (optional) A broker-provided identifier for the
               operation. When a value for operation is included with asynchronous
               responses for provision and update, and deprovision requests, the IBM Cloud
               platform will provide the same value using this query parameter as a
               URL-encoded string. If present, MUST be a non-empty string.
        :param str plan_id: (optional) ID of the plan from the catalog.json in your
               broker. If present, MUST be a non-empty string.
        :param str service_id: (optional) ID of the service from the catalog.json
               in your service broker. If present, MUST be a non-empty string.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Resp2079894Root` object
        """

        if instance_id is None:
            raise ValueError('instance_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_last_operation'
        )
        headers.update(sdk_headers)

        params = {'operation': operation, 'plan_id': plan_id, 'service_id': service_id}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['instance_id']
        path_param_values = self.encode_path_vars(instance_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/service_instances/{instance_id}/last_operation'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request)
        return response

    #########################
    # Bindings and Credentials
    #########################

    def replace_service_binding(
        self,
        binding_id: str,
        instance_id: str,
        *,
        plan_id: str = None,
        service_id: str = None,
        bind_resource: 'BindResource' = None,
        parameters: dict = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Bind a service instance to another resource.

        Create binding by GUID on service instance.
        If your service can be bound to applications in IBM Cloud, `bindable:true` must be
        specified in the catalog.json of your service broker. If bindable, it must be able
        to return API endpoints and credentials to your service consumers.
        **Note:** Brokers that do not offer any bindable services do not need to implement
        the endpoint for bind requests.
        See the OSB 2.12 spec for more details on
        [binding](https://github.com/openservicebrokerapi/servicebroker/blob/v2.12/spec.md#binding).

        :param str binding_id: The `binding_id` is provided by the IBM Cloud
               platform. This ID will be used for future unbind requests, so the broker
               can use it to correlate the resource it creates.
        :param str instance_id: The :`instance_id` is the ID of a previously
               provisioned service instance.
        :param str plan_id: (optional) The ID of the plan from the catalog.json in
               your broker. If present, it MUST be a non-empty string.
        :param str service_id: (optional) The ID of the service from the
               catalog.json in your broker. If present, it MUST be a non-empty string.
        :param BindResource bind_resource: (optional) A JSON object that contains
               data for platform resources associated with the binding to be created.
        :param dict parameters: (optional) Configuration options for the service
               instance. An opaque object, controller treats this as a blob. Brokers
               should ensure that the client has provided valid configuration parameters
               and values for the operation. If this field is not present in the request
               message, then the broker MUST NOT change the parameters of the instance as
               a result of this request.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Resp2079876Root` object
        """

        if binding_id is None:
            raise ValueError('binding_id must be provided')
        if instance_id is None:
            raise ValueError('instance_id must be provided')
        if bind_resource is not None:
            bind_resource = convert_model(bind_resource)
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='replace_service_binding'
        )
        headers.update(sdk_headers)

        data = {'plan_id': plan_id, 'service_id': service_id, 'bind_resource': bind_resource, 'parameters': parameters}
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['binding_id', 'instance_id']
        path_param_values = self.encode_path_vars(binding_id, instance_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/service_instances/{instance_id}/service_bindings/{binding_id}'.format(**path_param_dict)
        request = self.prepare_request(method='PUT', url=url, headers=headers, data=data)

        response = self.send(request)
        return response

    def delete_service_binding(
        self, binding_id: str, instance_id: str, plan_id: str, service_id: str, **kwargs
    ) -> DetailedResponse:
        """
        Delete (unbind) the credentials bound to a resource.

        Delete instance binding by GUID.
        When a broker receives an unbind request from the IBM Cloud platform, it MUST
        delete any resources associated with the binding. In the case where credentials
        were generated, this might result in requests to the service instance failing to
        authenticate.
        **Note**: Brokers that do not provide any bindable services or plans do not need
        to implement this endpoint.

        :param str binding_id: The `binding_id` is the ID of a previously
               provisioned binding for that service instance.
        :param str instance_id: The `instance_id` is the ID of a previously
               provisioned service instance.
        :param str plan_id: The ID of the plan from the catalog.json in the broker.
               It MUST be a non-empty string and should be a GUID.
        :param str service_id: The ID of the service from the catalog.json in the
               broker. It MUST be a non-empty string and should be a GUID.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if binding_id is None:
            raise ValueError('binding_id must be provided')
        if instance_id is None:
            raise ValueError('instance_id must be provided')
        if plan_id is None:
            raise ValueError('plan_id must be provided')
        if service_id is None:
            raise ValueError('service_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='delete_service_binding'
        )
        headers.update(sdk_headers)

        params = {'plan_id': plan_id, 'service_id': service_id}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['binding_id', 'instance_id']
        path_param_values = self.encode_path_vars(binding_id, instance_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/v2/service_instances/{instance_id}/service_bindings/{binding_id}'.format(**path_param_dict)
        request = self.prepare_request(method='DELETE', url=url, headers=headers, params=params)

        response = self.send(request)
        return response


##############################################################################
# Models
##############################################################################


class Resp1874644Root:
    """
    Check the active status of an enabled service.

    :attr bool active: (optional) Indicates (from the viewpoint of the provider)
          whether the service instance is active and is meaningful if enabled is true. The
          default value is true if not specified.
    :attr bool enabled: (optional) Indicates the current state of the service
          instance.
    :attr float last_active: (optional) Indicates when the service instance was last
          accessed/modified/etc., and is meaningful if enabled is true AND active is
          false. Represented as milliseconds since the epoch, but does not need to be
          accurate to the second/hour.
    """

    def __init__(self, *, active: bool = None, enabled: bool = None, last_active: float = None) -> None:
        """
        Initialize a Resp1874644Root object.

        :param bool active: (optional) Indicates (from the viewpoint of the
               provider) whether the service instance is active and is meaningful if
               enabled is true. The default value is true if not specified.
        :param bool enabled: (optional) Indicates the current state of the service
               instance.
        :param float last_active: (optional) Indicates when the service instance
               was last accessed/modified/etc., and is meaningful if enabled is true AND
               active is false. Represented as milliseconds since the epoch, but does not
               need to be accurate to the second/hour.
        """
        self.active = active
        self.enabled = enabled
        self.last_active = last_active

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Resp1874644Root':
        """Initialize a Resp1874644Root object from a json dictionary."""
        args = {}
        if 'active' in _dict:
            args['active'] = _dict.get('active')
        if 'enabled' in _dict:
            args['enabled'] = _dict.get('enabled')
        if 'last_active' in _dict:
            args['last_active'] = _dict.get('last_active')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Resp1874644Root object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'active') and self.active is not None:
            _dict['active'] = self.active
        if hasattr(self, 'enabled') and self.enabled is not None:
            _dict['enabled'] = self.enabled
        if hasattr(self, 'last_active') and self.last_active is not None:
            _dict['last_active'] = self.last_active
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Resp1874644Root object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Resp1874644Root') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Resp1874644Root') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Resp1874650Root:
    """
    Resp1874650Root.

    :attr List[Services] services: (optional) List of services.
    """

    def __init__(self, *, services: List['Services'] = None) -> None:
        """
        Initialize a Resp1874650Root object.

        :param List[Services] services: (optional) List of services.
        """
        self.services = services

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Resp1874650Root':
        """Initialize a Resp1874650Root object from a json dictionary."""
        args = {}
        if 'services' in _dict:
            args['services'] = [Services.from_dict(x) for x in _dict.get('services')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Resp1874650Root object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'services') and self.services is not None:
            _dict['services'] = [x.to_dict() for x in self.services]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Resp1874650Root object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Resp1874650Root') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Resp1874650Root') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Resp2079872Root:
    """
    OK - MUST be returned if the service instance already exists, is fully provisioned,
    and the requested parameters are identical to the existing service instance.

    :attr str dashboard_url: (optional) The URL of a web-based management user
          interface for the service instance; we refer to this as a service dashboard. The
          URL MUST contain enough information for the dashboard to identify the resource
          being accessed. Note: a broker that wishes to return `dashboard_url` for a
          service instance MUST return it with the initial response to the provision
          request, even if the service is provisioned asynchronously. If present, it MUST
          be a non-empty string.
    :attr str operation: (optional) For asynchronous responses, service brokers MAY
          return an identifier representing the operation. The value of this field MUST be
          provided by the platform with requests to the `last_operation` endpoint in a URL
          encoded query parameter. If present, MUST be a non-empty string.
    """

    def __init__(self, *, dashboard_url: str = None, operation: str = None) -> None:
        """
        Initialize a Resp2079872Root object.

        :param str dashboard_url: (optional) The URL of a web-based management user
               interface for the service instance; we refer to this as a service
               dashboard. The URL MUST contain enough information for the dashboard to
               identify the resource being accessed. Note: a broker that wishes to return
               `dashboard_url` for a service instance MUST return it with the initial
               response to the provision request, even if the service is provisioned
               asynchronously. If present, it MUST be a non-empty string.
        :param str operation: (optional) For asynchronous responses, service
               brokers MAY return an identifier representing the operation. The value of
               this field MUST be provided by the platform with requests to the
               `last_operation` endpoint in a URL encoded query parameter. If present,
               MUST be a non-empty string.
        """
        self.dashboard_url = dashboard_url
        self.operation = operation

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Resp2079872Root':
        """Initialize a Resp2079872Root object from a json dictionary."""
        args = {}
        if 'dashboard_url' in _dict:
            args['dashboard_url'] = _dict.get('dashboard_url')
        if 'operation' in _dict:
            args['operation'] = _dict.get('operation')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Resp2079872Root object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'dashboard_url') and self.dashboard_url is not None:
            _dict['dashboard_url'] = self.dashboard_url
        if hasattr(self, 'operation') and self.operation is not None:
            _dict['operation'] = self.operation
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Resp2079872Root object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Resp2079872Root') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Resp2079872Root') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Resp2079874Root:
    """
    Accepted - MUST be returned if the service instance provisioning is in progress. This
    triggers the IBM Cloud platform to poll the Service Instance `last_operation` Endpoint
    for operation status. Note that a re-sent `PUT` request MUST return a `202 Accepted`,
    not a `200 OK`, if the service instance is not yet fully provisioned.

    :attr str operation: (optional) For asynchronous responses, service brokers MAY
          return an identifier representing the operation. The value of this field MUST be
          provided by the platform with requests to the Last Operation endpoint in a URL
          encoded query parameter. If present, MUST be a non-empty string.
    """

    def __init__(self, *, operation: str = None) -> None:
        """
        Initialize a Resp2079874Root object.

        :param str operation: (optional) For asynchronous responses, service
               brokers MAY return an identifier representing the operation. The value of
               this field MUST be provided by the platform with requests to the Last
               Operation endpoint in a URL encoded query parameter. If present, MUST be a
               non-empty string.
        """
        self.operation = operation

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Resp2079874Root':
        """Initialize a Resp2079874Root object from a json dictionary."""
        args = {}
        if 'operation' in _dict:
            args['operation'] = _dict.get('operation')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Resp2079874Root object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'operation') and self.operation is not None:
            _dict['operation'] = self.operation
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Resp2079874Root object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Resp2079874Root') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Resp2079874Root') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Resp2079876Root:
    """
    Resp2079876Root.

    :attr object credentials: (optional) A free-form hash of credentials that can be
          used by applications or users to access the service.
    :attr str syslog_drain_url: (optional) A URL to which logs MUST be streamed.
          'requires':['syslog_drain'] MUST be declared in the Catalog endpoint or the
          platform MUST consider the response invalid.
    :attr str route_service_url: (optional) A URL to which the platform MUST proxy
          requests for the address sent with bind_resource.route in the request body.
          'requires':['route_forwarding'] MUST be declared in the Catalog endpoint or the
          platform can consider the response invalid.
    :attr List[VolumeMount] volume_mounts: (optional) An array of configuration for
          remote storage devices to be mounted into an application container filesystem.
          'requires':['volume_mount'] MUST be declared in the Catalog endpoint or the
          platform can consider the response invalid.
    """

    def __init__(
        self,
        *,
        credentials: object = None,
        syslog_drain_url: str = None,
        route_service_url: str = None,
        volume_mounts: List['VolumeMount'] = None
    ) -> None:
        """
        Initialize a Resp2079876Root object.

        :param object credentials: (optional) A free-form hash of credentials that
               can be used by applications or users to access the service.
        :param str syslog_drain_url: (optional) A URL to which logs MUST be
               streamed. 'requires':['syslog_drain'] MUST be declared in the Catalog
               endpoint or the platform MUST consider the response invalid.
        :param str route_service_url: (optional) A URL to which the platform MUST
               proxy requests for the address sent with bind_resource.route in the request
               body. 'requires':['route_forwarding'] MUST be declared in the Catalog
               endpoint or the platform can consider the response invalid.
        :param List[VolumeMount] volume_mounts: (optional) An array of
               configuration for remote storage devices to be mounted into an application
               container filesystem. 'requires':['volume_mount'] MUST be declared in the
               Catalog endpoint or the platform can consider the response invalid.
        """
        self.credentials = credentials
        self.syslog_drain_url = syslog_drain_url
        self.route_service_url = route_service_url
        self.volume_mounts = volume_mounts

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Resp2079876Root':
        """Initialize a Resp2079876Root object from a json dictionary."""
        args = {}
        if 'credentials' in _dict:
            args['credentials'] = _dict.get('credentials')
        if 'syslog_drain_url' in _dict:
            args['syslog_drain_url'] = _dict.get('syslog_drain_url')
        if 'route_service_url' in _dict:
            args['route_service_url'] = _dict.get('route_service_url')
        if 'volume_mounts' in _dict:
            args['volume_mounts'] = [VolumeMount.from_dict(x) for x in _dict.get('volume_mounts')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Resp2079876Root object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'credentials') and self.credentials is not None:
            _dict['credentials'] = self.credentials
        if hasattr(self, 'syslog_drain_url') and self.syslog_drain_url is not None:
            _dict['syslog_drain_url'] = self.syslog_drain_url
        if hasattr(self, 'route_service_url') and self.route_service_url is not None:
            _dict['route_service_url'] = self.route_service_url
        if hasattr(self, 'volume_mounts') and self.volume_mounts is not None:
            _dict['volume_mounts'] = [x.to_dict() for x in self.volume_mounts]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Resp2079876Root object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Resp2079876Root') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Resp2079876Root') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Resp2079894Root:
    """
    OK - MUST be returned upon successful processing of this request.

    :attr str description: (optional) A user-facing message displayed to the
          platform API client. Can be used to tell the user details about the status of
          the operation. If present, MUST be a non-empty string.
    :attr str state: Valid values are `in progress`, `succeeded`, and `failed`.
          While `â€ state": "in progressâ€ `, the platform SHOULD continue polling. A
          response with `â€ state": "succeededâ€ ` or `â€ state": "failedâ€ ` MUST cause
          the platform to cease polling.
    """

    def __init__(self, state: str, *, description: str = None) -> None:
        """
        Initialize a Resp2079894Root object.

        :param str state: Valid values are `in progress`, `succeeded`, and
               `failed`. While `â€ state": "in progressâ€ `, the platform SHOULD continue
               polling. A response with `â€ state": "succeededâ€ ` or `â€ state":
               "failedâ€ ` MUST cause the platform to cease polling.
        :param str description: (optional) A user-facing message displayed to the
               platform API client. Can be used to tell the user details about the status
               of the operation. If present, MUST be a non-empty string.
        """
        self.description = description
        self.state = state

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Resp2079894Root':
        """Initialize a Resp2079894Root object from a json dictionary."""
        args = {}
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        if 'state' in _dict:
            args['state'] = _dict.get('state')
        else:
            raise ValueError('Required property \'state\' not present in Resp2079894Root JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Resp2079894Root object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Resp2079894Root object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Resp2079894Root') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Resp2079894Root') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Resp2448145Root:
    """
    Check the enabled status of active service.

    :attr bool active: (optional) Indicates (from the viewpoint of the provider)
          whether the service instance is active and is meaningful if `enabled` is true.
          The default value is true if not specified.
    :attr bool enabled: Indicates the current state of the service instance.
    :attr int last_active: (optional) Indicates when the service instance was last
          accessed or modified, and is meaningful if `enabled` is true AND `active` is
          false.  Represented as milliseconds since the epoch, but does not need to be
          accurate to the second/hour.
    """

    def __init__(self, enabled: bool, *, active: bool = None, last_active: int = None) -> None:
        """
        Initialize a Resp2448145Root object.

        :param bool enabled: Indicates the current state of the service instance.
        :param bool active: (optional) Indicates (from the viewpoint of the
               provider) whether the service instance is active and is meaningful if
               `enabled` is true.  The default value is true if not specified.
        :param int last_active: (optional) Indicates when the service instance was
               last accessed or modified, and is meaningful if `enabled` is true AND
               `active` is false.  Represented as milliseconds since the epoch, but does
               not need to be accurate to the second/hour.
        """
        self.active = active
        self.enabled = enabled
        self.last_active = last_active

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Resp2448145Root':
        """Initialize a Resp2448145Root object from a json dictionary."""
        args = {}
        if 'active' in _dict:
            args['active'] = _dict.get('active')
        if 'enabled' in _dict:
            args['enabled'] = _dict.get('enabled')
        else:
            raise ValueError('Required property \'enabled\' not present in Resp2448145Root JSON')
        if 'last_active' in _dict:
            args['last_active'] = _dict.get('last_active')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Resp2448145Root object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'active') and self.active is not None:
            _dict['active'] = self.active
        if hasattr(self, 'enabled') and self.enabled is not None:
            _dict['enabled'] = self.enabled
        if hasattr(self, 'last_active') and self.last_active is not None:
            _dict['last_active'] = self.last_active
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Resp2448145Root object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Resp2448145Root') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Resp2448145Root') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class BindResource:
    """
    A JSON object that contains data for platform resources associated with the binding to
    be created.

    :attr str account_id: (optional) Account owner of resource to bind.
    :attr str serviceid_crn: (optional) Service ID of resource to bind.
    :attr str target_crn: (optional) Target ID of resource to bind.
    :attr str app_guid: (optional) GUID of an application associated with the
          binding. For credentials bindings.
    :attr str route: (optional) URL of the application to be intermediated. For
          route services bindings.
    """

    def __init__(
        self,
        *,
        account_id: str = None,
        serviceid_crn: str = None,
        target_crn: str = None,
        app_guid: str = None,
        route: str = None
    ) -> None:
        """
        Initialize a BindResource object.

        :param str account_id: (optional) Account owner of resource to bind.
        :param str serviceid_crn: (optional) Service ID of resource to bind.
        :param str target_crn: (optional) Target ID of resource to bind.
        :param str app_guid: (optional) GUID of an application associated with the
               binding. For credentials bindings.
        :param str route: (optional) URL of the application to be intermediated.
               For route services bindings.
        """
        self.account_id = account_id
        self.serviceid_crn = serviceid_crn
        self.target_crn = target_crn
        self.app_guid = app_guid
        self.route = route

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'BindResource':
        """Initialize a BindResource object from a json dictionary."""
        args = {}
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        if 'serviceid_crn' in _dict:
            args['serviceid_crn'] = _dict.get('serviceid_crn')
        if 'target_crn' in _dict:
            args['target_crn'] = _dict.get('target_crn')
        if 'app_guid' in _dict:
            args['app_guid'] = _dict.get('app_guid')
        if 'route' in _dict:
            args['route'] = _dict.get('route')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a BindResource object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'serviceid_crn') and self.serviceid_crn is not None:
            _dict['serviceid_crn'] = self.serviceid_crn
        if hasattr(self, 'target_crn') and self.target_crn is not None:
            _dict['target_crn'] = self.target_crn
        if hasattr(self, 'app_guid') and self.app_guid is not None:
            _dict['app_guid'] = self.app_guid
        if hasattr(self, 'route') and self.route is not None:
            _dict['route'] = self.route
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this BindResource object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'BindResource') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'BindResource') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Context:
    """
    Platform specific contextual information under which the service instance is to be
    provisioned.

    :attr str account_id: (optional) Returns the ID of the account in IBM Cloud that
          is provisioning the service instance.
    :attr str crn: (optional) When a customer provisions your service in IBM Cloud,
          a service instance is created and this instance is identified by its IBM Cloud
          Resource Name (CRN). The CRN is utilized in all aspects of the interaction with
          IBM Cloud including provisioning, binding (creating credentials and endpoints),
          metering, dashboard display, and access control. From a service provider
          perspective, the CRN can largely be treated as an opaque string to be utilized
          with the IBM Cloud APIs, but it can also be decomposed via the following
          structure:
          `crn:version:cname:ctype:service-name:location:scope:service-instance:resource-type:resource`.
    :attr str platform: (optional) Identifies the platform as "ibmcloud".
    """

    def __init__(self, *, account_id: str = None, crn: str = None, platform: str = None) -> None:
        """
        Initialize a Context object.

        :param str account_id: (optional) Returns the ID of the account in IBM
               Cloud that is provisioning the service instance.
        :param str crn: (optional) When a customer provisions your service in IBM
               Cloud, a service instance is created and this instance is identified by its
               IBM Cloud Resource Name (CRN). The CRN is utilized in all aspects of the
               interaction with IBM Cloud including provisioning, binding (creating
               credentials and endpoints), metering, dashboard display, and access
               control. From a service provider perspective, the CRN can largely be
               treated as an opaque string to be utilized with the IBM Cloud APIs, but it
               can also be decomposed via the following structure:
               `crn:version:cname:ctype:service-name:location:scope:service-instance:resource-type:resource`.
        :param str platform: (optional) Identifies the platform as "ibmcloud".
        """
        self.account_id = account_id
        self.crn = crn
        self.platform = platform

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Context':
        """Initialize a Context object from a json dictionary."""
        args = {}
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        if 'crn' in _dict:
            args['crn'] = _dict.get('crn')
        if 'platform' in _dict:
            args['platform'] = _dict.get('platform')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Context object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'account_id') and self.account_id is not None:
            _dict['account_id'] = self.account_id
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'platform') and self.platform is not None:
            _dict['platform'] = self.platform
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Context object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Context') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Context') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Plans:
    """
    Where is this in the source?.

    :attr str description: A short description of the plan. It MUST be a non-empty
          string. The description is NOT displayed in the IBM Cloud catalog or IBM Cloud
          CLI.
    :attr bool free: (optional) When false, service instances of this plan have a
          cost. The default is true.
    :attr str id: An identifier used to correlate this plan in future requests to
          the broker.  This MUST be globally unique within a platform marketplace. It MUST
          be a non-empty string and using a GUID is RECOMMENDED. If you define your
          service in the RMC, it will create a unique GUID for you to use. It is
          recommended to use the RMC to define and generate these values and then use them
          in your catalog.json metadata in your broker. This value is NOT displayed in the
          IBM Cloud catalog or IBM Cloud CLI.
    :attr str name: The programmatic name of the plan. It MUST be unique within the
          service. All lowercase, no spaces. It MUST be a non-empty string, and it's NOT
          displayed in the IBM Cloud catalog or IBM Cloud CLI.
    """

    def __init__(self, description: str, id: str, name: str, *, free: bool = None) -> None:
        """
        Initialize a Plans object.

        :param str description: A short description of the plan. It MUST be a
               non-empty string. The description is NOT displayed in the IBM Cloud catalog
               or IBM Cloud CLI.
        :param str id: An identifier used to correlate this plan in future requests
               to the broker.  This MUST be globally unique within a platform marketplace.
               It MUST be a non-empty string and using a GUID is RECOMMENDED. If you
               define your service in the RMC, it will create a unique GUID for you to
               use. It is recommended to use the RMC to define and generate these values
               and then use them in your catalog.json metadata in your broker. This value
               is NOT displayed in the IBM Cloud catalog or IBM Cloud CLI.
        :param str name: The programmatic name of the plan. It MUST be unique
               within the service. All lowercase, no spaces. It MUST be a non-empty
               string, and it's NOT displayed in the IBM Cloud catalog or IBM Cloud CLI.
        :param bool free: (optional) When false, service instances of this plan
               have a cost. The default is true.
        """
        self.description = description
        self.free = free
        self.id = id
        self.name = name

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Plans':
        """Initialize a Plans object from a json dictionary."""
        args = {}
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        else:
            raise ValueError('Required property \'description\' not present in Plans JSON')
        if 'free' in _dict:
            args['free'] = _dict.get('free')
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        else:
            raise ValueError('Required property \'id\' not present in Plans JSON')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        else:
            raise ValueError('Required property \'name\' not present in Plans JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Plans object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'free') and self.free is not None:
            _dict['free'] = self.free
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Plans object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Plans') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Plans') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Services:
    """
    The service object that describes the properties of your service.

    :attr bool bindable: Specifies whether or not your service can be bound to
          applications in IBM Cloud. If bindable, it must be able to return API endpoints
          and credentials to your service consumers.
    :attr str description: A short description of the service. It MUST be a
          non-empty string. Note that this description is not displayed by the the IBM
          Cloud console or IBM Cloud CLI.
    :attr str id: An identifier used to correlate this service in future requests to
          the broker. This MUST be globally unique within the IBM Cloud platform. It MUST
          be a non-empty string, and using a GUID is recommended. Recommended: If you
          define your service in the RMC, the RMC will generate a globally unique GUID
          service ID that you can use in your service broker.
    :attr str name: The service name is not your display name. Your service name
          must follow the follow these rules:
           - It must be all lowercase.
           - It can't include spaces but may include hyphens (`-`).
           - It must be less than 32 characters.
           Your service name should include your company name. If your company has more
          then one offering your service name should include both company and offering as
          part of the name. For example, the Compose company has offerings for Redis and
          Elasticsearch. Sample service names on IBM Cloud for these offerings would be
          `compose-redis` and `compose-elasticsearch`.  Each of these service names have
          associated display names that are shown in the IBM Cloud catalog: *Compose
          Redis* and *Compose Elasticsearch*. Another company (e.g. FastJetMail) may only
          have the single JetMail offering, in which case the service name should be
          `fastjetmail`. Recommended: If you define your service in RMC, you can export a
          catalog.json that will include the service name you defined within the RMC.
    :attr bool plan_updateable: (optional) The Default is false. This specifices
          whether or not you support plan changes for provisioned instances. If your
          offering supports multiple plans, and you want users to be able to change plans
          for a provisioned instance, you will need to enable the ability for users to
          update their service instance by using /v2/service_instances/{instance_id}
          PATCH.
    :attr List[Plans] plans: A list of plans for this service that must contain at
          least one plan.
    """

    def __init__(
        self,
        bindable: bool,
        description: str,
        id: str,
        name: str,
        plans: List['Plans'],
        *,
        plan_updateable: bool = None
    ) -> None:
        """
        Initialize a Services object.

        :param bool bindable: Specifies whether or not your service can be bound to
               applications in IBM Cloud. If bindable, it must be able to return API
               endpoints and credentials to your service consumers.
        :param str description: A short description of the service. It MUST be a
               non-empty string. Note that this description is not displayed by the the
               IBM Cloud console or IBM Cloud CLI.
        :param str id: An identifier used to correlate this service in future
               requests to the broker. This MUST be globally unique within the IBM Cloud
               platform. It MUST be a non-empty string, and using a GUID is recommended.
               Recommended: If you define your service in the RMC, the RMC will generate a
               globally unique GUID service ID that you can use in your service broker.
        :param str name: The service name is not your display name. Your service
               name must follow the follow these rules:
                - It must be all lowercase.
                - It can't include spaces but may include hyphens (`-`).
                - It must be less than 32 characters.
                Your service name should include your company name. If your company has
               more then one offering your service name should include both company and
               offering as part of the name. For example, the Compose company has
               offerings for Redis and Elasticsearch. Sample service names on IBM Cloud
               for these offerings would be `compose-redis` and `compose-elasticsearch`.
               Each of these service names have associated display names that are shown in
               the IBM Cloud catalog: *Compose Redis* and *Compose Elasticsearch*. Another
               company (e.g. FastJetMail) may only have the single JetMail offering, in
               which case the service name should be `fastjetmail`. Recommended: If you
               define your service in RMC, you can export a catalog.json that will include
               the service name you defined within the RMC.
        :param List[Plans] plans: A list of plans for this service that must
               contain at least one plan.
        :param bool plan_updateable: (optional) The Default is false. This
               specifices whether or not you support plan changes for provisioned
               instances. If your offering supports multiple plans, and you want users to
               be able to change plans for a provisioned instance, you will need to enable
               the ability for users to update their service instance by using
               /v2/service_instances/{instance_id} PATCH.
        """
        self.bindable = bindable
        self.description = description
        self.id = id
        self.name = name
        self.plan_updateable = plan_updateable
        self.plans = plans

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Services':
        """Initialize a Services object from a json dictionary."""
        args = {}
        if 'bindable' in _dict:
            args['bindable'] = _dict.get('bindable')
        else:
            raise ValueError('Required property \'bindable\' not present in Services JSON')
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        else:
            raise ValueError('Required property \'description\' not present in Services JSON')
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        else:
            raise ValueError('Required property \'id\' not present in Services JSON')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        else:
            raise ValueError('Required property \'name\' not present in Services JSON')
        if 'plan_updateable' in _dict:
            args['plan_updateable'] = _dict.get('plan_updateable')
        if 'plans' in _dict:
            args['plans'] = [Plans.from_dict(x) for x in _dict.get('plans')]
        else:
            raise ValueError('Required property \'plans\' not present in Services JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Services object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'bindable') and self.bindable is not None:
            _dict['bindable'] = self.bindable
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'plan_updateable') and self.plan_updateable is not None:
            _dict['plan_updateable'] = self.plan_updateable
        if hasattr(self, 'plans') and self.plans is not None:
            _dict['plans'] = [x.to_dict() for x in self.plans]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Services object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Services') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Services') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class VolumeMount:
    """
    VolumeMount.

    :attr str driver: A free-form hash of credentials that can be used by
          applications or users to access the service.
    :attr str container_dir: The path in the application container onto which the
          volume will be mounted. This specification does not mandate what action the
          platform is to take if the path specified already exists in the container.
    :attr str mode: 'r' to mount the volume read-only or 'rw' to mount it
          read-write.
    :attr str device_type: A string specifying the type of device to mount.
          Currently the only supported value is 'shared'.
    :attr str device: Device object containing device_type specific details.
          Currently only shared devices are supported.
    """

    def __init__(self, driver: str, container_dir: str, mode: str, device_type: str, device: str) -> None:
        """
        Initialize a VolumeMount object.

        :param str driver: A free-form hash of credentials that can be used by
               applications or users to access the service.
        :param str container_dir: The path in the application container onto which
               the volume will be mounted. This specification does not mandate what action
               the platform is to take if the path specified already exists in the
               container.
        :param str mode: 'r' to mount the volume read-only or 'rw' to mount it
               read-write.
        :param str device_type: A string specifying the type of device to mount.
               Currently the only supported value is 'shared'.
        :param str device: Device object containing device_type specific details.
               Currently only shared devices are supported.
        """
        self.driver = driver
        self.container_dir = container_dir
        self.mode = mode
        self.device_type = device_type
        self.device = device

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'VolumeMount':
        """Initialize a VolumeMount object from a json dictionary."""
        args = {}
        if 'driver' in _dict:
            args['driver'] = _dict.get('driver')
        else:
            raise ValueError('Required property \'driver\' not present in VolumeMount JSON')
        if 'container_dir' in _dict:
            args['container_dir'] = _dict.get('container_dir')
        else:
            raise ValueError('Required property \'container_dir\' not present in VolumeMount JSON')
        if 'mode' in _dict:
            args['mode'] = _dict.get('mode')
        else:
            raise ValueError('Required property \'mode\' not present in VolumeMount JSON')
        if 'device_type' in _dict:
            args['device_type'] = _dict.get('device_type')
        else:
            raise ValueError('Required property \'device_type\' not present in VolumeMount JSON')
        if 'device' in _dict:
            args['device'] = _dict.get('device')
        else:
            raise ValueError('Required property \'device\' not present in VolumeMount JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a VolumeMount object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'driver') and self.driver is not None:
            _dict['driver'] = self.driver
        if hasattr(self, 'container_dir') and self.container_dir is not None:
            _dict['container_dir'] = self.container_dir
        if hasattr(self, 'mode') and self.mode is not None:
            _dict['mode'] = self.mode
        if hasattr(self, 'device_type') and self.device_type is not None:
            _dict['device_type'] = self.device_type
        if hasattr(self, 'device') and self.device is not None:
            _dict['device'] = self.device
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this VolumeMount object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'VolumeMount') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'VolumeMount') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other
