# coding: utf-8

# (C) Copyright IBM Corp. 2021, 2022.
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

# IBM OpenAPI SDK Code Generator Version: 3.60.0-13f6e1ba-20221019-164457

"""
Case management API for creating cases, getting case statuses, adding comments to a case,
adding and removing users from a case watchlist, downloading and adding attachments, and
more.

API Version: 1.0.0
"""

from enum import Enum
from typing import BinaryIO, Dict, List
import json
import sys

from ibm_cloud_sdk_core import BaseService, DetailedResponse, get_query_param
from ibm_cloud_sdk_core.authenticators.authenticator import Authenticator
from ibm_cloud_sdk_core.get_authenticator import get_authenticator_from_environment
from ibm_cloud_sdk_core.utils import convert_list, convert_model

from .common import get_sdk_headers

##############################################################################
# Service
##############################################################################


class CaseManagementV1(BaseService):
    """The Case Management V1 service."""

    DEFAULT_SERVICE_URL = 'https://support-center.cloud.ibm.com/case-management/v1'
    DEFAULT_SERVICE_NAME = 'case_management'

    @classmethod
    def new_instance(
        cls,
        service_name: str = DEFAULT_SERVICE_NAME,
    ) -> 'CaseManagementV1':
        """
        Return a new client for the Case Management service using the specified
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
        Construct a new client for the Case Management service.

        :param Authenticator authenticator: The authenticator specifies the authentication mechanism.
               Get up to date information from https://github.com/IBM/python-sdk-core/blob/main/README.md
               about initializing the authenticator of your choice.
        """
        BaseService.__init__(self, service_url=self.DEFAULT_SERVICE_URL, authenticator=authenticator)

    #########################
    # default
    #########################

    def get_cases(
        self,
        *,
        offset: int = None,
        limit: int = None,
        search: str = None,
        sort: str = None,
        status: List[str] = None,
        fields: List[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Get cases in account.

        Get cases in the account that are specified by the content of the IAM token.

        :param int offset: (optional) Number of cases that are skipped.
        :param int limit: (optional) Number of cases that are returned.
        :param str search: (optional) String that a case might contain.
        :param str sort: (optional) Sort field and direction. If omitted, default
               to descending of updated date. Prefix "~" signifies sort in descending.
        :param List[str] status: (optional) Case status filter.
        :param List[str] fields: (optional) Selected fields of interest instead of
               all of the case information.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `CaseList` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_cases'
        )
        headers.update(sdk_headers)

        params = {
            'offset': offset,
            'limit': limit,
            'search': search,
            'sort': sort,
            'status': convert_list(status),
            'fields': convert_list(fields),
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/cases'
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def create_case(
        self,
        type: str,
        subject: str,
        description: str,
        *,
        severity: int = None,
        eu: 'CasePayloadEu' = None,
        offering: 'Offering' = None,
        resources: List['ResourcePayload'] = None,
        watchlist: List['User'] = None,
        invoice_number: str = None,
        sla_credit_request: bool = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Create a case.

        Create a support case to resolve issues in your account.

        :param str type: Case type.
        :param str subject: Short description used to identify the case.
        :param str description: Detailed description of the issue.
        :param int severity: (optional) Severity of the case. Smaller values mean
               higher severity.
        :param CasePayloadEu eu: (optional) Specify if the case should be treated
               as EU regulated. Only one of the following properties is required. Call EU
               support utility endpoint to determine which property must be specified for
               your account.
        :param Offering offering: (optional) Offering details.
        :param List[ResourcePayload] resources: (optional) List of resources to
               attach to case. If you attach Classic IaaS devices, use the type and id
               fields if the Cloud Resource Name (CRN) is unavailable. Otherwise, pass the
               resource CRN. The resource list must be consistent with the value that is
               selected for the resource offering.
        :param List[User] watchlist: (optional) Array of user IDs to add to the
               watchlist.
        :param str invoice_number: (optional) Invoice number of "Billing and
               Invoice" case type.
        :param bool sla_credit_request: (optional) Flag to indicate if case is for
               an Service Level Agreement (SLA) credit request.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Case` object
        """

        if type is None:
            raise ValueError('type must be provided')
        if subject is None:
            raise ValueError('subject must be provided')
        if description is None:
            raise ValueError('description must be provided')
        if eu is not None:
            eu = convert_model(eu)
        if offering is not None:
            offering = convert_model(offering)
        if resources is not None:
            resources = [convert_model(x) for x in resources]
        if watchlist is not None:
            watchlist = [convert_model(x) for x in watchlist]
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='create_case'
        )
        headers.update(sdk_headers)

        data = {
            'type': type,
            'subject': subject,
            'description': description,
            'severity': severity,
            'eu': eu,
            'offering': offering,
            'resources': resources,
            'watchlist': watchlist,
            'invoice_number': invoice_number,
            'sla_credit_request': sla_credit_request,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/cases'
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def get_case(self, case_number: str, *, fields: List[str] = None, **kwargs) -> DetailedResponse:
        """
        Get a case in account.

        View a case in the account that is specified by the case number.

        :param str case_number: Unique identifier of a case.
        :param List[str] fields: (optional) Selected fields of interest instead of
               all of the case information.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Case` object
        """

        if not case_number:
            raise ValueError('case_number must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_case'
        )
        headers.update(sdk_headers)

        params = {'fields': convert_list(fields)}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['case_number']
        path_param_values = self.encode_path_vars(case_number)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/cases/{case_number}'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def update_case_status(self, case_number: str, status_payload: 'StatusPayload', **kwargs) -> DetailedResponse:
        """
        Update case status.

        Mark the case as resolved or unresolved, or accept the provided resolution.

        :param str case_number: Unique identifier of a case.
        :param StatusPayload status_payload:
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Case` object
        """

        if not case_number:
            raise ValueError('case_number must be provided')
        if status_payload is None:
            raise ValueError('status_payload must be provided')
        if isinstance(status_payload, StatusPayload):
            status_payload = convert_model(status_payload)
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='update_case_status'
        )
        headers.update(sdk_headers)

        data = json.dumps(status_payload)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['case_number']
        path_param_values = self.encode_path_vars(case_number)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/cases/{case_number}/status'.format(**path_param_dict)
        request = self.prepare_request(method='PUT', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def add_comment(self, case_number: str, comment: str, **kwargs) -> DetailedResponse:
        """
        Add comment to case.

        Add a comment to a case to be viewed by a support engineer.

        :param str case_number: Unique identifier of a case.
        :param str comment: Comment to add to the case.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Comment` object
        """

        if not case_number:
            raise ValueError('case_number must be provided')
        if comment is None:
            raise ValueError('comment must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='add_comment'
        )
        headers.update(sdk_headers)

        data = {'comment': comment}
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['case_number']
        path_param_values = self.encode_path_vars(case_number)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/cases/{case_number}/comments'.format(**path_param_dict)
        request = self.prepare_request(method='PUT', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def add_watchlist(self, case_number: str, *, watchlist: List['User'] = None, **kwargs) -> DetailedResponse:
        """
        Add users to watchlist of case.

        Add users to the watchlist of case. By adding a user to the watchlist of the case,
        you are granting them read and write permissions, so the user can view the case,
        receive updates, and make updates to the case. Note that the user must be in the
        account to be added to the watchlist.

        :param str case_number: Unique identifier of a case.
        :param List[User] watchlist: (optional) Array of user ID objects.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `WatchlistAddResponse` object
        """

        if not case_number:
            raise ValueError('case_number must be provided')
        if watchlist is not None:
            watchlist = [convert_model(x) for x in watchlist]
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='add_watchlist'
        )
        headers.update(sdk_headers)

        data = {'watchlist': watchlist}
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['case_number']
        path_param_values = self.encode_path_vars(case_number)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/cases/{case_number}/watchlist'.format(**path_param_dict)
        request = self.prepare_request(method='PUT', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def remove_watchlist(self, case_number: str, *, watchlist: List['User'] = None, **kwargs) -> DetailedResponse:
        """
        Remove users from watchlist of case.

        Remove users from the watchlist of a case if you don't want them to view the case,
        receive updates, or make updates to the case.

        :param str case_number: Unique identifier of a case.
        :param List[User] watchlist: (optional) Array of user ID objects.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Watchlist` object
        """

        if not case_number:
            raise ValueError('case_number must be provided')
        if watchlist is not None:
            watchlist = [convert_model(x) for x in watchlist]
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='remove_watchlist'
        )
        headers.update(sdk_headers)

        data = {'watchlist': watchlist}
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['case_number']
        path_param_values = self.encode_path_vars(case_number)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/cases/{case_number}/watchlist'.format(**path_param_dict)
        request = self.prepare_request(method='DELETE', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def add_resource(
        self, case_number: str, *, crn: str = None, type: str = None, id: float = None, note: str = None, **kwargs
    ) -> DetailedResponse:
        """
        Add a resource to case.

        Add a resource to case by specifying the Cloud Resource Name (CRN), or id and type
        if attaching a class iaaS resource.

        :param str case_number: Unique identifier of a case.
        :param str crn: (optional) Cloud Resource Name of the resource.
        :param str type: (optional) Only used to attach Classic IaaS devices that
               have no CRN.
        :param float id: (optional) Deprecated: Only used to attach Classic IaaS
               devices that have no CRN. Id of Classic IaaS device. This is deprecated in
               favor of the crn field.
        :param str note: (optional) A note about this resource.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Resource` object
        """

        if not case_number:
            raise ValueError('case_number must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='add_resource'
        )
        headers.update(sdk_headers)

        data = {'crn': crn, 'type': type, 'id': id, 'note': note}
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['case_number']
        path_param_values = self.encode_path_vars(case_number)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/cases/{case_number}/resources'.format(**path_param_dict)
        request = self.prepare_request(method='PUT', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def upload_file(self, case_number: str, file: List[BinaryIO], **kwargs) -> DetailedResponse:
        """
        Add attachments to a support case.

        You can add attachments to a case to provide more information for the support team
        about the issue that you're experiencing.

        :param str case_number: Unique identifier of a case.
        :param list[FileWithMetadata] file: file of supported types, 8MB in size
               limit.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Attachment` object
        """

        if not case_number:
            raise ValueError('case_number must be provided')
        if file is None:
            raise ValueError('file must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='upload_file'
        )
        headers.update(sdk_headers)

        form_data = []
        for item in file:
            item = convert_model(item)
            _file = (item.get('filename') or None, item['data'], item.get('content_type') or 'application/octet-stream')
            form_data.append(('file', _file))

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['case_number']
        path_param_values = self.encode_path_vars(case_number)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/cases/{case_number}/attachments'.format(**path_param_dict)
        request = self.prepare_request(method='PUT', url=url, headers=headers, files=form_data)

        response = self.send(request, **kwargs)
        return response

    def download_file(self, case_number: str, file_id: str, **kwargs) -> DetailedResponse:
        """
        Download an attachment.

        Download an attachment from a case.

        :param str case_number: Unique identifier of a case.
        :param str file_id: Unique identifier of a file.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `BinaryIO` result
        """

        if not case_number:
            raise ValueError('case_number must be provided')
        if not file_id:
            raise ValueError('file_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='download_file'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/octet-stream'

        path_param_keys = ['case_number', 'file_id']
        path_param_values = self.encode_path_vars(case_number, file_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/cases/{case_number}/attachments/{file_id}'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def delete_file(self, case_number: str, file_id: str, **kwargs) -> DetailedResponse:
        """
        Remove attachment from case.

        Remove an attachment from a case.

        :param str case_number: Unique identifier of a case.
        :param str file_id: Unique identifier of a file.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AttachmentList` object
        """

        if not case_number:
            raise ValueError('case_number must be provided')
        if not file_id:
            raise ValueError('file_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='delete_file'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        path_param_keys = ['case_number', 'file_id']
        path_param_values = self.encode_path_vars(case_number, file_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/cases/{case_number}/attachments/{file_id}'.format(**path_param_dict)
        request = self.prepare_request(method='DELETE', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response


class GetCasesEnums:
    """
    Enums for get_cases parameters.
    """

    class Status(str, Enum):
        """
        Case status filter.
        """

        NEW = 'new'
        IN_PROGRESS = 'in_progress'
        WAITING_ON_CLIENT = 'waiting_on_client'
        RESOLUTION_PROVIDED = 'resolution_provided'
        RESOLVED = 'resolved'
        CLOSED = 'closed'

    class Fields(str, Enum):
        """
        Selected fields of interest instead of all of the case information.
        """

        NUMBER = 'number'
        SHORT_DESCRIPTION = 'short_description'
        DESCRIPTION = 'description'
        CREATED_AT = 'created_at'
        CREATED_BY = 'created_by'
        UPDATED_AT = 'updated_at'
        UPDATED_BY = 'updated_by'
        CONTACT = 'contact'
        CONTACT_TYPE = 'contact_type'
        STATUS = 'status'
        SEVERITY = 'severity'
        SUPPORT_TIER = 'support_tier'
        RESOLUTION = 'resolution'
        CLOSE_NOTES = 'close_notes'
        INVOICE_NUMBER = 'invoice_number'
        AGENT_CLOSE_ONLY = 'agent_close_only'
        EU = 'eu'
        WATCHLIST = 'watchlist'
        ATTACHMENTS = 'attachments'
        RESOURCES = 'resources'
        COMMENTS = 'comments'
        OFFERING = 'offering'


class GetCaseEnums:
    """
    Enums for get_case parameters.
    """

    class Fields(str, Enum):
        """
        Selected fields of interest instead of all of the case information.
        """

        NUMBER = 'number'
        SHORT_DESCRIPTION = 'short_description'
        DESCRIPTION = 'description'
        CREATED_AT = 'created_at'
        CREATED_BY = 'created_by'
        UPDATED_AT = 'updated_at'
        UPDATED_BY = 'updated_by'
        CONTACT = 'contact'
        CONTACT_TYPE = 'contact_type'
        STATUS = 'status'
        SEVERITY = 'severity'
        SUPPORT_TIER = 'support_tier'
        RESOLUTION = 'resolution'
        CLOSE_NOTES = 'close_notes'
        INVOICE_NUMBER = 'invoice_number'
        AGENT_CLOSE_ONLY = 'agent_close_only'
        EU = 'eu'
        WATCHLIST = 'watchlist'
        ATTACHMENTS = 'attachments'
        RESOURCES = 'resources'
        COMMENTS = 'comments'
        OFFERING = 'offering'


##############################################################################
# Models
##############################################################################


class Attachment:
    """
    Details of an attachment.

    :attr str id: (optional) Unique identifier of the attachment in database.
    :attr str filename: (optional) Name of the attachment.
    :attr int size_in_bytes: (optional) Size of the attachment in bytes.
    :attr str created_at: (optional) Date time of uploading in UTC.
    :attr str url: (optional) URL of the attachment used to download.
    """

    def __init__(
        self,
        *,
        id: str = None,
        filename: str = None,
        size_in_bytes: int = None,
        created_at: str = None,
        url: str = None,
    ) -> None:
        """
        Initialize a Attachment object.

        :param str id: (optional) Unique identifier of the attachment in database.
        :param str filename: (optional) Name of the attachment.
        :param int size_in_bytes: (optional) Size of the attachment in bytes.
        :param str created_at: (optional) Date time of uploading in UTC.
        :param str url: (optional) URL of the attachment used to download.
        """
        self.id = id
        self.filename = filename
        self.size_in_bytes = size_in_bytes
        self.created_at = created_at
        self.url = url

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Attachment':
        """Initialize a Attachment object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'filename' in _dict:
            args['filename'] = _dict.get('filename')
        if 'size_in_bytes' in _dict:
            args['size_in_bytes'] = _dict.get('size_in_bytes')
        if 'created_at' in _dict:
            args['created_at'] = _dict.get('created_at')
        if 'url' in _dict:
            args['url'] = _dict.get('url')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Attachment object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'filename') and self.filename is not None:
            _dict['filename'] = self.filename
        if hasattr(self, 'size_in_bytes') and self.size_in_bytes is not None:
            _dict['size_in_bytes'] = self.size_in_bytes
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = self.created_at
        if hasattr(self, 'url') and self.url is not None:
            _dict['url'] = self.url
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Attachment object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Attachment') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Attachment') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AttachmentList:
    """
    List of attachments in the case.

    :attr List[Attachment] attachments: (optional) New attachments array.
    """

    def __init__(self, *, attachments: List['Attachment'] = None) -> None:
        """
        Initialize a AttachmentList object.

        :param List[Attachment] attachments: (optional) New attachments array.
        """
        self.attachments = attachments

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AttachmentList':
        """Initialize a AttachmentList object from a json dictionary."""
        args = {}
        if 'attachments' in _dict:
            args['attachments'] = [Attachment.from_dict(x) for x in _dict.get('attachments')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AttachmentList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'attachments') and self.attachments is not None:
            _dict['attachments'] = [x.to_dict() for x in self.attachments]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AttachmentList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AttachmentList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AttachmentList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Case:
    """
    The support case.

    :attr str number: (optional) Identifying number of a created case.
    :attr str short_description: (optional) Short description of what the case is
          about.
    :attr str description: (optional) Full description of what the case is about.
    :attr str created_at: (optional) Date and time of case creation in UTC.
    :attr User created_by: (optional) User info in a case.
    :attr str updated_at: (optional) Date and time of the last update on the case in
          UTC.
    :attr User updated_by: (optional) User info in a case.
    :attr str contact_type: (optional) Name of the console to interact with the
          contact.
    :attr User contact: (optional) User info in a case.
    :attr str status: (optional) Status type of the case.
    :attr float severity: (optional) Severity level of the case.
    :attr str support_tier: (optional) Support tier of the account.
    :attr str resolution: (optional) Standard reasons of resolving case.
    :attr str close_notes: (optional) Notes of case closing.
    :attr CaseEu eu: (optional) EU support.
    :attr List[User] watchlist: (optional) List of users in the case watchlist.
    :attr List[Attachment] attachments: (optional) List of files that are attached
          to the case.
    :attr Offering offering: (optional) Offering details.
    :attr List[Resource] resources: (optional) List of attached resources.
    :attr List[Comment] comments: (optional) List of comments and updates that are
          sorted in chronological order.
    """

    def __init__(
        self,
        *,
        number: str = None,
        short_description: str = None,
        description: str = None,
        created_at: str = None,
        created_by: 'User' = None,
        updated_at: str = None,
        updated_by: 'User' = None,
        contact_type: str = None,
        contact: 'User' = None,
        status: str = None,
        severity: float = None,
        support_tier: str = None,
        resolution: str = None,
        close_notes: str = None,
        eu: 'CaseEu' = None,
        watchlist: List['User'] = None,
        attachments: List['Attachment'] = None,
        offering: 'Offering' = None,
        resources: List['Resource'] = None,
        comments: List['Comment'] = None,
    ) -> None:
        """
        Initialize a Case object.

        :param str number: (optional) Identifying number of a created case.
        :param str short_description: (optional) Short description of what the case
               is about.
        :param str description: (optional) Full description of what the case is
               about.
        :param str created_at: (optional) Date and time of case creation in UTC.
        :param User created_by: (optional) User info in a case.
        :param str updated_at: (optional) Date and time of the last update on the
               case in UTC.
        :param User updated_by: (optional) User info in a case.
        :param str contact_type: (optional) Name of the console to interact with
               the contact.
        :param User contact: (optional) User info in a case.
        :param str status: (optional) Status type of the case.
        :param float severity: (optional) Severity level of the case.
        :param str support_tier: (optional) Support tier of the account.
        :param str resolution: (optional) Standard reasons of resolving case.
        :param str close_notes: (optional) Notes of case closing.
        :param CaseEu eu: (optional) EU support.
        :param List[User] watchlist: (optional) List of users in the case
               watchlist.
        :param List[Attachment] attachments: (optional) List of files that are
               attached to the case.
        :param Offering offering: (optional) Offering details.
        :param List[Resource] resources: (optional) List of attached resources.
        :param List[Comment] comments: (optional) List of comments and updates that
               are sorted in chronological order.
        """
        self.number = number
        self.short_description = short_description
        self.description = description
        self.created_at = created_at
        self.created_by = created_by
        self.updated_at = updated_at
        self.updated_by = updated_by
        self.contact_type = contact_type
        self.contact = contact
        self.status = status
        self.severity = severity
        self.support_tier = support_tier
        self.resolution = resolution
        self.close_notes = close_notes
        self.eu = eu
        self.watchlist = watchlist
        self.attachments = attachments
        self.offering = offering
        self.resources = resources
        self.comments = comments

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Case':
        """Initialize a Case object from a json dictionary."""
        args = {}
        if 'number' in _dict:
            args['number'] = _dict.get('number')
        if 'short_description' in _dict:
            args['short_description'] = _dict.get('short_description')
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        if 'created_at' in _dict:
            args['created_at'] = _dict.get('created_at')
        if 'created_by' in _dict:
            args['created_by'] = User.from_dict(_dict.get('created_by'))
        if 'updated_at' in _dict:
            args['updated_at'] = _dict.get('updated_at')
        if 'updated_by' in _dict:
            args['updated_by'] = User.from_dict(_dict.get('updated_by'))
        if 'contact_type' in _dict:
            args['contact_type'] = _dict.get('contact_type')
        if 'contact' in _dict:
            args['contact'] = User.from_dict(_dict.get('contact'))
        if 'status' in _dict:
            args['status'] = _dict.get('status')
        if 'severity' in _dict:
            args['severity'] = _dict.get('severity')
        if 'support_tier' in _dict:
            args['support_tier'] = _dict.get('support_tier')
        if 'resolution' in _dict:
            args['resolution'] = _dict.get('resolution')
        if 'close_notes' in _dict:
            args['close_notes'] = _dict.get('close_notes')
        if 'eu' in _dict:
            args['eu'] = CaseEu.from_dict(_dict.get('eu'))
        if 'watchlist' in _dict:
            args['watchlist'] = [User.from_dict(x) for x in _dict.get('watchlist')]
        if 'attachments' in _dict:
            args['attachments'] = [Attachment.from_dict(x) for x in _dict.get('attachments')]
        if 'offering' in _dict:
            args['offering'] = Offering.from_dict(_dict.get('offering'))
        if 'resources' in _dict:
            args['resources'] = [Resource.from_dict(x) for x in _dict.get('resources')]
        if 'comments' in _dict:
            args['comments'] = [Comment.from_dict(x) for x in _dict.get('comments')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Case object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'number') and self.number is not None:
            _dict['number'] = self.number
        if hasattr(self, 'short_description') and self.short_description is not None:
            _dict['short_description'] = self.short_description
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'created_at') and self.created_at is not None:
            _dict['created_at'] = self.created_at
        if hasattr(self, 'created_by') and self.created_by is not None:
            _dict['created_by'] = self.created_by.to_dict()
        if hasattr(self, 'updated_at') and self.updated_at is not None:
            _dict['updated_at'] = self.updated_at
        if hasattr(self, 'updated_by') and self.updated_by is not None:
            _dict['updated_by'] = self.updated_by.to_dict()
        if hasattr(self, 'contact_type') and self.contact_type is not None:
            _dict['contact_type'] = self.contact_type
        if hasattr(self, 'contact') and self.contact is not None:
            _dict['contact'] = self.contact.to_dict()
        if hasattr(self, 'status') and self.status is not None:
            _dict['status'] = self.status
        if hasattr(self, 'severity') and self.severity is not None:
            _dict['severity'] = self.severity
        if hasattr(self, 'support_tier') and self.support_tier is not None:
            _dict['support_tier'] = self.support_tier
        if hasattr(self, 'resolution') and self.resolution is not None:
            _dict['resolution'] = self.resolution
        if hasattr(self, 'close_notes') and self.close_notes is not None:
            _dict['close_notes'] = self.close_notes
        if hasattr(self, 'eu') and self.eu is not None:
            _dict['eu'] = self.eu.to_dict()
        if hasattr(self, 'watchlist') and self.watchlist is not None:
            _dict['watchlist'] = [x.to_dict() for x in self.watchlist]
        if hasattr(self, 'attachments') and self.attachments is not None:
            _dict['attachments'] = [x.to_dict() for x in self.attachments]
        if hasattr(self, 'offering') and self.offering is not None:
            _dict['offering'] = self.offering.to_dict()
        if hasattr(self, 'resources') and self.resources is not None:
            _dict['resources'] = [x.to_dict() for x in self.resources]
        if hasattr(self, 'comments') and self.comments is not None:
            _dict['comments'] = [x.to_dict() for x in self.comments]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Case object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Case') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Case') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class ContactTypeEnum(str, Enum):
        """
        Name of the console to interact with the contact.
        """

        CLOUD_SUPPORT_CENTER = 'Cloud Support Center'
        IMS_CONSOLE = 'IMS Console'

    class SupportTierEnum(str, Enum):
        """
        Support tier of the account.
        """

        FREE = 'Free'
        BASIC = 'Basic'
        STANDARD = 'Standard'
        PREMIUM = 'Premium'


class CaseEu:
    """
    EU support.

    :attr bool support: (optional) Identifying whether the case has EU Support.
    :attr str data_center: (optional) Information about the data center.
    """

    def __init__(self, *, support: bool = None, data_center: str = None) -> None:
        """
        Initialize a CaseEu object.

        :param bool support: (optional) Identifying whether the case has EU
               Support.
        :param str data_center: (optional) Information about the data center.
        """
        self.support = support
        self.data_center = data_center

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CaseEu':
        """Initialize a CaseEu object from a json dictionary."""
        args = {}
        if 'support' in _dict:
            args['support'] = _dict.get('support')
        if 'data_center' in _dict:
            args['data_center'] = _dict.get('data_center')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CaseEu object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'support') and self.support is not None:
            _dict['support'] = self.support
        if hasattr(self, 'data_center') and self.data_center is not None:
            _dict['data_center'] = self.data_center
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this CaseEu object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CaseEu') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CaseEu') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class CaseList:
    """
    Response of a GET /cases request.

    :attr int total_count: (optional) Total number of cases that satisfy the query.
    :attr PaginationLink first: (optional) Container for URL pointer to related
          pages of cases.
    :attr PaginationLink next: (optional) Container for URL pointer to related pages
          of cases.
    :attr PaginationLink previous: (optional) Container for URL pointer to related
          pages of cases.
    :attr PaginationLink last: (optional) Container for URL pointer to related pages
          of cases.
    :attr List[Case] cases: (optional) List of cases.
    """

    def __init__(
        self,
        *,
        total_count: int = None,
        first: 'PaginationLink' = None,
        next: 'PaginationLink' = None,
        previous: 'PaginationLink' = None,
        last: 'PaginationLink' = None,
        cases: List['Case'] = None,
    ) -> None:
        """
        Initialize a CaseList object.

        :param int total_count: (optional) Total number of cases that satisfy the
               query.
        :param PaginationLink first: (optional) Container for URL pointer to
               related pages of cases.
        :param PaginationLink next: (optional) Container for URL pointer to related
               pages of cases.
        :param PaginationLink previous: (optional) Container for URL pointer to
               related pages of cases.
        :param PaginationLink last: (optional) Container for URL pointer to related
               pages of cases.
        :param List[Case] cases: (optional) List of cases.
        """
        self.total_count = total_count
        self.first = first
        self.next = next
        self.previous = previous
        self.last = last
        self.cases = cases

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CaseList':
        """Initialize a CaseList object from a json dictionary."""
        args = {}
        if 'total_count' in _dict:
            args['total_count'] = _dict.get('total_count')
        if 'first' in _dict:
            args['first'] = PaginationLink.from_dict(_dict.get('first'))
        if 'next' in _dict:
            args['next'] = PaginationLink.from_dict(_dict.get('next'))
        if 'previous' in _dict:
            args['previous'] = PaginationLink.from_dict(_dict.get('previous'))
        if 'last' in _dict:
            args['last'] = PaginationLink.from_dict(_dict.get('last'))
        if 'cases' in _dict:
            args['cases'] = [Case.from_dict(x) for x in _dict.get('cases')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CaseList object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'total_count') and self.total_count is not None:
            _dict['total_count'] = self.total_count
        if hasattr(self, 'first') and self.first is not None:
            _dict['first'] = self.first.to_dict()
        if hasattr(self, 'next') and self.next is not None:
            _dict['next'] = self.next.to_dict()
        if hasattr(self, 'previous') and self.previous is not None:
            _dict['previous'] = self.previous.to_dict()
        if hasattr(self, 'last') and self.last is not None:
            _dict['last'] = self.last.to_dict()
        if hasattr(self, 'cases') and self.cases is not None:
            _dict['cases'] = [x.to_dict() for x in self.cases]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this CaseList object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CaseList') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CaseList') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class CasePayloadEu:
    """
    Specify if the case should be treated as EU regulated. Only one of the following
    properties is required. Call EU support utility endpoint to determine which property
    must be specified for your account.

    :attr bool supported: (optional) indicating whether the case is EU supported.
    :attr int data_center: (optional) If EU supported utility endpoint specifies
          data center, then pass the data center id to mark a case as EU supported.
    """

    def __init__(self, *, supported: bool = None, data_center: int = None) -> None:
        """
        Initialize a CasePayloadEu object.

        :param bool supported: (optional) indicating whether the case is EU
               supported.
        :param int data_center: (optional) If EU supported utility endpoint
               specifies data center, then pass the data center id to mark a case as EU
               supported.
        """
        self.supported = supported
        self.data_center = data_center

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CasePayloadEu':
        """Initialize a CasePayloadEu object from a json dictionary."""
        args = {}
        if 'supported' in _dict:
            args['supported'] = _dict.get('supported')
        if 'data_center' in _dict:
            args['data_center'] = _dict.get('data_center')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CasePayloadEu object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'supported') and self.supported is not None:
            _dict['supported'] = self.supported
        if hasattr(self, 'data_center') and self.data_center is not None:
            _dict['data_center'] = self.data_center
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this CasePayloadEu object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CasePayloadEu') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CasePayloadEu') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Comment:
    """
    A comment in a case.

    :attr str value: (optional) The comment.
    :attr str added_at: (optional) Date time when comment was added in UTC.
    :attr User added_by: (optional) User info in a case.
    """

    def __init__(self, *, value: str = None, added_at: str = None, added_by: 'User' = None) -> None:
        """
        Initialize a Comment object.

        :param str value: (optional) The comment.
        :param str added_at: (optional) Date time when comment was added in UTC.
        :param User added_by: (optional) User info in a case.
        """
        self.value = value
        self.added_at = added_at
        self.added_by = added_by

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Comment':
        """Initialize a Comment object from a json dictionary."""
        args = {}
        if 'value' in _dict:
            args['value'] = _dict.get('value')
        if 'added_at' in _dict:
            args['added_at'] = _dict.get('added_at')
        if 'added_by' in _dict:
            args['added_by'] = User.from_dict(_dict.get('added_by'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Comment object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'value') and self.value is not None:
            _dict['value'] = self.value
        if hasattr(self, 'added_at') and self.added_at is not None:
            _dict['added_at'] = self.added_at
        if hasattr(self, 'added_by') and self.added_by is not None:
            _dict['added_by'] = self.added_by.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Comment object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Comment') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Comment') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Offering:
    """
    Offering details.

    :attr str name: Name of the offering.
    :attr OfferingType type: Offering type.
    """

    def __init__(self, name: str, type: 'OfferingType') -> None:
        """
        Initialize a Offering object.

        :param str name: Name of the offering.
        :param OfferingType type: Offering type.
        """
        self.name = name
        self.type = type

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Offering':
        """Initialize a Offering object from a json dictionary."""
        args = {}
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        else:
            raise ValueError('Required property \'name\' not present in Offering JSON')
        if 'type' in _dict:
            args['type'] = OfferingType.from_dict(_dict.get('type'))
        else:
            raise ValueError('Required property \'type\' not present in Offering JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Offering object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Offering object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Offering') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Offering') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class OfferingType:
    """
    Offering type.

    :attr str group: Offering type group. "crn_service_name" is preferred over
          "category" as the latter is legacy and will be deprecated in the future.
    :attr str key: CRN service name of the offering.
    :attr str kind: (optional) Optional. Platform kind of the offering.
    :attr str id: (optional) Offering id in the catalog. This alone is enough to
          identify the offering.
    """

    def __init__(self, group: str, key: str, *, kind: str = None, id: str = None) -> None:
        """
        Initialize a OfferingType object.

        :param str group: Offering type group. "crn_service_name" is preferred over
               "category" as the latter is legacy and will be deprecated in the future.
        :param str key: CRN service name of the offering.
        :param str kind: (optional) Optional. Platform kind of the offering.
        :param str id: (optional) Offering id in the catalog. This alone is enough
               to identify the offering.
        """
        self.group = group
        self.key = key
        self.kind = kind
        self.id = id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'OfferingType':
        """Initialize a OfferingType object from a json dictionary."""
        args = {}
        if 'group' in _dict:
            args['group'] = _dict.get('group')
        else:
            raise ValueError('Required property \'group\' not present in OfferingType JSON')
        if 'key' in _dict:
            args['key'] = _dict.get('key')
        else:
            raise ValueError('Required property \'key\' not present in OfferingType JSON')
        if 'kind' in _dict:
            args['kind'] = _dict.get('kind')
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a OfferingType object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'group') and self.group is not None:
            _dict['group'] = self.group
        if hasattr(self, 'key') and self.key is not None:
            _dict['key'] = self.key
        if hasattr(self, 'kind') and self.kind is not None:
            _dict['kind'] = self.kind
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this OfferingType object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'OfferingType') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'OfferingType') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class GroupEnum(str, Enum):
        """
        Offering type group. "crn_service_name" is preferred over "category" as the latter
        is legacy and will be deprecated in the future.
        """

        CRN_SERVICE_NAME = 'crn_service_name'
        CATEGORY = 'category'


class PaginationLink:
    """
    Container for URL pointer to related pages of cases.

    :attr str href: (optional) URL to related pages of cases.
    """

    def __init__(self, *, href: str = None) -> None:
        """
        Initialize a PaginationLink object.

        :param str href: (optional) URL to related pages of cases.
        """
        self.href = href

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'PaginationLink':
        """Initialize a PaginationLink object from a json dictionary."""
        args = {}
        if 'href' in _dict:
            args['href'] = _dict.get('href')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a PaginationLink object from a json dictionary."""
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
        """Return a `str` version of this PaginationLink object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'PaginationLink') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'PaginationLink') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Resource:
    """
    A resource record of a case.

    :attr str crn: (optional) ID of the resource.
    :attr str name: (optional) Name of the resource.
    :attr str type: (optional) Type of resource.
    :attr str url: (optional) URL of resource.
    :attr str note: (optional) Note about resource.
    """

    def __init__(
        self, *, crn: str = None, name: str = None, type: str = None, url: str = None, note: str = None
    ) -> None:
        """
        Initialize a Resource object.

        :param str crn: (optional) ID of the resource.
        :param str name: (optional) Name of the resource.
        :param str type: (optional) Type of resource.
        :param str url: (optional) URL of resource.
        :param str note: (optional) Note about resource.
        """
        self.crn = crn
        self.name = name
        self.type = type
        self.url = url
        self.note = note

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Resource':
        """Initialize a Resource object from a json dictionary."""
        args = {}
        if 'crn' in _dict:
            args['crn'] = _dict.get('crn')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'type' in _dict:
            args['type'] = _dict.get('type')
        if 'url' in _dict:
            args['url'] = _dict.get('url')
        if 'note' in _dict:
            args['note'] = _dict.get('note')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Resource object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'url') and self.url is not None:
            _dict['url'] = self.url
        if hasattr(self, 'note') and self.note is not None:
            _dict['note'] = self.note
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


class ResourcePayload:
    """
    Payload to add a resource to a case.

    :attr str crn: (optional) Cloud Resource Name of the resource.
    :attr str type: (optional) Only used to attach Classic IaaS devices that have no
          CRN.
    :attr float id: (optional) Deprecated: Only used to attach Classic IaaS devices
          that have no CRN. Id of Classic IaaS device. This is deprecated in favor of the
          crn field.
    :attr str note: (optional) A note about this resource.
    """

    def __init__(self, *, crn: str = None, type: str = None, id: float = None, note: str = None) -> None:
        """
        Initialize a ResourcePayload object.

        :param str crn: (optional) Cloud Resource Name of the resource.
        :param str type: (optional) Only used to attach Classic IaaS devices that
               have no CRN.
        :param float id: (optional) Deprecated: Only used to attach Classic IaaS
               devices that have no CRN. Id of Classic IaaS device. This is deprecated in
               favor of the crn field.
        :param str note: (optional) A note about this resource.
        """
        self.crn = crn
        self.type = type
        self.id = id
        self.note = note

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResourcePayload':
        """Initialize a ResourcePayload object from a json dictionary."""
        args = {}
        if 'crn' in _dict:
            args['crn'] = _dict.get('crn')
        if 'type' in _dict:
            args['type'] = _dict.get('type')
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'note' in _dict:
            args['note'] = _dict.get('note')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResourcePayload object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'note') and self.note is not None:
            _dict['note'] = self.note
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResourcePayload object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResourcePayload') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResourcePayload') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class StatusPayload:
    """
    Payload to update status of the case.

    :attr str action: action to perform on the case.
    """

    def __init__(self, action: str) -> None:
        """
        Initialize a StatusPayload object.

        :param str action: action to perform on the case.
        """
        msg = "Cannot instantiate base class. Instead, instantiate one of the defined subclasses: {0}".format(
            ", ".join(['ResolvePayload', 'UnresolvePayload', 'AcceptPayload'])
        )
        raise Exception(msg)

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'StatusPayload':
        """Initialize a StatusPayload object from a json dictionary."""
        disc_class = cls._get_class_by_discriminator(_dict)
        if disc_class != cls:
            return disc_class.from_dict(_dict)
        msg = (
            "Cannot convert dictionary into an instance of base class 'StatusPayload'. "
            + "The discriminator value should map to a valid subclass: {1}"
        ).format(", ".join(['ResolvePayload', 'UnresolvePayload', 'AcceptPayload']))
        raise Exception(msg)

    @classmethod
    def _from_dict(cls, _dict: Dict):
        """Initialize a StatusPayload object from a json dictionary."""
        return cls.from_dict(_dict)

    @classmethod
    def _get_class_by_discriminator(cls, _dict: Dict) -> object:
        mapping = {}
        mapping['resolve'] = 'ResolvePayload'
        mapping['unresolve'] = 'UnresolvePayload'
        mapping['accept'] = 'AcceptPayload'
        disc_value = _dict.get('action')
        if disc_value is None:
            raise ValueError('Discriminator property \'action\' not found in StatusPayload JSON')
        class_name = mapping.get(disc_value, disc_value)
        try:
            disc_class = getattr(sys.modules[__name__], class_name)
        except AttributeError:
            disc_class = cls
        if isinstance(disc_class, object):
            return disc_class
        raise TypeError('%s is not a discriminator class' % class_name)

    class ActionEnum(str, Enum):
        """
        action to perform on the case.
        """

        RESOLVE = 'resolve'
        UNRESOLVE = 'unresolve'
        ACCEPT = 'accept'


class User:
    """
    User info in a case.

    :attr str name: (optional) Full name of the user.
    :attr str realm: the ID realm.
    :attr str user_id: unique user ID in the realm specified by the type.
    """

    def __init__(self, realm: str, user_id: str, *, name: str = None) -> None:
        """
        Initialize a User object.

        :param str realm: the ID realm.
        :param str user_id: unique user ID in the realm specified by the type.
        """
        self.name = name
        self.realm = realm
        self.user_id = user_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'User':
        """Initialize a User object from a json dictionary."""
        args = {}
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'realm' in _dict:
            args['realm'] = _dict.get('realm')
        else:
            raise ValueError('Required property \'realm\' not present in User JSON')
        if 'user_id' in _dict:
            args['user_id'] = _dict.get('user_id')
        else:
            raise ValueError('Required property \'user_id\' not present in User JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a User object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'name') and getattr(self, 'name') is not None:
            _dict['name'] = getattr(self, 'name')
        if hasattr(self, 'realm') and self.realm is not None:
            _dict['realm'] = self.realm
        if hasattr(self, 'user_id') and self.user_id is not None:
            _dict['user_id'] = self.user_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this User object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'User') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'User') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class RealmEnum(str, Enum):
        """
        the ID realm.
        """

        IBMID = 'IBMid'
        SL = 'SL'
        BSS = 'BSS'


class Watchlist:
    """
    Payload to add/remove users to/from the case watchlist.

    :attr List[User] watchlist: (optional) Array of user ID objects.
    """

    def __init__(self, *, watchlist: List['User'] = None) -> None:
        """
        Initialize a Watchlist object.

        :param List[User] watchlist: (optional) Array of user ID objects.
        """
        self.watchlist = watchlist

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Watchlist':
        """Initialize a Watchlist object from a json dictionary."""
        args = {}
        if 'watchlist' in _dict:
            args['watchlist'] = [User.from_dict(x) for x in _dict.get('watchlist')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Watchlist object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'watchlist') and self.watchlist is not None:
            _dict['watchlist'] = [x.to_dict() for x in self.watchlist]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Watchlist object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Watchlist') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Watchlist') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class WatchlistAddResponse:
    """
    Response of a request when adding to watchlist.

    :attr List[User] added: (optional) List of added user.
    :attr List[User] failed: (optional) List of failed to add user.
    """

    def __init__(self, *, added: List['User'] = None, failed: List['User'] = None) -> None:
        """
        Initialize a WatchlistAddResponse object.

        :param List[User] added: (optional) List of added user.
        :param List[User] failed: (optional) List of failed to add user.
        """
        self.added = added
        self.failed = failed

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'WatchlistAddResponse':
        """Initialize a WatchlistAddResponse object from a json dictionary."""
        args = {}
        if 'added' in _dict:
            args['added'] = [User.from_dict(x) for x in _dict.get('added')]
        if 'failed' in _dict:
            args['failed'] = [User.from_dict(x) for x in _dict.get('failed')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a WatchlistAddResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'added') and self.added is not None:
            _dict['added'] = [x.to_dict() for x in self.added]
        if hasattr(self, 'failed') and self.failed is not None:
            _dict['failed'] = [x.to_dict() for x in self.failed]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this WatchlistAddResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'WatchlistAddResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'WatchlistAddResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AcceptPayload(StatusPayload):
    """
    Payload to accept the proposed resolution of the case.

    :attr str action: action to perform on the case.
    :attr str comment: (optional) Comment about accepting the proposed resolution.
    """

    def __init__(self, action: str, *, comment: str = None) -> None:
        """
        Initialize a AcceptPayload object.

        :param str action: action to perform on the case.
        :param str comment: (optional) Comment about accepting the proposed
               resolution.
        """
        # pylint: disable=super-init-not-called
        self.action = action
        self.comment = comment

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AcceptPayload':
        """Initialize a AcceptPayload object from a json dictionary."""
        args = {}
        if 'action' in _dict:
            args['action'] = _dict.get('action')
        else:
            raise ValueError('Required property \'action\' not present in AcceptPayload JSON')
        if 'comment' in _dict:
            args['comment'] = _dict.get('comment')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AcceptPayload object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'action') and self.action is not None:
            _dict['action'] = self.action
        if hasattr(self, 'comment') and self.comment is not None:
            _dict['comment'] = self.comment
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AcceptPayload object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AcceptPayload') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AcceptPayload') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class ActionEnum(str, Enum):
        """
        action to perform on the case.
        """

        RESOLVE = 'resolve'
        UNRESOLVE = 'unresolve'
        ACCEPT = 'accept'


class ResolvePayload(StatusPayload):
    """
    Payload to resolve the case.

    :attr str action: action to perform on the case.
    :attr str comment: (optional) comment of resolution.
    :attr int resolution_code: * 1: Client error
          * 2: Defect found with Component/Service
          * 3: Documentation Error
          * 4: Solution found in forums
          * 5: Solution found in public Documentation
          * 6: Solution no longer required
          * 7: Solution provided by IBM outside of support case
          * 8: Solution provided by IBM support engineer.
    """

    def __init__(self, action: str, resolution_code: int, *, comment: str = None) -> None:
        """
        Initialize a ResolvePayload object.

        :param str action: action to perform on the case.
        :param int resolution_code: * 1: Client error
               * 2: Defect found with Component/Service
               * 3: Documentation Error
               * 4: Solution found in forums
               * 5: Solution found in public Documentation
               * 6: Solution no longer required
               * 7: Solution provided by IBM outside of support case
               * 8: Solution provided by IBM support engineer.
        :param str comment: (optional) comment of resolution.
        """
        # pylint: disable=super-init-not-called
        self.action = action
        self.comment = comment
        self.resolution_code = resolution_code

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResolvePayload':
        """Initialize a ResolvePayload object from a json dictionary."""
        args = {}
        if 'action' in _dict:
            args['action'] = _dict.get('action')
        else:
            raise ValueError('Required property \'action\' not present in ResolvePayload JSON')
        if 'comment' in _dict:
            args['comment'] = _dict.get('comment')
        if 'resolution_code' in _dict:
            args['resolution_code'] = _dict.get('resolution_code')
        else:
            raise ValueError('Required property \'resolution_code\' not present in ResolvePayload JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResolvePayload object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'action') and self.action is not None:
            _dict['action'] = self.action
        if hasattr(self, 'comment') and self.comment is not None:
            _dict['comment'] = self.comment
        if hasattr(self, 'resolution_code') and self.resolution_code is not None:
            _dict['resolution_code'] = self.resolution_code
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ResolvePayload object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResolvePayload') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResolvePayload') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class ActionEnum(str, Enum):
        """
        action to perform on the case.
        """

        RESOLVE = 'resolve'
        UNRESOLVE = 'unresolve'
        ACCEPT = 'accept'


class UnresolvePayload(StatusPayload):
    """
    Payload to unresolve the case.

    :attr str action: action to perform on the case.
    :attr str comment: Comment why the case should be unresolved.
    """

    def __init__(self, action: str, comment: str) -> None:
        """
        Initialize a UnresolvePayload object.

        :param str action: action to perform on the case.
        :param str comment: Comment why the case should be unresolved.
        """
        # pylint: disable=super-init-not-called
        self.action = action
        self.comment = comment

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'UnresolvePayload':
        """Initialize a UnresolvePayload object from a json dictionary."""
        args = {}
        if 'action' in _dict:
            args['action'] = _dict.get('action')
        else:
            raise ValueError('Required property \'action\' not present in UnresolvePayload JSON')
        if 'comment' in _dict:
            args['comment'] = _dict.get('comment')
        else:
            raise ValueError('Required property \'comment\' not present in UnresolvePayload JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a UnresolvePayload object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'action') and self.action is not None:
            _dict['action'] = self.action
        if hasattr(self, 'comment') and self.comment is not None:
            _dict['comment'] = self.comment
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this UnresolvePayload object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'UnresolvePayload') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'UnresolvePayload') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class ActionEnum(str, Enum):
        """
        action to perform on the case.
        """

        RESOLVE = 'resolve'
        UNRESOLVE = 'unresolve'
        ACCEPT = 'accept'


class FileWithMetadata:
    """
    A file with its associated metadata.

    :attr BinaryIO data: The data / content for the file.
    :attr str filename: (optional) The filename of the file.
    :attr str content_type: (optional) The content type of the file.
    """

    def __init__(self, data: BinaryIO, *, filename: str = None, content_type: str = None) -> None:
        """
        Initialize a FileWithMetadata object.

        :param BinaryIO data: The data / content for the file.
        :param str filename: (optional) The filename of the file.
        :param str content_type: (optional) The content type of the file.
        """
        self.data = data
        self.filename = filename
        self.content_type = content_type

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'FileWithMetadata':
        """Initialize a FileWithMetadata object from a json dictionary."""
        args = {}
        if 'data' in _dict:
            args['data'] = _dict.get('data')
        else:
            raise ValueError('Required property \'data\' not present in FileWithMetadata JSON')
        if 'filename' in _dict:
            args['filename'] = _dict.get('filename')
        if 'content_type' in _dict:
            args['content_type'] = _dict.get('content_type')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a FileWithMetadata object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'data') and self.data is not None:
            _dict['data'] = self.data
        if hasattr(self, 'filename') and self.filename is not None:
            _dict['filename'] = self.filename
        if hasattr(self, 'content_type') and self.content_type is not None:
            _dict['content_type'] = self.content_type
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this FileWithMetadata object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'FileWithMetadata') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'FileWithMetadata') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


##############################################################################
# Pagers
##############################################################################


class GetCasesPager:
    """
    GetCasesPager can be used to simplify the use of the "get_cases" method.
    """

    def __init__(
        self,
        *,
        client: CaseManagementV1,
        limit: int = None,
        search: str = None,
        sort: str = None,
        status: List[str] = None,
        fields: List[str] = None,
    ) -> None:
        """
        Initialize a GetCasesPager object.
        :param int limit: (optional) Number of cases that are returned.
        :param str search: (optional) String that a case might contain.
        :param str sort: (optional) Sort field and direction. If omitted, default
               to descending of updated date. Prefix "~" signifies sort in descending.
        :param List[str] status: (optional) Case status filter.
        :param List[str] fields: (optional) Selected fields of interest instead of
               all of the case information.
        """
        self._has_next = True
        self._client = client
        self._page_context = {'next': None}
        self._limit = limit
        self._search = search
        self._sort = sort
        self._status = status
        self._fields = fields

    def has_next(self) -> bool:
        """
        Returns true if there are potentially more results to be retrieved.
        """
        return self._has_next

    def get_next(self) -> List[dict]:
        """
        Returns the next page of results.
        :return: A List[dict], where each element is a dict that represents an instance of Case.
        :rtype: List[dict]
        """
        if not self.has_next():
            raise StopIteration(message='No more results available')

        result = self._client.get_cases(
            limit=self._limit,
            search=self._search,
            sort=self._sort,
            status=self._status,
            fields=self._fields,
            offset=self._page_context.get('next'),
        ).get_result()

        next = None
        next_page_link = result.get('next')
        if next_page_link is not None:
            next = get_query_param(next_page_link.get('href'), 'offset')
        self._page_context['next'] = next
        if next is None:
            self._has_next = False

        return result.get('cases')

    def get_all(self) -> List[dict]:
        """
        Returns all results by invoking get_next() repeatedly
        until all pages of results have been retrieved.
        :return: A List[dict], where each element is a dict that represents an instance of Case.
        :rtype: List[dict]
        """
        results = []
        while self.has_next():
            next_page = self.get_next()
            results.extend(next_page)
        return results
