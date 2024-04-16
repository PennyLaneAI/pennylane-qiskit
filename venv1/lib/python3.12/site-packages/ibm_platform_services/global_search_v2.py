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
Search for resources with the global and shared resource properties repository that is
integrated in the IBM Cloud platform. The search repository stores and searches cloud
resources attributes, which categorize or classify resources. A resource is a physical or
logical component that can be created or reserved for an application or service instance.
They are owned by resource providers, such as IBM Kubernetes Service, or resource
controller in IBM Cloud. Resources are uniquely identified by a Cloud Resource Name (CRN)
or by an IMS ID. The properties of a resource include tags and system properties. Both
properties are defined in an IBM Cloud billing account, and span across many regions.

API Version: 2.0.1
"""

from enum import Enum
from typing import Dict, List, Optional
import json

from ibm_cloud_sdk_core import BaseService, DetailedResponse
from ibm_cloud_sdk_core.authenticators.authenticator import Authenticator
from ibm_cloud_sdk_core.get_authenticator import get_authenticator_from_environment
from ibm_cloud_sdk_core.utils import convert_list

from .common import get_sdk_headers

##############################################################################
# Service
##############################################################################


class GlobalSearchV2(BaseService):
    """The global_search V2 service."""

    DEFAULT_SERVICE_URL = 'https://api.global-search-tagging.cloud.ibm.com'
    DEFAULT_SERVICE_NAME = 'global_search'

    @classmethod
    def new_instance(
        cls,
        service_name: str = DEFAULT_SERVICE_NAME,
    ) -> 'GlobalSearchV2':
        """
        Return a new client for the global_search service using the specified
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
        Construct a new client for the global_search service.

        :param Authenticator authenticator: The authenticator specifies the authentication mechanism.
               Get up to date information from https://github.com/IBM/python-sdk-core/blob/main/README.md
               about initializing the authenticator of your choice.
        """
        BaseService.__init__(self, service_url=self.DEFAULT_SERVICE_URL, authenticator=authenticator)

    #########################
    # Search
    #########################

    def search(
        self,
        *,
        query: Optional[str] = None,
        fields: Optional[List[str]] = None,
        search_cursor: Optional[str] = None,
        x_request_id: Optional[str] = None,
        x_correlation_id: Optional[str] = None,
        account_id: Optional[str] = None,
        limit: Optional[int] = None,
        timeout: Optional[int] = None,
        sort: Optional[List[str]] = None,
        is_deleted: Optional[str] = None,
        is_reclaimed: Optional[str] = None,
        is_public: Optional[str] = None,
        impersonate_user: Optional[str] = None,
        can_tag: Optional[str] = None,
        **kwargs,
    ) -> DetailedResponse:
        """
        Find instances of resources (v3).

        Find IAM-enabled resources or storage and network resources that run on classic
        infrastructure in a specific account ID. You can apply query strings if necessary.
        To filter results, you can insert a string by using the Lucene syntax and the
        query string is parsed into a series of terms and operators. A term can be a
        single word or a phrase, in which case the search is performed for all the words,
        in the same order. To filter for a specific value regardless of the property that
        contains it, type the search term without specifying a field. Only resources that
        belong to the account ID and that are accessible by the client are returned.
        You must use `/v3/resources/search` when you need to fetch more than `10000`
        resource items. On the first call, the operation returns a live cursor on the data
        that you must use on all the subsequent calls to get the next batch of results
        until you get the empty result set.
        By default, the fields that are returned for every resource are `crn`, `name`,
        `family`, `type`, and `account_id`. You can specify the subset of the fields you
        want in your request using the `fields` request body attribute. Set `"fields":
        ["*"]` to discover the set of fields which are available to request.

        :param str query: (optional) The Lucene-formatted query string. Default to
               '*' if not set.
        :param List[str] fields: (optional) The list of the fields returned by the
               search. By default, the returned fields are the `account_id`, `name`,
               `type`, `family`, and `crn`. For all queries, `crn` is always returned. You
               may set `"fields": ["*"]` to discover the set of fields available to
               request.
        :param str search_cursor: (optional) An opaque cursor that is returned on
               each call and that must be set on the subsequent call to get the next batch
               of items. If the search returns no items, then the search_cursor is not
               present in the response.
        :param str x_request_id: (optional) An alphanumeric string that is used to
               trace the request. The value  may include ASCII alphanumerics and any of
               following segment separators: space ( ), comma (,), hyphen, (-), and
               underscore (_) and may have a length up to 1024 bytes. The value is
               considered invalid and must be ignored if that value includes any other
               character or is longer than 1024 bytes or is fewer than 8 characters. If
               not specified or invalid, it is automatically replaced by a random (version
               4) UUID.
        :param str x_correlation_id: (optional) An alphanumeric string that is used
               to trace the request as a part of a larger context: the same value is used
               for downstream requests and retries of those requests. The value may
               include ASCII alphanumerics and any of following segment separators: space
               ( ), comma (,), hyphen, (-), and underscore (_) and may have a length up to
               1024 bytes. The value is considered invalid and must be ignored if that
               value includes any other character or is longer than 1024 bytes or is fewer
               than 8 characters. If not specified or invalid, it is automatically
               replaced by a random (version 4) UUID.
        :param str account_id: (optional) The account ID to filter resources.
        :param int limit: (optional) The maximum number of hits to return. Defaults
               to 10.
        :param int timeout: (optional) A search timeout in milliseconds, bounding
               the search request to run within the specified time value and bail with the
               hits accumulated up to that point when expired. Defaults to the system
               defined timeout.
        :param List[str] sort: (optional) Comma separated properties names that are
               used for sorting.
        :param str is_deleted: (optional) Determines if deleted documents should be
               included in result set or not. Possible values are false (default), true or
               any. If false, only existing documents are returned; if true, only deleted
               documents are returned; If any, both existing and deleted documents are
               returned. (_for administrators only_).
        :param str is_reclaimed: (optional) Determines if reclaimed documents
               should be included in result set or not. Possible values are false
               (default), true or any. If false, only not reclaimed documents are
               returned; if true, only reclaimed documents are returned; If any, both
               reclaimed and not reclaimed documents are returned.
        :param str is_public: (optional) Determines if public resources should be
               included in result set or not. Possible values are false (default), true or
               any. If false, do not search public resources; if true, search only public
               resources; If any, search also public resources.
        :param str impersonate_user: (optional) The user on whose behalf the search
               must be performed. Only a GhoST admin can impersonate a user, so be sure
               you set a GhoST admin IAM token in the Authorization header if you set this
               parameter. (_for administrators only_).
        :param str can_tag: (optional) Determines if the result set must return the
               resources that the user can tag or the resources that the user can view
               (only a GhoST admin can use this parameter). If false (default), only
               resources user can view are returned; if true, only resources that user has
               permissions for tagging are returned (_for administrators only_).
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ScanResult` object
        """

        headers = {
            'x-request-id': x_request_id,
            'x-correlation-id': x_correlation_id,
        }
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME,
            service_version='V2',
            operation_id='search',
        )
        headers.update(sdk_headers)

        params = {
            'account_id': account_id,
            'limit': limit,
            'timeout': timeout,
            'sort': convert_list(sort),
            'is_deleted': is_deleted,
            'is_reclaimed': is_reclaimed,
            'is_public': is_public,
            'impersonate_user': impersonate_user,
            'can_tag': can_tag,
        }

        data = {
            'query': query,
            'fields': fields,
            'search_cursor': search_cursor,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
            del kwargs['headers']
        headers['Accept'] = 'application/json'

        url = '/v3/resources/search'
        request = self.prepare_request(
            method='POST',
            url=url,
            headers=headers,
            params=params,
            data=data,
        )

        response = self.send(request, **kwargs)
        return response


class SearchEnums:
    """
    Enums for search parameters.
    """

    class IsDeleted(str, Enum):
        """
        Determines if deleted documents should be included in result set or not. Possible
        values are false (default), true or any. If false, only existing documents are
        returned; if true, only deleted documents are returned; If any, both existing and
        deleted documents are returned. (_for administrators only_).
        """

        TRUE = 'true'
        FALSE = 'false'
        ANY = 'any'

    class IsReclaimed(str, Enum):
        """
        Determines if reclaimed documents should be included in result set or not.
        Possible values are false (default), true or any. If false, only not reclaimed
        documents are returned; if true, only reclaimed documents are returned; If any,
        both reclaimed and not reclaimed documents are returned.
        """

        TRUE = 'true'
        FALSE = 'false'
        ANY = 'any'

    class IsPublic(str, Enum):
        """
        Determines if public resources should be included in result set or not. Possible
        values are false (default), true or any. If false, do not search public resources;
        if true, search only public resources; If any, search also public resources.
        """

        TRUE = 'true'
        FALSE = 'false'
        ANY = 'any'

    class CanTag(str, Enum):
        """
        Determines if the result set must return the resources that the user can tag or
        the resources that the user can view (only a GhoST admin can use this parameter).
        If false (default), only resources user can view are returned; if true, only
        resources that user has permissions for tagging are returned (_for administrators
        only_).
        """

        TRUE = 'true'
        FALSE = 'false'


##############################################################################
# Models
##############################################################################


class ResultItem:
    """
    A resource returned in a search result, which is identified by its `crn`. It contains
    other properties that depend on the resource type.

    :param str crn: Resource identifier in CRN format.
    """

    # The set of defined properties for the class
    _properties = frozenset(['crn'])

    def __init__(
        self,
        crn: str,
        **kwargs,
    ) -> None:
        """
        Initialize a ResultItem object.

        :param str crn: Resource identifier in CRN format.
        :param **kwargs: (optional) Any additional properties.
        """
        self.crn = crn
        for _key, _value in kwargs.items():
            setattr(self, _key, _value)

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ResultItem':
        """Initialize a ResultItem object from a json dictionary."""
        args = {}
        if (crn := _dict.get('crn')) is not None:
            args['crn'] = crn
        else:
            raise ValueError('Required property \'crn\' not present in ResultItem JSON')
        args.update({k: v for (k, v) in _dict.items() if k not in cls._properties})
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ResultItem object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        for _key in [k for k in vars(self).keys() if k not in ResultItem._properties]:
            _dict[_key] = getattr(self, _key)
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def get_properties(self) -> Dict:
        """Return a dictionary of arbitrary properties from this instance of ResultItem"""
        _dict = {}

        for _key in [k for k in vars(self).keys() if k not in ResultItem._properties]:
            _dict[_key] = getattr(self, _key)
        return _dict

    def set_properties(self, _dict: dict):
        """Set a dictionary of arbitrary properties to this instance of ResultItem"""
        for _key in [k for k in vars(self).keys() if k not in ResultItem._properties]:
            delattr(self, _key)

        for _key, _value in _dict.items():
            if _key not in ResultItem._properties:
                setattr(self, _key, _value)

    def __str__(self) -> str:
        """Return a `str` version of this ResultItem object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ResultItem') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ResultItem') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ScanResult:
    """
    The search scan response.

    :param str search_cursor: (optional) The search cursor to use on all calls after
          the first one.
    :param int limit: Value of the limit parameter specified by the user.
    :param List[ResultItem] items: The array of results. Each item represents a
          resource. For each resource, the requested `fields` are returned. If you did not
          set the `fields` request body parameter, then the `account_id`, `name`, `type`,
          `family`, and `crn` are returned. An empty array signals the end of the result
          set, which means there are no more results to fetch.
    """

    def __init__(
        self,
        limit: int,
        items: List['ResultItem'],
        *,
        search_cursor: Optional[str] = None,
    ) -> None:
        """
        Initialize a ScanResult object.

        :param int limit: Value of the limit parameter specified by the user.
        :param List[ResultItem] items: The array of results. Each item represents a
               resource. For each resource, the requested `fields` are returned. If you
               did not set the `fields` request body parameter, then the `account_id`,
               `name`, `type`, `family`, and `crn` are returned. An empty array signals
               the end of the result set, which means there are no more results to fetch.
        :param str search_cursor: (optional) The search cursor to use on all calls
               after the first one.
        """
        self.search_cursor = search_cursor
        self.limit = limit
        self.items = items

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ScanResult':
        """Initialize a ScanResult object from a json dictionary."""
        args = {}
        if (search_cursor := _dict.get('search_cursor')) is not None:
            args['search_cursor'] = search_cursor
        if (limit := _dict.get('limit')) is not None:
            args['limit'] = limit
        else:
            raise ValueError('Required property \'limit\' not present in ScanResult JSON')
        if (items := _dict.get('items')) is not None:
            args['items'] = [ResultItem.from_dict(v) for v in items]
        else:
            raise ValueError('Required property \'items\' not present in ScanResult JSON')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ScanResult object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'search_cursor') and self.search_cursor is not None:
            _dict['search_cursor'] = self.search_cursor
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'items') and self.items is not None:
            items_list = []
            for v in self.items:
                if isinstance(v, dict):
                    items_list.append(v)
                else:
                    items_list.append(v.to_dict())
            _dict['items'] = items_list
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ScanResult object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ScanResult') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ScanResult') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other
