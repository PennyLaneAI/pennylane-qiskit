# coding: utf-8

# (C) Copyright IBM Corp. 2021.
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

# IBM OpenAPI SDK Code Generator Version: 3.43.0-49eab5c7-20211117-152138

"""
This is the API to use for managing private catalogs for IBM Cloud. Private catalogs
provide a way to centrally manage access to products in the IBM Cloud catalog and your own
catalogs.

API Version: 1.0
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List
import base64
import json

from ibm_cloud_sdk_core import BaseService, DetailedResponse
from ibm_cloud_sdk_core.authenticators.authenticator import Authenticator
from ibm_cloud_sdk_core.get_authenticator import get_authenticator_from_environment
from ibm_cloud_sdk_core.utils import convert_list, convert_model, datetime_to_string, string_to_datetime

from .common import get_sdk_headers

##############################################################################
# Service
##############################################################################


class CatalogManagementV1(BaseService):
    """The Catalog Management V1 service."""

    DEFAULT_SERVICE_URL = 'https://cm.globalcatalog.cloud.ibm.com/api/v1-beta'
    DEFAULT_SERVICE_NAME = 'catalog_management'

    @classmethod
    def new_instance(
        cls,
        service_name: str = DEFAULT_SERVICE_NAME,
    ) -> 'CatalogManagementV1':
        """
        Return a new client for the Catalog Management service using the specified
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
        Construct a new client for the Catalog Management service.

        :param Authenticator authenticator: The authenticator specifies the authentication mechanism.
               Get up to date information from https://github.com/IBM/python-sdk-core/blob/main/README.md
               about initializing the authenticator of your choice.
        """
        BaseService.__init__(self, service_url=self.DEFAULT_SERVICE_URL, authenticator=authenticator)

    #########################
    # Account
    #########################

    def get_catalog_account(self, **kwargs) -> DetailedResponse:
        """
        Get catalog account settings.

        Get the account level settings for the account for private catalog.

        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Account` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_catalog_account'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        url = '/catalogaccount'
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def update_catalog_account(
        self, *, id: str = None, hide_ibm_cloud_catalog: bool = None, account_filters: 'Filters' = None, **kwargs
    ) -> DetailedResponse:
        """
        Update account settings.

        Update the account level settings for the account for private catalog.

        :param str id: (optional) Account identification.
        :param bool hide_ibm_cloud_catalog: (optional) Hide the public catalog in
               this account.
        :param Filters account_filters: (optional) Filters for account and catalog
               filters.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if account_filters is not None:
            account_filters = convert_model(account_filters)
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='update_catalog_account'
        )
        headers.update(sdk_headers)

        data = {'id': id, 'hide_IBM_cloud_catalog': hide_ibm_cloud_catalog, 'account_filters': account_filters}
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        url = '/catalogaccount'
        request = self.prepare_request(method='PUT', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def get_catalog_account_audit(self, **kwargs) -> DetailedResponse:
        """
        Get catalog account audit log.

        Get the audit log associated with a catalog account.

        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AuditLog` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_catalog_account_audit'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        url = '/catalogaccount/audit'
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def get_catalog_account_filters(self, *, catalog: str = None, **kwargs) -> DetailedResponse:
        """
        Get catalog account filters.

        Get the accumulated filters of the account and of the catalogs you have access to.

        :param str catalog: (optional) catalog id. Narrow down filters to the
               account and just the one catalog.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccumulatedFilters` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_catalog_account_filters'
        )
        headers.update(sdk_headers)

        params = {'catalog': catalog}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        url = '/catalogaccount/filters'
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    #########################
    # Catalogs
    #########################

    def list_catalogs(self, **kwargs) -> DetailedResponse:
        """
        Get list of catalogs.

        Retrieves the available catalogs for a given account. This can be used by an
        unauthenticated user to retrieve the public catalog.

        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `CatalogSearchResult` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='list_catalogs'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        url = '/catalogs'
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def create_catalog(
        self,
        *,
        id: str = None,
        rev: str = None,
        label: str = None,
        short_description: str = None,
        catalog_icon_url: str = None,
        tags: List[str] = None,
        features: List['Feature'] = None,
        disabled: bool = None,
        resource_group_id: str = None,
        owning_account: str = None,
        catalog_filters: 'Filters' = None,
        syndication_settings: 'SyndicationResource' = None,
        kind: str = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Create a catalog.

        Create a catalog for a given account.

        :param str id: (optional) Unique ID.
        :param str rev: (optional) Cloudant revision.
        :param str label: (optional) Display Name in the requested language.
        :param str short_description: (optional) Description in the requested
               language.
        :param str catalog_icon_url: (optional) URL for an icon associated with
               this catalog.
        :param List[str] tags: (optional) List of tags associated with this
               catalog.
        :param List[Feature] features: (optional) List of features associated with
               this catalog.
        :param bool disabled: (optional) Denotes whether a catalog is disabled.
        :param str resource_group_id: (optional) Resource group id the catalog is
               owned by.
        :param str owning_account: (optional) Account that owns catalog.
        :param Filters catalog_filters: (optional) Filters for account and catalog
               filters.
        :param SyndicationResource syndication_settings: (optional) Feature
               information.
        :param str kind: (optional) Kind of catalog. Supported kinds are offering
               and vpe.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Catalog` object
        """

        if features is not None:
            features = [convert_model(x) for x in features]
        if catalog_filters is not None:
            catalog_filters = convert_model(catalog_filters)
        if syndication_settings is not None:
            syndication_settings = convert_model(syndication_settings)
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='create_catalog'
        )
        headers.update(sdk_headers)

        data = {
            'id': id,
            '_rev': rev,
            'label': label,
            'short_description': short_description,
            'catalog_icon_url': catalog_icon_url,
            'tags': tags,
            'features': features,
            'disabled': disabled,
            'resource_group_id': resource_group_id,
            'owning_account': owning_account,
            'catalog_filters': catalog_filters,
            'syndication_settings': syndication_settings,
            'kind': kind,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        url = '/catalogs'
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def get_catalog(self, catalog_identifier: str, **kwargs) -> DetailedResponse:
        """
        Get catalog.

        Get a catalog. This can also be used by an unauthenticated user to get the public
        catalog.

        :param str catalog_identifier: Catalog identifier.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Catalog` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_catalog'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def replace_catalog(
        self,
        catalog_identifier: str,
        *,
        id: str = None,
        rev: str = None,
        label: str = None,
        short_description: str = None,
        catalog_icon_url: str = None,
        tags: List[str] = None,
        features: List['Feature'] = None,
        disabled: bool = None,
        resource_group_id: str = None,
        owning_account: str = None,
        catalog_filters: 'Filters' = None,
        syndication_settings: 'SyndicationResource' = None,
        kind: str = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Update catalog.

        Update a catalog.

        :param str catalog_identifier: Catalog identifier.
        :param str id: (optional) Unique ID.
        :param str rev: (optional) Cloudant revision.
        :param str label: (optional) Display Name in the requested language.
        :param str short_description: (optional) Description in the requested
               language.
        :param str catalog_icon_url: (optional) URL for an icon associated with
               this catalog.
        :param List[str] tags: (optional) List of tags associated with this
               catalog.
        :param List[Feature] features: (optional) List of features associated with
               this catalog.
        :param bool disabled: (optional) Denotes whether a catalog is disabled.
        :param str resource_group_id: (optional) Resource group id the catalog is
               owned by.
        :param str owning_account: (optional) Account that owns catalog.
        :param Filters catalog_filters: (optional) Filters for account and catalog
               filters.
        :param SyndicationResource syndication_settings: (optional) Feature
               information.
        :param str kind: (optional) Kind of catalog. Supported kinds are offering
               and vpe.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Catalog` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if features is not None:
            features = [convert_model(x) for x in features]
        if catalog_filters is not None:
            catalog_filters = convert_model(catalog_filters)
        if syndication_settings is not None:
            syndication_settings = convert_model(syndication_settings)
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='replace_catalog'
        )
        headers.update(sdk_headers)

        data = {
            'id': id,
            '_rev': rev,
            'label': label,
            'short_description': short_description,
            'catalog_icon_url': catalog_icon_url,
            'tags': tags,
            'features': features,
            'disabled': disabled,
            'resource_group_id': resource_group_id,
            'owning_account': owning_account,
            'catalog_filters': catalog_filters,
            'syndication_settings': syndication_settings,
            'kind': kind,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}'.format(**path_param_dict)
        request = self.prepare_request(method='PUT', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def delete_catalog(self, catalog_identifier: str, **kwargs) -> DetailedResponse:
        """
        Delete catalog.

        Delete a catalog.

        :param str catalog_identifier: Catalog identifier.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='delete_catalog'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['catalog_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}'.format(**path_param_dict)
        request = self.prepare_request(method='DELETE', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def get_catalog_audit(self, catalog_identifier: str, **kwargs) -> DetailedResponse:
        """
        Get catalog audit log.

        Get the audit log associated with a catalog.

        :param str catalog_identifier: Catalog identifier.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AuditLog` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_catalog_audit'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/audit'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    #########################
    # Offerings
    #########################

    def get_consumption_offerings(
        self,
        *,
        digest: bool = None,
        catalog: str = None,
        select: str = None,
        include_hidden: bool = None,
        limit: int = None,
        offset: int = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Get consumption offerings.

        Retrieve the available offerings from both public and from the account that
        currently scoped for consumption. These copies cannot be used for updating. They
        are not complete and only return what is visible to the caller. This can be used
        by an unauthenticated user to retreive publicly available offerings.

        :param bool digest: (optional) true - Strip down the content of what is
               returned. For example don't return the readme. Makes the result much
               smaller. Defaults to false.
        :param str catalog: (optional) catalog id. Narrow search down to just a
               particular catalog. It will apply the catalog's public filters to the
               public catalog offerings on the result.
        :param str select: (optional) What should be selected. Default is 'all'
               which will return both public and private offerings. 'public' returns only
               the public offerings and 'private' returns only the private offerings.
        :param bool include_hidden: (optional) true - include offerings which have
               been marked as hidden. The default is false and hidden offerings are not
               returned.
        :param int limit: (optional) number or results to return.
        :param int offset: (optional) number of results to skip before returning
               values.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `OfferingSearchResult` object
        """

        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_consumption_offerings'
        )
        headers.update(sdk_headers)

        params = {
            'digest': digest,
            'catalog': catalog,
            'select': select,
            'includeHidden': include_hidden,
            'limit': limit,
            'offset': offset,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        url = '/offerings'
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def list_offerings(
        self,
        catalog_identifier: str,
        *,
        digest: bool = None,
        limit: int = None,
        offset: int = None,
        name: str = None,
        sort: str = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Get list of offerings.

        Retrieve the available offerings in the specified catalog. This can also be used
        by an unauthenticated user to retreive publicly available offerings.

        :param str catalog_identifier: Catalog identifier.
        :param bool digest: (optional) true - Strip down the content of what is
               returned. For example don't return the readme. Makes the result much
               smaller. Defaults to false.
        :param int limit: (optional) The maximum number of results to return.
        :param int offset: (optional) The number of results to skip before
               returning values.
        :param str name: (optional) Only return results that contain the specified
               string.
        :param str sort: (optional) The field on which the output is sorted. Sorts
               by default by **label** property. Available fields are **name**, **label**,
               **created**, and **updated**. By adding **-** (i.e. **-label**) in front of
               the query string, you can specify descending order. Default is ascending
               order.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `OfferingSearchResult` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='list_offerings'
        )
        headers.update(sdk_headers)

        params = {'digest': digest, 'limit': limit, 'offset': offset, 'name': name, 'sort': sort}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/offerings'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def create_offering(
        self,
        catalog_identifier: str,
        *,
        id: str = None,
        rev: str = None,
        url: str = None,
        crn: str = None,
        label: str = None,
        name: str = None,
        offering_icon_url: str = None,
        offering_docs_url: str = None,
        offering_support_url: str = None,
        tags: List[str] = None,
        keywords: List[str] = None,
        rating: 'Rating' = None,
        created: datetime = None,
        updated: datetime = None,
        short_description: str = None,
        long_description: str = None,
        features: List['Feature'] = None,
        kinds: List['Kind'] = None,
        permit_request_ibm_public_publish: bool = None,
        ibm_publish_approved: bool = None,
        public_publish_approved: bool = None,
        public_original_crn: str = None,
        publish_public_crn: str = None,
        portal_approval_record: str = None,
        portal_ui_url: str = None,
        catalog_id: str = None,
        catalog_name: str = None,
        metadata: dict = None,
        disclaimer: str = None,
        hidden: bool = None,
        provider: str = None,
        provider_info: 'ProviderInfo' = None,
        repo_info: 'RepoInfo' = None,
        support: 'Support' = None,
        media: List['MediaItem'] = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Create offering.

        Create an offering.

        :param str catalog_identifier: Catalog identifier.
        :param str id: (optional) unique id.
        :param str rev: (optional) Cloudant revision.
        :param str url: (optional) The url for this specific offering.
        :param str crn: (optional) The crn for this specific offering.
        :param str label: (optional) Display Name in the requested language.
        :param str name: (optional) The programmatic name of this offering.
        :param str offering_icon_url: (optional) URL for an icon associated with
               this offering.
        :param str offering_docs_url: (optional) URL for an additional docs with
               this offering.
        :param str offering_support_url: (optional) [deprecated] - Use
               offering.support instead.  URL to be displayed in the Consumption UI for
               getting support on this offering.
        :param List[str] tags: (optional) List of tags associated with this
               catalog.
        :param List[str] keywords: (optional) List of keywords associated with
               offering, typically used to search for it.
        :param Rating rating: (optional) Repository info for offerings.
        :param datetime created: (optional) The date and time this catalog was
               created.
        :param datetime updated: (optional) The date and time this catalog was last
               updated.
        :param str short_description: (optional) Short description in the requested
               language.
        :param str long_description: (optional) Long description in the requested
               language.
        :param List[Feature] features: (optional) list of features associated with
               this offering.
        :param List[Kind] kinds: (optional) Array of kind.
        :param bool permit_request_ibm_public_publish: (optional) Is it permitted
               to request publishing to IBM or Public.
        :param bool ibm_publish_approved: (optional) Indicates if this offering has
               been approved for use by all IBMers.
        :param bool public_publish_approved: (optional) Indicates if this offering
               has been approved for use by all IBM Cloud users.
        :param str public_original_crn: (optional) The original offering CRN that
               this publish entry came from.
        :param str publish_public_crn: (optional) The crn of the public catalog
               entry of this offering.
        :param str portal_approval_record: (optional) The portal's approval record
               ID.
        :param str portal_ui_url: (optional) The portal UI URL.
        :param str catalog_id: (optional) The id of the catalog containing this
               offering.
        :param str catalog_name: (optional) The name of the catalog.
        :param dict metadata: (optional) Map of metadata values for this offering.
        :param str disclaimer: (optional) A disclaimer for this offering.
        :param bool hidden: (optional) Determine if this offering should be
               displayed in the Consumption UI.
        :param str provider: (optional) Deprecated - Provider of this offering.
        :param ProviderInfo provider_info: (optional) Information on the provider
               for this offering, or omitted if no provider information is given.
        :param RepoInfo repo_info: (optional) Repository info for offerings.
        :param Support support: (optional) Offering Support information.
        :param List[MediaItem] media: (optional) A list of media items related to
               this offering.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Offering` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if rating is not None:
            rating = convert_model(rating)
        if created is not None:
            created = datetime_to_string(created)
        if updated is not None:
            updated = datetime_to_string(updated)
        if features is not None:
            features = [convert_model(x) for x in features]
        if kinds is not None:
            kinds = [convert_model(x) for x in kinds]
        if provider_info is not None:
            provider_info = convert_model(provider_info)
        if repo_info is not None:
            repo_info = convert_model(repo_info)
        if support is not None:
            support = convert_model(support)
        if media is not None:
            media = [convert_model(x) for x in media]
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='create_offering'
        )
        headers.update(sdk_headers)

        data = {
            'id': id,
            '_rev': rev,
            'url': url,
            'crn': crn,
            'label': label,
            'name': name,
            'offering_icon_url': offering_icon_url,
            'offering_docs_url': offering_docs_url,
            'offering_support_url': offering_support_url,
            'tags': tags,
            'keywords': keywords,
            'rating': rating,
            'created': created,
            'updated': updated,
            'short_description': short_description,
            'long_description': long_description,
            'features': features,
            'kinds': kinds,
            'permit_request_ibm_public_publish': permit_request_ibm_public_publish,
            'ibm_publish_approved': ibm_publish_approved,
            'public_publish_approved': public_publish_approved,
            'public_original_crn': public_original_crn,
            'publish_public_crn': publish_public_crn,
            'portal_approval_record': portal_approval_record,
            'portal_ui_url': portal_ui_url,
            'catalog_id': catalog_id,
            'catalog_name': catalog_name,
            'metadata': metadata,
            'disclaimer': disclaimer,
            'hidden': hidden,
            'provider': provider,
            'provider_info': provider_info,
            'repo_info': repo_info,
            'support': support,
            'media': media,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/offerings'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def import_offering_version(
        self,
        catalog_identifier: str,
        offering_id: str,
        *,
        tags: List[str] = None,
        target_kinds: List[str] = None,
        content: bytes = None,
        zipurl: str = None,
        target_version: str = None,
        include_config: bool = None,
        is_vsi: bool = None,
        repo_type: str = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Import offering version.

        Import new version to offering from a tgz.

        :param str catalog_identifier: Catalog identifier.
        :param str offering_id: Offering identification.
        :param List[str] tags: (optional) Tags array.
        :param List[str] target_kinds: (optional) Target kinds.  Current valid
               values are 'iks', 'roks', 'vcenter', and 'terraform'.
        :param bytes content: (optional) byte array representing the content to be
               imported.  Only supported for OVA images at this time.
        :param str zipurl: (optional) URL path to zip location.  If not specified,
               must provide content in the body of this call.
        :param str target_version: (optional) The semver value for this new
               version, if not found in the zip url package content.
        :param bool include_config: (optional) Add all possible configuration
               values to this version when importing.
        :param bool is_vsi: (optional) Indicates that the current terraform
               template is used to install a VSI Image.
        :param str repo_type: (optional) The type of repository containing this
               version.  Valid values are 'public_git' or 'enterprise_git'.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Offering` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if offering_id is None:
            raise ValueError('offering_id must be provided')
        if content is not None:
            content = str(base64.b64encode(content), 'utf-8')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='import_offering_version'
        )
        headers.update(sdk_headers)

        params = {
            'zipurl': zipurl,
            'targetVersion': target_version,
            'includeConfig': include_config,
            'isVSI': is_vsi,
            'repoType': repo_type,
        }

        data = {'tags': tags, 'target_kinds': target_kinds, 'content': content}
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier', 'offering_id']
        path_param_values = self.encode_path_vars(catalog_identifier, offering_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/offerings/{offering_id}/version'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers, params=params, data=data)

        response = self.send(request, **kwargs)
        return response

    def import_offering(
        self,
        catalog_identifier: str,
        *,
        tags: List[str] = None,
        target_kinds: List[str] = None,
        content: bytes = None,
        zipurl: str = None,
        offering_id: str = None,
        target_version: str = None,
        include_config: bool = None,
        is_vsi: bool = None,
        repo_type: str = None,
        x_auth_token: str = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Import offering.

        Import a new offering from a tgz.

        :param str catalog_identifier: Catalog identifier.
        :param List[str] tags: (optional) Tags array.
        :param List[str] target_kinds: (optional) Target kinds.  Current valid
               values are 'iks', 'roks', 'vcenter', and 'terraform'.
        :param bytes content: (optional) byte array representing the content to be
               imported.  Only supported for OVA images at this time.
        :param str zipurl: (optional) URL path to zip location.  If not specified,
               must provide content in this post body.
        :param str offering_id: (optional) Re-use the specified offeringID during
               import.
        :param str target_version: (optional) The semver value for this new
               version.
        :param bool include_config: (optional) Add all possible configuration items
               when creating this version.
        :param bool is_vsi: (optional) Indicates that the current terraform
               template is used to install a VSI Image.
        :param str repo_type: (optional) The type of repository containing this
               version.  Valid values are 'public_git' or 'enterprise_git'.
        :param str x_auth_token: (optional) Authentication token used to access the
               specified zip file.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Offering` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if content is not None:
            content = str(base64.b64encode(content), 'utf-8')
        headers = {'X-Auth-Token': x_auth_token}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='import_offering'
        )
        headers.update(sdk_headers)

        params = {
            'zipurl': zipurl,
            'offeringID': offering_id,
            'targetVersion': target_version,
            'includeConfig': include_config,
            'isVSI': is_vsi,
            'repoType': repo_type,
        }

        data = {'tags': tags, 'target_kinds': target_kinds, 'content': content}
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/import/offerings'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers, params=params, data=data)

        response = self.send(request, **kwargs)
        return response

    def reload_offering(
        self,
        catalog_identifier: str,
        offering_id: str,
        target_version: str,
        *,
        tags: List[str] = None,
        target_kinds: List[str] = None,
        content: bytes = None,
        zipurl: str = None,
        repo_type: str = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Reload offering.

        Reload an existing version in offering from a tgz.

        :param str catalog_identifier: Catalog identifier.
        :param str offering_id: Offering identification.
        :param str target_version: The semver value for this new version.
        :param List[str] tags: (optional) Tags array.
        :param List[str] target_kinds: (optional) Target kinds.  Current valid
               values are 'iks', 'roks', 'vcenter', and 'terraform'.
        :param bytes content: (optional) byte array representing the content to be
               imported.  Only supported for OVA images at this time.
        :param str zipurl: (optional) URL path to zip location.  If not specified,
               must provide content in this post body.
        :param str repo_type: (optional) The type of repository containing this
               version.  Valid values are 'public_git' or 'enterprise_git'.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Offering` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if offering_id is None:
            raise ValueError('offering_id must be provided')
        if target_version is None:
            raise ValueError('target_version must be provided')
        if content is not None:
            content = str(base64.b64encode(content), 'utf-8')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='reload_offering'
        )
        headers.update(sdk_headers)

        params = {'targetVersion': target_version, 'zipurl': zipurl, 'repoType': repo_type}

        data = {'tags': tags, 'target_kinds': target_kinds, 'content': content}
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier', 'offering_id']
        path_param_values = self.encode_path_vars(catalog_identifier, offering_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/offerings/{offering_id}/reload'.format(**path_param_dict)
        request = self.prepare_request(method='PUT', url=url, headers=headers, params=params, data=data)

        response = self.send(request, **kwargs)
        return response

    def get_offering(
        self, catalog_identifier: str, offering_id: str, *, type: str = None, digest: bool = None, **kwargs
    ) -> DetailedResponse:
        """
        Get offering.

        Get an offering. This can be used by an unauthenticated user for publicly
        available offerings.

        :param str catalog_identifier: Catalog identifier.
        :param str offering_id: Offering identification.
        :param str type: (optional) Offering Parameter Type.  Valid values are
               'name' or 'id'.  Default is 'id'.
        :param bool digest: (optional) Return the digest format of the specified
               offering.  Default is false.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Offering` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if offering_id is None:
            raise ValueError('offering_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_offering'
        )
        headers.update(sdk_headers)

        params = {'type': type, 'digest': digest}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier', 'offering_id']
        path_param_values = self.encode_path_vars(catalog_identifier, offering_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/offerings/{offering_id}'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def replace_offering(
        self,
        catalog_identifier: str,
        offering_id: str,
        *,
        id: str = None,
        rev: str = None,
        url: str = None,
        crn: str = None,
        label: str = None,
        name: str = None,
        offering_icon_url: str = None,
        offering_docs_url: str = None,
        offering_support_url: str = None,
        tags: List[str] = None,
        keywords: List[str] = None,
        rating: 'Rating' = None,
        created: datetime = None,
        updated: datetime = None,
        short_description: str = None,
        long_description: str = None,
        features: List['Feature'] = None,
        kinds: List['Kind'] = None,
        permit_request_ibm_public_publish: bool = None,
        ibm_publish_approved: bool = None,
        public_publish_approved: bool = None,
        public_original_crn: str = None,
        publish_public_crn: str = None,
        portal_approval_record: str = None,
        portal_ui_url: str = None,
        catalog_id: str = None,
        catalog_name: str = None,
        metadata: dict = None,
        disclaimer: str = None,
        hidden: bool = None,
        provider: str = None,
        provider_info: 'ProviderInfo' = None,
        repo_info: 'RepoInfo' = None,
        support: 'Support' = None,
        media: List['MediaItem'] = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Update offering.

        Update an offering.

        :param str catalog_identifier: Catalog identifier.
        :param str offering_id: Offering identification.
        :param str id: (optional) unique id.
        :param str rev: (optional) Cloudant revision.
        :param str url: (optional) The url for this specific offering.
        :param str crn: (optional) The crn for this specific offering.
        :param str label: (optional) Display Name in the requested language.
        :param str name: (optional) The programmatic name of this offering.
        :param str offering_icon_url: (optional) URL for an icon associated with
               this offering.
        :param str offering_docs_url: (optional) URL for an additional docs with
               this offering.
        :param str offering_support_url: (optional) [deprecated] - Use
               offering.support instead.  URL to be displayed in the Consumption UI for
               getting support on this offering.
        :param List[str] tags: (optional) List of tags associated with this
               catalog.
        :param List[str] keywords: (optional) List of keywords associated with
               offering, typically used to search for it.
        :param Rating rating: (optional) Repository info for offerings.
        :param datetime created: (optional) The date and time this catalog was
               created.
        :param datetime updated: (optional) The date and time this catalog was last
               updated.
        :param str short_description: (optional) Short description in the requested
               language.
        :param str long_description: (optional) Long description in the requested
               language.
        :param List[Feature] features: (optional) list of features associated with
               this offering.
        :param List[Kind] kinds: (optional) Array of kind.
        :param bool permit_request_ibm_public_publish: (optional) Is it permitted
               to request publishing to IBM or Public.
        :param bool ibm_publish_approved: (optional) Indicates if this offering has
               been approved for use by all IBMers.
        :param bool public_publish_approved: (optional) Indicates if this offering
               has been approved for use by all IBM Cloud users.
        :param str public_original_crn: (optional) The original offering CRN that
               this publish entry came from.
        :param str publish_public_crn: (optional) The crn of the public catalog
               entry of this offering.
        :param str portal_approval_record: (optional) The portal's approval record
               ID.
        :param str portal_ui_url: (optional) The portal UI URL.
        :param str catalog_id: (optional) The id of the catalog containing this
               offering.
        :param str catalog_name: (optional) The name of the catalog.
        :param dict metadata: (optional) Map of metadata values for this offering.
        :param str disclaimer: (optional) A disclaimer for this offering.
        :param bool hidden: (optional) Determine if this offering should be
               displayed in the Consumption UI.
        :param str provider: (optional) Deprecated - Provider of this offering.
        :param ProviderInfo provider_info: (optional) Information on the provider
               for this offering, or omitted if no provider information is given.
        :param RepoInfo repo_info: (optional) Repository info for offerings.
        :param Support support: (optional) Offering Support information.
        :param List[MediaItem] media: (optional) A list of media items related to
               this offering.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Offering` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if offering_id is None:
            raise ValueError('offering_id must be provided')
        if rating is not None:
            rating = convert_model(rating)
        if created is not None:
            created = datetime_to_string(created)
        if updated is not None:
            updated = datetime_to_string(updated)
        if features is not None:
            features = [convert_model(x) for x in features]
        if kinds is not None:
            kinds = [convert_model(x) for x in kinds]
        if provider_info is not None:
            provider_info = convert_model(provider_info)
        if repo_info is not None:
            repo_info = convert_model(repo_info)
        if support is not None:
            support = convert_model(support)
        if media is not None:
            media = [convert_model(x) for x in media]
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='replace_offering'
        )
        headers.update(sdk_headers)

        data = {
            'id': id,
            '_rev': rev,
            'url': url,
            'crn': crn,
            'label': label,
            'name': name,
            'offering_icon_url': offering_icon_url,
            'offering_docs_url': offering_docs_url,
            'offering_support_url': offering_support_url,
            'tags': tags,
            'keywords': keywords,
            'rating': rating,
            'created': created,
            'updated': updated,
            'short_description': short_description,
            'long_description': long_description,
            'features': features,
            'kinds': kinds,
            'permit_request_ibm_public_publish': permit_request_ibm_public_publish,
            'ibm_publish_approved': ibm_publish_approved,
            'public_publish_approved': public_publish_approved,
            'public_original_crn': public_original_crn,
            'publish_public_crn': publish_public_crn,
            'portal_approval_record': portal_approval_record,
            'portal_ui_url': portal_ui_url,
            'catalog_id': catalog_id,
            'catalog_name': catalog_name,
            'metadata': metadata,
            'disclaimer': disclaimer,
            'hidden': hidden,
            'provider': provider,
            'provider_info': provider_info,
            'repo_info': repo_info,
            'support': support,
            'media': media,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier', 'offering_id']
        path_param_values = self.encode_path_vars(catalog_identifier, offering_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/offerings/{offering_id}'.format(**path_param_dict)
        request = self.prepare_request(method='PUT', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def update_offering(
        self,
        catalog_identifier: str,
        offering_id: str,
        if_match: str,
        *,
        updates: List['JsonPatchOperation'] = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Update offering.

        Update an offering.

        :param str catalog_identifier: Catalog identifier.
        :param str offering_id: Offering identification.
        :param str if_match: Offering etag contained in quotes.
        :param List[JsonPatchOperation] updates: (optional)
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Offering` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if offering_id is None:
            raise ValueError('offering_id must be provided')
        if if_match is None:
            raise ValueError('if_match must be provided')
        if updates is not None:
            updates = [convert_model(x) for x in updates]
        headers = {'If-Match': if_match}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='update_offering'
        )
        headers.update(sdk_headers)

        data = json.dumps(updates)
        headers['content-type'] = 'application/json-patch+json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier', 'offering_id']
        path_param_values = self.encode_path_vars(catalog_identifier, offering_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/offerings/{offering_id}'.format(**path_param_dict)
        request = self.prepare_request(method='PATCH', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def delete_offering(self, catalog_identifier: str, offering_id: str, **kwargs) -> DetailedResponse:
        """
        Delete offering.

        Delete an offering.

        :param str catalog_identifier: Catalog identifier.
        :param str offering_id: Offering identification.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if offering_id is None:
            raise ValueError('offering_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='delete_offering'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['catalog_identifier', 'offering_id']
        path_param_values = self.encode_path_vars(catalog_identifier, offering_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/offerings/{offering_id}'.format(**path_param_dict)
        request = self.prepare_request(method='DELETE', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def get_offering_audit(self, catalog_identifier: str, offering_id: str, **kwargs) -> DetailedResponse:
        """
        Get offering audit log.

        Get the audit log associated with an offering.

        :param str catalog_identifier: Catalog identifier.
        :param str offering_id: Offering identifier.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AuditLog` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if offering_id is None:
            raise ValueError('offering_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_offering_audit'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier', 'offering_id']
        path_param_values = self.encode_path_vars(catalog_identifier, offering_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/offerings/{offering_id}/audit'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def replace_offering_icon(
        self, catalog_identifier: str, offering_id: str, file_name: str, **kwargs
    ) -> DetailedResponse:
        """
        Upload icon for offering.

        Upload an icon file to be stored in GC. File is uploaded as a binary payload - not
        as a form.

        :param str catalog_identifier: Catalog identifier.
        :param str offering_id: Offering identification.
        :param str file_name: Name of the file name that is being uploaded.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Offering` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if offering_id is None:
            raise ValueError('offering_id must be provided')
        if file_name is None:
            raise ValueError('file_name must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='replace_offering_icon'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier', 'offering_id', 'file_name']
        path_param_values = self.encode_path_vars(catalog_identifier, offering_id, file_name)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/offerings/{offering_id}/icon/{file_name}'.format(**path_param_dict)
        request = self.prepare_request(method='PUT', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def update_offering_ibm(
        self, catalog_identifier: str, offering_id: str, approval_type: str, approved: str, **kwargs
    ) -> DetailedResponse:
        """
        Allow offering to be published.

        Approve or disapprove the offering to be allowed to publish to the IBM Public
        Catalog. Options:
        * `allow_request` - (Allow requesting to publish to IBM)
        * `ibm` - (Allow publishing to be visible to IBM only)
        * `public` - (Allow publishing to be visible to everyone, including IBM)
        If disapprove `public`, then `ibm` approval will not  be changed. If disapprove
        `ibm` then `public` will automatically be disapproved. if disapprove
        `allow_request` then all rights to publish will be removed. This is because the
        process steps always go first through `allow` to `ibm` and then to `public`. `ibm`
        cannot be skipped. Only users with Approval IAM authority can use this. Approvers
        should use the catalog and offering id from the public catalog since they wouldn't
        have access to the private offering.

        :param str catalog_identifier: Catalog identifier.
        :param str offering_id: Offering identification.
        :param str approval_type: Type of approval, ibm or public.
        :param str approved: Approve (true) or disapprove (false).
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ApprovalResult` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if offering_id is None:
            raise ValueError('offering_id must be provided')
        if approval_type is None:
            raise ValueError('approval_type must be provided')
        if approved is None:
            raise ValueError('approved must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='update_offering_ibm'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier', 'offering_id', 'approval_type', 'approved']
        path_param_values = self.encode_path_vars(catalog_identifier, offering_id, approval_type, approved)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/offerings/{offering_id}/publish/{approval_type}/{approved}'.format(
            **path_param_dict
        )
        request = self.prepare_request(method='POST', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def deprecate_offering(
        self,
        catalog_identifier: str,
        offering_id: str,
        setting: str,
        *,
        description: str = None,
        days_until_deprecate: int = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Allows offering to be deprecated.

        Approve or disapprove the offering to be deprecated.

        :param str catalog_identifier: Catalog identifier.
        :param str offering_id: Offering identification.
        :param str setting: Set deprecation (true) or cancel deprecation (false).
        :param str description: (optional) Additional information that users can
               provide to be displayed in deprecation notification.
        :param int days_until_deprecate: (optional) Specifies the amount of days
               until product is not available in catalog.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if offering_id is None:
            raise ValueError('offering_id must be provided')
        if setting is None:
            raise ValueError('setting must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='deprecate_offering'
        )
        headers.update(sdk_headers)

        data = {'description': description, 'days_until_deprecate': days_until_deprecate}
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['catalog_identifier', 'offering_id', 'setting']
        path_param_values = self.encode_path_vars(catalog_identifier, offering_id, setting)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/offerings/{offering_id}/deprecate/{setting}'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def get_offering_updates(
        self,
        catalog_identifier: str,
        offering_id: str,
        kind: str,
        x_auth_refresh_token: str,
        *,
        target: str = None,
        version: str = None,
        cluster_id: str = None,
        region: str = None,
        resource_group_id: str = None,
        namespace: str = None,
        sha: str = None,
        channel: str = None,
        namespaces: List[str] = None,
        all_namespaces: bool = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Get version updates.

        Get available updates for the specified version.

        :param str catalog_identifier: Catalog identifier.
        :param str offering_id: Offering identification.
        :param str kind: The kind of offering (e.g, helm, ova, terraform ...).
        :param str x_auth_refresh_token: IAM Refresh token.
        :param str target: (optional) The target kind of the currently installed
               version (e.g. iks, roks, etc).
        :param str version: (optional) optionaly provide an existing version to
               check updates for if one is not given, all version will be returned.
        :param str cluster_id: (optional) The id of the cluster where this version
               was installed.
        :param str region: (optional) The region of the cluster where this version
               was installed.
        :param str resource_group_id: (optional) The resource group id of the
               cluster where this version was installed.
        :param str namespace: (optional) The namespace of the cluster where this
               version was installed.
        :param str sha: (optional) The sha value of the currently installed
               version.
        :param str channel: (optional) Optionally provide the channel value of the
               currently installed version.
        :param List[str] namespaces: (optional) Optionally provide a list of
               namespaces used for the currently installed version.
        :param bool all_namespaces: (optional) Optionally indicate that the current
               version was installed in all namespaces.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `List[VersionUpdateDescriptor]` result
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if offering_id is None:
            raise ValueError('offering_id must be provided')
        if kind is None:
            raise ValueError('kind must be provided')
        if x_auth_refresh_token is None:
            raise ValueError('x_auth_refresh_token must be provided')
        headers = {'X-Auth-Refresh-Token': x_auth_refresh_token}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_offering_updates'
        )
        headers.update(sdk_headers)

        params = {
            'kind': kind,
            'target': target,
            'version': version,
            'cluster_id': cluster_id,
            'region': region,
            'resource_group_id': resource_group_id,
            'namespace': namespace,
            'sha': sha,
            'channel': channel,
            'namespaces': convert_list(namespaces),
            'all_namespaces': all_namespaces,
        }

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier', 'offering_id']
        path_param_values = self.encode_path_vars(catalog_identifier, offering_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/offerings/{offering_id}/updates'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def get_offering_source(
        self,
        version: str,
        *,
        accept: str = None,
        catalog_id: str = None,
        name: str = None,
        id: str = None,
        kind: str = None,
        channel: str = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Get offering source.

        Get an offering's source.  This request requires authorization, even for public
        offerings.

        :param str version: The version being requested.
        :param str accept: (optional) The type of the response: application/yaml,
               application/json, or application/x-gzip.
        :param str catalog_id: (optional) Catlaog ID.  If not specified, this value
               will default to the public catalog.
        :param str name: (optional) Offering name.  An offering name or ID must be
               specified.
        :param str id: (optional) Offering id.  An offering name or ID must be
               specified.
        :param str kind: (optional) The kind of offering (e.g. helm, ova,
               terraform...).
        :param str channel: (optional) The channel value of the specified version.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `BinaryIO` result
        """

        if version is None:
            raise ValueError('version must be provided')
        headers = {'Accept': accept}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_offering_source'
        )
        headers.update(sdk_headers)

        params = {'version': version, 'catalogID': catalog_id, 'name': name, 'id': id, 'kind': kind, 'channel': channel}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        url = '/offering/source'
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    #########################
    # Versions
    #########################

    def get_offering_about(self, version_loc_id: str, **kwargs) -> DetailedResponse:
        """
        Get version about information.

        Get the about information, in markdown, for the current version.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `str` result
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_offering_about'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'text/markdown'

        path_param_keys = ['version_loc_id']
        path_param_values = self.encode_path_vars(version_loc_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}/about'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def get_offering_license(self, version_loc_id: str, license_id: str, **kwargs) -> DetailedResponse:
        """
        Get version license content.

        Get the license content for the specified license ID in the specified version.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param str license_id: The ID of the license, which maps to the file name
               in the 'licenses' directory of this verions tgz file.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `str` result
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        if license_id is None:
            raise ValueError('license_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_offering_license'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'text/plain'

        path_param_keys = ['version_loc_id', 'license_id']
        path_param_values = self.encode_path_vars(version_loc_id, license_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}/licenses/{license_id}'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def get_offering_container_images(self, version_loc_id: str, **kwargs) -> DetailedResponse:
        """
        Get version's container images.

        Get the list of container images associated with the specified version. The
        "image_manifest_url" property of the version should be the URL for the image
        manifest, and the operation will return that content.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ImageManifest` object
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_offering_container_images'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['version_loc_id']
        path_param_values = self.encode_path_vars(version_loc_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}/containerImages'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def deprecate_version(self, version_loc_id: str, **kwargs) -> DetailedResponse:
        """
        Deprecate version immediately.

        Deprecate the specified version.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='deprecate_version'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['version_loc_id']
        path_param_values = self.encode_path_vars(version_loc_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}/deprecate'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def set_deprecate_version(
        self, version_loc_id: str, setting: str, *, description: str = None, days_until_deprecate: int = None, **kwargs
    ) -> DetailedResponse:
        """
        Sets version to be deprecated in a certain time period.

        Set or cancel the version to be deprecated.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param str setting: Set deprecation (true) or cancel deprecation (false).
        :param str description: (optional) Additional information that users can
               provide to be displayed in deprecation notification.
        :param int days_until_deprecate: (optional) Specifies the amount of days
               until product is not available in catalog.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        if setting is None:
            raise ValueError('setting must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='set_deprecate_version'
        )
        headers.update(sdk_headers)

        data = {'description': description, 'days_until_deprecate': days_until_deprecate}
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['version_loc_id', 'setting']
        path_param_values = self.encode_path_vars(version_loc_id, setting)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}/deprecate/{setting}'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def account_publish_version(self, version_loc_id: str, **kwargs) -> DetailedResponse:
        """
        Publish version to account members.

        Publish the specified version so it is viewable by account members.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='account_publish_version'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['version_loc_id']
        path_param_values = self.encode_path_vars(version_loc_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}/account-publish'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def ibm_publish_version(self, version_loc_id: str, **kwargs) -> DetailedResponse:
        """
        Publish version to IBMers in public catalog.

        Publish the specified version so that it is visible to IBMers in the public
        catalog.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='ibm_publish_version'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['version_loc_id']
        path_param_values = self.encode_path_vars(version_loc_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}/ibm-publish'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def public_publish_version(self, version_loc_id: str, **kwargs) -> DetailedResponse:
        """
        Publish version to all users in public catalog.

        Publish the specified version so it is visible to all users in the public catalog.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='public_publish_version'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['version_loc_id']
        path_param_values = self.encode_path_vars(version_loc_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}/public-publish'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def commit_version(self, version_loc_id: str, **kwargs) -> DetailedResponse:
        """
        Commit version.

        Commit a working copy of the specified version.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='commit_version'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['version_loc_id']
        path_param_values = self.encode_path_vars(version_loc_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}/commit'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def copy_version(
        self,
        version_loc_id: str,
        *,
        tags: List[str] = None,
        target_kinds: List[str] = None,
        content: bytes = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Copy version to new target kind.

        Copy the specified version to a new target kind within the same offering.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param List[str] tags: (optional) Tags array.
        :param List[str] target_kinds: (optional) Target kinds.  Current valid
               values are 'iks', 'roks', 'vcenter', and 'terraform'.
        :param bytes content: (optional) byte array representing the content to be
               imported.  Only supported for OVA images at this time.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        if content is not None:
            content = str(base64.b64encode(content), 'utf-8')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='copy_version'
        )
        headers.update(sdk_headers)

        data = {'tags': tags, 'target_kinds': target_kinds, 'content': content}
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['version_loc_id']
        path_param_values = self.encode_path_vars(version_loc_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}/copy'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def get_offering_working_copy(self, version_loc_id: str, **kwargs) -> DetailedResponse:
        """
        Create working copy of version.

        Create a working copy of the specified version.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Version` object
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_offering_working_copy'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['version_loc_id']
        path_param_values = self.encode_path_vars(version_loc_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}/workingcopy'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def get_version(self, version_loc_id: str, **kwargs) -> DetailedResponse:
        """
        Get offering/kind/version 'branch'.

        Get the Offering/Kind/Version 'branch' for the specified locator ID.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Offering` object
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_version'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['version_loc_id']
        path_param_values = self.encode_path_vars(version_loc_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def delete_version(self, version_loc_id: str, **kwargs) -> DetailedResponse:
        """
        Delete version.

        Delete the specified version.  If the version is an active version with a working
        copy, the working copy will be deleted as well.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='delete_version'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['version_loc_id']
        path_param_values = self.encode_path_vars(version_loc_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}'.format(**path_param_dict)
        request = self.prepare_request(method='DELETE', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    #########################
    # Deploy
    #########################

    def get_cluster(self, cluster_id: str, region: str, x_auth_refresh_token: str, **kwargs) -> DetailedResponse:
        """
        Get kubernetes cluster.

        Get the contents of the specified kubernetes cluster.

        :param str cluster_id: ID of the cluster.
        :param str region: Region of the cluster.
        :param str x_auth_refresh_token: IAM Refresh token.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ClusterInfo` object
        """

        if cluster_id is None:
            raise ValueError('cluster_id must be provided')
        if region is None:
            raise ValueError('region must be provided')
        if x_auth_refresh_token is None:
            raise ValueError('x_auth_refresh_token must be provided')
        headers = {'X-Auth-Refresh-Token': x_auth_refresh_token}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_cluster'
        )
        headers.update(sdk_headers)

        params = {'region': region}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['cluster_id']
        path_param_values = self.encode_path_vars(cluster_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/deploy/kubernetes/clusters/{cluster_id}'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def get_namespaces(
        self,
        cluster_id: str,
        region: str,
        x_auth_refresh_token: str,
        *,
        limit: int = None,
        offset: int = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Get cluster namespaces.

        Get the namespaces associated with the specified kubernetes cluster.

        :param str cluster_id: ID of the cluster.
        :param str region: Cluster region.
        :param str x_auth_refresh_token: IAM Refresh token.
        :param int limit: (optional) The maximum number of results to return.
        :param int offset: (optional) The number of results to skip before
               returning values.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `NamespaceSearchResult` object
        """

        if cluster_id is None:
            raise ValueError('cluster_id must be provided')
        if region is None:
            raise ValueError('region must be provided')
        if x_auth_refresh_token is None:
            raise ValueError('x_auth_refresh_token must be provided')
        headers = {'X-Auth-Refresh-Token': x_auth_refresh_token}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_namespaces'
        )
        headers.update(sdk_headers)

        params = {'region': region, 'limit': limit, 'offset': offset}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['cluster_id']
        path_param_values = self.encode_path_vars(cluster_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/deploy/kubernetes/clusters/{cluster_id}/namespaces'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def deploy_operators(
        self,
        x_auth_refresh_token: str,
        *,
        cluster_id: str = None,
        region: str = None,
        namespaces: List[str] = None,
        all_namespaces: bool = None,
        version_locator_id: str = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Deploy operators.

        Deploy operators on a kubernetes cluster.

        :param str x_auth_refresh_token: IAM Refresh token.
        :param str cluster_id: (optional) Cluster ID.
        :param str region: (optional) Cluster region.
        :param List[str] namespaces: (optional) Kube namespaces to deploy
               Operator(s) to.
        :param bool all_namespaces: (optional) Denotes whether to install
               Operator(s) globally.
        :param str version_locator_id: (optional) A dotted value of
               `catalogID`.`versionID`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `List[OperatorDeployResult]` result
        """

        if x_auth_refresh_token is None:
            raise ValueError('x_auth_refresh_token must be provided')
        headers = {'X-Auth-Refresh-Token': x_auth_refresh_token}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='deploy_operators'
        )
        headers.update(sdk_headers)

        data = {
            'cluster_id': cluster_id,
            'region': region,
            'namespaces': namespaces,
            'all_namespaces': all_namespaces,
            'version_locator_id': version_locator_id,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        url = '/deploy/kubernetes/olm/operator'
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def list_operators(
        self, x_auth_refresh_token: str, cluster_id: str, region: str, version_locator_id: str, **kwargs
    ) -> DetailedResponse:
        """
        List operators.

        List the operators from a kubernetes cluster.

        :param str x_auth_refresh_token: IAM Refresh token.
        :param str cluster_id: Cluster identification.
        :param str region: Cluster region.
        :param str version_locator_id: A dotted value of `catalogID`.`versionID`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `List[OperatorDeployResult]` result
        """

        if x_auth_refresh_token is None:
            raise ValueError('x_auth_refresh_token must be provided')
        if cluster_id is None:
            raise ValueError('cluster_id must be provided')
        if region is None:
            raise ValueError('region must be provided')
        if version_locator_id is None:
            raise ValueError('version_locator_id must be provided')
        headers = {'X-Auth-Refresh-Token': x_auth_refresh_token}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='list_operators'
        )
        headers.update(sdk_headers)

        params = {'cluster_id': cluster_id, 'region': region, 'version_locator_id': version_locator_id}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        url = '/deploy/kubernetes/olm/operator'
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def replace_operators(
        self,
        x_auth_refresh_token: str,
        *,
        cluster_id: str = None,
        region: str = None,
        namespaces: List[str] = None,
        all_namespaces: bool = None,
        version_locator_id: str = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Update operators.

        Update the operators on a kubernetes cluster.

        :param str x_auth_refresh_token: IAM Refresh token.
        :param str cluster_id: (optional) Cluster ID.
        :param str region: (optional) Cluster region.
        :param List[str] namespaces: (optional) Kube namespaces to deploy
               Operator(s) to.
        :param bool all_namespaces: (optional) Denotes whether to install
               Operator(s) globally.
        :param str version_locator_id: (optional) A dotted value of
               `catalogID`.`versionID`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `List[OperatorDeployResult]` result
        """

        if x_auth_refresh_token is None:
            raise ValueError('x_auth_refresh_token must be provided')
        headers = {'X-Auth-Refresh-Token': x_auth_refresh_token}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='replace_operators'
        )
        headers.update(sdk_headers)

        data = {
            'cluster_id': cluster_id,
            'region': region,
            'namespaces': namespaces,
            'all_namespaces': all_namespaces,
            'version_locator_id': version_locator_id,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        url = '/deploy/kubernetes/olm/operator'
        request = self.prepare_request(method='PUT', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def delete_operators(
        self, x_auth_refresh_token: str, cluster_id: str, region: str, version_locator_id: str, **kwargs
    ) -> DetailedResponse:
        """
        Delete operators.

        Delete operators from a kubernetes cluster.

        :param str x_auth_refresh_token: IAM Refresh token.
        :param str cluster_id: Cluster identification.
        :param str region: Cluster region.
        :param str version_locator_id: A dotted value of `catalogID`.`versionID`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if x_auth_refresh_token is None:
            raise ValueError('x_auth_refresh_token must be provided')
        if cluster_id is None:
            raise ValueError('cluster_id must be provided')
        if region is None:
            raise ValueError('region must be provided')
        if version_locator_id is None:
            raise ValueError('version_locator_id must be provided')
        headers = {'X-Auth-Refresh-Token': x_auth_refresh_token}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='delete_operators'
        )
        headers.update(sdk_headers)

        params = {'cluster_id': cluster_id, 'region': region, 'version_locator_id': version_locator_id}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        url = '/deploy/kubernetes/olm/operator'
        request = self.prepare_request(method='DELETE', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def install_version(
        self,
        version_loc_id: str,
        x_auth_refresh_token: str,
        *,
        cluster_id: str = None,
        region: str = None,
        namespace: str = None,
        override_values: dict = None,
        entitlement_apikey: str = None,
        schematics: 'DeployRequestBodySchematics' = None,
        script: str = None,
        script_id: str = None,
        version_locator_id: str = None,
        vcenter_id: str = None,
        vcenter_user: str = None,
        vcenter_password: str = None,
        vcenter_location: str = None,
        vcenter_datastore: str = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Install version.

        Create an install for the specified version.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param str x_auth_refresh_token: IAM Refresh token.
        :param str cluster_id: (optional) Cluster ID.
        :param str region: (optional) Cluster region.
        :param str namespace: (optional) Kube namespace.
        :param dict override_values: (optional) Object containing Helm chart
               override values.  To use a secret for items of type password, specify a
               JSON encoded value of $ref:#/components/schemas/SecretInstance, prefixed
               with `cmsm_v1:`.
        :param str entitlement_apikey: (optional) Entitlement API Key for this
               offering.
        :param DeployRequestBodySchematics schematics: (optional) Schematics
               workspace configuration.
        :param str script: (optional) Script.
        :param str script_id: (optional) Script ID.
        :param str version_locator_id: (optional) A dotted value of
               `catalogID`.`versionID`.
        :param str vcenter_id: (optional) VCenter ID.
        :param str vcenter_user: (optional) VCenter User.
        :param str vcenter_password: (optional) VCenter Password.
        :param str vcenter_location: (optional) VCenter Location.
        :param str vcenter_datastore: (optional) VCenter Datastore.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        if x_auth_refresh_token is None:
            raise ValueError('x_auth_refresh_token must be provided')
        if schematics is not None:
            schematics = convert_model(schematics)
        headers = {'X-Auth-Refresh-Token': x_auth_refresh_token}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='install_version'
        )
        headers.update(sdk_headers)

        data = {
            'cluster_id': cluster_id,
            'region': region,
            'namespace': namespace,
            'override_values': override_values,
            'entitlement_apikey': entitlement_apikey,
            'schematics': schematics,
            'script': script,
            'script_id': script_id,
            'version_locator_id': version_locator_id,
            'vcenter_id': vcenter_id,
            'vcenter_user': vcenter_user,
            'vcenter_password': vcenter_password,
            'vcenter_location': vcenter_location,
            'vcenter_datastore': vcenter_datastore,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['version_loc_id']
        path_param_values = self.encode_path_vars(version_loc_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}/install'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def preinstall_version(
        self,
        version_loc_id: str,
        x_auth_refresh_token: str,
        *,
        cluster_id: str = None,
        region: str = None,
        namespace: str = None,
        override_values: dict = None,
        entitlement_apikey: str = None,
        schematics: 'DeployRequestBodySchematics' = None,
        script: str = None,
        script_id: str = None,
        version_locator_id: str = None,
        vcenter_id: str = None,
        vcenter_user: str = None,
        vcenter_password: str = None,
        vcenter_location: str = None,
        vcenter_datastore: str = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Pre-install version.

        Create a pre-install for the specified version.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param str x_auth_refresh_token: IAM Refresh token.
        :param str cluster_id: (optional) Cluster ID.
        :param str region: (optional) Cluster region.
        :param str namespace: (optional) Kube namespace.
        :param dict override_values: (optional) Object containing Helm chart
               override values.  To use a secret for items of type password, specify a
               JSON encoded value of $ref:#/components/schemas/SecretInstance, prefixed
               with `cmsm_v1:`.
        :param str entitlement_apikey: (optional) Entitlement API Key for this
               offering.
        :param DeployRequestBodySchematics schematics: (optional) Schematics
               workspace configuration.
        :param str script: (optional) Script.
        :param str script_id: (optional) Script ID.
        :param str version_locator_id: (optional) A dotted value of
               `catalogID`.`versionID`.
        :param str vcenter_id: (optional) VCenter ID.
        :param str vcenter_user: (optional) VCenter User.
        :param str vcenter_password: (optional) VCenter Password.
        :param str vcenter_location: (optional) VCenter Location.
        :param str vcenter_datastore: (optional) VCenter Datastore.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        if x_auth_refresh_token is None:
            raise ValueError('x_auth_refresh_token must be provided')
        if schematics is not None:
            schematics = convert_model(schematics)
        headers = {'X-Auth-Refresh-Token': x_auth_refresh_token}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='preinstall_version'
        )
        headers.update(sdk_headers)

        data = {
            'cluster_id': cluster_id,
            'region': region,
            'namespace': namespace,
            'override_values': override_values,
            'entitlement_apikey': entitlement_apikey,
            'schematics': schematics,
            'script': script,
            'script_id': script_id,
            'version_locator_id': version_locator_id,
            'vcenter_id': vcenter_id,
            'vcenter_user': vcenter_user,
            'vcenter_password': vcenter_password,
            'vcenter_location': vcenter_location,
            'vcenter_datastore': vcenter_datastore,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['version_loc_id']
        path_param_values = self.encode_path_vars(version_loc_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}/preinstall'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def get_preinstall(
        self,
        version_loc_id: str,
        x_auth_refresh_token: str,
        *,
        cluster_id: str = None,
        region: str = None,
        namespace: str = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Get version pre-install status.

        Get the pre-install status for the specified version.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param str x_auth_refresh_token: IAM Refresh token.
        :param str cluster_id: (optional) ID of the cluster.
        :param str region: (optional) Cluster region.
        :param str namespace: (optional) Required if the version's pre-install
               scope is `namespace`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `InstallStatus` object
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        if x_auth_refresh_token is None:
            raise ValueError('x_auth_refresh_token must be provided')
        headers = {'X-Auth-Refresh-Token': x_auth_refresh_token}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_preinstall'
        )
        headers.update(sdk_headers)

        params = {'cluster_id': cluster_id, 'region': region, 'namespace': namespace}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['version_loc_id']
        path_param_values = self.encode_path_vars(version_loc_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}/preinstall'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def validate_install(
        self,
        version_loc_id: str,
        x_auth_refresh_token: str,
        *,
        cluster_id: str = None,
        region: str = None,
        namespace: str = None,
        override_values: dict = None,
        entitlement_apikey: str = None,
        schematics: 'DeployRequestBodySchematics' = None,
        script: str = None,
        script_id: str = None,
        version_locator_id: str = None,
        vcenter_id: str = None,
        vcenter_user: str = None,
        vcenter_password: str = None,
        vcenter_location: str = None,
        vcenter_datastore: str = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Validate offering.

        Validate the offering associated with the specified version.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param str x_auth_refresh_token: IAM Refresh token.
        :param str cluster_id: (optional) Cluster ID.
        :param str region: (optional) Cluster region.
        :param str namespace: (optional) Kube namespace.
        :param dict override_values: (optional) Object containing Helm chart
               override values.  To use a secret for items of type password, specify a
               JSON encoded value of $ref:#/components/schemas/SecretInstance, prefixed
               with `cmsm_v1:`.
        :param str entitlement_apikey: (optional) Entitlement API Key for this
               offering.
        :param DeployRequestBodySchematics schematics: (optional) Schematics
               workspace configuration.
        :param str script: (optional) Script.
        :param str script_id: (optional) Script ID.
        :param str version_locator_id: (optional) A dotted value of
               `catalogID`.`versionID`.
        :param str vcenter_id: (optional) VCenter ID.
        :param str vcenter_user: (optional) VCenter User.
        :param str vcenter_password: (optional) VCenter Password.
        :param str vcenter_location: (optional) VCenter Location.
        :param str vcenter_datastore: (optional) VCenter Datastore.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        if x_auth_refresh_token is None:
            raise ValueError('x_auth_refresh_token must be provided')
        if schematics is not None:
            schematics = convert_model(schematics)
        headers = {'X-Auth-Refresh-Token': x_auth_refresh_token}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='validate_install'
        )
        headers.update(sdk_headers)

        data = {
            'cluster_id': cluster_id,
            'region': region,
            'namespace': namespace,
            'override_values': override_values,
            'entitlement_apikey': entitlement_apikey,
            'schematics': schematics,
            'script': script,
            'script_id': script_id,
            'version_locator_id': version_locator_id,
            'vcenter_id': vcenter_id,
            'vcenter_user': vcenter_user,
            'vcenter_password': vcenter_password,
            'vcenter_location': vcenter_location,
            'vcenter_datastore': vcenter_datastore,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['version_loc_id']
        path_param_values = self.encode_path_vars(version_loc_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}/validation/install'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def get_validation_status(self, version_loc_id: str, x_auth_refresh_token: str, **kwargs) -> DetailedResponse:
        """
        Get offering install status.

        Returns the install status for the specified offering version.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param str x_auth_refresh_token: IAM Refresh token.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `Validation` object
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        if x_auth_refresh_token is None:
            raise ValueError('x_auth_refresh_token must be provided')
        headers = {'X-Auth-Refresh-Token': x_auth_refresh_token}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_validation_status'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['version_loc_id']
        path_param_values = self.encode_path_vars(version_loc_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}/validation/install'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def get_override_values(self, version_loc_id: str, **kwargs) -> DetailedResponse:
        """
        Get override values.

        Returns the override values that were used to validate the specified offering
        version.

        :param str version_loc_id: A dotted value of `catalogID`.`versionID`.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result
        """

        if version_loc_id is None:
            raise ValueError('version_loc_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_override_values'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['version_loc_id']
        path_param_values = self.encode_path_vars(version_loc_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/versions/{version_loc_id}/validation/overridevalues'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    #########################
    # Objects
    #########################

    def search_objects(
        self, query: str, *, limit: int = None, offset: int = None, collapse: bool = None, digest: bool = None, **kwargs
    ) -> DetailedResponse:
        """
        List objects across catalogs.

        List the available objects from both public and private catalogs. These copies
        cannot be used for updating. They are not complete and only return what is visible
        to the caller.

        :param str query: Lucene query string.
        :param int limit: (optional) The maximum number of results to return.
        :param int offset: (optional) The number of results to skip before
               returning values.
        :param bool collapse: (optional) When true, hide private objects that
               correspond to public or IBM published objects.
        :param bool digest: (optional) Display a digests of search results, has
               default value of true.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ObjectSearchResult` object
        """

        if query is None:
            raise ValueError('query must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='search_objects'
        )
        headers.update(sdk_headers)

        params = {'query': query, 'limit': limit, 'offset': offset, 'collapse': collapse, 'digest': digest}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        url = '/objects'
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def list_objects(
        self,
        catalog_identifier: str,
        *,
        limit: int = None,
        offset: int = None,
        name: str = None,
        sort: str = None,
        **kwargs
    ) -> DetailedResponse:
        """
        List objects within a catalog.

        List the available objects within the specified catalog.

        :param str catalog_identifier: Catalog identifier.
        :param int limit: (optional) The number of results to return.
        :param int offset: (optional) The number of results to skip before
               returning values.
        :param str name: (optional) Only return results that contain the specified
               string.
        :param str sort: (optional) The field on which the output is sorted. Sorts
               by default by **label** property. Available fields are **name**, **label**,
               **created**, and **updated**. By adding **-** (i.e. **-label**) in front of
               the query string, you can specify descending order. Default is ascending
               order.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ObjectListResult` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='list_objects'
        )
        headers.update(sdk_headers)

        params = {'limit': limit, 'offset': offset, 'name': name, 'sort': sort}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/objects'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def create_object(
        self,
        catalog_identifier: str,
        *,
        id: str = None,
        name: str = None,
        rev: str = None,
        crn: str = None,
        url: str = None,
        parent_id: str = None,
        label_i18n: str = None,
        label: str = None,
        tags: List[str] = None,
        created: datetime = None,
        updated: datetime = None,
        short_description: str = None,
        short_description_i18n: str = None,
        kind: str = None,
        publish: 'PublishObject' = None,
        state: 'State' = None,
        catalog_id: str = None,
        catalog_name: str = None,
        data: dict = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Create catalog object.

        Create an object with a specific catalog.

        :param str catalog_identifier: Catalog identifier.
        :param str id: (optional) unique id.
        :param str name: (optional) The programmatic name of this offering.
        :param str rev: (optional) Cloudant revision.
        :param str crn: (optional) The crn for this specific object.
        :param str url: (optional) The url for this specific object.
        :param str parent_id: (optional) The parent for this specific object.
        :param str label_i18n: (optional) Translated display name in the requested
               language.
        :param str label: (optional) Display name in the requested language.
        :param List[str] tags: (optional) List of tags associated with this
               catalog.
        :param datetime created: (optional) The date and time this catalog was
               created.
        :param datetime updated: (optional) The date and time this catalog was last
               updated.
        :param str short_description: (optional) Short description in the requested
               language.
        :param str short_description_i18n: (optional) Short description
               translation.
        :param str kind: (optional) Kind of object.
        :param PublishObject publish: (optional) Publish information.
        :param State state: (optional) Offering state.
        :param str catalog_id: (optional) The id of the catalog containing this
               offering.
        :param str catalog_name: (optional) The name of the catalog.
        :param dict data: (optional) Map of data values for this object.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `CatalogObject` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if created is not None:
            created = datetime_to_string(created)
        if updated is not None:
            updated = datetime_to_string(updated)
        if publish is not None:
            publish = convert_model(publish)
        if state is not None:
            state = convert_model(state)
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='create_object'
        )
        headers.update(sdk_headers)

        data = {
            'id': id,
            'name': name,
            '_rev': rev,
            'crn': crn,
            'url': url,
            'parent_id': parent_id,
            'label_i18n': label_i18n,
            'label': label,
            'tags': tags,
            'created': created,
            'updated': updated,
            'short_description': short_description,
            'short_description_i18n': short_description_i18n,
            'kind': kind,
            'publish': publish,
            'state': state,
            'catalog_id': catalog_id,
            'catalog_name': catalog_name,
            'data': data,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/objects'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def get_object(self, catalog_identifier: str, object_identifier: str, **kwargs) -> DetailedResponse:
        """
        Get catalog object.

        Get the specified object from within the specified catalog.

        :param str catalog_identifier: Catalog identifier.
        :param str object_identifier: Object identifier.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `CatalogObject` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if object_identifier is None:
            raise ValueError('object_identifier must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_object'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier', 'object_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier, object_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/objects/{object_identifier}'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def replace_object(
        self,
        catalog_identifier: str,
        object_identifier: str,
        *,
        id: str = None,
        name: str = None,
        rev: str = None,
        crn: str = None,
        url: str = None,
        parent_id: str = None,
        label_i18n: str = None,
        label: str = None,
        tags: List[str] = None,
        created: datetime = None,
        updated: datetime = None,
        short_description: str = None,
        short_description_i18n: str = None,
        kind: str = None,
        publish: 'PublishObject' = None,
        state: 'State' = None,
        catalog_id: str = None,
        catalog_name: str = None,
        data: dict = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Update catalog object.

        Update an object within a specific catalog.

        :param str catalog_identifier: Catalog identifier.
        :param str object_identifier: Object identifier.
        :param str id: (optional) unique id.
        :param str name: (optional) The programmatic name of this offering.
        :param str rev: (optional) Cloudant revision.
        :param str crn: (optional) The crn for this specific object.
        :param str url: (optional) The url for this specific object.
        :param str parent_id: (optional) The parent for this specific object.
        :param str label_i18n: (optional) Translated display name in the requested
               language.
        :param str label: (optional) Display name in the requested language.
        :param List[str] tags: (optional) List of tags associated with this
               catalog.
        :param datetime created: (optional) The date and time this catalog was
               created.
        :param datetime updated: (optional) The date and time this catalog was last
               updated.
        :param str short_description: (optional) Short description in the requested
               language.
        :param str short_description_i18n: (optional) Short description
               translation.
        :param str kind: (optional) Kind of object.
        :param PublishObject publish: (optional) Publish information.
        :param State state: (optional) Offering state.
        :param str catalog_id: (optional) The id of the catalog containing this
               offering.
        :param str catalog_name: (optional) The name of the catalog.
        :param dict data: (optional) Map of data values for this object.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `CatalogObject` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if object_identifier is None:
            raise ValueError('object_identifier must be provided')
        if created is not None:
            created = datetime_to_string(created)
        if updated is not None:
            updated = datetime_to_string(updated)
        if publish is not None:
            publish = convert_model(publish)
        if state is not None:
            state = convert_model(state)
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='replace_object'
        )
        headers.update(sdk_headers)

        data = {
            'id': id,
            'name': name,
            '_rev': rev,
            'crn': crn,
            'url': url,
            'parent_id': parent_id,
            'label_i18n': label_i18n,
            'label': label,
            'tags': tags,
            'created': created,
            'updated': updated,
            'short_description': short_description,
            'short_description_i18n': short_description_i18n,
            'kind': kind,
            'publish': publish,
            'state': state,
            'catalog_id': catalog_id,
            'catalog_name': catalog_name,
            'data': data,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier', 'object_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier, object_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/objects/{object_identifier}'.format(**path_param_dict)
        request = self.prepare_request(method='PUT', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def delete_object(self, catalog_identifier: str, object_identifier: str, **kwargs) -> DetailedResponse:
        """
        Delete catalog object.

        Delete a specific object within a specific catalog.

        :param str catalog_identifier: Catalog identifier.
        :param str object_identifier: Object identifier.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if object_identifier is None:
            raise ValueError('object_identifier must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='delete_object'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['catalog_identifier', 'object_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier, object_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/objects/{object_identifier}'.format(**path_param_dict)
        request = self.prepare_request(method='DELETE', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def get_object_audit(self, catalog_identifier: str, object_identifier: str, **kwargs) -> DetailedResponse:
        """
        Get catalog object audit log.

        Get the audit log associated with a specific catalog object.

        :param str catalog_identifier: Catalog identifier.
        :param str object_identifier: Object identifier.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AuditLog` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if object_identifier is None:
            raise ValueError('object_identifier must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_object_audit'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier', 'object_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier, object_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/objects/{object_identifier}/audit'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def account_publish_object(self, catalog_identifier: str, object_identifier: str, **kwargs) -> DetailedResponse:
        """
        Publish object to account.

        Publish a catalog object to account.

        :param str catalog_identifier: Catalog identifier.
        :param str object_identifier: Object identifier.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if object_identifier is None:
            raise ValueError('object_identifier must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='account_publish_object'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['catalog_identifier', 'object_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier, object_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/objects/{object_identifier}/account-publish'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def shared_publish_object(self, catalog_identifier: str, object_identifier: str, **kwargs) -> DetailedResponse:
        """
        Publish object to share with allow list.

        Publish the specified object so that it is visible to those in the allow list.

        :param str catalog_identifier: Catalog identifier.
        :param str object_identifier: Object identifier.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if object_identifier is None:
            raise ValueError('object_identifier must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='shared_publish_object'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['catalog_identifier', 'object_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier, object_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/objects/{object_identifier}/shared-publish'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def ibm_publish_object(self, catalog_identifier: str, object_identifier: str, **kwargs) -> DetailedResponse:
        """
        Publish object to share with IBMers.

        Publish the specified object so that it is visible to IBMers in the public
        catalog.

        :param str catalog_identifier: Catalog identifier.
        :param str object_identifier: Object identifier.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if object_identifier is None:
            raise ValueError('object_identifier must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='ibm_publish_object'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['catalog_identifier', 'object_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier, object_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/objects/{object_identifier}/ibm-publish'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def public_publish_object(self, catalog_identifier: str, object_identifier: str, **kwargs) -> DetailedResponse:
        """
        Publish object to share with all users.

        Publish the specified object so it is visible to all users in the public catalog.

        :param str catalog_identifier: Catalog identifier.
        :param str object_identifier: Object identifier.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if object_identifier is None:
            raise ValueError('object_identifier must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='public_publish_object'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['catalog_identifier', 'object_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier, object_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/objects/{object_identifier}/public-publish'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def create_object_access(
        self, catalog_identifier: str, object_identifier: str, account_identifier: str, **kwargs
    ) -> DetailedResponse:
        """
        Add account ID to object access list.

        Add an account ID to an object's access list.

        :param str catalog_identifier: Catalog identifier.
        :param str object_identifier: Object identifier.
        :param str account_identifier: Account identifier.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if object_identifier is None:
            raise ValueError('object_identifier must be provided')
        if account_identifier is None:
            raise ValueError('account_identifier must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='create_object_access'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['catalog_identifier', 'object_identifier', 'account_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier, object_identifier, account_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/objects/{object_identifier}/access/{account_identifier}'.format(
            **path_param_dict
        )
        request = self.prepare_request(method='POST', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def get_object_access(
        self, catalog_identifier: str, object_identifier: str, account_identifier: str, **kwargs
    ) -> DetailedResponse:
        """
        Check for account ID in object access list.

        Determine if an account ID is in an object's access list.

        :param str catalog_identifier: Catalog identifier.
        :param str object_identifier: Object identifier.
        :param str account_identifier: Account identifier.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ObjectAccess` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if object_identifier is None:
            raise ValueError('object_identifier must be provided')
        if account_identifier is None:
            raise ValueError('account_identifier must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_object_access'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier', 'object_identifier', 'account_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier, object_identifier, account_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/objects/{object_identifier}/access/{account_identifier}'.format(
            **path_param_dict
        )
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def delete_object_access(
        self, catalog_identifier: str, object_identifier: str, account_identifier: str, **kwargs
    ) -> DetailedResponse:
        """
        Remove account ID from object access list.

        Delete the specified account ID from the specified object's access list.

        :param str catalog_identifier: Catalog identifier.
        :param str object_identifier: Object identifier.
        :param str account_identifier: Account identifier.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if object_identifier is None:
            raise ValueError('object_identifier must be provided')
        if account_identifier is None:
            raise ValueError('account_identifier must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='delete_object_access'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['catalog_identifier', 'object_identifier', 'account_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier, object_identifier, account_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/objects/{object_identifier}/access/{account_identifier}'.format(
            **path_param_dict
        )
        request = self.prepare_request(method='DELETE', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def get_object_access_list(
        self, catalog_identifier: str, object_identifier: str, *, limit: int = None, offset: int = None, **kwargs
    ) -> DetailedResponse:
        """
        Get object access list.

        Get the access list associated with the specified object.

        :param str catalog_identifier: Catalog identifier.
        :param str object_identifier: Object identifier.
        :param int limit: (optional) The maximum number of results to return.
        :param int offset: (optional) The number of results to skip before
               returning values.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `ObjectAccessListResult` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if object_identifier is None:
            raise ValueError('object_identifier must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_object_access_list'
        )
        headers.update(sdk_headers)

        params = {'limit': limit, 'offset': offset}

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier', 'object_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier, object_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/objects/{object_identifier}/access'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers, params=params)

        response = self.send(request, **kwargs)
        return response

    def delete_object_access_list(
        self, catalog_identifier: str, object_identifier: str, accounts: List[str], **kwargs
    ) -> DetailedResponse:
        """
        Delete accounts from object access list.

        Delete all or a set of accounts from an object's access list.

        :param str catalog_identifier: Catalog identifier.
        :param str object_identifier: Object identifier.
        :param List[str] accounts: A list of accounts to delete.  An entry with
               star["*"] will remove all accounts.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccessListBulkResponse` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if object_identifier is None:
            raise ValueError('object_identifier must be provided')
        if accounts is None:
            raise ValueError('accounts must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='delete_object_access_list'
        )
        headers.update(sdk_headers)

        data = json.dumps(accounts)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier', 'object_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier, object_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/objects/{object_identifier}/access'.format(**path_param_dict)
        request = self.prepare_request(method='DELETE', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def add_object_access_list(
        self, catalog_identifier: str, object_identifier: str, accounts: List[str], **kwargs
    ) -> DetailedResponse:
        """
        Add accounts to object access list.

        Add one or more accounts to the specified object's access list.

        :param str catalog_identifier: Catalog identifier.
        :param str object_identifier: Object identifier.
        :param List[str] accounts: A list of accounts to add.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccessListBulkResponse` object
        """

        if catalog_identifier is None:
            raise ValueError('catalog_identifier must be provided')
        if object_identifier is None:
            raise ValueError('object_identifier must be provided')
        if accounts is None:
            raise ValueError('accounts must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='add_object_access_list'
        )
        headers.update(sdk_headers)

        data = json.dumps(accounts)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['catalog_identifier', 'object_identifier']
        path_param_values = self.encode_path_vars(catalog_identifier, object_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/catalogs/{catalog_identifier}/objects/{object_identifier}/access'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    #########################
    # Instances
    #########################

    def create_offering_instance(
        self,
        x_auth_refresh_token: str,
        *,
        id: str = None,
        rev: str = None,
        url: str = None,
        crn: str = None,
        label: str = None,
        catalog_id: str = None,
        offering_id: str = None,
        kind_format: str = None,
        version: str = None,
        cluster_id: str = None,
        cluster_region: str = None,
        cluster_namespaces: List[str] = None,
        cluster_all_namespaces: bool = None,
        schematics_workspace_id: str = None,
        resource_group_id: str = None,
        install_plan: str = None,
        channel: str = None,
        metadata: dict = None,
        last_operation: 'OfferingInstanceLastOperation' = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Create an offering resource instance.

        Provision a new offering in a given account, and return its resource instance.

        :param str x_auth_refresh_token: IAM Refresh token.
        :param str id: (optional) provisioned instance ID (part of the CRN).
        :param str rev: (optional) Cloudant revision.
        :param str url: (optional) url reference to this object.
        :param str crn: (optional) platform CRN for this instance.
        :param str label: (optional) the label for this instance.
        :param str catalog_id: (optional) Catalog ID this instance was created
               from.
        :param str offering_id: (optional) Offering ID this instance was created
               from.
        :param str kind_format: (optional) the format this instance has (helm,
               operator, ova...).
        :param str version: (optional) The version this instance was installed from
               (not version id).
        :param str cluster_id: (optional) Cluster ID.
        :param str cluster_region: (optional) Cluster region (e.g., us-south).
        :param List[str] cluster_namespaces: (optional) List of target namespaces
               to install into.
        :param bool cluster_all_namespaces: (optional) designate to install into
               all namespaces.
        :param str schematics_workspace_id: (optional) Id of the schematics
               workspace, for offering instances provisioned through schematics.
        :param str resource_group_id: (optional) Id of the resource group to
               provision the offering instance into.
        :param str install_plan: (optional) Type of install plan (also known as
               approval strategy) for operator subscriptions. Can be either automatic,
               which automatically upgrades operators to the latest in a channel, or
               manual, which requires approval on the cluster.
        :param str channel: (optional) Channel to pin the operator subscription to.
        :param dict metadata: (optional) Map of metadata values for this offering
               instance.
        :param OfferingInstanceLastOperation last_operation: (optional) the last
               operation performed and status.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `OfferingInstance` object
        """

        if x_auth_refresh_token is None:
            raise ValueError('x_auth_refresh_token must be provided')
        if last_operation is not None:
            last_operation = convert_model(last_operation)
        headers = {'X-Auth-Refresh-Token': x_auth_refresh_token}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='create_offering_instance'
        )
        headers.update(sdk_headers)

        data = {
            'id': id,
            '_rev': rev,
            'url': url,
            'crn': crn,
            'label': label,
            'catalog_id': catalog_id,
            'offering_id': offering_id,
            'kind_format': kind_format,
            'version': version,
            'cluster_id': cluster_id,
            'cluster_region': cluster_region,
            'cluster_namespaces': cluster_namespaces,
            'cluster_all_namespaces': cluster_all_namespaces,
            'schematics_workspace_id': schematics_workspace_id,
            'resource_group_id': resource_group_id,
            'install_plan': install_plan,
            'channel': channel,
            'metadata': metadata,
            'last_operation': last_operation,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        url = '/instances/offerings'
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def get_offering_instance(self, instance_identifier: str, **kwargs) -> DetailedResponse:
        """
        Get Offering Instance.

        Get the resource associated with an installed offering instance.

        :param str instance_identifier: Version Instance identifier.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `OfferingInstance` object
        """

        if instance_identifier is None:
            raise ValueError('instance_identifier must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_offering_instance'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['instance_identifier']
        path_param_values = self.encode_path_vars(instance_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/instances/offerings/{instance_identifier}'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response

    def put_offering_instance(
        self,
        instance_identifier: str,
        x_auth_refresh_token: str,
        *,
        id: str = None,
        rev: str = None,
        url: str = None,
        crn: str = None,
        label: str = None,
        catalog_id: str = None,
        offering_id: str = None,
        kind_format: str = None,
        version: str = None,
        cluster_id: str = None,
        cluster_region: str = None,
        cluster_namespaces: List[str] = None,
        cluster_all_namespaces: bool = None,
        schematics_workspace_id: str = None,
        resource_group_id: str = None,
        install_plan: str = None,
        channel: str = None,
        metadata: dict = None,
        last_operation: 'OfferingInstanceLastOperation' = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Update Offering Instance.

        Update an installed offering instance.

        :param str instance_identifier: Version Instance identifier.
        :param str x_auth_refresh_token: IAM Refresh token.
        :param str id: (optional) provisioned instance ID (part of the CRN).
        :param str rev: (optional) Cloudant revision.
        :param str url: (optional) url reference to this object.
        :param str crn: (optional) platform CRN for this instance.
        :param str label: (optional) the label for this instance.
        :param str catalog_id: (optional) Catalog ID this instance was created
               from.
        :param str offering_id: (optional) Offering ID this instance was created
               from.
        :param str kind_format: (optional) the format this instance has (helm,
               operator, ova...).
        :param str version: (optional) The version this instance was installed from
               (not version id).
        :param str cluster_id: (optional) Cluster ID.
        :param str cluster_region: (optional) Cluster region (e.g., us-south).
        :param List[str] cluster_namespaces: (optional) List of target namespaces
               to install into.
        :param bool cluster_all_namespaces: (optional) designate to install into
               all namespaces.
        :param str schematics_workspace_id: (optional) Id of the schematics
               workspace, for offering instances provisioned through schematics.
        :param str resource_group_id: (optional) Id of the resource group to
               provision the offering instance into.
        :param str install_plan: (optional) Type of install plan (also known as
               approval strategy) for operator subscriptions. Can be either automatic,
               which automatically upgrades operators to the latest in a channel, or
               manual, which requires approval on the cluster.
        :param str channel: (optional) Channel to pin the operator subscription to.
        :param dict metadata: (optional) Map of metadata values for this offering
               instance.
        :param OfferingInstanceLastOperation last_operation: (optional) the last
               operation performed and status.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `OfferingInstance` object
        """

        if instance_identifier is None:
            raise ValueError('instance_identifier must be provided')
        if x_auth_refresh_token is None:
            raise ValueError('x_auth_refresh_token must be provided')
        if last_operation is not None:
            last_operation = convert_model(last_operation)
        headers = {'X-Auth-Refresh-Token': x_auth_refresh_token}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='put_offering_instance'
        )
        headers.update(sdk_headers)

        data = {
            'id': id,
            '_rev': rev,
            'url': url,
            'crn': crn,
            'label': label,
            'catalog_id': catalog_id,
            'offering_id': offering_id,
            'kind_format': kind_format,
            'version': version,
            'cluster_id': cluster_id,
            'cluster_region': cluster_region,
            'cluster_namespaces': cluster_namespaces,
            'cluster_all_namespaces': cluster_all_namespaces,
            'schematics_workspace_id': schematics_workspace_id,
            'resource_group_id': resource_group_id,
            'install_plan': install_plan,
            'channel': channel,
            'metadata': metadata,
            'last_operation': last_operation,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['instance_identifier']
        path_param_values = self.encode_path_vars(instance_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/instances/offerings/{instance_identifier}'.format(**path_param_dict)
        request = self.prepare_request(method='PUT', url=url, headers=headers, data=data)

        response = self.send(request, **kwargs)
        return response

    def delete_offering_instance(
        self, instance_identifier: str, x_auth_refresh_token: str, **kwargs
    ) -> DetailedResponse:
        """
        Delete a version instance.

        Delete and instance deployed out of a product version.

        :param str instance_identifier: Version Instance identifier.
        :param str x_auth_refresh_token: IAM Refresh token.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse
        """

        if instance_identifier is None:
            raise ValueError('instance_identifier must be provided')
        if x_auth_refresh_token is None:
            raise ValueError('x_auth_refresh_token must be provided')
        headers = {'X-Auth-Refresh-Token': x_auth_refresh_token}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='delete_offering_instance'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))

        path_param_keys = ['instance_identifier']
        path_param_values = self.encode_path_vars(instance_identifier)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/instances/offerings/{instance_identifier}'.format(**path_param_dict)
        request = self.prepare_request(method='DELETE', url=url, headers=headers)

        response = self.send(request, **kwargs)
        return response


class GetConsumptionOfferingsEnums:
    """
    Enums for get_consumption_offerings parameters.
    """

    class Select(str, Enum):
        """
        What should be selected. Default is 'all' which will return both public and
        private offerings. 'public' returns only the public offerings and 'private'
        returns only the private offerings.
        """

        ALL = 'all'
        PUBLIC = 'public'
        PRIVATE = 'private'


class UpdateOfferingIbmEnums:
    """
    Enums for update_offering_ibm parameters.
    """

    class ApprovalType(str, Enum):
        """
        Type of approval, ibm or public.
        """

        PC_MANAGED = 'pc_managed'
        ALLOW_REQUEST = 'allow_request'
        IBM = 'ibm'
        PUBLIC = 'public'

    class Approved(str, Enum):
        """
        Approve (true) or disapprove (false).
        """

        TRUE = 'true'
        FALSE = 'false'


class DeprecateOfferingEnums:
    """
    Enums for deprecate_offering parameters.
    """

    class Setting(str, Enum):
        """
        Set deprecation (true) or cancel deprecation (false).
        """

        TRUE = 'true'
        FALSE = 'false'


class GetOfferingSourceEnums:
    """
    Enums for get_offering_source parameters.
    """

    class Accept(str, Enum):
        """
        The type of the response: application/yaml, application/json, or
        application/x-gzip.
        """

        APPLICATION_YAML = 'application/yaml'
        APPLICATION_JSON = 'application/json'
        APPLICATION_X_GZIP = 'application/x-gzip'


class SetDeprecateVersionEnums:
    """
    Enums for set_deprecate_version parameters.
    """

    class Setting(str, Enum):
        """
        Set deprecation (true) or cancel deprecation (false).
        """

        TRUE = 'true'
        FALSE = 'false'


##############################################################################
# Models
##############################################################################


class AccessListBulkResponse:
    """
    Access List Add/Remove result.

    :attr dict errors: (optional) in the case of error on an account add/remove -
          account: error.
    """

    def __init__(self, *, errors: dict = None) -> None:
        """
        Initialize a AccessListBulkResponse object.

        :param dict errors: (optional) in the case of error on an account
               add/remove - account: error.
        """
        self.errors = errors

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccessListBulkResponse':
        """Initialize a AccessListBulkResponse object from a json dictionary."""
        args = {}
        if 'errors' in _dict:
            args['errors'] = _dict.get('errors')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccessListBulkResponse object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'errors') and self.errors is not None:
            _dict['errors'] = self.errors
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AccessListBulkResponse object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccessListBulkResponse') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccessListBulkResponse') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Account:
    """
    Account information.

    :attr str id: (optional) Account identification.
    :attr bool hide_ibm_cloud_catalog: (optional) Hide the public catalog in this
          account.
    :attr Filters account_filters: (optional) Filters for account and catalog
          filters.
    """

    def __init__(
        self, *, id: str = None, hide_ibm_cloud_catalog: bool = None, account_filters: 'Filters' = None
    ) -> None:
        """
        Initialize a Account object.

        :param str id: (optional) Account identification.
        :param bool hide_ibm_cloud_catalog: (optional) Hide the public catalog in
               this account.
        :param Filters account_filters: (optional) Filters for account and catalog
               filters.
        """
        self.id = id
        self.hide_ibm_cloud_catalog = hide_ibm_cloud_catalog
        self.account_filters = account_filters

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Account':
        """Initialize a Account object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'hide_IBM_cloud_catalog' in _dict:
            args['hide_ibm_cloud_catalog'] = _dict.get('hide_IBM_cloud_catalog')
        if 'account_filters' in _dict:
            args['account_filters'] = Filters.from_dict(_dict.get('account_filters'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Account object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'hide_ibm_cloud_catalog') and self.hide_ibm_cloud_catalog is not None:
            _dict['hide_IBM_cloud_catalog'] = self.hide_ibm_cloud_catalog
        if hasattr(self, 'account_filters') and self.account_filters is not None:
            _dict['account_filters'] = self.account_filters.to_dict()
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


class AccumulatedFilters:
    """
    The accumulated filters for an account. This will return the account filters plus a
    filter for each catalog the user has access to.

    :attr List[Filters] account_filters: (optional) Filters for accounts (at this
          time this will always be just one item array).
    :attr List[AccumulatedFiltersCatalogFiltersItem] catalog_filters: (optional) The
          filters for all of the accessible catalogs.
    """

    def __init__(
        self,
        *,
        account_filters: List['Filters'] = None,
        catalog_filters: List['AccumulatedFiltersCatalogFiltersItem'] = None
    ) -> None:
        """
        Initialize a AccumulatedFilters object.

        :param List[Filters] account_filters: (optional) Filters for accounts (at
               this time this will always be just one item array).
        :param List[AccumulatedFiltersCatalogFiltersItem] catalog_filters:
               (optional) The filters for all of the accessible catalogs.
        """
        self.account_filters = account_filters
        self.catalog_filters = catalog_filters

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccumulatedFilters':
        """Initialize a AccumulatedFilters object from a json dictionary."""
        args = {}
        if 'account_filters' in _dict:
            args['account_filters'] = [Filters.from_dict(x) for x in _dict.get('account_filters')]
        if 'catalog_filters' in _dict:
            args['catalog_filters'] = [
                AccumulatedFiltersCatalogFiltersItem.from_dict(x) for x in _dict.get('catalog_filters')
            ]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccumulatedFilters object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'account_filters') and self.account_filters is not None:
            _dict['account_filters'] = [x.to_dict() for x in self.account_filters]
        if hasattr(self, 'catalog_filters') and self.catalog_filters is not None:
            _dict['catalog_filters'] = [x.to_dict() for x in self.catalog_filters]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AccumulatedFilters object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccumulatedFilters') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccumulatedFilters') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AccumulatedFiltersCatalogFiltersItem:
    """
    AccumulatedFiltersCatalogFiltersItem.

    :attr AccumulatedFiltersCatalogFiltersItemCatalog catalog: (optional) Filters
          for catalog.
    :attr Filters filters: (optional) Filters for account and catalog filters.
    """

    def __init__(
        self, *, catalog: 'AccumulatedFiltersCatalogFiltersItemCatalog' = None, filters: 'Filters' = None
    ) -> None:
        """
        Initialize a AccumulatedFiltersCatalogFiltersItem object.

        :param AccumulatedFiltersCatalogFiltersItemCatalog catalog: (optional)
               Filters for catalog.
        :param Filters filters: (optional) Filters for account and catalog filters.
        """
        self.catalog = catalog
        self.filters = filters

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccumulatedFiltersCatalogFiltersItem':
        """Initialize a AccumulatedFiltersCatalogFiltersItem object from a json dictionary."""
        args = {}
        if 'catalog' in _dict:
            args['catalog'] = AccumulatedFiltersCatalogFiltersItemCatalog.from_dict(_dict.get('catalog'))
        if 'filters' in _dict:
            args['filters'] = Filters.from_dict(_dict.get('filters'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccumulatedFiltersCatalogFiltersItem object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'catalog') and self.catalog is not None:
            _dict['catalog'] = self.catalog.to_dict()
        if hasattr(self, 'filters') and self.filters is not None:
            _dict['filters'] = self.filters.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AccumulatedFiltersCatalogFiltersItem object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccumulatedFiltersCatalogFiltersItem') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccumulatedFiltersCatalogFiltersItem') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AccumulatedFiltersCatalogFiltersItemCatalog:
    """
    Filters for catalog.

    :attr str id: (optional) The ID of the catalog.
    :attr str name: (optional) The name of the catalog.
    """

    def __init__(self, *, id: str = None, name: str = None) -> None:
        """
        Initialize a AccumulatedFiltersCatalogFiltersItemCatalog object.

        :param str id: (optional) The ID of the catalog.
        :param str name: (optional) The name of the catalog.
        """
        self.id = id
        self.name = name

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccumulatedFiltersCatalogFiltersItemCatalog':
        """Initialize a AccumulatedFiltersCatalogFiltersItemCatalog object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccumulatedFiltersCatalogFiltersItemCatalog object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AccumulatedFiltersCatalogFiltersItemCatalog object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccumulatedFiltersCatalogFiltersItemCatalog') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccumulatedFiltersCatalogFiltersItemCatalog') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ApprovalResult:
    """
    Result of approval.

    :attr bool allow_request: (optional) Allowed to request to publish.
    :attr bool ibm: (optional) Visible to IBM.
    :attr bool public: (optional) Visible to everyone.
    :attr bool changed: (optional) Denotes whether approval has changed.
    """

    def __init__(
        self, *, allow_request: bool = None, ibm: bool = None, public: bool = None, changed: bool = None
    ) -> None:
        """
        Initialize a ApprovalResult object.

        :param bool allow_request: (optional) Allowed to request to publish.
        :param bool ibm: (optional) Visible to IBM.
        :param bool public: (optional) Visible to everyone.
        :param bool changed: (optional) Denotes whether approval has changed.
        """
        self.allow_request = allow_request
        self.ibm = ibm
        self.public = public
        self.changed = changed

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ApprovalResult':
        """Initialize a ApprovalResult object from a json dictionary."""
        args = {}
        if 'allow_request' in _dict:
            args['allow_request'] = _dict.get('allow_request')
        if 'ibm' in _dict:
            args['ibm'] = _dict.get('ibm')
        if 'public' in _dict:
            args['public'] = _dict.get('public')
        if 'changed' in _dict:
            args['changed'] = _dict.get('changed')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ApprovalResult object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'allow_request') and self.allow_request is not None:
            _dict['allow_request'] = self.allow_request
        if hasattr(self, 'ibm') and self.ibm is not None:
            _dict['ibm'] = self.ibm
        if hasattr(self, 'public') and self.public is not None:
            _dict['public'] = self.public
        if hasattr(self, 'changed') and self.changed is not None:
            _dict['changed'] = self.changed
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ApprovalResult object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ApprovalResult') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ApprovalResult') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AuditLog:
    """
    A collection of audit records.

    :attr List[AuditRecord] list: (optional) A list of audit records.
    """

    def __init__(self, *, list: List['AuditRecord'] = None) -> None:
        """
        Initialize a AuditLog object.

        :param List[AuditRecord] list: (optional) A list of audit records.
        """
        self.list = list

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AuditLog':
        """Initialize a AuditLog object from a json dictionary."""
        args = {}
        if 'list' in _dict:
            args['list'] = [AuditRecord.from_dict(x) for x in _dict.get('list')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AuditLog object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'list') and self.list is not None:
            _dict['list'] = [x.to_dict() for x in self.list]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AuditLog object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AuditLog') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AuditLog') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class AuditRecord:
    """
    An audit record which describes a change made to a catalog or associated resource.

    :attr str id: (optional) The identifier of the audit record.
    :attr datetime created: (optional) The time at which the change was made.
    :attr str change_type: (optional) The type of change described by the audit
          record.
    :attr str target_type: (optional) The resource type associated with the change.
    :attr str target_id: (optional) The identifier of the resource that was changed.
    :attr str who_delegate_email: (optional) The email address of the user that made
          the change.
    :attr str message: (optional) A message which describes the change.
    """

    def __init__(
        self,
        *,
        id: str = None,
        created: datetime = None,
        change_type: str = None,
        target_type: str = None,
        target_id: str = None,
        who_delegate_email: str = None,
        message: str = None
    ) -> None:
        """
        Initialize a AuditRecord object.

        :param str id: (optional) The identifier of the audit record.
        :param datetime created: (optional) The time at which the change was made.
        :param str change_type: (optional) The type of change described by the
               audit record.
        :param str target_type: (optional) The resource type associated with the
               change.
        :param str target_id: (optional) The identifier of the resource that was
               changed.
        :param str who_delegate_email: (optional) The email address of the user
               that made the change.
        :param str message: (optional) A message which describes the change.
        """
        self.id = id
        self.created = created
        self.change_type = change_type
        self.target_type = target_type
        self.target_id = target_id
        self.who_delegate_email = who_delegate_email
        self.message = message

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AuditRecord':
        """Initialize a AuditRecord object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'created' in _dict:
            args['created'] = string_to_datetime(_dict.get('created'))
        if 'change_type' in _dict:
            args['change_type'] = _dict.get('change_type')
        if 'target_type' in _dict:
            args['target_type'] = _dict.get('target_type')
        if 'target_id' in _dict:
            args['target_id'] = _dict.get('target_id')
        if 'who_delegate_email' in _dict:
            args['who_delegate_email'] = _dict.get('who_delegate_email')
        if 'message' in _dict:
            args['message'] = _dict.get('message')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AuditRecord object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'created') and self.created is not None:
            _dict['created'] = datetime_to_string(self.created)
        if hasattr(self, 'change_type') and self.change_type is not None:
            _dict['change_type'] = self.change_type
        if hasattr(self, 'target_type') and self.target_type is not None:
            _dict['target_type'] = self.target_type
        if hasattr(self, 'target_id') and self.target_id is not None:
            _dict['target_id'] = self.target_id
        if hasattr(self, 'who_delegate_email') and self.who_delegate_email is not None:
            _dict['who_delegate_email'] = self.who_delegate_email
        if hasattr(self, 'message') and self.message is not None:
            _dict['message'] = self.message
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AuditRecord object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AuditRecord') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AuditRecord') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Catalog:
    """
    Catalog information.

    :attr str id: (optional) Unique ID.
    :attr str rev: (optional) Cloudant revision.
    :attr str label: (optional) Display Name in the requested language.
    :attr str short_description: (optional) Description in the requested language.
    :attr str catalog_icon_url: (optional) URL for an icon associated with this
          catalog.
    :attr List[str] tags: (optional) List of tags associated with this catalog.
    :attr str url: (optional) The url for this specific catalog.
    :attr str crn: (optional) CRN associated with the catalog.
    :attr str offerings_url: (optional) URL path to offerings.
    :attr List[Feature] features: (optional) List of features associated with this
          catalog.
    :attr bool disabled: (optional) Denotes whether a catalog is disabled.
    :attr datetime created: (optional) The date-time this catalog was created.
    :attr datetime updated: (optional) The date-time this catalog was last updated.
    :attr str resource_group_id: (optional) Resource group id the catalog is owned
          by.
    :attr str owning_account: (optional) Account that owns catalog.
    :attr Filters catalog_filters: (optional) Filters for account and catalog
          filters.
    :attr SyndicationResource syndication_settings: (optional) Feature information.
    :attr str kind: (optional) Kind of catalog. Supported kinds are offering and
          vpe.
    """

    def __init__(
        self,
        *,
        id: str = None,
        rev: str = None,
        label: str = None,
        short_description: str = None,
        catalog_icon_url: str = None,
        tags: List[str] = None,
        url: str = None,
        crn: str = None,
        offerings_url: str = None,
        features: List['Feature'] = None,
        disabled: bool = None,
        created: datetime = None,
        updated: datetime = None,
        resource_group_id: str = None,
        owning_account: str = None,
        catalog_filters: 'Filters' = None,
        syndication_settings: 'SyndicationResource' = None,
        kind: str = None
    ) -> None:
        """
        Initialize a Catalog object.

        :param str id: (optional) Unique ID.
        :param str rev: (optional) Cloudant revision.
        :param str label: (optional) Display Name in the requested language.
        :param str short_description: (optional) Description in the requested
               language.
        :param str catalog_icon_url: (optional) URL for an icon associated with
               this catalog.
        :param List[str] tags: (optional) List of tags associated with this
               catalog.
        :param List[Feature] features: (optional) List of features associated with
               this catalog.
        :param bool disabled: (optional) Denotes whether a catalog is disabled.
        :param str resource_group_id: (optional) Resource group id the catalog is
               owned by.
        :param str owning_account: (optional) Account that owns catalog.
        :param Filters catalog_filters: (optional) Filters for account and catalog
               filters.
        :param SyndicationResource syndication_settings: (optional) Feature
               information.
        :param str kind: (optional) Kind of catalog. Supported kinds are offering
               and vpe.
        """
        self.id = id
        self.rev = rev
        self.label = label
        self.short_description = short_description
        self.catalog_icon_url = catalog_icon_url
        self.tags = tags
        self.url = url
        self.crn = crn
        self.offerings_url = offerings_url
        self.features = features
        self.disabled = disabled
        self.created = created
        self.updated = updated
        self.resource_group_id = resource_group_id
        self.owning_account = owning_account
        self.catalog_filters = catalog_filters
        self.syndication_settings = syndication_settings
        self.kind = kind

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Catalog':
        """Initialize a Catalog object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if '_rev' in _dict:
            args['rev'] = _dict.get('_rev')
        if 'label' in _dict:
            args['label'] = _dict.get('label')
        if 'short_description' in _dict:
            args['short_description'] = _dict.get('short_description')
        if 'catalog_icon_url' in _dict:
            args['catalog_icon_url'] = _dict.get('catalog_icon_url')
        if 'tags' in _dict:
            args['tags'] = _dict.get('tags')
        if 'url' in _dict:
            args['url'] = _dict.get('url')
        if 'crn' in _dict:
            args['crn'] = _dict.get('crn')
        if 'offerings_url' in _dict:
            args['offerings_url'] = _dict.get('offerings_url')
        if 'features' in _dict:
            args['features'] = [Feature.from_dict(x) for x in _dict.get('features')]
        if 'disabled' in _dict:
            args['disabled'] = _dict.get('disabled')
        if 'created' in _dict:
            args['created'] = string_to_datetime(_dict.get('created'))
        if 'updated' in _dict:
            args['updated'] = string_to_datetime(_dict.get('updated'))
        if 'resource_group_id' in _dict:
            args['resource_group_id'] = _dict.get('resource_group_id')
        if 'owning_account' in _dict:
            args['owning_account'] = _dict.get('owning_account')
        if 'catalog_filters' in _dict:
            args['catalog_filters'] = Filters.from_dict(_dict.get('catalog_filters'))
        if 'syndication_settings' in _dict:
            args['syndication_settings'] = SyndicationResource.from_dict(_dict.get('syndication_settings'))
        if 'kind' in _dict:
            args['kind'] = _dict.get('kind')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Catalog object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'rev') and self.rev is not None:
            _dict['_rev'] = self.rev
        if hasattr(self, 'label') and self.label is not None:
            _dict['label'] = self.label
        if hasattr(self, 'short_description') and self.short_description is not None:
            _dict['short_description'] = self.short_description
        if hasattr(self, 'catalog_icon_url') and self.catalog_icon_url is not None:
            _dict['catalog_icon_url'] = self.catalog_icon_url
        if hasattr(self, 'tags') and self.tags is not None:
            _dict['tags'] = self.tags
        if hasattr(self, 'url') and getattr(self, 'url') is not None:
            _dict['url'] = getattr(self, 'url')
        if hasattr(self, 'crn') and getattr(self, 'crn') is not None:
            _dict['crn'] = getattr(self, 'crn')
        if hasattr(self, 'offerings_url') and getattr(self, 'offerings_url') is not None:
            _dict['offerings_url'] = getattr(self, 'offerings_url')
        if hasattr(self, 'features') and self.features is not None:
            _dict['features'] = [x.to_dict() for x in self.features]
        if hasattr(self, 'disabled') and self.disabled is not None:
            _dict['disabled'] = self.disabled
        if hasattr(self, 'created') and getattr(self, 'created') is not None:
            _dict['created'] = datetime_to_string(getattr(self, 'created'))
        if hasattr(self, 'updated') and getattr(self, 'updated') is not None:
            _dict['updated'] = datetime_to_string(getattr(self, 'updated'))
        if hasattr(self, 'resource_group_id') and self.resource_group_id is not None:
            _dict['resource_group_id'] = self.resource_group_id
        if hasattr(self, 'owning_account') and self.owning_account is not None:
            _dict['owning_account'] = self.owning_account
        if hasattr(self, 'catalog_filters') and self.catalog_filters is not None:
            _dict['catalog_filters'] = self.catalog_filters.to_dict()
        if hasattr(self, 'syndication_settings') and self.syndication_settings is not None:
            _dict['syndication_settings'] = self.syndication_settings.to_dict()
        if hasattr(self, 'kind') and self.kind is not None:
            _dict['kind'] = self.kind
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Catalog object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Catalog') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Catalog') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class CatalogObject:
    """
    object information.

    :attr str id: (optional) unique id.
    :attr str name: (optional) The programmatic name of this offering.
    :attr str rev: (optional) Cloudant revision.
    :attr str crn: (optional) The crn for this specific object.
    :attr str url: (optional) The url for this specific object.
    :attr str parent_id: (optional) The parent for this specific object.
    :attr str label_i18n: (optional) Translated display name in the requested
          language.
    :attr str label: (optional) Display name in the requested language.
    :attr List[str] tags: (optional) List of tags associated with this catalog.
    :attr datetime created: (optional) The date and time this catalog was created.
    :attr datetime updated: (optional) The date and time this catalog was last
          updated.
    :attr str short_description: (optional) Short description in the requested
          language.
    :attr str short_description_i18n: (optional) Short description translation.
    :attr str kind: (optional) Kind of object.
    :attr PublishObject publish: (optional) Publish information.
    :attr State state: (optional) Offering state.
    :attr str catalog_id: (optional) The id of the catalog containing this offering.
    :attr str catalog_name: (optional) The name of the catalog.
    :attr dict data: (optional) Map of data values for this object.
    """

    def __init__(
        self,
        *,
        id: str = None,
        name: str = None,
        rev: str = None,
        crn: str = None,
        url: str = None,
        parent_id: str = None,
        label_i18n: str = None,
        label: str = None,
        tags: List[str] = None,
        created: datetime = None,
        updated: datetime = None,
        short_description: str = None,
        short_description_i18n: str = None,
        kind: str = None,
        publish: 'PublishObject' = None,
        state: 'State' = None,
        catalog_id: str = None,
        catalog_name: str = None,
        data: dict = None
    ) -> None:
        """
        Initialize a CatalogObject object.

        :param str id: (optional) unique id.
        :param str name: (optional) The programmatic name of this offering.
        :param str rev: (optional) Cloudant revision.
        :param str crn: (optional) The crn for this specific object.
        :param str url: (optional) The url for this specific object.
        :param str parent_id: (optional) The parent for this specific object.
        :param str label_i18n: (optional) Translated display name in the requested
               language.
        :param str label: (optional) Display name in the requested language.
        :param List[str] tags: (optional) List of tags associated with this
               catalog.
        :param datetime created: (optional) The date and time this catalog was
               created.
        :param datetime updated: (optional) The date and time this catalog was last
               updated.
        :param str short_description: (optional) Short description in the requested
               language.
        :param str short_description_i18n: (optional) Short description
               translation.
        :param str kind: (optional) Kind of object.
        :param PublishObject publish: (optional) Publish information.
        :param State state: (optional) Offering state.
        :param str catalog_id: (optional) The id of the catalog containing this
               offering.
        :param str catalog_name: (optional) The name of the catalog.
        :param dict data: (optional) Map of data values for this object.
        """
        self.id = id
        self.name = name
        self.rev = rev
        self.crn = crn
        self.url = url
        self.parent_id = parent_id
        self.label_i18n = label_i18n
        self.label = label
        self.tags = tags
        self.created = created
        self.updated = updated
        self.short_description = short_description
        self.short_description_i18n = short_description_i18n
        self.kind = kind
        self.publish = publish
        self.state = state
        self.catalog_id = catalog_id
        self.catalog_name = catalog_name
        self.data = data

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CatalogObject':
        """Initialize a CatalogObject object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if '_rev' in _dict:
            args['rev'] = _dict.get('_rev')
        if 'crn' in _dict:
            args['crn'] = _dict.get('crn')
        if 'url' in _dict:
            args['url'] = _dict.get('url')
        if 'parent_id' in _dict:
            args['parent_id'] = _dict.get('parent_id')
        if 'label_i18n' in _dict:
            args['label_i18n'] = _dict.get('label_i18n')
        if 'label' in _dict:
            args['label'] = _dict.get('label')
        if 'tags' in _dict:
            args['tags'] = _dict.get('tags')
        if 'created' in _dict:
            args['created'] = string_to_datetime(_dict.get('created'))
        if 'updated' in _dict:
            args['updated'] = string_to_datetime(_dict.get('updated'))
        if 'short_description' in _dict:
            args['short_description'] = _dict.get('short_description')
        if 'short_description_i18n' in _dict:
            args['short_description_i18n'] = _dict.get('short_description_i18n')
        if 'kind' in _dict:
            args['kind'] = _dict.get('kind')
        if 'publish' in _dict:
            args['publish'] = PublishObject.from_dict(_dict.get('publish'))
        if 'state' in _dict:
            args['state'] = State.from_dict(_dict.get('state'))
        if 'catalog_id' in _dict:
            args['catalog_id'] = _dict.get('catalog_id')
        if 'catalog_name' in _dict:
            args['catalog_name'] = _dict.get('catalog_name')
        if 'data' in _dict:
            args['data'] = _dict.get('data')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CatalogObject object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'rev') and self.rev is not None:
            _dict['_rev'] = self.rev
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'url') and self.url is not None:
            _dict['url'] = self.url
        if hasattr(self, 'parent_id') and self.parent_id is not None:
            _dict['parent_id'] = self.parent_id
        if hasattr(self, 'label_i18n') and self.label_i18n is not None:
            _dict['label_i18n'] = self.label_i18n
        if hasattr(self, 'label') and self.label is not None:
            _dict['label'] = self.label
        if hasattr(self, 'tags') and self.tags is not None:
            _dict['tags'] = self.tags
        if hasattr(self, 'created') and self.created is not None:
            _dict['created'] = datetime_to_string(self.created)
        if hasattr(self, 'updated') and self.updated is not None:
            _dict['updated'] = datetime_to_string(self.updated)
        if hasattr(self, 'short_description') and self.short_description is not None:
            _dict['short_description'] = self.short_description
        if hasattr(self, 'short_description_i18n') and self.short_description_i18n is not None:
            _dict['short_description_i18n'] = self.short_description_i18n
        if hasattr(self, 'kind') and self.kind is not None:
            _dict['kind'] = self.kind
        if hasattr(self, 'publish') and self.publish is not None:
            _dict['publish'] = self.publish.to_dict()
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state.to_dict()
        if hasattr(self, 'catalog_id') and self.catalog_id is not None:
            _dict['catalog_id'] = self.catalog_id
        if hasattr(self, 'catalog_name') and self.catalog_name is not None:
            _dict['catalog_name'] = self.catalog_name
        if hasattr(self, 'data') and self.data is not None:
            _dict['data'] = self.data
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this CatalogObject object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CatalogObject') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CatalogObject') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class CatalogSearchResult:
    """
    Paginated catalog search result.

    :attr int total_count: (optional) The overall total number of resources in the
          search result set.
    :attr List[Catalog] resources: (optional) Resulting objects.
    """

    def __init__(self, *, total_count: int = None, resources: List['Catalog'] = None) -> None:
        """
        Initialize a CatalogSearchResult object.

        :param int total_count: (optional) The overall total number of resources in
               the search result set.
        :param List[Catalog] resources: (optional) Resulting objects.
        """
        self.total_count = total_count
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CatalogSearchResult':
        """Initialize a CatalogSearchResult object from a json dictionary."""
        args = {}
        if 'total_count' in _dict:
            args['total_count'] = _dict.get('total_count')
        if 'resources' in _dict:
            args['resources'] = [Catalog.from_dict(x) for x in _dict.get('resources')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CatalogSearchResult object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'total_count') and self.total_count is not None:
            _dict['total_count'] = self.total_count
        if hasattr(self, 'resources') and self.resources is not None:
            _dict['resources'] = [x.to_dict() for x in self.resources]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this CatalogSearchResult object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CatalogSearchResult') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CatalogSearchResult') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class CategoryFilter:
    """
    Filter on a category. The filter will match against the values of the given category
    with include or exclude.

    :attr bool include: (optional) -> true - This is an include filter, false - this
          is an exclude filter.
    :attr FilterTerms filter: (optional) Offering filter terms.
    """

    def __init__(self, *, include: bool = None, filter: 'FilterTerms' = None) -> None:
        """
        Initialize a CategoryFilter object.

        :param bool include: (optional) -> true - This is an include filter, false
               - this is an exclude filter.
        :param FilterTerms filter: (optional) Offering filter terms.
        """
        self.include = include
        self.filter = filter

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'CategoryFilter':
        """Initialize a CategoryFilter object from a json dictionary."""
        args = {}
        if 'include' in _dict:
            args['include'] = _dict.get('include')
        if 'filter' in _dict:
            args['filter'] = FilterTerms.from_dict(_dict.get('filter'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a CategoryFilter object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'include') and self.include is not None:
            _dict['include'] = self.include
        if hasattr(self, 'filter') and self.filter is not None:
            _dict['filter'] = self.filter.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this CategoryFilter object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'CategoryFilter') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'CategoryFilter') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ClusterInfo:
    """
    Cluster information.

    :attr str resource_group_id: (optional) Resource Group ID.
    :attr str resource_group_name: (optional) Resource Group name.
    :attr str id: (optional) Cluster ID.
    :attr str name: (optional) Cluster name.
    :attr str region: (optional) Cluster region.
    """

    def __init__(
        self,
        *,
        resource_group_id: str = None,
        resource_group_name: str = None,
        id: str = None,
        name: str = None,
        region: str = None
    ) -> None:
        """
        Initialize a ClusterInfo object.

        :param str resource_group_id: (optional) Resource Group ID.
        :param str resource_group_name: (optional) Resource Group name.
        :param str id: (optional) Cluster ID.
        :param str name: (optional) Cluster name.
        :param str region: (optional) Cluster region.
        """
        self.resource_group_id = resource_group_id
        self.resource_group_name = resource_group_name
        self.id = id
        self.name = name
        self.region = region

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ClusterInfo':
        """Initialize a ClusterInfo object from a json dictionary."""
        args = {}
        if 'resource_group_id' in _dict:
            args['resource_group_id'] = _dict.get('resource_group_id')
        if 'resource_group_name' in _dict:
            args['resource_group_name'] = _dict.get('resource_group_name')
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'region' in _dict:
            args['region'] = _dict.get('region')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ClusterInfo object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'resource_group_id') and self.resource_group_id is not None:
            _dict['resource_group_id'] = self.resource_group_id
        if hasattr(self, 'resource_group_name') and self.resource_group_name is not None:
            _dict['resource_group_name'] = self.resource_group_name
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'region') and self.region is not None:
            _dict['region'] = self.region
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ClusterInfo object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ClusterInfo') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ClusterInfo') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Configuration:
    """
    Configuration description.

    :attr str key: (optional) Configuration key.
    :attr str type: (optional) Value type (string, boolean, int).
    :attr object default_value: (optional) The default value.  To use a secret when
          the type is password, specify a JSON encoded value of
          $ref:#/components/schemas/SecretInstance, prefixed with `cmsm_v1:`.
    :attr str value_constraint: (optional) Constraint associated with value, e.g.,
          for string type - regx:[a-z].
    :attr str description: (optional) Key description.
    :attr bool required: (optional) Is key required to install.
    :attr List[object] options: (optional) List of options of type.
    :attr bool hidden: (optional) Hide values.
    """

    def __init__(
        self,
        *,
        key: str = None,
        type: str = None,
        default_value: object = None,
        value_constraint: str = None,
        description: str = None,
        required: bool = None,
        options: List[object] = None,
        hidden: bool = None
    ) -> None:
        """
        Initialize a Configuration object.

        :param str key: (optional) Configuration key.
        :param str type: (optional) Value type (string, boolean, int).
        :param object default_value: (optional) The default value.  To use a secret
               when the type is password, specify a JSON encoded value of
               $ref:#/components/schemas/SecretInstance, prefixed with `cmsm_v1:`.
        :param str value_constraint: (optional) Constraint associated with value,
               e.g., for string type - regx:[a-z].
        :param str description: (optional) Key description.
        :param bool required: (optional) Is key required to install.
        :param List[object] options: (optional) List of options of type.
        :param bool hidden: (optional) Hide values.
        """
        self.key = key
        self.type = type
        self.default_value = default_value
        self.value_constraint = value_constraint
        self.description = description
        self.required = required
        self.options = options
        self.hidden = hidden

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Configuration':
        """Initialize a Configuration object from a json dictionary."""
        args = {}
        if 'key' in _dict:
            args['key'] = _dict.get('key')
        if 'type' in _dict:
            args['type'] = _dict.get('type')
        if 'default_value' in _dict:
            args['default_value'] = _dict.get('default_value')
        if 'value_constraint' in _dict:
            args['value_constraint'] = _dict.get('value_constraint')
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        if 'required' in _dict:
            args['required'] = _dict.get('required')
        if 'options' in _dict:
            args['options'] = _dict.get('options')
        if 'hidden' in _dict:
            args['hidden'] = _dict.get('hidden')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Configuration object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'key') and self.key is not None:
            _dict['key'] = self.key
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'default_value') and self.default_value is not None:
            _dict['default_value'] = self.default_value
        if hasattr(self, 'value_constraint') and self.value_constraint is not None:
            _dict['value_constraint'] = self.value_constraint
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'required') and self.required is not None:
            _dict['required'] = self.required
        if hasattr(self, 'options') and self.options is not None:
            _dict['options'] = self.options
        if hasattr(self, 'hidden') and self.hidden is not None:
            _dict['hidden'] = self.hidden
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Configuration object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Configuration') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Configuration') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class DeployRequestBodySchematics:
    """
    Schematics workspace configuration.

    :attr str name: (optional) Schematics workspace name.
    :attr str description: (optional) Schematics workspace description.
    :attr List[str] tags: (optional) Schematics workspace tags.
    :attr str resource_group_id: (optional) Resource group to use when creating the
          schematics workspace.
    """

    def __init__(
        self, *, name: str = None, description: str = None, tags: List[str] = None, resource_group_id: str = None
    ) -> None:
        """
        Initialize a DeployRequestBodySchematics object.

        :param str name: (optional) Schematics workspace name.
        :param str description: (optional) Schematics workspace description.
        :param List[str] tags: (optional) Schematics workspace tags.
        :param str resource_group_id: (optional) Resource group to use when
               creating the schematics workspace.
        """
        self.name = name
        self.description = description
        self.tags = tags
        self.resource_group_id = resource_group_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'DeployRequestBodySchematics':
        """Initialize a DeployRequestBodySchematics object from a json dictionary."""
        args = {}
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        if 'tags' in _dict:
            args['tags'] = _dict.get('tags')
        if 'resource_group_id' in _dict:
            args['resource_group_id'] = _dict.get('resource_group_id')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a DeployRequestBodySchematics object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'tags') and self.tags is not None:
            _dict['tags'] = self.tags
        if hasattr(self, 'resource_group_id') and self.resource_group_id is not None:
            _dict['resource_group_id'] = self.resource_group_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this DeployRequestBodySchematics object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'DeployRequestBodySchematics') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'DeployRequestBodySchematics') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Deployment:
    """
    Deployment for offering.

    :attr str id: (optional) unique id.
    :attr str label: (optional) Display Name in the requested language.
    :attr str name: (optional) The programmatic name of this offering.
    :attr str short_description: (optional) Short description in the requested
          language.
    :attr str long_description: (optional) Long description in the requested
          language.
    :attr dict metadata: (optional) open ended metadata information.
    :attr List[str] tags: (optional) list of tags associated with this catalog.
    :attr datetime created: (optional) the date'time this catalog was created.
    :attr datetime updated: (optional) the date'time this catalog was last updated.
    """

    def __init__(
        self,
        *,
        id: str = None,
        label: str = None,
        name: str = None,
        short_description: str = None,
        long_description: str = None,
        metadata: dict = None,
        tags: List[str] = None,
        created: datetime = None,
        updated: datetime = None
    ) -> None:
        """
        Initialize a Deployment object.

        :param str id: (optional) unique id.
        :param str label: (optional) Display Name in the requested language.
        :param str name: (optional) The programmatic name of this offering.
        :param str short_description: (optional) Short description in the requested
               language.
        :param str long_description: (optional) Long description in the requested
               language.
        :param dict metadata: (optional) open ended metadata information.
        :param List[str] tags: (optional) list of tags associated with this
               catalog.
        :param datetime created: (optional) the date'time this catalog was created.
        :param datetime updated: (optional) the date'time this catalog was last
               updated.
        """
        self.id = id
        self.label = label
        self.name = name
        self.short_description = short_description
        self.long_description = long_description
        self.metadata = metadata
        self.tags = tags
        self.created = created
        self.updated = updated

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Deployment':
        """Initialize a Deployment object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'label' in _dict:
            args['label'] = _dict.get('label')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'short_description' in _dict:
            args['short_description'] = _dict.get('short_description')
        if 'long_description' in _dict:
            args['long_description'] = _dict.get('long_description')
        if 'metadata' in _dict:
            args['metadata'] = _dict.get('metadata')
        if 'tags' in _dict:
            args['tags'] = _dict.get('tags')
        if 'created' in _dict:
            args['created'] = string_to_datetime(_dict.get('created'))
        if 'updated' in _dict:
            args['updated'] = string_to_datetime(_dict.get('updated'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Deployment object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'label') and self.label is not None:
            _dict['label'] = self.label
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'short_description') and self.short_description is not None:
            _dict['short_description'] = self.short_description
        if hasattr(self, 'long_description') and self.long_description is not None:
            _dict['long_description'] = self.long_description
        if hasattr(self, 'metadata') and self.metadata is not None:
            _dict['metadata'] = self.metadata
        if hasattr(self, 'tags') and self.tags is not None:
            _dict['tags'] = self.tags
        if hasattr(self, 'created') and self.created is not None:
            _dict['created'] = datetime_to_string(self.created)
        if hasattr(self, 'updated') and self.updated is not None:
            _dict['updated'] = datetime_to_string(self.updated)
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Deployment object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Deployment') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Deployment') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Feature:
    """
    Feature information.

    :attr str title: (optional) Heading.
    :attr str description: (optional) Feature description.
    """

    def __init__(self, *, title: str = None, description: str = None) -> None:
        """
        Initialize a Feature object.

        :param str title: (optional) Heading.
        :param str description: (optional) Feature description.
        """
        self.title = title
        self.description = description

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Feature':
        """Initialize a Feature object from a json dictionary."""
        args = {}
        if 'title' in _dict:
            args['title'] = _dict.get('title')
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Feature object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'title') and self.title is not None:
            _dict['title'] = self.title
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Feature object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Feature') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Feature') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class FilterTerms:
    """
    Offering filter terms.

    :attr List[str] filter_terms: (optional) List of values to match against. If
          include is true, then if the offering has one of the values then the offering is
          included. If include is false, then if the offering has one of the values then
          the offering is excluded.
    """

    def __init__(self, *, filter_terms: List[str] = None) -> None:
        """
        Initialize a FilterTerms object.

        :param List[str] filter_terms: (optional) List of values to match against.
               If include is true, then if the offering has one of the values then the
               offering is included. If include is false, then if the offering has one of
               the values then the offering is excluded.
        """
        self.filter_terms = filter_terms

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'FilterTerms':
        """Initialize a FilterTerms object from a json dictionary."""
        args = {}
        if 'filter_terms' in _dict:
            args['filter_terms'] = _dict.get('filter_terms')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a FilterTerms object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'filter_terms') and self.filter_terms is not None:
            _dict['filter_terms'] = self.filter_terms
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this FilterTerms object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'FilterTerms') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'FilterTerms') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Filters:
    """
    Filters for account and catalog filters.

    :attr bool include_all: (optional) -> true - Include all of the public catalog
          when filtering. Further settings will specifically exclude some offerings. false
          - Exclude all of the public catalog when filtering. Further settings will
          specifically include some offerings.
    :attr dict category_filters: (optional) Filter against offering properties.
    :attr IDFilter id_filters: (optional) Filter on offering ID's. There is an
          include filter and an exclule filter. Both can be set.
    """

    def __init__(
        self, *, include_all: bool = None, category_filters: dict = None, id_filters: 'IDFilter' = None
    ) -> None:
        """
        Initialize a Filters object.

        :param bool include_all: (optional) -> true - Include all of the public
               catalog when filtering. Further settings will specifically exclude some
               offerings. false - Exclude all of the public catalog when filtering.
               Further settings will specifically include some offerings.
        :param dict category_filters: (optional) Filter against offering
               properties.
        :param IDFilter id_filters: (optional) Filter on offering ID's. There is an
               include filter and an exclule filter. Both can be set.
        """
        self.include_all = include_all
        self.category_filters = category_filters
        self.id_filters = id_filters

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Filters':
        """Initialize a Filters object from a json dictionary."""
        args = {}
        if 'include_all' in _dict:
            args['include_all'] = _dict.get('include_all')
        if 'category_filters' in _dict:
            args['category_filters'] = {
                k: CategoryFilter.from_dict(v) for k, v in _dict.get('category_filters').items()
            }
        if 'id_filters' in _dict:
            args['id_filters'] = IDFilter.from_dict(_dict.get('id_filters'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Filters object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'include_all') and self.include_all is not None:
            _dict['include_all'] = self.include_all
        if hasattr(self, 'category_filters') and self.category_filters is not None:
            _dict['category_filters'] = {k: v.to_dict() for k, v in self.category_filters.items()}
        if hasattr(self, 'id_filters') and self.id_filters is not None:
            _dict['id_filters'] = self.id_filters.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Filters object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Filters') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Filters') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class IDFilter:
    """
    Filter on offering ID's. There is an include filter and an exclule filter. Both can be
    set.

    :attr FilterTerms include: (optional) Offering filter terms.
    :attr FilterTerms exclude: (optional) Offering filter terms.
    """

    def __init__(self, *, include: 'FilterTerms' = None, exclude: 'FilterTerms' = None) -> None:
        """
        Initialize a IDFilter object.

        :param FilterTerms include: (optional) Offering filter terms.
        :param FilterTerms exclude: (optional) Offering filter terms.
        """
        self.include = include
        self.exclude = exclude

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'IDFilter':
        """Initialize a IDFilter object from a json dictionary."""
        args = {}
        if 'include' in _dict:
            args['include'] = FilterTerms.from_dict(_dict.get('include'))
        if 'exclude' in _dict:
            args['exclude'] = FilterTerms.from_dict(_dict.get('exclude'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a IDFilter object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'include') and self.include is not None:
            _dict['include'] = self.include.to_dict()
        if hasattr(self, 'exclude') and self.exclude is not None:
            _dict['exclude'] = self.exclude.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this IDFilter object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'IDFilter') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'IDFilter') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Image:
    """
    Image.

    :attr str image: (optional) Image.
    """

    def __init__(self, *, image: str = None) -> None:
        """
        Initialize a Image object.

        :param str image: (optional) Image.
        """
        self.image = image

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Image':
        """Initialize a Image object from a json dictionary."""
        args = {}
        if 'image' in _dict:
            args['image'] = _dict.get('image')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Image object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'image') and self.image is not None:
            _dict['image'] = self.image
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Image object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Image') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Image') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ImageManifest:
    """
    Image Manifest.

    :attr str description: (optional) Image manifest description.
    :attr List[Image] images: (optional) List of images.
    """

    def __init__(self, *, description: str = None, images: List['Image'] = None) -> None:
        """
        Initialize a ImageManifest object.

        :param str description: (optional) Image manifest description.
        :param List[Image] images: (optional) List of images.
        """
        self.description = description
        self.images = images

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ImageManifest':
        """Initialize a ImageManifest object from a json dictionary."""
        args = {}
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        if 'images' in _dict:
            args['images'] = [Image.from_dict(x) for x in _dict.get('images')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ImageManifest object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        if hasattr(self, 'images') and self.images is not None:
            _dict['images'] = [x.to_dict() for x in self.images]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ImageManifest object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ImageManifest') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ImageManifest') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class InstallStatus:
    """
    Installation status.

    :attr InstallStatusMetadata metadata: (optional) Installation status metadata.
    :attr InstallStatusRelease release: (optional) Release information.
    :attr InstallStatusContentMgmt content_mgmt: (optional) Content management
          information.
    """

    def __init__(
        self,
        *,
        metadata: 'InstallStatusMetadata' = None,
        release: 'InstallStatusRelease' = None,
        content_mgmt: 'InstallStatusContentMgmt' = None
    ) -> None:
        """
        Initialize a InstallStatus object.

        :param InstallStatusMetadata metadata: (optional) Installation status
               metadata.
        :param InstallStatusRelease release: (optional) Release information.
        :param InstallStatusContentMgmt content_mgmt: (optional) Content management
               information.
        """
        self.metadata = metadata
        self.release = release
        self.content_mgmt = content_mgmt

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'InstallStatus':
        """Initialize a InstallStatus object from a json dictionary."""
        args = {}
        if 'metadata' in _dict:
            args['metadata'] = InstallStatusMetadata.from_dict(_dict.get('metadata'))
        if 'release' in _dict:
            args['release'] = InstallStatusRelease.from_dict(_dict.get('release'))
        if 'content_mgmt' in _dict:
            args['content_mgmt'] = InstallStatusContentMgmt.from_dict(_dict.get('content_mgmt'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a InstallStatus object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'metadata') and self.metadata is not None:
            _dict['metadata'] = self.metadata.to_dict()
        if hasattr(self, 'release') and self.release is not None:
            _dict['release'] = self.release.to_dict()
        if hasattr(self, 'content_mgmt') and self.content_mgmt is not None:
            _dict['content_mgmt'] = self.content_mgmt.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this InstallStatus object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'InstallStatus') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'InstallStatus') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class InstallStatusContentMgmt:
    """
    Content management information.

    :attr List[dict] pods: (optional) Pods.
    :attr List[dict] errors: (optional) Errors.
    """

    def __init__(self, *, pods: List[dict] = None, errors: List[dict] = None) -> None:
        """
        Initialize a InstallStatusContentMgmt object.

        :param List[dict] pods: (optional) Pods.
        :param List[dict] errors: (optional) Errors.
        """
        self.pods = pods
        self.errors = errors

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'InstallStatusContentMgmt':
        """Initialize a InstallStatusContentMgmt object from a json dictionary."""
        args = {}
        if 'pods' in _dict:
            args['pods'] = _dict.get('pods')
        if 'errors' in _dict:
            args['errors'] = _dict.get('errors')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a InstallStatusContentMgmt object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'pods') and self.pods is not None:
            _dict['pods'] = self.pods
        if hasattr(self, 'errors') and self.errors is not None:
            _dict['errors'] = self.errors
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this InstallStatusContentMgmt object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'InstallStatusContentMgmt') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'InstallStatusContentMgmt') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class InstallStatusMetadata:
    """
    Installation status metadata.

    :attr str cluster_id: (optional) Cluster ID.
    :attr str region: (optional) Cluster region.
    :attr str namespace: (optional) Cluster namespace.
    :attr str workspace_id: (optional) Workspace ID.
    :attr str workspace_name: (optional) Workspace name.
    """

    def __init__(
        self,
        *,
        cluster_id: str = None,
        region: str = None,
        namespace: str = None,
        workspace_id: str = None,
        workspace_name: str = None
    ) -> None:
        """
        Initialize a InstallStatusMetadata object.

        :param str cluster_id: (optional) Cluster ID.
        :param str region: (optional) Cluster region.
        :param str namespace: (optional) Cluster namespace.
        :param str workspace_id: (optional) Workspace ID.
        :param str workspace_name: (optional) Workspace name.
        """
        self.cluster_id = cluster_id
        self.region = region
        self.namespace = namespace
        self.workspace_id = workspace_id
        self.workspace_name = workspace_name

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'InstallStatusMetadata':
        """Initialize a InstallStatusMetadata object from a json dictionary."""
        args = {}
        if 'cluster_id' in _dict:
            args['cluster_id'] = _dict.get('cluster_id')
        if 'region' in _dict:
            args['region'] = _dict.get('region')
        if 'namespace' in _dict:
            args['namespace'] = _dict.get('namespace')
        if 'workspace_id' in _dict:
            args['workspace_id'] = _dict.get('workspace_id')
        if 'workspace_name' in _dict:
            args['workspace_name'] = _dict.get('workspace_name')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a InstallStatusMetadata object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'cluster_id') and self.cluster_id is not None:
            _dict['cluster_id'] = self.cluster_id
        if hasattr(self, 'region') and self.region is not None:
            _dict['region'] = self.region
        if hasattr(self, 'namespace') and self.namespace is not None:
            _dict['namespace'] = self.namespace
        if hasattr(self, 'workspace_id') and self.workspace_id is not None:
            _dict['workspace_id'] = self.workspace_id
        if hasattr(self, 'workspace_name') and self.workspace_name is not None:
            _dict['workspace_name'] = self.workspace_name
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this InstallStatusMetadata object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'InstallStatusMetadata') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'InstallStatusMetadata') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class InstallStatusRelease:
    """
    Release information.

    :attr List[dict] deployments: (optional) Kube deployments.
    :attr List[dict] replicasets: (optional) Kube replica sets.
    :attr List[dict] statefulsets: (optional) Kube stateful sets.
    :attr List[dict] pods: (optional) Kube pods.
    :attr List[dict] errors: (optional) Kube errors.
    """

    def __init__(
        self,
        *,
        deployments: List[dict] = None,
        replicasets: List[dict] = None,
        statefulsets: List[dict] = None,
        pods: List[dict] = None,
        errors: List[dict] = None
    ) -> None:
        """
        Initialize a InstallStatusRelease object.

        :param List[dict] deployments: (optional) Kube deployments.
        :param List[dict] replicasets: (optional) Kube replica sets.
        :param List[dict] statefulsets: (optional) Kube stateful sets.
        :param List[dict] pods: (optional) Kube pods.
        :param List[dict] errors: (optional) Kube errors.
        """
        self.deployments = deployments
        self.replicasets = replicasets
        self.statefulsets = statefulsets
        self.pods = pods
        self.errors = errors

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'InstallStatusRelease':
        """Initialize a InstallStatusRelease object from a json dictionary."""
        args = {}
        if 'deployments' in _dict:
            args['deployments'] = _dict.get('deployments')
        if 'replicasets' in _dict:
            args['replicasets'] = _dict.get('replicasets')
        if 'statefulsets' in _dict:
            args['statefulsets'] = _dict.get('statefulsets')
        if 'pods' in _dict:
            args['pods'] = _dict.get('pods')
        if 'errors' in _dict:
            args['errors'] = _dict.get('errors')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a InstallStatusRelease object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'deployments') and self.deployments is not None:
            _dict['deployments'] = self.deployments
        if hasattr(self, 'replicasets') and self.replicasets is not None:
            _dict['replicasets'] = self.replicasets
        if hasattr(self, 'statefulsets') and self.statefulsets is not None:
            _dict['statefulsets'] = self.statefulsets
        if hasattr(self, 'pods') and self.pods is not None:
            _dict['pods'] = self.pods
        if hasattr(self, 'errors') and self.errors is not None:
            _dict['errors'] = self.errors
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this InstallStatusRelease object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'InstallStatusRelease') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'InstallStatusRelease') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class JsonPatchOperation:
    """
    This model represents an individual patch operation to be performed on a JSON
    document, as defined by RFC 6902.

    :attr str op: The operation to be performed.
    :attr str path: The JSON Pointer that identifies the field that is the target of
          the operation.
    :attr str from_: (optional) The JSON Pointer that identifies the field that is
          the source of the operation.
    :attr object value: (optional) The value to be used within the operation.
    """

    def __init__(self, op: str, path: str, *, from_: str = None, value: object = None) -> None:
        """
        Initialize a JsonPatchOperation object.

        :param str op: The operation to be performed.
        :param str path: The JSON Pointer that identifies the field that is the
               target of the operation.
        :param str from_: (optional) The JSON Pointer that identifies the field
               that is the source of the operation.
        :param object value: (optional) The value to be used within the operation.
        """
        self.op = op
        self.path = path
        self.from_ = from_
        self.value = value

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'JsonPatchOperation':
        """Initialize a JsonPatchOperation object from a json dictionary."""
        args = {}
        if 'op' in _dict:
            args['op'] = _dict.get('op')
        else:
            raise ValueError('Required property \'op\' not present in JsonPatchOperation JSON')
        if 'path' in _dict:
            args['path'] = _dict.get('path')
        else:
            raise ValueError('Required property \'path\' not present in JsonPatchOperation JSON')
        if 'from' in _dict:
            args['from_'] = _dict.get('from')
        if 'value' in _dict:
            args['value'] = _dict.get('value')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a JsonPatchOperation object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'op') and self.op is not None:
            _dict['op'] = self.op
        if hasattr(self, 'path') and self.path is not None:
            _dict['path'] = self.path
        if hasattr(self, 'from_') and self.from_ is not None:
            _dict['from'] = self.from_
        if hasattr(self, 'value') and self.value is not None:
            _dict['value'] = self.value
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this JsonPatchOperation object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'JsonPatchOperation') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'JsonPatchOperation') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other

    class OpEnum(str, Enum):
        """
        The operation to be performed.
        """

        ADD = 'add'
        REMOVE = 'remove'
        REPLACE = 'replace'
        MOVE = 'move'
        COPY = 'copy'
        TEST = 'test'


class Kind:
    """
    Offering kind.

    :attr str id: (optional) Unique ID.
    :attr str format_kind: (optional) content kind, e.g., helm, vm image.
    :attr str target_kind: (optional) target cloud to install, e.g., iks,
          open_shift_iks.
    :attr dict metadata: (optional) Open ended metadata information.
    :attr str install_description: (optional) Installation instruction.
    :attr List[str] tags: (optional) List of tags associated with this catalog.
    :attr List[Feature] additional_features: (optional) List of features associated
          with this offering.
    :attr datetime created: (optional) The date and time this catalog was created.
    :attr datetime updated: (optional) The date and time this catalog was last
          updated.
    :attr List[Version] versions: (optional) list of versions.
    :attr List[Plan] plans: (optional) list of plans.
    """

    def __init__(
        self,
        *,
        id: str = None,
        format_kind: str = None,
        target_kind: str = None,
        metadata: dict = None,
        install_description: str = None,
        tags: List[str] = None,
        additional_features: List['Feature'] = None,
        created: datetime = None,
        updated: datetime = None,
        versions: List['Version'] = None,
        plans: List['Plan'] = None
    ) -> None:
        """
        Initialize a Kind object.

        :param str id: (optional) Unique ID.
        :param str format_kind: (optional) content kind, e.g., helm, vm image.
        :param str target_kind: (optional) target cloud to install, e.g., iks,
               open_shift_iks.
        :param dict metadata: (optional) Open ended metadata information.
        :param str install_description: (optional) Installation instruction.
        :param List[str] tags: (optional) List of tags associated with this
               catalog.
        :param List[Feature] additional_features: (optional) List of features
               associated with this offering.
        :param datetime created: (optional) The date and time this catalog was
               created.
        :param datetime updated: (optional) The date and time this catalog was last
               updated.
        :param List[Version] versions: (optional) list of versions.
        :param List[Plan] plans: (optional) list of plans.
        """
        self.id = id
        self.format_kind = format_kind
        self.target_kind = target_kind
        self.metadata = metadata
        self.install_description = install_description
        self.tags = tags
        self.additional_features = additional_features
        self.created = created
        self.updated = updated
        self.versions = versions
        self.plans = plans

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Kind':
        """Initialize a Kind object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'format_kind' in _dict:
            args['format_kind'] = _dict.get('format_kind')
        if 'target_kind' in _dict:
            args['target_kind'] = _dict.get('target_kind')
        if 'metadata' in _dict:
            args['metadata'] = _dict.get('metadata')
        if 'install_description' in _dict:
            args['install_description'] = _dict.get('install_description')
        if 'tags' in _dict:
            args['tags'] = _dict.get('tags')
        if 'additional_features' in _dict:
            args['additional_features'] = [Feature.from_dict(x) for x in _dict.get('additional_features')]
        if 'created' in _dict:
            args['created'] = string_to_datetime(_dict.get('created'))
        if 'updated' in _dict:
            args['updated'] = string_to_datetime(_dict.get('updated'))
        if 'versions' in _dict:
            args['versions'] = [Version.from_dict(x) for x in _dict.get('versions')]
        if 'plans' in _dict:
            args['plans'] = [Plan.from_dict(x) for x in _dict.get('plans')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Kind object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'format_kind') and self.format_kind is not None:
            _dict['format_kind'] = self.format_kind
        if hasattr(self, 'target_kind') and self.target_kind is not None:
            _dict['target_kind'] = self.target_kind
        if hasattr(self, 'metadata') and self.metadata is not None:
            _dict['metadata'] = self.metadata
        if hasattr(self, 'install_description') and self.install_description is not None:
            _dict['install_description'] = self.install_description
        if hasattr(self, 'tags') and self.tags is not None:
            _dict['tags'] = self.tags
        if hasattr(self, 'additional_features') and self.additional_features is not None:
            _dict['additional_features'] = [x.to_dict() for x in self.additional_features]
        if hasattr(self, 'created') and self.created is not None:
            _dict['created'] = datetime_to_string(self.created)
        if hasattr(self, 'updated') and self.updated is not None:
            _dict['updated'] = datetime_to_string(self.updated)
        if hasattr(self, 'versions') and self.versions is not None:
            _dict['versions'] = [x.to_dict() for x in self.versions]
        if hasattr(self, 'plans') and self.plans is not None:
            _dict['plans'] = [x.to_dict() for x in self.plans]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Kind object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Kind') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Kind') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class License:
    """
    BSS license.

    :attr str id: (optional) License ID.
    :attr str name: (optional) license name.
    :attr str type: (optional) type of license e.g., Apache xxx.
    :attr str url: (optional) URL for the license text.
    :attr str description: (optional) License description.
    """

    def __init__(
        self, *, id: str = None, name: str = None, type: str = None, url: str = None, description: str = None
    ) -> None:
        """
        Initialize a License object.

        :param str id: (optional) License ID.
        :param str name: (optional) license name.
        :param str type: (optional) type of license e.g., Apache xxx.
        :param str url: (optional) URL for the license text.
        :param str description: (optional) License description.
        """
        self.id = id
        self.name = name
        self.type = type
        self.url = url
        self.description = description

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'License':
        """Initialize a License object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'type' in _dict:
            args['type'] = _dict.get('type')
        if 'url' in _dict:
            args['url'] = _dict.get('url')
        if 'description' in _dict:
            args['description'] = _dict.get('description')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a License object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'url') and self.url is not None:
            _dict['url'] = self.url
        if hasattr(self, 'description') and self.description is not None:
            _dict['description'] = self.description
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this License object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'License') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'License') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class MediaItem:
    """
    Offering Media information.

    :attr str url: (optional) URL of the specified media item.
    :attr str caption: (optional) Caption for this media item.
    :attr str type: (optional) Type of this media item.
    :attr str thumbnail_url: (optional) Thumbnail URL for this media item.
    """

    def __init__(self, *, url: str = None, caption: str = None, type: str = None, thumbnail_url: str = None) -> None:
        """
        Initialize a MediaItem object.

        :param str url: (optional) URL of the specified media item.
        :param str caption: (optional) Caption for this media item.
        :param str type: (optional) Type of this media item.
        :param str thumbnail_url: (optional) Thumbnail URL for this media item.
        """
        self.url = url
        self.caption = caption
        self.type = type
        self.thumbnail_url = thumbnail_url

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'MediaItem':
        """Initialize a MediaItem object from a json dictionary."""
        args = {}
        if 'url' in _dict:
            args['url'] = _dict.get('url')
        if 'caption' in _dict:
            args['caption'] = _dict.get('caption')
        if 'type' in _dict:
            args['type'] = _dict.get('type')
        if 'thumbnail_url' in _dict:
            args['thumbnail_url'] = _dict.get('thumbnail_url')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a MediaItem object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'url') and self.url is not None:
            _dict['url'] = self.url
        if hasattr(self, 'caption') and self.caption is not None:
            _dict['caption'] = self.caption
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'thumbnail_url') and self.thumbnail_url is not None:
            _dict['thumbnail_url'] = self.thumbnail_url
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this MediaItem object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'MediaItem') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'MediaItem') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class NamespaceSearchResult:
    """
    Paginated list of namespace search results.

    :attr int offset: The offset (origin 0) of the first resource in this page of
          search results.
    :attr int limit: The maximum number of resources returned in each page of search
          results.
    :attr int total_count: (optional) The overall total number of resources in the
          search result set.
    :attr int resource_count: (optional) The number of resources returned in this
          page of search results.
    :attr str first: (optional) A URL for retrieving the first page of search
          results.
    :attr str last: (optional) A URL for retrieving the last page of search results.
    :attr str prev: (optional) A URL for retrieving the previous page of search
          results.
    :attr str next: (optional) A URL for retrieving the next page of search results.
    :attr List[str] resources: (optional) Resulting objects.
    """

    def __init__(
        self,
        offset: int,
        limit: int,
        *,
        total_count: int = None,
        resource_count: int = None,
        first: str = None,
        last: str = None,
        prev: str = None,
        next: str = None,
        resources: List[str] = None
    ) -> None:
        """
        Initialize a NamespaceSearchResult object.

        :param int offset: The offset (origin 0) of the first resource in this page
               of search results.
        :param int limit: The maximum number of resources returned in each page of
               search results.
        :param int total_count: (optional) The overall total number of resources in
               the search result set.
        :param int resource_count: (optional) The number of resources returned in
               this page of search results.
        :param str first: (optional) A URL for retrieving the first page of search
               results.
        :param str last: (optional) A URL for retrieving the last page of search
               results.
        :param str prev: (optional) A URL for retrieving the previous page of
               search results.
        :param str next: (optional) A URL for retrieving the next page of search
               results.
        :param List[str] resources: (optional) Resulting objects.
        """
        self.offset = offset
        self.limit = limit
        self.total_count = total_count
        self.resource_count = resource_count
        self.first = first
        self.last = last
        self.prev = prev
        self.next = next
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'NamespaceSearchResult':
        """Initialize a NamespaceSearchResult object from a json dictionary."""
        args = {}
        if 'offset' in _dict:
            args['offset'] = _dict.get('offset')
        else:
            raise ValueError('Required property \'offset\' not present in NamespaceSearchResult JSON')
        if 'limit' in _dict:
            args['limit'] = _dict.get('limit')
        else:
            raise ValueError('Required property \'limit\' not present in NamespaceSearchResult JSON')
        if 'total_count' in _dict:
            args['total_count'] = _dict.get('total_count')
        if 'resource_count' in _dict:
            args['resource_count'] = _dict.get('resource_count')
        if 'first' in _dict:
            args['first'] = _dict.get('first')
        if 'last' in _dict:
            args['last'] = _dict.get('last')
        if 'prev' in _dict:
            args['prev'] = _dict.get('prev')
        if 'next' in _dict:
            args['next'] = _dict.get('next')
        if 'resources' in _dict:
            args['resources'] = _dict.get('resources')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a NamespaceSearchResult object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'total_count') and self.total_count is not None:
            _dict['total_count'] = self.total_count
        if hasattr(self, 'resource_count') and self.resource_count is not None:
            _dict['resource_count'] = self.resource_count
        if hasattr(self, 'first') and self.first is not None:
            _dict['first'] = self.first
        if hasattr(self, 'last') and self.last is not None:
            _dict['last'] = self.last
        if hasattr(self, 'prev') and self.prev is not None:
            _dict['prev'] = self.prev
        if hasattr(self, 'next') and self.next is not None:
            _dict['next'] = self.next
        if hasattr(self, 'resources') and self.resources is not None:
            _dict['resources'] = self.resources
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this NamespaceSearchResult object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'NamespaceSearchResult') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'NamespaceSearchResult') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ObjectAccess:
    """
    object access.

    :attr str id: (optional) unique id.
    :attr str account: (optional) account id.
    :attr str catalog_id: (optional) unique id.
    :attr str target_id: (optional) object id.
    :attr datetime create: (optional) date and time create.
    """

    def __init__(
        self,
        *,
        id: str = None,
        account: str = None,
        catalog_id: str = None,
        target_id: str = None,
        create: datetime = None
    ) -> None:
        """
        Initialize a ObjectAccess object.

        :param str id: (optional) unique id.
        :param str account: (optional) account id.
        :param str catalog_id: (optional) unique id.
        :param str target_id: (optional) object id.
        :param datetime create: (optional) date and time create.
        """
        self.id = id
        self.account = account
        self.catalog_id = catalog_id
        self.target_id = target_id
        self.create = create

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ObjectAccess':
        """Initialize a ObjectAccess object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'account' in _dict:
            args['account'] = _dict.get('account')
        if 'catalog_id' in _dict:
            args['catalog_id'] = _dict.get('catalog_id')
        if 'target_id' in _dict:
            args['target_id'] = _dict.get('target_id')
        if 'create' in _dict:
            args['create'] = string_to_datetime(_dict.get('create'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ObjectAccess object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'account') and self.account is not None:
            _dict['account'] = self.account
        if hasattr(self, 'catalog_id') and self.catalog_id is not None:
            _dict['catalog_id'] = self.catalog_id
        if hasattr(self, 'target_id') and self.target_id is not None:
            _dict['target_id'] = self.target_id
        if hasattr(self, 'create') and self.create is not None:
            _dict['create'] = datetime_to_string(self.create)
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ObjectAccess object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ObjectAccess') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ObjectAccess') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ObjectAccessListResult:
    """
    Paginated object search result.

    :attr int offset: The offset (origin 0) of the first resource in this page of
          search results.
    :attr int limit: The maximum number of resources returned in each page of search
          results.
    :attr int total_count: (optional) The overall total number of resources in the
          search result set.
    :attr int resource_count: (optional) The number of resources returned in this
          page of search results.
    :attr str first: (optional) A URL for retrieving the first page of search
          results.
    :attr str last: (optional) A URL for retrieving the last page of search results.
    :attr str prev: (optional) A URL for retrieving the previous page of search
          results.
    :attr str next: (optional) A URL for retrieving the next page of search results.
    :attr List[ObjectAccess] resources: (optional) Resulting objects.
    """

    def __init__(
        self,
        offset: int,
        limit: int,
        *,
        total_count: int = None,
        resource_count: int = None,
        first: str = None,
        last: str = None,
        prev: str = None,
        next: str = None,
        resources: List['ObjectAccess'] = None
    ) -> None:
        """
        Initialize a ObjectAccessListResult object.

        :param int offset: The offset (origin 0) of the first resource in this page
               of search results.
        :param int limit: The maximum number of resources returned in each page of
               search results.
        :param int total_count: (optional) The overall total number of resources in
               the search result set.
        :param int resource_count: (optional) The number of resources returned in
               this page of search results.
        :param str first: (optional) A URL for retrieving the first page of search
               results.
        :param str last: (optional) A URL for retrieving the last page of search
               results.
        :param str prev: (optional) A URL for retrieving the previous page of
               search results.
        :param str next: (optional) A URL for retrieving the next page of search
               results.
        :param List[ObjectAccess] resources: (optional) Resulting objects.
        """
        self.offset = offset
        self.limit = limit
        self.total_count = total_count
        self.resource_count = resource_count
        self.first = first
        self.last = last
        self.prev = prev
        self.next = next
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ObjectAccessListResult':
        """Initialize a ObjectAccessListResult object from a json dictionary."""
        args = {}
        if 'offset' in _dict:
            args['offset'] = _dict.get('offset')
        else:
            raise ValueError('Required property \'offset\' not present in ObjectAccessListResult JSON')
        if 'limit' in _dict:
            args['limit'] = _dict.get('limit')
        else:
            raise ValueError('Required property \'limit\' not present in ObjectAccessListResult JSON')
        if 'total_count' in _dict:
            args['total_count'] = _dict.get('total_count')
        if 'resource_count' in _dict:
            args['resource_count'] = _dict.get('resource_count')
        if 'first' in _dict:
            args['first'] = _dict.get('first')
        if 'last' in _dict:
            args['last'] = _dict.get('last')
        if 'prev' in _dict:
            args['prev'] = _dict.get('prev')
        if 'next' in _dict:
            args['next'] = _dict.get('next')
        if 'resources' in _dict:
            args['resources'] = [ObjectAccess.from_dict(x) for x in _dict.get('resources')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ObjectAccessListResult object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'total_count') and self.total_count is not None:
            _dict['total_count'] = self.total_count
        if hasattr(self, 'resource_count') and self.resource_count is not None:
            _dict['resource_count'] = self.resource_count
        if hasattr(self, 'first') and self.first is not None:
            _dict['first'] = self.first
        if hasattr(self, 'last') and self.last is not None:
            _dict['last'] = self.last
        if hasattr(self, 'prev') and self.prev is not None:
            _dict['prev'] = self.prev
        if hasattr(self, 'next') and self.next is not None:
            _dict['next'] = self.next
        if hasattr(self, 'resources') and self.resources is not None:
            _dict['resources'] = [x.to_dict() for x in self.resources]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ObjectAccessListResult object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ObjectAccessListResult') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ObjectAccessListResult') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ObjectListResult:
    """
    Paginated object search result.

    :attr int offset: The offset (origin 0) of the first resource in this page of
          search results.
    :attr int limit: The maximum number of resources returned in each page of search
          results.
    :attr int total_count: (optional) The overall total number of resources in the
          search result set.
    :attr int resource_count: (optional) The number of resources returned in this
          page of search results.
    :attr str first: (optional) A URL for retrieving the first page of search
          results.
    :attr str last: (optional) A URL for retrieving the last page of search results.
    :attr str prev: (optional) A URL for retrieving the previous page of search
          results.
    :attr str next: (optional) A URL for retrieving the next page of search results.
    :attr List[CatalogObject] resources: (optional) Resulting objects.
    """

    def __init__(
        self,
        offset: int,
        limit: int,
        *,
        total_count: int = None,
        resource_count: int = None,
        first: str = None,
        last: str = None,
        prev: str = None,
        next: str = None,
        resources: List['CatalogObject'] = None
    ) -> None:
        """
        Initialize a ObjectListResult object.

        :param int offset: The offset (origin 0) of the first resource in this page
               of search results.
        :param int limit: The maximum number of resources returned in each page of
               search results.
        :param int total_count: (optional) The overall total number of resources in
               the search result set.
        :param int resource_count: (optional) The number of resources returned in
               this page of search results.
        :param str first: (optional) A URL for retrieving the first page of search
               results.
        :param str last: (optional) A URL for retrieving the last page of search
               results.
        :param str prev: (optional) A URL for retrieving the previous page of
               search results.
        :param str next: (optional) A URL for retrieving the next page of search
               results.
        :param List[CatalogObject] resources: (optional) Resulting objects.
        """
        self.offset = offset
        self.limit = limit
        self.total_count = total_count
        self.resource_count = resource_count
        self.first = first
        self.last = last
        self.prev = prev
        self.next = next
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ObjectListResult':
        """Initialize a ObjectListResult object from a json dictionary."""
        args = {}
        if 'offset' in _dict:
            args['offset'] = _dict.get('offset')
        else:
            raise ValueError('Required property \'offset\' not present in ObjectListResult JSON')
        if 'limit' in _dict:
            args['limit'] = _dict.get('limit')
        else:
            raise ValueError('Required property \'limit\' not present in ObjectListResult JSON')
        if 'total_count' in _dict:
            args['total_count'] = _dict.get('total_count')
        if 'resource_count' in _dict:
            args['resource_count'] = _dict.get('resource_count')
        if 'first' in _dict:
            args['first'] = _dict.get('first')
        if 'last' in _dict:
            args['last'] = _dict.get('last')
        if 'prev' in _dict:
            args['prev'] = _dict.get('prev')
        if 'next' in _dict:
            args['next'] = _dict.get('next')
        if 'resources' in _dict:
            args['resources'] = [CatalogObject.from_dict(x) for x in _dict.get('resources')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ObjectListResult object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'total_count') and self.total_count is not None:
            _dict['total_count'] = self.total_count
        if hasattr(self, 'resource_count') and self.resource_count is not None:
            _dict['resource_count'] = self.resource_count
        if hasattr(self, 'first') and self.first is not None:
            _dict['first'] = self.first
        if hasattr(self, 'last') and self.last is not None:
            _dict['last'] = self.last
        if hasattr(self, 'prev') and self.prev is not None:
            _dict['prev'] = self.prev
        if hasattr(self, 'next') and self.next is not None:
            _dict['next'] = self.next
        if hasattr(self, 'resources') and self.resources is not None:
            _dict['resources'] = [x.to_dict() for x in self.resources]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ObjectListResult object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ObjectListResult') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ObjectListResult') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class ObjectSearchResult:
    """
    Paginated object search result.

    :attr int offset: The offset (origin 0) of the first resource in this page of
          search results.
    :attr int limit: The maximum number of resources returned in each page of search
          results.
    :attr int total_count: (optional) The overall total number of resources in the
          search result set.
    :attr int resource_count: (optional) The number of resources returned in this
          page of search results.
    :attr str first: (optional) A URL for retrieving the first page of search
          results.
    :attr str last: (optional) A URL for retrieving the last page of search results.
    :attr str prev: (optional) A URL for retrieving the previous page of search
          results.
    :attr str next: (optional) A URL for retrieving the next page of search results.
    :attr List[CatalogObject] resources: (optional) Resulting objects.
    """

    def __init__(
        self,
        offset: int,
        limit: int,
        *,
        total_count: int = None,
        resource_count: int = None,
        first: str = None,
        last: str = None,
        prev: str = None,
        next: str = None,
        resources: List['CatalogObject'] = None
    ) -> None:
        """
        Initialize a ObjectSearchResult object.

        :param int offset: The offset (origin 0) of the first resource in this page
               of search results.
        :param int limit: The maximum number of resources returned in each page of
               search results.
        :param int total_count: (optional) The overall total number of resources in
               the search result set.
        :param int resource_count: (optional) The number of resources returned in
               this page of search results.
        :param str first: (optional) A URL for retrieving the first page of search
               results.
        :param str last: (optional) A URL for retrieving the last page of search
               results.
        :param str prev: (optional) A URL for retrieving the previous page of
               search results.
        :param str next: (optional) A URL for retrieving the next page of search
               results.
        :param List[CatalogObject] resources: (optional) Resulting objects.
        """
        self.offset = offset
        self.limit = limit
        self.total_count = total_count
        self.resource_count = resource_count
        self.first = first
        self.last = last
        self.prev = prev
        self.next = next
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ObjectSearchResult':
        """Initialize a ObjectSearchResult object from a json dictionary."""
        args = {}
        if 'offset' in _dict:
            args['offset'] = _dict.get('offset')
        else:
            raise ValueError('Required property \'offset\' not present in ObjectSearchResult JSON')
        if 'limit' in _dict:
            args['limit'] = _dict.get('limit')
        else:
            raise ValueError('Required property \'limit\' not present in ObjectSearchResult JSON')
        if 'total_count' in _dict:
            args['total_count'] = _dict.get('total_count')
        if 'resource_count' in _dict:
            args['resource_count'] = _dict.get('resource_count')
        if 'first' in _dict:
            args['first'] = _dict.get('first')
        if 'last' in _dict:
            args['last'] = _dict.get('last')
        if 'prev' in _dict:
            args['prev'] = _dict.get('prev')
        if 'next' in _dict:
            args['next'] = _dict.get('next')
        if 'resources' in _dict:
            args['resources'] = [CatalogObject.from_dict(x) for x in _dict.get('resources')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ObjectSearchResult object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'total_count') and self.total_count is not None:
            _dict['total_count'] = self.total_count
        if hasattr(self, 'resource_count') and self.resource_count is not None:
            _dict['resource_count'] = self.resource_count
        if hasattr(self, 'first') and self.first is not None:
            _dict['first'] = self.first
        if hasattr(self, 'last') and self.last is not None:
            _dict['last'] = self.last
        if hasattr(self, 'prev') and self.prev is not None:
            _dict['prev'] = self.prev
        if hasattr(self, 'next') and self.next is not None:
            _dict['next'] = self.next
        if hasattr(self, 'resources') and self.resources is not None:
            _dict['resources'] = [x.to_dict() for x in self.resources]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ObjectSearchResult object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ObjectSearchResult') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ObjectSearchResult') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Offering:
    """
    Offering information.

    :attr str id: (optional) unique id.
    :attr str rev: (optional) Cloudant revision.
    :attr str url: (optional) The url for this specific offering.
    :attr str crn: (optional) The crn for this specific offering.
    :attr str label: (optional) Display Name in the requested language.
    :attr str name: (optional) The programmatic name of this offering.
    :attr str offering_icon_url: (optional) URL for an icon associated with this
          offering.
    :attr str offering_docs_url: (optional) URL for an additional docs with this
          offering.
    :attr str offering_support_url: (optional) [deprecated] - Use offering.support
          instead.  URL to be displayed in the Consumption UI for getting support on this
          offering.
    :attr List[str] tags: (optional) List of tags associated with this catalog.
    :attr List[str] keywords: (optional) List of keywords associated with offering,
          typically used to search for it.
    :attr Rating rating: (optional) Repository info for offerings.
    :attr datetime created: (optional) The date and time this catalog was created.
    :attr datetime updated: (optional) The date and time this catalog was last
          updated.
    :attr str short_description: (optional) Short description in the requested
          language.
    :attr str long_description: (optional) Long description in the requested
          language.
    :attr List[Feature] features: (optional) list of features associated with this
          offering.
    :attr List[Kind] kinds: (optional) Array of kind.
    :attr bool permit_request_ibm_public_publish: (optional) Is it permitted to
          request publishing to IBM or Public.
    :attr bool ibm_publish_approved: (optional) Indicates if this offering has been
          approved for use by all IBMers.
    :attr bool public_publish_approved: (optional) Indicates if this offering has
          been approved for use by all IBM Cloud users.
    :attr str public_original_crn: (optional) The original offering CRN that this
          publish entry came from.
    :attr str publish_public_crn: (optional) The crn of the public catalog entry of
          this offering.
    :attr str portal_approval_record: (optional) The portal's approval record ID.
    :attr str portal_ui_url: (optional) The portal UI URL.
    :attr str catalog_id: (optional) The id of the catalog containing this offering.
    :attr str catalog_name: (optional) The name of the catalog.
    :attr dict metadata: (optional) Map of metadata values for this offering.
    :attr str disclaimer: (optional) A disclaimer for this offering.
    :attr bool hidden: (optional) Determine if this offering should be displayed in
          the Consumption UI.
    :attr str provider: (optional) Deprecated - Provider of this offering.
    :attr ProviderInfo provider_info: (optional) Information on the provider for
          this offering, or omitted if no provider information is given.
    :attr RepoInfo repo_info: (optional) Repository info for offerings.
    :attr Support support: (optional) Offering Support information.
    :attr List[MediaItem] media: (optional) A list of media items related to this
          offering.
    """

    def __init__(
        self,
        *,
        id: str = None,
        rev: str = None,
        url: str = None,
        crn: str = None,
        label: str = None,
        name: str = None,
        offering_icon_url: str = None,
        offering_docs_url: str = None,
        offering_support_url: str = None,
        tags: List[str] = None,
        keywords: List[str] = None,
        rating: 'Rating' = None,
        created: datetime = None,
        updated: datetime = None,
        short_description: str = None,
        long_description: str = None,
        features: List['Feature'] = None,
        kinds: List['Kind'] = None,
        permit_request_ibm_public_publish: bool = None,
        ibm_publish_approved: bool = None,
        public_publish_approved: bool = None,
        public_original_crn: str = None,
        publish_public_crn: str = None,
        portal_approval_record: str = None,
        portal_ui_url: str = None,
        catalog_id: str = None,
        catalog_name: str = None,
        metadata: dict = None,
        disclaimer: str = None,
        hidden: bool = None,
        provider: str = None,
        provider_info: 'ProviderInfo' = None,
        repo_info: 'RepoInfo' = None,
        support: 'Support' = None,
        media: List['MediaItem'] = None
    ) -> None:
        """
        Initialize a Offering object.

        :param str id: (optional) unique id.
        :param str rev: (optional) Cloudant revision.
        :param str url: (optional) The url for this specific offering.
        :param str crn: (optional) The crn for this specific offering.
        :param str label: (optional) Display Name in the requested language.
        :param str name: (optional) The programmatic name of this offering.
        :param str offering_icon_url: (optional) URL for an icon associated with
               this offering.
        :param str offering_docs_url: (optional) URL for an additional docs with
               this offering.
        :param str offering_support_url: (optional) [deprecated] - Use
               offering.support instead.  URL to be displayed in the Consumption UI for
               getting support on this offering.
        :param List[str] tags: (optional) List of tags associated with this
               catalog.
        :param List[str] keywords: (optional) List of keywords associated with
               offering, typically used to search for it.
        :param Rating rating: (optional) Repository info for offerings.
        :param datetime created: (optional) The date and time this catalog was
               created.
        :param datetime updated: (optional) The date and time this catalog was last
               updated.
        :param str short_description: (optional) Short description in the requested
               language.
        :param str long_description: (optional) Long description in the requested
               language.
        :param List[Feature] features: (optional) list of features associated with
               this offering.
        :param List[Kind] kinds: (optional) Array of kind.
        :param bool permit_request_ibm_public_publish: (optional) Is it permitted
               to request publishing to IBM or Public.
        :param bool ibm_publish_approved: (optional) Indicates if this offering has
               been approved for use by all IBMers.
        :param bool public_publish_approved: (optional) Indicates if this offering
               has been approved for use by all IBM Cloud users.
        :param str public_original_crn: (optional) The original offering CRN that
               this publish entry came from.
        :param str publish_public_crn: (optional) The crn of the public catalog
               entry of this offering.
        :param str portal_approval_record: (optional) The portal's approval record
               ID.
        :param str portal_ui_url: (optional) The portal UI URL.
        :param str catalog_id: (optional) The id of the catalog containing this
               offering.
        :param str catalog_name: (optional) The name of the catalog.
        :param dict metadata: (optional) Map of metadata values for this offering.
        :param str disclaimer: (optional) A disclaimer for this offering.
        :param bool hidden: (optional) Determine if this offering should be
               displayed in the Consumption UI.
        :param str provider: (optional) Deprecated - Provider of this offering.
        :param ProviderInfo provider_info: (optional) Information on the provider
               for this offering, or omitted if no provider information is given.
        :param RepoInfo repo_info: (optional) Repository info for offerings.
        :param Support support: (optional) Offering Support information.
        :param List[MediaItem] media: (optional) A list of media items related to
               this offering.
        """
        self.id = id
        self.rev = rev
        self.url = url
        self.crn = crn
        self.label = label
        self.name = name
        self.offering_icon_url = offering_icon_url
        self.offering_docs_url = offering_docs_url
        self.offering_support_url = offering_support_url
        self.tags = tags
        self.keywords = keywords
        self.rating = rating
        self.created = created
        self.updated = updated
        self.short_description = short_description
        self.long_description = long_description
        self.features = features
        self.kinds = kinds
        self.permit_request_ibm_public_publish = permit_request_ibm_public_publish
        self.ibm_publish_approved = ibm_publish_approved
        self.public_publish_approved = public_publish_approved
        self.public_original_crn = public_original_crn
        self.publish_public_crn = publish_public_crn
        self.portal_approval_record = portal_approval_record
        self.portal_ui_url = portal_ui_url
        self.catalog_id = catalog_id
        self.catalog_name = catalog_name
        self.metadata = metadata
        self.disclaimer = disclaimer
        self.hidden = hidden
        self.provider = provider
        self.provider_info = provider_info
        self.repo_info = repo_info
        self.support = support
        self.media = media

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Offering':
        """Initialize a Offering object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if '_rev' in _dict:
            args['rev'] = _dict.get('_rev')
        if 'url' in _dict:
            args['url'] = _dict.get('url')
        if 'crn' in _dict:
            args['crn'] = _dict.get('crn')
        if 'label' in _dict:
            args['label'] = _dict.get('label')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'offering_icon_url' in _dict:
            args['offering_icon_url'] = _dict.get('offering_icon_url')
        if 'offering_docs_url' in _dict:
            args['offering_docs_url'] = _dict.get('offering_docs_url')
        if 'offering_support_url' in _dict:
            args['offering_support_url'] = _dict.get('offering_support_url')
        if 'tags' in _dict:
            args['tags'] = _dict.get('tags')
        if 'keywords' in _dict:
            args['keywords'] = _dict.get('keywords')
        if 'rating' in _dict:
            args['rating'] = Rating.from_dict(_dict.get('rating'))
        if 'created' in _dict:
            args['created'] = string_to_datetime(_dict.get('created'))
        if 'updated' in _dict:
            args['updated'] = string_to_datetime(_dict.get('updated'))
        if 'short_description' in _dict:
            args['short_description'] = _dict.get('short_description')
        if 'long_description' in _dict:
            args['long_description'] = _dict.get('long_description')
        if 'features' in _dict:
            args['features'] = [Feature.from_dict(x) for x in _dict.get('features')]
        if 'kinds' in _dict:
            args['kinds'] = [Kind.from_dict(x) for x in _dict.get('kinds')]
        if 'permit_request_ibm_public_publish' in _dict:
            args['permit_request_ibm_public_publish'] = _dict.get('permit_request_ibm_public_publish')
        if 'ibm_publish_approved' in _dict:
            args['ibm_publish_approved'] = _dict.get('ibm_publish_approved')
        if 'public_publish_approved' in _dict:
            args['public_publish_approved'] = _dict.get('public_publish_approved')
        if 'public_original_crn' in _dict:
            args['public_original_crn'] = _dict.get('public_original_crn')
        if 'publish_public_crn' in _dict:
            args['publish_public_crn'] = _dict.get('publish_public_crn')
        if 'portal_approval_record' in _dict:
            args['portal_approval_record'] = _dict.get('portal_approval_record')
        if 'portal_ui_url' in _dict:
            args['portal_ui_url'] = _dict.get('portal_ui_url')
        if 'catalog_id' in _dict:
            args['catalog_id'] = _dict.get('catalog_id')
        if 'catalog_name' in _dict:
            args['catalog_name'] = _dict.get('catalog_name')
        if 'metadata' in _dict:
            args['metadata'] = _dict.get('metadata')
        if 'disclaimer' in _dict:
            args['disclaimer'] = _dict.get('disclaimer')
        if 'hidden' in _dict:
            args['hidden'] = _dict.get('hidden')
        if 'provider' in _dict:
            args['provider'] = _dict.get('provider')
        if 'provider_info' in _dict:
            args['provider_info'] = ProviderInfo.from_dict(_dict.get('provider_info'))
        if 'repo_info' in _dict:
            args['repo_info'] = RepoInfo.from_dict(_dict.get('repo_info'))
        if 'support' in _dict:
            args['support'] = Support.from_dict(_dict.get('support'))
        if 'media' in _dict:
            args['media'] = [MediaItem.from_dict(x) for x in _dict.get('media')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Offering object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'rev') and self.rev is not None:
            _dict['_rev'] = self.rev
        if hasattr(self, 'url') and self.url is not None:
            _dict['url'] = self.url
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'label') and self.label is not None:
            _dict['label'] = self.label
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'offering_icon_url') and self.offering_icon_url is not None:
            _dict['offering_icon_url'] = self.offering_icon_url
        if hasattr(self, 'offering_docs_url') and self.offering_docs_url is not None:
            _dict['offering_docs_url'] = self.offering_docs_url
        if hasattr(self, 'offering_support_url') and self.offering_support_url is not None:
            _dict['offering_support_url'] = self.offering_support_url
        if hasattr(self, 'tags') and self.tags is not None:
            _dict['tags'] = self.tags
        if hasattr(self, 'keywords') and self.keywords is not None:
            _dict['keywords'] = self.keywords
        if hasattr(self, 'rating') and self.rating is not None:
            _dict['rating'] = self.rating.to_dict()
        if hasattr(self, 'created') and self.created is not None:
            _dict['created'] = datetime_to_string(self.created)
        if hasattr(self, 'updated') and self.updated is not None:
            _dict['updated'] = datetime_to_string(self.updated)
        if hasattr(self, 'short_description') and self.short_description is not None:
            _dict['short_description'] = self.short_description
        if hasattr(self, 'long_description') and self.long_description is not None:
            _dict['long_description'] = self.long_description
        if hasattr(self, 'features') and self.features is not None:
            _dict['features'] = [x.to_dict() for x in self.features]
        if hasattr(self, 'kinds') and self.kinds is not None:
            _dict['kinds'] = [x.to_dict() for x in self.kinds]
        if hasattr(self, 'permit_request_ibm_public_publish') and self.permit_request_ibm_public_publish is not None:
            _dict['permit_request_ibm_public_publish'] = self.permit_request_ibm_public_publish
        if hasattr(self, 'ibm_publish_approved') and self.ibm_publish_approved is not None:
            _dict['ibm_publish_approved'] = self.ibm_publish_approved
        if hasattr(self, 'public_publish_approved') and self.public_publish_approved is not None:
            _dict['public_publish_approved'] = self.public_publish_approved
        if hasattr(self, 'public_original_crn') and self.public_original_crn is not None:
            _dict['public_original_crn'] = self.public_original_crn
        if hasattr(self, 'publish_public_crn') and self.publish_public_crn is not None:
            _dict['publish_public_crn'] = self.publish_public_crn
        if hasattr(self, 'portal_approval_record') and self.portal_approval_record is not None:
            _dict['portal_approval_record'] = self.portal_approval_record
        if hasattr(self, 'portal_ui_url') and self.portal_ui_url is not None:
            _dict['portal_ui_url'] = self.portal_ui_url
        if hasattr(self, 'catalog_id') and self.catalog_id is not None:
            _dict['catalog_id'] = self.catalog_id
        if hasattr(self, 'catalog_name') and self.catalog_name is not None:
            _dict['catalog_name'] = self.catalog_name
        if hasattr(self, 'metadata') and self.metadata is not None:
            _dict['metadata'] = self.metadata
        if hasattr(self, 'disclaimer') and self.disclaimer is not None:
            _dict['disclaimer'] = self.disclaimer
        if hasattr(self, 'hidden') and self.hidden is not None:
            _dict['hidden'] = self.hidden
        if hasattr(self, 'provider') and self.provider is not None:
            _dict['provider'] = self.provider
        if hasattr(self, 'provider_info') and self.provider_info is not None:
            _dict['provider_info'] = self.provider_info.to_dict()
        if hasattr(self, 'repo_info') and self.repo_info is not None:
            _dict['repo_info'] = self.repo_info.to_dict()
        if hasattr(self, 'support') and self.support is not None:
            _dict['support'] = self.support.to_dict()
        if hasattr(self, 'media') and self.media is not None:
            _dict['media'] = [x.to_dict() for x in self.media]
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


class OfferingInstance:
    """
    A offering instance resource (provision instance of a catalog offering).

    :attr str id: (optional) provisioned instance ID (part of the CRN).
    :attr str rev: (optional) Cloudant revision.
    :attr str url: (optional) url reference to this object.
    :attr str crn: (optional) platform CRN for this instance.
    :attr str label: (optional) the label for this instance.
    :attr str catalog_id: (optional) Catalog ID this instance was created from.
    :attr str offering_id: (optional) Offering ID this instance was created from.
    :attr str kind_format: (optional) the format this instance has (helm, operator,
          ova...).
    :attr str version: (optional) The version this instance was installed from (not
          version id).
    :attr str cluster_id: (optional) Cluster ID.
    :attr str cluster_region: (optional) Cluster region (e.g., us-south).
    :attr List[str] cluster_namespaces: (optional) List of target namespaces to
          install into.
    :attr bool cluster_all_namespaces: (optional) designate to install into all
          namespaces.
    :attr str schematics_workspace_id: (optional) Id of the schematics workspace,
          for offering instances provisioned through schematics.
    :attr str resource_group_id: (optional) Id of the resource group to provision
          the offering instance into.
    :attr str install_plan: (optional) Type of install plan (also known as approval
          strategy) for operator subscriptions. Can be either automatic, which
          automatically upgrades operators to the latest in a channel, or manual, which
          requires approval on the cluster.
    :attr str channel: (optional) Channel to pin the operator subscription to.
    :attr dict metadata: (optional) Map of metadata values for this offering
          instance.
    :attr OfferingInstanceLastOperation last_operation: (optional) the last
          operation performed and status.
    """

    def __init__(
        self,
        *,
        id: str = None,
        rev: str = None,
        url: str = None,
        crn: str = None,
        label: str = None,
        catalog_id: str = None,
        offering_id: str = None,
        kind_format: str = None,
        version: str = None,
        cluster_id: str = None,
        cluster_region: str = None,
        cluster_namespaces: List[str] = None,
        cluster_all_namespaces: bool = None,
        schematics_workspace_id: str = None,
        resource_group_id: str = None,
        install_plan: str = None,
        channel: str = None,
        metadata: dict = None,
        last_operation: 'OfferingInstanceLastOperation' = None
    ) -> None:
        """
        Initialize a OfferingInstance object.

        :param str id: (optional) provisioned instance ID (part of the CRN).
        :param str rev: (optional) Cloudant revision.
        :param str url: (optional) url reference to this object.
        :param str crn: (optional) platform CRN for this instance.
        :param str label: (optional) the label for this instance.
        :param str catalog_id: (optional) Catalog ID this instance was created
               from.
        :param str offering_id: (optional) Offering ID this instance was created
               from.
        :param str kind_format: (optional) the format this instance has (helm,
               operator, ova...).
        :param str version: (optional) The version this instance was installed from
               (not version id).
        :param str cluster_id: (optional) Cluster ID.
        :param str cluster_region: (optional) Cluster region (e.g., us-south).
        :param List[str] cluster_namespaces: (optional) List of target namespaces
               to install into.
        :param bool cluster_all_namespaces: (optional) designate to install into
               all namespaces.
        :param str schematics_workspace_id: (optional) Id of the schematics
               workspace, for offering instances provisioned through schematics.
        :param str resource_group_id: (optional) Id of the resource group to
               provision the offering instance into.
        :param str install_plan: (optional) Type of install plan (also known as
               approval strategy) for operator subscriptions. Can be either automatic,
               which automatically upgrades operators to the latest in a channel, or
               manual, which requires approval on the cluster.
        :param str channel: (optional) Channel to pin the operator subscription to.
        :param dict metadata: (optional) Map of metadata values for this offering
               instance.
        :param OfferingInstanceLastOperation last_operation: (optional) the last
               operation performed and status.
        """
        self.id = id
        self.rev = rev
        self.url = url
        self.crn = crn
        self.label = label
        self.catalog_id = catalog_id
        self.offering_id = offering_id
        self.kind_format = kind_format
        self.version = version
        self.cluster_id = cluster_id
        self.cluster_region = cluster_region
        self.cluster_namespaces = cluster_namespaces
        self.cluster_all_namespaces = cluster_all_namespaces
        self.schematics_workspace_id = schematics_workspace_id
        self.resource_group_id = resource_group_id
        self.install_plan = install_plan
        self.channel = channel
        self.metadata = metadata
        self.last_operation = last_operation

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'OfferingInstance':
        """Initialize a OfferingInstance object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if '_rev' in _dict:
            args['rev'] = _dict.get('_rev')
        if 'url' in _dict:
            args['url'] = _dict.get('url')
        if 'crn' in _dict:
            args['crn'] = _dict.get('crn')
        if 'label' in _dict:
            args['label'] = _dict.get('label')
        if 'catalog_id' in _dict:
            args['catalog_id'] = _dict.get('catalog_id')
        if 'offering_id' in _dict:
            args['offering_id'] = _dict.get('offering_id')
        if 'kind_format' in _dict:
            args['kind_format'] = _dict.get('kind_format')
        if 'version' in _dict:
            args['version'] = _dict.get('version')
        if 'cluster_id' in _dict:
            args['cluster_id'] = _dict.get('cluster_id')
        if 'cluster_region' in _dict:
            args['cluster_region'] = _dict.get('cluster_region')
        if 'cluster_namespaces' in _dict:
            args['cluster_namespaces'] = _dict.get('cluster_namespaces')
        if 'cluster_all_namespaces' in _dict:
            args['cluster_all_namespaces'] = _dict.get('cluster_all_namespaces')
        if 'schematics_workspace_id' in _dict:
            args['schematics_workspace_id'] = _dict.get('schematics_workspace_id')
        if 'resource_group_id' in _dict:
            args['resource_group_id'] = _dict.get('resource_group_id')
        if 'install_plan' in _dict:
            args['install_plan'] = _dict.get('install_plan')
        if 'channel' in _dict:
            args['channel'] = _dict.get('channel')
        if 'metadata' in _dict:
            args['metadata'] = _dict.get('metadata')
        if 'last_operation' in _dict:
            args['last_operation'] = OfferingInstanceLastOperation.from_dict(_dict.get('last_operation'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a OfferingInstance object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'rev') and self.rev is not None:
            _dict['_rev'] = self.rev
        if hasattr(self, 'url') and self.url is not None:
            _dict['url'] = self.url
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'label') and self.label is not None:
            _dict['label'] = self.label
        if hasattr(self, 'catalog_id') and self.catalog_id is not None:
            _dict['catalog_id'] = self.catalog_id
        if hasattr(self, 'offering_id') and self.offering_id is not None:
            _dict['offering_id'] = self.offering_id
        if hasattr(self, 'kind_format') and self.kind_format is not None:
            _dict['kind_format'] = self.kind_format
        if hasattr(self, 'version') and self.version is not None:
            _dict['version'] = self.version
        if hasattr(self, 'cluster_id') and self.cluster_id is not None:
            _dict['cluster_id'] = self.cluster_id
        if hasattr(self, 'cluster_region') and self.cluster_region is not None:
            _dict['cluster_region'] = self.cluster_region
        if hasattr(self, 'cluster_namespaces') and self.cluster_namespaces is not None:
            _dict['cluster_namespaces'] = self.cluster_namespaces
        if hasattr(self, 'cluster_all_namespaces') and self.cluster_all_namespaces is not None:
            _dict['cluster_all_namespaces'] = self.cluster_all_namespaces
        if hasattr(self, 'schematics_workspace_id') and self.schematics_workspace_id is not None:
            _dict['schematics_workspace_id'] = self.schematics_workspace_id
        if hasattr(self, 'resource_group_id') and self.resource_group_id is not None:
            _dict['resource_group_id'] = self.resource_group_id
        if hasattr(self, 'install_plan') and self.install_plan is not None:
            _dict['install_plan'] = self.install_plan
        if hasattr(self, 'channel') and self.channel is not None:
            _dict['channel'] = self.channel
        if hasattr(self, 'metadata') and self.metadata is not None:
            _dict['metadata'] = self.metadata
        if hasattr(self, 'last_operation') and self.last_operation is not None:
            _dict['last_operation'] = self.last_operation.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this OfferingInstance object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'OfferingInstance') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'OfferingInstance') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class OfferingInstanceLastOperation:
    """
    the last operation performed and status.

    :attr str operation: (optional) last operation performed.
    :attr str state: (optional) state after the last operation performed.
    :attr str message: (optional) additional information about the last operation.
    :attr str transaction_id: (optional) transaction id from the last operation.
    :attr str updated: (optional) Date and time last updated.
    """

    def __init__(
        self,
        *,
        operation: str = None,
        state: str = None,
        message: str = None,
        transaction_id: str = None,
        updated: str = None
    ) -> None:
        """
        Initialize a OfferingInstanceLastOperation object.

        :param str operation: (optional) last operation performed.
        :param str state: (optional) state after the last operation performed.
        :param str message: (optional) additional information about the last
               operation.
        :param str transaction_id: (optional) transaction id from the last
               operation.
        :param str updated: (optional) Date and time last updated.
        """
        self.operation = operation
        self.state = state
        self.message = message
        self.transaction_id = transaction_id
        self.updated = updated

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'OfferingInstanceLastOperation':
        """Initialize a OfferingInstanceLastOperation object from a json dictionary."""
        args = {}
        if 'operation' in _dict:
            args['operation'] = _dict.get('operation')
        if 'state' in _dict:
            args['state'] = _dict.get('state')
        if 'message' in _dict:
            args['message'] = _dict.get('message')
        if 'transaction_id' in _dict:
            args['transaction_id'] = _dict.get('transaction_id')
        if 'updated' in _dict:
            args['updated'] = _dict.get('updated')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a OfferingInstanceLastOperation object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'operation') and self.operation is not None:
            _dict['operation'] = self.operation
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        if hasattr(self, 'message') and self.message is not None:
            _dict['message'] = self.message
        if hasattr(self, 'transaction_id') and self.transaction_id is not None:
            _dict['transaction_id'] = self.transaction_id
        if hasattr(self, 'updated') and self.updated is not None:
            _dict['updated'] = self.updated
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this OfferingInstanceLastOperation object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'OfferingInstanceLastOperation') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'OfferingInstanceLastOperation') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class OfferingSearchResult:
    """
    Paginated offering search result.

    :attr int offset: The offset (origin 0) of the first resource in this page of
          search results.
    :attr int limit: The maximum number of resources returned in each page of search
          results.
    :attr int total_count: (optional) The overall total number of resources in the
          search result set.
    :attr int resource_count: (optional) The number of resources returned in this
          page of search results.
    :attr str first: (optional) A URL for retrieving the first page of search
          results.
    :attr str last: (optional) A URL for retrieving the last page of search results.
    :attr str prev: (optional) A URL for retrieving the previous page of search
          results.
    :attr str next: (optional) A URL for retrieving the next page of search results.
    :attr List[Offering] resources: (optional) Resulting objects.
    """

    def __init__(
        self,
        offset: int,
        limit: int,
        *,
        total_count: int = None,
        resource_count: int = None,
        first: str = None,
        last: str = None,
        prev: str = None,
        next: str = None,
        resources: List['Offering'] = None
    ) -> None:
        """
        Initialize a OfferingSearchResult object.

        :param int offset: The offset (origin 0) of the first resource in this page
               of search results.
        :param int limit: The maximum number of resources returned in each page of
               search results.
        :param int total_count: (optional) The overall total number of resources in
               the search result set.
        :param int resource_count: (optional) The number of resources returned in
               this page of search results.
        :param str first: (optional) A URL for retrieving the first page of search
               results.
        :param str last: (optional) A URL for retrieving the last page of search
               results.
        :param str prev: (optional) A URL for retrieving the previous page of
               search results.
        :param str next: (optional) A URL for retrieving the next page of search
               results.
        :param List[Offering] resources: (optional) Resulting objects.
        """
        self.offset = offset
        self.limit = limit
        self.total_count = total_count
        self.resource_count = resource_count
        self.first = first
        self.last = last
        self.prev = prev
        self.next = next
        self.resources = resources

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'OfferingSearchResult':
        """Initialize a OfferingSearchResult object from a json dictionary."""
        args = {}
        if 'offset' in _dict:
            args['offset'] = _dict.get('offset')
        else:
            raise ValueError('Required property \'offset\' not present in OfferingSearchResult JSON')
        if 'limit' in _dict:
            args['limit'] = _dict.get('limit')
        else:
            raise ValueError('Required property \'limit\' not present in OfferingSearchResult JSON')
        if 'total_count' in _dict:
            args['total_count'] = _dict.get('total_count')
        if 'resource_count' in _dict:
            args['resource_count'] = _dict.get('resource_count')
        if 'first' in _dict:
            args['first'] = _dict.get('first')
        if 'last' in _dict:
            args['last'] = _dict.get('last')
        if 'prev' in _dict:
            args['prev'] = _dict.get('prev')
        if 'next' in _dict:
            args['next'] = _dict.get('next')
        if 'resources' in _dict:
            args['resources'] = [Offering.from_dict(x) for x in _dict.get('resources')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a OfferingSearchResult object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'offset') and self.offset is not None:
            _dict['offset'] = self.offset
        if hasattr(self, 'limit') and self.limit is not None:
            _dict['limit'] = self.limit
        if hasattr(self, 'total_count') and self.total_count is not None:
            _dict['total_count'] = self.total_count
        if hasattr(self, 'resource_count') and self.resource_count is not None:
            _dict['resource_count'] = self.resource_count
        if hasattr(self, 'first') and self.first is not None:
            _dict['first'] = self.first
        if hasattr(self, 'last') and self.last is not None:
            _dict['last'] = self.last
        if hasattr(self, 'prev') and self.prev is not None:
            _dict['prev'] = self.prev
        if hasattr(self, 'next') and self.next is not None:
            _dict['next'] = self.next
        if hasattr(self, 'resources') and self.resources is not None:
            _dict['resources'] = [x.to_dict() for x in self.resources]
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this OfferingSearchResult object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'OfferingSearchResult') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'OfferingSearchResult') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class OperatorDeployResult:
    """
    Operator deploy result.

    :attr str phase: (optional) Status phase.
    :attr str message: (optional) Status message.
    :attr str link: (optional) Operator API path.
    :attr str name: (optional) Name of Operator.
    :attr str version: (optional) Operator version.
    :attr str namespace: (optional) Kube namespace.
    :attr str package_name: (optional) Package Operator exists in.
    :attr str catalog_id: (optional) Catalog identification.
    """

    def __init__(
        self,
        *,
        phase: str = None,
        message: str = None,
        link: str = None,
        name: str = None,
        version: str = None,
        namespace: str = None,
        package_name: str = None,
        catalog_id: str = None
    ) -> None:
        """
        Initialize a OperatorDeployResult object.

        :param str phase: (optional) Status phase.
        :param str message: (optional) Status message.
        :param str link: (optional) Operator API path.
        :param str name: (optional) Name of Operator.
        :param str version: (optional) Operator version.
        :param str namespace: (optional) Kube namespace.
        :param str package_name: (optional) Package Operator exists in.
        :param str catalog_id: (optional) Catalog identification.
        """
        self.phase = phase
        self.message = message
        self.link = link
        self.name = name
        self.version = version
        self.namespace = namespace
        self.package_name = package_name
        self.catalog_id = catalog_id

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'OperatorDeployResult':
        """Initialize a OperatorDeployResult object from a json dictionary."""
        args = {}
        if 'phase' in _dict:
            args['phase'] = _dict.get('phase')
        if 'message' in _dict:
            args['message'] = _dict.get('message')
        if 'link' in _dict:
            args['link'] = _dict.get('link')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'version' in _dict:
            args['version'] = _dict.get('version')
        if 'namespace' in _dict:
            args['namespace'] = _dict.get('namespace')
        if 'package_name' in _dict:
            args['package_name'] = _dict.get('package_name')
        if 'catalog_id' in _dict:
            args['catalog_id'] = _dict.get('catalog_id')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a OperatorDeployResult object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'phase') and self.phase is not None:
            _dict['phase'] = self.phase
        if hasattr(self, 'message') and self.message is not None:
            _dict['message'] = self.message
        if hasattr(self, 'link') and self.link is not None:
            _dict['link'] = self.link
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'version') and self.version is not None:
            _dict['version'] = self.version
        if hasattr(self, 'namespace') and self.namespace is not None:
            _dict['namespace'] = self.namespace
        if hasattr(self, 'package_name') and self.package_name is not None:
            _dict['package_name'] = self.package_name
        if hasattr(self, 'catalog_id') and self.catalog_id is not None:
            _dict['catalog_id'] = self.catalog_id
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this OperatorDeployResult object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'OperatorDeployResult') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'OperatorDeployResult') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Plan:
    """
    Offering plan.

    :attr str id: (optional) unique id.
    :attr str label: (optional) Display Name in the requested language.
    :attr str name: (optional) The programmatic name of this offering.
    :attr str short_description: (optional) Short description in the requested
          language.
    :attr str long_description: (optional) Long description in the requested
          language.
    :attr dict metadata: (optional) open ended metadata information.
    :attr List[str] tags: (optional) list of tags associated with this catalog.
    :attr List[Feature] additional_features: (optional) list of features associated
          with this offering.
    :attr datetime created: (optional) the date'time this catalog was created.
    :attr datetime updated: (optional) the date'time this catalog was last updated.
    :attr List[Deployment] deployments: (optional) list of deployments.
    """

    def __init__(
        self,
        *,
        id: str = None,
        label: str = None,
        name: str = None,
        short_description: str = None,
        long_description: str = None,
        metadata: dict = None,
        tags: List[str] = None,
        additional_features: List['Feature'] = None,
        created: datetime = None,
        updated: datetime = None,
        deployments: List['Deployment'] = None
    ) -> None:
        """
        Initialize a Plan object.

        :param str id: (optional) unique id.
        :param str label: (optional) Display Name in the requested language.
        :param str name: (optional) The programmatic name of this offering.
        :param str short_description: (optional) Short description in the requested
               language.
        :param str long_description: (optional) Long description in the requested
               language.
        :param dict metadata: (optional) open ended metadata information.
        :param List[str] tags: (optional) list of tags associated with this
               catalog.
        :param List[Feature] additional_features: (optional) list of features
               associated with this offering.
        :param datetime created: (optional) the date'time this catalog was created.
        :param datetime updated: (optional) the date'time this catalog was last
               updated.
        :param List[Deployment] deployments: (optional) list of deployments.
        """
        self.id = id
        self.label = label
        self.name = name
        self.short_description = short_description
        self.long_description = long_description
        self.metadata = metadata
        self.tags = tags
        self.additional_features = additional_features
        self.created = created
        self.updated = updated
        self.deployments = deployments

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Plan':
        """Initialize a Plan object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'label' in _dict:
            args['label'] = _dict.get('label')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'short_description' in _dict:
            args['short_description'] = _dict.get('short_description')
        if 'long_description' in _dict:
            args['long_description'] = _dict.get('long_description')
        if 'metadata' in _dict:
            args['metadata'] = _dict.get('metadata')
        if 'tags' in _dict:
            args['tags'] = _dict.get('tags')
        if 'additional_features' in _dict:
            args['additional_features'] = [Feature.from_dict(x) for x in _dict.get('additional_features')]
        if 'created' in _dict:
            args['created'] = string_to_datetime(_dict.get('created'))
        if 'updated' in _dict:
            args['updated'] = string_to_datetime(_dict.get('updated'))
        if 'deployments' in _dict:
            args['deployments'] = [Deployment.from_dict(x) for x in _dict.get('deployments')]
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Plan object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'label') and self.label is not None:
            _dict['label'] = self.label
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'short_description') and self.short_description is not None:
            _dict['short_description'] = self.short_description
        if hasattr(self, 'long_description') and self.long_description is not None:
            _dict['long_description'] = self.long_description
        if hasattr(self, 'metadata') and self.metadata is not None:
            _dict['metadata'] = self.metadata
        if hasattr(self, 'tags') and self.tags is not None:
            _dict['tags'] = self.tags
        if hasattr(self, 'additional_features') and self.additional_features is not None:
            _dict['additional_features'] = [x.to_dict() for x in self.additional_features]
        if hasattr(self, 'created') and self.created is not None:
            _dict['created'] = datetime_to_string(self.created)
        if hasattr(self, 'updated') and self.updated is not None:
            _dict['updated'] = datetime_to_string(self.updated)
        if hasattr(self, 'deployments') and self.deployments is not None:
            _dict['deployments'] = [x.to_dict() for x in self.deployments]
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


class ProviderInfo:
    """
    Information on the provider for this offering, or omitted if no provider information
    is given.

    :attr str id: (optional) The id of this provider.
    :attr str name: (optional) The name of this provider.
    """

    def __init__(self, *, id: str = None, name: str = None) -> None:
        """
        Initialize a ProviderInfo object.

        :param str id: (optional) The id of this provider.
        :param str name: (optional) The name of this provider.
        """
        self.id = id
        self.name = name

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'ProviderInfo':
        """Initialize a ProviderInfo object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a ProviderInfo object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this ProviderInfo object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'ProviderInfo') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'ProviderInfo') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class PublishObject:
    """
    Publish information.

    :attr bool permit_ibm_public_publish: (optional) Is it permitted to request
          publishing to IBM or Public.
    :attr bool ibm_approved: (optional) Indicates if this offering has been approved
          for use by all IBMers.
    :attr bool public_approved: (optional) Indicates if this offering has been
          approved for use by all IBM Cloud users.
    :attr str portal_approval_record: (optional) The portal's approval record ID.
    :attr str portal_url: (optional) The portal UI URL.
    """

    def __init__(
        self,
        *,
        permit_ibm_public_publish: bool = None,
        ibm_approved: bool = None,
        public_approved: bool = None,
        portal_approval_record: str = None,
        portal_url: str = None
    ) -> None:
        """
        Initialize a PublishObject object.

        :param bool permit_ibm_public_publish: (optional) Is it permitted to
               request publishing to IBM or Public.
        :param bool ibm_approved: (optional) Indicates if this offering has been
               approved for use by all IBMers.
        :param bool public_approved: (optional) Indicates if this offering has been
               approved for use by all IBM Cloud users.
        :param str portal_approval_record: (optional) The portal's approval record
               ID.
        :param str portal_url: (optional) The portal UI URL.
        """
        self.permit_ibm_public_publish = permit_ibm_public_publish
        self.ibm_approved = ibm_approved
        self.public_approved = public_approved
        self.portal_approval_record = portal_approval_record
        self.portal_url = portal_url

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'PublishObject':
        """Initialize a PublishObject object from a json dictionary."""
        args = {}
        if 'permit_ibm_public_publish' in _dict:
            args['permit_ibm_public_publish'] = _dict.get('permit_ibm_public_publish')
        if 'ibm_approved' in _dict:
            args['ibm_approved'] = _dict.get('ibm_approved')
        if 'public_approved' in _dict:
            args['public_approved'] = _dict.get('public_approved')
        if 'portal_approval_record' in _dict:
            args['portal_approval_record'] = _dict.get('portal_approval_record')
        if 'portal_url' in _dict:
            args['portal_url'] = _dict.get('portal_url')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a PublishObject object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'permit_ibm_public_publish') and self.permit_ibm_public_publish is not None:
            _dict['permit_ibm_public_publish'] = self.permit_ibm_public_publish
        if hasattr(self, 'ibm_approved') and self.ibm_approved is not None:
            _dict['ibm_approved'] = self.ibm_approved
        if hasattr(self, 'public_approved') and self.public_approved is not None:
            _dict['public_approved'] = self.public_approved
        if hasattr(self, 'portal_approval_record') and self.portal_approval_record is not None:
            _dict['portal_approval_record'] = self.portal_approval_record
        if hasattr(self, 'portal_url') and self.portal_url is not None:
            _dict['portal_url'] = self.portal_url
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this PublishObject object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'PublishObject') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'PublishObject') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Rating:
    """
    Repository info for offerings.

    :attr int one_star_count: (optional) One start rating.
    :attr int two_star_count: (optional) Two start rating.
    :attr int three_star_count: (optional) Three start rating.
    :attr int four_star_count: (optional) Four start rating.
    """

    def __init__(
        self,
        *,
        one_star_count: int = None,
        two_star_count: int = None,
        three_star_count: int = None,
        four_star_count: int = None
    ) -> None:
        """
        Initialize a Rating object.

        :param int one_star_count: (optional) One start rating.
        :param int two_star_count: (optional) Two start rating.
        :param int three_star_count: (optional) Three start rating.
        :param int four_star_count: (optional) Four start rating.
        """
        self.one_star_count = one_star_count
        self.two_star_count = two_star_count
        self.three_star_count = three_star_count
        self.four_star_count = four_star_count

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Rating':
        """Initialize a Rating object from a json dictionary."""
        args = {}
        if 'one_star_count' in _dict:
            args['one_star_count'] = _dict.get('one_star_count')
        if 'two_star_count' in _dict:
            args['two_star_count'] = _dict.get('two_star_count')
        if 'three_star_count' in _dict:
            args['three_star_count'] = _dict.get('three_star_count')
        if 'four_star_count' in _dict:
            args['four_star_count'] = _dict.get('four_star_count')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Rating object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'one_star_count') and self.one_star_count is not None:
            _dict['one_star_count'] = self.one_star_count
        if hasattr(self, 'two_star_count') and self.two_star_count is not None:
            _dict['two_star_count'] = self.two_star_count
        if hasattr(self, 'three_star_count') and self.three_star_count is not None:
            _dict['three_star_count'] = self.three_star_count
        if hasattr(self, 'four_star_count') and self.four_star_count is not None:
            _dict['four_star_count'] = self.four_star_count
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Rating object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Rating') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Rating') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class RepoInfo:
    """
    Repository info for offerings.

    :attr str token: (optional) Token for private repos.
    :attr str type: (optional) Public or enterprise GitHub.
    """

    def __init__(self, *, token: str = None, type: str = None) -> None:
        """
        Initialize a RepoInfo object.

        :param str token: (optional) Token for private repos.
        :param str type: (optional) Public or enterprise GitHub.
        """
        self.token = token
        self.type = type

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'RepoInfo':
        """Initialize a RepoInfo object from a json dictionary."""
        args = {}
        if 'token' in _dict:
            args['token'] = _dict.get('token')
        if 'type' in _dict:
            args['type'] = _dict.get('type')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a RepoInfo object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'token') and self.token is not None:
            _dict['token'] = self.token
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this RepoInfo object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'RepoInfo') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'RepoInfo') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Resource:
    """
    Resource requirements.

    :attr str type: (optional) Type of requirement.
    :attr object value: (optional) mem, disk, cores, and nodes can be parsed as an
          int.  targetVersion will be a semver range value.
    """

    def __init__(self, *, type: str = None, value: object = None) -> None:
        """
        Initialize a Resource object.

        :param str type: (optional) Type of requirement.
        :param object value: (optional) mem, disk, cores, and nodes can be parsed
               as an int.  targetVersion will be a semver range value.
        """
        self.type = type
        self.value = value

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Resource':
        """Initialize a Resource object from a json dictionary."""
        args = {}
        if 'type' in _dict:
            args['type'] = _dict.get('type')
        if 'value' in _dict:
            args['value'] = _dict.get('value')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Resource object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'value') and self.value is not None:
            _dict['value'] = self.value
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

    class TypeEnum(str, Enum):
        """
        Type of requirement.
        """

        MEM = 'mem'
        DISK = 'disk'
        CORES = 'cores'
        TARGETVERSION = 'targetVersion'
        NODES = 'nodes'


class Script:
    """
    Script information.

    :attr str instructions: (optional) Instruction on step and by whom (role) that
          are needed to take place to prepare the target for installing this version.
    :attr str script: (optional) Optional script that needs to be run post any
          pre-condition script.
    :attr str script_permission: (optional) Optional iam permissions that are
          required on the target cluster to run this script.
    :attr str delete_script: (optional) Optional script that if run will remove the
          installed version.
    :attr str scope: (optional) Optional value indicating if this script is scoped
          to a namespace or the entire cluster.
    """

    def __init__(
        self,
        *,
        instructions: str = None,
        script: str = None,
        script_permission: str = None,
        delete_script: str = None,
        scope: str = None
    ) -> None:
        """
        Initialize a Script object.

        :param str instructions: (optional) Instruction on step and by whom (role)
               that are needed to take place to prepare the target for installing this
               version.
        :param str script: (optional) Optional script that needs to be run post any
               pre-condition script.
        :param str script_permission: (optional) Optional iam permissions that are
               required on the target cluster to run this script.
        :param str delete_script: (optional) Optional script that if run will
               remove the installed version.
        :param str scope: (optional) Optional value indicating if this script is
               scoped to a namespace or the entire cluster.
        """
        self.instructions = instructions
        self.script = script
        self.script_permission = script_permission
        self.delete_script = delete_script
        self.scope = scope

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Script':
        """Initialize a Script object from a json dictionary."""
        args = {}
        if 'instructions' in _dict:
            args['instructions'] = _dict.get('instructions')
        if 'script' in _dict:
            args['script'] = _dict.get('script')
        if 'script_permission' in _dict:
            args['script_permission'] = _dict.get('script_permission')
        if 'delete_script' in _dict:
            args['delete_script'] = _dict.get('delete_script')
        if 'scope' in _dict:
            args['scope'] = _dict.get('scope')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Script object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'instructions') and self.instructions is not None:
            _dict['instructions'] = self.instructions
        if hasattr(self, 'script') and self.script is not None:
            _dict['script'] = self.script
        if hasattr(self, 'script_permission') and self.script_permission is not None:
            _dict['script_permission'] = self.script_permission
        if hasattr(self, 'delete_script') and self.delete_script is not None:
            _dict['delete_script'] = self.delete_script
        if hasattr(self, 'scope') and self.scope is not None:
            _dict['scope'] = self.scope
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Script object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Script') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Script') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class State:
    """
    Offering state.

    :attr str current: (optional) one of: new, validated, account-published,
          ibm-published, public-published.
    :attr datetime current_entered: (optional) Date and time of current request.
    :attr str pending: (optional) one of: new, validated, account-published,
          ibm-published, public-published.
    :attr datetime pending_requested: (optional) Date and time of pending request.
    :attr str previous: (optional) one of: new, validated, account-published,
          ibm-published, public-published.
    """

    def __init__(
        self,
        *,
        current: str = None,
        current_entered: datetime = None,
        pending: str = None,
        pending_requested: datetime = None,
        previous: str = None
    ) -> None:
        """
        Initialize a State object.

        :param str current: (optional) one of: new, validated, account-published,
               ibm-published, public-published.
        :param datetime current_entered: (optional) Date and time of current
               request.
        :param str pending: (optional) one of: new, validated, account-published,
               ibm-published, public-published.
        :param datetime pending_requested: (optional) Date and time of pending
               request.
        :param str previous: (optional) one of: new, validated, account-published,
               ibm-published, public-published.
        """
        self.current = current
        self.current_entered = current_entered
        self.pending = pending
        self.pending_requested = pending_requested
        self.previous = previous

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'State':
        """Initialize a State object from a json dictionary."""
        args = {}
        if 'current' in _dict:
            args['current'] = _dict.get('current')
        if 'current_entered' in _dict:
            args['current_entered'] = string_to_datetime(_dict.get('current_entered'))
        if 'pending' in _dict:
            args['pending'] = _dict.get('pending')
        if 'pending_requested' in _dict:
            args['pending_requested'] = string_to_datetime(_dict.get('pending_requested'))
        if 'previous' in _dict:
            args['previous'] = _dict.get('previous')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a State object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'current') and self.current is not None:
            _dict['current'] = self.current
        if hasattr(self, 'current_entered') and self.current_entered is not None:
            _dict['current_entered'] = datetime_to_string(self.current_entered)
        if hasattr(self, 'pending') and self.pending is not None:
            _dict['pending'] = self.pending
        if hasattr(self, 'pending_requested') and self.pending_requested is not None:
            _dict['pending_requested'] = datetime_to_string(self.pending_requested)
        if hasattr(self, 'previous') and self.previous is not None:
            _dict['previous'] = self.previous
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this State object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'State') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'State') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Support:
    """
    Offering Support information.

    :attr str url: (optional) URL to be displayed in the Consumption UI for getting
          support on this offering.
    :attr str process: (optional) Support process as provided by an ISV.
    :attr List[str] locations: (optional) A list of country codes indicating where
          support is provided.
    """

    def __init__(self, *, url: str = None, process: str = None, locations: List[str] = None) -> None:
        """
        Initialize a Support object.

        :param str url: (optional) URL to be displayed in the Consumption UI for
               getting support on this offering.
        :param str process: (optional) Support process as provided by an ISV.
        :param List[str] locations: (optional) A list of country codes indicating
               where support is provided.
        """
        self.url = url
        self.process = process
        self.locations = locations

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Support':
        """Initialize a Support object from a json dictionary."""
        args = {}
        if 'url' in _dict:
            args['url'] = _dict.get('url')
        if 'process' in _dict:
            args['process'] = _dict.get('process')
        if 'locations' in _dict:
            args['locations'] = _dict.get('locations')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Support object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'url') and self.url is not None:
            _dict['url'] = self.url
        if hasattr(self, 'process') and self.process is not None:
            _dict['process'] = self.process
        if hasattr(self, 'locations') and self.locations is not None:
            _dict['locations'] = self.locations
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Support object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Support') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Support') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class SyndicationAuthorization:
    """
    Feature information.

    :attr str token: (optional) Array of syndicated namespaces.
    :attr datetime last_run: (optional) Date and time last updated.
    """

    def __init__(self, *, token: str = None, last_run: datetime = None) -> None:
        """
        Initialize a SyndicationAuthorization object.

        :param str token: (optional) Array of syndicated namespaces.
        :param datetime last_run: (optional) Date and time last updated.
        """
        self.token = token
        self.last_run = last_run

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SyndicationAuthorization':
        """Initialize a SyndicationAuthorization object from a json dictionary."""
        args = {}
        if 'token' in _dict:
            args['token'] = _dict.get('token')
        if 'last_run' in _dict:
            args['last_run'] = string_to_datetime(_dict.get('last_run'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SyndicationAuthorization object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'token') and self.token is not None:
            _dict['token'] = self.token
        if hasattr(self, 'last_run') and self.last_run is not None:
            _dict['last_run'] = datetime_to_string(self.last_run)
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this SyndicationAuthorization object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SyndicationAuthorization') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SyndicationAuthorization') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class SyndicationCluster:
    """
    Feature information.

    :attr str region: (optional) Cluster region.
    :attr str id: (optional) Cluster ID.
    :attr str name: (optional) Cluster name.
    :attr str resource_group_name: (optional) Resource group ID.
    :attr str type: (optional) Syndication type.
    :attr List[str] namespaces: (optional) Syndicated namespaces.
    :attr bool all_namespaces: (optional) Syndicated to all namespaces on cluster.
    """

    def __init__(
        self,
        *,
        region: str = None,
        id: str = None,
        name: str = None,
        resource_group_name: str = None,
        type: str = None,
        namespaces: List[str] = None,
        all_namespaces: bool = None
    ) -> None:
        """
        Initialize a SyndicationCluster object.

        :param str region: (optional) Cluster region.
        :param str id: (optional) Cluster ID.
        :param str name: (optional) Cluster name.
        :param str resource_group_name: (optional) Resource group ID.
        :param str type: (optional) Syndication type.
        :param List[str] namespaces: (optional) Syndicated namespaces.
        :param bool all_namespaces: (optional) Syndicated to all namespaces on
               cluster.
        """
        self.region = region
        self.id = id
        self.name = name
        self.resource_group_name = resource_group_name
        self.type = type
        self.namespaces = namespaces
        self.all_namespaces = all_namespaces

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SyndicationCluster':
        """Initialize a SyndicationCluster object from a json dictionary."""
        args = {}
        if 'region' in _dict:
            args['region'] = _dict.get('region')
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if 'name' in _dict:
            args['name'] = _dict.get('name')
        if 'resource_group_name' in _dict:
            args['resource_group_name'] = _dict.get('resource_group_name')
        if 'type' in _dict:
            args['type'] = _dict.get('type')
        if 'namespaces' in _dict:
            args['namespaces'] = _dict.get('namespaces')
        if 'all_namespaces' in _dict:
            args['all_namespaces'] = _dict.get('all_namespaces')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SyndicationCluster object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'region') and self.region is not None:
            _dict['region'] = self.region
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'name') and self.name is not None:
            _dict['name'] = self.name
        if hasattr(self, 'resource_group_name') and self.resource_group_name is not None:
            _dict['resource_group_name'] = self.resource_group_name
        if hasattr(self, 'type') and self.type is not None:
            _dict['type'] = self.type
        if hasattr(self, 'namespaces') and self.namespaces is not None:
            _dict['namespaces'] = self.namespaces
        if hasattr(self, 'all_namespaces') and self.all_namespaces is not None:
            _dict['all_namespaces'] = self.all_namespaces
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this SyndicationCluster object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SyndicationCluster') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SyndicationCluster') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class SyndicationHistory:
    """
    Feature information.

    :attr List[str] namespaces: (optional) Array of syndicated namespaces.
    :attr List[SyndicationCluster] clusters: (optional) Array of syndicated
          namespaces.
    :attr datetime last_run: (optional) Date and time last syndicated.
    """

    def __init__(
        self, *, namespaces: List[str] = None, clusters: List['SyndicationCluster'] = None, last_run: datetime = None
    ) -> None:
        """
        Initialize a SyndicationHistory object.

        :param List[str] namespaces: (optional) Array of syndicated namespaces.
        :param List[SyndicationCluster] clusters: (optional) Array of syndicated
               namespaces.
        :param datetime last_run: (optional) Date and time last syndicated.
        """
        self.namespaces = namespaces
        self.clusters = clusters
        self.last_run = last_run

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SyndicationHistory':
        """Initialize a SyndicationHistory object from a json dictionary."""
        args = {}
        if 'namespaces' in _dict:
            args['namespaces'] = _dict.get('namespaces')
        if 'clusters' in _dict:
            args['clusters'] = [SyndicationCluster.from_dict(x) for x in _dict.get('clusters')]
        if 'last_run' in _dict:
            args['last_run'] = string_to_datetime(_dict.get('last_run'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SyndicationHistory object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'namespaces') and self.namespaces is not None:
            _dict['namespaces'] = self.namespaces
        if hasattr(self, 'clusters') and self.clusters is not None:
            _dict['clusters'] = [x.to_dict() for x in self.clusters]
        if hasattr(self, 'last_run') and self.last_run is not None:
            _dict['last_run'] = datetime_to_string(self.last_run)
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this SyndicationHistory object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SyndicationHistory') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SyndicationHistory') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class SyndicationResource:
    """
    Feature information.

    :attr bool remove_related_components: (optional) Remove related components.
    :attr List[SyndicationCluster] clusters: (optional) Syndication clusters.
    :attr SyndicationHistory history: (optional) Feature information.
    :attr SyndicationAuthorization authorization: (optional) Feature information.
    """

    def __init__(
        self,
        *,
        remove_related_components: bool = None,
        clusters: List['SyndicationCluster'] = None,
        history: 'SyndicationHistory' = None,
        authorization: 'SyndicationAuthorization' = None
    ) -> None:
        """
        Initialize a SyndicationResource object.

        :param bool remove_related_components: (optional) Remove related
               components.
        :param List[SyndicationCluster] clusters: (optional) Syndication clusters.
        :param SyndicationHistory history: (optional) Feature information.
        :param SyndicationAuthorization authorization: (optional) Feature
               information.
        """
        self.remove_related_components = remove_related_components
        self.clusters = clusters
        self.history = history
        self.authorization = authorization

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'SyndicationResource':
        """Initialize a SyndicationResource object from a json dictionary."""
        args = {}
        if 'remove_related_components' in _dict:
            args['remove_related_components'] = _dict.get('remove_related_components')
        if 'clusters' in _dict:
            args['clusters'] = [SyndicationCluster.from_dict(x) for x in _dict.get('clusters')]
        if 'history' in _dict:
            args['history'] = SyndicationHistory.from_dict(_dict.get('history'))
        if 'authorization' in _dict:
            args['authorization'] = SyndicationAuthorization.from_dict(_dict.get('authorization'))
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a SyndicationResource object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'remove_related_components') and self.remove_related_components is not None:
            _dict['remove_related_components'] = self.remove_related_components
        if hasattr(self, 'clusters') and self.clusters is not None:
            _dict['clusters'] = [x.to_dict() for x in self.clusters]
        if hasattr(self, 'history') and self.history is not None:
            _dict['history'] = self.history.to_dict()
        if hasattr(self, 'authorization') and self.authorization is not None:
            _dict['authorization'] = self.authorization.to_dict()
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this SyndicationResource object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'SyndicationResource') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'SyndicationResource') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Validation:
    """
    Validation response.

    :attr datetime validated: (optional) Date and time of last successful
          validation.
    :attr datetime requested: (optional) Date and time of last validation was
          requested.
    :attr str state: (optional) Current validation state - <empty>, in_progress,
          valid, invalid, expired.
    :attr str last_operation: (optional) Last operation (e.g. submit_deployment,
          generate_installer, install_offering.
    :attr dict target: (optional) Validation target information (e.g. cluster_id,
          region, namespace, etc).  Values will vary by Content type.
    """

    def __init__(
        self,
        *,
        validated: datetime = None,
        requested: datetime = None,
        state: str = None,
        last_operation: str = None,
        target: dict = None
    ) -> None:
        """
        Initialize a Validation object.

        :param datetime validated: (optional) Date and time of last successful
               validation.
        :param datetime requested: (optional) Date and time of last validation was
               requested.
        :param str state: (optional) Current validation state - <empty>,
               in_progress, valid, invalid, expired.
        :param str last_operation: (optional) Last operation (e.g.
               submit_deployment, generate_installer, install_offering.
        :param dict target: (optional) Validation target information (e.g.
               cluster_id, region, namespace, etc).  Values will vary by Content type.
        """
        self.validated = validated
        self.requested = requested
        self.state = state
        self.last_operation = last_operation
        self.target = target

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Validation':
        """Initialize a Validation object from a json dictionary."""
        args = {}
        if 'validated' in _dict:
            args['validated'] = string_to_datetime(_dict.get('validated'))
        if 'requested' in _dict:
            args['requested'] = string_to_datetime(_dict.get('requested'))
        if 'state' in _dict:
            args['state'] = _dict.get('state')
        if 'last_operation' in _dict:
            args['last_operation'] = _dict.get('last_operation')
        if 'target' in _dict:
            args['target'] = _dict.get('target')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Validation object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'validated') and self.validated is not None:
            _dict['validated'] = datetime_to_string(self.validated)
        if hasattr(self, 'requested') and self.requested is not None:
            _dict['requested'] = datetime_to_string(self.requested)
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state
        if hasattr(self, 'last_operation') and self.last_operation is not None:
            _dict['last_operation'] = self.last_operation
        if hasattr(self, 'target') and self.target is not None:
            _dict['target'] = self.target
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Validation object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Validation') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Validation') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Version:
    """
    Offering version information.

    :attr str id: (optional) Unique ID.
    :attr str rev: (optional) Cloudant revision.
    :attr str crn: (optional) Version's CRN.
    :attr str version: (optional) Version of content type.
    :attr str sha: (optional) hash of the content.
    :attr datetime created: (optional) The date and time this version was created.
    :attr datetime updated: (optional) The date and time this version was last
          updated.
    :attr str offering_id: (optional) Offering ID.
    :attr str catalog_id: (optional) Catalog ID.
    :attr str kind_id: (optional) Kind ID.
    :attr List[str] tags: (optional) List of tags associated with this catalog.
    :attr str repo_url: (optional) Content's repo URL.
    :attr str source_url: (optional) Content's source URL (e.g git repo).
    :attr str tgz_url: (optional) File used to on-board this version.
    :attr List[Configuration] configuration: (optional) List of user solicited
          overrides.
    :attr dict metadata: (optional) Open ended metadata information.
    :attr Validation validation: (optional) Validation response.
    :attr List[Resource] required_resources: (optional) Resource requirments for
          installation.
    :attr bool single_instance: (optional) Denotes if single instance can be
          deployed to a given cluster.
    :attr Script install: (optional) Script information.
    :attr List[Script] pre_install: (optional) Optional pre-install instructions.
    :attr VersionEntitlement entitlement: (optional) Entitlement license info.
    :attr List[License] licenses: (optional) List of licenses the product was built
          with.
    :attr str image_manifest_url: (optional) If set, denotes a url to a YAML file
          with list of container images used by this version.
    :attr bool deprecated: (optional) read only field, indicating if this version is
          deprecated.
    :attr str package_version: (optional) Version of the package used to create this
          version.
    :attr State state: (optional) Offering state.
    :attr str version_locator: (optional) A dotted value of `catalogID`.`versionID`.
    :attr str console_url: (optional) Console URL.
    :attr str long_description: (optional) Long description for version.
    :attr List[str] whitelisted_accounts: (optional) Whitelisted accounts for
          version.
    """

    def __init__(
        self,
        *,
        id: str = None,
        rev: str = None,
        crn: str = None,
        version: str = None,
        sha: str = None,
        created: datetime = None,
        updated: datetime = None,
        offering_id: str = None,
        catalog_id: str = None,
        kind_id: str = None,
        tags: List[str] = None,
        repo_url: str = None,
        source_url: str = None,
        tgz_url: str = None,
        configuration: List['Configuration'] = None,
        metadata: dict = None,
        validation: 'Validation' = None,
        required_resources: List['Resource'] = None,
        single_instance: bool = None,
        install: 'Script' = None,
        pre_install: List['Script'] = None,
        entitlement: 'VersionEntitlement' = None,
        licenses: List['License'] = None,
        image_manifest_url: str = None,
        deprecated: bool = None,
        package_version: str = None,
        state: 'State' = None,
        version_locator: str = None,
        console_url: str = None,
        long_description: str = None,
        whitelisted_accounts: List[str] = None
    ) -> None:
        """
        Initialize a Version object.

        :param str id: (optional) Unique ID.
        :param str rev: (optional) Cloudant revision.
        :param str crn: (optional) Version's CRN.
        :param str version: (optional) Version of content type.
        :param str sha: (optional) hash of the content.
        :param datetime created: (optional) The date and time this version was
               created.
        :param datetime updated: (optional) The date and time this version was last
               updated.
        :param str offering_id: (optional) Offering ID.
        :param str catalog_id: (optional) Catalog ID.
        :param str kind_id: (optional) Kind ID.
        :param List[str] tags: (optional) List of tags associated with this
               catalog.
        :param str repo_url: (optional) Content's repo URL.
        :param str source_url: (optional) Content's source URL (e.g git repo).
        :param str tgz_url: (optional) File used to on-board this version.
        :param List[Configuration] configuration: (optional) List of user solicited
               overrides.
        :param dict metadata: (optional) Open ended metadata information.
        :param Validation validation: (optional) Validation response.
        :param List[Resource] required_resources: (optional) Resource requirments
               for installation.
        :param bool single_instance: (optional) Denotes if single instance can be
               deployed to a given cluster.
        :param Script install: (optional) Script information.
        :param List[Script] pre_install: (optional) Optional pre-install
               instructions.
        :param VersionEntitlement entitlement: (optional) Entitlement license info.
        :param List[License] licenses: (optional) List of licenses the product was
               built with.
        :param str image_manifest_url: (optional) If set, denotes a url to a YAML
               file with list of container images used by this version.
        :param bool deprecated: (optional) read only field, indicating if this
               version is deprecated.
        :param str package_version: (optional) Version of the package used to
               create this version.
        :param State state: (optional) Offering state.
        :param str version_locator: (optional) A dotted value of
               `catalogID`.`versionID`.
        :param str console_url: (optional) Console URL.
        :param str long_description: (optional) Long description for version.
        :param List[str] whitelisted_accounts: (optional) Whitelisted accounts for
               version.
        """
        self.id = id
        self.rev = rev
        self.crn = crn
        self.version = version
        self.sha = sha
        self.created = created
        self.updated = updated
        self.offering_id = offering_id
        self.catalog_id = catalog_id
        self.kind_id = kind_id
        self.tags = tags
        self.repo_url = repo_url
        self.source_url = source_url
        self.tgz_url = tgz_url
        self.configuration = configuration
        self.metadata = metadata
        self.validation = validation
        self.required_resources = required_resources
        self.single_instance = single_instance
        self.install = install
        self.pre_install = pre_install
        self.entitlement = entitlement
        self.licenses = licenses
        self.image_manifest_url = image_manifest_url
        self.deprecated = deprecated
        self.package_version = package_version
        self.state = state
        self.version_locator = version_locator
        self.console_url = console_url
        self.long_description = long_description
        self.whitelisted_accounts = whitelisted_accounts

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Version':
        """Initialize a Version object from a json dictionary."""
        args = {}
        if 'id' in _dict:
            args['id'] = _dict.get('id')
        if '_rev' in _dict:
            args['rev'] = _dict.get('_rev')
        if 'crn' in _dict:
            args['crn'] = _dict.get('crn')
        if 'version' in _dict:
            args['version'] = _dict.get('version')
        if 'sha' in _dict:
            args['sha'] = _dict.get('sha')
        if 'created' in _dict:
            args['created'] = string_to_datetime(_dict.get('created'))
        if 'updated' in _dict:
            args['updated'] = string_to_datetime(_dict.get('updated'))
        if 'offering_id' in _dict:
            args['offering_id'] = _dict.get('offering_id')
        if 'catalog_id' in _dict:
            args['catalog_id'] = _dict.get('catalog_id')
        if 'kind_id' in _dict:
            args['kind_id'] = _dict.get('kind_id')
        if 'tags' in _dict:
            args['tags'] = _dict.get('tags')
        if 'repo_url' in _dict:
            args['repo_url'] = _dict.get('repo_url')
        if 'source_url' in _dict:
            args['source_url'] = _dict.get('source_url')
        if 'tgz_url' in _dict:
            args['tgz_url'] = _dict.get('tgz_url')
        if 'configuration' in _dict:
            args['configuration'] = [Configuration.from_dict(x) for x in _dict.get('configuration')]
        if 'metadata' in _dict:
            args['metadata'] = _dict.get('metadata')
        if 'validation' in _dict:
            args['validation'] = Validation.from_dict(_dict.get('validation'))
        if 'required_resources' in _dict:
            args['required_resources'] = [Resource.from_dict(x) for x in _dict.get('required_resources')]
        if 'single_instance' in _dict:
            args['single_instance'] = _dict.get('single_instance')
        if 'install' in _dict:
            args['install'] = Script.from_dict(_dict.get('install'))
        if 'pre_install' in _dict:
            args['pre_install'] = [Script.from_dict(x) for x in _dict.get('pre_install')]
        if 'entitlement' in _dict:
            args['entitlement'] = VersionEntitlement.from_dict(_dict.get('entitlement'))
        if 'licenses' in _dict:
            args['licenses'] = [License.from_dict(x) for x in _dict.get('licenses')]
        if 'image_manifest_url' in _dict:
            args['image_manifest_url'] = _dict.get('image_manifest_url')
        if 'deprecated' in _dict:
            args['deprecated'] = _dict.get('deprecated')
        if 'package_version' in _dict:
            args['package_version'] = _dict.get('package_version')
        if 'state' in _dict:
            args['state'] = State.from_dict(_dict.get('state'))
        if 'version_locator' in _dict:
            args['version_locator'] = _dict.get('version_locator')
        if 'console_url' in _dict:
            args['console_url'] = _dict.get('console_url')
        if 'long_description' in _dict:
            args['long_description'] = _dict.get('long_description')
        if 'whitelisted_accounts' in _dict:
            args['whitelisted_accounts'] = _dict.get('whitelisted_accounts')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Version object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and self.id is not None:
            _dict['id'] = self.id
        if hasattr(self, 'rev') and self.rev is not None:
            _dict['_rev'] = self.rev
        if hasattr(self, 'crn') and self.crn is not None:
            _dict['crn'] = self.crn
        if hasattr(self, 'version') and self.version is not None:
            _dict['version'] = self.version
        if hasattr(self, 'sha') and self.sha is not None:
            _dict['sha'] = self.sha
        if hasattr(self, 'created') and self.created is not None:
            _dict['created'] = datetime_to_string(self.created)
        if hasattr(self, 'updated') and self.updated is not None:
            _dict['updated'] = datetime_to_string(self.updated)
        if hasattr(self, 'offering_id') and self.offering_id is not None:
            _dict['offering_id'] = self.offering_id
        if hasattr(self, 'catalog_id') and self.catalog_id is not None:
            _dict['catalog_id'] = self.catalog_id
        if hasattr(self, 'kind_id') and self.kind_id is not None:
            _dict['kind_id'] = self.kind_id
        if hasattr(self, 'tags') and self.tags is not None:
            _dict['tags'] = self.tags
        if hasattr(self, 'repo_url') and self.repo_url is not None:
            _dict['repo_url'] = self.repo_url
        if hasattr(self, 'source_url') and self.source_url is not None:
            _dict['source_url'] = self.source_url
        if hasattr(self, 'tgz_url') and self.tgz_url is not None:
            _dict['tgz_url'] = self.tgz_url
        if hasattr(self, 'configuration') and self.configuration is not None:
            _dict['configuration'] = [x.to_dict() for x in self.configuration]
        if hasattr(self, 'metadata') and self.metadata is not None:
            _dict['metadata'] = self.metadata
        if hasattr(self, 'validation') and self.validation is not None:
            _dict['validation'] = self.validation.to_dict()
        if hasattr(self, 'required_resources') and self.required_resources is not None:
            _dict['required_resources'] = [x.to_dict() for x in self.required_resources]
        if hasattr(self, 'single_instance') and self.single_instance is not None:
            _dict['single_instance'] = self.single_instance
        if hasattr(self, 'install') and self.install is not None:
            _dict['install'] = self.install.to_dict()
        if hasattr(self, 'pre_install') and self.pre_install is not None:
            _dict['pre_install'] = [x.to_dict() for x in self.pre_install]
        if hasattr(self, 'entitlement') and self.entitlement is not None:
            _dict['entitlement'] = self.entitlement.to_dict()
        if hasattr(self, 'licenses') and self.licenses is not None:
            _dict['licenses'] = [x.to_dict() for x in self.licenses]
        if hasattr(self, 'image_manifest_url') and self.image_manifest_url is not None:
            _dict['image_manifest_url'] = self.image_manifest_url
        if hasattr(self, 'deprecated') and self.deprecated is not None:
            _dict['deprecated'] = self.deprecated
        if hasattr(self, 'package_version') and self.package_version is not None:
            _dict['package_version'] = self.package_version
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state.to_dict()
        if hasattr(self, 'version_locator') and self.version_locator is not None:
            _dict['version_locator'] = self.version_locator
        if hasattr(self, 'console_url') and self.console_url is not None:
            _dict['console_url'] = self.console_url
        if hasattr(self, 'long_description') and self.long_description is not None:
            _dict['long_description'] = self.long_description
        if hasattr(self, 'whitelisted_accounts') and self.whitelisted_accounts is not None:
            _dict['whitelisted_accounts'] = self.whitelisted_accounts
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this Version object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'Version') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'Version') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class VersionEntitlement:
    """
    Entitlement license info.

    :attr str provider_name: (optional) Provider name.
    :attr str provider_id: (optional) Provider ID.
    :attr str product_id: (optional) Product ID.
    :attr List[str] part_numbers: (optional) list of license entitlement part
          numbers, eg. D1YGZLL,D1ZXILL.
    :attr str image_repo_name: (optional) Image repository name.
    """

    def __init__(
        self,
        *,
        provider_name: str = None,
        provider_id: str = None,
        product_id: str = None,
        part_numbers: List[str] = None,
        image_repo_name: str = None
    ) -> None:
        """
        Initialize a VersionEntitlement object.

        :param str provider_name: (optional) Provider name.
        :param str provider_id: (optional) Provider ID.
        :param str product_id: (optional) Product ID.
        :param List[str] part_numbers: (optional) list of license entitlement part
               numbers, eg. D1YGZLL,D1ZXILL.
        :param str image_repo_name: (optional) Image repository name.
        """
        self.provider_name = provider_name
        self.provider_id = provider_id
        self.product_id = product_id
        self.part_numbers = part_numbers
        self.image_repo_name = image_repo_name

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'VersionEntitlement':
        """Initialize a VersionEntitlement object from a json dictionary."""
        args = {}
        if 'provider_name' in _dict:
            args['provider_name'] = _dict.get('provider_name')
        if 'provider_id' in _dict:
            args['provider_id'] = _dict.get('provider_id')
        if 'product_id' in _dict:
            args['product_id'] = _dict.get('product_id')
        if 'part_numbers' in _dict:
            args['part_numbers'] = _dict.get('part_numbers')
        if 'image_repo_name' in _dict:
            args['image_repo_name'] = _dict.get('image_repo_name')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a VersionEntitlement object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'provider_name') and self.provider_name is not None:
            _dict['provider_name'] = self.provider_name
        if hasattr(self, 'provider_id') and self.provider_id is not None:
            _dict['provider_id'] = self.provider_id
        if hasattr(self, 'product_id') and self.product_id is not None:
            _dict['product_id'] = self.product_id
        if hasattr(self, 'part_numbers') and self.part_numbers is not None:
            _dict['part_numbers'] = self.part_numbers
        if hasattr(self, 'image_repo_name') and self.image_repo_name is not None:
            _dict['image_repo_name'] = self.image_repo_name
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this VersionEntitlement object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'VersionEntitlement') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'VersionEntitlement') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class VersionUpdateDescriptor:
    """
    Indicates if the current version can be upgraded to the version identified by the
    descriptor.

    :attr str version_locator: (optional) A dotted value of `catalogID`.`versionID`.
    :attr str version: (optional) the version number of this version.
    :attr State state: (optional) Offering state.
    :attr List[Resource] required_resources: (optional) Resource requirments for
          installation.
    :attr str package_version: (optional) Version of package.
    :attr str sha: (optional) The SHA value of this version.
    :attr bool can_update: (optional) true if the current version can be upgraded to
          this version, false otherwise.
    :attr dict messages: (optional) If can_update is false, this map will contain
          messages for each failed check, otherwise it will be omitted.  Possible keys
          include nodes, cores, mem, disk, targetVersion, and install-permission-check.
    """

    def __init__(
        self,
        *,
        version_locator: str = None,
        version: str = None,
        state: 'State' = None,
        required_resources: List['Resource'] = None,
        package_version: str = None,
        sha: str = None,
        can_update: bool = None,
        messages: dict = None
    ) -> None:
        """
        Initialize a VersionUpdateDescriptor object.

        :param str version_locator: (optional) A dotted value of
               `catalogID`.`versionID`.
        :param str version: (optional) the version number of this version.
        :param State state: (optional) Offering state.
        :param List[Resource] required_resources: (optional) Resource requirments
               for installation.
        :param str package_version: (optional) Version of package.
        :param str sha: (optional) The SHA value of this version.
        :param bool can_update: (optional) true if the current version can be
               upgraded to this version, false otherwise.
        :param dict messages: (optional) If can_update is false, this map will
               contain messages for each failed check, otherwise it will be omitted.
               Possible keys include nodes, cores, mem, disk, targetVersion, and
               install-permission-check.
        """
        self.version_locator = version_locator
        self.version = version
        self.state = state
        self.required_resources = required_resources
        self.package_version = package_version
        self.sha = sha
        self.can_update = can_update
        self.messages = messages

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'VersionUpdateDescriptor':
        """Initialize a VersionUpdateDescriptor object from a json dictionary."""
        args = {}
        if 'version_locator' in _dict:
            args['version_locator'] = _dict.get('version_locator')
        if 'version' in _dict:
            args['version'] = _dict.get('version')
        if 'state' in _dict:
            args['state'] = State.from_dict(_dict.get('state'))
        if 'required_resources' in _dict:
            args['required_resources'] = [Resource.from_dict(x) for x in _dict.get('required_resources')]
        if 'package_version' in _dict:
            args['package_version'] = _dict.get('package_version')
        if 'sha' in _dict:
            args['sha'] = _dict.get('sha')
        if 'can_update' in _dict:
            args['can_update'] = _dict.get('can_update')
        if 'messages' in _dict:
            args['messages'] = _dict.get('messages')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a VersionUpdateDescriptor object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'version_locator') and self.version_locator is not None:
            _dict['version_locator'] = self.version_locator
        if hasattr(self, 'version') and self.version is not None:
            _dict['version'] = self.version
        if hasattr(self, 'state') and self.state is not None:
            _dict['state'] = self.state.to_dict()
        if hasattr(self, 'required_resources') and self.required_resources is not None:
            _dict['required_resources'] = [x.to_dict() for x in self.required_resources]
        if hasattr(self, 'package_version') and self.package_version is not None:
            _dict['package_version'] = self.package_version
        if hasattr(self, 'sha') and self.sha is not None:
            _dict['sha'] = self.sha
        if hasattr(self, 'can_update') and self.can_update is not None:
            _dict['can_update'] = self.can_update
        if hasattr(self, 'messages') and self.messages is not None:
            _dict['messages'] = self.messages
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this VersionUpdateDescriptor object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'VersionUpdateDescriptor') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'VersionUpdateDescriptor') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other
