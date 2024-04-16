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

# IBM OpenAPI SDK Code Generator Version: 3.33.0-caf29bd0-20210603-225214

"""
API docs for IBM Cloud Shell repository
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


class IbmCloudShellV1(BaseService):
    """The IBM Cloud Shell V1 service."""

    DEFAULT_SERVICE_URL = 'https://api.shell.cloud.ibm.com'
    DEFAULT_SERVICE_NAME = 'ibm_cloud_shell'

    @classmethod
    def new_instance(
        cls,
        service_name: str = DEFAULT_SERVICE_NAME,
    ) -> 'IbmCloudShellV1':
        """
        Return a new client for the IBM Cloud Shell service using the specified
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
        Construct a new client for the IBM Cloud Shell service.

        :param Authenticator authenticator: The authenticator specifies the authentication mechanism.
               Get up to date information from https://github.com/IBM/python-sdk-core/blob/master/README.md
               about initializing the authenticator of your choice.
        """
        BaseService.__init__(self, service_url=self.DEFAULT_SERVICE_URL, authenticator=authenticator)

    #########################
    # account_settings
    #########################

    def get_account_settings(self, account_id: str, **kwargs) -> DetailedResponse:
        """
        Get account settings.

        Retrieve account settings for the given account ID. Call this method to get
        details about a particular account setting, whether Cloud Shell is enabled, the
        list of enabled regions and the list of enabled features. Users need to be an
        account owner or users need to be assigned an IAM policy with the Administrator
        role for the Cloud Shell account management service.

        :param str account_id: The account ID in which the account settings belong
               to.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccountSettings` object
        """

        if account_id is None:
            raise ValueError('account_id must be provided')
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='get_account_settings'
        )
        headers.update(sdk_headers)

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id']
        path_param_values = self.encode_path_vars(account_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/api/v1/user/accounts/{account_id}/settings'.format(**path_param_dict)
        request = self.prepare_request(method='GET', url=url, headers=headers)

        response = self.send(request)
        return response

    def update_account_settings(
        self,
        account_id: str,
        *,
        rev: str = None,
        default_enable_new_features: bool = None,
        default_enable_new_regions: bool = None,
        enabled: bool = None,
        features: List['Feature'] = None,
        regions: List['RegionSetting'] = None,
        **kwargs
    ) -> DetailedResponse:
        """
        Update account settings.

        Update account settings for the given account ID. Call this method to update
        account settings configuration, you can enable or disable Cloud Shell, enable or
        disable available regions and enable and disable features. To update account
        settings, users need to be an account owner or users need to be assigned an IAM
        policy with the Administrator role for the Cloud Shell account management service.

        :param str account_id: The account ID in which the account settings belong
               to.
        :param str rev: (optional) Unique revision number for the settings object.
        :param bool default_enable_new_features: (optional) You can choose which
               Cloud Shell features are available in the account and whether any new
               features are enabled as they become available. The feature settings apply
               only to the enabled Cloud Shell locations.
        :param bool default_enable_new_regions: (optional) Set whether Cloud Shell
               is enabled in a specific location for the account. The location determines
               where user and session data are stored. By default, users are routed to the
               nearest available location.
        :param bool enabled: (optional) When enabled, Cloud Shell is available to
               all users in the account.
        :param List[Feature] features: (optional) List of Cloud Shell features.
        :param List[RegionSetting] regions: (optional) List of Cloud Shell region
               settings.
        :param dict headers: A `dict` containing the request headers
        :return: A `DetailedResponse` containing the result, headers and HTTP status code.
        :rtype: DetailedResponse with `dict` result representing a `AccountSettings` object
        """

        if account_id is None:
            raise ValueError('account_id must be provided')
        if features is not None:
            features = [convert_model(x) for x in features]
        if regions is not None:
            regions = [convert_model(x) for x in regions]
        headers = {}
        sdk_headers = get_sdk_headers(
            service_name=self.DEFAULT_SERVICE_NAME, service_version='V1', operation_id='update_account_settings'
        )
        headers.update(sdk_headers)

        data = {
            '_rev': rev,
            'default_enable_new_features': default_enable_new_features,
            'default_enable_new_regions': default_enable_new_regions,
            'enabled': enabled,
            'features': features,
            'regions': regions,
        }
        data = {k: v for (k, v) in data.items() if v is not None}
        data = json.dumps(data)
        headers['content-type'] = 'application/json'

        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['Accept'] = 'application/json'

        path_param_keys = ['account_id']
        path_param_values = self.encode_path_vars(account_id)
        path_param_dict = dict(zip(path_param_keys, path_param_values))
        url = '/api/v1/user/accounts/{account_id}/settings'.format(**path_param_dict)
        request = self.prepare_request(method='POST', url=url, headers=headers, data=data)

        response = self.send(request)
        return response


##############################################################################
# Models
##############################################################################


class AccountSettings:
    """
    Definition of Cloud Shell account settings.

    :attr str id: (optional) Unique id of the settings object.
    :attr str rev: (optional) Unique revision number for the settings object.
    :attr str account_id: (optional) The id of the account the settings belong to.
    :attr int created_at: (optional) Creation timestamp in Unix epoch time.
    :attr str created_by: (optional) IAM ID of creator.
    :attr bool default_enable_new_features: (optional) You can choose which Cloud
          Shell features are available in the account and whether any new features are
          enabled as they become available. The feature settings apply only to the enabled
          Cloud Shell locations.
    :attr bool default_enable_new_regions: (optional) Set whether Cloud Shell is
          enabled in a specific location for the account. The location determines where
          user and session data are stored. By default, users are routed to the nearest
          available location.
    :attr bool enabled: (optional) When enabled, Cloud Shell is available to all
          users in the account.
    :attr List[Feature] features: (optional) List of Cloud Shell features.
    :attr List[RegionSetting] regions: (optional) List of Cloud Shell region
          settings.
    :attr str type: (optional) Type of api response object.
    :attr int updated_at: (optional) Timestamp of last update in Unix epoch time.
    :attr str updated_by: (optional) IAM ID of last updater.
    """

    def __init__(
        self,
        *,
        id: str = None,
        rev: str = None,
        account_id: str = None,
        created_at: int = None,
        created_by: str = None,
        default_enable_new_features: bool = None,
        default_enable_new_regions: bool = None,
        enabled: bool = None,
        features: List['Feature'] = None,
        regions: List['RegionSetting'] = None,
        type: str = None,
        updated_at: int = None,
        updated_by: str = None
    ) -> None:
        """
        Initialize a AccountSettings object.

        :param str rev: (optional) Unique revision number for the settings object.
        :param bool default_enable_new_features: (optional) You can choose which
               Cloud Shell features are available in the account and whether any new
               features are enabled as they become available. The feature settings apply
               only to the enabled Cloud Shell locations.
        :param bool default_enable_new_regions: (optional) Set whether Cloud Shell
               is enabled in a specific location for the account. The location determines
               where user and session data are stored. By default, users are routed to the
               nearest available location.
        :param bool enabled: (optional) When enabled, Cloud Shell is available to
               all users in the account.
        :param List[Feature] features: (optional) List of Cloud Shell features.
        :param List[RegionSetting] regions: (optional) List of Cloud Shell region
               settings.
        """
        self.id = id
        self.rev = rev
        self.account_id = account_id
        self.created_at = created_at
        self.created_by = created_by
        self.default_enable_new_features = default_enable_new_features
        self.default_enable_new_regions = default_enable_new_regions
        self.enabled = enabled
        self.features = features
        self.regions = regions
        self.type = type
        self.updated_at = updated_at
        self.updated_by = updated_by

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'AccountSettings':
        """Initialize a AccountSettings object from a json dictionary."""
        args = {}
        if '_id' in _dict:
            args['id'] = _dict.get('_id')
        if '_rev' in _dict:
            args['rev'] = _dict.get('_rev')
        if 'account_id' in _dict:
            args['account_id'] = _dict.get('account_id')
        if 'created_at' in _dict:
            args['created_at'] = _dict.get('created_at')
        if 'created_by' in _dict:
            args['created_by'] = _dict.get('created_by')
        if 'default_enable_new_features' in _dict:
            args['default_enable_new_features'] = _dict.get('default_enable_new_features')
        if 'default_enable_new_regions' in _dict:
            args['default_enable_new_regions'] = _dict.get('default_enable_new_regions')
        if 'enabled' in _dict:
            args['enabled'] = _dict.get('enabled')
        if 'features' in _dict:
            args['features'] = [Feature.from_dict(x) for x in _dict.get('features')]
        if 'regions' in _dict:
            args['regions'] = [RegionSetting.from_dict(x) for x in _dict.get('regions')]
        if 'type' in _dict:
            args['type'] = _dict.get('type')
        if 'updated_at' in _dict:
            args['updated_at'] = _dict.get('updated_at')
        if 'updated_by' in _dict:
            args['updated_by'] = _dict.get('updated_by')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a AccountSettings object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'id') and getattr(self, 'id') is not None:
            _dict['_id'] = getattr(self, 'id')
        if hasattr(self, 'rev') and self.rev is not None:
            _dict['_rev'] = self.rev
        if hasattr(self, 'account_id') and getattr(self, 'account_id') is not None:
            _dict['account_id'] = getattr(self, 'account_id')
        if hasattr(self, 'created_at') and getattr(self, 'created_at') is not None:
            _dict['created_at'] = getattr(self, 'created_at')
        if hasattr(self, 'created_by') and getattr(self, 'created_by') is not None:
            _dict['created_by'] = getattr(self, 'created_by')
        if hasattr(self, 'default_enable_new_features') and self.default_enable_new_features is not None:
            _dict['default_enable_new_features'] = self.default_enable_new_features
        if hasattr(self, 'default_enable_new_regions') and self.default_enable_new_regions is not None:
            _dict['default_enable_new_regions'] = self.default_enable_new_regions
        if hasattr(self, 'enabled') and self.enabled is not None:
            _dict['enabled'] = self.enabled
        if hasattr(self, 'features') and self.features is not None:
            _dict['features'] = [x.to_dict() for x in self.features]
        if hasattr(self, 'regions') and self.regions is not None:
            _dict['regions'] = [x.to_dict() for x in self.regions]
        if hasattr(self, 'type') and getattr(self, 'type') is not None:
            _dict['type'] = getattr(self, 'type')
        if hasattr(self, 'updated_at') and getattr(self, 'updated_at') is not None:
            _dict['updated_at'] = getattr(self, 'updated_at')
        if hasattr(self, 'updated_by') and getattr(self, 'updated_by') is not None:
            _dict['updated_by'] = getattr(self, 'updated_by')
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this AccountSettings object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'AccountSettings') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'AccountSettings') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other


class Feature:
    """
    Describes a Cloud Shell feature.

    :attr bool enabled: (optional) State of the feature.
    :attr str key: (optional) Name of the feature.
    """

    def __init__(self, *, enabled: bool = None, key: str = None) -> None:
        """
        Initialize a Feature object.

        :param bool enabled: (optional) State of the feature.
        :param str key: (optional) Name of the feature.
        """
        self.enabled = enabled
        self.key = key

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'Feature':
        """Initialize a Feature object from a json dictionary."""
        args = {}
        if 'enabled' in _dict:
            args['enabled'] = _dict.get('enabled')
        if 'key' in _dict:
            args['key'] = _dict.get('key')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a Feature object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'enabled') and self.enabled is not None:
            _dict['enabled'] = self.enabled
        if hasattr(self, 'key') and self.key is not None:
            _dict['key'] = self.key
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


class RegionSetting:
    """
    Describes a Cloud Shell region setting.

    :attr bool enabled: (optional) State of the region.
    :attr str key: (optional) Name of the region.
    """

    def __init__(self, *, enabled: bool = None, key: str = None) -> None:
        """
        Initialize a RegionSetting object.

        :param bool enabled: (optional) State of the region.
        :param str key: (optional) Name of the region.
        """
        self.enabled = enabled
        self.key = key

    @classmethod
    def from_dict(cls, _dict: Dict) -> 'RegionSetting':
        """Initialize a RegionSetting object from a json dictionary."""
        args = {}
        if 'enabled' in _dict:
            args['enabled'] = _dict.get('enabled')
        if 'key' in _dict:
            args['key'] = _dict.get('key')
        return cls(**args)

    @classmethod
    def _from_dict(cls, _dict):
        """Initialize a RegionSetting object from a json dictionary."""
        return cls.from_dict(_dict)

    def to_dict(self) -> Dict:
        """Return a json dictionary representing this model."""
        _dict = {}
        if hasattr(self, 'enabled') and self.enabled is not None:
            _dict['enabled'] = self.enabled
        if hasattr(self, 'key') and self.key is not None:
            _dict['key'] = self.key
        return _dict

    def _to_dict(self):
        """Return a json dictionary representing this model."""
        return self.to_dict()

    def __str__(self) -> str:
        """Return a `str` version of this RegionSetting object."""
        return json.dumps(self.to_dict(), indent=2)

    def __eq__(self, other: 'RegionSetting') -> bool:
        """Return `true` when self and other are equal, false otherwise."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: 'RegionSetting') -> bool:
        """Return `true` when self and other are not equal, false otherwise."""
        return not self == other
