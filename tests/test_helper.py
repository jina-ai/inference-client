"""
import os
from unittest.mock import Mock, patch

import pytest

from client.helper import available_models, fetch_host, login, validate_model

@patch('client.helper.Auth.validate_token', Mock(return_value=None))
def test_login_valid_token():
    assert login('valid token') == 'valid token'


def test_login_invalid_token():
    with pytest.raises(Exception):
        login('invalid token')


@patch.dict(os.environ, {'JINA_AUTH_TOKEN': 'valid session'})
@patch('client.helper.hubble.login', Mock(return_value=None))
@patch('client.helper.hubble.get_token', Mock(return_value='valid session'))
def test_login_no_token_valid_session():
    assert login() == 'valid session'


@patch.dict(os.environ, {'JINA_AUTH_TOKEN': 'invalid session'})
@patch('client.helper.hubble.login', Mock(return_value=None))
@patch('client.helper.hubble.get_token', Mock(return_value='valid session'))
def test_login_no_token_invalid_session():
    assert login() == 'valid session'


@patch('client.helper.requests.post', Mock(return_value=Mock(status_code=200)))
def test_validate_model_valid_token_valid_model():
    assert validate_model('valid token', 'valid model')


@patch('client.helper.requests.post', Mock(return_value=Mock(status_code=404)))
def test_validate_model_invalid_token_valid_model():
    with pytest.raises(Exception):
        validate_model('invalid token', 'valid model')


@patch('client.helper.requests.post', Mock(return_value=Mock(status_code=404)))
def test_validate_model_valid_token_invalid_model():
    with pytest.raises(Exception):
        validate_model('valid token', 'invalid model')


@patch('client.helper.requests.post', Mock(return_value=Mock(status_code=404)))
def test_validate_model_invalid_token_invalid_model():
    with pytest.raises(Exception):
        validate_model('invalid token', 'invalid model')
"""
