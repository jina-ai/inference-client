import os
from unittest.mock import Mock, patch

import pytest

from client import Client


@patch('client.Client._login', Mock(return_value='valid token'))
@patch('client.Client._fetch_metadata', Mock(return_value={'model': 'model name'}))
def test_valid_token():
    client = Client(model='model name', token='valid token', force=True)
    assert client.token == 'valid token'


def test_invalid_token():
    with pytest.raises(Exception):
        Client(model='model name', token='invalid token', force=True)


@patch.dict(os.environ, {'JINA_AUTH_TOKEN': 'valid session'})
@patch('client.Client._login', Mock(return_value='valid session'))
@patch('client.Client._fetch_metadata', Mock(return_value={'model': 'model name'}))
def test_no_token_valid_session():
    client = Client(model='model name')
    assert client.token == 'valid session'


@patch.dict(os.environ, {'JINA_AUTH_TOKEN': 'invalid session'})
@patch('client.Client._login', Mock(return_value='valid session'))
@patch('client.Client._fetch_metadata', Mock(return_value={'model': 'model name'}))
def test_no_token_invalid_session():
    client = Client(model='model name')
    assert client.token == 'valid session'
