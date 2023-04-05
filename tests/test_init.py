import os
from unittest.mock import Mock, patch

import pytest

from inference_client import Client


@patch('client.client.login', Mock(return_value='valid token'))
def test_valid_token():
    client = Client(token='valid token')
    assert client.token == 'valid token'


def test_invalid_token():
    with pytest.raises(Exception):
        Client(token='invalid token')


@patch.dict(os.environ, {'JINA_AUTH_TOKEN': 'valid session'})
@patch('client.client.login', Mock(return_value='valid session'))
def test_no_token_valid_session():
    client = Client()
    assert client.token == 'valid session'


@patch.dict(os.environ, {'JINA_AUTH_TOKEN': 'invalid session'})
@patch('client.client.login', Mock(return_value='valid session'))
def test_no_token_invalid_session():
    client = Client()
    assert client.token == 'valid session'
