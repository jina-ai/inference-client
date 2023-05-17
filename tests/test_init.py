import os
from unittest.mock import Mock, patch

import pytest
from fastapi.responses import JSONResponse

from inference_client import Client


@patch('inference_client.client.login', Mock(return_value='valid token'))
def test_valid_token():
    client = Client(token='valid token')
    assert client._auth_token == 'valid token'


def test_invalid_token():
    with pytest.raises(Exception) as e:
        Client(token='invalid token')
    assert (
        str(e.value)
        == 'Invalid or expired auth token. Please re-enter your token and try again.'
    )


@patch.dict(os.environ, {'JINA_AUTH_TOKEN': 'valid session'})
@patch('inference_client.client.login', Mock(return_value='valid session'))
def test_no_token_valid_session():
    client = Client()
    assert client._auth_token == 'valid session'


@patch.dict(os.environ, {'JINA_AUTH_TOKEN': 'invalid session'})
@patch('inference_client.client.login', Mock(return_value='valid session'))
def test_no_token_invalid_session():
    client = Client()
    assert client._auth_token == 'valid session'


@patch('inference_client.client.login', Mock(return_value='valid token'))
@patch(
    'inference_client.helper.requests.get',
    Mock(return_value=JSONResponse(status_code=404, content={})),
)
def test_invalid_model_name():
    client = Client(token='valid token')
    with pytest.raises(ValueError) as e:
        client.get_model('invalid model')
    assert (
        str(e.value)
        == 'Invalid model name `invalid model` provided. Please visit https://cloud.jina.ai/user/inference to create and use the model names listed there.'
    )
