from unittest.mock import Mock, patch

import pytest
from docarray import Document, DocumentArray

from inference_client import Client


@pytest.mark.parametrize(
    'inputs',
    [
        DocumentArray(
            [
                Document(uri='https://picsum.photos/id/233/100'),
            ]
        ),
        [
            Document(uri='https://picsum.photos/id/233/100'),
        ],
    ],
)
@patch('client.client.fetch_host', Mock(return_value='grpc://mock.inference.jina.ai'))
@patch('client.client.login', Mock(return_value='valid_token'))
@patch(
    'client.base.BaseClient.caption',
    Mock(
        return_value=DocumentArray(
            [
                Document(
                    uri='https://picsum.photos/id/233/100',
                    tags={'response': 'a image of two rails'},
                ).load_uri_to_blob(),
            ]
        )
    ),
)
def test_caption_document(inputs):
    model = Client().get_model('mock-model')
    res = model.caption(docs=inputs)
    res.summary()
    res[0].summary()


@pytest.mark.parametrize(
    'inputs',
    [
        'https://picsum.photos/id/233/100',
        Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob,
        Document(uri='https://picsum.photos/id/233/100')
        .load_uri_to_image_tensor()
        .tensor,
    ],
)
@patch('client.client.fetch_host', Mock(return_value='grpc://mock.inference.jina.ai'))
@patch('client.client.login', Mock(return_value='valid_token'))
@patch(
    'client.base.BaseClient.caption',
    Mock(
        return_value=DocumentArray(
            [
                Document(
                    uri='https://picsum.photos/id/233/100',
                    tags={'response': 'a image of two rails'},
                ).load_uri_to_blob(),
            ]
        )
    ),
)
def test_encode_plain_image_str(inputs):
    model = Client().get_model('mock-model')
    res = model.caption(image=inputs)
    res.summary()
    res[0].summary()


@pytest.mark.parametrize(
    'inputs',
    [
        Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob,
    ],
)
@patch('client.client.fetch_host', Mock(return_value='grpc://mock.inference.jina.ai'))
@patch('client.client.login', Mock(return_value='valid_token'))
@patch(
    'client.base.BaseClient.caption',
    Mock(
        return_value=DocumentArray(
            [
                Document(
                    uri='https://picsum.photos/id/233/100',
                    tags={'response': 'a image of two rails'},
                ).load_uri_to_blob(),
            ]
        )
    ),
)
def test_encode_plain_image_blob(inputs):
    model = Client().get_model('mock-model')
    res = model.caption(image=inputs)
    res.summary()
    res[0].summary()


@pytest.mark.parametrize(
    'inputs',
    [
        Document(uri='https://picsum.photos/id/233/100')
        .load_uri_to_image_tensor()
        .tensor,
    ],
)
@patch('client.client.fetch_host', Mock(return_value='grpc://mock.inference.jina.ai'))
@patch('client.client.login', Mock(return_value='valid_token'))
@patch(
    'client.base.BaseClient.caption',
    Mock(
        return_value=DocumentArray(
            [
                Document(
                    uri='https://picsum.photos/id/233/100',
                    tags={'response': 'a image of two rails'},
                ).load_uri_to_image_tensor(),
            ]
        )
    ),
)
def test_encode_plain_image_tensor(inputs):
    model = Client().get_model('mock-model')
    res = model.caption(image=inputs)
    res.summary()
    res[0].summary()
