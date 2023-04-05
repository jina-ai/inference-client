from unittest.mock import Mock, patch

import numpy as np
import pytest
from docarray import Document, DocumentArray

from inference_client import Client


@pytest.mark.parametrize(
    'inputs',
    [
        DocumentArray(
            [
                Document(text='hello world'),
                Document(uri='https://picsum.photos/id/233/100'),
            ]
        ),
        [
            Document(text='hello world'),
            Document(uri='https://picsum.photos/id/233/100'),
        ],
    ],
)
@patch(
    'inference_client.client.fetch_host',
    Mock(return_value='grpc://mock.inference.jina.ai'),
)
@patch('inference_client.client.login', Mock(return_value='valid_token'))
@patch(
    'inference_client.base.BaseClient.encode',
    Mock(
        return_value=DocumentArray(
            [
                Document(text='hello world', embedding=np.random.random((512,))),
                Document(
                    uri='https://picsum.photos/id/233/100',
                    embedding=np.random.random((512,)),
                ).load_uri_to_blob(),
            ]
        )
    ),
)
def test_encode_document(inputs):
    model = Client().get_model('mock-model')
    res = model.encode(docs=inputs)
    res.summary()
    res[0].summary()


@pytest.mark.parametrize('inputs', ['hello world'])
@patch(
    'inference_client.client.fetch_host',
    Mock(return_value='grpc://mock.inference.jina.ai'),
)
@patch('inference_client.client.login', Mock(return_value='valid_token'))
@patch(
    'inference_client.base.BaseClient.encode',
    Mock(
        return_value=DocumentArray(
            [Document(text='hello world', embedding=np.random.random((512,)))]
        )
    ),
)
def test_encode_plain_text(inputs):
    model = Client().get_model('mock-model')
    res = model.encode(text=inputs)
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
@patch(
    'inference_client.client.fetch_host',
    Mock(return_value='grpc://mock.inference.jina.ai'),
)
@patch('inference_client.client.login', Mock(return_value='valid_token'))
@patch(
    'inference_client.base.BaseClient.encode',
    Mock(
        return_value=DocumentArray(
            [
                Document(
                    uri='https://picsum.photos/id/233/100',
                    embedding=np.random.random((512,)),
                ).load_uri_to_blob(),
            ]
        )
    ),
)
def test_encode_plain_image_str(inputs):
    model = Client().get_model('mock-model')
    res = model.encode(image=inputs)
    res.summary()
    res[0].summary()


@pytest.mark.parametrize(
    'inputs',
    [
        Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob,
    ],
)
@patch(
    'inference_client.client.fetch_host',
    Mock(return_value='grpc://mock.inference.jina.ai'),
)
@patch('inference_client.client.login', Mock(return_value='valid_token'))
@patch(
    'inference_client.base.BaseClient.encode',
    Mock(
        return_value=DocumentArray(
            [
                Document(
                    uri='https://picsum.photos/id/233/100',
                    embedding=np.random.random((512,)),
                ).load_uri_to_blob(),
            ]
        )
    ),
)
def test_encode_plain_image_blob(inputs):
    model = Client().get_model('mock-model')
    res = model.encode(image=inputs)
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
@patch(
    'inference_client.client.fetch_host',
    Mock(return_value='grpc://mock.inference.jina.ai'),
)
@patch('inference_client.client.login', Mock(return_value='valid_token'))
@patch(
    'inference_client.base.BaseClient.encode',
    Mock(
        return_value=DocumentArray(
            [
                Document(
                    uri='https://picsum.photos/id/233/100',
                    embedding=np.random.random((512,)),
                ).load_uri_to_image_tensor(),
            ]
        )
    ),
)
def test_encode_plain_image_tensor(inputs):
    model = Client().get_model('mock-model')
    res = model.encode(image=inputs)
    res.summary()
    res[0].summary()
