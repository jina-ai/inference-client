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
    'inference_client.client.get_model_spec',
    Mock(return_value={'endpoints': {'grpc': 'grpc://mock.inference.jina.ai'}}),
)
@patch('inference_client.client.login', Mock(return_value='valid_token'))
@patch(
    'inference_client.base.BaseClient._post',
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
    assert isinstance(res, DocumentArray)
    assert len(res) == 2
    assert isinstance(res[0], Document)
    assert isinstance(res[1], Document)
    assert res[0].embedding.shape == (512,)
    assert res[1].embedding.shape == (512,)


@pytest.mark.parametrize('inputs', ['hello world'])
@patch(
    'inference_client.client.get_model_spec',
    Mock(return_value={'endpoints': {'grpc': 'grpc://mock.inference.jina.ai'}}),
)
@patch('inference_client.client.login', Mock(return_value='valid_token'))
@patch(
    'inference_client.base.BaseClient._post',
    Mock(
        return_value=DocumentArray(
            [Document(text='hello world', embedding=np.random.random((512,)))]
        )
    ),
)
def test_encode_plain_text_single(inputs):
    model = Client().get_model('mock-model')
    res = model.encode(text=inputs)
    assert res.shape == (512,)


@pytest.mark.parametrize('inputs', [['hello world', 'hello jina']])
@patch(
    'inference_client.client.get_model_spec',
    Mock(return_value={'endpoints': {'grpc': 'grpc://mock.inference.jina.ai'}}),
)
@patch('inference_client.client.login', Mock(return_value='valid_token'))
@patch(
    'inference_client.base.BaseClient._post',
    Mock(
        return_value=DocumentArray(
            [
                Document(text='hello world', embedding=np.random.random((512,))),
                Document(text='hello jina', embedding=np.random.random((512,))),
            ]
        )
    ),
)
def test_encode_plain_text_list(inputs):
    model = Client().get_model('mock-model')
    res = model.encode(text=inputs)
    assert len(res) == 2
    assert res[0].shape == (512,)
    assert res[1].shape == (512,)


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
    'inference_client.client.get_model_spec',
    Mock(return_value={'endpoints': {'grpc': 'grpc://mock.inference.jina.ai'}}),
)
@patch('inference_client.client.login', Mock(return_value='valid_token'))
@patch(
    'inference_client.base.BaseClient._post',
    Mock(
        return_value=DocumentArray(
            [
                Document(
                    uri='https://picsum.photos/id/233/100',
                    embedding=np.random.random((512,)),
                ),
            ]
        )
    ),
)
def test_encode_plain_image(inputs):
    model = Client().get_model('mock-model')
    res = model.encode(image=inputs)
    assert res.shape == (512,)


@pytest.mark.parametrize(
    'inputs',
    [
        [
            'https://picsum.photos/id/233/100',
            Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob,
        ],
        [
            Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob,
            Document(uri='https://picsum.photos/id/233/100')
            .load_uri_to_image_tensor()
            .tensor,
        ],
        [
            Document(uri='https://picsum.photos/id/233/100')
            .load_uri_to_image_tensor()
            .tensor,
            'https://picsum.photos/id/233/100',
        ],
    ],
)
@patch(
    'inference_client.client.get_model_spec',
    Mock(return_value={'endpoints': {'grpc': 'grpc://mock.inference.jina.ai'}}),
)
@patch('inference_client.client.login', Mock(return_value='valid_token'))
@patch(
    'inference_client.base.BaseClient._post',
    Mock(
        return_value=DocumentArray(
            [
                Document(
                    uri='https://picsum.photos/id/233/100',
                    embedding=np.random.random((512,)),
                ),
                Document(
                    uri='https://picsum.photos/id/233/100',
                    embedding=np.random.random((512,)),
                ),
            ]
        )
    ),
)
def test_encode_plain_image_list(inputs):
    model = Client().get_model('mock-model')
    res = model.encode(image=inputs)
    assert len(res) == 2
    assert res[0].shape == (512,)
    assert res[1].shape == (512,)
