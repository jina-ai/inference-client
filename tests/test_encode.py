from unittest.mock import Mock, patch

import pytest
from docarray import Document, DocumentArray

from inference_client.base import BaseClient


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
def test_encode_document(make_flow, inputs):
    model = BaseClient(
        model_name='dummy-model',
        token='valid_token',
        host=f'grpc://0.0.0.0:{make_flow.port}',
    )
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
def test_encode_plain_text_single(make_flow, inputs):
    model = BaseClient(
        model_name='dummy-model',
        token='valid_token',
        host=f'grpc://0.0.0.0:{make_flow.port}',
    )
    res = model.encode(text=inputs)
    assert res.shape == (512,)


@pytest.mark.parametrize('inputs', [['hello world', 'hello jina']])
@patch(
    'inference_client.client.get_model_spec',
    Mock(return_value={'endpoints': {'grpc': 'grpc://mock.inference.jina.ai'}}),
)
def test_encode_plain_text_list(make_flow, inputs):
    model = BaseClient(
        model_name='dummy-model',
        token='valid_token',
        host=f'grpc://0.0.0.0:{make_flow.port}',
    )
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
def test_encode_plain_image(make_flow, inputs):
    model = BaseClient(
        model_name='dummy-model',
        token='valid_token',
        host=f'grpc://0.0.0.0:{make_flow.port}',
    )
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
def test_encode_plain_image_list(make_flow, inputs):
    model = BaseClient(
        model_name='dummy-model',
        token='valid_token',
        host=f'grpc://0.0.0.0:{make_flow.port}',
    )
    res = model.encode(image=inputs)
    assert len(res) == 2
    assert res[0].shape == (512,)
    assert res[1].shape == (512,)
