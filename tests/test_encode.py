import os
from unittest.mock import Mock, patch

import pytest
from docarray import Document, DocumentArray


@pytest.mark.parametrize(
    'inputs',
    [
        DocumentArray(
            [
                Document(text='hello world'),
                Document(uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg'),
            ]
        ),
        [
            Document(text='hello world'),
            Document(uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg'),
        ],
    ],
)
def test_encode_document(make_client, inputs):
    res = make_client.encode(docs=inputs)
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
def test_encode_plain_text_single(make_client, inputs):
    res = make_client.encode(text=inputs)
    assert res.shape == (512,)


@pytest.mark.parametrize('inputs', [['hello world', 'hello jina']])
@patch(
    'inference_client.client.get_model_spec',
    Mock(return_value={'endpoints': {'grpc': 'grpc://mock.inference.jina.ai'}}),
)
def test_encode_plain_text_list(make_client, inputs):
    res = make_client.encode(text=inputs)
    assert len(res) == 2
    assert res[0].shape == (512,)
    assert res[1].shape == (512,)


@pytest.mark.parametrize(
    'inputs',
    [
        f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
        Document(uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg')
        .load_uri_to_blob()
        .blob,
        Document(uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg')
        .load_uri_to_image_tensor()
        .tensor,
    ],
)
def test_encode_plain_image(make_client, inputs):
    res = make_client.encode(image=inputs)
    assert res.shape == (512,)


@pytest.mark.parametrize(
    'inputs',
    [
        [
            f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
            Document(uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg')
            .load_uri_to_blob()
            .blob,
        ],
        [
            Document(uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg')
            .load_uri_to_blob()
            .blob,
            Document(uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg')
            .load_uri_to_image_tensor()
            .tensor,
        ],  # Commented out for a wierd GitHub Actions timeout error with Python 3.10
        [
            Document(uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg')
            .load_uri_to_image_tensor()
            .tensor,
            f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
        ],
    ],
)
def test_encode_plain_image_list(make_client, inputs):
    res = make_client.encode(image=inputs)
    assert len(res) == 2
    assert res[0].shape == (512,)
    assert res[1].shape == (512,)
