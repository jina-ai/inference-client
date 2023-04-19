from unittest.mock import Mock, patch

import pytest
from docarray import Document, DocumentArray

from inference_client import Client


@pytest.mark.parametrize(
    'inputs',
    [
        DocumentArray(
            [
                Document(
                    uri='https://picsum.photos/id/233/100',
                    tags={'prompt': 'Question: how many cats are there? Answer:'},
                ),
            ]
        ),
        [
            Document(
                uri='https://picsum.photos/id/233/100',
                tags={'prompt': 'Question: how many cats are there? Answer:'},
            ),
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
                    tags={'response': 'a lot more than you think'},
                ),
            ]
        ),
    ),
)
def test_vqa_document(inputs):
    model = Client().get_model('mock-model')
    res = model.vqa(docs=inputs)
    assert isinstance(res, DocumentArray)
    assert res[0].tags['response'] == 'a lot more than you think'


@pytest.mark.parametrize(
    'inputs',
    [
        [
            'https://picsum.photos/id/233/100',
            'Question: how many cats are there? Answer:',
        ],
        [
            Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob,
            'Question: how many cats are there? Answer:',
        ],
        [
            Document(uri='https://picsum.photos/id/233/100')
            .load_uri_to_image_tensor()
            .tensor,
            'Question: how many cats are there? Answer:',
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
                    tags={'response': 'a lot more than you think'},
                ),
            ]
        ),
    ),
)
def test_vqa_plain_image(inputs):
    model = Client().get_model('mock-model')
    res = model.vqa(image=inputs[0], question=inputs[1])
    assert res == 'a lot more than you think'
