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
@patch('client.client.fetch_host', Mock(return_value='grpc://mock.inference.jina.ai'))
@patch('client.client.login', Mock(return_value='valid_token'))
@patch(
    'client.base.BaseClient.vqa',
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
    res.summary()
    res[0].summary()


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
@patch('client.client.fetch_host', Mock(return_value='grpc://mock.inference.jina.ai'))
@patch('client.client.login', Mock(return_value='valid_token'))
@patch(
    'client.base.BaseClient.vqa',
    Mock(
        return_value=DocumentArray(
            [
                Document(
                    uri='https://picsum.photos/id/233/100',
                    tags={'response': 'a lot more than you think'},
                ).load_uri_to_blob(),
            ]
        ),
    ),
)
def test_vqa_plain_str(inputs):
    model = Client().get_model('mock-model')
    res = model.vqa(image=inputs[0], question=inputs[1])
    res.summary()
    res[0].summary()


@pytest.mark.parametrize(
    'inputs',
    [
        [
            Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob,
            'Question: how many cats are there? Answer:',
        ],
    ],
)
@patch('client.client.fetch_host', Mock(return_value='grpc://mock.inference.jina.ai'))
@patch('client.client.login', Mock(return_value='valid_token'))
@patch(
    'client.base.BaseClient.vqa',
    Mock(
        return_value=DocumentArray(
            [
                Document(
                    uri='https://picsum.photos/id/233/100',
                    tags={'response': 'a lot more than you think'},
                ).load_uri_to_blob(),
            ]
        ),
    ),
)
def test_vqa_plain_blob(inputs):
    model = Client().get_model('mock-model')
    res = model.vqa(image=inputs[0], question=inputs[1])
    res.summary()
    res[0].summary()


@pytest.mark.parametrize(
    'inputs',
    [
        [
            Document(uri='https://picsum.photos/id/233/100')
            .load_uri_to_image_tensor()
            .tensor,
            'Question: how many cats are there? Answer:',
        ],
    ],
)
@patch('client.client.fetch_host', Mock(return_value='grpc://mock.inference.jina.ai'))
@patch('client.client.login', Mock(return_value='valid_token'))
@patch(
    'client.base.BaseClient.vqa',
    Mock(
        return_value=DocumentArray(
            [
                Document(
                    uri='https://picsum.photos/id/233/100',
                    tags={'response': 'a lot more than you think'},
                ).load_uri_to_image_tensor(),
            ]
        ),
    ),
)
def test_vqa_plain_tensor(inputs):
    model = Client().get_model('mock-model')
    res = model.vqa(image=inputs[0], question=inputs[1])
    res.summary()
    res[0].summary()
