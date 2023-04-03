from unittest.mock import Mock, patch

import pytest
from docarray import Document, DocumentArray

from client import Client


@pytest.mark.parametrize(
    'inputs',
    [
        DocumentArray(
            [
                Document(
                    text='a black and white photo of nature',
                    matches=DocumentArray(
                        [
                            Document(text='a colorful photo of nature'),
                            Document(text='a black and white photo of a dog'),
                            Document(text='a black and white photo of a cat'),
                        ]
                    ),
                ),
            ]
        ),
        [
            Document(
                text='a black and white photo of nature',
                matches=DocumentArray(
                    [
                        Document(text='a colorful photo of nature'),
                        Document(text='a black and white photo of a dog'),
                        Document(text='a black and white photo of a cat'),
                    ]
                ),
            ),
        ],
    ],
)
@patch('client.client.fetch_host', Mock(return_value='grpc://mock.inference.jina.ai'))
@patch('client.client.login', Mock(return_value='valid_token'))
@patch(
    'client.base.BaseClient.rank',
    Mock(
        return_value=DocumentArray(
            [
                Document(
                    text='a black and white photo of nature',
                    matches=DocumentArray(
                        [
                            Document(text='a colorful photo of nature'),
                            Document(text='a black and white photo of a cat'),
                            Document(text='a black and white photo of a dog'),
                        ]
                    ),
                ),
            ]
        ),
    ),
)
def test_rank_document(inputs):
    model = Client().get_model('mock-model')
    res = model.rank(docs=inputs)
    res.summary()
    res[0].summary()


@pytest.mark.parametrize(
    'inputs',
    [
        [
            'a black and white photo of nature',
            [
                'a colorful photo of nature',
                'a black and white photo of a dog',
                'a black and white photo of a cat',
            ],
        ],
    ],
)
@patch('client.client.fetch_host', Mock(return_value='grpc://mock.inference.jina.ai'))
@patch('client.client.login', Mock(return_value='valid_token'))
@patch(
    'client.base.BaseClient.rank',
    Mock(
        return_value=DocumentArray(
            [
                Document(
                    text='a black and white photo of nature',
                    matches=DocumentArray(
                        [
                            Document(text='a colorful photo of nature'),
                            Document(text='a black and white photo of a cat'),
                            Document(text='a black and white photo of a dog'),
                        ]
                    ),
                ),
            ]
        ),
    ),
)
def test_rank_plain_input(inputs):
    model = Client().get_model('mock-model')
    res = model.rank(reference=inputs[0], candidates=inputs[1])
    res.summary()
    res[0].summary()
