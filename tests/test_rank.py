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
                'https://picsum.photos/id/232/100',
                Document(uri='https://picsum.photos/id/233/100')
                .load_uri_to_blob()
                .blob,
                Document(uri='https://picsum.photos/id/234/100')
                .load_uri_to_image_tensor()
                .tensor,
            ],
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
                    text='a black and white photo of nature',
                    matches=DocumentArray(
                        [
                            Document(
                                tensor=Document(uri='https://picsum.photos/id/234/100')
                                .load_uri_to_image_tensor()
                                .tensor
                            ),
                            Document(
                                blob=Document(uri='https://picsum.photos/id/233/100')
                                .load_uri_to_blob()
                                .blob
                            ),
                            Document(uri='https://picsum.photos/id/232/100'),
                            Document(text='a colorful photo of nature'),
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
    assert isinstance(res, list)
    assert len(res) == 4
    assert (
        res[0]
        == Document(uri='https://picsum.photos/id/234/100')
        .load_uri_to_image_tensor()
        .tensor
    ).all()
    assert (
        res[1]
        == Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob
    )
    assert res[2] == 'https://picsum.photos/id/232/100'
    assert res[3] == 'a colorful photo of nature'
