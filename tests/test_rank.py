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
                            Document(
                                text='a colorful photo of nature',
                                scores={
                                    'clip_score_cosine': {
                                        'value': 0.1,
                                        'op_name': 'cosine',
                                    },
                                    'clip_score': {'value': 0.4, 'op_name': 'softmax'},
                                },
                            ),
                            Document(
                                text='a black and white photo of a cat',
                                scores={
                                    'clip_score_cosine': {
                                        'value': 0.2,
                                        'op_name': 'cosine',
                                    },
                                    'clip_score': {'value': 0.5, 'op_name': 'softmax'},
                                },
                            ),
                            Document(
                                text='a black and white photo of a dog',
                                scores={
                                    'clip_score_cosine': {
                                        'value': 0.3,
                                        'op_name': 'cosine',
                                    },
                                    'clip_score': {'value': 0.6, 'op_name': 'softmax'},
                                },
                            ),
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
    assert isinstance(res, DocumentArray)
    assert res[0].matches[0].text == 'a colorful photo of nature'
    assert res[0].matches[0].scores['clip_score_cosine']['value'] == 0.1
    assert res[0].matches[0].scores['clip_score']['value'] == 0.4
    assert res[0].matches[1].text == 'a black and white photo of a cat'
    assert res[0].matches[1].scores['clip_score_cosine']['value'] == 0.2
    assert res[0].matches[1].scores['clip_score']['value'] == 0.5
    assert res[0].matches[2].text == 'a black and white photo of a dog'
    assert res[0].matches[2].scores['clip_score_cosine']['value'] == 0.3
    assert res[0].matches[2].scores['clip_score']['value'] == 0.6


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
                                .tensor,
                                scores={
                                    'clip_score_cosine': {
                                        'value': 0.1,
                                        'op_name': 'cosine',
                                    },
                                    'clip_score': {'value': 0.5, 'op_name': 'softmax'},
                                },
                            ),
                            Document(
                                blob=Document(uri='https://picsum.photos/id/233/100')
                                .load_uri_to_blob()
                                .blob,
                                scores={
                                    'clip_score_cosine': {
                                        'value': 0.2,
                                        'op_name': 'cosine',
                                    },
                                    'clip_score': {'value': 0.6, 'op_name': 'softmax'},
                                },
                            ),
                            Document(
                                uri='https://picsum.photos/id/232/100',
                                scores={
                                    'clip_score_cosine': {
                                        'value': 0.3,
                                        'op_name': 'cosine',
                                    },
                                    'clip_score': {'value': 0.7, 'op_name': 'softmax'},
                                },
                            ),
                            Document(
                                text='a colorful photo of nature',
                                scores={
                                    'clip_score_cosine': {
                                        'value': 0.4,
                                        'op_name': 'cosine',
                                    },
                                    'clip_score': {'value': 0.8, 'op_name': 'softmax'},
                                },
                            ),
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
    assert isinstance(res[0], tuple)
    assert (
        res[0][0]
        == Document(uri='https://picsum.photos/id/234/100')
        .load_uri_to_image_tensor()
        .tensor
    ).all()
    assert isinstance(res[0][1], dict)
    assert res[0][1]['clip_score_cosine']['value'] == 0.1
    assert res[0][1]['clip_score']['value'] == 0.5
    assert isinstance(res[1], tuple)
    assert (
        res[1][0]
        == Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob
    )
    assert isinstance(res[1][1], dict)
    assert res[1][1]['clip_score_cosine']['value'] == 0.2
    assert res[1][1]['clip_score']['value'] == 0.6
    assert isinstance(res[2], tuple)
    assert res[2][0] == 'https://picsum.photos/id/232/100'
    assert isinstance(res[2][1], dict)
    assert res[2][1]['clip_score_cosine']['value'] == 0.3
    assert res[2][1]['clip_score']['value'] == 0.7
    assert isinstance(res[3], tuple)
    assert res[3][0] == 'a colorful photo of nature'
    assert isinstance(res[3][1], dict)
    assert res[3][1]['clip_score_cosine']['value'] == 0.4
    assert res[3][1]['clip_score']['value'] == 0.8
