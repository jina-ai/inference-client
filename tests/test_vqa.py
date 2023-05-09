import pytest
from docarray import Document, DocumentArray

from inference_client.base import BaseClient


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
def test_vqa_document(make_flow, inputs):
    model = BaseClient(
        model_name='dummy-model',
        token='valid_token',
        host=f'grpc://0.0.0.0:{make_flow.port}',
    )
    res = model.vqa(docs=inputs)
    assert isinstance(res, DocumentArray)
    assert res[0].tags['response'] == 'Yes, it is a cat'


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
def test_vqa_plain_image(make_flow, inputs):
    model = BaseClient(
        model_name='dummy-model',
        token='valid_token',
        host=f'grpc://0.0.0.0:{make_flow.port}',
    )
    res = model.vqa(image=inputs[0], question=inputs[1])
    assert res == 'Yes, it is a cat'
