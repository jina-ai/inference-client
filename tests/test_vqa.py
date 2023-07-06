import os

import pytest
from docarray import Document, DocumentArray


@pytest.mark.parametrize(
    'inputs',
    [
        DocumentArray(
            [
                Document(
                    uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
                    tags={'prompt': 'Question: how many cats are there? Answer:'},
                ),
            ]
        ),
        [
            Document(
                uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
                tags={'prompt': 'Question: how many cats are there? Answer:'},
            ),
        ],
    ],
)
def test_vqa_document(make_client, inputs):
    res = make_client.vqa(docs=inputs)
    assert isinstance(res, DocumentArray)
    assert res[0].tags['response'] == 'Yes, it is a cat'


@pytest.mark.parametrize(
    'inputs',
    [
        [
            f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
            'Question: how many cats are there? Answer:',
        ],
        [
            Document(uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg')
            .load_uri_to_blob()
            .blob,
            'Question: how many cats are there? Answer:',
        ],
        [
            Document(uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg')
            .load_uri_to_image_tensor()
            .tensor,
            'Question: how many cats are there? Answer:',
        ],
    ],
)
def test_vqa_plain_image(make_client, inputs):
    res = make_client.vqa(image=inputs[0], question=inputs[1])
    assert res == 'Yes, it is a cat'
