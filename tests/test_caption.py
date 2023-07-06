import os

import pytest
from docarray import Document, DocumentArray


@pytest.mark.parametrize(
    'inputs',
    [
        DocumentArray(
            [
                Document(uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg'),
            ]
        ),
        [
            Document(uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg'),
        ],
    ],
)
def test_caption_document(make_client, inputs):
    res = make_client.caption(docs=inputs)
    assert isinstance(res, DocumentArray)
    assert res[0].tags['response'] == 'A image of something very nice'


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
def test_caption_plain_image(make_client, inputs):
    res = make_client.caption(image=inputs)
    assert res == 'A image of something very nice'
