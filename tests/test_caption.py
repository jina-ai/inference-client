import pytest
from docarray import Document, DocumentArray


@pytest.mark.parametrize(
    'inputs',
    [
        DocumentArray(
            [
                Document(uri='https://picsum.photos/id/233/100'),
            ]
        ),
        [
            Document(uri='https://picsum.photos/id/233/100'),
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
        'https://picsum.photos/id/233/100',
        Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob,
        Document(uri='https://picsum.photos/id/233/100')
        .load_uri_to_image_tensor()
        .tensor,
    ],
)
def test_encode_plain_image(make_client, inputs):
    res = make_client.caption(image=inputs)
    assert res == 'A image of something very nice'
