import pytest
from docarray import Document, DocumentArray


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
def test_rank_document(make_client, inputs):
    res = make_client.rank(docs=inputs)
    assert isinstance(res, DocumentArray)
    assert len(res[0].matches) == 3
    for m in res[0].matches:
        assert m.text is not None
        assert m.scores['cosine'].value is not None


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
def test_rank_plain_text(make_client, inputs):
    res = make_client.rank(text=inputs[0], candidates=inputs[1])
    assert isinstance(res, list)
    assert len(res) == 4
    assert isinstance(res[0], tuple)
    for r in res:
        assert r[0] is not None
        assert r[1]['cosine'].value is not None


@pytest.mark.parametrize(
    'inputs',
    [
        [
            'https://picsum.photos/id/232/100',
            [
                'a colorful photo of nature',
                'a lovely photo of cat',
                'a black and white photo of dog',
                'a cat playing with a dog',
            ],
        ],
    ],
)
def test_rank_plain_image(make_client, inputs):
    res = make_client.rank(image=inputs[0], candidates=inputs[1])
    assert isinstance(res, list)
    assert len(res) == 4
    assert isinstance(res[0], tuple)
    for r in res:
        assert r[0] is not None
        assert r[1]['cosine'].value is not None
