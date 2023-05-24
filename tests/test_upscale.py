import pytest
from docarray import Document, DocumentArray


@pytest.mark.parametrize(
    'inputs',
    [
        (
            DocumentArray(
                [
                    Document(uri='https://picsum.photos/id/237/200/300'),
                ]
            ),
            None,
            (2400, 1600, 3),
        ),
        (
            DocumentArray(
                [
                    Document(uri='https://picsum.photos/id/237/200/300'),
                ]
            ),
            '600:800',
            (800, 600, 3),
        ),
        (
            DocumentArray(
                [
                    Document(uri='https://picsum.photos/id/237/200/300'),
                ]
            ),
            '600:-1',
            (900, 600, 3),
        ),
        (
            DocumentArray(
                [
                    Document(uri='https://picsum.photos/id/237/200/300'),
                ]
            ),
            '-1:600',
            (600, 400, 3),
        ),
        (
            DocumentArray(
                [
                    Document(uri='https://picsum.photos/id/237/200/300'),
                ]
            ),
            '600:-7',
            (903, 600, 3),
        ),
        (
            DocumentArray(
                [
                    Document(uri='https://picsum.photos/id/237/200/300'),
                ]
            ),
            '-7:600',
            (600, 406, 3),
        ),
        (
            DocumentArray(
                [
                    Document(uri='https://picsum.photos/id/237/200/300'),
                ]
            ),
            '600:0',
            (300, 600, 3),
        ),
        (
            DocumentArray(
                [
                    Document(uri='https://picsum.photos/id/237/200/300'),
                ]
            ),
            '0:600',
            (600, 200, 3),
        ),
        (
            DocumentArray(
                [
                    Document(uri='https://picsum.photos/id/237/200/300'),
                ]
            ),
            '0:0',
            (300, 200, 3),
        ),
        (
            DocumentArray(
                [
                    Document(uri='https://picsum.photos/id/237/200/300'),
                ]
            ),
            '-1:-1',
            (300, 200, 3),
        ),
        (
            [
                Document(uri='https://picsum.photos/id/237/200/300'),
            ],
            None,
            (2400, 1600, 3),
        ),
        (
            [
                Document(uri='https://picsum.photos/id/237/200/300'),
            ],
            '600:800',
            (800, 600, 3),
        ),
        (
            [
                Document(uri='https://picsum.photos/id/237/200/300'),
            ],
            '600:-1',
            (900, 600, 3),
        ),
        (
            [
                Document(uri='https://picsum.photos/id/237/200/300'),
            ],
            '-1:600',
            (600, 400, 3),
        ),
        (
            [
                Document(uri='https://picsum.photos/id/237/200/300'),
            ],
            '600:-7',
            (903, 600, 3),
        ),
        (
            [
                Document(uri='https://picsum.photos/id/237/200/300'),
            ],
            '-7:600',
            (600, 406, 3),
        ),
        (
            [
                Document(uri='https://picsum.photos/id/237/200/300'),
            ],
            '600:0',
            (300, 600, 3),
        ),
        (
            [
                Document(uri='https://picsum.photos/id/237/200/300'),
            ],
            '0:600',
            (600, 200, 3),
        ),
        (
            [
                Document(uri='https://picsum.photos/id/237/200/300'),
            ],
            '0:0',
            (300, 200, 3),
        ),
        (
            [
                Document(uri='https://picsum.photos/id/237/200/300'),
            ],
            '-1:-1',
            (300, 200, 3),
        ),
    ],
)
def test_upscale_document(make_client, inputs):
    res = make_client.upscale(docs=inputs[0], scale=inputs[1])
    assert isinstance(res, DocumentArray)
    assert res[0].convert_blob_to_image_tensor().tensor.shape == inputs[2]


@pytest.mark.parametrize(
    'inputs',
    [
        (
            'https://picsum.photos/id/237/200/300',
            None,
            (2400, 1600, 3),
        ),
        (
            'https://picsum.photos/id/237/200/300',
            '600:800',
            (800, 600, 3),
        ),
        (
            'https://picsum.photos/id/237/200/300',
            '600:-1',
            (900, 600, 3),
        ),
        (
            'https://picsum.photos/id/237/200/300',
            '-1:600',
            (600, 400, 3),
        ),
        (
            'https://picsum.photos/id/237/200/300',
            '600:-7',
            (903, 600, 3),
        ),
        (
            'https://picsum.photos/id/237/200/300',
            '-7:600',
            (600, 406, 3),
        ),
        (
            'https://picsum.photos/id/237/200/300',
            '600:0',
            (300, 600, 3),
        ),
        (
            'https://picsum.photos/id/237/200/300',
            '0:600',
            (600, 200, 3),
        ),
        (
            'https://picsum.photos/id/237/200/300',
            '0:0',
            (300, 200, 3),
        ),
        (
            'https://picsum.photos/id/237/200/300',
            '-1:-1',
            (300, 200, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_blob()
            .blob,
            None,
            (2400, 1600, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_blob()
            .blob,
            '600:800',
            (800, 600, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_blob()
            .blob,
            '600:-1',
            (900, 600, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_blob()
            .blob,
            '-1:600',
            (600, 400, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_blob()
            .blob,
            '600:-7',
            (903, 600, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_blob()
            .blob,
            '-7:600',
            (600, 406, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_blob()
            .blob,
            '600:0',
            (300, 600, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_blob()
            .blob,
            '0:600',
            (600, 200, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_blob()
            .blob,
            '0:0',
            (300, 200, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_blob()
            .blob,
            '-1:-1',
            (300, 200, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_image_tensor()
            .tensor,
            None,
            (2400, 1600, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_image_tensor()
            .tensor,
            '600:800',
            (800, 600, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_image_tensor()
            .tensor,
            '600:-1',
            (900, 600, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_image_tensor()
            .tensor,
            '-1:600',
            (600, 400, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_image_tensor()
            .tensor,
            '600:-7',
            (903, 600, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_image_tensor()
            .tensor,
            '-7:600',
            (600, 406, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_image_tensor()
            .tensor,
            '600:0',
            (300, 600, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_image_tensor()
            .tensor,
            '0:600',
            (600, 200, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_image_tensor()
            .tensor,
            '0:0',
            (300, 200, 3),
        ),
        (
            Document(uri='https://picsum.photos/id/237/200/300')
            .load_uri_to_image_tensor()
            .tensor,
            '-1:-1',
            (300, 200, 3),
        ),
    ],
)
def test_upscale_plain_image(make_client, inputs):
    res = make_client.upscale(image=inputs[0], scale=inputs[1])
    assert isinstance(res, bytes)
    assert Document(blob=res).convert_blob_to_image_tensor().tensor.shape == inputs[2]


def test_upscale_invalid_input(make_client):
    with pytest.raises(ValueError) as e:
        make_client.upscale(image='invalid', docs='invalid')
        assert (
            str(e.value)
            == 'More than one input type provided. Please provide only docs or image input.'
        )


@pytest.mark.parametrize(
    'inputs',
    [
        (123, 'Scale should be a string.'),
        ('1:2:3', 'Scale should be in the format of `width:height`.'),
        ('100:12.34', 'Both width and height should be integers.'),
        ('12.34:100', 'Both width and height should be integers.'),
        ('12.34:45.78', 'Both width and height should be integers.'),
    ],
)
def test_upscale_invalid_scale(make_client, inputs):
    with pytest.raises(ValueError) as e:
        make_client.upscale(
            image='https://picsum.photos/id/237/200/300', scale=inputs[0]
        )
        assert str(e.value) == inputs[1]
