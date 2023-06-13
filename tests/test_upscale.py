import os

import pytest
from docarray import Document, DocumentArray

from .helper import what


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


@pytest.mark.parametrize(
    'inputs',
    [
        (
            DocumentArray(
                [
                    Document(
                        uri='https://picsum.photos/id/237/200/300',
                        tags={'image_format': 'jpeg', 'output_path': 'test0.jpeg'},
                    )
                ]
            ),
            'test0.jpeg',
            'jpeg',
        ),
        (
            DocumentArray(
                [
                    Document(
                        uri='https://picsum.photos/id/237/200/300',
                        tags={'image_format': 'png', 'output_path': 'test1.png'},
                    )
                ]
            ),
            'test1.png',
            'png',
        ),
        (
            DocumentArray(
                [
                    Document(
                        uri='https://picsum.photos/id/237/200/300',
                        tags={'output_path': 'test2.jpeg'},
                    )
                ]
            ),
            'test2.jpeg',
            'jpeg',
        ),
    ],
)
def test_upscale_document_savefile(make_client, inputs):
    res = make_client.upscale(docs=inputs[0])
    assert isinstance(res, DocumentArray)
    assert os.path.exists(inputs[1])
    assert what(inputs[1]) == inputs[2]


@pytest.mark.parametrize(
    'inputs',
    [
        ('https://picsum.photos/id/237/200/300', 'jpeg', 'test0.jpeg', 'jpeg'),
        ('https://picsum.photos/id/237/200/300', 'png', 'test1.png', 'png'),
        ('https://picsum.photos/id/237/200/300', None, 'test2.jpeg', 'jpeg'),
    ],
)
def test_upscale_plain_image_savefile(make_client, inputs):
    res = make_client.upscale(
        image=inputs[0], image_format=inputs[1], output_path=inputs[2]
    )
    assert os.path.exists(inputs[2])
    assert what(inputs[2]) == inputs[3]


def test_upscale_document_quality(make_client):
    doc0 = Document(
        uri='https://picsum.photos/id/237/200/300',
        tags={'image_format': 'jpeg', 'output_path': 'test0.jpeg'},
    )
    doc1 = Document(
        uri='https://picsum.photos/id/237/200/300',
        tags={'image_format': 'jpeg', 'output_path': 'test1.jpeg'},
    )
    doc2 = Document(
        uri='https://picsum.photos/id/237/200/300',
        tags={'image_format': 'jpeg', 'output_path': 'test2.jpeg'},
    )
    make_client.upscale(docs=[doc0], quality=75)
    make_client.upscale(docs=[doc1], quality=50)
    make_client.upscale(docs=[doc2], quality=25)
    assert os.path.exists('test0.jpeg')
    assert os.path.exists('test1.jpeg')
    assert os.path.exists('test2.jpeg')
    assert os.path.getsize('test0.jpeg') > os.path.getsize('test1.jpeg')
    assert os.path.getsize('test1.jpeg') > os.path.getsize('test2.jpeg')
    os.remove('test0.jpeg')
    os.remove('test1.jpeg')
    os.remove('test2.jpeg')


def test_upscale_plain_image_quality(make_client):
    make_client.upscale(
        image='https://picsum.photos/id/237/200/300',
        quality=75,
        output_path='test0.jpeg',
    )
    make_client.upscale(
        image='https://picsum.photos/id/237/200/300',
        quality=50,
        output_path='test1.jpeg',
    )
    make_client.upscale(
        image='https://picsum.photos/id/237/200/300',
        quality=25,
        output_path='test2.jpeg',
    )
    assert os.path.exists('test0.jpeg')
    assert os.path.exists('test1.jpeg')
    assert os.path.exists('test2.jpeg')
    assert os.path.getsize('test0.jpeg') > os.path.getsize('test1.jpeg')
    assert os.path.getsize('test1.jpeg') > os.path.getsize('test2.jpeg')
    os.remove('test0.jpeg')
    os.remove('test1.jpeg')
    os.remove('test2.jpeg')


@pytest.mark.parametrize(
    'inputs',
    [
        (123, 'Quality should be an integer between 0 and 100.'),
        (-123, 'Quality should be an integer between 0 and 100.'),
    ],
)
def test_upscale_invalid_quality(make_client, inputs):
    with pytest.raises(ValueError) as e:
        make_client.upscale(
            image='https://picsum.photos/id/237/200/300', quality=inputs[0]
        )
        assert str(e.value) == inputs[1]
