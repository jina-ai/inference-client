import os

import numpy as np
import pytest
from docarray import Document, DocumentArray


@pytest.mark.parametrize(
    'inputs',
    [
        DocumentArray(
            [
                Document(
                    uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
                    tags={'prompt': 'A dog is sleeping on the floor.'},
                ),
            ]
        ),
        [
            Document(
                uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
                tags={'prompt': 'A dog is sleeping on the floor.'},
            ),
        ],
    ],
)
def test_image_to_image_document(make_client, inputs):
    res = make_client.image_to_image(docs=inputs)
    assert isinstance(res, DocumentArray)
    assert len(res[0].blob) > 0
    assert len(res[0].matches[0].blob) > 0


@pytest.mark.parametrize(
    'inputs',
    [
        DocumentArray(
            [
                Document(
                    uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
                    tags={'prompt': 'A dog is sleeping on the floor.'},
                ),
            ]
        ),
        [
            Document(
                uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
                tags={'prompt': 'A dog is sleeping on the floor.'},
            ),
        ],
    ],
)
def test_image_to_image_document_2_images_per_prompt(make_client, inputs):
    res = make_client.image_to_image(
        docs=inputs, parameters={'num_images_per_prompt': 2}
    )
    assert isinstance(res, DocumentArray)
    assert len(res[0].blob) > 0
    assert len(res[0].matches) == 2
    assert len(res[0].matches[0].blob) > 0
    assert len(res[0].matches[1].blob) > 0


@pytest.mark.parametrize(
    'inputs',
    [
        DocumentArray(
            [
                Document(
                    uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
                    tags={'prompt': 'A dog is sleeping on the floor.'},
                ),
            ]
        ),
        [
            Document(
                uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
                tags={'prompt': 'A dog is sleeping on the floor.'},
            ),
        ],
    ],
)
def test_image_to_image_document_latent_output(make_client, inputs):
    res = make_client.image_to_image(docs=inputs, parameters={'output_type': 'latent'})
    assert isinstance(res, DocumentArray)
    assert len(res[0].blob) > 0
    assert res[0].matches[0].tensor is not None


@pytest.mark.parametrize(
    'inputs',
    [
        DocumentArray(
            [
                Document(
                    uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
                    tags={'prompt': 'A dog is sleeping on the floor.'},
                ),
            ]
        ),
        [
            Document(
                uri=f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
                tags={'prompt': 'A dog is sleeping on the floor.'},
            ),
        ],
    ],
)
def test_image_to_image_document_latent_output_2_images_per_prompt(make_client, inputs):
    res = make_client.image_to_image(
        docs=inputs, parameters={'output_type': 'latent', 'num_images_per_prompt': 2}
    )
    assert isinstance(res, DocumentArray)
    assert len(res[0].blob) > 0
    assert len(res[0].matches) == 2
    assert res[0].matches[0].tensor is not None
    assert res[0].matches[1].tensor is not None


@pytest.mark.parametrize(
    'inputs',
    [
        [
            f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
            'A dog is sleeping on the floor.',
        ]
    ],
)
def test_image_to_image_plain(make_client, inputs):
    res = make_client.image_to_image(image=inputs[0], prompt=inputs[1])
    assert isinstance(res, bytes)


@pytest.mark.parametrize(
    'inputs',
    [
        [
            f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
            'A dog is sleeping on the floor.',
        ]
    ],
)
def test_image_to_image_plain_2_images_per_prompt(make_client, inputs):
    res = make_client.image_to_image(
        image=inputs[0], prompt=inputs[1], parameters={'num_images_per_prompt': 2}
    )
    assert isinstance(res, list)
    assert len(res) == 2
    assert isinstance(res[0], bytes)
    assert isinstance(res[1], bytes)


@pytest.mark.parametrize(
    'inputs',
    [
        [
            f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
            'A dog is sleeping on the floor.',
        ]
    ],
)
def test_image_to_image_plain_latent_output(make_client, inputs):
    res = make_client.image_to_image(
        image=inputs[0], prompt=inputs[1], parameters={'output_type': 'latent'}
    )
    assert isinstance(res, np.ndarray)


@pytest.mark.parametrize(
    'inputs',
    [
        [
            f'{os.path.dirname(os.path.abspath(__file__))}/test.jpeg',
            'A dog is sleeping on the floor.',
        ]
    ],
)
def test_text_to_image_plain_latent_output_2_images_per_prompt(make_client, inputs):
    res = make_client.image_to_image(
        image=inputs[0],
        prompt=inputs[1],
        parameters={'output_type': 'latent', 'num_images_per_prompt': 2},
    )
    assert isinstance(res, list)
    assert len(res) == 2
    assert isinstance(res[0], np.ndarray)
    assert isinstance(res[1], np.ndarray)
