from unittest.mock import Mock, patch

import pytest
from docarray import Document, DocumentArray

from client import Client


@pytest.mark.parametrize(
    'inputs',
    [
        DocumentArray(
            [
                Document(text='hello world'),
                Document(uri='https://picsum.photos/id/233/100'),
            ]
        ),
        [
            Document(text='hello world'),
            Document(uri='https://picsum.photos/id/233/100'),
        ],
    ],
)
@patch('client.client.fetch_host')
def test_encode_document(mock_fetch_host, make_model_name_and_host, inputs):
    mock_fetch_host.return_value = make_model_name_and_host[1]

    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    model = client.get_model(make_model_name_and_host[0])
    res = model.encode(docs=inputs)
    res.summary()
    res[0].summary()


@pytest.mark.parametrize('inputs', ['hello world'])
@patch('client.client.fetch_host')
def test_encode_plain_text(mock_fetch_host, make_model_name_and_host, inputs):
    mock_fetch_host.return_value = make_model_name_and_host[1]

    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    model = client.get_model(make_model_name_and_host[0])
    res = model.encode(text=inputs)
    res.summary()
    res[0].summary()


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
@patch('client.client.fetch_host')
def test_encode_plain_image(mock_fetch_host, make_model_name_and_host, inputs):
    mock_fetch_host.return_value = make_model_name_and_host[1]

    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    model = client.get_model(make_model_name_and_host[0])
    res = model.encode(image=inputs)
    res.summary()
    res[0].summary()


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
@patch('client.client.fetch_host')
def test_caption_document(mock_fetch_host, make_model_name_and_host, inputs):
    mock_fetch_host.return_value = make_model_name_and_host[1]

    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    model = client.get_model(make_model_name_and_host[0])
    res = model.caption(docs=inputs)
    res.summary()
    res[0].summary()


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
@patch('client.client.fetch_host')
def test_caption_plain(mock_fetch_host, make_model_name_and_host, inputs):
    mock_fetch_host.return_value = make_model_name_and_host[1]

    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    model = client.get_model(make_model_name_and_host[0])
    res = model.caption(image=inputs)
    res.summary()
    res[0].summary()


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
        DocumentArray(
            [
                Document(
                    text='a black and white photo of nature',
                    matches=DocumentArray(
                        [
                            Document(uri='https://picsum.photos/id/233/100'),
                            Document(uri='https://picsum.photos/id/234/100'),
                            Document(uri='https://picsum.photos/id/235/100'),
                        ]
                    ),
                ),
            ]
        ),
        DocumentArray(
            [
                Document(
                    uri='https://picsum.photos/id/233/100',
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
        DocumentArray(
            [
                Document(
                    uri='https://picsum.photos/id/233/100',
                    matches=DocumentArray(
                        [
                            Document(uri='https://picsum.photos/id/233/100'),
                            Document(uri='https://picsum.photos/id/234/100'),
                            Document(uri='https://picsum.photos/id/235/100'),
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
        [
            Document(
                text='a black and white photo of nature',
                matches=DocumentArray(
                    [
                        Document(uri='https://picsum.photos/id/233/100'),
                        Document(uri='https://picsum.photos/id/234/100'),
                        Document(uri='https://picsum.photos/id/235/100'),
                    ]
                ),
            ),
        ],
        [
            Document(
                uri='https://picsum.photos/id/233/100',
                matches=DocumentArray(
                    [
                        Document(text='a colorful photo of nature'),
                        Document(text='a black and white photo of a dog'),
                        Document(text='a black and white photo of a cat'),
                    ]
                ),
            ),
        ],
        [
            Document(
                uri='https://picsum.photos/id/233/100',
                matches=DocumentArray(
                    [
                        Document(uri='https://picsum.photos/id/233/100'),
                        Document(uri='https://picsum.photos/id/234/100'),
                        Document(uri='https://picsum.photos/id/235/100'),
                    ]
                ),
            ),
        ],
    ],
)
@patch('client.client.fetch_host')
def test_rank_document(mock_fetch_host, make_model_name_and_host, inputs):
    mock_fetch_host.return_value = make_model_name_and_host[1]

    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    model = client.get_model(make_model_name_and_host[0])
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
                'a black and white photo of a dog',
                'a black and white photo of a cat',
            ],
        ],
        [
            'a black and white photo of nature',
            [
                'https://picsum.photos/id/233/100',
                'https://picsum.photos/id/234/100',
                'https://picsum.photos/id/235/100',
            ],
        ],
        [
            'https://picsum.photos/id/233/100',
            [
                'a colorful photo of nature',
                'a black and white photo of a dog',
                'a black and white photo of a cat',
            ],
        ],
        [
            'https://picsum.photos/id/233/100',
            [
                'https://picsum.photos/id/233/100',
                'https://picsum.photos/id/234/100',
                'https://picsum.photos/id/235/100',
            ],
        ],
    ],
)
@patch('client.client.fetch_host')
def test_rank_plain_input(mock_fetch_host, make_model_name_and_host, inputs):
    mock_fetch_host.return_value = make_model_name_and_host[1]

    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    model = client.get_model(make_model_name_and_host[0])
    res = model.rank(reference=inputs[0], candidates=inputs[1])
    res.summary()
    res[0].summary()


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
@patch('client.client.fetch_host')
def test_vqa_document(mock_fetch_host, make_model_name_and_host, inputs):
    mock_fetch_host.return_value = make_model_name_and_host[1]

    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    model = client.get_model(make_model_name_and_host[0])
    res = model.vqa(docs=inputs)
    res.summary()
    res[0].summary()


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
@patch('client.client.fetch_host')
def test_vqa_plain(mock_fetch_host, make_model_name_and_host, inputs):
    mock_fetch_host.return_value = make_model_name_and_host[1]

    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    model = client.get_model(make_model_name_and_host[0])
    res = model.vqa(image=inputs[0], question=inputs[1])
    res.summary()
    res[0].summary()
