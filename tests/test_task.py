from unittest.mock import Mock, patch

import pytest
from docarray import Document, DocumentArray

from client import Client


@pytest.mark.parametrize(
    'input',
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
@pytest.mark.parametrize(
    ('model_name', 'model_host'),
    [
        ('ViT-B-32::openai', 'grpcs://api.clip.jina.ai:2096'),
        (
            'Salesforce/blip2-opt-2.7b',
            'grpcs://crucial-gazelle-779d1c8739-grpc.wolf.jina.ai',
        ),
    ],
)
@patch('client.client.fetch_host')
def test_encode_document(mock_fetch_host, model_name, model_host, input):
    mock_fetch_host.return_value = model_host

    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    model = client.get_model(model_name)
    res = model.encode(docs=input)
    res.summary()
    res[0].summary()


@pytest.mark.parametrize('input', ['hello world'])
@pytest.mark.parametrize(
    ('model_name', 'model_host'),
    [
        ('ViT-B-32::openai', 'grpcs://api.clip.jina.ai:2096'),
        (
            'Salesforce/blip2-opt-2.7b',
            'grpcs://crucial-gazelle-779d1c8739-grpc.wolf.jina.ai',
        ),
    ],
)
@patch('client.client.fetch_host')
def test_encode_plain_text(mock_fetch_host, model_name, model_host, input):
    mock_fetch_host.return_value = model_host

    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    model = client.get_model(model_name)
    res = model.encode(text=input)
    res.summary()
    res[0].summary()


@pytest.mark.parametrize(
    'input',
    [
        'https://picsum.photos/id/233/100',
        Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob,
        Document(uri='https://picsum.photos/id/233/100')
        .load_uri_to_image_tensor()
        .tensor,
    ],
)
@pytest.mark.parametrize(
    ('model_name', 'model_host'),
    [
        ('ViT-B-32::openai', 'grpcs://api.clip.jina.ai:2096'),
        (
            'Salesforce/blip2-opt-2.7b',
            'grpcs://crucial-gazelle-779d1c8739-grpc.wolf.jina.ai',
        ),
    ],
)
@patch('client.client.fetch_host')
def test_encode_plain_image(mock_fetch_host, model_name, model_host, input):
    mock_fetch_host.return_value = model_host

    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    model = client.get_model(model_name)
    res = model.encode(image=input)
    res.summary()
    res[0].summary()


@pytest.mark.parametrize(
    'input',
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
@pytest.mark.parametrize(
    ('model_name', 'model_host'),
    [
        ('ViT-B-32::openai', 'grpcs://api.clip.jina.ai:2096'),
        (
            'Salesforce/blip2-opt-2.7b',
            'grpcs://crucial-gazelle-779d1c8739-grpc.wolf.jina.ai',
        ),
    ],
)
@patch('client.client.fetch_host')
def test_caption_document(mock_fetch_host, model_name, model_host, input):
    mock_fetch_host.return_value = model_host

    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    model = client.get_model(model_name)
    res = model.caption(docs=input)
    res.summary()
    res[0].summary()


@pytest.mark.parametrize(
    'input',
    [
        'https://picsum.photos/id/233/100',
        Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob,
        Document(uri='https://picsum.photos/id/233/100')
        .load_uri_to_image_tensor()
        .tensor,
    ],
)
@pytest.mark.parametrize(
    ('model_name', 'model_host'),
    [
        ('ViT-B-32::openai', 'grpcs://api.clip.jina.ai:2096'),
        (
            'Salesforce/blip2-opt-2.7b',
            'grpcs://crucial-gazelle-779d1c8739-grpc.wolf.jina.ai',
        ),
    ],
)
@patch('client.client.fetch_host')
def test_caption_plain_image(mock_fetch_host, model_name, model_host, input):
    mock_fetch_host.return_value = model_host

    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    model = client.get_model(model_name)
    res = model.caption(image=input)
    res.summary()
    res[0].summary()
