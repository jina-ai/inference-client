from unittest.mock import Mock, patch

import pytest
from docarray import Document, DocumentArray

from client import Client


@patch('client.client.fetch_host', Mock(return_value='grpcs://api.clip.jina.ai:2096'))
def test_clip_encode_documentarray():
    docs = DocumentArray([Document(text='hello world'), Document(text='hello jina')])
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    clip = client.get_model('ViT-B-32::openai')
    res = clip.encode(docs=docs)
    res.summary()
    res[0].summary()


@patch('client.client.fetch_host', Mock(return_value='grpcs://api.clip.jina.ai:2096'))
def test_clip_encode_list_of_document():
    docs = [Document(text='hello world'), Document(text='hello jina')]
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    clip = client.get_model('ViT-B-32::openai')
    res = clip.encode(docs=docs)
    res.summary()
    res[0].summary()


@patch('client.client.fetch_host', Mock(return_value='grpcs://api.clip.jina.ai:2096'))
def test_clip_encode_plain_text():
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    clip = client.get_model('ViT-B-32::openai')
    res = clip.encode(text='hello world')
    res.summary()
    res[0].summary()


@patch('client.client.fetch_host', Mock(return_value='grpcs://api.clip.jina.ai:2096'))
def test_clip_encode_plain_image_uri():
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    clip = client.get_model('ViT-B-32::openai')
    res = clip.encode(image='https://picsum.photos/id/233/100')
    res.summary()
    res[0].summary()


@patch('client.client.fetch_host', Mock(return_value='grpcs://api.clip.jina.ai:2096'))
def test_clip_encode_plain_image_blob():
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    clip = client.get_model('ViT-B-32::openai')
    res = clip.encode(
        image=Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob
    )
    res.summary()
    res[0].summary()


@patch('client.client.fetch_host', Mock(return_value='grpcs://api.clip.jina.ai:2096'))
def test_clip_encode_plain_image_tensor():
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    clip = client.get_model('ViT-B-32::openai')
    res = clip.encode(
        image=Document(uri='https://picsum.photos/id/233/100')
        .load_uri_to_image_tensor()
        .tensor
    )
    res.summary()
    res[0].summary()


@patch('client.client.fetch_host', Mock(return_value='grpcs://api.clip.jina.ai:2096'))
def test_clip_caption():
    docs = DocumentArray(
        [Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob()]
    )
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    clip = client.get_model('ViT-B-32::openai')
    res = clip.caption(
        docs=docs, parameters={'num_captions': 3, 'use_nucleus_sampling': True}
    )
    res.summary()
    res[0].summary()


@patch(
    'client.client.fetch_host',
    Mock(return_value='grpcs://crucial-gazelle-779d1c8739-grpc.wolf.jina.ai'),
)
def test_blip2_encode_documentarray():
    docs = DocumentArray([Document(text='hello world'), Document(text='hello jina')])
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    blip2 = client.get_model('Salesforce/blip2-opt-2.7b')
    res = blip2.encode(docs=docs)
    res.summary()
    res[0].summary()


@patch(
    'client.client.fetch_host',
    Mock(return_value='grpcs://crucial-gazelle-779d1c8739-grpc.wolf.jina.ai'),
)
def test_blip2_encode_list_of_document():
    docs = [Document(text='hello world'), Document(text='hello jina')]
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    blip2 = client.get_model('Salesforce/blip2-opt-2.7b')
    res = blip2.encode(docs=docs)
    res.summary()
    res[0].summary()


@patch(
    'client.client.fetch_host',
    Mock(return_value='grpcs://crucial-gazelle-779d1c8739-grpc.wolf.jina.ai'),
)
def test_blip2_encode_plain_text():
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    blip2 = client.get_model('Salesforce/blip2-opt-2.7b')
    res = blip2.encode(text='hello world')
    res.summary()
    res[0].summary()


@patch(
    'client.client.fetch_host',
    Mock(return_value='grpcs://crucial-gazelle-779d1c8739-grpc.wolf.jina.ai'),
)
def test_blip2_encode_plain_image_uri():
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    blip2 = client.get_model('Salesforce/blip2-opt-2.7b')
    res = blip2.encode(image='https://picsum.photos/id/233/100')
    res.summary()
    res[0].summary()


@patch(
    'client.client.fetch_host',
    Mock(return_value='grpcs://crucial-gazelle-779d1c8739-grpc.wolf.jina.ai'),
)
def test_blip2_encode_plain_image_blob():
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    blip2 = client.get_model('Salesforce/blip2-opt-2.7b')
    res = blip2.encode(
        image=Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob
    )
    res.summary()
    res[0].summary()


@patch(
    'client.client.fetch_host',
    Mock(return_value='grpcs://crucial-gazelle-779d1c8739-grpc.wolf.jina.ai'),
)
def test_blip2_encode_plain_image_tensor():
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    blip2 = client.get_model('Salesforce/blip2-opt-2.7b')
    res = blip2.encode(
        image=Document(uri='https://picsum.photos/id/233/100')
        .load_uri_to_image_tensor()
        .tensor
    )
    res.summary()
    res[0].summary()


@patch(
    'client.client.fetch_host',
    Mock(return_value='grpcs://crucial-gazelle-779d1c8739-grpc.wolf.jina.ai'),
)
def test_blip2_caption_documentarray():
    docs = DocumentArray(
        [Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob()]
    )
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    blip2 = client.get_model('Salesforce/blip2-opt-2.7b')
    res = blip2.caption(docs=docs)
    res.summary()
    res[0].summary()


@patch(
    'client.client.fetch_host',
    Mock(return_value='grpcs://crucial-gazelle-779d1c8739-grpc.wolf.jina.ai'),
)
def test_blip2_caption_list_of_document():
    docs = [Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob()]
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    blip2 = client.get_model('Salesforce/blip2-opt-2.7b')
    res = blip2.caption(docs=docs)
    res.summary()
    res[0].summary()


@patch(
    'client.client.fetch_host',
    Mock(return_value='grpcs://crucial-gazelle-779d1c8739-grpc.wolf.jina.ai'),
)
def test_blip2_caption_plain_image_uri():
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    blip2 = client.get_model('Salesforce/blip2-opt-2.7b')
    res = blip2.caption(image='https://picsum.photos/id/233/100')
    res.summary()
    res[0].summary()


@patch(
    'client.client.fetch_host',
    Mock(return_value='grpcs://crucial-gazelle-779d1c8739-grpc.wolf.jina.ai'),
)
def test_blip2_caption_plain_image_blob():
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    blip2 = client.get_model('Salesforce/blip2-opt-2.7b')
    res = blip2.caption(
        image=Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob
    )
    res.summary()
    res[0].summary()


@patch(
    'client.client.fetch_host',
    Mock(return_value='grpcs://crucial-gazelle-779d1c8739-grpc.wolf.jina.ai'),
)
def test_blip2_caption_plain_image_tensor():
    client = Client(token='ebf1afcf5c9432ed5662d8b1d6e20303')
    blip2 = client.get_model('Salesforce/blip2-opt-2.7b')
    res = blip2.caption(
        image=Document(uri='https://picsum.photos/id/233/100')
        .load_uri_to_image_tensor()
        .tensor
    )
    res.summary()
    res[0].summary()
