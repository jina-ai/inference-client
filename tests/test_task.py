import pytest
from docarray import Document, DocumentArray

from client import Client


def test_clip_encode():
    docs = DocumentArray([Document(text='hello world'), Document(text='hello jina')])
    client = Client('clip', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.encode(docs)
    res.summary()
    res[0].summary()


def test_clip_caption():
    docs = DocumentArray(
        [Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob()]
    )
    client = Client('clip', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.caption(
        docs, parameters={'num_captions': 3, 'use_nucleus_sampling': True}
    )
    res.summary()
    res[0].summary()


def test_blip_encode():
    docs = DocumentArray([Document(text='hello world'), Document(text='hello jina')])
    client = Client('blip', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.encode(docs)
    res.summary()
    res[0].summary()


def test_blip_caption():
    docs = DocumentArray(
        [Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob()]
    )
    client = Client('blip', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.caption(
        docs, parameters={'num_captions': 3, 'use_nucleus_sampling': True}
    )
    res.summary()
    res[0].summary()
