import pytest
from docarray import Document, DocumentArray

from client import Client


def test_encode():
    docs = DocumentArray([Document(text='hello world'), Document(text='hello jina')])
    client = Client('CLIP/ViT-B-32::openai', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.encode(docs)
    res.summary()
    res[0].summary()
