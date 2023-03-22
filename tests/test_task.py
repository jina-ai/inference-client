import pytest
from docarray import Document, DocumentArray

from client import Client


def test_clip_encode_documentarray():
    docs = DocumentArray([Document(text='hello world'), Document(text='hello jina')])
    client = Client('clip', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.encode(docs=docs)
    res.summary()
    res[0].summary()


def test_clip_encode_list_of_document():
    docs = [Document(text='hello world'), Document(text='hello jina')]
    client = Client('clip', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.encode(docs=docs)
    res.summary()
    res[0].summary()


def test_clip_encode_plain_text():
    client = Client('clip', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.encode(text='hello world')
    res.summary()
    res[0].summary()


def test_clip_encode_plain_image_uri():
    client = Client('clip', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.encode(image='https://picsum.photos/id/233/100')
    res.summary()
    res[0].summary()


def test_clip_encode_plain_image_blob():
    client = Client('clip', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.encode(
        image=Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob
    )
    res.summary()
    res[0].summary()


def test_clip_encode_plain_image_tensor():
    client = Client('clip', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.encode(
        image=Document(uri='https://picsum.photos/id/233/100')
        .load_uri_to_image_tensor()
        .tensor
    )
    res.summary()
    res[0].summary()


def test_clip_caption():
    docs = DocumentArray(
        [Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob()]
    )
    client = Client('clip', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.caption(
        docs=docs, parameters={'num_captions': 3, 'use_nucleus_sampling': True}
    )
    res.summary()
    res[0].summary()


# def test_blip_encode():
#     docs = DocumentArray([Document(text='hello world'), Document(text='hello jina')])
#     client = Client('blip', token='ebf1afcf5c9432ed5662d8b1d6e20303')
#     res = client.encode(docs=docs)
#     res.summary()
#     res[0].summary()
#
#
# def test_blip_caption():
#     docs = DocumentArray(
#         [Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob()]
#     )
#     client = Client('blip', token='ebf1afcf5c9432ed5662d8b1d6e20303')
#     res = client.caption(
#         docs=docs, parameters={'num_captions': 3, 'use_nucleus_sampling': True}
#     )
#     res.summary()
#     res[0].summary()


def test_blip2_encode_documentarray():
    docs = DocumentArray([Document(text='hello world'), Document(text='hello jina')])
    client = Client('blip2', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.encode(docs=docs)
    res.summary()
    res[0].summary()


def test_blip2_encode_list_of_document():
    docs = [Document(text='hello world'), Document(text='hello jina')]
    client = Client('blip2', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.encode(docs=docs)
    res.summary()
    res[0].summary()


def test_blip2_encode_plain_text():
    client = Client('blip2', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.encode(text='hello world')
    res.summary()
    res[0].summary()


def test_blip2_encode_plain_image_uri():
    client = Client('blip2', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.encode(image='https://picsum.photos/id/233/100')
    res.summary()
    res[0].summary()


def test_blip2_encode_plain_image_blob():
    client = Client('blip2', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.encode(
        image=Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob
    )
    res.summary()
    res[0].summary()


def test_blip2_encode_plain_image_tensor():
    client = Client('blip2', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.encode(
        image=Document(uri='https://picsum.photos/id/233/100')
        .load_uri_to_image_tensor()
        .tensor
    )
    res.summary()
    res[0].summary()


def test_blip2_caption_documentarray():
    docs = DocumentArray(
        [Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob()]
    )
    client = Client('blip2', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.caption(docs=docs)
    res.summary()
    res[0].summary()


def test_blip2_caption_list_of_document():
    docs = [Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob()]
    client = Client('blip2', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.caption(docs=docs)
    res.summary()
    res[0].summary()


def test_blip2_caption_plain_image_uri():
    client = Client('blip2', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.caption(image='https://picsum.photos/id/233/100')
    res.summary()
    res[0].summary()


def test_blip2_caption_plain_image_blob():
    client = Client('blip2', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.caption(
        image=Document(uri='https://picsum.photos/id/233/100').load_uri_to_blob().blob
    )
    res.summary()
    res[0].summary()


def test_blip2_caption_plain_image_tensor():
    client = Client('blip2', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    res = client.caption(
        image=Document(uri='https://picsum.photos/id/233/100')
        .load_uri_to_image_tensor()
        .tensor
    )
    res.summary()
    res[0].summary()
