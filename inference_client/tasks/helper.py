import mimetypes
from typing import Optional

from docarray import Document


def load_plain_into_document(content, mime_type: Optional[str] = None):
    """
    Load plain input into document. If the raw input is a str, it will automatically load into text or image Document
    based on the provided content type, or the guessed mime type.

    :param content: input
    :param mime_type: the mime type of the input, if None, it will be guessed based on the input
    :return: a text or image document with content loaded
    """
    import numpy
    import torch

    if isinstance(content, str):
        if mime_type is None:
            _mime = mimetypes.guess_type(content)[0]
            if _mime and _mime.startswith('image'):
                mime_type = 'image'
            else:
                mime_type = 'text'
        if mime_type == 'image':
            return Document(
                uri=content,
            ).load_uri_to_blob()
        elif mime_type == 'text':
            return Document(text=content)
    elif isinstance(content, bytes):
        return Document(blob=content)
    elif isinstance(content, (numpy.ndarray, torch.Tensor)):
        return Document(tensor=content)
    else:
        raise TypeError(f"Cannot convert content to Document")


def iter_doc(content):
    """
    Iterate over the input content and yield Document.

    :param content: input content to be converted/loaded to Document
    :yield: a Document
    """
    for c in content:
        if c.content_type in ('text', 'blob'):
            d = c
        elif not c.blob and c.uri:
            c.load_uri_to_blob()
            d = c
        elif c.tensor is not None:
            d = c
        else:
            raise TypeError(f'unsupported input type {c!r} {c.content_type}')
        yield d
