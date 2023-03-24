import mimetypes
from typing import TYPE_CHECKING, Optional, Union, overload

import numpy
import torch
from docarray import Document, DocumentArray
from jina import Client

if TYPE_CHECKING:  # pragma: no cover
    from docarray.typing import ArrayType


class BaseClient:
    """
    Base client of inference-client.
    """

    def __init__(self, model_name: str, token: str, host: str, **kwargs):
        self.model_name = model_name
        self.token = token
        self.host = host
        self.client = Client(host=self.host)

    @overload
    def encode(self, text: str, **kwargs):
        """
        Encode plain text
        :param text: the text to encode
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def encode(self, image: Union[str, bytes, 'ArrayType'], **kwargs):
        """
        Encode image # TODO: add image type
        :param image: the image to encode, can be a `ndarray`, 'bytes' or uri of the image
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def encode(self, docs: Union['Document', 'DocumentArray'], **kwargs):
        """
        Encode documents
        :param docs: the documents to encode
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def encode(
        self,
        docs: Optional[Union['Document', 'DocumentArray']] = None,
        text: Optional[str] = None,
        image: Optional[Union[str, bytes, 'ArrayType']] = None,
        **kwargs,
    ):
        """
        Encode text, image, or documents using a pre-trained model.

        :param docs: The documents to encode. Default: None.
        :param text: The text to encode. Default: None.
        :param image: The image to encode, can be a `ndarray`, 'bytes' or uri of the image. Default: None.
        :param kwargs: Additional arguments to pass to the model.
        """
        ...

    def encode(self, **kwargs):
        """
        Encode the documents using the model.
        :param kwargs: additional arguments to pass to the model
        :return: encoded content
        """
        return self._post(endpoint='/encode', **kwargs)

    @overload
    def caption(self, image: Union[str, bytes, 'ArrayType'], **kwargs):
        """
        caption image # TODO: add image type
        :param image: the image to caption, can be a `ndarray`, 'bytes' or uri of the image
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def caption(self, docs: Union['Document', 'DocumentArray'], **kwargs):
        """
        caption documents
        :param docs: the documents to caption
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def caption(
        self,
        docs: Optional[Union['Document', 'DocumentArray']] = None,
        image: Optional[Union[str, bytes, 'ArrayType']] = None,
        **kwargs,
    ):
        """
        Generate a caption for an image or a set of documents using a pre-trained model.

        :param docs: The documents to caption. Default: None.
        :param image: The image to caption, can be a `ndarray`, 'bytes' or uri of the image. Default: None.
        :param kwargs: Additional arguments to pass to the model.
        """
        ...

    def caption(self, **kwargs):
        """
        Caption the documents using the model.
        :param kwargs: additional arguments to pass to the model
        :return: captioned content
        """
        # TODO get from args/kwargs

        return self._post(endpoint='/caption', **kwargs)

    def _iter_doc(self, content):
        from docarray import Document

        for c in content:
            if isinstance(c, str):
                _mime = mimetypes.guess_type(c)[0]
                if _mime and _mime.startswith('image'):
                    d = Document(
                        uri=c,
                    ).load_uri_to_blob()
                else:
                    d = Document(text=c)
            elif isinstance(c, Document):
                if c.content_type in ('text', 'blob'):
                    d = c
                elif not c.blob and c.uri:
                    c.load_uri_to_blob()
                    d = c
                elif c.tensor is not None:
                    d = c
                else:
                    raise TypeError(f'unsupported input type {c!r} {c.content_type}')
            else:
                raise TypeError(f'unsupported input type {c!r}')

            yield d

    def _get_post_payload(self, **kwargs):
        payload = dict(
            on=kwargs.pop('endpoint', '/'),
            request_size=kwargs.pop('request_size', 1),
            metadata=(('authorization', self.token),),
        )

        if 'docs' in kwargs:
            total_docs = (
                len(kwargs.get('docs'))
                if hasattr(kwargs.get('docs'), '__len__')
                else None
            )
            payload.update(total_docs=total_docs)
            payload.update(inputs=self._iter_doc(kwargs.pop('docs')))
        elif 'text' in kwargs:
            payload.update(inputs=DocumentArray([Document(text=kwargs.pop('text'))]))
            payload.update(total_docs=1)
        elif 'image' in kwargs:
            image = kwargs.pop('image')
            if isinstance(image, str):
                payload.update(inputs=DocumentArray([Document(uri=image)]))
            elif isinstance(image, bytes):
                payload.update(inputs=DocumentArray([Document(blob=image)]))
            elif isinstance(image, (numpy.ndarray, torch.Tensor)):
                payload.update(inputs=DocumentArray([Document(tensor=image)]))
            payload.update(total_docs=1)

        return payload

    def _post(self, **kwargs):
        return self.client.post(**self._get_post_payload(**kwargs))
