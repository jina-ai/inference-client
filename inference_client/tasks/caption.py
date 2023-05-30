from typing import TYPE_CHECKING, Iterable, Optional, Union, overload

import numpy
from docarray import Document, DocumentArray
from jina import Client

from .helper import get_base_payload, iter_doc, load_plain_into_document

if TYPE_CHECKING:
    from docarray.typing import ArrayType


class CaptionMixin:
    """
    Mixin class for captioning documents.
    """

    token: str
    client: Client

    @overload
    def caption(self, *, image: Union[str, bytes, 'ArrayType'], **kwargs):
        """
        caption image # TODO: add image type

        :param image: the image to caption, can be a `ndarray`, 'bytes' or uri of the image
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def caption(self, *, docs: Union[Iterable['Document'], 'DocumentArray'], **kwargs):
        """
        caption documents

        :param docs: the documents to caption
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def caption(
        self,
        *,
        docs: Optional[Union[Iterable['Document'], 'DocumentArray']] = None,
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

        :param kwargs: additional arguments to pass to the model.
        :return: captioned content.
        """
        payload, content_type = self._get_caption_payload(**kwargs)
        result = self.client.post(**payload)
        return self._unbox_caption_result(
            result=result,
            content_type=content_type,
        )

    def _get_caption_payload(self, **kwargs):
        payload = get_base_payload('/caption', self.token, **kwargs)

        if 'docs' in kwargs:
            if 'image' in kwargs:
                raise ValueError(
                    'More than one input type provided. Please provide only docs or image input.'
                )
            content_type = 'docarray'
            total_docs = (
                len(kwargs.get('docs'))
                if hasattr(kwargs.get('docs'), '__len__')
                else None
            )
            payload.update(total_docs=total_docs)
            payload.update(inputs=iter_doc(kwargs.pop('docs')))

        elif 'image' in kwargs:
            content_type = 'plain'
            image_content = kwargs.pop('image')
            if isinstance(image_content, (str, bytes, numpy.ndarray)):
                image_doc = load_plain_into_document(image_content, mime_type='image')
                payload.update(inputs=DocumentArray([image_doc]))
                payload.update(total_docs=1)
            else:
                raise ValueError('Only single image input is supported.')

        else:
            raise ValueError('Please provide either image or docs input.')

        return payload, content_type

    def _unbox_caption_result(
        self,
        result: 'DocumentArray' = None,
        content_type: str = 'docarray',
    ):
        if content_type == 'plain':
            return result[0].tags['response']
        else:
            return result
