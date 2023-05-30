from typing import TYPE_CHECKING, Iterable, Optional, Union, overload

import numpy
from docarray import Document, DocumentArray
from jina import Client

from .helper import get_base_payload, iter_doc, load_plain_into_document

if TYPE_CHECKING:
    from docarray.typing import ArrayType


class VQAMixin:
    """
    Mixin class for VQA documents.
    """

    token: str
    client: Client

    @overload
    def vqa(self, *, image: Union[str, bytes, 'ArrayType'], question: str, **kwargs):
        """
        Answer the question using the model.

        :param image: the image that the question is about.
        :param question: the question to be answered.
        :param kwargs: additional arguments to pass to the model.
        """
        ...

    @overload
    def vqa(self, *, docs: Union[Iterable['Document'], 'DocumentArray'], **kwargs):
        """
        Answer the question using the model.

        :param docs: the documents to be answered with image as root and question stored in the tags.
        :param kwargs: additional arguments to pass to the model.
        """
        ...

    @overload
    def vqa(
        self,
        *,
        docs: Optional[Union[Iterable['Document'], 'DocumentArray']] = None,
        image: Optional[Union[str, bytes, 'ArrayType']] = None,
        question: Optional[str] = None,
        **kwargs,
    ):
        """
        Answer the question using the model.

        :param docs: the documents to be answered with image as root and question stored in the tags. Default: None.
        :param image: the image that the question is about. Default: None.
        :param question: the question to be answered. Default: None.
        :param kwargs: additional arguments to pass to the model.
        """
        ...

    def vqa(self, **kwargs):
        """
        Answer the question using the model.

        :param kwargs: additional arguments to pass to the model.
        :return: answered content.
        """
        payload, content_type = self._get_vqa_payload(**kwargs)
        result = self.client.post(**payload)
        return self._unbox_vqa_result(
            result=result,
            content_type=content_type,
        )

    def _get_vqa_payload(self, **kwargs):
        payload = get_base_payload('/vqa', self.token, **kwargs)

        if 'docs' in kwargs:
            if 'image' in kwargs or 'question' in kwargs:
                raise ValueError(
                    'More than one input type provided. Please provide only docs or image and question.'
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
            if 'question' not in kwargs:
                raise ValueError('Please provide a question for the image input.')
            content_type = 'plain'
            image_content = kwargs.pop('image')
            if isinstance(image_content, (str, bytes, numpy.ndarray)):
                image_doc = load_plain_into_document(image_content, mime_type='image')
                image_doc.tags.update(prompt=kwargs.pop('question'))
                payload.update(inputs=DocumentArray([image_doc]))
                payload.update(total_docs=1)
            else:
                raise ValueError('Only single image input is supported.')

        else:
            raise ValueError('Please provide either image and question or docs input.')

        return payload, content_type

    def _unbox_vqa_result(
        self,
        result: 'DocumentArray' = None,
        content_type: str = 'docarray',
    ):
        if content_type == 'plain':
            return result[0].tags['response']
        else:
            return result
