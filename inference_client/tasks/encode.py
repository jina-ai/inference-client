from typing import TYPE_CHECKING, Iterable, Optional, Union, overload

import numpy
from docarray import Document, DocumentArray
from jina import Client

from .helper import get_base_payload, iter_doc, load_plain_into_document

if TYPE_CHECKING:
    from docarray.typing import ArrayType


class EncodeMixin:
    """
    Mixin class for encoding documents.
    """

    token: str
    client: Client

    @overload
    def encode(self, *, text: Union[str, Iterable[str]], **kwargs):
        """
        Encode plain text.

        :param text: the text to encode.
        :param kwargs: additional arguments to pass to the model.
        """
        ...

    @overload
    def encode(
        self,
        *,
        image: Union[
            str,
            bytes,
            'ArrayType',
            Iterable[str],
            Iterable[bytes],
            Iterable['ArrayType'],
        ],
        **kwargs,
    ):
        """
        Encode image.

        :param image: the image to encode, can be a `ndarray`, 'bytes' or uri of the image.
        :param kwargs: additional arguments to pass to the model.
        """
        ...

    @overload
    def encode(self, *, docs: Union[Iterable['Document'], 'DocumentArray'], **kwargs):
        """
        Encode documents

        :param docs: the documents to encode
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def encode(
        self,
        *,
        docs: Optional[Union[Iterable['Document'], 'DocumentArray']] = None,
        text: Optional[Union[str, Iterable[str]]] = None,
        image: Optional[
            Union[
                str,
                bytes,
                'ArrayType',
                Iterable[str],
                Iterable[bytes],
                Iterable['ArrayType'],
            ]
        ] = None,
        **kwargs,
    ):
        """
        Encode text, image, or documents using a pre-trained model.

        :param docs: the documents to encode. Default: None.
        :param text: the text to encode. Default: None.
        :param image: the image to encode, can be a `ndarray`, 'bytes' or uri of the image. Default: None.
        :param kwargs: additional arguments to pass to the model.
        """
        ...

    def encode(self, **kwargs):
        """
        Encode the documents using the model.

        :param kwargs: additional arguments to pass to the model.
        :return: encoded content.
        """
        payload, content_type, is_list = self._get_enocde_payload(**kwargs)
        result = self.client.post(**payload)
        return self._unbox_encode_result(
            result=result,
            content_type=content_type,
            is_list=is_list,
        )

    def _get_enocde_payload(self, **kwargs):
        payload = get_base_payload('/encode', self.token, **kwargs)
        is_list = False

        if 'docs' in kwargs:
            if 'text' in kwargs or 'image' in kwargs:
                raise ValueError(
                    'More than one input type provided. Please provide only text, image or docs input.'
                )
            content_type = 'docarray'
            total_docs = (
                len(kwargs.get('docs'))
                if hasattr(kwargs.get('docs'), '__len__')
                else None
            )
            payload.update(total_docs=total_docs)
            payload.update(inputs=iter_doc(kwargs.pop('docs')))

        elif 'text' in kwargs:
            if 'image' in kwargs:
                raise ValueError(
                    'Multi-modal input not supported. Please provide only text or image input.'
                )
            content_type = 'plain'
            text_content = kwargs.pop('text')
            if isinstance(text_content, str):
                is_list = False
                text_doc = Document(text=text_content)
                payload.update(inputs=DocumentArray([text_doc]))
                payload.update(total_docs=1)
            else:
                is_list = True
                text_docs = DocumentArray([Document(text=c) for c in text_content])
                payload.update(inputs=text_docs)
                payload.update(total_docs=len(text_docs))
                payload.update(results_in_order=True)

        elif 'image' in kwargs:
            if 'text' in kwargs:
                raise ValueError(
                    'Multi-modal input not supported. Please provide only text or image input.'
                )
            content_type = 'plain'
            image_content = kwargs.pop('image')
            if isinstance(image_content, (str, bytes, numpy.ndarray)):
                is_list = False
                image_doc = load_plain_into_document(image_content, mime_type='image')
                payload.update(inputs=DocumentArray([image_doc]))
                payload.update(total_docs=1)
            else:
                is_list = True
                image_docs = DocumentArray(
                    [
                        load_plain_into_document(c, mime_type='image')
                        for c in image_content
                    ]
                )
                payload.update(inputs=image_docs)
                payload.update(total_docs=len(image_docs))
                payload.update(results_in_order=True)

        else:
            raise ValueError(
                'Please provide either text, image or docs input to encode.'
            )

        return payload, content_type, is_list

    def _unbox_encode_result(
        self,
        result: 'DocumentArray' = None,
        content_type: str = 'docarray',
        is_list: bool = False,
    ):
        if content_type == 'plain':
            if is_list:
                return result.embeddings
            else:
                return result[0].embedding
        else:
            return result
