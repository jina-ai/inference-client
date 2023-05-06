from typing import TYPE_CHECKING, Iterable, Optional, Union, overload

from docarray import Document, DocumentArray
from helper import iter_doc, load_plain_into_document
from jina import Client

if TYPE_CHECKING:
    from docarray.typing import ArrayType


class RankMixin:
    """
    Mixin class for encoding documents.
    """

    token: str
    client: Client

    @overload
    def rank(
        self,
        text: str,
        candidates: Iterable[Union[str, bytes, 'ArrayType']],
        **kwargs,
    ):
        """
        Rank the documents using the model.

        :param text: the reference text
        :param candidates: the candidates to be ranked, can be either a list of strings or a list of images
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def rank(
        self,
        image: Union[str, bytes, 'ArrayType'],
        candidates: Iterable[Union[str, bytes, 'ArrayType']],
        **kwargs,
    ):
        """
        Rank the documents using the model.

        :param image: the reference image, can be a `ndarray`, 'bytes' or uri of the image
        :param candidates: the candidates to be ranked, can be either a list of strings or a list of images
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def rank(self, docs: Union[Iterable['Document'], 'DocumentArray'], **kwargs):
        """
        Rank the documents using the model.

        :param docs: the documents to be ranked with candidates stored in the matches
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def rank(
        self,
        docs: Optional[Union[Iterable['Document'], 'DocumentArray']] = None,
        text: Optional[str] = None,
        image: Optional[Union[str, bytes, 'ArrayType']] = None,
        candidates: Optional[Iterable[Union[str, bytes, 'ArrayType']]] = None,
        **kwargs,
    ):
        """
        Rank the documents using the model.

        :param docs: the documents to be ranked with candidates stored in the matches. Default: None.
        :param text: the reference text. Default: None.
        :param image: the reference image, can be a `ndarray`, 'bytes' or uri of the image. Default: None.
        :param candidates: the candidates to be ranked, can be either a list of strings or a list of images. Default: None.
        :param kwargs: additional arguments to pass to the model
        """
        ...

    def rank(self, **kwargs):
        """
        Rank the documents using the model.

        :param kwargs: additional arguments to pass to the model
        :return: ranked content
        """
        payload, content_type = self._get_rank_payload(endpoint='/rank', **kwargs)
        result = self.client.post(payload=payload)
        return self._unbox_rank_result(
            result=result,
            content_type=content_type,
        )

    def _get_rank_payload(self, **kwargs):
        payload = dict(
            on=kwargs.pop('endpoint', '/'),
            request_size=kwargs.pop('request_size', 1),
            metadata=(('authorization', self.token),),
        )

        content_type = None

        if 'docs' in kwargs:
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
                text_doc = Document(text=text_content)
                if 'candidates' in kwargs:
                    candidates = kwargs.pop('candidates')
                    text_doc.matches = DocumentArray(
                        [load_plain_into_document(c) for c in candidates]
                    )
                payload.update(inputs=DocumentArray([text_doc]))
                payload.update(total_docs=1)
            else:
                raise ValueError('Text input must be a string.')

        elif 'image' in kwargs:
            if 'text' in kwargs:
                raise ValueError(
                    'Multi-modal input not supported. Please provide only text or image input.'
                )

            content_type = 'plain'
            image_content = kwargs.pop('image')
            if isinstance(image_content, str):
                image_doc = load_plain_into_document(image_content, mime_type='image')
                if 'candidates' in kwargs:
                    candidates = kwargs.pop('candidates')
                    image_doc.matches = DocumentArray(
                        [load_plain_into_document(c) for c in candidates]
                    )
                elif 'question' in kwargs:
                    image_doc.tags.update(prompt=kwargs.pop('question'))
                payload.update(inputs=DocumentArray([image_doc]))
                payload.update(total_docs=1)
            else:
                raise ValueError('Only single image input is supported.')

        return payload, content_type

    def _unbox_rank_result(
        self,
        result: 'DocumentArray' = None,
        content_type: str = 'docarray',
    ):
        if content_type == 'plain':
            return [
                (d.uri, d.scores) if d.uri else (d.content, d.scores)
                for d in result[0].matches
            ]
        else:
            return result
