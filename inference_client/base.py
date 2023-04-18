from typing import TYPE_CHECKING, Iterable, Optional, Union, overload

from docarray import Document, DocumentArray
from jina import Client

from .helper import load_plain_into_document

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
    def encode(self, text: Union[str, Iterable[str]], **kwargs):
        """
        Encode plain text

        :param text: the text to encode
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def encode(
        self,
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
        Encode image # TODO: add image type

        :param image: the image to encode, can be a `ndarray`, 'bytes' or uri of the image
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def encode(self, docs: Union[Iterable['Document'], 'DocumentArray'], **kwargs):
        """
        Encode documents

        :param docs: the documents to encode
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def encode(
        self,
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

        :param kwargs: additional arguments to pass to the model
        :return: encoded content
        """
        payload, input_type = self._get_post_payload(endpoint='/encode', **kwargs)
        return self._post(payload=payload)

    @overload
    def caption(self, image: Union[str, bytes, 'ArrayType'], **kwargs):
        """
        caption image # TODO: add image type

        :param image: the image to caption, can be a `ndarray`, 'bytes' or uri of the image
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def caption(self, docs: Union[Iterable['Document'], 'DocumentArray'], **kwargs):
        """
        caption documents

        :param docs: the documents to caption
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def caption(
        self,
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

        :param kwargs: additional arguments to pass to the model
        :return: captioned content
        """
        return self._post(endpoint='/caption', **kwargs)

    @overload
    def rank(
        self,
        reference: Union[str, bytes, 'ArrayType'],
        candidates: Iterable[Union[str, bytes, 'ArrayType']],
        **kwargs,
    ):
        """
        Rank the documents using the model.

        :param reference: the reference image or text
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
        reference: Optional[Union[str, bytes, 'ArrayType']] = None,
        candidates: Optional[Iterable[Union[str, bytes, 'ArrayType']]] = None,
        **kwargs,
    ):
        """
        Rank the documents using the model.

        :param docs: the documents to be ranked with candidates stored in the matches. Default: None.
        :param reference: the reference image or text. Default: None.
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
        return self._post(endpoint='/rank', **kwargs)

    @overload
    def vqa(self, image: Union[str, bytes, 'ArrayType'], question: str, **kwargs):
        """
        Answer the question using the model.

        :param image: the image that the question is about
        :param question: the question to be answered
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def vqa(self, docs: Union[Iterable['Document'], 'DocumentArray'], **kwargs):
        """
        Answer the question using the model.

        :param docs: the documents to be answered with image as root and question stored in the tags
        :param kwargs: additional arguments to pass to the model
        """
        ...

    @overload
    def vqa(
        self,
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
        :param kwargs: additional arguments to pass to the model
        """
        ...

    def vqa(self, **kwargs):
        """
        Answer the question using the model.

        :param kwargs: additional arguments to pass to the model
        :return: answered content
        """
        return self._post(endpoint='/vqa', **kwargs)

    def _iter_doc(self, content):
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

    def _get_post_payload(self, **kwargs):
        payload = dict(
            on=kwargs.pop('endpoint', '/'),
            request_size=kwargs.pop('request_size', 1),
            metadata=(('authorization', self.token),),
        )

        input_type = None
        is_list = False

        if 'docs' in kwargs:
            total_docs = (
                len(kwargs.get('docs'))
                if hasattr(kwargs.get('docs'), '__len__')
                else None
            )
            payload.update(total_docs=total_docs)
            payload.update(inputs=self._iter_doc(kwargs.pop('docs')))
            input_type = 'docarray'

        elif 'text' in kwargs:
            text_content = kwargs.pop('text')
            if not isinstance(text_content, list):
                is_list = False
                text_doc = Document(text_content)
                if 'candidates' in kwargs:
                    candidates = kwargs.pop('candidates')
                    text_doc.matches = DocumentArray(
                        [load_plain_into_document(c) for c in candidates]
                    )
                payload.update(inputs=DocumentArray([text_doc]))
                payload.update(total_docs=1)
            else:
                is_list = True
                text_docs = DocumentArray(
                    [Document(text=text) for text in text_content]
                )
                payload.update(inputs=text_docs)
                payload.update(total_docs=len(text_docs))

            input_type = 'plain'

        elif 'image' in kwargs:
            image_content = kwargs.pop('image')
            if not isinstance(image_content, list):
                is_list = False
                image_doc = load_plain_into_document(image_content, is_image=True)
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
                is_list = True
                image_docs = DocumentArray(
                    [load_plain_into_document(c, is_image=True) for c in image_content]
                )
                payload.update(inputs=image_docs)
                payload.update(total_docs=len(image_docs))
            input_type = 'plain'

        elif 'reference' in kwargs:
            reference_doc = load_plain_into_document(kwargs.pop('reference'))
            candidates = kwargs.pop('candidates')
            reference_doc.matches = DocumentArray(
                [load_plain_into_document(c) for c in candidates]
            )
            payload.update(inputs=DocumentArray([reference_doc]))
            payload.update(total_docs=1)
            input_type = 'plain'

        return payload, input_type, is_list

    def _post(self, payload):
        return self.client.post(payload)

    def _unboxed_result(
        results: Optional['DocumentArray'] = None, output_type: str = 'docarray'
    ):
        if results is not None:
            if results.embeddings is None:
                raise ValueError(
                    'Empty embedding returned from the server. '
                    'This often due to a mis-config of the server, '
                    'restarting the server or changing the serving port number often solves the problem'
                )
            return results.embeddings
