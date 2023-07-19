from typing import TYPE_CHECKING, Iterable, Optional, Union, overload

import numpy
from docarray import Document, DocumentArray
from jina import Client

from .helper import get_base_payload, iter_doc, load_plain_into_document

if TYPE_CHECKING:
    from docarray.typing import ArrayType


class TextToImageMixin:
    """
    Mixin class for text to image generation.
    """

    token: str
    client: Client

    @overload
    def text_to_image(self, prompt: str, **kwargs):
        """
        Generate an image from prompt.

        :param prompt: The prompt or prompts to guide the image generation..
        :param kwargs: Additional arguments to pass to the model.
        """
        ...

    @overload
    def text_to_image(
        self, *, docs: Union[Iterable['Document'], 'DocumentArray'], **kwargs
    ):
        """
        Generate an image from documents containing prompts.

        :param docs: The documents containing prompts to guide the image generation.
        :param kwargs: Additional arguments to pass to the model.
        """
        ...

    @overload
    def text_to_image(
        self,
        prompt: Optional[str] = None,
        docs: Optional[Union[Iterable['Document'], 'DocumentArray']] = None,
        **kwargs,
    ):
        """
        Generate an image from prompt or documents containing prompts.

        :param prompt: The prompt or prompts to guide the image generation. Default: None.
        :param docs: The documents containing prompts to guide the image generation. Default: None.
        :param kwargs: Additional arguments to pass to the model.
        """
        ...

    def text_to_image(self, prompt: str = None, **kwargs):
        """
        Generate an image from prompt or documents containing prompts.

        :param prompt: The prompt or prompts to guide the image generation.
        :param kwargs: Additional arguments to pass to the model.

        :return: The generated image.
        """
        self._get_text_to_image_payload(prompt=prompt, **kwargs)
        return None

    def _get_text_to_image_payload(self, **kwargs):
        payload = get_base_payload('/text-to-image', self.token, **kwargs)
        print(payload)

        if kwargs.get('prompt') is not None:
            if kwargs.get('docs') is not None:
                raise ValueError(
                    'More than one input type provided. Please provide either prompt or docs input.'
                )
            content_type = 'plain'
            prompt_doc = Document(tags={'prompt': kwargs.get('prompt')})
            payload.update(inputs=DocumentArray([prompt_doc]))
            payload.update(total_docs=1)

        elif kwargs.get('docs') is not None:
            content_type = 'docarray'
            total_docs = (
                len(kwargs.get('docs'))
                if hasattr(kwargs.get('docs'), '__len__')
                else None
            )
            payload.update(total_docs=total_docs)
            payload.update(inputs=iter_doc(kwargs.pop('docs')))

        return payload, content_type
