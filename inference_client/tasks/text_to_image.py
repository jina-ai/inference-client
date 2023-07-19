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
    def text_to_image(self, prompt: Optional[str], **kwargs):
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
        *,
        prompt: Optional[str],
        docs: Union[Iterable['Document'], 'DocumentArray'],
        **kwargs
    ):
        """
        Generate an image from documents containing prompts.

        :param prompt: The prompt or prompts to guide the image generation.
        :param docs: The documents containing prompts to guide the image generation.
        :param kwargs: Additional arguments to pass to the model.
        """
        ...

    # def text_to_image(self, prompt, *, docs, **kwargs):
    #     pass
