from typing import TYPE_CHECKING, Iterable, Optional, Union, overload

import numpy
from docarray import Document, DocumentArray
from jina import Client

from .helper import get_base_payload, iter_doc, load_plain_into_document

if TYPE_CHECKING:
    from docarray.typing import ArrayType


class ImageToImageMixin:
    """
    Mixin class for text to image generation.
    """

    token: str
    client: Client

    @overload
    def image_to_image(
        self,
        image: Union[str, bytes, 'ArrayType'],
        prompt: str,
        *,
        negative_prompt: Optional[str],
        **kwargs,
    ):
        """
        Generate an image from a base image and prompt.

        :param image: The base image to generate from.
        :param prompt: The prompt or prompts to guide the image generation.
        :param negative_prompt: The prompt or prompts not to guide the image generation.
        :param kwargs: Additional arguments to pass to the model.
        """
        ...

    @overload
    def image_to_image(
        self, *, docs: Union[Iterable['Document'], 'DocumentArray'], **kwargs
    ):
        """
        Generate an image from documents containing base images and prompts.

        :param docs: The documents containing base images and prompts to guide the image generation.
        :param kwargs: Additional arguments to pass to the model.
        """
        ...

    @overload
    def image_to_image(
        self,
        image: Optional[Union[str, bytes, 'ArrayType']] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        docs: Optional[Union[Iterable['Document'], 'DocumentArray']] = None,
        **kwargs,
    ):
        """
        Generate an image from prompt or documents containing prompts.

        :param image: The base image to generate from.
        :param prompt: The prompt or prompts to guide the image generation.
        :param negative_prompt: The prompt or prompts not to guide the image generation.
        :param docs: The documents containing base images and prompts to guide the image generation.
        :param kwargs: Additional arguments to pass to the model.
        """
        ...

    def image_to_image(
        self, image: Union[str, bytes, 'ArrayType'] = None, prompt: str = None, **kwargs
    ):
        """
        Generate an image from prompt or documents containing prompts.

        :param image: The base image to generate from.
        :param prompt: The prompt or prompts to guide the image generation.
        :param kwargs: Additional arguments to pass to the model.

        :return: The generated image.
        """
        payload, content_type = self._get_image_to_image_payload(
            image=image, prompt=prompt, **kwargs
        )
        result = self.client.post(**payload)
        return self._unbox_image_to_image_result(result, content_type)

    def _get_image_to_image_payload(self, **kwargs):
        payload = get_base_payload('/image-to-image', self.token, **kwargs)

        if (image_content := kwargs.pop('image', None)) is not None:
            if kwargs.get('docs') is not None:
                raise ValueError(
                    'More than one input type provided. Please provide either prompt or docs input.'
                )
            if (prompt := kwargs.pop('prompt', None)) is None:
                raise ValueError('Please provide a prompt input.')
            content_type = 'plain'
            image_doc = load_plain_into_document(image_content, mime_type='image')
            image_doc.tags.update(
                prompt=prompt, negative_prompt=kwargs.pop('negative_prompt', None)
            )
            payload.update(inputs=DocumentArray([image_doc]))
            payload.update(total_docs=1)

        elif (docs := kwargs.pop('docs', None)) is not None:
            content_type = 'docarray'
            total_docs = len(docs) if hasattr(docs, '__len__') else None
            payload.update(total_docs=total_docs)
            payload.update(inputs=iter_doc(docs))
        else:
            raise ValueError('Please provide either docs or image and prompt input.')

        return payload, content_type

    def _unbox_image_to_image_result(self, result, content_type):
        if content_type == 'plain':
            matches = result[0].matches
            if len(matches[0].blob) > 0:
                output = [m.blob for m in matches]
            elif matches[0].tensor is not None:
                output = [m.tensor for m in matches]
            else:
                raise ValueError('No image found in the result.')
            return output[0] if len(output) == 1 else output
        else:
            return result
