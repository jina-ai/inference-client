from typing import TYPE_CHECKING, Iterable, Optional, Tuple, Union, overload

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
        prompt: str,
        image: Union[str, bytes, 'ArrayType'],
        *,
        strength: Optional[float],
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        negative_prompt: Optional[str],
        num_images_per_prompt: Optional[int],
        eta: Optional[float],
        output_type: Optional[str],
        return_dict: Optional[bool],
        cross_attention_kwargs: Optional[dict],
        guidance_rescale: Optional[float],
        original_size: Optional[Tuple[int]],
        crops_coords_top_left: Optional[Tuple[int]],
        target_size: Optional[Tuple[int]],
        aesthetic_score: Optional[float],
        negative_aesthetic_score: Optional[float],
        **kwargs,
    ):
        """
        Generate an image from a base image and prompt.

        :param prompt: The prompt to guide the image generation.
        :param image: Image, or tensor representing an image, that will be used as the starting point for the process.
        Can also accpet image latents as image, if passing latents directly, it will not be encoded again.
        :param strength: Conceptually, indicates how much to transform the reference image. Must be between 0 and 1.
        `image` will be used as a starting point, adding more noise to it the larger the strength. The number of
        denoising steps depends on the amount of noise initially added. When strength is 1, added noise will be maximum
        and the denoising process will run for the full number of iterations specified in num_inference_steps. A value
        of 1, therefore, essentially ignores image.
        :param num_inference_steps: The number of denoising steps. More denoising steps usually lead to a higher quality
        image at the expense of slower inference. This parameter will be modulated by strength.
        :param guidance_scale: Guidance scale as defined in Classifier-Free Diffusion Guidance. Higher guidance scale
        encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image
        quality.
        :param negative_prompt: The prompt or prompts not to guide the image generation. Ignored when not using
        guidance.
        :param num_images_per_prompt: The number of images to generate per prompt.
        :param eta: Corresponds to parameter eta (η) in the DDIM paper. Only applies to schedulers.DDIMScheduler, will
        be ignored for others.
        :param output_type: The output format of the generate image.
        :param return_dict: Whether or not to return a StableDiffusionPipelineOutput instead of a plain tuple.
        :param cross_attention_kwargs: A kwargs dictionary that if specified is passed along to the AttentionProcessor
        as defined under self.processor in diffusers.cross_attention.
        :param guidance_rescale: Guidance rescale factor proposed by Common Diffusion Noise Schedules and Sample Steps
        are Flawed. Guidance rescale factor should fix overexposure when using zero terminal SNR.
        :param original_size: Parameter used by Stable Diffusion XL.
        :param crops_coords_top_left: Parameter used by Stable Diffusion XL.
        :param target_size: Parameter used by Stable Diffusion XL.
        :param aesthetic_score: Parameter used by Stable Diffusion XL.
        :param negative_aesthetic_score: Parameter used by Stable Diffusion XL.
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
        prompt: Optional[str] = None,
        image: Optional[Union[str, bytes, 'ArrayType']] = None,
        negative_prompt: Optional[str] = None,
        docs: Optional[Union[Iterable['Document'], 'DocumentArray']] = None,
        **kwargs,
    ):
        """
        Generate an image from prompt or documents containing prompts.

        :param prompt: The prompt or prompts to guide the image generation.
        :param image: The base image to generate from.
        :param negative_prompt: The prompt or prompts not to guide the image generation.
        :param docs: The documents containing base images and prompts to guide the image generation.
        :param kwargs: Additional arguments to pass to the model.
        """
        ...

    def image_to_image(
        self, prompt: str = None, image: Union[str, bytes, 'ArrayType'] = None, **kwargs
    ):
        """
        Generate an image from prompt or documents containing prompts.

        :param prompt: The prompt or prompts to guide the image generation.
        :param image: The base image to generate from.
        :param kwargs: Additional arguments to pass to the model.

        :return: The generated image.
        """
        payload, content_type = self._get_image_to_image_payload(
            prompt=prompt, image=image, **kwargs
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
