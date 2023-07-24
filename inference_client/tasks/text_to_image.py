from typing import TYPE_CHECKING, Iterable, Optional, Tuple, Union, overload

from docarray import Document, DocumentArray
from jina import Client

from .helper import get_base_payload, iter_doc

if TYPE_CHECKING:
    import torch


class TextToImageMixin:
    """
    Mixin class for text to image generation.
    """

    token: str
    client: Client

    @overload
    def text_to_image(
        self,
        prompt: str,
        *,
        height: Optional[int],
        width: Optional[int],
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        negative_prompt: Optional[str],
        num_images_per_prompt: Optional[int],
        eta: Optional[float],
        latents: Optional['torch.FloatTensor'],
        output_type: Optional[str],
        return_dict: Optional[bool],
        cross_attention_kwargs: Optional[dict],
        guidance_rescale: Optional[float],
        original_size: Optional[Tuple[int]],
        crops_coords_top_left: Optional[Tuple[int]],
        target_size: Optional[Tuple[int]],
        **kwargs,
    ):
        """
        Generate an image from prompt.

        :param prompt: The prompt to guide the image generation.
        :param height: The height in pixels of the generated image.
        :param width: The width in pixels of the generated image.
        :param num_inference_steps: The number of denoising steps. More denoising steps usually lead to a higher quality
        image at the expense of slower inference.
        :param guidance_scale: Guidance scale as defined in Classifier-Free Diffusion Guidance. Higher guidance scale
        encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image
        quality.
        :param negative_prompt: The prompt or prompts not to guide the image generation. Ignored when not using
        guidance.
        :param num_images_per_prompt: The number of images to generate per prompt.
        :param eta: Corresponds to parameter eta (η) in the DDIM paper. Only applies to schedulers.DDIMScheduler, will
        be ignored for others.
        :param latents: Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for
        image generation. Can be used to tweak the same generation with different prompts.
        :param output_type: The output format of the generate image.
        :param return_dict: Whether or not to return a StableDiffusionPipelineOutput instead of a plain tuple.
        :param cross_attention_kwargs: A kwargs dictionary that if specified is passed along to the AttentionProcessor
        as defined under self.processor in diffusers.cross_attention.
        :param guidance_rescale:  Guidance rescale factor proposed by Common Diffusion Noise Schedules and Sample Steps
        are Flawed. Guidance rescale factor should fix overexposure when using zero terminal SNR.
        :param original_size: Parameter used by Stable Diffusion XL.
        :param crops_coords_top_left: Parameter used by Stable Diffusion XL.
        :param target_size: Parameter used by Stable Diffusion XL.
        :param kwargs: Additional arguments to pass to the model.
        """
        ...

    @overload
    def text_to_image(
        self,
        *,
        docs: Union[Iterable['Document'], 'DocumentArray'],
        height: Optional[int],
        width: Optional[int],
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        num_images_per_prompt: Optional[int],
        eta: Optional[float],
        latents: Optional['torch.FloatTensor'],
        output_type: Optional[str],
        return_dict: Optional[bool],
        cross_attention_kwargs: Optional[dict],
        guidance_rescale: Optional[float],
        original_size: Optional[Tuple[int]],
        crops_coords_top_left: Optional[Tuple[int]],
        target_size: Optional[Tuple[int]],
        **kwargs,
    ):
        """
        Generate an image from documents containing prompts.

        :param docs: The documents containing prompts and/or negative_prompt to guide the image generation.
        :param height: The height in pixels of the generated image.
        :param width: The width in pixels of the generated image.
        :param num_inference_steps: The number of denoising steps. More denoising steps usually lead to a higher quality
        image at the expense of slower inference.
        :param guidance_scale: Guidance scale as defined in Classifier-Free Diffusion Guidance. Higher guidance scale
        encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image
        quality.
        :param num_images_per_prompt: The number of images to generate per prompt.
        :param eta: Corresponds to parameter eta (η) in the DDIM paper. Only applies to schedulers.DDIMScheduler, will
        be ignored for others.
        :param latents: Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for
        image generation. Can be used to tweak the same generation with different prompts.
        :param output_type: The output format of the generate image.
        :param return_dict: Whether or not to return a StableDiffusionPipelineOutput instead of a plain tuple.
        :param cross_attention_kwargs: A kwargs dictionary that if specified is passed along to the AttentionProcessor
        as defined under self.processor in diffusers.cross_attention.
        :param guidance_rescale:  Guidance rescale factor proposed by Common Diffusion Noise Schedules and Sample Steps
        are Flawed. Guidance rescale factor should fix overexposure when using zero terminal SNR.
        :param original_size: Parameter used by Stable Diffusion XL.
        :param crops_coords_top_left: Parameter used by Stable Diffusion XL.
        :param target_size: Parameter used by Stable Diffusion XL.
        :param kwargs: Additional arguments to pass to the model.
        """
        ...

    @overload
    def text_to_image(
        self,
        prompt: str,
        height: Optional[int],
        width: Optional[int],
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        negative_prompt: Optional[str],
        num_images_per_prompt: Optional[int],
        eta: Optional[float],
        latents: Optional['torch.FloatTensor'],
        output_type: Optional[str],
        return_dict: Optional[bool],
        cross_attention_kwargs: Optional[dict],
        guidance_rescale: Optional[float],
        original_size: Optional[Tuple[int]],
        crops_coords_top_left: Optional[Tuple[int]],
        target_size: Optional[Tuple[int]],
        docs: Optional[Union[Iterable['Document'], 'DocumentArray']] = None,
        **kwargs,
    ):
        """
        Generate an image from prompt or documents containing prompts.

        :param prompt: The prompt to guide the image generation.
        :param height: The height in pixels of the generated image.
        :param width: The width in pixels of the generated image.
        :param num_inference_steps: The number of denoising steps. More denoising steps usually lead to a higher quality
        image at the expense of slower inference.
        :param guidance_scale: Guidance scale as defined in Classifier-Free Diffusion Guidance. Higher guidance scale
        encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image
        quality.
        :param negative_prompt: The prompt or prompts not to guide the image generation. Ignored when not using
        guidance.
        :param num_images_per_prompt: The number of images to generate per prompt.
        :param eta: Corresponds to parameter eta (η) in the DDIM paper. Only applies to schedulers.DDIMScheduler, will
        be ignored for others.
        :param latents: Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for
        image generation. Can be used to tweak the same generation with different prompts.
        :param output_type: The output format of the generate image.
        :param return_dict: Whether or not to return a StableDiffusionPipelineOutput instead of a plain tuple.
        :param cross_attention_kwargs: A kwargs dictionary that if specified is passed along to the AttentionProcessor
        as defined under self.processor in diffusers.cross_attention.
        :param guidance_rescale: Guidance rescale factor proposed by Common Diffusion Noise Schedules and Sample Steps
        are Flawed. Guidance rescale factor should fix overexposure when using zero terminal SNR.
        :param original_size: Parameter used by Stable Diffusion XL.
        :param crops_coords_top_left: Parameter used by Stable Diffusion XL.
        :param target_size: Parameter used by Stable Diffusion XL.
        :param docs: The documents containing prompts and/or negative_prompt to guide the image generation.
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
        payload, content_type = self._get_text_to_image_payload(prompt=prompt, **kwargs)
        result = self.client.post(**payload)
        return self._unbox_text_to_image_result(result, content_type)

    def _get_text_to_image_payload(self, **kwargs):
        payload = get_base_payload('/text-to-image', self.token, **kwargs)

        if (prompt := kwargs.pop('prompt', None)) is not None:
            if kwargs.get('docs') is not None:
                raise ValueError(
                    'More than one input type provided. Please provide either prompt or docs input.'
                )
            content_type = 'plain'
            prompt_doc = Document(
                tags={
                    'prompt': prompt,
                    'negative_prompt': kwargs.pop('negative_prompt', None),
                }
            )
            payload.update(inputs=DocumentArray([prompt_doc]))
            payload.update(total_docs=1)

        elif (docs := kwargs.pop('docs', None)) is not None:
            content_type = 'docarray'
            total_docs = len(docs) if hasattr(docs, '__len__') else None
            payload.update(total_docs=total_docs)
            payload.update(inputs=iter_doc(docs))
        else:
            raise ValueError('Please provide either prompt or docs input.')

        if (parameters := payload.get('parameters', None)) is not None:
            parameters.update(kwargs)
        else:
            payload.update(parameters=kwargs)

        return payload, content_type

    def _unbox_text_to_image_result(self, result, content_type):
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
