from typing import TYPE_CHECKING, List, Optional, Union, overload

from docarray import Document, DocumentArray

from .helper import get_base_payload

if TYPE_CHECKING:
    from jina import Client


class GenerationMixin:
    """
    Mixin class for generation tasks.
    """

    token: str
    client: 'Client'

    @overload
    def generate(self, prompts: Union[str, List[str]], **kwargs):
        """
        Generate text from prompts using the model.

        :param prompts: the prompts to generate text from.
        :param kwargs: additional arguments to pass to the model.
        """
        ...

    @overload
    def generate(self, prompts: str, *, inplace_images: List = [], **kwargs):
        """
        Generate text from prompts using the model.

        :param prompts: the prompt to generate text from.
        :param inplace_images: the images to generate text from.
        :param kwargs: additional arguments to pass to the model.
        """
        ...

    @overload
    def generate(
        self,
        prompts: Union[str, List[str]],
        *,
        max_new_tokens: Optional[int] = None,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        **kwargs
    ):
        """Generate text from the given prompt.

        :param prompts: The prompt(s) to generate from.
        :param max_new_tokens: The maximum number of tokens to generate, not including the prompt.
        :param num_beams: Number of beams for beam search. 1 means no beam search.
        :param do_sample: Whether to use sampling instead of greedy decoding.
        :param temperature: The temperature to use for sampling. Only relevant if do_sample is True. Higher means more stochastic.
        :param top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering. Only relevant if do_sample is True.
        :param top_p: The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Only relevant if do_sample is True.
        :param repetition_penalty: The parameter for repetition penalty. 1.0 means no penalty.
        :param length_penalty: Exponential penalty to the length that is used with beam-based generation.
                It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence.
                Since the score is the log likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences,
                while length_penalty < 0.0 encourages shorter sequences.
        :param no_repeat_ngram_size: If set to int > 0, all ngrams of that size can only occur once.
        :param kwargs: additional arguments to pass to the model.
        """
        ...

    def generate(self, prompts: Union[str, List[str]], **kwargs):
        """Generate text from the given prompt.

        :param prompts: The prompt(s) to generate from.
        :param kwargs: The arguments to pass to the model.
        :return: The generated text.
        """
        prompts = [prompts] if isinstance(prompts, str) else prompts
        payload = get_base_payload('/generate', self.token, **kwargs)
        payload.update(
            inputs=DocumentArray([Document(text=prompt) for prompt in prompts])
        )
        result = self.client.post(**payload)
        text_out = [r.tags['generated_text'] or r.tags['response'] for r in result]
        return text_out if len(text_out) > 1 else text_out[0]
