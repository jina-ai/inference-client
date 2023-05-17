import mimetypes
import os
from functools import lru_cache
from typing import Optional

import hubble
import numpy
import requests
from docarray import Document
from hubble.utils.auth import Auth

from .config import settings
from .logging import logger


def login(token: Optional[str] = None) -> str:
    """
    Try to login using the token.

    :param token: An optional token to use for authentication. If not set, it will try to login using the auth token
        in the env, or guide the user to login from a pop-out window
    :return: The validated token.
    """
    if token:
        os.environ['JINA_AUTH_TOKEN'] = token
        Auth.validate_token(token)
        logger.info(f'successfully validated token: {token}')
        return token
    else:
        hubble.login()
        token = hubble.get_token()
        logger.info(f'successfully logged in with token: {token}')
        return token


@lru_cache(maxsize=10)
def get_model_spec(model_name: str, token: str):
    """
    Retrieves the model spec for the specified model.

    :param model_name: The name of the model to retrieve spec for.
    :param token: The token to use for authentication.
    :return: A dict containing the model spec.
    """
    try:
        resp = requests.get(
            f"{settings.api_endpoint}/models/?model_name={model_name}",
            headers={"Authorization": token},
        )

        if resp.status_code == 401:
            raise ValueError(
                "The given Jina auth token is invalid. Please check your Jina auth token."
            )
        elif resp.status_code == 404:
            raise ValueError(
                f"Invalid model name `{model_name}` provided. "
                f"Please visit https://cloud.jina.ai/user/inference to create and use the model names listed there."
            )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f'Failed to fetch the model spec for {model_name}')
        raise e from None


def load_plain_into_document(content, is_image: bool = False):
    """
    Load plain input into document. If the raw input is a str, it will automatically load into text or image Document
    based on the mime type.

    :param content: input
    :param is_image: whether the input is an image when the input content is of string type, if True, it will force to
        load into image Document
    :return: a text or image document with content loaded
    """
    import torch

    if isinstance(content, str):
        if is_image:
            return Document(
                uri=content,
            ).load_uri_to_blob()

        _mime = mimetypes.guess_type(content)[0]
        if _mime and _mime.startswith('image'):
            return Document(
                uri=content,
            ).load_uri_to_blob()
        else:
            return Document(text=content)
    elif isinstance(content, bytes):
        return Document(blob=content)
    elif isinstance(content, (numpy.ndarray, torch.Tensor)):
        return Document(tensor=content)
    else:
        raise TypeError(f"Cannot convert content to Document")
