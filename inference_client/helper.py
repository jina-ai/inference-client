import mimetypes
import os
from typing import Optional

import hubble
import numpy
import requests
import torch
from docarray import Document
from hubble.utils.auth import Auth
from jina.logging.logger import JinaLogger

INFERENCE_API = 'https://api.clip.jina.ai/api/v1'
INFERENCE_API_STAGE = 'https://api-stage.clip.jina.ai/api/v1'
logger = JinaLogger('inference-client')


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


def validate_model(token: str, model_name: str):
    """
    Validate whether the user has access to the specified model.

    :param token: The token to use for authentication.
    :param model_name: The name of the model to connect to.
    """
    # TODO: combine with fetch_metadata
    pass
    # try:
    #     resp = requests.post(
    #         f'{INFERENCE_API}/validate',
    #         json={'model': model_name},
    #         headers={'Authorization': token},
    #     )
    #
    #     if resp.status_code == 200:
    #         logger.info(f'successfully validated model {model_name} with token {token}')
    #     else:
    #         raise Exception(f'failed to validate model')
    # except Exception as e:
    #     logger.error(f'failed to validate model {model_name} with token {token}')
    #     raise Exception(f'You do not have access to {model_name}: {e}')


def available_models(token: str):
    """
    Retrieves a list of models that the user has access to.

    :param token: The token to use for authentication.
    :return: A list of model names.
    """
    return ['clip', 'blip']
    # try:
    #     resp = requests.get(
    #         f'{INFERENCE_API}/charts/', headers={'Authorization': token}
    #     )
    #
    #     if resp.status_code == 200:
    #         available = []
    #         for res in resp.json():
    #             name = res['name']
    #             for model_name in res['params_matrix'][0]['model_name']:
    #                 available.append(f'{name}/{model_name}')
    #         logger.info(
    #             f'successfully fetched model list: {available} with token {token}'
    #         )
    #         return available
    #     else:
    #         raise Exception(f'failed to fetch the model list')
    # except Exception as e:
    #     logger.error(f'failed to fetch the model list with token {token}')
    #     raise Exception(f'failed to fetch the model list: {e}')


def fetch_host(token: str, model_name: str):
    """
    Retrieves host for the specified model.

    :param token: The token to use for authentication.
    :param model_name: The name of the model to retrieve host for.
    :return: A string containing the host.
    """
    try:
        resp = requests.get(
            f"https://api.clip.jina.ai/api/v1/models/?model_name={model_name}",
            headers={"Authorization": token},
        )

        if resp.status_code == 401:
            raise ValueError(
                "The given Jina auth token is invalid. Please check your Jina auth token."
            )
        elif resp.status_code == 404:
            raise ValueError(
                f"The given model name `{model_name}` is not valid. "
                f"Please go to https://cloud.jina.ai/user/inference "
                f"and create a model with the given model name."
            )
        resp.raise_for_status()
        return resp.json()["endpoints"]["grpc"]
    except requests.exceptions.HTTPError as err:
        raise ValueError(f"Error: {err!r}")


def load_plain_into_document(content, is_image: bool = False):
    """
    Load plain input into document. If the raw input is a str, it will automatically load into text or image Document
    based on the mime type.

    :param content: input
    :param is_image: whether the input is an image when the input content is of string type, if True, it will force to
    load into image Document
    :return: a text or image document with content loaded
    """
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
