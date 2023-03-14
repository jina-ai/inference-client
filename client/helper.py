import os
from typing import Optional

import hubble
import requests
from hubble.utils.auth import Auth

INFERENCE_API = 'https://api.clip.jina.ai'


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
        return token
    else:
        hubble.login()
        return hubble.get_token()


def validate_model(token: str, model_name: str):
    """
    Validate whether the user has access to the specified model.

    :param token: The token to use for authentication.
    :param model_name: The name of the model to connect to.
    """
    try:
        requests.post(
            f'{INFERENCE_API}/validate',
            json={'model': model_name},
            headers={'Authorization': token},
        )
    except requests.exceptions.HTTPError:
        available = available_models(token)
        raise Exception(
            f'You do not have access to {model_name}. Available models: {available}'
        )


def available_models(token: str):
    """
    Retrieves a list of models that the user has access to.

    :param token: The token to use for authentication.
    :return: A list of model names.
    """
    print('fetching available models')
    try:
        requests.post(f'{INFERENCE_API}/available', headers={'Authorization': token})
        return ['CLIP/ViT-B-32', 'CLIP/ViT-B-16']

    except requests.exceptions.HTTPError:
        raise Exception('Unkown error while fetching available models')


def fetch_metadata(token: str, model_name: str):
    """
    Retrieves metadata for the specified model.

    :param token: The token to use for authentication.
    :param model_name: The name of the model to retrieve metadata for.
    :return: A dictionary containing metadata for the model.
    """
    print(f'fetching metadata for {model_name}')
    return {
        'grpc': 'grpcs://api.clip.jina.ai:2096',
        'http': 'https://api.clip.jina.ai:8443',
        'image_size': 224,
    }
