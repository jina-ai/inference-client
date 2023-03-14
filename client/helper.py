import os
from typing import Optional

import hubble
import requests
from hubble.utils.auth import Auth
from jina.logging.logger import JinaLogger

INFERENCE_API = 'https://api.clip.jina.ai/api/v1'
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
    try:
        resp = requests.post(
            f'{INFERENCE_API}/validate',
            json={'model': model_name},
            headers={'Authorization': token},
        )

        if resp.status_code == 200:
            logger.info(f'successfully validated model {model_name} with token {token}')
        else:
            raise Exception(f'failed to validate model')
    except Exception as e:
        logger.error(f'failed to validate model {model_name} with token {token}')
        raise Exception(f'You do not have access to {model_name}: {e}')


def available_models(token: str):
    """
    Retrieves a list of models that the user has access to.

    :param token: The token to use for authentication.
    :return: A list of model names.
    """
    try:
        resp = requests.get(
            f'{INFERENCE_API}/charts/', headers={'Authorization': token}
        )

        if resp.status_code == 200:
            available = []
            for res in resp.json():
                name = res['name']
                for model_name in res['params_matrix'][0]['model_name']:
                    available.append(f'{name}/{model_name}')
            logger.info(
                f'successfully fetched model list: {available} with token {token}'
            )
            return available
        else:
            raise Exception(f'failed to fetch the model list')
    except Exception as e:
        logger.error(f'failed to fetch the model list with token {token}')
        raise Exception(f'failed to fetch the model list: {e}')


def fetch_metadata(token: str, model_name: str):
    """
    Retrieves metadata for the specified model.

    :param token: The token to use for authentication.
    :param model_name: The name of the model to retrieve metadata for.
    :return: A dictionary containing metadata for the model.
    """
    return {
        'grpc': 'grpcs://api.clip.jina.ai:2096',
        'http': 'https://api.clip.jina.ai:8443',
        'image_size': 224,
    }
