import os
from typing import Optional

import hubble
from hubble.utils.auth import Auth


def login(token: Optional[str] = None):
    """
    :param token: A optional token to use for authentication. If not set, it will try to login using the auth token
    in the cache or env, or guide the user to login from a pop-out window
    :return: The token used for authentication.
    """
    if token:
        os.environ['JINA_AUTH_TOKEN'] = token
        Auth.validate_token(token)
        return token
    else:
        hubble.login()
        return hubble.get_token()


def get_model(token: str, model_name: str):
    """
    Validate whether the user has access to the specified model. Retrieves metadata for the specified model.

    :param model: The name of the model to connect to.
    :return: None.
    """
    cfg = fetch_metadata(token, model_name)
    print(cfg)
    return cfg


def fetch_metadata(token: str, model_name: str):
    """
    Retrieves metadata for the specified model.

    :param model: The name of the model to retrieve metadata for.
    :return: A dictionary containing metadata for the model.
    """
    print(f'fetching metadata for {model_name}')
    return {
        'grpc': 'grpcs://api.clip.jina.ai:2096',
        'http': 'https://api.clip.jina.ai:8443',
        'image_size': 224,
    }
