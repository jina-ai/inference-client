import os
from typing import Optional

import hubble
from hubble.utils.auth import Auth


class Client:
    def __init__(
        self,
        model_name: str = None,
        *,
        token: Optional[str] = None,
    ):
        """
        Initializes the client with the desired model and user token.

        :param model_name: The name of the model to connect to.
        :param token: An optional user token for authentication.
        :return: None
        """
        self.model_name = model_name
        self.token = self._login(token=token)
        self._connect(self.model_name)

    def _login(self, token: Optional[str] = None):
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

    def _connect(self, model_name):
        """
        Validate whether the user has access to the specified model. Retrieves metadata for the specified model.

        :param model: The name of the model to connect to.
        :return: None.
        """
        cfg = self._fetch_metadata(model_name)
        print(cfg)

    def _fetch_metadata(self, model_name):
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
