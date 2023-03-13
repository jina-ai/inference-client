import os
from typing import Optional

import hubble
from base_client import BaseClient
from hubble.utils.auth import Auth


class Client:
    def __init__(
        self,
        model: str = None,
        *,
        token: Optional[str] = None,
        force: Optional[bool] = False,
    ):
        """
        Initializes the client with the desired model and user token.

        :param model: The name of the model to connect to.
        :param token: An optional user token for authentication.
        :param force: An optional boolean to decide whether to force the login. If set and no token is provided, force
        to login regardless of the current system env session
        :return: None
        """
        self.token = self._login(token=token, force=force)
        self.model = model
        self._connect(self.model)

    def _login(self, token: Optional[str] = None, force: Optional[bool] = False):
        """

        :param token: A optional token to use for authentication. If not set, it will try to login using the auth token
        in the cache or env, or guide the user to login from a pop-out window
        :param force: An optional boolean to decide whether to force the login. If set and no token is provided, force
        to login regardless of the current system env session
        :return: The token used for authentication.
        """
        if token:
            os.environ['JINA_AUTH_TOKEN'] = token
            Auth.validate_token(token)
            return token
        else:
            hubble.login(force=force)
            return hubble.get_token()

    def _connect(self, model):
        """
        Validate whether the user has access to the specified model. Retrieves metadata for the specified model.

        :param model: The name of the model to connect to.
        :return: None.
        """
        cfg = self._fetch_metadata(model)
        print(cfg)

    def _fetch_metadata(self, model):
        """
        Retrieves metadata for the specified model.

        :param model: The name of the model to retrieve metadata for.
        :return: A dictionary containing metadata for the model.
        """
        print(f'fetching metadata for {model}')
        return {
            'grpc': 'grpcs://api.clip.jina.ai:2096',
            'http': 'https://api.clip.jina.ai:8443',
            'image_size': 224,
        }


if __name__ == '__main__':
    # ic = Client(token='asdfasdf')
    # os.environ['JINA_AUTH_TOKEN'] = '123412341234'
    # hubble.login()
    # ebf1afcf5c9432ed5662d8b1d6e20303

    # ic = Client(model='AAAA', token='asdfasdf')
    ic = Client(model='AAAA', token='ebf1afcf5c9432ed5662d8b1d6e20303')
    # ic = Client(model='AAAA')
