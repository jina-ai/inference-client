import os
from typing import Optional

import hubble
from base_client import BaseClient
from hubble.utils.auth import Auth


class Client:
    def __init__(
        self,
        model: Optional[str] = None,
        *,
        token: Optional[str] = None,
        force: Optional[bool] = False,
    ):
        self.token = self._login(token=token, force=force)
        self.model = model
        self._connect(self.model)

    def _login(self, token: Optional[str] = None, force: Optional[bool] = False):
        if token:
            os.environ['JINA_AUTH_TOKEN'] = token
            Auth.validate_token(token)
            return token
        else:
            hubble.login(force=force)
            return hubble.get_token()

    def _connect(self, model):
        cfg = self._fetch_metadata(model)
        print(cfg)

    def _fetch_metadata(self, model):
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
