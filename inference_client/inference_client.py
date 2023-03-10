import os

import hubble
from hubble.utils.auth import Auth


class InferenceClient:
    def __init__(self, token: str = None):
        print('InferenceClient.__init__()')
        if token:
            os.environ['JINA_AUTH_TOKEN'] = token
            Auth.validate_token(token)
        else:
            print('InferenceClient.__init__() - calling hubble.login()')
            hubble.login()
            print('InferenceClient.__init__() - calling hubble.get_token()')
            token = hubble.get_token()
            print('InferenceClient.__init__() - token:', token)


if __name__ == '__main__':
    ic = InferenceClient(token='asdfasdf')
