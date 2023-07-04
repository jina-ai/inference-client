from functools import lru_cache
from typing import Optional

from .helper import get_model_spec, login
from .model import Model


class Client:
    """
    A Python client for accessing models that are hosted on Jina Cloud.
    """

    def __init__(
        self,
        *,
        token: Optional[str] = None,
    ):
        """
        Initializes the client with the desired model and user token.

        :param token: An optional user token for authentication.
        """

        try:
            self._auth_token = login(token) if token else None
        except Exception:
            raise ValueError(
                f'Invalid or expired auth token. Please re-enter your token and try again.'
            ) from None

    @lru_cache(maxsize=10)
    def get_model(
        self, model_name: Optional[str] = None, endpoint: Optional[str] = None
    ):
        """
        Get a model by name or endpoint. Returns a cached model if it exists.

        Example:

        ```python
        from jina import Client

        client = Client()

        # get model by name
        model = client.get_model('Salesforce/blip2-flan-t5-xl')

        # or get model by endpoint
        model = client.get_model('grpc://localhost:12345')
        ```

        :param model_name: The name of the model.
        :param endpoint: The endpoint of the model.
        :return: The model.
        """

        if not self._auth_token and not endpoint:
            raise ValueError(
                'Please provide an endpoint or a valid user token to access the model.'
            )

        from urllib.parse import urlparse

        o = urlparse(model_name or endpoint)
        if o.scheme and o.netloc:
            endpoint = model_name or endpoint
        elif model_name:
            spec = get_model_spec(model_name, self._auth_token)
            endpoint = spec['endpoints']['grpc']

        return Model(
            model_name=model_name,
            token=self._auth_token,
            host=endpoint,
        )
