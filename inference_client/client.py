from functools import lru_cache
from typing import Optional

from .base import BaseClient
from .helper import get_model_spec, login


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
    def get_model(self, model_name_or_endpoint: str):
        """
        Get a model by name or endpoint. Returns a cached model if it exists.

        :param model_name_or_endpoint: The name of the model or the endpoint.
        :return: The model.
        """

        from urllib.parse import urlparse

        scheme = urlparse(model_name_or_endpoint).scheme

        if not scheme:
            spec = get_model_spec(model_name_or_endpoint, self._auth_token)
            endpoint = spec['endpoints']['grpc']
        else:
            endpoint = model_name_or_endpoint

        return BaseClient(
            model_name=model_name_or_endpoint,
            token=self._auth_token,
            host=endpoint,
        )
