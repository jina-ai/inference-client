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
        host: Optional[str] = None,
    ):
        """
        Initializes the client with the desired model and user token.

        :param token: An optional user token for authentication.
        :param host: An optional host to connect to.
        """

        if host:
            assert host.startswith('grpc'), 'Host must be a gRPC endpoint.'
        self._host = host

        try:
            self._auth_token = login(token) if not host else token
        except Exception:
            raise ValueError(
                f'Invalid or expired auth token. Please re-enter your token and try again.'
            ) from None

    @lru_cache(maxsize=10)
    def get_model(self, model_name: str):
        """
        Get a model by name. Returns a cached model if it exists.

        :param model_name: The name of the model to connect to.
        :return: The model.
        """

        spec = (
            get_model_spec(model_name, self._auth_token)
            if not self._host
            else {"endpoints": self._host}
        )
        return BaseClient(
            model_name=model_name,
            token=self._auth_token,
            host=spec["endpoints"]["grpc"],
        )
