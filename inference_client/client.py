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

        self._auth_token = login(token)

    @lru_cache(maxsize=10)
    def get_model(self, model_name: str):
        """
        Get a model by name. Returns a cached model if it exists.

        :param model_name: The name of the model to connect to.
        :return: The model.
        """

        spec = get_model_spec(model_name, self._auth_token)
        return BaseClient(
            model_name=model_name,
            token=self._auth_token,
            host=spec["endpoints"]["grpc"],
        )
