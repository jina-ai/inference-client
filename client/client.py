from typing import Optional

from .base import BaseClient
from .helper import fetch_host, login


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

        self.token = login(token)
        self.models = {}

    def get_model(self, model_name):
        """
        Get a model by name. Returns a cached model if it exists.

        :param model_name: The name of the model to connect to.
        :return: The model.
        """
        if (model := self.models.get(model_name)) is None:
            host = fetch_host(self.token, model_name)
            model = BaseClient(model_name=model_name, token=self.token, host=host)
            self.models[model_name] = model
        return model
