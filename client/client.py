from typing import Optional

from .base import BaseClient
from .helper import fetch_metadata, login, validate_model


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
        if model_name in self.models:
            return self.models.get(model_name)

        config = fetch_metadata(self.token, model_name)
        model = BaseClient(model_name=model_name, token=self.token, config=config)
        self.models[model_name] = model
        return model
