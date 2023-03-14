from typing import Optional

from .base_client import BaseClient
from .helper import fetch_metadata, login, validate_model


class Client:
    """
    A Python client for accessing models that are hosted on Jina Cloud.
    """

    def __init__(
        self,
        model_name: str,
        *,
        token: Optional[str] = None,
    ):
        """
        Initializes the client with the desired model and user token.

        :param model_name: The name of the model to access.
        :param token: An optional user token for authentication.
        """
        assert model_name, '`model_name` is required to create a client'

        self.model_name = model_name
        self.token = login(token)

        validate_model(self.token, self.model_name)

        config = fetch_metadata(self.token, self.model_name)
        self.endpoint = config['grpc']
        self.image_size = config['image_size']
        self.model = BaseClient(self.endpoint, self.token)

    def encode(self, docs):
        """
        Encodes the documents using the model.

        :param docs: The documents to encode.
        :return: The encoded documents.
        """
        return self.model.encode(docs)
