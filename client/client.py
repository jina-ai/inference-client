from typing import Optional

from .helper import get_model, login


class Client:
    def __init__(
        self,
        model_name: str = None,
        *,
        token: Optional[str] = None,
    ):
        """
        Initializes the client with the desired model and user token.

        :param model_name: The name of the model to connect to.
        :param token: An optional user token for authentication.
        :return: None
        """
        self.model_name = model_name
        self.token = login(token)
        self.cfg = get_model(self.token, self.model_name)
