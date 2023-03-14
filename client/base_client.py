from jina import Client


class BaseClient:
    """
    Base client of inference-client.
    """

    def __init__(self, endpoint: str, token: str):
        self.client = Client(host=endpoint)
        self.token = token

    def encode(self, docs):
        """
        Encode the documents using the model.
        :param docs: docs
        :return: encoded docs
        """
        res = self.client.post(
            on='/encode',
            inputs=docs,
            return_results=True,
            metadata=(('authorization', self.token),),
        )
        return res
