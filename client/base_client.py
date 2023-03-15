from jina import Client


class BaseClient:
    """
    Base client of inference-client.
    """

    def __init__(self, host: str, token: str):
        self.client = Client(host=host)
        self.token = token

    def _encode(self, docs, **kwargs):
        """
        Encode the documents using the model.
        :param docs: docs
        :param kwargs: additional arguments to pass to the model
        :return: encoded docs
        """
        # res = self.client.post(
        #     on='/encode',
        #     inputs=docs,
        #     metadata=(('authorization', self.token),),
        # )
        # return res
        return self._post(docs, endpoint='/encode', **kwargs)

    def _caption(self, docs, **kwargs):
        """
        Caption the documents using the model.
        :param docs: docs
        :param kwargs: additional arguments to pass to the model
        :return: captioned docs
        """
        return self._post(docs, endpoint='/caption', **kwargs)

    def _get_post_payload(self, docs, **kwargs):
        endpoint = kwargs.pop('endpoint', '/')
        inputs = docs  # TODO: add preprocess
        request_size = kwargs.pop('request_size', 1)
        total_docs = (len(docs) if hasattr(docs, '__len__') else None,)
        metadata = (('authorization', self.token),)

        print(f">>>>>> {kwargs}")

        return dict(
            on=endpoint,
            inputs=inputs,
            request_size=request_size,
            total_docs=total_docs,
            metadata=metadata,
            **kwargs,
        )

    def _post(self, docs, **kwargs):
        return self.client.post(**self._get_post_payload(docs, **kwargs))
