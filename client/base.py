import mimetypes
from io import BytesIO
from typing import Optional

from jina import Client
from PIL import Image


class BaseClient:
    """
    Base client of inference-client.
    """

    def __init__(
        self, host: str, token: str, image_size: Optional[int] = None, **kwargs
    ):
        self.client = Client(host=host)
        self.token = token
        self.image_size = image_size

    def _encode(self, content, **kwargs):
        """
        Encode the documents using the model.
        :param content: content
        :param kwargs: additional arguments to pass to the model
        :return: encoded content
        """
        # res = self.client.post(
        #     on='/encode',
        #     inputs=content,
        #     metadata=(('authorization', self.token),),
        # )
        # return res
        return self._post(content, endpoint='/encode', **kwargs)

    def _caption(self, **kwargs):
        """
        Caption the documents using the model.
        :param kwargs: additional arguments to pass to the model
        :return: captioned content
        """
        # TODO get from args/kwargs

        return self._post(None, endpoint='/caption', **kwargs)

    def _iter_doc(self, content):
        from docarray import Document

        for c in content:
            if isinstance(c, str):
                _mime = mimetypes.guess_type(c)[0]
                if _mime and _mime.startswith('image'):
                    if self.image_size:
                        im = Image.open(c).resize((self.image_size, self.image_size))
                        imb = BytesIO()
                        im.save(imb, format='JPEG')
                        d = Document(blob=imb.getvalue())
                    else:
                        d = Document(
                            uri=c,
                        ).load_uri_to_blob()
                else:
                    d = Document(text=c)
            elif isinstance(c, Document):
                if c.content_type in ('text', 'blob'):
                    d = c
                elif not c.blob and c.uri:
                    if self.image_size:
                        im = Image.open(c.uri).resize(
                            (self.image_size, self.image_size)
                        )
                        imb = BytesIO()
                        im.save(imb, format='JPEG')
                        c.blob = imb.getvalue()
                    else:
                        c.load_uri_to_blob()
                    d = c
                elif c.tensor is not None:
                    if self.image_size:
                        c.set_image_tensor_shape(
                            shape=(self.image_size, self.image_size)
                        )
                    d = c
                else:
                    raise TypeError(f'unsupported input type {c!r} {c.content_type}')
            else:
                raise TypeError(f'unsupported input type {c!r}')

            yield d

    def _get_post_payload(self, docs, **kwargs):
        endpoint = kwargs.pop('endpoint', '/')
        inputs = self._iter_doc(docs)  # TODO: add preprocess
        request_size = kwargs.pop('request_size', 1)
        total_docs = len(docs) if hasattr(docs, '__len__') else None
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
