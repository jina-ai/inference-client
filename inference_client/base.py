from jina import Client

from .tasks.caption import CaptionMixin
from .tasks.encode import EncodeMixin
from .tasks.rank import RankMixin
from .tasks.vqa import VQAMixin


class BaseClient(CaptionMixin, EncodeMixin, RankMixin, VQAMixin):
    """
    Base client of inference-client.
    """

    def __init__(self, model_name: str, token: str, host: str, **kwargs):
        self.model_name = model_name
        self.token = token
        self.host = host
        self.client = Client(host=self.host)
