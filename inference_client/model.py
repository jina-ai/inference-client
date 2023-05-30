from jina import Client

from .tasks.caption import CaptionMixin
from .tasks.encode import EncodeMixin
from .tasks.rank import RankMixin
from .tasks.upscale import UpscaleMixin
from .tasks.vqa import VQAMixin


class Model(CaptionMixin, EncodeMixin, RankMixin, UpscaleMixin, VQAMixin):
    """
    The model to be used for inference.
    """

    def __init__(self, model_name: str, token: str, host: str, **kwargs):
        self.model_name = model_name
        self.token = token
        self.host = host
        self.client = Client(host=self.host)
